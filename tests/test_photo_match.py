import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from harpor.eval import photo_match as P

H, W = 48, 64  # the manifest's photo size
WINDOW = (12, 6, 52, 42)  # u0, v0, u1, v1, end exclusive
RH, RW = 54, 60  # the render: the 40 x 36 window at 1.5x
SHA_A, SHA_B = "a" * 64, "b" * 64
DEVDATA = Path(os.environ.get(
    "GATOR_DEVDATA",
    "/private/tmp/claude-501/-Users-andrewli-lunar-lab-research/60864a71-d343-43d1-9009-0a27ee1f6e39/scratchpad/devdata"))


def photo_and_mask():
    """A red box with a dark grid on a noisy background, and the box's mask."""
    rng = np.random.default_rng(0)
    photo = rng.integers(60, 200, (H, W, 3)).astype(np.uint8)
    mask = np.zeros((H, W), bool)
    mask[10:38, 16:48] = True
    obj = np.zeros((H, W, 3), np.uint8)
    obj[...] = (180, 50, 50)
    obj[::4] = obj[:, ::4] = (25, 12, 12)
    photo[mask] = obj[mask]
    return photo, mask


def crop(a):
    u0, v0, u1, v1 = WINDOW
    return P.fit(a[v0:v1, u0:u1], RH, RW)


def good_render():
    """The photo's object pixels on a transparent background: a perfect match."""
    photo, mask = photo_and_mask()
    rgba = np.zeros((RH, RW, 4), np.uint8)
    m = crop(mask)
    rgba[m, :3], rgba[m, 3] = crop(photo)[m], 255
    return rgba


def as_float(rgba):
    return rgba.astype(np.float64) / 255


def write_png(path, a):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a).save(path)


def session(root, reviews, sha=SHA_A, result=True, clay=False, file_scale=1):
    """A session folder with one input view. reviews: {receipt: {asset: RGBA uint8 render}}."""
    photo, mask = photo_and_mask()
    work = root / "work"
    big = lambda a: np.repeat(np.repeat(a, file_scale, 0), file_scale, 1)  # noqa: E731
    write_png(work / "input" / "images" / "000.png", big(photo))
    write_png(work / "input" / "masks" / "000.png", big(np.where(mask, 255, 0).astype(np.uint8)))
    (work / "input" / "manifest.json").write_text(json.dumps({"views": [
        {"index": 0, "width": W, "height": H, "image": "input/images/000.png", "mask": "input/masks/000.png"}]}))
    for receipt, renders in reviews.items():
        d = work / "reviews" / receipt
        rec = {"name": "view00", "crop": list(WINDOW), "scale": 1.5}
        for n, a in renders.items():
            write_png(d / f"view00_{n}.png", a)
            rec[n] = f"view00_{n}.png"
            rec[f"iou_{n}"] = 0.5
        orbit = {n: f"orbit0_{n}.png" for n in renders}  # no photo: never scored, so no file is needed
        (d / "review.json").write_text(json.dumps({"clay": clay, "views": [rec, {"name": "orbit0", **orbit}]}))
    if result:
        (root / "result.json").write_text(json.dumps({"verdict": "complete", "glb": "work/exports/001.glb",
                                                      "sha256": sha, "agent": "claude", "model": "claude-opus-5-5"}))
    return root


def test_lab_of_known_colours():
    lab = P.srgb_to_lab([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert np.allclose(lab[0], [100, 0, 0], atol=0.01)
    assert np.allclose(lab[1], [0, 0, 0], atol=1e-9)
    assert np.allclose(lab[2], [53.24, 80.09, 67.20], atol=0.02)


def test_matching_render_beats_a_wrong_one_on_every_metric():
    photo, mask = photo_and_mask()
    p, m = as_float(crop(photo)), crop(mask)
    good = as_float(good_render())
    ref = P.view_scores(p, m, good)
    assert ref["iou"] == 1.0 and ref["color_de"] == pytest.approx(0, abs=1e-9) and ref["pix_de"] == 0
    assert ref["edge"] > 0.8

    shifted = np.roll(good, 6, axis=1)  # silhouette off
    assert P.view_scores(p, m, shifted)["iou"] < ref["iou"]

    recolored = good.copy()
    recolored[..., :3] = good[..., 2::-1]  # red box rendered blue
    s = P.view_scores(p, m, recolored)
    assert s["iou"] == 1.0 and s["color_de"] > 20 and s["pix_de"] > 20

    scrambled = good.copy()  # the same pixels in another order: same mean colour, wrong per pixel
    idx = np.flatnonzero(m)
    scrambled.reshape(-1, 4)[idx, :3] = good.reshape(-1, 4)[np.random.default_rng(1).permutation(idx), :3]
    s = P.view_scores(p, m, scrambled)
    assert s["color_de"] == pytest.approx(0, abs=1e-6) and s["pix_de"] > 10 and s["edge"] < ref["edge"]

    flat = good.copy()  # the grid painted over in the mean colour
    flat[m, :3] = good[m, :3].mean(0)
    s = P.view_scores(p, m, flat)
    assert s["iou"] == 1.0 and s["edge"] < ref["edge"] - 0.2


def test_edge_is_zero_without_gradient_and_scores_are_nan_without_pixels():
    photo, mask = photo_and_mask()
    p, m = as_float(crop(photo)), crop(mask)
    empty = np.zeros((RH, RW, 4))
    s = P.view_scores(p, m, empty)
    assert s["iou"] == 0.0 and math.isnan(s["color_de"]) and math.isnan(s["pix_de"])
    assert P.pearson(np.ones(5), np.arange(5.0)) == 0.0


def test_session_reproduces_the_review_crop(tmp_path):
    """A render made from the photo's own crop scores IoU 1 and delta E 0, also when the stored photo is
    larger than the manifest says (review.py fits it to the manifest's size first)."""
    bad = good_render()
    bad[..., :3] = bad[..., 2::-1]
    for scale in (1, 2):
        root = session(tmp_path / f"s{scale}", {f"r001-{SHA_A[:12]}": {"original": bad, "candidate": good_render()}},
                       file_scale=scale)
        s = P.score_session(root)
        v = s["views"][0]
        assert s["n_views"] == 1 and v["name"] == "view00" and v["size_wh"] == [RW, RH]
        assert v["candidate"]["iou"] == 1.0 and v["candidate"]["pix_de"] == 0.0
        assert s["delta"]["color_de"] < -20 and s["delta"]["pix_de"] < -20 and s["delta"]["iou"] == 0.0
        assert s["mean"]["candidate"]["edge"] == v["candidate"]["edge"]
        assert s["verdict"] == "complete" and s["agent"] == "claude" and s["warnings"] == []


def test_review_of_the_delivered_glb_is_picked(tmp_path):
    g = good_render()
    renders = {"original": g, "candidate": g}
    work = session(tmp_path / "a", {f"r001-{SHA_A[:12]}": renders, f"r002-{SHA_B[:12]}": renders,
                                    f"r003-{SHA_A[:12]}": renders, "r004-nothex": renders}, sha=SHA_A) / "work"
    (work / "reviews" / f"r005-{SHA_A[:12]}").mkdir()  # a review that failed: no review.json
    assert P.pick_review(work, SHA_A) == (work / "reviews" / f"r003-{SHA_A[:12]}", [])
    assert P.pick_review(work, SHA_B) == (work / "reviews" / f"r002-{SHA_B[:12]}", [])
    folder, warnings = P.pick_review(work, "c" * 64)
    assert folder.name == f"r003-{SHA_A[:12]}" and "no review of the delivered GLB" in warnings[0]
    folder, warnings = P.pick_review(work, None)
    assert folder.name == f"r003-{SHA_A[:12]}" and "no sha256" in warnings[0]


def test_latest_review_is_by_number_and_no_review_exits(tmp_path):
    g = good_render()
    work = session(tmp_path / "a", {f"r999-{SHA_B[:12]}": {"candidate": g},
                                    f"r1000-{SHA_B[:12]}": {"candidate": g}}) / "work"
    folder, warnings = P.pick_review(work, SHA_A)
    assert folder.name == f"r1000-{SHA_B[:12]}" and warnings
    empty = session(tmp_path / "b", {}) / "work"
    with pytest.raises(SystemExit, match="no review"):
        P.pick_review(empty, SHA_A)


def test_session_without_result_or_original_render_and_clay_review(tmp_path):
    root = session(tmp_path / "s", {f"r001-{SHA_A[:12]}": {"candidate": good_render()}}, result=False, clay=True)
    s = P.score_session(root)
    assert "original" not in s["mean"] and "delta" not in s and "delta" not in s["views"][0]
    assert s["verdict"] is None and len(s["warnings"]) == 3  # no result.json, no sha, clay
    row = P.table([s])[1]
    assert row.split()[1:4] == ["None", "1", "r001"] and " -" in row  # named after the run folder, not "s"


def test_dino_with_a_fake_embedder(tmp_path):
    def embed(images):  # a stand-in for DINOv2: the downsampled image as a unit vector
        f = np.stack([P.fit(im, 8, 8).ravel() for im in images])
        return f / np.linalg.norm(f, axis=1, keepdims=True)

    bad = good_render()
    bad[..., :3] = bad[..., 2::-1]
    s = P.score_session(session(tmp_path / "s", {f"r001-{SHA_A[:12]}": {"original": bad, "candidate": good_render()}}),
                        embed)
    v = s["views"][0]
    assert v["candidate"]["dino"] == pytest.approx(1.0) and v["original"]["dino"] < 0.99
    assert s["delta"]["dino"] > 0 and "dino" in P.table([s], dino=True)[0]


def test_dino_is_skipped_without_the_cache(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setenv("TORCH_HOME", str(tmp_path))
    embed, note = P.dino_embedder()
    assert embed is None and note.startswith("dino skipped") and str(tmp_path) in note


def test_cli_prints_rows_and_writes_json(tmp_path, capsys, monkeypatch):
    empty = np.zeros((RH, RW, 4), np.uint8)  # a candidate that vanished: NaN colour scores
    a = session(tmp_path / "opus", {f"r001-{SHA_A[:12]}": {"original": good_render(), "candidate": empty}})
    b = session(tmp_path / "astra", {f"r002-{SHA_A[:12]}": {"original": good_render(), "candidate": good_render()}})
    monkeypatch.setattr(P, "dino_embedder", lambda: (None, "dino skipped: test"))
    out = tmp_path / "out" / "scores.json"
    P.main([str(a), str(b), "--dino", "--json", str(out)])
    lines = capsys.readouterr().out.splitlines()
    assert [ln.split()[:2] for ln in lines[2:4]] == [["opus", "complete"], ["astra", "complete"]]
    assert "dino skipped: test" in lines and "dino" not in lines[1]
    data = json.loads(out.read_text())  # strict JSON: NaN became null
    assert [s["name"] for s in data["sessions"]] == ["opus", "astra"]
    assert data["sessions"][0]["views"][0]["candidate"]["color_de"] is None
    assert data["sessions"][0]["delta"]["iou"] == -1.0


@pytest.mark.skipif(not (DEVDATA / "chair_opus" / "result.json").exists(), reason="the chair sessions are not on this machine")
def test_real_chair_sessions():
    sessions = [P.score_session(DEVDATA / n) for n in ("chair_opus", "chair_astra")]
    for s in sessions:
        assert s["warnings"] == [] and s["review"].endswith(s["sha256"][:12]) and s["n_views"] == 5
        for v in s["views"]:  # the crop and the mask match review.py: the same IoU
            for n in P.ASSETS:
                assert v[n]["iou"] == pytest.approx(v["review_iou"][n], abs=1e-12)
        for n in P.ASSETS:
            assert all(math.isfinite(x) for x in s["mean"][n].values())
    # Both sessions start from the same original.glb under the same cameras and lights.
    for k, x in sessions[0]["mean"]["original"].items():
        assert x == pytest.approx(sessions[1]["mean"]["original"][k], rel=1e-3, abs=1e-3)


def test_background_composite_credits_a_real_opening():
    """A see-through part: the photo shows the floor through a hole that the filled mask covers. A render
    with the same opening beats an opaque one on bg_de, and a render that drops solid geometry loses."""
    h, w = 120, 160
    yy, xx = np.mgrid[0:h, 0:w]
    floor = np.stack([0.3 + 0.002 * xx, 0.35 + 0.001 * yy, np.full((h, w), 0.4)], -1)
    obj = np.zeros((h, w), bool)
    obj[30:90, 40:120] = True
    hole = np.zeros((h, w), bool)
    hole[45:75, 65:95] = True
    photo = floor.copy()
    photo[obj & ~hole] = (0.1, 0.2, 0.7)
    mask = obj  # SAM-style: the mask fills the opening

    def render(alpha):
        r = np.zeros((h, w, 4))
        r[..., :3] = (0.1, 0.2, 0.7)
        r[..., 3] = alpha
        return r

    opening = P.view_scores(photo, mask, render((obj & ~hole).astype(float)))
    opaque = P.view_scores(photo, mask, render(obj.astype(float)))
    chunk = np.zeros((h, w), bool)
    chunk[30:45, 40:120] = True
    missing = P.view_scores(photo, mask, render((obj & ~hole & ~chunk).astype(float)))
    assert opening["bg_de"] < opaque["bg_de"] and opening["bg_de"] < missing["bg_de"]
    assert opening["bg_edge"] > opaque["bg_edge"]
    assert opaque["iou"] > opening["iou"]  # the IoU alone prefers the opaque render, which is the point
