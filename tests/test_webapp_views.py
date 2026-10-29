import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from harpor.stage_a import mvsam3d as M
from harpor.webapp import views as V

H, W = 48, 64
K = np.array([[60.0, 0, 31.5], [0, 60.0, 23.5], [0, 0, 1]])


def look_at(C, target=(0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0)):
    """OpenCV world-to-camera 3x4 of a camera at C looking at target, world z up."""
    f = V.unit(np.asarray(target) - C)
    x = V.unit(np.cross(f, up))
    R = np.stack([x, np.cross(f, x), f])
    return np.hstack([R, (-R @ C)[:, None]])


def circle(n, arc=360.0, radius=2.0, height=0.5):
    a = np.radians(np.arange(n) * arc / (n if arc == 360 else n - 1))
    return np.stack([radius * np.cos(a), radius * np.sin(a), np.full(n, height)], 1)


def sphere(E, r=0.4):
    """Pointmap (camera coords) and mask of a sphere of radius r at the origin; background at depth 10."""
    v, u = np.mgrid[0:H, 0:W]
    d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones((H, W))], -1)
    R, t = E[:, :3], E[:, 3]
    o, dw = -R.T @ t, d @ R
    a, b, c = (dw ** 2).sum(-1), 2 * dw @ o, o @ o - r * r
    disc = b * b - 4 * a * c
    hit = disc > 0
    s = np.where(hit, (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a), 10.0)
    return (d * s[..., None]).astype(np.float32), hit


def corner_mask():
    v, u = np.mgrid[0:H, 0:W]
    return (u - 8) ** 2 + (v - 8) ** 2 < 36


def lower_mask():
    """The bottom two thirds of the frame: SAM 3 took the floor along with the object."""
    return np.repeat((np.arange(H) >= H // 3)[:, None], W, 1)


def make_pool(root, names, E, masks, P):
    for d in ("input/images", "input/object", "masks"):
        (root / d).mkdir(parents=True, exist_ok=True)
    for i, (n, m) in enumerate(zip(names, masks)):
        Image.fromarray(np.full((H, W, 3), (100 + i, 120, 140), np.uint8)).save(root / "input" / "images" / f"{n}.png")
        Image.fromarray((m * 255).astype(np.uint8)).save(root / "masks" / f"{n}.png")
        rgba = np.zeros((H, W, 4), np.uint8)
        rgba[..., 3] = m * 255
        Image.fromarray(rgba).save(root / "input" / "object" / f"{n}.png")
    P = np.array(P)
    np.savez(root / "da3_output.npz", depth=P[..., 2], conf=np.ones(P.shape[:3], np.float32), pointmaps=P,
             pointmaps_sam3d=P.transpose(0, 3, 1, 2).copy(), extrinsics=np.array(E, np.float32),
             intrinsics=np.repeat(K[None], len(names), 0).astype(np.float32),
             image_files=np.array([str(root / "input" / "images" / f"{n}.png") for n in names]), process_res=504)


def make_job(tmp_path, n=24, arc=360.0, bad=(), big=(), small=(), no_mask=(), scores=None, **job):
    """A job after masks, prep and da3: pool frames 0, 10, 20, ... on a circle around a sphere."""
    j = tmp_path / "job"
    frames = [10 * i for i in range(n)]
    names = [f"f{f:05d}" for f in frames]
    E = [look_at(C) for C in circle(n, arc)]
    P, masks = zip(*(sphere(e) for e in E))
    masks = [corner_mask() if i in bad else lower_mask() if i in big
             else np.pad(np.ones((2, 2), bool), ((23, 23), (31, 31))) if i in small else m for i, m in enumerate(masks)]
    keep = [i for i in range(n) if i not in no_mask]
    make_pool(j / "pool", [names[i] for i in keep], [E[i] for i in keep], [masks[i] for i in keep], [P[i] for i in keep])
    scores = scores or {}
    sam = {"views": [{"name": nm, "picked": [] if i in no_mask else [0],
                      "instances": [] if i in no_mask else [{"index": 0, "score": scores.get(i, 0.9)}]}
                     for i, nm in enumerate(names)]}
    (j / "scene" / "sam3_toy").mkdir(parents=True)
    (j / "scene" / "sam3_toy" / "sam3.json").write_text(json.dumps(sam))
    (j / "frames").mkdir()
    total = frames[-1] + 1
    (j / "frames" / "meta.json").write_text(json.dumps({"times": [i / 30 for i in range(total)], "rel": [1.0] * total}))
    picks = {str(p): p for p in job.get("picks", [])}
    (j / "frames" / "pool.json").write_text(json.dumps({"pool": frames, "names": {str(f): nm for f, nm in zip(frames, names)},
                                                        "picks": picks}))
    (j / "job.json").write_text(json.dumps({"mode": "auto", "prompt": "toy", "slug": "toy", "n_views": 6, **job}))
    return j, E, P, masks


def test_kcenter_spreads_the_views_and_keeps_the_anchor():
    dirs = V.unit(circle(36))
    S = V.kcenter(dirs, np.ones(36), 6, 7)
    D = V.angles(dirs)
    assert S[0] == 7 and len(set(S)) == 6 and S == V.kcenter(dirs, np.ones(36), 6, 7)
    assert D[np.ix_(S, S)][np.triu_indices(6, 1)].min() >= 45 and D[:, S].min(1).max() <= 30
    q = np.ones(36)
    q[[25, 26]] = (0.5, 2.0)
    S = V.kcenter(dirs, q, 4, 0, locked=[3])
    assert S[:2] == [0, 3] and len(S) == 4
    assert V.kcenter(dirs, np.ones(36), 40, 0)[0] == 0 and len(V.kcenter(dirs, np.ones(36), 40, 0)) == 36
    S = V.kcenter(dirs[:4], np.ones(4), 6, 2)
    assert sorted(S) == [0, 1, 2, 3] and S[0] == 2


def test_thin_drops_near_duplicates():
    dirs = V.unit(circle(12, arc=44))
    D, q = V.angles(dirs), np.linspace(1, 0.5, 12)
    S = V.kcenter(dirs, q, 6, 5)
    kept = V.thin(S, D, q, fixed=[5])
    assert 5 in kept and len(kept) == 3
    assert D[np.ix_(kept, kept)][np.triu_indices(3, 1)].min() >= 15
    wide = V.kcenter(V.unit(circle(36)), np.ones(36), 6, 0)
    assert V.thin(wide, V.angles(V.unit(circle(36))), np.ones(36)) == wide


def test_thin_goes_on_past_a_close_fixed_pair():
    a = np.radians([0, 5, 100, 108, 200, 280])  # 0 and 5 are fixed; 100 and 108 are free and close
    D = V.angles(np.stack([np.cos(a), np.sin(a), np.zeros(6)], 1))
    kept = V.thin(range(6), D, np.ones(6), fixed=[0, 1])
    assert kept[:2] == [0, 1] and len(kept) == 5 and len({2, 3} & set(kept)) == 1
    assert V.thin([0, 1, 4], D, np.ones(6), fixed=[0, 1], keep=2) == [0, 1, 4]


def test_consistency_flags_a_wrong_mask():
    E = [look_at(C) for C in circle(8)]
    P, masks = map(list, zip(*(sphere(e) for e in E)))
    masks[3] = corner_mask()
    s = V.consistency(P, masks, np.array(E), [K] * 8)
    assert s[3] < 0.1 and np.delete(s, 3).min() > 0.8
    c = V.object_center(P, [np.ones((H, W))] * 8, masks, np.array(E))
    assert np.abs(c).max() < 0.25
    assert np.allclose(V.camera_centers(np.array(E)), circle(8), atol=1e-9)
    big = [np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize((2 * W, 2 * H), Image.NEAREST)) > 0 for m in masks]
    assert np.allclose(V.consistency(P, big, np.array(E), [K] * 8), s, atol=0.05)  # masks at another size


def test_consistency_flags_a_mask_with_more_than_the_object():
    E = [look_at(C) for C in circle(16)]
    P, masks = map(list, zip(*(sphere(e) for e in E)))
    assert V.consistency(P, masks, np.array(E), [K] * 16).min() > 0.95
    for wide in (lower_mask(), np.ones((H, W), bool)):  # the sphere and the floor; the whole frame
        masks[3] = wide
        s = V.consistency(P, masks, np.array(E), [K] * 16)
        assert s[3] < 0.3 and np.delete(s, 3).min() > 0.7


def test_quality():
    q, terms = V.quality(0.9, 0.0125, True, 0.45, 0.05)
    assert q == pytest.approx(0.9 * 0.5 ** 4) and terms["area"] == 0.5 and terms["sharp"] == 0.5
    assert V.quality(1.0, 0.2, False, 2.0, 0.0)[0] == 1.0
    m = np.zeros((10, 10), bool)
    m[5, 5] = True
    assert not V.truncated(m)
    m[1, 5] = True
    assert V.truncated(m)


def test_build_run_slices_the_pool(tmp_path):
    names = ["f00000", "f00010", "f00020", "f00030", "f00040"]
    E = [look_at(C) for C in circle(5)]
    P, masks = zip(*(sphere(e) for e in E))
    make_pool(tmp_path / "pool", names, E, masks, P)
    run = tmp_path / "run"
    V.build_run(tmp_path / "pool", ["f00030", "f00010"], run)
    da = np.load(run / "da3_output.npz")
    assert np.allclose(da["extrinsics"], np.array(E)[[3, 1]], atol=1e-6) and da["pointmaps"].shape == (2, H, W, 3)
    assert da["pointmaps_sam3d"].shape == (2, 3, H, W) and int(da["process_res"]) == 504
    assert [str(f) for f in da["image_files"]] == [str(run / "input" / "images" / f"{i}.png") for i in (0, 1)]
    assert M.view_names(run) == ["0", "1"]
    cams = M.cameras(run)
    assert np.allclose(cams[0]["w2c"], np.eye(4), atol=1e-6) and np.allclose(np.array(cams[0]["K"]), K, atol=1e-4)
    assert (np.array(Image.open(run / "masks" / "0.png")) > 0).sum() == masks[3].sum()
    prep = json.loads((run / "prep.json").read_text())
    assert prep["views"][0] == {"name": "0", "pool_name": "f00030", "frame": 30}
    V.build_run(tmp_path / "pool", ["f00040", "f00000", "f00020"], run)
    assert M.view_names(run) == ["0", "1", "2"] and len(M.cameras(run)) == 3
    assert not (run / "lidar").exists()  # clouds=None: no LiDAR points

    # The lidar stage's points of the picks (lidar/cloud/<name>.glb) go to run/lidar/<i>.glb in pick order; a
    # pick without a file gets none.
    clouds = tmp_path / "cloud"
    clouds.mkdir()
    for n in ("f00000", "f00020", "f00030"):
        (clouds / f"{n}.glb").write_bytes(b"glTF of " + n.encode())
    V.build_run(tmp_path / "pool", ["f00040", "f00000", "f00020"], run, clouds=clouds)
    assert sorted(p.name for p in (run / "lidar").iterdir()) == ["1.glb", "2.glb"]
    assert [(run / "lidar" / f"{i}.glb").read_bytes() for i in (1, 2)] == [b"glTF of f00000", b"glTF of f00020"]
    V.build_run(tmp_path / "pool", ["f00040"], run, clouds=clouds)
    assert not (run / "lidar").exists()


def test_select_auto(tmp_path):
    j, E, _, masks = make_job(tmp_path, bad=(4,), small=(9,), no_mask=(13,), scores={5: 0.95})
    (j / "agent" / "s").mkdir(parents=True)  # a stale agent run is removed
    sel = V.select(j)
    st = {c["name"]: c["status"] for c in sel["candidates"]}
    assert sel["ok"] and sel["anchor"] == "f00050" and len(sel["picks"]) == 6 and sel["picks"][0]["name"] == "f00050"
    assert (st["f00040"], st["f00090"], st["f00130"]) == ("inconsistent", "small", "no_mask")
    assert sum(s == "picked" for s in st.values()) == 6 and not (j / "agent").exists()
    assert sel["min_spacing_deg"] >= 40 and sel["coverage_deg"] <= 35
    assert sel["picks"][0]["q_terms"]["score"] == 0.95 and len(sel["ring"]["points"]) == 24
    assert sel["ring"]["points"][13] is None and np.abs(sel["ring"]["center"]).max() < 0.3
    r = np.linalg.norm(np.array([p for p in sel["ring"]["points"] if p]) - sel["ring"]["center"], axis=1)
    assert np.allclose(r, 2.0, atol=0.3)  # the cameras' circle around the object, seen from above
    uv = np.array([p for p in sel["ring"]["points"] if p]) - sel["ring"]["center"]
    assert (np.diff(np.unwrap(np.arctan2(uv[:, 1], uv[:, 0]))) > 0).all()  # counterclockwise, as in the world
    assert json.loads((j / "selection.json").read_text()) == sel
    assert M.view_names(j / "run") == [str(i) for i in range(6)] and len(M.cameras(j / "run")) == 6
    prep = json.loads((j / "run" / "prep.json").read_text())
    assert [v["pool_name"] for v in prep["views"]] == [p["name"] for p in sel["picks"]]
    assert (np.array(Image.open(j / "run" / "masks" / "0.png")) > 0).sum() == masks[5].sum()

    job = json.loads((j / "job.json").read_text())
    job.update(view0=70, overrides={"locked": ["f00090", "f00080"], "excluded": ["f00050", 0]})
    (j / "job.json").write_text(json.dumps(job))
    sel = V.select(j)
    names = [p["name"] for p in sel["picks"]]
    assert names[:3] == ["f00070", "f00090", "f00080"] and "f00050" not in names and "f00000" not in names
    assert any("f00090 is small" in w for w in sel["warnings"])


def test_select_thins_a_short_arc(tmp_path):
    j, *_ = make_job(tmp_path, n=12, arc=44)
    sel = V.select(j)
    assert sel["ok"] and len(sel["picks"]) == 3 and sel["picks"][0]["name"] == "f00000"
    assert sel["min_spacing_deg"] >= 15  # 13.8 when the guard dropped views by q instead of rerunning k-center
    assert any("covers only 50 deg" in w and "kept 3 views instead of 6" in w for w in sel["warnings"])
    assert not any("only" in w and "apart" in w for w in sel["warnings"])

    # A view locked 4.5 deg from view 0 stays, and the free views still keep 15 deg from every pick.
    job = json.loads((j / "job.json").read_text())
    job["overrides"] = {"locked": ["f00010"]}
    (j / "job.json").write_text(json.dumps(job))
    sel = V.select(j)
    names = [p["name"] for p in sel["picks"]]
    D = V.angles([p["dir"] for p in sel["picks"]])
    assert names[:2] == ["f00000", "f00010"] and len(names) == 4
    assert min(D[a, b] for a in range(4) for b in range(max(a + 1, 2), 4)) >= 15  # every pair with a free view
    assert sel["min_spacing_deg"] < 5 and any("two picks are only 4 deg apart" in w for w in sel["warnings"])


def test_select_leaves_an_oversized_mask_out_of_the_center(tmp_path):
    j, *_ = make_job(tmp_path, big=(3,))
    sel = V.select(j)
    st = {c["name"]: c for c in sel["candidates"]}
    assert st["f00030"]["status"] == "inconsistent" and st["f00030"]["consistency"] < 0.3
    assert sum(c["status"] == "inconsistent" for c in sel["candidates"]) == 1
    assert np.abs(sel["ring"]["center"]).max() < 0.04  # 0.08 with the wide mask's floor points in the median


def test_select_manual(tmp_path):
    j, *_ = make_job(tmp_path, bad=(4,), no_mask=(13,), mode="manual", picks=[0, 40, 130, 120], view0=120)
    sel = V.select(j)
    assert [p["name"] for p in sel["picks"]] == ["f00120", "f00000", "f00040"] and sel["n_requested"] == 3
    assert any("frame 130 has no mask" in w for w in sel["warnings"])
    assert any("frame 40 is inconsistent" in w for w in sel["warnings"])
    assert M.view_names(j / "run") == ["0", "1", "2"]


def test_select_stops_when_masks_disagree(tmp_path):
    j, *_ = make_job(tmp_path, bad=tuple(range(0, 18, 2)))
    (j / "run").mkdir()
    with pytest.raises(SystemExit) as e:
        V.select(j)
    assert e.value.code == 2 and not (j / "run").exists()
    sel = json.loads((j / "selection.json").read_text())
    assert not sel["ok"] and "9 of 24 masked views disagree" in sel["warnings"][-1] and sel["picks"] == []
    assert sum(c["status"] == "inconsistent" for c in sel["candidates"]) == 9
    j2, *_ = make_job(tmp_path / "b", mode="manual", picks=[0], no_mask=(1,))
    with pytest.raises(SystemExit):
        V.select(j2)
    assert "only 1 of your frames" in json.loads((j2 / "selection.json").read_text())["warnings"][-1]


def test_build_run_applies_the_metric_scale_once(tmp_path):
    names = ["f00000", "f00010", "f00020", "f00030"]
    E = [look_at(C) for C in circle(4)]
    P, masks = zip(*(sphere(e) for e in E))
    make_pool(tmp_path / "pool", names, E, masks, P)
    V.build_run(tmp_path / "pool", ["f00020", "f00000", "f00030"], tmp_path / "raw")
    for _ in range(2):  # a second build does not compound: it starts from the raw pool again
        V.build_run(tmp_path / "pool", ["f00020", "f00000", "f00030"], tmp_path / "run", metric=1.25)
    raw, run = np.load(tmp_path / "raw" / "da3_output.npz"), np.load(tmp_path / "run" / "da3_output.npz")
    assert "metric_scale" not in raw.files and float(run["metric_scale"]) == 1.25
    for k in ("depth", "pointmaps", "pointmaps_sam3d"):
        assert np.allclose(run[k], 1.25 * raw[k])
    assert np.allclose(run["extrinsics"][..., 3], 1.25 * raw["extrinsics"][..., 3])
    assert np.array_equal(run["intrinsics"], raw["intrinsics"]) and np.array_equal(run["conf"], raw["conf"])
    assert json.loads((tmp_path / "raw" / "prep.json").read_text())["metric_scale"] is None
    assert json.loads((tmp_path / "run" / "prep.json").read_text())["metric_scale"] == 1.25
    with np.load(tmp_path / "pool" / "da3_output.npz") as pool:
        assert "metric_scale" not in pool.files and np.allclose(pool["depth"][[2, 0, 3]], raw["depth"])
    for a, b in zip(M.cameras(tmp_path / "raw"), M.cameras(tmp_path / "run")):
        Ta, Tb = np.array(a["w2c"]), np.array(b["w2c"])
        assert np.allclose(Ta[:3, :3], Tb[:3, :3], atol=1e-6) and np.allclose(1.25 * Ta[:3, 3], Tb[:3, 3], atol=1e-5)
        assert np.allclose(a["K"], b["K"])
    assert np.abs(np.array(M.cameras(tmp_path / "run")[1]["w2c"])[:3, 3]).max() > 0.5  # a real baseline was scaled


def write_lidar(j, **rec):
    (j / "pool" / "lidar.json").write_text(json.dumps({"version": 1, "ok": True, "s": 1.25, "reason": None,
                                                       "combine": {"views_used": 20, "spread_rel": 0.012}, **rec}))


def test_metric_scale_reads_the_lidar_record(tmp_path):
    j = tmp_path / "job"
    (j / "pool").mkdir(parents=True)
    assert V.metric_scale(j, {"lidar": None}) is None and V.metric_scale(j, {}) is None
    on = {"lidar": {"kind": "rosbag", "topic": "/livox/lidar"}}
    with pytest.raises(SystemExit, match="pool/lidar.json is missing"):
        V.metric_scale(j, on)
    write_lidar(j)
    assert V.metric_scale(j, on) == 1.25 and V.metric_scale(j, {"lidar": False}) is None
    write_lidar(j, ok=False, s=3.0, reason="the LiDAR scale 3.00 is far from Depth Anything 3's metric estimate")
    with pytest.raises(SystemExit, match="LiDAR metric scale failed: the LiDAR scale 3.00 is far"):
        V.metric_scale(j, on)
    for bad in (None, float("nan"), -1.0, 0, True, "1.25"):
        write_lidar(j, s=bad)
        with pytest.raises(SystemExit, match="no usable scale"):
            V.metric_scale(j, on)
    for bad in ("false", 1, "true", None):  # only ok true passes
        write_lidar(j, ok=bad)
        with pytest.raises(SystemExit, match="LiDAR metric scale failed"):
            V.metric_scale(j, on)
    (j / "pool" / "lidar.json").write_text("{")
    with pytest.raises(SystemExit, match="not valid JSON"):
        V.metric_scale(j, on)
    for text in ("null", "[]", "1.25"):  # valid JSON, but not a record
        (j / "pool" / "lidar.json").write_text(text)
        with pytest.raises(SystemExit, match="not a LiDAR record"):
            V.metric_scale(j, on)


def test_select_applies_the_lidar_scale(tmp_path):
    j, *_ = make_job(tmp_path, lidar={"kind": "rosbag", "topic": "/livox/lidar"})
    write_lidar(j)
    (j / "lidar" / "cloud").mkdir(parents=True)  # the lidar stage's points of every pool view
    for p in (j / "pool" / "masks").glob("*.png"):
        (j / "lidar" / "cloud" / f"{p.stem}.glb").write_bytes(p.stem.encode())
    sel = V.select(j)
    assert sel["ok"] and sel["metric_scale"] == {"s": 1.25, "views_used": 20, "spread_rel": 0.012}
    assert json.loads((j / "selection.json").read_text())["metric_scale"] == sel["metric_scale"]
    assert json.loads((j / "run" / "prep.json").read_text())["metric_scale"] == 1.25
    picks = [p["name"] for p in sel["picks"]]
    assert [(j / "run" / "lidar" / f"{i}.glb").read_bytes().decode() for i in range(len(picks))] == picks
    assert len(list((j / "run" / "lidar").iterdir())) == len(picks)
    with np.load(j / "pool" / "da3_output.npz") as pool, np.load(j / "run" / "da3_output.npz") as run:
        stem = [Path(str(f)).stem for f in pool["image_files"]]
        assert np.allclose(run["depth"], 1.25 * pool["depth"][[stem.index(n) for n in picks]])
        assert float(run["metric_scale"]) == 1.25

    write_lidar(j, ok=False, reason="LiDAR points cover the object in only 1 of 24 views")
    with pytest.raises(SystemExit, match="only 1 of 24 views"):
        V.select(j)
    assert (j / "run" / "da3_output.npz").is_file()  # nothing was removed
    assert json.loads((j / "selection.json").read_text())["metric_scale"]["s"] == 1.25

    j2, *_ = make_job(tmp_path / "b")
    (j2 / "lidar").mkdir()  # an earlier build with LiDAR left its points; LiDAR is off now
    (j / "lidar" / "cloud").rename(j2 / "lidar" / "cloud")
    assert V.select(j2)["metric_scale"] is None
    with np.load(j2 / "run" / "da3_output.npz") as run:
        assert "metric_scale" not in run.files
    assert not (j2 / "run" / "lidar").exists()
