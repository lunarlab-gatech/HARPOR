import json

import numpy as np
import pytest
import trimesh
from PIL import Image

from gator import frames
from gator.parts import render
from gator.parts import vote as V

W, H = 200, 150
K = np.array([[220.0, 0, 99.5], [0, 220.0, 74.5], [0, 0, 1]])
CENTER = np.array([0.0, 0.0, 2.0])
EXTENTS = (0.4, 0.3, 0.3)  # W: x right, y down, z away from view 0


def look_at(eye, target=CENTER):
    """OpenCV world-to-camera 4x4 from eye toward target, image down along +y of W."""
    z = (target - eye) / np.linalg.norm(target - eye)
    x = np.cross([0.0, 1.0, 0.0], z)
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, np.cross(z, x), z])
    T[:3, 3] = -T[:3, :3] @ eye
    return T


def eye(az, el, r=1.3):
    a, e = np.radians(az), np.radians(el)
    return CENTER + r * np.array([np.sin(a) * np.cos(e), -np.sin(e), -np.cos(a) * np.cos(e)])


def make_run(root, eyes, proposal, instances):
    """A run folder: a textured box (8 x 8 quads per side) as posed.glb, cameras.json, masks/,
    input/images/, proposal.json and parts/sam3 with the instances. instances(view, ids, region) gives
    {prompt: [mask, ...]} for one view; region(name) is the face mask of a named face set."""
    run = root / "run"
    for d in ("masks", "input/images", "parts/sam3"):
        (run / d).mkdir(parents=True, exist_ok=True)
    box = trimesh.creation.box(extents=EXTENTS)
    for _ in range(3):
        box = box.subdivide()
    uv = (box.vertices[:, :2] - box.vertices[:, :2].min(0)) / np.ptp(box.vertices[:, :2], 0)
    tex = Image.fromarray(np.full((8, 8, 3), (180, 120, 70), np.uint8))
    visual = trimesh.visual.TextureVisuals(uv=uv, material=trimesh.visual.material.PBRMaterial(baseColorTexture=tex))
    trimesh.Trimesh(frames.w_to_gltf(box.vertices + CENTER), box.faces, visual=visual,
                    process=False).export(run / "posed.glb")
    x_w, faces = V.load_mesh(run / "posed.glb")
    c = x_w[faces].mean(1) - CENTER
    regions = {"lid": c[:, 1] < -0.075 + 1e-9,
               "front": np.isclose(c[:, 2], -0.15),
               "top_drawer": np.isclose(c[:, 2], -0.15) & (np.abs(c[:, 0]) < 0.1) & (c[:, 1] > -0.1125) & (c[:, 1] < -0.0375),
               "bottom_drawer": np.isclose(c[:, 2], -0.15) & (np.abs(c[:, 0]) < 0.1) & (c[:, 1] > 0.0375) & (c[:, 1] < 0.1125),
               "low_door": np.isclose(c[:, 2], -0.15) & (np.abs(c[:, 0]) < 0.1) & (c[:, 1] > 0.0375),
               "bottom": np.isclose(c[:, 1], 0.15),
               "stand": c[:, 1] > 0.075 + 1e-9}
    regions["upper"] = ~regions["stand"]
    knob = np.flatnonzero(np.isclose(c[:, 2], -0.15) & (c[:, 0] > 0.15) & (c[:, 1] > -0.0375) & (c[:, 1] < 0))[:1]
    regions["knob"] = np.isin(np.arange(len(faces)), knob)
    cams, records = [], {}
    for k, e in enumerate(eyes):
        name = str(k)
        T = look_at(e)
        ids = render.face_ids(x_w, faces, K, T, (W, H))
        Image.fromarray(np.where(ids >= 0, 255, 0).astype(np.uint8)).save(run / "masks" / f"{name}.png")
        Image.fromarray(np.full((H, W, 3), (90 + k, 100, 120), np.uint8)).save(run / "input" / "images" / f"{name}.png")
        cams.append({"name": name, "K": K.tolist(), "w2c": T.tolist(), "width": W, "height": H,
                     "image": str(run / "input" / "images" / f"{name}.png"), "mask": "/nowhere/mask.png"})
        for prompt, masks in instances(name, ids, lambda r: np.isin(ids, np.flatnonzero(regions[r]))).items():
            rec = records.setdefault(prompt, {"name": prompt, "slug": prompt, "instances": 0, "views": {}})
            rec["views"][name] = []
            (run / "parts" / "sam3" / prompt).mkdir(exist_ok=True)
            for i, m in enumerate(masks):
                f = f"{prompt}/{name}_{i}.png"
                Image.fromarray(np.where(m, 255, 0).astype(np.uint8)).save(run / "parts" / "sam3" / f)
                rec["views"][name].append({"index": i, "score": 0.9 - 0.1 * i, "box_xyxy": [0, 0, 1, 1],
                                           "mask_px": int(m.sum()), "file": f})
                rec["instances"] += 1
    (run / "cameras.json").write_text(json.dumps(cams))
    (run / "proposal.json").write_text(json.dumps(proposal))
    (run / "parts" / "sam3" / "part_masks.json").write_text(json.dumps(
        {"model": "fake", "views": [str(k) for k in range(len(eyes))], "parts": list(records.values()),
         "not_found": [p for p, r in records.items() if not r["instances"]]}))
    return run, regions


def part_list(base, *parts):
    return {"source": "user", "noun": "box", "base": base,
            "parts": [{"name": base, "parent": None, "joint": "fixed", "count": 1, "axis_hint": None},
                      *[{"name": n, "parent": base, "joint": j, "count": c, "axis_hint": None} for n, j, c in parts]]}


def test_a_lid_seen_in_every_view_is_cut_off_the_box(tmp_path):
    def instances(view, ids, region):
        found = [region("lid")]
        if view == "0":
            found.append(ids >= 0)  # SAM 3 took the whole box for the lid once
        return {"lid": found}

    run, regions = make_run(tmp_path, [eye(az, 35) for az in range(0, 360, 60)],
                            part_list("box body", ("lid", "revolute", 1)), instances)
    rec = V.run(run)
    data = np.load(run / "parts" / "faces.npz")
    assert rec["names"] == ["box_body", "lid"] == data["names"].tolist() and rec["base"] == "box_body"
    assert (data["part"] == regions["lid"]).all()
    assert data["centroids"].shape == (768, 3) and data["score"].shape == (768,)
    assert np.isclose(data["score"][regions["lid"]], 1).all()
    lid = next(c for c in rec["clusters"] if c["name"] == "lid")
    assert lid["status"] == "found" and lid["views"] == [str(k) for k in range(6)] and lid["confidence"] > 0.9
    assert lid["instances"]["0"] == [0] and set(lid["visibility"]) == {str(k) for k in range(6)}
    assert rec["dropped_instances"] == [{"prompt": "lid", "view": "0", "index": 1, "why": "covers 100% of the object"}]
    assert rec["counts"]["unseen"] > 0  # the bottom: no camera sees it, so it takes its neighbours' part
    assert rec["clusters"][0]["role"] == "base" and rec["warnings"] == []


def test_two_drawers_listed_in_either_order_and_a_tiny_knob(tmp_path):
    def instances(view, ids, region):
        drawers = [region("top_drawer"), region("bottom_drawer")]
        out = {"drawer": drawers if int(view) % 2 == 0 else drawers[::-1]}
        if view == "0":
            out["knob"] = [region("knob")]
        return out

    proposal = part_list("cabinet body", ("drawer", "prismatic", 2), ("knob", "revolute", 1))
    run, regions = make_run(tmp_path, [eye(az, 15) for az in (-40, -15, 15, 40)], proposal, instances)
    rec = V.run(run)
    data = np.load(run / "parts" / "faces.npz")
    assert rec["names"] == ["cabinet_body", "drawer_1", "drawer_2"]
    part = data["part"]
    assert (part[regions["top_drawer"]] == 1).all() and (part[regions["bottom_drawer"]] == 2).all()
    assert (part[~(regions["top_drawer"] | regions["bottom_drawer"])] == 0).all()
    status = {(c["prompt"], c["name"]): c["status"] for c in rec["clusters"]}
    assert status == {("cabinet body", "cabinet_body"): "found", ("drawer", "drawer_1"): "found",
                      ("drawer", "drawer_2"): "found", ("knob", None): "dropped_small"}
    d1 = next(c for c in rec["clusters"] if c["name"] == "drawer_1")
    assert d1["instances"] == {"0": [0], "1": [1], "2": [0], "3": [1]}


def test_an_unseen_bottom_stays_with_the_base(tmp_path):
    # A door down to the bottom edge, filmed from above: no camera sees the bottom, which touches the door.
    run, regions = make_run(tmp_path, [eye(az, 35) for az in (-50, -20, 20, 50)],
                            part_list("box body", ("door", "revolute", 1)),
                            lambda view, ids, region: {"door": [region("low_door")]})
    rec = V.run(run)
    part = np.load(run / "parts" / "faces.npz")["part"]
    assert rec["counts"]["unseen"] >= regions["bottom"].sum()
    assert (part[regions["low_door"]] == 1).all() and (part[~regions["low_door"]] == 0).all()


def test_a_part_that_sam3_never_found(tmp_path):
    proposal = part_list("box body", ("lid", "revolute", 1), ("door", "revolute", 2))
    run, regions = make_run(tmp_path, [eye(az, 35) for az in (0, 90, 180, 270)], proposal,
                            lambda view, ids, region: {"lid": [region("lid")], "door": []})
    rec = V.run(run)
    assert rec["names"] == ["box_body", "lid"]
    assert {"name": None, "role": "part", "prompt": "door", "slug": "door", "status": "not_found",
            "faces": 0} in rec["clusters"]
    assert (np.load(run / "parts" / "faces.npz")["part"] == regions["lid"]).all()


def test_cluster_never_joins_two_instances_of_one_view():
    area = np.ones(6)
    seen = np.ones((2, 6), bool)

    def inst(prompt, view, S):
        return {"prompt": prompt, "view": view, "v": int(view), "S": np.array(S)}

    insts = [inst("a", "0", [0, 1, 2]), inst("a", "0", [2, 3]), inst("a", "1", [0, 1, 2]), inst("b", "1", [0, 1, 2]),
             inst("a", "1", [4, 5])]
    assert V.cluster(insts, seen, area) == [[0, 2], [1], [3], [4]]
    assert V.cluster(insts, seen, area, min_iou=0.2) == [[0, 2], [1], [3], [4]]  # 1 and 2 overlap, but 2 is taken


def test_propagate_and_cleanup_on_a_strip():
    # Faces 0..9 in a row: 5 is unseen; then 8 is a small second piece of part 1.
    adj = np.array([[i, i + 1] for i in range(9)])
    label, n = V.propagate(np.array([1, 1, 1, 1, 1, -1, 0, 0, 0, 0]), adj, 2)
    assert n == 1 and label[5] in (0, 1)
    allowed = np.ones((10, 2), bool)
    allowed[5:, 1] = False  # part 1 may not reach past face 4
    label, n = V.propagate(np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, 0]), adj, 2, allowed)
    assert n == 4 and label.tolist() == [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    area = np.ones(10)
    label, moved = V.cleanup(np.array([1, 1, 1, 1, 1, 0, 0, 0, 1, 0]), adj, area, 2, piece=0.2)
    assert moved == 1 and label.tolist() == [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]


def test_min_cut_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = 7
        c0, c1 = rng.random(n), rng.random(n)
        pairs = np.array([(i, j) for i in range(n) for j in range(n) if i != j and rng.random() < 0.3]).reshape(-1, 2)
        pw = rng.random(len(pairs))

        def cost(x):
            return c0[~x].sum() + c1[x].sum() + pw[~x[pairs[:, 0]] & x[pairs[:, 1]]].sum()

        best = min(cost(np.array([(k >> i) & 1 for i in range(n)], bool)) for k in range(2 ** n))
        assert np.isclose(cost(V.min_cut(c0, c1, pairs[:, 0], pairs[:, 1], pw)), best, atol=1e-6)


def grid(n=20):
    """The pairs of a n x n grid of cells (cell r * n + c), 4-neighbours."""
    idx = np.arange(n * n).reshape(n, n)
    return np.concatenate([np.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()], 1),
                           np.stack([idx[:-1].ravel(), idx[1:].ravel()], 1)])


def test_regions_turn_speckle_into_whole_parts():
    # The left half votes weakly for the part, with every third cell weakly against it; the right half votes
    # against it, but for a firm 2 x 2 island.
    n = 20
    adj = grid(n)
    r, c = np.divmod(np.arange(n * n), n)
    p = np.where(c < 10, np.where(np.arange(n * n) % 3 == 0, 0.4, 0.6), 0.2)
    island = (r >= 5) & (r < 7) & (c >= 14) & (c < 16)
    p[island] = 1.0
    D = np.stack([p, 1 - p], 1)
    start = D.argmin(1)
    label, changed = V.regions(start, D, adj, np.full(len(adj), 0.3))
    assert (label == ((c < 10) | island)).all() and changed == int((start != label).sum()) > 0
    # A dearer border takes the island too; a face that may not take the part never does.
    assert (V.regions(start, D, adj, np.full(len(adj), 0.6))[0] == (c < 10)).all()
    allowed = np.ones((n * n, 2), bool)
    held = (c == 4) & (r < 3)
    allowed[held, 1] = False
    label, _ = V.regions(np.where(held, 0, start), D, adj, np.full(len(adj), 0.3), allowed)
    assert (label == ((c < 10) & ~held | island)).all()


@pytest.mark.parametrize("n,expect", [(1, ["lid"]), (3, ["lid_1", "lid_2", "lid_3"])])
def test_names_are_slugs_numbered_when_repeated(n, expect):
    clusters = [{"prompt": "Lid", "status": "found"} for _ in range(n)] + [{"prompt": "Lid", "status": "dropped_small"}]
    names = V.names_for("box", clusters, ["Lid"])
    assert [names[id(c)] for c in clusters[:n]] == expect and len(names) == n
    clash = [{"prompt": "box", "status": "found"}]
    assert list(V.names_for("box", clash, ["box"]).values()) == ["box_2"]


def test_a_wrong_instance_in_one_view_is_outvoted(tmp_path):
    # View 0 also takes the front of the box for a lid; views 1 and 5 see the front well and found the lid elsewhere.
    def instances(view, ids, region):
        return {"lid": [region("lid"), region("front")] if view == "0" else [region("lid")]}

    run, regions = make_run(tmp_path, [eye(az, 35) for az in range(0, 360, 60)],
                            part_list("box body", ("lid", "revolute", 1)), instances)
    rec = V.run(run)
    assert rec["names"] == ["box_body", "lid"] and rec["warnings"] == []
    assert (np.load(run / "parts" / "faces.npz")["part"] == regions["lid"]).all()
    (wrong,) = [c for c in rec["clusters"] if c["status"] == "unconfirmed"]
    assert wrong["name"] is None and wrong["views"] == ["0"] and wrong["instances"] == {"0": [1]}
    assert wrong["why"] == "only 1 of the 3 views that see its faces well and found 'lid' hold it"


def _top_left(region_mask):
    r, c = np.nonzero(region_mask)
    return region_mask & (np.arange(region_mask.shape[1])[None, :] < np.median(c))


@pytest.mark.parametrize("extra,why", [
    ("half", "a duplicate of instance 0"),  # a lower-scored second mask of the left half of the top drawer
    ("stack", "the union of instances [1, 2]"),  # one mask over both drawers, the best-scored one
])
def test_a_duplicate_or_a_union_in_one_view_is_dropped(tmp_path, extra, why):
    def instances(view, ids, region):
        drawers = [region("top_drawer"), region("bottom_drawer")]
        if extra == "half":
            return {"drawer": drawers + [_top_left(region("top_drawer"))] if view == "0" else drawers}
        both = drawers[0] | drawers[1]
        r, c = np.nonzero(both)
        stack = np.zeros_like(both)
        stack[r.min():r.max() + 1, c.min():c.max() + 1] = True
        return {"drawer": [stack & (ids >= 0)] + drawers}

    run, regions = make_run(tmp_path, [eye(az, 15) for az in (-40, -15, 15, 40)],
                            part_list("cabinet body", ("drawer", "prismatic", 2)), instances)
    rec = V.run(run)
    part = np.load(run / "parts" / "faces.npz")["part"]
    assert rec["names"] == ["cabinet_body", "drawer_1", "drawer_2"] and rec["warnings"] == []
    assert (part[regions["top_drawer"]] == 1).all() and (part[regions["bottom_drawer"]] == 2).all()
    assert (part[~(regions["top_drawer"] | regions["bottom_drawer"])] == 0).all()
    index = 2 if extra == "half" else 0
    views = ["0"] if extra == "half" else ["0", "1", "2", "3"]
    assert [d for d in rec["dropped_instances"] if d["why"] == why] == [
        {"prompt": "drawer", "view": v, "index": index, "why": why} for v in views]


def test_overlaps_finds_unions_before_duplicates():
    def box(r0, r1, c0, c1):
        m = np.zeros((10, 10), bool)
        m[r0:r1, c0:c1] = True
        return m

    top, low, stack = box(0, 4, 0, 10), box(6, 10, 0, 10), box(0, 10, 0, 10)
    # The stack holds two distinct drawers: a union, although it has the best score.
    assert V.overlaps([stack, top, low], [0.9, 0.8, 0.7], [0, 1, 2]) == ["the union of instances [1, 2]", None, None]
    # Two masks of one drawer are not two parts: the one with the lower score goes when half of it lies in
    # the other, whichever is larger.
    assert V.overlaps([top, box(0, 4, 0, 6), low], [0.9, 0.8, 0.7], [0, 1, 2]) == [
        None, "a duplicate of instance 0", None]
    assert V.overlaps([top, box(0, 4, 0, 6)], [0.6, 0.9], [0, 1]) == ["a duplicate of instance 1", None]
    assert V.overlaps([box(0, 4, 0, 6), box(0, 4, 4, 10)], [0.9, 0.8], [0, 1]) == [None, None]
    assert V.overlaps([], [], []) == []


@pytest.mark.parametrize("with_base", [True, False])
def test_a_part_that_is_most_of_the_object_needs_the_base_masks(tmp_path, with_base):
    # The top of a box above a low stand, like a chair's seat and backrest above its base: its masks cover
    # more than 80% of the object, so they pass only where a base mask shows that they hold none of the base.
    def instances(view, ids, region):
        out = {"top": [region("upper")] + ([ids >= 0] if view == "0" else [])}
        if with_base:
            out["stand"] = [region("stand")]
        return out

    run, regions = make_run(tmp_path, [eye(az, 35) for az in range(0, 360, 60)],
                            part_list("stand", ("top", "revolute", 1)), instances)
    rec = V.run(run)
    part = np.load(run / "parts" / "faces.npz")["part"]
    top = [d for d in rec["dropped_instances"] if d["prompt"] == "top"]
    if with_base:
        assert rec["names"] == ["stand", "top"] and (part == regions["upper"]).all()
        assert top == [{"prompt": "top", "view": "0", "index": 1, "why": "holds 100% of the base"}]
        assert rec["clusters"][0]["views"] == [str(k) for k in range(6)] and rec["clusters"][0]["support_share"] > 0
    else:
        assert rec["names"] == ["stand"] and not part.any()
        assert len(top) == 7 and all(d["why"].startswith("covers") for d in top)
        assert rec["clusters"][0]["views"] == [] and rec["clusters"][1]["status"] == "not_found"


def test_mesh_pixels_outside_the_object_mask_do_not_count(tmp_path):
    # The photo's object is smaller than the mesh: a rim of the mesh lies outside the grown mask in view 0.
    from scipy.ndimage import binary_erosion

    run, _ = make_run(tmp_path, [eye(0, 35)], part_list("box body", ("lid", "revolute", 1)),
                      lambda view, ids, region: {"lid": [region("lid")]})
    x_w, faces = V.load_mesh(run / "posed.glb")
    full = V.view_maps(x_w, faces, V.cameras(run))[0]
    small = binary_erosion(np.array(Image.open(run / "masks" / "0.png")) > 0, iterations=8)
    Image.fromarray(np.where(small, 255, 0).astype(np.uint8)).save(run / "masks" / "0.png")
    (v,) = V.view_maps(x_w, faces, V.cameras(run))
    assert (v["ids"][~v["grown"]] == -1).all() and (v["ids"][v["grown"]] == full["ids"][v["grown"]]).all()
    assert (v["vis"] == render.visible_pixels(v["ids"], len(faces))).all() and v["vis"].sum() < full["vis"].sum()


def test_no_part_is_named_world():
    clusters = [{"prompt": "World", "status": "found"}]
    assert list(V.names_for("globe_stand", clusters, ["World"]).values()) == ["world_2"]
