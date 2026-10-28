import json

import numpy as np
import pytest
from PIL import Image

from gator.stage_a import mvsam3d as M


def scene(tmp_path, names=("10", "2", "1"), size=(400, 300)):
    (tmp_path / "scene" / "images").mkdir(parents=True)
    (tmp_path / "scene" / "toy").mkdir()
    for i, n in enumerate(names):
        Image.new("RGB", size, (i * 40, 100, 200)).save(tmp_path / "scene" / "images" / f"{n}.png")
        m = np.zeros((size[1], size[0], 4), np.uint8)
        m[100:200, 150:250, 3] = 255
        Image.fromarray(m, "RGBA").save(tmp_path / "scene" / "toy" / f"{n}_mask.png")
    return tmp_path / "scene"


def test_prep_orders_views_and_writes_both_mask_forms(tmp_path):
    out = tmp_path / "run"
    M.prep(scene(tmp_path), "toy", out, max_side=200)
    assert M.view_names(out) == ["1", "2", "10"]
    rgba = np.array(Image.open(out / "input" / "object" / "1.png"))
    plain = np.array(Image.open(out / "masks" / "1.png"))
    assert rgba.shape == (150, 200, 4) and plain.shape == (150, 200)
    assert (rgba[..., 3] > 0).sum() == (plain > 0).sum() == 50 * 50
    assert json.loads((out / "prep.json").read_text())["views"][0]["mask_px"] == 2500


def test_prep_gives_mixed_sizes_one_size(tmp_path):
    (tmp_path / "scene" / "images").mkdir(parents=True)
    (tmp_path / "scene" / "toy").mkdir()
    for n, (w, h) in {"1": (400, 300), "2": (300, 240), "3": (500, 400)}.items():
        Image.new("RGB", (w, h), (90, 100, 200)).save(tmp_path / "scene" / "images" / f"{n}.png")
        m = np.zeros((h, w), np.uint8)
        m[h // 4:3 * h // 4, w // 4:3 * w // 4] = 255
        Image.fromarray(m).save(tmp_path / "scene" / "toy" / f"{n}.png")
    out = tmp_path / "run"
    M.prep(tmp_path / "scene", "toy", out, max_side=1024)
    info = json.loads((out / "prep.json").read_text())
    assert info["aspect"] == 1.25 and {tuple(v["size"]) for v in info["views"]} == {(375, 300)}
    assert info["views"][0]["crop"] == [12, 0, 387, 300]  # 400x300 center-cropped to 375x300
    for n in ("1", "2", "3"):
        assert Image.open(out / "input" / "images" / f"{n}.png").size == (375, 300)
        assert np.array(Image.open(out / "masks" / f"{n}.png")).shape == (300, 375)


def test_cameras_are_relative_to_view_0_at_image_size(tmp_path):
    out = tmp_path / "run"
    M.prep(scene(tmp_path, names=("1", "2")), "toy", out, max_side=400)
    rng = np.random.default_rng(0)

    def rigid():
        q = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        q *= np.sign(np.linalg.det(q))
        return np.hstack([q, rng.normal(size=(3, 1))])

    E = np.stack([rigid(), rigid()]).astype(np.float32)
    K = np.array([[[100.0, 0, 49.5], [0, 100.0, 37.0], [0, 0, 1]]] * 2, np.float32)
    files = [str(out / "input" / "images" / f"{n}.png") for n in ("2", "1")]  # DA3 order differs from ours
    np.savez(out / "da3_output.npz", depth=np.ones((2, 75, 100), np.float32), extrinsics=E[::-1], intrinsics=K,
             image_files=np.array(files))
    cams = M.cameras(out)
    assert [c["name"] for c in cams] == ["1", "2"] and (cams[0]["width"], cams[0]["height"]) == (400, 300)
    assert np.allclose(cams[0]["w2c"], np.eye(4), atol=1e-6)
    E4 = [np.vstack([e, [0, 0, 0, 1]]) for e in E]
    assert np.allclose(cams[1]["w2c"], E4[1] @ np.linalg.inv(E4[0]), atol=1e-5)
    # A world point seen by DA3 in view 1 projects to the same place through our camera, scaled 4x.
    X = np.array([0.3, -0.2, 3.0, 1.0])
    p_da3 = K[0] @ (E4[1] @ X)[:3]
    p_ours = np.array(cams[1]["K"]) @ (np.array(cams[1]["w2c"]) @ (E4[0] @ X))[:3]  # X in view 0's camera, then ours
    u_da3, u_ours = p_da3[:2] / p_da3[2], p_ours[:2] / p_ours[2]
    assert np.allclose(u_ours, (u_da3 + 0.5) * 4 - 0.5, atol=1e-3)


def look_at_w2c(eye):
    """test_parts_vote.look_at toward the origin."""
    from test_parts_vote import look_at  # it imports trimesh, which the callers need anyway

    return look_at(np.asarray(eye, dtype=float), np.zeros(3))


def box_views():
    import trimesh

    from gator.stage_a.place import silhouette

    box = trimesh.creation.box(extents=(0.2, 0.1, 0.15))
    x_true, faces = box.vertices + [0.02, -0.01, 0.0], box.faces
    K = np.array([[150.0, 0, 79.5], [0, 150.0, 59.5], [0, 0, 1]])
    cams, masks = [], []
    for eye in [(0, 0, -0.6), (0.6, 0, -0.1), (-0.4, -0.2, 0.45)]:
        T = look_at_w2c(eye)
        cams.append({"K": K.tolist(), "w2c": T.tolist(), "width": 160, "height": 120})
        masks.append(silhouette(x_true @ T[:3, :3].T + T[:3, 3], faces, K, (160, 120), (160, 120)))
    return x_true, faces, cams, masks


@pytest.mark.parametrize("wrong", [1.15, 0.8, 1.6])  # shrink, grow, and past the first range (0.7 to 1.3)
def test_fit_scale_recovers_a_wrong_size(wrong):
    x_true, faces, cams, masks = box_views()
    c = (x_true.min(0) + x_true.max(0)) / 2
    k, center, before, after, at_limit = M.fit_scale(c + wrong * (x_true - c), faces, cams, masks)
    assert np.allclose(center, c) and not at_limit
    assert abs(k * wrong - 1) < 0.02
    assert after.mean() > 0.95 > before.mean()


def test_fit_scale_without_overlap_keeps_scale_1():
    x_true, faces, cams, masks = box_views()
    k, _, before, after, _ = M.fit_scale(x_true, faces, cams, [np.zeros_like(m) for m in masks], lo=0.5, hi=0.95)
    assert k == 1.0 and before.max() == 0 and after.max() == 0


def test_scaled_pose_reproduces_the_fitted_mesh():
    from gator.stage_a.place import canonical_glb_to_w

    rng = np.random.default_rng(1)
    pose = {"rotation_wxyz": rng.normal(size=4).tolist(), "scale": [0.3, 0.3, 0.3], "translation": [0.01, 0.02, 0.5]}
    V = rng.normal(size=(50, 3))
    x = canonical_glb_to_w(V, pose)
    c = (x.min(0) + x.max(0)) / 2
    assert np.allclose(canonical_glb_to_w(V, M.scaled_pose(pose, 0.9, c)), c + 0.9 * (x - c))


def test_load_mask_reads_alpha_and_opaque_exports(tmp_path):
    from gator.stage_a.sam3d import load_mask

    obj = np.zeros((40, 50), bool)
    obj[10:30, 15:35] = True
    gray = np.where(obj, 255, 0).astype(np.uint8)
    edge = gray.copy()
    edge[9, 15:35] = 160  # an anti-aliased edge row, more than half covered
    edge[30, 15:35] = 60  # less than half covered
    cases = {
        "alpha": np.dstack([np.full((40, 50, 3), 255, np.uint8), gray]),  # RGBA, the object in alpha
        "opaque_rgba": np.dstack([edge, edge, edge, np.full((40, 50), 255, np.uint8)]),  # SAM 3 demo export
        "rgb": np.dstack([gray, gray, gray]),
        "gray": gray,
        "binary": obj.astype(np.uint8),
    }
    for name, a in cases.items():
        Image.fromarray(a).save(tmp_path / f"{name}.png")
        m = load_mask(tmp_path / f"{name}.png")
        want = obj.copy()
        if name == "opaque_rgba":
            want[9, 15:35] = True
        assert m.shape == (40, 50) and (m == want).all(), name


def test_matrix_to_quat_inverts_quat_to_matrix():
    from gator.stage_a.place import quat_wxyz_to_matrix

    rng = np.random.default_rng(2)
    for q in [rng.normal(size=4) for _ in range(20)] + [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]:
        R = quat_wxyz_to_matrix(q)
        assert np.allclose(quat_wxyz_to_matrix(M.matrix_to_quat_wxyz(R)), R, atol=1e-9)


def test_similar_pose_reproduces_the_moved_mesh():
    from gator.stage_a.place import canonical_glb_to_w

    rng = np.random.default_rng(3)
    pose = {"rotation_wxyz": rng.normal(size=4).tolist(), "scale": [0.3, 0.3, 0.3], "translation": [0.01, 0.02, 0.5]}
    V = rng.normal(size=(50, 3))
    x = canonical_glb_to_w(V, pose)
    c, R, d = (x.min(0) + x.max(0)) / 2, M.rotation_from_vector([4.0, -7.0, 2.0]), np.array([0.03, -0.01, 0.05])
    assert np.allclose(canonical_glb_to_w(V, M.similar_pose(pose, 0.85, R, d, c)), c + d + 0.85 * (x - c) @ R.T)


def fit_both(x, faces, cams, masks):
    k, cs, _, _, _ = M.fit_scale(x, faces, cams, masks)
    x = cs + k * (x - cs)
    (k2, R, d, rv), c2, start, best, evals, at_bound = M.fit_pose(x, faces, cams, masks)
    return c2 + d + k2 * (x - c2) @ R.T, R, rv, start, best, evals, at_bound


def test_fit_pose_adds_no_rotation_to_a_shifted_mesh():
    x_true, faces, cams, masks = box_views()
    c = (x_true.min(0) + x_true.max(0)) / 2
    x_fit, R, rv, start, best, evals, at_bound = fit_both(c + np.array([0.03, -0.02, 0.04]) + 1.1 * (x_true - c), faces, cams, masks)
    diag = np.linalg.norm(np.ptp(x_true, 0))
    assert np.abs(rv).max() < 2.0 and np.abs(x_fit - x_true).max() < 0.03 * diag
    assert best.mean() > 0.95 > start.mean() and not at_bound and evals <= 400


def test_fit_pose_recovers_a_rotated_mesh():
    x_true, faces, cams, masks = box_views()
    c = (x_true.min(0) + x_true.max(0)) / 2
    R0 = M.rotation_from_vector([4.0, -5.0, 0.0])
    x_fit, R, rv, start, best, evals, at_bound = fit_both(c + np.array([-0.02, 0.03, -0.03]) + 0.9 * (x_true - c) @ R0.T, faces, cams, masks)
    diag = np.linalg.norm(np.ptp(x_true, 0))
    angle = np.degrees(np.arccos(np.clip((np.trace(R @ R0) - 1) / 2, -1, 1)))  # what is left of R0 after the fit
    # Three 160 x 120 masks of a box cannot pin a rotation below a few degrees: the true pose scores only
    # about 0.965, and a box turned 5 degrees about the vertical can score as well. Check to that precision.
    truth = M.view_ious(x_true, faces, cams, masks, 384).mean()
    assert angle < 4.0 and np.abs(x_fit - x_true).max() < 0.05 * diag
    assert best.mean() >= truth - 0.015 and start.mean() < 0.6 and not at_bound and evals <= 400


def valley_views(spread, extents=(0.2, 0.1, 0.15), dist=0.6, f=300.0, wh=(320, 240)):
    """The box at the origin, view 0's camera dist in front of it, and two more views at +-spread degrees
    about the box, all dist away, with focal length f and images of wh."""
    import trimesh

    from gator.stage_a.place import silhouette

    box = trimesh.creation.box(extents=extents)
    K = np.array([[f, 0, wh[0] / 2 - 0.5], [0, f, wh[1] / 2 - 0.5], [0, 0, 1]])
    cams, masks = [], []
    for a in np.radians([0.0, -spread, spread]):
        T = look_at_w2c(dist * np.array([np.sin(a), 0.0, -np.cos(a)]))
        cams.append({"K": K.tolist(), "w2c": T.tolist(), "width": wh[0], "height": wh[1]})
        masks.append(silhouette(box.vertices @ T[:3, :3].T + T[:3, 3], box.faces, K, wh, wh))
    return box.vertices, box.faces, cams, masks


@pytest.mark.parametrize("spread", [10, 20])
def test_fit_pose_recovers_a_mesh_too_large_and_too_far(spread):
    # Scaled 5% about view 0's camera, the mesh is 5% too large and 5% too far. View 0's silhouette is
    # unchanged and the other narrow views change little, so size and distance must move together.
    x_true, faces, cams, masks = valley_views(spread)
    o = np.array([0.0, 0.0, -0.6])  # view 0's camera
    x_fit, R, rv, start, best, evals, at_bound = fit_both(o + 1.05 * (x_true - o), faces, cams, masks)
    assert np.abs(np.ptp(x_fit, 0) / np.ptp(x_true, 0) - 1).max() < 0.01
    assert np.abs(x_fit.min(0) + x_fit.max(0)).max() / 2 < 0.01  # the box center within 1 cm of the origin
    assert not at_bound and evals <= 400


def test_fit_pose_keeps_a_far_small_mesh_off_the_bounds():
    # A 6 cm box 1.2 m away, about 15 box diagonals from view 0's camera, 0.5% large about that camera. A scale
    # about the camera by the first step would move the shift past its bound; clipped, it would no longer keep
    # view 0's silhouette, and the fit would end on the bound (mean IoU 0.966, 2% small). It is not taken.
    x_true, faces, cams, masks = valley_views(5, (0.06, 0.03, 0.045), 1.2, 600.0, (640, 480))
    o = np.array([0.0, 0.0, -1.2])
    x0 = o + 1.005 * (x_true - o)
    k, cs, before, after, _ = M.fit_scale(x0, faces, cams, masks)
    x = cs + k * (x0 - cs) if after.mean() - before.mean() >= 0.005 else x0  # as place() applies it
    (k2, R, d, rv), c2, start, best, evals, at_bound = M.fit_pose(x, faces, cams, masks)
    x_fit = c2 + d + k2 * (x - c2) @ R.T
    assert best.mean() > 0.99 and not at_bound and evals <= 400
    assert np.abs(np.ptp(x_fit, 0) / np.ptp(x_true, 0) - 1).max() < 0.01


def test_place_rejects_an_unknown_fit(tmp_path):
    import pytest as _pytest

    with _pytest.raises(ValueError):
        M.place(tmp_path, fit="Pose")


def test_place_records_the_metric_scale(tmp_path):
    import trimesh

    from gator.stage_a.place import canonical_glb_to_w, silhouette

    out = tmp_path / "run"
    for d in ("input/images", "masks"):
        (out / d).mkdir(parents=True)
    box = trimesh.creation.box(extents=(0.2, 0.1, 0.15))
    box.export(out / "canonical.glb")
    pose = {"rotation_wxyz": [1.0, 0.0, 0.0, 0.0], "translation": [0.0, 0.0, 0.6], "scale": [1.0, 1.0, 1.0]}
    (out / "pose.json").write_text(json.dumps(pose))
    mesh = trimesh.load(out / "canonical.glb", force="mesh", process=False)
    x_w = canonical_glb_to_w(mesh.vertices, pose)  # the box 0.6 m in front of view 0
    K = np.array([[150.0, 0, 79.5], [0, 150.0, 59.5], [0, 0, 1]])
    shift = np.eye(4)
    shift[2, 3] = -0.6
    E = [np.eye(4), look_at_w2c([0.3, 0.0, -0.4]) @ shift]  # view 1 looks at the box from the side
    for i, T in enumerate(E):
        m = silhouette(x_w @ T[:3, :3].T + T[:3, 3], mesh.faces, K, (160, 120), (160, 120))
        Image.fromarray((m * 255).astype(np.uint8)).save(out / "masks" / f"{i}.png")
        Image.new("RGB", (160, 120), (90, 100, 110)).save(out / "input" / "images" / f"{i}.png")
    v, u = np.mgrid[0:120, 0:160]
    P = np.stack([(u - 79.5) / 150 * 0.6, (v - 59.5) / 150 * 0.6, np.full(u.shape, 0.6)], -1)
    arrays = dict(depth=np.full((2, 120, 160), 0.6, np.float32), pointmaps=np.stack([P, P]).astype(np.float32),
                  extrinsics=np.stack([T[:3] for T in E]).astype(np.float32), intrinsics=np.stack([K, K]).astype(np.float32),
                  image_files=np.array([str(out / "input" / "images" / f"{i}.png") for i in range(2)]))
    np.savez(out / "da3_output.npz", **arrays)
    M.place(out, samples=2000, fit="none")
    p = json.loads((out / "placement.json").read_text())
    assert p["metric_scale"] is None and p["units"] == "DA3 units (meters)" and p["check"]["mean_mask_iou"] > 0.9
    assert p["frame_W"].endswith("in DA3 units (meters)")
    np.savez(out / "da3_output.npz", **arrays, metric_scale=np.float64(1.25))  # select applied a LiDAR scale
    M.place(out, samples=2000, fit="none")
    p = json.loads((out / "placement.json").read_text())
    assert p["metric_scale"] == 1.25 and p["units"] == "meters (LiDAR scale applied in select)"
    assert p["frame_W"].endswith("in meters (LiDAR scale applied in select)")  # the same units in both places
