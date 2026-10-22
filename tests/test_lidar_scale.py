import json
import re

import numpy as np
import pytest
from PIL import Image

from gator.stage_a import lidar_scale as L
from gator.stage_a import mvsam3d as M


def rigid(rng):
    q = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    q *= np.sign(np.linalg.det(q))
    return np.hstack([q, rng.normal(size=(3, 1))])


def make_run(root, n=2, size=(400, 300), grid=(100, 75)):
    """A run folder as select writes it: input/images, masks, prep.json and da3_output.npz."""
    run = root / "run"
    for d in ("input/images", "masks"):
        (run / d).mkdir(parents=True)
    names = [str(k) for k in range(n)]
    for k in names:
        Image.new("RGB", size, (100, 120, 140)).save(run / "input" / "images" / f"{k}.png")
        Image.new("L", size, 255).save(run / "masks" / f"{k}.png")
    (run / "prep.json").write_text(json.dumps({"views": [{"name": k} for k in names]}))
    rng = np.random.default_rng(0)
    w, h = grid
    depth = rng.uniform(1, 3, (n, h, w)).astype(np.float32)
    P = np.stack([depth * 0.1, depth * 0.2, depth], -1)
    np.savez(run / "da3_output.npz", depth=depth, conf=np.ones_like(depth), pointmaps=P,
             pointmaps_sam3d=np.moveaxis(P, -1, 1), extrinsics=np.stack([rigid(rng) for _ in names]).astype(np.float32),
             intrinsics=np.array([[[100.0, 0, 49.5], [0, 100.0, 37.0], [0, 0, 1]]] * n, np.float32),
             image_files=np.array([str(run / "input" / "images" / f"{k}.png") for k in names]), process_res=504)
    return run


def test_apply_scale_scales_depth_points_and_translations_only(tmp_path):
    run = make_run(tmp_path)
    with np.load(run / "da3_output.npz") as da:
        raw = {k: da[k] for k in da.files}
    out = L.apply_scale(raw, 1.25)
    for k in ("depth", "pointmaps", "pointmaps_sam3d"):
        assert np.allclose(out[k], 1.25 * raw[k]) and out[k].dtype == raw[k].dtype
    assert np.allclose(out["extrinsics"][..., :3], raw["extrinsics"][..., :3])  # rotations stay
    assert np.allclose(out["extrinsics"][..., 3], 1.25 * raw["extrinsics"][..., 3])
    for k in ("conf", "intrinsics", "image_files", "process_res"):
        assert np.array_equal(out[k], raw[k])
    assert float(out["metric_scale"]) == 1.25 and "metric_scale" not in raw  # the input is not changed
    assert float(L.apply_scale(out, 2.0)["metric_scale"]) == 2.5  # scales applied in turn multiply
    for bad in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive"):
            L.apply_scale(raw, bad)


def test_scale_run_copies_the_inputs_and_scales_the_cameras(tmp_path):
    run = make_run(tmp_path, n=3)
    L.scale_run(run, 2.0, tmp_path / "s2")
    out = tmp_path / "s2"
    assert sorted(p.name for p in (out / "input" / "images").iterdir()) == ["0.png", "1.png", "2.png"]
    assert (out / "masks" / "2.png").is_file() and (out / "prep.json").is_file()
    with np.load(out / "da3_output.npz") as da:
        assert all(str(f).startswith(str(out)) for f in da["image_files"]) and float(da["metric_scale"]) == 2.0
    a, b = M.cameras(run), M.cameras(out)
    for ca, cb in zip(a, b):  # the same views and K; every camera's translation from view 0 doubles
        assert ca["name"] == cb["name"] and np.allclose(ca["K"], cb["K"])
        Ta, Tb = np.array(ca["w2c"]), np.array(cb["w2c"])
        assert np.allclose(Ta[:3, :3], Tb[:3, :3], atol=1e-5) and np.allclose(2 * Ta[:3, 3], Tb[:3, 3], atol=1e-4)
    with pytest.raises(SystemExit, match="exists"):
        L.scale_run(run, 2.0, out)


def placement(t, scale, extent, center, iou, q=(1.0, 0.0, 0.0, 0.0)):
    pose = {"rotation_wxyz": list(q), "translation": list(t), "scale": [scale] * 3}
    return {"pose": pose, "pose_mvsam3d": pose, "posed_extent_W": list(extent), "posed_center_W": list(center),
            "scale_fit": {"applied": False, "scale": 1.0}, "pose_fit": {"applied": False}, "check": {"mean_mask_iou": iou}}


def test_compare_runs_reports_the_ratios(tmp_path):
    for name, k, iou, q in (("a", 1.0, 0.80, (1.0, 0.0, 0.0, 0.0)), ("b", 2.0, 0.81, (np.cos(0.005), np.sin(0.005), 0.0, 0.0))):
        (tmp_path / name).mkdir()
        (tmp_path / name / "placement.json").write_text(json.dumps(
            placement([0.1 * k, -0.2 * k, 1.5 * k], 0.4 * k, [0.5 * k, 0.6 * k, 0.4 * k], [0.1 * k, 0.0, 1.5 * k], iou, q)))
    rec = L.compare_runs(tmp_path / "a", tmp_path / "b", 2.0)
    assert np.isclose(rec["mvsam3d"]["translation_ratio"], 2.0) and np.isclose(rec["mvsam3d"]["scale_ratio"], 2.0)
    assert np.allclose(rec["placed"]["extent_ratio"], 2.0) and np.isclose(rec["placed"]["center_ratio"], 2.0)
    assert np.isclose(rec["mvsam3d"]["rotation_deg"], np.degrees(0.01), atol=1e-6)
    assert all(rec["within_2pct"].values())
    rec = L.compare_runs(tmp_path / "a", tmp_path / "b", 1.9)  # 5% off: the ratios fail, the rotation and IoU pass
    assert rec["within_2pct"] == {"translation": False, "scale": False, "extent": False, "rotation_2deg": True,
                                  "iou_0.02": True}


# The estimate. The sphere scene of test_webapp_views: DA3 grid 64x48 with K. The source frames are
# 220x150; prep center-cropped them to CROP (160x120) and resized that to 128x96 (the pool images and
# masks), and DA3 worked at 64x48, so the source camera K_SRC is K through the crop.
from test_webapp_views import H, W, K, circle, look_at, make_pool, sphere  # noqa: E402

SRC = (220, 150)
CROP = (30, 15, 190, 135)
K_SRC = np.array([[150.0, 0, 109.5], [0, 150.0, 74.5], [0, 0, 1]])


def true_depth(E, Kc, wh, r=0.4):
    """Depth of the sphere scene (radius r at the origin, background at 10) at every pixel centre."""
    w, h = wh
    v, u = np.mgrid[0:h, 0:w]
    d = np.stack([(u - Kc[0, 2]) / Kc[0, 0], (v - Kc[1, 2]) / Kc[1, 1], np.ones((h, w))], -1)
    R, t = E[:, :3], E[:, 3]
    o, dw = -R.T @ t, d @ R
    a, b, c = (dw ** 2).sum(-1), 2 * dw @ o, o @ o - r * r
    disc = b * b - 4 * a * c
    return np.where(disc > 0, (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a), 10.0)


def lidar_pool(root, n=8, factors=None, step=(), empty_mask=(), no_samples=(), few=()):
    """A pool of n views around the sphere whose DA3 depth is 0.8 x the true depth (and divided by
    factors[i] more), and LiDAR samples of the true depth at every other source pixel. step: views whose
    DA3 depth is 0.7 x on the left half and 1.3 x on the right; empty_mask: views with an empty mask;
    no_samples: views without samples; few: views with only 10 samples on the sphere."""
    names = [f"f{10 * i:05d}" for i in range(n)]
    E = [look_at(C) for C in circle(n)]
    P, masks = map(list, zip(*(sphere(e) for e in E)))
    f = np.ones(n) if factors is None else np.asarray(factors, float)
    P = [0.8 * p / k for p, k in zip(P, f)]
    for i in step:
        P[i][:, :W // 2] *= 0.7
        P[i][:, W // 2:] *= 1.3
    masks = [np.zeros_like(m) if i in empty_mask else m for i, m in enumerate(masks)]
    pool = root / "pool"
    make_pool(pool, names, E, masks, P)
    for i, (nm, m) in enumerate(zip(names, masks)):  # the prep images are twice DA3's grid
        Image.new("RGB", (2 * W, 2 * H), (100 + i, 120, 140)).save(pool / "input" / "images" / f"{nm}.png")
        Image.fromarray(np.kron(m, np.ones((2, 2), np.uint8)).astype(np.uint8) * 255).save(pool / "masks" / f"{nm}.png")
    (pool / "prep.json").write_text(json.dumps({"views": [{"name": nm, "size": [2 * W, 2 * H], "source_size": list(SRC),
                                                            "crop": list(CROP)} for nm in names]}))
    samples = {}
    for i, (nm, e) in enumerate(zip(names, E)):
        if i in no_samples:
            continue
        z = true_depth(e, K_SRC, SRC)
        v, u = np.mgrid[0:SRC[1]:2, 0:SRC[0]:2]
        s = np.stack([u.ravel(), v.ravel(), z[v, u].ravel()], 1)
        samples[nm] = s[s[:, 2] < 10][:10] if i in few else s
    return pool, samples


def test_source_to_grid_matches_cameras_scaled_K(tmp_path):
    K_grid = np.array([[60.0, 0, 31.5], [0, 60.0, 23.5], [0, 0, 1]])  # K_SRC through CROP at 64x48
    run = tmp_path / "run"
    (run / "input" / "images").mkdir(parents=True)
    Image.new("RGB", (128, 96)).save(run / "input" / "images" / "0.png")  # the prep image: CROP resized to 128x96
    np.savez(run / "da3_output.npz", depth=np.ones((1, 48, 64), np.float32), extrinsics=np.eye(4)[None, :3],
             intrinsics=K_grid[None].astype(np.float32), image_files=np.array([str(run / "input" / "images" / "0.png")]))
    K_img = np.array(M.cameras(run)[0]["K"])
    X = np.random.default_rng(0).uniform([-0.3, -0.2, 1.0], [0.3, 0.2, 2.0], (50, 3))
    u, v, z, ok = L.project(X, K_SRC, None, None, SRC)
    assert ok.all() and np.allclose(z, X[:, 2])

    def pinhole(Kc):
        return Kc[0, 0] * X[:, 0] / X[:, 2] + Kc[0, 2], Kc[1, 1] * X[:, 1] / X[:, 2] + Kc[1, 2]

    assert np.allclose(L.source_to_grid(u, v, CROP, (64, 48)), pinhole(K_grid), atol=1e-9)
    assert np.allclose(L.source_to_grid(u, v, CROP, (128, 96)), pinhole(K_img), atol=1e-4)


def test_pixel_rounds_half_up():
    assert L.pixel([-0.51, -0.5, 0.0, 0.4999, 0.5, 1.49, 1.5]).tolist() == [-1, 0, 0, 0, 1, 1, 2]


def test_view_ratio_registers_samples_on_the_da3_grid():
    # Every DA3 pixel has its own depth, so a sample that lands on the wrong pixel gets a wrong ratio.
    rng = np.random.default_rng(3)
    depth = rng.uniform(1, 3, (H, W))
    crop = (37, 21, 247, 171)  # off centre; 210x150 to 64x48, so sx = 0.305 and sy = 0.32
    v, u = np.mgrid[0:200, 0:300]
    u, v = u.ravel(), v.ravel()
    gu, gv = (u + 0.5 - 37) * W / 210 - 0.5, (v + 0.5 - 21) * H / 150 - 0.5  # pixel centres at integers
    iu, iv = np.floor(gu + 0.5).astype(int), np.floor(gv + 0.5).astype(int)
    sel = (np.abs(gu - iu) <= 0.45) & (np.abs(gv - iv) <= 0.45) & (iu >= 0) & (iu < W) & (iv >= 0) & (iv < H)
    assert (gu[sel] - np.floor(gu[sel]) >= 0.5).any() and ((gu[sel] >= -0.5) & (gu[sel] < 0)).any()  # both roundings
    uvz = np.stack([u[sel], v[sel], 1.25 * depth[iv[sel], iu[sel]]], 1)
    rec, _ = L.view_ratio(uvz, crop, depth, None, np.ones((H, W), bool), zbuffer_tol=10.0)
    assert rec["points_grid"] == sel.sum() and rec["points_used"] > 0.95 * sel.sum()
    assert rec["s"] == 1.25 and rec["iqr_rel"] == 0 and rec["status"] == "candidate"


def test_view_ratio_takes_the_median_and_the_relative_iqr():
    r = 1.25 * (1 + 0.4 * np.linspace(0, 1, 100) ** 3)  # skewed: the mean is above the median
    i = np.arange(100)
    uvz = np.stack([10 + i % 40, 10 + i // 40, 2.0 * r], 1)
    rec, _ = L.view_ratio(uvz, (0, 0, W, H), np.full((H, W), 2.0), None, np.ones((H, W), bool), zbuffer_tol=10.0)
    p25, p50, p75 = np.percentile(r, [25, 50, 75])
    assert rec["points_used"] == 100 and np.isclose(rec["s"], p50) and np.mean(r) > p50 + 0.01
    assert np.isclose(rec["iqr_rel"], (p75 - p25) / p50, atol=1e-4)


def test_rot90_points_matches_np_rot90():
    a = np.arange(35).reshape(5, 7)
    v, u = np.mgrid[0:5, 0:7]
    for k in range(-1, 5):
        b = np.rot90(a, k)
        u2, v2 = L.rot90_points(u, v, (7, 5), k)
        assert np.array_equal(b[v2.astype(int), u2.astype(int)], a)


KP = np.array([[100.0, 0, 63.5], [0, 100.0, 47.5], [0, 0, 1]])


def test_project_without_distortion_is_pinhole():
    X = np.random.default_rng(1).uniform([-1, -1, 0.5], [1, 1, 3], (200, 3))
    u0, v0 = KP[0, 0] * X[:, 0] / X[:, 2] + 63.5, KP[1, 1] * X[:, 1] / X[:, 2] + 47.5
    inside = (u0 >= -0.5) & (u0 < 127.5) & (v0 >= -0.5) & (v0 < 95.5)
    assert 0 < inside.sum() < len(X)
    for D, model in ((None, None), ([0.0] * 5, "plumb_bob"), ([0.0] * 8, "rational_polynomial"), ([0.0] * 4, "radtan")):
        u, v, z, ok = L.project(X, KP, D, model, (128, 96))
        assert np.allclose(u[ok], u0[ok]) and np.allclose(v[ok], v0[ok]) and np.array_equal(ok, inside)
    u, v, z, ok = L.project([[0, 0, 0.05], [0, 0, -1], [0, 0, 0.11], [-0.641, 0, 1], [-0.639, 0, 1]], KP, None, None,
                            (128, 96))
    assert ok.tolist() == [False, False, True, False, True]  # too near, behind; u = -0.6 is outside, -0.4 inside
    with pytest.raises(ValueError, match="no distortion_model"):
        L.project(X, KP, [0.1, 0, 0, 0, 0], None, (128, 96))
    with pytest.raises(ValueError, match="unknown distortion model"):
        L.project(X, KP, [0.1], "division", (128, 96))


def test_project_distortion_by_hand():
    # plumb_bob at x = 0.2, y = -0.1: r2 = 0.05, radial = 1 - 0.3 r2 + 0.1 r2^2 + 0.01 r2^3 = 0.98525125,
    # x' = x radial + 2 p1 x y + p2 (r2 + 2 x^2) = 0.19675025, y' = y radial + p1 (r2 + 2 y^2) + 2 p2 x y = -0.098375125.
    D = [-0.3, 0.1, 0.001, -0.002, 0.01]
    for X in ([0.2, -0.1, 1.0], [0.4, -0.2, 2.0]):
        u, v, z, ok = L.project([X], KP, D, "plumb_bob", (128, 96))
        assert ok[0] and np.allclose([u[0], v[0]], [83.175025, 37.6624875]) and z[0] == X[2]
    # rational_polynomial with k4 = 0.05: radial divided by 1 + 0.05 r2 = 1.0025.
    u, v, _, _ = L.project([[0.2, -0.1, 1.0]], KP, D + [0.05, 0.0, 0.0], "rational_polynomial", (128, 96))
    assert np.allclose([u[0], v[0]], [83.12588528678305, 37.68705735660848])
    # equidistant at r = 0.5: theta = atan(0.5) = 0.4636476, theta_d = theta (1 + k1 theta^2 + k2 theta^4).
    for D, at in (([0.0] * 4, (109.86476090008061, 91.31885654004836, 84.59180872006449)),
                  ([0.1, 0.01, 0.0, 0.0], (110.88288596878951, 91.9297315812737, 85.4063087750316))):
        u, v, _, ok = L.project([[0.5, 0.0, 1.0], [0.3, 0.4, 1.0]], KP, D, "equidistant", (128, 96))
        assert ok.all() and np.allclose([u[0], v[0], u[1], v[1]], [at[0], 47.5, at[1], at[2]])


def test_project_fold_guard():
    # k1 = -0.5 folds x = 1.5 (pinhole u = 213.5, far right) back to x' = 1.5 (1 - 0.5 x 2.25) = -0.1875.
    xd, _ = L.distort(np.array([1.5]), np.array([0.0]), [-0.5, 0, 0, 0, 0], "plumb_bob")
    assert np.isclose(100 * xd[0] + 63.5, 44.75)  # inside the image
    u, v, z, ok = L.project([[1.5, 0.0, 1.0], [0.5, 0.0, 1.0]], KP, [-0.5, 0, 0, 0, 0], "plumb_bob", (128, 96))
    assert ok.tolist() == [False, True] and np.isclose(u[1], 0.4375 * 100 + 63.5)
    # x (1 - 0.5 x^2) stops increasing at r = sqrt(2/3) = 0.816. x = 0.85 is inside the grown image (pinhole
    # u = 148.5 < 153.1) but folds to u = 117.8, the pixel of x = 0.77; the corner point (r = 1.06) folds too.
    assert abs(L.fold_limit([-0.5, 0, 0, 0, 0], "plumb_bob", 2.0) - np.sqrt(2 / 3)) < 1e-3
    u, v, z, ok = L.project([[0.85, 0.0, 1.0], [0.85, 0.63, 1.0], [0.8, 0.0, 1.0]], KP, [-0.5, 0, 0, 0, 0], "plumb_bob",
                            (128, 96))
    assert ok.tolist() == [False, False, True]
    assert L.fold_limit([0.1, 0.01, 0, 0, 0], "plumb_bob", 5.0) == np.inf and L.fold_limit(None, None, 5.0) == np.inf


def test_project_wide_fisheye_keeps_its_outer_ring():
    # A 848x800 fisheye, f = 286: 75 deg off the axis lands 361 px from the centre, inside the image, while
    # its pinhole pixel (tan 75 deg = 3.73, 1067 px) is far outside any grown image.
    Kf = np.array([[286.0, 0, 423.5], [0, 286.0, 399.5], [0, 0, 1]])
    D = [-0.01, 0.05, -0.05, 0.01]
    th = np.radians([30.0, 60.0, 75.0, 85.0])
    X = np.stack([np.sin(th), np.zeros(4), np.cos(th)], 1)
    u, v, z, ok = L.project(X, Kf, D, "equidistant", (848, 800))
    t2 = th * th
    thd = th * (1 + t2 * (D[0] + t2 * (D[1] + t2 * (D[2] + t2 * D[3]))))
    assert ok.tolist() == [True, True, True, False]  # 85 deg lands 424.9 px out: off the right edge
    assert np.allclose(u[:3], 423.5 + 286 * thd[:3]) and np.allclose(v[:3], 399.5)
    # theta (1 - 0.5 theta^2) stops increasing at 46.8 deg: 60 deg would fold to the pixel of about 28 deg.
    u, v, z, ok = L.project([[np.tan(np.radians(30)), 0, 1], [np.tan(np.radians(60)), 0, 1]], KP, [-0.5, 0, 0, 0],
                            "equidistant", (128, 96))
    assert ok.tolist() == [True, False]


@pytest.mark.parametrize("at", [1.0, 3.0])  # the tolerance is relative: at 3 m, 3% is 9 cm
def test_zbuffer_drops_points_behind_a_foreground_plane(at):
    v, u = np.mgrid[0:10, 0:6]  # a plane at z = at (to 1.01 at) over columns 0..5
    iu, iv, z = u.ravel(), v.ravel(), at * (1.0 + 0.01 * (u.ravel() % 2))
    iu = np.r_[iu, 3, 6, 9, 2, 2]
    iv = np.r_[iv, 4, 4, 4, 7, 7]
    z = np.r_[z, at * np.array([3.0, 3.0, 3.0, 1.02, 1.05])]
    keep = L.zbuffer(iu, iv, z, (10, 12))
    assert keep[:60].all()  # the plane
    assert keep[60:].tolist() == [False, False, True, True, False]  # behind it, next to its edge, clear; 2% and 5% behind


def test_view_ratio_erodes_the_mask_and_drops_low_conf():
    from scipy.ndimage import distance_transform_edt

    mask = np.zeros((H, W), bool)
    mask[10:30, 20:40] = True
    conf = np.ones((H, W))
    conf[10:14, 20:40] = 0.1  # 20% of the mask: under the 30th percentile
    depth = np.full((H, W), 2.0)
    depth[(mask & (distance_transform_edt(mask) <= 1)) | (conf < 1)] = 1.0  # wrong DA3 depth on the rim and at low conf
    v, u = np.mgrid[0:H, 0:W]
    uvz = np.stack([u.ravel(), v.ravel(), np.full(u.size, 2.5)], 1)  # the source frame is the grid here
    rec, pts = L.view_ratio(uvz, (0, 0, W, H), depth, conf, mask)
    assert rec["erode_px"] == 1 and rec["conf_min"] == 1.0 and rec["points_mask"] == 18 * 18
    assert rec["points_used"] == 15 * 18 and rec["s"] == 1.25 and rec["iqr_rel"] == 0 and rec["status"] == "candidate"
    assert pts["used"].sum() == 270 and len(pts["u"]) == H * W

    small = np.zeros((H, W), bool)
    small[20:23, 30:33] = True  # eroding 1 px keeps 1 of 9 px: no erosion
    rec, _ = L.view_ratio(uvz, (0, 0, W, H), np.full((H, W), 2.0), None, small)
    assert rec["erode_px"] == 0 and rec["points_used"] == 9 and rec["status"] == "few_points" and rec["s"] == 1.25
    assert L.view_ratio(uvz, (0, 0, W, H), depth, conf, np.zeros((H, W), bool))[0]["status"] == "no_mask"
    rec, _ = L.view_ratio(np.zeros((0, 3)), (0, 0, W, H), depth, conf, mask)
    assert rec["status"] == "no_samples" and rec["s"] is None


def test_view_ratio_takes_the_conf_percentile_over_the_whole_mask():
    from scipy.ndimage import distance_transform_edt

    mask = np.zeros((H, W), bool)
    mask[10:30, 20:40] = True
    inner = distance_transform_edt(mask) > 1  # the mask eroded by 1 px
    conf = np.zeros((H, W))
    conf[mask] = 0.1  # the rim: 19% of the mask
    conf[inner] = np.random.default_rng(0).permutation(np.linspace(0.5, 1.0, inner.sum()))
    v, u = np.mgrid[0:H, 0:W]
    uvz = np.stack([u.ravel(), v.ravel(), np.full(u.size, 2.5)], 1)
    rec, _ = L.view_ratio(uvz, (0, 0, W, H), np.full((H, W), 2.0), conf, mask)
    p = np.percentile(conf[mask], 30)  # over the eroded mask it would be higher and drop more points
    assert rec["erode_px"] == 1 and rec["conf_min"] == round(p, 4) and p < np.percentile(conf[inner], 30)
    assert rec["points_used"] == (inner & (conf >= p)).sum() == 280


def test_weighted_median_ties():
    assert L.weighted_median([0, 1, 2], [0.3, 0.1, 0.2]) == 0.5  # half the weight is reached exactly at 0
    assert L.weighted_median([0, 1, 2], [0.1, 0.2, 0.3]) == 1.5
    assert L.weighted_median([2, 0, 1], [1, 1, 1]) == 1.0 and L.weighted_median([5.0], [3.0]) == 5.0


def test_combine_weights_are_capped_square_roots():
    def views(*pairs):
        return [{"s": s, "points_used": n, "status": "candidate"} for s, n in pairs]

    # Weights sqrt(2000), sqrt(1000), sqrt(1000): 1.03 is the median. Without the cap 1.0 would be, and with
    # weights n instead of sqrt(n) the two would tie.
    c = L.combine(views((1.0, 8000), (1.03, 1000), (1.04, 1000)))
    assert c["ok"] and np.isclose(c["s"], 1.03) and c["views_used"] == 3


def test_combine_needs_half_of_the_views_with_points():
    def views(*pairs, status="candidate"):
        return [{"s": s, "points_used": n, "status": status} for s, n in pairs]

    vs = views((1.0, 500), (1.01, 500), (0.99, 500)) + views(*[(1.0, 500)] * 4, status="scattered")
    c = L.combine(vs)
    assert not c["ok"] and c["reason"].startswith("the per-view scales disagree: 3 of the 7 views with LiDAR points "
                                                  "agree (4 scattered, 0 outliers) and at least 4 must")
    vs = views(*[(1.0, 2000)] * 4, (1.5, 30), (0.6, 30), (1.4, 30), (0.7, 30), (2.0, 30))
    c = L.combine(vs)
    assert not c["ok"] and c["reason"].startswith("the per-view scales disagree: 4 of the 9 views with LiDAR points "
                                                  "agree (0 scattered, 5 outliers) and at least 5 must")


def test_combine_outlier_cut_uses_the_normal_mad():
    # ln s_i: median 0, MAD 0.06, cut 3 x 1.4826 x 0.06 = 0.267. 0.22 is inside it (but beyond 3 MAD = 0.18).
    x = [0.0, 0.04, -0.04, 0.06, -0.06, 0.22, -0.07]
    vs = [{"s": float(np.exp(a)), "points_used": 100, "status": "candidate"} for a in x]
    c = L.combine(vs)
    assert [v["status"] for v in vs] == ["used"] * 7 and np.isclose(c["cut_rel"], np.expm1(3 * 1.4826 * 0.06), atol=1e-5)
    assert np.isclose(c["mad_rel"], np.expm1(0.06), atol=1e-5)


def test_combine_weights_and_checks():
    def views(*pairs):
        return [{"s": s, "points_used": n, "status": "candidate"} for s, n in pairs]

    vs = views((1.0, 10000), (1.02, 100), (1.03, 100))  # weights sqrt(2000), 10, 10
    c = L.combine(vs)
    assert c["ok"] and c["s"] == 1.0 and c["views_used"] == 3 and [v["status"] for v in vs] == ["used"] * 3
    vs = views((1.0, 100), (1.01, 100), (0.99, 100), (1.3, 100))
    c = L.combine(vs)
    assert c["ok"] and np.isclose(c["s"], 1.0) and vs[3]["status"] == "outlier" and c["cut_rel"] == 0.05
    c = L.combine(views((1.0, 100), (1.0, 100)))
    assert not c["ok"] and c["reason"].startswith("LiDAR points cover the object in only 2 of 2 views")
    assert L.combine(views((1.0, 100), (1.0, 100)), min_views=2)["ok"]


def test_estimate_recovers_the_scale(tmp_path):
    pool, samples = lidar_pool(tmp_path)
    rec = L.estimate(pool, samples, check_dir=tmp_path / "check")
    assert rec["ok"] and rec["reason"] is None and abs(rec["s"] / 1.25 - 1) < 0.005
    assert rec["version"] == 1 and rec["da3"] == {"process_res": 504, "grid": [W, H]} and rec["warnings"] == []
    assert [v["status"] for v in rec["views"]] == ["used"] * 8 and [v["frame"] for v in rec["views"]] == list(range(0, 80, 10))
    assert min(v["points_used"] for v in rec["views"]) > 300 and rec["combine"]["views_used"] == 8
    assert rec["combine"]["spread_rel"] < 0.005 and rec["combine"]["thresholds"]["min_points"] == 30
    json.dumps(rec)  # plain JSON types
    for v in rec["views"]:
        assert Image.open(tmp_path / "check" / f"{v['name']}.jpg").size == (640, 480)
    assert L.estimate(pool, samples, min_points=600)["reason"].startswith("LiDAR points cover the object in only 0 of 8")
    with pytest.raises(TypeError, match="min_point"):
        L.estimate(pool, samples, min_point=10)


def test_estimate_leaves_out_outliers_and_odd_views(tmp_path):
    pool, samples = lidar_pool(tmp_path, n=10, factors=[1, 1, 1, 1, 1, 1, 1.5, 1.5, 1, 1], step=(8,), empty_mask=(5,),
                               no_samples=(9,))
    rec = L.estimate(pool, samples)
    st = [v["status"] for v in rec["views"]]
    assert st == ["used"] * 5 + ["no_mask", "outlier", "outlier", "scattered", "no_samples"]
    assert rec["ok"] and abs(rec["s"] / 1.25 - 1) < 0.005 and rec["views"][8]["iqr_rel"] > 0.25
    c = rec["combine"]
    assert (c["views_with_points"], c["views_used"], c["outliers"], c["scattered"]) == (8, 5, 2, 1)
    assert any("f00060: its scale 1.87" in w and "+50%" in w for w in rec["warnings"])
    assert any("no LiDAR samples in 1 of 10 views: f00090" in w for w in rec["warnings"])


def test_estimate_names_samples_that_match_no_view(tmp_path):
    pool, samples = lidar_pool(tmp_path)
    rec = L.estimate(pool, {**samples, "f99999": samples["f00000"]})
    assert rec["ok"] and rec["warnings"] == [
        "LiDAR samples for frames that are not pool views: f99999 (pool views are named like f00012)"]
    rec = L.estimate(pool, {str(int(k[1:])): a for k, a in samples.items()})  # "0", "10", ...
    assert not rec["ok"] and rec["reason"].startswith(
        "none of the LiDAR samples' frame names (0, 10, 20, 30, 40 and 3 more) is a pool view")


def test_check_image_shows_points_off_the_mask_in_colour(tmp_path):
    Image.new("RGB", (64, 48), (120, 120, 120)).save(tmp_path / "img.png")
    mask = np.zeros((48, 64), bool)
    mask[16:32, 24:40] = True
    pts = {"u": np.array([5.0, 32.0, 58.0]), "v": np.array([5.0, 24.0, 40.0]), "r": np.array([1.5, 1.0, 0.6]),
           "used": np.array([False, True, False])}
    L.check_image(tmp_path / "img.png", mask, (0, 0, 64, 48), pts, 1.0, tmp_path / "c.jpg")
    a = np.asarray(Image.open(tmp_path / "c.jpg"), int)
    assert a.shape == (480, 640, 3)
    red, blue, white = a[55:57, 55:57], a[405:407, 585:587], a[244:247, 324:327]  # pixel 10 u + 4.5 at 10x
    assert (red[..., 0] - red[..., 1] > 80).all() and (blue[..., 2] - blue[..., 0] > 80).all()
    assert (white > 200).all() and a[100, 100].max() < 90  # the background is dimmed


def test_estimate_fails_with_too_few_views_and_the_cli_says_why(tmp_path, capsys):
    pool, samples = lidar_pool(tmp_path, few=(2,), no_samples=(3, 4, 5, 6, 7))
    np.savez(tmp_path / "samples.npz", **samples)
    with pytest.raises(SystemExit, match="LiDAR points cover the object in only 2 of 8 views"):
        L.main(["estimate", "--pool", str(pool), "--samples", str(tmp_path / "samples.npz"), "--out", str(tmp_path / "lidar.json"),
                "--check", str(tmp_path / "check")])
    rec = json.loads((tmp_path / "lidar.json").read_text())
    assert not rec["ok"] and [v["status"] for v in rec["views"]] == ["used", "used", "few_points"] + ["no_samples"] * 5
    assert len(list((tmp_path / "check").glob("*.jpg"))) == 8
    assert re.search(r"f00020: few_points, \d of 10 points\n", capsys.readouterr().out)


def test_estimate_fails_when_views_disagree(tmp_path):
    pool, samples = lidar_pool(tmp_path, n=6, step=(2, 3, 4), no_samples=(5,))
    rec = L.estimate(pool, samples)
    assert not rec["ok"] and rec["reason"].startswith(
        "the per-view scales disagree: 2 of the 5 views with LiDAR points agree (3 scattered, 0 outliers) and at least 3 must")


def test_estimate_fails_on_spread(tmp_path):
    pool, samples = lidar_pool(tmp_path, factors=[0.85, 1.0, 1.15, 0.9, 1.1, 1.2, 0.95, 1.05])
    rec = L.estimate(pool, samples)
    assert not rec["ok"] and rec["combine"]["views_used"] == 8 and rec["combine"]["spread_rel"] > 0.08
    assert rec["reason"].startswith("the per-view scales spread by 12.")


def test_estimate_fails_far_from_da3(tmp_path):
    pool, samples = lidar_pool(tmp_path, factors=[2.4] * 8)
    rec = L.estimate(pool, samples)
    assert not rec["ok"] and abs(rec["s"] / 3 - 1) < 0.005
    assert rec["reason"] == ("the LiDAR scale 3.00 is far from Depth Anything 3's metric estimate; "
                             "check T_cam_lidar (is it T_lidar_cam?) and the units")


# Calibration files.
R_TEST = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]) @ np.array(
    [[np.cos(0.1), -np.sin(0.1), 0.0], [np.sin(0.1), np.cos(0.1), 0.0], [0.0, 0.0, 1.0]])
T_TEST = np.vstack([np.hstack([R_TEST, [[0.05], [-0.1], [0.02]]]), [0, 0, 0, 1]])
K_TEST = [[600.0, 0.0, 320.0], [0.0, 610.0, 240.0], [0.0, 0.0, 1.0]]


def calib(T=T_TEST, key="T_cam_lidar", **extra):
    return "\n".join([f"{key}: {json.dumps(np.asarray(T).tolist())}"] + [f"{k}: {json.dumps(v)}" for k, v in extra.items()])


def test_read_calibration_canonical_yaml_and_json():
    extra = {"camera_axes": "optical", "time_offset_s": 0.02, "K": K_TEST, "D": [-0.1, 0.01, 0.0, 0.0, 0.0],
             "distortion_model": "plumb_bob", "image_size": [640, 480]}
    c = L.read_calibration(calib(**extra), "rig.yaml")
    assert np.allclose(c["T_cam_lidar"], T_TEST, atol=1e-12) and c["given_as"] == "T_cam_lidar"
    assert (c["camera_axes"], c["time_offset_s"], c["K"], c["D"], c["distortion_model"], c["image_size"]) == (
        "optical", 0.02, K_TEST, [-0.1, 0.01, 0.0, 0.0, 0.0], "plumb_bob", [640, 480])
    assert L.read_calibration(json.dumps({"T_cam_lidar": T_TEST.tolist(), **extra}), "rig.json") == c
    assert L.read_calibration(calib(**extra).encode(), "rig.yml") == c
    bare = L.read_calibration(calib(), "rig.yaml")
    assert (bare["camera_axes"], bare["time_offset_s"], bare["K"], bare["D"], bare["distortion_model"]) == (
        "optical", 0.0, None, None, None)


def test_read_calibration_aliases():
    def T_of(text, name="c.yaml"):
        return np.array(L.read_calibration(text, name)["T_cam_lidar"])

    assert np.allclose(T_of(calib(np.linalg.inv(T_TEST), "T_lidar_cam")), T_TEST)
    assert np.allclose(T_of(calib(T_TEST.ravel())), T_TEST)
    data = ", ".join(f"{x:.17g}" for x in T_TEST.ravel())
    opencv = (f"%YAML:1.0\n---\nT_cam_lidar: !!opencv-matrix\n   rows: 4\n   cols: 4\n   dt: d\n   data: [ {data} ]\n"
              "K: !!opencv-matrix\n   rows: 3\n   cols: 3\n   dt: d\n   data: [ 600., 0., 320., 0., 610., 240., 0., 0., 1. ]\n")
    c = L.read_calibration(opencv, "calib.yaml")
    assert np.allclose(c["T_cam_lidar"], T_TEST) and c["K"] == K_TEST
    livo = f"Rcl: {json.dumps(R_TEST.ravel().tolist())}\nPcl: [0.05, -0.1, 0.02]\n"
    c = L.read_calibration(livo)
    assert np.allclose(c["T_cam_lidar"], T_TEST) and c["given_as"] == "Rcl"
    nested = "extrin_calib:\n  extrinsic_T: [0, 0, 0]\n" + "".join("  " + line + "\n" for line in livo.splitlines())
    assert np.allclose(T_of(nested), T_TEST)
    body = T_of(calib(camera_axes="body"))
    assert np.allclose(body[:3], L.R_OPT_LINK @ T_TEST[:3]) and np.allclose(body[3], [0, 0, 0, 1])
    ros = calib(camera_matrix={"rows": 3, "cols": 3, "data": sum(K_TEST, [])}, distortion_model="radtan",
                distortion_coefficients={"rows": 1, "cols": 4, "data": [0.1, 0, 0, 0]}, image_width=640, image_height=480)
    c = L.read_calibration(ros)
    assert (c["K"], c["D"], c["distortion_model"], c["image_size"]) == (K_TEST, [0.1, 0, 0, 0], "plumb_bob", [640, 480])
    assert L.read_calibration(calib(K=[600, 610, 320, 240]))["K"] == K_TEST  # fx, fy, cx, cy
    assert L.read_calibration(calib(D=[0.1, 0, 0, 0], distortion_model="fisheye"))["distortion_model"] == "equidistant"
    assert L.read_calibration(calib(D=[0.1, 0, 0, 0, 0], distortion_model=" Plumb_Bob"))["distortion_model"] == "plumb_bob"
    ros2 = "/**:\n  ros__parameters:\n    extrin_calib:\n" + "".join("      " + line + "\n" for line in livo.splitlines())
    assert np.allclose(T_of(ros2), T_TEST)
    js = json.dumps({"T_cam_lidar": T_TEST.tolist()})
    assert np.allclose(T_of(b"\xef\xbb\xbf" + js.encode(), "c.json"), T_TEST)  # a byte order mark
    assert np.allclose(T_of("\ufeff" + js, "c.json"), T_TEST)


def test_read_calibration_opencv_distortion_without_a_model():
    def cv(n):
        data = ", ".join(f"{x:.17g}" for x in T_TEST.ravel())
        d = ", ".join(["-0.1", "0.01"] + ["0.001"] * (n - 2))
        return (f"%YAML:1.0\n---\nT_cam_lidar: !!opencv-matrix\n   rows: 4\n   cols: 4\n   dt: d\n   data: [ {data} ]\n"
                f"distortion_coefficients: !!opencv-matrix\n   rows: 1\n   cols: {n}\n   dt: d\n   data: [ {d} ]\n")

    assert L.read_calibration(cv(5))["distortion_model"] == "plumb_bob" and len(L.read_calibration(cv(5))["D"]) == 5
    assert L.read_calibration(cv(8))["distortion_model"] == "rational_polynomial"
    with pytest.raises(ValueError, match="no distortion_model"):  # only OpenCV's own files get a model
        L.read_calibration(cv(5).replace("%YAML:1.0\n", ""))


def test_read_calibration_reorthonormalises_the_rotation():
    T = T_TEST.copy()
    T[:3, :3] += 2e-4 * np.random.default_rng(0).normal(size=(3, 3))
    R = np.array(L.read_calibration(calib(T))["T_cam_lidar"])[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12) and np.allclose(R, R_TEST, atol=1e-3)


def with_row(T, row):
    T = np.array(T, float)
    T[3] = row
    return T


@pytest.mark.parametrize("text, name, match", [
    (calib(with_row(T_TEST, [0, 0, 1, 1])), "c.yaml", "last row of T_cam_lidar"),
    (calib(np.diag([1.1, 1.0, 1.0, 1.0])), "c.yaml", "not orthonormal"),
    (calib(np.diag([1.0, 1.0, -1.0, 1.0])), "c.yaml", "reflection"),
    (calib(np.hstack([np.eye(4)[:, :3], [[35.0], [0], [0], [1]]])), "c.yaml", "the translation is 35 m: is it in millimeters"),
    (calib(np.linalg.inv(np.hstack([np.eye(4)[:, :3], [[0], [3.0], [0], [1]]])), "T_lidar_cam"), "c.yaml", "translation is 3 m"),
    (calib(K=[[-600.0, 0, 320], [0, 610, 240], [0, 0, 1]]), "c.yaml", "K must have positive"),
    (calib(K=[600.0, 610, 320]), "c.yaml", "K has 3 numbers"),
    ("K: [600, 610, 320, 240]\n", "c.yaml", "no T_cam_lidar, T_lidar_cam or Rcl and Pcl"),
    (calib() + "\n" + calib(key="T_lidar_cam"), "c.yaml", "give one of them"),
    ("T_cam_lidar: [1, 2\n", "c.yaml", re.escape("not valid YAML: expected ',' or ']', but got '<stream end>' (line 2)")),
    ("{'T_cam_lidar': 1}", "c.json", "not valid JSON"),
    ("- 1\n- 2\n", "c.yaml", "has no keys"),
    (calib(T_TEST.ravel()[:12]), "c.yaml", "T_cam_lidar has 12 numbers; give a 4x4 matrix or 16 numbers"),
    (calib(camera_axes="camera"), "c.yaml", "camera_axes is 'camera'"),
    (calib(D=[0.1, 0, 0, 0, 0]), "c.yaml", "no distortion_model"),
    (calib(D=[0.1], distortion_model="division"), "c.yaml", "unknown distortion model"),
    (calib(D=[0.1, 0, 0, 0, 0.2], distortion_model="equidistant"), "c.yaml", "equidistant takes 4 coefficients"),
    ("Rcl: [1, 0, 0, 0, 1, 0, 0, 0, 1]\n", "c.yaml", "both Rcl and Pcl"),
    ("T_cam_lidar: {rows: 4, cols: 4, data: [1, 0, 0, 0]}\n", "c.yaml", "4 numbers in data for 4x4"),
    ("T_cam_lidar: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, one]]\n", "c.yaml", "not a list of numbers"),
    (calib(image_size=[640.5, 480]), "c.yaml", "image_size is"),
    (calib(time_offset_s="soon"), "c.yaml", "time_offset_s is not a number"),
    (calib(distortion_model="plumb_bob"), "c.yaml", "distortion_model is plumb_bob but D is missing"),
    (calib(intrinsics=[600, 610, 320, 240], distortion_coeffs=[0.1, 0, 0, 0]), "c.yaml",
     "intrinsics and distortion_coeffs \\(Kalibr's names\\); give K"),
])
def test_read_calibration_errors(text, name, match):
    with pytest.raises(ValueError, match=match):
        L.read_calibration(text, name)


def test_read_calibration_without_ruamel(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "ruamel", None)
    monkeypatch.setitem(sys.modules, "ruamel.yaml", None)
    try:
        import yaml  # noqa: F401
    except ImportError:
        with pytest.raises(ValueError, match="needs ruamel.yaml or PyYAML"):
            L.read_calibration(calib())
        assert np.allclose(L.read_calibration(json.dumps({"T_cam_lidar": T_TEST.tolist()}), "c.json")["T_cam_lidar"], T_TEST)
    else:
        assert np.allclose(L.read_calibration(calib())["T_cam_lidar"], T_TEST)
