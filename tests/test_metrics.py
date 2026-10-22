import numpy as np
import pytest
import trimesh

from gator import frames as F
from gator.eval import metrics as M
from gator.eval import similarity as S

rng = np.random.default_rng(0)
N = 20_000


def rotation(deg, axis):
    axis = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    K = np.cross(np.eye(3), axis)
    a = np.deg2rad(deg)
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def random_rotation():
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def square(z=0.0):
    """Unit square in the plane z, as two triangles."""
    v = np.array([[0, 0, z], [1, 0, z], [1, 1, z], [0, 1, z]], dtype=np.float64)
    return trimesh.Trimesh(v, [[0, 1, 2], [0, 2, 3]], process=False)


def box():
    return trimesh.creation.box(extents=(1.0, 0.6, 0.3))


def lump():
    """A shape with no symmetry: a box with a ball on one corner."""
    ball = trimesh.creation.icosphere(subdivisions=2, radius=0.25)
    ball.apply_translation([0.45, 0.25, 0.1])
    return trimesh.util.concatenate([box(), ball])


def test_identical_meshes_score_perfectly():
    mesh = lump()
    D = M.diameter(mesh.vertices)
    r = M.shape_metrics(mesh, mesh, D, M.TAU_REAL, n=N)
    assert r["cd"] == pytest.approx(0.0, abs=1e-12)
    assert r["f1"] == 1.0
    assert r["nc"] == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(r["T"], np.eye(4), atol=1e-12)
    assert M.add_sb(mesh, mesh, D, n=N) == 0.0


def test_chamfer_by_hand():
    a = np.array([[0.0, 0, 0], [3, 0, 0], [0, 4, 0]])
    b = np.zeros((1, 3))
    assert M.directed_distance(a, b) == pytest.approx(7 / 3)
    assert M.directed_distance(b, a) == 0.0
    assert M.chamfer(a, b, D=2.0) == pytest.approx((7 / 3) / 4)


def test_translation_gives_expected_cd_and_cd_is_not_squared():
    # Same seed, so each sample of the lifted square sits straight above its partner, at distance dz.
    D = np.sqrt(2)
    cd1 = M.add_sb(square(0.05), square(0.0), D, n=N)
    cd2 = M.add_sb(square(0.10), square(0.0), D, n=N)
    assert cd1 == pytest.approx(0.05 / D, rel=1e-9)
    assert cd2 == pytest.approx(2 * cd1, rel=1e-9)


def test_f1_threshold_is_relative_to_d():
    # A 0.1 grid in the plane z = 0, lifted by 0.03: every nearest neighbor is the lifted partner.
    x, y = np.meshgrid(np.arange(10) * 0.1, np.arange(10) * 0.1)
    pa = np.stack([x.ravel(), y.ravel(), np.zeros(100)], 1)
    na = np.tile([0.0, 0.0, 1.0], (100, 1))
    pb = pa + [0.0, 0.0, 0.03]
    D = 2.0
    assert M.surface_scores(pa, na, pb, na, D, tau=0.02)["f1"] == 1.0  # 0.03 < 0.04
    assert M.surface_scores(pa, na, pb, na, D, tau=0.01)["f1"] == 0.0  # 0.03 > 0.02
    assert M.surface_scores(pa, na, pb, -na, D, tau=0.02)["nc"] == 1.0  # orientation is ignored


def test_icp_recovers_similarity():
    src = M.sample_surface(lump(), 4000, seed=1)[0]
    T_true = S.compose(1.3, rotation(35, [1, 2, 3]), [0.1, -0.05, 0.08])
    dst = rng.permutation(F.transform_points(T_true, src))
    T = S.icp(src, dst)
    assert np.allclose(T, T_true, atol=1e-9)
    s, R, t = S.decompose(T)
    assert s == pytest.approx(1.3)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_icp_starts_from_init():
    src = M.sample_surface(lump(), 4000, seed=1)[0]
    T_true = S.compose(1.3, rotation(120, [0, 1, 1]), [0.4, 0.2, -0.3])
    dst = F.transform_points(T_true, src)
    assert not np.allclose(S.icp(src, dst), T_true, atol=1e-3)
    init = S.compose(1.2, rotation(110, [0, 1, 1.2]), [0.35, 0.25, -0.3])
    assert np.allclose(S.icp(src, dst, init), T_true, atol=1e-9)


def test_shape_fit_does_not_shrink_an_offset_prediction():
    # A perfect shape, turned, scaled and placed half a diameter away. One-way scaled ICP shrinks
    # it onto the near side of the reference (s about 0.44 here), so the shape scores look bad.
    ref = lump()
    D = M.diameter(ref.vertices)
    T_true = S.compose(0.8, rotation(30, [1, 2, 3]), [0.5 * D, 0.0, 0.0])
    r = M.shape_metrics(ref.copy().apply_transform(T_true), ref, D, M.TAU_REAL, n=N)
    assert np.allclose(r["T"], np.linalg.inv(T_true), atol=1e-9)
    assert r["cd"] == pytest.approx(0.0, abs=1e-12)
    assert r["nc"] == pytest.approx(1.0, abs=1e-12)
    assert r["f1"] == 1.0


def test_fits_never_reflect():
    # Principal axes 3, 2, 1, mirrored in x. The best proper fit turns 180 degrees about y and
    # flips the weakest axis, so s = (18 + 8 - 2) / (18 + 8 + 2).
    src = np.array([[3.0, 0, 0], [-3, 0, 0], [0, 2, 0], [0, -2, 0], [0, 0, 1], [0, 0, -1]])
    mirror = src * [-1, 1, 1]
    s, R, t = S.decompose(S.umeyama(src, mirror))
    assert s == pytest.approx(24 / 28)
    assert np.allclose(R, np.diag([-1.0, 1, -1]))
    assert np.allclose(t, 0)

    cloud = M.sample_surface(lump(), 2000, seed=2)[0]
    for T in (S.umeyama(cloud, cloud * [-1, 1, 1]), S.icp(cloud, cloud * [1, -1, 1])):
        s, R, _ = S.decompose(T)
        assert s > 0
        assert np.linalg.det(R) == pytest.approx(1.0)
        assert np.allclose(R @ R.T, np.eye(3))


def test_add_sb_is_not_reduced_by_icp():
    D = np.sqrt(2)
    pred, ref = square(0.05), square(0.0)
    r = M.shape_metrics(pred, ref, D, M.TAU_REAL, n=N)
    assert r["cd"] == pytest.approx(0.0, abs=1e-12)
    assert r["f1"] == 1.0
    assert np.allclose(r["T"], F.rigid(np.eye(3), [0, 0, -0.05]))
    assert M.add_sb(pred, ref, D, n=N) == pytest.approx(0.05 / D, rel=1e-9)


def test_observed_variant_fits_from_the_scan_only():
    # The scan saw the box but missed the ball, and the prediction is the whole lump, shifted.
    offset = np.array([0.03, -0.02, 0.015])
    pred, scan = lump().apply_translation(offset), box()
    D = M.diameter(scan.vertices)
    r = M.shape_metrics_observed(pred, scan, D, n=N)
    s, R, t = S.decompose(r["T"])
    assert s == pytest.approx(1.0, abs=1e-3)
    assert np.allclose(R, np.eye(3), atol=1e-3)
    assert np.allclose(t, -offset, atol=1e-3)
    # The ball is penalized: CD and F1 are scored in both directions after the fit.
    pa, na = M.sample_surface(pred, N)
    pb, nb = M.sample_surface(scan, N)
    expect = M.surface_scores(F.transform_points(r["T"], pa), na @ R.T, pb, nb, D, M.TAU_REAL)
    assert r["cd"] == pytest.approx(expect["cd"])
    assert r["f1"] == pytest.approx(expect["f1"])
    assert r["f1"] < 0.95
    # Fitting the whole prediction to the scan instead drags the ball onto the box.
    s_full = S.decompose(M.shape_metrics(pred, scan, D, M.TAU_REAL, n=N)["T"])[0]
    assert s_full < 0.95


def test_observed_variant_uses_init_from_prediction_to_reference():
    # 120 degrees is too far for ICP from the identity. init maps the prediction to the scan.
    scan = box()
    T_true = S.compose(1.2, rotation(120, [0, 1, 1]), [0.4, 0.2, -0.3])  # scan -> prediction
    pred = scan.copy().apply_transform(T_true)
    D = M.diameter(scan.vertices)
    init = np.linalg.inv(S.compose(1.1, rotation(110, [0, 1, 1.2]), [0.35, 0.25, -0.3]))
    r = M.shape_metrics_observed(pred, scan, D, init=init, n=N)
    assert np.allclose(r["T"], np.linalg.inv(T_true), atol=1e-9)
    assert r["cd"] == pytest.approx(0.0, abs=1e-12)


def test_camera_similarity():
    V = 5
    R_ref = np.stack([random_rotation() for _ in range(V)])
    c_ref = rng.normal(size=(V, 3)) * 2
    T_true = S.compose(0.4, random_rotation(), [1.0, 2.0, 3.0])  # estimated frame -> reference frame
    c_est = F.transform_points(np.linalg.inv(T_true), c_ref)
    R_est = R_ref @ S.decompose(T_true)[1]

    # The estimated cameras see a point exactly where the reference cameras see its image under T.
    K = np.array([[500.0, 0, 320], [0, 500, 240], [0, 0, 1]])
    x_est = rng.normal(size=(10, 3)) * 0.3 + c_est.mean(0)
    x_ref = F.transform_points(T_true, x_est)
    for i in range(V):
        uv_est, _ = F.project(x_est, K, F.rigid(R_est[i], -R_est[i] @ c_est[i]))
        uv_ref, _ = F.project(x_ref, K, F.rigid(R_ref[i], -R_ref[i] @ c_ref[i]))
        assert np.allclose(uv_est, uv_ref)

    assert np.allclose(S.camera_similarity(c_est, c_ref), T_true)
    assert np.allclose(S.camera_similarity(c_est, c_ref, R_est, R_ref), T_true)
    # Two centers leave the roll about the baseline open. The rotations pin it.
    assert np.allclose(S.camera_similarity(c_est[:2], c_ref[:2], R_est[:2], R_ref[:2]), T_true)


def test_diameter():
    cube = trimesh.creation.box()
    assert M.diameter(cube.vertices) == pytest.approx(np.sqrt(3))
    inside = rng.uniform(-0.4, 0.4, size=(1000, 3))
    assert M.diameter(np.vstack([cube.vertices, inside])) == pytest.approx(np.sqrt(3))
    assert M.diameter(trimesh.creation.icosphere(subdivisions=3).vertices) == pytest.approx(2.0)


def test_diameter_of_flat_and_collinear_points():
    assert M.diameter(square(0.3).vertices) == pytest.approx(np.sqrt(2))
    assert M.diameter(np.outer(np.arange(5.0), [1.0, 2.0, 2.0])) == pytest.approx(12.0)


def test_aggregation_counts_failures():
    add = [0.05, None, 0.2, 0.08, float("nan")]
    assert M.add_sb_recall(add) == pytest.approx((2 / 5, 3 / 5))
    assert M.mean_valid(add) == pytest.approx(((0.05 + 0.2 + 0.08) / 3, 3 / 5))
    assert M.iou_mean([0.8, None, 0.6]) == pytest.approx(((0.8 + 0.0 + 0.6) / 3, 2 / 3))
    assert M.add_sb_recall([0.1]) == (0.0, 1.0)  # strictly below the threshold
    assert M.add_sb_recall([0.05, float("inf")]) == (0.5, 0.5)


def test_aggregation_with_no_valid_output():
    mean, rate = M.mean_valid([None, float("nan")])
    assert np.isnan(mean) and rate == 0.0
    assert M.iou_mean([None, None]) == (0.0, 0.0)
    assert M.add_sb_recall([None]) == (0.0, 0.0)
