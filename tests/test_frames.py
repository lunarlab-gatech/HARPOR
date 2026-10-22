import numpy as np

from gator import frames as F

rng = np.random.default_rng(0)


def random_rotation():
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def test_g_from_w_is_proper_rotation():
    R = F.G_FROM_W
    assert np.allclose(R @ R.T, np.eye(3))
    assert np.isclose(np.linalg.det(R), 1.0)


def test_g_axes():
    assert np.allclose(F.w_to_g([0.0, -1.0, 0.0]), [0, 0, 1])  # OpenCV up -> G up (+z)
    assert np.allclose(F.w_to_g([0.0, 0.0, 1.0]), [0, 1, 0])  # camera forward -> G +y
    assert np.allclose(F.w_to_g([1.0, 0.0, 0.0]), [1, 0, 0])


def test_gltf_equals_g():
    p_w = rng.normal(size=(50, 3))
    assert np.allclose(F.gltf_to_w(F.g_to_gltf(F.w_to_g(p_w))), p_w)
    assert np.allclose(F.W_FROM_GLTF @ F.G_FROM_W, np.eye(3))


def test_to_glb_rotation_is_minus_90_about_x():
    a = np.deg2rad(-90)
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    assert np.allclose(F.TO_GLB_ROTATION, Rx)
    assert np.allclose(F.rotate(F.TO_GLB_ROTATION, [1.0, 2.0, 3.0]), [1, 3, -2])


def test_to_glb_input_lands_on_g():
    p_g = rng.normal(size=(20, 3))
    assert np.allclose(F.rotate(F.TO_GLB_ROTATION, F.to_glb_input_from_g(p_g)), p_g)


def test_normalize_roundtrip_has_no_rotation():
    p = rng.normal(size=(30, 3))
    t, s = np.array([0.3, -1.0, 2.0]), 0.7
    q = F.normalize(p, t, s)
    assert np.allclose(F.denormalize(q, t, s), p)
    d = p[1] - p[0]
    assert np.allclose(q[1] - q[0], d / s)


def test_invert_rigid():
    T = F.rigid(random_rotation(), rng.normal(size=3))
    assert np.allclose(T @ F.invert_rigid(T), np.eye(4))


def test_rebase_to_view0_keeps_pixels():
    K = np.array([[500.0, 0, 320], [0, 510.0, 240], [0, 0, 1]])
    w2c = np.stack([F.rigid(random_rotation(), rng.normal(size=3)) for _ in range(3)])
    p_old = rng.normal(size=(40, 3)) + w2c[0][:3, :3].T @ np.array([0, 0, 5.0])
    new = F.rebase_to_view0(w2c)
    assert np.allclose(new[0], np.eye(4))
    p_new = F.transform_points(w2c[0], p_old)
    for i in range(3):
        uv_a, z_a = F.project(p_old, K, w2c[i])
        uv_b, z_b = F.project(p_new, K, new[i])
        assert np.allclose(uv_a, uv_b) and np.allclose(z_a, z_b)


def test_project_unproject_roundtrip():
    K = np.array([[300.0, 0, 31.5], [0, 300.0, 23.5], [0, 0, 1]])
    w2c = F.rigid(random_rotation(), rng.normal(size=3))
    depth = rng.uniform(1.0, 3.0, size=(48, 64))
    depth[0, 0] = np.nan
    depth[1, 1] = 0.0
    mask = np.ones_like(depth, dtype=bool)
    mask[2, 2] = False
    p, uv = F.unproject_depth(depth, K, w2c, mask)
    assert len(p) == depth.size - 3
    uv2, z = F.project(p, K, w2c)
    assert np.allclose(uv2, uv)
    assert np.allclose(z, depth[uv[:, 1].astype(int), uv[:, 0].astype(int)])


def test_w2c_in_g_matches_w():
    K = np.eye(3)
    w2c = F.rigid(random_rotation(), rng.normal(size=3))
    p_w = rng.normal(size=(10, 3))
    uv_w, z_w = F.project(p_w, K, w2c)
    uv_g, z_g = F.project(F.w_to_g(p_w), K, F.w2c_in_g(w2c))
    assert np.allclose(uv_w, uv_g) and np.allclose(z_w, z_g)


def test_crop_intrinsics_moves_pixels_consistently():
    K = np.array([[400.0, 0, 300.2], [0, 410.0, 200.7], [0, 0, 1]])
    box, out = (100.0, 50.0, 356.0, 306.0), (384, 384)
    K2 = F.crop_intrinsics(K, box, out)
    p = rng.normal(size=(20, 3)) * 0.2 + np.array([0, 0, 3.0])
    uv, _ = F.project(p, K, np.eye(4))
    uv2, _ = F.project(p, K2, np.eye(4))
    sx = out[0] / (box[2] - box[0])
    sy = out[1] / (box[3] - box[1])
    expected = np.stack([(uv[:, 0] + 0.5 - box[0]) * sx - 0.5, (uv[:, 1] + 0.5 - box[1]) * sy - 0.5], -1)
    assert np.allclose(uv2, expected)
