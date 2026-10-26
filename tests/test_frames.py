import numpy as np

from gator import frames as F

rng = np.random.default_rng(0)


def test_gltf_w_roundtrip():
    assert np.allclose(F.gltf_to_w([1.0, 2.0, 3.0]), [1, -3, 2])  # (x, y, z)_W = (x, -z, y)_glTF
    p_w = rng.normal(size=(50, 3))
    assert np.allclose(F.gltf_to_w(F.w_to_gltf(p_w)), p_w)
    assert np.allclose(F.W_FROM_GLTF @ F.GLTF_FROM_W, np.eye(3))


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
