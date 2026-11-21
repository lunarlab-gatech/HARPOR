import numpy as np

from harpor import frames as F

rng = np.random.default_rng(0)


def test_gltf_w_roundtrip():
    assert np.allclose(F.gltf_to_w([1.0, 2.0, 3.0]), [1, -3, 2])  # (x, y, z)_W = (x, -z, y)_glTF
    p_w = rng.normal(size=(50, 3))
    assert np.allclose(F.gltf_to_w(F.w_to_gltf(p_w)), p_w)
    assert np.allclose(F.W_FROM_GLTF @ F.GLTF_FROM_W, np.eye(3))
