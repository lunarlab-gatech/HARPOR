"""Coordinate frames (build_plan.md note 2).

Frames:
  W     OpenCV frame of input view 0: x right, y down, z forward. Metric if the depth is metric.
  glTF  Prompt 2 maps glTF to W with (x, y, z)_W = (x, -z, y)_glTF, which is also what Blender's
        stock glTF importer does.

Cameras are OpenCV world-to-camera 4x4 matrices acting on column vectors. Pixel centers sit at
integer coordinates, so pixel (0, 0) covers [-0.5, 0.5] x [-0.5, 0.5].

Point arrays have shape (..., 3) and are row vectors, so a rotation R applies as p @ R.T.
"""

import numpy as np

# Prompt 2: (x, y, z)_world = (x, -z, y)_glTF.
W_FROM_GLTF = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
GLTF_FROM_W = W_FROM_GLTF.T


def rotate(R, p):
    return np.asarray(p) @ np.asarray(R).T


def gltf_to_w(p):
    return rotate(W_FROM_GLTF, p)


def w_to_gltf(p):
    return rotate(GLTF_FROM_W, p)


def rigid(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def transform_points(T, p):
    T = np.asarray(T)
    return np.asarray(p) @ T[:3, :3].T + T[:3, 3]


def project(p_world, K, w2c):
    """Project world points with an OpenCV camera. Returns pixel coordinates (N, 2) and depth (N,)."""
    p_cam = transform_points(w2c, p_world)
    z = p_cam[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = (p_cam[..., :2] / z[..., None]) * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])
    return uv, z


def crop_intrinsics(K, box, out_size):
    """Intrinsics after cropping box = (x0, y0, x1, y1) and resizing it to out_size = (w, h).

    Box edges are continuous pixel coordinates, so the crop covers [x0, x1) in edge units, where
    pixel centers sit at integer coordinates.
    """
    x0, y0, x1, y1 = box
    sx = out_size[0] / (x1 - x0)
    sy = out_size[1] / (y1 - y0)
    K2 = np.array(K, dtype=np.float64, copy=True)
    K2[0, 0] *= sx
    K2[1, 1] *= sy
    K2[0, 2] = (K[0, 2] + 0.5 - x0) * sx - 0.5
    K2[1, 2] = (K[1, 2] + 0.5 - y0) * sy - 0.5
    return K2
