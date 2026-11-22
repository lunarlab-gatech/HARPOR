"""Coordinate frames (README.md note 1).

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
