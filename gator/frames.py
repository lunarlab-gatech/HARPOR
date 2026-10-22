"""Coordinate frames for the GATOR clone (build_plan.md note 2).

Frames:
  W     OpenCV frame of input view 0: x right, y down, z forward. Metric if the depth is metric.
  G     Generation frame: (x, y, z)_G = (x, z, -y)_W. x right, y forward, z up, as in TRELLIS.2.
  q     Anchor-normalized G: q = (p_G - t) / s. TRELLIS.2 decodes meshes in [-0.5, 0.5]^3 of q.
  glTF  Numerically equal to G. Prompt 2 maps glTF to W with (x, y, z)_W = (x, -z, y)_glTF,
        which is also what Blender's stock glTF importer does.

Cameras are OpenCV world-to-camera 4x4 matrices acting on column vectors. Pixel centers sit at
integer coordinates, so pixel (0, 0) covers [-0.5, 0.5] x [-0.5, 0.5].

Point arrays have shape (..., 3) and are row vectors, so a rotation R applies as p @ R.T.
"""

import numpy as np

G_FROM_W = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
W_FROM_G = G_FROM_W.T

# Prompt 2: (x, y, z)_world = (x, -z, y)_glTF.
W_FROM_GLTF = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
GLTF_FROM_W = W_FROM_GLTF.T

# TRELLIS.2 o_voxel.postprocess.to_glb maps (x, y, z) to (x, z, -y), a -90 degree turn about X.
TO_GLB_ROTATION = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


def rotate(R, p):
    return np.asarray(p) @ np.asarray(R).T


def w_to_g(p):
    return rotate(G_FROM_W, p)


def g_to_w(p):
    return rotate(W_FROM_G, p)


def gltf_to_w(p):
    return rotate(W_FROM_GLTF, p)


def w_to_gltf(p):
    return rotate(GLTF_FROM_W, p)


def g_to_gltf(p):
    """G and glTF are the same numbers. Write G-frame vertices into the GLB as they are."""
    return np.array(p, copy=True)


def to_glb_input_from_g(p_g):
    """Points to pass to TRELLIS.2 to_glb so that its rotation lands them back on G."""
    return rotate(TO_GLB_ROTATION.T, p_g)


def normalize(p_g, t, s):
    """Anchor normalization (paper Eq. 3). Translates and scales, never rotates."""
    return (np.asarray(p_g) - np.asarray(t)) / s


def denormalize(q, t, s):
    return np.asarray(q) * s + np.asarray(t)


def rigid(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def invert_rigid(T):
    T = np.asarray(T)
    R, t = T[:3, :3], T[:3, 3]
    return rigid(R.T, -R.T @ t)


def transform_points(T, p):
    T = np.asarray(T)
    return np.asarray(p) @ T[:3, :3].T + T[:3, 3]


def rebase_to_view0(w2c):
    """Re-express world-to-camera matrices (V, 4, 4) so that view 0 becomes the world frame W."""
    w2c = np.asarray(w2c)
    return w2c @ invert_rigid(w2c[0])


def w2c_in_g(w2c_w):
    """World-to-camera matrices that take G-frame points instead of W-frame points."""
    return np.asarray(w2c_w) @ rigid(W_FROM_G, np.zeros(3))


def project(p_world, K, w2c):
    """Project world points with an OpenCV camera. Returns pixel coordinates (N, 2) and depth (N,)."""
    p_cam = transform_points(w2c, p_world)
    z = p_cam[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = (p_cam[..., :2] / z[..., None]) * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])
    return uv, z


def unproject_depth(depth, K, w2c, mask=None):
    """Back-project a depth map: p_cam = z K^-1 [u, v, 1], p_world = inv(w2c) p_cam.

    Depth is valid where it is finite and positive (and inside the mask, if one is given).
    Returns world points (N, 3) and their pixel coordinates (N, 2).
    """
    depth = np.asarray(depth, dtype=np.float64)
    valid = np.isfinite(depth) & (depth > 0)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    v, u = np.nonzero(valid)
    z = depth[v, u]
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    p_cam = np.stack([x, y, z], axis=-1)
    return transform_points(invert_rigid(w2c), p_cam), np.stack([u, v], axis=-1).astype(np.float64)


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
