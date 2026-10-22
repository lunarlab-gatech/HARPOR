"""Crop windows for the input-view rows of the review sheet (build_plan.md note 5).

Plain numpy, so review.py (inside Blender) and the tests share it. Pixel centers sit at integer
coordinates, as in the manifest's K.
"""

import numpy as np


def project(points, K, w2c):
    """Pixel coordinates (u, v) of the world points in front of the camera."""
    p = np.asarray(points, float) @ np.asarray(w2c, float)[:3, :3].T + np.asarray(w2c, float)[:3, 3]
    p = p[p[:, 2] > 1e-9]
    uv = p @ np.asarray(K, float).T
    return uv[:, :2] / uv[:, 2:3]


def crop_window(mask, uv, width, height, margin=0.08, min_size=32):
    """Integer window (u0, v0, u1, v1), end exclusive, that holds the mask's box and the projected
    points, grown by margin of its long side and clipped to the image. The whole image if both are empty."""
    boxes = []
    if mask is not None and mask.any():
        rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
        boxes.append([cols[0], rows[0], cols[-1] + 1, rows[-1] + 1])
    uv = np.asarray(uv, float).reshape(-1, 2)
    uv = uv[np.isfinite(uv).all(1)]
    if len(uv):
        boxes.append([np.floor(uv[:, 0].min() + 0.5), np.floor(uv[:, 1].min() + 0.5),
                      np.floor(uv[:, 0].max() + 0.5) + 1, np.floor(uv[:, 1].max() + 0.5) + 1])
    if not boxes:
        return 0, 0, width, height
    b = np.array(boxes, float)
    u0, v0, u1, v1 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    pad = max(margin * max(u1 - u0, v1 - v0), (min_size - min(u1 - u0, v1 - v0)) / 2, 0)
    u0, v0 = max(int(np.floor(u0 - pad)), 0), max(int(np.floor(v0 - pad)), 0)
    u1, v1 = min(int(np.ceil(u1 + pad)), width), min(int(np.ceil(v1 + pad)), height)
    if u1 <= u0 or v1 <= v0:
        return 0, 0, width, height
    return u0, v0, u1, v1


def crop_camera(K, window, long_side):
    """K, width and height that render the window at long_side pixels on its long side.

    A window pixel u maps to (u - u0 + 0.5) * s - 0.5, so fx scales by s and cx moves the same way.
    """
    u0, v0, u1, v1 = window
    s = long_side / max(u1 - u0, v1 - v0)
    K = np.array(K, float)
    K2 = K.copy()
    K2[0, 0], K2[1, 1] = K[0, 0] * s, K[1, 1] * s
    K2[0, 2], K2[1, 2] = (K[0, 2] - u0 + 0.5) * s - 0.5, (K[1, 2] - v0 + 0.5) * s - 0.5
    return K2, max(round((u1 - u0) * s), 1), max(round((v1 - v0) * s), 1), s
