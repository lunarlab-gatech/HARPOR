"""Observation anchor in frame G (build_plan.md notes 2 and 3).

The anchor (t, s) centers the 2nd to 98th percentile box of the masked depth points and scales its
longest side times a margin m to 1, so q = (p_G - t) / s puts the object in [-0.5, 0.5]^3. It
never rotates. Masks are eroded by 5 px first, and depth is valid where finite and positive.
"""

import numpy as np
from scipy import ndimage

from gator import frames

ERODE_PX = 5
PERCENTILES = (2.0, 98.0)


def erode(mask):
    """Drop mask pixels within ERODE_PX of the background. The image border is not background."""
    y, x = np.mgrid[-ERODE_PX:ERODE_PX + 1, -ERODE_PX:ERODE_PX + 1]
    return ndimage.binary_erosion(mask, structure=x * x + y * y <= ERODE_PX ** 2, border_value=1)


def observed_points(masks, depths, Ks, w2cs):
    """G-frame points of valid depth inside the eroded masks of all views. w2cs act on W points."""
    p = np.concatenate([frames.unproject_depth(d, K, T, erode(m))[0]
                        for m, d, K, T in zip(masks, depths, Ks, w2cs)])
    if len(p) == 0:
        raise ValueError("no valid depth inside the eroded masks, so the anchor is undefined")
    return frames.w_to_g(p)


def box_frame(lo, hi, margin):
    """(t, s) that center the box [lo, hi] and make its longest side times margin span the unit cube."""
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    side = float(np.max(hi - lo))
    if not side > 0:
        raise ValueError("the points have no extent, so the frame is undefined")
    return (lo + hi) / 2, margin * side


def percentile_frame(points_g, margin):
    """box_frame of the per-axis 2nd to 98th percentile box of the points."""
    lo, hi = np.percentile(points_g, PERCENTILES, axis=0)
    return box_frame(lo, hi, margin)


def anchor(masks, depths, Ks, w2cs, margin):
    """Observation anchor (t_g, s_g) from per-view masks, depth maps, intrinsics and W-frame w2c."""
    return percentile_frame(observed_points(masks, depths, Ks, w2cs), margin)


def pointmap_q(depth, K, w2c, t, s):
    """Anchor-normalized pointmap (H, W, 3) of the whole depth map for the pointmap tokens.

    Pixels with invalid depth are NaN. The mask is not applied, so the scene context stays.
    """
    p, uv = frames.unproject_depth(depth, K, w2c)
    out = np.full(np.shape(depth) + (3,), np.nan)
    out[uv[:, 1].astype(int), uv[:, 0].astype(int)] = frames.normalize(frames.w_to_g(p), t, s)
    return out
