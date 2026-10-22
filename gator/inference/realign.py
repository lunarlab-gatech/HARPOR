"""Re-alignment of the generated structure (build_plan.md note 5, research.md section 4.5).

Cleans the decoded 64^3 occupancy, fits its translation and isotropic scale to the observed G
points, recomputes the normalization from the occupied box and re-voxelizes to 32^3. Grid axes
index (x, y, z) of q, and cell i of a res grid has center (i + 0.5) / res - 0.5.
"""

import itertools

import numpy as np
from scipy import ndimage

from gator import frames
from gator.inference.anchor import box_frame

MIN_COMPONENT = 0.05  # components smaller than this fraction of the largest are floaters
TRUNC = 0.1  # the fit ignores points farther than this from the structure, in anchor q units
SCALE_PRIOR = 1e-5  # weight of (log scale change)^2, so scale the points cannot see stays as generated
MIN_GAIN = 0.1  # the fit is kept only if it cuts the mean truncated distance by this fraction
FIT_ITERS = 30
MAX_POINTS = 20000
BOX_MARGIN = 1 / 0.99999  # TRELLIS.2 dual_grid.py scales by 0.99999 / longest side


def clean_occupancy(occ):
    """Keep the largest 26-connected component and every component of at least MIN_COMPONENT of it."""
    labels, n = ndimage.label(occ, structure=np.ones((3, 3, 3)))
    if n == 0:
        raise ValueError("the decoded occupancy is empty")
    sizes = np.bincount(labels.ravel())[1:]
    return np.isin(labels, 1 + np.flatnonzero(sizes >= MIN_COMPONENT * sizes.max()))


def centers(occ):
    """q coordinates of the occupied cell centers."""
    return (np.argwhere(occ) + 0.5) / occ.shape[0] - 0.5


def trilinear(f, g):
    """Value and gradient of the trilinear interpolant of the grid f at grid coordinates g (N, 3)."""
    i = np.clip(np.floor(g).astype(int), 0, np.array(f.shape) - 2)
    w1 = g - i
    val, grad = np.zeros(len(g)), np.zeros_like(w1)
    for k in itertools.product((0, 1), repeat=3):
        w = np.where(k, w1, 1 - w1)
        fk = f[tuple((i + k).T)]
        val += w.prod(1) * fk
        dw = np.stack([w[:, 1] * w[:, 2], w[:, 0] * w[:, 2], w[:, 0] * w[:, 1]], 1)
        grad += (2 * np.array(k) - 1) * dw * fk[:, None]
    return val, grad


def fit(occ, t, s, points_g):
    """Fit the G frame of occ, given in q of the anchor (t, s), to the observed points.

    The one-sided distance from a point to the structure is the trilinear interpolant of the
    distance transform of the boundary voxel centers, so it is exact on faces through those centers
    and has no lattice ripple. Gauss-Newton runs over a shift and a log-scale about the median
    point (no rotation), ignores points beyond TRUNC or outside the grid, and adds SCALE_PRIOR.
    Returns the fitted (t, s), or the anchor if the fit cuts the mean truncated distance by less
    than MIN_GAIN.
    """
    res = occ.shape[0]
    dist = ndimage.distance_transform_edt(~(occ & ~ndimage.binary_erosion(occ))) / res
    a = frames.normalize(points_g[::int(np.ceil(len(points_g) / MAX_POINTS))], t, s)
    o = np.median(a, axis=0)

    def residuals(x):
        """Distances (inf outside the grid), their gradients in structure q, and the points there."""
        e = np.exp(x[3])
        c = o + (a - x[:3] - o) / e
        g = (c + 0.5) * res - 0.5
        d, grad = trilinear(dist, g)
        return np.where(((g >= 0) & (g <= res - 1)).all(1), e * d, np.inf), grad * res, c

    def cost(x):
        return np.minimum(residuals(x)[0], TRUNC).mean()

    x, prior = np.zeros(4), np.diag([0.0, 0.0, 0.0, SCALE_PRIOR])
    for _ in range(FIT_ITERS):
        r, grad, c = residuals(x)
        k = r < TRUNC
        J = np.c_[-grad[k], r[k] - np.exp(x[3]) * (grad[k] * (c[k] - o)).sum(1)]
        x = x - np.linalg.lstsq(J.T @ J / len(r) + prior, J.T @ r[k] / len(r) + prior @ x, rcond=None)[0]
    if cost(x) < (1 - MIN_GAIN) * cost(np.zeros(4)):
        e = np.exp(x[3])
        return t + s * (x[:3] + (1 - e) * o), s * e
    return t, s


def normalization(points_g):
    """(t, s) of the full bounding box of points in G. Re-alignment and the training targets share it."""
    points_g = np.asarray(points_g)
    return box_frame(points_g.min(0), points_g.max(0), BOX_MARGIN)


def revoxelize(occ, t_occ, s_occ, t, s):
    """Splat the cells of occ (frame t_occ, s_occ) into a grid of the same size in the frame (t, s).

    Each cell is sampled n^3 times, with samples at most half a new cell apart. Samples outside the
    cube are dropped.
    """
    res = occ.shape[0]
    n = max(2, int(np.ceil(2 * s_occ / s)))
    off = ((np.arange(n) + 0.5) / n - 0.5) / res
    off = np.stack(np.meshgrid(off, off, off, indexing="ij"), -1).reshape(-1, 3)
    p = frames.denormalize((centers(occ)[:, None] + off).reshape(-1, 3), t_occ, s_occ)
    idx = np.floor((frames.normalize(p, t, s) + 0.5) * res).astype(int)
    out = np.zeros_like(occ, dtype=bool)
    out[tuple(idx[((idx >= 0) & (idx < res)).all(1)].T)] = True
    return out


def max_pool(occ):
    r = occ.shape[0] // 2
    return occ.reshape(r, 2, r, 2, r, 2).any(axis=(1, 3, 5))


def realign(occ, t, s, points_g):
    """Note 5 for a 64^3 occ in q of the anchor (t, s). Returns the new (t, s) and the 32^3 grid.

    The new box spans the occupied cells, not only their centers, so the whole structure stays inside.
    """
    occ = clean_occupancy(occ)
    t_fit, s_fit = fit(occ, t, s, points_g)
    c, h = centers(occ), 0.5 / occ.shape[0]
    box = frames.denormalize(np.stack([c.min(0) - h, c.max(0) + h]), t_fit, s_fit)
    t_new, s_new = normalization(box)
    return t_new, s_new, max_pool(revoxelize(occ, t_fit, s_fit, t_new, s_new))
