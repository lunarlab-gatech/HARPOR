"""Similarity fits for scoring (build_plan.md note 12, paper Appendix C).

A similarity is a 4x4 matrix [[s R, t], [0, 1]] with s > 0 and det(R) = +1. It acts on row-vector
points through frames.transform_points: p -> s p @ R.T + t. No fit here returns a reflection.
"""

import numpy as np
from scipy.spatial import cKDTree

from gator.frames import transform_points


def compose(s, R, t):
    T = np.eye(4)
    T[:3, :3] = s * np.asarray(R)
    T[:3, 3] = t
    return T


def decompose(T):
    """Scale, rotation and translation of a similarity."""
    T = np.asarray(T)
    s = np.cbrt(np.linalg.det(T[:3, :3]))
    return s, T[:3, :3] / s, T[:3, 3].copy()


def nearest_rotation(M):
    """Proper rotation closest to M. Where the closest orthogonal matrix is a reflection, the
    weakest singular direction is flipped instead (the Umeyama sign fix)."""
    U, _, Vt = np.linalg.svd(M)
    U[:, 2] *= np.sign(np.linalg.det(U @ Vt))
    return U @ Vt


def umeyama(src, dst):
    """Least-squares similarity T with dst ~ T(src) for matched points (Umeyama 1991)."""
    src, dst = np.asarray(src, dtype=np.float64), np.asarray(dst, dtype=np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    x = src - mu_s
    cov = (dst - mu_d).T @ x
    R = nearest_rotation(cov)
    s = np.sum(R * cov) / np.sum(x**2)
    return compose(s, R, mu_d - s * R @ mu_s)


def icp(src, dst, init=np.eye(4), symmetric=True, max_iter=50, tol=1e-6):
    """Point-to-point similarity ICP that moves src onto dst, starting from init.

    Each step matches every src point to its nearest dst point and, if symmetric, every dst point
    to its nearest moved src point, then refits umeyama on all pairs. One-way matching lets the
    scale shrink src onto the part of dst nearest to it. It stops when the mean squared match
    distance improves by less than tol, relative.
    """
    src, dst = np.asarray(src, dtype=np.float64), np.asarray(dst, dtype=np.float64)
    tree = cKDTree(dst)

    def match(T):
        moved = transform_points(T, src)
        d, i = tree.query(moved, workers=-1)
        if not symmetric:
            return np.mean(d**2), src, dst[i]
        e, j = cKDTree(moved).query(dst, workers=-1)
        return np.mean(np.concatenate([d, e]) ** 2), np.vstack([src, src[j]]), np.vstack([dst[i], dst])

    T = np.asarray(init, dtype=np.float64)
    err, a, b = match(T)
    for _ in range(max_iter):
        T = umeyama(a, b)
        prev, (err, a, b) = err, match(T)
        if prev - err <= tol * prev:
            break
    return T


def camera_similarity(c_est, c_ref, R_est=None, R_ref=None):
    """Similarity that maps the estimated reconstruction frame onto the reference frame (paper C.1).

    c_est and c_ref are (V, 3) camera centers of the same views. R_est and R_ref are optional
    (V, 3, 3) world-to-camera rotations. Without them this is umeyama on the centers, which needs 3
    views that are not collinear. With them, R is the chordal mean of R_ref_i^T R_est_i, and s and t
    fit the centers given R, which needs 2 views.
    """
    c_est, c_ref = np.asarray(c_est, dtype=np.float64), np.asarray(c_ref, dtype=np.float64)
    if R_est is None:
        return umeyama(c_est, c_ref)
    R = nearest_rotation(np.einsum("vji,vjk->ik", R_ref, R_est))
    x, y = c_est - c_est.mean(0), c_ref - c_ref.mean(0)
    s = np.sum(y * (x @ R.T)) / np.sum(x**2)
    return compose(s, R, c_ref.mean(0) - s * R @ c_est.mean(0))
