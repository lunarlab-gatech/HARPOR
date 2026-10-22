"""Geometry metrics of paper Appendix C (build_plan.md note 12).

Meshes are compared through seeded, area-weighted surface samples with face normals. Distances are
not squared and are divided by the reference diameter D. In the aggregates a failed output is None
or NaN, and every value comes with the valid rate.
"""

import numpy as np
import trimesh
from scipy.spatial import ConvexHull, QhullError, cKDTree
from scipy.spatial.distance import cdist

from gator.eval.similarity import decompose, icp
from gator.frames import transform_points

N_SAMPLES = 100_000
TAU_TOYS4K = 0.01
TAU_REAL = 0.02
ADD_SB_THRESHOLD = 0.1


def sample_surface(mesh, n=N_SAMPLES, seed=0):
    """Area-weighted surface samples (n, 3) and their unit face normals (n, 3)."""
    p, face = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return p, mesh.face_normals[face]


def directed_distance(a, b):
    """d(A, B): mean distance from the points of A to their nearest point of B. Not squared."""
    return float(cKDTree(b).query(a, workers=-1)[0].mean())


def chamfer(a, b, D):
    """CD(A, B) = (d(A, B) + d(B, A)) / (2 D)."""
    return (directed_distance(a, b) + directed_distance(b, a)) / (2 * D)


def surface_scores(pa, na, pb, nb, D, tau):
    """CD, NC and F1 at tau * D between prediction samples A and reference samples B, as placed."""
    da, ia = cKDTree(pb).query(pa, workers=-1)
    db, ib = cKDTree(pa).query(pb, workers=-1)
    nc = 0.5 * (np.abs(np.sum(na * nb[ia], 1)).mean() + np.abs(np.sum(nb * na[ib], 1)).mean())
    precision, recall = np.mean(da < tau * D), np.mean(db < tau * D)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"cd": (da.mean() + db.mean()) / (2 * D), "nc": nc, "f1": f1}


def add_sb(pred, ref, D, n=N_SAMPLES, seed=0):
    """ADD-SB: CD between the posed prediction and the posed reference, with no fitting."""
    return chamfer(sample_surface(pred, n, seed)[0], sample_surface(ref, n, seed)[0], D)


def _move(T, p, normals):
    return transform_points(T, p), normals @ decompose(T)[1].T


def shape_metrics(pred, ref, D, tau, init=np.eye(4), n=N_SAMPLES, seed=0):
    """Similarity ICP from the prediction to the reference, then CD, NC and F1 (paper C.2).

    The ICP matches points in both directions. init is a first guess of the similarity that aligns the prediction. The result holds the
    scores and T, the fitted similarity, which the IoU and appearance renders reuse.
    """
    pa, na = sample_surface(pred, n, seed)
    pb, nb = sample_surface(ref, n, seed)
    T = icp(pa, pb, init)
    return {**surface_scores(*_move(T, pa, na), pb, nb, D, tau), "T": T}


def shape_metrics_observed(pred, observed, D, tau=TAU_REAL, init=np.eye(4), n=N_SAMPLES, seed=0):
    """ScanNet++ variant: ICP fits only the observed surface onto the prediction, so completed parts
    that the scan missed do not pull the fit. CD and F1 stay symmetric, so those parts are penalized.
    """
    pa, na = sample_surface(pred, n, seed)
    pb, nb = sample_surface(observed, n, seed)
    T = np.linalg.inv(icp(pb, pa, np.linalg.inv(init), symmetric=False))
    return {**surface_scores(*_move(T, pa, na), pb, nb, D, tau), "T": T}


def diameter(points):
    """Largest distance between two points, exact over the convex-hull vertices.

    On CAD vertices this is the BOP CAD diameter. On an observed mesh it is the ScanNet++ D.
    """
    p = np.asarray(points, dtype=np.float64)
    try:
        h = p[ConvexHull(p).vertices]
    except QhullError:  # flat or collinear points: joggle, but keep the original coordinates
        h = p[ConvexHull(p, qhull_options="QJ").vertices]
    return float(max(cdist(h[i : i + 2048], h).max() for i in range(0, len(h), 2048)))


def _valid(values):
    v = np.asarray(values, dtype=np.float64)
    return v, np.isfinite(v)


def mean_valid(values):
    """Mean over the valid outputs, and the valid rate."""
    v, ok = _valid(values)
    return (float(v[ok].mean()) if ok.any() else float("nan")), float(ok.mean())


def iou_mean(values):
    """Mean silhouette IoU with a failed output scored as 0, and the valid rate."""
    v, ok = _valid(values)
    return float(np.where(ok, v, 0.0).mean()), float(ok.mean())


def add_sb_recall(values, threshold=ADD_SB_THRESHOLD):
    """ADD-SB@0.1: share of outputs with ADD-SB below threshold, a failure counted as a miss. Also
    returns the valid rate."""
    v, ok = _valid(values)
    return float(np.mean(ok & (v < threshold))), float(ok.mean())
