"""Boxes and seams of the parts of a split mesh, in frame W (build_plan.md note 11).

aabb and obb describe one part's points. seams lists the edges where two parts meet: for every pair of
parts that share edges, their count, length and center, up to 4 straight lines fitted to them (the
hinge-axis candidates of the agent) and the mean normal of the faces on each side.

The mesh of a textured GLB has its vertices split along the texture seams, so its faces do not share
edges there. topology() merges coincident vertices first, keeping the order of the faces.
"""

import numpy as np


def topology(vertices, faces):
    """A trimesh.Trimesh of the faces with coincident vertices merged, so face_adjacency also crosses
    texture seams. Face i is still face i."""
    import trimesh

    mesh = trimesh.Trimesh(np.asarray(vertices, dtype=float), np.asarray(faces), process=False)
    mesh.merge_vertices()
    return mesh


def pieces(mesh, label, n_labels):
    """[(pieces, largest)] per label 0 to n_labels - 1 of the faces of mesh (topology): how many connected pieces
    the label's faces form, by shared edges, and the share of its area in the largest one. A part of two or more
    sizeable pieces often holds the surface of two things. A label without faces has (0, 0.0)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    label = np.asarray(label, int)
    adj, area, F = np.asarray(mesh.face_adjacency).reshape(-1, 2), np.asarray(mesh.area_faces, float), len(label)
    same = label[adj[:, 0]] == label[adj[:, 1]]
    g = coo_matrix((np.ones(int(same.sum())), (adj[same, 0], adj[same, 1])), shape=(F, F))
    n, comp = connected_components(g, directed=False)
    comp_area = np.bincount(comp, weights=area, minlength=n)
    out = []
    for k in range(n_labels):
        ids = np.unique(comp[label == k])
        total = float(comp_area[ids].sum())
        out.append((len(ids), float(comp_area[ids].max() / total) if len(ids) and total > 0 else 0.0))
    return out


def aabb(points):
    p = np.asarray(points, dtype=float).reshape(-1, 3)
    lo, hi = p.min(0), p.max(0)
    return {"min": lo.tolist(), "max": hi.tolist(), "center": ((lo + hi) / 2).tolist(), "extent": (hi - lo).tolist()}


def _canonical(axes):
    """Rows as unit vectors whose largest component is positive, then made right-handed."""
    axes = np.array(axes, dtype=float)
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    for a in axes:
        if a[np.argmax(np.abs(a))] < 0:
            a *= -1
    if np.linalg.det(axes) < 0:
        axes[2] *= -1
    return axes


def obb(points):
    """Oriented box (trimesh.bounds.oriented_bounds): center, axes (rows, longest first), extents along
    them and the 8 corners. Falls back to the axis-aligned box when the points are too few or flat."""
    import trimesh

    p = np.asarray(points, dtype=float).reshape(-1, 3)
    try:
        T, ext = trimesh.bounds.oriented_bounds(p)
        M = np.linalg.inv(T)
        axes, center, ext = M[:3, :3].T, M[:3, 3], np.asarray(ext, dtype=float)
        if not (np.isfinite(axes).all() and np.isfinite(center).all() and np.isfinite(ext).all()):
            raise ValueError("non-finite box")
    except Exception:
        lo, hi = p.min(0), p.max(0)
        axes, center, ext = np.eye(3), (lo + hi) / 2, hi - lo
    order = np.argsort(-ext, kind="stable")
    axes, ext = _canonical(axes[order]), ext[order]
    signs = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1], indexing="ij")).reshape(3, -1).T
    corners = center + (signs * ext / 2) @ axes
    return {"center": center.tolist(), "axes": axes.tolist(), "extents": ext.tolist(), "corners": corners.tolist()}


def _direction(u):
    u = np.asarray(u, dtype=float) / np.linalg.norm(u)
    return u if u[np.argmax(np.abs(u))] > 0 else -u


def fit_lines(a, b, max_lines=4, tol=None, min_share=0.05, pairs=400, seed=0, inliers=False):
    """Greedy RANSAC of straight lines on edges with endpoints a, b (N, 3). An edge is held by a line
    when both its ends lie within tol of it, and a line scores the held length weighted by
    1 - (distance / tol)^2 (MSAC), so a line that runs exactly along the edges wins over a slanted one
    that holds a few more. The best line through two edge midpoints is refined by PCA of the held
    endpoints; its edges are then removed and the next line found, up to max_lines, while a line holds
    at least min_share of the length. tol defaults to 2% of the edges' bounding diagonal or half the
    median edge length, whichever is larger. Deterministic. Returns [{p0, p1, direction, length,
    inlier_share}], and with inliers=True also the held edge indices of each line."""
    a, b = np.asarray(a, dtype=float).reshape(-1, 3), np.asarray(b, dtype=float).reshape(-1, 3)
    L = np.linalg.norm(b - a, axis=1)
    total = float(L.sum())
    if len(a) == 0 or total <= 0:
        return ([], []) if inliers else []
    if tol is None:
        tol = max(0.02 * float(np.linalg.norm(np.ptp(np.concatenate([a, b]), 0))), 0.5 * float(np.median(L)))
    mid = (a + b) / 2
    rng = np.random.default_rng(seed)
    left = np.ones(len(a), bool)

    def score(c, u, idx):
        da = np.linalg.norm((a[idx] - c) - np.outer((a[idx] - c) @ u, u), axis=1)
        db = np.linalg.norm((b[idx] - c) - np.outer((b[idx] - c) @ u, u), axis=1)
        d = np.maximum(da, db)
        h = d <= tol
        return h, float((L[idx] * np.where(h, 1 - (d / tol) ** 2, 0.0)).sum())

    lines, held = [], []
    while len(lines) < max_lines and left.any() and L[left].sum() > min_share * total:
        idx = np.flatnonzero(left)
        if len(idx) == 1:
            cand = [(idx[0], idx[0])]
        elif len(idx) <= 30:
            cand = [(i, j) for k, i in enumerate(idx) for j in idx[k + 1:]]
        else:
            cand = [tuple(rng.choice(idx, 2, replace=False)) for _ in range(pairs)]
        best, best_s = None, 0.0
        for i, j in cand:
            d = mid[j] - mid[i] if i != j else b[i] - a[i]
            if np.linalg.norm(d) < 1e-12:
                continue
            u = d / np.linalg.norm(d)
            s = score(mid[i], u, idx)[1]
            if s > best_s + 1e-12:
                best, best_s = (mid[i], u), s
        if best is None:
            break
        h, _ = score(best[0], best[1], idx)
        inl = idx[h]
        pts, wts = np.concatenate([a[inl], b[inl]]), np.concatenate([L[inl], L[inl]])
        c = (pts * wts[:, None]).sum(0) / wts.sum()
        _, _, vt = np.linalg.svd((pts - c) * np.sqrt(wts)[:, None], full_matrices=False)
        u = _direction(vt[0])
        h2, s2 = score(c, u, idx)
        if s2 >= best_s:  # keep the refined line when it scores no less
            inl = idx[h2]
        if L[inl].sum() < min_share * total:
            break
        pts = np.concatenate([a[inl], b[inl]])
        c = pts.mean(0)
        t = (pts - c) @ u
        lines.append({"p0": (c + t.min() * u).tolist(), "p1": (c + t.max() * u).tolist(), "direction": u.tolist(),
                      "length": float(t.max() - t.min()), "inlier_share": round(float(L[inl].sum() / total), 4)})
        held.append(inl)
        left[inl] = False
    return (lines, held) if inliers else lines


def _mean_normal(N, area, f):
    n = (N[f] * area[f, None]).sum(0)
    return (n / np.linalg.norm(n)).tolist() if np.linalg.norm(n) > 1e-12 else [0.0, 0.0, 0.0]


def seams(mesh, part_ids, names=None, max_lines=4):
    """Every pair of parts that share edges in mesh (a topology() mesh) under part_ids (one per face):
    [{parts, edges, length, center, lines, normals}], sorted by the pair. names maps ids to names
    (default: the ids). normals holds each side's area-weighted mean face normal along the seam, and
    each line has the same for its own edges (along a closed seam, such as a lid's, the seam's own
    normals cancel out)."""
    part_ids = np.asarray(part_ids)
    label = (lambda k: names[k]) if names is not None else (lambda k: k)
    adj, adj_e = np.asarray(mesh.face_adjacency), np.asarray(mesh.face_adjacency_edges)
    if len(adj) == 0:
        return []
    cross = part_ids[adj[:, 0]] != part_ids[adj[:, 1]]
    adj, adj_e = adj[cross], adj_e[cross]
    pa, pb = part_ids[adj[:, 0]], part_ids[adj[:, 1]]
    lo, hi = np.minimum(pa, pb), np.maximum(pa, pb)
    V, N, area = np.asarray(mesh.vertices), np.asarray(mesh.face_normals), np.asarray(mesh.area_faces)
    out = []
    for p, q in sorted({(int(x), int(y)) for x, y in zip(lo, hi)}):
        rows = np.flatnonzero((lo == p) & (hi == q))
        _, first = np.unique(np.sort(adj_e[rows], axis=1), axis=0, return_index=True)
        rows = rows[np.sort(first)]  # one row per edge (a non-manifold edge has more)
        a, b = V[adj_e[rows, 0]], V[adj_e[rows, 1]]
        L = np.linalg.norm(b - a, axis=1)
        f_p = np.where(part_ids[adj[rows, 0]] == p, adj[rows, 0], adj[rows, 1])  # the face on p's side
        f_q = np.where(part_ids[adj[rows, 0]] == p, adj[rows, 1], adj[rows, 0])
        lines, held = fit_lines(a, b, max_lines, inliers=True)
        for line, k in zip(lines, held):
            line["normals"] = {str(label(p)): _mean_normal(N, area, f_p[k]), str(label(q)): _mean_normal(N, area, f_q[k])}
        center = ((a + b) / 2 * L[:, None]).sum(0) / max(L.sum(), 1e-12)
        out.append({"parts": [label(p), label(q)], "edges": int(len(rows)), "length": float(L.sum()),
                    "center": center.tolist(), "lines": lines,
                    "normals": {str(label(p)): _mean_normal(N, area, f_p), str(label(q)): _mean_normal(N, area, f_q)}})
    return out
