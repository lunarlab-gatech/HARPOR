"""The two bounded LiDAR sculpt operations, in plain numpy (README.md note 5, LiDAR geometry).

lidar_sculpt.py runs them inside Blender on the agent's candidate.blend, and the tests run them on trimesh meshes. Blender's
bundled Python has numpy but not scipy, so this file uses numpy and the standard library only. Every position is in W
(Blender world coordinates); the caller maps the result back into each object's local coordinates.

  resolve  finds the patch in the current mesh from a review region's seed, normal and bounds: the connected faces that
           face the region's way, grown from the face nearest the seed across edges that bend less than a crease. Face
           indices of a former export are not used, since a GLB reimport, a material split or a triangulation renumbers
           them. A second surface of the same facing as near the seed makes the patch ambiguous.
  plane    fits a robust plane to the measured endpoints (each view weighs the same) and moves the patch's interior
           vertices toward it, the rim pinned and a smooth transition near it.
  push     takes the normal part of the measured displacement, dot(p - h, n), smooths it over the patch and moves each
           interior vertex along its normal, tapered toward the boundary.

Both pin every vertex that a face outside the patch also uses (the rim, seams with other surfaces and attachments) and
move coincident vertices (a UV seam) as one. A pass moves no vertex farther than the least of max_displacement, 1.5
times the measured discrepancy and a quarter of the patch's size. A result whose positions are not finite or that
flips a face is refused.
"""

import heapq
import math

import numpy as np

CREASE_DEG = 35.0
FACING_DEG = 40.0
MIN_MOVABLE = 4


class SculptError(Exception):
    """The operation cannot run: args[0] says why, for the agent."""


def weld(V, eps):
    """(group of each vertex, position of each group): vertices within eps of each other's grid cell are one."""
    key = np.round(np.asarray(V, float) / eps).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    G = np.zeros((inv.max() + 1, 3))
    np.add.at(G, inv, V)
    G /= np.bincount(inv)[:, None]
    return inv, G


def triangles(V, F):
    """(unit normals, centroids, areas) of the triangles F of V."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cr = np.cross(b - a, c - a)
    area = np.linalg.norm(cr, axis=1) / 2
    return cr / np.maximum(2 * area, 1e-300)[:, None], (a + b + c) / 3, area


def edges(Fg):
    """(sorted welded edges (E, 2), faces of each edge [list]) of triangles in welded vertex groups."""
    e = np.sort(np.concatenate([Fg[:, [0, 1]], Fg[:, [1, 2]], Fg[:, [2, 0]]]), axis=1)
    face = np.tile(np.arange(len(Fg)), 3)
    uniq, inv = np.unique(e, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    order = np.argsort(inv, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(inv[order])) + 1, len(order)]
    return uniq, [face[order[starts[k]:starts[k + 1]]] for k in range(len(uniq))]


def _components(nodes, pairs):
    """Connected components of nodes (a list) under pairs: {node: component}."""
    adj = {n: [] for n in nodes}
    for a, b in pairs:
        adj[a].append(b)
        adj[b].append(a)
    comp, k = {}, 0
    for n in nodes:
        if n in comp:
            continue
        stack = [n]
        comp[n] = k
        while stack:
            x = stack.pop()
            for y in adj[x]:
                if y not in comp:
                    comp[y] = k
                    stack.append(y)
        k += 1
    return comp


def resolve(V, F, seed, normal, bounds, *, facing_deg=FACING_DEG, crease_deg=CREASE_DEG):
    """The patch of a review region in the mesh (V, F in W): {faces, gid, G, edges, normal, size, spacing}. SculptError
    when no face matches or when two surfaces match about equally (ambiguous)."""
    V, F = np.asarray(V, float), np.asarray(F, np.int64)
    lo, hi = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
    diag = float(np.linalg.norm(hi - lo)) or 1e-6
    span = float(np.linalg.norm(V.max(0) - V.min(0))) or 1.0
    gid, G = weld(V, span * 1e-6)
    Fg = gid[F]
    nf, cf, af = triangles(V, F)
    normal = np.asarray(normal, float) / np.linalg.norm(normal)
    margin = 0.05 * diag + 1e-9
    inside = ((cf >= lo - margin) & (cf <= hi + margin)).all(1)
    cand = np.flatnonzero(inside & (nf @ normal > math.cos(math.radians(facing_deg))) & (af > 0))
    if not len(cand):
        raise SculptError("no face of the current mesh faces the region's normal inside its bounds; the surface moved "
                          "or was removed since the review")
    E, faces_of = edges(Fg)
    is_cand = np.zeros(len(F), bool)
    is_cand[cand] = True
    cos_crease = math.cos(math.radians(crease_deg))
    pairs = []
    for fs in faces_of:
        fs = [f for f in fs if is_cand[f]]
        for i in range(len(fs)):
            for j in range(i + 1, len(fs)):
                if nf[fs[i]] @ nf[fs[j]] > cos_crease:
                    pairs.append((int(fs[i]), int(fs[j])))
    comp = _components([int(f) for f in cand], pairs)
    seed = np.asarray(seed, float)
    dist = np.linalg.norm(cf[cand] - seed, axis=1)
    best = int(cand[np.argmin(dist)])
    mine = [f for f in cand if comp[int(f)] == comp[best]]
    area = af[mine].sum()
    edge_len = np.linalg.norm(G[E[:, 0]] - G[E[:, 1]], axis=1)
    spacing = float(np.median(edge_len)) if len(edge_len) else diag
    near = {}
    for f, d in zip(cand, dist):
        c = comp[int(f)]
        if c != comp[best]:
            near[c] = min(near.get(c, np.inf), d)
    d0 = float(dist.min())
    for c, d in near.items():
        other = af[[f for f in cand if comp[int(f)] == c]].sum()
        if other > 0.2 * area and d - d0 < max(2 * spacing, 0.02 * diag):
            raise SculptError(f"ambiguous patch: two separate surfaces face the region's normal near its seed "
                              f"({d0:.4f} m and {d:.4f} m away); review with a detail view and pick a seed on one")
    faces = np.array(sorted(mine), np.int64)
    pts = G[np.unique(Fg[faces])]
    nrm = nf[faces].T @ af[faces]
    nrm /= np.linalg.norm(nrm)
    return {"faces": faces, "gid": gid, "G": G, "Fg": Fg, "edges": E, "normal": nrm,
            "size": float(np.linalg.norm(pts.max(0) - pts.min(0))), "spacing": spacing, "area": float(area)}


def movable(patch, n_faces):
    """(movable groups, pinned groups) of a patch: a group that a face outside the patch also uses is pinned."""
    Fg, faces = patch["Fg"], patch["faces"]
    inside = np.zeros(len(Fg), bool)
    inside[faces] = True
    used_in = np.unique(Fg[inside])
    used_out = np.unique(Fg[~inside])
    pinned = np.intersect1d(used_in, used_out)
    # The patch's own boundary edges (used by one patch face) pin their ends too, for an open surface.
    e = np.sort(np.concatenate([Fg[faces][:, [0, 1]], Fg[faces][:, [1, 2]], Fg[faces][:, [2, 0]]]), axis=1)
    u, cnt = np.unique(e, axis=0, return_counts=True)
    pinned = np.union1d(pinned, np.unique(u[cnt == 1]))
    return np.setdiff1d(used_in, pinned), pinned


def rim_distance(patch, move, pinned):
    """Distance along the mesh's edges from each movable group to the nearest pinned one (Dijkstra)."""
    G, E = patch["G"], patch["edges"]
    keep = np.isin(E, np.r_[move, pinned]).all(1)
    E = E[keep]
    w = np.linalg.norm(G[E[:, 0]] - G[E[:, 1]], axis=1)
    adj = {}
    for (a, b), x in zip(E.tolist(), w.tolist()):
        adj.setdefault(a, []).append((b, x))
        adj.setdefault(b, []).append((a, x))
    dist = {int(p): 0.0 for p in pinned}
    heap = [(0.0, int(p)) for p in pinned]
    heapq.heapify(heap)
    while heap:
        d, a = heapq.heappop(heap)
        if d > dist.get(a, np.inf):
            continue
        for b, x in adj.get(a, ()):
            if d + x < dist.get(b, np.inf):
                dist[b] = d + x
                heapq.heappush(heap, (d + x, b))
    return np.array([dist.get(int(g), np.inf) for g in move])


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def balanced(view, weight=None):
    """Per-ray weights in which each view weighs sqrt(its rays), but no view more than the median view, as the LiDAR
    scale weighs its views (lidar_scale.combine): distinct views weigh about the same, dense depth cannot dominate a
    sparse view, and a view with a handful of rays on the patch cannot dominate either. The given weights keep their
    proportions inside a view."""
    view = np.asarray(view)
    w = np.ones(len(view)) if weight is None else np.asarray(weight, float)
    views, counts = np.unique(view, return_counts=True)
    total = np.minimum(np.sqrt(counts), np.median(np.sqrt(counts)))
    out = np.zeros(len(view))
    for v, tv in zip(views, total):
        s = view == v
        out[s] = tv * w[s] / max(w[s].sum(), 1e-300)
    return out / max(out.sum(), 1e-300)


def _wpca(p, w):
    c = (p * w[:, None]).sum(0) / w.sum()
    X = p - c
    return c, np.linalg.eigh((X * w[:, None]).T @ X)[1][:, 0]


def fit_plane(p, w, iters=8, floor=1e-5):
    """A robust plane (center, unit normal, RMS residual of the kept points) of the points p, weights w: weighted PCA,
    reweighted by Huber on the residuals, then refit on the points within 3 robust sigmas (at least floor meters), so
    that a few returns of a neighboring surface cannot tilt it."""
    p, w0 = np.asarray(p, float), np.asarray(w, float)
    w = w0.copy()
    for _ in range(iters):
        c, n = _wpca(p, w)
        r = (p - c) @ n
        s = 1.4826 * np.median(np.abs(r)) + 1e-12
        w = w0 * np.minimum(1.0, 2 * s / np.maximum(np.abs(r), 1e-300))
    for _ in range(3):
        keep = np.abs(r) <= max(3 * 1.4826 * np.median(np.abs(r)), floor)
        if keep.sum() < 3:
            break
        c, n = _wpca(p[keep], w0[keep])
        r = (p - c) @ n
    keep = np.abs(r) <= max(3 * 1.4826 * np.median(np.abs(r)), floor)
    return c, n, float(np.sqrt(np.average(r[keep] ** 2, weights=w0[keep])))


def normal_offsets(V, F, p, n, chunk=256):
    """(t, foot) of each point p along the direction n to the triangles (V, F): p = foot + t n, with foot on the
    nearest crossing of the line through p (either side), nan where the line misses every triangle. Moller-Trumbore
    in plain numpy."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    e1, e2 = b - a, c - a
    h = np.cross(n, e2)
    det = (e1 * h).sum(1)
    ok = np.abs(det) > 1e-14
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    out = np.full(len(p), np.nan)
    for s0 in range(0, len(p), chunk):
        q = p[s0:s0 + chunk]
        s = q[:, None, :] - a[None]
        u = (s * h[None]).sum(2) * inv[None]
        qv = np.cross(s, e1[None])
        v = (qv @ n) * inv[None]
        tt = -(qv * e2[None]).sum(2) * inv[None]  # the crossing is at q - tt n
        hit = ok[None] & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
        tt = np.where(hit, tt, np.inf)
        best = np.take_along_axis(tt, np.argmin(np.abs(tt), 1)[:, None], 1)[:, 0]
        out[s0:s0 + chunk] = np.where(np.isfinite(best), best, np.nan)
    return out, p - out[:, None] * n


def _vertex_normals(patch, V_patch_faces_normals, areas, move):
    """Area-weighted normals of the movable groups over the patch's faces."""
    Fg, faces, G = patch["Fg"], patch["faces"], patch["G"]
    N = np.zeros((len(G), 3))
    for k in range(3):
        np.add.at(N, Fg[faces, k], V_patch_faces_normals[faces] * areas[faces, None])
    N = N[move]
    return N / np.maximum(np.linalg.norm(N, axis=1), 1e-300)[:, None]


def sculpt(V, F, region, corr, op, strength=1.0, max_displacement=0.05, transition=0.06):
    """One sculpt pass of a review region (lidar.json's region and its correspondences {end, hit, normal, view,
    weight}): ("plane" or "push", strength in (0, 1]). Returns (new V, report). SculptError for an operation that cannot
    run: a missing or ambiguous patch, too few movable vertices, too few correspondences, or a result that is not finite
    or flips a face."""
    if op not in ("plane", "push"):
        raise SculptError(f"op is plane or push, not {op!r}")
    if not 0 < strength <= 1:
        raise SculptError("strength is in (0, 1]")
    V = np.asarray(V, float)
    F = np.asarray(F, np.int64)
    end = np.asarray(corr.get("end") or [], float).reshape(-1, 3)
    if len(end) < 10:
        raise SculptError(f"region {region.get('id')} has {len(end)} fitting correspondences; at least 10 are needed")
    hit = np.asarray(corr["hit"], float).reshape(-1, 3)
    hn = np.asarray(corr["normal"], float).reshape(-1, 3)
    w = balanced(corr["view"], corr.get("weight"))
    patch = resolve(V, F, region["seed"], region["normal"], region["bounds"])
    move, pinned = movable(patch, len(F))
    if len(move) < MIN_MOVABLE:
        raise SculptError(f"the patch has {len(move)} movable vertices (its rim and seams are pinned); it is too coarse "
                          "for a LiDAR sculpt: subdivide it first or choose another repair")
    G, gid = patch["G"], patch["gid"]
    nf, _, af = triangles(V, F)
    vn = _vertex_normals(patch, nf, af, move)
    disc = np.abs(((end - hit) * hn).sum(1))
    measured = float(np.percentile(disc, 95))
    cap = min(max_displacement, 1.5 * measured, 0.25 * patch["size"])
    d_rim = rim_distance(patch, move, pinned)
    rep = {"op": op, "patch_faces": int(len(patch["faces"])), "movable": int(len(move)), "pinned": int(len(pinned)),
           "patch_size_m": patch["size"], "measured_p95_m": measured, "cap_m": cap}
    if op == "plane":
        c, n, rms = fit_plane(end, w)
        if n @ patch["normal"] < 0:
            n = -n
        dist = (G[move] - c) @ n
        fall = smoothstep(d_rim / max(transition * patch["size"], 1e-9))
        disp = -(dist * fall * strength)[:, None] * n
        rep.update(plane_center=c.tolist(), plane_normal=n.tolist(), plane_rms_m=rms)
    else:
        # The measured displacement along the patch's normal at each endpoint: the endpoint's offset from the patch
        # along that normal (normal_offsets), where the line crosses the patch, else dot(p - h, n) at the ray's hit.
        nrm = patch["normal"]
        off, foot = normal_offsets(V, F[patch["faces"]], end, nrm)
        ray = ((end - hit) * hn).sum(1)
        d = np.where(np.isfinite(off), off, ray)
        hit = np.where(np.isfinite(off)[:, None], foot, hit)
        sigma = max(1.0 * patch["spacing"], 0.03 * patch["size"])
        field = np.zeros(len(move))
        order = np.argsort(d)
        ds, ws = d[order], w[order]
        for a in range(0, len(move), 512):  # a kernel-weighted median: robust to returns of a neighboring surface
            q = G[move[a:a + 512]]
            k = np.exp(-((q[:, None, :] - hit[order][None]) ** 2).sum(2) / (2 * sigma ** 2)) * ws[None]
            cum = np.cumsum(k, 1)
            half = cum[:, -1:] / 2
            j = np.minimum((cum < half).sum(1), len(ds) - 1)
            near = k.max(1) > math.exp(-2.0) * ws.max()  # a correspondence within 2 sigma; else no motion
            field[a:a + 512] = np.where(near & (cum[:, -1] > 1e-300), ds[j], 0.0)
        fall = smoothstep(d_rim / max(2.5 * transition * patch["size"], 1e-9))
        # Along the vertex normals smoothed over 2 sigma, so that a dent's slopes move out, not sideways.
        q = G[move]
        sn = np.zeros_like(vn)
        for a in range(0, len(move), 512):
            k = np.exp(-((q[a:a + 512, None, :] - q[None]) ** 2).sum(2) / (2 * (2 * sigma) ** 2))
            sn[a:a + 512] = k @ vn
        sn /= np.maximum(np.linalg.norm(sn, axis=1), 1e-300)[:, None]
        disp = (field * fall * strength)[:, None] * sn
        rep.update(kernel_sigma_m=sigma)
    size = np.linalg.norm(disp, axis=1)
    over = size > cap
    disp[over] *= (cap / size[over])[:, None]
    newG = G.copy()
    newG[move] += disp
    out = V + (newG - G)[gid]
    if not np.isfinite(out).all():
        raise SculptError("the result is not finite; nothing was changed")
    n1, _, a1 = triangles(out, F)
    touched = np.isin(patch["Fg"], move).any(1)
    flipped = touched & ((n1 * nf).sum(1) <= 0) & (af > 0)
    if flipped.any() or (touched & (a1 <= 1e-14 * max(af.max(), 1e-300))).any():
        raise SculptError(f"the result flips or collapses {int(flipped.sum())} faces; try a lower strength")
    moved = np.linalg.norm(disp, axis=1)
    rep.update(moved=int((moved > 1e-9).sum()), max_move_m=float(moved.max()), median_move_m=float(np.median(moved)),
               clamped=int(over.sum()))
    return out, rep
