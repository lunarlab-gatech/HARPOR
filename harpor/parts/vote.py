"""The face vote: which part each face of the posed mesh belongs to (build_plan.md note 11).

Runs in envs/sam3d through `python -m harpor.parts.stage vote --job J`. Reads from the run folder
posed.glb (as trimesh loads it with force="mesh", taken to W with frames.gltf_to_w), cameras.json,
masks/, proposal.json and parts/sam3 (harpor.stage_a.part_masks), and writes:
  parts/faces.npz  part (the part id of every face), names (part id -> name, the base first),
                   centroids (every face's centroid in W, for the split in Blender) and score (the
                   probability of the face's part from the vote, 0 for faces that no view sees)
  parts/vote.json  the clusters with their instances per view (for the base, the views with a base
                   mask), status (found, not_found, dropped_small, unconfirmed) and why, confidence
                   and visibility per view, the dropped
                   instances with the reason, the views and the counts of every step

Rules: build_plan.md note 11, step 3 (Vote). A face is in a SAM 3 instance when at least half of its pixels in
that view lie inside it.
No part is named "world" (joints.WORLD, the parent of a fixed base).
Neighbours come from mesh.face_adjacency of the welded mesh (geometry.topology), so they cross the
texture seams. Everything is deterministic.
"""

import json
import time
from pathlib import Path

import numpy as np

from harpor import frames
from harpor.fileio import replace, write_json
from harpor.parts import geometry, joints, render
from harpor.stage_a.sam3_masks import slug

MIN_SHARED, MIN_IOU, MAX_COVER = 0.1, 0.5, 0.8
INSIDE = 0.5  # of an instance's pixels, for unions and duplicates in one view
WHOLE = 0.8  # of the view's base mask: a part instance that holds this much is the whole object
SEES = 0.25  # of the best view's visible pixels of a cluster's faces, for a view that sees the cluster
MIN_AREA = 0.003  # of the surface
PIECE = 0.1  # of the part's area
GROW, MIN_GROW = 0.1, 0.02  # of the part's box extent, and of the object's diagonal
DILATE = 0.01  # of the long side
SMOOTH = 0.004  # of the object's diagonal: the cost of a border per unit of length, against area x probability


def load_mesh(glb):
    """Vertices in W and faces of posed.glb, as trimesh loads it with force="mesh" (the face order of
    faces.npz)."""
    import trimesh

    mesh = trimesh.load(glb, force="mesh", process=False)
    return frames.gltf_to_w(np.asarray(mesh.vertices, dtype=float)), np.asarray(mesh.faces)


def cameras(run):
    """cameras.json of mvsam3d.place, every mask taken from the run folder when it is there."""
    run = Path(run)
    cams = json.loads((run / "cameras.json").read_text())
    for c in cams:
        m = run / "masks" / f"{c['name']}.png"
        c["mask"] = str(m) if m.exists() else c.get("mask")
    return cams


def view_maps(x_w, faces, cams, max_side=render.MAX_SIDE, dilate=DILATE):
    """Per view: name, size, the face-id map inside the grown object mask, visible pixels per face, the
    object mask and its dilation (grown)."""
    from scipy.ndimage import binary_dilation

    from harpor.stage_a.place import resize_mask
    from harpor.stage_a.sam3d import load_mask

    out = []
    for c in cams:
        K, size = render.scaled(c["K"], (c["width"], c["height"]), max_side)
        ids = render.face_ids(x_w, faces, K, c["w2c"], size)
        mask = c.get("mask")
        obj = resize_mask(load_mask(mask), size) if mask and Path(mask).exists() else ids >= 0
        grown = binary_dilation(obj, iterations=max(1, round(dilate * max(size))))
        ids = np.where(grown, ids, -1)
        out.append({"name": c["name"], "size": size, "ids": ids, "vis": render.visible_pixels(ids, len(faces)),
                    "object": obj, "grown": grown})
    return out


def overlaps(masks, scores, index, inside=INSIDE):
    """Per instance of one view and prompt: None to keep it, else why it is dropped. An instance that
    holds at least `inside` of each of two or more other instances, which are not duplicates of each
    other, is their union; then, in order of score, an instance with at least `inside` of its pixels in
    a kept instance of a higher score is a duplicate. index: the instances' indices, for the reasons."""
    n = len(masks)
    px = np.array([m.sum() for m in masks], float)
    inter = np.array([[np.logical_and(a, b).sum() for b in masks] for a in masks], float)
    within = inter / np.maximum(px[:, None], 1)  # within[a, b]: the share of a inside b
    order = sorted(range(n), key=lambda k: (-scores[k], k))
    why = [None] * n
    for u in range(n):
        held = []
        for a in order:
            if a != u and within[a, u] >= inside and all(within[a, b] < inside and within[b, a] < inside for b in held):
                held.append(a)
        if len(held) >= 2:
            why[u] = f"the union of instances {sorted(index[a] for a in held)}"
    kept = []
    for a in order:
        if why[a] is None:
            dup = next((b for b in kept if within[a, b] >= inside), None)
            if dup is None:
                kept.append(a)
            else:
                why[a] = f"a duplicate of instance {index[dup]}"
    return why


def instances(sam3_dir, prompts, views, n_faces, max_cover=MAX_COVER, base=None, whole=WHOLE):
    """(kept, dropped, segmented, base_masks): the instances of part_masks.json for these prompts, clipped
    to the grown object mask, with the faces they hold; the dropped ones with the reason; the prompts that
    part_masks.json has; and per view the union of the base's kept instances, None where it has none (all
    None when part_masks.json does not have the prompt base). A base instance that covers more than
    max_cover of the object is dropped. A part instance that holds at least `whole` of the view's base
    mask is the whole object and dropped; in a view without a base mask, one that covers more than
    max_cover of the object is."""
    from harpor.stage_a.place import resize_mask
    from harpor.stage_a.sam3d import load_mask

    sam3_dir = Path(sam3_dir)
    pm = json.loads((sam3_dir / "part_masks.json").read_text())
    by_view = {v["name"]: k for k, v in enumerate(views)}
    kept, dropped = [], []
    records = {p["name"]: p for p in pm["parts"]}
    base_masks = [None] * len(views)

    def clipped(prompt, vname, recs):
        """[(record, score, mask)]: the instances of one prompt in one view that the cover rules keep."""
        here = []
        for r in recs:
            rec = {"prompt": prompt, "view": vname, "index": r["index"]}
            if vname not in by_view:
                dropped.append({**rec, "why": "the view is not in cameras.json"})
                continue
            v = views[by_view[vname]]
            bm = base_masks[by_view[vname]] if prompt != base else None
            m = resize_mask(load_mask(sam3_dir / r["file"]), v["size"]) & v["grown"]
            px, obj_px = int(m.sum()), int(v["object"].sum())
            held = np.logical_and(m, bm).sum() / bm.sum() if bm is not None else 0.0
            if px == 0:
                dropped.append({**rec, "why": "outside the object"})
            elif bm is not None and held >= whole:
                dropped.append({**rec, "why": f"holds {held:.0%} of the base"})
            elif bm is None and px > max_cover * obj_px:
                dropped.append({**rec, "why": f"covers {px / max(obj_px, 1):.0%} of the object"})
            else:
                here.append((rec, r["score"], m))
        return here

    if base is not None and base in records:
        for vname, recs in records[base]["views"].items():
            here = clipped(base, vname, recs)
            if here:
                base_masks[by_view[vname]] = np.logical_or.reduce([m for _, _, m in here])
    for prompt in prompts:
        part = records.get(prompt)
        for vname, recs in (part["views"].items() if part else []):
            here = clipped(prompt, vname, recs)
            whys = overlaps([m for _, _, m in here], [s for _, s, _ in here], [r["index"] for r, _, _ in here])
            for (rec, score, m), why in zip(here, whys):
                if why:
                    dropped.append({**rec, "why": why})
                    continue
                k = by_view[vname]
                v = views[k]
                inside_px = render.inside_pixels(v["ids"], m, n_faces)
                f = np.flatnonzero(inside_px)
                kept.append({**rec, "v": k, "score": score, "px": int(m.sum()), "faces": f, "counts": inside_px[f],
                             "S": f[2 * inside_px[f] >= v["vis"][f]]})
    return kept, dropped, set(records), base_masks


def cluster(insts, seen, area, min_shared=MIN_SHARED, min_iou=MIN_IOU):
    """Clusters of instances (lists of indices into insts, in order of their first member): same prompt
    only, never two of one view. seen: (views, faces) bool, the faces each view sees."""
    n = len(insts)
    root = list(range(n))
    views = {i: {insts[i]["view"]} for i in range(n)}

    def find(i):
        while root[i] != i:
            root[i] = root[root[i]]
            i = root[i]
        return i

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = insts[i], insts[j]
            if a["prompt"] != b["prompt"] or a["view"] == b["view"] or not len(a["S"]) or not len(b["S"]):
                continue
            co = seen[a["v"]] & seen[b["v"]]
            A, B = a["S"][co[a["S"]]], b["S"][co[b["S"]]]
            inter = area[np.intersect1d(A, B, assume_unique=True)].sum()
            if inter <= 0:
                continue
            iou = inter / area[np.union1d(A, B)].sum()
            shared = inter / min(area[a["S"]].sum(), area[b["S"]].sum())
            if shared >= min_shared and iou >= min_iou:
                pairs.append((-iou, i, j))
    for _, i, j in sorted(pairs):
        ri, rj = find(i), find(j)
        if ri != rj and not views[ri] & views[rj]:
            ri, rj = min(ri, rj), max(ri, rj)
            root[rj] = ri
            views[ri] |= views.pop(rj)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def agreement(S, members, found_in, views, sees=SEES):
    """(yes, seeing) of a cluster with the faces S and its views members (indices): seeing is the
    number of the views found_in (those that found the cluster's prompt) whose visible pixels of S are
    at least `sees` of the best view's, and yes how many of those are members."""
    px = np.array([views[k]["vis"][S].sum() for k in found_in], float)
    best = px.max(initial=0.0)
    seeing = [k for k, n in zip(found_in, px) if best > 0 and n >= sees * best]
    return sum(k in members for k in seeing), len(seeing)


def _neighbour_counts(label, adj, n_labels, mask_a=None, mask_b=None):
    """(faces, labels) counts of each face's neighbours per label; mask_a/mask_b pick the pairs that
    count for the first and the second face of each pair."""
    a, b = adj[:, 0], adj[:, 1]
    ka = np.ones(len(a), bool) if mask_a is None else mask_a
    kb = np.ones(len(b), bool) if mask_b is None else mask_b
    idx = np.concatenate([a[ka] * n_labels + label[b[ka]], b[kb] * n_labels + label[a[kb]]])
    return np.bincount(idx, minlength=len(label) * n_labels).reshape(len(label), n_labels)


def reach(centroids, x_w, faces, label, n_labels, grow=GROW, min_grow=MIN_GROW):
    """(faces, labels) bool: the faces each label may take by propagation. The base takes any face; a
    part those whose centroid lies in the oriented box of its labelled faces, grown on every side by
    `grow` of the box's extent along that axis, and by at least min_grow of the object's diagonal."""
    out = np.ones((len(centroids), n_labels), bool)
    diag = float(np.linalg.norm(np.ptp(x_w, 0)))
    for j in range(1, n_labels):
        f = label == j
        if not f.any():
            out[:, j] = False
            continue
        box = geometry.obb(x_w[faces[f]].reshape(-1, 3))
        half = np.maximum(np.array(box["extents"]) * (0.5 + grow), np.array(box["extents"]) / 2 + min_grow * diag)
        local = (centroids - box["center"]) @ np.array(box["axes"]).T
        out[:, j] = (np.abs(local) <= half).all(1)
    return out


def propagate(label, adj, n_labels, allowed=None):
    """Faces labelled -1 take the majority label of their labelled neighbours, ring by ring, among the
    labels that allowed (faces, labels) permits them; those that never get one become the base (0).
    Returns the labels and how many faces got one this way."""
    label = label.copy()
    n = 0
    while (label < 0).any():
        un = label < 0
        ka, kb = un[adj[:, 0]] & ~un[adj[:, 1]], un[adj[:, 1]] & ~un[adj[:, 0]]
        if not (ka.any() or kb.any()):
            break
        counts = _neighbour_counts(np.maximum(label, 0), adj, n_labels, ka, kb)
        if allowed is not None:
            counts = counts * allowed
        got = un & (counts.sum(1) > 0)
        if not got.any():
            break
        label[got] = counts[got].argmax(1)
        n += int(got.sum())
    label[label < 0] = 0
    return label, n


def edge_lengths(topo):
    """The length of the edge that each pair of topo.face_adjacency shares."""
    e = np.asarray(topo.face_adjacency_edges).reshape(-1, 2)
    v = np.asarray(topo.vertices, dtype=float)
    return np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1)


def min_cut(cost0, cost1, pa, pb, pw):
    """x (bool per node) with the least sum(cost0[~x]) + sum(cost1[x]) + sum(pw[~x[pa] & x[pb]]), all costs
    at least 0: a minimum s-t cut by scipy's maximum_flow, on capacities scaled to integers."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import breadth_first_order, maximum_flow

    n = len(cost0)
    s, t = n, n + 1
    cap = np.concatenate([cost1, cost0, pw])
    scale = 2.0 ** 30 / max(float(cap.sum()), 1e-300)  # every flow stays below 2**31 (scipy uses int32)
    cap = np.rint(cap * scale).astype(np.int64)
    rows = np.concatenate([np.full(n, s), np.arange(n), pa])
    cols = np.concatenate([np.arange(n), np.full(n, t), pb])
    k = cap > 0
    G = csr_matrix((cap[k].astype(np.int32), (rows[k], cols[k])), shape=(n + 2, n + 2))
    G.sum_duplicates()
    res = maximum_flow(G, s, t, method="dinic")
    R = (G - res.flow).tocsr()  # residual capacities; the flow is skew-symmetric
    R.data = (R.data > 0).astype(np.int8)
    R.eliminate_zeros()
    side0 = np.zeros(n + 2, bool)
    side0[breadth_first_order(R, s, directed=True, return_predecessors=False)] = True
    return ~side0[:n]


def regions(label, D, adj, w, allowed=None, cycles=5):
    """Labels with the least sum(D[f, label[f]]) + sum(w[label[a] != label[b]]) over the pairs of adj, by
    alpha-expansion (graph cuts) from `label`. D: (faces, labels) costs; w: the weight of each pair;
    allowed: (faces, labels) bool, False where a face may not take a label (label must respect it).
    Returns the labels and the number of faces that changed."""
    F, L = D.shape
    a, b = adj[:, 0], adj[:, 1]
    rows = np.arange(F)
    lab = label.copy()
    big = 4 * (np.bincount(a, w, F) + np.bincount(b, w, F)) + np.ptp(D, 1)  # more than taking alpha can save

    def energy(x):
        return float(D[rows, x].sum() + w[x[a] != x[b]].sum())

    E = energy(lab)
    for _ in range(cycles):
        better = False
        for alpha in range(L):
            if not (lab != alpha).any():
                continue
            # x = 1: the face takes alpha. Pair costs A (0, 0), B (0, 1), C (1, 0), 0 (1, 1) are
            # A + (C - A) x_a - C x_b + (B + C - A)(1 - x_a) x_b, and B + C - A >= 0 (Potts).
            A, B, C = w * (lab[a] != lab[b]), w * (lab[a] != alpha), w * (lab[b] != alpha)
            c0 = D[rows, lab]
            c1 = D[:, alpha] + np.bincount(a, C - A, F) - np.bincount(b, C, F)
            if allowed is not None:
                c1 = np.where(allowed[:, alpha], c1, c1 + big)
            m = np.minimum(c0, c1)
            x = min_cut(c0 - m, c1 - m, a, b, B + C - A)
            new = np.where(x, alpha, lab)
            if allowed is not None:
                new = np.where(allowed[rows, new], new, lab)
            En = energy(new)
            if En < E - 1e-9 * max(abs(E), 1e-300):
                lab, E, better = new, En, True
        if not better:
            break
    return lab, int((lab != label).sum())


def cleanup(label, adj, area, n_labels, piece=PIECE):
    """A connected piece of a label that is not its largest and holds less than `piece` of its area
    joins the neighbouring label it shares the most edges with. Returns the labels and the faces moved."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    a, b = adj[:, 0], adj[:, 1]
    F = len(label)
    same = label[a] == label[b]
    g = coo_matrix((np.ones(int(same.sum())), (a[same], b[same])), shape=(F, F))
    n, comp = connected_components(g, directed=False)
    comp_area = np.bincount(comp, weights=area, minlength=n)
    comp_label = np.zeros(n, int)
    comp_label[comp] = label
    label_area = np.bincount(label, weights=area, minlength=n_labels)
    largest = np.full(n_labels, -1)
    for c in np.lexsort((np.arange(n), -comp_area)):  # largest first, then lowest index
        if largest[comp_label[c]] < 0:
            largest[comp_label[c]] = c
    small = (comp_area < piece * label_area[comp_label]) & (np.arange(n) != largest[comp_label])
    cross = label[a] != label[b]
    ka, kb = cross & small[comp[a]], cross & small[comp[b]]
    keys = np.concatenate([comp[a[ka]] * n_labels + label[b[ka]], comp[b[kb]] * n_labels + label[a[kb]]])
    if not len(keys):
        return label, 0
    counts = np.bincount(keys, minlength=n * n_labels).reshape(n, n_labels)
    move = small & (counts.sum(1) > 0)
    new = np.where(move, counts.argmax(1), comp_label)
    out = new[comp]
    return out, int((out != label).sum())


def names_for(base, clusters, prompts):
    """The node name of every found cluster: the prompt's slug, or <slug>_1, <slug>_2, ... when the
    prompt has more than one found cluster; never the base's name, another part's or "world"."""
    used = {base, joints.WORLD}
    out = {}
    for prompt in prompts:
        found = [c for c in clusters if c["prompt"] == prompt and c["status"] == "found"]
        for k, c in enumerate(found, 1):
            name = slug(prompt) if len(found) == 1 else f"{slug(prompt)}_{k}"
            n = 2
            while name in used:
                name = f"{slug(prompt)}_{k}_{n}" if len(found) > 1 else f"{slug(prompt)}_{n}"
                n += 1
            used.add(name)
            out[id(c)] = name
    return out


def run(run_dir, max_side=render.MAX_SIDE):
    """The vote of the run folder; see the module docstring. Returns the vote.json record."""
    R = Path(run_dir)
    t0 = time.time()
    proposal = json.loads((R / "proposal.json").read_text())
    base_slug = slug(proposal["base"])
    if base_slug == joints.WORLD:
        raise SystemExit(f'the base cannot be called "{joints.WORLD}"; run propose again')
    prompts = [p["name"] for p in proposal["parts"] if p["name"] != proposal["base"]]
    x_w, faces = load_mesh(R / "posed.glb")
    F = len(faces)
    topo = geometry.topology(x_w, faces)
    adj = np.asarray(topo.face_adjacency).reshape(-1, 2)
    area = np.asarray(topo.area_faces, dtype=float)
    total = float(area.sum())
    views = view_maps(x_w, faces, cameras(R), max_side)
    seen = np.array([v["vis"] > 0 for v in views]).reshape(len(views), F)
    vis_total = np.sum([v["vis"] for v in views], axis=0) if views else np.zeros(F)
    insts, dropped_inst, segmented, base_masks = instances(R / "parts" / "sam3", prompts, views, F,
                                                           base=proposal["base"])
    warnings = [f"part {p!r} is not in part_masks.json" for p in prompts if p not in segmented]

    # Clusters: found; dropped_small when their faces cover too little of the surface; unconfirmed when
    # fewer than half of the views that see their faces well and found the prompt hold them.
    found_in = {}
    for i in insts:
        found_in.setdefault(i["prompt"], set()).add(i["v"])
    clusters = []
    for members in cluster(insts, seen, area):
        S = np.unique(np.concatenate([insts[i]["S"] for i in members]))
        prompt = insts[members[0]]["prompt"]
        yes, seeing = agreement(S, {insts[i]["v"] for i in members}, sorted(found_in[prompt]), views)
        c = {"prompt": prompt, "members": members, "support": float(area[S].sum()) / total,
             "views": [insts[i]["view"] for i in members], "status": "found"}
        if area[S].sum() < MIN_AREA * total:
            c.update(status="dropped_small", why=f"its instances hold {c['support']:.2%} of the surface, "
                                                 f"less than {MIN_AREA:.1%}")
        elif 2 * yes < seeing:
            c.update(status="unconfirmed", why=f"only {yes} of the {seeing} views that see its faces well and "
                                               f"found {prompt!r} hold it")
        clusters.append(c)
    order = {p: k for k, p in enumerate(prompts)}
    first = {id(c): min(c["members"]) for c in clusters}
    clusters.sort(key=lambda c: (order[c["prompt"]], -len(c["views"]), -c["support"], first[id(c)]))
    found = [c for c in clusters if c["status"] == "found"]

    # Shares: pixels of each face inside each found cluster's instances (column 0: inside the base masks),
    # over its pixels in the views where SAM 3 found the cluster's prompt (that have a base mask).
    P = 1 + len(found)
    votes, visc = np.zeros((F, P)), np.zeros((F, P))
    base_views, base_S = [], np.zeros(F, bool)
    for k, m in enumerate(base_masks):
        if m is not None:
            inside_px = render.inside_pixels(views[k]["ids"], m, F)
            votes[:, 0] += inside_px
            visc[:, 0] += views[k]["vis"]
            base_S |= (views[k]["vis"] > 0) & (2 * inside_px >= views[k]["vis"])
            base_views.append(views[k]["name"])
    for j, c in enumerate(found, 1):
        for i in c["members"]:
            votes[insts[i]["faces"], j] += insts[i]["counts"]
        for k in sorted(found_in[c["prompt"]]):
            visc[:, j] += views[k]["vis"]
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(visc > 0, votes / np.maximum(visc, 1), 0.0)
    rows = np.arange(F)
    # Of the clusters of one prompt, the one with the highest share keeps it (then the one with more views:
    # found is in that order).
    for p in prompts:
        cols = np.array([j for j, c in enumerate(found, 1) if c["prompt"] == p], int)
        if len(cols) > 1:
            win = cols[share[:, cols].argmax(1)]
            share[:, cols] = np.where(cols[None, :] == win[:, None], share[:, cols], 0.0)
    # Probabilities, least area first; the rest goes to the base.
    support = np.array([float(area[base_S].sum()) / total] + [c["support"] for c in found])
    prob, rest = np.zeros((F, P)), np.ones(F)
    for j in np.lexsort((np.arange(P), support)):
        prob[:, j] = share[:, j] * rest
        rest = rest * (1 - share[:, j])
    prob[:, 0] += rest
    seen_any = vis_total > 0
    label = np.where(seen_any, prob.argmax(1), -1)
    unseen = int((~seen_any).sum())
    centroids = x_w[faces].mean(1)
    allowed = reach(centroids, x_w, faces, label, P)
    label, propagated = propagate(label, adj, P, allowed)
    allowed[seen_any] = True
    cost = np.where(seen_any[:, None], area[:, None] * (1 - prob), 0.0)
    diag = float(np.linalg.norm(np.ptp(x_w, 0)))
    label, relabelled = regions(label, cost, adj, SMOOTH * diag * edge_lengths(topo), allowed)
    label, absorbed = cleanup(label, adj, area, P)

    # Parts left too small go to their neighbours.
    part_area = np.bincount(label, weights=area, minlength=P)
    small = [j for j in range(1, P) if part_area[j] < MIN_AREA * total]
    for j in small:
        found[j - 1].update(status="dropped_small", why=f"the vote left it {part_area[j] / total:.2%} of the surface "
                            f"(its instances hold {found[j - 1]['support']:.2%}), less than {MIN_AREA:.1%}")
    if small:
        label = np.where(np.isin(label, small), -1, label)
        label, n = propagate(label, adj, P)
        absorbed += n

    # Scores with the old ids, then the found parts renumbered 1.. in order.
    score = np.where(seen_any, prob[rows, label], 0.0)
    keep = [0] + [j for j in range(1, P) if found[j - 1]["status"] == "found"]
    remap = np.zeros(P, int)
    remap[keep] = np.arange(len(keep))
    label = remap[label]
    found = [c for c in found if c["status"] == "found"]
    names_of = names_for(base_slug, found, prompts)
    names = [base_slug] + [names_of[id(c)] for c in found]

    def record(j, c=None):
        f = label == j
        a = float(area[f].sum())
        rec = {"faces": int(f.sum()), "area_share": round(a / total, 5),
               "confidence": round(float((score[f] * area[f]).sum() / a), 4) if a > 0 else 0.0,
               "visibility": {v["name"]: round(float(area[f & s].sum() / a), 4) if a > 0 else 0.0
                              for v, s in zip(views, seen)}}
        if c is not None:
            rec.update(views=c["views"], instances={v: [insts[i]["index"] for i in c["members"] if insts[i]["view"] == v]
                                                    for v in c["views"]},
                       support_share=round(c["support"], 5))
        return rec

    out = [{"name": base_slug, "role": "base", "prompt": proposal["base"], "slug": base_slug, "status": "found",
            **record(0), "views": base_views, "support_share": round(float(support[0]), 5)}]
    for j, c in enumerate(found, 1):
        out.append({"name": names[j], "role": "part", "prompt": c["prompt"], "slug": slug(c["prompt"]),
                    "status": "found", **record(j, c)})
    for c in clusters:
        if c["status"] != "found":
            out.append({"name": None, "role": "part", "prompt": c["prompt"], "slug": slug(c["prompt"]),
                        "status": c["status"], "why": c["why"], "faces": 0, "views": c["views"],
                        "instances": {v: [insts[i]["index"] for i in c["members"] if insts[i]["view"] == v]
                                      for v in c["views"]}, "support_share": round(c["support"], 5)})
    for p in prompts:
        if not any(c["prompt"] == p for c in clusters):
            out.append({"name": None, "role": "part", "prompt": p, "slug": slug(p), "status": "not_found", "faces": 0})
    counts = {p["name"]: p.get("count", 1) for p in proposal["parts"]}
    for p in prompts:
        n = sum(c["prompt"] == p for c in found)
        if n and n != counts[p]:
            warnings.append(f"part {p!r}: {n} found, the proposal says {counts[p]}")

    (R / "parts").mkdir(exist_ok=True)
    replace(R / "parts" / "faces.npz", np.savez, part=label.astype(np.int32), names=np.array(names),
            centroids=centroids, score=score.astype(np.float32))
    vote = {"base": base_slug, "names": names, "clusters": out, "dropped_instances": dropped_inst,
            "views": [{"name": v["name"], "size_wh": list(v["size"]), "visible_faces": int(s.sum()),
                       "object_px": int(v["object"].sum())} for v, s in zip(views, seen)],
            "counts": {"faces": F, "instances": len(insts), "unseen": unseen, "propagated": propagated,
                       "relabelled": relabelled, "absorbed": absorbed},
            "settings": {"max_side": max_side, "dilate": DILATE, "max_cover": MAX_COVER, "min_shared": MIN_SHARED,
                         "min_iou": MIN_IOU, "inside": INSIDE, "whole": WHOLE, "sees": SEES, "grow": GROW,
                         "min_grow": MIN_GROW, "min_area": MIN_AREA, "piece": PIECE, "smooth": SMOOTH},
            "warnings": warnings, "seconds": round(time.time() - t0, 1)}
    write_json(R / "parts" / "vote.json", vote)
    return vote


def whole(run_dir):
    """The vote of a job that skips the split (job.json parts.split false): every face of posed.glb is the base,
    and every other part of the proposal is not found, so that harpor.parts.stage finish gives each one a
    requirement with a reserve and the agent cuts it out itself (tools.add_part). No SAM 3 part mask is read.
    Writes run/parts/faces.npz and vote.json in the forms of run, and returns the vote.json record."""
    R = Path(run_dir)
    t0 = time.time()
    proposal = json.loads((R / "proposal.json").read_text())
    base_slug = slug(proposal["base"])
    if base_slug == joints.WORLD:
        raise SystemExit(f'the base cannot be called "{joints.WORLD}"; run propose again')
    x_w, faces = load_mesh(R / "posed.glb")
    F = len(faces)
    why = "the split was skipped: the agent cuts the parts"
    out = [{"name": base_slug, "role": "base", "prompt": proposal["base"], "slug": base_slug, "status": "found",
            "faces": F, "area_share": 1.0, "confidence": 1.0, "visibility": {}, "views": [], "support_share": 1.0}]
    out += [{"name": None, "role": "part", "prompt": p["name"], "slug": slug(p["name"]), "status": "not_found",
             "why": why, "faces": 0} for p in proposal["parts"] if p["name"] != proposal["base"]]
    (R / "parts").mkdir(exist_ok=True)
    replace(R / "parts" / "faces.npz", np.savez, part=np.zeros(F, np.int32), names=np.array([base_slug]),
            centroids=x_w[faces].mean(1), score=np.ones(F, np.float32))
    vote = {"base": base_slug, "names": [base_slug], "clusters": out, "dropped_instances": [], "views": [],
            "counts": {"faces": F, "instances": 0, "unseen": 0, "propagated": 0, "relabelled": 0, "absorbed": 0},
            "settings": {"split": False}, "warnings": [why], "seconds": round(time.time() - t0, 1)}
    write_json(R / "parts" / "vote.json", vote)
    return vote
