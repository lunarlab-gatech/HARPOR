"""LiDAR geometry evidence of an agent session: the host's scorer, its repair regions and its acceptance check (README.md
note 5, LiDAR geometry; docs/lidar_sculpting_plan.md).

The lidar stage keeps each pool view's measured returns (harpor.webapp.lidar.write_geometry), select copies those of
the picks into J/run/lidar/geometry (harpor.webapp.views.build_run), and harpor.webapp.agent_step hands their index to
controller.prepare_session. freeze() then moves the returns into frame W, once per session:

  origin_W = inverse(w2c) origin_c and end_W = inverse(w2c) xyz_c, with w2c the final view-relative camera of
  run/cameras.json. The returns are already in meters and select already scaled DA3's translations, so no metric
  scale, no fitted object pose and no second glTF axis conversion apply. The stock glTF importer gives Blender world
  coordinates equal to W, so the same numbers hold in Blender.

The authoritative arrays, the split and the settings stay in root/control/lidar (the agent never sees them); the
agent gets the fitting returns only, in work/input/lidar. Views form viewpoint groups (overlapping acquisition windows
or nearly the same pose), and every third group by azimuth, counted from view 0's, is reserved for verification.

The scorer (score) follows each eligible return's ray from its sensor origin o through its measured endpoint p to the
candidate's first triangle h: the signed error is |h - o| - |p - o| in meters. Negative: the mesh is too close to that
sensor (it sticks out); positive: it lies too far away (a dent, or a jump to a far surface). A ray with no hit is a
miss. The ray IDs, weights (each view weighs the same) and region assignments are frozen from the original, so every
candidate is scored on the same evidence. baseline() scores the original and builds the repair regions; review_outputs()
adds overlays and a region summary to a review; evaluate() is the host's check of a stored candidate
(harpor.agent.validate), on the reserved rays only.

Modes: off (nothing), inspect (overlays and informational findings) and sculpt (the agent gets the sculpt helper and
the host's acceptance policy decides). Only a calibrated profile (PROFILES) allows sculpt; a real source has none
until its range repeatability is measured, so its jobs stay at inspect.
"""

import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np

from harpor import frames as F
from harpor.fileio import read_json, replace, write_json

VERSION = 1  # of the evidence files, the scorer and the settings: another version is a cache miss
SURFACES = 2  # of the surface split in baseline.json; 2 grows each surface against its mean normal
MODES = ("off", "inspect", "sculpt")
DEFAULT_MODE = "inspect"  # job.json's lidar_geometry when a record has none (README.md note 8)
# Per-return flags (harpor.webapp.lidar.write_geometry). BLOCKING returns are never scored.
IN_MASK, BOUNDARY, MIXED, CONF_MISSING, CONF_LOW, EXCLUDED, OFFSET, HIDDEN = 1, 2, 4, 8, 16, 32, 64, 128
BLOCKING = BOUNDARY | MIXED | CONF_LOW | EXCLUDED | HIDDEN
FITTING, VERIFY = "fitting", "verify"
# The settings of every source; a profile changes some of them.
BASE = {
    "min_views": 2,            # distinct fitting viewpoint groups that must see a region's displacement
    "min_verify_views": 1,     # reserved groups that must observe a region for automatic acceptance
    "min_samples": 25,         # rays of one view on a region (spatially spread by the sample below)
    "grazing_deg": 75.0,       # rays this far or farther from a face's normal are not scored
    "coverage_tol": 0.02,      # share of a region's reserved hits that a candidate may lose
    "silhouette_tol": 0.01,    # mask IoU that a view may lose
    "max_displacement_m": 0.05,  # the most a sculpt pass may move a vertex
    "sample_cap": 3000,        # rays per view at most, a deterministic spatially balanced sample
    "sample_grid_px": 6,       # its grid cells in prepared-image pixels
    "miss_error_rel": 0.25,    # a miss counts as this share of the original's diagonal
    "crease_deg": 35.0,        # faces meeting at a sharper angle are different surfaces
    "region_min_share": 0.005,  # surfaces under this share of the area are left out
    "pose_tol_deg": 2.0,       # two views closer than this and pose_tol_rel of the distance are one viewpoint
    "pose_tol_rel": 0.02,
    "silhouette_px": 256,      # long side of the silhouettes that evaluate compares
    "correspondences": 1500,   # fitting correspondences per region in a review, at most
}
# Calibrated profiles: tolerances that were measured, so a job of that source may sculpt. range_tol_m is the range
# repeatability (a gain or a loss inside it is a tie), endpoint_tol_m the distance at which two views' endpoints
# disagree, legend_m the end of the overlay's color scale.
PROFILES = {
    "synthetic": {"range_tol_m": 0.002, "endpoint_tol_m": 0.004, "legend_m": 0.02,
                  "from": "fixture tolerances: noiseless analytic rays (tests/test_lidar_geometry.py)"},
    # The Livox LiDAR of the box, cage and pallet bags (harpor.webapp.lidar.sensor_profile). Measured 2026-11-22: the
    # range scatter of 278,805 returns about 173 fitted floor, wall and table planes in the three bags, along each ray:
    # 1.4826 MAD 10.5 mm (9.7, 11.6 and 12.3 mm by bag; 12.7 mm beyond 50 degrees of incidence), 95% within 25 mm. It
    # holds the surfaces' own flatness and the motion within a scan, so it bounds the range noise from above. Pose and
    # mask variability were not measured.
    "livox": {"range_tol_m": 0.010, "endpoint_tol_m": 0.025, "legend_m": 0.05,
              "from": "Livox range scatter on flat surfaces of the box, cage and pallet bags, 2026-11-22: "
                      "1.4826 MAD 10.5 mm, 95% within 25 mm (results/ice_runs.md note 5)"},
}
# Real sources until a capture of a static plane calibrates them: diagnostic tolerances only, inspect at most.
NOMINAL = {"rosbag": {"range_tol_m": 0.03, "endpoint_tol_m": 0.05, "legend_m": 0.08,
                      "from": "nominal spinning-LiDAR range noise, not calibrated"},
           "stray": {"range_tol_m": 0.02, "endpoint_tol_m": 0.03, "legend_m": 0.06,
                     "from": "nominal iPhone depth noise, not calibrated"}}


class Unavailable(Exception):
    """The session has no usable LiDAR geometry: args[0] says why."""


# ---- settings, digests and files ----

def settings(kind, profile=None):
    """The settings of a source kind (rosbag, stray) or of an explicit profile (synthetic, for tests), with
    calibrated, profile and version."""
    name = profile or kind
    if name in PROFILES:
        return {**BASE, **PROFILES[name], "profile": name, "calibrated": True, "version": VERSION}
    nominal = NOMINAL.get(kind) or {"range_tol_m": 0.03, "endpoint_tol_m": 0.05, "legend_m": 0.08,
                                    "from": "nominal, not calibrated"}
    return {**BASE, **nominal, "profile": None, "calibrated": False, "version": VERSION}


def effective_mode(requested, available, calibrated):
    """The mode that runs: off when asked, unavailable without evidence, and sculpt only with a calibrated profile."""
    if requested not in MODES:
        raise ValueError(f"lidar_geometry is one of {', '.join(MODES)}, not {requested!r}")
    if requested == "off":
        return "off"
    if not available:
        return "unavailable"
    return "sculpt" if requested == "sculpt" and calibrated else "inspect"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(obj):
    """sha256 of obj as canonical JSON."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def save_npz(path, **arrays):
    """arrays as an NPZ, written under a temporary name and renamed (fileio.replace)."""
    write_bytes(path, npz_bytes(**arrays))


def npz_bytes(**arrays):
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def write_bytes(path, data):
    """data at path, written under a temporary name and renamed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    replace(Path(path), lambda tmp: Path(tmp).write_bytes(data))


def manifest(control):
    """control/lidar/manifest.json, or None for a session without LiDAR geometry."""
    m = read_json(Path(control) / "lidar" / "manifest.json")
    return m if isinstance(m, dict) else None


def active(control):
    """The manifest when the session has evidence to score (inspect or sculpt), else None."""
    m = manifest(control)
    return m if m and m.get("mode", {}).get("effective") in ("inspect", "sculpt") else None


def load_evidence(control):
    """{name: array} of control/lidar/evidence.npz."""
    with np.load(Path(control) / "lidar" / "evidence.npz") as f:
        return {k: f[k] for k in f.files}


# ---- freezing the evidence (controller.prepare_session) ----

def _groups(views, cams, s):
    """Viewpoint groups of the available views: views whose acquisition windows overlap or whose poses are nearly
    the same share a group. Returns ([[view index]], {view index: group}, [whether each group's independence rests
    on known acquisition times])."""
    idx = [v["index"] for v in views]
    parent = {i: i for i in idx}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    c2w = {i: np.linalg.inv(np.asarray(cams[i]["w2c"], float)) for i in idx}
    C, Z = {i: c2w[i][:3, 3] for i in idx}, {i: c2w[i][:3, 2] for i in idx}
    dist = np.median([np.linalg.norm(C[i] - np.mean([C[j] for j in idx], 0)) for i in idx]) or 1.0
    win = {v["index"]: (v.get("time") or {}).get("window_ns") for v in views}
    scans = {v["index"]: set((v.get("time") or {}).get("scans") or []) for v in views}
    for a in idx:
        for b in idx:
            if a >= b:
                continue
            wa, wb = win[a], win[b]
            overlap = bool(wa and wb and wa[0] <= wb[1] and wb[0] <= wa[1]) or bool(scans[a] & scans[b])
            angle = math.degrees(math.acos(float(np.clip(Z[a] @ Z[b], -1, 1))))
            near = angle < s["pose_tol_deg"] and np.linalg.norm(C[a] - C[b]) < s["pose_tol_rel"] * dist
            if overlap or near:
                parent[find(a)] = find(b)
    roots = sorted({find(i) for i in idx}, key=lambda r: min(i for i in idx if find(i) == r))
    groups = [[i for i in idx if find(i) == r] for r in roots]
    timed = [all(win[i] for i in g) for g in groups]
    return groups, {i: k for k, g in enumerate(groups) for i in g}, timed


def _roles(groups, cams, center):
    """FITTING or VERIFY per group: every third group by azimuth about the object, counted from view 0's group (or the
    first), is reserved. With fewer than three groups nothing is reserved."""
    if len(groups) < 3:
        return [FITTING] * len(groups), "fewer than 3 viewpoint groups: nothing is reserved"
    C = np.array([np.mean([np.linalg.inv(np.asarray(cams[i]["w2c"]))[:3, 3] for i in g], 0) for g in groups])
    X = C - C.mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    a, b = Vt[0], Vt[1]
    az = np.arctan2((C - center) @ b, (C - center) @ a)
    order = list(np.argsort(az))
    start = next((k for k, g in enumerate(groups) if 0 in g), 0)
    order = order[order.index(start):] + order[:order.index(start)]
    roles = [FITTING] * len(groups)
    for pos, k in enumerate(order):
        if pos % 3 == 2:
            roles[k] = VERIFY
    return roles, "every third viewpoint group by azimuth about the object, counted from view 0's group"


def _balanced(pixel, rows, cap, cell):
    """A deterministic, spatially balanced subset of at most cap rays (indices): the image is cut into cells of cell
    px, and the cells give one ray each in turn, each cell's rays in the order of a hash of their row IDs."""
    n = len(rows)
    if n <= cap:
        return np.arange(n)
    key = np.floor(pixel / cell).astype(np.int64)
    cid = np.unique(key, axis=0, return_inverse=True)[1].reshape(-1)
    h = (rows.astype(np.uint64) * np.uint64(2654435761)) % np.uint64(1 << 32)
    order = np.lexsort((h, cid))
    rank = np.empty(n, np.int64)
    first = np.r_[0, np.flatnonzero(np.diff(cid[order])) + 1]
    sizes = np.diff(np.r_[first, n])
    rank[order] = np.arange(n) - np.repeat(first, sizes)
    pick = np.lexsort((h, cid, rank))[:cap]
    return np.sort(pick)


def freeze(root, cams, lidar):
    """Write root/control/lidar (the session's evidence) and root/work/input/lidar (the fitting returns and a small
    manifest) from lidar, harpor.webapp.agent_step.geometry_request's {requested, index (run/lidar/geometry/index.json
    or None), reason, profile}. cams: the session's views, in the manifest's order (K, w2c, width, height). Returns the
    manifest. A session without evidence gets a manifest with status unavailable and the reason; a view that fails
    is listed with its reason and the others go on."""
    root = Path(root)
    control, work = root / "control" / "lidar", root / "work" / "input" / "lidar"
    control.mkdir(parents=True, exist_ok=True)
    requested = lidar.get("requested") or DEFAULT_MODE
    index_path = lidar.get("index")
    index = read_json(index_path) if index_path else None
    kind = (index or {}).get("kind")
    s = settings(kind, (index or {}).get("profile") or lidar.get("profile"))
    m = {"version": VERSION, "status": "unavailable", "reason": lidar.get("reason"), "kind": kind, "settings": s,
         "mode": {"requested": requested, "effective": "off" if requested == "off" else "unavailable"}, "views": []}
    if requested == "off" or not isinstance(index, dict):
        if requested != "off" and not m["reason"]:
            m["reason"] = "no LiDAR geometry index for this run"
        write_json(control / "manifest.json", m)
        return m
    folder = Path(index_path).parent
    ends, origins, pixels, rows, flags, views_of = [], [], [], [], [], []
    good = []
    for v in index.get("views") or []:
        i = v.get("index")
        rec = {"index": i, "pool_name": v.get("pool_name"), "model": v.get("model"),
               "confidence": v.get("confidence"), "time": v.get("time")}
        try:
            if v.get("status") != "available":
                raise Unavailable(v.get("reason") or "no measured XYZ for this view")
            if not isinstance(i, int) or not 0 <= i < len(cams):
                raise Unavailable(f"view {i!r} is not a view of this session")
            f = folder / v["file"]
            if sha256_file(f) != v.get("sha256"):
                raise Unavailable(f"{v['file']} does not match the index's sha256")
            with np.load(f) as d:
                xyz, origin = np.asarray(d["xyz"], float).reshape(-1, 3), np.asarray(d["origin"], float).reshape(3)
                pixel, row = np.asarray(d["pixel"], float).reshape(-1, 2), np.asarray(d["row"], np.int64).reshape(-1)
                flag = np.asarray(d["flags"], np.uint16).reshape(-1)
            if not (len(xyz) == len(pixel) == len(row) == len(flag)):
                raise Unavailable("its arrays are not row for row")
            c2w = np.linalg.inv(np.asarray(cams[i]["w2c"], float))
            end = F.transform_points(c2w, xyz)
            o = F.transform_points(c2w, origin[None])[0]
            ok = np.isfinite(end).all(1) & (np.linalg.norm(end - o, axis=1) > 1e-6)
            flag = np.where(ok, flag, flag | EXCLUDED).astype(np.uint16)
            ends.append(end)
            origins.append(np.repeat(o[None], len(end), 0))
            pixels.append(pixel)
            rows.append(row)
            flags.append(flag)
            views_of.append(np.full(len(end), i, np.int64))
            rec.update(status="available", returns=int(len(end)),
                       eligible=int((((flag & IN_MASK) > 0) & ((flag & BLOCKING) == 0)).sum()))
            good.append(rec)
        except (Unavailable, OSError, KeyError, ValueError) as e:
            rec.update(status="unavailable", reason=str(e))
        m["views"].append(rec)
    if not good:
        m["reason"] = "no selected view has usable LiDAR returns"
        write_json(control / "manifest.json", m)
        return m
    end, origin, pixel = np.concatenate(ends), np.concatenate(origins), np.concatenate(pixels)
    row, flag, view = np.concatenate(rows), np.concatenate(flags), np.concatenate(views_of)
    eligible = ((flag & IN_MASK) > 0) & ((flag & BLOCKING) == 0)
    keep = np.zeros(len(end), bool)  # the balanced sample of each view's eligible rays
    for i in np.unique(view):
        at = np.flatnonzero((view == i) & eligible)
        keep[at[_balanced(pixel[at], row[at], s["sample_cap"], s["sample_grid_px"])]] = True
    center = np.median(end[keep], 0) if keep.any() else end.mean(0)
    groups, group_of, timed = _groups(good, {v["index"]: cams[v["index"]] for v in good}, s)
    roles, rule = _roles(groups, cams, center)
    group = np.array([group_of[i] for i in view], np.int64)
    role = np.array([roles[g] == VERIFY for g in group])
    n_view = {int(i): max(int((keep & (view == i)).sum()), 1) for i in np.unique(view)}
    weight = np.array([1.0 / n_view[int(i)] for i in view]) * keep
    ev = {"ray": np.arange(len(end), dtype=np.int64), "view": view, "row": row, "origin": origin, "end": end,
          "pixel": pixel, "flags": flag, "group": group, "verify": role, "keep": keep, "weight": weight}
    data = npz_bytes(**ev)
    write_bytes(control / "evidence.npz", data)
    import shutil

    for k, c in enumerate(cams):  # the host's own masks: the agent can write work/input
        if c.get("mask"):
            (control / "masks").mkdir(exist_ok=True)
            shutil.copyfile(root / "work" / c["mask"], control / "masks" / f"{k:03d}.png")
    for rec in m["views"]:
        if rec["status"] == "available":
            g = group_of[rec["index"]]
            rec.update(group=g, role=roles[g], sampled=int((keep & (view == rec["index"])).sum()))
    m.update(status="available", reason=None, groups=[{"group": k, "views": g, "role": roles[k], "timed": timed[k]}
                                                      for k, g in enumerate(groups)],
             split={"rule": rule, "reserved_views": [i for k, g in enumerate(groups) if roles[k] == VERIFY for i in g],
                    "cap_per_view": s["sample_cap"]},
             pool_index_sha256=index.get("pool_index_sha256"), metric_scale=index.get("metric_scale"),
             evidence_sha256=hashlib.sha256(data).hexdigest())
    m["mode"]["effective"] = effective_mode(requested, True, s["calibrated"])
    m["key"] = digest({k: m[k] for k in ("version", "evidence_sha256", "settings", "groups", "split", "mode")})
    write_json(control / "manifest.json", m)
    fit = keep & ~role
    work.mkdir(parents=True, exist_ok=True)
    write_bytes(work / "fitting.npz", npz_bytes(ray=ev["ray"][fit], view=view[fit], origin=origin[fit],
                                                 end=end[fit], pixel=pixel[fit]))
    write_json(work / "manifest.json", {
        "frame": "W (as input/manifest.json); meters; Blender world coordinates are the same numbers",
        "mode": m["mode"]["effective"], "kind": kind,
        "files": {"fitting.npz": "ray (ID), view, origin and end (the sensor origin and the measured endpoint, W), "
                                 "pixel (the input image's pixel)"},
        "sign": "signed error = |hit - origin| - |end - origin|: negative, the mesh is too close to the sensor",
        "fitting_views": sorted({int(i) for i in view[fit]}), "reserved_views": "kept by the host",
        "rays": int(fit.sum())})
    return m


# ---- meshes and the scorer ----

class Mesh:
    """Every triangle of a GLB in W (harpor.physics.mjcf.load_s applies the node transforms and the stock glTF import),
    with a ray index built once."""

    def __init__(self, glb):
        import trimesh

        from harpor.physics.mjcf import load_s

        parts = [m for _, m in load_s(glb)]
        V = np.concatenate([np.asarray(p.vertices, float) for p in parts])
        off = np.cumsum([0] + [len(p.vertices) for p in parts])[:-1]
        Fc = np.concatenate([np.asarray(p.faces, np.int64) + o for p, o in zip(parts, off)])
        self.mesh = trimesh.Trimesh(V, Fc, process=False)
        self.vertices, self.faces = self.mesh.vertices, self.mesh.faces
        self.normals = np.asarray(self.mesh.face_normals, float)
        self.areas = np.asarray(self.mesh.area_faces, float)
        lo, hi = V.min(0), V.max(0)
        self.diagonal = float(np.linalg.norm(hi - lo))
        self._ray = trimesh.ray.ray_triangle.RayMeshIntersector(self.mesh)
        self._adjacency = None

    def first_hits(self, origin, direction, chunk=20000):
        """(distance along the unit direction, triangle) of each ray's first hit; nan and -1 for a miss."""
        n = len(origin)
        t, tri = np.full(n, np.nan), np.full(n, -1, np.int64)
        for a in range(0, n, chunk):
            o, d = origin[a:a + chunk], direction[a:a + chunk]
            if not len(o):
                continue
            it, ir, loc = self._ray.intersects_id(o, d, multiple_hits=False, return_locations=True)
            if len(it):
                t[a + ir] = ((loc - o[ir]) * d[ir]).sum(1)
                tri[a + ir] = it
        return t, tri

    def adjacency(self):
        """(pairs of faces that share an edge, once the vertices are welded by position)."""
        if self._adjacency is None:
            import trimesh

            eps = max(self.diagonal, 1e-9) * 1e-6
            _, inv = np.unique(np.round(self.vertices / eps), axis=0, return_inverse=True)
            welded = trimesh.Trimesh(self.vertices, inv.reshape(-1)[self.faces], process=False)
            self._adjacency = np.asarray(welded.face_adjacency, np.int64).reshape(-1, 2)
        return self._adjacency


def score(mesh, ev):
    """Per ray of the evidence: err (signed error in meters, nan for a miss), hit, tri (the first triangle, -1),
    cos (|cos| of the angle between the ray and the triangle's normal) and front (the ray meets the triangle's
    front)."""
    o, p = ev["origin"], ev["end"]
    rng = np.linalg.norm(p - o, axis=1)
    d = (p - o) / np.maximum(rng, 1e-12)[:, None]
    t, tri = mesh.first_hits(o, d)
    hit = tri >= 0
    nrm = np.zeros_like(d)
    nrm[hit] = mesh.normals[tri[hit]]
    dot = (nrm * d).sum(1)
    return {"err": np.where(hit, t - rng, np.nan), "hit": hit, "tri": tri, "cos": np.abs(dot), "front": dot < 0,
            "range": rng}


def cached_score(control, glb, ev, key, mesh=None):
    """score of the GLB at glb, cached in control/lidar/scores by the GLB's sha256 and the evidence key. Returns
    (score, mesh or None when it came from the cache)."""
    folder = Path(control) / "lidar" / "scores"
    sha = sha256_file(glb)
    path = folder / f"{digest({'glb': sha, 'key': key, 'version': VERSION})[:24]}.npz"
    if path.is_file():
        with np.load(path) as f:
            return {k: f[k] for k in f.files}, mesh
    mesh = mesh or Mesh(glb)
    sc = score(mesh, ev)
    save_npz(path, **sc)
    return sc, mesh


# ---- regions (baseline) ----

def _sparse_adj(n, pairs):
    from scipy.sparse import coo_matrix

    if not len(pairs):
        return coo_matrix((n, n)).tocsr()
    a, b = pairs[:, 0], pairs[:, 1]
    return coo_matrix((np.ones(2 * len(a)), (np.r_[a, b], np.r_[b, a])), shape=(n, n)).tocsr()


def surfaces(mesh, s):
    """Connected surfaces of similarly oriented faces: (surface per face, smoothed face normals). A surface grows from
    the flattest face not yet in one, across shared edges. A face joins when its smoothed normal is within crease_deg
    of the face that reaches it and of the surface's mean normal so far. The second test cuts a rounded edge: there
    each face turns only a little from the next, but the faces past the middle of the edge face the other side."""
    from collections import deque

    n = len(mesh.faces)
    A = _sparse_adj(n, mesh.adjacency()).tocsr()
    A.sort_indices()
    w = mesh.normals * mesh.areas[:, None]
    sm = w + A @ w
    sm /= np.maximum(np.linalg.norm(sm, axis=1), 1e-12)[:, None]
    cos_t = math.cos(math.radians(s["crease_deg"]))
    flat = (mesh.normals * sm).sum(1)  # 1 where the face's neighbors face its way: the middle of a side
    order = np.lexsort((np.arange(n), -flat))
    N, area, ptr, nbr = sm.tolist(), mesh.areas.tolist(), A.indptr.tolist(), A.indices.tolist()
    comp = [-1] * n
    k = 0
    for seed in order.tolist():
        if comp[seed] >= 0:
            continue
        comp[seed] = k
        ax, ay, az = (c * area[seed] for c in N[seed])
        mean = N[seed]
        q = deque([seed])
        while q:
            f = q.popleft()
            nfx, nfy, nfz = N[f]
            for g in nbr[ptr[f]:ptr[f + 1]]:
                if comp[g] >= 0:
                    continue
                gx, gy, gz = N[g]
                if nfx * gx + nfy * gy + nfz * gz > cos_t and mean[0] * gx + mean[1] * gy + mean[2] * gz > cos_t:
                    comp[g] = k
                    ax, ay, az = ax + gx * area[g], ay + gy * area[g], az + gz * area[g]
                    r = math.sqrt(ax * ax + ay * ay + az * az)
                    if r > 0:
                        mean = (ax / r, ay / r, az / r)
                    q.append(g)
        k += 1
    return np.asarray(comp, np.int64), sm


def usable(ev, sc, s):
    """The rays that are scored on the original: sampled, a front-facing hit, not grazing."""
    return ev["keep"] & sc["hit"] & sc["front"] & (sc["cos"] > math.cos(math.radians(s["grazing_deg"])))


def _median(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else None


def _icp(mesh, o, p, w, tol, iters=8):
    """The rigid motion (rotation vector, translation) of the measurements (origins o, endpoints p) that best puts
    the endpoints on the mesh: point-to-plane ICP, each pass following the moved rays to their first hits again,
    Huber-weighted. Returns (x, share of rays beyond tol before, after)."""
    def residuals(x):
        R = _rotation(x[:3])
        oo, pp = o @ R.T + x[3:], p @ R.T + x[3:]
        rng = np.linalg.norm(pp - oo, axis=1)
        d = (pp - oo) / np.maximum(rng, 1e-12)[:, None]
        t, tri = mesh.first_hits(oo, d)
        hit = tri >= 0
        h = oo + d * np.nan_to_num(t)[:, None]
        n = np.zeros_like(h)
        n[hit] = mesh.normals[tri[hit]]
        return pp, h, n, hit, np.where(hit, ((pp - h) * n).sum(1), np.nan)

    x = np.zeros(6)
    pp, h, n, hit, r = residuals(x)
    before = float(np.mean(~hit | (np.abs(np.nan_to_num(r)) > tol)))
    for _ in range(iters):
        ok = hit & np.isfinite(r)
        if ok.sum() < 6:
            break
        scale = 1.4826 * np.median(np.abs(r[ok])) + 1e-9
        hub = np.minimum(1.0, 2 * scale / np.maximum(np.abs(r[ok]), 1e-12))
        J = np.c_[np.cross(pp[ok], n[ok]), n[ok]]
        Wt = w[ok] * hub
        dx = -np.linalg.solve(J.T @ (J * Wt[:, None]) + 1e-9 * np.eye(6), J.T @ (Wt * r[ok]))
        R = _rotation(dx[:3])
        Rx = _rotation(x[:3])
        x = np.r_[_rotvec(R @ Rx), R @ x[3:] + dx[3:]]
        pp, h, n, hit, r = residuals(x)
        if np.linalg.norm(dx) < 1e-7:
            break
    after = float(np.mean(~hit | (np.abs(np.nan_to_num(r)) > tol)))
    return x, before, after


def _rotvec(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = math.acos(c)
    if th < 1e-12:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return v * th


def _rotation(v):
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def alignment(ev, sc, mesh, s, use):
    """Diagnostic rigid fits (_icp) of the scored rays, all fitting views together and each view alone: a common shift
    of the whole object, or one view that disagrees with the others, is an alignment finding. The fitted motions are
    reported and never applied. Returns {common, views: [{view, before, after, translation_m, rotation_deg,
    misregistered}], excluded, common_shift}; before and after are the shares of rays beyond range_tol_m (a miss
    counts)."""
    tol = s["range_tol_m"]
    fit = use & ~ev["verify"]

    def one(sel):
        if sel.sum() < 6:
            return None
        x, b, a = _icp(mesh, ev["origin"][sel], ev["end"][sel], ev["weight"][sel] / ev["weight"][sel].sum(), tol)
        return {"before": round(b, 4), "after": round(a, 4), "translation_m": float(np.linalg.norm(x[3:])),
                "rotation_deg": float(math.degrees(np.linalg.norm(x[:3]))),
                "explained": bool(b > 0.5 and a < 0.5 * b)}

    common = one(fit)
    views, excluded = [], []
    for i in np.unique(ev["view"][use]):
        share = float(np.mean(np.abs(sc["err"][use & (ev["view"] == i)]) > tol))
        if share <= 0.5:  # most of its rays agree: nothing to explain
            views.append({"view": int(i), "before": round(share, 4), "misregistered": False})
            continue
        r = one(ev["keep"] & (ev["view"] == i) & sc["hit"])
        if r is None:
            continue
        r["view"] = int(i)
        r["misregistered"] = bool(r["explained"] and not (common and common["explained"]))
        views.append(r)
    others = [v for v in views if not v["misregistered"]]
    if len(views) >= 3 and len(others) >= 2:
        excluded = [v["view"] for v in views if v["misregistered"]]
    return {"common": common, "views": views, "excluded": excluded,
            "common_shift": bool(common and common["explained"])}


def _endpoint_gap(pa, pb, k=8):
    """Median distance of the endpoints pb to the local planes of the endpoints pa (where they overlap), or None."""
    from scipy.spatial import cKDTree

    if len(pa) < k or len(pb) < 3:
        return None
    tree = cKDTree(pa)
    spacing = float(np.median(tree.query(pa, 2)[0][:, 1])) or 1e-6
    d, j = tree.query(pb, k)
    near = d[:, 0] < 3 * spacing
    if near.sum() < 3:
        return None
    nb = pa[j[near]]
    c = nb.mean(1)
    X = nb - c[:, None]
    _, _, Vt = np.linalg.svd(X)
    nrm = Vt[:, -1]
    return float(np.median(np.abs(((pb[near] - c) * nrm).sum(1))))


def regions(mesh, ev, sc, s, excluded=()):
    """The surfaces of the original (surfaces) with their support, and the repair regions among them. A region is a
    connected, similarly oriented surface (or the part of a large one where the error gathers, with a band around it)
    where at least min_views distinct fitting groups show a displacement of one sign beyond range_tol_m and their
    endpoints agree on one physical surface. Its status: target (sculptable) with verifiable when a reserved group
    observes it, conflict (the fitting groups disagree: an alignment finding) or observation. Returns (surfaces,
    regions, face_region: a region index per face, -1)."""
    from scipy.sparse.csgraph import connected_components

    tol = s["range_tol_m"]
    comp, sm = surfaces(mesh, s)
    use = usable(ev, sc, s) & ~np.isin(ev["view"], list(excluded))
    fit, ver = use & ~ev["verify"], use & ev["verify"]
    total = float(mesh.areas.sum())
    tri = sc["tri"]
    n = len(mesh.faces)
    pairs = mesh.adjacency()
    A = _sparse_adj(n, pairs)
    same = _sparse_adj(n, pairs[comp[pairs[:, 0]] == comp[pairs[:, 1]]])  # adjacency within a surface
    s_f = np.bincount(tri[fit], weights=sc["err"][fit] * ev["weight"][fit], minlength=n)
    w_f = np.bincount(tri[fit], weights=ev["weight"][fit], minlength=n)
    S, W = s_f.copy(), w_f.copy()
    S, W = S + same @ S, W + same @ W  # one ring of smoothing: faces with few rays borrow from their neighbors
    e_f = np.where(W > 0, S / np.maximum(W, 1e-12), 0.0)
    surf, regs = [], []
    face_region = np.full(n, -1, np.int64)
    for c in np.unique(comp):
        faces = np.flatnonzero(comp == c)
        area = float(mesh.areas[faces].sum())
        if area < s["region_min_share"] * total:
            continue
        nrm = sm[faces].T @ mesh.areas[faces]
        nrm /= max(np.linalg.norm(nrm), 1e-12)
        on = np.isin(tri, faces)
        rec = {"surface": int(c), "area_share": round(area / total, 4), "normal": np.round(nrm, 4).tolist(),
               "center": np.round(mesh.vertices[mesh.faces[faces]].reshape(-1, 3).mean(0), 5).tolist(),
               "fitting_rays": int((fit & on).sum()), "reserved_rays": int((ver & on).sum()),
               "median_signed_m": _median(sc["err"][fit & on])}
        surf.append(rec)
        flagged = faces[(np.abs(e_f[faces]) > tol) & (W[faces] > 0)]
        if not len(flagged):
            continue
        sub = same[flagged][:, flagged]
        k, lab = connected_components(sub, directed=False)
        for b in range(k):
            blob = flagged[lab == b]
            if mesh.areas[blob].sum() < s["region_min_share"] * total:
                continue
            if mesh.areas[blob].sum() > 0.7 * area:  # the error covers the surface: the whole surface
                core = patch = faces
            else:  # the core is where the error is; the patch adds 3 rings of the same surface around it
                mark = np.zeros(n, bool)
                mark[blob] = True
                for _ in range(3):
                    mark |= (same @ mark.astype(float)) > 0
                core, patch = blob, np.flatnonzero(mark & (comp == c))
            regs.append(_region(mesh, ev, sc, s, core, patch, int(c), fit, ver, sm))
    regs.sort(key=lambda r: (-(r["status"] == "target"), -r["area_share"]))
    keep = []
    face_patch = np.full(n, -1, np.int64)
    for r in regs:  # one region per face: the first (a target) wins
        if face_patch[r["patch"]].max() >= 0:
            continue
        r["id"] = f"L{len(keep) + 1}"
        face_region[r["faces"]] = len(keep)
        face_patch[r["patch"]] = len(keep)
        keep.append(r)
    for r in keep:  # the comparison band: the patch's margin and the faces next to it, on any surface
        mark = np.zeros(n, bool)
        mark[r["patch"]] = True
        for _ in range(3):
            mark |= (A @ mark.astype(float)) > 0
        mark[r["faces"]] = False
        r["band"] = np.flatnonzero(mark)
    return surf, keep, face_region, face_patch


def _region(mesh, ev, sc, s, faces, patch, comp, fit, ver, sm):
    """One region's record (regions): its core faces (where the error is: they decide its support, its status and the
    host's gain), its patch (the core and a margin: the bounds, normal and seed that the sculpt helper selects, and
    its correspondences), support by fitting and reserved group, and its status."""
    tol = s["range_tol_m"]
    on = np.isin(sc["tri"], faces)
    nrm = sm[patch].T @ mesh.areas[patch]
    nrm /= max(np.linalg.norm(nrm), 1e-12)
    pts = mesh.vertices[mesh.faces[patch]].reshape(-1, 3)
    support, gaps = [], []
    for g in np.unique(ev["group"][fit & on]):
        sel = fit & on & (ev["group"] == g)
        support.append({"group": int(g), "views": sorted({int(i) for i in ev["view"][sel]}), "rays": int(sel.sum()),
                        "median_signed_m": _median(sc["err"][sel])})
    strong = [x for x in support if x["rays"] >= s["min_samples"] and abs(x["median_signed_m"] or 0) > tol]
    signs = {np.sign(x["median_signed_m"]) for x in strong}
    ends = {x["group"]: ev["end"][fit & on & (ev["group"] == x["group"])] for x in strong}
    for a in range(len(strong)):
        for b in range(a + 1, len(strong)):
            g = _endpoint_gap(ends[strong[a]["group"]], ends[strong[b]["group"]])
            if g is not None:
                gaps.append(g)
    verify = []
    for g in np.unique(ev["group"][ver & on]):
        sel = ver & on & (ev["group"] == g)
        verify.append({"group": int(g), "views": sorted({int(i) for i in ev["view"][sel]}), "rays": int(sel.sum())})
    verifiable = sum(x["rays"] >= s["min_samples"] for x in verify) >= s["min_verify_views"]
    if len(strong) >= s["min_views"] and len(signs) == 1 and all(g <= s["endpoint_tol_m"] for g in gaps):
        status = "target"
    elif len(strong) >= 2 and (len(signs) > 1 or any(g > s["endpoint_tol_m"] for g in gaps)):
        status = "conflict"
    else:
        status = "observation"
    med = _median(sc["err"][fit & on])
    seed_face = faces[np.argmax(np.abs(np.bincount(sc["tri"][fit & on], weights=sc["err"][fit & on],
                                                   minlength=len(mesh.faces))[faces]))] if (fit & on).any() else faces[0]
    seed = mesh.vertices[mesh.faces[seed_face]].mean(0)
    flat = _flatness(ev["end"][fit & on])
    return {"faces": faces, "patch": patch, "surface": comp, "status": status,
            "verifiable": bool(verifiable and status == "target"),
            "area_share": round(float(mesh.areas[faces].sum() / mesh.areas.sum()), 4),
            "patch_share": round(float(mesh.areas[patch].sum() / mesh.areas.sum()), 4),
            "normal": np.round(nrm, 5).tolist(), "seed": np.round(seed, 6).tolist(),
            "bounds": [np.round(pts.min(0), 6).tolist(), np.round(pts.max(0), 6).tolist()],
            "median_signed_m": med, "support": support, "verification": verify,
            "endpoint_gap_m": max(gaps) if gaps else None,
            "direction": None if med is None else ("inward: the mesh sticks out toward the sensors" if med < 0 else
                                                   "outward: the mesh lies behind the measured surface"),
            "suggested_op": "plane" if flat is not None and flat < s["range_tol_m"] * 1.5 else "push",
            "measured_flatness_m": flat}


def _flatness(p):
    """The RMS distance of the endpoints p to their best plane, or None for too few."""
    if len(p) < 10:
        return None
    X = p - p.mean(0)
    return float(np.linalg.svd(X, full_matrices=False)[1][-1] / math.sqrt(len(p)))


def view_summary(ev, sc, s, use):
    out = []
    for i in np.unique(ev["view"]):
        sel = ev["keep"] & (ev["view"] == i)
        u = use & (ev["view"] == i)
        out.append({"view": int(i), "rays": int(sel.sum()), "hits": int((sel & sc["hit"]).sum()),
                    "misses": int((sel & ~sc["hit"]).sum()), "scored": int(u.sum()),
                    "median_signed_m": _median(sc["err"][u]), "median_abs_m": _median(np.abs(sc["err"][u])),
                    "p90_abs_m": None if not u.any() else float(np.percentile(np.abs(sc["err"][u]), 90)),
                    "reserved": bool(ev["verify"][sel].any()) if sel.any() else False})
    return out


def baseline(root):
    """Score root/original.glb and build the repair regions, before the clock starts (controller.make_previews):
    control/lidar/baseline.json and baseline.npz (the region of each face and each ray's original hit), and for the
    agent work/original/previews/lidar/ (summary.json without reserved rays, and an overlay per view). Returns the
    baseline record, or None for a session without evidence."""
    root = Path(root)
    control = root / "control"
    m = active(control)
    if m is None:
        return None
    ev = load_evidence(control)
    s = m["settings"]
    sc, mesh = cached_score(control, root / "original.glb", ev, m["key"])
    mesh = mesh or Mesh(root / "original.glb")
    use = usable(ev, sc, s)
    al = alignment(ev, sc, mesh, s, use)
    surf, regs, face_region, face_patch = regions(mesh, ev, sc, s, al["excluded"])
    if al["common_shift"]:  # a shift of the whole object is not a local repair: no target is acceptable
        for r in regs:
            if r["status"] == "target":
                r["status"], r["verifiable"] = "observation", False
                r["why"] = "a rigid shift of the whole object explains the errors: an alignment finding"
    ray_region = np.where(sc["hit"], face_region[np.maximum(sc["tri"], 0)], -1)
    ray_patch = np.where(sc["hit"], face_patch[np.maximum(sc["tri"], 0)], -1)
    ray_band = np.full(len(ray_region), -1)
    for k, r in enumerate(regs):
        ray_band[np.isin(sc["tri"], r["band"]) & (ray_region < 0)] = k
    comp, _ = surfaces(mesh, s)
    ray_surface = np.where(sc["hit"], comp[np.maximum(sc["tri"], 0)], -1)
    save_npz(control / "lidar" / "baseline.npz", ray_region=ray_region, ray_patch=ray_patch, ray_band=ray_band,
             ray_surface=ray_surface, use=use, face_region=face_region)
    observed = {x["surface"] for x in surf if x["fitting_rays"] + x["reserved_rays"] > 0}
    try:
        sil = np.round(silhouettes(control, root / "original.glb", s["silhouette_px"]), 4).tolist()
    except Exception as e:  # noqa: BLE001 - evaluate then has no silhouette check
        sil = {"error": f"{type(e).__name__}: {e}"}
    rec = {"version": VERSION, "surfaces_version": SURFACES, "key": m["key"],
           "glb_sha256": sha256_file(root / "original.glb"),
           "diagonal_m": mesh.diagonal, "mode": m["mode"]["effective"], "alignment": al,
           "regions": [{k: v for k, v in r.items() if k not in ("faces", "patch", "band")}
                       | {"faces": int(len(r["faces"])), "patch_faces": int(len(r["patch"]))} for r in regs],
           "surfaces": surf, "unobserved": [x for x in surf if x["surface"] not in observed],
           "views": view_summary(ev, sc, s, use), "silhouettes": sil}
    write_json(control / "lidar" / "baseline.json", rec)
    out = root / "work" / "original" / "previews" / "lidar"
    out.mkdir(parents=True, exist_ok=True)
    agent = agent_summary(rec, m)
    write_json(out / "summary.json", agent)
    previews = read_json(root / "work" / "original" / "previews" / "review.json") or {}
    for name, data in overlays(root, ev, sc, s, use, previews).items():
        (out / name).write_bytes(data)
    return rec


def agent_summary(rec, m):
    """What the agent sees of a baseline or a review: no reserved ray, no verification detail beyond its counts."""
    reserved = set((m.get("split") or {}).get("reserved_views") or [])
    regs = []
    for r in rec["regions"]:
        x = {k: r[k] for k in ("id", "status", "area_share", "normal", "seed", "bounds", "median_signed_m", "direction",
                               "suggested_op", "measured_flatness_m") if k in r}
        x["fitting_support"] = [{k: g[k] for k in ("views", "rays", "median_signed_m")} for g in r["support"]]
        x["host_verifiable"] = r.get("verifiable", False)
        if "current" in r:
            x["current"] = r["current"]
        regs.append(x)
    return {"mode": m["mode"]["effective"], "units": "meters; W = Blender world coordinates",
            "sign": "signed error = |hit - origin| - |end - origin|: negative (red), the mesh is too close to the "
                    "sensor; positive (blue), it lies behind the measured surface; a cross is a miss",
            "range_tol_m": m["settings"]["range_tol_m"], "calibrated": m["settings"]["calibrated"],
            "regions": regs,
            "surfaces": [{k: x[k] for k in ("surface", "area_share", "normal", "center", "fitting_rays",
                                            "median_signed_m")} for x in rec["surfaces"]],
            "unobserved_surfaces": [{k: x[k] for k in ("surface", "area_share", "normal", "center")}
                                    for x in rec["unobserved"]],
            "alignment": {"common_shift": rec["alignment"]["common_shift"],
                          "excluded_views": [v for v in rec["alignment"]["excluded"] if v not in reserved]},
            "views": [v for v in rec["views"] if not v["reserved"]]}


def summary_lines(rec, current=None):
    """A few lines for the agent: each region with its original (and current) fit, and the alignment notice."""
    lines = []
    for r in rec["regions"]:
        cur = (current or {}).get(r["id"])
        now = f", now {cur['median_signed_m'] * 1000:+.1f} mm" if cur and cur.get("median_signed_m") is not None else ""
        orig = r["median_signed_m"]
        lines.append(f"lidar {r['id']} {r['status']}{' (host-verifiable)' if r.get('verifiable') else ''}: "
                     f"normal {fmt(r['normal'])}, seed {fmt(r['seed'])}, median signed "
                     f"{'none' if orig is None else f'{orig * 1000:+.1f} mm'}{now}; "
                     f"{len(r['support'])} fitting groups; suggested {r['suggested_op']}")
    if rec["alignment"]["common_shift"]:
        lines.append("lidar alignment: a rigid shift of the whole object explains the errors; do not sculpt for it")
    for i in rec["alignment"]["excluded"]:
        lines.append(f"lidar alignment: view {i} disagrees with the others and is left out")
    if rec.get("unobserved"):
        lines.append(f"lidar: {len(rec['unobserved'])} surfaces have no returns (any repair there is inferred)")
    return lines or ["lidar: every observed surface agrees with the returns; no LiDAR repair region"]


def fmt(v):
    return "[" + ", ".join(f"{x:.4g}" for x in v) + "]"


# ---- overlays ----

LEGEND = ((180, 30, 30), (255, 255, 255), (30, 60, 200))  # too close, agrees, too far


def _colors(err, legend):
    x = np.clip(np.nan_to_num(err) / legend, -1, 1)
    neg, pos = np.clip(-x, 0, 1)[:, None], np.clip(x, 0, 1)[:, None]
    white, red, blue = (np.array(c, float) for c in (LEGEND[1], LEGEND[0], LEGEND[2]))
    return (white * (1 - neg - pos) + red * neg + blue * pos).astype(np.uint8)


def overlays(root, ev, sc, s, use, review=None, reserved=False):
    """{file name: PNG bytes}: each view's photo (work/input/images, cropped and scaled as review.py's crop of that
    view when review has it) with its returns at their stored pixels, colored by signed error on a fixed scale of
    +-legend_m, a cross for a miss and a small gray dot for a return that is not scored. reserved: also the reserved
    views (the host's own copies; the agent's never have them)."""
    from PIL import Image, ImageDraw

    root = Path(root)
    m = read_json(root / "control" / "manifest.json") or {}
    crops = {v["name"]: v for v in (review or {}).get("views") or [] if "crop" in v}
    out = {}
    for i in np.unique(ev["view"]):
        sel = ev["view"] == i
        if ev["verify"][sel].any() and not reserved:
            continue
        entry = next((v for v in m.get("views") or [] if v["index"] == i), None)
        if entry is None:
            continue
        photo = Image.open(root / "work" / entry["image"]).convert("RGB")
        px = ev["pixel"][sel].astype(float)
        c = crops.get(f"view{int(i):02d}")
        if c:
            u0, v0, u1, v1 = c["crop"]
            k = c["scale"]
            photo = photo.crop((u0, v0, u1, v1))
            photo = photo.resize((max(round((u1 - u0) * k), 1), max(round((v1 - v0) * k), 1)))
            px = (px - [u0, v0] + 0.5) * k - 0.5
        img = Image.blend(photo, Image.new("RGB", photo.size, (0, 0, 0)), 0.35)
        draw = ImageDraw.Draw(img)
        u = use[sel]
        hit = sc["hit"][sel]
        keep = ev["keep"][sel]
        col = _colors(sc["err"][sel], s["legend_m"])
        r = max(1, round(min(img.size) / 250))
        for (x, y), ok, h, kp, cl in zip(px, u, hit, keep, col):
            if not kp:
                continue
            if not h:
                draw.line([(x - r - 1, y - r - 1), (x + r + 1, y + r + 1)], fill=(230, 0, 230))
                draw.line([(x - r - 1, y + r + 1), (x + r + 1, y - r - 1)], fill=(230, 0, 230))
            elif ok:
                draw.rectangle([x - r, y - r, x + r, y + r], fill=tuple(int(v) for v in cl))
            else:
                draw.point((x, y), fill=(150, 150, 150))
        _legend(draw, img.size, s["legend_m"])
        buf = io.BytesIO()
        img.save(buf, "PNG")
        out[f"lidar_view{int(i):02d}.png"] = buf.getvalue()
    return out


def _legend(draw, size, legend):
    """The color scale in the lower left: -legend (the mesh too close) to +legend (too far), in mm."""
    w, h = size
    bw, bh = max(120, w // 3), max(6, h // 40)
    x0, y0 = 6, h - bh - 30
    for k in range(bw):
        e = (k / (bw - 1) * 2 - 1) * legend
        draw.line([(x0 + k, y0), (x0 + k, y0 + bh)], fill=tuple(int(v) for v in _colors(np.array([e]), legend)[0]))
    mm = legend * 1000
    right = f"+{mm:.0f} mm"
    draw.text((x0, y0 + bh + 2), f"-{mm:.0f} mm", fill=(255, 255, 255))
    draw.text((x0 + bw - draw.textlength(right), y0 + bh + 2), right, fill=(255, 255, 255))
    draw.text((x0, y0 + bh + 14), "red: mesh too close, blue: too far, x: miss", fill=(255, 255, 255))


# ---- review and validation ----

def _region_rays(base, k):
    return base["ray_region"] == k


def current_fit(rec, ev, sc, base, s, reserved=False):
    """{region id: {median_signed_m, median_abs_m, hits, rays}} of a candidate's score on the region's fitting (or
    reserved) rays."""
    out = {}
    sel0 = base["use"] & (ev["verify"] if reserved else ~ev["verify"])
    for k, r in enumerate(rec["regions"]):
        sel = sel0 & _region_rays(base, k)
        out[r["id"]] = {"median_signed_m": _median(sc["err"][sel]), "median_abs_m": _median(np.abs(sc["err"][sel])),
                        "hits": int((sel & sc["hit"]).sum()), "rays": int(sel.sum())}
    return out


def correspondences(rec, ev, sc, base, mesh, s):
    """{region id: {end, hit, normal, view, weight}} of the fitting rays on each region, at the candidate (sc, mesh),
    for the sculpt helper: the measured endpoint, the candidate's first hit and its normal, all in W."""
    out = {}
    fit = base["use"] & ~ev["verify"] & sc["hit"]
    for k, r in enumerate(rec["regions"]):
        at = np.flatnonzero(fit & (base["ray_patch"] == k))  # the whole patch: the margin pins the push field
        views, counts = np.unique(ev["view"][at], return_counts=True)  # views with too few rays there are left out
        at = at[np.isin(ev["view"][at], views[counts >= s["min_samples"]])]
        if len(at) > s["correspondences"]:
            at = at[np.linspace(0, len(at) - 1, s["correspondences"]).round().astype(int)]
        d = (ev["end"][at] - ev["origin"][at]) / sc["range"][at, None]
        h = ev["origin"][at] + d * (sc["range"][at] + sc["err"][at])[:, None]
        n = mesh.normals[sc["tri"][at]] if mesh is not None else np.zeros_like(h)
        out[r["id"]] = {"end": np.round(ev["end"][at], 6).tolist(), "hit": np.round(h, 6).tolist(),
                        "normal": np.round(n, 5).tolist(), "view": ev["view"][at].tolist(),
                        "weight": np.round(ev["weight"][at] / max(ev["weight"][at].sum(), 1e-12), 8).tolist()}
    return out


def review_outputs(work, control, glb, review=None, extra=None):
    """The LiDAR part of a review of the host's copy glb (tools.review): ({file name: bytes} for the receipt folder,
    summary lines), or (None, lines) for a session without evidence or when scoring fails (the review still works).
    lidar.json holds each region's original and current fit on the fitting rays, the coordinates to select it and its
    fitting correspondences at this candidate; extra: fields to add (receipt, blend_sha256, ...)."""
    root = Path(control).parent
    m = active(control)
    if m is None:
        return None, []
    try:
        rec = read_json(Path(control) / "lidar" / "baseline.json")
        if not isinstance(rec, dict) or rec.get("key") != m["key"] or rec.get("surfaces_version") != SURFACES:
            rec = baseline(root)
        ev, s = load_evidence(control), m["settings"]
        with np.load(Path(control) / "lidar" / "baseline.npz") as f:
            base = {k: f[k] for k in f.files}
        sc, mesh = cached_score(control, glb, ev, m["key"])
        mesh = mesh or Mesh(glb)
        cur = current_fit(rec, ev, sc, base, s)
        regs = [{**r, "current": cur[r["id"]]} for r in rec["regions"]]
        corr = correspondences(rec, ev, sc, base, mesh, s)
        doc = {**agent_summary({**rec, "regions": regs}, m), **(extra or {}), "evidence_key": m["key"],
               "correspondences": corr}
        files = {"lidar.json": json.dumps(doc, indent=1).encode()}
        files.update(overlays(root, ev, sc, s, base["use"], review))
        lines = summary_lines(rec, cur)
        return files, lines
    except Exception as e:  # noqa: BLE001 - the review itself still works
        return None, [f"lidar: scoring failed ({type(e).__name__}: {e}); the review has no LiDAR overlay"]


def silhouettes(control, glb, size):
    """Mask IoU per view of the GLB's silhouette under the host's cameras and masks (mvsam3d.view_ious), at size px."""
    from harpor.stage_a.mvsam3d import view_ious
    from harpor.stage_a.place import load_mask

    m = read_json(Path(control) / "manifest.json")
    mesh = Mesh(glb)
    folder = Path(control) / "lidar" / "masks"
    masks = [load_mask(folder / f"{k:03d}.png") if (folder / f"{k:03d}.png").is_file() else None
             for k in range(len(m["views"]))]
    cams = [v for v, k in zip(m["views"], masks) if k is not None]
    return view_ious(np.asarray(mesh.vertices), np.asarray(mesh.faces), cams, [k for k in masks if k is not None],
                     size)


def evaluate(glb, control, original):
    """The host's LiDAR check of a stored candidate glb against original (root/original.glb), on the reserved rays:
    {status, key, regions: [{id, status, gain_m, before, after, coverage, see_through, views}], protected,
    silhouettes, see_through, counts, resolved, gain_m, regressed, mode, findings}. A see-through ray (its measured
    point lies more than max_displacement_m beyond the original's hit) does not count toward coverage. Statuses of a
    region: improved, tie, regressed, insufficient (no reserved view observes it) or inconsistent (its fitting groups
    disagree). The whole: unavailable, error, regressed, improved or tie. Never raises: a scorer failure is
    {status: error}."""
    m = active(control)
    if m is None:
        mm = manifest(control)
        return {"status": "unavailable", "mode": (mm or {}).get("mode"), "reason": (mm or {}).get("reason"),
                "findings": []}
    try:
        return _evaluate(glb, control, original, m)
    except Exception as e:  # noqa: BLE001 - the report says it, and the round decides
        return {"status": "error", "mode": m["mode"], "key": m["key"], "error": f"{type(e).__name__}: {e}",
                "findings": [_finding("lidar_error", why=f"the LiDAR scorer failed: {type(e).__name__}: {e}; the "
                                                         "candidate's geometry is not verified", counts=True)]}


def _finding(code, measured=None, expected=None, why="", counts=False):
    return {"code": code, "measured": measured, "expected": expected, "why": why, "counts": bool(counts)}


def _per_view_abs(err, hit, sel, view, miss):
    out = {}
    for i in np.unique(view[sel]):
        e = np.where(hit, np.abs(err), miss)[sel & (view == i)]
        out[int(i)] = float(np.median(e)) if len(e) else None
    return out


def _evaluate(glb, control, original, m):
    control = Path(control)
    root = control.parent
    s = m["settings"]
    tol, sculpt = s["range_tol_m"], m["mode"]["effective"] == "sculpt"
    rec = read_json(control / "lidar" / "baseline.json")
    if not isinstance(rec, dict) or rec.get("key") != m["key"] or rec.get("surfaces_version") != SURFACES:
        rec = baseline(root)
    with np.load(control / "lidar" / "baseline.npz") as f:
        base = {k: f[k] for k in f.files}
    ev = load_evidence(control)
    sc0, _ = cached_score(control, original, ev, m["key"])
    sc1, _ = cached_score(control, glb, ev, m["key"])
    miss = s["miss_error_rel"] * rec["diagonal_m"]
    ver = base["use"] & ev["verify"]
    view = ev["view"]
    # A see-through ray: its measured point lies farther beyond the original's first hit than a sculpt pass may move a
    # surface (max_displacement_m), so the laser passed where the original has a surface (a wire cage, slats, a gap)
    # rather than showing a shape error. A candidate that opens the surface there agrees with that ray, so losing its
    # hit is no coverage loss, and its miss keeps the original's error (neutral: opening a surface gains nothing
    # either). Every other lost hit counts, a bulge of a few centimeters included.
    see0 = sc0["hit"] & (sc0["err"] < -s["max_displacement_m"])
    open1 = see0 & ~sc1["hit"]
    err1, hit1 = np.where(open1, sc0["err"], sc1["err"]), sc1["hit"] | open1
    findings, out_regions = [], []
    regressed = False

    def judge(sel, min_n):
        """(before {view: median abs}, after, coverage, losses beyond tol, gain, see-through rays) of the rays sel, by
        reserved view. The coverage is the share of the original's hits that the candidate keeps, of the rays that
        are not see-through."""
        counts = {int(i): int((sel & (view == i)).sum()) for i in np.unique(view[sel])}
        strong = sel & np.isin(view, [i for i, c in counts.items() if c >= min_n])
        b = _per_view_abs(sc0["err"], sc0["hit"], strong, view, miss)
        a = _per_view_abs(err1, hit1, strong, view, miss)
        solid = strong & sc0["hit"] & ~see0
        hits0 = int(solid.sum())
        cov = None if not hits0 else float((solid & sc1["hit"]).sum() / hits0)
        loss = {i: a[i] - b[i] for i in b if b[i] is not None and a[i] is not None and a[i] - b[i] > tol}
        gain = None if not b else float(np.median(list(b.values())) - np.median(list(a.values())))
        return b, a, cov, loss, gain, {"rays": int((strong & see0).sum()), "opened": int((strong & open1).sum())}

    for k, r in enumerate(rec["regions"]):
        sel = ver & (base["ray_region"] == k)
        b, a, cov, loss, gain, see = judge(sel, s["min_samples"])
        x = {"id": r["id"], "region_status": r["status"], "before_m": b, "after_m": a, "coverage": cov,
             "gain_m": gain, "losses_m": loss, "see_through": see}
        fit_sel = base["use"] & ~ev["verify"] & (base["ray_region"] == k)
        x["fitting_gain_m"] = None if not fit_sel.any() else float(
            np.median(np.where(sc0["hit"], np.abs(sc0["err"]), miss)[fit_sel])
            - np.median(np.where(hit1, np.abs(err1), miss)[fit_sel]))
        if r["status"] == "conflict":
            x["status"] = "inconsistent"
        elif not b:
            x["status"] = "insufficient"
        elif cov is not None and cov < 1 - s["coverage_tol"]:
            x["status"] = "regressed"
        elif loss:
            x["status"] = "regressed"
        elif gain is not None and gain > tol:
            x["status"] = "improved"
        else:
            x["status"] = "tie"
        if x["status"] == "regressed":
            regressed = True
        out_regions.append(x)
    # Protected: reserved rays on no region, by surface; and each region's band.
    protected = []
    for c in np.unique(base["ray_surface"][ver & (base["ray_region"] < 0)]):
        if c < 0:
            continue
        sel = ver & (base["ray_region"] < 0) & (base["ray_surface"] == c)
        b, a, cov, loss, _, see = judge(sel, s["min_samples"])
        bad = bool(loss) or (cov is not None and cov < 1 - s["coverage_tol"])
        protected.append({"surface": int(c), "before_m": b, "after_m": a, "coverage": cov, "losses_m": loss,
                          "see_through": see, "regressed": bad})
        regressed |= bad
    for k, r in enumerate(rec["regions"]):
        sel = ver & (base["ray_band"] == k)
        if not sel.any():
            continue
        b, a, cov, loss, _, see = judge(sel, max(5, s["min_samples"] // 5))
        bad = bool(loss) or (cov is not None and cov < 1 - s["coverage_tol"])
        protected.append({"band_of": r["id"], "before_m": b, "after_m": a, "coverage": cov, "losses_m": loss,
                          "see_through": see, "regressed": bad})
        regressed |= bad
    sil = None
    try:
        if not isinstance(rec.get("silhouettes"), list):
            raise ValueError(f"the original has no silhouettes: {rec.get('silhouettes')}")
        i0 = np.asarray(rec["silhouettes"], float)
        i1 = silhouettes(control, glb, s["silhouette_px"])
        drop = (i0 - i1).tolist()
        sil = {"before": np.round(i0, 4).tolist(), "after": np.round(i1, 4).tolist(),
               "regressed": [k for k, d in enumerate(drop) if d > s["silhouette_tol"]]}
        regressed |= bool(sil["regressed"])
    except Exception as e:  # noqa: BLE001 - recorded; the ray check still decides
        sil = {"error": f"{type(e).__name__}: {e}"}
    targets = [x for x, r in zip(out_regions, rec["regions"]) if r["status"] == "target" and r.get("verifiable")]
    resolved = sum(x["status"] == "improved" for x in targets)
    gains = [x["gain_m"] or 0.0 for x in targets]
    for x, r in zip(out_regions, rec["regions"]):
        if x["status"] == "regressed":
            findings.append(_finding("lidar_regression", measured={"gain_m": x["gain_m"], "coverage": x["coverage"],
                                                                   "losses_m": x["losses_m"]},
                                     expected={"gain_m": f"> {tol}", "coverage": f">= {1 - s['coverage_tol']}"},
                                     why=f"region {r['id']} got worse on reserved returns", counts=sculpt))
        elif r["status"] == "target" and r.get("verifiable") and x["status"] != "improved":
            findings.append(_finding("lidar_target", measured={"gain_m": x["gain_m"], "status": x["status"]},
                                     expected={"gain_m": f"> {tol}"},
                                     why=f"region {r['id']} ({r['direction']}) still disagrees with the returns: "
                                         f"seed {fmt(r['seed'])}, normal {fmt(r['normal'])}; reselect it from that "
                                         "seed and normal", counts=sculpt))
        elif r["status"] == "target" and not r.get("verifiable"):
            findings.append(_finding("lidar_coverage", measured=x["status"],
                                     why=f"region {r['id']} has no reserved view: inspection only"))
        elif r["status"] == "conflict":
            findings.append(_finding("lidar_alignment", measured=r.get("endpoint_gap_m"),
                                     why=f"the views disagree on region {r['id']}: a registration finding, not a "
                                         "repair"))
    for p in protected:
        if p["regressed"]:
            where = f"surface {p['surface']}" if "surface" in p else f"the band around {p['band_of']}"
            findings.append(_finding("lidar_regression", measured={"losses_m": p["losses_m"], "coverage": p["coverage"]},
                                     why=f"{where} got worse on reserved returns", counts=sculpt))
    if sil and sil.get("regressed"):
        findings.append(_finding("lidar_regression", measured={"views": sil["regressed"]},
                                 expected=f"IoU loss <= {s['silhouette_tol']}", why="silhouettes got worse",
                                 counts=sculpt))
    if rec["alignment"]["common_shift"] or rec["alignment"]["excluded"]:
        findings.append(_finding("lidar_alignment", measured={"common_shift": rec["alignment"]["common_shift"],
                                                              "excluded_views": rec["alignment"]["excluded"]},
                                 why="registration: the LiDAR evidence does not support a local repair there"))
    status = "regressed" if regressed else "improved" if resolved else "tie"
    return {"status": status, "mode": m["mode"], "key": m["key"], "settings": s, "regions": out_regions,
            "see_through": {"rays": int((ver & see0).sum()), "opened": int((ver & open1).sum()),
                            "of": int((ver & sc0["hit"]).sum())},
            "protected": protected, "silhouettes": sil, "targets": len(targets), "resolved": resolved,
            "gain_m": float(np.mean(gains)) if gains else 0.0, "regressed": regressed, "findings": findings,
            "candidate_sha256": sha256_file(glb)}


def rank_terms(report):
    """(resolved targets, the mean gain over the targets in whole multiples of range_tol_m) of a report's LiDAR check,
    for controller.rank: a gain inside the tolerance is a tie."""
    lid = (report or {}).get("lidar") or {}
    if lid.get("status") not in ("improved", "tie", "regressed"):
        return 0, 0
    tol = (lid.get("settings") or {}).get("range_tol_m") or 1.0
    return int(lid.get("resolved") or 0), int(math.floor((lid.get("gain_m") or 0.0) / tol))


def demonstrably_worse(report):
    """Whether a report's LiDAR check found the candidate worse in sculpt mode, where that excludes it from delivery."""
    lid = (report or {}).get("lidar") or {}
    return bool(lid.get("regressed")) and (lid.get("mode") or {}).get("effective") == "sculpt"
