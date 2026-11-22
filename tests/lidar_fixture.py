"""A known-geometry LiDAR job for the sculpting tests (tests/test_lidar_geometry.py; docs/lidar_sculpting_plan.md step 1).

An asymmetric, textured box (36 x 18 x 24 cm) with triangulated panels, filmed by 8 cameras on a ring 30 degrees above
it, 1 m away. Each view's LiDAR returns are the analytic first hits of rays on the six clean box planes (box_hit), not
the production scorer, so a mistake there cannot also make the fixture pass. View OFFSET_VIEW's LiDAR sits 8 cm to the
side of its camera; the others measure along the camera's own rays (the aligned-camera model). The masks come from the
same analytic box. The bottom panel is never measured. W is view 0's OpenCV camera, as in a real job.

job() writes a job folder as the stages leave it (job.json, pool/lidar.json, selection.json, run/ with cameras.json,
posed.glb, images, masks, prep.json, placement.json and lidar/geometry/), from one of the meshes: "clean", "bowed" (the
right panel bowed outward by BOW_M at its center), "dent" (the right panel dented inward by DENT_M), "deleted" (the right
panel removed), "no_top" (the top panel removed) or "shifted" (the whole box moved by SHIFT_M). With see_through, the
returns on that panel lie SEE_M farther along their rays: the laser went through it, as through a wire cage, to a far
surface. truth() holds what the agent must not see: the damaged panel, the clean dimensions and the clean vertices.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from harpor import frames as Fr

DIMS = np.array([0.36, 0.18, 0.24])  # x right, y up, z depth in the box's own frame
YAW_DEG = 20.0
RADIUS, ELEV_DEG, N_VIEWS = 1.0, 30.0, 8
SIZE, FOCAL = (320, 240), 300.0
STEP_PX = 2  # one LiDAR ray every 2 px
OFFSET_VIEW, OFFSET_M = 5, 0.08
SPACING = 0.02  # panel grid
BOW_M, DENT_M, DENT_SIGMA, SHIFT_M = 0.020, 0.025, 0.12, 0.012
SEE_M = 0.30  # how far beyond a see-through panel its returns lie
RIGHT = (0, +1)  # the damaged panel: +x
TOP = (1, +1)
PANELS = [(0, +1), (0, -1), (1, +1), (1, -1), (2, +1), (2, -1)]  # (axis, sign); (1, -1) is the bottom
COLORS = {(0, 1): (200, 70, 60), (0, -1): (60, 160, 90), (1, 1): (230, 200, 80), (1, -1): (90, 90, 90),
          (2, 1): (70, 110, 200), (2, -1): (180, 120, 200)}


def yaw():
    a = np.radians(YAW_DEG)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def look_at(C, target=np.zeros(3), up=np.array([0.0, 1.0, 0.0])):
    """OpenCV world-to-camera (x right, y down, z forward) from C toward target, in the design frame (y up)."""
    z = (target - C) / np.linalg.norm(target - C)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, -R @ C
    return T


def design_cameras():
    el = np.radians(ELEV_DEG)
    out = []
    for k in range(N_VIEWS):
        az = np.radians(-30.0 + 45.0 * k)
        C = RADIUS * np.array([np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)])
        out.append(look_at(C))
    return out


def K():
    w, h = SIZE
    return np.array([[FOCAL, 0, (w - 1) / 2], [0, FOCAL, (h - 1) / 2], [0, 0, 1.0]])


ANCHOR = [0]  # the design camera that is view 0 (job(anchor=...) sets it): W is its frame


def world():
    """(W from the design frame, the box's design pose R, the cameras' w2c in W), views in order from ANCHOR[0]."""
    D = design_cameras()
    D = D[ANCHOR[0]:] + D[:ANCHOR[0]]
    W_from_D = D[0]
    return W_from_D, yaw(), [Dk @ np.linalg.inv(W_from_D) for Dk in D]


# ---- the box mesh ----

def panel_grid(axis, sign, spacing=SPACING):
    """(vertices in the box frame, triangles, uv) of one panel, a grid with its own vertices (UV seams at the edges)."""
    a, b = [i for i in range(3) if i != axis]
    na, nb = max(2, int(round(DIMS[a] / spacing))), max(2, int(round(DIMS[b] / spacing)))
    s, t = np.meshgrid(np.linspace(-0.5, 0.5, na + 1), np.linspace(-0.5, 0.5, nb + 1), indexing="ij")
    V = np.zeros((s.size, 3))
    V[:, axis] = sign * DIMS[axis] / 2
    V[:, a], V[:, b] = s.ravel() * DIMS[a], t.ravel() * DIMS[b]
    idx = np.arange(s.size).reshape(na + 1, nb + 1)
    q = np.stack([idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()], 1)
    Fc = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
    n = np.zeros(3)
    n[axis] = sign
    tri = V[Fc]
    if (np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]) @ n < 0).any():
        Fc = Fc[:, ::-1]
    k = PANELS.index((axis, sign))
    uv = np.c_[(s.ravel() + 0.5 + (k % 3)) / 3, (t.ravel() + 0.5 + (k // 3)) / 2]
    return V, Fc, uv, (s.ravel(), t.ravel())


def damage(V, st, panel, kind):
    """Panel vertices after the damage kind (in the box frame)."""
    s, t = st
    axis, sign = panel
    V = V.copy()
    if kind == "bowed":
        V[:, axis] += sign * BOW_M * (1 - (2 * s) ** 2) * (1 - (2 * t) ** 2)
    elif kind == "dent":
        V[:, axis] -= sign * DENT_M * np.exp(-((s - 0.1) ** 2 + (t + 0.05) ** 2) / (2 * DENT_SIGMA ** 2)) \
            * (1 - (2 * s) ** 2) * (1 - (2 * t) ** 2)
    return V


def box_parts(kind="clean"):
    """[(panel, vertices in W, triangles, uv)] of the mesh of kind (see the module docstring)."""
    W_from_D, R, _ = world()
    shift = np.array([SHIFT_M, -0.004, 0.006]) if kind == "shifted" else np.zeros(3)
    out = []
    for p in PANELS:
        if (kind == "deleted" and p == RIGHT) or (kind == "no_top" and p == TOP):
            continue
        V, Fc, uv, st = panel_grid(*p)
        if p == RIGHT and kind in ("bowed", "dent"):
            V = damage(V, st, p, kind)
        Vd = V @ R.T + shift
        out.append((p, Fr.transform_points(W_from_D, Vd), Fc, uv))
    return out


def texture():
    """An asymmetric texture: six colored tiles with a checker and a mark in one corner."""
    img = np.zeros((128, 192, 3), np.uint8)
    for k, p in enumerate(PANELS):
        r, c = k // 3, k % 3
        tile = np.array(COLORS[p], np.uint8) * np.ones((64, 64, 1), np.uint8)
        yy, xx = np.mgrid[:64, :64]
        tile[((yy // 8 + xx // 8) % 2) == 0] = (tile[((yy // 8 + xx // 8) % 2) == 0] * 0.75).astype(np.uint8)
        tile[4:14, 4:20] = 255
        img[64 * (1 - r):64 * (2 - r), 64 * c:64 * (c + 1)] = tile
    return Image.fromarray(img)


def node_matrix(k):
    """A nonidentity node transform per panel (nodes=True): a turn about glTF y, a nonuniform scale and a shift."""
    a = np.radians(15.0 + 10 * k)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    M = np.eye(4)
    M[:3, :3] = R @ np.diag([1.0, 1.3, 0.8])
    M[:3, 3] = [0.05 * k, -0.02, 0.03]
    return M


def write_glb(path, kind="clean", nodes=False, tint=None):
    """The mesh of kind as a GLB in glTF axes (frames.w_to_gltf), one primitive per panel with a shared texture. nodes:
    store each panel in local coordinates under a nonidentity node transform (node_matrix), so that a scorer must
    apply the node transforms. tint: a color that replaces the texture's (a texture-only repair)."""
    import trimesh

    tex = texture()
    if tint is not None:
        tex = Image.fromarray((np.asarray(tex) * 0.5 + np.array(tint) * 0.5).astype(np.uint8))
    scene = trimesh.Scene()
    for k, (p, V, Fc, uv) in enumerate(box_parts(kind)):
        mat = trimesh.visual.material.PBRMaterial(baseColorTexture=tex, metallicFactor=0.0, roughnessFactor=0.8)
        G = Fr.w_to_gltf(V)
        M = node_matrix(k) if nodes else np.eye(4)
        Minv = np.linalg.inv(M)
        m = trimesh.Trimesh(G @ Minv[:3, :3].T + Minv[:3, 3], Fc, visual=trimesh.visual.TextureVisuals(uv=uv, material=mat),
                            process=False)
        scene.add_geometry(m, node_name=f"panel{k}", geom_name=f"panel{k}", transform=M)
    Path(path).write_bytes(scene.export(file_type="glb"))
    return path


# ---- analytic rays ----

def box_hit(origin_W, dirs_W):
    """(distance, panel index) of each ray's first hit on the clean box, nan and -1 for a miss: the slab method in the
    box's frame. Independent of trimesh."""
    W_from_D, R, _ = world()
    D_from_W = np.linalg.inv(W_from_D)
    o = Fr.transform_points(D_from_W, origin_W) @ R  # R^T applied as row vectors: into the box frame
    d = dirs_W @ D_from_W[:3, :3].T @ R
    half = DIMS / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (-half - o) / d, (half - o) / d
    tn, tf = np.minimum(t1, t2), np.maximum(t1, t2)
    near = np.nanmax(tn, 1)
    far = np.nanmin(tf, 1)
    axis = np.nanargmax(tn, 1)
    ok = (near <= far) & (near > 1e-9)
    sign = -np.sign(d[np.arange(len(d)), axis])
    panel = np.array([PANELS.index((int(a), int(s))) if s else -1 for a, s in zip(axis, sign)])
    return np.where(ok, near, np.nan), np.where(ok, panel, -1)


def pixels():
    w, h = SIZE
    u, v = np.meshgrid(np.arange(0, w, STEP_PX), np.arange(0, h, STEP_PX))
    return np.c_[u.ravel(), v.ravel()].astype(float)


def rays_of(view, w2c, offset=None):
    """(origin_W, dir_W, origin_c) of a view's LiDAR rays: through every STEP_PX-th pixel, from the camera or from
    a sensor offset by offset (camera frame), parallel to the camera's rays."""
    px = pixels()
    d_c = np.c_[(px[:, 0] - K()[0, 2]) / FOCAL, (px[:, 1] - K()[1, 2]) / FOCAL, np.ones(len(px))]
    d_c /= np.linalg.norm(d_c, axis=1)[:, None]
    c2w = np.linalg.inv(w2c)
    o_c = np.zeros(3) if offset is None else np.asarray(offset, float)
    return np.repeat(Fr.transform_points(c2w, o_c[None]), len(px), 0), d_c @ c2w[:3, :3].T, o_c


def render(w2c):
    """(photo, mask) of the clean box from a camera: an analytic ray per pixel, textured by panel."""
    w, h = SIZE
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    px = np.c_[u.ravel(), v.ravel()].astype(float)
    d_c = np.c_[(px[:, 0] - K()[0, 2]) / FOCAL, (px[:, 1] - K()[1, 2]) / FOCAL, np.ones(len(px))]
    d_c /= np.linalg.norm(d_c, axis=1)[:, None]
    c2w = np.linalg.inv(w2c)
    t, panel = box_hit(np.repeat(c2w[:3, 3][None], len(px), 0), d_c @ c2w[:3, :3].T)
    img = np.full((len(px), 3), 120, np.uint8)
    hit = panel >= 0
    P = c2w[:3, 3] + d_c @ c2w[:3, :3].T * np.nan_to_num(t)[:, None]
    chk = ((np.floor(P[:, 0] / 0.03) + np.floor(P[:, 1] / 0.03) + np.floor(P[:, 2] / 0.03)) % 2) == 0
    for k, p in enumerate(PANELS):
        sel = panel == k
        col = np.array(COLORS[p], float)
        img[sel] = np.where(chk[sel, None], col * 0.75, col).astype(np.uint8)
    return img.reshape(h, w, 3), hit.reshape(h, w)


# ---- the job folder ----

def returns(view, w2c, mask, see_through=None):
    """{xyz (camera), origin (camera), pixel, row, flags} of a view's LiDAR returns on the clean box; those on the panel
    see_through lie SEE_M farther along their rays."""
    from harpor.agent import lidar_geometry as G

    offset = np.array([OFFSET_M, 0.0, 0.0]) if view == OFFSET_VIEW else None
    o_W, d_W, o_c = rays_of(view, w2c, offset)
    t, panel = box_hit(o_W, d_W)
    if see_through is not None:
        t = np.where(panel == PANELS.index(see_through), t + SEE_M, t)
    hit = panel >= 0
    p_W = o_W[hit] + d_W[hit] * t[hit][:, None]
    xyz = Fr.transform_points(w2c, p_W)
    uv = xyz[:, :2] / xyz[:, 2:3] * FOCAL + K()[:2, 2]
    iu, iv = np.floor(uv + 0.5).astype(int).T
    w, h = SIZE
    inside = (iu >= 0) & (iu < w) & (iv >= 0) & (iv < h)
    iu, iv = np.clip(iu, 0, w - 1), np.clip(iv, 0, h - 1)
    flags = np.zeros(len(xyz), np.uint16)
    flags[inside & mask[iv, iu]] |= G.IN_MASK
    pad = np.pad(mask, 1, mode="edge")
    edge = np.zeros_like(mask)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            edge |= pad[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] != mask
    flags[edge[iv, iu]] |= G.BOUNDARY
    flags[~inside] |= G.EXCLUDED
    if offset is not None:  # the camera must see the endpoint the offset sensor hit
        flags |= G.OFFSET
        c2w = np.linalg.inv(w2c)
        dc = p_W - c2w[:3, 3]
        rng = np.linalg.norm(dc, axis=1)
        tc, _ = box_hit(np.repeat(c2w[:3, 3][None], len(p_W), 0), dc / rng[:, None])
        flags[~(np.abs(tc - rng) < 1e-6)] |= G.HIDDEN
    return {"xyz": xyz, "origin": o_c, "pixel": uv, "row": np.arange(len(xyz), dtype=np.int64), "flags": flags}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def job(J, kind="clean", misregister=None, profile="synthetic", anchor=0, nodes=False, see_through=None):
    """A job folder after place (see the module docstring), with the posed mesh of kind. misregister: a view whose
    camera in cameras.json is turned by 2 degrees and moved 1.5 cm, its returns unchanged (a registration error).
    profile: the geometry index's profile (None: an uncalibrated source). anchor: the design camera that is view 0.
    nodes: the GLB keeps its panels under node transforms. see_through: a panel that the laser goes through (returns).
    Returns J."""
    from harpor.agent import lidar_geometry as G

    ANCHOR[0] = anchor
    J = Path(J)
    run = J / "run"
    for d in ("input/images", "masks", "lidar/geometry"):
        (run / d).mkdir(parents=True, exist_ok=True)
    (J / "lidar" / "geometry").mkdir(parents=True, exist_ok=True)
    (J / "pool").mkdir(parents=True, exist_ok=True)
    _, _, cams = world()
    entries, views = [], []
    for i, w2c in enumerate(cams):
        photo, mask = render(w2c)
        Image.fromarray(photo).save(run / "input" / "images" / f"{i}.png")
        Image.fromarray(mask.astype(np.uint8) * 255).save(run / "masks" / f"{i}.png")
        r = returns(i, w2c, mask, see_through)
        f = run / "lidar" / "geometry" / f"{i}.npz"
        np.savez(f, **r)
        views.append({"index": i, "pool_name": f"f{10 * i:05d}", "status": "available", "file": f.name,
                      "sha256": sha(f), "model": "lidar_ray" if i == OFFSET_VIEW else "aligned_depth_camera",
                      "confidence": "none", "returns": int(len(r["xyz"])),
                      "time": {"window_ns": [i * 10**9, i * 10**9 + 10**8], "scans": [i]}})
        T = np.asarray(w2c, float)
        if misregister == i:
            a = np.radians(2.0)
            T = np.array([[np.cos(a), 0, np.sin(a), 0.015], [0, 1, 0, 0], [-np.sin(a), 0, np.cos(a), 0],
                          [0, 0, 0, 1]]) @ T
        entries.append({"name": str(i), "K": K().tolist(), "w2c": T.tolist(), "width": SIZE[0], "height": SIZE[1],
                        "image": str(run / "input" / "images" / f"{i}.png"), "mask": str(run / "masks" / f"{i}.png")})
    kind_of = "synthetic" if profile else "stray"  # an uncalibrated source without a profile
    pool_index = {"version": G.VERSION, "kind": kind_of, "profile": profile,
                  "views": {v["pool_name"]: {k: v[k] for k in ("status", "model", "confidence", "returns")}
                            for v in views}}
    (J / "lidar" / "geometry" / "index.json").write_text(json.dumps(pool_index))
    pool_sha = sha(J / "lidar" / "geometry" / "index.json")
    (run / "lidar" / "geometry" / "index.json").write_text(json.dumps({
        "version": G.VERSION, "kind": kind_of, "profile": profile, "pool_index_sha256": pool_sha,
        "metric_scale": 1.0, "views": views}))
    (run / "cameras.json").write_text(json.dumps(entries))
    write_glb(run / "posed.glb", kind, nodes=nodes)
    (run / "prep.json").write_text(json.dumps({"source": "webapp pool", "metric_scale": 1.0,
                                               "views": [{"name": str(i), "pool_name": v["pool_name"]}
                                                         for i, v in enumerate(views)]}))
    (run / "placement.json").write_text(json.dumps({"metric_scale": 1.0, "check": {"mean_mask_iou": 0.95}}))
    (J / "selection.json").write_text(json.dumps({"ok": True, "metric_scale": {"s": 1.0}}))
    (J / "pool" / "lidar.json").write_text(json.dumps({
        "version": 2, "ok": True, "s": 1.0, "extrinsic": {"calibration_sha256": None},
        "geometry": {"index": "lidar/geometry/index.json", "sha256": pool_sha}}))
    (J / "job.json").write_text(json.dumps({
        "id": "20261122-000000-abcdef", "prompt": "box", "slug": "box", "agent": "opus",
        "source": {"kind": "stray", "topic": None}, "lidar": {"kind": "stray"}, "calibration": None,
        "lidar_geometry": "sculpt"}))
    return J


def truth():
    """What the tests know and the agent must not: the damaged panel, the clean dimensions and clean vertices."""
    return {"panel": RIGHT, "dims": DIMS.copy(),
            "clean": {p: V for p, V, _, _ in box_parts("clean")}}
