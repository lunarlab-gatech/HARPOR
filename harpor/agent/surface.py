"""How much of the original's surface a candidate keeps: the source map rho of paper Sec. 3.2 and A.2 (build_plan.md
note 5).

compare(glb, original) reads both GLBs with harpor.physics.mjcf.load_s. The session's GLBs are all in
W with glTF axes and in the same units, so their numbers compare one to one. It never raises: any failure gives
{"error": ...}, which the caller reports as surface_error.

Matching: D is the diagonal of the original's box. The original's vertices merge where their positions are the same
numbers, so the copies of a vertex on a UV seam are one, and go into a cKDTree. Each candidate vertex is the merged
vertex within VERTEX_TOL D of it. A candidate face matches an original face on the same three vertices (sorted ids),
preferring one that winds the same way. Unlike quantized positions, this cannot flip a vertex across a bin edge after
a float round trip through Blender.

Categories, as shares of the native area:
- retained: matched faces with the same UVs (every corner within UV_TOL), the same winding and the same material.
- modified: matched faces with a change, and unmatched faces whose centroid and three edge midpoints all lie within
  NEAR D of the candidate's surface, sampled at SAMPLES points per (NEAR D)^2, where one of the NEIGHBOURS samples
  nearest the centroid lies on a candidate face that faces the same way. A cut or a remesh makes new faces on the
  same surface, and the facing rule keeps the back of a panel thinner than NEAR D from standing in for its deleted
  front. Such a face's counterpart is the candidate face of that sample. uv_changed and material_changed are the
  faces whose counterpart has that change, flipped the matched faces that wind the other way; a candidate face
  without UVs where the original has them has changed its UVs. On a re-cut face only lost UVs show, since its
  corners are not the original's.
- deleted: the other unmatched faces. present is 1 - deleted.
- unresolved: matched faces without UVs in the original and with no other change, since their UVs cannot be checked.
- added: candidate faces that match no original face. It never blocks, whatever its material, because the prompts
  ask for cavities and trays.
Measured with this rule on the meshes of an office chair: a plane through the seat crosses 0.2% to 12.5% of the
native area, yet a 1.2% cut was judged fully present, and a deleted 2.97% patch reported 2.76%, in 0.07 s.
On a synthetic sheet of 9,800 faces compare takes 0.02 to 0.06 s with 64-pixel images, and 0.4 s with three
2048-pixel images, because each pair of primitives compares its images.

Faces without UV area are not native: a cap that closes a cut gets one UV at every corner, so such faces of the
original are left out of every share and of the native area, and a GLB without UVs has none. A primitive whose UVs
are all one point has no UVs: Blender's join gives an object without UVs a UV of 0 at every corner, and its faces
are native.

Materials: each pair of an original primitive and a candidate primitive among the faces and their counterparts
compares the factors (within FACTOR_TOL) and the decoded base color, normal and metallic-roughness images at full
resolution. An image differs when only one material has it, when its size changed, or when the mean difference in
some tile of a TILES x TILES grid is above IMAGE_TOL. Measured on the office chair's texture: JPEG at quality 95, 90 and
75 gives 1.38, 1.92 and 2.68, a blur 30.1, an 8x downscale 12.5 and a patch of 2% of the image in its mean colour 18.9.
So a renamed or re-encoded image is the same image. An image of more than MAX_IMAGE_PIXELS pixels is not decoded: its
GLB is an error here (large_image, which reads the size from the image's header), and harpor.agent.controller.
check_glb refuses it, since a small GLB can hold one.

Findings (LIMITS, shares of the native area): deleted above 0.02 is surface_deleted, uv_changed above 0.01
surface_uv, material_changed above 0.01 surface_material, flipped above 0.01 surface_flipped, and present below
IDENTITY surface_identity. The limits come from reasoning, not from paired data. ENFORCE = False keeps them
report-only (blocks is then always False) until a real run calibrates them.

The report holds diagonal, native_area, caps_area, shares {retained, modified, uv_changed, material_changed,
flipped, deleted, unresolved, added, present}, lost_regions [{min, max, share}] (up to REGIONS boxes in W around the
largest connected sets of deleted faces), materials [{original, candidate, why, share}] of the pairs that differ,
findings [{code, measured, expected, why}], blocks and seconds.

It runs on the host in envs/sam3d, with numpy, scipy and trimesh, which it imports only when it runs.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

VERTEX_TOL = 1e-5  # of D: a candidate vertex this close to an original vertex is that vertex
UV_TOL = 1e-4  # a UV corner further than this from the original's has changed
NEAR = 0.005  # of D: an unmatched face this close to the candidate's surface is modified, further is deleted
SAMPLES = 4  # of the candidate's surface, per (NEAR D)^2
MAX_SAMPLES = 2_000_000
NEIGHBOURS = 16  # samples near a centroid that may face its way: on a panel thinner than a sample's spacing both
# sides interleave, and all 16 lie on the back with odds of 2^-16
CAP_UV_AREA = 1e-12  # of the UV square, far below a texel: a face with less is a cap
TILES = 16  # per side of an image
IMAGE_TOL = 6.0  # of 255: the largest mean difference in a tile of two images that are the same
FACTOR_TOL = 6 / 255
IMAGES = ("baseColorTexture", "normalTexture", "metallicRoughnessTexture")
MAX_IMAGE_PIXELS = 8192 * 8192  # of one image, which tile_difference holds as RGBA in int16, 8 bytes a pixel
# glTF's defaults; trimesh keeps the base colour factor in 0..255
FACTORS = {"baseColorFactor": [255, 255, 255, 255], "metallicFactor": 1.0, "roughnessFactor": 1.0,
           "emissiveFactor": [0.0, 0.0, 0.0]}
LIMITS = {"deleted": (0.02, "surface_deleted", "deleted"), "uv_changed": (0.01, "surface_uv", "with other UVs"),
          "material_changed": (0.01, "surface_material", "with another material or image"),
          "flipped": (0.01, "surface_flipped", "facing the other way")}  # share: (limit, finding, what it is)
IDENTITY = 0.5  # the smallest present share
REGIONS = 5
ENFORCE = False


@dataclass
class Surface:
    """Every face of a GLB as one mesh: uv is NaN for a primitive without UVs, prim the primitive of each face, and
    materials that of each primitive."""

    vertices: np.ndarray
    faces: np.ndarray
    uv: np.ndarray
    prim: np.ndarray
    materials: list


def _uv(mesh):
    """The UVs of a primitive, NaN without them or when they are all one point (see the module docstring)."""
    uv = getattr(mesh.visual, "uv", None)
    if uv is None or len(uv) != len(mesh.vertices) or not np.ptp(np.asarray(uv, float), 0).max(initial=0) > 0:
        return np.full((len(mesh.vertices), 2), np.nan)
    return np.asarray(uv, float)


def large_image(meshes):
    """Why an image of the meshes' materials is too large to decode (over MAX_IMAGE_PIXELS), None when none is.
    trimesh opens the images without decoding them, so the size comes from the image's header."""
    for m in meshes:
        material = getattr(getattr(m, "visual", None), "material", None)
        for k in IMAGES:
            w, h = getattr(getattr(material, k, None), "size", (0, 0))
            if w * h > MAX_IMAGE_PIXELS:
                return f"its {k} is {w}x{h} pixels, over {MAX_IMAGE_PIXELS:,}"
    return None


def read_surface(glb):
    """The Surface of a GLB in W (harpor.physics.mjcf.load_s); primitives without faces are left out. ValueError for an
    image too large to decode (large_image)."""
    from harpor.physics.mjcf import load_s

    meshes = [m for _, m in load_s(glb) if len(getattr(m, "faces", ()))]
    why = large_image(meshes)
    if why:
        raise ValueError(f"{Path(glb).name}: {why}")
    offsets = np.cumsum([0] + [len(m.vertices) for m in meshes])
    return Surface(np.concatenate([np.zeros((0, 3))] + [m.vertices for m in meshes]),
                   np.concatenate([np.zeros((0, 3), int)] + [m.faces + o for m, o in zip(meshes, offsets)]),
                   np.concatenate([np.zeros((0, 2))] + [_uv(m) for m in meshes]),
                   np.repeat(np.arange(len(meshes)), [len(m.faces) for m in meshes]),
                   [getattr(m.visual, "material", None) for m in meshes])


def normals(points, faces):
    """Each face's normal, twice its area long (F, 3)."""
    a, b, c = (points[faces[:, k]] for k in range(3))
    return np.cross(b - a, c - a)


def areas(points, faces):
    return 0.5 * np.linalg.norm(normals(points, faces), axis=1)


def uv_areas(uv, faces):
    """The UV area of each face, NaN without UVs."""
    a, b, c = (uv[faces[:, k]] for k in range(3))
    return 0.5 * np.abs((b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0])


def parity(ids):
    """The parity of the permutation that sorts each face's three vertex ids: two faces on the same vertices wind the
    same way when theirs is equal."""
    return ((ids[:, 0] > ids[:, 1]).astype(int) + (ids[:, 0] > ids[:, 2]) + (ids[:, 1] > ids[:, 2])) % 2


def match(orig, cand, tol):
    """(points, ids, cand_ids, j, flipped, used): the original's merged vertices, those of each original face and of
    each candidate face (-1 for a candidate vertex with none within tol), the candidate face that matches each
    original face (-1 for none), whether that face winds the other way, and the mask of candidate faces that match
    some original face. A face with a repeated vertex matches nothing."""
    from scipy.spatial import cKDTree

    points, inverse = np.unique(orig.vertices, axis=0, return_inverse=True)
    ids = inverse.reshape(-1)[orig.faces]
    dist, near = cKDTree(points).query(cand.vertices, distance_upper_bound=tol)
    cand_ids = np.where(dist <= tol, near, -1)[cand.faces]
    so, sc = np.sort(ids, 1), np.sort(cand_ids, 1)
    o = np.flatnonzero((so[:, 0] < so[:, 1]) & (so[:, 1] < so[:, 2]))
    c = np.flatnonzero((sc[:, 0] >= 0) & (sc[:, 0] < sc[:, 1]) & (sc[:, 1] < sc[:, 2]))
    keys, key = np.unique(np.concatenate([so[o], sc[c]]), axis=0, return_inverse=True)
    okey, ckey = key.reshape(-1)[:len(o)], key.reshape(-1)[len(o):]
    best = np.full((len(keys), 2), -1)  # a candidate face of each key, by parity
    best[ckey, parity(cand_ids[c])] = c
    same, other = best[okey, parity(ids[o])], best[okey, 1 - parity(ids[o])]
    j, flipped = np.full(len(ids), -1), np.zeros(len(ids), bool)
    j[o], flipped[o] = np.where(same >= 0, same, other), (same < 0) & (other >= 0)
    seen, used = np.zeros(len(keys), bool), np.zeros(len(cand_ids), bool)
    seen[okey] = True
    used[c] = seen[ckey]
    return points, ids, cand_ids, j, flipped, used


def corners(uv, faces, ids):
    """The UVs of each face's corners in the order of their merged vertex ids (F, 3, 2)."""
    return np.take_along_axis(uv[faces], np.argsort(ids, 1)[..., None], 1)


def uv_moved(orig, cand, ids, cand_ids, i, j):
    """Whether candidate face j[k] moved a UV corner of original face i[k] by more than UV_TOL; a candidate face
    without UVs moved them all."""
    a, b = corners(orig.uv, orig.faces[i], ids[i]), corners(cand.uv, cand.faces[j], cand_ids[j])
    return np.isnan(b).any((1, 2)) | (np.linalg.norm(a - b, axis=2) > UV_TOL).any(1)


def tile_difference(a, b):
    """The largest mean absolute difference, of 255, over a TILES x TILES grid of tiles of two decoded images of one
    size (fewer tiles for an image under TILES pixels across)."""
    d = np.abs(np.asarray(a.convert("RGBA"), np.int16) - np.asarray(b.convert("RGBA"), np.int16)).sum(2)
    rows, cols = (np.unique(np.linspace(0, n, TILES + 1).astype(int)) for n in d.shape)
    sums = np.add.reduceat(np.add.reduceat(d, rows[:-1], 0), cols[:-1], 1)
    return float((sums / np.outer(np.diff(rows), np.diff(cols)) / 4).max())


def image_change(a, b):
    """Why decoded image b differs from a, None when it does not."""
    if a is None or b is None:
        return None if a is b else ("removed" if b is None else "added")
    if a.size != b.size:
        return f"resized from {a.size[0]}x{a.size[1]} to {b.size[0]}x{b.size[1]}"
    worst = tile_difference(a, b)
    return f"differs by {worst:.1f}/255 in a tile" if worst > IMAGE_TOL else None


def factor(material, name):
    """A material's factor in 0..1, glTF's default when it has none."""
    v = getattr(material, name, None)
    v = np.asarray(FACTORS[name] if v is None else v, float).ravel()
    return v / 255 if name == "baseColorFactor" else v


def material_change(a, b):
    """Why material b differs from material a ([] when it does not): a factor beyond FACTOR_TOL or a changed image."""
    if a is None or b is None:
        return [] if a is b else ["no material" if b is None else "a material where the original has none"]
    why = [f"{k} {np.round(factor(a, k), 3).tolist()} to {np.round(factor(b, k), 3).tolist()}" for k in FACTORS
           if factor(a, k).shape != factor(b, k).shape or np.abs(factor(a, k) - factor(b, k)).max() > FACTOR_TOL]
    return why + [f"{k} {w}" for k in IMAGES if (w := image_change(getattr(a, k, None), getattr(b, k, None)))]


def material_pairs(orig, cand, i, j, weight):
    """(changed, changes): whether the materials of original face i[k] and candidate face j[k] differ, and
    [{original, candidate, why, share}] of each primitive pair that does; weight[k] is face i[k]'s share."""
    pairs, pair = np.unique(np.c_[orig.prim[i], cand.prim[j]].reshape(-1, 2), axis=0, return_inverse=True)
    pair = pair.reshape(-1)
    whys = [material_change(orig.materials[a], cand.materials[b]) for a, b in pairs]
    shares = np.bincount(pair, weight, len(pairs))
    changes = [{"original": getattr(orig.materials[a], "name", None),
                "candidate": getattr(cand.materials[b], "name", None), "why": why, "share": round(float(s), 5)}
               for (a, b), why, s in zip(pairs, whys, shares) if why]
    return np.array([bool(w) for w in whys], bool)[pair], changes


def probes(points, faces):
    """Each face's centroid and three edge midpoints (F, 4, 3)."""
    a, b, c = (points[faces[:, k]] for k in range(3))
    return np.stack([(a + b + c) / 3, (a + b) / 2, (b + c) / 2, (c + a) / 2], 1)


def near_surface(probe, normal, cand, tol):
    """(near, face): whether every probe point of each face lies within tol of the candidate's surface, sampled at
    SAMPLES points per tol^2, with one of the NEIGHBOURS samples nearest its centroid within tol on a candidate face
    that faces the same way (normal: each face's), and for a face that is near, the candidate face of the nearest
    such sample (else -1)."""
    import trimesh
    from scipy.spatial import cKDTree

    mesh = trimesh.Trimesh(cand.vertices, cand.faces, process=False)
    if not len(probe) or not mesh.area > 0:
        return np.zeros(len(probe), bool), np.full(len(probe), -1)
    n = int(np.clip(SAMPLES * mesh.area / tol ** 2, 1000, MAX_SAMPLES))
    points, faces = trimesh.sample.sample_surface(mesh, n, seed=0)
    tree = cKDTree(points)
    dist, k = tree.query(probe[:, 0], NEIGHBOURS)
    face = faces[k]
    facing = (dist <= tol) & (np.einsum("fkx,fx->fk", normals(cand.vertices, cand.faces)[face], normal) > 0)
    near = facing.any(1) & (tree.query(probe[:, 1:].reshape(-1, 3))[0].reshape(-1, 3) <= tol).all(1)
    return near, np.where(near, face[np.arange(len(face)), facing.argmax(1)], -1)


def lost_regions(points, ids, lost, area, total):
    """Up to REGIONS boxes {min, max, share} in W around the largest sets of lost faces that share vertices, largest
    first."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    f = ids[lost]
    if not len(f):
        return []
    edges = (np.r_[f[:, 0], f[:, 1]], np.r_[f[:, 1], f[:, 2]])
    graph = coo_matrix((np.ones(2 * len(f)), edges), shape=(len(points), len(points)))
    label = connected_components(graph, directed=False)[1][f[:, 0]]
    sizes = np.bincount(label, area[lost])
    out = []
    for c in np.argsort(-sizes)[:REGIONS]:
        if sizes[c] > 0:
            p = points[f[label == c].ravel()]
            out.append({"min": np.round(p.min(0), 4).tolist(), "max": np.round(p.max(0), 4).tolist(),
                        "share": round(float(sizes[c] / total), 5)})
    return out


def finding(code, measured, expected, why):
    return {"code": code, "measured": measured, "expected": expected, "why": why}


def findings(shares):
    """The findings of the shares, the one that changes the object's identity first."""
    out = [finding("surface_identity", shares["present"], f"at least {IDENTITY}",
                   "under half of the native surface is still present")] if shares["present"] < IDENTITY else []
    return out + [finding(code, shares[k], f"at most {limit}", f"share of the native surface {what}")
                  for k, (limit, code, what) in LIMITS.items() if shares[k] > limit]


def _compare(glb, original):
    t0 = time.time()
    orig, cand = read_surface(original), read_surface(glb)
    if not len(orig.faces):
        raise ValueError("the original has no faces")
    diag = float(np.linalg.norm(np.ptp(orig.vertices[orig.faces.ravel()], 0)))
    area = areas(orig.vertices, orig.faces)
    has_uv = ~np.isnan(orig.uv[orig.faces]).any((1, 2))
    native = ~(uv_areas(orig.uv, orig.faces) < CAP_UV_AREA)  # NaN, without UVs, is native
    total = float(area[native].sum())
    if not total > 0:
        raise ValueError("the original has no native area")
    points, ids, cand_ids, j, flipped, used = match(orig, cand, VERTEX_TOL * diag)
    matched = j >= 0
    unmatched = np.flatnonzero(native & ~matched)
    near, other = np.zeros(len(j), bool), j.copy()
    near[unmatched], other[unmatched] = near_surface(probes(orig.vertices, orig.faces[unmatched]),
                                                     normals(orig.vertices, orig.faces[unmatched]), cand, NEAR * diag)
    i, m, k = np.flatnonzero(matched), np.flatnonzero(near), np.flatnonzero(matched | near)
    uv_changed, material_changed = np.zeros(len(j), bool), np.zeros(len(j), bool)
    uv_changed[i] = has_uv[i] & uv_moved(orig, cand, ids, cand_ids, i, j[i])
    uv_changed[m] = has_uv[m] & np.isnan(cand.uv[cand.faces[other[m]]]).any((1, 2))
    material_changed[k], materials = material_pairs(orig, cand, k, other[k], area[k] * native[k] / total)
    changed = uv_changed | material_changed | flipped
    deleted = native & ~matched & ~near

    def share(mask):
        return float(area[mask & native].sum() / total)

    shares = {"retained": share(matched & ~changed & has_uv), "modified": share((matched & changed) | near),
              "uv_changed": share(uv_changed), "material_changed": share(material_changed), "flipped": share(flipped),
              "deleted": share(deleted), "unresolved": share(matched & ~changed & ~has_uv),
              "added": float(areas(cand.vertices, cand.faces)[~used].sum() / total), "present": 1 - share(deleted)}
    shares = {k: round(v, 5) for k, v in shares.items()}
    found = findings(shares)
    return {"diagonal": diag, "native_area": float(f"{total:.6g}"), "caps_area": float(f"{area[~native].sum():.6g}"),
            "shares": shares,
            "lost_regions": lost_regions(points, ids, deleted, area, total), "materials": materials,
            "findings": found, "blocks": ENFORCE and bool(found), "seconds": round(time.time() - t0, 2)}


def compare(glb, original):
    """The surface report of candidate glb against original (see the module docstring). It never raises: a failure
    gives {"error": ...}."""
    try:
        return _compare(glb, original)
    except Exception as e:  # noqa: BLE001 - the caller records the report, so every failure goes there
        return {"error": f"{type(e).__name__}: {e}"}
