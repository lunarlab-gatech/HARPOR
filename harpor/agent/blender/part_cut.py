"""What add_part.py and move_faces.py share (build_plan.md note 11): the tag layers, the face selection on a bmesh,
and the piece of a tagged mesh with its cut holes capped.

The scripts import it after putting their own folder first on sys.path, as they import render_common. A face's
tag (the int face layer TAG) says which side of the cut it is on: NEW or REST; CANDIDATE is used only while planes
select. A cut edge has 1 in the int edge layer SEAM (cutting.py). The selection maths is part_select.py, in
plain numpy. Only bpy, bmesh, mathutils and numpy.
"""
# Untested: needs Blender 5.2, which the test environment does not have.

import bmesh
import bpy
import numpy as np
from mathutils import Vector

import part_select as S
from cutting import cap

TAG, SEAM = "gator_new", "gator_seam"
CAP, CANDIDATE, NEW, REST = 3, 2, 1, 0  # CAP: move_faces.py tags the caps of earlier cuts, which it rebuilds


def layers(bm):
    """(tag layer, seam layer) of bm, both at zero. They must exist before any element is read: a new layer
    invalidates the elements. A mesh that still has them (a cut that another tool left half done) starts from zero."""
    tl = bm.faces.layers.int.get(TAG) or bm.faces.layers.int.new(TAG)
    sl = bm.edges.layers.int.get(SEAM) or bm.edges.layers.int.new(SEAM)
    for f in bm.faces:
        f[tl] = REST
    for e in bm.edges:
        e[sl] = 0
    return tl, sl


def arrays(bm, M):
    """(vertices in W, vertex indices per face, face centres in W, face pairs per shared edge) of bm."""
    bm.verts.index_update()
    bm.faces.index_update()
    co = np.array([v.co[:] for v in bm.verts], float).reshape(-1, 3)
    vw = co @ M[:3, :3].T + M[:3, 3]
    fv = [[v.index for v in f.verts] for f in bm.faces]
    centers = np.array([vw[i].mean(0) for i in fv], float).reshape(-1, 3)
    pairs = []
    for e in bm.edges:
        fs = [f.index for f in e.link_faces]
        pairs += list(zip(fs, fs[1:]))
    return vw, fv, centers, pairs


def diagonals(bm, M):
    """(the diagonal of bm's box in its own frame, and in W); 1.0 for a mesh without extent."""
    co = np.array([v.co[:] for v in bm.verts], float).reshape(-1, 3)
    return (float(np.linalg.norm(np.ptp(co, 0))) or 1.0), (float(np.linalg.norm(np.ptp(co @ M[:3, :3].T, 0))) or 1.0)


def select(bm, tl, spec, M, source):
    """Tag the faces of bm that spec selects NEW and weld the vertices that the glTF import split, so that shared
    edges are shared (as split_parts.py does). spec: {"faces": PATH}, polygon indices of the mesh as the .blend
    holds it, tagged before the weld, which can drop a face; or {"planes": [...], "seed": [...]} in W: the faces on
    the normal's side of every plane, and of those the connected piece nearest the seed. Only the faces near that
    piece are cut along the planes (bisect_plane), so the cut edge is straight and the rest keeps its faces; a face
    that crosses the region without a vertex in it is missed. M: the object's world matrix; source: its name, for
    the errors (ValueError)."""
    bm.faces.ensure_lookup_table()
    local_diag, diag = diagonals(bm, M)
    tol = 1e-6 * diag
    if "faces" in spec:
        for i in S.face_indices(np.load(spec["faces"], allow_pickle=False), len(bm.faces)):
            bm.faces[int(i)][tl] = NEW
        bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=1e-6 * local_diag)
        return
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=1e-6 * local_diag)
    planes, seed = S.unit_planes(spec["planes"]), np.array(spec["seed"], float)
    vw, fv, centers, pairs = arrays(bm, M)
    labels, _ = S.components(len(fv), pairs, S.touching(S.inside(vw, planes, tol), fv))
    k = S.nearest_piece(labels, centers, seed)
    if k < 0:
        raise ValueError(f"no face of {source} has a vertex on the normal's side of every plane")
    for f in bm.faces:
        f[tl] = CANDIDATE if labels[f.index] == k else REST
    Minv = np.linalg.inv(M)
    for p in planes:  # the plane in the object's own frame
        point = Minv[:3, :3] @ p[:3] + Minv[:3, 3]
        normal = M[:3, :3].T @ p[3:]
        faces = [f for f in bm.faces if f[tl] == CANDIDATE]
        geom = list({v for f in faces for v in f.verts}) + list({e for f in faces for e in f.edges}) + faces
        bmesh.ops.bisect_plane(bm, geom=geom, dist=1e-6 * local_diag, plane_co=Vector(point.tolist()),
                               plane_no=Vector(normal.tolist()).normalized(), clear_inner=False, clear_outer=False)
    vw, fv, centers, pairs = arrays(bm, M)
    tags = np.array([f[tl] for f in bm.faces])
    labels, _ = S.components(len(fv), pairs, (tags == CANDIDATE) & S.inside(centers, planes, tol))
    k = S.nearest_piece(labels, centers, seed)
    if k < 0:
        raise ValueError("the planes select no face")
    for f in bm.faces:
        f[tl] = NEW if labels[f.index] == k else REST


def pieces(bm, tl, tag, M):
    """How many connected pieces the faces of bm with this tag form, by shared edges."""
    _, fv, _, pairs = arrays(bm, M)
    return S.components(len(fv), pairs, np.array([f[tl] == tag for f in bm.faces]))[1]


def mark_cut(bm, tl, sl):
    """Make every edge between faces of different tags a cut edge, sharp."""
    for e in bm.edges:
        if len({f[tl] for f in e.link_faces}) > 1:
            e[sl] = 1
            e.smooth = False


def piece(me, new_name, keep, warnings):
    """(mesh, counts): a copy of the tagged mesh me with only the faces tagged keep, its cut holes capped
    (cutting.cap) and the two layers removed. counts: faces, cap_faces, holes, fill and open_cut_edges, the cut
    edges that no cap could close. A mesh that Blender had to correct adds a line to warnings."""
    pm = me.copy()
    pm.name = new_name
    bm = bmesh.new()
    bm.from_mesh(pm)
    tl, sl = bm.faces.layers.int.get(TAG), bm.edges.layers.int.get(SEAM)
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if f[tl] != keep], context="FACES")
    faces = len(bm.faces)
    holes, methods = cap(bm, sl)
    counts = {"faces": faces, "cap_faces": len(bm.faces) - faces, "holes": holes, "fill": methods,
              "open_cut_edges": sum(1 for e in bm.edges if e.is_boundary and e[sl])}
    bm.faces.layers.int.remove(tl)
    bm.edges.layers.int.remove(sl)
    bm.to_mesh(pm)
    bm.free()
    if pm.validate(clean_customdata=False):  # True when Blender had to correct the mesh data
        warnings.append(f"{new_name}: Blender corrected invalid mesh data")
    return pm, counts


def mesh_area(me):
    a = np.empty(len(me.polygons))
    me.polygons.foreach_get("area", a)
    return float(a.sum())


def is_cap(f, uv):
    """Whether a face is a cap of an earlier cut: every corner has one UV (cutting.cap), which is also how
    harpor.agent.surface tells a cap from the object's own surface. False for a mesh without UVs."""
    if uv is None:
        return False
    first = f.loops[0][uv].uv
    return all((lp[uv].uv - first).length < 1e-9 for lp in f.loops)


def shown_meshes():
    return [o for o in bpy.context.scene.objects if o.visible_get() and o.type == "MESH"]


def link(name, what="object"):
    """The visible mesh object of this name; ValueError without one."""
    obj = bpy.data.objects.get(name)
    if obj is None or obj.type != "MESH":
        raise ValueError(f"no mesh {what} is named {name!r}")
    if not obj.visible_get():
        raise ValueError(f"{name} is hidden, so export_glb leaves it out; show it first")
    return obj


def save(warnings):
    try:
        bpy.ops.file.pack_all()
    except RuntimeError as e:  # an image without a file; the .blend still saves
        warnings.append(f"pack_all: {e}")
    bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
