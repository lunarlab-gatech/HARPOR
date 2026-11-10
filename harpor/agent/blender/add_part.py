"""Cut a new part out of a part object in the agent's .blend (build_plan.md note 11).

blender -b --factory-startup --disable-autoexec candidate.blend --python-exit-code 1 --python add_part.py -- OUT.json NAME SOURCE SPEC

For a part of the part list that the split did not find. NAME is the new object (tools.add_part checks that
it is the reserve of a requirement), SOURCE the visible mesh object to cut it from, and SPEC JSON text with
the selection, one of:

  {"faces": PATH}                  a .npy file: polygon indices of SOURCE as the .blend holds them, or one bool
                                   per polygon. The agent chose them with its own code, so any shape works.
  {"planes": [[px, py, pz, nx, ny, nz], ...], "seed": [x, y, z]}
                                   in W: the part is on the normal's side of every plane, and of those faces the
                                   connected piece nearest the seed. Only the faces near that piece are cut along
                                   the planes (bisect_plane), so the cut edge is straight and the rest of SOURCE
                                   keeps its faces. A face that crosses the region without a vertex in it is missed.

The selection maths is part_select.py. The steps: tag the faces, weld the vertices that the glTF import split
(as split_parts.py does, so that shared edges are shared), mark the edges between the two sides as cut edges,
and make two meshes from the tagged one: the new part and what is left of SOURCE, each with its cut holes
capped (cutting.py). The new object gets SOURCE's materials, UVs, transform, parent and collections, and no
joint. An empty selection, or one that holds every face of SOURCE, is an error.

Writes OUT.json: ok, error, name, source, the faces, cap faces, holes and open cut edges of both sides, the
new part's pieces (connected pieces of its faces; more than 1 is usually a mistake), its area, its share of
SOURCE's area and of all visible meshes', its box in W, and the visible mesh objects (the links). OUT.json is
created exclusively before anything changes, and the .blend is saved only when the cut succeeded, so an error
leaves it as it was.
"""
# Untested: needs Blender 5.2, which the test environment does not have.

import json
import sys
import time
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
import part_select as S  # noqa: E402
from cutting import cap  # noqa: E402

TAG, SEAM = "gator_new", "gator_seam"
CANDIDATE, NEW, REST = 2, 1, 0
out, name, source, spec = sys.argv[sys.argv.index("--") + 1:][:4]
spec = json.loads(spec)
t0 = time.time()
record = open(out, "x")  # first, so an existing OUT.json stops the script before the .blend changes
rec = {"ok": False, "error": None, "name": name, "source": source, "warnings": []}


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


def piece(me, new_name, keep):
    """(mesh, counts): a copy of the tagged mesh me with only the faces tagged keep, its cut holes capped."""
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
        rec["warnings"].append(f"{new_name}: Blender corrected invalid mesh data")
    return pm, counts


def mesh_area(me):
    a = np.empty(len(me.polygons))
    me.polygons.foreach_get("area", a)
    return float(a.sum())


def cut():
    obj = bpy.data.objects.get(source)
    if obj is None or obj.type != "MESH":
        raise ValueError(f"no mesh object is named {source!r}")
    if not obj.visible_get():
        raise ValueError(f"{source} is hidden, so export_glb leaves it out; show it first")
    if bpy.data.objects.get(name) is not None:
        raise ValueError(f"an object named {name!r} exists already")
    shown = [o for o in bpy.context.scene.objects if o.visible_get() and o.type == "MESH"]
    total = sum(mesh_area(o.data) for o in shown)
    me = obj.data
    if me.users > 1:  # another object shares the mesh; only SOURCE is cut
        me = obj.data = me.copy()
    M = np.array(obj.matrix_world, float)
    bm = bmesh.new()
    bm.from_mesh(me)
    # The layers come before any element is read: a new layer invalidates the elements. A mesh that still has
    # them (a cut that another tool left half done) starts from zero.
    tl = bm.faces.layers.int.get(TAG) or bm.faces.layers.int.new(TAG)
    sl = bm.edges.layers.int.get(SEAM) or bm.edges.layers.int.new(SEAM)
    bm.faces.ensure_lookup_table()
    if not len(bm.faces):
        raise ValueError(f"{source} has no faces")
    for f in bm.faces:
        f[tl] = REST
    for e in bm.edges:
        e[sl] = 0
    co = np.array([v.co[:] for v in bm.verts], float).reshape(-1, 3)
    local_diag = float(np.linalg.norm(np.ptp(co, 0))) or 1.0
    diag = float(np.linalg.norm(np.ptp(co @ M[:3, :3].T, 0))) or 1.0
    tol = 1e-6 * diag

    if "faces" in spec:
        for i in S.face_indices(np.load(spec["faces"], allow_pickle=False), len(bm.faces)):
            bm.faces[int(i)][tl] = NEW
        bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=1e-6 * local_diag)
    else:
        bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=1e-6 * local_diag)
        planes, seed = S.unit_planes(spec["planes"]), np.array(spec["seed"], float)
        vw, fv, centers, pairs = arrays(bm, M)
        near = S.touching(S.inside(vw, planes, tol), fv)
        labels, _ = S.components(len(fv), pairs, near)
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
                                   plane_no=Vector(normal.tolist()).normalized(), clear_inner=False,
                                   clear_outer=False)
        vw, fv, centers, pairs = arrays(bm, M)
        tags = np.array([f[tl] for f in bm.faces])
        labels, _ = S.components(len(fv), pairs, (tags == CANDIDATE) & S.inside(centers, planes, tol))
        k = S.nearest_piece(labels, centers, seed)
        if k < 0:
            raise ValueError("the planes select no face")
        for f in bm.faces:
            f[tl] = NEW if labels[f.index] == k else REST

    new = [f for f in bm.faces if f[tl] == NEW]
    if not new:
        raise ValueError("the selection holds no face")
    if len(new) == len(bm.faces):
        raise ValueError(f"the selection holds every face of {source}; a part is a piece of it")
    _, fv, _, pairs = arrays(bm, M)
    _, pieces = S.components(len(fv), pairs, np.array([f[tl] == NEW for f in bm.faces]))
    area, source_area = sum(f.calc_area() for f in new), sum(f.calc_area() for f in bm.faces)
    for e in bm.edges:
        if len({f[tl] for f in e.link_faces}) > 1:
            e[sl] = 1
            e.smooth = False
    bm.to_mesh(me)
    bm.free()

    old_name = me.name
    part, part_counts = piece(me, name, NEW)
    rest, rest_counts = piece(me, old_name + "_cut", REST)
    obj.data = rest
    bpy.data.meshes.remove(me)
    rest.name = old_name
    ob = bpy.data.objects.new(name, part)
    for c in obj.users_collection:
        c.objects.link(ob)
    ob.parent, ob.parent_type, ob.parent_bone = obj.parent, obj.parent_type, obj.parent_bone
    ob.matrix_parent_inverse = obj.matrix_parent_inverse.copy()
    ob.matrix_basis = obj.matrix_basis.copy()
    if ob.name != name:
        raise ValueError(f"Blender named the new object {ob.name}, not {name}")
    pv = np.empty(len(part.vertices) * 3)
    part.vertices.foreach_get("co", pv)
    pw = pv.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
    rec.update(part=part_counts, rest=rest_counts, pieces=pieces, area=area,
               share_of_source=area / source_area if source_area else None,
               share_of_object=area / total if total else None,
               box={"min": pw.min(0).tolist(), "max": pw.max(0).tolist()})
    try:
        bpy.ops.file.pack_all()
    except RuntimeError as e:  # an image without a file; the .blend still saves
        rec["warnings"].append(f"pack_all: {e}")
    bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
    rec["ok"] = True


try:
    cut()
except Exception as e:  # the .blend on disk is as it was: it is saved only at the end of cut()
    rec["error"] = f"{type(e).__name__}: {e}" if not isinstance(e, ValueError) else str(e)
rec.update(links=[o.name for o in bpy.context.scene.objects if o.visible_get() and o.type == "MESH"],
           seconds=round(time.time() - t0, 2))
with record as f:
    json.dump(rec, f, indent=2)
