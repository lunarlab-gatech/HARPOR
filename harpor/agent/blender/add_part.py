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

The selection is part_cut.select (its maths is part_select.py), which move_faces.py shares. The steps: tag the
faces, weld the vertices that the glTF import split (as split_parts.py does, so that shared edges are shared),
mark the edges between the two sides as cut edges, and make two meshes from the tagged one: the new part and
what is left of SOURCE, each with its cut holes capped (part_cut.piece, cutting.py). The new object gets SOURCE's materials, UVs, transform, parent and collections, and no
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import part_cut as P  # noqa: E402

out, name, source, spec = sys.argv[sys.argv.index("--") + 1:][:4]
spec = json.loads(spec)
t0 = time.time()
record = open(out, "x")  # first, so an existing OUT.json stops the script before the .blend changes
rec = {"ok": False, "error": None, "name": name, "source": source, "warnings": []}


def cut():
    obj = P.link(source)
    if bpy.data.objects.get(name) is not None:
        raise ValueError(f"an object named {name!r} exists already")
    total = sum(P.mesh_area(o.data) for o in P.shown_meshes())
    me = obj.data
    if me.users > 1:  # another object shares the mesh; only SOURCE is cut
        me = obj.data = me.copy()
    M = np.array(obj.matrix_world, float)
    bm = bmesh.new()
    bm.from_mesh(me)
    tl, sl = P.layers(bm)
    if not len(bm.faces):
        raise ValueError(f"{source} has no faces")
    P.select(bm, tl, spec, M, source)
    new = [f for f in bm.faces if f[tl] == P.NEW]
    if not new:
        raise ValueError("the selection holds no face")
    if len(new) == len(bm.faces):
        raise ValueError(f"the selection holds every face of {source}; a part is a piece of it")
    pieces = P.pieces(bm, tl, P.NEW, M)
    area, source_area = sum(f.calc_area() for f in new), sum(f.calc_area() for f in bm.faces)
    P.mark_cut(bm, tl, sl)
    bm.to_mesh(me)
    bm.free()

    old_name = me.name
    part, part_counts = P.piece(me, name, P.NEW, rec["warnings"])
    rest, rest_counts = P.piece(me, old_name + "_cut", P.REST, rec["warnings"])
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
    P.save(rec["warnings"])
    rec["ok"] = True


try:
    cut()
except Exception as e:  # the .blend on disk is as it was: it is saved only at the end of cut()
    rec["error"] = f"{type(e).__name__}: {e}" if not isinstance(e, ValueError) else str(e)
rec.update(links=[o.name for o in P.shown_meshes()], seconds=round(time.time() - t0, 2))
with record as f:
    json.dump(rec, f, indent=2)
