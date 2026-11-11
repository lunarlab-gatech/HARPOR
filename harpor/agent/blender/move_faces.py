"""Move faces from one part object to another in the agent's .blend (build_plan.md note 11).

blender -b --factory-startup --disable-autoexec candidate.blend --python-exit-code 1 --python move_faces.py -- OUT.json SOURCE TARGET SPEC

For a part that the split got wrong: SOURCE holds surface that belongs to TARGET. Both are visible mesh
objects, such as two parts, or a part and the body. SPEC is the selection on SOURCE, as add_part.py takes it:
{"faces": PATH} or {"planes": [...], "seed": [...]} (part_cut.select).

The two objects are split again as one surface with a new labelling, as split_parts.py splits the whole object,
so neither is left with faces inside it (a link that is not closed makes the host's clearance check
clearance_missing):
1. The selected faces of a copy of SOURCE's mesh are tagged for TARGET, the others stay SOURCE's, and the caps
   of both copies are tagged as caps (a cap has one UV for all its corners, part_cut.is_cap).
2. The copies are joined (bpy.ops.object.join, which merges material slots, UV layers and the face tags) and
   welded: the seam between the two closes, because both came from one surface.
3. The caps are deleted, and the edges that bounded them become cut edges where they are open, so the holes
   toward other parts are capped again in step 5.
4. The edges between faces of different tags become cut edges.
5. Two meshes are made from the joined one (part_cut.piece), each capped, and given to SOURCE and TARGET.
Both objects keep their name, joint (the custom property gator_joint), transform, parent and collections. The
objects must have the same world transform (the parts of a split do). An empty selection, one that holds all
of SOURCE's own surface (that would remove the link), and SOURCE == TARGET are errors.

Writes OUT.json: ok, error, source, target, moved (the faces that changed sides, caps not counted), the moved
area and its share of SOURCE's surface before, after ({object: its faces, cap faces, holes, open cut edges and
connected pieces now}), and the visible mesh objects (the links). OUT.json is created exclusively
before anything changes, and the .blend is saved only when the move succeeded, so an error leaves it as it was.
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

out, source, target, spec = sys.argv[sys.argv.index("--") + 1:][:4]
spec = json.loads(spec)
t0 = time.time()
record = open(out, "x")  # first, so an existing OUT.json stops the script before the .blend changes
rec = {"ok": False, "error": None, "source": source, "target": target, "warnings": []}


def tagged(mesh, M, name, tag=None):
    """(a copy of mesh with every face tagged, the number of its own faces tagged NEW, their area, the area and the
    number of all its own faces). A cap of an earlier cut is tagged CAP, whatever the selection says: move() deletes
    the caps after the join and caps the new parts again. tag None: the faces that spec selects are NEW and the
    others REST (the copy of SOURCE); else every face of its own gets tag (the copy of TARGET). Face tags survive
    bpy.ops.object.join; edge marks set before it did not (seen in Blender 5.2), so no edge is marked here."""
    me = mesh.copy()
    bm = bmesh.new()
    bm.from_mesh(me)
    tl, _ = P.layers(bm)
    if not len(bm.faces):
        raise ValueError(f"{name} has no faces")
    if tag is None:
        P.select(bm, tl, spec, M, name)
    else:
        for f in bm.faces:
            f[tl] = tag
    uv = bm.loops.layers.uv.active
    for f in bm.faces:
        if P.is_cap(f, uv):
            f[tl] = P.CAP
    own = [f for f in bm.faces if f[tl] != P.CAP]
    new = [f for f in own if f[tl] == P.NEW]
    counts = (len(new), sum(f.calc_area() for f in new), sum(f.calc_area() for f in own), len(own))
    bm.to_mesh(me)
    bm.free()
    return (me, *counts)


def move():
    if source == target:
        raise ValueError(f"{source} cannot give faces to itself")
    a, b = P.link(source, "object (--from)"), P.link(target, "object (--to)")
    Ma, Mb = np.array(a.matrix_world, float), np.array(b.matrix_world, float)
    if not np.allclose(Ma, Mb, atol=1e-9):
        raise ValueError(f"{source} and {target} have different transforms; a part object is never moved, rotated "
                         "or scaled")
    ma, moved, area, own, faces = tagged(a.data, Ma, source)
    if not moved:
        raise ValueError("the selection holds no face of the part's own surface")
    if moved == faces:
        raise ValueError(f"the selection holds all of {source}'s surface; that would remove the link")
    mb = tagged(b.data, Mb, target, P.NEW)[0]

    # Join copies, not the objects: join deletes every object but the active one, and the two must stay.
    ta, tb = bpy.data.objects.new("__gator_a", ma), bpy.data.objects.new("__gator_b", mb)
    for o in (ta, tb):
        bpy.context.scene.collection.objects.link(o)
        o.select_set(True)
    bpy.context.view_layer.objects.active = ta
    with bpy.context.temp_override(object=ta, active_object=ta, selected_objects=[ta, tb],
                                   selected_editable_objects=[ta, tb]):
        bpy.ops.object.join()
    joined = ta.data
    bm = bmesh.new()
    bm.from_mesh(joined)
    sl = bm.edges.layers.int.get(P.SEAM) or bm.edges.layers.int.new(P.SEAM)  # before any element is read
    tl = bm.faces.layers.int.get(P.TAG)
    if tl is None:
        raise ValueError("the joined mesh lost its face tags")
    for e in bm.edges:
        e[sl] = 0
    local_diag, _ = P.diagonals(bm, Ma)
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=1e-6 * local_diag)
    # The caps go, and the edges that bounded them are cut edges where they are open now: the holes toward the
    # other parts, which piece() caps again. The seam between the two objects closed in the weld.
    caps = [f for f in bm.faces if f[tl] == P.CAP]
    edges = {e for f in caps for e in f.edges}
    bmesh.ops.delete(bm, geom=caps, context="FACES")
    for e in edges:
        if e.is_valid and e.is_boundary:
            e[sl] = 1
    P.mark_cut(bm, tl, sl)
    counts = {source: P.pieces(bm, tl, P.REST, Ma), target: P.pieces(bm, tl, P.NEW, Ma)}
    bm.to_mesh(joined)
    bm.free()

    after = {}
    for obj, keep in ((a, P.REST), (b, P.NEW)):
        old = obj.data
        name = old.name
        new, c = P.piece(joined, name + "_cut", keep, rec["warnings"])
        if not c["faces"]:
            raise ValueError(f"{obj.name} would have no faces left")
        obj.data = new
        if old.users == 0:
            bpy.data.meshes.remove(old)
        new.name = name
        after[obj.name] = {**c, "pieces": counts[obj.name]}
    bpy.data.objects.remove(ta, do_unlink=True)  # join already removed the other copy and its mesh
    if joined.users == 0:
        bpy.data.meshes.remove(joined)
    rec.update(moved=moved, area=area, share_of_source=area / own if own else None, after=after)
    P.save(rec["warnings"])
    rec["ok"] = True


try:
    move()
except Exception as e:  # the .blend on disk is as it was: it is saved only at the end of move()
    rec["error"] = f"{type(e).__name__}: {e}" if not isinstance(e, ValueError) else str(e)
rec.update(links=[o.name for o in P.shown_meshes()], seconds=round(time.time() - t0, 2))
with record as f:
    json.dump(rec, f, indent=2)
