"""Split the posed mesh into one object per part, cut and capped, and export parts.glb (build_plan.md note 11).

blender -b --factory-startup --python-exit-code 1 --python split_parts.py -- POSED.glb FACES.npz OUT.glb REPORT.json

FACES.npz is the face vote of gator.parts.vote: part (one id per face of POSED.glb as trimesh loads
it), names (id -> part name, the base first) and centroids (every face's centroid in W). The stock glTF
importer maps glTF to W, so Blender world coordinates are W. Steps:
1. Import POSED.glb, apply every mesh object's transform to its mesh and join them, so one mesh holds
   the whole object in W.
2. Give every polygon its part: by index when the polygon centers equal the centroids in order, else
   by the nearest centroid (mathutils.kdtree). A polygon farther than 1e-4 of the diagonal from every
   centroid goes to the base. Custom normals are cleared, or the caps would get copies of their
   neighbours', and the faces are shaded smooth.
3. Weld coincident vertices (texture seams split them; UVs are per corner, so they stay) and mark the
   edges between faces of different parts, also as sharp.
4. Per part: a copy of the mesh without the other parts' faces. Every hole bounded by marked edges is
   filled (fill(): edgeloop_fill, then holes_fill, then a fan from the loop's centre), turned to face the
   same way as its neighbours and fanned from its centre (bmesh.ops.poke). Caps
   take material slot 0, flat shading and the mean UV of the corners around them, so they show one
   colour of the texture. The object and its mesh take the part's name.
5. Export the part objects with the exporter call of export_glb.py (export_extras=True), then re-import
   the GLB and check that its mesh objects carry the part names.
Writes REPORT.json (the face mapping, faces, cap faces, holes and open seam edges per part, the names
found again, warnings) before it fails on a name mismatch, so the report is there either way.
Only bpy, bmesh, mathutils and numpy: the SIF's Blender has no trimesh.
"""
# Checked on ICE with Blender 5.2.2 (2026-10-20) on the chair's MV-SAM3D mesh; the unit tests use the fake Blender.

import json
import sys
import time

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, kdtree

posed, faces_npz, out, report = sys.argv[sys.argv.index("--") + 1:][:4]
t0 = time.time()
with np.load(faces_npz) as data:
    part, names, centroids = data["part"].astype(int), [str(n) for n in data["names"]], data["centroids"].astype(float)
warnings = []

# 1. Import, bake the transforms, join.
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=posed)
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
if not meshes:
    raise RuntimeError("the posed GLB has no meshes")
for o in meshes:
    M = o.matrix_world.copy()
    o.parent = None
    if o.data.users > 1:
        o.data = o.data.copy()
    o.data.transform(M)
    o.matrix_world = Matrix.Identity(4)
for o in [o for o in bpy.context.scene.objects if o.type != "MESH"]:
    bpy.data.objects.remove(o, do_unlink=True)
src = meshes[0]
if len(meshes) > 1:
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = src
    with bpy.context.temp_override(object=src, active_object=src, selected_objects=meshes,
                                   selected_editable_objects=meshes):
        bpy.ops.object.join()
src.name = src.data.name = "__gator_source"  # so no part name is taken
me = src.data

# 2. The part of every polygon.
n = len(me.polygons)
centers = np.empty(n * 3)
me.polygons.foreach_get("center", centers)
centers = centers.reshape(-1, 3)
diag = float(np.linalg.norm(np.ptp(centroids, 0))) or 1.0
tol = 1e-4 * diag
unmatched = 0
if n == len(centroids) and np.abs(centers - centroids).max() <= tol:
    ids, mapping = part.copy(), "order"
else:
    tree = kdtree.KDTree(len(centroids))
    for i, c in enumerate(centroids.tolist()):
        tree.insert(c, i)
    tree.balance()
    ids, mapping = np.zeros(n, int), "nearest"
    for k, c in enumerate(centers.tolist()):
        _, i, d = tree.find(c)
        if i is None or d > tol:
            unmatched += 1
        else:
            ids[k] = part[i]
    warnings.append(f"the polygons of the import are not the faces of faces.npz in order ({n} and {len(centroids)}); "
                    f"matched by the nearest centroid, {unmatched} to the base")
attr = me.attributes.new("gator_part", "INT", "FACE")
attr.data.foreach_set("value", ids.astype(np.int32))
cleared = False
if getattr(me, "has_custom_normals", False):
    try:
        with bpy.context.temp_override(object=src, active_object=src):
            bpy.ops.mesh.customdata_custom_splitnormals_clear()
        cleared = True
    except Exception as e:  # the caps keep copied normals; they still render
        warnings.append(f"custom normals not cleared: {e}")
me.shade_smooth()

# 3. Weld and mark the cut edges.
bm = bmesh.new()
bm.from_mesh(me)
verts_before = len(bm.verts)
bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=1e-6 * diag)
welded = verts_before - len(bm.verts)
pl = bm.faces.layers.int.get("gator_part")
sl = bm.edges.layers.int.new("gator_seam")
for e in bm.edges:
    if len({f[pl] for f in e.link_faces}) > 1:
        e[sl] = 1
        e.smooth = False
bm.to_mesh(me)
bm.free()


def orient(bm, f, caps):
    """Reverse f when it runs along a shared edge in the same direction as its non-cap neighbour."""
    for loop in f.loops:
        for g in loop.edge.link_faces:
            if g is not f and g not in caps:
                other = next(x for x in g.loops if x.edge == loop.edge)
                if other.vert == loop.vert:
                    bmesh.ops.reverse_faces(bm, faces=[f])
                return


def loops_of(edges):
    """The closed chains of vertices that the given boundary edges form; a chain that does not close is left out."""
    nbr = {}
    for e in edges:
        a, b = e.verts
        nbr.setdefault(a, []).append(b)
        nbr.setdefault(b, []).append(a)
    seen, loops = set(), []
    for start in nbr:
        if start in seen or len(nbr[start]) != 2:
            continue
        chain, prev, v = [start], None, start
        while True:
            seen.add(v)
            nxt = [w for w in nbr[v] if w is not prev]
            if not nxt or len(nbr[v]) != 2:
                chain = None
                break
            prev, v = v, nxt[0]
            if v is start:
                break
            chain.append(v)
        if chain and len(chain) >= 3:
            loops.append(chain)
    return loops


def fill(bm, sl):
    """Close every hole bounded by cut edges. Blender 5.2's holes_fill and edgenet_fill leave some simple loops
    open that edgeloop_fill closes (seen on a real MV-SAM3D chair), so edgeloop_fill comes first, then
    holes_fill, then a fan of triangles from a new vertex at the loop's centre. Returns (faces, methods)."""
    faces, methods = [], []
    for name, op in (("edgeloop_fill", lambda es: bmesh.ops.edgeloop_fill(bm, edges=es)["faces"]),
                     ("holes_fill", lambda es: bmesh.ops.holes_fill(bm, edges=es, sides=0)["faces"])):
        es = [e for e in bm.edges if e.is_boundary and e[sl]]
        if not es:
            break
        try:
            new = op(es)
        except Exception:  # an edge set that the operator cannot take; the next one may
            new = []
        if new:
            faces += new
            methods.append(name)
    for chain in loops_of([e for e in bm.edges if e.is_boundary and e[sl]]):
        c = bm.verts.new(np.mean([v.co for v in chain], 0).tolist())
        for a, b in zip(chain, chain[1:] + chain[:1]):
            try:
                faces.append(bm.faces.new((a, b, c)))
            except ValueError:  # the face exists already
                pass
        methods.append("fan")
    return faces, methods


def mean_uv(f, uv, caps):
    acc = [tuple(lp[uv].uv) for v in f.verts for lp in v.link_loops if lp.face is not f and lp.face not in caps]
    if acc:
        m = np.mean(acc, 0).tolist()
        for lp in f.loops:
            lp[uv].uv = m


# 4. One object per part.
if not me.materials:
    warnings.append("the mesh has no material; the caps have none either")
results, objects = [], []
for k, name in enumerate(names):
    pm = me.copy()
    pm.name = name
    bm = bmesh.new()
    bm.from_mesh(pm)
    pl, sl = bm.faces.layers.int.get("gator_part"), bm.edges.layers.int.get("gator_seam")
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if f[pl] != k], context="FACES")
    rec = {"name": name, "faces": len(bm.faces), "cap_faces": 0, "holes": 0, "open_seam_edges": 0, "fill": []}
    if not bm.faces:
        warnings.append(f"part {name} has no faces; left out")
        bm.free()
        bpy.data.meshes.remove(pm)
        results.append(rec)
        continue
    caps, rec["fill"] = fill(bm, sl)
    capset = set(caps)
    uv = bm.loops.layers.uv.active
    for f in caps:
        orient(bm, f, capset)
        f.material_index, f.smooth = 0, False
        if uv is not None:
            mean_uv(f, uv, capset)
    ngons = [f for f in caps if len(f.verts) > 3]
    if ngons:  # a fan from each cap's centre: BEAUTY triangulation of a bent cut loop left zero-area slivers
        bmesh.ops.poke(bm, faces=ngons, center_mode="MEAN")
    rec.update(cap_faces=len(bm.faces) - rec["faces"], holes=len(caps),
               open_seam_edges=sum(1 for e in bm.edges if e.is_boundary and e[sl]))
    bm.faces.layers.int.remove(pl)
    bm.edges.layers.int.remove(sl)
    bm.to_mesh(pm)
    bm.free()
    if pm.validate(clean_customdata=False):  # True when Blender had to correct the mesh data
        warnings.append(f"part {name}: Blender corrected invalid mesh data")
    ob = bpy.data.objects.new(name, pm)
    bpy.context.scene.collection.objects.link(ob)
    if ob.name != name:
        warnings.append(f"Blender named part {name} {ob.name}")
    objects.append(ob)
    results.append(rec)
bpy.data.objects.remove(src, do_unlink=True)
bpy.data.meshes.remove(me)

# 5. Export as export_glb.py does, re-import and check the names.
bpy.ops.export_scene.gltf(filepath=out, export_format="GLB", export_yup=True, export_apply=True, use_visible=True,
                          export_materials="EXPORT", export_image_format="AUTO", export_cameras=False,
                          export_lights=False, export_extras=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=out)
again = sorted(o.name for o in bpy.context.scene.objects if o.type == "MESH")
expected = sorted(r["name"] for r in results if r["faces"])
with open(report, "w") as f:
    json.dump({"faces_in": n, "faces_npz": len(centroids), "mapping": mapping, "unmatched": unmatched,
               "welded_vertices": welded, "custom_normals_cleared": cleared, "parts": results,
               "names_again": again, "warnings": warnings, "seconds": round(time.time() - t0, 1)}, f, indent=2)
if again != expected:
    raise RuntimeError(f"the re-imported GLB has the mesh objects {again}, not the parts {expected}")
