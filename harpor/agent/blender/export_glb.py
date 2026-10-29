"""Export the open .blend to GLB once and re-import it to check it (build_plan.md note 5).

blender -b --factory-startup candidate.blend --python-exit-code 1 --python export_glb.py -- OUT.glb REPORT.json

The stock exporter with export_yup=True applies the axis conversion exactly once. No manual rotation.
Only visible objects are exported, so hidden reference copies and helper objects stay out. Custom
properties go to the extras (export_extras=True), so a part's joint, which set_joint stores as the
object's string property gator_joint (build_plan.md note 11), travels inside the GLB. The report adds
the GLB's mesh node names (nodes) and its joints ({node: joint}, read from the JSON chunk).
"""
# Untested: needs Blender 5.2, which the test environment does not have.

import json
import struct
import sys

import bpy
import numpy as np

out, report = sys.argv[sys.argv.index("--") + 1:][:2]
KEY = "gator_joint"


def mesh_box():
    """World box of the evaluated meshes, so modifiers count as export_apply=True exports them."""
    dg, pts = bpy.context.evaluated_depsgraph_get(), []
    for o in bpy.context.scene.objects:
        if o.type == "MESH" and o.visible_get():
            e = o.evaluated_get(dg)
            m = e.to_mesh()
            co = np.empty(len(m.vertices) * 3)
            m.vertices.foreach_get("co", co)
            M = np.array(e.matrix_world)
            pts.append(co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3])
            e.to_mesh_clear()
    p = np.concatenate(pts)
    return p.min(0), p.max(0)


def glb_nodes(path):
    """Mesh node names and {node name: joint} of a GLB's JSON chunk."""
    with open(path, "rb") as f:
        data = f.read()
    doc = json.loads(data[20:20 + struct.unpack_from("<I", data, 12)[0]])
    joints = {}
    for n in doc.get("nodes", []):
        j = n["extras"].get(KEY) if isinstance(n.get("extras"), dict) else None
        if j is not None:
            try:
                joints[n.get("name")] = json.loads(j) if isinstance(j, str) else j
            except ValueError:
                joints[n.get("name")] = j
    return [n.get("name") for n in doc.get("nodes", []) if "mesh" in n], joints


lo0, hi0 = mesh_box()
in_scene = [o.name for o in bpy.context.scene.objects if o.visible_get() and KEY in o.keys()]
bpy.ops.export_scene.gltf(filepath=out, export_format="GLB", export_yup=True, export_apply=True, use_visible=True,
                          export_materials="EXPORT", export_image_format="AUTO", export_cameras=False,
                          export_lights=False, export_extras=True)
nodes, joints = glb_nodes(out)
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=out)
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
if not meshes:
    raise RuntimeError("the re-imported GLB has no meshes")
lo1, hi1 = mesh_box()
warnings = []
if max(np.abs(lo1 - lo0).max(), np.abs(hi1 - hi0).max()) > 0.01 * np.linalg.norm(hi0 - lo0):
    warnings.append("re-imported bounding box differs from the .blend; a 90 degree X turn means a wrong axis conversion")
if any(not o.material_slots or any(s.material is None for s in o.material_slots) for o in meshes):
    warnings.append("a mesh has no material")
if sorted(joints) != sorted(in_scene):
    warnings.append(f"joints are on the scene objects {sorted(in_scene)} but on the GLB nodes {sorted(joints)}; "
                    "an object under a hidden parent may be left out")
triangles = 0
for o in meshes:
    o.data.calc_loop_triangles()
    triangles += len(o.data.loop_triangles)
with open(report, "w") as f:
    json.dump({"meshes": len(meshes), "materials": len(bpy.data.materials), "images": len(bpy.data.images),
               "triangles": triangles, "bbox_blend": [lo0.tolist(), hi0.tolist()],
               "bbox_glb": [lo1.tolist(), hi1.tolist()], "nodes": nodes, "joints": joints, "warnings": warnings},
              f, indent=2)
