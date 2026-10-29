"""Render an articulated candidate at each joint state (build_plan.md note 11).

blender -b --factory-startup --python-exit-code 1 --python review_articulation.py -- ARGS.json

ARGS.json comes from tools.review_articulation: root, out, clay, views (from the manifest), candidate (the
exported GLB), links (its mesh node names), orbit (the azimuths of the orbit views, in degrees) and
states, each with its name, t and motions: every link's 4x4 motion in W at that state, which the host
computed from the joints (harpor.parts.joints), so this script does no joint math. Only the candidate is
imported. The stock glTF importer maps glTF to W, so Blender world coordinates are W. Imported
animation is cleared, or its action would set the poses at render time (export_glb exports
animations). Parents are cleared, keeping each object's world matrix, and each link is placed at its
motion times its rest matrix. Input views are cropped to the object over all states, as in review.py.
Writes one PNG per view and state, sheet.png (one row per view: each input view with the photo, then
the states in order; then the orbit views with the states) and render.json (the crop windows and the
file of each render).
"""
# Untested: needs Blender 5.2, which the test environment does not have.

import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_common import (import_glb, input_camera, on_white, orbit_views, place_camera, points, render,  # noqa: E402
                           setup, write_sheet)

A = json.loads(Path(sys.argv[sys.argv.index("--") + 1]).read_text())
ROOT, OUT = Path(A["root"]), Path(A["out"])

scene, cam = setup(A["clay"])
for o in list(bpy.data.objects):  # the sun and the camera, so that no part name is taken
    o.name = "__gator_" + o.name
objs = import_glb(A["candidate"])
for o in objs:
    o.animation_data_clear()
    if o.type == "MESH" and o.data.shape_keys:
        o.data.shape_keys.animation_data_clear()
bpy.context.view_layer.update()
world = {o: o.matrix_world.copy() for o in objs}
for o in objs:
    o.parent = None
for o, M in world.items():
    o.matrix_world = M
by_name = {o.name: o for o in objs if o.type == "MESH"}
missing = sorted(set(A["links"]) - set(by_name))
if missing:
    raise RuntimeError(f"the candidate has no mesh objects named {missing}; found {sorted(by_name)}")
rest = {n: by_name[n].matrix_world.copy() for n in A["links"]}


def pose(state):
    for n, M in state["motions"].items():
        by_name[n].matrix_world = Matrix(M) @ rest[n]
    bpy.context.view_layer.update()


pts = []
for s in A["states"]:
    pose(s)
    pts.append(points(objs))
allp = np.concatenate(pts)
center, diag = (allp.min(0) + allp.max(0)) / 2, float(np.linalg.norm(allp.max(0) - allp.min(0)))
views = [{**v, "name": f"view{v['index']:02d}"} for v in A["views"]] + orbit_views(center, diag, azimuths=A["orbit"])
assets = {"candidate": objs}
rows, records = [], []
for v in views:
    rec, photo = {"name": v["name"]}, None
    K, width, height = v["K"], v["width"], v["height"]
    if v.get("image"):
        K, width, height, photo, _, crop = input_camera(v, ROOT, allp)
        rec.update(crop)
    place_camera(scene, cam, K, v["w2c"], width, height, (1e-3 * diag, 1e3 * diag))
    row = []
    for s in A["states"]:
        pose(s)
        name = f"{v['name']}_{s['name']}.png"
        row.append(on_white(render(scene, assets, "candidate", OUT / name)))
        rec[s["name"]] = name
    rows.append(([photo] + row if photo is not None else row, photo is not None))
    records.append(rec)
write_sheet(rows, OUT / "sheet.png")
(OUT / "render.json").write_text(json.dumps({"clay": A["clay"], "states": [s["name"] for s in A["states"]],
                                             "views": records, "center": center.tolist(), "diagonal": diag},
                                            indent=2))
