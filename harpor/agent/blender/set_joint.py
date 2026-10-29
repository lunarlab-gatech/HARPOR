"""Set or remove the joint of one part in the agent's .blend (build_plan.md note 11).

blender -b --factory-startup --disable-autoexec candidate.blend --python-exit-code 1 --python set_joint.py -- OUT.json CHILD SPEC

SPEC is JSON text: the joint (harpor.parts.joints) to store, null to remove it, or a string to store as
it is (tools.set_joint undoes a change with the child's previous value). The joint is the child object's
custom property gator_joint, a JSON string, which export_glb.py writes to the glTF node's extras. The
.blend is packed and saved, so its sha256 changes and export_glb exports it again.

Writes OUT.json: ok, error, the child's previous value, and what export_glb.py would export: the
joints of the visible objects, the visible mesh objects (the links) and, per link, its nearest visible
mesh ancestor (holders). OUT.json is created exclusively before anything changes, so an existing file
stops the script with the .blend as it was. tools.set_joint validates the joints on the host, because
harpor.parts.joints is not inside this container. A child that is not a visible mesh object is an error,
and the .blend is then left as it was.
"""
# Untested: needs Blender 5.2, which the test environment does not have.

import json
import sys
import time

import bpy

KEY = "gator_joint"
out, child, spec = sys.argv[sys.argv.index("--") + 1:][:3]
spec = json.loads(spec)
t0 = time.time()
record = open(out, "x")  # first, so an existing OUT.json stops the script before the .blend changes


def value(o):
    """The object's joint as a dict when its string parses, else as stored; None without one."""
    v = o.get(KEY)
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v.to_dict() if hasattr(v, "to_dict") else v


rec = {"ok": False, "error": None, "child": child, "previous": None, "warnings": []}
obj = bpy.data.objects.get(child)
if obj is None or obj.type != "MESH":
    rec["error"] = f"no mesh object is named {child!r}"
elif not obj.visible_get():
    rec["error"] = f"{child} is hidden, so export_glb leaves it out; show it first"
else:
    rec["previous"] = value(obj)
    if spec is None:
        if KEY in obj.keys():
            del obj[KEY]
    else:
        obj[KEY] = spec if isinstance(spec, str) else json.dumps(spec)
    try:
        bpy.ops.file.pack_all()
    except RuntimeError as e:  # an image without a file; the .blend still saves
        rec["warnings"].append(f"pack_all: {e}")
    bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
    rec["ok"] = True

shown = [o for o in bpy.context.scene.objects if o.visible_get()]
links = [o.name for o in shown if o.type == "MESH"]
holders = {}
for o in shown:
    p = o.parent
    while p is not None and p.name not in links:
        p = p.parent
    if o.type == "MESH" and p is not None:
        holders[o.name] = p.name
rec.update(joints={o.name: value(o) for o in shown if KEY in o.keys()}, links=links, holders=holders,
           seconds=round(time.time() - t0, 2))
with record as f:
    json.dump(rec, f, indent=2)
