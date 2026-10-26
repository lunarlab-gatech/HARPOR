"""Scripted stand-in for the Codex CLI agent, so tests drive every controller path (build_plan.md note 5).

Run `python -m gator.agent.fake_agent SCRIPT.json` with the work folder as the current directory and
GATOR_SOCKET set to the tool server's socket. The script is a list of actions:

  {"do": "edit", "kind": "good", "size": 1.0, "seconds": 0}   save candidate.blend, a JSON scene that
                                                             the fake Blender exports in `seconds`; the
                                                             scene's other keys (objects, joints) stay
  {"do": "glb", "path": "exports/009.glb", "kind": "nan"}     write a GLB directly
  {"do": "tool", "args": ["finish", "--receipt", "{receipt}", ...]}   call a helper through the socket,
                                                             as bin/ does; {receipt} is the newest review
                                                             receipt, {areceipt} the newest of
                                                             review_articulation
  {"do": "tool_nowait", "args": ["export_glb"]}             start a helper call and go on after 1 s
  {"do": "write", "path": "run_budget.json", "text": "{}"}
  {"do": "sleep", "seconds": 30}, {"do": "ignore_sigint"}, {"do": "crash"}
  {"do": "read_prompt"}                                      copy stdin to prompt_seen.txt

GLB kinds: good (a box of edge `size` with a PBR material), nan, no_material, garbage and parts (the
same cube as two named nodes in W: body, the lower three quarters, and lid on top; with joints in the
lid's extras when the scene has them). The fake .blend of parts also lists its objects and joints.
`python -m gator.agent.fake_agent --blender ...` stands in for Blender, following the file contracts
of blender/export_glb.py, review.py, set_joint.py and review_articulation.py without bpy, those of
blender/split_parts.py for gator.parts.stage split, and the frames_dir contract of blender/turntable.py
for gator.webapp.turntable render.
"""

import contextlib
import json
import math
import os
import signal
import sys
import threading
import time
from pathlib import Path

from gator.agent.bin import gator_tool

PARTS = ("body", "lid")


def write_parts(path, size=1.0, joints=None):
    """The parts GLB: a cube of edge size about the origin of W (y down), split at a quarter of its height
    into body (below) and lid (on top), as named root nodes with a PBR material; joints go to the extras."""
    import numpy as np
    import trimesh

    from gator import frames
    from gator.parts.joints import write_joints
    from gator.physics.align import read_glb, write_glb as write_doc

    h = size / 2
    scene = trimesh.Scene()
    for name, (y0, y1), color in (("body", (-h / 2, h), [200, 120, 60, 255]),
                                  ("lid", (-h, -h / 2), [60, 120, 200, 255])):
        box = trimesh.creation.box(bounds=np.array([[-h, y0, -h], [h, y1, h]]))
        mesh = trimesh.Trimesh(frames.w_to_gltf(box.vertices), box.faces, process=False)
        mesh.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=color, metallicFactor=0.0, roughnessFactor=0.7))
        scene.add_geometry(mesh, node_name=name, geom_name=name)
    scene.export(path, file_type="glb")
    if joints:
        doc, binary = read_glb(path)
        write_doc(path, write_joints(doc, joints), binary)


def write_glb(path, kind="good", size=1.0, joints=None):
    if kind == "garbage":
        Path(path).write_bytes(b"this is not a GLB file")
        return
    if kind == "parts":
        write_parts(path, size, joints)
        return
    import trimesh

    mesh = trimesh.creation.box(extents=[size] * 3)
    if kind != "no_material":
        mesh.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=[200, 120, 60, 255], metallicFactor=0.0, roughnessFactor=0.7))
    if kind == "nan":
        mesh.vertices[0] = float("nan")
    mesh.export(path, file_type="glb")


def split_parts(posed, faces_npz, out, report):
    """blender/split_parts.py without bpy: one trimesh submesh per part of faces.npz as a named node,
    textures kept, no caps."""
    import numpy as np
    import trimesh

    with np.load(faces_npz) as data:
        part, names = data["part"], [str(n) for n in data["names"]]
    mesh = trimesh.load(posed, force="mesh", process=False)
    scene, parts = trimesh.Scene(), []
    for k, name in enumerate(names):
        f = np.flatnonzero(part == k)
        if len(f):
            scene.add_geometry(mesh.submesh([f], append=True), node_name=name, geom_name=name)
        parts.append({"name": name, "faces": int(len(f)), "cap_faces": 0, "holes": 0, "open_seam_edges": 0})
    scene.export(out, file_type="glb")
    Path(report).write_text(json.dumps({"faces_in": len(part), "faces_npz": len(part), "mapping": "order",
                                        "unmatched": 0, "welded_vertices": 0, "custom_normals_cleared": False,
                                        "parts": parts, "names_again": sorted(p["name"] for p in parts if p["faces"]),
                                        "warnings": ["fake Blender: no caps"], "seconds": 0.0}))


def set_joint(blend, out, child, spec):
    """blender/set_joint.py on the fake .blend of parts: its objects are the links, all visible. OUT.json
    is created first, so an existing one leaves the .blend as it was."""
    with open(out, "x") as f:
        scene, spec = json.loads(Path(blend).read_text()), json.loads(spec)
        objects, joints = scene.get("objects", []), scene.setdefault("joints", {})
        rec = {"ok": child in objects, "error": None if child in objects else f"no mesh object is named {child!r}",
               "child": child, "previous": joints.get(child) if child in objects else None, "warnings": []}
        if rec["ok"]:
            if spec is None:
                joints.pop(child, None)
            else:
                joints[child] = spec
            Path(blend).write_text(json.dumps(scene))
        rec.update(joints=joints, links=objects, holders=scene.get("holders", {}), seconds=0.0)
        json.dump(rec, f)


def turntable_frames(args, side=96):
    """blender/turntable.py with frames_dir, without bpy: args["frames"] PNGs of side px (0001.png, ...) on the
    0.82-gray background, each the convex hull of the GLB's vertices in dark gray, seen orthographically as the
    real camera sees it (along +z of W after the asset's matrix, args["elevation"] degrees from above), as the
    asset turns once about W's y."""
    import numpy as np
    from PIL import Image, ImageDraw
    from scipy.spatial import ConvexHull

    from gator.physics.align import load_w

    assert len(args["assets"]) == 1 and args.get("frames_dir"), "frames_dir takes exactly one asset"
    a, out, n = args["assets"][0], Path(args["frames_dir"]), args["frames"]
    M = np.eye(4) if a.get("matrix") is None else np.asarray(a["matrix"], float)
    v = load_w(a["path"])[0] @ M[:3, :3].T + M[:3, 3]  # glTF -> W is Blender's import
    c, r = (v.min(0) + v.max(0)) / 2, np.linalg.norm(v.max(0) - v.min(0)) / 2
    e = math.radians(args.get("elevation", 0))
    right, down = np.array([1.0, 0.0, 0.0]), np.array([0.0, math.cos(e), -math.sin(e)])  # the camera's x and y in W
    out.mkdir(parents=True, exist_ok=True)
    for k in range(n):
        t = -2 * math.pi * k / n  # as turntable.py keys the pivot's turn
        R = np.array([[math.cos(t), 0.0, math.sin(t)], [0.0, 1.0, 0.0], [-math.sin(t), 0.0, math.cos(t)]])
        p = (v - c) @ R.T
        uv = side / 2 + np.stack([p @ right, p @ down], 1) * side / (2.6 * r)  # the panel is 2.6 r wide
        img = Image.new("RGB", (side, side), (209, 209, 209))
        ImageDraw.Draw(img).polygon([tuple(q) for q in uv[ConvexHull(uv).vertices]], fill=(64, 64, 64))
        img.save(out / f"{k + 1:04d}.png")


def fake_blender(argv):
    i = argv.index("--python")
    # Blender applies --python-exit-code only to scripts after it; a later flag lets failures exit 0.
    assert "--python-exit-code" in argv[:i], "--python-exit-code must come before --python"
    script, rest = Path(argv[i + 1]).name, argv[argv.index("--") + 1:]
    blend = next((x for x in argv[:i] if x.endswith(".blend")), None)
    if script == "make_candidate.py":
        import trimesh

        from gator.parts.joints import mesh_nodes
        from gator.physics.align import read_glb

        size = float(trimesh.load(rest[0], file_type="glb", force="scene", process=False).extents[0])
        scene = {"kind": "good", "size": size, "seconds": 0.0}
        names = mesh_nodes(read_glb(rest[0])[0])
        if len(names) > 1:
            scene.update(kind="parts", objects=names, joints={})
        Path(rest[1]).write_text(json.dumps(scene))
    elif script == "export_glb.py":
        scene = json.loads(Path(blend).read_text())
        time.sleep(scene["seconds"])
        write_glb(rest[0], scene["kind"], scene["size"], scene.get("joints"))
        rep = {"meshes": 1, "materials": int(scene["kind"] != "no_material"), "images": 0, "triangles": 12,
               "warnings": []}
        if scene["kind"] == "parts":
            rep.update(meshes=len(PARTS), triangles=12 * len(PARTS), nodes=list(PARTS),
                       joints=scene.get("joints") or {})
        Path(rest[1]).write_text(json.dumps(rep))
    elif script == "split_parts.py":
        split_parts(*rest[:4])
    elif script == "set_joint.py":
        set_joint(blend, *rest[:3])
    elif script == "turntable.py":
        turntable_frames(json.loads(Path(rest[0]).read_text()))
    elif script == "review_articulation.py":
        args = json.loads(Path(rest[0]).read_text())
        assert all(sorted(s["motions"]) == sorted(args["links"]) for s in args["states"])
        out, states = Path(args["out"]), [s["name"] for s in args["states"]]
        views = [f"view{v['index']:02d}" for v in args["views"]] + [f"orbit{k}" for k in range(len(args["orbit"]))]
        (out / "sheet.png").write_bytes(b"")
        (out / "render.json").write_text(json.dumps({"clay": args["clay"], "states": states, "views": [
            {"name": v, **{s: f"{v}_{s}.png" for s in states}} for v in views]}))
    else:
        args = json.loads(Path(rest[0]).read_text())
        out = Path(args["out"])
        (out / "sheet.png").write_bytes(b"")
        review = {"views": [{"name": f"view{v['index']:02d}", "iou_candidate": 1.0} for v in args["views"]]}
        if args["original"]:
            review.update(center_shift=0.0, extent_ratio=[1.0, 1.0, 1.0])
        (out / "review.json").write_text(json.dumps(review))


def run(script):
    receipt = areceipt = None
    for a in script:
        do = a["do"]
        if do == "edit":
            scene = {}
            with contextlib.suppress(OSError, ValueError):
                scene = json.loads(Path("candidate.blend").read_text())
            scene = scene if isinstance(scene, dict) else {}
            scene.update(kind=a.get("kind", "good"), size=a.get("size", 1.0), seconds=a.get("seconds", 0.0))
            if scene["kind"] == "parts":
                scene.setdefault("objects", list(PARTS))
                scene.setdefault("joints", {})
            Path("candidate.blend").write_text(json.dumps(scene))
        elif do == "glb":
            write_glb(a["path"], a.get("kind", "good"), a.get("size", 1.0))
        elif do == "tool":
            args = [s.replace("{receipt}", receipt or "").replace("{areceipt}", areceipt or "") for s in a["args"]]
            print("$", *args, flush=True)
            output, code = gator_tool.call(args)
            sys.stdout.write(output)
            if code:
                print(f"exit {code}", flush=True)
            receipts = sorted(Path("reviews").glob("r*"))
            receipt = receipts[-1].name if receipts else None
            receipts = sorted(Path("reviews_articulation").glob("a*"))
            areceipt = receipts[-1].name if receipts else None
        elif do == "tool_nowait":
            threading.Thread(target=gator_tool.call, args=(a["args"],), daemon=True).start()
            time.sleep(1.0)
        elif do == "write":
            Path(a["path"]).parent.mkdir(parents=True, exist_ok=True)
            Path(a["path"]).write_text(a["text"])
        elif do == "sleep":
            time.sleep(a["seconds"])
        elif do == "ignore_sigint":
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        elif do == "crash":
            os._exit(3)
        elif do == "read_prompt":
            Path("prompt_seen.txt").write_bytes(sys.stdin.buffer.read())
        sys.stdout.flush()


if __name__ == "__main__":
    if sys.argv[1] == "--blender":
        fake_blender(sys.argv[2:])
    else:
        run(json.loads(Path(sys.argv[1]).read_text()))
