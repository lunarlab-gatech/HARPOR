"""Scripted stand-in for the Codex CLI agent, so tests drive every controller path (build_plan.md note 5).

Run `python -m harpor.agent.fake_agent SCRIPT.json` with the work folder as the current directory and
GATOR_SOCKET set to the tool server's socket. The script is a list of actions:

  {"do": "edit", "kind": "good", "size": 1.0, "seconds": 0}   save candidate.blend, a JSON scene that
                                                             the fake Blender exports in `seconds`; the
                                                             scene's other keys (objects, joints, and
                                                             pause, the seconds between the GLB and its
                                                             report) stay
  {"do": "glb", "path": "exports/009.glb", "kind": "nan"}     write a GLB directly
  {"do": "tool", "args": ["finish", "--receipt", "{receipt}", ...]}   call a helper through the socket,
                                                             as bin/ does; {receipt} is the newest review
                                                             receipt, {areceipt} the newest of
                                                             review_articulation
  {"do": "tool_nowait", "args": ["export_glb"]}             start a helper call and go on after 1 s
  {"do": "write", "path": "run_budget.json", "text": "{}"}
  {"do": "sleep", "seconds": 30}, {"do": "ignore_sigint"}, {"do": "crash"}
  {"do": "read_prompt"}                                      copy stdin to prompt_seen.txt

A session with rounds (controller.run_rounds) runs one agent per round: the script may then be
{"rounds": [[...], [...]]}, and each agent runs the list of its round, run_budget.json's round (none
past the last).

GLB kinds: good (a box of edge `size` with a PBR material), nan, no_material, garbage and parts (the
same cube as two named nodes in W: body, the lower three quarters, and lid on top; with joints in the
nodes' extras when the scene has them). The fake .blend of parts also lists its objects and joints.
An object that the fake add_part.py added is a thin slab on one side of the body (SIDES), a node of
its own in every export.
`python -m harpor.agent.fake_agent --blender ...` stands in for Blender, following the file contracts
of blender/export_glb.py, review.py, set_joint.py, add_part.py, move_faces.py and review_articulation.py without
bpy, those of
blender/split_parts.py for harpor.parts.stage split, and the frames_dir contract of blender/turntable.py
for harpor.webapp.turntable render and, with "soft", harpor.webapp.softvideo render.
`python -m harpor.agent.fake_agent --sim` stands in for the python of envs/mujoco: harpor.agent.validate
runs GATOR_MUJOCO, split like a shell as GATOR_BLENDER is, with `-m harpor.physics.sim_check ...` after
it, and fake_sim writes the report of that contract without MuJoCo, with the outcomes of GATOR_FAKE_SIM.
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

from harpor.agent.bin import gator_tool

PARTS = ("body", "lid")
SIDES = ((1, 0), (-1, 0), (0, 1), (0, -1))  # the body's sides in x and z, where the added parts sit, in this order
SLAB = 0.05  # the thickness of an added part, of the cube's edge


def added(objects):
    """The objects of a fake scene that add_part made: those that are not the two parts of the split."""
    return [o for o in objects or [] if o not in PARTS]


def slab(k, size=1.0):
    """The box [min, max] in W of the k-th added part: a slab against the body's side SIDES[k], as high as the
    body. Its top edge on the body is the line that a hinge of it would turn about."""
    h, (sx, sz) = size / 2, SIDES[k]
    lo, hi = [-h, -h / 2, -h], [h, h, h]
    axis = 0 if sx else 2
    lo[axis], hi[axis] = (h, h + SLAB * size) if sx + sz > 0 else (-h - SLAB * size, -h)
    return [lo, hi]


def write_parts(path, size=1.0, joints=None, objects=None):
    """The parts GLB: a cube of edge size about the origin of W (y down), split at a quarter of its height
    into body (below) and lid (on top), as named root nodes with a PBR material; joints go to the extras.
    objects: the scene's objects; each one that add_part added is a slab (slab) with its own node."""
    import numpy as np
    import trimesh

    from harpor import frames
    from harpor.parts.joints import write_joints
    from harpor.physics.align import read_glb, write_glb as write_doc

    h = size / 2
    scene = trimesh.Scene()
    boxes = [("body", [[-h, -h / 2, -h], [h, h, h]], [200, 120, 60, 255]),
             ("lid", [[-h, -h, -h], [h, -h / 2, h]], [60, 120, 200, 255])]
    boxes += [(name, slab(k, size), [200, 120, 60, 255]) for k, name in enumerate(added(objects))]
    for name, bounds, color in boxes:
        box = trimesh.creation.box(bounds=np.array(bounds))
        mesh = trimesh.Trimesh(frames.w_to_gltf(box.vertices), box.faces, process=False)
        mesh.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=color, metallicFactor=0.0, roughnessFactor=0.7))
        scene.add_geometry(mesh, node_name=name, geom_name=name)
    scene.export(path, file_type="glb")
    if joints:
        doc, binary = read_glb(path)
        write_doc(path, write_joints(doc, joints), binary)


def write_glb(path, kind="good", size=1.0, joints=None, objects=None):
    if kind == "garbage":
        Path(path).write_bytes(b"this is not a GLB file")
        return
    if kind == "parts":
        write_parts(path, size, joints, objects)
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


def add_part(blend, out, name, source, spec):
    """blender/add_part.py on the fake .blend of parts. The source is a box of 12 triangles, so a faces file holds
    indices 0 to 11 (checked as the real script checks them), and planes select by the box's 8 corners: none on
    the planes' side is an empty selection, all 8 the whole source. The new object joins the scene's objects and
    is a slab in the exports (slab). OUT.json is created first, and an error leaves the .blend as it was."""
    with open(out, "x") as f:
        scene, spec = json.loads(Path(blend).read_text()), json.loads(spec)
        objects = scene.setdefault("objects", [])
        rec = {"ok": False, "error": None, "name": name, "source": source, "warnings": []}
        try:
            if source not in objects:
                raise ValueError(f"no mesh object is named {source!r}")
            if name in objects:
                raise ValueError(f"an object named {name!r} exists already")
            if len(added(objects)) >= len(SIDES):
                raise ValueError("the fake Blender has no side left for another part")
            n = _selected(scene, spec, source)
            if n == 12:
                raise ValueError(f"the selection holds every face of {source}; a part is a piece of it")
            lo, hi = slab(len(added(objects)), scene["size"])
            objects.append(name)
            Path(blend).write_text(json.dumps(scene))
            counts = {"cap_faces": 2, "holes": 1, "fill": ["edgeloop_fill"], "open_cut_edges": 0}
            rec.update(ok=True, part={"faces": n, **counts}, rest={"faces": 12 - n, **counts}, pieces=1,
                       area=n / 12, share_of_source=n / 12, share_of_object=n / 24, box={"min": lo, "max": hi})
        except ValueError as e:
            rec["error"] = str(e)
        rec.update(links=objects, seconds=0.0)
        json.dump(rec, f)


def _selected(scene, spec, source):
    """How many of the 12 triangles of a fake part object a selection takes, as add_part and move_faces count
    them: a faces file holds indices 0 to 11 (checked as the real scripts check them), and planes select by the
    box's 8 corners. ValueError for a selection of no face."""
    import numpy as np

    from harpor.agent.blender import part_select as S

    if "faces" in spec:
        return len(S.face_indices(np.load(spec["faces"], allow_pickle=False), 12))
    h = scene["size"] / 2
    corners = np.array([[x, y, z] for x in (-h, h) for y in (-h / 2, h) for z in (-h, h)])
    n = int(S.inside(corners, S.unit_planes(spec["planes"])).sum()) * 12 // 8
    if not n:
        raise ValueError(f"no face of {source} has a vertex on the normal's side of every plane")
    return n


def move_faces(blend, out, source, target, spec):
    """blender/move_faces.py on the fake .blend of parts: both objects must be in the scene, and the selection is
    counted as add_part's (_selected). The move is recorded in the scene's "moved" list, which changes the .blend
    and so the next export's name; the boxes of the exports stay as they are. OUT.json is created first, and an
    error leaves the .blend as it was."""
    with open(out, "x") as f:
        scene, spec = json.loads(Path(blend).read_text()), json.loads(spec)
        objects = scene.setdefault("objects", [])
        rec = {"ok": False, "error": None, "source": source, "target": target, "warnings": []}
        try:
            if source == target:
                raise ValueError(f"{source} cannot give faces to itself")
            for name, flag in ((source, "--from"), (target, "--to")):
                if name not in objects:
                    raise ValueError(f"no mesh object ({flag}) is named {name!r}")
            n = _selected(scene, spec, source)
            if n == 12:
                raise ValueError(f"the selection holds all of {source}'s surface; that would remove the link")
            scene.setdefault("moved", []).append({"from": source, "to": target, "faces": n})
            Path(blend).write_text(json.dumps(scene))
            counts = {"cap_faces": 2, "holes": 1, "fill": ["edgeloop_fill"], "open_cut_edges": 0, "pieces": 1}
            rec.update(ok=True, moved=n, area=n / 12, share_of_source=n / 12,
                       after={source: {"faces": 12 - n, **counts}, target: {"faces": 12 + n, **counts}})
        except ValueError as e:
            rec["error"] = str(e)
        rec.update(links=objects, seconds=0.0)
        json.dump(rec, f)


def turntable_frames(args, side=96):
    """blender/turntable.py without bpy: args["frames"] PNGs of side px (0001.png, ...) on the
    0.82-gray background, each the convex hull of the GLB's vertices in dark gray, seen orthographically as the
    real camera sees it (along +z of W after the asset's matrix, args["elevation"] degrees from above), as the
    asset turns once about W's y. With args["soft"] the asset stays, framed on the soft part, whose vertices of each
    frame (matrix @ (rest + disp) of the motion) are drawn on it in white."""
    import numpy as np
    from PIL import Image, ImageDraw
    from scipy.spatial import ConvexHull

    from harpor.physics.align import load_w

    assert len(args["assets"]) == 1 and args.get("frames_dir"), "turntable.py takes exactly one asset and frames_dir"
    a, out, n, soft = args["assets"][0], Path(args["frames_dir"]), args["frames"], args.get("soft")
    M = np.eye(4) if a.get("matrix") is None else np.asarray(a["matrix"], float)
    v = load_w(a["path"])[0] @ M[:3, :3].T + M[:3, 3]  # glTF -> W is Blender's import
    part = None
    if soft:
        S = np.asarray(soft["matrix"], float)
        with np.load(soft["motion"]) as z:
            part = (z["rest"][None] + z["disp"]) @ S[:3, :3].T + S[:3, 3]
        assert len(part) == n, f"the motion has {len(part)} frames, not {n}"
    box = v if part is None else part[0]
    c, r = (box.min(0) + box.max(0)) / 2, np.linalg.norm(box.max(0) - box.min(0)) / 2
    e = math.radians(args.get("elevation", 0))
    right, down = np.array([1.0, 0.0, 0.0]), np.array([0.0, math.cos(e), -math.sin(e)])  # the camera's x and y in W
    out.mkdir(parents=True, exist_ok=True)

    def panel(p):  # the panel is 2.6 r wide
        return side / 2 + np.stack([(p - c) @ right, (p - c) @ down], 1) * side / (2.6 * r)

    for k in range(n):
        t = 0.0 if soft else -2 * math.pi * k / n  # as turntable.py keys the pivot's turn
        R = np.array([[math.cos(t), 0.0, math.sin(t)], [0.0, 1.0, 0.0], [-math.sin(t), 0.0, math.cos(t)]])
        uv = panel((v - c) @ R.T + c)
        img = Image.new("RGB", (side, side), (209, 209, 209))
        draw = ImageDraw.Draw(img)
        draw.polygon([tuple(q) for q in uv[ConvexHull(uv).vertices]], fill=(64, 64, 64))
        if part is not None:
            draw.point([tuple(q) for q in panel(part[k])], fill=(255, 255, 255))
        img.save(out / f"{k + 1:04d}.png")


def fake_blender(argv):
    i = argv.index("--python")
    # Blender applies --python-exit-code only to scripts after it; a later flag lets failures exit 0.
    assert "--python-exit-code" in argv[:i], "--python-exit-code must come before --python"
    script, rest = Path(argv[i + 1]).name, argv[argv.index("--") + 1:]
    blend = next((x for x in argv[:i] if x.endswith(".blend")), None)
    if script == "make_candidate.py":
        import trimesh

        from harpor.parts.joints import mesh_nodes
        from harpor.physics.align import read_glb

        size = float(trimesh.load(rest[0], file_type="glb", force="scene", process=False).extents[0])
        scene = {"kind": "good", "size": size, "seconds": 0.0}
        names = mesh_nodes(read_glb(rest[0])[0])
        if len(names) > 1:
            scene.update(kind="parts", objects=names, joints={})
        Path(rest[1]).write_text(json.dumps(scene))
    elif script == "export_glb.py":
        scene = json.loads(Path(blend).read_text())
        time.sleep(scene["seconds"])
        write_glb(rest[0], scene["kind"], scene["size"], scene.get("joints"), scene.get("objects"))
        time.sleep(scene.get("pause", 0.0))  # the real script re-imports the GLB before it writes the report
        rep = {"meshes": 1, "materials": int(scene["kind"] != "no_material"), "images": 0, "triangles": 12,
               "warnings": []}
        if scene["kind"] == "parts":
            nodes = [*PARTS, *added(scene.get("objects"))]
            rep.update(meshes=len(nodes), triangles=12 * len(nodes), nodes=nodes, joints=scene.get("joints") or {})
        Path(rest[1]).write_text(json.dumps(rep))
    elif script == "split_parts.py":
        split_parts(*rest[:4])
    elif script == "set_joint.py":
        set_joint(blend, *rest[:3])
    elif script == "add_part.py":
        add_part(blend, *rest[:4])
    elif script == "move_faces.py":
        move_faces(blend, *rest[:4])
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


def fake_sim(argv):
    """harpor.physics.sim_check without MuJoCo. argv is what follows GATOR_MUJOCO: `-m harpor.physics.sim_check
    --glb GLB --out DIR --asset-sha SHA [--intervals FILE]`. Each hinge and slide of the intervals (each one
    without them) and each soft part gets a run of 3 states. GATOR_FAKE_SIM is JSON: {link: code} gives the
    link's run a finding of that code ("pass": none), and {"error": text} gives a report of only that error."""
    import argparse

    from harpor.parts import joints as J

    assert argv[:2] == ["-m", "harpor.physics.sim_check"], argv
    ap = argparse.ArgumentParser()
    for flag in ("--glb", "--out", "--asset-sha"):
        ap.add_argument(flag, required=True)
    ap.add_argument("--intervals")
    a = ap.parse_args(argv[2:])
    outcomes, out = json.loads(os.environ.get("GATOR_FAKE_SIM") or "{}"), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"asset_sha256": a.asset_sha}
    if "error" in outcomes:
        report["error"] = outcomes["error"]
    else:
        joints, _ = J.load(a.glb)
        intervals = json.loads(Path(a.intervals).read_text()) if a.intervals else None
        states = [{"t": 1.25 * k, "ctrl": 0.0, "q": 0.0} for k in range(3)]
        report.update(
            joints=[{"link": c, "type": j["type"], "interval": intervals[c] if intervals else [j["lower"], j["upper"]],
                     "states": states} for c, j in joints.items()
                    if j["type"] in ("revolute", "prismatic") and (intervals is None or c in intervals)],
            soft=[{"link": c, "dmax_m": 0.004, "visible_dmax_m": 0.004, "residual_m": 0.0, "cap_limited": False,
                   "states": states} for c, j in joints.items() if j["type"] == "soft"],
            findings=[{"code": code, "link": link, "measured": None, "expected": None, "why": "fake MuJoCo"}
                      for link, code in outcomes.items() if code != "pass" and link in joints])
    (out / "report.json").write_text(json.dumps(report))


def round_actions(script):
    """The actions of this agent: script itself, or of {"rounds": [...]} the list of run_budget.json's round."""
    if isinstance(script, list):
        return script
    n = json.loads(Path("run_budget.json").read_text())["round"]
    return script["rounds"][n - 1] if n <= len(script["rounds"]) else []


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
    elif sys.argv[1] == "--sim":
        fake_sim(sys.argv[2:])
    else:
        run(round_actions(json.loads(Path(sys.argv[1]).read_text())))
