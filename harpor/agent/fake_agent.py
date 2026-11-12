"""Scripted stand-in for the Codex CLI agent, so tests drive every controller path (build_plan.md note 5).

Run `python -m harpor.agent.fake_agent SCRIPT.json` with the work folder as the current directory and
GATOR_SOCKET set to the tool server's socket. The script is a list of actions:

  {"do": "edit", "kind": "good", "size": 1.0, "seconds": 0}   save candidate.blend, a JSON scene that
                                                             the fake Blender exports in `seconds`; the
                                                             scene's other keys (pause, the seconds
                                                             between the GLB and its report) stay
  {"do": "glb", "path": "exports/009.glb", "kind": "nan"}     write a GLB directly
  {"do": "tool", "args": ["finish", "--receipt", "{receipt}", ...]}   call a helper through the socket,
                                                             as bin/ does; {receipt} is the newest review
                                                             receipt
  {"do": "tool_nowait", "args": ["export_glb"]}             start a helper call and go on after 1 s
  {"do": "write", "path": "run_budget.json", "text": "{}"}
  {"do": "sleep", "seconds": 30}, {"do": "ignore_sigint"}, {"do": "crash"}
  {"do": "read_prompt"}                                      copy stdin to prompt_seen.txt

A session with rounds (controller.run_rounds) runs one agent per round: the script may then be
{"rounds": [[...], [...]]}, and each agent runs the list of its round, run_budget.json's round (none
past the last).

GLB kinds: good (a box of edge `size` with a PBR material), nan, no_material and garbage.
`python -m harpor.agent.fake_agent --blender ...` stands in for Blender, following the file contracts
of blender/make_candidate.py, export_glb.py and review.py without bpy, and the frames_dir contract of
blender/turntable.py for harpor.webapp.turntable render.
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


def write_glb(path, kind="good", size=1.0):
    if kind == "garbage":
        Path(path).write_bytes(b"this is not a GLB file")
        return
    import trimesh

    mesh = trimesh.creation.box(extents=[size] * 3)
    if kind != "no_material":
        mesh.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=[200, 120, 60, 255], metallicFactor=0.0, roughnessFactor=0.7))
    if kind == "nan":
        mesh.vertices[0] = float("nan")
    mesh.export(path, file_type="glb")


def turntable_frames(args, side=96):
    """blender/turntable.py without bpy: args["frames"] PNGs of side px (0001.png, ...) on the
    0.82-gray background, each the convex hull of the GLB's vertices in dark gray, seen orthographically as the
    real camera sees it (along +z of W after the asset's matrix, args["elevation"] degrees from above), as the
    asset turns once about W's y."""
    import numpy as np
    from PIL import Image, ImageDraw
    from scipy.spatial import ConvexHull

    from harpor.physics.align import load_w

    assert len(args["assets"]) == 1 and args.get("frames_dir"), "turntable.py takes exactly one asset and frames_dir"
    a, out, n = args["assets"][0], Path(args["frames_dir"]), args["frames"]
    M = np.eye(4) if a.get("matrix") is None else np.asarray(a["matrix"], float)
    v = load_w(a["path"])[0] @ M[:3, :3].T + M[:3, 3]  # glTF -> W is Blender's import
    c, r = (v.min(0) + v.max(0)) / 2, np.linalg.norm(v.max(0) - v.min(0)) / 2
    e = math.radians(args.get("elevation", 0))
    right, down = np.array([1.0, 0.0, 0.0]), np.array([0.0, math.cos(e), -math.sin(e)])  # the camera's x and y in W
    out.mkdir(parents=True, exist_ok=True)

    def panel(p):  # the panel is 2.6 r wide
        return side / 2 + np.stack([(p - c) @ right, (p - c) @ down], 1) * side / (2.6 * r)

    for k in range(n):
        t = -2 * math.pi * k / n  # as turntable.py keys the pivot's turn
        R = np.array([[math.cos(t), 0.0, math.sin(t)], [0.0, 1.0, 0.0], [-math.sin(t), 0.0, math.cos(t)]])
        uv = panel((v - c) @ R.T + c)
        img = Image.new("RGB", (side, side), (209, 209, 209))
        draw = ImageDraw.Draw(img)
        draw.polygon([tuple(q) for q in uv[ConvexHull(uv).vertices]], fill=(64, 64, 64))
        img.save(out / f"{k + 1:04d}.png")


def fake_blender(argv):
    i = argv.index("--python")
    # Blender applies --python-exit-code only to scripts after it; a later flag lets failures exit 0.
    assert "--python-exit-code" in argv[:i], "--python-exit-code must come before --python"
    script, rest = Path(argv[i + 1]).name, argv[argv.index("--") + 1:]
    blend = next((x for x in argv[:i] if x.endswith(".blend")), None)
    if script == "make_candidate.py":
        import trimesh

        size = float(trimesh.load(rest[0], file_type="glb", force="scene", process=False).extents[0])
        Path(rest[1]).write_text(json.dumps({"kind": "good", "size": size, "seconds": 0.0}))
    elif script == "export_glb.py":
        scene = json.loads(Path(blend).read_text())
        time.sleep(scene["seconds"])
        write_glb(rest[0], scene["kind"], scene["size"])
        time.sleep(scene.get("pause", 0.0))  # the real script re-imports the GLB before it writes the report
        rep = {"meshes": 1, "materials": int(scene["kind"] != "no_material"), "images": 0, "triangles": 12,
               "nodes": ["mesh"], "warnings": []}
        Path(rest[1]).write_text(json.dumps(rep))
    elif script == "turntable.py":
        turntable_frames(json.loads(Path(rest[0]).read_text()))
    else:
        args = json.loads(Path(rest[0]).read_text())
        out = Path(args["out"])
        (out / "sheet.png").write_bytes(b"")
        review = {"views": [{"name": f"view{v['index']:02d}", "iou_candidate": 1.0} for v in args["views"]]}
        if args["original"]:
            review.update(center_shift=0.0, extent_ratio=[1.0, 1.0, 1.0])
        (out / "review.json").write_text(json.dumps(review))


def round_actions(script):
    """The actions of this agent: script itself, or of {"rounds": [...]} the list of run_budget.json's round."""
    if isinstance(script, list):
        return script
    n = json.loads(Path("run_budget.json").read_text())["round"]
    return script["rounds"][n - 1] if n <= len(script["rounds"]) else []


def run(script):
    receipt = None
    for a in script:
        do = a["do"]
        if do == "edit":
            scene = {}
            with contextlib.suppress(OSError, ValueError):
                scene = json.loads(Path("candidate.blend").read_text())
            scene = scene if isinstance(scene, dict) else {}
            scene.update(kind=a.get("kind", "good"), size=a.get("size", 1.0), seconds=a.get("seconds", 0.0))
            Path("candidate.blend").write_text(json.dumps(scene))
        elif do == "glb":
            write_glb(a["path"], a.get("kind", "good"), a.get("size", 1.0))
        elif do == "tool":
            args = [s.replace("{receipt}", receipt or "") for s in a["args"]]
            print("$", *args, flush=True)
            output, code = gator_tool.call(args)
            sys.stdout.write(output)
            if code:
                print(f"exit {code}", flush=True)
            receipts = sorted(Path("reviews").glob("r*"))
            receipt = receipts[-1].name if receipts else None
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
        run(round_actions(json.loads(Path(sys.argv[1]).read_text())))
