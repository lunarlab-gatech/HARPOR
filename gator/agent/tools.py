"""Helper commands export_glb, review, checkpoint and finish (build_plan.md note 5), and set_joint and
review_articulation of the articulation mode (note 11).

The agent calls them by name inside its container. The client in bin/ sends each command line to the
controller's tool server (server.py), which runs this module on the host. The host-only control
folder holds the events and the clock record. The events are the record of what the helpers did:
exports, reviews, joints, checkpoints and finish, each with a sha256 of the GLB or the .blend. The
controller trusts them and not the agent-writable copies in work/. Each command prints about 20 lines
at most and writes details to files under work/. Every command but finish ends with the seconds left
before the deadline, also when it fails, but not after a usage error. Paths in output and events are
relative to work/, so they read the same on the host and in the container.

set_joint and review_articulation work only in the articulation mode. Blender in the agent's SIF has
no trimesh and cannot see gator/parts, so the joint checks (gator.parts.joints), the joint motions and
the penetration test run here on the host, and the Blender scripts only store the joint and render.

These helpers run on the host but act on a folder the agent controls. So they read and write work/
without following symlinks: folders must be real folders, new files are created exclusively, and
reads take regular files only.

Environment: GATOR_WORK, GATOR_CONTROL (default: work/../control) and GATOR_BLENDER (default blender,
split like a shell).
"""

import argparse
import contextlib
import datetime
import hashlib
import json
import math
import os
import re
import shlex
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

from gator.fileio import write_json

BLENDER_SCRIPTS = Path(__file__).resolve().parent / "blender"
# The agents' models, here because this module needs only the standard library: claude_agent, the
# controller, gator.parts.propose and the web app's key check all read them.
CODEX_MODEL, CLAUDE_MODEL = "gpt-6-astra", "claude-opus-5-5"
EFFORTS = ("low", "medium", "high", "xhigh", "max")  # Claude's effort levels
FALLBACK_BETA = "server-side-fallback-2026-07-01"  # Claude's server-side refusal fallback
# Claude Opus 5.5's row of claude_agent.PRICES (USD per million tokens; the columns are explained there).
OPUS_PRICES = {"input_tokens": 4.0, "output_tokens": 20.0, "cache_read_input_tokens": 0.20,
               "cache_creation_input_tokens": 5.0, "cache_creation_1h": 8.0}
EXPORT_TIMEOUT_S = 180.0
MAX_EXCLUDED_S = 600.0  # cap on the export time that moves the deadline
REVIEW_TIMEOUT_S = 300.0  # render only
SET_JOINT_TIMEOUT_S = 120.0
VERDICTS = {"refinement": ("complete", "unchanged", "needs_review"),
            "articulation": ("articulated", "unchanged", "needs_review")}
MAX_LIMITATIONS = 3
MAX_DETAIL_VIEWS = 4
MAX_READ_BYTES = 2 << 30
RECEIPT = re.compile(r"[ra]\d{3}-[0-9a-f]{12}")  # r: review, a: review_articulation
REVIEW_EVENTS = ("review", "review_articulation")
AXIS_TOLERANCE = 1e-3  # a set_joint axis this close to unit length is normalized
MAX_JOINT_ROWS = 15
MAX_REVIEW_ROWS = 11
STATES, MIN_STATES, MAX_STATES = 3, 2, 5  # review_articulation --states
ORBIT_AZIMUTHS = (-45.0, 45.0)  # degrees from the front (view 0's side), 25 degrees up
SAMPLES = 2000  # surface samples per moving link
DEPTH_TOLERANCE = 0.005  # of the object's diagonal: the cut faces touch at rest, which is not penetration
THIN_TOLERANCE = 0.25  # of a link's thinnest obb extent: a sample inside a thin link is at most half as deep
# Rays of trimesh's contains test run near +x: the box of a ray along an axis meets few triangle boxes in
# the rtree, so the pure-Python test is about 50 times faster than with trimesh's default diagonal.
RAY = (1.0, 0.0123, 0.0071)
_children = []


def read_file(path, limit=MAX_READ_BYTES):
    """Bytes of a regular file. A final symlink, a FIFO, a device or a file over limit raises OSError."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_size > limit:
            raise OSError(f"{path} is not a regular file of at most {limit} bytes")
        with os.fdopen(os.dup(fd), "rb") as f:
            return f.read(limit + 1)
    finally:
        os.close(fd)


def open_new(path, mode="w"):
    """A new file for writing. Fails if anything, a symlink included, is already at path."""
    return os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644), mode)


def real_dir(work, name):
    """work/name as a real folder, created if missing. Exits if it is a symlink or a file."""
    d = Path(work) / name
    with contextlib.suppress(FileExistsError):
        d.mkdir()
    if not stat.S_ISDIR(os.lstat(d).st_mode):
        sys.exit(f"{name}/ must be a folder, not a link or a file")
    return d


def in_work(work, path):
    """A path the agent gave, as /work/... or relative to work, mapped to the host. Exits if it leaves work."""
    p, work = Path(path), Path(work).resolve()
    if p.is_absolute():
        try:
            p = p.relative_to("/work")
        except ValueError:
            sys.exit(f"{path} is outside /work")
    host = (work / p).resolve()
    if not host.is_relative_to(work):
        sys.exit(f"{path} is outside /work")
    return host


def sha256(path):
    return hashlib.sha256(read_file(path)).hexdigest()


def iso(t):
    return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).isoformat(timespec="seconds")


def env_values(path, names):
    """The values of the lines NAME=value of an env file whose NAME is in names, in file order, without
    one pair of quotes around them (as a shell reads them)."""
    values = []
    for line in Path(path).read_text().splitlines():
        name, sep, value = line.partition("=")
        if sep and name in names:
            value = value.strip()
            values.append(value[1:-1] if len(value) > 1 and value[0] == value[-1] and value[0] in "'\"" else value)
    return values


def log_event(control, event, **fields):
    with open(Path(control) / "events.jsonl", "a") as f:
        f.write(json.dumps({"time": time.time(), "event": event, **fields}) + "\n")


def json_lines(text):
    """The JSON objects among the lines of text. Other lines are skipped."""
    for line in text.splitlines():
        with contextlib.suppress(ValueError):
            e = json.loads(line)
            if isinstance(e, dict):
                yield e


def read_jsonl(path):
    """The JSON objects of a JSONL file. Other lines, and a missing or unreadable file, give nothing."""
    with contextlib.suppress(OSError):
        yield from json_lines(Path(path).read_text(errors="replace"))


def read_events(control):
    """Events from control/events.jsonl. Lines that do not parse are skipped. A missing file gives none,
    but one that cannot be read raises OSError, so a failed read never looks like an empty log."""
    path = Path(control) / "events.jsonl"
    return list(json_lines(path.read_text(errors="replace"))) if path.exists() else []


def write_budget(work, started_at, budget_s, excluded_s=0.0, exports=()):
    """Write work/run_budget.json, the agent's view of the deadline. Returns the deadline."""
    deadline = started_at + budget_s + excluded_s
    write_json(Path(work) / "run_budget.json", {
        "budget_s": budget_s, "started_at": started_at, "started_at_iso": iso(started_at),
        "deadline": deadline, "deadline_iso": iso(deadline), "excluded_s": excluded_s,
        "exports": list(exports), "export_timeout_s": EXPORT_TIMEOUT_S, "max_excluded_s": MAX_EXCLUDED_S,
    })
    return deadline


def start_budget(work, control, started_at, budget_s):
    """Start the clock: control/budget.json is the record, run_budget.json the agent's view."""
    write_json(Path(control) / "budget.json", {"started_at": started_at, "budget_s": budget_s})
    return write_budget(work, started_at, budget_s)


def _extend_budget(work, control):
    """Rewrite the agent's view from the control record and the export events, never from the view."""
    b = json.loads((Path(control) / "budget.json").read_text())
    exports = [{"export": e["export"], "seconds": e["seconds"]}
               for e in read_events(control) if e.get("event") == "export_done"]
    excluded = min(sum(x["seconds"] for x in exports), MAX_EXCLUDED_S)
    return write_budget(work, b["started_at"], b["budget_s"], excluded, exports)


def time_left(work, control):
    """Seconds before the deadline, counted as the controller counts them. Returns (left, budget_s)."""
    budget_s = json.loads((Path(control) / "budget.json").read_text())["budget_s"]
    return max(_extend_budget(work, control) - time.time(), 0.0), budget_s


def _kill_group(p):
    with contextlib.suppress(ProcessLookupError):
        os.killpg(p.pid, signal.SIGKILL)
    p.wait()


def _run(cmd, log, timeout):
    """Run cmd in its own process group and kill the group on timeout or SIGTERM.

    Blender may run inside Apptainer, so the whole group goes, not only the direct child.
    """
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    _children.append(p)
    try:
        return p.wait(timeout)
    except subprocess.TimeoutExpired:
        _kill_group(p)
        raise
    finally:
        _children.remove(p)


def run_ok(cmd, log, timeout):
    """True when _run exits 0. A timeout, or a command that does not start, goes to log and gives False."""
    try:
        return _run(cmd, log, timeout) == 0
    except (subprocess.TimeoutExpired, OSError) as e:
        log.write(f"{e}\n")
        return False


def blender_argv(blender, script, *args, blend=None):
    """The command that runs BLENDER_SCRIPTS/script in Blender without a window, with args after "--".
    blend: a .blend to open first, with the auto-run of the scripts stored in it off (the agent wrote
    it). --python-exit-code must come before --python: Blender applies it only to the scripts that
    follow it."""
    return [*blender, "-b", "--factory-startup", *(["--disable-autoexec", str(blend)] if blend else []),
            "--python-exit-code", "1", "--python", str(BLENDER_SCRIPTS / script), "--", *map(str, args)]


def _on_sigterm(signum, frame):
    # No wait here: the interrupted code may hold the child's wait lock.
    for p in list(_children):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(p.pid, signal.SIGKILL)
    sys.exit(128 + signum)


def _count(control, event):
    return sum(e.get("event") == event for e in read_events(control))


def export_glb(work, control, blender, blend=None, timeout=EXPORT_TIMEOUT_S):
    """Export the saved .blend to exports/NNN.glb outside the time budget and re-import it to check it.

    Returns the cached GLB when the .blend and that GLB are unchanged since the last export. Every
    export_started gets an export_done, also when the export fails. Returns (glb, report, cached).
    """
    work = Path(work).resolve()
    exports = real_dir(work, "exports")
    blend = in_work(work, blend or "candidate.blend")
    try:
        blend_sha = sha256(blend)
    except OSError:
        sys.exit(f"{blend.relative_to(work)} is missing or not a regular file; save the scene first")
    done = [e for e in read_events(control) if e.get("event") == "export_done"]
    if done and done[-1]["ok"] and done[-1]["blend_sha256"] == blend_sha:
        glb = work / done[-1]["glb"]
        with contextlib.suppress(OSError, ValueError):
            if sha256(glb) == done[-1]["glb_sha256"]:
                return glb, json.loads(read_file(glb.with_suffix(".json"), 1 << 20)), True
    n = _count(control, "export_started") + 1
    glb = exports / f"{n:03d}.glb"
    report = glb.with_suffix(".json")
    cmd = blender_argv(blender, "export_glb.py", glb, report, blend=blend)
    log_event(control, "export_started", export=n)
    t, ok, glb_sha, rep, why = time.time(), False, None, None, ""
    try:
        with open_new(glb.with_suffix(".log")) as log:
            ok = run_ok(cmd, log, timeout)
        if ok:
            glb_sha, rep = sha256(glb), json.loads(read_file(report, 1 << 20))
    except (OSError, ValueError) as e:
        ok, why = False, f" ({e})"
    finally:
        seconds = min(time.time() - t, timeout)
        log_event(control, "export_done", export=n, seconds=seconds, ok=ok, glb=str(glb.relative_to(work)),
                  blend_sha256=blend_sha, glb_sha256=glb_sha)
        _extend_budget(work, control)
    if not ok:
        sys.exit(f"export {n:03d} failed after {seconds:.1f} s; see {glb.with_suffix('.log').relative_to(work)}{why}")
    return glb, rep, False


def manifest(control):
    """The controller's copy of input/manifest.json, which the agent cannot edit."""
    return json.loads((Path(control) / "manifest.json").read_text())


def check_detail(detail):
    """Close-up cameras as [x, y, z, radius, dx, dy, dz] in W. Exits on a bad rig."""
    detail = [list(map(float, d)) for d in detail or []]
    if len(detail) > MAX_DETAIL_VIEWS:
        sys.exit(f"{len(detail)} detail views given; at most {MAX_DETAIL_VIEWS} are allowed")
    for d in detail:
        if len(d) != 7 or not all(map(math.isfinite, d)) or d[3] <= 0 or not any(d[4:]):
            sys.exit(f"bad detail view {d}: give X Y Z RADIUS DX DY DZ with RADIUS > 0 and a nonzero direction")
    return detail


def review(work, control, blender, clay=False, detail=()):
    """Render the original and the candidate with the same cameras and lights. Returns the receipt."""
    work = Path(work).resolve()
    detail = check_detail(detail)
    reviews = real_dir(work, "reviews")
    glb, _, cached = export_glb(work, control, blender)
    m = manifest(control)
    sha = sha256(glb)
    receipt = f"r{_count(control, 'review') + 1:03d}-{sha[:12]}"
    out = reviews / receipt
    try:
        out.mkdir()
    except FileExistsError:
        sys.exit(f"reviews/{receipt} already exists; remove it and run review again")
    write_json(out / "args.json", {
        "root": str(work), "out": str(out), "clay": clay, "views": m["views"], "candidate": str(glb),
        "original": str(work / "original" / "original.glb"), "detail": detail,
    })
    with open_new(out / "blender.log") as log:
        ok = run_ok(blender_argv(blender, "review.py", out / "args.json"), log, REVIEW_TIMEOUT_S)
    if not ok:
        sys.exit(f"review render failed; see {(out / 'blender.log').relative_to(work)}")
    rec = {"receipt": receipt, "glb": str(glb.relative_to(work)), "sha256": sha, "clay": clay, "cached": cached,
           "sheet": str((out / "sheet.png").relative_to(work)), "review": str((out / "review.json").relative_to(work)),
           "time": time.time()}
    write_json(out / "receipt.json", rec)
    log_event(control, "review", receipt=receipt, glb=rec["glb"], sha256=sha)
    return rec


def _articulation_only(control, command):
    if manifest(control)["mode"] != "articulation":
        sys.exit(f"{command} works only in the articulation mode")


def glb_joints(data):
    """(joints, mesh node names, glTF dict) of GLB bytes (gator.parts.joints). Exits when they cannot be read."""
    from gator.parts import joints as J

    try:
        if data[:4] != b"glTF" or data[16:20] != b"JSON":
            raise ValueError("not a binary glTF 2.0 file")
        doc = json.loads(data[20:20 + int.from_bytes(data[12:16], "little")])
        return J.read_joints(doc), J.mesh_nodes(doc), doc
    except (ValueError, AttributeError, TypeError) as e:
        sys.exit(f"cannot read the joints of the GLB: {e}")


def joint_spec(parent, kind, origin=None, axis=None, lower=None, upper=None):
    """The joint of set_joint's arguments in frame W (gator.parts.joints). An axis within AXIS_TOLERANCE
    of unit length is normalized. Exits with the reason on a bad argument."""
    from gator.parts import joints as J

    if kind == "fixed":
        if any(x is not None for x in (origin, axis, lower, upper)):
            sys.exit("a fixed joint takes only --child, --parent and --type")
        return {"version": 1, "frame": "W", "parent": parent, "type": "fixed"}
    if parent == J.WORLD:
        sys.exit("only a fixed joint attaches a part to the world: give --type fixed with --parent world")
    given = {"origin": origin, "axis": axis, "lower": lower, "upper": upper}
    missing = [f"--{k}" for k, v in given.items() if v is None]
    if missing:
        sys.exit(f"a {kind} joint needs {', '.join(missing)}")
    if not all(map(math.isfinite, [*origin, *axis, lower, upper])):
        sys.exit("origin, axis, lower and upper must be finite numbers")
    n = math.sqrt(sum(a * a for a in axis))
    if abs(n - 1) > AXIS_TOLERANCE:
        sys.exit(f"--axis must be a unit vector; its length is {n:.6g}")
    if not lower <= 0 <= upper or not lower < upper:
        sys.exit(f"the range [{lower:g}, {upper:g}] must hold 0 with lower < upper: the part as it is now is the "
                 "pose 0; use type fixed for a part that cannot move")
    if kind == "revolute" and max(-lower, upper) > 2 * math.pi:
        sys.exit("a revolute range is in radians, at most 2 pi each way")
    return {"version": 1, "frame": "W", "parent": parent, "type": kind, "origin": [float(x) for x in origin],
            "axis": [a / n for a in axis], "lower": float(lower), "upper": float(upper)}


def _set_joint_run(work, blender, blend, child, spec, label):
    """Run blender/set_joint.py once on blend. Returns its record, set_joint/<label>.json."""
    d = real_dir(work, "set_joint")
    out, log_path = d / f"{label}.json", d / f"{label}.log"
    if os.path.lexists(out):  # set_joint.py would change the .blend and then fail to write its record
        sys.exit(f"{out.relative_to(work)} already exists; remove it and run set_joint again")
    cmd = blender_argv(blender, "set_joint.py", out, child, json.dumps(spec), blend=blend)
    rec = None
    try:
        with open_new(log_path) as log:
            ok = run_ok(cmd, log, SET_JOINT_TIMEOUT_S)
        if ok:
            rec = json.loads(read_file(out, 1 << 22))
    except FileExistsError:
        sys.exit(f"{log_path.relative_to(work)} already exists; remove it and run set_joint again")
    except (OSError, ValueError):
        pass
    if not isinstance(rec, dict) or not isinstance(rec.get("joints"), dict) or not isinstance(rec.get("links"), list):
        sys.exit(f"set_joint failed; see {log_path.relative_to(work)}")
    return rec


def _problem_key(problem):
    """A problem of gator.parts.joints.validate as a key: a cycle is its set of links, because validate
    words the same cycle from the first of its links that it meets."""
    m = re.fullmatch(r"the joints form a cycle: (.+)", problem)
    return ("cycle", frozenset(m.group(1).split(" -> "))) if m else problem


def set_joint(work, control, blender, child, spec):
    """Set the joint of child in candidate.blend, or remove it (spec None), then check the joints of the
    whole scene with gator.parts.joints.validate. A change that adds a problem is undone: the child gets
    its previous joint back. Problems from before the change do not block it. Writes work/joints.json and
    returns it: joints, links, holders, the problems left (all from before), and root, fixed_base and the
    welded loose links when the tree is valid."""
    from gator.parts import joints as J

    work = Path(work).resolve()
    _articulation_only(control, "set_joint")
    if child == J.WORLD:
        sys.exit(f'"{J.WORLD}" is the fixed base, not a part')
    if spec and spec["parent"] == child:
        sys.exit(f"{child} cannot be its own parent")
    blend = in_work(work, "candidate.blend")
    try:
        sha256(blend)
    except OSError:
        sys.exit("candidate.blend is missing or not a regular file; save the scene first")
    label, action = f"{_count(control, 'joint') + 1:03d}", "set" if spec else "remove"
    try:
        rec = _set_joint_run(work, blender, blend, child, spec, label)
        if not rec.get("ok"):
            sys.exit(str(rec.get("error") or "set_joint failed"))
    except SystemExit as e:
        log_event(control, "joint", child=child, action=action, ok=False, why=str(e.code))
        raise
    joints, links, previous = rec["joints"], rec["links"], rec.get("previous")
    before = dict(joints)
    if previous is None:
        before.pop(child, None)
    else:
        before[child] = previous
    known = {_problem_key(p) for p in J.validate(before, links)}
    problems = J.validate(joints, links)
    added = [p for p in problems if _problem_key(p) not in known]
    if added:
        try:
            _set_joint_run(work, blender, blend, child, previous, label + "_undo")
            undone = f"{child} has no joint again" if previous is None else f"the joint of {child} is back as it was"
        except SystemExit as e:
            undone = f"the undo failed ({e.code}); remove the joint with set_joint --child {child} --remove"
        log_event(control, "joint", child=child, action=action, ok=False, why=added)
        sys.exit(f"refused: {added[0]}" + (f" (and {len(added) - 1} more)" if len(added) > 1 else "") + f"\n{undone}")
    out = {"joints": joints, "links": links, "holders": rec.get("holders") or {}, "problems": problems}
    if joints and not problems:
        root, _, fixed = J.tree(joints, links)
        out.update(root=root, fixed_base=fixed, loose=J.welds(joints, links, root, out["holders"]))
    write_json(work / "joints.json", out)
    log_event(control, "joint", child=child, action=action, ok=True, joint=spec, joints=len(joints),
              blend_sha256=sha256(blend))
    return out


def carriers(joints, links, held):
    """{link: the link whose motion it has}: itself for the root and the child of a moving joint; for
    the child of a fixed joint, the carrier of its parent; for a loose link, the carrier of the link it
    is welded to (gator.parts.joints.welds; held: holders). Links that move alike but hang on different
    joints get different carriers."""
    from gator.parts import joints as J

    root, _, _ = J.tree(joints, links)
    weld = J.welds(joints, links, root, held)
    out = {}
    for link in links:
        chain, k = [], link
        while k not in out and k != root and not (k in joints and joints[k]["type"] != "fixed"):
            chain.append(k)
            k = joints[k]["parent"] if k in joints else weld[k]  # holders lead up the glTF tree
        top = out.get(k, k)
        for c in [*chain, k]:
            out[c] = top
    return out


def articulation_poses(joints, links, held, states=STATES):
    """[{name, t, q, motions}] for t from 0 to 1 in `states` steps: every joint at lower + t (upper - lower).
    motions holds each link's 4x4 motion in W: a jointed link moves with its parent, a loose link with the
    link it is welded to (carriers; held: holders)."""
    import numpy as np

    from gator.parts import joints as J

    root, children, _ = J.tree(joints, links)
    carrier = carriers(joints, links, held)
    full = {c: J.full(j) for c, j in joints.items()}
    poses = []
    for k in range(states):
        t = k / (states - 1)
        q = {c: j["lower"] + t * (j["upper"] - j["lower"]) for c, j in full.items()}
        M, todo = {root: np.eye(4)}, [root]
        while todo:
            p = todo.pop()
            for c in children[p]:
                M[c] = M[p] @ J.joint_transform(joints[c], q[c])
                todo.append(c)
        poses.append({"name": f"s{k}", "t": t, "q": q,
                      "motions": {link: M[link] if link in M else M[carrier[link]] for link in links}})
    return poses


def link_meshes(glb):
    """{link: trimesh.Trimesh in W} of a GLB: the primitives of each mesh node (gator.physics.mjcf.load_s),
    as one mesh without textures and with coincident vertices merged, so a closed part is watertight."""
    import numpy as np
    import trimesh

    from gator.physics.mjcf import load_s

    parts = {}
    for link, _, m in load_s(glb, by_link=True):
        parts.setdefault(link, []).append(m)
    out = {}
    for link, ms in parts.items():
        offsets = np.cumsum([0] + [len(m.vertices) for m in ms[:-1]])
        out[link] = trimesh.Trimesh(np.concatenate([m.vertices for m in ms]),
                                    np.concatenate([m.faces + o for m, o in zip(ms, offsets)]))
    return out


def _thinnest(mesh):
    """The smallest extent of a mesh's oriented box (0 for a flat or empty mesh)."""
    import trimesh

    try:
        return float(min(trimesh.bounds.oriented_bounds(mesh)[1]))
    except Exception:  # qhull fails on a flat or degenerate mesh
        return 0.0


def penetration(meshes, joints, poses, carrier, samples=SAMPLES, seed=0):
    """Per moving joint and pose: the share of the surface samples of the joint's child, and of the links
    it carries (carriers), that lie inside another link deeper than that link's tolerance, the largest
    depth among them and the link that holds the most of them. meshes: link_meshes at rest. The
    tolerance is DEPTH_TOLERANCE of the diagonal, at most THIN_TOLERANCE of the link's thinnest obb
    extent, so a part that passes through a thin part is found. Inside is trimesh's ray parity test
    (contains), depth the distance to that link's surface (closest_point). The samples move into each
    other link's rest frame, so each mesh builds its ray and proximity trees once."""
    import numpy as np
    import trimesh
    from trimesh.ray.ray_util import contains_points

    t0 = time.time()
    allv = np.concatenate([m.vertices for m in meshes.values()])
    diag = float(np.linalg.norm(np.ptp(allv, 0)))
    tol = DEPTH_TOLERANCE * diag
    tols = {k: min(tol, THIN_TOLERANCE * _thinnest(m)) or tol for k, m in meshes.items()}  # flat: nothing inside
    ray = np.array(RAY) / np.linalg.norm(RAY)
    report = {}
    for child, j in joints.items():
        if j["type"] == "fixed":
            continue
        group = [k for k in meshes if carrier[k] == child]
        pts = np.concatenate([trimesh.sample.sample_surface(meshes[k], samples, seed=seed)[0] for k in group])
        states = []
        for p in poses:
            deep, depth, count = np.zeros(len(pts), bool), np.zeros(len(pts)), {}
            for other, m in meshes.items():
                if other in group:
                    continue
                T = np.linalg.inv(p["motions"][other]) @ p["motions"][child]
                x = pts @ T[:3, :3].T + T[:3, 3]
                inside = contains_points(m.ray, x, check_direction=ray)
                if not inside.any():
                    continue
                d = np.zeros(len(x))
                d[inside] = trimesh.proximity.closest_point(m, x[inside])[1]
                hit = d > tols[other]
                count[other] = int(hit.sum())
                deep |= hit
                depth = np.maximum(depth, np.where(hit, d, 0.0))
            worst = max(count, key=count.get) if any(count.values()) else None
            states.append({"name": p["name"], "inside_share": round(float(deep.mean()), 4),
                           "max_depth": float(depth.max()), "worst_other_link": worst})
        report[child] = {"type": j["type"], "parent": j["parent"], "moving_links": group, "samples": len(pts),
                         "states": states}
    return {"tolerance": tol, "diagonal": diag, "samples_per_link": samples,
            "links": {k: {"faces": len(m.faces), "watertight": bool(m.is_watertight), "tolerance": tols[k]}
                      for k, m in meshes.items()},
            "joints": report, "seconds": round(time.time() - t0, 2)}


def review_articulation(work, control, blender, states=STATES, clay=False):
    """Export the scene, move every joint from lower to upper in `states` steps (all joints at once),
    check the moving parts for penetration on the host and render each state under every input camera
    and 2 orbit views. Returns (the receipt record, report.json)."""
    import tempfile

    from gator.parts import joints as J

    work = Path(work).resolve()
    _articulation_only(control, "review_articulation")
    if not MIN_STATES <= states <= MAX_STATES:
        sys.exit(f"--states {states}: give {MIN_STATES} to {MAX_STATES}")
    reviews = real_dir(work, "reviews_articulation")
    glb, _, cached = export_glb(work, control, blender)
    data = read_file(glb)
    sha = hashlib.sha256(data).hexdigest()
    joints, links, doc = glb_joints(data)
    if not joints:
        sys.exit(f"{glb.relative_to(work)} holds no joint; set the joints with set_joint first, or run review when no "
                 "part moves")
    problems = J.validate(joints, links)
    if problems:
        sys.exit(f"the joints of {glb.relative_to(work)} are not a valid tree: {problems[0]}")
    if all(j["type"] == "fixed" for j in joints.values()):
        sys.exit("every joint is fixed, so nothing moves; review_articulation needs a revolute or prismatic joint "
                 "(run review when no part moves)")
    receipt = f"a{_count(control, 'review_articulation') + 1:03d}-{sha[:12]}"
    out = reviews / receipt
    try:
        out.mkdir()
    except FileExistsError:
        sys.exit(f"reviews_articulation/{receipt} already exists; remove it and run review_articulation again")
    held = J.holders(doc)
    poses = articulation_poses(joints, links, held, states)
    try:
        with tempfile.NamedTemporaryFile(dir=control, suffix=".glb") as f:  # the bytes that were hashed, host only
            f.write(data)
            f.flush()
            meshes = link_meshes(f.name)
        if sorted(meshes) != sorted(links):
            raise ValueError(f"the mesh nodes {sorted(links)} load as the meshes {sorted(meshes)}")
        check = penetration(meshes, joints, poses, carriers(joints, links, held))
    except Exception as e:  # a mesh trimesh cannot test; the renders still show the motion
        check = {"error": f"{type(e).__name__}: {e}", "tolerance": None, "links": {}, "joints": {}}
    report = {"receipt": receipt, "glb": str(glb.relative_to(work)), "sha256": sha,
              "states": [{k: p[k] for k in ("name", "t", "q")} for p in poses], **check}
    write_json(out / "report.json", report)
    write_json(out / "args.json", {
        "root": str(work), "out": str(out), "clay": clay, "views": manifest(control)["views"], "candidate": str(glb),
        "links": links, "orbit": list(ORBIT_AZIMUTHS),
        "states": [{"name": p["name"], "t": p["t"], "motions": {k: M.tolist() for k, M in p["motions"].items()}}
                   for p in poses]})
    with open_new(out / "blender.log") as log:
        ok = run_ok(blender_argv(blender, "review_articulation.py", out / "args.json"), log, REVIEW_TIMEOUT_S)
    if not ok:
        sys.exit(f"review_articulation render failed; see {(out / 'blender.log').relative_to(work)}")
    rec = {"receipt": receipt, "glb": str(glb.relative_to(work)), "sha256": sha, "clay": clay, "cached": cached,
           "joints": len(joints), "states": len(poses), "sheet": str((out / "sheet.png").relative_to(work)),
           "report": str((out / "report.json").relative_to(work)), "time": time.time()}
    write_json(out / "receipt.json", rec)
    log_event(control, "review_articulation", receipt=receipt, glb=rec["glb"], sha256=sha, joints=len(joints))
    return rec, report


def _receipt(work, control, receipt):
    """The review or review_articulation event of a receipt, checked against the GLB as it is now.
    receipt.json is not read."""
    rec = next((e for e in read_events(control) if e.get("event") in REVIEW_EVENTS and e.get("receipt") == receipt),
               None)
    if not RECEIPT.fullmatch(receipt) or rec is None:
        sys.exit(f"unknown review receipt {receipt!r}")
    try:
        data = read_file(Path(work) / rec["glb"])
    except OSError:
        data = b""
    if hashlib.sha256(data).hexdigest() != rec["sha256"]:
        sys.exit(f"{rec['glb']} changed after review {receipt}")
    return rec, data


def checkpoint(work, control, receipt, defect):
    """Read-only copy of a reviewed GLB in checkpoints/, labeled needs_review with the open defect."""
    work = Path(work).resolve()
    rec, data = _receipt(work, control, receipt)
    dst = real_dir(work, "checkpoints") / f"{_count(control, 'checkpoint') + 1:03d}-{receipt}.glb"
    try:
        with open_new(dst, "wb") as f:
            f.write(data)
            os.fchmod(f.fileno(), 0o444)
    except FileExistsError:
        sys.exit(f"{dst.relative_to(work)} already exists")
    meta = {"receipt": receipt, "glb": str(dst.relative_to(work)), "sha256": rec["sha256"],
            "status": "needs_review", "defect": defect, "time": time.time()}
    write_json(dst.with_suffix(".json"), meta)
    log_event(control, "checkpoint", glb=meta["glb"], sha256=rec["sha256"], defect=defect, receipt=receipt)
    return meta


def _articulated(control, rec, data):
    """Exits unless the GLB of a receipt holds a valid joint set that review_articulation reviewed."""
    from gator.parts import joints as J

    joints, links, _ = glb_joints(data)
    if not joints:
        sys.exit(f"{rec['glb']} holds no joint; verdict articulated needs at least one (set_joint)")
    problems = J.validate(joints, links)
    if problems:
        sys.exit(f"the joints of {rec['glb']} are not a valid tree: {problems[0]}")
    if not any(e.get("event") == "review_articulation" and e.get("sha256") == rec["sha256"]
               for e in read_events(control)):
        sys.exit(f"{rec['glb']} has no articulation review; run review_articulation and finish with its receipt")


def finish(work, control, receipt, verdict, reason, limitations=()):
    """Log the hand-off and write work/finish.json for the agent. The controller then ends the session.

    Verdict articulated needs a GLB with a valid joint set that review_articulation reviewed."""
    work = Path(work).resolve()
    mode = manifest(control)["mode"]
    allowed = VERDICTS[mode]
    if verdict not in allowed:
        sys.exit(f"unknown verdict {verdict!r} for {mode}; use one of {', '.join(allowed)}")
    if len(limitations) > MAX_LIMITATIONS:
        sys.exit(f"{len(limitations)} limitations given; at most {MAX_LIMITATIONS} are allowed")
    rec, data = _receipt(work, control, receipt)
    if verdict == "articulated":
        _articulated(control, rec, data)
    out = {"receipt": receipt, "verdict": verdict, "reason": reason, "limitations": list(limitations),
           "glb": "original/original.glb" if verdict == "unchanged" else rec["glb"],
           "sha256": None if verdict == "unchanged" else rec["sha256"], "time": time.time()}
    write_json(work / "finish.json", out)
    log_event(control, "finish", **{k: v for k, v in out.items() if k != "time"})
    return out


def _time_left_line(work, control):
    left, budget_s = time_left(work, control)
    return f"time left: {left:.0f} s of the {budget_s:.0f} s budget"


def _fmt(x):
    """A number or a list of numbers, short; anything else as repr."""
    if isinstance(x, (list, tuple)):
        return "(" + ", ".join(_fmt(v) for v in x) + ")"
    return f"{x:.4g}" if isinstance(x, (int, float)) and not isinstance(x, bool) else repr(x)


def _joint_row(child, j):
    """One line of set_joint's table."""
    if not isinstance(j, dict):
        return f"{child}: {j!r} is not a joint"
    row = f"{child} > {j.get('parent')}: {j.get('type')}"
    if j.get("type") != "fixed":
        row += (f", origin {_fmt(j.get('origin'))}, axis {_fmt(j.get('axis'))}, range [{_fmt(j.get('lower'))}, "
                f"{_fmt(j.get('upper'))}] {'rad' if j.get('type') == 'revolute' else 'm'}")
    return row


def _state_cell(s):
    if not s["inside_share"]:
        return f"{s['name']} none"
    return f"{s['name']} {s['inside_share']:.1%}, deepest {s['max_depth']:.3g}, most in {s['worst_other_link']}"


def _command(a, work, control, blender):
    """Run one parsed helper command and print its result."""
    if a.cmd == "export_glb":
        glb, rep, cached = export_glb(work, control, blender, a.blend)
        budget = json.loads((control / "budget.json").read_text())
        deadline = _extend_budget(work, control)
        print(f"{glb.relative_to(work)} ({'cached, scene unchanged' if cached else 'new'})")
        print(f"sha256 {sha256(glb)}")
        print(f"meshes {rep['meshes']}, materials {rep['materials']}, images {rep['images']}, "
              f"triangles {rep['triangles']}")
        if "nodes" in rep and manifest(control)["mode"] == "articulation":
            nodes = [str(n) for n in rep["nodes"]]
            print(f"nodes {len(nodes)}: {', '.join(nodes[:10])}{' ...' if len(nodes) > 10 else ''}; "
                  f"joints {len(rep.get('joints') or {})}")
        print(f"deadline {iso(deadline)} (unix {deadline:.0f}), excluded {deadline - budget['started_at'] - budget['budget_s']:.1f} s")
        for w in rep["warnings"][:5]:
            print(f"warning: {w}")
    elif a.cmd == "review":
        rec = review(work, control, blender, a.clay, a.detail)
        print(f"receipt {rec['receipt']}")
        print(f"candidate {rec['glb']} ({'cached' if rec['cached'] else 'new'} export)")
        print(f"sheet {rec['sheet']}")
        print(f"details {rec['review']}")
        rev = json.loads(read_file(work / rec["review"], 1 << 24))
        if "center_shift" in rev:
            print(f"evidence, not proof: center shift {rev['center_shift']:.3f} of the original diagonal, "
                  f"extent ratio {' '.join(f'{x:.2f}' for x in rev['extent_ratio'])}")
        ious = [f"{v['name']} {v['iou_candidate']:.2f}" for v in rev["views"] if "iou_candidate" in v]
        if ious:
            print("candidate mask IoU: " + ", ".join(ious[:12]))
    elif a.cmd == "set_joint":
        if a.remove:
            if any(x is not None for x in (a.parent, a.type, a.origin, a.axis, a.lower, a.upper)):
                sys.exit("--remove takes only --child")
            spec = None
        elif a.parent is None or a.type is None:
            sys.exit("give --parent and --type, or --remove")
        else:
            spec = joint_spec(a.parent, a.type, a.origin, a.axis, a.lower, a.upper)
        rec = set_joint(work, control, blender, a.child, spec)
        js = rec["joints"]
        print(f"joint of {a.child} {'removed' if spec is None else 'set'}; candidate.blend holds {len(js)} "
              f"joint{'' if len(js) == 1 else 's'} (all in joints.json)")
        if "root" in rec:
            welded = [f"{k} > {v}" for k, v in rec["loose"].items()]
            print(f"root {rec['root']} ({'fixed base' if rec['fixed_base'] else 'free base'})"
                  + (f"; welded without a joint: {', '.join(welded[:6])}{' ...' if len(welded) > 6 else ''}"
                     if welded else ""))
        for child, j in list(js.items())[:MAX_JOINT_ROWS]:
            print("  " + _joint_row(child, j))
        if len(js) > MAX_JOINT_ROWS:
            print(f"  ... and {len(js) - MAX_JOINT_ROWS} more")
        if rec["problems"]:
            n = len(rec["problems"])
            print(f"problem left from before this change: {rec['problems'][0]}"
                  + (f" (and {n - 1} more)" if n > 1 else ""))
    elif a.cmd == "review_articulation":
        rec, rep = review_articulation(work, control, blender, a.states, a.clay)
        print(f"receipt {rec['receipt']}")
        n = rec["joints"]
        print(f"candidate {rec['glb']} ({'cached' if rec['cached'] else 'new'} export), {n} joint{'' if n == 1 else 's'}")
        print(f"sheet {rec['sheet']}: one row per view: each input view (the photo, then the states), then 2 orbit "
              "views (the states)")
        print(f"details {rec['report']}")
        names, ts = (", ".join(x) for x in zip(*[(s["name"], f"{s['t']:.2g}") for s in rep["states"]]))
        print(f"states {names}: every joint at {ts} of the way from lower to upper")
        if rep.get("error"):
            print(f"penetration check failed: {rep['error'][:300]}")
        else:
            print(f"penetration: the share of a moving part's surface samples deeper than {rep['tolerance']:.3g} "
                  "(less in thin parts) inside another part, the largest depth, and the part that holds the most")
        rows = list(rep["joints"].items())
        for child, j in rows[:MAX_REVIEW_ROWS]:
            print(f"  {child} ({j['type']} > {j['parent']}): " + "; ".join(_state_cell(s) for s in j["states"]))
        if len(rows) > MAX_REVIEW_ROWS:
            print(f"  ... and {len(rows) - MAX_REVIEW_ROWS} more")
        unclosed = [k for k, v in rep["links"].items() if not v["watertight"]]
        if unclosed:
            print(f"warning: not closed, so inside and outside are uncertain: {', '.join(unclosed[:8])}")
    elif a.cmd == "checkpoint":
        meta = checkpoint(work, control, a.receipt, a.defect)
        print(f"checkpoint {meta['glb']} (read-only, needs_review)")
        print(f"defect: {meta['defect']}")
    else:
        out = finish(work, control, a.receipt, a.verdict, a.reason, a.limitation)
        print(f"finished: {out['verdict']}, {out['glb']}")
        if out["sha256"]:
            print(f"sha256 {out['sha256']}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="gator-tools", description="GATOR agent helpers (build_plan.md note 5).")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export_glb", help="export the saved .blend to a GLB, outside the time budget")
    e.add_argument("--blend", help="a .blend inside /work; default /work/candidate.blend")
    r = sub.add_parser("review", help="render original and candidate with the same cameras and lights")
    r.add_argument("--clay", action="store_true", help="opaque gray material on every surface")
    r.add_argument("--detail", nargs=7, type=float, action="append", metavar=("X", "Y", "Z", "R", "DX", "DY", "DZ"),
                   help="a close-up camera on the sphere at X Y Z with radius R, looking along DX DY DZ (all in W); "
                        f"repeat up to {MAX_DETAIL_VIEWS} times")
    j = sub.add_parser("set_joint", help="set or remove the joint of one part in candidate.blend (articulation mode)")
    j.add_argument("--child", required=True, help="the part: its object's name")
    j.add_argument("--parent", help="the part it moves against, or world (with --type fixed) for a fixed base")
    j.add_argument("--type", choices=("revolute", "prismatic", "fixed"))
    j.add_argument("--origin", nargs=3, type=float, metavar=("X", "Y", "Z"), help="a point on the axis, in W")
    j.add_argument("--axis", nargs=3, type=float, metavar=("X", "Y", "Z"), help="the unit direction of the axis, in W")
    j.add_argument("--lower", type=float, help="radians (revolute) or meters (prismatic), at most 0")
    j.add_argument("--upper", type=float, help="at least 0 and more than --lower")
    j.add_argument("--remove", action="store_true", help="remove the joint of --child")
    v = sub.add_parser("review_articulation", help="render the joints from lower to upper and check for penetration")
    v.add_argument("--states", type=int, default=STATES, help=f"poses from lower to upper, {MIN_STATES} to "
                                                               f"{MAX_STATES}")
    v.add_argument("--clay", action="store_true", help="opaque gray material on every surface")
    c = sub.add_parser("checkpoint", help="read-only copy of a reviewed GLB")
    c.add_argument("--receipt", required=True)
    c.add_argument("--status", choices=["needs_review"], default="needs_review")
    c.add_argument("--defect", required=True, help="the open defect")
    f = sub.add_parser("finish", help="hand off the reviewed GLB; the last action")
    f.add_argument("--receipt", required=True)
    f.add_argument("--verdict", required=True, help=f"refinement: {', '.join(VERDICTS['refinement'])}; "
                                                    f"articulation: {', '.join(VERDICTS['articulation'])}")
    f.add_argument("--reason", required=True)
    f.add_argument("--limitation", action="append", default=[], help="repeat up to 3 times")
    a = p.parse_args(argv)
    work = Path(os.environ.get("GATOR_WORK", "/work")).resolve()
    control = Path(os.environ.get("GATOR_CONTROL", work.parent / "control"))
    blender = shlex.split(os.environ.get("GATOR_BLENDER", "blender"))

    try:
        _command(a, work, control, blender)
    except SystemExit as e:
        # A failed helper other than finish also ends with the time left; argparse and SIGTERM exits do not.
        if a.cmd == "finish" or not isinstance(e.code, str):
            raise
        try:
            line = _time_left_line(work, control)
        except (OSError, ValueError, KeyError):
            raise e from None
        sys.exit(f"{e.code}\n{line}")
    if a.cmd != "finish":
        print(_time_left_line(work, control))


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _on_sigterm)
    main()
