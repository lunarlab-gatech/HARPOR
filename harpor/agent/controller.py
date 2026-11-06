"""Agent-stage controller (build_plan.md note 5).

Prepares the session folder, runs the agent under a watchdog with a 600 s budget (the caller passes
900 s for the articulation mode, build_plan.md note 11), serves the helper commands from the host,
checks GLBs as paper A.4 describes and against the original's size, and picks the deliverable with the
fallback order. An articulation whose parts.json has a found requirement runs in rounds, as
PhysAg3D's repair loop does (paper A.3, A.4; run_rounds): a build, then repairs or one planning round
(replan) while the host's check (harpor.agent.validate) finds something an agent can fix, each round
with a new agent. Each build or repair keeps its pick as a candidate in control/candidates, and the
session delivers the best of them (select_candidate, paper A.6). The agent is one of two, chosen per
session in prepare_session and recorded in control/agent.json:

  codex    GPT-6-Astra through the Codex CLI, as in the paper. The CLI runs in an Apptainer
           container that holds the API key (apptainer_command).
  claude   Claude Opus 5.5 through our own Messages API loop, harpor/agent/claude_agent.py. The loop
           runs on the host with the key in its memory only, and runs the model's shell commands in
           an Apptainer container without the key and without network (shell_command).

Session layout:

  root/original.glb   the controller's own copy of the generator output (refinement), or of the split
                      parts.glb (articulation)
  root/result.json    the result record
  root/control/       host only: events.jsonl (what the helpers did, with GLB hashes; the deliverable
                      and the deadline come from it, and from a round's own events in rounds),
                      budget.json, manifest.json, agent.json, candidate.log, previews.log, exports/
                      (the host's copy of each export and its report) and staging/ (where Blender
                      writes them first, tools.export_glb); in
                      articulation also parts.json, and validate/ and ranges.json of the host's checks
                      (harpor.agent.validate); in rounds also candidates/vNN/ (store_candidate) and
                      evidence.json (harpor.agent.evidence)
  root/work/          mounted as /work: AGENTS.md (codex only), input/, original/, candidate.blend,
                      exports/, reviews/, checkpoints/, run_budget.json, finish.json; in articulation
                      also input/parts.json, set_joint/, joints.json, reviews_articulation/ and
                      simulate/; in rounds also evidence/round_NN.json (the check of each build) and
                      the plan.md of a planning round
  root/session/       prompt.txt and stderr.log; for codex the CLI's event log codex.jsonl and
                      codex_home/ for CODEX_HOME, mounted as /session in its container; for claude
                      the agent's log claude.jsonl, its stdout claude_stdout.log and system.md, a copy
                      of its system prompt, which its shell container does not see (the cost and the
                      tool calls in result.json come from that log); in rounds, rounds/NN/ holds the
                      logs and prompt of each round but the last (start_round)

Both containers also see the helper client at /opt/gator/bin and the tool server's socket folder at
/opt/gator/sock, read-only. The Codex container also sees DNS_FILES, read-only. Neither container
sees root/control; the helpers' Blender container sees only its staging/ and, read-only, its
exports/ (blender_command).
"""

import contextlib
import hashlib
import io
import json
import math
import os
import re
import shutil
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import trimesh

from harpor import frames as F
from harpor.fileio import read_json, write_json, write_text
from harpor.agent import evidence, prompts, surface, tools, validate
from harpor.agent.server import ToolServer
from harpor.parts import joints as joint_spec

BUDGET_S = 600.0
MAX_BUILDS = 3  # paper A.4: the build and at most 2 repairs; a planning round (replan) is not a build
ROUND_BUDGET_S = {"repair": 450.0, "replan": 300.0}  # a build takes the budget_s of the caller
# What a round leaves in root/session, moved to session/rounds/NN/ when the next round starts (start_round).
ROUND_FILES = ("codex.jsonl", "claude.jsonl", "stderr.log", "claude_stdout.log", "last_message.txt", "prompt.txt")
# Of result.json's rounds: no why, a finding's text. link and measured can still name links of the agent's GLB.
FINDING_KEYS = ("code", "requirement", "link", "measured", "expected")
OUTCOMES = ("pass", "fail", "inconclusive", "unassessed")  # of a requirement (paper A.3)
# What a stored candidate's rank compares, in order (rank): physics comes first, ahead of paper A.6's order.
RANK = ("round outcome pass", "surface not blocking", "required requirements passing", "source not an export",
        "version")
AGENT_DIR = Path(__file__).resolve().parent
REPO = AGENT_DIR.parents[1]  # claude_agent runs from here, as `python -m harpor.agent.claude_agent`
CODEX_ENV = {"CODEX_HOME": "/session/codex_home"}
HELPER_ENV = {"GATOR_SOCKET": "/opt/gator/sock/tools.sock", "PREPEND_PATH": "/opt/gator/bin"}
CODEX_MODEL, CLAUDE_MODEL, EFFORTS = tools.CODEX_MODEL, tools.CLAUDE_MODEL, tools.EFFORTS
CLAUDE_EFFORT = "xhigh"  # the effort of the Astra comparison runs; Opus 5.5's own default is medium, so it is always sent
LOGS = {"codex": "codex.jsonl", "claude": "claude.jsonl"}  # each agent's event log, in root/session
# --no-mount bind-paths turns off every bind path in the site's apptainer.conf: on ICE, /scratch and /storage.
CONTAINER = ["apptainer", "exec", "--nv", "--containall", "--no-mount", "bind-paths"]
DNS_FILES = ("/etc/resolv.conf", "/etc/hosts")  # only the Codex container binds these, since only it needs the network
# Shell commands that reach the network. Nothing in the container blocks them, so result.json lists
# the ones that ran.
NETWORK = re.compile(
    r"(?:^|(?<=[\s;&|(`\"']))(?:(?:pip3|pipx|pip|conda|mamba|micromamba|apt-get|apt|npm|pnpm|yarn|poetry|uv)\s+"
    r"(?:install|add|download|i)\b|(?:curl|wget|aria2c)\s+(?:-|https?://|ftp://)|git\s+(?:clone|fetch|pull|ls-remote"
    r"|submodule)\b)|urllib\.request|urlopen\(|urlretrieve\(|requests\.(?:get|post)\(|http\.client", re.M)
# The bounds of a delivered GLB's size change (check_size): a GLB whose box grew by more than the upper bound on
# every side, or shrank below the lower bound on every side, is not delivered. The prompts ask the agent to keep
# the original's numeric scale. A unit slip in Blender (2.54 for inches, or more) scales every side by the same
# factor. A repair seldom changes every side that much: in synthetic tests in the tilted frame W (view pitch up to
# 30 deg), legs under a desk or stool top, splayed stool legs and a long pan handle changed the shortest side by at
# most 1.4 times, and a window of 0.8 to 1.25 refused them. The teddy run of note 5 changed the sides by under 1.5%.
SIZE_RATIO = (0.5, 2.0)
CODEX_EXIT = {0: "success", 1: "error_or_interrupted", 130: "cancelled", 143: "terminated"}
CLAUDE_EXIT = {0: "ended", 2: "refusal", 3: "api_error", 130: "interrupted"}
KEY_NAMES = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
# Codex exec reads its key only from CODEX_API_KEY. The key file holds OPENAI_API_KEY, so a shell in the
# container copies it and then runs codex; the controller never reads the key for this.
CODEX_KEY_WRAPPER = ["sh", "-c", 'export CODEX_API_KEY="$OPENAI_API_KEY"; unset OPENAI_API_KEY; exec "$@"', "codex"]


def codex_command():
    """The codex exec command of note 5, as the paper ran GPT-6-Astra: highest reasoning effort, no
    Codex sandbox or approvals inside our container. The prompt comes on stdin ("-")."""
    return ["codex", "exec", "--dangerously-bypass-approvals-and-sandbox", "--skip-git-repo-check", "--strict-config",
            "-C", "/work", "-m", CODEX_MODEL, "-c", 'model_reasoning_effort="xhigh"', "--json",
            "-o", "/session/last_message.txt", "-"]


def _bind(src, dst, ro=False):
    src = str(src)
    if any(c in src for c in ":,"):
        raise ValueError(f"Apptainer cannot bind a path with ':' or ',': {src}")
    return ["--bind", f"{src}:{dst}" + (":ro" if ro else "")]


def _mounts(root, sock_dir, session=True):
    root = Path(root).resolve()
    return [*_bind(root / "work", "/work"), *(_bind(root / "session", "/session") if session else []),
            *_bind(AGENT_DIR / "bin", "/opt/gator/bin", ro=True), *_bind(sock_dir, "/opt/gator/sock", ro=True)]


def _env_flags(env):
    return [a for k, v in env.items() for a in ("--env", f"{k}={v}")]


def apptainer_command(root, sif, key_file, sock_dir, agent=None):
    """The codex agent's container (note 5): /work and /session writable, the helper client, the
    socket folder and DNS_FILES read-only, no /scratch or /storage, GPU for Blender. agent replaces
    the CLI command in tests."""
    return [*CONTAINER, *_mounts(root, sock_dir), *(a for p in DNS_FILES for a in _bind(p, p, ro=True)),
            "--env-file", str(key_file), *_env_flags({**CODEX_ENV, **HELPER_ENV}), "--pwd", "/work", str(sif),
            *(agent or [*CODEX_KEY_WRAPPER, *codex_command()])]


def shell_command(root, sif, sock_dir):
    """The claude agent's shell (note 5): one `bash --noprofile --norc` in a container with the mounts
    and helper variables of apptainer_command, but no /session, no DNS_FILES, no key file and no
    network (--network none; with --containall the container keeps only loopback). root/session holds
    claude_agent's log, which result.json takes the cost and the tool calls from, so the model's
    commands must not reach it. claude_agent writes the commands to its stdin."""
    return [*CONTAINER, "--net", "--network", "none", *_mounts(root, sock_dir, session=False),
            *_env_flags(HELPER_ENV), "--pwd", "/work", str(sif), "bash", "--noprofile", "--norc"]


def claude_command(root, sif, sock_dir, key_file, effort=CLAUDE_EFFORT):
    """The claude agent (note 5): harpor.agent.claude_agent with the controller's own interpreter, on
    the host. It reads the key from key_file into its memory and starts its shell with shell_command.
    Paths are absolute, because it runs from the repository and not from the work folder."""
    return [sys.executable, "-m", "harpor.agent.claude_agent", os.path.abspath(root), "--sif", os.path.abspath(sif),
            "--sock-dir", str(sock_dir), "--key-file", os.path.abspath(key_file), "--effort", effort]


def sif_blender(sif, rw=(), ro=(), aliases=()):
    """Blender in the agent's SIF, in a container that sees only the paths rw (read-write) and ro
    (read-only), each at its host path, and the Blender scripts (read-only). aliases: (host path,
    container path) pairs, bound read-write."""
    binds = [*(_bind(p, p) for p in rw), *(_bind(src, dst) for src, dst in aliases),
             *(_bind(p, p, ro=True) for p in ro), _bind(tools.BLENDER_SCRIPTS, tools.BLENDER_SCRIPTS, ro=True)]
    return [*CONTAINER, *(a for b in binds for a in b), str(sif), "blender"]


def blender_command(root, sif):
    """Blender for the host-side helpers: a separate container that sees only the work folder,
    control/staging, control/exports (read-only) and the Blender scripts. The work folder is at its
    host path, so helper arguments need no mapping, and at /work, so paths the agent saved in the .blend
    still resolve. export_glb.py writes each export into control/staging, and the host moves it into
    control/exports, where the reviews render the host's copy (tools.export_glb). Every Blender run
    writes through the links that the agent puts in work/, so only staging/ is writable: no run can
    change the host's copies. Both are at their host paths. The tool server hands this command to the
    helpers as GATOR_BLENDER."""
    work, control = (Path(root) / "work").resolve(), (Path(root) / "control").resolve()
    return sif_blender(sif, rw=[work, control / "staging"], ro=[control / "exports"], aliases=[(work, "/work")])


def cli_exit(returncode, agent="codex"):
    """Name of the agent's exit code (note 5), for agent "codex" (the Codex CLI) or "claude"."""
    if returncode is None:
        return None
    if returncode < 0:
        with contextlib.suppress(ValueError):
            return f"signal {signal.Signals(-returncode).name}"
        return f"signal {-returncode}"
    return (CLAUDE_EXIT if agent == "claude" else CODEX_EXIT).get(returncode, f"exit {returncode}")


def agent_info(root):
    """control/agent.json: agent, model and effort."""
    return json.loads((Path(root) / "control" / "agent.json").read_text())


def codex_config():
    """config.toml for CODEX_HOME."""
    return (AGENT_DIR / "codex" / "config.toml").read_text()


def agent_record(agent, effort=None):
    """control/agent.json (agent, model, effort) for prepare_session, or ValueError for a setting the
    agent does not take."""
    if agent == "codex":
        if effort not in (None, "xhigh"):
            raise ValueError("effort is a setting of the claude agent; Codex runs at xhigh (codex/config.toml)")
        return {"agent": "codex", "model": CODEX_MODEL, "effort": "xhigh"}
    if agent != "claude":
        raise ValueError(f"agent is codex or claude, not {agent!r}")
    effort = CLAUDE_EFFORT if effort is None else effort
    if effort not in EFFORTS:
        raise ValueError(f"effort is one of {', '.join(EFFORTS)}, not {effort!r}")
    return {"agent": "claude", "model": CLAUDE_MODEL, "effort": effort}


def prepare_session(root, mode, prompt, views, *, source, metric, original=None, agent="codex", effort=None,
                    parts=None):
    """Write the session folder for mode 'refinement' or 'articulation'.

    Articulation (build_plan.md note 11) works like refinement on the split parts.glb as the original,
    and parts (run/parts.json of the parts stage) goes to work/input/parts.json, which the manifest names,
    and to control/parts.json, the copy that the host's checks read (harpor.agent.validate).
    prompt: None for the mode's prompt (harpor.agent.prompts). views: dicts with K, w2c
    (OpenCV world-to-camera in W), width, height, and image and mask files. agent: "codex" (GPT-6-Astra
    through the Codex CLI, at the xhigh effort of codex/config.toml) or "claude" (Claude Opus 5.5
    through claude_agent at effort, default xhigh).
    """
    if mode not in tools.VERDICTS:
        raise ValueError(f"mode is one of {', '.join(tools.VERDICTS)}, not {mode!r}")
    if (parts is not None) != (mode == "articulation"):
        raise ValueError("parts (run/parts.json) goes with the articulation mode, and only with it")
    record = agent_record(agent, effort)
    prompt = prompts.load(mode) if prompt is None else prompt
    root, work = Path(root), Path(root) / "work"
    for d in ("input", "exports", "reviews", "checkpoints"):
        (work / d).mkdir(parents=True, exist_ok=True)
    for d in ("exports", "staging"):  # blender_command binds them, so they must exist
        (root / "control" / d).mkdir(parents=True, exist_ok=True)
    (root / "control" / "events.jsonl").touch()
    write_json(root / "control" / "agent.json", record)
    (root / "session").mkdir(exist_ok=True)
    if agent == "codex":
        (root / "session" / "codex_home").mkdir(exist_ok=True)
        (root / "session" / "codex_home" / "config.toml").write_text(codex_config())
        shutil.copyfile(AGENT_DIR / "codex" / "AGENTS.md", work / "AGENTS.md")
    else:  # claude_agent sends this file as the system prompt; the copy is the record
        shutil.copyfile(AGENT_DIR / "claude" / "SYSTEM.md", root / "session" / "system.md")
    (root / "session" / "prompt.txt").write_text(prompt, encoding="utf-8")
    entries = []
    for i, v in enumerate(views):
        e = {"index": i, "K": np.asarray(v["K"], float).tolist(), "w2c": np.asarray(v["w2c"], float).tolist(),
             "width": v["width"], "height": v["height"]}
        for key, folder in (("image", "images"), ("mask", "masks")):
            dst = work / "input" / folder / f"{i:03d}{Path(v[key]).suffix}"
            dst.parent.mkdir(exist_ok=True)
            shutil.copyfile(v[key], dst)
            e[key] = str(dst.relative_to(work))
        entries.append(e)
    m = {
        "mode": mode,
        "frame": "W, the OpenCV frame of input view 0: x right, y down, z forward",
        "glb_axes": "glTF; (x, y, z)_W = (x, -z, y)_glTF, which Blender's stock glTF importer applies",
        "w_from_gltf": F.W_FROM_GLTF.tolist(),
        "cameras": "K and OpenCV world-to-camera w2c (4x4, column vectors); pixel centers at integer coordinates",
        "source": source,
        "units": "meters" if metric else "relative: one unknown scale shared by all views",
        "views": entries,
    }
    if parts is not None:
        shutil.copyfile(parts, work / "input" / "parts.json")
        shutil.copyfile(parts, root / "control" / "parts.json")
        m["parts"] = "input/parts.json"
    write_json(work / "input" / "manifest.json", m)
    write_json(root / "control" / "manifest.json", m)
    shutil.copyfile(original, root / "original.glb")
    (work / "original" / "previews").mkdir(parents=True)
    shutil.copyfile(original, work / "original" / "original.glb")
    return work


def _finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def _capped(s, cap):
    """Seconds clipped to [0, cap]. A non-number counts as 0, so a bad events line cannot stop the clock."""
    return min(max(float(s), 0.0), cap) if _finite(s) else 0.0


def deadline(t0, budget_s, events, now):
    """The controller's own deadline: t0 + budget + the time of the helpers that run off the clock
    (tools.OFF_CLOCK: export_glb, and simulate's check after its export), each run capped at its kind's
    timeout and the total at tools.MAX_EXCLUDED_S.

    The tool server runs one request at a time, so only the newest run without its _done event can be
    running; it counts up to now, so the clock pauses during it. events: this round's (tools.round_events).
    run_budget.json is never read here. Returns (deadline, excluded_s).
    """
    running, excluded, last = {}, 0.0, None
    for e in events:
        for kind, cap in tools.OFF_CLOCK.items():
            key = (kind, repr(e.get(kind)))
            if e.get("event") == f"{kind}_started":
                last = key
                running[key] = e["time"] if _finite(e.get("time")) else now
            elif e.get("event") == f"{kind}_done":
                running.pop(key, None)
                excluded += _capped(e.get("seconds"), cap)
    if last in running:
        excluded += _capped(now - running[last], tools.OFF_CLOCK[last[0]])
    excluded = min(excluded, tools.MAX_EXCLUDED_S)
    return t0 + budget_s + excluded, excluded


def _signal_group(proc, sig):
    with contextlib.suppress(ProcessLookupError, PermissionError):  # macOS refuses a group of finished processes
        os.killpg(proc.pid, sig)


def _stop(proc, kill_after_s):
    _signal_group(proc, signal.SIGINT)
    try:
        proc.wait(kill_after_s)
        return "sigint"
    except subprocess.TimeoutExpired:
        _signal_group(proc, signal.SIGKILL)
        proc.wait()
        return "killed"


def run_agent(root, cmd, env=None, budget_s=BUDGET_S, poll_s=0.5, kill_after_s=5.0, finish_wait_s=3.0):
    """Run the agent in its own process group with a watchdog.

    Codex runs in the work folder with session/prompt.txt on stdin and its events going to
    session/codex.jsonl. claude_agent runs from the repository; it reads the prompt and writes
    session/claude.jsonl itself, and its stdout goes to session/claude_stdout.log. Its shell
    container stays in its process group.

    SIGINT goes to the group at the deadline, or finish_wait_s after the finish helper ran, and SIGKILL
    follows kill_after_s later. Under Apptainer the container has its own PID namespace, so the group
    signal also ends processes the agent started in a new session. The deadline moves by the export
    and simulate time that the helpers record in root/control (deadline). Both read only this round's
    events (tools.round_events), so an earlier round's finish does not end this one. The group is
    always killed on the way out, also when the watchdog itself fails.
    """
    root = Path(root)
    work, session, control = root / "work", root / "session", root / "control"
    claude = agent_info(root)["agent"] == "claude"
    t0 = time.time()
    tools.start_budget(work, control, t0, budget_s)
    ended, finished_at = "exited", None
    with open(os.devnull if claude else session / "prompt.txt", "rb") as stdin, \
            open(session / ("claude_stdout.log" if claude else LOGS["codex"]), "wb") as out, \
            open(session / "stderr.log", "wb") as err:
        proc = subprocess.Popen(cmd, cwd=REPO if claude else work, stdin=stdin, stdout=out, stderr=err, env=env,
                                start_new_session=True)
        try:
            while proc.poll() is None:
                now = time.time()
                events = tools.round_events(control)
                if finished_at is None and any(e.get("event") == "finish" for e in events):
                    finished_at = now
                if finished_at is not None and now - finished_at >= finish_wait_s:
                    ended = "finish_" + _stop(proc, kill_after_s)
                    break
                if now >= deadline(t0, budget_s, events, now)[0]:
                    ended = "deadline_" + _stop(proc, kill_after_s)
                    break
                time.sleep(poll_s)
        finally:
            _signal_group(proc, signal.SIGKILL)
            proc.wait()
    t1 = time.time()
    end, excluded = deadline(t0, budget_s, tools.round_events(control), t1)
    return {"exit": ended, "returncode": proc.returncode, "started_at": t0, "ended_at": t1, "wall_s": t1 - t0,
            "budget_s": budget_s, "excluded_s": excluded, "deadline_s": end - t0}


def check_glb(path):
    """Validity per paper A.4: parses, re-imports, finite vertices, materials. No quality filter.

    Vertices must be finite after the node transforms too, because the pose is baked into the asset.
    Its nodes must form trees, as glTF requires (harpor.parts.joints.parents), and no image may be too large to
    decode (harpor.agent.surface.large_image). A GLB with joints (harpor.parts.joints) must also hold a valid
    joint tree; one without is not checked for them. The file is read once, without following a symlink.
    Returns None for a valid GLB, else the reason.
    """
    try:
        data = tools.read_file(path)
        magic, version, _, n, kind = struct.unpack_from("<4sIII4s", data)
        if magic != b"glTF" or version != 2 or kind != b"JSON":
            return "unparsable: not a binary glTF 2.0 file"
        gltf = json.loads(data[20:20 + n])
        prims = [p for m in gltf.get("meshes", []) for p in m.get("primitives", [])]
        unmaterialed = [p for p in prims if p.get("material") not in range(len(gltf.get("materials", [])))]
    except Exception as e:
        return f"unparsable: {e!r}"
    if not prims:
        return "no meshes"
    if unmaterialed:
        return "no materials: a primitive has no material"
    try:
        scene = trimesh.load(io.BytesIO(data), file_type="glb", force="scene", process=False)
    except Exception as e:
        return f"re-import failed: {e}"
    if not scene.geometry:
        return "re-import has no meshes"
    if not all(np.isfinite(m.vertices).all() for m in scene.geometry.values()) or \
            not all(np.isfinite(scene.graph[n][0]).all() for n in scene.graph.nodes_geometry):
        return "non-finite vertices"
    why = surface.large_image(scene.geometry.values())
    if why:
        return f"image too large: {why}"
    try:
        joint_spec.parents(gltf)
    except Exception as e:
        return f"bad nodes: {e}"
    try:
        joints = joint_spec.read_joints(gltf)
        problems = joint_spec.validate(joints, joint_spec.mesh_nodes(gltf)) if joints else []
    except Exception as e:
        return f"bad joints: {e}"
    if problems:
        return "bad joints: " + "; ".join(problems[:3])
    return None


def box_extents(path):
    """The sides of a GLB's bounding box after its node transforms, in the GLB's own axes and units. The
    file is read without following a symlink."""
    scene = trimesh.load(io.BytesIO(tools.read_file(path)), file_type="glb", force="scene", process=False)
    return np.asarray(scene.extents, dtype=float)


def check_size(path, original):
    """None when the GLB keeps the original's size, else the reason. It does not when every side of its box
    is more than SIZE_RATIO's upper bound times the original's side, or every side is less than the lower
    bound times it (see SIZE_RATIO: a repair leaves at least one side as it was).

    original is the GLB the agent started from (root/original.glb, a copy of posed.glb or of the split
    parts.glb). Both are in frame W with glTF axes and in the same units, so their sides compare one to one.
    A side under 1% of the original's longest is left out: a flat original's thin side says nothing.
    """
    try:
        new, old = box_extents(path), box_extents(original)
        keep = old > 0.01 * old.max()
        ratio = new[keep] / old[keep]
    except Exception as e:
        return f"size not checked: {e!r}"
    lo, hi = SIZE_RATIO
    if not len(ratio) or not np.isfinite(ratio).all():  # check_glb gives the reason of a GLB that is not finite
        return "size not checked: its box or the original's has no finite size"
    if ratio.min() > hi:
        return f"wrong size: every side of its box is at least {ratio.min():.3g} times the original's (over {hi})"
    if ratio.max() < lo:
        return f"wrong size: every side of its box is at most {ratio.max():.3g} times the original's (under {lo})"
    return None


def joint_count(glb):
    """Number of joints in a GLB (harpor.parts.joints); None without a GLB or when they cannot be read."""
    try:
        return len(joint_spec.load(glb)[0]) if glb else None
    except Exception:  # an unreadable GLB; check_glb names the reason
        return None


def original_pick(root):
    """The last of the fallback order: the unchanged original, root/original.glb."""
    return dict(source="fallback_original", glb=Path(root) / "original.glb", verdict="fallback_original",
                reason="no valid agent output; the unchanged original is returned", limitations=[])


def select_deliverable(root, mode):
    """Fallback order: finish GLB, newest valid checkpoint, newest valid export, then the unchanged
    original. Returns (pick, rejected).

    Candidates come from the helpers' host-side events of this round (tools.round_events), and each GLB
    must still have the sha256 logged there: for an export, that of the bytes Blender wrote (tools.export_glb).
    Files that the agent wrote into work/ by other means are never delivered. Each GLB must also pass check_glb
    and keep the size of root/original.glb (check_size). The agent's processes have ended by now (run_agent), so
    those checks and store_candidate read the bytes whose sha256 was compared.
    """
    root = Path(root)
    work = root / "work"
    events = tools.round_events(root / "control")
    rejected = []

    def valid(e, label, glb=None):
        glb = glb or work / e["glb"]
        why = None
        if e.get("sha256") is not None:
            try:
                why = None if tools.sha256(glb) == e["sha256"] else f"changed after {label}"
            except OSError:
                why = f"missing or not a regular file after {label}"
        why = why or check_glb(glb) or check_size(glb, root / "original.glb")
        if why:
            rejected.append({"from": label, "glb": os.path.relpath(glb, root), "why": why})
        return why is None

    finishes = [e for e in events if e.get("event") == "finish"]
    if finishes:
        fin = finishes[-1]
        pick = dict(source="finish", verdict=fin.get("verdict"), reason=fin.get("reason"),
                    limitations=fin.get("limitations", []))
        if pick["verdict"] not in tools.VERDICTS[mode]:
            rejected.append({"from": "finish", "glb": None, "why": f"unknown verdict {pick['verdict']!r} for {mode}"})
        elif pick["verdict"] == "unchanged":
            return dict(pick, glb=root / "original.glb"), rejected
        elif valid(fin, "finish"):
            return dict(pick, glb=work / fin["glb"]), rejected
    for e in reversed([e for e in events if e.get("event") == "checkpoint"]):
        if valid(e, "checkpoint"):
            return dict(source="checkpoint", glb=work / e["glb"], verdict="needs_review", reason=e.get("defect"),
                        limitations=[]), rejected
    for e in reversed([e for e in events if e.get("event") == "export_done" and e.get("ok")]):
        if valid({**e, "sha256": e.get("glb_sha256")}, "export"):
            return dict(source="export", glb=work / e["glb"], verdict="needs_review", limitations=[],
                        reason="uninspected export recovered; the session ended without finish"), rejected
    return original_pick(root), rejected


def _claude_bash_calls(e):
    """(id, command) of the bash calls that claude_agent ran from one response line: none unless the
    turn stopped for its tool calls (stop_reason tool_use), and none before the last fallback block
    (the declined model's calls; see claude_agent.echo_content)."""
    if e.get("stop_reason") != "tool_use":
        return []
    blocks = [b for b in e.get("blocks") or [] if isinstance(b, dict)]
    last = max((i for i, b in enumerate(blocks) if b.get("type") == "fallback"), default=0)
    return [(b.get("id"), (b.get("input") or {}).get("command")) for b in blocks[last:]
            if b.get("type") == "tool_use" and b.get("name") == "bash"]


def shell_commands(log):
    """Shell commands in the agent's JSONL log, each counted once: Codex --json command_execution
    items (logged when they start and when they complete), or the bash calls that claude_agent ran
    from its response lines."""
    cmds, seen = [], set()
    for e in tools.read_jsonl(log):
        with contextlib.suppress(AttributeError, TypeError):
            if e.get("type") == "response":
                calls = _claude_bash_calls(e)
            else:
                item = e.get("item") or {}
                calls = [(item.get("id"), item.get("command"))] if item.get("type") == "command_execution" else []
            for call_id, cmd in calls:
                if call_id in seen:
                    continue
                seen.add(call_id)
                if isinstance(cmd, list):
                    cmd = " ".join(map(str, cmd))
                if isinstance(cmd, str):
                    cmds.append(cmd)
    return cmds


def network_commands(*logs, limit=5):
    """Shell commands in the agent's logs, one per round, that match NETWORK (at most limit of all of them).
    Each log is read alone, since shell_commands counts a command once by its id, and a new agent may use
    the same ids again."""
    return [c[:200] for log in logs for c in shell_commands(log) if NETWORK.search(c)][:limit]


def _number(x):
    return x if type(x) in (int, float) and math.isfinite(x) else 0


def claude_usage(*logs):
    """turns, tokens, cost_usd and end_reason of claude_agent's logs, one per round: the sums of each log's
    (log_usage), with the end_reason of the last."""
    parts = [log_usage(log) for log in logs]
    return {"turns": sum(_number(p["turns"]) for p in parts),
            "tokens": {k: sum(_number((p["tokens"] if isinstance(p["tokens"], dict) else {}).get(k)) for p in parts)
                       for k in tools.TOKENS},
            "cost_usd": round(sum(_number(p["cost_usd"]) for p in parts), 4), "end_reason": parts[-1]["end_reason"]}


def log_usage(log):
    """turns, tokens (the four usage totals), cost_usd and end_reason of a claude_agent log, from its
    "end" line. Without one (the agent was killed first) they are summed over its response lines:
    their billed tokens (every attempt of a turn with a refusal fallback, else usage) and their own
    cost_usd (each attempt at its model's prices)."""
    totals, turns, cost = dict.fromkeys(tools.TOKENS, 0), 0, 0.0
    for e in tools.read_jsonl(log):
        if e.get("type") == "end":
            return {"turns": e.get("turns"), "tokens": e.get("totals"), "cost_usd": e.get("cost_usd"),
                    "end_reason": e.get("reason")}
        if e.get("type") == "response":
            turns += 1
            usage = next((e[k] for k in ("billed", "usage") if isinstance(e.get(k), dict)), {})
            for k in totals:
                totals[k] += usage[k] if type(usage.get(k)) is int else 0
            line_cost = e.get("cost_usd")
            if type(line_cost) in (int, float) and math.isfinite(line_cost):
                cost += line_cost
    return {"turns": turns, "tokens": totals, "cost_usd": round(cost, 4), "end_reason": None}


def make_previews(root, blender, timeout=300):
    """Render the original under every input camera and 6 orbit views into work/original/previews,
    before the clock starts. Prompt 1 asks the agent to inspect them. Returns True if the sheet was
    written."""
    root = Path(root).resolve()
    out = root / "work" / "original" / "previews"
    m = json.loads((root / "control" / "manifest.json").read_text())
    # In work/: the helpers' Blender container sees the work folder and, of control/, only staging/ and exports/.
    args = out / "args.json"
    write_json(args, {"root": str(root / "work"), "out": str(out), "clay": False, "views": m["views"],
                      "candidate": None, "original": str(out.parent / "original.glb"), "detail": []})
    with open(root / "control" / "previews.log", "w") as log:
        ok = tools.run_ok(tools.blender_argv(blender, "review.py", args), log, timeout)
    args.unlink()
    return ok and (out / "sheet.png").is_file()


def build_candidate(root, blender, timeout=300):
    """Write work/candidate.blend from the original with the stock glTF importer, before the clock
    starts. Returns True if the file was written."""
    work, control = Path(root).resolve() / "work", Path(root).resolve() / "control"
    cmd = tools.blender_argv(blender, "make_candidate.py", work / "original" / "original.glb", work / "candidate.blend")
    with open(control / "candidate.log", "w") as log:
        ok = tools.run_ok(cmd, log, timeout)
    return ok and (work / "candidate.blend").is_file()


def has_rounds(root, mode):
    """Rounds for sessions with physics supervision, or an articulation with a found requirement."""
    from harpor.agent.physics_review import enabled

    return enabled(Path(root) / "control") or (mode == "articulation" and any(
        r.get("status") == "found" for r in validate.requirements(Path(root) / "control")))


def next_round(rounds):
    """The next round's kind, or None. At most 3 builds and 1 replan; a replan is always followed by a
    build and does not count as one (paper A.3, A.4). scope "none" means no agent action can help."""
    last = rounds[-1]
    if last["kind"] == "replan":
        return "repair"
    if last["outcome"] == "pass" or last["scope"] == "none":
        return None
    if sum(r["kind"] != "replan" for r in rounds) >= MAX_BUILDS:
        return None
    if last["scope"] == "structural" and all(r["kind"] != "replan" for r in rounds):
        return "replan"
    return "repair"


def start_round(root, n, prompt):
    """Begin round n > 1: move round n - 1's logs and prompt (ROUND_FILES) from session/ into
    session/rounds/NN/, then write session/prompt.txt, and for codex work/AGENTS.md and
    codex_home/config.toml again, through fileio.write_text.

    The agent may have put a link at any of these names. os.replace moves a link, not the file it points
    to, and write_text replaces one; session/rounds and codex_home must be real folders (tools.real_dir).
    Each codex exec reads these files again, so an edit of an earlier round does not reach this one."""
    root = Path(root)
    session = root / "session"
    done = tools.real_dir(tools.real_dir(session, "rounds"), f"{n - 1:02d}")
    for name in ROUND_FILES:
        with contextlib.suppress(FileNotFoundError):
            os.replace(session / name, done / name)
    write_text(session / "prompt.txt", prompt)
    if agent_info(root)["agent"] == "codex":
        write_text(tools.real_dir(session, "codex_home") / "config.toml", codex_config())
        write_text(root / "work" / "AGENTS.md", (AGENT_DIR / "codex" / "AGENTS.md").read_text(encoding="utf-8"))


def run_round(root, cmd, env, blender, sock_dir, n, kind, prompt, watchdog):
    """Run round n, of kind build, repair or replan, and return run_agent's record. Round 1 runs in the
    prepared session; a later one begins with start_round and prompt. The round_start event makes every
    round-scoped reader (tools.round_events) start here. The round has a tool server of its own, since
    stop() is final. A build takes watchdog's budget_s, a repair or replan ROUND_BUDGET_S."""
    root = Path(root)
    if n > 1:
        start_round(root, n, prompt)
    tools.log_event(root / "control", "round_start", round=n, kind=kind)
    budget_s = watchdog.get("budget_s", BUDGET_S) if kind == "build" else ROUND_BUDGET_S[kind]
    server = ToolServer(Path(sock_dir) / "tools.sock", root / "work", root / "control", blender).start()
    try:
        return run_agent(root, cmd, env, **{**watchdog, "budget_s": budget_s})
    finally:
        server.stop()


def _host_step(errors, fn, *args, **kw):
    """fn(*args, **kw), or None after its failure went to errors. A step after a round reads files that the
    agent made, so a string SystemExit (tools.real_dir, tools.glb_joints) and any Exception are caught; an
    integer SystemExit, such as Cancel's sys.exit(143) (tools._on_sigterm), goes on."""
    try:
        return fn(*args, **kw)
    except SystemExit as e:
        if not isinstance(e.code, str):
            raise
        errors.append(e.code)
    except Exception as e:  # noqa: BLE001 - one failed check must not stop the session
        errors.append(f"{type(e).__name__}: {e}")
    return None


def store_candidate(root, n, kind, pick):
    """Keep the pick of build or repair round n as the next version of control/candidates, since work/ stays the
    agent's and earlier candidates must stay available when a revision fails (paper A.3): vNN/candidate.glb, a
    read-only copy, and vNN/report.json, the record {version, round, kind, glb, from (where the round left it),
    sha256, source, agent_verdict, reason, limitations, report}, whose report (the round's) check_round adds. The
    GLB is read once, and every later step reads the copy. Each round gets its own version, also with the bytes of
    an earlier one, so that each keeps its own verdict. Returns the record, or None for fallback_original and
    unchanged, which are no agent's GLB."""
    root = Path(root)
    mode = (read_json(root / "control/manifest.json") or {}).get("mode", "articulation")
    if pick["source"] == "fallback_original" or (pick["verdict"] == "unchanged" and mode != "refinement"):
        return None
    data = tools.read_file(pick["glb"])
    folder = root / "control" / "candidates"
    folder.mkdir(exist_ok=True)
    version = len(list(folder.glob("v[0-9][0-9]"))) + 1
    out = folder / f"v{version:02d}"
    out.mkdir()
    with tools.open_new(out / "candidate.glb", "wb") as f:
        f.write(data)
        os.fchmod(f.fileno(), 0o444)
    rec = {"version": version, "round": n, "kind": kind, "glb": os.path.relpath(out / "candidate.glb", root),
           "from": os.path.relpath(pick["glb"], root), "sha256": hashlib.sha256(data).hexdigest(),
           "source": pick["source"], "agent_verdict": pick["verdict"], "reason": pick.get("reason"),
           "limitations": list(pick.get("limitations") or []), "report": None}
    write_json(out / "report.json", rec)
    return rec


def _written(errors, path, data):
    """Whether write_json(path, data) went through; its failure goes to errors (_host_step)."""
    before = len(errors)
    _host_step(errors, write_json, path, data)
    return len(errors) == before


def check_round(root, n, kind):
    """The end of build or repair round n: the round's pick (select_deliverable on its events), stored as a
    candidate (store_candidate), the host's check of the stored copy (validate.check, which pins the references of
    control/ranges.json) and validate.round_report of them. An unchanged pick is not stored, and its check is of
    root/original.glb, whose joint_missing tells the next agent what is open. A copy of the report goes to
    work/evidence/round_NN.json for the next round's agent, and then into the candidate's record. A step that fails
    is a validation_error of the round (_host_step). Returns {pick, rejected, version, sha256, report}; pick is None
    when select_deliverable failed, and version when nothing was stored."""
    root = Path(root)
    control, errors, report = root / "control", [], None
    mode = (read_json(control / "manifest.json") or {}).get("mode", "articulation")
    pick, rejected = _host_step(errors, select_deliverable, root, mode) or (None, [])
    cand = pick and _host_step(errors, store_candidate, root, n, kind, pick)
    glb = root / cand["glb"] if cand else None
    if pick and pick["verdict"] == "unchanged":  # root/original.glb, the host's own copy
        glb = pick["glb"]
    if glb:
        report = _host_step(errors, validate.check, glb, control, root / "original.glb", pin_ranges=True)
        from harpor.agent import physics_review

        if report is not None and physics_review.enabled(control):
            judgment = _host_step(errors, physics_review.check, root, glb, report)
            if judgment is not None:
                report = physics_review.apply(report, judgment)
    folder = _host_step(errors, tools.real_dir, root / "work", "evidence")
    final = validate.round_report(control, report, pick, errors)
    if folder is not None and not _written(errors, folder / f"round_{n:02d}.json", final):
        final = validate.round_report(control, report, pick, errors)  # the record names the copy that failed
    if cand and not _written(errors, (root / cand["glb"]).with_name("report.json"), {**cand, "report": final}):
        final = validate.round_report(control, report, pick, errors)  # the candidate stays unchecked, its round fails
    return {"pick": pick, "rejected": rejected, "version": cand and cand["version"],
            "sha256": cand["sha256"] if cand else final["sha256"], "report": final}


def round_record(root, n, kind, run, end=None):
    """A round of result.json: round, kind, budget_s, exit and wall_s, and for a build or repair (end:
    check_round's) its candidate {version (in control/candidates, None when nothing was stored), source, verdict,
    glb (where the round left it), sha256}, outcome, scope and findings, each with only FINDING_KEYS, so that no
    finding's why reaches result.json."""
    rec = {"round": n, "kind": kind, "budget_s": run["budget_s"], "exit": run["exit"], "wall_s": run["wall_s"],
           "candidate": None, "outcome": None, "scope": None, "findings": []}
    if end:
        pick, report = end["pick"], end["report"]
        if pick:
            rec["candidate"] = {"version": end["version"], "source": pick["source"], "verdict": pick["verdict"],
                                "glb": os.path.relpath(pick["glb"], root), "sha256": end["sha256"]}
        rec.update(outcome=report["outcome"], scope=report["scope"],
                   findings=[{k: f.get(k) for k in FINDING_KEYS} for f in report["findings"]])
    return rec


def stored_candidates(root):
    """The records of control/candidates (store_candidate), oldest first."""
    recs = [read_json(d / "report.json") for d in sorted((Path(root) / "control" / "candidates").glob("v[0-9][0-9]"))]
    return [r for r in recs if isinstance(r, dict)]


def _records(root, cand):
    """The records of a stored candidate's report; every requirement unassessed when its check did not end."""
    return ((cand.get("report") or {}).get("records") or
            [validate.record(r) for r in validate.requirements(Path(root) / "control")])


def passing(cand):
    """The required found requirements that pass in a stored candidate's report with a trajectory of at least
    validate.MIN_STATES states (paper A.6), none when the report is of other bytes than the candidate's."""
    report = cand.get("report") or {}
    if report.get("sha256") != cand["sha256"]:
        return 0
    return sum(r["required"] and bool(r["target"]) and r["outcome"] == "pass"
               and r["measured"].get("states", 0) >= validate.MIN_STATES for r in report["records"])


def rank(cand):
    """A stored candidate's rank, highest first (RANK). A passing round always wins, which puts physics ahead of
    paper A.6's order on purpose: that order would let a reviewed failure beat a later pass."""
    report = cand.get("report") or {}
    return (report.get("sha256") == cand["sha256"] and report.get("outcome") == "pass",
            not (report.get("surface") or {}).get("blocks"), passing(cand), cand["source"] != "export", cand["version"])


def eligibility(root, cand):
    """None when a stored candidate may be delivered (paper A.6), else why not: its copy must still have its sha256,
    pass check_glb and keep the original's size (check_size), and the surface check must not find most of the
    original gone (surface_identity, which counts only while surface.ENFORCE is on, as every surface finding). A
    surface that only blocks does not make a candidate ineligible: it ranks lower (rank)."""
    root = Path(root)
    glb = root / cand["glb"]
    try:
        if tools.sha256(glb) != cand["sha256"]:
            return "changed after it was stored"
    except OSError:
        return "missing or not a regular file"
    why = check_glb(glb) or check_size(glb, root / "original.glb")
    if why:
        return why
    lost = [f for f in (cand.get("report") or {}).get("findings") or []
            if f["code"] == "surface_identity" and validate.counts(f)]
    return f"surface_identity: {lost[0]['why']}" if lost else None


def ranking(root):
    """(the eligible stored candidates, best first by rank; [{from, glb, why}] of the others, as rejected)."""
    eligible, others = [], []
    for c in stored_candidates(root):
        why = eligibility(root, c)
        if why:
            others.append({"from": f"candidate v{c['version']:02d}", "glb": c["glb"], "why": why})
        else:
            eligible.append(c)
    return sorted(eligible, key=rank, reverse=True), others


def best_candidate(root):
    """The best stored candidate so far (ranking) where the agent can reach it, for prompts.round_prompt: {glb (where
    its round left it, relative to work/), round, passing, required}, or None."""
    root = Path(root)
    ranked, _ = ranking(root)
    if not ranked:
        return None
    best = ranked[0]
    return {"glb": os.path.relpath(root / best["from"], root / "work"), "round": best["round"],
            "passing": passing(best), "required": sum(r["required"] and bool(r["target"]) for r in _records(root, best))}


def why_first(ranked):
    """Why the first of ranked won: the first element of RANK in which it beats the second."""
    if len(ranked) == 1:
        return "the only eligible candidate"
    a, b = rank(ranked[0]), rank(ranked[1])
    i = next(i for i in range(len(RANK)) if a[i] != b[i])  # the versions differ
    return f"first of {len(ranked)} eligible candidates; it beats v{ranked[1]['version']:02d} on {RANK[i]}"


def _finding_line(f):
    where = ", ".join(f"{k} {f[k]}" for k in ("requirement", "link") if f.get(k))
    return f"{f['code']}: {where}" if where else f["code"]


def review_note(root, cand):
    """The reason and limitations of a stored candidate delivered as needs_review, written by the host: the reason
    names the required requirements that do not pass, with their outcome, and the codes of the findings that count;
    the limitations are the first of those findings (validate.ranked, at most tools.MAX_LIMITATIONS), by code,
    requirement and link. Neither holds the agent's text."""
    report = cand.get("report") or {}
    unmet = [f"{r['requirement']} {r['outcome']}" for r in _records(root, cand)
             if r["required"] and r["target"] and r["outcome"] != "pass"]
    found = [f for f in validate.ranked(report.get("findings") or []) if validate.counts(f)]
    codes = list(dict.fromkeys(f["code"] for f in found))
    reason = (f"the host's check of candidate v{cand['version']:02d} (round {cand['round']}) did not pass: "
              + (f"requirements not met: {', '.join(unmet)}" if unmet else "every required requirement passes")
              + (f"; open: {', '.join(codes)}" if codes else ""))
    return {"reason": reason, "limitations": list(dict.fromkeys(map(_finding_line, found)))[:tools.MAX_LIMITATIONS]}


def select_candidate(root, last=None):
    """The deliverable of a session with rounds (paper A.6): the best eligible stored candidate (ranking). Its
    verdict is articulated only when its round passed, which needs the agent's articulated too, and it keeps the
    agent's reason and limitations; else it is needs_review with the host's (review_note). source keeps the
    candidate's value and glb is its copy in control/candidates, so harpor.webapp.steps.agent_glb finds it in the
    session folder as before. Without an eligible candidate it is the unchanged original as fallback_original, the
    paper's static fallback, unless last, the pick of the last build or repair, has the verdict unchanged, which is
    kept.

    Returns (pick, chosen, selected): pick as select_deliverable's, chosen the stored candidate's record or None,
    and selected the choice for control/evidence.json: {version, verdict, why, rank [{version, rank}] best first,
    ineligible (ranking's others), unresolved (the requirements that do not pass)}."""
    root = Path(root)
    ranked, others = ranking(root)
    order = [{"version": c["version"], "rank": list(rank(c))} for c in ranked]
    if not ranked:
        pick = dict(last) if last and last["verdict"] == "unchanged" else original_pick(root)
        selected = {"version": None, "verdict": pick["verdict"], "rank": order, "ineligible": others,
                    "why": "no stored candidate is eligible" if others else "no build or repair handed off a GLB",
                    "unresolved": [r["id"] for r in validate.requirements(root / "control")]}
        return pick, None, selected
    best = ranked[0]
    pick = dict(source=best["source"], glb=root / best["glb"])
    if rank(best)[0]:
        mode = (read_json(root / "control/manifest.json") or {}).get("mode", "articulation")
        verdict = "articulated" if mode == "articulation" else best["agent_verdict"]
        pick.update(verdict=verdict, reason=best["reason"], limitations=best["limitations"])
    else:
        pick.update(verdict="needs_review", **review_note(root, best))
    selected = {"version": best["version"], "verdict": pick["verdict"], "why": why_first(ranked), "rank": order,
                "ineligible": others, "unresolved": [r["requirement"] for r in _records(root, best)
                                                     if r["outcome"] != "pass"]}
    return pick, best, selected


def candidate_summary(chosen, selected):
    """result.json's candidate: the stored candidate delivered, {version, round, kind, from, agent_verdict, outcome
    (of its round), why (select_candidate's)}, or None."""
    if chosen is None:
        return None
    return {**{k: chosen[k] for k in ("version", "round", "kind", "from", "agent_verdict")},
            "outcome": (chosen["report"] or {}).get("outcome"), "why": selected["why"]}


def run_rounds(root, cmd, env, blender, sock_dir, watchdog):
    """The rounds of a session with a found requirement (has_rounds; paper A.3, A.4): the build, then while
    next_round says so a repair or a replan, each a new agent with a prompt (prompts.round_prompt) that carries
    the feedback of the last check and the best stored candidate (best_candidate); only the repair right after a
    replan is told to follow its /work/plan.md, which a later one would take for a plan of its own check. Each
    build or repair ends with check_round, and every round with control/evidence.json (harpor.agent.evidence), so a
    session that stops keeps the evidence of the rounds that ran.

    A failure of the loop itself ends the loop but keeps the rounds that ran, since root/result.json is written
    only at the end and a retry of the job deletes the session. It is raised when no round ran, and an integer
    SystemExit such as Cancel's always is. Returns (runs, rounds, ends): run_agent's records, result.json's
    rounds and check_round's {n: end}."""
    root = Path(root)
    runs, rounds, ends, kind, prompt, last = [], [], {}, "build", None, None
    mode = (read_json(root / "control/manifest.json") or {}).get("mode", "articulation")
    try:
        while kind:
            n = len(runs) + 1
            runs.append(run_round(root, cmd, env, blender, sock_dir, n, kind, prompt, watchdog))
            if kind != "replan":
                ends[n], last = check_round(root, n, kind), n
            rounds.append(round_record(root, n, kind, runs[-1], ends.get(n)))
            evidence.write(root, rounds, stored_candidates(root))
            kind = next_round(rounds)
            if kind:
                prompt = prompts.round_prompt(kind, ROUND_BUDGET_S[kind], validate.feedback(ends[last]["report"]),
                                              best_candidate(root), f"evidence/round_{last:02d}.json",
                                              planned=rounds[-1]["kind"] == "replan", mode=mode)
                from harpor.agent.physics_review import GUIDANCE, enabled

                if enabled(root / "control"):
                    prompt += GUIDANCE
    except SystemExit as e:
        if not isinstance(e.code, str) or not runs:
            raise
        print(f"rounds stopped after round {len(runs)}: {e.code}", file=sys.stderr)
    except Exception as e:  # noqa: BLE001 - deliver what the rounds that ran made
        if not runs:
            raise
        print(f"rounds stopped after round {len(runs)}: {type(e).__name__}: {e}", file=sys.stderr)
    return runs, rounds, ends


def total_timings(runs):
    """run_agent's timings of the rounds: started_at of the first, ended_at of the last, the others summed."""
    return {"started_at": runs[0]["started_at"], "ended_at": runs[-1]["ended_at"],
            **{k: sum(r[k] for r in runs) for k in ("wall_s", "budget_s", "excluded_s", "deadline_s")}}


def round_logs(root, agent):
    """The agent's log of each round, in order: an earlier round's is in session/rounds/NN/ (start_round), the last
    one's in session/. The folders are listed rather than counted from the rounds that ran, since a round that fails
    to start has already moved the log of the round before it."""
    session, name = Path(root) / "session", LOGS[agent]
    return [*sorted(session.glob(f"rounds/[0-9][0-9]/{name}")), session / name]


def requirement_counts(root, report):
    """{pass, fail, inconclusive, unassessed}: the outcomes of the requirements in a round's report; without one
    every requirement of control/parts.json is unassessed."""
    outcomes = ([r["outcome"] for r in report["records"]] if report else
                ["unassessed"] * len(validate.requirements(Path(root) / "control")))
    return {k: outcomes.count(k) for k in OUTCOMES}


def run_session(root, cmd, env=None, *, blender, sock_dir, **watchdog):
    """Run one prepared session and write root/result.json.

    The tool server listens on sock_dir/tools.sock while the agent runs and is stopped, with any
    running helper, before the deliverable is picked. blender: the argv prefix that runs Blender for
    the helpers. watchdog: keyword arguments of run_agent. A refinement result adds surface, the
    deliverable's harpor.agent.surface report against root/original.glb, which only reports: it never
    blocks, and a comparison that fails is recorded as {"error": ...}.

    A session with rounds (has_rounds, run_rounds) delivers the best of its stored candidates
    (select_candidate), whose choice goes to control/evidence.json too. Its result adds rounds, candidate
    (candidate_summary), requirements (the outcome counts of that candidate's check, or without one of the
    last round's) and surface (that check's). rejected holds what each round's select_deliverable rejected and
    the stored candidates that are not eligible. exit, returncode and cli_exit are the last round's; timings,
    the usage of a claude agent and network_commands cover every round. An articulation result adds
    simulations, the simulate runs that gave a report.
    """
    root = Path(root)
    work = root / "work"
    mode = json.loads((root / "control" / "manifest.json").read_text())["mode"]
    info = agent_info(root)
    candidate = build_candidate(root, blender)
    previews = make_previews(root, blender)
    rounds, report = None, None
    if has_rounds(root, mode):
        runs, rounds, ends = run_rounds(root, cmd, env, blender, sock_dir, watchdog)
        end = ends[max(ends)] if ends else None
        pick, chosen, selected = select_candidate(root, end and end["pick"])
        evidence.write(root, rounds, stored_candidates(root), selected)
        report = chosen["report"] if chosen else end and end["report"]
        rejected = [*(x for n in sorted(ends) for x in ends[n]["rejected"]), *selected["ineligible"]]
    else:
        server = ToolServer(Path(sock_dir) / "tools.sock", work, root / "control", blender).start()
        try:
            runs = [run_agent(root, cmd, env, **watchdog)]
        finally:
            server.stop()
        pick, rejected = select_deliverable(root, mode)
    glb, last = pick.pop("glb"), runs[-1]
    logs, events = round_logs(root, info["agent"]), tools.read_events(root / "control")
    result = {
        "mode": mode, "agent": info["agent"], "model": info["model"], "effort": info["effort"], **pick,
        "glb": os.path.relpath(glb, root) if glb else None,
        "sha256": tools.sha256(glb) if glb else None,
        "exit": last["exit"], "returncode": last["returncode"], "cli_exit": cli_exit(last["returncode"], info["agent"]),
        "timings": total_timings(runs),
        "counts": {k: sum(e.get("event") == ev and e.get("ok", True) for e in events)
                   for k, ev in (("exports", "export_done"), ("reviews", "review"), ("checkpoints", "checkpoint"),
                                 ("joints", "joint"), ("articulation_reviews", "review_articulation"))},
        "rejected": rejected,
        "candidate_built": candidate,
        "previews_built": previews,
        "network_commands": network_commands(*logs),
    }
    if mode == "articulation":  # counts.joints counts set_joint changes; this is what the deliverable holds
        result["joints_delivered"] = joint_count(glb)
        result["simulations"] = sum(e.get("event") == "simulate_done" and bool(e.get("ok")) for e in events)
    if rounds is not None:
        result.update(rounds=rounds, candidate=candidate_summary(chosen, selected),
                      requirements=requirement_counts(root, report),
                      surface=(report or {}).get("surface") or surface.compare(glb, root / "original.glb"))
    if mode == "refinement":
        result["surface"] = surface.compare(glb, root / "original.glb")
    if info["agent"] == "claude":
        result.update(claude_usage(*logs))
    write_json(root / "result.json", result)
    return result


def run_container_session(root, sif, key_file, agent=None, **watchdog):
    """Run one prepared session with the agent that control/agent.json names (note 5). key_file: the
    env file with OPENAI_API_KEY (codex) or ANTHROPIC_API_KEY (claude). agent replaces the agent's
    command, for tests of the launcher: for codex the CLI inside the container, for claude the whole
    host command."""
    info = agent_info(root)
    sock_dir = tempfile.mkdtemp(prefix="gator-")  # short path: Unix socket paths are limited to 107 bytes
    try:
        if info["agent"] == "claude":
            cmd = agent or claude_command(root, sif, sock_dir, key_file, info["effort"])
        else:
            cmd = apptainer_command(root, sif, key_file, sock_dir, agent)
        result = run_session(root, cmd, blender=blender_command(root, sif), sock_dir=sock_dir, **watchdog)
    finally:
        shutil.rmtree(sock_dir, ignore_errors=True)
    result["key_found_in"] = find_key(root, key_file)
    write_json(Path(root) / "result.json", result)
    return result


def read_keys(key_file):
    """Every API key of an env file: the values of its OPENAI_API_KEY= and ANTHROPIC_API_KEY= lines, in
    file order, without quotes around them (as a shell and claude_agent read them). A file can hold
    both, and the Codex container gets all of them through --env-file."""
    return list(dict.fromkeys(k for k in tools.env_values(key_file, KEY_NAMES) if k))


def find_key(root, key_file, chunk=50 << 20):
    """Files under root that contain any API key from key_file (read_keys). The codex agent's shell can
    read the keys from /proc, so these files must not be shared. No key is ever printed. Files of any
    size are scanned, chunk bytes at a time (has_key)."""
    keys = [k.encode() for k in read_keys(key_file) if len(k) >= 16]
    hits = []
    if not keys:
        return hits
    for d, _, files in os.walk(root):
        for name in files:
            p = Path(d) / name
            with contextlib.suppress(OSError):
                if has_key(p, keys, chunk):
                    hits.append(os.path.relpath(p, root))
    return sorted(hits)


def has_key(path, keys, chunk=50 << 20):
    """True when the regular file at path holds any of keys (bytes). It reads chunk bytes at a time and
    keeps the last len(longest key) - 1 bytes of each for the next, so a key across two chunks is found.
    A final symlink, a FIFO or a device raises OSError, as tools.read_file does."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError(f"{path} is not a regular file")
        keep, tail = max(map(len, keys)) - 1, b""
        with os.fdopen(os.dup(fd), "rb") as f:
            while block := f.read(chunk):
                data = tail + block
                if any(k in data for k in keys):
                    return True
                tail = data[-keep:] if keep else b""
        return False
    finally:
        os.close(fd)
