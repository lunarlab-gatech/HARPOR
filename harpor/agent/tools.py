"""Helper commands export_glb, review, checkpoint and finish (build_plan.md note 5).

The agent calls them by name inside its container. The client in bin/ sends each command line to the
controller's tool server (server.py), which runs this module on the host. The host-only control
folder holds the events and the clock record. The events are the record of what the helpers did:
exports, reviews, checkpoints and finish, each with a sha256 of the GLB or the .blend. The
controller trusts them and not the agent-writable copies in work/. Each command prints about 20 lines
at most and writes details to files under work/. Every command but finish ends with the seconds left
before the deadline, also when it fails, but not after a usage error. Paths in output and events are
relative to work/, so they read the same on the host and in the container. export_glb runs outside
the time budget (OFF_CLOCK).

A session with physics supervision runs in rounds (controller.run_rounds), each with a new agent. The
round_start event begins one, and the deadline, the finish that ends a round and its deliverable come
from the events since then (round_events). In a planning round (kind replan) every helper is refused.

These helpers run on the host but act on a folder the agent controls. So they read and write work/
without following symlinks: folders must be real folders, new files are created exclusively, and
reads take regular files only. Files written after a wait, during an export or a render, go through a
descriptor of their folder (folder, new_folder), since the agent can swap the folder for a link meanwhile.

Blender writes each export and its report into control/staging/. The host reads and hashes them, moves them into
control/exports/, which the agent's containers never see and the helpers' Blender sees read-only, and only then
gives the agent its copies in work/exports/ (export_glb). review reads the host's copy (host_copy); checkpoint,
finish and the controller check the agent's copy against the logged sha256. Agent text that the host stores, the
reason and limitations of finish and the defect of checkpoint, is kept as one line of printable characters
(printable).

Environment: GATOR_WORK, GATOR_CONTROL (default: work/../control; both are resolved) and GATOR_BLENDER
(default blender, split like a shell).
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
import tempfile
import time
from pathlib import Path

from harpor.fileio import write_json

BLENDER_SCRIPTS = Path(__file__).resolve().parent / "blender"
# The agents' models, here because this module needs only the standard library: claude_agent, the
# controller, harpor.agent.vlm and the web app's key check all read them. The claude agent runs
# CLAUDE_MODEL (Claude Opus 5.5) by default, or FABLE_MODEL (Claude Fable 5.2).
CODEX_MODEL, CLAUDE_MODEL, FABLE_MODEL = "gpt-6-astra", "claude-opus-5-5", "claude-fable-5-2"
CLAUDE_MODELS = (CLAUDE_MODEL, FABLE_MODEL)  # the models that the claude agent takes
# The web app's agents (steps.AGENTS, without "none") and the model of each one. The same model also
# judges the physics (physics_review).
MODELS = {"astra": CODEX_MODEL, "opus": CLAUDE_MODEL, "fable": FABLE_MODEL}
EFFORTS = ("low", "medium", "high", "xhigh", "max")  # Claude's effort levels
FALLBACK_BETA = "server-side-fallback-2026-07-01"  # Claude's server-side refusal fallback
TOKENS = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
# USD per million tokens of the models that can serve a turn: the claude agent's models and the refusal
# fallbacks that "default" picks for them (Claude Opus 5, Claude Opus 4.8). cache_creation_input_tokens
# is the 5-minute write (1.25x input), cache_creation_1h the 1-hour write (2x input). A model not listed
# is priced as the dearest one.
PRICES = {
    CLAUDE_MODEL: {"input_tokens": 4.0, "output_tokens": 20.0, "cache_read_input_tokens": 0.20,
                   "cache_creation_input_tokens": 5.0, "cache_creation_1h": 8.0},
    FABLE_MODEL: {"input_tokens": 8.0, "output_tokens": 40.0, "cache_read_input_tokens": 0.20,
                  "cache_creation_input_tokens": 10.0, "cache_creation_1h": 16.0},
    "claude-opus-5": {"input_tokens": 5.0, "output_tokens": 25.0, "cache_read_input_tokens": 0.50,
                      "cache_creation_input_tokens": 6.25, "cache_creation_1h": 10.0},
    "claude-opus-4-8": {"input_tokens": 5.0, "output_tokens": 25.0, "cache_read_input_tokens": 0.50,
                        "cache_creation_input_tokens": 6.25, "cache_creation_1h": 10.0},
}
EXPORT_TIMEOUT_S = 180.0
OFF_CLOCK = {"export": EXPORT_TIMEOUT_S}  # {kind: cap of one run} of the helpers whose <kind>_started and <kind>_done
# events move the deadline
MAX_EXCLUDED_S = 600.0  # cap on the export time that moves the deadline
REVIEW_TIMEOUT_S = 300.0  # render only
VERDICTS = ("complete", "unchanged", "needs_review")
MAX_LIMITATIONS = 3
MAX_DETAIL_VIEWS = 4
MAX_READ_BYTES = 2 << 30
RECEIPT = re.compile(r"r\d{3}-[0-9a-f]{12}")  # a review's receipt
DIRECTORY = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW  # a real folder, not a link to one
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


def open_new(path, mode="w", dir_fd=None):
    """A new file for writing. Fails if anything, a symlink included, is already at path. dir_fd: the descriptor
    of the folder that a relative path is in."""
    return os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644, dir_fd=dir_fd), mode)


def real_dir(work, name):
    """work/name as a real folder, created if missing. Exits if it is a symlink or a file."""
    d = Path(work) / name
    with contextlib.suppress(FileExistsError):
        d.mkdir()
    if not stat.S_ISDIR(os.lstat(d).st_mode):
        sys.exit(f"{name}/ must be a folder, not a link or a file")
    return d


@contextlib.contextmanager
def folder(path, dir_fd=None):
    """A descriptor of the real folder at path, for the files that a helper writes there after a wait (open_new and
    put with it as dir_fd). It is opened without following a link, so a link that the agent swapped in for the
    folder meanwhile raises OSError and cannot send those files out of work/. dir_fd: as open_new's."""
    fd = os.open(path, DIRECTORY, dir_fd=dir_fd)
    try:
        yield fd
    finally:
        os.close(fd)


@contextlib.contextmanager
def new_folder(work, name, sub, command):
    """A descriptor of the new folder work/name/sub, made and opened through a descriptor of work/name (folder). The
    files that a helper writes through it stay in work/, whatever links the agent swaps in for either folder, also
    while Blender renders there. Exits when work/name is not a real folder or sub is there already."""
    try:
        with folder(real_dir(work, name)) as d:
            os.mkdir(sub, dir_fd=d)
            fd = os.open(sub, DIRECTORY, dir_fd=d)
    except FileExistsError:
        sys.exit(f"{name}/{sub} already exists; remove it and run {command} again")
    except OSError:
        sys.exit(f"{name}/ must be a folder, not a link or a file")
    try:
        yield fd
    finally:
        os.close(fd)


def put(fd, name, data):
    """Write data (bytes) as name in the folder of descriptor fd: under a new name (open_new), then moved over name
    with os.replace, which swaps the name itself, so a file or link that the agent put there is replaced, never
    written through."""
    tmp = f".{name}.{os.urandom(6).hex()}"
    with open_new(tmp, "wb", dir_fd=fd) as f:
        f.write(data)
    os.replace(tmp, name, src_dir_fd=fd, dst_dir_fd=fd)


def put_json(fd, name, obj):
    """obj as JSON through put, as harpor.fileio.write_json writes it."""
    put(fd, name, json.dumps(obj, indent=2).encode())


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


@contextlib.contextmanager
def host_file(control, data):
    """The path of a temporary GLB in control/ that holds data, for a check that reads a path: it reads the bytes
    that were hashed, which the agent cannot change."""
    with tempfile.NamedTemporaryFile(dir=control, suffix=".glb") as f:
        f.write(data)
        f.flush()
        yield f.name


def printable(text):
    """text as one line of printable characters, for agent text that the host stores or prints. Tabs, line breaks
    and other whitespace become one space, and every character that str.isprintable refuses is dropped: C0 and C1
    controls and DEL, so ESC and with it every terminal sequence (OSC 52, OSC 0, CSI), format characters such as
    bidi overrides (Cf), surrogates, private-use and unassigned code points. Letters of any script stay. No cap: the
    tool server already caps each argument at 4096 characters (server.MAX_ARG_CHARS)."""
    kept = "".join(c for c in str(text) if c.isspace() or c.isprintable())
    return " ".join(kept.split())


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
    """The JSON objects of a JSONL file, read as read_file reads: the Codex agent can write session/, so a link, a
    FIFO or a device there gives nothing, as do other lines and a missing or unreadable file."""
    with contextlib.suppress(OSError):
        yield from json_lines(read_file(path).decode(errors="replace"))


def read_events(control):
    """Events from control/events.jsonl. Lines that do not parse are skipped. A missing file gives none,
    but one that cannot be read raises OSError, so a failed read never looks like an empty log."""
    path = Path(control) / "events.jsonl"
    return list(json_lines(path.read_text(errors="replace"))) if path.exists() else []


def round_events(control):
    """Events since the last round_start; all of them in a session without rounds (controller.run_rounds).

    The finish that ends a round, its deadline and its deliverable come from these. File numbering (_count),
    the export cache, receipts and the result's counts stay global: an earlier round's receipt is still valid
    while its GLB is intact."""
    events = read_events(control)
    start = max((i for i, e in enumerate(events) if e.get("event") == "round_start"), default=-1)
    return events[start + 1:]


def round_info(control):
    """{round, kind} of the last round_start (kind build, repair or replan), else {}."""
    e = next((e for e in reversed(read_events(control)) if e.get("event") == "round_start"), None)
    return {"round": e.get("round"), "kind": e.get("kind")} if e else {}


def write_budget(work, started_at, budget_s, excluded_s=0.0, exports=(), **round_keys):
    """Write work/run_budget.json, the agent's view of the deadline. round_keys: round and kind of the round
    (round_info), none in a session without rounds. Returns the deadline."""
    deadline = started_at + budget_s + excluded_s
    write_json(Path(work) / "run_budget.json", {
        "budget_s": budget_s, "started_at": started_at, "started_at_iso": iso(started_at),
        "deadline": deadline, "deadline_iso": iso(deadline), "excluded_s": excluded_s,
        "exports": list(exports), "export_timeout_s": EXPORT_TIMEOUT_S, "max_excluded_s": MAX_EXCLUDED_S,
        **round_keys,
    })
    return deadline


def start_budget(work, control, started_at, budget_s):
    """Start the clock: control/budget.json is the record, run_budget.json the agent's view."""
    write_json(Path(control) / "budget.json", {"started_at": started_at, "budget_s": budget_s})
    return write_budget(work, started_at, budget_s, **round_info(control))


def _extend_budget(work, control):
    """Rewrite the agent's view from the control record and this round's export events (OFF_CLOCK), never from
    the view."""
    b = json.loads((Path(control) / "budget.json").read_text())
    events = round_events(control)
    runs = {kind: [{kind: e[kind], "seconds": e["seconds"]} for e in events if e.get("event") == f"{kind}_done"]
            for kind in OFF_CLOCK}
    excluded = min(sum(x["seconds"] for xs in runs.values() for x in xs), MAX_EXCLUDED_S)
    return write_budget(work, b["started_at"], b["budget_s"], excluded, runs["export"], **round_info(control))


def time_left(work, control):
    """Seconds before this round's deadline, counted as the controller counts them. Returns (left, budget_s)."""
    budget_s = json.loads((Path(control) / "budget.json").read_text())["budget_s"]
    return max(_extend_budget(work, control) - time.time(), 0.0), budget_s


def _kill_group(p):
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(p.pid, signal.SIGKILL)
    p.wait()


def _run(cmd, log, timeout, **popen):
    """Run cmd in its own process group and kill the group on timeout or SIGTERM. popen: more arguments of
    subprocess.Popen, such as cwd and env.

    Blender may run inside Apptainer, so the whole group goes, not only the direct child.
    """
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, **popen)
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
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(p.pid, signal.SIGKILL)
    sys.exit(128 + signum)


def _count(control, event):
    return sum(e.get("event") == event for e in read_events(control))


def host_copy(control, glb):
    """The host's copy of the export glb (work/exports/NNN.glb, or that path relative to work/): control/exports/
    under the same name, the file that the export's Blender wrote. The agent's containers never see it, and the
    helpers' Blender sees it read-only (controller.blender_command)."""
    return Path(control) / "exports" / Path(glb).name


def give(glb, data, report):
    """Give the agent its copies of an export: data at glb (work/exports/NNN.glb) and report at NNN.json.

    Each is put over its own name (put), so a file or link that the agent put there is replaced, never written
    through. The names are the host's, numbered from its events, so a file in the way does not fail an export that
    Blender finished. Both go through one descriptor of exports/ (folder), opened now: the agent may have swapped the
    folder for a link while Blender ran, and a check by path (real_dir) could lose that race. Such a folder fails the
    export."""
    with folder(glb.parent) as fd:
        put(fd, glb.name, data)
        put(fd, glb.with_suffix(".json").name, report)


def export_glb(work, control, blender, blend=None, timeout=EXPORT_TIMEOUT_S):
    """Export the saved .blend to exports/NNN.glb outside the time budget and re-import it to check it.

    Blender writes the GLB and its report into control/staging/, the one folder of control/ that the helpers'
    Blender can write. After Blender exits, the host reads each once, hashes the GLB, moves both into control/exports/
    (host_copy) and only then gives the agent its copies (give). So export_done logs the sha256 of the bytes Blender
    wrote, whatever the agent does in work/exports/ meanwhile, and no later Blender run can change the host's copies,
    whatever links the agent puts in work/ for it to write through. Returns the cached export when the .blend and both
    copies of the last export are unchanged since it ran. Every export_started gets an export_done, also when the
    export fails; ok means that the agent has its copies. Returns (glb, report, cached), glb the agent's copy.
    """
    work = Path(work).resolve()
    exports = real_dir(work, "exports")
    blend = in_work(work, blend or "candidate.blend")
    # A known limit: Blender opens the .blend after this hash, so the agent can swap the file in between. The .blend
    # stays in work/, since Blender resolves the agent's relative // paths against its folder. Only the agent's own
    # export cache suffers: blend_sha256 then names another scene than the export's, whose bytes are still Blender's.
    try:
        blend_sha = sha256(blend)
    except OSError:
        sys.exit(f"{blend.relative_to(work)} is missing or not a regular file; save the scene first")
    done = [e for e in read_events(control) if e.get("event") == "export_done"]
    if done and done[-1]["ok"] and done[-1]["blend_sha256"] == blend_sha:
        glb, host = work / done[-1]["glb"], host_copy(control, done[-1]["glb"])
        with contextlib.suppress(OSError, ValueError):
            # The host's copy is the one the helpers read, the agent's the one checkpoint and finish check.
            if sha256(host) == sha256(glb) == done[-1]["glb_sha256"]:
                return glb, json.loads(read_file(host.with_suffix(".json"), 1 << 20)), True
    n = _count(control, "export_started") + 1
    glb, host = exports / f"{n:03d}.glb", host_copy(control, f"{n:03d}.glb")
    staged = [Path(control) / "staging" / f"{n:03d}{suffix}" for suffix in (".glb", ".json")]
    for d in (staged[0].parent, host.parent):
        d.mkdir(exist_ok=True)
    cmd = blender_argv(blender, "export_glb.py", *staged, blend=blend)
    log_event(control, "export_started", export=n)
    t, ok, glb_sha, rep, why = time.time(), False, None, None, ""
    try:
        for p in staged:  # what an earlier Blender run may have left there, through a link that the agent put in work/
            p.unlink(missing_ok=True)
        with folder(exports) as fd, open_new(glb.with_suffix(".log").name, dir_fd=fd) as log:
            ran = run_ok(cmd, log, timeout)
        if ran:
            data, text = read_file(staged[0]), read_file(staged[1], 1 << 20)
            glb_sha, rep = hashlib.sha256(data).hexdigest(), json.loads(text)
            for p in staged:  # no Blender runs between the read and the move
                os.replace(p, host.with_name(p.name))
            give(glb, data, text)
            ok = True
    except (OSError, ValueError) as e:
        why = f" ({e})"
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
    """Render the original and the candidate with the same cameras and lights. Returns the receipt. The receipt's
    sha256 and the render are of the host's copy of the export, which the agent cannot change meanwhile. The host
    writes the receipt's folder through descriptors (new_folder): the export and the render take minutes, in which
    the agent can swap reviews/ for a link."""
    work = Path(work).resolve()
    detail = check_detail(detail)
    real_dir(work, "reviews")  # checked before the export too, so that a link there costs no export
    glb, _, cached = export_glb(work, control, blender)
    m = manifest(control)
    host = host_copy(control, glb)
    sha = sha256(host)
    receipt = f"r{_count(control, 'review') + 1:03d}-{sha[:12]}"
    out = work / "reviews" / receipt
    with new_folder(work, "reviews", receipt, "review") as fd:
        put_json(fd, "args.json", {
            "root": str(work), "out": str(out), "clay": clay, "views": m["views"], "candidate": str(host),
            "original": str(work / "original" / "original.glb"), "detail": detail,
        })
        with open_new("blender.log", dir_fd=fd) as log:
            ok = run_ok(blender_argv(blender, "review.py", out / "args.json"), log, REVIEW_TIMEOUT_S)
        if not ok:
            sys.exit(f"review render failed; see {(out / 'blender.log').relative_to(work)}")
        rec = {"receipt": receipt, "glb": str(glb.relative_to(work)), "sha256": sha, "clay": clay, "cached": cached,
               "sheet": str((out / "sheet.png").relative_to(work)),
               "review": str((out / "review.json").relative_to(work)), "time": time.time()}
        put_json(fd, "receipt.json", rec)
    log_event(control, "review", receipt=receipt, glb=rec["glb"], sha256=sha)
    return rec


def _receipt(work, control, receipt):
    """The review event of a receipt, checked against the agent's copy of its GLB as it is now, and those bytes, read
    once: that copy is the one the controller delivers. receipt.json is not read."""
    rec = next((e for e in read_events(control) if e.get("event") == "review" and e.get("receipt") == receipt), None)
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
    """Read-only copy of a reviewed GLB in checkpoints/, labeled needs_review with the open defect, kept as one
    printable line (printable)."""
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
            "status": "needs_review", "defect": printable(defect), "time": time.time()}
    write_json(dst.with_suffix(".json"), meta)
    log_event(control, "checkpoint", glb=meta["glb"], sha256=rec["sha256"], defect=meta["defect"], receipt=receipt)
    return meta


def finish(work, control, receipt, verdict, reason, limitations=()):
    """Log the hand-off and write work/finish.json for the agent. The controller then ends the session.

    A GLB must keep the size of the controller's original.glb (controller.check_size), so the agent sees a unit
    slip here, while it can still fix it. The check reads the bytes that _receipt checked. The reason and each
    limitation are kept as one printable line (printable)."""
    work = Path(work).resolve()
    if verdict not in VERDICTS:
        sys.exit(f"unknown verdict {verdict!r}; use one of {', '.join(VERDICTS)}")
    if len(limitations) > MAX_LIMITATIONS:
        sys.exit(f"{len(limitations)} limitations given; at most {MAX_LIMITATIONS} are allowed")
    rec, data = _receipt(work, control, receipt)
    if verdict != "unchanged":
        from harpor.agent.controller import check_size  # the controller imports this module

        # root/original.glb, not work/original/original.glb, which the agent can write. A GLB whose size cannot
        # be read is left to the controller, whose check_glb names what is wrong with it.
        with host_file(control, data) as path:
            why = check_size(path, Path(control).parent / "original.glb")
        if why and why.startswith("wrong size"):
            sys.exit(f"{rec['glb']} cannot be delivered: {why}. A unit slip in Blender changes every side by the "
                     "same factor; undo it")
    out = {"receipt": receipt, "verdict": verdict, "reason": printable(reason),
           "limitations": [printable(x) for x in limitations],
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


def _command(a, work, control, blender):
    """Run one parsed helper command and print its result."""
    if a.cmd == "export_glb":
        glb, rep, cached = export_glb(work, control, blender, a.blend)
        budget = json.loads((control / "budget.json").read_text())
        deadline = _extend_budget(work, control)
        print(f"{glb.relative_to(work)} ({'cached, scene unchanged' if cached else 'new'})")
        print(f"sha256 {sha256(host_copy(control, glb))}")
        print(f"meshes {rep['meshes']}, materials {rep['materials']}, images {rep['images']}, "
              f"triangles {rep['triangles']}")
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
    c = sub.add_parser("checkpoint", help="read-only copy of a reviewed GLB")
    c.add_argument("--receipt", required=True)
    c.add_argument("--defect", required=True, help="the open defect")
    f = sub.add_parser("finish", help="hand off the reviewed GLB; the last action")
    f.add_argument("--receipt", required=True)
    f.add_argument("--verdict", required=True, help=", ".join(VERDICTS))
    f.add_argument("--reason", required=True)
    f.add_argument("--limitation", action="append", default=[], help="repeat up to 3 times")
    a = p.parse_args(argv)
    # Resolved, as controller.blender_command binds them: Blender's container has no path through a symlink, such as
    # a home folder's link to a scratch file system.
    work = Path(os.environ.get("GATOR_WORK", "/work")).resolve()
    control = Path(os.environ.get("GATOR_CONTROL", work.parent / "control")).resolve()
    blender = shlex.split(os.environ.get("GATOR_BLENDER", "blender"))

    try:
        if round_info(control).get("kind") == "replan":  # no export, receipt or off-clock time in a planning round
            sys.exit(f"{a.cmd} refused: this is a planning round, in which no helper runs; write your plan to "
                     "/work/plan.md and stop")
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
    # Run as harpor.agent.tools, the module that the controller imports, so that _on_sigterm sees the children
    # that any importer starts as well as this module's (Blender): under -m this file is __main__, a second copy
    # with its own _children.
    from harpor.agent import tools

    signal.signal(signal.SIGTERM, tools._on_sigterm)
    tools.main()
