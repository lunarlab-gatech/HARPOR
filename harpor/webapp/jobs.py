"""Job folders, job state and the runner that runs the stages of every web-app job.

A job lives in RUNS/<job_id>/ (job_id YYYYmmdd-HHMMSS-<6 hex>, folder mode 0700). job.json holds its
state and is replaced atomically on every change; it never holds the API key. The upload is a video,
a rosbag or a zipped Stray Scanner export (job.json "source": {"kind": "video", "rosbag" or "stray",
"topic"}, by the file suffix: VIDEOS, BAGS, STRAY). The Runner runs the analysis ("topics", then
"frames") at once for each upload, in a thread of its own, and the build stages (extract to package)
of one job at a time in one worker thread, because there is one GPU; other jobs wait as "queued".
"topics" lists a rosbag's camera image topics (a video and a Stray Scanner capture skip it): with
none the job fails, with one "frames" runs on it, with more the job waits in "choose_topic" until
submit_analyze gets the user's topic. "frames" unpacks a Stray Scanner zip into video/stray first
(stray.py).
While its upload runs, a job is busy too (Runner.new_upload to upload_done): it cannot be deleted.
Each stage is `bash -c SCRIPT` from steps.command in its own process group, with stdout and stderr
appended to logs/<stage>.log, where a line "== <stage> <time>" starts each run.

job.json "agent" is the agent stage's agent (steps.AGENTS): "astra" (GPT-6-Astra through the Codex
CLI, with the user's OpenAI API key), "opus" (Claude Opus 5.5 through claude_agent, with the user's
Anthropic API key) or "none" (no key: the agent stage is skipped and physics uses run/posed.glb, or
run/parts.glb for an articulated job, whose parts then get no joints).

job.json "lidar" (steps.lidar) turns on the metric scale from LiDAR: null, {"kind": "rosbag", "topic":
T} for a LiDAR topic of the job's bag, or {"kind": "stray"} for the depth of a Stray Scanner capture,
which a new Stray job has (on by default). lidar_settings checks it at the start. "calibration" is the
LiDAR-camera calibration file the user uploaded for a bag, or null: {name, bytes, sha256, lidar_topic}
and the fields that lidar_scale.read_calibration parsed from J/lidar/calibration/<name>. lidar_topic
is the LiDAR topic that the upload named, or null: lidar_settings then refuses another LiDAR topic. An
upload or a delete of the calibration makes the lidar stage's record stale (steps.done), and so does a
record of an older estimator (lidar_scale.VERSIONS), so the next Build or retry starts from lidar
(Runner.resume). A Stray Scanner capture needs no calibration: its depth is aligned with the color camera.

job.json "articulated" (false for a rigid object) and "parts" {"mode", "list", "confirmed"} set up the
parts of an articulated object (steps.py). The job's VLM lists the parts in the propose stage
(mode "vlm"), or proposes them and the job waits in "review" after propose until resume gets
confirm_parts (mode "vlm_edit"; whatever auto_continue says), or the user typed the list at the start
(mode "user", which is also the mode of a job without an agent). "confirmed" turns true when the user
confirms the list, and false again whenever propose runs. A rigid job skips propose and parts.

job.json "turntable" (false at first) asks for the 360 video: the turntable stage
(steps.py) renders it after urdf. It is optional (steps.OPTIONAL): when it fails, the job gets a warning
and the build goes on to package. So is the softvideo stage after it, which needs no setting: it runs when the
MuJoCo model has a soft part.

A job folder of an earlier version may lack the record of a newer stage: Runner._recover adds it, skipped, at
every start of the server.

The API key stays in the runner's memory, with the agent it was checked for. For the stages that take
it (steps.needs_key: the agent stage, and the propose stage when the VLM lists the parts) it is
written to /dev/shm/gator-webapp-<uid>/<job_id>.env (folder 0700, file 0600, OPENAI_API_KEY=<key>
for astra, ANTHROPIC_API_KEY=<key> for opus) and deleted when the stage ends. A server that restarts
has no key, and a change of agent drops the key held for the other one, so a job that reaches such a
stage then waits in "review" until the key is given again. A build may also be queued without the key
on purpose (submit_build with api_key None, resume with key_later): it then waits there with
NO_KEY_LATER.

One Runner runs the jobs of a runs folder. It writes RUNS/.runner.json (pid, host, Slurm job), and a
second Runner on that folder refuses to start while the first one's server is known to run (Runner._claim,
holder_alive); a file it cannot check counts as stale.

    python -m harpor.webapp.jobs package --job J

writes J/results.zip: posed.glb, refined.glb (the agent's GLB; physics/up.json glb_rule says whether
physics used it), parts.glb and parts.json (an articulated job's split parts), aligned.glb,
turntable.mp4 (the 360 video, when the job asked for it), softvideo.mp4 (the soft-part video, when the MuJoCo model
has a soft part), summary.json, mjcf/, usd/, urdf/, placement.json, selection.json, lidar.json (the metric
scale from LiDAR), agent_result.json and evidence.json (the agent's evidence of an articulation in rounds),
without any file that the agent result lists in key_found_in (agent_result.json too, when the session's
result.json is listed, and evidence.json when its control/evidence.json is).
"""

import argparse
import contextlib
import json
import os
import re
import secrets
import shutil
import signal
import socket
import stat
import subprocess
import tempfile
import threading
import time
import zipfile
from pathlib import Path

from harpor.agent.tools import CLAUDE_MODEL as OPUS
from harpor.fileio import write_json
from harpor.webapp import steps

ID = re.compile(r"\d{8}-\d{6}-[0-9a-f]{6}")
KEY = re.compile(r"[A-Za-z0-9_-]{1,512}")  # one line, since it goes into an env file
STAGE = {"status": "pending", "started": None, "ended": None, "returncode": None, "note": None}
ACTIVE = ("analyzing", "queued", "running")  # a job in these statuses is busy
BUILD = steps.NAMES[steps.NAMES.index("extract"):]
VIDEOS = (".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm")
BAGS = (".bag", ".mcap", ".db3")
STRAY = (".zip",)  # a Stray Scanner export folder, zipped
INPUTS = {"video": "a video", "rosbag": "a rosbag", "stray": "a Stray Scanner capture"}  # for messages
# What a stage's rerun must not find from an earlier run; the other stages replace their own files.
# The LiDAR samples cache (lidar/samples.* and lidar/clouds.npz) stays: the lidar stage checks its key. A killed
# frames stage leaves its half-written preview video (.tmp_preview.mp4).
DERIVED = {"frames": ("frames/meta.json", "frames/thumbs", "frames/stamps.json", "frames/preview.mp4",
                      "frames/.tmp_preview.mp4"),
           "extract": ("frames/pool.json", "scene", "pool"), "lidar": ("pool/lidar.json", "lidar/check", "lidar/cloud"),
           "select": ("selection.json", "run"), "vomp": ("run/vomp",), "vomp_parts": ("run/vomp_parts",),
           "propose": ("run/proposal.json",),
           "parts": ("run/parts", "run/parts.glb", "run/parts.json"), "agent": ("agent",), "align": ("physics",),
           "turntable": ("turntable",), "softvideo": ("softvideo",), "package": ("results.zip",)}
OLDER = "older than this stage"  # the note of a stage that a job folder of an earlier version has no record of
NO_KEY = {a: f"The server no longer holds the {p} API key (it restarted); paste the key again and press Build."
          for a, p in steps.PROVIDER.items()}
NO_KEY_LATER = {a: f"The job needs the {p} API key now; paste it and press Build." for a, p in steps.PROVIDER.items()}
RUNNER_FILE = ".runner.json"  # in the runs folder: the server whose Runner runs its jobs (Runner._claim)
# squeue states of a Slurm job whose server may still run (holder_alive)
SLURM_ALIVE = ("RUNNING", "COMPLETING", "CONFIGURING", "SUSPENDED", "STOPPED", "SIGNALING")
PARTS = {"mode": "vlm", "list": [], "confirmed": False}  # job.json "parts" of a new job
_LOCK = threading.RLock()  # job.json read-modify-write, for the runner's threads and the server


def kind(filename):
    """"video", "rosbag" or "stray" by the file suffix (any case), else None."""
    suffix = Path(str(filename)).suffix.lower()
    return "video" if suffix in VIDEOS else "rosbag" if suffix in BAGS else "stray" if suffix in STRAY else None


def safe_name(filename):
    """The name an upload is saved under: ASCII letters, digits, _ . - and no leading dot; the suffix
    stays ("..bag" -> "video.bag"), so kind() of it is that of the saved file."""
    p = Path(re.sub(r"[^\w.-]+", "_", Path(str(filename)).name, flags=re.ASCII))
    return (p.stem.lstrip(".") or "video") + p.suffix


def create_job(runs_dir, filename):
    """A new job folder for an upload. Returns (job_id, the path to write the video, rosbag or zip to). A
    Stray Scanner capture starts with LiDAR on."""
    name = safe_name(filename)
    k = kind(name) or "video"
    job_id = time.strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(3)
    J = Path(runs_dir) / job_id
    J.parent.mkdir(parents=True, exist_ok=True)
    J.mkdir(mode=0o700)
    os.chmod(J, 0o700)
    for d in ("video", "logs"):
        (J / d).mkdir(mode=0o700)
    path = J / "video" / name
    save(J, {"id": job_id, "created": stamp(),
             "video": {"name": name, "path": str(path.resolve()), "bytes": None},
             "source": {"kind": k, "topic": None}, "agent": "astra", "status": "uploaded",
             "mode": "auto", "prompt": None, "slug": None, "n_views": 6, "picks": [], "view0": None,
             "auto_continue": False, "overrides": {"locked": [], "excluded": [], "instance": {}},
             "articulated": False, "parts": dict(PARTS), "lidar": {"kind": "stray"} if k == "stray" else None,
             "calibration": None, "turntable": False,
             "stages": {s: dict(STAGE) for s in steps.NAMES}, "current": None, "error": None, "warnings": [],
             "key_ok": False})
    return job_id, path


def stamp():
    """Now, in job.json's "created" format (local time)."""
    return time.strftime("%Y-%m-%dT%H:%M:%S")


load = steps.job


def save(job_dir, job):
    write_json(Path(job_dir) / "job.json", job, private=True, indent=1)


def update(job_dir, **fields):
    """Set top-level fields of job.json under the lock that the runner uses. Returns the job."""
    with _LOCK:
        job = load(job_dir)
        job.update(fields)
        save(job_dir, job)
        return job


def update_stage(job_dir, stage, **fields):
    with _LOCK:
        job = load(job_dir)
        job["stages"][stage].update(fields)
        save(job_dir, job)


def _clear(job_dir, stages):
    for s in stages:
        for rel in DERIVED.get(s, ()):
            p = Path(job_dir) / rel
            if p.is_dir() and not p.is_symlink():
                shutil.rmtree(p)
            else:
                p.unlink(missing_ok=True)


def parts_settings(job):
    """{"articulated", "parts"} of a job.json dict in the form the build runs with: parts {"mode", "list",
    "confirmed": False}, the mode as steps.parts_mode gives it (agent "none" means "user"). Only an
    articulated job in mode "user" keeps its list, in harpor.parts.propose.validate's form; the VLM
    modes start without one. ValueError (with propose's reason for a bad list) when articulated is not
    a bool, the mode is unknown, or the mode is "user" and the list is empty or breaks a rule."""
    from harpor.parts import propose

    art = job.get("articulated", False)
    if not isinstance(art, bool):
        raise ValueError("articulated is true or false")
    parts = job.get("parts") or {}
    if not isinstance(parts, dict):
        raise ValueError('parts is {"mode", "list"}')
    mode, items = steps.parts_mode({**job, "parts": parts}), []
    if art and mode == "user":
        if not parts.get("list"):
            raise ValueError("type the parts of the object" + (": without an agent there is no VLM to list them"
                                                               if steps.agent(job) == "none" else ""))
        items = propose.validate(parts["list"], job["prompt"])
    return {"articulated": art, "parts": {"mode": mode, "list": items, "confirmed": False}}


def lidar_settings(job_dir, job, value, read_bag=True):
    """job.json's "lidar" in the form the build runs with: None, {"kind": "rosbag", "topic": T} or
    {"kind": "stray"}. value is None for no LiDAR. For a rosbag job it is {"topic": T}
    ("kind" may say "rosbag"). For a Stray Scanner capture it is {"kind": "stray"}, its depth. job is
    the job.json dict, whose source and calibration count.

    ValueError with a plain reason when:
      - a video job asks for LiDAR;
      - a Stray job asks for anything else, or a bag job for Stray's;
      - T is not a LiDAR topic of frames/topics.json (lidar_topic_error);
      - the calibration was uploaded for another LiDAR topic than T (its lidar_topic; a calibration
        without one is not checked);
      - the bag's TF does not link the camera topic's frame to T's frame (tf_links), and no
        calibration was uploaded;
      - the calibration's K is for images of another aspect ratio than the camera topic's;
      - the uploaded bag, or a Stray job's export, is not on the server;
      - neither the calibration's K nor the camera topic's camera_info (rosbag.camera_model) gives the
        intrinsics.
    read_bag False leaves out the check that opens the bag, the camera_info. That is for submit_build under
    the lock, after the server's Start checked it all."""
    if value is None:
        return None
    from harpor.webapp import lidar

    input_kind = steps.source(job)["kind"]
    if input_kind == "stray":  # its depth is aligned with the color camera: only the export must be there
        if not isinstance(value, dict) or value.get("kind") != "stray":
            raise ValueError('lidar is null or {"kind": "stray"} for a Stray Scanner capture')
        export = Path(job["video"]["path"])
        if not export.is_file():
            raise ValueError(f"the uploaded export {export.name} is not on the server; upload it again")
        return {"kind": "stray"}
    if input_kind != "rosbag":
        raise ValueError(f"LiDAR needs a rosbag or a Stray Scanner input; this job's input is "
                         f"{INPUTS.get(input_kind, input_kind)}, so turn LiDAR off")
    if not isinstance(value, dict) or value.get("kind", "rosbag") != "rosbag" \
            or not isinstance(value.get("topic"), str) or not value["topic"]:
        raise ValueError('lidar is null or {"topic": T}, a LiDAR topic of the bag')
    topic, f = value["topic"], Path(job_dir) / "frames" / "topics.json"
    found = json.loads(f.read_text()) if f.is_file() else {}
    why = lidar_topic_error(found, topic)
    if why:
        raise ValueError(why)
    found_lidar = {x["topic"]: x for x in found["lidar"]}
    cam = steps.source(job).get("topic")
    camera = next((t for t in found.get("topics") or [] if t["topic"] == cam), None)
    if camera is None:
        raise ValueError("choose the bag's camera topic first")
    cal = job.get("calibration") or None
    if cal and cal.get("lidar_topic") and cal["lidar_topic"] != topic:  # its T is for that LiDAR, not this one
        raise ValueError(f"the calibration {cal.get('name')} is for {cal['lidar_topic']}, not {topic}; choose "
                         f"{cal['lidar_topic']}, or upload a calibration for {topic}")
    cam_frame, lidar_frame = camera.get("frame_id"), found_lidar[topic].get("frame_id")
    if cal is None and not camera["tf_links"].get(topic):
        raise ValueError(lidar.tf_reason(cam_frame, lidar_frame, topic))
    if cal and cal.get("K") and cal.get("image_size") and camera.get("width") and camera.get("height"):
        try:  # the stage's own check (lidar.fit_matrix), before any GPU stage runs
            lidar.fit_matrix(cal["K"], cal["image_size"], (camera["width"], camera["height"]),
                             lidar.calibration_source(cal, cal.get("name")))
        except lidar.Failed as e:
            raise ValueError(str(e)) from None
    bag = Path(job["video"]["path"])
    if not bag.is_file():  # the lidar stage's own reason (lidar.rosbag_samples), without the server's path
        raise ValueError(f"the uploaded bag {bag.name} is not on the server; upload it again")
    if read_bag and not (cal and cal.get("K")):
        from harpor.webapp import rosbag

        try:  # its ValueError (an unknown distortion model, another aspect ratio) passes on
            model = rosbag.camera_model(bag, cam)
        except SystemExit as e:
            raise ValueError(f"cannot read the intrinsics of {cam}: {e}") from None
        if model is None:
            raise ValueError(f"{cam} has no usable camera_info ({camera.get('camera_info') or rosbag.info_topic(cam)}); "
                             "give K (and D) in the calibration")
    return {"kind": "rosbag", "topic": topic}


def lidar_topic_error(found, topic):
    """Why topic is not a LiDAR topic of frames/topics.json (found, its dict) that the lidar stage can
    use, else None."""
    names = [x["topic"] for x in found.get("lidar") or []]
    if topic in names:
        return None
    why = {x["topic"]: x["why"] for x in found.get("lidar_skipped") or []}
    if topic in why:
        return f"the LiDAR topic {topic} cannot be used: {why[topic]}"
    return f"{topic} is not a LiDAR topic of the bag; " + (f"its LiDAR topics are {', '.join(names)}" if names
                                                           else "the bag has none")


def last_progress(log):
    """The last `PROGRESS x` of the newest run in a stage log, as a float in [0, 1], else None."""
    try:
        with open(log, "rb") as f:
            f.seek(max(0, f.seek(0, 2) - 65536))
            text = "\n" + f.read().decode(errors="replace")
    except OSError:
        return None
    found = re.findall(r"^PROGRESS\s+([0-9.]+)\s*$", text.rsplit("\n== ", 1)[-1], re.M)
    with contextlib.suppress(ValueError, IndexError):
        return min(max(float(found[-1]), 0.0), 1.0)
    return None


def progress(job_dir):
    """Every stage's state (name, gpu, status, started, ended, returncode, note, seconds), with progress
    from the last PROGRESS line of the running stage's log."""
    J = Path(job_dir)
    job, now, out = load(J), time.time(), []
    for s in steps.STAGES:
        st = {**job["stages"][s["name"]], "name": s["name"], "gpu": s["gpu"], "seconds": None, "progress": None}
        if st["started"]:
            st["seconds"] = round((st["ended"] or now) - st["started"], 1)
        if st["status"] == "running":
            st["progress"] = last_progress(J / "logs" / f"{s['name']}.log")
        out.append(st)
    return out


def key_kind_error(key, agent):
    """Why key cannot be a key of agent's provider, judged by its prefix alone, else None: Anthropic
    keys start with sk-ant-, OpenAI keys with sk- (and not sk-ant-). check_key asks this before any
    request, so a key pasted under the wrong agent never goes to the other company's API."""
    if agent == "opus":
        return None if key.startswith("sk-ant-") else "Anthropic keys start with sk-ant-"
    if key.startswith("sk-ant-"):
        return "this is an Anthropic key (sk-ant-); this agent needs an OpenAI key"
    return None if key.startswith("sk-") else "OpenAI keys start with sk-"


def check_key(key, agent="astra"):
    """(ok, message) for the API key of agent astra or opus. A key that is not one word, or whose prefix belongs to
    the other provider (key_kind_error), is refused before any request; ask_provider then asks the provider. The
    message never holds the key."""
    if not isinstance(key, str) or not KEY.fullmatch(key):
        return False, "an API key is one word of letters, digits, - and _"
    why = key_kind_error(key, agent)
    if why:
        return False, why
    return ask_provider(key, agent)


def ask_provider(key, agent):
    """(ok, message) from the provider of agent astra (OpenAI: GET https://api.openai.com/v1/models with the key) or
    opus (Anthropic: the model record of claude-opus-5-5, with the anthropic SDK)."""
    import urllib.error
    import urllib.request

    if agent == "opus":
        return check_anthropic_key(key)
    req = urllib.request.Request("https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=10):
            return True, "OpenAI accepted the key"
    except urllib.error.HTTPError as e:  # its body quotes part of the key, so only the code is kept
        return False, f"OpenAI rejected the key (HTTP {e.code})"
    except (urllib.error.URLError, OSError) as e:
        return False, f"could not reach api.openai.com: {getattr(e, 'reason', e)}"


def check_anthropic_key(key):
    """(ok, message) for an Anthropic API key: GET /v1/models/claude-opus-5-5 with it, one retry, 20 s.
    Only the error's kind goes into the message, never the error's text."""
    try:
        import anthropic
    except ImportError:
        return False, "the server cannot check Anthropic keys: envs/web needs anthropic==1.15.0"
    try:
        anthropic.Anthropic(api_key=key, max_retries=1, timeout=20).models.retrieve(OPUS)
        return True, f"Anthropic accepted the key for {OPUS}"
    except anthropic.AuthenticationError:
        return False, "the Anthropic key was rejected"
    except (anthropic.PermissionDeniedError, anthropic.NotFoundError):
        return False, f"this key cannot use {OPUS}"
    except anthropic.APIStatusError as e:
        return False, f"the Anthropic API answered HTTP {e.status_code}"
    except anthropic.APIConnectionError:
        return False, "could not reach the Anthropic API"


def default_key_dir():
    base = "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()  # macOS has no /dev/shm
    return Path(base) / f"gator-webapp-{os.getuid()}"


def _signal_group(pgid, sig):
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, sig)


def _alive(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def command_of(pid):
    """The command line of process pid ("" when it is gone), or None when it cannot be read."""
    with contextlib.suppress(OSError):
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    try:  # macOS has no /proc
        return subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True,
                              timeout=5).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None


def holder_alive(info):
    """True when the server that a .runner.json names ({pid, host, slurm_job}) is known to run: on this host, its pid
    runs as this user and is a HARPOR server (its command line names harpor.webapp); on another host, squeue says its
    Slurm job runs. What cannot be known (a file without a Slurm job from another host, or no answer from squeue)
    counts as gone, with a line that says so: a stale file never keeps a server from starting."""
    if info.get("host") == socket.gethostname():
        pid = info.get("pid")
        if type(pid) is not int or pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError):  # gone, or the pid is another user's process now
            return False
        cmd = command_of(pid)
        return cmd is None or "harpor.webapp" in cmd or "harpor/webapp" in cmd  # a pid reused by another program
    job = info.get("slurm_job")
    if not isinstance(job, str) or not job.isascii() or not job.isdigit():
        return False
    try:
        out = subprocess.run(["squeue", "-h", "-j", job, "-o", "%T"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        out = None
    if out is None or out.returncode and "Invalid job id" not in out.stderr:
        print(f"Slurm did not say whether job {job} (the server in {RUNNER_FILE} on {info.get('host')}) still runs; "
              "this server takes the runs folder", flush=True)
        return False
    return out.returncode == 0 and out.stdout.strip() in SLURM_ALIVE


class Runner:
    """Runs the stages of every job (see the module docstring). One per server."""

    kill_after_s = 10.0

    def __init__(self, cfg, key_dir=None):
        self.cfg, self.runs = cfg, Path(cfg.runs)
        self.key_dir = Path(key_dir) if key_dir else default_key_dir()
        self._keys, self._procs, self._cancelled, self._queue, self._uploads = {}, {}, {}, [], set()
        self._later = set()  # jobs whose key comes later
        self._cv = threading.Condition(_LOCK)
        self.runs.mkdir(parents=True, exist_ok=True)
        self._claim()  # before _recover, which would fail the running jobs of another server
        self._key_folder()
        for f in self.key_dir.glob("*.env"):  # left by a server that died
            f.unlink()
        self._recover()
        threading.Thread(target=self._worker, daemon=True).start()

    def job_dir(self, job_id):
        if not isinstance(job_id, str) or not ID.fullmatch(job_id) or not (self.runs / job_id / "job.json").is_file():
            raise ValueError(f"no job {job_id!r}")
        return self.runs / job_id

    def check_key(self, key, agent="astra"):
        return check_key(key, agent)

    # ---- requests from the server ----

    def new_upload(self, filename):
        """create_job for an upload. The job is busy (no delete, no stage) until upload_done."""
        with self._cv:
            job_id, path = create_job(self.runs, filename)
            self._uploads.add(job_id)
            return job_id, path

    def upload_done(self, job_id, n):
        """The upload of job_id ended: with n (its bytes, the file in place) the analysis starts; with None
        (the upload failed) the job folder is deleted."""
        with self._cv:
            self._uploads.discard(job_id)
            J = self.runs / job_id
            if n is None:
                shutil.rmtree(J, ignore_errors=True)
                return
            update(J, video={**load(J)["video"], "bytes": n})
            self.submit_analyze(job_id)

    def uploading(self, job_id):
        with self._cv:
            return job_id in self._uploads

    def locked(self):
        """The runner's lock, for a server request that checks that a job is idle and then changes its files (the
        calibration): while it is held, no build or resume starts and no other such request runs. Hold it only for
        quick file work, since the runner's threads wait for it."""
        return self._cv

    def submit_analyze(self, job_id, topic=None):
        """Run the analysis now, in its own thread (CPU only, so it may run while another job builds):
        "topics" unless it is done, then "frames". topic, one of a rosbag job's frames/topics.json, sets
        source.topic and deletes the outputs and records of every build stage and the picks first, as
        for a new upload; not while the job is in review or done. A calibration uploaded for another
        camera topic is deleted then."""
        J = self.job_dir(job_id)
        with self._cv:
            job = load(J)
            self._idle(job_id, job)
            stages, fields = job["stages"], {}
            if topic is not None:
                if steps.source(job)["kind"] != "rosbag":
                    raise ValueError(f"the job's input is {INPUTS.get(steps.source(job)['kind'], 'a video')}, not a rosbag")
                if job["status"] in ("review", "done"):
                    raise ValueError(f"job {job_id} is {job['status']}")
                _clear(J, BUILD)
                stages.update({s: dict(STAGE) for s in BUILD})
                fields = {"source": {**steps.source(job), "topic": topic}, "picks": [], "view0": None,
                          "overrides": {"locked": [], "excluded": [], "instance": {}}}
                if job.get("calibration") and steps.source(job).get("topic") not in (None, topic):
                    shutil.rmtree(J / "lidar" / "calibration", ignore_errors=True)  # its T and K are another camera's
                    fields["calibration"] = None
            _clear(J, ["frames"])
            stages["frames"] = dict(STAGE)
            if not (stages["topics"]["status"] == "done" and steps.done("topics", J)):
                stages["topics"] = dict(STAGE)
            self._cancelled.pop(job_id, None)
            update(J, **fields, status="analyzing", stages=stages, current=None, error=None, warnings=[])
        threading.Thread(target=self._analyze, args=(job_id,), daemon=True).start()

    def has_key(self, job_id, agent=None):
        """True when the runner holds a key for the job (checked for agent, when given)."""
        with self._cv:
            return job_id in self._keys and agent in (None, self._keys[job_id][0])

    def forget_key(self, job_id):
        """Drop the key held for the job; job.json key_ok turns false. Returns True when one was held. A
        stage that already has its key file keeps it until it ends; a later stage that needs the key
        waits in review for it."""
        with self._cv:
            held, J = self._keys.pop(job_id, None) is not None, self.runs / str(job_id)
            if held and ID.fullmatch(str(job_id)) and (J / "job.json").is_file():
                update(J, key_ok=False)
            return held

    def queued(self):
        """The builds that wait, oldest first, as [(job_id, the stage each starts from)]: a copy, taken
        under the lock. The running build is not in it."""
        with self._cv:
            return [(job_id, first) for job_id, first, _ in self._queue]

    def delete(self, job_id):
        """Delete the job folder and the key held for it, unless the job is uploading, analyzing, queued
        or running."""
        J = self.job_dir(job_id)
        with self._cv:
            self._idle(job_id, load(J))
            self._keys.pop(job_id, None)
            self._cancelled.pop(job_id, None)
            self._later.discard(job_id)
            shutil.rmtree(J)

    def submit_build(self, job_id, api_key, checked_calibration=None, **settings):
        """Queue the build from extract with the settings in job.json (mode, prompt, n_views, picks, view0,
        auto_continue, overrides, agent, articulated, parts, lidar, turntable); settings are written to
        job.json first, under the same lock. parts is {"mode": steps.PARTS_MODES, "list": the typed parts}; it
        is stored with the mode the job runs with (parts_settings). lidar is None, {"topic": T} or
        {"kind": "stray"}, stored as lidar_settings gives it (its ValueError passes on); the camera_info check,
        which opens the bag, is left to the caller (the server's Start makes it). checked_calibration is
        {"sha256": the job's calibration sha256 or None} when the caller made that check: a calibration that
        another tab changed since then is refused, since the check ran on the other one. turntable is true or
        false; the server checks its type, as it does for mode and n_views. api_key is for the agent (and the VLM of
        the part list), unless the agent is "none". None queues the build without a key (a key held for the job is
        dropped, key_ok is false): the job waits in review with NO_KEY_LATER when it reaches a stage that
        needs the key. An earlier build's outputs are deleted first."""
        from harpor.stage_a.sam3_masks import slug

        J = self.job_dir(job_id)
        with self._cv:
            job = load(J)
            self._idle(job_id, job)
            if not steps.done("frames", J):
                raise ValueError("the input has not been analyzed")
            job.update(settings)
            try:
                s = slug(job.get("prompt") or "")
            except SystemExit as e:
                raise ValueError(str(e)) from None
            settings.update(parts_settings(job))
            if "lidar" in settings:  # the bag is not opened under the lock; the server's Start did that check
                if checked_calibration is not None and settings["lidar"] and \
                        (job.get("calibration") or {}).get("sha256") != checked_calibration["sha256"]:
                    raise ValueError("the calibration changed while the job was starting; press Start again")
                settings["lidar"] = lidar_settings(J, job, settings["lidar"], read_bag=False)
            agent = steps.agent(job)
            if agent != "none" and api_key is not None:
                self._take_key(job_id, api_key, agent)
            else:  # no agent, or the key comes later
                self._keys.pop(job_id, None)
            self._key_later(job_id, agent != "none" and api_key is None)
            _clear(J, BUILD)
            self._cancelled.pop(job_id, None)
            update(J, **settings, status="queued", slug=s, stages={**job["stages"], **{b: dict(STAGE) for b in BUILD}},
                   current=None, error=None, warnings=[], key_ok=self._key_for(job_id, agent) is not None)
            self._queue.append((job_id, "extract", not job.get("auto_continue")))
            self._cv.notify()

    def resume(self, job_id, from_stage, api_key=None, confirm_parts=False, parts_list=None, key_later=False,
               checked_calibration=None, **fields):
        """Queue the stages from from_stage on; api_key replaces the key held for the job, and fields
        (overrides, view0, agent) are written to job.json first, under the same lock. key_later, without
        api_key: the key was left out on purpose, so a job without one that reaches a stage that needs
        it waits in review with NO_KEY_LATER instead of NO_KEY.

        From a stage up to select (the user changed an instance or the picks): the stages from there
        rerun, except da3, which reruns only when prep made a view it has not seen; the outputs of
        select and later are deleted; and the job stops at the review again (always when it was in
        review, else unless auto_continue). From sam3d on (Build, or a retry after a failure): the
        stages that finished are skipped, except that another agent than the job's reruns the agent
        stage and every stage after it (the part list stays). Another agent drops the key held for the
        old one. From an optional stage (steps.OPTIONAL; a done job's Retry after its 360 video failed):
        it and every stage after it run again, since package zipped its old output. Returns the stage
        the run starts from: from_stage, or the agent stage when another agent moves it earlier.

        confirm_parts, while the job waits for its part list (steps.waits_for_parts): parts_list, or the
        VLM's own list when it is None, becomes the confirmed list in run/proposal.json
        (harpor.parts.propose.confirm, whose ValueError passes on with the reason) and in job.json parts
        (list, confirmed); the outputs of parts and later are deleted, and the run starts from parts.
        Without confirm_parts such a job resumes only from propose or earlier (it stops there again).
        A job that was cancelled or failed while its part list waited (steps.pause_after propose) starts
        from propose when from_stage is later, so it stops at the part list again.

        A job with LiDAR whose lidar stage is not done (it failed or was cancelled, the calibration was
        uploaded or deleted since it ran, or an older estimator made its record: steps.done) starts from
        lidar when from_stage is later, as a run from a stage up to select does: the new scale is shown
        in the review before the build goes on. A confirm_parts then raises ValueError, since the run
        from lidar would drop the list. checked_calibration is {"sha256": the job's calibration sha256 or
        None} when the caller checked the LiDAR settings outside the lock (server.lidar_check): a
        calibration that another tab changed since then raises ValueError, as in submit_build.
        """
        J = self.job_dir(job_id)
        if confirm_parts:
            from_stage = "parts"
        elif parts_list is not None:
            raise ValueError("a part list is given only with confirm_parts")
        if from_stage not in BUILD:
            raise ValueError(f"cannot resume from {from_stage!r}")
        i, r = steps.NAMES.index(from_stage), steps.NAMES.index(steps.REVIEW_AFTER)
        with self._cv:
            job = load(J)
            self._idle(job_id, job)
            agent = steps.agent({**job, **fields})
            changed = agent != steps.agent(job)
            waits = steps.waits_for_parts(J, job)
            if confirm_parts and not waits:
                raise ValueError("the job does not wait for its part list")
            p = steps.NAMES.index(steps.PARTS_AFTER)
            if waits and not confirm_parts and i > p:
                raise ValueError("the job waits for its part list; confirm or edit it first")
            if not confirm_parts and i > p and steps.pause_after(steps.PARTS_AFTER, J):
                from_stage, i = steps.PARTS_AFTER, p  # cancelled or failed at the part list: back to it
            if checked_calibration is not None and steps.lidar(job) is not None and \
                    (job.get("calibration") or {}).get("sha256") != checked_calibration["sha256"]:
                raise ValueError("the calibration changed while the run was starting; try again")
            li = steps.NAMES.index("lidar")
            if i > li and steps.lidar({**job, **fields}) is not None and not steps.done("lidar", J):
                if confirm_parts:
                    raise ValueError("the LiDAR scale is not current (the calibration or the app's scale estimate "
                                     "changed since it ran); cancel the job, then retry from the lidar stage")
                from_stage, i = "lidar", li  # failed, cancelled or stale: the scale first
            if confirm_parts:
                from harpor.parts import propose

                if parts_list is None:
                    parts_list = json.loads((J / steps.MARKERS["propose"]).read_text())["parts"]
                propose.validate(parts_list, job["prompt"])  # a bad list changes nothing
            self._check_part_list(J, {**job, **fields}, i)
            stages = job["stages"]
            if i <= r:
                _clear(J, steps.NAMES[i:])
                stages.update({s: dict(STAGE) for s in steps.NAMES[i:] if s != "da3" or s == from_stage})
                pause = job["status"] == "review" or not job.get("auto_continue")
            elif stages[steps.REVIEW_AFTER]["status"] != "done" or not steps.done(steps.REVIEW_AFTER, J):
                raise ValueError("the views are not selected yet")
            else:
                pause = False
                if from_stage in steps.OPTIONAL:  # results.zip holds its old output, or none
                    _clear(J, steps.NAMES[i:])
                    stages.update({s: dict(STAGE) for s in steps.NAMES[i:]})
            if confirm_parts:
                proposal = propose.confirm(J, parts_list)
                _clear(J, steps.NAMES[i:])
                stages.update({s: dict(STAGE) for s in steps.NAMES[i:]})
                fields["parts"] = {**job["parts"], "list": proposal["parts"], "confirmed": True}
            if changed and i > r:  # the other agent runs, and the stages after it take its GLB
                a = steps.NAMES.index("agent")
                _clear(J, steps.NAMES[a:])
                stages.update({s: dict(STAGE) for s in steps.NAMES[a:]})
                from_stage = steps.NAMES[min(i, a)]
            if api_key:
                self._take_key(job_id, api_key, agent)
            elif changed:
                self._keys.pop(job_id, None)
            self._key_later(job_id, key_later and not api_key)
            self._cancelled.pop(job_id, None)
            update(J, **fields, status="queued", stages=stages, current=None, error=None, warnings=[],
                   key_ok=self._key_for(job_id, agent) is not None)
            self._queue.append((job_id, from_stage, pause))
            self._cv.notify()
        return from_stage

    def cancel(self, job_id):
        """Drop the job from the queue, or SIGTERM its running stage's process group (SIGKILL follows
        kill_after_s later); the job ends as "cancelled". Returns at once."""
        J = self.job_dir(job_id)
        with self._cv:
            if load(J)["status"] not in ACTIVE + ("review",):
                return
            self._cancelled[job_id] = time.time()
            self._keys.pop(job_id, None)
            self._queue = [q for q in self._queue if q[0] != job_id]
            proc = self._procs.get(job_id)
            if proc is None:
                update(J, status="cancelled", current=None, key_ok=False)
            else:
                _signal_group(proc.pid, signal.SIGTERM)

    def shutdown(self, timeout=None):
        """Cancel every running stage, wait for them to end and delete the key files and .runner.json
        (server exit)."""
        with self._cv:
            self._queue.clear()
            running = list(self._procs)
        for job_id in running:
            self.cancel(job_id)
        t0 = time.time()
        while self._procs and time.time() - t0 < (timeout or self.kill_after_s + 5):
            time.sleep(0.1)
        for f in self.key_dir.glob("*.env"):
            f.unlink(missing_ok=True)
        with contextlib.suppress(OSError, ValueError, AttributeError):  # unless another Runner took the folder since
            if json.loads((self.runs / RUNNER_FILE).read_text()).get("id") == self._id:
                (self.runs / RUNNER_FILE).unlink()

    # ---- inside the runner ----

    def _claim(self):
        """Write RUNS/.runner.json ({pid, host, slurm_job, started, id}), or SystemExit while the server
        it names still runs (holder_alive): two Runners on one runs folder would fail each other's jobs
        (_recover) and run two GPU stages at once. A Runner of this same process takes over, as a
        restart in the tests does. This guards against a second server started by mistake; it is not a
        lock against two that start in the same second."""
        f, old = self.runs / RUNNER_FILE, None
        with contextlib.suppress(OSError, ValueError):
            old = json.loads(f.read_text())
        this = isinstance(old, dict) and old.get("host") == socket.gethostname() and old.get("pid") == os.getpid()
        if isinstance(old, dict) and not this and holder_alive(old):
            where = f"pid {old.get('pid')} on {old.get('host')}" + (
                f", Slurm job {old['slurm_job']}" if old.get("slurm_job") else "")
            raise SystemExit(f"another HARPOR server runs the jobs in {self.runs} ({where}); stop it first, "
                             f"or delete {f} if it is gone")
        self._id = secrets.token_hex(8)
        info = {"pid": os.getpid(), "host": socket.gethostname(), "slurm_job": os.environ.get("SLURM_JOB_ID"),
                "started": stamp(), "id": self._id}
        write_json(f, info, private=True, indent=1)

    def _key_later(self, job_id, later):
        """Remember whether the newest build or resume of the job left its key for later (NO_KEY_LATER)."""
        (self._later.add if later else self._later.discard)(job_id)

    def _idle(self, job_id, job):
        if job_id in self._uploads:
            raise ValueError(f"job {job_id} is still uploading")
        if job["status"] in ACTIVE or job_id in self._procs:
            raise ValueError(f"job {job_id} is {job['status']}")

    def _check_part_list(self, J, job, i):
        """ValueError when the propose stage would run from stage index i on a job whose parts mode is
        "user" (also an agent "none") but whose job.json has no part list: it would fail."""
        p = steps.NAMES.index("propose")
        done = job["stages"]["propose"]["status"] == "done" and steps.done("propose", J)
        runs = i <= p and (i <= steps.NAMES.index(steps.REVIEW_AFTER) or not done)
        if runs and steps.articulated(job) and steps.parts_mode(job) == "user" and not job["parts"]["list"]:
            raise ValueError("this job has no part list" + (": without an agent there is no VLM to list the parts"
                                                            if steps.agent(job) == "none" else ""))

    def _take_key(self, job_id, key, agent):
        if not isinstance(key, str) or not KEY.fullmatch(key):
            raise ValueError("an API key is one word of letters, digits, - and _")
        self._keys[job_id] = (agent, key)

    def _key_for(self, job_id, agent):
        """The key held for the job if it was given for agent, else None."""
        held, key = self._keys.get(job_id, (None, None))
        return key if held == agent else None

    def _key_folder(self):
        self.key_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        st = os.lstat(self.key_dir)
        if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid():
            raise SystemExit(f"{self.key_dir} is not a folder of this user; remove it")
        os.chmod(self.key_dir, 0o700)

    def _write_key(self, job_id, agent, key):
        self._key_folder()
        path = self.key_dir / f"{job_id}.env"
        path.unlink(missing_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(f"{steps.KEY_VAR[agent]}={key}\n")
        return path

    def _recover(self):
        """Jobs that were running when an earlier server stopped are failed; no job holds a key. A job folder of an
        earlier version gets a record of each stage it has none of, skipped (OLDER), so that a done job does not look
        unfinished and progress and _build find every stage; load stays strict."""
        for f in sorted(self.runs.glob("*/job.json")):
            with contextlib.suppress(OSError, ValueError, KeyError):
                job = load(f.parent)
                fields = {"key_ok": False}
                older = {s: {**STAGE, "status": "skipped", "note": OLDER} for s in steps.NAMES
                         if s not in job["stages"]}
                if older:
                    job["stages"].update(older)
                    fields["stages"] = job["stages"]
                if job["status"] in ACTIVE:
                    for name, st in job["stages"].items():
                        if st["status"] == "running":  # it ended about when its log last changed
                            log = f.parent / "logs" / f"{name}.log"
                            ended = log.stat().st_mtime if log.is_file() else time.time()
                            st.update(status="failed", note="the server stopped", ended=max(ended, st["started"] or 0))
                    fields.update(status="failed", current=None, stages=job["stages"],
                                  error="the server stopped while this job ran; resume it")
                update(f.parent, **fields)

    def _status(self, job_id, status, **fields):
        """Set the job's status unless it was cancelled meanwhile. Returns False if it was."""
        with self._cv:
            if job_id in self._cancelled:
                return False
            if status == "done":
                self._keys.pop(job_id, None)
                fields["key_ok"] = False
            update(self.runs / job_id, status=status, **fields)
            return True

    def _worker(self):
        while True:
            with self._cv:
                while not self._queue:
                    self._cv.wait()
                job_id, first, pause = self._queue.pop(0)
            try:
                self._build(job_id, first, pause)
            except Exception as e:  # one broken job must not stop the queue
                self._crashed(job_id, e)

    def _analyze(self, job_id):
        try:
            if load(self.runs / job_id)["stages"]["topics"]["status"] != "done" \
                    and self._run_stage(job_id, "topics") not in ("done", "skipped"):
                return
            if self._has_topic(job_id) and self._run_stage(job_id, "frames") == "done":
                self._status(job_id, "ready", current=None)
        except Exception as e:
            self._crashed(job_id, e)

    def _has_topic(self, job_id):
        """True when "frames" can run: on a video, or on a rosbag's topic (the only one, when the bag has
        one camera topic). Else the job waits in "choose_topic"."""
        J = self.runs / job_id
        with self._cv:
            job = load(J)
            src = steps.source(job)
            if src["kind"] != "rosbag":
                return True
            found = [t["topic"] for t in json.loads((J / "frames" / "topics.json").read_text())["topics"]]
            if src.get("topic") in found:
                return True
            if len(found) == 1:
                update(J, source={**src, "topic": found[0]})
                return True
            self._status(job_id, "choose_topic", current=None)
            return False

    def _crashed(self, job_id, e):
        J = self.runs / job_id
        with contextlib.suppress(Exception):
            job = load(J)
            if job["current"]:
                update_stage(J, job["current"], status="failed", ended=time.time(), note=f"runner error: {e!r}")
            self._status(job_id, "failed", current=None, error=f"runner error: {e!r}")

    def _build(self, job_id, first, pause):
        J = self.runs / job_id
        if not self._status(job_id, "running"):
            return
        for stage in steps.NAMES[steps.NAMES.index(first):]:
            ran = not (load(J)["stages"][stage]["status"] == "done" and steps.done(stage, J))
            if ran and self._run_stage(job_id, stage) not in ("done", "skipped", "warned"):
                return
            if (ran and stage == steps.REVIEW_AFTER and pause) or steps.pause_after(stage, J):
                self._status(job_id, "review", current=None)  # after propose: the part list waits for the user
                return
        self._status(job_id, "done", current=None)

    def _run_stage(self, job_id, stage):
        """Run one stage and record it. Returns done, skipped, warned (an optional stage failed: steps.OPTIONAL;
        the job gets a warning and the build goes on), failed, review or cancelled."""
        J = self.runs / job_id
        reason = steps.skip(stage, J)
        if reason:
            update_stage(J, stage, **{**STAGE, "status": "skipped", "note": reason})
            return "skipped"
        key_file, job = None, load(J)
        if stage == "propose":  # a new proposal is not confirmed
            with self._cv:
                job = update(J, parts={**load(J)["parts"], "confirmed": False})
        if steps.needs_key(stage, job):
            agent = steps.agent(job)
            with self._cv:
                key, later = self._key_for(job_id, agent), job_id in self._later
            if key is None:
                update_stage(J, stage, note=f"waiting for the {steps.PROVIDER[agent]} API key")
                self._status(job_id, "review", current=None, key_ok=False,
                             warnings=[(NO_KEY_LATER if later else NO_KEY)[agent]])
                return "review"
            key_file = self._write_key(job_id, agent, key)
        log_path = J / "logs" / f"{stage}.log"
        log_path.parent.mkdir(exist_ok=True)
        try:
            argv, cwd, env = steps.command(stage, load(J), self.cfg, key_file)
            with open(log_path, "ab") as log:
                log.write(f"== {stage} {time.strftime('%Y-%m-%d %H:%M:%S')}\n".encode())
                log.flush()
                with self._cv:
                    if job_id in self._cancelled:
                        return "cancelled"
                    update_stage(J, stage, **{**STAGE, "status": "running", "started": time.time()})
                    update(J, current=stage)
                    proc = self._procs[job_id] = subprocess.Popen(
                        argv, cwd=cwd, env={**os.environ, **env}, stdin=subprocess.DEVNULL, stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True)
                rc = self._wait(job_id, proc)
        finally:
            with self._cv:
                self._procs.pop(job_id, None)
            if key_file is not None:
                key_file.unlink(missing_ok=True)
        ended = {"ended": time.time(), "returncode": rc}
        with self._cv:
            if job_id in self._cancelled:
                update_stage(J, stage, status="failed", note="cancelled", **ended)
                update(J, status="cancelled", current=None, key_ok=False)
                return "cancelled"
        why = steps.guard(stage, J)
        if rc == 0 and why is None and steps.done(stage, J):
            update_stage(J, stage, status="done", **ended)
            return "done"
        if stage == "select" and rc == 2 and (J / "selection.json").is_file():
            warnings = json.loads((J / "selection.json").read_text()).get("warnings", [])
            update_stage(J, stage, status="failed", note="the automatic selection needs help", **ended)
            self._status(job_id, "review", current=None,
                         warnings=warnings + ["Pick the frames by hand, or lock or exclude views and select again."])
            return "review"
        why = why or (f"exit {rc}: {_last_line(log_path)}" if rc else "it finished without writing its output")
        update_stage(J, stage, status="failed", note=why, **ended)
        if stage in steps.OPTIONAL:
            with self._cv:
                update(J, warnings=[*(load(J).get("warnings") or []), f"{steps.OPTIONAL[stage]}: {why}"])
            return "warned"
        self._status(job_id, "failed", current=None, error=f"{stage}: {why}")
        return "failed"

    def _wait(self, job_id, proc):
        """Wait for a stage. After a cancel its process group has kill_after_s to end before SIGKILL; whatever
        it leaves running otherwise is killed at once."""
        while True:
            try:
                rc = proc.wait(0.2)
                break
            except subprocess.TimeoutExpired:
                self._grace(job_id, proc.pid)
        while self._grace(job_id, proc.pid) and _alive(proc.pid):
            time.sleep(0.1)  # the agent step is still stopping its container
        _signal_group(proc.pid, signal.SIGKILL)
        return rc

    def _grace(self, job_id, pgid):
        """True while a cancelled stage is within kill_after_s of the cancel; SIGKILL to its group after."""
        with self._cv:
            t = self._cancelled.get(job_id)
        if t is None:
            return False
        if time.time() - t < self.kill_after_s:
            return True
        _signal_group(pgid, signal.SIGKILL)
        return False


def _last_line(log):
    """The last line that the newest run in a stage log printed, other than PROGRESS lines."""
    text, last = "", ""
    with contextlib.suppress(OSError):
        with open(log, "rb") as f:
            f.seek(max(0, f.seek(0, 2) - 8192))
            text = ("\n" + f.read().decode(errors="replace")).rsplit("\n== ", 1)[-1]
    for line in text.splitlines()[1:]:  # the first is the header or a cut line
        if line.strip() and not line.startswith("PROGRESS"):
            last = line.strip()
    return last[:300] or "no output"


def package(job_dir):
    """Write J/results.zip; see the module docstring."""
    J = Path(job_dir)
    result, evidence = J / "agent" / "result.json", J / "agent" / "evidence.json"
    keyed = set()
    if result.is_file():
        found = json.loads(result.read_text()).get("key_found_in") or []
        keyed = {(J / "agent" / "s" / p).resolve() for p in found}
        # agent/result.json and agent/evidence.json are copies of these files with the key scrubbed out, so they stay
        # out too.
        copies = {"result.json": result, "control/evidence.json": evidence}
        keyed |= {copy.resolve() for src, copy in copies.items() if src in found}
    files = [("posed.glb", J / "run" / "posed.glb"), ("refined.glb", steps.agent_glb(J)),
             ("parts.glb", J / "run" / "parts.glb"), ("parts.json", J / "run" / "parts.json"),
             ("aligned.glb", J / "physics" / "aligned.glb"), ("turntable.mp4", J / "turntable" / "turntable.mp4"),
             ("softvideo.mp4", J / "softvideo" / "softvideo.mp4"), ("summary.json", J / "physics" / "summary.json"),
             ("placement.json", J / "run" / "placement.json"), ("selection.json", J / "selection.json"),
             ("lidar.json", J / "pool" / "lidar.json"), ("agent_result.json", result), ("evidence.json", evidence)]
    for d in ("mjcf", "usd", "urdf"):
        files += [(f"{d}/{p.relative_to(J / 'physics' / d)}", p) for p in sorted((J / "physics" / d).rglob("*"))]
    # align copies the GLB byte for byte, MJCF, USD and URDF copy its meshes and textures, and the 360 video and the
    # soft-part video show it.
    used = (json.loads((J / "physics" / "up.json").read_text()).get("glb") if (J / "physics" / "up.json").is_file() else None)
    if used and Path(used).resolve() in keyed:
        keyed |= {p.resolve() for a, p in files
                  if a in ("aligned.glb", "turntable.mp4", "softvideo.mp4") or a.startswith(("mjcf/", "usd/", "urdf/"))}
    tmp = J / ".results.zip.tmp"
    names = []
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for arc, p in files:
            if p is not None and p.is_file() and not p.is_symlink() and p.resolve() not in keyed:
                z.write(p, arc)
                names.append(arc)
    os.replace(tmp, J / "results.zip")
    left = [a for a, p in files if p is not None and p.resolve() in keyed]
    print(f"results.zip: {len(names)} files, {(J / 'results.zip').stat().st_size / 1e6:.1f} MB"
          + (f"; left out (they hold the API key): {left}" if left else ""))
    for arc in ("posed.glb", "aligned.glb", "summary.json"):
        if arc not in names:
            print(f"  missing: {arc}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="step", required=True)
    p = sub.add_parser("package")
    p.add_argument("--job", required=True)
    a = ap.parse_args(argv)
    package(a.job)


if __name__ == "__main__":
    main()
