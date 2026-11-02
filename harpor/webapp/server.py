"""The HARPOR web app's server: upload a phone video, a rosbag or a Stray Scanner capture, choose the
views, build a simulation-ready object.

Runs in envs/web (fastapi, uvicorn, av, numpy, pillow, scipy, rosbags) on the GPU machine, from the repo
root (start_ice.sh starts it there, inside a Slurm job):

    python -m harpor.webapp [--runs DIR] [--host 127.0.0.1] [--port 0] [--fake] [--worker [--local-keys A]]
                           [--demos DIR]

It makes a login token, writes ~/.config/harpor/webapp.json (0600: node, port, token, url_path, and
slurm_job and slurm_node from SLURM_JOB_ID and SLURMD_NODENAME, so start_ice.sh reuses the server
only in the same job, and worker), prints the URL and the port forward from another computer, and serves the app
with one uvicorn worker. When the server stops (Ctrl-C or SIGTERM), it runs the hooks that on_stop
added, then cancels the running stages, and deletes webapp.json if it still holds this server's token.
--worker also serves the public site: a personal worker (worker.py) in this process, on the same Runner,
signed in as the machine's user (python -m harpor.webapp.worker login). --local-keys names the agents whose
key the machine keeps in its own key file. webapp.json's "worker" is true while the worker runs, else false.
--host defaults to 127.0.0.1; start_ice.sh passes 0.0.0.0, so that the login node can reach the server.
--port 0 takes a free port. --fake runs every stage with harpor.webapp.fake_steps, so the app works
on a Mac without GPUs (OpenAI keys that start with sk- and Anthropic keys that start with sk-ant-
pass). The runs folder defaults to steps.config_from_env(): WEBAPP_RUNS, else $G/webapp_runs. --demos DIR (default:
WEBAPP_DEMOS, else none) names a demos folder (demos.py), which the server reads and checks once when it starts: add
or change a demo, then start it again.

Login: GET /?token=T sets an HttpOnly SameSite=Strict cookie and redirects to /. Every other
request needs the cookie and gets 401 without it, static files included.

Routes (JSON unless said otherwise; {id} is a job id):
  GET    /demos.html                  static/demos.html with the cards of the demos and of the placeholders, and
                                      the demos' sections (demos.page), or "No demos yet."
  GET    /demos/{name}/{file}         a file that the demo's demo.json lists (demos.TYPES), else 404; Range and
                                      HEAD as for a job's file
  PUT    /api/upload?name=N           the file as the raw body: a video (jobs.VIDEOS), a rosbag (jobs.BAGS)
                                      or a zipped Stray Scanner export (jobs.STRAY, .zip) by the suffix it
                                      is saved under, else 415; 411 without a Content-Length; at most 20
                                      GB, and 413 up front when the Content-Length is larger or the runs
                                      folder has less than 1.2 x Content-Length + 2 GB free (2.2 x for a
                                      zip, which the frames stage unpacks next to it); makes a job,
                                      analyzes it. A Stray job starts with LiDAR on
  GET    /api/jobs                    every job: id, created, status ("uploading" while its upload runs),
                                      prompt, video (the file name), kind
  GET    /api/jobs/{id}               job.json with "kind" (video, rosbag or stray), the stages' progress,
                                      the agent's verdict, model and, for opus, tokens and cost estimate
                                      ("agent_result"; "agent" is the job's choice: astra, opus or none;
                                      an articulation result adds mode, counts, joints_delivered and
                                      simulations, one in rounds also rounds, requirements, candidate and
                                      surface, and a refinement's surface, controller.run_session), the
                                      output files that exist and then the LiDAR points of the run's views
                                      (run/lidar/<i>.glb, by i) and, while the scale has failed, of the
                                      pool's views (lidar/cloud/<view>.glb, pool_clouds) as "files", their
                                      sizes in bytes ("sizes", sizes_of),
                                      "articulated", "parts_mode" and "parts_review" (the job waits for its
                                      part list), "lidar" (steps.lidar), "calibration" (the uploaded
                                      calibration's summary, calibration_summary, or null), "lidar_current"
                                      (steps.done("lidar") for a job with LiDAR whose lidar stage is done:
                                      false when the calibration was uploaded or deleted after the scale
                                      was found, or an older estimator found it; else null), "lidar_stale"
                                      (why it is false: "calibration" or "estimator", lidar_stale; else
                                      null) and "preview" (the input's preview video and where each view
                                      is in it, preview_of)
  DELETE /api/jobs/{id}               the job folder, unless the job is uploading, analyzing, queued or
                                      running
  GET    /api/jobs/{id}/frames        frames/meta.json, with frames/pool.json as "pool" once extracted
  GET    /api/jobs/{id}/thumb/{idx}   a thumbnail (JPEG)
  POST   /api/jobs/{id}/topic         {topic}: a rosbag job's camera topic (one of frames/topics.json), in
                                      choose_topic, ready, failed or cancelled; analyzes the frames again.
                                      409 for a job whose input is not a rosbag
  PUT    /api/jobs/{id}/calibration?name=N&lidar_topic=L
                                      a rosbag job's LiDAR-camera calibration as the raw body: .yaml, .yml
                                      or .json (else 415), at most 64 KB (else 413); 409 for a video job, for
                                      a Stray Scanner capture (its depth is aligned with the color camera),
                                      while the job is analyzing, queued or running, and while a job with
                                      LiDAR waits for its part list (the new scale would drop the list:
                                      Runner.resume); 400
                                      with the reason of lidar_scale.read_calibration. L (optional; empty
                                      for none) is the LiDAR topic that the file is for, one of
                                      frames/topics.json "lidar" (else 400); jobs.lidar_settings then
                                      refuses another LiDAR topic. Saved as lidar/calibration/<name> (0600)
                                      and in job.json "calibration", with "lidar_topic" (L or null);
                                      answers calibration_summary. A lidar stage that ran with another
                                      calibration is then stale: the next Build or retry starts from
                                      lidar and stops in review (Runner.resume)
  DELETE /api/jobs/{id}/calibration   the calibration (409 for a job that is not a rosbag's, while the job
                                      is analyzing, queued or running, and while a job with LiDAR waits for
                                      its part list); a lidar stage that ran with it is stale, as after a PUT
  POST   /api/jobs/{id}/start         {mode, prompt, n_views, picks, view0, auto_continue, agent, api_key,
                                      articulated?, parts_mode?, parts?, lidar?, turntable?} (agent astra: an
                                      OpenAI key; opus: an Anthropic key; none: no key, physics uses the Stage A
                                      mesh). articulated (default false) builds an object with moving parts;
                                      parts_mode (steps.PARTS_MODES, default vlm) says who lists them, and parts is the
                                      typed list of mode user ([{name, parent, joint, count}], checked with
                                      harpor.parts.propose); agent none means mode user. lidar is null,
                                      {"topic": T}, a LiDAR topic of a rosbag job, or {"kind": "stray"}, the
                                      depth of a Stray Scanner capture, for the metric scale;
                                      jobs.lidar_settings checks it (400 with the reason). Without it a
                                      Stray job keeps its own (LiDAR on, unless a Start turned it off) and
                                      the other jobs get null. turntable (default false) renders a 360
                                      video of the object at the end (the turntable stage)
  GET    /api/jobs/{id}/selection     selection.json with "sam3": the SAM 3 instances of every pool frame,
                                      and "lidar": the metric scale's record in short (lidar_compact) for a
                                      job with LiDAR, else null; its calibration_changed is true when the
                                      calibration was uploaded or deleted after the scale was found, and its
                                      estimator_changed when an older estimator found it (lidar_stale)
  POST   /api/jobs/{id}/overrides     {locked, excluded, instance, view0, agent?, api_key?}: select again
                                      (masks first when an instance changed); the job stops in review. A
                                      run that includes the lidar stage (from masks, or from lidar when the
                                      scale is stale) answers 400 when the job's LiDAR would fail that
                                      stage's checks, as for continue
  GET    /api/jobs/{id}/parts         run/proposal.json ("proposal"), the job's parts settings, "waiting"
                                      (the job waits for its part list), "photo" (view 0's SAM 3 check
                                      image) and "checks" (the part masks' check images)
  POST   /api/jobs/{id}/parts         {parts, api_key?}: only while the job waits for its part list (else
                                      409); the list (400 with the reason when it breaks a rule) becomes
                                      the confirmed list, and the build goes on from the parts stage
  POST   /api/jobs/{id}/continue      {agent?, api_key?, from?}: build on from sam3d, or retry from a
                                      stage ("frames": analyze the input again); another agent than the
                                      job's reruns the agent stage and the stages after it. While the job
                                      waits for its part list, a continue without "from" confirms the
                                      VLM's list as it is; a retry of a job that was cancelled at its
                                      part list stops at the list again (it answers "from": "propose"). A
                                      run that starts at or before select (or from lidar, when the scale
                                      is stale: another calibration or estimator) needs the API key when
                                      the agent stage is done, since a done job has dropped it, and answers
                                      400 when the job's LiDAR would fail the lidar stage's checks
                                      (jobs.lidar_settings), before anything is deleted
  POST   /api/jobs/{id}/cancel
  GET    /api/jobs/{id}/events        SSE: "state" (status and stages) when it changes, "log" (whole
                                      lines of the running stage's log; id <stage>:<byte offset>, so a
                                      reconnect resumes with Last-Event-ID), "end" when the job stops
  GET    /api/jobs/{id}/file/{path}   only the paths that ALLOW matches; refined.glb is the agent's GLB
                                      (steps.agent_glb), never the Stage A or split parts' GLB, and
                                      agent/evidence.json not when its source held the key (resolve). A Range
                                      request (a video's seek) gets 206 with Content-Range, and 416
                                      when it starts past the end (Starlette's FileResponse). HEAD gets the
                                      headers alone

picks and view0 of the start request are frames as the user clicked them; video extract snaps them
to the sharpest frame within 0.25 s (frames/pool.json "picks"). The overrides name pool frames
("f00012").

The API key of the job's agent (OpenAI for astra, Anthropic for opus) arrives in the start,
overrides, parts and continue bodies. It is checked with jobs.check_key before anything runs and
then lives only in the runner's memory (jobs.py). The VLM that lists the parts is the agent's model,
so it takes the same key. A job with "agent": "none" takes no key. With "key_later": true and no
api_key, start, parts and continue go on without the key: the job asks for it when it reaches a stage
that needs it (jobs.NO_KEY_LATER), and the answer's "key" says KEY_LATER. Bodies are parsed by hand, and
FastAPI's telemetry is off. Only the reason for a bad part list quotes the body: it names the parts,
and it is cut to PART_REASON characters.
"""

import argparse
import hashlib
import json
import math
import os
import re
import secrets
import shutil
import signal
import socket
import time
from contextlib import asynccontextmanager, suppress
from dataclasses import replace
from pathlib import Path

from harpor.fileio import config_dir, read_json, write_json
from harpor.webapp import jobs, steps

STATIC = Path(__file__).resolve().parent / "static"
COOKIE = "gator_webapp"
MAX_UPLOAD = 20 << 30
FREE_FACTOR = {"stray": 2.2}  # free space an upload needs, times its size (+ 2 GB); 1.2 for the other kinds
# Path segments are words, so no "..", absolute path or backslash ever matches.
ALLOW = re.compile(r"frames/thumbs/\d+\.jpg|frames/topic_previews/\d+\.jpg|frames/topics\.json|frames/preview\.mp4"
                   r"|pool/lidar\.json|lidar/check/f\d{5,9}\.jpg|lidar/cloud/f\d{5,9}\.glb"
                   r"|scene/sam3_\w+/check/[\w-]+\.jpg"
                   r"|run/input/images/\d{1,3}\.png|run/lidar/\d{1,3}\.glb|run/overlay_\d+\.png|run/posed\.glb"
                   r"|run/parts\.glb|run/parts\.json|run/parts/sam3/check/\d+\.jpg"
                   r"|refined\.glb|physics/aligned\.glb|turntable/turntable\.mp4|softvideo/softvideo\.mp4"
                   r"|physics/summary\.json|physics/mjcf/mjcf\.json|physics/urdf/model\.urdf|run/placement\.json"
                   r"|selection\.json|results\.zip|agent/evidence\.json|logs/\w+\.log")
TYPES = {".glb": "model/gltf-binary", ".jpg": "image/jpeg", ".png": "image/png", ".json": "application/json",
         ".zip": "application/zip", ".log": "text/plain; charset=utf-8", ".urdf": "application/xml",
         ".mp4": "video/mp4"}
# run/parts.* exist only for an articulated job; mjcf.json and model.urdf for every job that got that far.
# pool/lidar.json only for a job with LiDAR. turntable/turntable.mp4 only when the job asked for the 360 video and
# its render worked; softvideo/softvideo.mp4 only when the MuJoCo model has a soft part and its video worked.
# agent/evidence.json only for an articulation whose requirements ran in rounds.
OUTPUTS = ("run/posed.glb", "run/parts.glb", "run/parts.json", "refined.glb", "physics/aligned.glb",
           "turntable/turntable.mp4", "softvideo/softvideo.mp4", "physics/summary.json", "physics/mjcf/mjcf.json",
           "physics/urdf/model.urdf", "run/placement.json", "selection.json", "pool/lidar.json", "agent/evidence.json",
           "results.zip")
INSTANCE = re.compile(r"none|\d{1,3}(,\d{1,3})*")
MAX_CALIBRATION = 64 << 10  # bytes of a calibration file
CALIBRATION_SUFFIXES = (".yaml", ".yml", ".json")
LIDAR_COUNTS = ("views", "views_with_points", "views_used", "outliers", "scattered", "few_points", "no_mask",
                "no_samples")
PART_REASON = 200  # the longest reason for a bad part list (part_reason)
KEY_LATER = "no key yet: the job asks for it when it needs it"  # the answer to a body with key_later
NO_TELEMETRY = {"tracing": False, "metrics": False, "logs": False, "operation_spans": False, "auto_configure": False}


class Bad(Exception):
    """A request the server refuses: (HTTP status, message). Messages never hold the API key."""


def agent_of(b, job=None):
    """The agent a body names ("agent": astra, opus or none), else the job's (job: the job.json dict, when
    given), else astra; 400 for anything else."""
    if job is not None and b.get("agent") is None:
        return steps.agent(job)
    try:
        return steps.agent({"agent": b.get("agent", "astra")})
    except ValueError:
        raise Bad(400, f"agent is {', '.join(steps.AGENTS)}") from None


def key_later(b):
    """True when a start, continue or parts body says "key_later": true and holds no API key: the build
    goes on without the key, and the job asks for it when a stage needs it (jobs.NO_KEY_LATER). A key in
    the body wins. 400 when key_later is not true or false."""
    later = b.get("key_later", False)
    if type(later) is not bool:
        raise Bad(400, "key_later is true or false")
    return later and not b.get("api_key")


def on_stop(app, hook):
    """Run hook() (no arguments, in a thread) when the server stops, before the runner cancels the
    stages. app is make_app's app or the FastAPI app inside it. Hooks run in the order they were added;
    one that fails does not keep the others or the shutdown from running."""
    (app.app if isinstance(app, TokenGate) else app).state.stop_hooks.append(hook)


def part_reason(e):
    """The reason harpor.parts.propose gives for a bad part list. It quotes the parts, so every long quoted
    value is shortened first (the rule after it stays), then the whole text is cut."""
    text = re.sub(r"""(?:^|(?<=[\s(\[,:]))(['"])([^'"]{37})[^'"]{4,}\1(?=[\s,.;:)\]]|$)""", r"\1\2...\1", str(e))
    return text if len(text) <= PART_REASON else text[:PART_REASON - 3] + "..."


def content_length(request):
    """The request's Content-Length in bytes, else None (missing, or not a whole number). One of more
    than 15 digits counts as 10**15, which is over every limit here."""
    size = request.headers.get("content-length", "")
    if not re.fullmatch(r"[0-9]+", size):
        return None
    size = size.lstrip("0") or "0"
    return 10**15 if len(size) > 15 else int(size)


def calibration_summary(cal):
    """The short form of job.json "calibration" for the page, or None: name, bytes, translation_m (the
    length of T_cam_lidar's translation), rotation_deg (the angle of its rotation), has_intrinsics (K
    is given), camera_axes, time_offset_s (None when the file gives none) and lidar_topic (the LiDAR
    topic it was uploaded for, or None)."""
    if not cal:
        return None
    T = cal["T_cam_lidar"]
    c = max(-1.0, min(1.0, (T[0][0] + T[1][1] + T[2][2] - 1) / 2))
    return {"name": cal.get("name"), "bytes": cal.get("bytes"),
            "translation_m": round(math.hypot(T[0][3], T[1][3], T[2][3]), 4),
            "rotation_deg": round(math.degrees(math.acos(c)), 2), "has_intrinsics": cal.get("K") is not None,
            "camera_axes": cal.get("camera_axes"), "time_offset_s": cal.get("time_offset_s"),
            "lidar_topic": cal.get("lidar_topic")}


def lidar_stale(J):
    """Why the scale in J/pool/lidar.json is no longer the job's, or None (no record, not ok, or current):
    "calibration" when the calibration was uploaded or deleted since the scale was found, else
    "estimator" (steps.done also wants its source's current lidar_scale.VERSIONS). The page names the
    cause, so a scale that an older estimator found does not say that the calibration changed."""
    rec = read_json(Path(J) / "pool" / "lidar.json")
    if not isinstance(rec, dict) or rec.get("ok") is not True or steps.done("lidar", J):
        return None
    cal = (jobs.load(J).get("calibration") or {}).get("sha256")
    return "calibration" if (rec.get("extrinsic") or {}).get("calibration_sha256") != cal else "estimator"


def lidar_compact(rec, stale=None):
    """pool/lidar.json in short for the review page, or None: ok, s, reason, the view counts, spread_rel,
    source, the extrinsic's source, calibration_changed and estimator_changed (stale, lidar_stale's cause:
    the record is ok but steps.done says that the calibration changed since, or that an older estimator
    made it), warnings, every view's name, s, points_used and status, and the folder of the check images
    (lidar/check/<view>.jpg)."""
    if not isinstance(rec, dict):
        return None
    c = rec.get("combine") or {}
    return {"ok": rec.get("ok") is True, "s": rec.get("s"), "reason": rec.get("reason"),
            "counts": {k: c.get(k) for k in LIDAR_COUNTS}, "spread_rel": c.get("spread_rel"),
            "source": rec.get("source"), "extrinsic": (rec.get("extrinsic") or {}).get("source"),
            "calibration_changed": stale == "calibration", "estimator_changed": stale == "estimator",
            "warnings": rec.get("warnings") or [],
            "views": [{k: v.get(k) for k in ("name", "s", "points_used", "status")} for v in rec.get("views") or []],
            "check_dir": "lidar/check"}


def preview_of(J):
    """frames/meta.json "preview" for the page, or None before the frames stage wrote meta.json.
    {"error": why} when the frames stage could not write the preview, {"missing": true} when the file is gone
    since. Else the record of video.Preview without "moved", and "seek": {pick's frame: {frame, at}} for every
    pick of selection.json: the frame that the preview shows for it (the kept frame at or before it, since the
    preview keeps every stride-th frame) and that frame's time in the preview, in s."""
    meta = read_json(J / "frames" / "meta.json")
    rec = meta.get("preview") if isinstance(meta, dict) else None
    if not isinstance(rec, dict):
        return None
    if rec.get("error") is not None:
        return {"error": str(rec["error"])[:300]}
    if not resolve(J, "frames/preview.mp4"):
        return {"missing": True}
    out, seek = {k: v for k, v in rec.items() if k != "moved"}, {}
    times, moved, stride = meta.get("times") or [], rec.get("moved") or {}, max(1, int(rec.get("stride") or 1))
    sel = read_json(J / "selection.json")
    for pk in (sel.get("picks") if isinstance(sel, dict) else None) or []:
        f = pk.get("frame") if isinstance(pk, dict) else None
        if type(f) is int and 0 <= f < len(times):
            k = f - f % stride
            seek[str(f)] = {"frame": k, "at": moved.get(str(k), round(max(0.0, times[k] - (rec.get("t0") or 0)), 4))}
    return {**out, "seek": seek}


def key_in(J, rel):
    """Whether the agent result of job folder J lists rel, a file of the session folder, in key_found_in."""
    r = read_json(J / "agent" / "result.json")
    return isinstance(r, dict) and rel in (r.get("key_found_in") or [])


def resolve(J, rel):
    """The file that rel names in job folder J, if ALLOW has it and it stays inside J, else None. The pool's LiDAR
    points (lidar/cloud/) only while the scale has failed (pool_clouds), and the agent's evidence (agent_step) only
    while its source, the session's control/evidence.json, does not hold the API key."""
    if not ALLOW.fullmatch(rel) or rel.startswith("lidar/cloud/") and not scale_failed(J):
        return None
    if rel == "agent/evidence.json" and key_in(J, "control/evidence.json"):
        return None
    p = J / rel
    if rel == "refined.glb":
        p = steps.agent_glb(J)  # the agent's own GLB, never a file that holds the key (agent result key_found_in)
        if p is None:
            return None
    try:
        real = p.resolve(strict=True)
    except OSError:
        return None
    return real if real.is_file() and real.is_relative_to(J.resolve()) else None


def lidar_points(J):
    """{run/lidar/<i>.glb: the file} of the run's views that have LiDAR points (views.build_run), by i."""
    d = J / "run" / "lidar"
    rels = [f"run/lidar/{p.name}" for p in d.glob("*.glb")] if d.is_dir() else []
    rels = sorted((r for r in rels if ALLOW.fullmatch(r)), key=lambda r: int(Path(r).stem))
    return {r: p for r in rels if (p := resolve(J, r))}


def scale_failed(J):
    """True when pool/lidar.json says that the LiDAR scale failed (ok false)."""
    rec = read_json(J / "pool" / "lidar.json")
    return isinstance(rec, dict) and rec.get("ok") is False


def pool_clouds(J):
    """{lidar/cloud/<view>.glb: the file} of the pool's views that have LiDAR points (lidar.write_clouds), by name,
    while the scale has failed: the Inputs tab shows them, since there are no views used. With a scale, the views
    used have theirs in run/lidar (lidar_points)."""
    if not scale_failed(J):
        return {}
    rels = sorted(f"lidar/cloud/{p.name}" for p in (J / "lidar" / "cloud").glob("*.glb"))
    return {r: p for r in rels if (p := resolve(J, r))}


def sizes_of(found):
    """{path: bytes} of the files in found ({path: the file that resolve gave, or None}), one stat each, for
    the Downloads tab. A file that goes between resolve and the stat is left out."""
    out = {}
    for rel, p in found.items():
        if p is not None:
            with suppress(OSError):
                out[rel] = p.stat().st_size
    return out


def read_log(log, pos, limit=1 << 16):
    """(text, new pos): the whole lines of log after byte pos (a cut line if one is longer than limit)."""
    try:
        with open(log, "rb") as f:
            f.seek(pos)
            data = f.read(limit)
    except OSError:
        return "", pos
    cut = data.rfind(b"\n") + 1 or (len(data) if len(data) == limit else 0)
    return data[:cut].decode(errors="replace"), pos + cut


def log_start(log):
    """Byte offset of the newest run in a stage log (its "== <stage> <time>" line)."""
    try:
        data = Path(log).read_bytes()
    except OSError:
        return 0
    return data.rfind(b"\n== ") + 1


def started_after(job, t):
    """(started, stage) of the stage run that started first after time t, else None."""
    runs = [(st["started"], name) for name, st in job["stages"].items() if st.get("started") and st["started"] > t]
    return min(runs) if runs else None


def sse(event, data, id=None):
    return (f"id: {id}\n" if id else "") + f"event: {event}\ndata: {json.dumps(data)}\n\n"


def state(J, job):
    stages = [{k: v for k, v in s.items() if k != "seconds"} for s in jobs.progress(J)]
    return {"status": job["status"], "current": job["current"], "error": job["error"], "warnings": job["warnings"],
            "key_ok": job["key_ok"], "agent": steps.agent(job), "articulated": steps.articulated(job),
            "parts_mode": steps.parts_mode(job), "parts_review": steps.waits_for_parts(J, job), "stages": stages}


def tick(J, cursor, last):
    """One look at job J for its event stream. cursor is (stage, byte offset in its log, the start
    time of that stage run) or None. Stage logs are followed in the order the runs started, so a
    stage that ran between two looks is not skipped. Returns (events, cursor, last state, ended)."""
    out = []
    job = jobs.load(J)  # before the log: a stage that started after this one means this log is complete
    if cursor is None:  # a new client: from the newest run of the stage that started last (the running one)
        run = max(((s["started"], n) for n, s in job["stages"].items() if s.get("started")), default=None)
        cursor = (run[1], log_start(J / "logs" / f"{run[1]}.log"), run[0]) if run else None
    moved = False
    if cursor:
        stage, pos, t = cursor
        text, pos = read_log(J / "logs" / f"{stage}.log", pos)
        if text:
            out.append(sse("log", {"stage": stage, "text": text}, f"{stage}:{pos}"))
            cursor = (stage, pos, t)
        elif nxt := started_after(job, t):
            start = log_start(J / "logs" / f"{nxt[1]}.log")
            # The same stage again (a retry): the log only grows, so pos may already be past the new run's
            # header, and what was sent from there must not be sent twice.
            cursor, moved = (nxt[1], max(pos, start) if nxt[1] == stage else start, nxt[0]), True
    now = state(J, job)
    if now != last:
        out.append(sse("state", now))
    ended = not out and not moved and job["status"] not in jobs.ACTIVE
    if ended:
        out.append(sse("end", {"status": job["status"]}))
    return out, cursor, now, ended


def parse_event_id(J, value):
    """The cursor (see tick) of a Last-Event-ID "<stage>:<offset>", else None."""
    stage, _, pos = (value or "").partition(":")
    ok = stage in steps.NAMES and re.fullmatch(r"[0-9]{1,15}", pos)  # not int("²"), nor a seek past what a file holds
    t = jobs.load(J)["stages"][stage]["started"] if ok else None
    return (stage, int(pos), t) if t else None  # a stage that was reset since: start over as a new client


def make_app(cfg, token, key_dir=None):
    """The ASGI app (FastAPI behind the token gate) with its own jobs.Runner for cfg. The FastAPI app is the
    gate's .app; its state.runner is the Runner, and on_stop adds to its state.stop_hooks."""
    import anyio
    from fastapi import FastAPI, Request
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
    from starlette.concurrency import run_in_threadpool
    from starlette.requests import ClientDisconnect

    from harpor.webapp import demos

    shown = demos.load(cfg.demos)  # before the Runner claims the runs folder; DemoError says what is wrong
    soon = demos.placeholders(cfg.demos, shown)
    demo_files = {tuple(rel.split("/")[1:]): p for rel, p in demos.files(cfg.demos, shown).items()}
    runner = jobs.Runner(cfg, key_dir=key_dir)

    @asynccontextmanager
    async def lifespan(app):
        yield
        for hook in list(app.state.stop_hooks):  # on_stop: a worker stops before the stages are cancelled
            try:
                await run_in_threadpool(hook)
            except Exception as e:
                print(f"a stop hook failed: {e!r}", flush=True)
        await run_in_threadpool(runner.shutdown)  # no stage keeps the GPU after the server stops

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan, telemetry=NO_TELEMETRY)
    app.state.runner = runner
    app.state.stop_hooks = []  # on_stop adds to it
    app.state.demos = shown

    @app.exception_handler(Bad)
    async def bad(request, e):
        return JSONResponse({"detail": e.args[1]}, status_code=e.args[0])

    def job_dir(job_id):
        try:
            return runner.job_dir(job_id)
        except ValueError:
            raise Bad(404, "no such job") from None

    async def body(request):
        try:
            data = json.loads(await request.body() or b"{}")
        except (ValueError, RecursionError):  # RecursionError: nested deeper than the parser goes
            raise Bad(400, "the request body is not JSON") from None
        if not isinstance(data, dict):
            raise Bad(400, "the request body is not a JSON object")
        return data

    def runner_call(fn, *args, **kw):
        """A runner request; its ValueError means the job's state does not allow it."""
        try:
            return fn(*args, **kw)
        except ValueError as e:
            raise Bad(409, str(e)) from None

    def check_key(key, agent):
        ok, msg = runner.check_key(key, agent)
        if not ok:
            raise Bad(400, msg)
        return msg

    def idle(J):
        job = jobs.load(J)
        if job["status"] in jobs.ACTIVE:
            raise Bad(409, f"the job is {job['status']}")
        return job

    def lidar_check(J, job, first):
        """(stale, checked): whether the job's scale is stale (it has LiDAR and its lidar stage is not done:
        Runner.resume then starts from lidar), and {"sha256": the calibration's} for Runner.resume's
        checked_calibration (None without LiDAR). 400 with the reason when a run from first includes the
        lidar stage (first is at or before it, or the scale is stale) and the job's LiDAR fails that stage's
        checks (jobs.lidar_settings): the run deletes the results before that stage could fail. The checks
        include the camera_info, read from the bag when the calibration has no K, so they run outside the
        runner's lock, as Start's do. A calibration that another tab changes in between (which also makes
        the scale stale) is refused under the lock, by the sha256 that this check saw."""
        if steps.lidar(job) is None:
            return False, None
        stale = not steps.done("lidar", J)
        if stale or steps.NAMES.index(first) <= steps.NAMES.index("lidar"):
            try:
                jobs.lidar_settings(J, job, job["lidar"])
            except ValueError as e:
                raise Bad(400, str(e)) from None
        return stale, {"sha256": (job.get("calibration") or {}).get("sha256")}

    # ---- pages ----

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/help.html")
    def help_page():  # the Help link opens it in a new tab
        return FileResponse(STATIC / "help.html", headers={"Cache-Control": "no-cache"})

    @app.get("/demos.html")
    def demos_page():  # the header's Demos link opens it in a new tab
        return HTMLResponse(demos.page((STATIC / "demos.html").read_text(), shown, soon),
                            headers={"Cache-Control": "no-cache"})

    @app.api_route("/demos/{name}/{file}", methods=["GET", "HEAD"])
    def demo_file(name: str, file: str):
        p = demo_files.get((name, file))
        if p is None:
            raise Bad(404, "no such demo file")
        return FileResponse(p, media_type=demos.TYPES[p.suffix], headers={"Cache-Control": "no-cache"})

    @app.get("/static/{name}")
    def static(name: str):  # no-cache, so the browser takes a new app.js after an update
        if name not in ("app.js", "style.css", "demos.js"):
            raise Bad(404, "no such file")
        return FileResponse(STATIC / name, headers={"Cache-Control": "no-cache"})

    # ---- jobs ----

    @app.put("/api/upload")
    async def upload(request: Request, name: str = ""):
        if not 1 <= len(name) <= 200:
            raise Bad(400, "name the file: ?name=<file name>")
        kind = jobs.kind(jobs.safe_name(name))  # the name the file gets, which create_job classifies
        if kind is None:
            raise Bad(415, f"upload a video ({' '.join(jobs.VIDEOS)}), a rosbag ({' '.join(jobs.BAGS)}) or a Stray "
                           f"Scanner export ({' '.join(jobs.STRAY)}); {Path(name).suffix or 'a file without a suffix'} "
                           "is none of these")
        too_big = f"the file is larger than {MAX_UPLOAD >> 30} GB"
        size = content_length(request)
        if size is None:  # a chunked upload would skip the free-space check; the page always sends one
            raise Bad(411, "send the file with a Content-Length header")
        if size > MAX_UPLOAD:
            raise Bad(413, too_big)
        free, need = shutil.disk_usage(runner.runs).free, FREE_FACTOR.get(kind, 1.2) * size + (2 << 30)
        if free < need:
            raise Bad(413, f"the server has {free / 2**30:.1f} GB free and this upload needs {need / 2**30:.1f} GB")
        job_id, path = await run_in_threadpool(runner.new_upload, name)  # no delete until upload_done
        part, n = path.with_name(path.name + ".part"), 0
        try:
            fd = os.open(part, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            async with await anyio.open_file(fd, "wb") as f:  # streamed in chunks, never whole in memory
                async for chunk in request.stream():
                    n += len(chunk)
                    if n > MAX_UPLOAD:
                        raise Bad(413, too_big)
                    await f.write(chunk)
            if not n:
                raise Bad(400, "the upload is empty")
            os.replace(part, path)
        except BaseException as e:  # also a dropped connection: no half file and no job stays
            runner.upload_done(job_id, None)
            if isinstance(e, ClientDisconnect):
                return JSONResponse({"detail": "the upload was cut off"}, status_code=400)
            raise
        runner.upload_done(job_id, n)  # starts the analysis
        return {"id": job_id, "bytes": n}

    @app.get("/api/jobs")
    def list_jobs():
        out = []
        for f in sorted(runner.runs.glob("*/job.json"), reverse=True):
            job = read_json(f) if jobs.ID.fullmatch(f.parent.name) else None
            if job:
                status = "uploading" if runner.uploading(job["id"]) else job["status"]
                out.append({"id": job["id"], "created": job["created"], "status": status,
                            "prompt": job["prompt"], "video": job["video"]["name"], "kind": steps.source(job)["kind"]})
        return out

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        J = job_dir(job_id)
        job = jobs.load(J)
        # parts_review: the job waits for its part list.
        job.update(kind=steps.source(job)["kind"], parts_mode=steps.parts_mode(job),
                   parts_review=steps.waits_for_parts(J, job),
                   calibration=calibration_summary(job.get("calibration")), preview=preview_of(J))
        # The scale was found with the job's calibration as it is now (steps.done compares their sha256), and by
        # this estimator (lidar_scale.VERSIONS, by source); lidar_stale says which of the two changed.
        ran = job["stages"]["lidar"]["status"] == "done"
        job["lidar_current"] = steps.done("lidar", J) if job["lidar"] and ran else None
        job["lidar_stale"] = lidar_stale(J) if job["lidar_current"] is False else None
        job["progress"] = jobs.progress(J)
        found = {f: resolve(J, f) for f in OUTPUTS}
        found.update(lidar_points(J))
        found.update(pool_clouds(J))
        job["files"] = [f for f, p in found.items() if p]
        job["sizes"] = sizes_of(found)
        r = read_json(J / "agent" / "result.json")
        if r and "result.json" in (r.get("key_found_in") or []):  # the agent's own text held the key
            r = {**r, "reason": "left out: the agent's result held the API key", "limitations": []}
        keys = ("source", "verdict", "reason", "limitations", "exit")
        # turns, tokens and cost_usd are only in Claude's result, joints_delivered and simulations only in an
        # articulation's, rounds, requirements and candidate only in one that ran in rounds, and surface only in that
        # one's and a refinement's (controller.run_session). An older result has none of the last five.
        extra = ("agent", "model", "turns", "tokens", "cost_usd", "mode", "counts", "joints_delivered", "simulations",
                 "rounds", "requirements", "candidate", "surface")
        job["agent_result"] = {**{k: r.get(k) for k in keys}, **{k: r[k] for k in extra if k in r}} if r else None
        return job

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str):
        job_dir(job_id)
        runner_call(runner.delete, job_id)
        return {"deleted": job_id}

    @app.get("/api/jobs/{job_id}/frames")
    def get_frames(job_id: str):
        J = job_dir(job_id)
        meta = read_json(J / "frames" / "meta.json")
        if meta is None:
            raise Bad(404, "the frames are not analyzed yet")
        pool = read_json(J / "frames" / "pool.json")
        return {**meta, **({"pool": pool} if pool else {})}

    @app.get("/api/jobs/{job_id}/thumb/{idx}")
    def thumb(job_id: str, idx: int):
        f = job_dir(job_id) / "frames" / "thumbs" / f"{idx}.jpg"
        if not f.is_file():
            raise Bad(404, f"no thumbnail of frame {idx}")
        return FileResponse(f, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})

    def set_topic(job_id, b):
        J = job_dir(job_id)
        job = idle(J)
        if steps.source(job)["kind"] != "rosbag":
            raise Bad(409, f"this job's input is {jobs.INPUTS.get(steps.source(job)['kind'], 'a video')}, not a rosbag")
        if job["status"] in ("review", "done"):
            raise Bad(409, f"the job is {job['status']}")
        found = [t["topic"] for t in (read_json(J / "frames" / "topics.json") or {}).get("topics", [])]
        if b.get("topic") not in found:
            raise Bad(400, "the topic is not one of the bag's camera topics (frames/topics.json)")
        runner_call(runner.submit_analyze, job_id, topic=b["topic"])
        return {"status": "analyzing", "topic": b["topic"]}

    @app.post("/api/jobs/{job_id}/topic")
    async def topic(job_id: str, request: Request):
        return await run_in_threadpool(set_topic, job_id, await body(request))

    def start_job(job_id, b):
        J = job_dir(job_id)
        job = idle(J)
        meta = read_json(J / "frames" / "meta.json")
        if meta is None:
            raise Bad(409, "the input is not analyzed yet")
        n = len(meta["times"])
        mode, prompt, n_views = b.get("mode"), b.get("prompt"), b.get("n_views", 6)
        picks, view0, auto_continue = b.get("picks") or [], b.get("view0"), b.get("auto_continue", False)
        if mode not in ("manual", "auto"):
            raise Bad(400, "mode is manual or auto")
        if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 60:
            raise Bad(400, "the object noun has 1 to 60 characters")
        if not re.search(r"[A-Za-z0-9]", prompt):  # sam3_masks.slug needs one
            raise Bad(400, "the object noun needs a letter or digit")
        if type(n_views) is not int or not 2 <= n_views <= 12:
            raise Bad(400, "the number of views is 2 to 12")
        if not isinstance(picks, list) or any(type(p) is not int or not 0 <= p < n for p in picks) \
                or len(set(picks)) < len(picks):
            raise Bad(400, f"picks are different frames from 0 to {n - 1}")
        if mode == "manual" and not 2 <= len(picks) <= 12:
            raise Bad(400, "pick 2 to 12 frames")
        if view0 is not None and (type(view0) is not int or view0 not in (picks if mode == "manual" else range(n))):
            raise Bad(400, "view 0 is one of the picked frames")
        if type(auto_continue) is not bool:
            raise Bad(400, "auto_continue is true or false")
        turntable = b.get("turntable", False)
        if type(turntable) is not bool:
            raise Bad(400, "turntable is true or false")
        agent, later = agent_of(b), key_later(b)  # astra when the body names none
        articulated, parts = b.get("articulated", False), {"mode": b.get("parts_mode"), "list": b.get("parts") or []}
        if not isinstance(parts["list"], list):
            raise Bad(400, "parts is a list of parts")
        try:  # the settings the build would store; the typed list must pass harpor.parts.propose.validate
            jobs.parts_settings({**job, "prompt": prompt.strip(), "agent": agent, "articulated": articulated,
                                 "parts": parts})
        except ValueError as e:
            raise Bad(400, part_reason(e)) from None
        # A Stray job keeps its own setting (on when it was made) unless the body says otherwise.
        value = b.get("lidar") if "lidar" in b or steps.source(job)["kind"] != "stray" else steps.lidar(job)
        try:  # the LiDAR topic, its TF link (or the calibration) and the intrinsics
            lidar = jobs.lidar_settings(J, job, value)
        except ValueError as e:
            raise Bad(400, str(e)) from None
        checked = {"sha256": (job.get("calibration") or {}).get("sha256")}  # submit_build refuses another one
        if agent == "none":
            key, msg = None, "no key: the agent is turned off"
        elif later:  # the job asks for the key when it reaches a stage that needs it
            key, msg = None, KEY_LATER
        else:
            key, msg = b.get("api_key"), check_key(b.get("api_key"), agent)
        runner_call(runner.submit_build, job_id, key, checked_calibration=checked, mode=mode,
                    prompt=prompt.strip(), n_views=n_views, picks=picks if mode == "manual" else [], view0=view0,
                    auto_continue=auto_continue, agent=agent, overrides={"locked": [], "excluded": [], "instance": {}},
                    articulated=articulated, parts=parts, lidar=lidar, turntable=turntable)
        return {"status": "queued", "key": msg}

    @app.post("/api/jobs/{job_id}/start")
    async def start(job_id: str, request: Request):
        return await run_in_threadpool(start_job, job_id, await body(request))

    @app.get("/api/jobs/{job_id}/selection")
    def selection(job_id: str):
        J = job_dir(job_id)
        sel = read_json(J / "selection.json")
        if sel is None:
            raise Bad(404, "the views are not selected yet")
        job = jobs.load(J)
        sam = read_json(J / "scene" / f"sam3_{job['slug']}" / "sam3.json") or {"views": []}
        sel["sam3"] = {v["name"]: {"instances": [{"index": i["index"], "score": i["score"]} for i in v["instances"]],
                                   "picked": v.get("picked"), "pick": v.get("pick"), "mask": v.get("mask") is not None,
                                   "warnings": v.get("warnings") or []} for v in sam["views"]}
        sel["check_dir"] = f"scene/sam3_{job['slug']}/check"
        sel["overrides"], sel["view0"] = job["overrides"], job["view0"]
        sel["lidar"] = lidar_compact(read_json(J / "pool" / "lidar.json"), lidar_stale(J)) if steps.lidar(job) else None
        return sel

    def calibration_job(J, job_id):
        """The job.json of a rosbag job that may change its calibration now; 409 for another kind of input,
        while its upload runs, while it is analyzing, queued or running, and while a job with LiDAR waits for
        its part list: the stale scale would then block the list's confirmation (Runner.resume). The change
        calls it again under runner.locked(), so that no Start comes between the check and the change."""
        if runner.uploading(job_id):
            raise Bad(409, "the job is still uploading")
        job = idle(J)
        kind = steps.source(job)["kind"]
        if kind == "stray":
            raise Bad(409, "a Stray Scanner capture needs no calibration: its depth is aligned with the color camera")
        if kind != "rosbag":
            raise Bad(409, "a calibration is for the LiDAR of a rosbag job; this job's input is a video")
        if steps.lidar(job) is not None and steps.waits_for_parts(J, job):
            raise Bad(409, "the job waits for its part list; confirm the list or cancel the job first, then change "
                           "the calibration")
        return job

    def set_calibration(job_id, name, data, lidar_topic=None):
        from harpor.stage_a import lidar_scale

        J = job_dir(job_id)
        calibration_job(J, job_id)
        if not data:
            raise Bad(400, "the calibration file is empty")
        if lidar_topic:  # the LiDAR that the file's T is for
            why = jobs.lidar_topic_error(read_json(J / "frames" / "topics.json") or {}, lidar_topic)
            if why:
                raise Bad(400, why)
        safe = jobs.safe_name(name)
        try:
            parsed = lidar_scale.read_calibration(data, safe)
        except ValueError as e:
            raise Bad(400, str(e)[:300]) from None
        cal = {"name": safe, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
               "lidar_topic": lidar_topic or None, **parsed}
        # The check again and the write in one step: a Start or another upload may have come during the parse.
        with runner.locked():
            calibration_job(J, job_id)
            d = J / "lidar" / "calibration"
            d.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(J / "lidar", 0o700)
            for old in d.iterdir():  # one calibration at a time
                old.unlink()
            tmp = d / f".tmp_{safe}"
            with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as f:
                f.write(data)
            os.replace(tmp, d / safe)
            jobs.update(J, calibration=cal)
        return calibration_summary(cal)

    @app.put("/api/jobs/{job_id}/calibration")
    async def put_calibration(job_id: str, request: Request, name: str = "", lidar_topic: str = ""):
        job_dir(job_id)
        if not 1 <= len(name) <= 200:
            raise Bad(400, "name the file: ?name=<file name>")
        if Path(jobs.safe_name(name)).suffix.lower() not in CALIBRATION_SUFFIXES:
            raise Bad(415, f"upload the calibration as {', '.join(CALIBRATION_SUFFIXES)}")
        too_big = f"a calibration file has at most {MAX_CALIBRATION >> 10} KB"
        size = content_length(request)
        if size is not None and size > MAX_CALIBRATION:
            raise Bad(413, too_big)
        data = bytearray()
        async for chunk in request.stream():  # never more than the limit in memory
            data += chunk
            if len(data) > MAX_CALIBRATION:
                raise Bad(413, too_big)
        return await run_in_threadpool(set_calibration, job_id, name, bytes(data), lidar_topic)

    @app.delete("/api/jobs/{job_id}/calibration")
    def delete_calibration(job_id: str):
        J = job_dir(job_id)
        with runner.locked():  # the check and the change in one step, as for an upload
            calibration_job(J, job_id)
            shutil.rmtree(J / "lidar" / "calibration", ignore_errors=True)
            jobs.update(J, calibration=None)
        return {"calibration": None}

    def set_overrides(job_id, b):
        J = job_dir(job_id)
        job = idle(J)
        pool = read_json(J / "frames" / "pool.json")
        sam = read_json(J / "scene" / f"sam3_{job['slug']}" / "sam3.json")
        if not pool or not sam:
            raise Bad(409, "the frame pool has no masks yet")
        names = set(pool["names"].values())
        found = {v["name"]: len(v["instances"]) for v in sam["views"]}
        locked, excluded = b.get("locked") or [], b.get("excluded") or []
        instance, view0 = b.get("instance") or {}, b.get("view0")
        for x in (locked, excluded):
            if not isinstance(x, list) or not all(isinstance(n, str) and n in names for n in x):
                raise Bad(400, "locked and excluded are lists of pool frames such as f00012")
            if len(set(x)) < len(x):
                raise Bad(400, "locked and excluded name each pool frame once")
        if set(locked) & set(excluded):
            raise Bad(400, f"{sorted(set(locked) & set(excluded))} are both locked and excluded")
        if not isinstance(instance, dict):
            raise Bad(400, "instance maps pool frames to K, K,L or none")
        for name, k in instance.items():
            if name not in names or not isinstance(k, str) or not INSTANCE.fullmatch(k):
                raise Bad(400, "instance maps pool frames to K, K,L or none")
            if k != "none" and max(map(int, k.split(","))) >= found.get(name, 0):
                raise Bad(400, f"SAM 3 found {found.get(name, 0)} instances in {name}")
        if view0 is not None and (not isinstance(view0, str) or view0 not in names):
            raise Bad(400, "view 0 is a pool frame such as f00012")
        first = "masks" if instance != job["overrides"].get("instance", {}) else "select"
        _, checked = lidar_check(J, job, first)  # from masks, or from lidar when the scale is stale, as for continue
        agent = agent_of(b, job)
        key = (b.get("api_key") or None) if agent != "none" else None  # no key when the agent is off
        if key is not None:
            check_key(key, agent)
        first = runner_call(runner.resume, job_id, first, key, view0=view0, agent=agent, checked_calibration=checked,
                            overrides={"locked": locked, "excluded": excluded, "instance": instance})
        return {"status": "queued", "from": first}

    @app.post("/api/jobs/{job_id}/overrides")
    async def overrides(job_id: str, request: Request):
        return await run_in_threadpool(set_overrides, job_id, await body(request))

    @app.get("/api/jobs/{job_id}/parts")
    def get_parts(job_id: str):
        J = job_dir(job_id)
        job = jobs.load(J)
        anchor = (read_json(J / "selection.json") or {}).get("anchor")  # run view 0, the photo the VLM sees
        photo = f"scene/sam3_{job['slug']}/check/{anchor}.jpg" if job.get("slug") and isinstance(anchor, str) else None
        checks = sorted((J / "run" / "parts" / "sam3" / "check").glob("*.jpg"), key=lambda p: (len(p.stem), p.stem))
        return {"articulated": steps.articulated(job), "noun": job["prompt"],
                "parts": {**job["parts"], "mode": steps.parts_mode(job)},
                "waiting": steps.waits_for_parts(J, job), "proposal": read_json(J / "run" / "proposal.json"),
                "photo": photo if photo and resolve(J, photo) else None,
                "checks": [rel for rel in (p.relative_to(J).as_posix() for p in checks) if resolve(J, rel)]}

    def held_key(job_id, b, agent, later=False):
        """The body's API key for agent (checked), else None; 400 when the server holds none for the job,
        unless later (key_later: the job asks for the key when it needs it)."""
        key = (b.get("api_key") or None) if agent != "none" else None  # no key when the agent is off
        if key is not None:
            check_key(key, agent)
        elif agent != "none" and not later and not runner.has_key(job_id, agent):
            raise Bad(400, f"the server holds no {steps.PROVIDER[agent]} API key for this job (it restarted, the agent "
                           "changed, or the job finished, which drops the key); paste the key again")
        return key

    def asks(job_id, agent, later):
        """{"key": KEY_LATER} when a build went on with key_later and the runner holds no key for the
        job's agent, else {}."""
        return {"key": KEY_LATER} if later and agent != "none" and not runner.has_key(job_id, agent) else {}

    def set_parts(job_id, b):
        from harpor.parts import propose

        J = job_dir(job_id)
        job = idle(J)
        if not steps.waits_for_parts(J, job):
            raise Bad(409, "the job does not wait for its part list")
        if not isinstance(b.get("parts"), list):
            raise Bad(400, "parts is a list of parts")
        try:
            parts = propose.validate(b["parts"], job["prompt"])
        except ValueError as e:
            raise Bad(400, part_reason(e)) from None
        agent, later = steps.agent(job), key_later(b)
        key = held_key(job_id, b, agent, later)  # the agent stage runs after the parts
        first = runner_call(runner.resume, job_id, "parts", key, confirm_parts=True, parts_list=parts, key_later=later)
        return {"status": "queued", "from": first, **asks(job_id, agent, later)}

    @app.post("/api/jobs/{job_id}/parts")
    async def parts(job_id: str, request: Request):
        return await run_in_threadpool(set_parts, job_id, await body(request))

    def continue_job(job_id, b):
        J = job_dir(job_id)
        job = idle(J)
        first, agent, later = b.get("from") or "sam3d", agent_of(b, job), key_later(b)
        key = (b.get("api_key") or None) if agent != "none" else None  # no key when the agent is off
        if first == "frames":  # analyze the input again (it failed, was cancelled or the server stopped); no key
            if job["stages"]["frames"]["status"] == "done":
                raise Bad(409, "the input is already analyzed")
            if not Path(job["video"]["path"]).is_file():
                raise Bad(409, "the uploaded file is not on the server; upload it again")
            runner_call(runner.submit_analyze, job_id)
            return {"status": "analyzing", "from": "frames"}
        if first not in jobs.BUILD:
            raise Bad(400, f"from is frames or one of {jobs.BUILD}")
        changed = agent != steps.agent(job)  # the other agent's stage runs anew (Runner.resume)
        # Runner.resume runs every stage again from a start at or before select, also from lidar when the
        # scale is stale (the calibration or the estimator changed after the scale was found). A done agent
        # stage then needs the key, and the job's LiDAR must pass the lidar stage's checks (lidar_check).
        stale, checked = lidar_check(J, job, first)
        again = steps.NAMES.index(first) <= steps.NAMES.index(steps.REVIEW_AFTER) or stale
        if changed or again or job["stages"]["agent"]["status"] not in ("done", "skipped"):
            key = held_key(job_id, b, agent, later)
        elif key is not None:
            check_key(key, agent)
        else:
            later = False  # the agent stage is done: no stage needs a key
        # Build while the job waits for its part list confirms the VLM's list as it is.
        confirm = steps.waits_for_parts(J, job) and not b.get("from")
        first = runner_call(runner.resume, job_id, first, key, confirm_parts=confirm, agent=agent, key_later=later,
                            checked_calibration=checked)
        # from: the agent stage when the agent changed, parts when confirmed
        return {"status": "queued", "from": first, **asks(job_id, agent, later)}

    @app.post("/api/jobs/{job_id}/continue")
    async def continue_(job_id: str, request: Request):
        return await run_in_threadpool(continue_job, job_id, await body(request))

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        J = job_dir(job_id)
        runner_call(runner.cancel, job_id)
        return {"status": jobs.load(J)["status"]}

    @app.get("/api/jobs/{job_id}/events")
    def events(job_id: str, request: Request):
        J = job_dir(job_id)
        cursor = parse_event_id(J, request.headers.get("last-event-id"))

        async def stream():
            nonlocal cursor
            last, quiet = None, time.time()
            while True:
                out, cursor, last, ended = await run_in_threadpool(tick, J, cursor, last)
                for event in out:
                    yield event
                if ended:
                    return
                if out:
                    quiet = time.time()
                elif time.time() - quiet > 15:
                    yield ": ping\n\n"
                    quiet = time.time()
                await anyio.sleep(0.5)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.api_route("/api/jobs/{job_id}/file/{rel:path}", methods=["GET", "HEAD"])  # HEAD: the headers alone
    def file(job_id: str, rel: str):
        J = job_dir(job_id)
        p = resolve(J, rel)
        if p is None:
            raise Bad(404, "not a file of this job that the app serves")
        name = f"{job_id}_results.zip" if rel == "results.zip" else None
        return FileResponse(p, media_type=TYPES[Path(rel).suffix], filename=name, headers={"Cache-Control": "no-cache"})

    return TokenGate(app, token)


class TokenGate:
    """ASGI middleware: the login cookie passes; GET /?token=T sets it and redirects to /; else 401. The help page and
    its stylesheet pass without the cookie: they hold no job data, and their troubleshooting says how to log in."""

    OPEN = ("/help.html", "/static/style.css")

    def __init__(self, app, token):
        self.app, self.token = app, token.encode()

    def ok(self, value):
        return bool(value) and secrets.compare_digest(value.encode(errors="replace"), self.token)

    async def __call__(self, scope, receive, send):
        from starlette.requests import Request
        from starlette.responses import PlainTextResponse, RedirectResponse

        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        req = Request(scope)
        if scope["method"] == "GET" and scope["path"] == "/" and self.ok(req.query_params.get("token")):
            resp = RedirectResponse("/", status_code=303)  # the token leaves the address bar
            resp.set_cookie(COOKIE, self.token.decode(), httponly=True, samesite="strict")
            return await resp(scope, receive, send)
        if self.ok(req.cookies.get(COOKIE)) or (scope["method"] == "GET" and scope["path"] in self.OPEN):
            return await self.app(scope, receive, send)
        resp = PlainTextResponse("open the URL with ?token= that the server printed", status_code=401)
        return await resp(scope, receive, send)


def write_info(info):
    """~/.config/harpor/webapp.json (0600), replaced atomically, for start_ice.sh. The folder is made with
    mode 0700 when it is missing; an existing folder is used as it is."""
    d = config_dir()
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    write_json(d / "webapp.json", info, private=True, indent=1)
    return d / "webapp.json"


def remove_info(path, token):
    """Delete webapp.json if it is still this server's (a later start may have replaced it)."""
    with suppress(OSError, ValueError):
        if json.loads(Path(path).read_text()).get("token") == token:
            Path(path).unlink()


def _exit_on_sigterm(sig, frame):
    # uvicorn shuts down on SIGTERM and then raises the signal again with the handler it found, which
    # is this one; without it the process dies there and main's finally does not run.
    raise SystemExit(128 + sig)


def worker_of(local_keys):
    """The personal worker's session (session.Session, which takes the session file's lock). A worker that
    cannot start leaves the server in tunnel mode: it prints why and returns None."""
    from harpor.webapp import session, supa, worker

    if local_keys:
        worker.check_local_keys(local_keys)  # the server refuses to start with --local-keys and a bad key file
    try:
        return session.Session()
    except (SystemExit, supa.Error) as e:
        print(f"the personal worker does not start: {e}\n{worker.SIGNED_OUT}", flush=True)
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m harpor.webapp", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", help="folder of the job folders (default: WEBAPP_RUNS, else $G/webapp_runs)")
    ap.add_argument("--host", default="127.0.0.1", help="the address to listen on (default 127.0.0.1)")
    ap.add_argument("--port", type=int, default=0, help="0 takes a free port")
    ap.add_argument("--fake", action="store_true", help="run every stage with harpor.webapp.fake_steps (no GPU)")
    ap.add_argument("--worker", action="store_true",
                    help="also serve the public site: a personal worker on this GPU machine (sign in first: "
                         "python -m harpor.webapp.worker login)")
    ap.add_argument("--local-keys", metavar="AGENTS",
                    help="with --worker: astra, opus or astra,opus, whose API key this machine keeps in "
                         "~/.config/harpor/openai.env or anthropic.env")
    ap.add_argument("--demos", metavar="DIR", help="a demos folder to show at /demos.html (default: WEBAPP_DEMOS)")
    a = ap.parse_args(argv)
    local_keys = ()
    if a.local_keys is not None:
        if not a.worker:
            ap.error("--local-keys needs --worker")
        local_keys = tuple(dict.fromkeys(k.strip() for k in a.local_keys.split(",") if k.strip()))
        if not local_keys or set(local_keys) - {"astra", "opus"}:
            ap.error("--local-keys takes astra, opus or astra,opus")
    import uvicorn

    cfg = steps.config_from_env()
    cfg = replace(cfg, runs=Path(a.runs).resolve() if a.runs else cfg.runs, fake=cfg.fake or a.fake,
                  demos=Path(a.demos).expanduser().resolve() if a.demos else cfg.demos)
    # The socket is bound here, so the port is known (also for --port 0) before webapp.json is written.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((a.host, a.port))
    except OSError as e:
        raise SystemExit(f"cannot listen on {a.host}:{a.port}: {e}")
    port, node, token = sock.getsockname()[1], socket.getfqdn(), secrets.token_urlsafe(32)
    session = worker_of(local_keys) if a.worker else None  # before the Runner claims the runs folder
    from harpor.webapp import demos

    try:
        app = make_app(cfg, token)
    except demos.DemoError as e:
        raise SystemExit(f"the demos folder: {e}")
    w = None
    if session is not None:
        from harpor.webapp import worker

        w = worker.Worker(app, cfg, local_keys, session=session)
        on_stop(app, w.stop_thread)  # the worker stops before the runner cancels the stages
    signal.signal(signal.SIGTERM, _exit_on_sigterm)
    info = write_info({"node": node, "port": port, "token": token, "url_path": f"/?token={token}",
                       "slurm_job": os.environ.get("SLURM_JOB_ID"), "slurm_node": os.environ.get("SLURMD_NODENAME"),
                       "worker": w is not None})
    print(f"HARPOR web app on {node}:{port}, runs in {cfg.runs}" + (" (fake stages)" if cfg.fake else ""))
    print(f"login details in {info}")
    if cfg.demos:
        print(f"demos: {len(app.app.state.demos)} from {cfg.demos}")
    print(f"on this machine: http://localhost:{port}/?token={token}")
    target, via = ("127.0.0.1", "<this machine>") if a.worker or a.host == "127.0.0.1" else (node, "<login node>")
    print(f"from another computer: ssh -L {port}:{target}:{port} {via}, then open the same URL there", flush=True)
    if w is not None:
        print(f"the personal worker of {w.place} runs in this server", flush=True)
        w.start_thread()
    try:
        with suppress(KeyboardInterrupt):  # uvicorn shuts down on Ctrl-C, then raises it again
            uvicorn.Server(uvicorn.Config(app, log_level="info", access_log=False, timeout_graceful_shutdown=3)).run(
                sockets=[sock])
    finally:
        remove_info(info, token)


if __name__ == "__main__":
    main()
