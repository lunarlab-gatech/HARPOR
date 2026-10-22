"""The HARPOR web app's server: upload a phone video, a rosbag or a Stray Scanner capture, choose the
views, build a simulation-ready object.

Runs in envs/web (fastapi, uvicorn, av, numpy, pillow, scipy, rosbags) inside the ICE GPU job, from the
gator_clone root (start_ice.sh starts it there):

    python -m gator.webapp [--runs DIR] [--host 0.0.0.0] [--port 0] [--fake]

It makes a login token, writes ~/.config/gator/webapp.json (0600: node, port, token, url_path, and
slurm_job and slurm_node from SLURM_JOB_ID and SLURMD_NODENAME, so start_ice.sh reuses the server
only in the same job), prints the port forward for the Mac and the URL, and serves the app with one
uvicorn worker. When the server stops (Ctrl-C or SIGTERM), it deletes webapp.json if it still
holds this server's token.
--port 0 takes a free port. --fake runs every stage with gator.webapp.fake_steps, so the app works
on a Mac without GPUs (OpenAI keys that start with sk- and Anthropic keys that start with sk-ant-
pass). The runs folder defaults to steps.config_from_env(): WEBAPP_RUNS, else $G/webapp_runs.

Login: GET /?token=T sets an HttpOnly SameSite=Strict cookie and redirects to /. Every other
request needs the cookie and gets 401 without it, static files included.

Routes (JSON unless said otherwise; {id} is a job id):
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
                                      an articulation result adds mode, counts and joints_delivered), the
                                      output files that exist,
                                      "articulated", "parts_mode" and "parts_review" (the job waits for its
                                      part list), "lidar" (steps.lidar) and "calibration" (the uploaded
                                      calibration's summary, calibration_summary, or null)
  DELETE /api/jobs/{id}               the job folder, unless the job is uploading, analyzing, queued or
                                      running
  GET    /api/jobs/{id}/frames        frames/meta.json, with frames/pool.json as "pool" once extracted
  GET    /api/jobs/{id}/thumb/{idx}   a thumbnail (JPEG)
  POST   /api/jobs/{id}/topic         {topic}: a rosbag job's camera topic (one of frames/topics.json), in
                                      choose_topic, ready, failed or cancelled; analyzes the frames again.
                                      409 for a job whose input is not a rosbag
  PUT    /api/jobs/{id}/calibration?name=N
                                      a rosbag job's LiDAR-camera calibration as the raw body: .yaml, .yml
                                      or .json (else 415), at most 64 KB (else 413); 409 for a video job, for
                                      a Stray Scanner capture (its depth is aligned with the color camera)
                                      and while the job is analyzing, queued or running; 400 with the
                                      reason of lidar_scale.read_calibration. Saved as
                                      lidar/calibration/<name> (0600) and in job.json "calibration";
                                      answers calibration_summary. A lidar stage that ran with another
                                      calibration is then stale: the next Build or retry starts from
                                      lidar and stops in review (Runner.resume)
  DELETE /api/jobs/{id}/calibration   the calibration (409 for a job that is not a rosbag's, and while the
                                      job is analyzing, queued or running); a lidar stage that ran with it
                                      is stale, as after a PUT
  POST   /api/jobs/{id}/start         {mode, prompt, n_views, picks, view0, auto_continue, agent, api_key,
                                      articulated?, parts_mode?, parts?, lidar?} (agent astra: an OpenAI
                                      key; opus: an Anthropic key; none: no key, physics uses the Stage A
                                      mesh; true and false mean astra and none). articulated (default
                                      false) builds an object with moving parts; parts_mode
                                      (steps.PARTS_MODES, default vlm) says who lists them, and parts is the
                                      typed list of mode user ([{name, parent, joint, count}], checked with
                                      gator.parts.propose); agent none means mode user. lidar is null,
                                      {"topic": T}, a LiDAR topic of a rosbag job, or {"kind": "stray"}, the
                                      depth of a Stray Scanner capture, for the metric scale;
                                      jobs.lidar_settings checks it (400 with the reason). Without it a
                                      Stray job keeps its own (LiDAR on, unless a Start turned it off) and
                                      the other jobs get null
  GET    /api/jobs/{id}/selection     selection.json with "sam3": the SAM 3 instances of every pool frame,
                                      and "lidar": the metric scale's record in short (lidar_compact) for a
                                      job with LiDAR, else null; its calibration_changed is true when the
                                      calibration was uploaded or deleted after the scale was found
  POST   /api/jobs/{id}/overrides     {locked, excluded, instance, view0, agent?, api_key?}: select again
                                      (masks first when an instance changed); the job stops in review
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
                                      part list stops at the list again (it answers "from": "propose")
  POST   /api/jobs/{id}/cancel
  GET    /api/jobs/{id}/events        SSE: "state" (status and stages) when it changes, "log" (whole
                                      lines of the running stage's log; id <stage>:<byte offset>, so a
                                      reconnect resumes with Last-Event-ID), "end" when the job stops
  GET    /api/jobs/{id}/file/{path}   only the paths that ALLOW matches; refined.glb is the agent's GLB
                                      (steps.agent_glb), never the Stage A or split parts' GLB

picks and view0 of the start request are frames as the user clicked them; video extract snaps them
to the sharpest frame within 0.25 s (frames/pool.json "picks"). The overrides name pool frames
("f00012").

The API key of the job's agent (OpenAI for astra, Anthropic for opus) arrives in the start,
overrides, parts and continue bodies. It is checked with jobs.check_key before anything runs and
then lives only in the runner's memory (jobs.py). The VLM that lists the parts is the agent's model,
so it takes the same key. A job with "agent": "none" takes no key. Bodies are parsed by hand, and
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

from gator.webapp import jobs, steps

STATIC = Path(__file__).resolve().parent / "static"
COOKIE = "gator_webapp"
MAX_UPLOAD = 20 << 30
FREE_FACTOR = {"stray": 2.2}  # free space an upload needs, times its size (+ 2 GB); 1.2 for the other kinds
ACTIVE = ("analyzing", "queued", "running")
# Path segments are words, so no "..", absolute path or backslash ever matches.
ALLOW = re.compile(r"frames/thumbs/\d+\.jpg|frames/topic_previews/\d+\.jpg|frames/topics\.json"
                   r"|pool/lidar\.json|lidar/check/f\d{5,9}\.jpg"
                   r"|scene/sam3_\w+/check/[\w-]+\.jpg|run/overlay_\d+\.png|run/posed\.glb"
                   r"|run/parts\.glb|run/parts\.json|run/parts/sam3/check/\d+\.jpg"
                   r"|refined\.glb|physics/aligned\.glb|physics/summary\.json|physics/mjcf/mjcf\.json"
                   r"|physics/urdf/model\.urdf|run/placement\.json|selection\.json|results\.zip|logs/\w+\.log")
TYPES = {".glb": "model/gltf-binary", ".jpg": "image/jpeg", ".png": "image/png", ".json": "application/json",
         ".zip": "application/zip", ".log": "text/plain; charset=utf-8", ".urdf": "application/xml"}
# run/parts.* exist only for an articulated job; mjcf.json and model.urdf for every job that got that far.
# pool/lidar.json only for a job with LiDAR.
OUTPUTS = ("run/posed.glb", "run/parts.glb", "run/parts.json", "refined.glb", "physics/aligned.glb",
           "physics/summary.json", "physics/mjcf/mjcf.json", "physics/urdf/model.urdf", "run/placement.json",
           "selection.json", "pool/lidar.json", "results.zip")
INSTANCE = re.compile(r"none|\d{1,3}(,\d{1,3})*")
MAX_CALIBRATION = 64 << 10  # bytes of a calibration file
CALIBRATION_SUFFIXES = (".yaml", ".yml", ".json")
LIDAR_COUNTS = ("views", "views_with_points", "views_used", "outliers", "scattered", "few_points", "no_mask",
                "no_samples")
PART_REASON = 200  # the longest reason for a bad part list (part_reason)
FORWARD = "ssh -S ~/.ssh/cm-ice -O forward -L 8765:{node}:{port} ali497@login-ice.pace.gatech.edu"
NO_TELEMETRY = {"tracing": False, "metrics": False, "logs": False, "operation_spans": False, "auto_configure": False}


class Bad(Exception):
    """A request the server refuses: (HTTP status, message). Messages never hold the API key."""


def part_reason(e):
    """The reason gator.parts.propose gives for a bad part list. It quotes the parts, so every long quoted
    value is shortened first (the rule after it stays), then the whole text is cut."""
    text = re.sub(r"""(?:^|(?<=[\s(\[,:]))(['"])([^'"]{37})[^'"]{4,}\1(?=[\s,.;:)\]]|$)""", r"\1\2...\1", str(e))
    return text if len(text) <= PART_REASON else text[:PART_REASON - 3] + "..."


def calibration_summary(cal):
    """The short form of job.json "calibration" for the page, or None: name, bytes, translation_m (the
    length of T_cam_lidar's translation), rotation_deg (the angle of its rotation), has_intrinsics (K
    is given), camera_axes and time_offset_s."""
    if not cal:
        return None
    T = cal["T_cam_lidar"]
    c = max(-1.0, min(1.0, (T[0][0] + T[1][1] + T[2][2] - 1) / 2))
    return {"name": cal.get("name"), "bytes": cal.get("bytes"),
            "translation_m": round(math.hypot(T[0][3], T[1][3], T[2][3]), 4),
            "rotation_deg": round(math.degrees(math.acos(c)), 2), "has_intrinsics": cal.get("K") is not None,
            "camera_axes": cal.get("camera_axes"), "time_offset_s": cal.get("time_offset_s")}


def lidar_compact(rec, stale=False):
    """pool/lidar.json in short for the review page, or None: ok, s, reason, the view counts, spread_rel,
    source, the extrinsic's source, calibration_changed (stale: the record is ok but steps.done says the
    calibration changed since), warnings, every view's name, s, points_used and status, and the folder
    of the check images (lidar/check/<view>.jpg)."""
    if not isinstance(rec, dict):
        return None
    c = rec.get("combine") or {}
    return {"ok": rec.get("ok") is True, "s": rec.get("s"), "reason": rec.get("reason"),
            "counts": {k: c.get(k) for k in LIDAR_COUNTS}, "spread_rel": c.get("spread_rel"),
            "source": rec.get("source"), "extrinsic": (rec.get("extrinsic") or {}).get("source"),
            "calibration_changed": bool(stale and rec.get("ok") is True), "warnings": rec.get("warnings") or [],
            "views": [{k: v.get(k) for k in ("name", "s", "points_used", "status")} for v in rec.get("views") or []],
            "check_dir": "lidar/check"}


def lidar_of(job):
    """steps.lidar of a job, or job.json's own value when it is malformed (the lidar stage says why)."""
    try:
        return steps.lidar(job)
    except ValueError:
        return job.get("lidar")


def read_json(path):
    with suppress(OSError, ValueError):
        return json.loads(Path(path).read_text())
    return None


def resolve(J, rel):
    """The file that rel names in job folder J, if ALLOW has it and it stays inside J, else None."""
    if not ALLOW.fullmatch(rel):
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
    ended = not out and not moved and job["status"] not in ACTIVE
    if ended:
        out.append(sse("end", {"status": job["status"]}))
    return out, cursor, now, ended


def parse_event_id(J, value):
    """The cursor (see tick) of a Last-Event-ID "<stage>:<offset>", else None."""
    stage, _, pos = (value or "").partition(":")
    t = jobs.load(J)["stages"].get(stage, {}).get("started") if stage in steps.NAMES and pos.isdigit() else None
    return (stage, int(pos), t) if t else None  # a stage that was reset since: start over as a new client


def make_app(cfg, token, key_dir=None):
    """The ASGI app (FastAPI behind the token gate) with its own jobs.Runner for cfg."""
    import anyio
    from fastapi import FastAPI, Request
    from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
    from starlette.concurrency import run_in_threadpool
    from starlette.requests import ClientDisconnect

    runner = jobs.Runner(cfg, key_dir=key_dir)

    @asynccontextmanager
    async def lifespan(app):
        yield
        await run_in_threadpool(runner.shutdown)  # no stage keeps the GPU after the server stops

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan, telemetry=NO_TELEMETRY)
    app.state.runner = runner

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
        except ValueError:
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

    def agent_of(b, job=None):
        """The agent a body names ("agent": astra, opus, none, or true or false), else the job's."""
        if job is not None and b.get("agent") is None:
            return steps.agent(job)
        try:
            return steps.agent({"agent": b.get("agent", "astra")})
        except ValueError:
            raise Bad(400, f"agent is {', '.join(steps.AGENTS)}") from None

    def idle(J):
        job = jobs.load(J)
        if job["status"] in ACTIVE:
            raise Bad(409, f"the job is {job['status']}")
        return job

    # ---- pages ----

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/static/{name}")
    def static(name: str):  # no-cache, so the browser takes a new app.js after an update
        if name not in ("app.js", "style.css"):
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
        size = request.headers.get("content-length", "")
        if not size.isdigit():  # a chunked upload would skip the free-space check; the page always sends one
            raise Bad(411, "send the file with a Content-Length header")
        if int(size) > MAX_UPLOAD:
            raise Bad(413, too_big)
        free, need = shutil.disk_usage(runner.runs).free, FREE_FACTOR.get(kind, 1.2) * int(size) + (2 << 30)
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
        # Also for jobs made before these fields; parts_review: the job waits for its part list.
        job.update(source=steps.source(job), kind=steps.source(job)["kind"], agent=steps.agent(job),
                   articulated=steps.articulated(job), parts_mode=steps.parts_mode(job),
                   parts_review=steps.waits_for_parts(J, job), lidar=lidar_of(job),
                   calibration=calibration_summary(job.get("calibration")))
        job["progress"] = jobs.progress(J)
        job["files"] = [f for f in OUTPUTS if resolve(J, f)]
        r = read_json(J / "agent" / "result.json")
        if r and "result.json" in (r.get("key_found_in") or []):  # the agent's own text held the key
            r = {**r, "reason": "left out: the agent's result held the API key", "limitations": []}
        keys = ("source", "verdict", "reason", "limitations", "exit")
        # Older results lack some of them; joints_delivered is only in an articulation's result.
        extra = ("agent", "model", "turns", "tokens", "cost_usd", "mode", "counts", "joints_delivered")
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
        agent = agent_of(b)  # astra when the body names none
        articulated, parts = b.get("articulated", False), {"mode": b.get("parts_mode"), "list": b.get("parts") or []}
        if not isinstance(parts["list"], list):
            raise Bad(400, "parts is a list of parts")
        try:  # the settings the build would store; the typed list must pass gator.parts.propose.validate
            jobs.parts_settings({**job, "prompt": prompt.strip(), "agent": agent, "articulated": articulated,
                                 "parts": parts})
        except ValueError as e:
            raise Bad(400, part_reason(e)) from None
        # A Stray job keeps its own setting (on when it was made) unless the body says otherwise.
        value = b.get("lidar") if "lidar" in b or steps.source(job)["kind"] != "stray" else lidar_of(job)
        try:  # the LiDAR topic, its TF link (or the calibration) and the intrinsics
            lidar = jobs.lidar_settings(J, job, value)
        except ValueError as e:
            raise Bad(400, str(e)) from None
        msg = check_key(b.get("api_key"), agent) if agent != "none" else "no key: the agent is turned off"
        runner_call(runner.submit_build, job_id, b["api_key"] if agent != "none" else None, mode=mode,
                    prompt=prompt.strip(), n_views=n_views, picks=picks if mode == "manual" else [], view0=view0,
                    auto_continue=auto_continue, agent=agent, overrides={"locked": [], "excluded": [], "instance": {}},
                    articulated=articulated, parts=parts, lidar=lidar)
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
        sel["lidar"] = lidar_compact(read_json(J / "pool" / "lidar.json"), not steps.done("lidar", J)) \
            if lidar_of(job) else None
        return sel

    def calibration_job(J, job_id):
        """The job.json of a rosbag job that may change its calibration now; 409 for another kind of input,
        while its upload runs or while it is analyzing, queued or running."""
        if runner.uploading(job_id):
            raise Bad(409, "the job is still uploading")
        job = idle(J)
        kind = steps.source(job)["kind"]
        if kind == "stray":
            raise Bad(409, "a Stray Scanner capture needs no calibration: its depth is aligned with the color camera")
        if kind != "rosbag":
            raise Bad(409, "a calibration is for the LiDAR of a rosbag job; this job's input is a video")
        return job

    def set_calibration(job_id, name, data):
        from gator.stage_a import lidar_scale

        J = job_dir(job_id)
        calibration_job(J, job_id)
        if not data:
            raise Bad(400, "the calibration file is empty")
        safe = jobs.safe_name(name)
        try:
            parsed = lidar_scale.read_calibration(data, safe)
        except ValueError as e:
            raise Bad(400, str(e)[:300]) from None
        d = J / "lidar" / "calibration"
        d.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(J / "lidar", 0o700)
        for old in d.iterdir():  # one calibration at a time
            old.unlink()
        tmp = d / f".tmp_{safe}"
        with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as f:
            f.write(data)
        os.replace(tmp, d / safe)
        cal = {"name": safe, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), **parsed}
        jobs.update(J, calibration=cal)
        return calibration_summary(cal)

    @app.put("/api/jobs/{job_id}/calibration")
    async def put_calibration(job_id: str, request: Request, name: str = ""):
        job_dir(job_id)
        if not 1 <= len(name) <= 200:
            raise Bad(400, "name the file: ?name=<file name>")
        if Path(jobs.safe_name(name)).suffix.lower() not in CALIBRATION_SUFFIXES:
            raise Bad(415, f"upload the calibration as {', '.join(CALIBRATION_SUFFIXES)}")
        too_big = f"a calibration file has at most {MAX_CALIBRATION >> 10} KB"
        size = request.headers.get("content-length", "")
        if size.isdigit() and int(size) > MAX_CALIBRATION:
            raise Bad(413, too_big)
        data = bytearray()
        async for chunk in request.stream():  # never more than the limit in memory
            data += chunk
            if len(data) > MAX_CALIBRATION:
                raise Bad(413, too_big)
        return await run_in_threadpool(set_calibration, job_id, name, bytes(data))

    @app.delete("/api/jobs/{job_id}/calibration")
    def delete_calibration(job_id: str):
        J = job_dir(job_id)
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
        agent = agent_of(b, job)
        key = (b.get("api_key") or None) if agent != "none" else None  # no key when the agent is off
        if key is not None:
            check_key(key, agent)
        first = "masks" if instance != job["overrides"].get("instance", {}) else "select"
        first = runner_call(runner.resume, job_id, first, key, view0=view0, agent=agent,
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
                "parts": {**jobs.PARTS, **(job.get("parts") or {}), "mode": steps.parts_mode(job)},
                "waiting": steps.waits_for_parts(J, job), "proposal": read_json(J / "run" / "proposal.json"),
                "photo": photo if photo and resolve(J, photo) else None,
                "checks": [rel for rel in (p.relative_to(J).as_posix() for p in checks) if resolve(J, rel)]}

    def held_key(job_id, b, agent):
        """The body's API key for agent (checked), else None; 400 when the server holds none for the job."""
        key = (b.get("api_key") or None) if agent != "none" else None  # no key when the agent is off
        if key is not None:
            check_key(key, agent)
        elif agent != "none" and not runner.has_key(job_id, agent):
            raise Bad(400, f"the server holds no {steps.PROVIDER[agent]} API key for this job (it restarted, or the "
                           "agent changed); paste the key again")
        return key

    def set_parts(job_id, b):
        from gator.parts import propose

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
        key = held_key(job_id, b, steps.agent(job))  # the agent stage runs after the parts
        first = runner_call(runner.resume, job_id, "parts", key, confirm_parts=True, parts_list=parts)
        return {"status": "queued", "from": first}

    @app.post("/api/jobs/{job_id}/parts")
    async def parts(job_id: str, request: Request):
        return await run_in_threadpool(set_parts, job_id, await body(request))

    def continue_job(job_id, b):
        J = job_dir(job_id)
        job = idle(J)
        first, agent = b.get("from") or "sam3d", agent_of(b, job)
        key = (b.get("api_key") or None) if agent != "none" else None  # no key when the agent is off
        if first == "frames":  # analyze the input again (it failed, was cancelled or the server stopped); no key
            if job["stages"].get("frames", {}).get("status") == "done":
                raise Bad(409, "the input is already analyzed")
            if not Path(job["video"]["path"]).is_file():
                raise Bad(409, "the uploaded file is not on the server; upload it again")
            runner_call(runner.submit_analyze, job_id)
            return {"status": "analyzing", "from": "frames"}
        if first not in jobs.BUILD:
            raise Bad(400, f"from is frames or one of {jobs.BUILD}")
        changed = agent != steps.agent(job)  # the other agent's stage runs anew (Runner.resume)
        if changed or job["stages"]["agent"]["status"] not in ("done", "skipped"):
            key = held_key(job_id, b, agent)
        elif key is not None:
            check_key(key, agent)
        # Build while the job waits for its part list confirms the VLM's list as it is.
        confirm = steps.waits_for_parts(J, job) and not b.get("from")
        first = runner_call(runner.resume, job_id, first, key, confirm_parts=confirm, agent=agent)
        return {"status": "queued", "from": first}  # the agent stage when the agent changed, parts when confirmed

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

    @app.get("/api/jobs/{job_id}/file/{rel:path}")
    def file(job_id: str, rel: str):
        J = job_dir(job_id)
        p = resolve(J, rel)
        if p is None:
            raise Bad(404, "not a file of this job that the app serves")
        name = f"{job_id}_results.zip" if rel == "results.zip" else None
        return FileResponse(p, media_type=TYPES[Path(rel).suffix], filename=name, headers={"Cache-Control": "no-cache"})

    return TokenGate(app, token)


class TokenGate:
    """ASGI middleware: the login cookie passes; GET /?token=T sets it and redirects to /; else 401."""

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
        if self.ok(req.cookies.get(COOKIE)):
            return await self.app(scope, receive, send)
        resp = PlainTextResponse("open the URL with ?token= that the server printed", status_code=401)
        return await resp(scope, receive, send)


def write_info(info):
    """~/.config/gator/webapp.json (0600), replaced atomically, for start_ice.sh."""
    d = Path("~/.config/gator").expanduser()
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = d / ".webapp.json.tmp"
    tmp.unlink(missing_ok=True)
    with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as f:
        json.dump(info, f, indent=1)
    os.replace(tmp, d / "webapp.json")
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


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m gator.webapp", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", help="folder of the job folders (default: WEBAPP_RUNS, else $G/webapp_runs)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=0, help="0 takes a free port")
    ap.add_argument("--fake", action="store_true", help="run every stage with gator.webapp.fake_steps (no GPU)")
    a = ap.parse_args(argv)
    import uvicorn

    cfg = steps.config_from_env()
    cfg = replace(cfg, runs=Path(a.runs).resolve() if a.runs else cfg.runs, fake=cfg.fake or a.fake)
    # The socket is bound here, so the port is known (also for --port 0) before webapp.json is written.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((a.host, a.port))
    except OSError as e:
        raise SystemExit(f"cannot listen on {a.host}:{a.port}: {e}")
    port, node, token = sock.getsockname()[1], socket.getfqdn(), secrets.token_urlsafe(32)
    app = make_app(cfg, token)
    signal.signal(signal.SIGTERM, _exit_on_sigterm)
    info = write_info({"node": node, "port": port, "token": token, "url_path": f"/?token={token}",
                       "slurm_job": os.environ.get("SLURM_JOB_ID"), "slurm_node": os.environ.get("SLURMD_NODENAME")})
    print(f"HARPOR web app on {node}:{port}, runs in {cfg.runs}" + (" (fake stages)" if cfg.fake else ""))
    print(f"login details in {info}")
    print(f"on the Mac: {FORWARD.format(node=node, port=port)}")
    print(f"then open http://localhost:8765/?token={token}")
    print(f"(on this machine: http://localhost:{port}/?token={token})", flush=True)
    try:
        with suppress(KeyboardInterrupt):  # uvicorn shuts down on Ctrl-C, then raises it again
            uvicorn.Server(uvicorn.Config(app, log_level="info", access_log=False, timeout_graceful_shutdown=3)).run(
                sockets=[sock])
    finally:
        remove_info(info, token)


if __name__ == "__main__":
    main()
