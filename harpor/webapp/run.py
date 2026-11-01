"""One terminal command that runs a whole HARPOR job, with no web server and no browser.

Runs in envs/web on the GPU machine, from the repo root (on ICE inside a Slurm job, as the server does):

    python -m harpor.webapp.run INPUT [--prompt NOUN] [settings]
    python -m harpor.webapp.run --resume ID [answers]

INPUT is a video, a rosbag or a zipped Stray Scanner export. The command makes a job folder the way an upload does,
then runs the web app's own code in this process and calls its routes directly (server.make_app, with
httpx.ASGITransport, as worker.py does). It opens no port. So every setting passes the page's checks, and every stage
is the Runner's stage, from the same stage scripts (steps.py); only the watching and the printing are new here.

The input is a hard link in the job folder, or a copy when it is on another file system. No size limit and no
free-space check apply: the person chose this machine's disk. A job of this command holds job.json "public":
{"local": true}, so that a worker that starts later on the same runs folder leaves it off the public site
(worker.local_jobs). The page can still open it.

The command never asks a question. When a job needs a person, it prints what to look at and the exact next command,
and it exits with code 3:

  - A bag with several camera topics waits for --camera-topic.
  - Without --prompt the command stops after the analysis, so that the person can look at the thumbnails.
  - With --review, the job stops after select, and also when the automatic selection fails. --resume ID goes on.
    --resume ID with --lock, --exclude, --view0 or --instance selects again and stops again (masks first when an
    instance changed). These flags add to the choices made earlier.
  - With --edit-parts, the job stops after the VLM's part list. --resume ID accepts the list, and --resume ID --parts F
    replaces it. F is a JSON list of parts ([{name, parent, joint, count}], harpor.parts.propose) or an object with a
    "parts" list, such as a copy of run/proposal.json.

--resume ID takes the next step for the job's state. A job whose build has not started takes the settings of its
resume, and the printed command repeats them. A job that failed or was cancelled goes on from the first stage not done,
as the page's Retry does. After the build has started the settings stay as they are, except --calibration, which
starts again from the lidar stage: for other settings, start a new job.

Ctrl-C, SIGTERM and a closed terminal cancel the running stage and leave a job that can resume. A SIGHUP that the
caller ignores (nohup) stays ignored, and the job goes on.

The key of --agent astra or opus comes from --key-file or from ~/.config/harpor/openai.env or anthropic.env, in the
format of the worker's key files (session.local_key), and it is read again on each resume. The command never prints it.

Exit codes: 0 for a finished job, 1 for a failed or cancelled job, 2 for bad arguments or a refused setting, 3 for a job
that waits for a person, and 128 + the signal after Ctrl-C, SIGTERM or SIGHUP.

Two rules come from jobs.py. A Runner deletes every *.env in its key folder when it starts and stops, and all of one
user's servers share the default folder, so this command has a folder of its own for each runs folder: a key that a
killed run left there is deleted by the next run. And a Runner sees a live holder of a runs folder by "harpor.webapp" in
its command line, so this module must stay in that package. While this command runs, a server or a second command on
the same runs folder refuses to start, and this command refuses while they run.
"""

import argparse
import contextlib
import hashlib
import json
import os
import re
import secrets
import shlex
import shutil
import signal
import sys
import time
from dataclasses import replace
from pathlib import Path

import anyio
import httpx

from harpor.fileio import read_json
from harpor.webapp import jobs, relay, server, session, steps, views

POLL_S = 0.5
DONE, FAILED, BAD, WAITS = 0, 1, 2, 3  # exit codes
ALWAYS = ("runs", "key_file", "fake")  # the flags that every next command repeats
# The flags that reach job.json at the Start route, so a next command repeats them until then (Run.again).
SETTINGS = ("prompt", "camera_topic", "lidar_topic", "calibration", "no_lidar", "n_views", "picks", "view0", "review",
            "articulated", "parts", "edit_parts", "turntable", "agent")
LATE_SETTINGS = ("calibration", "parts", "view0")  # those that a job whose build has started still takes
LEFT_OUT = {"available": "usable, not picked", "excluded": "excluded", "no_mask": "no mask of the object",
            "small": "the mask is under 1% of the image", "inconsistent": "the mask disagrees with the other views"}


class Stop(Exception):
    """The command stops: (exit code, the reason to print). Reasons never hold the API key."""


def say(text):
    with contextlib.suppress(OSError):  # a closed terminal must not keep the job from stopping cleanly
        print(text, flush=True)


def flag(name):
    return "--" + name.replace("_", "-")


def given(a, names):
    """The flags among names that a holds, each as one word."""
    values = [(n, getattr(a, n)) for n in names]
    return [flag(n) if v is True else f"{flag(n)}={v}" for n, v in values if v not in (None, False)]


def started(job):
    """True when the job's build has started, so job.json holds its settings."""
    return job["stages"]["extract"]["status"] != "pending"


def part_list(path):
    """The parts of a --parts file; see the module docstring."""
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError) as e:
        raise Stop(BAD, f"cannot read the part list: {e}") from None
    parts = data.get("parts") if isinstance(data, dict) else data
    if not isinstance(parts, list):
        raise Stop(BAD, f'{path} is not a JSON list of parts or an object with a "parts" list')
    return parts


def names(values):
    """The pool names of --lock or --exclude values, each a frame or a list such as 12,f00080."""
    return [views.pool_name(x) for v in values or () for x in v.split(",")]


def instances(values):
    """{pool name: K, "K,L" or "none"} of --instance values such as f00012=1,2."""
    out = {}
    for v in values or ():
        name, sep, k = v.partition("=")
        if not sep:
            raise Stop(BAD, f"--instance takes FRAME=K, such as f00012=1,2, not {v}")
        out[views.pool_name(name)] = k
    return out


def check_input(a):
    """Stops when the command has no job to work on, or its input cannot be a job's. The upload route makes the same
    check of the suffix; create_job alone would take an unknown suffix for a video."""
    if bool(a.input) == bool(a.resume):
        raise Stop(BAD, "give an input file, or --resume ID")
    if a.resume:
        return
    p = Path(a.input)
    if not p.is_file():
        raise Stop(BAD, f"{p} is not a file")
    if not p.stat().st_size:
        raise Stop(BAD, f"{p} is empty")
    if jobs.kind(jobs.safe_name(p.name)) is None:
        raise Stop(BAD, f"the input is a video ({' '.join(jobs.VIDEOS)}), a rosbag ({' '.join(jobs.BAGS)}) or a Stray "
                        f"Scanner export ({' '.join(jobs.STRAY)}); {p.suffix or 'no suffix'} is none of these")


def check_settings(a, kind):
    """Stops on settings that cannot work, so that a mistake costs no analysis. The rules that need the job, such as
    the frame range, are the Start route's."""
    agent = a.agent or "none"
    for bad, why in (
            (kind != "rosbag" and (a.camera_topic or a.lidar_topic or a.calibration),
             "--camera-topic, --lidar-topic and --calibration are for a rosbag"),
            (kind != "stray" and a.no_lidar, "--no-lidar is for a Stray Scanner capture, whose depth is on by default"),
            (a.lidar_topic and a.no_lidar, "--lidar-topic and --no-lidar do not go together"),
            (a.calibration and not a.lidar_topic, "--calibration needs --lidar-topic"),
            (a.calibration and not Path(a.calibration or "").is_file(), f"{a.calibration} is not a file"),
            (a.calibration and Path(a.calibration or "").suffix.lower() not in server.CALIBRATION_SUFFIXES,
             f"the calibration is {', '.join(server.CALIBRATION_SUFFIXES)}"),
            (a.picks and not re.fullmatch(r"\d+(,\d+)*", a.picks or ""), "--picks takes frames such as 12,80,150"),
            (a.view0 is not None and not re.fullmatch(r"f?\d+", a.view0 or ""), "--view0 takes a frame such as 12"),
            (a.lock or a.exclude or a.instance, "--lock, --exclude and --instance answer a view review, on a resume"),
            ((a.parts or a.edit_parts) and not a.articulated, "--parts and --edit-parts need --articulated"),
            (a.parts and a.edit_parts, "--parts gives the list and --edit-parts asks the VLM for it: choose one"),
            (a.edit_parts and agent == "none", "--edit-parts needs --agent astra or opus: there is no VLM without one"),
            (a.articulated and agent == "none" and not a.parts,
             "without an agent there is no VLM to list the parts: give --parts, or --agent astra or opus"),
            (a.key_file and agent == "none", "--key-file needs --agent astra or opus")):
        if bad:
            raise Stop(BAD, why)
    if a.parts:
        try:  # the Start route's check
            jobs.parts_settings({"prompt": a.prompt, "agent": agent, "articulated": True,
                                 "parts": {"mode": "user", "list": part_list(a.parts)}})
        except ValueError as e:
            raise Stop(BAD, f"{a.parts}: {server.part_reason(e)}") from None


def place(src, path):
    """Puts the input file src at path; see the module docstring."""
    try:
        os.link(src, path)
    except OSError:  # another file system, or a file of another user
        say(f"copying {src} into the job folder")
        part = path.with_name(path.name + ".part")
        os.close(os.open(part, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
        shutil.copyfile(src, part)
        os.replace(part, path)


def key_folder(cfg):
    """This command's key folder for cfg's runs folder; see the module docstring."""
    folder = jobs.default_key_dir()
    digest = hashlib.sha256(str(Path(cfg.runs).resolve()).encode()).hexdigest()[:12]
    return folder.with_name(f"{folder.name}-run-{digest}")


def view_line(c, sam):
    """A view of selection.json, with its SAM 3 instances when there is a choice. sam: the selection route's "sam3"."""
    found = sam.get(c["name"]) or {}
    text = f"{c['name']} at {c['time_s']:.2f} s" + (f", q {c['q']:.2f}" if c.get("q") is not None else "")
    if len(found.get("instances") or ()) > 1:
        text += f", SAM 3 found {len(found['instances'])} instances and used {found.get('picked')}"
    return text


class Run:
    """The web app in this process, with the one job that this command works on."""

    def __init__(self, a, cfg):
        self.a, self.cfg, self.key_dir = a, cfg, key_folder(cfg)
        self.job_id, self.J, self.key = None, None, None
        self.seen, self.tenths = {}, {}  # what show printed last, per stage
        self.app = server.make_app(cfg, secrets.token_urlsafe(32), key_dir=self.key_dir).app  # no token gate
        self.runner = self.app.state.runner
        self.handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
        for sig in (signal.SIGTERM, signal.SIGHUP):
            if self.handlers[sig] != signal.SIG_IGN:  # nohup: a closed terminal does not stop the job
                signal.signal(sig, server._exit_on_sigterm)

    def go(self):
        """Works on the job and stops the app. Returns the exit code."""
        try:
            return self.drive()
        except Stop as e:
            code, why = e.args
            say(why)
            if self.job_id:
                say(f"to go on: {self.again()}")
            return code
        except KeyboardInterrupt:
            return 128 + signal.SIGINT
        finally:
            self.close()

    def close(self):
        """Cancels a job that is still busy, so that it can resume, and stops the runner. A second Ctrl-C does not cut
        this short: the stages would stay on the GPU."""
        for sig in self.handlers:
            signal.signal(sig, signal.SIG_IGN)
        busy = self.job_id is not None and jobs.load(self.J)["status"] in jobs.ACTIVE
        if busy:
            say("cancelling the job")
            self.runner.cancel(self.job_id)
            t0 = time.time()
            while jobs.load(self.J)["status"] in jobs.ACTIVE and time.time() - t0 < self.runner.kill_after_s + 5:
                time.sleep(0.1)
        self.runner.shutdown()
        with contextlib.suppress(OSError):
            self.key_dir.rmdir()
        if busy:
            self.show()
            say(f"to go on: {self.again()}")
        for sig, handler in self.handlers.items():
            signal.signal(sig, handler)

    def call(self, method, route, **kw):
        """The answer of the job's route ("" is GET /api/jobs/{id}). A refusal stops the command with its reason."""
        async def go():
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://run", timeout=None) as c:
                return await c.request(method, f"/api/jobs/{self.job_id}{route}", **kw)

        r = anyio.run(go)
        if r.status_code >= 400:
            raise Stop(BAD, r.json()["detail"])
        return r.json()

    def again(self, *more):
        """The command that goes on with this job, with the words of more at its end."""
        words = ["--resume", self.job_id, *given(self.a, ALWAYS)]
        if not started(jobs.load(self.J)):
            words += given(self.a, SETTINGS)
        return shlex.join([sys.executable, "-m", "harpor.webapp.run", *words, *more])

    # ---- the job's progress ----

    def show(self):
        """One line for each stage that changed since the last look, and for each 10% of a running stage."""
        for s in jobs.progress(self.J):
            name, now = s["name"], (s["status"], s["started"], s["note"])
            if self.seen.get(name) != now:
                self.seen[name] = now
                self.tenths.pop(name, None)
                if s["status"] != "pending":
                    after = f" after {s['seconds']} s" if s["seconds"] is not None and s["status"] != "running" else ""
                    say(f"{name}: {s['status']}{after}" + (f": {s['note']}" if s["note"] else ""))
            tenth = int((s["progress"] or 0) * 10)
            if tenth > self.tenths.get(name, 0):
                self.tenths[name] = tenth
                say(f"{name}: {tenth * 10}%")

    def watch(self):
        """Prints the stages' changes until the job stops being busy. Returns job.json."""
        while True:
            job = jobs.load(self.J)
            self.show()
            if job["status"] not in jobs.ACTIVE:
                return job
            time.sleep(POLL_S)

    def follow(self):
        return self.ended(self.watch())

    def ended(self, job):
        """Says how the job stopped. Returns the exit code."""
        for w in job["warnings"]:
            say(f"warning: {w}")
        if job["status"] == "review":
            return self.part_review() if steps.waits_for_parts(self.J, job) else self.view_review(job)
        if job["status"] == "done":
            return self.finished()
        if job["status"] == "failed":
            failed = [s["name"] for s in jobs.progress(self.J) if s["status"] == "failed"]
            say(f"the job failed: {job['error']}")
            if failed:
                say(f"log file: {self.J / 'logs' / f'{failed[-1]}.log'}")
        else:
            say(f"the job is {job['status']}")
        say(f"to go on: {self.again()}")
        return FAILED

    def finished(self):
        r = self.call("GET", "")["agent_result"]
        if r:
            say(f"agent: {r.get('verdict')}: {r.get('reason')}")
        say(f"results: {self.J / 'results.zip'}")
        return DONE

    def view_review(self, job):
        """Prints the view selection and the next commands."""
        sel, J = self.call("GET", "/selection"), self.J
        for w in sel["warnings"]:
            if w not in job["warnings"]:
                say(f"warning: {w}")
        for i, p in enumerate(sel["picks"]):
            say(f"view {i}: {view_line(p, sel['sam3'])}")
        for c in sel["candidates"]:
            if c["status"] != "picked":
                say(f"left out: {view_line(c, sel['sam3'])}: {LEFT_OUT.get(c['status'], c['status'])}")
        lidar = sel["lidar"]
        if lidar:
            say(f"LiDAR scale: {lidar['s']} from {lidar['counts']['views_used']} views" if lidar["ok"]
                else f"LiDAR scale: {lidar['reason']}")
            say(f"LiDAR check images: {J / lidar['check_dir']}, named <view>.jpg")
        say(f"mask check images: {J / sel['check_dir']}, named <view>.jpg")
        change = "--exclude, --lock, --view0 or --instance FRAME=K"
        if job["stages"][steps.REVIEW_AFTER]["status"] == "done":
            say(f"to go on: {self.again()}")
            say(f"to change the views, add {change} to that command. The job selects again and stops here again")
        else:
            say(f"the automatic selection needs help. Add {change} to this command: {self.again()}")
            say("or start a new job with --picks")
        return WAITS

    def part_review(self):
        """Prints the VLM's part list and the next commands."""
        parts, J = self.call("GET", "/parts"), self.J
        for p in parts["proposal"]["parts"]:
            count = f", {p['count']} of them" if p["count"] > 1 else ""
            hint = f", {p['axis_hint']}" if p.get("axis_hint") else ""
            role = f"{p['joint']} on {p['parent']}" if p["parent"] else "the base"
            say(f"part: {p['name']}: {role}{count}{hint}")
        if parts["photo"]:
            say(f"the VLM saw this photo: {J / parts['photo']}")
        say(f"to accept the list: {self.again()}")
        proposal, replacing = J / steps.MARKERS["propose"], self.again("--parts", "EDITED.json")
        say(f"to change it, edit a copy of {proposal} and run: {replacing}")
        return WAITS

    def analyzed(self):
        """Prints what the analysis found, for the person who chooses the settings."""
        found, meta = read_json(self.J / "frames" / "topics.json") or {}, read_json(self.J / "frames" / "meta.json")
        for t in found.get("topics") or ():
            say(f"camera topic: {t['topic']}, {t.get('width')} x {t.get('height')}, {t.get('count')} images")
        for t in found.get("skipped") or ():
            say(f"unusable topic: {t['topic']}: {t['why']}")
        for t in found.get("lidar") or ():
            say(f"LiDAR topic: {t['topic']}")
        for t in found.get("lidar_skipped") or ():
            say(f"unusable LiDAR topic: {t['topic']}: {t['why']}")
        if meta:
            say(f"the input has {len(meta['times'])} frames; the thumbnails are in {self.J / 'frames' / 'thumbs'}, "
                "named <frame>.jpg")

    # ---- the steps ----

    def take_key(self, agent):
        """Reads the key of agent and checks it, so that a bad key costs no stages."""
        if agent == "none":
            return
        try:
            self.key = session.local_key(agent, self.a.key_file)
        except (OSError, ValueError, SystemExit) as e:
            raise Stop(BAD, f"cannot read the {steps.PROVIDER[agent]} API key: {e}") from None
        ok, why = self.runner.check_key(self.key, agent)
        if not ok:
            raise Stop(BAD, why)

    def upload(self):
        """A new job for the input, as an upload makes one. Its analysis starts here."""
        src = Path(self.a.input)
        self.job_id, path = self.runner.new_upload(src.name)
        self.J = self.runner.runs / self.job_id
        jobs.update(self.J, public={"local": True})
        say(f"job {self.job_id} in {self.J}")
        try:
            place(src.resolve(), path)
        except BaseException as e:  # no half file and no job stays
            self.runner.upload_done(self.job_id, None)
            self.job_id = None
            if isinstance(e, OSError):
                raise Stop(FAILED, f"cannot put the input into the job folder: {e}") from None
            raise
        self.runner.upload_done(self.job_id, path.stat().st_size)

    def calibration(self, lidar_topic):
        path = Path(self.a.calibration)
        try:
            data = path.read_bytes()
        except OSError as e:
            raise Stop(BAD, f"cannot read the calibration: {e}") from None
        self.call("PUT", "/calibration", params={"name": path.name, "lidar_topic": lidar_topic}, content=data)

    def start(self):
        """From the analysis to a running build: the camera topic, the calibration, then the Start route."""
        a, job = self.a, self.watch()
        if job["status"] in ("choose_topic", "ready") and a.camera_topic not in (None, steps.source(job)["topic"]):
            self.call("POST", "/topic", json={"topic": a.camera_topic})
            job = self.watch()
        if job["status"] == "choose_topic":
            self.analyzed()
            say(f"the bag has several camera topics. To choose one: {self.again('--camera-topic', 'TOPIC')}")
            return WAITS
        if job["stages"]["frames"]["status"] != "done":
            return self.ended(job)
        if not a.prompt:
            self.analyzed()
            say(f"to start the build: {self.again('--prompt', 'NOUN')}")
            say("to choose the frames yourself, add --picks with the frames, such as --picks 12,80,150")
            return WAITS
        if a.calibration:
            self.calibration(a.lidar_topic)
        body = {"mode": "manual" if a.picks else "auto", "prompt": a.prompt, "agent": a.agent or "none",
                "api_key": self.key, "auto_continue": not a.review, "turntable": a.turntable,
                "articulated": a.articulated, "picks": [int(x) for x in a.picks.split(",")] if a.picks else []}
        if a.n_views is not None:
            body["n_views"] = a.n_views
        if a.view0 is not None:
            body["view0"] = views.frame_of(views.pool_name(a.view0))
        if a.parts:
            body.update(parts_mode="user", parts=part_list(a.parts))
        elif a.edit_parts:
            body["parts_mode"] = "vlm_edit"
        if a.lidar_topic:
            body["lidar"] = {"topic": a.lidar_topic}
        elif a.no_lidar:
            body["lidar"] = None
        self.call("POST", "/start", json=body)
        return self.follow()

    def overrides(self):
        """Selects again with the --lock, --exclude, --instance and --view0 of this resume, on top of the job's own."""
        a, sel = self.a, self.call("GET", "/selection")
        locked, excluded, old = names(a.lock), names(a.exclude), sel["overrides"]
        view0 = sel["view0"] if isinstance(sel["view0"], str) else sel["anchor"] if sel["view0"] is not None else None
        self.call("POST", "/overrides", json={
            "locked": sorted(set(old["locked"]).difference(excluded).union(locked)),
            "excluded": sorted(set(old["excluded"]).difference(locked).union(excluded)),
            "instance": {**old["instance"], **instances(a.instance)},
            "view0": view0 if a.view0 is None else views.pool_name(a.view0), "api_key": self.key})
        return self.follow()

    def go_on(self, job):
        """--resume of a job whose build has started: the next step for its state."""
        a, J = self.a, self.J
        same = ("agent",) if a.agent == steps.agent(job) else ()  # naming the job's own agent changes nothing
        late = given(a, [n for n in SETTINGS if n not in (*LATE_SETTINGS, *same)])
        if late:
            raise Stop(BAD, f"the build of job {self.job_id} has started, so it keeps its settings. Start a new job to "
                            f"use {' '.join(late)}")
        viewed = a.lock or a.exclude or a.instance or a.view0 is not None
        if job["status"] == "done" and (viewed or a.calibration or a.parts):
            raise Stop(BAD, f"job {self.job_id} is done. Start a new job to change it")
        lidar = steps.lidar(job)
        if a.calibration and not (lidar or {}).get("topic"):
            raise Stop(BAD, f"job {self.job_id} uses no LiDAR topic, so it takes no calibration")
        if viewed or a.calibration or (lidar and not steps.done("lidar", J)) \
                or job["stages"]["agent"]["status"] not in ("done", "skipped"):  # the routes' rule for the key
            self.take_key(steps.agent(job))
        if a.calibration:
            self.calibration(lidar["topic"])
        if viewed:
            return self.overrides()
        if a.parts:
            self.call("POST", "/parts", json={"parts": part_list(a.parts), "api_key": self.key})
            return self.follow()
        if job["status"] == "review":
            if job["stages"][steps.REVIEW_AFTER]["status"] != "done":
                return self.ended(job)  # the automatic selection failed, so the build cannot go on yet
            self.call("POST", "/continue", json={"api_key": self.key})  # also accepts a waiting part list
            return self.follow()
        first = relay.first_not_done({"stages": jobs.progress(J), "articulated": steps.articulated(job),
                                      "parts_mode": steps.parts_mode(job)}, job["parts"]["confirmed"] is True)
        if first is None:
            return self.ended({**job, "status": "done"})
        self.call("POST", "/continue", json={"from": first, "api_key": self.key})
        return self.follow()

    def drive(self):
        """The new job's analysis and Start, or the resume of a job. Returns the exit code."""
        a = self.a
        if a.input:
            if a.prompt:
                self.take_key(a.agent or "none")
            self.upload()
            return self.start()
        try:
            self.J = self.runner.job_dir(a.resume)
        except ValueError:
            raise Stop(BAD, f"no job {a.resume} in {self.cfg.runs}") from None
        self.job_id, job = a.resume, jobs.load(self.J)
        self.seen = {s["name"]: (s["status"], s["started"], s["note"]) for s in jobs.progress(self.J)}
        if started(job):
            return self.go_on(job)
        check_settings(a, steps.source(job)["kind"])
        if a.prompt:
            self.take_key(a.agent or "none")
        if job["status"] not in ("choose_topic", "ready"):  # the analysis failed or was cancelled
            if a.camera_topic:
                self.call("POST", "/topic", json={"topic": a.camera_topic})
            elif job["stages"]["frames"]["status"] != "done":
                self.call("POST", "/continue", json={"from": "frames"})
        return self.start()


def parser():
    ap = argparse.ArgumentParser(prog="python -m harpor.webapp.run", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?", help="a video, a rosbag or a zipped Stray Scanner export")
    ap.add_argument("--resume", metavar="ID", help="go on with a job; see the text above")
    ap.add_argument("--prompt", metavar="NOUN", help="the object noun; without it the command stops after the analysis")
    ap.add_argument("--camera-topic", metavar="T", help="a rosbag's camera topic")
    ap.add_argument("--lidar-topic", metavar="T", help="a rosbag's LiDAR topic, for the metric scale")
    ap.add_argument("--calibration", metavar="F", help="a rosbag's LiDAR-camera calibration (.yaml, .yml or .json)")
    ap.add_argument("--no-lidar", action="store_true", help="turn off the depth of a Stray Scanner capture")
    ap.add_argument("--n-views", type=int, metavar="N", help="how many views to select, 2 to 12 (default 6)")
    ap.add_argument("--picks", metavar="FRAMES", help="choose the views yourself: 2 to 12 frames, such as 12,80,150")
    ap.add_argument("--view0", metavar="F", help="the frame of view 0; in a review, a pool frame such as f00012")
    ap.add_argument("--review", action="store_true", help="stop after the views are selected")
    ap.add_argument("--articulated", action="store_true", help="build an object with moving parts")
    ap.add_argument("--parts", metavar="F", help="the part list as a JSON file; on a resume, replace the VLM's list")
    ap.add_argument("--edit-parts", action="store_true", help="stop after the VLM's part list")
    ap.add_argument("--turntable", action="store_true", help="render a 360 video of the object")
    ap.add_argument("--agent", choices=steps.AGENTS, help="the agent that refines the object (default none)")
    ap.add_argument("--key-file", metavar="F",
                    help="the agent's key file (default ~/.config/harpor/openai.env or anthropic.env)")
    ap.add_argument("--lock", action="append", metavar="FRAMES", help="on a resume: always use these views")
    ap.add_argument("--exclude", action="append", metavar="FRAMES", help="on a resume: never use these views")
    ap.add_argument("--instance", action="append", metavar="FRAME=K",
                    help="on a resume: use SAM 3's instance K of a view (K, K,L or none)")
    ap.add_argument("--runs", metavar="DIR",
                    help="the folder of the job folders (default: WEBAPP_RUNS, else $G/webapp_runs)")
    ap.add_argument("--fake", action="store_true", help="run every stage with harpor.webapp.fake_steps (no GPU)")
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    cfg = steps.config_from_env()
    cfg = replace(cfg, runs=Path(a.runs).resolve() if a.runs else cfg.runs, fake=cfg.fake or a.fake, demos=None)
    try:
        check_input(a)
        if a.input:
            check_settings(a, jobs.kind(jobs.safe_name(Path(a.input).name)))
    except Stop as e:
        say(e.args[1])
        return e.args[0]
    return Run(a, cfg).go()


if __name__ == "__main__":
    sys.exit(main())
