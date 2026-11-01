"""The personal worker: a thread inside the HARPOR server on a user's own GPU machine that serves the public site
(docs/public_site.md, What the worker does).

    python -m harpor.webapp --worker [--local-keys astra,opus]

Worker(app, cfg, local_keys, session=session) runs on make_app's app: it uses the inner FastAPI app through
httpx.ASGITransport and that app's one jobs.Runner, in a thread with its own event loop
(start_thread). It never listens on a port. The server's lifespan calls stop_thread before
runner.shutdown (server.on_stop), so the worker never copies the "cancelled" that shutdown writes.
It signs in as the machine's user (session.Session) and talks to Supabase only through one supa.Store
over that session: the user's own rows under RLS, and the own_* functions.

A relayed request: the worker claims it, takes its sealed key off Supabase in every case, and checks it
against fresh rows: the user is approved (a delete is exempt), the request is this user's, and the job is
of this machine with its folder here (a delete is exempt from both). The agent that server.py will run
(relay.effective_agent) must be in access.agents, which holds only astra and opus. A sealed key is opened
with this start's KeyBox, the AAD built from the request, this worker's id and that agent; the worker may
instead read the machine's own key file (--local-keys). The request goes to the app as relay.build makes
it, the job is published, and only then is the answer written, so the page's refresh after the answer
sees the new view.

Publishing a job, in this order: the log text of its running stage (server.tick, without PROGRESS lines,
scrubbed, at most LOG_CAP per stage run), the files of stages that are not running (relay.mirror_plan:
uploaded under names that hold their hash, scanned for the job's keys, lean review files, left out with a
note when too large), then the view, the live state, the snapshots and the jobs row in one own_publish
call. A change of status, or of a stage's status, publishes at once; progress alone at most every 2 s.

The start: the workers row, then the jobs rows against the job folders here (reconcile), orphan objects,
interrupted jobs resumed at most RESUMES times, and a first publish of every job. Inputs stay on the
machine: a local upload (its own UI, or `worker add`) gets a row (local_jobs). Near the end of a Slurm
job the worker takes no builds; at the stop, active jobs keep their status with a note, and the next
start resumes them. Housekeeping, every minute: the settings row again, and the key of a job that waited
in review over KEY_HOURS goes; every hour, retention (review files and logs REVIEW_DAYS, results
RESULT_DAYS after a job stopped, also for a job whose folder is gone from the machine) and the orphan
objects of the user's folder (own_orphans). pg_cron sweeps the rows (private.sweep); no worker does.
"""

import argparse
import hashlib
import logging
import os
import re
import stat
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import anyio
import httpx

from harpor.fileio import config_dir, private_folder
from harpor.webapp import jobs, keybox, relay, server, steps, supa

log = logging.getLogger("harpor.worker")
VERSION = "harpor-worker-1"
PARALLEL = 4  # requests at once; never two of one job (the claim and the job's lock)
OWN_IDLE = 2.0  # s between own_claim_request calls while nothing waits (no long poll from a user session)
LOG_CAP = 256 << 10  # characters of job_logs text per stage run
LOG_CUT = "\n[the rest is in the full log when the stage ends]\n"
TICKS = 64  # server.tick calls in one publish at most: each reads up to 64 KB of one stage's log, or moves on
THROTTLE = 2.0  # s between publishes of progress alone
OLD_OBJECT_S = 600  # an object that a path no longer names goes after 10 minutes (open pages keep their images)
STOP_WAIT = 30  # s that a delete waits for a cancelled job to stop
SETTLE = 5  # publishes the mirror again at most this often while stages change under it
PAUSE_S = 300  # a worker whose account lost its approval tries again after 5 minutes
OFFLINE_S = 60  # a worker that cannot reach Supabase at its start tries again after this long
RESUMES = 2  # automatic resumes of one job, so a job that crashes the server does not loop
HOUSEKEEPING_S = 60
HOURLY_S = 3600
SQUEUE_S = 600  # the Slurm job's end is read again this often (the limit can change)
REVIEW_DAYS, RESULT_DAYS = 14, 90  # retention on the site, counted from when a job stopped
KEY_HOURS = 24  # a key held for a job in review goes after this long
ORPHAN_ROUNDS = 20  # own_orphans calls of 1,000 names in one pass at most
SIGNED_OUT = ("signed out of the HARPOR site: run python -m harpor.webapp.worker login, then start the server "
              "again")
ENDED = "{place} stopped. This job goes on when HARPOR runs there again."
TWICE = "{place} stopped twice while this job ran; press Retry"
GONE = "the job folder is gone from {place}"
KEY_GONE = "your machine restarted after you sent the key; paste the key again"  # own_worker_start's text
REVIEW_EXPIRED = f"the review images and logs expired {REVIEW_DAYS} days after the job stopped"
RESULTS_EXPIRED = f"the result files expired {RESULT_DAYS} days after the job stopped"
STOPPED = re.compile(r"worker \S+ was stopped")


def now():
    return datetime.now(timezone.utc)


def slurm_end():
    """The end of this Slurm job (squeue %e), or None outside Slurm or without an answer."""
    job = os.environ.get("SLURM_JOB_ID")
    if not job or not job.isdigit():
        return None
    try:
        out = subprocess.run(["squeue", "-h", "-j", job, "-o", "%e"], capture_output=True, text=True, timeout=20)
        return datetime.fromisoformat(out.stdout.strip()).astimezone(timezone.utc)
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None


@dataclass
class Pub:
    """What the worker keeps of one job it serves."""
    uuid: str
    local: str
    created: str | None = None  # the row's created_at, the view's "created"
    cursor: tuple | None = None  # server.tick's
    logged: dict = field(default_factory=dict)  # (stage, run start) -> characters sent
    keys: set = field(default_factory=set)  # every key held for the job, for the scrub and the key scan
    manifest: dict | None = None  # path -> {"object", "sha256"} on the site
    seen: dict = field(default_factory=dict)  # path -> (stat and keys, sha256 or None, note or None)
    view: dict | None = None  # what was published last
    live: dict | None = None
    snaps: dict = field(default_factory=dict)
    row: dict | None = None
    at: float = 0.0  # time of the last publish
    mtime: int | None = None  # job.json's at the last publish
    gone: bool = False  # the site no longer has the job: no more copies (a delete may still come)
    logs_full: bool = False  # the site's log of the job is full


class Worker:
    """The personal worker of one server (see the module docstring). local_keys names the agents whose key the
    machine keeps in its own key file. session: the session.Session that server.worker_of started. store and
    local_key are for tests: a supa.Store and the function that reads a local key (session.local_key by default)."""

    def __init__(self, app, cfg, local_keys=(), *, session, store=None, local_key=None):
        local_keys = tuple(local_keys or ())
        if set(local_keys) - set(steps.PROVIDER):
            raise ValueError(f"local keys are for {' and '.join(steps.PROVIDER)}")
        self.inner = app.app if isinstance(app, server.TokenGate) else app
        self.runner = self.inner.state.runner
        self.cfg, self.local_keys, self.session = cfg, local_keys, session
        if store is None:
            store = supa.Store.of(self.session)
        self.store, self._local_key = store, local_key
        self.place = self.session.machine_name or "your machine"
        self.box = keybox.KeyBox()  # a new ECDH P-256 pair, in memory only
        self.jobs, self.locks, self.settings = {}, {}, {}
        self.trash = {}  # job-files object -> when it goes
        self.folderless = set()  # this machine's jobs whose folder is gone: retention reads their rows
        self.refused = {}  # local job id -> when own_job_create may be tried again
        self.ends_at = slurm_end()
        self.paused_until = 0.0
        self.stopping = threading.Event()
        self.thread, self.client = None, None

    def __repr__(self):
        return f"Worker({self.store!r})"

    # ---- life cycle ----

    def start_thread(self):
        self.thread = threading.Thread(target=self._run, name="harpor-worker", daemon=True)
        self.thread.start()

    def _run(self):
        try:
            anyio.run(self.main)
        except Exception:
            log.exception("the HARPOR worker stopped")

    def stop_thread(self, timeout=12):
        """Stop the loops, mark active jobs as interrupted and stop the workers row; waits up to timeout s."""
        self.stopping.set()
        if self.thread:
            self.thread.join(timeout=timeout)

    async def main(self):
        """start (again after OFFLINE_S without Supabase, or PAUSE_S while the account waits for approval), the
        loops until the worker stops, then end, also after a failed start."""
        try:
            if await self.begin():
                async with anyio.create_task_group() as tg:
                    for loop in self.loops():
                        tg.start_soon(self._forever, loop)
                    while not self.stopping.is_set():
                        await anyio.sleep(0.2)
                    tg.cancel_scope.cancel()
        finally:
            self.stopping.set()
            with anyio.move_on_after(8, shield=True):
                await self.end()

    async def begin(self):
        """start, tried again as main says; False when the worker does not start (it printed why)."""
        while not self.stopping.is_set():
            try:
                await self.start()
                return True
            except supa.SignedOut:
                print(SIGNED_OUT, flush=True)
                return False
            except supa.Error as e:
                if not (self.paused(e) or isinstance(e, supa.Offline)):
                    print(f"the HARPOR worker could not start: {e}", flush=True)
                    return False
                wait = PAUSE_S if self.paused() else OFFLINE_S
                print(f"the HARPOR worker waits {wait:g} s and starts again: {e}", flush=True)
                await self.nap(wait)
        return False

    def loops(self):
        return [self.request_loop, self.sync_loop, self.heartbeat_loop, self.housekeeping_loop, self.local_loop]

    async def _forever(self, loop):
        """Run one loop; an error is logged and the loop starts again after 5 s, unless the worker stops."""
        while not self.stopping.is_set():
            try:
                await loop()
                return
            except Exception as e:
                if self.stopped(e):
                    return
                log.exception("the worker's %s failed; it starts again in 5 s", loop.__name__)
                await self.nap(5)

    async def nap(self, seconds):
        """Sleep, but wake up when the worker stops."""
        with anyio.move_on_after(seconds):
            while not self.stopping.is_set():
                await anyio.sleep(0.2)

    async def start(self):
        """The workers row, the jobs rows against the job folders (reconcile), orphan objects, the interrupted
        jobs resumed, and a first publish of every job."""
        if self.client is None:
            self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.inner), base_url="http://worker",
                                            timeout=None)
        self.settings = await self.store.settings()
        if not self.store.machine_id:  # the login came before the approval
            await self.store.bind(self.place)
            self.session.set_machine(self.store.machine_id)
        await self.store.start(self.box.public_b64, self.ends_at, VERSION, list(self.local_keys))
        rows = await self.reconcile()
        try:
            await self.delete_orphans()
        except supa.Error as e:  # the hourly housekeeping tries again
            log.warning("orphan objects stay for now: %s", e)
        await self.resume_interrupted(rows)
        for jid in list(self.jobs):
            await self.publish_safely(jid, force=True)

    async def end(self):
        """At the stop: active jobs keep their status with a note (the next start resumes them), the workers row
        stops, and the clients close."""
        try:
            if self.store.worker_id is None:  # it never started: no row to stop
                return
            for jid, pub in list(self.jobs.items()):
                if not pub.gone and (pub.live or {}).get("status") in jobs.ACTIVE:
                    await self.store.publish(jid, row={"note": ENDED.format(place=self.place)})
            await self.store.stop()
        except Exception as e:
            log.warning("the worker's stop did not reach Supabase: %s", e)
        finally:
            if self.client is not None:
                await self.client.aclose()
            await self.store.close()
            self.session.close()  # the session's lock: login and logout work again while the server runs on

    def stopped(self, e):
        """True (and the worker stops) when Supabase says another worker took over."""
        if isinstance(e, supa.Error) and STOPPED.search(e.message or ""):
            log.warning("%s; this worker stops", e.message)
            self.stopping.set()
            return True
        return False

    def paused(self, e=None):
        """A worker whose account lost its approval pauses the copy for PAUSE_S."""
        if isinstance(e, supa.Error) and (e.message or "").startswith("your account is waiting for approval"):
            log.warning("the account waits for approval again; the copy to the site pauses")
            self.paused_until = time.monotonic() + PAUSE_S
        return time.monotonic() < self.paused_until

    def accepting(self):
        """False in the last settings.near_end_minutes of the Slurm job: builds wait for the next start."""
        if self.ends_at is None:
            return True
        return self.ends_at - now() > timedelta(minutes=self.settings.get("near_end_minutes", 60))

    # ---- the jobs this worker serves ----

    def folders(self):
        """{job uuid: (local id, job.json "public")} of every job folder here whose "public" names a job of this
        user and machine that is not "gone". A job of the tunnel alone has no "public" and is never touched."""
        out = {}
        for f in sorted(self.runner.runs.glob("*/job.json")):
            job = server.read_json(f) if jobs.ID.fullmatch(f.parent.name) else None
            pub = (job or {}).get("public") or {}
            if isinstance(pub, dict) and self.mine(pub):
                out.setdefault(pub["job"], (f.parent.name, pub))
        return out

    async def reconcile(self):
        """The jobs rows against the job folders at a start; the jobs it serves are tracked. A copied job whose
        row is gone gets job.json public.gone and is no longer copied (its folder stays; the user may delete it in
        the local UI), unless its delete request waits: RLS hides a row in "deleting", and the delete still needs
        the folder. A row whose folder is missing here becomes failed, and its files on the site wait for retention
        (folderless). Returns {uuid: row} of the rows it read; it never deletes a folder."""
        rows = {r["id"]: r for r in await self.store.job_rows()}
        rest = dict(rows)  # the rows without a folder here; rows itself goes on to resume_interrupted
        for jid, (local, pub) in self.folders().items():
            row = rest.pop(jid, None)
            if row is not None:
                self.track(jid, local).created = row.get("created_at")
            elif await self.delete_waits(jid):
                self.track(jid, local).gone = True
            else:
                jobs.update(self.runner.runs / local, public={**pub, "gone": True})
        note = GONE.format(place=self.place)
        self.folderless = set(rest)
        for jid, row in rest.items():
            if (row["status"], row.get("note")) != ("failed", note):
                await self.store.publish(jid, row={"status": "failed", "note": note, "queue_pos": None})
        return rows

    async def delete_waits(self, jid):
        """True when a delete request of the job waits or runs."""
        return bool(await self.store.supa.select("requests", "id", job_id=jid, route="delete",
                                                 filters={"state": "in.(pending,taken)"}, limit=1))

    async def resume_interrupted(self, rows):
        """A row that says analyzing, queued or running while job.json says failed (Runner._recover after a crash
        or SIGKILL) or cancelled (the stop) goes on: the analysis again while the frames are not done, else the
        build from the first stage not done (app.js retryFrom), without a key, so a stage that needs one waits in
        review. A job whose cancel was answered after the row's last update stays. RESUMES at most per job, counted
        in job.json public.resumes."""
        order = sorted((rows[jid].get("updated_at") or "", jid) for jid in self.jobs if jid in rows)
        for _, jid in order:
            row, pub = rows[jid], self.jobs[jid]
            if pub.gone or row.get("deleted_at") or row["status"] not in jobs.ACTIVE:
                continue
            try:
                J = self.runner.job_dir(pub.local)
            except ValueError:
                continue
            job = jobs.load(J)
            if job["status"] not in ("failed", "cancelled") or await self.cancelled_since(jid, row.get("updated_at")):
                continue
            n = (job.get("public") or {}).get("resumes", 0)
            if not isinstance(n, int) or n >= RESUMES:
                await self.store.publish(jid, row={"note": TWICE.format(place=self.place)})
                continue
            stages = jobs.progress(J)
            first = relay.first_not_done({"stages": stages, "articulated": steps.articulated(job),
                                          "parts_mode": steps.parts_mode(job)},
                                         job["parts"]["confirmed"] is True)
            status = {s["name"]: s["status"] for s in stages}
            if row["status"] == "analyzing" and first != "frames":
                continue
            if first is None and status["frames"] == "done" and status["extract"] == "pending":
                first = "extract"  # a Start whose build never began
            try:
                if first == "frames":
                    self.runner.submit_analyze(pub.local)
                elif first:
                    self.runner.resume(pub.local, first, None)
                else:
                    continue
            except ValueError as e:
                await self.store.publish(jid, row={"note": relay.scrub(str(e), self.ctx(pub))[:300]})
                continue
            jobs.update(J, public={**job["public"], "resumes": n + 1})
            await self.store.publish(jid, row={"note": None})
            log.info("job %s goes on from %s", jid, first)

    async def cancelled_since(self, jid, t):
        """True when a cancel of the job was answered after t (the row's updated_at)."""
        if not t:
            return False
        return bool(await self.store.supa.select("requests", "id", job_id=jid, route="cancel", state="done",
                                                 filters={"done_at": f"gt.{t}"}, limit=1))

    def mine(self, pub):
        """True for a job.json "public" of this user and machine that is not gone."""
        return isinstance(pub.get("job"), str) and pub.get("user") == self.store.user_id \
            and pub.get("machine") == self.store.machine_id and not pub.get("gone")

    def track(self, jid, local):
        """Serve job jid (local job folder local)."""
        self.jobs.setdefault(jid, Pub(jid, local))
        return self.jobs[jid]

    def lock(self, jid):
        return self.locks.setdefault(jid, anyio.Lock())

    def queue_pos(self, local):
        q = [job_id for job_id, _ in self.runner.queued()]
        return q.index(local) + 1 if local in q else None

    def ctx(self, pub):
        return relay.Ctx(uuid=pub.uuid, local=pub.local, G=str(self.cfg.G), runs=str(self.runner.runs),
                         keys=frozenset(pub.keys), created=pub.created)

    # ---- relayed requests ----

    async def request_loop(self):
        slots = anyio.Semaphore(PARALLEL)
        async with anyio.create_task_group() as tg:
            while not self.stopping.is_set():
                if self.paused():
                    await self.nap(5)
                    continue
                await slots.acquire()
                try:
                    r = await self.store.claim(None if self.accepting() else list(relay.NO_BUILD))
                except Exception as e:  # any error here: the requests in progress go on (no cancel of the group)
                    slots.release()
                    if self.stopped(e):
                        return
                    if not self.paused(e):
                        log.warning("claim failed: %s", e)
                    await self.nap(5)
                    continue
                if r is None:
                    slots.release()
                    await self.nap(OWN_IDLE)
                    continue
                tg.start_soon(self.handle, r, slots)

    async def poll(self):
        """Claim and handle one request, if one waits; True when it did (the loop's step, for tests)."""
        r = await self.store.claim(None if self.accepting() else list(relay.NO_BUILD))
        if r is None:
            return False
        await self.handle(r)
        return True

    async def handle(self, r, slots=None):
        """One claimed request: run it, publish the job, then write the answer."""
        jid = r["job_id"]
        try:
            async with self.lock(jid):
                try:
                    status, out = await self.run_request(r)
                except relay.Refuse as e:
                    status, out = e.status, {"detail": e.detail}
                except Exception:
                    log.exception("request %s (%s) failed", r.get("id"), r.get("route"))  # the full error stays here
                    status, out = 500, {"detail": relay.FAILED}
                pub = self.jobs.get(jid)
                if pub is not None:
                    await self.publish_safely(jid, force=True, locked=True)
                status, out = relay.public_answer(status, out, self.ctx(pub) if pub else relay.Ctx(uuid=jid))
                await self.store.answer(r["id"], status, out)
        except Exception as e:
            if not self.stopped(e):
                log.exception("the answer to request %s did not reach Supabase", r.get("id"))
        finally:
            if slots is not None:
                slots.release()

    async def run_request(self, r):
        jid, route = r["job_id"], r["route"]
        sealed = await self.store.take_key(r)  # off Supabase in every case
        pub = self.jobs.get(jid)
        acc = await self.store.access()
        row = await self.store.job_row(jid)
        self.check(r, row, acc, pub)
        if route == "delete":
            return await self.delete(r, pub)
        J = self.runner.job_dir(pub.local)
        job = jobs.load(J)
        body = dict(r.get("body") or {})
        agent = relay.effective_agent(route, body, job)
        if agent is not None and (no := relay.agent_refusal(agent, acc)):
            raise no
        key = None
        if sealed is not None and agent is not None and route in relay.KEY_ROUTES:
            key = self.open_key(r, sealed, agent)
        elif agent in self.local_keys and route in relay.KEY_ROUTES and (
                route == "start" or not self.runner.has_key(pub.local, agent)):
            key = self.local_key(agent)
        elif r.get("key_ref") and agent is not None and route in relay.KEY_ROUTES and agent not in self.local_keys:
            raise relay.Refuse(400, KEY_GONE)  # the key was sealed to an earlier start of this machine
        if key:
            body["api_key"] = key
            body.pop("key_later", None)
            pub.keys.add(key)
        method, path, content, headers = relay.build(route, pub.local, body, r.get("query"), r.get("body_text"))
        resp = await self.client.request(method, path, content=content, headers=headers)
        try:
            return resp.status_code, resp.json()
        except ValueError:
            return resp.status_code, {"detail": resp.text[:300]}

    def check(self, r, row, acc, pub):
        """Refuse a request that this worker must not run (see the module docstring)."""
        route = r["route"]
        if route != "delete" and not (acc or {}).get("approved"):
            raise relay.Refuse(403, "your account is waiting for approval")
        if r["user_id"] != self.store.user_id:
            raise relay.Refuse(404, "no such job")
        if route != "delete" and (row is None or row.get("machine_id") != self.store.machine_id):
            raise relay.Refuse(404, "no such job on this machine")
        if pub is None and route != "delete":
            raise relay.Refuse(404, f"the job's folder is not on {self.place}")
        if pub is not None and row is not None and pub.created is None:
            pub.created = row.get("created_at")

    def open_key(self, r, sealed, agent):
        """The API key sealed to this worker start, checked like jobs.check_key's first steps."""
        if sealed.get("agent") != agent:
            raise relay.Refuse(400, f"the key was sealed for {sealed.get('agent')}, but this request runs {agent}")
        try:
            aad = keybox.aad(r["job_id"], r["user_id"], agent, self.store.worker_id, r.get("key_ref") or "")
            key = self.box.open(sealed.get("epk"), sealed.get("iv"), sealed.get("ct"), aad)
        except keybox.Invalid:
            raise relay.Refuse(400, "the sealed key does not open; paste the key again") from None
        why = None if jobs.KEY.fullmatch(key) else "an API key is one word of letters, digits, - and _"
        why = why or jobs.key_kind_error(key, agent)
        if why:
            raise relay.Refuse(400, why)
        return key

    def local_key(self, agent):
        """The machine's own key for agent (--local-keys), read now."""
        try:
            if self._local_key is not None:
                return self._local_key(agent)
            from harpor.webapp import session as sessions

            return sessions.local_key(agent)
        except (OSError, ValueError, SystemExit) as e:
            raise relay.Refuse(400, f"{self.place} could not read its {steps.PROVIDER[agent]} key: {e}") from None

    async def delete(self, r, pub):
        """A delete: cancel the job here if it still runs (a machine that was offline), delete its folder, its
        objects and its row."""
        jid = r["job_id"]
        if pub is not None:
            status, out = await self.drop_folder(pub.local)
            if status not in (200, 404):
                return status, out
        await self.remove_objects(jid)
        await self.store.delete_row(jid)
        self.jobs.pop(jid, None)
        return 200, {"deleted": jid}

    async def drop_folder(self, local):
        """Delete a job folder here with DELETE /api/jobs/{local}, after a cancel when the job still runs (waiting up
        to STOP_WAIT s for it to stop). server.py's (status, JSON); 404 when the folder is gone already."""
        J = self.runner.runs / str(local)
        if not jobs.ID.fullmatch(str(local)) or not (J / "job.json").is_file():
            return 404, {"detail": "no such job"}
        if jobs.load(J)["status"] in jobs.ACTIVE:
            await self.client.post(f"/api/jobs/{local}/cancel")
            with anyio.move_on_after(STOP_WAIT):
                while (J / "job.json").is_file() and jobs.load(J)["status"] in jobs.ACTIVE:
                    await anyio.sleep(0.2)
        resp = await self.client.delete(f"/api/jobs/{local}")
        try:
            return resp.status_code, resp.json()
        except ValueError:
            return resp.status_code, {"detail": resp.text[:300]}

    async def remove_objects(self, jid):
        """Delete every Storage object of a job (job_objects), through the Storage API, in rounds of 1,000 until
        fewer come back or a round deletes nothing (the page may have deleted them already)."""
        while True:
            names = await self.store.job_objects(jid)
            if not names or not await self.remove(names) or len(names) < 1000:
                return

    async def remove(self, names):
        """Delete job-files objects; how many went. A name that the Storage API cannot take (supa.quote) is logged
        and stays, so one bad name never stops the rest of a batch."""
        bad = {n for n in names if not supa.object_ok(n)}
        if bad:
            log.warning("%d names that the Storage API cannot take stay: %s", len(bad), sorted(bad)[:3])
        names = [n for n in names if n not in bad]
        return await self.store.remove(names) if names else 0

    # ---- publishing ----

    async def publish(self, jid, force=False):
        async with self.lock(jid):
            await self._publish(jid, force)

    async def publish_safely(self, jid, force=False, locked=False):
        """publish, with errors logged; a stopped worker, a paused account or a job gone from the site stops,
        pauses or ends the copy."""
        try:
            if locked:
                await self._publish(jid, force)
            else:
                await self.publish(jid, force)
        except supa.Error as e:
            if self.stopped(e) or self.paused(e):
                return
            if (e.message or "").startswith("no such job on this machine") and jid in self.jobs:
                self.jobs[jid].gone = True  # deleted on the site; a delete request may still come
                return
            log.warning("publishing job %s failed: %s", jid, e)
        except Exception:
            log.exception("publishing job %s failed", jid)

    async def _publish(self, jid, force=False):
        """Logs, files, view, live state, snapshots and row of one job, in that order (see the module docstring).
        force: publish now, even when only the progress changed."""
        pub = self.jobs.get(jid)
        if pub is None or pub.gone:
            return
        try:
            J = self.runner.job_dir(pub.local)
        except ValueError:  # the folder went (a delete in the tunnel): the next start's reconcile says so
            return
        if pub.created is None:
            row = await self.store.job_row(jid)
            if row is None:
                pub.gone = True
                return
            pub.created = row.get("created_at")
        state = await self.push_logs(pub, J)
        if not force and pub.live is not None and relay.coarse(state) == relay.coarse(pub.live) \
                and time.monotonic() - pub.at < THROTTLE:
            return
        mtime = (J / "job.json").stat().st_mtime_ns
        for _ in range(SETTLE):  # the answers must not show a stage that ended after its files were looked at
            job = jobs.load(J)
            notes = await self.mirror(pub, J, job)
            raw = (await self.client.get(f"/api/jobs/{pub.local}")).json()
            answers = {path: await self.client.get(f"/api/jobs/{pub.local}/{path}") for path in supa.SNAPSHOTS}
            if stage_states(jobs.load(J)) == stage_states(job):
                break
        ctx = self.ctx(pub)
        view = relay.public_view(raw, ctx, pub.manifest, notes)
        live = relay.state_of(view)
        thumbs = {int(p[len("frames/thumbs/"):-4]) for p in pub.manifest if p.startswith("frames/thumbs/")} \
            if self.settings.get("lean_files", True) else None
        snaps = {}
        for path, resp in answers.items():
            status, body = relay.public_snapshot(path, resp.status_code, resp.json(), ctx, thumbs)
            if pub.snaps.get(path) != {"status": status, "body": body}:
                snaps[path] = {"status": status, "body": body}
        row = relay.row_fields(view, self.queue_pos(pub.local))
        if relay.size_of(view) > relay.VIEW_LIMIT or relay.size_of(live) > relay.LIVE_LIMIT:
            log.warning("the view of job %s is too large for the site", jid)
            view, live = pub.view, pub.live
        await self.store.publish(jid, view=view if view != pub.view else None,
                                 live=live if live != pub.live else None, snapshots=snaps or None,
                                 row=row if row != pub.row else None)
        pub.view, pub.live, pub.row, pub.at, pub.mtime = view, live, row, time.monotonic(), mtime
        pub.snaps.update(snaps)
        if live["status"] in ("done", "cancelled") and not self.runner.has_key(pub.local):
            pub.keys.clear()  # the runner dropped its key; every file of the job was scanned with it

    async def push_logs(self, pub, J):
        """The new log text of the job's running stage (server.tick), scrubbed and capped; the job's state."""
        rows, state = [], None
        for _ in range(TICKS):  # until the cursor stops: a tick reads one stage's log, or moves to the next stage
            before = pub.cursor
            events, pub.cursor, state, _ = await anyio.to_thread.run_sync(server.tick, J, pub.cursor, None)
            rows += relay.log_rows(events, pub.cursor)
            if pub.cursor == before:
                break
        if pub.logs_full or not rows:
            return state
        ctx, out = self.ctx(pub), []
        for x in rows:
            k = (x["stage"], x["run_started"])
            sent = pub.logged.get(k, 0)
            if sent >= LOG_CAP:
                continue
            text = relay.scrub(x["text"], ctx)[:LOG_CAP - sent]
            if sent + len(text) >= LOG_CAP:
                text += LOG_CUT
            pub.logged[k] = sent + len(text)
            out += [{**x, "text": part} for part in relay.split_text(text)]
        if out:
            try:
                await self.store.logs(pub.uuid, out)
            except supa.Error as e:
                if not (e.message or "").startswith("the job's log on the site is full"):
                    raise
                pub.logs_full = True  # the full log stays on the machine
        return state

    async def mirror(self, pub, J, job):
        """Upload the files the site should hold and change job_files (relay.mirror_plan); notes on the files
        left out, for the view's warnings."""
        if pub.manifest is None:
            pub.manifest = await self.store.manifest(pub.uuid)
        lean, limit = self.settings.get("lean_files", True), self.settings.get("plan_file_limit", 50 << 20)
        ctx, notes, put, left, full = self.ctx(pub), [], [], set(), None
        stop = stop_time(job)  # retention: the files of a job that stopped long ago leave the site
        roles = expired_roles((time.time() - stop) / 86400) if stop else set()
        expired = {r: RESULTS_EXPIRED if r == "result" else REVIEW_EXPIRED for r in roles}
        for f in await anyio.to_thread.run_sync(relay.mirror_plan, J, job, lean):
            if f.role in expired:
                left.add(f.rel)
                notes += [expired[f.role]] if expired[f.role] not in notes else []
                continue
            try:
                st = f.path.stat()
            except OSError:
                continue
            stamp = (st.st_size, st.st_mtime_ns, frozenset(pub.keys), lean, limit)
            seen = pub.seen.get(f.rel)
            data = None
            if seen is None or seen[0] != stamp:
                sha, note, data, ctype = await anyio.to_thread.run_sync(self.read_file, f, ctx, lean, limit, pub.keys)
                pub.seen[f.rel] = seen = (stamp, sha, note)
            _, sha, note = seen
            if note:
                notes.append(note)
                left.add(f.rel)
                continue
            if pub.manifest.get(f.rel, {}).get("sha256") == sha:
                continue
            if data is None:
                sha, note, data, ctype = await anyio.to_thread.run_sync(self.read_file, f, ctx, lean, limit, pub.keys)
                if note or sha is None:
                    continue
            name = supa.object_name(self.store.user_id, pub.uuid, sha, f.rel)
            try:
                await self.store.upload(name, data, ctype)
            except supa.Error as e:
                if not refused_upload(e):  # no answer, a 5xx or a session that ended: the next publish tries again
                    raise
                if refused_by_policy(e):  # tried again at each publish: room may come
                    full = full or await self.quota_note()
                    notes.append(f"{f.rel} left out: {full}")
                else:  # too large, a type the bucket refuses, ...: not tried again until the file changes
                    note = f"{f.rel} left out: the site's storage refused it ({e.message or e.status})"
                    pub.seen[f.rel] = (stamp, sha, note)
                    notes.append(note)
                left.add(f.rel)
                continue
            self.trash.pop(name, None)
            put.append({"path": f.rel, "object": name, "bytes": len(data), "sha256": sha, "role": f.role,
                        "content_type": ctype})
        drop = [p for p in pub.manifest if p in left or server.resolve(J, p) is None]
        drop = [p for p in drop if p not in {x["path"] for x in put}]
        if put or drop:
            await self.store.files(pub.uuid, put, drop)
            due = time.monotonic() + OLD_OBJECT_S
            for p in drop:
                self.trash[pub.manifest.pop(p)["object"]] = due
            for x in put:
                old = pub.manifest.get(x["path"])
                if old and old["object"] != x["object"]:
                    self.trash[old["object"]] = due
                pub.manifest[x["path"]] = {"object": x["object"], "sha256": x["sha256"]}
        return notes

    def read_file(self, f, ctx, lean, limit, keys):
        """(sha256, note, data, content type) of a mirrored file as the site gets it; a note instead when it
        stays here: too large, over the lean preview limit, or it holds a key of the job."""
        try:
            data, ctype = relay.file_bytes(f, ctx, lean)
        except relay.Skip as e:
            return None, str(e), None, None
        except OSError:
            return None, None, None, None
        if len(data) > limit:
            return None, f"{f.rel} ({mb(len(data))} MB) is over the site's {mb(limit)} MB file limit; it stays on " \
                         f"{self.place}", None, None
        if relay.holds_key(data, keys, shape=True):  # also after a restart, when the worker holds no key
            return None, f"{f.rel} left out: it holds the API key", None, None
        return hashlib.sha256(data).hexdigest(), None, data, ctype

    async def quota_note(self):
        """Why an upload was refused: the user's storage on the site is full."""
        try:
            used, acc = await self.store.storage_used(), await self.store.access()
            cap = (acc or {}).get("max_result_bytes", 0)
            return f"your storage on the site is full ({mb(used)} of {mb(cap)} MB); the file stays on {self.place}"
        except supa.Error:
            return f"your storage on the site is full; the file stays on {self.place}"

    async def empty_trash(self):
        """Delete the objects whose 10 minutes are over."""
        due = [name for name, t in self.trash.items() if t <= time.monotonic()]
        if due:
            await self.remove(due)
            for name in due:
                self.trash.pop(name, None)

    # ---- the other loops ----

    async def sync_loop(self):
        """Every second: publish the jobs that run (or just stopped); every 5 s also those whose job.json
        changed, and old objects go."""
        n = 0
        while not self.stopping.is_set():
            n += 1
            if not self.paused():
                await self.sync_once(every=n % 5 == 0)
            await self.nap(1)

    async def sync_once(self, every=True):
        """One step of the sync loop: jobs that were active at their last publish (their logs and progress
        change outside job.json); jobs whose job.json changed when it now says active, or with every."""
        for jid, pub in list(self.jobs.items()):
            if pub.gone:
                continue
            active = (pub.live or {}).get("status") in jobs.ACTIVE
            if not active:
                try:
                    J = self.runner.job_dir(pub.local)
                    if (J / "job.json").stat().st_mtime_ns == pub.mtime:
                        continue
                    active = jobs.load(J)["status"] in jobs.ACTIVE
                except (ValueError, OSError):
                    continue
            if active or every:
                await self.publish_safely(jid)
        try:
            await self.empty_trash()
        except supa.Error as e:
            log.warning("old objects stay for now: %s", e)

    async def heartbeat_loop(self):
        """Every 10 s: the heartbeat, and the queue places of queued jobs; every SQUEUE_S the Slurm job's end."""
        read = time.monotonic()
        while not self.stopping.is_set():
            if time.monotonic() - read >= SQUEUE_S:
                read = time.monotonic()
                self.ends_at = await anyio.to_thread.run_sync(slurm_end) or self.ends_at
            await self.beat()
            await self.nap(10)

    async def beat(self):
        if self.paused():  # the workers row goes stale: the site shows the machine offline
            return
        try:
            await self.store.beat(len(self.runner.queued()), self.accepting(), self.ends_at)
        except supa.Error as e:
            if self.stopped(e):
                return
            if not self.paused(e):
                log.warning("heartbeat failed: %s", e)
        for jid, pub in list(self.jobs.items()):
            pos = self.queue_pos(pub.local)
            if pub.row is not None and pub.row.get("queue_pos") != pos and pub.row.get("status") == "queued":
                await self.publish_safely(jid, force=True)

    async def local_loop(self):
        """Every 5 s: new local jobs become rows on the site."""
        while not self.stopping.is_set():
            if not self.paused():
                await self.local_jobs()
            await self.nap(5)

    async def local_jobs(self):
        """Local jobs whose upload ended at or after the session's mirror_since, with no job.json "public", get
        a row (own_job_create) and "public": {job, user, machine, resumes: 0}. Jobs made before the first
        login stay local."""
        since = self.session.mirror_since
        for f in sorted(self.runner.runs.glob("*/job.json")):
            local = f.parent.name
            job = server.read_json(f) if jobs.ID.fullmatch(local) else None
            video = (job or {}).get("video") or {}
            if not job or job.get("public") or not video.get("bytes") or self.runner.uploading(local):
                continue
            if since and job.get("created", "") < since or self.refused.get(local, 0) > time.monotonic():
                continue
            try:
                jid = await self.store.create_job(relay.site_name(video["name"]), steps.source(job)["kind"],
                                                  video["bytes"])
            except supa.Error as e:
                if self.stopped(e) or self.paused(e):
                    return
                log.warning("job %s stays on this machine for now: %s", local, e.message or e)
                self.refused[local] = time.monotonic() + PAUSE_S
                continue
            jobs.update(f.parent, public={"job": jid, "user": self.store.user_id, "machine": self.store.machine_id,
                                          "resumes": 0})
            self.track(jid, local)
            await self.publish_safely(jid, force=True)

    # ---- housekeeping ----

    async def housekeeping_loop(self):
        """Every HOUSEKEEPING_S; the hourly part every HOURLY_S (start() deleted the orphans already)."""
        hour = time.monotonic()
        while not self.stopping.is_set():
            await self.nap(HOUSEKEEPING_S)
            if self.stopping.is_set():
                return
            hourly = time.monotonic() - hour >= HOURLY_S
            hour = time.monotonic() if hourly else hour
            await self.housekeeping(hourly)

    async def housekeeping(self, hourly=False):
        """One round: the settings row again (the owner may change it), and keys of jobs in review over KEY_HOURS;
        hourly also retention and orphan objects. A step that fails is logged, and the others still run."""
        if self.paused():
            return
        steps_ = [self.read_settings, self.forget_old_keys] + ([self.retention, self.delete_orphans] if hourly else [])
        for step in steps_:
            try:
                await step()
            except supa.Error as e:
                if self.stopped(e):
                    return
                log.warning("housekeeping (%s): %s", step.__name__, e)
            except Exception:
                log.exception("housekeeping (%s) failed", step.__name__)

    async def read_settings(self):
        self.settings = await self.store.settings()

    async def retention(self):
        """Hourly, counted from when a job stopped: a job whose files on the site still
        include review files or logs after REVIEW_DAYS, or results after RESULT_DAYS, is published again. mirror()
        leaves those files out with a note and drops them from job_files, and their objects go 10 minutes later. A job
        whose folder is gone (reconcile, or a delete in the local UI while the server runs) goes through
        expire_files instead."""
        for jid, pub in list(self.jobs.items()):
            if pub.gone:
                continue
            job = server.read_json(self.runner.runs / pub.local / "job.json")
            if job is None:
                self.folderless.add(jid)
                continue
            stop = stop_time(job) if pub.manifest else None
            expired = expired_roles((time.time() - stop) / 86400) if stop is not None else set()
            if any(relay.role_of(p) in expired for p in pub.manifest or {}):
                await self.publish_safely(jid, force=True)
        for jid in sorted(self.folderless):
            try:
                await self.expire_files(jid)
            except supa.Error as e:
                if self.stopped(e) or self.paused(e):
                    return
                log.warning("retention of job %s: %s", jid, e)

    async def expire_files(self, jid):
        """Retention of a job whose folder is gone from this machine, counted from its row's stopped_at, else its
        updated_at: the expired files leave job_files, and their objects go 10 minutes later. The job leaves the list
        when its row is gone or no file is left on the site."""
        row = await self.store.job_row(jid)
        t = (row or {}).get("stopped_at") or (row or {}).get("updated_at")
        if not t:
            self.folderless.discard(jid)
            return
        expired = expired_roles((now() - datetime.fromisoformat(t)).total_seconds() / 86400)
        manifest = await self.store.manifest(jid)
        drop = [p for p in manifest if relay.role_of(p) in expired]
        if drop:
            await self.store.files(jid, [], drop)
            due = time.monotonic() + OLD_OBJECT_S
            for p in drop:
                self.trash[manifest[p]["object"]] = due
        if len(drop) == len(manifest):
            self.folderless.discard(jid)

    async def delete_orphans(self):
        """Objects in the user's folder that nothing names (own_orphans), 1,000 a round. An object of the trash
        keeps its 10 minutes."""
        for _ in range(ORPHAN_ROUNDS):
            names = await self.store.orphans(1000)
            gone = await self.remove([n for n in names if n not in self.trash])
            if len(names) < 1000 or not gone:
                return

    async def forget_old_keys(self):
        """The key of a job that waited in review for more than KEY_HOURS goes (Runner.forget_key); the job asks for
        it again at Build."""
        for jid, pub in list(self.jobs.items()):
            job = None if pub.gone else server.read_json(self.runner.runs / pub.local / "job.json")
            stop = stop_time(job) if job and job.get("status") == "review" else None
            if stop is not None and time.time() - stop > KEY_HOURS * 3600 and self.runner.forget_key(pub.local):
                log.info("job %s waited in review for over %d h; its key is dropped", jid, KEY_HOURS)
                await self.publish_safely(jid, force=True)


def stop_time(job):
    """When the job last stopped (its newest stage end, in seconds since the epoch), or None while it is active or
    before any stage ended."""
    if not job or job.get("status") in jobs.ACTIVE:
        return None
    ends = [st.get("ended") for st in (job.get("stages") or {}).values()
            if isinstance(st, dict) and isinstance(st.get("ended"), (int, float))]
    return max(ends) if ends else None


def expired_roles(days):
    """The file roles whose time on the site is over, days after a job stopped."""
    return ({"review", "log"} if days > REVIEW_DAYS else set()) | ({"result"} if days > RESULT_DAYS else set())


def check_local_keys(agents):
    """Refuse (SystemExit) a start with --local-keys unless each agent's key file (session.KEY_FILES) is a file of
    this user with mode 0600 in a folder of mode 0700. Only the modes are looked at here; the key is read when a
    job needs it (session.local_key)."""
    from harpor.webapp.session import KEY_FILES

    folder = config_dir()
    private_folder(folder)
    for agent in agents:
        path = folder / KEY_FILES[agent]
        try:
            st = os.lstat(path)
        except FileNotFoundError:
            raise SystemExit(f"--local-keys {agent} needs {path} with the {steps.PROVIDER[agent]} key "
                             f"({steps.KEY_VAR[agent]}=...), mode 0600") from None
        if not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid() or st.st_mode & 0o077:
            raise SystemExit(f"{path} must be a file of this user with mode 0600 (chmod 600 {path})")


def stage_states(job):
    """The status of each stage in job.json."""
    return {name: st.get("status") for name, st in (job.get("stages") or {}).items()}


def mb(n):
    """n bytes in MB for a message: one decimal below 10 MB."""
    return f"{n / 1e6:.1f}" if n < 10e6 else f"{n / 1e6:.0f}"


def refused_by_policy(e):
    """True for Storage's refusal of an upload by the bucket policies (the user's storage is full)."""
    return e.status in (400, 403) and ("row-level security" in (e.message or "") or e.status == 403)


def refused_upload(e):
    """True when Storage refused one upload for good (a 4xx: the policies, the file limit, the content type), so the
    file is left out and the rest of the publish goes on. False for no answer, a 5xx, a 429 and an ended session,
    which the next publish tries again."""
    return not isinstance(e, supa.SignedOut) and e.status is not None and 400 <= e.status < 500 \
        and e.status not in (401, 408, 429)


def main(argv=None):
    """python -m harpor.webapp.worker login | status | logout | add: the personal worker's commands.
    Exit codes: 0 for success, 1 for an error with a one-line reason, 2 for wrong arguments."""
    ap = argparse.ArgumentParser(prog="python -m harpor.webapp.worker",
                                 description="The personal worker's commands. Start the worker itself with python -m "
                                             "harpor.webapp --worker.")
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("login", help="sign in to the HARPOR site and add this machine")
    p.add_argument("--provider", choices=supa.PROVIDERS, default="github")
    p.add_argument("--name", help="this machine's name on the site (default: the host name)")
    p.add_argument("--port", type=int, default=8767, help="the sign-in's loopback port on 127.0.0.1")
    p.add_argument("--no-browser", action="store_true", help="only print the sign-in page (forward the port first)")
    p.add_argument("--site-config", help="the site's site.json (default: harpor/webapp/site.json)")
    sub.add_parser("status", help="the account, this machine and the session")
    sub.add_parser("logout", help="sign out on this machine and delete the session file")
    p = sub.add_parser("add", help="add a local input to the HARPOR server on this machine")
    p.add_argument("path")
    p.add_argument("--name", help="the file name the job gets (default: the file's own)")
    a = ap.parse_args(argv)
    from harpor.webapp import session as sessions

    if a.command == "login":
        if not 0 <= a.port <= 65535:
            ap.error("--port is 0 to 65535")
        return sessions.login(a.provider, a.name, a.port, not a.no_browser, a.site_config)
    if a.command == "add":
        return sessions.add(a.path, a.name)
    return sessions.status() if a.command == "status" else sessions.logout()


if __name__ == "__main__":
    raise SystemExit(main())
