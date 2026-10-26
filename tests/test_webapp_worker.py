"""The personal worker (gator/webapp/worker.py) end to end in fake mode: a real server app and Runner on tmp_path,
the fake Supabase of tests/fake_supabase.py, and the worker driven from the test's event loop (start, poll,
sync_once, end) or in its own thread. Also the server's --worker flags."""

import io
import json
import re
import secrets
import shutil
import threading
import time
import uuid
import zipfile
from datetime import timedelta
from pathlib import Path

import anyio
import httpx
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("av")
pytest.importorskip("cryptography")
from fastapi.testclient import TestClient  # noqa: E402

import fake_supabase as fs  # noqa: E402
from gator.webapp import jobs, relay, server, steps, supa, worker  # noqa: E402
from test_webapp_server import write_video  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "tok-" + secrets.token_hex(16)
OPUS = "sk-ant-test-" + secrets.token_hex(16)
ASTRA = "sk-test-" + secrets.token_hex(16)
START = {"mode": "auto", "prompt": "cup", "n_views": 4, "auto_continue": False}


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    return write_video(tmp_path_factory.mktemp("video") / "cup.mp4").read_bytes()


@pytest.fixture(scope="module")
def template(tmp_path_factory, video):
    """A job folder of a video that the frames stage analyzed ("ready"), made by a real server."""
    d = tmp_path_factory.mktemp("template")
    cfg = steps.Config(G=d / "G", runs=d / "runs", fake=True)
    with TestClient(server.make_app(cfg, TOKEN, key_dir=d / "keys")) as c:
        c.get(f"/?token={TOKEN}")
        job_id = c.put("/api/upload", params={"name": "cup.mp4"}, content=video).json()["id"]
        t0 = time.time()
        while c.get(f"/api/jobs/{job_id}").json()["status"] != "ready":
            assert time.time() - t0 < 60
            time.sleep(0.1)
    return cfg.runs / job_id


class Server:
    """One HARPOR server: make_app with its Runner on a runs folder of its own."""

    def __init__(self, tmp_path, name="runs"):
        self.cfg = steps.Config(G=tmp_path / "G", runs=tmp_path / name, fake=True)
        self.gate = server.make_app(self.cfg, TOKEN, key_dir=tmp_path / f"{name}_keys")
        self.runner = self.gate.app.state.runner

    def clone(self, template, **fields):
        """A copy of the template job under a new id; returns (local id, job folder)."""
        local = time.strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(3)
        J = self.cfg.runs / local
        shutil.copytree(template, J)
        job = jobs.load(J)
        job.update(id=local, video={**job["video"], "path": str(J / "video" / job["video"]["name"])}, **fields)
        jobs.save(J, job)
        return local, J

    def job(self, local):
        return jobs.load(self.cfg.runs / local)


@pytest.fixture
def srv(tmp_path):
    s = Server(tmp_path)
    yield s
    s.runner.shutdown(timeout=5)


@pytest.fixture
def fake():
    return fs.FakeSupabase()


def own_worker(srv, fake, user, machine=None, local_keys=(), local_key=None, mirror_since="2000-01-01T00:00:00"):
    """The personal worker of srv, signed in as user on machine (None: it binds a new one at its start)."""
    session = fake.session(user, machine_id=machine, machine_name="lab-box", mirror_since=mirror_since)
    store = supa.Store.of(session, transport=fake.transport, wait=0)
    return worker.Worker(srv.gate, srv.cfg, local_keys, store=store, session=session, local_key=local_key)


def own_job(fake, srv, local, user, machine, status="uploaded"):
    """A local job that the worker copied to the site: its row and job.json "public"."""
    J = srv.cfg.runs / local
    row = fake.put("jobs", {"user_id": user, "machine_id": machine, "input_name": "cup.mp4", "input_kind": "video",
                            "input_bytes": jobs.load(J)["video"]["bytes"], "status": status})
    jobs.update(J, public={"job": row["id"], "user": user, "machine": machine, "resumes": 0})
    return row["id"]


def machine(fake, user, name="lab-box"):
    return fake.put("machines", {"user_id": user, "name": name})["id"]


def served(srv, fake, template):
    """(user, machine, local job, its folder, its uuid, worker): an approved user's machine, a copy of template
    that the worker copied to the site, and the worker."""
    c = fake.add_user("c@example.org", approved=True)
    mid = machine(fake, c)
    local, J = srv.clone(template)
    jid = own_job(fake, srv, local, c, mid)
    return c, mid, local, J, jid, own_worker(srv, fake, c, mid)


def sealed(fake, w, key, jid, user, agent="opus", **kw):
    return fs.sealed(fake.row("workers", id=w.store.worker_id), key, jid, user, agent, **kw)


def answer(fake, rid):
    r = fake.row("requests", id=rid)
    return r["state"], r["status"], r["response"]


def leaked(fake, *secret):
    """The secrets that Supabase holds anywhere users or the page could read them: every table but the sealed
    job_secrets rows, and every Storage object."""
    with fake.lock:
        text = json.dumps({t: [fake._out(r) for r in rows.values()] for t, rows in fake.t.items()
                           if t != "job_secrets"})
        blobs = [o["data"] for o in fake.objects.values()]
    return [s for s in secret if s in text or any(s.encode() in b for b in blobs)]


async def until(cond, w=None, timeout=90):
    """Wait for cond(); the worker's sync loop runs meanwhile, one step each 0.2 s, and once more after cond()."""
    t0 = time.monotonic()
    while True:
        ok = cond()
        if w is not None:
            await w.sync_once()
        if ok:
            return
        if time.monotonic() - t0 > timeout:
            raise AssertionError("timed out")
        await anyio.sleep(0.2)


def status_is(srv, local, *statuses):
    return lambda: srv.job(local)["status"] in statuses


def files(fake, jid):
    return {f["path"]: f for f in fake.rows("job_files", job_id=jid)}


def watch_live(fake, jid, kind, name):
    """[(state, paths in job_files)] at each write of the job's live state (kind, name: the write that does it)."""
    seen = []

    def watch(caller, k, n):
        if (k, n) == (kind, name):
            live = fake.row("job_live", job_id=jid)
            if live:
                seen.append((live["state"], set(files(fake, jid))))

    fake.watch = watch
    return seen


def check_files_before_states(seen):
    """Every published state whose stage is done already had that stage's files on the site."""
    needs = {"frames": "frames/thumbs/0.jpg", "select": "selection.json", "place": "run/posed.glb",
             "package": "results.zip"}
    checked = set()
    for state, paths in seen:
        done = {s["name"] for s in state["stages"] if s["status"] == "done"}
        for stage, path in needs.items():
            if stage in done:
                assert path in paths, (stage, state["status"])
                checked.add(stage)
    return checked


def stopped_days_ago(J, days):
    """Set every stage end of job folder J to days ago."""
    t = time.time() - days * 86400
    for name, st in jobs.load(J)["stages"].items():
        if st.get("ended"):
            jobs.update_stage(J, name, ended=t)


# ---- relayed requests and publishing ----

def test_worker_relays_publishes_and_answers_last(srv, fake, template):
    c, mid, local, J, jid, w = served(srv, fake, template)
    seen = watch_live(fake, jid, "rpc", "own_publish")

    async def main():
        await w.start()
        try:
            wid = w.store.worker_id
            assert fake.row("workers", id=wid)["public_key"] == w.box.public_b64
            # the first publish: the view as GET /api/jobs/{id} after the rewrite, the review files, the state
            raw = (await w.client.get(f"/api/jobs/{local}")).json()
            pub = w.jobs[jid]
            want = relay.public_view(raw, w.ctx(pub), pub.manifest)
            assert fake.row("job_views", job_id=jid)["body"] == want
            assert want["id"] == jid and "public" not in want and "path" not in want["video"]
            assert want["created"] == fake.row("jobs", id=jid)["created_at"].isoformat()
            assert fake.row("job_live", job_id=jid)["state"] == relay.state_of(want)
            frames = fake.row("job_snapshots", job_id=jid, path="frames")
            assert frames["status"] == 200 and frames["body"]["thumbs"] == (await w.client.get(
                f"/api/jobs/{local}/frames")).json()["thumbs"]
            assert {f"frames/thumbs/{i}.jpg" for i in frames["body"]["thumbs"]} | {"frames/preview.mp4"} <= set(
                files(fake, jid))
            assert fake.row("job_snapshots", job_id=jid, path="selection")["status"] == 404
            assert fake.row("jobs", id=jid)["status"] == "ready" and fake.row("jobs", id=jid)["stopped_at"]
            # Start with an Anthropic key sealed in the page
            key = sealed(fake, w, OPUS, jid, c)
            sent = fake.rpc(c, "relay", p_job=jid, p_route="start", p_body={**START, "agent": "opus"}, p_sealed=key)
            assert sent["worker_online"] is True and sent["accepting"] is True
            assert fake.rows("job_secrets")
            fake.calls.clear()
            assert await w.poll() is True
            assert answer(fake, sent["id"]) == ("done", 200, {"status": "queued", "key": "fake mode: key accepted"})
            assert fake.rows("job_secrets") == []  # taken with the claim and deleted
            # the order of writes: logs, uploads and the file list, then one own_publish, and the answer last
            calls = [(k, n) for _, k, n in fake.calls]
            pub_at, ans_at = calls.index(("rpc", "own_publish")), calls.index(("rpc", "own_answer"))
            assert pub_at < ans_at and calls.count(("rpc", "own_publish")) == 1
            assert all(i < pub_at for i, (k, n) in enumerate(calls) if k == "upload" or n in ("own_logs", "own_files"))
            assert w.jobs[jid].keys == {OPUS}
            await until(status_is(srv, local, "review"), w)
            await w.sync_once()
            got = files(fake, jid)
            assert {"selection.json", "logs/select.log", "run/input/images/0.png"} <= set(got)
            assert any(p.startswith("scene/sam3_cup/check/") for p in got)
            assert got["run/input/images/0.png"]["content_type"] == "image/jpeg"  # lean mode
            assert got["selection.json"]["role"] == "result" and got["logs/select.log"]["role"] == "log"
            for f in got.values():
                assert f["object"] == supa.object_name(c, jid, f["sha256"], f["path"])
                assert ("job-files", f["object"]) in fake.objects
            sel = fake.row("job_snapshots", job_id=jid, path="selection")
            assert sel["status"] == 200 and sel["body"]["check_dir"] == "scene/sam3_cup/check"
            logs = fake.rows("job_logs", job_id=jid)
            assert logs and {x["stage"] for x in logs} >= {"select"}
            assert not any(line.startswith("PROGRESS") for x in logs for line in x["text"].splitlines())
            assert fake.row("job_views", job_id=jid)["body"]["status"] == "review"
            # Build: the runner holds the key, so no key travels
            sent = fake.rpc(c, "relay", p_job=jid, p_route="continue", p_body={})
            assert await w.poll() is True
            assert answer(fake, sent["id"])[:2] == ("done", 200)
            await until(status_is(srv, local, "done", "failed"), w)
            await w.sync_once()
            assert srv.job(local)["status"] == "done", srv.job(local)["error"]
            got = files(fake, jid)
            local_files = [p for p in server.OUTPUTS if server.resolve(J, p)]
            assert set(local_files) <= set(got) and "results.zip" in local_files
            zipped = fake.objects[("job-files", got["results.zip"]["object"])]["data"]
            with zipfile.ZipFile(io.BytesIO(zipped)) as site, zipfile.ZipFile(J / "results.zip") as here:
                assert site.namelist() == here.namelist() and "summary.json" in site.namelist()
                for name in site.namelist():
                    if Path(name).suffix in relay.TEXT:
                        text = site.read(name).decode()
                        assert local not in text and str(srv.cfg.G) not in text, name
                    else:
                        assert site.read(name) == here.read(name), name
                assert str(J) in here.read("summary.json").decode()  # the folder's own copy keeps its paths
            body = fake.row("job_views", job_id=jid)["body"]
            assert body["status"] == "done" and set(body["files"]) == set(local_files)
            assert fake.row("jobs", id=jid)["status"] == "done"
        finally:
            await w.end()

    anyio.run(main)
    assert fake.row("workers", id=w.store.worker_id)["stopped_at"] is not None
    assert check_files_before_states(seen) >= {"frames", "select", "place", "package"}
    assert leaked(fake, OPUS, str(srv.cfg.G), local) == []
    assert {who for who, _, _ in fake.calls} == {c}  # the user's own session, and nothing else


def test_requests_the_worker_refuses(srv, fake, template):
    c = fake.add_user("c@example.org", approved=True)
    d = fake.add_user("d@example.org", approved=True)
    mid = machine(fake, c)
    local, J = srv.clone(template)
    jid = own_job(fake, srv, local, c, mid)
    w = own_worker(srv, fake, c, mid)

    async def main():
        await w.start()

        async def relayed(route="start", body=None, key=None, agent="opus", **kw):
            sent = fake.rpc(c, "relay", p_job=jid, p_route=route, p_body=body if body is not None else
                            {**START, "agent": agent}, p_sealed=key, **kw)
            assert await w.poll() is True
            assert fake.rows("job_secrets") == []  # gone in every case
            return answer(fake, sent["id"])[1:]

        try:
            # the agent: access.agents, also for a start that names none (astra) and a job whose agent is none
            fake.set("access", {"user_id": c}, agents=["opus"])
            assert await relayed(body={**START}) == (403, {"detail": "this account may use the agents opus"})
            assert await relayed(agent="astra") == (403, {"detail": "this account may use the agents opus"})
            fake.set("access", {"user_id": c}, agents=["astra", "opus"])
            jobs.update(J, agent="none")
            assert await relayed("continue", body={}) == (
                403, {"detail": "this account may use the agents astra, opus"})
            jobs.update(J, agent="astra")
            # sealed keys: for another job, for another agent, of the other provider
            other = sealed(fake, w, OPUS, "6f1c0d2e-0000-4000-8000-00000000abcd", c)
            assert await relayed(key={**other, "ref": other["ref"]}) == (
                400, {"detail": "the sealed key does not open; paste the key again"})
            assert await relayed(key=sealed(fake, w, ASTRA, jid, c, "astra")) == (
                400, {"detail": "the key was sealed for astra, but this request runs opus"})
            assert await relayed(key=sealed(fake, w, ASTRA, jid, c)) == (
                400, {"detail": "Anthropic keys start with sk-ant-"})
            # approval taken back after the request was sent: the claim is refused, and the request waits
            sent = fake.rpc(c, "relay", p_job=jid, p_route="cancel")
            fake.set("access", {"user_id": c}, approved=False)
            with pytest.raises(supa.Error, match="your account is waiting for approval"):
                await w.poll()
            fake.set("access", {"user_id": c}, approved=True)
            assert await w.poll() and answer(fake, sent["id"])[1:] == (200, {"status": "ready"})
            # a row of this machine whose folder is not here
            ghost = fake.put("jobs", {"user_id": c, "machine_id": mid, "input_name": "x.mp4", "input_kind": "video",
                                      "input_bytes": 5, "status": "ready"})["id"]
            sent = fake.rpc(c, "relay", p_job=ghost, p_route="cancel")
            assert await w.poll() and answer(fake, sent["id"])[1:] == (
                404, {"detail": "the job's folder is not on lab-box"})
            # what no claim hands this worker is refused again: another user's request, another machine's row
            row = fake.row("jobs", id=jid)
            for r, rw, why in (({"route": "cancel", "user_id": d}, row, "no such job"),
                               ({"route": "cancel", "user_id": c}, {**row, "machine_id": str(uuid.uuid4())},
                                "no such job on this machine"),
                               ({"route": "cancel", "user_id": c}, row, "your account is waiting for approval")):
                with pytest.raises(relay.Refuse) as e:
                    w.check(r, rw, {"approved": why != "your account is waiting for approval"}, w.jobs[jid])
                assert e.value.detail == why
            # what server.py refuses goes back as it is, rewritten
            assert await relayed(body={"mode": "auto", "prompt": "", "agent": "opus", "key_later": True}) == (
                400, {"detail": "the object noun has 1 to 60 characters"})
            assert srv.job(local)["status"] == "ready"  # nothing ran
        finally:
            await w.end()

    anyio.run(main)


def test_near_end_and_offline_requests(srv, fake, template):
    """Requests wait as pending while the machine is offline, and run when its worker starts. Near the end of the
    Slurm job the worker takes only requests that start no build; a Start waits for the next start."""
    c = fake.add_user("c@example.org", approved=True)
    mid = machine(fake, c)
    jid1 = own_job(fake, srv, srv.clone(template)[0], c, mid, status="ready")
    jid2 = own_job(fake, srv, srv.clone(template)[0], c, mid, status="ready")
    start = fake.rpc(c, "relay", p_job=jid1, p_route="start", p_body={**START, "agent": "opus", "key_later": True})
    cancel = fake.rpc(c, "relay", p_job=jid2, p_route="cancel")
    assert start["worker_online"] is False and cancel["worker_online"] is False
    w = own_worker(srv, fake, c, mid)
    w.ends_at = worker.now() + timedelta(minutes=30)
    assert w.accepting() is False

    async def main():
        await w.start()
        try:
            assert fake.row("workers", id=w.store.worker_id)["ends_at"] == w.ends_at
            assert await w.poll() is True and answer(fake, cancel["id"])[:2] == ("done", 200)
            assert await w.poll() is False and answer(fake, start["id"])[0] == "pending"
            await w.store.beat(0, w.accepting(), w.ends_at)
            assert fake.row("workers", id=w.store.worker_id)["accepting"] is False
            with pytest.raises(fs.Fail, match="worker_ending: the machine stops soon"):
                fake.rpc(c, "relay", p_job=jid2, p_route="start", p_body={**START, "agent": "opus"},
                         p_sealed=sealed(fake, w, OPUS, jid2, c))
            w.ends_at = None
            assert await w.poll() is True
            assert answer(fake, start["id"]) == ("done", 200, {"status": "queued", "key": server.KEY_LATER})
            await w.client.post(f"/api/jobs/{w.jobs[jid1].local}/cancel")
        finally:
            await w.end()

    anyio.run(main)


def test_delete_removes_the_folder_the_objects_and_the_row(srv, fake, template):
    c, mid, local, J, jid, w = served(srv, fake, template)

    async def main():
        await w.start()
        try:
            assert files(fake, jid) and any(n.startswith(f"{c}/{jid}/") for _, n in fake.objects)
            for i in range(1200):  # objects that no job_files row names: job_objects hands them out 1,000 a round
                fake.object("job-files", f"{c}/{jid}/{'1' * 16}/left_{i}.json", b"{}")
            assert fake.rpc(c, "delete_job", p_job=jid) == "deleting"
            assert await w.poll() is True
        finally:
            await w.end()

    anyio.run(main)
    assert not J.exists() and fake.row("jobs", id=jid) is None and jid not in w.jobs
    assert not [n for _, n in fake.objects if n.startswith(f"{c}/{jid}/")]
    assert fake.rows("requests", job_id=jid) == []


def test_files_over_the_limit_stay_with_a_note(srv, fake, template):
    c = fake.add_user("c@example.org", approved=True)
    mid = machine(fake, c)
    local, J = srv.clone(template)
    jid = own_job(fake, srv, local, c, mid)
    fake.set("settings", {"id": 1}, plan_file_limit=(J / "frames" / "preview.mp4").stat().st_size - 1)
    w = own_worker(srv, fake, c, mid)

    async def main():
        await w.start()
        await w.end()

    anyio.run(main)
    got = files(fake, jid)
    assert "frames/preview.mp4" not in got and "frames/thumbs/0.jpg" in got
    warnings = fake.row("job_views", job_id=jid)["body"]["warnings"]
    assert [x for x in warnings if re.fullmatch(r"frames/preview\.mp4 \([\d.]+ MB\) is over the site's [\d.]+ MB file "
                                                r"limit; it stays on lab-box", x)]
    assert fake.row("job_live", job_id=jid)["state"]["warnings"] == warnings


def test_keys_and_paths_never_leave_the_server(srv, fake, template):
    """The scrub (views, logs, log files) and the key scan (every other file)."""
    c = fake.add_user("c@example.org", approved=True)
    mid = machine(fake, c)
    local, J = srv.clone(template)
    jid = own_job(fake, srv, local, c, mid)
    with open(J / "logs" / "frames.log", "a") as f:
        f.write(f"the key {OPUS} in {J}/video and {srv.cfg.G}/envs and sk-{'z' * 20}\n")
    (J / "pool").mkdir(exist_ok=True)
    (J / "pool" / "lidar.json").write_text(json.dumps({"ok": False, "note": f"held {OPUS} in {J}/pool"}))
    (J / "run").mkdir(exist_ok=True)
    (J / "run" / "posed.glb").write_bytes(b"glTF\x02\x00\x00\x00" + OPUS.encode() + b"\x00" * 20)
    jobs.update(J, warnings=[f"warned about {OPUS} at {J}/frames"])
    w = own_worker(srv, fake, c, mid)
    w.track(jid, local).keys.add(OPUS)

    async def main():
        await w.start()
        await w.end()

    anyio.run(main)
    got = files(fake, jid)
    log = fake.objects[("job-files", got["logs/frames.log"]["object"])]["data"].decode()
    assert "the key [key] in <job>/video and $G/envs and [key]\n" in log
    lidar = json.loads(fake.objects[("job-files", got["pool/lidar.json"]["object"])]["data"])
    assert lidar == {"ok": False, "note": "held [key] in <job>/pool"}  # text is scrubbed
    assert "run/posed.glb" not in got  # binary: the key scan leaves it out
    warnings = fake.row("job_views", job_id=jid)["body"]["warnings"]
    assert "warned about [key] at <job>/frames" in warnings
    assert "run/posed.glb left out: it holds the API key" in warnings
    rows = "".join(x["text"] for x in fake.rows("job_logs", job_id=jid))
    assert "the key [key] in <job>/video and $G/envs and [key]" in rows
    assert leaked(fake, OPUS, str(srv.cfg.G), local, "sk-" + "z" * 20) == []


def test_tunnel_and_worker_at_once_and_the_stop_hook(srv, fake, template):
    """The worker thread and the tunnel's TestClient use one app at once; the lifespan stops the worker before the
    runner, so an active job keeps its status with a note."""
    c, mid, local, J, jid, w = served(srv, fake, template)
    server.on_stop(srv.gate, w.stop_thread)
    with TestClient(srv.gate) as tc:
        tc.get(f"/?token={TOKEN}")
        w.start_thread()
        t0 = time.time()
        while not fake.row("job_views", job_id=jid):
            assert time.time() - t0 < 30
            time.sleep(0.1)
        sent = fake.rpc(c, "relay", p_job=jid, p_route="start", p_body={**START, "agent": "opus", "key_later": True})
        while fake.row("requests", id=sent["id"])["state"] != "done":
            assert tc.get("/api/jobs").status_code == 200 and tc.get(f"/api/jobs/{local}").status_code == 200
            assert time.time() - t0 < 30
            time.sleep(0.02)
        assert answer(fake, sent["id"])[1] == 200
        while srv.job(local)["status"] not in ("running", "review"):
            time.sleep(0.05)
        while fake.row("jobs", id=jid)["status"] not in ("queued", "running"):
            assert time.time() - t0 < 60
            time.sleep(0.05)
    assert not w.thread.is_alive()
    row = fake.row("jobs", id=jid)
    if row["status"] in ("queued", "running"):  # the stop came while the build ran
        assert row["note"] == "lab-box stopped. This job goes on when HARPOR runs there again."
    assert fake.row("workers", id=w.store.worker_id)["stopped_at"] is not None


def test_another_start_stops_this_worker(srv, fake):
    c = fake.add_user("c@example.org", approved=True)
    mid = machine(fake, c)
    w = own_worker(srv, fake, c, mid)

    async def main():
        await w.start()
        fake.rpc(c, "own_worker_start", p_machine=mid, p_public_key="K" * 88, p_ends_at=None, p_version="other")
        with anyio.fail_after(10):
            await w.heartbeat_loop()  # the beat is refused: the loop ends and the worker stops
            await w.request_loop()  # so is the claim
        await w.end()

    anyio.run(main)
    assert w.stopping.is_set()


# ---- local jobs, keys, the site's deletes, approval ----

def test_own_worker_copies_local_jobs_and_relays_with_a_local_key(srv, fake, template, video):
    c = fake.add_user("c@example.org", approved=True)
    old, _ = srv.clone(template, created="2020-01-01T00:00:00")  # made before the first login: it stays local
    reads = []
    w = own_worker(srv, fake, c, local_keys=("opus",), local_key=lambda agent: reads.append(agent) or OPUS,
                   mirror_since=time.strftime("%Y-%m-%dT%H:%M:%S"))

    async def main():
        await w.start()
        try:
            mid = w.store.machine_id
            assert fake.row("machines", id=mid)["name"] == "lab-box" and w.session.machine_id == mid
            assert fake.row("workers", id=w.store.worker_id)["local_keys"] == ["opus"]
            assert reads == []
            r = await w.client.put("/api/upload", params={"name": "box clip.mp4"}, content=video,
                                   headers={"Content-Length": str(len(video))})
            local = r.json()["id"]
            await w.local_jobs()
            jid = jobs.load(srv.cfg.runs / local)["public"]["job"]
            assert jobs.load(srv.cfg.runs / local)["public"] == {"job": jid, "user": c, "machine": mid, "resumes": 0}
            assert "public" not in srv.job(old)
            row = fake.row("jobs", id=jid)
            assert (row["machine_id"], row["input_name"], row["input_bytes"]) == (mid, "box_clip.mp4", len(video))
            await until(lambda: fake.row("jobs", id=jid)["status"] == "ready", w)
            assert fake.row("usage_daily", user_id=c)["own_jobs"] == 1
            seen = watch_live(fake, jid, "rpc", "own_publish")
            # Start: the page sends no key, since the machine holds it; the worker reads it now
            sent = fake.rpc(c, "relay", p_job=jid, p_route="start", p_body={**START, "agent": "opus"})
            assert sent["worker_online"] is True
            fake.calls.clear()
            assert await w.poll() is True
            assert reads == ["opus"]
            assert answer(fake, sent["id"]) == ("done", 200, {"status": "queued", "key": "fake mode: key accepted"})
            calls = [(k, n) for _, k, n in fake.calls]
            pub, ans = calls.index(("rpc", "own_publish")), calls.index(("rpc", "own_answer"))
            assert pub < ans and calls.count(("rpc", "own_publish")) == 1
            assert all(k == "rpc" and n in ("own_claim_request", "own_logs", "own_files", "own_publish", "own_answer")
                       or k == "upload" for k, n in calls)  # the user's session: no table write
            await until(status_is(srv, local, "review"), w)
            await w.sync_once()
            # Build, with the key the runner holds: no second read
            sent = fake.rpc(c, "relay", p_job=jid, p_route="continue", p_body={})
            assert await w.poll() is True and answer(fake, sent["id"])[1] == 200
            assert reads == ["opus"]
            await until(status_is(srv, local, "done", "failed"), w)
            await w.sync_once()
            assert srv.job(local)["status"] == "done"
            got = files(fake, jid)
            assert "results.zip" in got and got["results.zip"]["object"].startswith(f"{c}/{jid}/")
            assert fake.objects[("job-files", got["results.zip"]["object"])]["owner"] == c
            assert fake.row("job_views", job_id=jid)["body"]["status"] == "done"
            assert fake.row("jobs", id=jid)["status"] == "done"
            assert check_files_before_states(seen) >= {"select", "place", "package"}
        finally:
            await w.end()

    anyio.run(main)
    assert fake.row("workers", id=w.store.worker_id)["stopped_at"] is not None
    assert leaked(fake, OPUS, str(srv.cfg.G)) == []
    assert fake.rows("job_secrets") == [] and {who for who, _, _ in fake.calls} == {c}


def test_own_worker_opens_a_sealed_key_and_refuses_none(srv, fake, template):
    c, mid, local, J, jid, w = served(srv, fake, template)

    async def main():
        await w.start()
        try:
            assert fake.row("jobs", id=jid)["status"] == "ready"
            key = sealed(fake, w, OPUS, jid, c)
            assert key["worker_id"] == w.store.worker_id
            sent = fake.rpc(c, "relay", p_job=jid, p_route="start", p_body={**START, "agent": "opus"}, p_sealed=key)
            assert await w.poll() is True
            assert answer(fake, sent["id"]) == ("done", 200, {"status": "queued", "key": "fake mode: key accepted"})
            assert fake.rows("job_secrets") == []
            await w.client.post(f"/api/jobs/{local}/cancel")
            await until(status_is(srv, local, "cancelled"), w)
            jobs.update(J, agent="none")  # set in the local UI
            sent = fake.rpc(c, "relay", p_job=jid, p_route="continue", p_body={})
            assert await w.poll() is True
            assert answer(fake, sent["id"])[1:] == (403, {"detail": "this account may use the agents astra, opus"})
            sent = fake.rpc(c, "relay", p_job=jid, p_route="start", p_body={**START, "agent": "opus"})
            assert await w.poll() is True  # no sealed key, no local key, no key_later: server.py's own 400
            assert answer(fake, sent["id"])[1] == 400
            # a Start whose key was sealed to an earlier start of this machine (a restart while relay() ran)
            lost = fake.put("requests", {"user_id": c, "job_id": jid, "route": "start",
                                         "body": {**START, "agent": "opus"}, "key_ref": str(uuid.uuid4())})
            assert await w.poll() is True
            assert answer(fake, lost["id"])[1:] == (400, {"detail": worker.KEY_GONE}) == (400, {
                "detail": "your machine restarted after you sent the key; paste the key again"})
        finally:
            await w.end()

    anyio.run(main)
    assert leaked(fake, OPUS) == []


def test_own_worker_leaves_files_out_when_the_storage_is_full(srv, fake, template):
    c = fake.add_user("c@example.org", approved=True, max_result_bytes=0)
    mid = machine(fake, c)
    local, _ = srv.clone(template)
    jid = own_job(fake, srv, local, c, mid)
    w = own_worker(srv, fake, c, mid)

    async def main():
        await w.start()
        await w.end()

    anyio.run(main)
    assert files(fake, jid) == {} and not fake.objects
    warnings = fake.row("job_views", job_id=jid)["body"]["warnings"]
    assert "frames/thumbs/0.jpg left out: your storage on the site is full (0.0 of 0.0 MB); the file stays on lab-box" \
        in warnings


def test_own_worker_stops_copying_a_job_deleted_on_the_site_and_deletes_it_later(srv, fake, template):
    c, mid, local, J, jid, w = served(srv, fake, template)

    async def main():
        await w.start()
        try:
            assert files(fake, jid)
            fake.set("jobs", {"id": jid}, deleted_at=fake.now(), status="deleting")
            jobs.update(J, warnings=["changed"])
            await w.publish_safely(jid, force=True)
            assert w.jobs[jid].gone and J.exists()  # the folder stays
            fake.put("requests", {"user_id": c, "job_id": jid, "route": "delete"})
            assert await w.poll() is True
        finally:
            await w.end()

    anyio.run(main)
    assert not J.exists() and fake.row("jobs", id=jid) is None
    assert not [n for _, n in fake.objects if n.startswith(f"{c}/{jid}/")]


def test_own_delete_cancels_a_job_that_still_runs(srv, fake, template):
    """A machine that was offline gets a delete for a job that runs there: it cancels the job, then deletes it."""
    c, mid, local, J, jid, w = served(srv, fake, template)

    async def first():
        await w.start()
        sent = fake.rpc(c, "relay", p_job=jid, p_route="start", p_body={**START, "agent": "opus", "key_later": True,
                                                                         "auto_continue": True})
        assert await w.poll() is True and answer(fake, sent["id"])[1] == 200
        await w.end()  # the machine goes offline while the build runs

    anyio.run(first)
    assert srv.job(local)["status"] in jobs.ACTIVE
    assert fake.row("jobs", id=jid)["note"] == "lab-box stopped. This job goes on when HARPOR runs there again."
    assert fake.rpc(c, "delete_job", p_job=jid) == "deleting"
    w2 = own_worker(srv, fake, c, mid)
    posted = []

    async def second():
        await w2.start()
        post = w2.client.post

        async def record(url, *args, **kw):
            posted.append(url)
            return await post(url, *args, **kw)

        w2.client.post = record
        try:
            active = srv.job(local)["status"] in jobs.ACTIVE
            assert await w2.poll() is True
            assert posted == ([f"/api/jobs/{local}/cancel"] if active else [])
        finally:
            await w2.end()

    anyio.run(second)
    assert not J.exists() and fake.row("jobs", id=jid) is None


def test_own_worker_pauses_without_approval_and_stops_when_replaced(srv, fake, template):
    c = fake.add_user("c@example.org", approved=True)
    mid = machine(fake, c)
    jid = own_job(fake, srv, srv.clone(template)[0], c, mid)
    w = own_worker(srv, fake, c, mid)

    async def main():
        await w.start()
        try:
            fake.set("access", {"user_id": c}, approved=False)
            await w.beat()
            assert w.paused() and not w.stopping.is_set()
            fake.calls.clear()
            await w.sync_once()
            assert fake.calls == []  # the copy pauses
            w.paused_until = 0
            jobs.update(srv.cfg.runs / w.jobs[jid].local, warnings=["changed"])
            await w.publish_safely(jid, force=True)  # refused too: the copy pauses again
            assert w.paused()
            fake.set("access", {"user_id": c}, approved=True)
            w.paused_until = 0
            fake.rpc(c, "own_worker_start", p_machine=mid, p_public_key="K" * 88, p_ends_at=None, p_version="x")
            await w.beat()  # refused: another worker took over
            assert w.stopping.is_set()
        finally:
            await w.end()

    anyio.run(main)


def test_own_worker_waits_for_approval_at_its_start(srv, fake, monkeypatch):
    monkeypatch.setattr(worker, "PAUSE_S", 0.5)
    c = fake.add_user("c@example.org", approved=True)
    mid = machine(fake, c)
    fake.set("access", {"user_id": c}, approved=False)
    w = own_worker(srv, fake, c, mid)

    async def main():
        async with anyio.create_task_group() as tg:
            tg.start_soon(w.main)
            await anyio.sleep(0.3)
            assert w.store.worker_id is None and not fake.rows("workers")
            fake.set("access", {"user_id": c}, approved=True)
            with anyio.fail_after(10):
                while w.store.worker_id is None:
                    await anyio.sleep(0.05)
            w.stopping.set()

    anyio.run(main)
    assert fake.row("workers", id=w.store.worker_id)["stopped_at"] is not None


def test_a_signed_out_session_leaves_the_server_running(srv, fake, capsys):
    c = fake.add_user("c@example.org", approved=True)
    session = fake.session(c, machine_id=machine(fake, c))

    async def refused():
        raise supa.SignedOut("the HARPOR site signed this machine out")

    session.bearer = refused
    w = worker.Worker(srv.gate, srv.cfg, store=supa.Store.of(session, transport=fake.transport, wait=0),
                      session=session)
    anyio.run(w.main)
    assert worker.SIGNED_OUT in capsys.readouterr().out and not fake.rows("workers")


def test_fake_keeps_a_users_machines_apart(fake):
    """Two machines of one user: a personal worker claims, and writes, only its own machine's jobs (the case table
    has no such pair, so this is checked here)."""
    c = fake.add_user("c@example.org", approved=True)
    m1, m2 = machine(fake, c, "one"), machine(fake, c, "two")
    w1 = fake.rpc(c, "own_worker_start", p_machine=m1, p_public_key="A" * 88, p_ends_at=None, p_version="t")
    fake.rpc(c, "own_worker_start", p_machine=m2, p_public_key="B" * 88, p_ends_at=None, p_version="t")
    job2 = fake.put("jobs", {"user_id": c, "machine_id": m2, "input_name": "x.mp4", "input_kind": "video",
                             "input_bytes": 5, "status": "ready"})["id"]
    fake.rpc(c, "relay", p_job=job2, p_route="cancel")
    assert fake.rpc(c, "own_claim_request", p_worker=w1) == []
    for fn, args in (("own_publish", {"p_row": {"status": "done"}}), ("own_logs", {"p_rows": [
            {"stage": "frames", "pos": 1, "text": "x"}]}), ("own_files", {})):
        with pytest.raises(fs.Fail, match="no such job on this machine"):
            fake.rpc(c, fn, p_worker=w1, p_job=job2, **args)
    assert fake.row("jobs", id=job2)["status"] == "ready"
    fake.object("job-files", f"{c}/{job2}/{'0' * 16}/x.json", b"{}", age=7200)
    assert fake.rpc(c, "own_orphans", p_worker=w1) == []  # another machine's job: its own worker cleans it


def test_fake_refuses_the_secret_key_and_anon(fake):
    """The fake answers a secret key with 401 (no part of HARPOR sends one), and gives anon nothing."""
    c = fake.add_user("c@example.org", approved=True)

    async def call(key, token, path, method="GET", body=None):
        async with httpx.AsyncClient(transport=fake.transport) as client:
            return await client.request(method, fs.URL + path, json=body,
                                        headers={"apikey": key, "Authorization": f"Bearer {token}"})

    token = fake.access_token(c)
    assert anyio.run(call, fs.SECRET, fs.SECRET, "/rest/v1/jobs").status_code == 401
    assert anyio.run(call, fs.PUBLISHABLE, fs.SECRET, "/rest/v1/jobs").status_code == 401
    assert anyio.run(call, fs.PUBLISHABLE, token, "/rest/v1/jobs?select=*").json() == []
    for path, method in (("/rest/v1/jobs", "GET"), ("/rest/v1/rpc/storage_used", "POST")):
        r = anyio.run(call, fs.PUBLISHABLE, fs.PUBLISHABLE, path, method, {})
        assert r.status_code == 401 and "permission denied" in r.json()["message"]
    r = anyio.run(call, fs.PUBLISHABLE, token, "/rest/v1/jobs", "POST", {})
    assert r.status_code == 403 and "permission denied" in r.json()["message"]  # users write no table


def test_fake_sweep_follows_private_sweep(fake):
    """fake.sweep() applies private.sweep's rules (pg_cron runs it on Supabase); tests/pgcheck checks the SQL."""
    c = fake.add_user("c@example.org", approved=True)
    mid = machine(fake, c)
    wid = fake.rpc(c, "own_worker_start", p_machine=mid, p_public_key="K" * 88, p_ends_at=None, p_version="t")

    def job(status, days, **kw):
        jid = fake.put("jobs", {"user_id": c, "machine_id": mid, "input_name": "x.mp4", "input_kind": "video",
                                "input_bytes": 5, "status": status, "stopped_at": fake.now() - timedelta(days=days),
                                **kw})["id"]
        fake.put("job_logs", {"job_id": jid, "user_id": c, "stage": "frames", "pos": 1, "text": "x"})
        return jid

    waits, deleting = job("ready", 0), job("deleting", 0, deleted_at=fake.now())
    stopped, recent, running = job("done", 15), job("failed", 13), job("running", 30)
    pending = fake.put("requests", {"user_id": c, "job_id": waits, "route": "cancel"})["id"]
    big = fake.put("requests", {"user_id": c, "job_id": stopped, "route": "start", "state": "done",
                                "done_at": fake.now(), "response": {"detail": "x" * 2000}})["id"]
    fake.put("job_secrets", {"ref": str(uuid.uuid4()), "job_id": waits, "user_id": c, "worker_id": wid,
                             "agent": "opus", "epk": "B" * 88, "iv": "C" * 16, "ct": "D" * 40})
    fake.bump_rate(c)
    assert fake.sweep() == 0 and fake.rows("job_secrets")
    assert {r["job_id"] for r in fake.rows("job_logs")} == {waits, deleting, recent, running}  # 14 days after a stop
    fake.advance(2 * 3600)
    assert fake.sweep() == 0 and fake.rows("job_secrets") == [] and fake.row("requests", id=big)["response"] is None
    fake.advance(8 * 86400)
    assert fake.sweep() == 1
    r = fake.row("requests", id=pending)
    assert (r["state"], r["status"], r["response"]) == (
        "expired", 410, {"detail": "the request waited more than 7 days for its machine; send it again"})
    assert fake.row("jobs", id=deleting) is None and fake.row("requests", id=big) is None and fake.rows("rate") == []
    assert {r["job_id"] for r in fake.rows("job_logs")} == {waits, running}  # waits stopped 8 days ago


def test_worker_arguments():
    class App:
        class state:
            runner = None

    cfg = steps.Config(G=Path("/G"), runs=Path("/runs"), fake=True)
    with pytest.raises(ValueError, match="local keys are for astra and opus"):
        worker.Worker(App, cfg, ("none",), store=object())
    w = worker.Worker(App, cfg, ["opus", "astra"], store=object())
    assert w.local_keys == ("opus", "astra") and w.place == "your machine"


def test_fake_supabase_passes_the_shared_cases():
    """tests/supabase_cases.json, which the SQL passes too (tests/test_supabase_sql.py)."""
    failed = []
    for case in fs.load_cases():
        ok, what = fs.run_case(case)
        if not ok:
            failed.append((case["id"], what))
    assert failed == []
    assert len(fs.load_cases()) >= 55


# ---- the life cycle: start, resume, stop and housekeeping ----

def test_a_restart_resumes_interrupted_jobs_at_most_twice(tmp_path, fake, template):
    """The stop (SIGTERM) keeps the row's status with a note; the next start goes on from the first stage not
    done, without a key, at most RESUMES times (job.json public.resumes). An analysis goes on as an analysis, and
    a job whose cancel was answered after its row last changed stays cancelled."""
    c = fake.add_user("c@example.org", approved=True)
    mid = machine(fake, c)
    servers = [Server(tmp_path)]
    local, J = servers[0].clone(template)
    jid = own_job(fake, servers[0], local, c, mid)
    an_local, A = servers[0].clone(template)
    analysis = own_job(fake, servers[0], an_local, c, mid)
    held_local, H = servers[0].clone(template)
    held = own_job(fake, servers[0], held_local, c, mid)

    def restart():
        """The runner's stop when the machine's server stops, and a new server on the same runs folder."""
        servers[-1].runner.shutdown(timeout=10)
        servers.append(Server(tmp_path))
        return servers[-1]

    try:
        w = own_worker(servers[0], fake, c, mid)

        async def first():
            await w.start()
            fake.rpc(c, "relay", p_job=jid, p_route="start", p_body={**START, "agent": "opus", "key_later": True,
                                                                     "auto_continue": True})
            assert await w.poll() is True
            await until(lambda: servers[0].job(local)["status"] == "running", w)
            await w.end()

        anyio.run(first)
        row = fake.row("jobs", id=jid)
        assert row["status"] in jobs.ACTIVE and row["note"] == worker.ENDED.format(place="lab-box")
        # a crash during an analysis (Runner._recover failed it), and a cancel answered after the row last changed
        fake.set("jobs", {"id": analysis}, status="analyzing")
        jobs.update_stage(A, "frames", status="failed")
        jobs.update(A, status="failed")
        fake.set("jobs", {"id": held}, status="running")
        jobs.update(H, status="cancelled")
        fake.put("requests", {"user_id": c, "job_id": held, "route": "cancel", "state": "done", "status": 200,
                              "done_at": fake.now() + timedelta(seconds=5)})
        two = restart()
        assert two.job(local)["status"] in ("cancelled", "failed")
        w2 = own_worker(two, fake, c, mid)

        async def second():
            await w2.start()
            try:
                row = fake.row("jobs", id=jid)
                assert row["note"] is None and row["status"] in jobs.ACTIVE
                assert two.job(local)["status"] in ("queued", "running") and two.job(local)["public"]["resumes"] == 1
                assert two.job(an_local)["status"] == "analyzing" and two.job(an_local)["public"]["resumes"] == 1
                assert two.job(held_local)["status"] == "cancelled" and two.job(held_local)["public"]["resumes"] == 0
                assert fake.row("jobs", id=held)["status"] == "cancelled"
                await until(status_is(two, an_local, "ready", "failed"), w2)
                assert two.job(an_local)["status"] == "ready" and fake.row("jobs", id=analysis)["status"] == "ready"
            finally:
                await w2.end()

        anyio.run(second)
        three = restart()
        jobs.update(J, status="cancelled", public={**jobs.load(J)["public"], "resumes": worker.RESUMES})  # a third stop
        fake.set("jobs", {"id": jid}, status="running")
        w3 = own_worker(three, fake, c, mid)

        async def third():
            await w3.start()
            await w3.end()

        anyio.run(third)
        row = fake.row("jobs", id=jid)
        assert (row["status"], row["note"]) == ("cancelled", worker.TWICE.format(place="lab-box"))
        assert three.job(local)["status"] == "cancelled" and three.job(local)["public"]["resumes"] == worker.RESUMES
    finally:
        for s in servers:
            s.runner.shutdown(timeout=10)


def test_a_personal_worker_start_keeps_local_data_and_resumes(srv, fake, template):
    """A copied job whose row went is marked gone and stays on the machine, a row whose folder is missing fails,
    another machine's folder is left alone, and an interrupted job goes on, counted in job.json public.resumes."""
    c = fake.add_user("c@example.org", approved=True)
    mid = machine(fake, c)
    gone_local, _ = srv.clone(template)
    gone = own_job(fake, srv, gone_local, c, mid, status="ready")
    assert fake.rpc(c, "delete_job", p_job=gone) == "deleting"  # the machine is offline, and stays away a week
    fake.advance(8 * 86400)
    assert fake.sweep() == 1  # the delete request expired, and the row went
    assert fake.row("jobs", id=gone) is None
    missing = fake.put("jobs", {"user_id": c, "machine_id": mid, "input_name": "cup.mp4", "input_kind": "video",
                                "input_bytes": 5, "status": "ready"})["id"]
    other_local, _ = srv.clone(template)
    other = {"job": str(uuid.uuid4()), "user": c, "machine": str(uuid.uuid4()), "resumes": 0}
    jobs.update(srv.cfg.runs / other_local, public=other)
    resumed_local, R = srv.clone(template)
    resumed = own_job(fake, srv, resumed_local, c, mid, status="analyzing")
    jobs.update_stage(R, "frames", status="failed")
    jobs.update(R, status="failed")
    w = own_worker(srv, fake, c, mid)

    async def main():
        await w.start()
        try:
            assert set(w.jobs) == {resumed}
            assert srv.job(resumed_local)["public"]["resumes"] == 1 and srv.job(resumed_local)["status"] == "analyzing"
            assert fake.row("jobs", id=resumed)["status"] == "analyzing"
            await until(status_is(srv, resumed_local, "ready", "failed"), w)
        finally:
            await w.end()

    anyio.run(main)
    assert srv.job(gone_local)["public"] == {"job": gone, "user": c, "machine": mid, "resumes": 0, "gone": True}
    row = fake.row("jobs", id=missing)
    assert (row["status"], row["note"]) == ("failed", "the job folder is gone from lab-box")
    assert srv.job(other_local)["public"] == other
    assert fake.row("jobs", id=resumed)["status"] == "ready"


def test_housekeeping_applies_retention_and_deletes_orphans(srv, fake, template):
    """Hourly: a job that stopped over REVIEW_DAYS ago loses its review files and logs on the site (with a note in its
    view), and its results only after RESULT_DAYS; the orphan objects of the user's folder go (own_orphans), but a
    name that the Storage API cannot take stays. Every minute: a key held in review over KEY_HOURS goes."""
    c = fake.add_user("c@example.org", approved=True)
    mid = machine(fake, c)
    made = {}
    for name in ("old", "older", "review"):
        local, J = srv.clone(template)
        (J / "run").mkdir(exist_ok=True)
        (J / "run" / "placement.json").write_text('{"ok": true}')  # a result file
        made[name] = (own_job(fake, srv, local, c, mid), local, J)
    jid = {name: x[0] for name, x in made.items()}
    folder = {name: x[2] for name, x in made.items()}
    elsewhere = fake.put("jobs", {"user_id": c, "machine_id": machine(fake, c, "box-2"), "input_name": "cup.mp4",
                                  "input_kind": "video", "input_bytes": 5, "status": "done"})["id"]
    gone = [f"{c}/{uuid.uuid4()}/{'2' * 16}/c.json", f"{c}/{uuid.uuid4()}/{'2' * 16}/a..b.json"]  # their rows went
    kept = [f"{c}/{elsewhere}/{'3' * 16}/x.json", f"{c}/{uuid.uuid4()}/{'4' * 16}/new.json"]
    for n in gone + kept[:1]:
        fake.object("job-files", n, b"{}", age=7200)
    fake.object("job-files", kept[1], b"{}")  # younger than an hour: its own_files may still come
    w = own_worker(srv, fake, c, mid)

    async def main():
        await w.start()  # the start deletes orphans too
        try:
            assert ("job-files", gone[0]) not in fake.objects and ("job-files", gone[1]) in fake.objects
            assert all(("job-files", n) in fake.objects for n in kept)
            stale = f"{c}/{jid['review']}/{'5' * 16}/run/placement.json"  # this machine's, but no row names it
            fake.object("job-files", stale, b"{}", age=7200)
            # a key held for a job that waits in review for over KEY_HOURS
            jobs.update(folder["review"], status="review")
            stopped_days_ago(folder["review"], 25 / 24)
            with srv.runner._cv:
                srv.runner._take_key(made["review"][1], OPUS, "opus")
            await w.housekeeping()
            assert not srv.runner.has_key(made["review"][1]) and srv.job(made["review"][1])["key_ok"] is False
            assert ("job-files", stale) in fake.objects  # orphans go hourly
            # retention, counted from when a job stopped: 15 days (review files and logs), 91 days (results too)
            for name, days in (("old", 15), ("older", 91)):
                stopped_days_ago(folder[name], days)
                assert {"run/placement.json", "logs/frames.log", "frames/thumbs/0.jpg"} <= set(files(fake, jid[name]))
            await w.housekeeping(hourly=True)
            assert ("job-files", stale) not in fake.objects
            for name in ("old", "older"):
                view = fake.row("job_views", job_id=jid[name])["body"]
                assert worker.REVIEW_EXPIRED in view["warnings"] and set(view["files"]) <= set(files(fake, jid[name]))
                assert folder[name].exists()  # the job folder stays on the machine
            assert set(files(fake, jid["old"])) == {"run/placement.json"} and files(fake, jid["older"]) == {}
            assert worker.RESULTS_EXPIRED in fake.row("job_views", job_id=jid["older"])["body"]["warnings"]
            assert {"run/placement.json", "frames/thumbs/0.jpg"} <= set(files(fake, jid["review"]))  # a day ago
            for name in w.trash:  # the objects of dropped files go 10 minutes later: open pages keep their images
                w.trash[name] = 0
            await w.empty_trash()
            assert not [n for _, n in fake.objects if n.startswith(f"{c}/{jid['older']}/")]
            calls = len(fake.calls)
            await w.housekeeping(hourly=True)  # nothing more to publish
            assert [n for _, k, n in fake.calls[calls:] if k == "rpc"] == ["own_orphans"]
        finally:
            await w.end()

    anyio.run(main)


def test_retention_reaches_a_job_whose_folder_is_gone(srv, fake, template):
    """A job deleted in the machine's own UI keeps its row (failed) and its files on the site only until their
    retention ends, counted from the row's stopped_at: review files and logs after REVIEW_DAYS, results after
    RESULT_DAYS. The same holds when the folder goes while the server runs."""
    c = fake.add_user("c@example.org", approved=True)
    mid = machine(fake, c)
    made = {}
    for name in ("before", "while", "fresh"):
        local, J = srv.clone(template)
        (J / "run").mkdir(exist_ok=True)
        (J / "run" / "placement.json").write_text('{"ok": true}')  # a result file
        made[name] = (own_job(fake, srv, local, c, mid), J)
    jid = {name: x[0] for name, x in made.items()}
    w = own_worker(srv, fake, c, mid)

    async def first():
        await w.start()
        await w.end()

    anyio.run(first)
    for name in made:
        assert {"run/placement.json", "logs/frames.log", "frames/thumbs/0.jpg"} <= set(files(fake, jid[name]))
    shutil.rmtree(made["before"][1])  # deleted in the local UI while the server was down
    shutil.rmtree(made["fresh"][1])
    fake.set("jobs", {"id": jid["before"]}, stopped_at=worker.now() - timedelta(days=15))
    w2 = own_worker(srv, fake, c, mid)

    async def second():
        await w2.start()
        try:
            assert w2.folderless == {jid["before"], jid["fresh"]}
            assert fake.row("jobs", id=jid["before"])["note"] == "the job folder is gone from lab-box"
            fake.set("jobs", {"id": jid["while"]}, stopped_at=worker.now() - timedelta(days=91))
            shutil.rmtree(made["while"][1])  # deleted in the local UI while the server runs
            await w2.housekeeping(hourly=True)
            assert set(files(fake, jid["before"])) == {"run/placement.json"}  # 15 days: the result stays
            assert files(fake, jid["while"]) == {}  # 91 days: everything goes
            assert {"run/placement.json", "logs/frames.log"} <= set(files(fake, jid["fresh"]))  # stopped just now
            assert w2.folderless == {jid["before"], jid["fresh"]}
            for name in w2.trash:
                w2.trash[name] = 0
            await w2.empty_trash()
            assert [n for _, n in fake.objects if n.startswith(f"{c}/{jid['before']}/")] == [
                files(fake, jid["before"])["run/placement.json"]["object"]]
            assert not [n for _, n in fake.objects if n.startswith(f"{c}/{jid['while']}/")]
            fake.set("jobs", {"id": jid["before"]}, stopped_at=worker.now() - timedelta(days=91))
            await w2.housekeeping(hourly=True)
            assert files(fake, jid["before"]) == {} and w2.folderless == {jid["fresh"]}
        finally:
            await w2.end()

    anyio.run(second)


def test_the_claim_loop_goes_on_after_an_answer_it_cannot_read(srv, fake, monkeypatch):
    """A claim whose answer Python cannot decode (a body nested deeper than its JSON parser goes) is logged; the
    loop, and the requests it is handling, go on."""
    c = fake.add_user("c@example.org", approved=True)
    w = own_worker(srv, fake, c, machine(fake, c))
    seen = []

    async def claim(kinds):
        seen.append(kinds)
        if len(seen) == 1:
            raise RecursionError("maximum recursion depth exceeded while decoding a JSON array")
        if len(seen) >= 3:
            w.stopping.set()
        await anyio.sleep(0.01)
        return None

    async def nap(seconds):
        await anyio.sleep(0.01)

    async def main():
        await w.start()
        try:
            monkeypatch.setattr(w.store, "claim", claim)
            monkeypatch.setattr(w, "nap", nap)
            with anyio.fail_after(10):
                await w.request_loop()
        finally:
            await w.end()

    anyio.run(main)
    assert len(seen) >= 3


def test_a_refused_upload_leaves_the_file_out_and_the_rest_goes_on(srv, fake, template):
    """Storage refuses one upload for good (here a content type that job-files does not take): the file stays on the
    machine with a note, the other files, the view and the row go up, and the file is not tried again until it
    changes."""
    c = fake.add_user("c@example.org", approved=True)
    mid = machine(fake, c)
    local, J = srv.clone(template)
    jid = own_job(fake, srv, local, c, mid)
    fake.buckets["job-files"]["mime"] = ("image/jpeg", "application/json")  # no logs, no preview video
    tries = []

    def handle(request):
        if request.method == "POST" and request.url.path.endswith("/logs/frames.log"):
            tries.append(request.url.path)
        return fake.handle(request)

    session = fake.session(c, machine_id=mid)
    w = worker.Worker(srv.gate, srv.cfg, store=supa.Store.of(session, transport=httpx.MockTransport(handle), wait=0),
                      session=session)

    async def main():
        await w.start()
        try:
            await w.publish(jid, force=True)
        finally:
            await w.end()

    anyio.run(main)
    warnings = fake.row("job_views", job_id=jid)["body"]["warnings"]
    assert any(x.startswith("logs/frames.log left out: the site's storage refused it (mime type text/plain")
               for x in warnings), warnings
    assert "frames/thumbs/0.jpg" in files(fake, jid) and "logs/frames.log" not in files(fake, jid)
    assert fake.row("jobs", id=jid)["status"] == "ready" and len(tries) == 1


def test_a_local_job_name_with_two_dots_gets_the_sites_name(srv, fake, template):
    c = fake.add_user("c@example.org", approved=True)
    mid = machine(fake, c)
    local, J = srv.clone(template)
    jobs.update(J, video={**jobs.load(J)["video"], "name": "clip..final.mp4"})
    w = own_worker(srv, fake, c, mid)

    async def main():
        await w.start()
        try:
            await w.local_jobs()
        finally:
            await w.end()

    anyio.run(main)
    row = fake.row("jobs", user_id=c)
    assert row["input_name"] == "clip.final.mp4" and jobs.load(J)["public"]["job"] == row["id"]


def test_the_heartbeat_reads_the_end_of_the_gpu_job_again(srv, fake, monkeypatch):
    """A machine inside a Slurm job: every SQUEUE_S the heartbeat reads the job's end again (the limit can change);
    within settings.near_end_minutes of it the workers row says accepting = false, and the page sends no key."""
    c = fake.add_user("c@example.org", approved=True)
    w = own_worker(srv, fake, c, machine(fake, c))
    assert w.ends_at is None and w.accepting() is True  # outside Slurm
    end = (worker.now() + timedelta(minutes=20)).replace(microsecond=0)
    monkeypatch.setattr(worker, "slurm_end", lambda: end)
    monkeypatch.setattr(worker, "SQUEUE_S", 0)

    async def main():
        await w.start()
        try:
            assert fake.row("workers", id=w.store.worker_id)["accepting"] is True
            async with anyio.create_task_group() as tg:
                tg.start_soon(w.heartbeat_loop)
                with anyio.fail_after(10):
                    while fake.row("workers", id=w.store.worker_id)["accepting"] is not False:
                        await anyio.sleep(0.05)
                w.stopping.set()
        finally:
            await w.end()

    anyio.run(main)
    row = fake.row("workers", id=w.store.worker_id)
    assert w.ends_at == end and row["ends_at"] == end and row["accepting"] is False


# ---- the server with a worker: main() ----

def test_main_starts_the_worker_it_is_asked_for(tmp_path, monkeypatch, capsys):
    import uvicorn

    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("GATOR_ROOT", str(tmp_path / "G"))
    monkeypatch.setattr(jobs, "default_key_dir", lambda: tmp_path / "keys")
    info = home / ".config" / "gator" / "webapp.json"
    ran, started = [], []  # (host, webapp.json) while the server runs; the workers it started
    monkeypatch.setattr(uvicorn.Server, "run", lambda self, sockets: ran.append(
        (sockets[0].getsockname()[0], json.loads(info.read_text()))))
    monkeypatch.setattr(worker.Worker, "start_thread", lambda self: started.append(self))
    fake = fs.FakeSupabase()
    c = fake.add_user("c@example.org", approved=True)
    from gator.webapp import session as sessions  # its Session, with the fake's session in its place

    monkeypatch.setattr(sessions, "Session", lambda: fake.session(c, machine_id=machine(fake, c)))
    args = ["--fake", "--runs", str(tmp_path / "runs")]
    for wrong, why in ((["--local-keys", "opus"], "--local-keys needs --worker"),
                       (["--worker", "--local-keys", "none"], "--local-keys takes astra, opus or astra,opus")):
        with pytest.raises(SystemExit) as e:
            server.main(args + wrong)
        assert e.value.code == 2 and why in capsys.readouterr().err
    assert ran == []

    server.main(args)  # the tunnel as before
    assert ran[-1][0] == "0.0.0.0" and ran[-1][1]["worker"] is False and started == []
    assert "on the Mac: ssh -S ~/.ssh/cm-ice -O forward" in capsys.readouterr().out
    server.main(args + ["--worker"])  # the local UI stays on the machine
    w = started[-1]
    assert ran[-1][0] == "127.0.0.1" and ran[-1][1]["worker"] is True and w.local_keys == ()
    assert w.inner.state.stop_hooks == [w.stop_thread] and isinstance(w.store, supa.Store)  # it stops before the runner
    out = capsys.readouterr().out
    assert "the personal worker of lab-box runs in this server" in out
    assert re.search(r"from another computer: ssh -L (\d+):127\.0\.0\.1:\1 <this machine>", out)
    assert "on the Mac" not in out and "cm-ice" not in out and "login-ice" not in out  # nothing of the owner's cluster
    server.main(args + ["--worker", "--host", "0.0.0.0"])
    assert ran[-1][0] == "0.0.0.0"

    def held():
        raise SystemExit(f"another HARPOR server holds {home}/.config/gator/harpor_session.json (pid 7)")

    monkeypatch.setattr(sessions, "Session", held)  # a session that cannot start leaves the server in tunnel mode
    n = len(started)
    server.main(args + ["--worker"])
    out = capsys.readouterr().out
    assert ran[-1][1]["worker"] is False and len(started) == n
    assert "the personal worker does not start: another HARPOR server holds" in out and worker.SIGNED_OUT in out
    # --local-keys: the server refuses to start unless each key file is 0600 in a 0700 folder (it is not read)
    monkeypatch.setattr(sessions, "Session", lambda: fake.session(c, machine_id=machine(fake, c, "box-2")))
    conf = home / ".config" / "gator"
    conf.chmod(0o700)
    with pytest.raises(SystemExit, match=r"anthropic\.env with the Anthropic key \(ANTHROPIC_API_KEY=\.\.\.\)"):
        server.main(args + ["--worker", "--local-keys", "opus"])
    (conf / "anthropic.env").write_text("ANTHROPIC_API_KEY=sk-ant-test\n")
    (conf / "anthropic.env").chmod(0o644)
    with pytest.raises(SystemExit, match="must be a file of this user with mode 0600"):
        server.main(args + ["--worker", "--local-keys", "opus"])
    (conf / "anthropic.env").chmod(0o600)
    server.main(args + ["--worker", "--local-keys", "opus"])
    assert started[-1].local_keys == ("opus",) and ran[-1][1]["worker"] is True
    for w in started:
        anyio.run(w.store.close)


class FakeHTTP:
    """The fake Supabase over real HTTP on 127.0.0.1, for a server process of its own."""

    def __init__(self, fake):
        import http.server

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def answer(self):
                n = int(self.headers.get("Content-Length") or 0)
                req = httpx.Request(self.command, fs.URL + self.path, headers=list(self.headers.items()),
                                    content=self.rfile.read(n) if n else b"")
                r = fake.handle(req)
                data = r.read()
                self.send_response(r.status_code)
                if "content-type" in r.headers:
                    self.send_header("content-type", r.headers["content-type"])
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            do_GET = do_POST = do_DELETE = answer

            def log_message(self, *a):
                pass

        self.http = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.http.server_address[1]}"
        threading.Thread(target=self.http.serve_forever, daemon=True).start()

    def close(self):
        self.http.shutdown()
        self.http.server_close()
