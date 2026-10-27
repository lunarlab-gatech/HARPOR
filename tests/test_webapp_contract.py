"""The contract between the tunnel and the public site: a request sent the way the public page sends it,
through the fake Supabase and a worker (RelayClient), gets the answer that the same request gets from the server
directly (TestClient), after the rewrite. Hostile bodies too: either both give the same answer, or relay() refuses
the body before any worker sees it and the server refuses it as well.

Each request goes to twin jobs, two copies of one job folder on one server: the TestClient asks about one, the
personal worker serves the other, with the user's session (supa.Store)."""

import secrets
import time

import anyio
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("av")
pytest.importorskip("cryptography")
from fastapi.testclient import TestClient  # noqa: E402

import fake_supabase as fs  # noqa: E402
from gator.webapp import jobs, relay, server, steps  # noqa: E402
from test_webapp_server import HOSTILE_CONTINUE, hostile_overrides, hostile_start  # noqa: E402
from test_webapp_server import calibration_yaml, scene_bag  # noqa: E402
from test_webapp_worker import TOKEN, Server, machine, own_job, own_worker  # noqa: E402
from test_webapp_worker import template, video  # noqa: E402, F401 (fixtures)

OPUS = "sk-ant-test-" + secrets.token_hex(16)
BASE = {"mode": "auto", "prompt": "box", "n_views": 4, "agent": "opus", "key_later": True}


class RelayClient:
    """A request as the public page sends it: relay() as the job's user, with an API key sealed to the job's
    worker (never in the body), then one claim by the worker. send() returns the requests row's answer as
    (status, JSON), or (None, message) when relay() refused the request itself."""

    def __init__(self, fake, w, user, jid, folder):
        self.fake, self.w, self.user, self.jid, self.folder = fake, w, user, jid, folder

    async def send(self, route, body=None, query=None, text=None):
        body, sealed = (dict(body) if isinstance(body, dict) else body), None
        if route in relay.KEY_ROUTES and isinstance(body, dict) and isinstance(body.get("api_key"), str):
            agent = relay.effective_agent(route, body, jobs.load(self.folder)) or "opus"  # the page's agent
            sealed = fs.sealed(self.fake.row("workers", id=self.w.store.worker_id), body.pop("api_key"),
                                   self.jid, self.user, agent)
        try:
            sent = self.fake.rpc(self.user, "relay", p_job=self.jid, p_route=route, p_query=query or {},
                                 p_body={} if body is None else body, p_text=text, p_sealed=sealed)
        except fs.Fail as e:
            return None, e.message
        assert await self.w.poll() is True
        r = self.fake.row("requests", id=sent["id"])
        assert r["state"] == "done"
        return r["status"], r["response"]


class Tunnel:
    """The same request to the server directly, on the tunnel's twin, as app.js sends it; the answer rewritten
    for the worker's job (its uuid, and $G, <runs> and [key] for what the machine must not show)."""

    def __init__(self, srv, local, jid):
        self.c, self.local = TestClient(srv.gate), local
        self.c.get(f"/?token={TOKEN}")
        self.ctx = relay.Ctx(uuid=jid, local=local, G=str(srv.cfg.G), runs=str(srv.runner.runs),
                             keys=frozenset({OPUS}))

    def send(self, route, body=None, query=None, text=None):
        base = f"/api/jobs/{self.local}"
        if route == "calibration_put":
            r = self.c.put(f"{base}/calibration", params=query, content=(text or "").encode())
        elif route == "calibration_delete":
            r = self.c.delete(f"{base}/calibration")
        elif route == "cancel":
            r = self.c.post(f"{base}/cancel")
        else:
            r = self.c.post(f"{base}/{route}", json=body)
        return relay.public_answer(r.status_code, r.json(), self.ctx)


def test_relayed_requests_get_the_tunnel_answers(tmp_path, template):
    fake = fs.FakeSupabase()
    srv = Server(tmp_path)
    tlocal, TJ = srv.clone(template)
    wlocal, WJ = srv.clone(template)
    user = fake.add_user("c@example.org", approved=True, max_requests_per_hour=1000)
    mid = machine(fake, user)
    jid, w = own_job(fake, srv, wlocal, user, mid), own_worker(srv, fake, user, mid)
    tunnel, page = Tunnel(srv, tlocal, jid), RelayClient(fake, w, user, jid, WJ)
    refused = []  # (route, body, relay()'s reason) of the requests that never reached a worker

    async def both(route, body=None, query=None, text=None):
        want = await anyio.to_thread.run_sync(lambda: tunnel.send(route, body, query, text))
        got = await page.send(route, body, query, text)
        if got[0] is None:  # relay() refused it; the server refuses it too
            assert want[0] >= 400, (route, str(body)[:200], want, got)
            refused.append((route, body, got[1]))
        else:
            assert got == want, (route, str(body)[:200])
        return got

    async def settled(*statuses):
        """Wait until both twins stopped in one of statuses, and the site's row says so (the sync loop's step)."""
        with anyio.fail_after(120):
            while True:
                await w.sync_once()
                if srv.job(tlocal)["status"] in statuses and fake.row("jobs", id=jid)["status"] in statuses:
                    return
                await anyio.sleep(0.2)

    async def main():
        await w.start()
        try:
            # The server's hostile bodies, with opus in place of the agent "none". The one intended difference: the
            # tunnel runs the agent "none", and the public site refuses it.
            n = len(tunnel.c.get(f"/api/jobs/{tlocal}/frames").json()["times"])
            for wrong in hostile_start(n, OPUS):
                await both("start", {**BASE, **wrong})
            for route, body in (("start", {**BASE, "agent": "none"}), ("start", {**BASE, "agent": False}),
                                ("continue", {"agent": "none"}), ("overrides", {"agent": False})):
                assert await page.send(route, body) == (None, "the public site runs the agents astra and opus")
            assert srv.job(tlocal)["status"] == srv.job(wlocal)["status"] == "ready"  # nothing started
            await both("topic", {"topic": "/cam"})  # a video has no topic
            await both("calibration_put", query={"name": "cam.yaml"}, text="K: [1, 0, 0]\n")
            await both("parts", {"parts": [{"name": "body"}]})  # the job does not wait for a part list
            assert await both("cancel") == (200, {"status": "ready"})
            # Start with the key: sealed in the page, in the body for the tunnel
            got = await both("start", {**BASE, "key_later": False, "api_key": OPUS, "auto_continue": False})
            assert got == (200, {"status": "queued", "key": "fake mode: key accepted"})
            await settled("review", "failed")
            assert srv.job(tlocal)["status"] == srv.job(wlocal)["status"] == "review"
            name = sorted(tunnel.c.get(f"/api/jobs/{tlocal}/selection").json()["sam3"])[0]
            assert name in page.w.runner.job_dir(wlocal).joinpath("selection.json").read_text()
            for wrong in hostile_overrides(name):
                await both("overrides", wrong)
            for wrong in HOSTILE_CONTINUE:
                await both("continue", wrong)
            await both("parts", {"parts": [{"name": "body"}] * 10000})
            assert srv.job(tlocal)["status"] == srv.job(wlocal)["status"] == "review"
            # Build: each server holds its twin's key, so no key travels
            assert await both("continue", {}) == (200, {"status": "queued", "from": "sam3d"})
            await settled("done", "failed")
            assert srv.job(tlocal)["status"] == srv.job(wlocal)["status"] == "done"
            assert await both("continue", {"from": "package", "key_later": True}) == (
                200, {"status": "queued", "from": "package"})
            await settled("done", "failed")
        finally:
            await w.end()

    try:
        anyio.run(main)
    finally:
        srv.runner.shutdown(timeout=10)
    reasons = {why for _, _, why in refused}
    assert reasons <= {"the request body is not a JSON object of at most 128 KB", "send the API key sealed, not in the "
                       "body", "the sealed key is malformed", "the public site runs the agents astra and opus",
                       "this job's input is not a rosbag", "the job does not wait for its part list"}, reasons
    assert len(refused) < 20  # most hostile bodies reach the worker, and server.py answers them
    assert OPUS not in str(fake.rows("requests"))


def test_a_relayed_calibration_keeps_its_lidar_topic(tmp_path):
    """A calibration upload from the public page names the LiDAR topic that it is for, as app.js sends it to the
    tunnel (?lidar_topic=). The worker passes the topic on: both twins give the same answer and keep the same topic
    in job.json, and a topic that is not one of the bag's LiDAR topics gets the server's refusal."""
    from test_webapp_lidar import POINTS, T_CAM_LIDAR

    d = tmp_path / "template"
    cfg = steps.Config(G=d / "G", runs=d / "runs", fake=True)
    with TestClient(server.make_app(cfg, TOKEN, key_dir=d / "keys")) as c:  # a bag that the frames stage analyzed
        c.get(f"/?token={TOKEN}")
        bag = scene_bag(d / "no_tf.bag", tf=None)
        template = cfg.runs / c.put("/api/upload", params={"name": "no_tf.bag"}, content=bag.read_bytes()).json()["id"]
        t0 = time.time()
        while c.get(f"/api/jobs/{template.name}").json()["status"] != "ready":
            assert time.time() - t0 < 60
            time.sleep(0.1)
    fake = fs.FakeSupabase()
    srv = Server(tmp_path)
    tlocal, _ = srv.clone(template)
    wlocal, WJ = srv.clone(template)
    user = fake.add_user("c@example.org", approved=True)
    mid = machine(fake, user)
    jid = own_job(fake, srv, wlocal, user, mid, name="no_tf.bag", kind="rosbag")
    w = own_worker(srv, fake, user, mid)
    tunnel, page = Tunnel(srv, tlocal, jid), RelayClient(fake, w, user, jid, WJ)
    yaml = calibration_yaml(T_CAM_LIDAR)

    async def both(query):
        want = await anyio.to_thread.run_sync(lambda: tunnel.send("calibration_put", query=query, text=yaml))
        got = await page.send("calibration_put", query=query, text=yaml)
        assert got == want, query
        return got

    def topics():
        return [(srv.job(local)["calibration"] or {}).get("lidar_topic") for local in (tlocal, wlocal)]

    async def main():
        await w.start()
        try:
            assert await both({"name": "calib.yaml", "lidar_topic": "/nope"}) == (
                400, {"detail": "/nope is not a LiDAR topic of the bag; its LiDAR topics are /lidar/points"})
            assert topics() == [None, None]  # nothing saved
            status, summary = await both({"name": "calib.yaml", "lidar_topic": POINTS})
            assert status == 200 and summary["lidar_topic"] == POINTS and topics() == [POINTS, POINTS]
            assert (await both({"name": "calib.yaml"}))[1]["lidar_topic"] is None and topics() == [None, None]
        finally:
            await w.end()

    try:
        anyio.run(main)
    finally:
        srv.runner.shutdown(timeout=10)
    assert [r["query"] for r in fake.rows("requests")] == [{"name": "calib.yaml", "lidar_topic": "/nope"},
                                                          {"name": "calib.yaml", "lidar_topic": POINTS},
                                                          {"name": "calib.yaml"}]
