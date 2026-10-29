import fnmatch
import io
import json
import re
import secrets
import zipfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("av")
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from harpor.webapp import jobs, relay, server, steps  # noqa: E402
from test_webapp_server import run_app_js, wait, write_video  # noqa: E402

TOKEN = "tok-" + secrets.token_hex(16)
UUID = "6f1c0d2e-0000-4000-8000-000000000001"
LOCAL = "20261024-120000-abcdef"
KEY = "sk-ant-test-" + secrets.token_hex(16)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """A fake job in review (agent none, views selected), its TestClient and config."""
    d = tmp_path_factory.mktemp("relay")
    cfg = steps.Config(G=d / "G", runs=d / "runs", fake=True)
    with TestClient(server.make_app(cfg, TOKEN, key_dir=d / "keys")) as c:
        c.get(f"/?token={TOKEN}")
        r = c.put("/api/upload", params={"name": "cup.mp4"}, content=write_video(d / "src.mp4").read_bytes())
        job_id = r.json()["id"]
        wait(c, job_id, "ready")
        r = c.post(f"/api/jobs/{job_id}/start", json={"mode": "auto", "prompt": "cup", "n_views": 4, "agent": "none"})
        assert r.status_code == 200, r.text
        wait(c, job_id, "review")
        yield c, cfg, job_id


# ---- the route table ----

def test_build_makes_each_route_from_the_table():
    body = {"mode": "auto", "prompt": "cup", "api_key": "k"}
    assert relay.build("start", LOCAL, body) == (
        "POST", f"/api/jobs/{LOCAL}/start", json.dumps(body).encode(), {"Content-Type": "application/json"})
    for route in ("topic", "overrides", "parts", "continue"):
        method, path, content, _ = relay.build(route, LOCAL, {"x": 1})
        assert (method, path, json.loads(content)) == ("POST", f"/api/jobs/{LOCAL}/{route}", {"x": 1})
    assert relay.build("cancel", LOCAL, {"ignored": 1}) == ("POST", f"/api/jobs/{LOCAL}/cancel", None, {})
    assert relay.build("delete", LOCAL) == ("DELETE", f"/api/jobs/{LOCAL}", None, {})
    assert relay.build("calibration_delete", LOCAL) == ("DELETE", f"/api/jobs/{LOCAL}/calibration", None, {})
    method, path, content, headers = relay.build("calibration_put", LOCAL, query={"name": "cal &x=1/../a.yaml"},
                                                 text="T: [1, 2]\nµ: 1\n")
    assert method == "PUT" and path == f"/api/jobs/{LOCAL}/calibration?name=cal+%26x%3D1%2F..%2Fa.yaml"
    assert content == "T: [1, 2]\nµ: 1\n".encode() and headers == {"Content-Type": "application/octet-stream"}
    # The LiDAR topic that the calibration is for goes in the query string too, encoded; "" names none.
    _, path, _, _ = relay.build("calibration_put", LOCAL, query={"name": "c.yaml", "lidar_topic": "/a b&name=x.yaml"},
                                text="x")
    assert path == f"/api/jobs/{LOCAL}/calibration?name=c.yaml&lidar_topic=%2Fa+b%26name%3Dx.yaml"
    for none in ({"name": "c.yaml"}, {"name": "c.yaml", "lidar_topic": ""}, {"name": "c.yaml", "lidar_topic": None}):
        _, path, _, _ = relay.build("calibration_put", LOCAL, query=none, text="x")
        assert path == f"/api/jobs/{LOCAL}/calibration?name=c.yaml"
    assert set(relay.ROUTES) == {"topic", "calibration_put", "calibration_delete", "start", "overrides", "parts",
                                 "continue", "cancel", "delete"}
    assert set(relay.NO_BUILD) | set(relay.KEY_ROUTES) == set(relay.ROUTES)
    assert not set(relay.NO_BUILD) & set(relay.KEY_ROUTES)


def test_build_refuses_what_is_not_in_the_table():
    with pytest.raises(relay.Refuse) as e:
        relay.build("file", LOCAL)
    assert (e.value.status, e.value.detail) == (400, "unknown request")
    with pytest.raises(relay.Refuse, match="name the file"):
        relay.build("calibration_put", LOCAL, query={"name": 5}, text="x")
    with pytest.raises(relay.Refuse, match="the LiDAR topic is not text"):
        relay.build("calibration_put", LOCAL, query={"name": "c.yaml", "lidar_topic": ["/lidar/points"]}, text="x")
    for bad in ("../x", f"{LOCAL}/../y", "", None, LOCAL + "\n"):
        with pytest.raises(ValueError):
            relay.build("cancel", bad)


def test_the_routes_cover_the_relayed_routes_of_the_migrations():
    migrations = Path(__file__).resolve().parents[1] / "supabase" / "migrations"
    sql = (migrations / "20261101000100_harpor_tables.sql").read_text()
    routes = re.search(r"route in \(([^)]*)\)", sql, re.S)[1]
    assert set(re.findall(r"'(\w+)'", routes)) == set(relay.ROUTES)


# ---- the agent that server.py runs ----

BODIES = [{}, {"agent": "astra"}, {"agent": "opus"}, {"agent": "none"}, {"agent": True}, {"agent": False},
          {"agent": None}, {"agent": "bogus"}, {"agent": 5}, {"agent": ["opus"]}]
JOBS = [{"agent": "astra"}, {"agent": "opus"}, {"agent": "none"}, {"agent": True}, {"agent": False}, {}]


def test_effective_agent_follows_server_agent_of():
    def server_says(fn):
        try:
            return fn()
        except (server.Bad, ValueError):
            return None

    for b in BODIES:
        assert relay.effective_agent("start", b, {}) == server_says(lambda: server.agent_of(b))
        for job in JOBS:
            for route in ("overrides", "continue"):
                assert relay.effective_agent(route, b, job) == server_says(lambda: server.agent_of(b, job)), (route, b)
            assert relay.effective_agent("parts", b, job) == steps.agent(job)
            for route in relay.NO_BUILD:
                assert relay.effective_agent(route, b, job) is None
    assert relay.effective_agent("start", {}, {"agent": "opus"}) == "astra"  # a start that names none runs astra
    assert relay.effective_agent("continue", {}, {"agent": "none"}) == "none"
    assert relay.effective_agent("parts", {"agent": "opus"}, {"agent": "none"}) == "none"
    assert relay.effective_agent("parts", {}, {"agent": "bogus"}) is None


def test_effective_agent_is_the_agent_that_server_py_stores(built):
    """For every start body in the table, the agent that the server writes into job.json is the one the
    worker checked."""
    c, cfg, _ = built
    r = c.put("/api/upload", params={"name": "two.mp4"}, content=write_video(cfg.runs.parent / "two.mp4").read_bytes())
    job_id = r.json()["id"]
    wait(c, job_id, "ready")
    seen = []
    for b in BODIES:
        agent = relay.effective_agent("start", b, jobs.load(cfg.runs / job_id))
        key = {"astra": "sk-test-" + "a" * 20, "opus": "sk-ant-" + "b" * 20}.get(agent)
        r = c.post(f"/api/jobs/{job_id}/start", json={"mode": "auto", "prompt": "cup", **b,
                                                      **({"api_key": key} if key else {})})
        if agent is None:
            assert r.status_code == 400 and r.json()["detail"].startswith("agent is"), (b, r.text)
            continue
        assert r.status_code == 200, (b, r.text)
        assert steps.agent(jobs.load(cfg.runs / job_id)) == agent, b
        seen.append(agent)
        c.post(f"/api/jobs/{job_id}/cancel")
        wait(c, job_id, "cancelled")
    assert set(seen) == {"astra", "opus", "none"}


def test_agent_refusal_allows_only_the_accounts_public_agents():
    assert relay.agent_refusal("opus", {"agents": ["astra", "opus"]}) is None
    assert relay.agent_refusal("astra", {"agents": ["astra", "opus"]}) is None
    for agent, access in (("none", {"agents": ["astra", "opus"]}), ("astra", {"agents": ["opus"]}),
                          ("none", {"agents": ["none", "opus"]}), ("opus", None), ("opus", {})):
        e = relay.agent_refusal(agent, access)
        assert isinstance(e, relay.Refuse) and e.status == 403, (agent, access)
    assert relay.agent_refusal("astra", {"agents": ["opus"]}).detail == "this account may use the agents opus"
    assert relay.agent_refusal("none", {"agents": ["none", "opus"]}).detail == "this account may use the agents opus"


# ---- the rewrite ----

def ctx(tmp_path, keys=(KEY,), created="2026-10-24T10:00:00+00:00"):
    G = tmp_path / "G"
    return relay.Ctx(uuid=UUID, local=LOCAL, G=str(G), runs=str(G / "webapp_runs"), keys=frozenset(keys),
                     created=created)


def test_scrub_takes_out_keys_paths_and_the_local_id(tmp_path):
    c = ctx(tmp_path, keys=("plain-key-without-prefix",))
    G, J = tmp_path / "G", tmp_path / "G" / "webapp_runs" / LOCAL
    text = (f"key plain-key-without-prefix and sk-{'x' * 16} and sk-ant-api03-{'y' * 30} but sk-short; "
            f"job {LOCAL} in {J}/frames/meta.json, runs {G}/webapp_runs/other, env {G}/envs/web/bin/python")
    out = relay.scrub(text, c)
    assert out == (f"key [key] and [key] and [key] but sk-short; job {UUID} in <job>/frames/meta.json, runs "
                   "<runs>/other, env $G/envs/web/bin/python")
    assert relay.scrub(5, c) == 5 and relay.scrub(None, c) is None
    link = tmp_path / "link"
    link.symlink_to(G)
    c2 = relay.Ctx(uuid=UUID, local=LOCAL, G=str(link), runs=str(link / "webapp_runs"))
    assert relay.scrub(f"{G}/x {link}/y", c2) == "$G/x $G/y"  # the path as given and as resolved
    assert relay.scrub("/", relay.Ctx(uuid=UUID, G="/")) == "/"  # never the root alone


def test_rewrite_goes_through_keys_values_and_lists(tmp_path):
    c = ctx(tmp_path)
    got = relay.rewrite({LOCAL: [KEY, {"n": 1, "ok": True, "p": f"{tmp_path}/G/x"}], "none": None}, c)
    assert got == {UUID: ["[key]", {"n": 1, "ok": True, "p": "$G/x"}], "none": None}


def test_public_view_keeps_only_what_the_site_holds(tmp_path):
    c = ctx(tmp_path)
    view = {"id": LOCAL, "created": "2026-10-24T12:00:00", "status": "done", "warnings": ["w"],
            "video": {"name": "cup.mp4", "path": f"{tmp_path}/G/webapp_runs/{LOCAL}/video/cup.mp4", "bytes": 5},
            "public": {"job": UUID, "user": "u"}, "files": ["run/posed.glb", "results.zip"],
            "sizes": {"run/posed.glb": 3, "results.zip": 70_000_000}, "error": f"the key {KEY} failed"}
    got = relay.public_view(view, c, manifest={"run/posed.glb": {}}, notes=["results.zip is too large", "w"])
    assert got == {"id": UUID, "created": "2026-10-24T10:00:00+00:00", "status": "done",
                   "warnings": ["w", "results.zip is too large"], "video": {"name": "cup.mp4", "bytes": 5},
                   "files": ["run/posed.glb"], "sizes": {"run/posed.glb": 3}, "error": "the key [key] failed"}
    assert "public" in view and "path" in view["video"]  # the view itself is left as it was


def test_snapshots_and_answers(tmp_path):
    c = ctx(tmp_path)
    meta = {"thumbs": [0, 1, 2, 3], "candidates": [2], "pool": {"names": {"2": f"f{2:05d}"}}, "note": LOCAL}
    assert relay.public_snapshot("frames", 200, meta, c) == (200, {**meta, "note": UUID})
    assert relay.public_snapshot("frames", 200, meta, c, thumbs={0, 2})[1]["thumbs"] == [0, 2]
    assert relay.public_snapshot("selection", 404, {"detail": "the views are not selected yet"}, c) == (
        404, {"detail": "the views are not selected yet"})
    status, body = relay.public_snapshot("frames", 200, {"times": [0.1] * 300_000}, c)
    assert status == 413 and "too large" in body["detail"]
    assert relay.public_answer(200, {"status": "queued", "key": f"checked {KEY}"}, c) == (
        200, {"status": "queued", "key": "checked [key]"})
    assert relay.public_answer(400, {"detail": "x" * 70_000}, c) == (
        400, {"detail": "the answer is too large for the site"})


def test_state_of_a_view_is_server_state(built, tmp_path):
    c, cfg, job_id = built
    J = cfg.runs / job_id
    cx = relay.Ctx(uuid=UUID, local=job_id, G=str(cfg.G), runs=str(cfg.runs))
    for change in ({}, {"warnings": ["a warning"], "error": f"failed in {J}", "key_ok": True}):
        job = jobs.load(J)
        jobs.update(J, **change)
        try:
            view = c.get(f"/api/jobs/{job_id}").json()
            want = relay.rewrite(server.state(J, jobs.load(J)), cx)
            assert relay.state_of(relay.public_view(view, cx)) == want
            assert relay.state_of(view) == server.state(J, jobs.load(J))
        finally:
            jobs.save(J, job)


def test_coarse_and_the_row(tmp_path):
    st = {"status": "running", "stages": [{"name": "masks", "status": "running", "progress": 0.2}]}
    assert relay.coarse(st) == relay.coarse({**st, "stages": [{**st["stages"][0], "progress": 0.9}]})
    assert relay.coarse(st) != relay.coarse({**st, "stages": [{**st["stages"][0], "status": "done"}]})
    assert relay.coarse(None) is None
    progress = [{"name": "frames", "ended": 1_790_000_000.5}, {"name": "extract", "ended": 1_790_000_100.0},
                {"name": "masks", "ended": None}]
    assert relay.row_fields({"status": "review", "prompt": "cup", "progress": progress}, 3) == {
        "status": "review", "prompt": "cup", "queue_pos": None, "stopped_at": "2026-09-21T14:15:00+00:00"}
    assert relay.row_fields({"status": "queued", "prompt": None, "progress": progress}, 2) == {
        "status": "queued", "prompt": None, "queue_pos": 2}
    assert relay.row_fields({"status": "uploaded", "progress": []}) == {"status": "uploaded", "prompt": None,
                                                                         "queue_pos": None}


# ---- the retry stage: app.js retryFrom in Python ----

def retry_cases():
    names = ["topics", "frames", *jobs.BUILD]
    stages = lambda **o: [{"name": n, "status": o.get(n, "pending")} for n in names]  # noqa: E731
    done = {n: "done" for n in names}
    at_list = {**done, "lidar": "skipped", "parts": "failed", "agent": "pending", "align": "pending",
               "vomp": "pending", "mjcf": "pending", "usd": "pending", "urdf": "pending", "package": "pending"}
    out = [(stages(frames="failed"), False, False), (stages(), False, False),
           (stages(frames="done", extract="done", masks="failed"), False, False), (stages(frames="done"), False, False),
           (stages(**done), False, False), (stages(**{**done, "agent": "skipped", "usd": "failed"}), False, False),
           (stages(frames="done", extract="done", masks="done", prep="done", da3="done", lidar="failed"), False, False)]
    for confirmed in (False, True):
        for mode in ("vlm_edit", "vlm"):
            out.append(({"articulated": True, "parts_mode": mode, "stages": stages(**at_list)}, confirmed, False))
    for stale in (False, True):
        out.append((stages(**{**done, "sam3d": "failed", "place": "pending"}), False, stale))
        out.append((stages(frames="done", extract="done", masks="done", prep="failed"), False, stale))
    # A done job whose scale is stale runs again from lidar (Run lidar and later stages again); before any build, nothing.
    out += [(stages(**done), False, True), (stages(**{**done, "turntable": "skipped"}), False, True),
            (stages(frames="done"), False, True)]
    return [({"stages": s} if isinstance(s, list) else s, confirmed, stale) for s, confirmed, stale in out]


def test_first_not_done_is_app_js_retry_from():
    cases = retry_cases()
    py = [relay.first_not_done(st, confirmed, stale) for st, confirmed, stale in cases]
    assert py[:7] == ["frames", "frames", "masks", None, None, "usd", "lidar"]
    assert py[7:11] == ["propose", "parts", "parts", "parts"] and py[11:15] == ["sam3d", "prep", "lidar", "prep"]
    assert py[15:] == ["lidar", "lidar", None]
    assert run_app_js({"/api/jobs": []}, f"""
        const out = [];
        for (const [st, confirmed, stale] of {json.dumps(cases)}) {{
          V = {{ job: {{ parts: {{ confirmed }} }}, scaleStale: stale }};
          out.push(retryFrom(st));
        }}
        console.log(JSON.stringify(out));
        """) == py


def test_app_js_build_stages_are_jobs_build():
    text = (server.STATIC / "app.js").read_text()
    names = re.findall(r'"(\w+)"', re.search(r"const BUILD = \[([^\]]*)\]", text)[1])
    assert names == jobs.BUILD


# ---- logs ----

def test_log_rows_drop_progress_lines_and_keep_the_offset():
    events = [server.sse("log", {"stage": "masks", "text": "one\nPROGRESS 0.25\ntwo\nPROGRESS 0.5\r\n"}, "masks:120"),
              server.sse("state", {"status": "running"}), server.sse("log", {"stage": "masks", "text": "PROGRESS 1\n"},
                                                                     "masks:140")]
    assert relay.log_rows(events, ("masks", 140, 1790000000.5)) == [
        {"stage": "masks", "run_started": 1790000000.5, "pos": 120, "text": "one\ntwo\n"}]
    assert relay.log_rows([server.sse("end", {"status": "done"})], None) == []
    text = "\n".join(f"line {i}" for i in range(20000)) + "\n"
    parts = relay.split_text(text, 1000)
    assert "".join(parts) == text and all(len(p) <= 1000 and p.endswith("\n") for p in parts)
    assert relay.split_text("x" * 2500, 1000) == ["x" * 1000, "x" * 1000, "x" * 500] and relay.split_text("") == []


def test_tick_rows_of_a_real_log(built):
    _, cfg, job_id = built
    J = cfg.runs / job_id
    events, cursor, _, _ = server.tick(J, None, None)
    rows = relay.log_rows(events, cursor)
    assert rows and rows[0]["stage"] == cursor[0] and rows[0]["pos"] == cursor[1]
    assert rows[0]["run_started"] == jobs.load(J)["stages"][cursor[0]]["started"]
    assert not any(line.startswith("PROGRESS") for r in rows for line in r["text"].splitlines())


# ---- files ----

def alternative_sample(alt):
    """A path that one ALLOW alternative matches."""
    return (alt.replace(r"\d{5,9}", "00012").replace(r"\d{1,3}", "3").replace(r"\d+", "12")
            .replace(r"[\w-]+", "f00012").replace(r"\w+", "masks").replace("\\.", "."))


def test_globs_cover_every_allow_path():
    """A new ALLOW path fails here until the worker mirrors it (relay.GLOBS) and knows its stage (OWNERS)."""
    alternatives = server.ALLOW.pattern.split("|")
    assert sorted(relay.GLOBS) == sorted(alternatives)
    for alt, glob in relay.GLOBS.items():
        sample = alternative_sample(alt)
        assert re.fullmatch(alt, sample) and server.ALLOW.fullmatch(sample), alt
        assert fnmatch.fnmatchcase(sample, glob), (alt, glob)
        assert relay.owner(sample) in steps.NAMES, sample
    assert relay.owner("logs/masks.log") == "masks" and relay.owner("agent/s/x") is None
    assert {relay.role_of(p) for p in server.OUTPUTS} == {"result"}
    assert relay.role_of("logs/frames.log") == "log" and relay.role_of("frames/thumbs/0.jpg") == "review"
    # The LiDAR points of the run's views: select copies them, and they leave the site with the review images.
    assert relay.owner("run/lidar/0.glb") == "select" and relay.role_of("run/lidar/11.glb") == "review"


def job_folder(tmp_path, running=()):
    """A job folder with one file per ALLOW path, a few that ALLOW does not serve, and job.json. Its LiDAR scale
    failed, so the pool's points are served too (server.pool_clouds)."""
    J = tmp_path / LOCAL
    for alt in relay.GLOBS:
        p = J / alternative_sample(alt)
        p.parent.mkdir(parents=True, exist_ok=True)
        if alt != r"refined\.glb":  # not a file of the folder: resolve maps it to the agent's GLB
            p.write_bytes(b"data of " + p.name.encode())
    (J / "pool" / "lidar.json").write_text(json.dumps({"ok": False}))
    (J / "agent" / "s").mkdir(parents=True)
    (J / "agent" / "s" / "final.glb").write_bytes(b"glb")
    (J / "agent" / "result.json").write_text(json.dumps({"source": "s", "verdict": "improved", "glb": "final.glb"}))
    (J / "frames" / "thumbs" / "x.jpg").write_bytes(b"not a frame")
    (J / "frames" / "meta.json").write_text(json.dumps({"thumbs": [12]}))
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    (J / "physics" / "summary.json").unlink()
    (J / "physics" / "summary.json").symlink_to(outside)
    stages = {n: {"status": "running" if n in running else "done"} for n in steps.NAMES}
    (J / "job.json").write_text(json.dumps({"id": LOCAL, "stages": stages}))
    return J, jobs.load(J)


def test_mirror_plan_takes_every_allow_path_that_resolve_accepts(tmp_path):
    J, job = job_folder(tmp_path)
    plan = relay.mirror_plan(J, job)
    got = {f.rel: f for f in plan}
    want = {alternative_sample(a) for a in relay.GLOBS} - {"physics/summary.json"}  # a link out of the folder
    assert set(got) == want
    assert got["refined.glb"].path == (J / "agent" / "s" / "final.glb").resolve()  # steps.agent_glb
    assert got["results.zip"].role == "result" and got["logs/masks.log"].role == "log"
    assert got["frames/thumbs/12.jpg"].role == "review"
    assert [(f.rel, f.role) for f in plan if f.rel.startswith("run/lidar/")] == [(alternative_sample(r"run/lidar/\d{1,3}\.glb"),
                                                                                  "review")]
    assert got["turntable/turntable.mp4"].role == "result"  # the 360 video: a result, kept 90 days


def test_mirror_plan_holds_back_the_files_of_a_running_stage(tmp_path):
    J, job = job_folder(tmp_path, running=("frames", "place", "masks", "turntable"))
    rels = {f.rel for f in relay.mirror_plan(J, job)}
    assert not {"frames/thumbs/12.jpg", "frames/preview.mp4", "run/posed.glb", "run/overlay_12.png",
                "run/placement.json", "logs/masks.log", "scene/sam3_masks/check/f00012.jpg",
                "turntable/turntable.mp4"} & rels
    assert {"frames/topics.json", "run/input/images/3.png", "selection.json", "results.zip"} <= rels


def test_lean_thumbs_and_the_lean_plan(tmp_path):
    meta = {"thumbs": list(range(0, 1200, 2)), "candidates": [5, 333, 1001, 1198]}
    keep = relay.lean_thumbs(meta)
    assert 150 <= len(keep) <= 155 and 0 in keep and {1198} <= keep and 5 not in keep and 333 not in keep
    assert keep <= set(meta["thumbs"])
    assert relay.lean_thumbs({"thumbs": [1, 2, 3], "candidates": [2]}) == {1, 2, 3}
    assert relay.lean_thumbs(None) == set()
    J, job = job_folder(tmp_path)
    for i in range(400):
        (J / "frames" / "thumbs" / f"{i}.jpg").write_bytes(b"t")
    (J / "frames" / "meta.json").write_text(json.dumps({"thumbs": list(range(400)), "candidates": [7]}))
    lean = {f.rel for f in relay.mirror_plan(J, job, lean=True) if f.rel.startswith("frames/thumbs/")}
    full = {f.rel for f in relay.mirror_plan(J, job) if f.rel.startswith("frames/thumbs/")}
    assert len(full) == 400 and len(lean) == len(relay.lean_thumbs(json.loads((J / "frames/meta.json").read_text())))
    assert "frames/thumbs/7.jpg" in lean


def image(size, fmt):
    buf = io.BytesIO()
    a = (np.random.default_rng(1).random((size[1], size[0], 3)) * 255).astype(np.uint8)
    Image.fromarray(a).save(buf, fmt)
    return buf.getvalue()


def test_lean_review_images_go_up_as_jpeg(tmp_path):
    c = ctx(tmp_path)
    J = tmp_path / "J"
    files = {"run/input/images/3.png": image((1600, 900), "PNG"), "run/overlay_12.png": image((300, 200), "PNG"),
             "scene/sam3_cup/check/f00012.jpg": image((2000, 1000), "JPEG"),
             "lidar/check/f00012.jpg": image((640, 480), "JPEG"), "run/posed.glb": b"glTF..."}
    for rel, data in files.items():
        (J / rel).parent.mkdir(parents=True, exist_ok=True)
        (J / rel).write_bytes(data)
    out = {rel: relay.file_bytes(relay.Mirror(rel, J / rel, relay.role_of(rel)), c, lean=True) for rel in files}
    for rel in ("run/input/images/3.png", "run/overlay_12.png", "scene/sam3_cup/check/f00012.jpg"):
        data, ctype = out[rel]
        with Image.open(io.BytesIO(data)) as im:
            assert (im.format, ctype) == ("JPEG", "image/jpeg") and max(im.size) <= 1280, rel
    with Image.open(io.BytesIO(out["run/input/images/3.png"][0])) as im:
        assert im.size == (1280, 720)
    assert out["lidar/check/f00012.jpg"] == (files["lidar/check/f00012.jpg"], "image/jpeg")  # small enough: as it is
    assert out["run/posed.glb"] == (b"glTF...", "model/gltf-binary")
    full = relay.file_bytes(relay.Mirror("run/overlay_12.png", J / "run/overlay_12.png", "review"), c)
    assert full == (files["run/overlay_12.png"], "image/png")  # not lean: the file as it is


def test_a_large_preview_is_left_out_in_lean_mode(tmp_path):
    c = ctx(tmp_path)
    p = tmp_path / "frames" / "preview.mp4"
    p.parent.mkdir()
    with open(p, "wb") as f:
        f.truncate(relay.LEAN_PREVIEW + 1)  # sparse
    f = relay.Mirror("frames/preview.mp4", p, "review")
    with pytest.raises(relay.Skip, match="preview video .* over 30 MB"):
        relay.file_bytes(f, c, lean=True)
    p.write_bytes(b"small")
    assert relay.file_bytes(f, c, lean=True) == (b"small", "video/mp4")


def test_stage_logs_are_scrubbed_and_cut_to_their_end(tmp_path):
    c = ctx(tmp_path)
    J = tmp_path / "G" / "webapp_runs" / LOCAL
    log = J / "logs" / "agent.log"
    log.parent.mkdir(parents=True)
    log.write_text(f"== agent\nkey {KEY} at {J}/agent/s\n")
    data, ctype = relay.file_bytes(relay.Mirror("logs/agent.log", log, "log"), c)
    assert data == b"== agent\nkey [key] at <job>/agent/s\n" and ctype == "text/plain; charset=utf-8"
    with open(log, "w") as f:
        f.write("old line\n" * 700_000 + "the end\n")
    data, _ = relay.file_bytes(relay.Mirror("logs/agent.log", log, "log"), c)
    assert len(data) <= relay.LOG_FILE_CAP + 64 and data.startswith(b"[the start of this log is left out]\nold line\n")
    assert data.endswith(b"the end\n")


def test_text_files_and_the_results_zip_are_scrubbed(tmp_path):
    c = ctx(tmp_path)
    J = tmp_path / "G" / "webapp_runs" / LOCAL
    (J / "physics").mkdir(parents=True)
    summary = {"glb": f"{J}/physics/aligned.glb", "note": f"key {KEY}"}
    (J / "physics" / "summary.json").write_text(json.dumps(summary))
    data, ctype = relay.file_bytes(relay.Mirror("physics/summary.json", J / "physics/summary.json", "result"), c)
    assert json.loads(data) == {"glb": "<job>/physics/aligned.glb", "note": "key [key]"}
    assert ctype == "application/json"
    (J / "physics" / "urdf").mkdir()
    (J / "physics" / "urdf" / "model.urdf").write_text(f'<robot name="cup"><!-- {J} --></robot>')
    data, ctype = relay.file_bytes(relay.Mirror("physics/urdf/model.urdf", J / "physics/urdf/model.urdf", "result"), c)
    assert data == b'<robot name="cup"><!-- <job> --></robot>'
    assert ctype == "text/plain; charset=utf-8" and server.TYPES[".urdf"] == "application/xml"  # the tunnel keeps XML
    assert relay.scrub_text(b"\xff\xfe not utf-8 " + KEY.encode(), c) == b"\xff\xfe not utf-8 " + KEY.encode()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("summary.json", json.dumps(summary))
        z.writestr("mjcf/scene.xml", f"<mujoco><!-- {J} --></mujoco>")
        z.writestr(zipfile.ZipInfo("posed.glb", (2026, 10, 24, 12, 0, 0)), b"glTF binary")
    (J / "results.zip").write_bytes(buf.getvalue())
    data, ctype = relay.file_bytes(relay.Mirror("results.zip", J / "results.zip", "result"), c)
    assert ctype == "application/zip"
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        assert z.namelist() == ["summary.json", "mjcf/scene.xml", "posed.glb"]
        assert json.loads(z.read("summary.json")) == {"glb": "<job>/physics/aligned.glb", "note": "key [key]"}
        assert z.read("mjcf/scene.xml") == b"<mujoco><!-- <job> --></mujoco>"
        assert z.read("posed.glb") == b"glTF binary" and z.getinfo("posed.glb").date_time == (2026, 10, 24, 12, 0, 0)
        assert z.getinfo("summary.json").compress_type == zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(J / "results.zip", "a") as z:
        z.writestr("refined.glb", b"glTF " + KEY.encode())
    with pytest.raises(relay.Skip, match="results.zip left out: it holds the API key"):
        relay.file_bytes(relay.Mirror("results.zip", J / "results.zip", "result"), c)
    assert relay.scrub_zip(b"not a zip", c) == b"not a zip"


def test_the_key_scan_finds_keys_across_chunk_edges(tmp_path):
    k = "sk-ant-" + "k" * 30
    for at in (0, 5, 1000 - 10, 1000 - len(k), 1000, 2999 - len(k) + 1):
        data = bytearray(b"." * 3000)
        data[at:at + len(k)] = k.encode()
        for chunk in (1000, 7, len(k), len(k) - 1):
            assert relay.holds_key(bytes(data), [k], chunk=chunk), (at, chunk)
    cut = (b"." * 995 + k.encode())[:1000]  # a key cut short at the end is no key
    assert not relay.holds_key(cut, [k], chunk=100)
    assert relay.holds_key(b"x" * 10 + b"second", ["first-key", "second"], chunk=4)
    assert not relay.holds_key(b"anything", []) and not relay.holds_key(b"anything", ["", None])
    p = tmp_path / "big.bin"
    with open(p, "wb") as f:
        f.truncate((3 << 20) - 3)
        f.write(k.encode())
    assert relay.holds_key(p, [k]) and relay.holds_key(str(p), [k])
    assert not relay.holds_key(p, ["sk-ant-other-key-xxxxxxxx"])


def test_the_shape_scan_finds_a_key_the_worker_no_longer_holds(tmp_path):
    """With shape, the scan finds anything shaped like a key in binary data (after a restart the worker holds no key),
    also across a chunk edge, but not a word that only ends in "sk-", such as a part name."""
    k = b"sk-ant-api03-" + b"Q" * 40
    glb = b"glTF\x02\x00\x00\x00" + bytes(range(256)) * 8
    for at in (0, 100, 1000 - 5, len(glb)):
        data = glb[:at] + k + glb[at:]
        assert relay.holds_key(data, [], chunk=1000, shape=True) and not relay.holds_key(data, []), at
    assert relay.holds_key(glb + b'"sk-proj-' + b"a" * 30 + b'"', [], shape=True)
    for word in (b"desk-lamp_with_a_long_shade_name", b"mask-0001_component_part_left", b"sk-short"):
        assert not relay.holds_key(glb + word + glb, [], chunk=64, shape=True), word
    c = relay.Ctx(uuid="u", keys=frozenset())
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("refined.glb", glb + k)
    with pytest.raises(relay.Skip, match="results.zip left out: it holds the API key"):
        relay.scrub_zip(buf.getvalue(), c)


def test_relay_does_no_network_io_and_reads_no_secret():
    """relay.py imports no client and reads no key file or environment: it works on the job's files."""
    text = Path(relay.__file__).read_text()
    assert not re.search(r"^\s*(import|from) (httpx|socket|subprocess|urllib\.request|harpor\.webapp\.supa)", text, re.M)
    for word in ("supabase.env", "openai.env", "anthropic.env", "os.environ", "getenv"):
        assert word not in text, word
