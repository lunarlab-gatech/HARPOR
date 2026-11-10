import hashlib
import io
import json
import math
import os
import secrets
import shutil
import stat
import subprocess
import sys
import textwrap
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("av")
from fastapi.testclient import TestClient  # noqa: E402

from harpor.webapp import jobs, server, steps  # noqa: E402

TOKEN = "tok-" + secrets.token_hex(16)
KEY = "sk-test-" + secrets.token_hex(16)
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def client(tmp_path, fake_stages):
    cfg = steps.Config(G=tmp_path / "G", runs=tmp_path / "runs")
    with TestClient(server.make_app(cfg, TOKEN, key_dir=tmp_path / "keys")) as c:
        yield c, cfg


def login(c):
    r = c.get(f"/?token={TOKEN}", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/"
    return r


def write_video(path, n=64, size=(192, 128)):
    import av

    rng = np.random.default_rng(0)
    bg = (rng.random((size[1] // 4, size[0] // 4, 3)) * 180 + 30).astype(np.uint8).repeat(4, 0).repeat(4, 1)
    with av.open(str(path), "w", format="mp4") as c:
        s = c.add_stream("libx264", rate=24, options={"crf": "12"})
        s.width, s.height, s.pix_fmt = size[0], size[1], "yuv420p"
        for i in range(n):
            a = bg.copy()
            a[40:80, 10 + 2 * i:50 + 2 * i] = 240
            for p in s.encode(av.VideoFrame.from_ndarray(a, format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    return path


def write_bag(path, topics, n=48, size=(192, 128)):
    """A ROS1 bag with one CompressedImage (JPEG) topic per name in topics: n frames at 10 Hz of a textured
    background with a white square moving right, with header stamps."""
    from PIL import Image
    from rosbags.rosbag1 import Writer
    from rosbags.typesys import Stores, get_typestore

    ts = get_typestore(Stores.ROS1_NOETIC)
    T = ts.types
    rng = np.random.default_rng(0)
    bg = (rng.random((size[1] // 4, size[0] // 4, 3)) * 180 + 30).astype(np.uint8).repeat(4, 0).repeat(4, 1)
    with Writer(path) as w:
        conns = [w.add_connection(t, "sensor_msgs/msg/CompressedImage", typestore=ts) for t in topics]
        for i in range(n):
            a = bg.copy()
            a[40:80, 10 + 2 * i:50 + 2 * i] = 240
            buf = io.BytesIO()
            Image.fromarray(a).save(buf, "JPEG", quality=90)
            ns = 1_749_907_120_000_000_000 + i * 100_000_000
            header = T["std_msgs/msg/Header"](seq=i, stamp=T["builtin_interfaces/msg/Time"](sec=ns // 10**9, nanosec=ns % 10**9),
                                              frame_id="camera")
            msg = T["sensor_msgs/msg/CompressedImage"](header=header, format="jpeg", data=np.frombuffer(buf.getvalue(), np.uint8))
            for conn in conns:
                w.write(conn, ns + 50_000_000, ts.serialize_ros1(msg, conn.msgtype))
    return path


def wait(c, job_id, *statuses, timeout=90):
    t0 = time.time()
    while time.time() - t0 < timeout:
        job = c.get(f"/api/jobs/{job_id}").json()
        if job["status"] in statuses:
            return job
        time.sleep(0.2)
    raise AssertionError(f"job is {job['status']}, not {statuses}: {job['error']}")


def uploaded(c, tmp_path, name="clip one.mp4"):
    video = write_video(tmp_path / "src.mp4")
    r = c.put("/api/upload", params={"name": name}, content=video.read_bytes())
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]
    wait(c, job_id, "ready")
    return job_id


def events(c, job_id, last_id=None):
    """(event, data, id) of the job's event stream until its end event."""
    out, ev = [], {}
    headers = {"Last-Event-ID": last_id} if last_id else {}
    with c.stream("GET", f"/api/jobs/{job_id}/events", headers=headers) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        for line in r.iter_lines():
            if not line:
                if ev:
                    out.append((ev.get("event"), json.loads(ev["data"]), ev.get("id")))
                    if ev.get("event") == "end":
                        return out
                ev = {}
            elif not line.startswith(":"):
                k, _, v = line.partition(": ")
                ev[k] = v
    return out


def all_files(root):
    return [p for p in Path(root).rglob("*") if p.is_file() and not p.is_symlink()]


def test_token_gate(client):
    c, _ = client
    for path in ("/", "/static/app.js", "/static/index.html", "/api/jobs", "/api/jobs/x/file/selection.json", "/demos.html",
                 "/demos/x/y.mp4"):
        assert c.get(path).status_code == 401
    # The help page and its stylesheet hold no job data: they pass, so the page's troubleshooting reads without a login.
    assert c.get("/help.html").status_code == c.get("/static/style.css").status_code == 200
    assert c.head("/static/app.js").status_code == c.post("/help.html").status_code == 401
    assert c.put("/api/upload?name=a.mp4", content=b"x").status_code == 401
    assert c.get("/?token=wrong").status_code == 401 and c.get("/?token=").status_code == 401
    cookie = login(c).headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie
    assert "/static/app.js" in c.get("/").text and c.get("/static/app.js").status_code == 200
    assert c.get("/api/jobs").json() == [] and c.get("/api/jobs/20260101-000000-abcdef").status_code == 404
    c.cookies.clear()  # newer starlette sets the login cookie for its own domain: a second set() would not replace it
    c.cookies.set(server.COOKIE, "wrong")
    assert c.get("/api/jobs").status_code == 401


def test_demos_pages(tmp_path, capsys, fake_stages):
    """/demos.html shows the demos of the server's demos folder as cards, with the cards of its placeholders.json (or
    "No demos yet." without either), and /demos/<name>/<file> serves only the files that a demo.json lists, with their
    type, Range and HEAD. A demos folder that breaks a rule, or a bad placeholders.json, stops the server with the
    reason."""
    import re

    from test_webapp_demos import hand_demo

    from harpor.webapp import demos

    def app(folder, runs="runs"):
        cfg = steps.Config(G=tmp_path / "G", runs=tmp_path / runs, demos=folder)
        return server.make_app(cfg, TOKEN, key_dir=tmp_path / "keys")

    with TestClient(app(None)) as c:
        login(c)
        r = c.get("/demos.html")
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
        assert '<p class="emptytab">No demos yet.</p>' in r.text and "<!-- demos -->" not in r.text
    (tmp_path / "demos").mkdir()
    (tmp_path / "demos" / "placeholders.json").write_text('{"version": 1, "placeholders": ["Laptop"]}')
    with TestClient(app(tmp_path / "demos")) as c:
        login(c)
        r = c.get("/demos.html")
        assert '<span class="dctitle">Laptop</span><span class="dcnote">Coming soon</span>' in r.text
        assert "No demos yet." not in r.text and "<section" not in r.text
    d = hand_demo(tmp_path / "demos")
    (d / ".DS_Store").write_bytes(b"\0")
    with TestClient(app(tmp_path / "demos")) as c:
        login(c)
        r = c.get("/demos.html")
        assert 'id="demo-office-chair"' in r.text and r.headers["cache-control"] == "no-cache"
        assert r.text.index('<a class="democard" href="#demo-office-chair">') < r.text.index(">Laptop<")
        assert 'href="/static/style.css"' in r.text and 'href="help.html#demos"' in r.text
        assert c.get("/static/demos.js").content == (server.STATIC / "demos.js").read_bytes()
        mp4 = (d / "turntable.mp4").read_bytes()
        r = c.get("/demos/office-chair/turntable.mp4")
        assert r.status_code == 200 and r.headers["content-type"] == "video/mp4" and r.content == mp4
        r = c.get("/demos/office-chair/turntable.mp4", headers={"Range": "bytes=0-99"})
        assert r.status_code == 206 and r.content == mp4[:100] and r.headers["content-range"] == f"bytes 0-99/{len(mp4)}"
        r = c.head("/demos/office-chair/aligned.glb")
        assert r.status_code == 200 and r.headers["content-type"] == "model/gltf-binary" and not r.content
        assert c.get("/demos/office-chair/results.zip").headers["content-type"] == "application/zip"
        for bad in ("/demos/office-chair/demo.json", "/demos/office-chair/.DS_Store", "/demos/other/turntable.mp4",
                    "/demos/placeholders.json",
                    "/demos/office-chair/../office-chair/demo.json", "/demos/office-chair%2Fturntable.mp4/x",
                    "/demos/office-chair/%2E%2E%2Fdemo.json", "/demos/office-chair"):
            assert c.get(bad).status_code == 404, bad
    (tmp_path / "demos" / "placeholders.json").write_text('{"version": 1, "placeholders": ["Cage"]}')
    with pytest.raises(demos.DemoError, match=re.escape("placeholders.json: looks private (matches 'cage')")):
        app(tmp_path / "demos", runs="runs2")
    (tmp_path / "demos" / "placeholders.json").unlink()
    (d / "notes.zip").write_bytes(b"PK\x03\x04")
    with pytest.raises(demos.DemoError, match="office-chair/notes.zip is not listed in demo.json"):
        app(tmp_path / "demos", runs="runs2")
    assert not (tmp_path / "runs2").exists()  # stopped before the runner claims the runs folder
    with pytest.raises(SystemExit, match="the demos folder: office-chair/notes.zip is not listed in demo.json"):
        server.main(["--runs", str(tmp_path / "runs2"), "--host", "127.0.0.1", "--demos", str(tmp_path / "demos")])
    with pytest.raises(SystemExit, match="the demos folder: no demos folder .*missing"):  # mistyped
        server.main(["--runs", str(tmp_path / "runs2"), "--host", "127.0.0.1", "--demos", str(tmp_path / "missing")])
    assert not (tmp_path / "runs2").exists()


def test_upload_review_overrides_and_results(client, tmp_path):
    c, cfg = client
    login(c)
    assert c.put("/api/upload", content=b"").status_code == 400  # no name
    job_id = uploaded(c, tmp_path)
    J = cfg.runs / job_id
    assert c.get("/api/jobs").json()[0]["video"] == "clip_one.mp4" and not list(cfg.runs.glob("*/video/*.part"))
    meta = c.get(f"/api/jobs/{job_id}/frames").json()
    assert len(meta["times"]) == 64 and meta["candidates"] and meta["video"]["width"] == 192
    r = c.get(f"/api/jobs/{job_id}/thumb/{meta['thumbs'][0]}")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"

    start = {"mode": "auto", "prompt": "Wooden box", "n_views": 4, "auto_continue": False}
    bad = "bad-" + secrets.token_hex(8)
    r = c.post(f"/api/jobs/{job_id}/start", json={**start, "api_key": bad})
    assert r.status_code == 400 and bad not in r.text and c.get(f"/api/jobs/{job_id}").json()["status"] == "ready"
    for wrong in ({"n_views": 13}, {"prompt": ""}, {"mode": "manual", "picks": [3]}, {"picks": [0, 99]}):
        assert c.post(f"/api/jobs/{job_id}/start", json={**start, **wrong, "api_key": KEY}).status_code == 400, wrong
    assert c.post(f"/api/jobs/{job_id}/start", content=b"[1, 2]").status_code == 400
    r = c.post(f"/api/jobs/{job_id}/start", json={**start, "api_key": KEY})
    assert r.status_code == 200 and KEY not in r.text

    seen = events(c, job_id)  # the build up to the review, live
    kinds = {e for e, _, _ in seen}
    assert {"state", "log", "end"} <= kinds and seen[-1][1] == {"status": "review"}
    logs = [(d, i) for e, d, i in seen if e == "log"]
    assert all(i == f"{d['stage']}:{i.split(':')[1]}" for d, i in logs)
    assert {d["stage"] for d, _ in logs} >= {"extract", "masks", "prep", "da3", "select"}  # also stages between two looks
    again = events(c, job_id, last_id=logs[-1][1])  # resumed after the last log line: nothing twice
    assert [e for e, _, _ in again] == ["state", "end"]

    job = c.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "review" and job["slug"] == "wooden_box" and job["key_ok"] is True
    assert job["files"] == ["selection.json"] and job["sizes"] == {"selection.json": (J / "selection.json").stat().st_size}
    sel = c.get(f"/api/jobs/{job_id}/selection").json()
    assert sel["ok"] and len(sel["picks"]) >= 3 and set(sel["sam3"]) == {x["name"] for x in sel["candidates"]}
    assert sel["lidar"] is None and sel["metric_scale"] is None and job["lidar"] is None and job["calibration"] is None
    check = f"{sel['check_dir']}/{sel['picks'][0]['name']}.jpg"
    assert c.get(f"/api/jobs/{job_id}/file/{check}").headers["content-type"] == "image/jpeg"

    # Leave one pick out and say another frame does not show the object: masks, prep and select rerun.
    out, gone = sel["picks"][1]["name"], sel["picks"][2]["name"]
    bad_ov = [{"excluded": ["f99999"]}, {"locked": [out], "excluded": [out]}, {"instance": {gone: "7"}},
              {"instance": {gone: "x"}}, {"view0": "nope"}]
    for ov in bad_ov:
        assert c.post(f"/api/jobs/{job_id}/overrides", json=ov).status_code == 400, ov
    r = c.post(f"/api/jobs/{job_id}/overrides", json={"locked": [], "excluded": [out], "instance": {gone: "none"}})
    assert r.json() == {"status": "queued", "from": "masks"}
    job = wait(c, job_id, "review")
    sel = c.get(f"/api/jobs/{job_id}/selection").json()
    status = {x["name"]: x["status"] for x in sel["candidates"]}
    assert status[out] == "excluded" and status[gone] == "no_mask" and sel["overrides"]["excluded"] == [out]
    assert sel["sam3"][gone]["mask"] is False and job["stages"]["da3"]["status"] == "done"

    assert c.post(f"/api/jobs/{job_id}/continue", json={}).json() == {"status": "queued", "from": "sam3d"}
    job = wait(c, job_id, "done", "failed")
    assert job["status"] == "done", job["error"]
    # A rigid object without LiDAR or the 360 video has every output but the split parts, the LiDAR record, the videos
    # and the evidence of an articulation's rounds.
    assert set(job["files"]) == set(server.OUTPUTS) - {"run/parts.glb", "run/parts.json", "pool/lidar.json",
                                                      "turntable/turntable.mp4", "softvideo/softvideo.mp4",
                                                      "agent/evidence.json"}
    assert job["turntable"] is False and job["stages"]["turntable"]["note"] == "turned off"
    # The Downloads tab's sizes: the bytes that the file route sends for each file (refined.glb is the agent's GLB).
    assert set(job["sizes"]) == set(job["files"])
    assert all(job["sizes"][f] == len(c.get(f"/api/jobs/{job_id}/file/{f}").content) > 0 for f in job["files"]), job["sizes"]
    assert job["sizes"]["refined.glb"] == (J / "agent/s/work/exports/001.glb").stat().st_size
    assert job["agent_result"]["verdict"] == "complete" and job["articulated"] is False and job["parts_review"] is False
    assert [job["stages"][s]["note"] for s in ("propose", "parts")] == ["turned off: rigid object"] * 2
    names = set(zipfile.ZipFile(io.BytesIO(c.get(f"/api/jobs/{job_id}/file/results.zip").content)).namelist())
    assert {"posed.glb", "refined.glb", "aligned.glb", "summary.json", "selection.json", "mjcf/mjcf.json",
            "urdf/model.urdf"} <= names and "parts.glb" not in names
    for p in ("run/posed.glb", "refined.glb", "physics/aligned.glb"):
        r = c.get(f"/api/jobs/{job_id}/file/{p}")
        assert r.status_code == 200 and r.headers["content-type"] == "model/gltf-binary" and r.content[:4] == b"glTF"
    assert c.get(f"/api/jobs/{job_id}/file/refined.glb").content == (J / "agent/s/work/exports/001.glb").read_bytes()
    assert "== package" in c.get(f"/api/jobs/{job_id}/file/logs/package.log").text

    # Only the allowlist is served: no job.json, no video, no way out of the job folder.
    (tmp_path / "outside.jpg").write_bytes(b"secret")
    (J / "frames" / "thumbs" / "9999.jpg").symlink_to(tmp_path / "outside.jpg")
    for p in ("job.json", "video/clip_one.mp4", "frames/thumbs/../../job.json", "%2e%2e/%2e%2e/outside.jpg",
              "..%2F..%2Foutside.jpg", "/etc/passwd", "frames/thumbs/9999.jpg", "agent/result.json", "logs/../job.json"):
        assert c.get(f"/api/jobs/{job_id}/file/{p}").status_code == 404, p
    assert not any(KEY.encode() in p.read_bytes() for p in all_files(tmp_path))


def test_sizes_of_leaves_out_a_file_that_went(tmp_path):
    """sizes_of stats each file that resolve found; a file that goes between resolve and the stat, and a path
    that resolve did not find, have no size."""
    (tmp_path / "a.json").write_text("{}")
    (tmp_path / "b.glb").write_bytes(b"glTF" + bytes(96))
    found = {"a.json": tmp_path / "a.json", "b.glb": tmp_path / "b.glb", "gone.zip": tmp_path / "gone.zip", "none": None}
    assert server.sizes_of(found) == {"a.json": 2, "b.glb": 100}


def test_rosbag_upload_topic_choice_and_no_agent(client, tmp_path):
    """A bag with two camera topics waits for the choice; the build without the agent needs no key and
    gives physics the Stage A mesh."""
    pytest.importorskip("rosbags")
    c, cfg = client
    login(c)
    raw, rect = "/cam_a/image_raw/compressed", "/cam_b/image_rect/compressed"
    bag = write_bag(tmp_path / "src.bag", [raw, rect])
    r = c.put("/api/upload", params={"name": "test.bag"}, content=bag.read_bytes())
    assert r.status_code == 200, r.text
    job_id, J = r.json()["id"], cfg.runs / r.json()["id"]
    job = wait(c, job_id, "choose_topic", "failed")
    assert job["status"] == "choose_topic", job["error"]
    assert job["source"] == {"kind": "rosbag", "topic": None} and job["agent"] == "astra"
    assert job["stages"]["topics"]["status"] == "done" and job["stages"]["frames"]["status"] == "pending"
    assert c.get("/api/jobs").json()[0]["kind"] == "rosbag"
    found = c.get(f"/api/jobs/{job_id}/file/frames/topics.json").json()
    assert [t["topic"] for t in found["topics"]] == [rect, raw]  # rectified first
    assert [t["rectified"] for t in found["topics"]] == [True, None] and found["topics"][0]["count"] == 48
    r = c.get(f"/api/jobs/{job_id}/file/{found['topics'][0]['preview']}")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert c.get(f"/api/jobs/{job_id}/frames").status_code == 404
    assert c.post(f"/api/jobs/{job_id}/start", json={"mode": "auto", "prompt": "box", "agent": "none"}).status_code == 409

    for wrong in ({"topic": "/lidar/points"}, {"topic": None}, {}):
        assert c.post(f"/api/jobs/{job_id}/topic", json=wrong).status_code == 400, wrong
    assert c.post(f"/api/jobs/{job_id}/topic", json={"topic": raw}).json() == {"status": "analyzing", "topic": raw}
    job = wait(c, job_id, "ready", "failed")
    assert job["status"] == "ready", job["error"]
    meta = c.get(f"/api/jobs/{job_id}/frames").json()
    assert meta["video"]["kind"] == "rosbag" and meta["video"]["topic"] == raw and len(meta["times"]) == 48

    # Another topic while ready: frames runs again on it, topics does not.
    started = {s: job["stages"][s]["started"] for s in ("topics", "frames")}
    assert c.post(f"/api/jobs/{job_id}/topic", json={"topic": rect}).status_code == 200
    job = wait(c, job_id, "ready", "failed")
    assert job["source"]["topic"] == rect and job["stages"]["frames"]["started"] > started["frames"]
    assert job["stages"]["topics"]["started"] == started["topics"]
    assert c.get(f"/api/jobs/{job_id}/frames").json()["video"]["topic"] == rect

    start = {"mode": "auto", "prompt": "Wooden box", "n_views": 4, "auto_continue": False, "agent": "none"}
    assert c.post(f"/api/jobs/{job_id}/start", json={**start, "agent": "no"}).status_code == 400
    r = c.post(f"/api/jobs/{job_id}/start", json=start)  # no api_key at all
    assert r.status_code == 200, r.text
    job = wait(c, job_id, "review", "failed")
    assert job["status"] == "review", job["error"]
    assert job["agent"] == "none" and job["key_ok"] is False
    assert c.post(f"/api/jobs/{job_id}/topic", json={"topic": raw}).status_code == 409
    assert c.post(f"/api/jobs/{job_id}/continue", json={}).json() == {"status": "queued", "from": "sam3d"}
    job = wait(c, job_id, "done", "failed")
    assert job["status"] == "done", job["error"]
    assert job["stages"]["agent"]["status"] == "skipped" and job["stages"]["agent"]["note"] == "turned off: physics uses the Stage A mesh"
    assert job["agent_result"] is None and "refined.glb" not in job["files"] and "physics/aligned.glb" in job["files"]
    up = json.loads((J / "physics" / "up.json").read_text())
    assert up["glb"] == str(J / "run" / "posed.glb") and up["glb_rule"] == "turned off: physics uses the Stage A mesh"
    names = set(zipfile.ZipFile(io.BytesIO(c.get(f"/api/jobs/{job_id}/file/results.zip").content)).namelist())
    assert {"posed.glb", "aligned.glb", "summary.json", "mjcf/mjcf.json"} <= names
    assert "refined.glb" not in names and "agent_result.json" not in names
    assert c.post(f"/api/jobs/{job_id}/topic", json={"topic": raw}).status_code == 409  # done
    assert not list((tmp_path / "keys").iterdir())


def test_rosbag_with_one_topic_goes_straight_to_frames(client, tmp_path):
    pytest.importorskip("rosbags")
    c, cfg = client
    login(c)
    bag = write_bag(tmp_path / "src.bag", ["/cam/image_rect/compressed"], n=20)
    job_id = c.put("/api/upload", params={"name": "one.BAG"}, content=bag.read_bytes()).json()["id"]
    job = wait(c, job_id, "ready", "failed", "choose_topic")
    assert job["status"] == "ready", job["error"]
    assert job["source"] == {"kind": "rosbag", "topic": "/cam/image_rect/compressed"}
    assert [job["stages"][s]["status"] for s in ("topics", "frames")] == ["done", "done"]
    assert len(c.get(f"/api/jobs/{job_id}/frames").json()["times"]) == 20
    # A video job has no topic to choose.
    vid = uploaded(c, tmp_path)
    r = c.post(f"/api/jobs/{vid}/topic", json={"topic": "/cam/image_rect/compressed"})
    assert r.status_code == 409 and "video" in r.json()["detail"]
    assert c.get(f"/api/jobs/{vid}").json()["stages"]["topics"]["note"] == "video input"


def scene_bag(path, **kw):
    """A bag of fake_steps' scene with a LiDAR (test_webapp_lidar.write_scene_bag): 48 frames at 10 Hz,
    192x128, like write_bag's."""
    pytest.importorskip("rosbags")
    from test_webapp_lidar import write_scene_bag

    return write_scene_bag(path, n=48, size=(192, 128), dt=0.1, step=2, **kw)


def test_calibration_upload_and_the_lidar_start(client, tmp_path):
    """A bag whose TF does not link the camera to the LiDAR: Start with LiDAR asks for the calibration; the
    calibration's upload is checked, saved and summarised; then the build finds the scale with it."""
    from test_webapp_lidar import POINTS, T_CAM_LIDAR

    bag = scene_bag(tmp_path / "no_tf.bag", tf=None)
    c, cfg = client
    login(c)
    start = {"mode": "auto", "prompt": "Wooden box", "n_views": 4, "auto_continue": False, "agent": "none"}
    good = json.dumps({"T_cam_lidar": T_CAM_LIDAR.tolist()})
    vid = uploaded(c, tmp_path)
    r = c.post(f"/api/jobs/{vid}/start", json={**start, "lidar": {"topic": POINTS}})
    assert r.status_code == 400
    assert r.json()["detail"] == "LiDAR needs a rosbag or a Stray Scanner input; this job's input is a video, so turn LiDAR off"
    r = c.put(f"/api/jobs/{vid}/calibration", params={"name": "calib.json"}, content=good)
    assert r.status_code == 409 and "video" in r.json()["detail"]

    job_id = c.put("/api/upload", params={"name": "no_tf.bag"}, content=bag.read_bytes()).json()["id"]
    J = cfg.runs / job_id
    job = wait(c, job_id, "ready", "failed")
    assert job["status"] == "ready", job["error"]
    assert job["lidar"] is None and job["calibration"] is None
    found = c.get(f"/api/jobs/{job_id}/file/frames/topics.json").json()
    assert [x["topic"] for x in found["lidar"]] == [POINTS] and found["topics"][0]["tf_links"] == {POINTS: None}
    for lidar, why in (({"topic": "/nope"}, "/nope is not a LiDAR topic of the bag; its LiDAR topics are /lidar/points"),
                       ({"topic": POINTS}, "the bag's TF does not link camera_optical_frame to lidar; upload the calibration"),
                       (POINTS, 'lidar is null or {"topic": T}, a LiDAR topic of the bag')):
        r = c.post(f"/api/jobs/{job_id}/start", json={**start, "lidar": lidar})
        assert r.status_code == 400 and r.json()["detail"] == why, lidar

    # Refused uploads change nothing.
    url = f"/api/jobs/{job_id}/calibration"
    for params, content, code, detail in (
            ({"name": "calib.txt"}, good, 415, "upload the calibration as .yaml, .yml, .json"),
            ({"name": "calib.yaml"}, b"x" * (64 * 1024 + 1), 413, "a calibration file has at most 64 KB"),
            ({"name": "calib.yaml"}, b"T_cam_lidar: [1, 2, 3]", 400,
             "T_cam_lidar has 3 numbers; give a 4x4 matrix or 16 numbers"),
            ({"name": "calib.yaml"}, b"", 400, "the calibration file is empty"),
            ({}, good, 400, "name the file: ?name=<file name>")):
        r = c.put(url, params=params, content=content)
        assert (r.status_code, r.json()["detail"]) == (code, detail), params
    r = c.put(url, params={"name": "calib.yaml"}, content=(b"#" * 8192 for _ in range(9)))  # streamed, no length
    assert "content-length" not in r.request.headers and r.status_code == 413
    assert c.get(f"/api/jobs/{job_id}").json()["calibration"] is None and not (J / "lidar").exists()
    runner = c.app.app.state.runner
    busy, _ = runner.new_upload("busy.bag")  # while a job's upload runs, its calibration cannot change
    for refused in (c.put(f"/api/jobs/{busy}/calibration", params={"name": "calib.json"}, content=good),
                    c.delete(f"/api/jobs/{busy}/calibration")):
        assert (refused.status_code, refused.json()["detail"]) == (409, "the job is still uploading")
    runner.upload_done(busy, None)

    # A good one is saved (0600) and summarised; it can be removed and given again.
    r = c.put(url, params={"name": "my calib.json"}, content=good)
    assert r.status_code == 200, r.text
    summary = r.json()
    angle = np.degrees(np.arccos((np.trace(T_CAM_LIDAR[:3, :3]) - 1) / 2))
    assert summary == {"name": "my_calib.json", "bytes": len(good), "translation_m": round(np.linalg.norm(T_CAM_LIDAR[:3, 3]), 4),
                       "rotation_deg": pytest.approx(angle, abs=0.01), "has_intrinsics": False, "camera_axes": "optical",
                       "time_offset_s": None, "lidar_topic": None}  # the file gives no time offset, the upload no topic
    f = J / "lidar" / "calibration" / "my_calib.json"
    assert f.read_text() == good and stat.S_IMODE(f.stat().st_mode) == 0o600
    assert stat.S_IMODE((J / "lidar").stat().st_mode) == stat.S_IMODE(f.parent.stat().st_mode) == 0o700
    assert c.get(f"/api/jobs/{job_id}").json()["calibration"] == summary
    cal = jobs.load(J)["calibration"]
    assert cal["sha256"] == hashlib.sha256(good.encode()).hexdigest() and np.allclose(cal["T_cam_lidar"], T_CAM_LIDAR)
    assert c.delete(url).json() == {"calibration": None} and not f.exists()
    assert c.get(f"/api/jobs/{job_id}").json()["calibration"] is None
    assert c.post(f"/api/jobs/{job_id}/start", json={**start, "lidar": {"topic": POINTS}}).status_code == 400

    # T_lidar_cam given as T_cam_lidar: the lidar stage fails with the estimate's reason. With the right
    # calibration, a retry from the lidar stage stops at the review.
    inverted = json.dumps({"T_cam_lidar": np.linalg.inv(T_CAM_LIDAR).tolist()})
    assert c.put(url, params={"name": "calib.json"}, content=inverted).status_code == 200
    r = c.post(f"/api/jobs/{job_id}/start", json={**start, "lidar": {"topic": POINTS}})
    assert r.status_code == 200, r.text
    for refused in (c.put(url, params={"name": "calib.json"}, content=good), c.delete(url)):  # queued or running
        assert refused.status_code == 409 and refused.json()["detail"].startswith("the job is ")
    job = wait(c, job_id, "failed", "review")
    assert job["status"] == "failed" and job["error"].startswith("lidar: LiDAR scale: LiDAR points cover the object in only")
    assert job["stages"]["lidar"]["note"] == job["error"][len("lidar: "):] and job["stages"]["select"]["status"] == "pending"
    assert c.get(f"/api/jobs/{job_id}/file/pool/lidar.json").json()["ok"] is False
    yaml = "T_cam_lidar:\n" + "".join(f"  - [{', '.join(f'{x:.12f}' for x in row)}]\n" for row in T_CAM_LIDAR)
    r = c.put(url, params={"name": "calib.yaml"}, content=yaml)
    assert r.status_code == 200 and r.json()["name"] == "calib.yaml" and [p.name for p in f.parent.iterdir()] == ["calib.yaml"]
    assert c.post(f"/api/jobs/{job_id}/continue", json={"from": "lidar"}).json() == {"status": "queued", "from": "lidar"}
    job = wait(c, job_id, "review", "failed")
    assert job["status"] == "review", job["error"]
    assert job["lidar"] == {"kind": "rosbag", "topic": POINTS} and job["stages"]["lidar"]["status"] == "done"
    assert job["calibration"]["name"] == "calib.yaml" and "pool/lidar.json" in job["files"]

    # The review page's LiDAR record and the files it shows.
    sel = c.get(f"/api/jobs/{job_id}/selection").json()
    lid = sel["lidar"]
    assert lid["ok"] and lid["s"] == pytest.approx(1.25, rel=0.02) and sel["metric_scale"]["s"] == lid["s"]
    assert (lid["extrinsic"], lid["check_dir"], lid["reason"], lid["warnings"]) == ("calibration", "lidar/check", None, [])
    assert lid["source"]["topic"] == POINTS and lid["counts"]["views_used"] == len(lid["views"]) >= 3
    assert set(lid["counts"]) == set(server.LIDAR_COUNTS) and lid["spread_rel"] < 0.02
    assert all(set(v) == {"name", "s", "points_used", "status"} and v["status"] == "used" for v in lid["views"])
    r = c.get(f"/api/jobs/{job_id}/file/pool/lidar.json")
    assert r.status_code == 200 and r.headers["content-type"] == "application/json" and r.json()["s"] == lid["s"]
    r = c.get(f"/api/jobs/{job_id}/file/lidar/check/{lid['views'][0]['name']}.jpg")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    for p in ("lidar/samples.npz", "lidar/samples.json", "lidar/calibration/calib.yaml", "lidar/check/../../job.json",
              "lidar/check/x.jpg", "lidar/check/f1.jpg", "pool/da3_output.npz", "pool/prep.json"):
        assert c.get(f"/api/jobs/{job_id}/file/{p}").status_code == 404, p

    # A calibration uploaded in review makes the scale stale: Build finds it again first (and here the
    # inverted one fails); after the right one, Build finds the scale again and stops in review.
    assert c.put(url, params={"name": "calib.json"}, content=inverted).status_code == 200
    assert c.get(f"/api/jobs/{job_id}/selection").json()["lidar"]["calibration_changed"] is True
    assert c.post(f"/api/jobs/{job_id}/continue", json={}).json() == {"status": "queued", "from": "lidar"}
    job = wait(c, job_id, "failed", "review", "done")
    assert job["status"] == "failed" and job["error"].startswith("lidar: LiDAR scale: LiDAR points cover the object")
    assert c.put(url, params={"name": "calib.yaml"}, content=yaml).status_code == 200
    assert c.post(f"/api/jobs/{job_id}/continue", json={}).json() == {"status": "queued", "from": "lidar"}
    job = wait(c, job_id, "review", "failed", "done")
    assert job["status"] == "review", job["error"]
    lid = c.get(f"/api/jobs/{job_id}/selection").json()["lidar"]
    assert lid["ok"] and lid["calibration_changed"] is False and lid["s"] == pytest.approx(1.25, rel=0.02)
    assert c.get(f"/api/jobs/{job_id}/file/pool/lidar.json").json()["extrinsic"]["file"] == "calib.yaml"


def calibration_yaml(T, extra=""):
    """A calibration file that gives T as T_cam_lidar, with extra lines after it."""
    return "T_cam_lidar:\n" + "".join(f"  - [{', '.join(f'{x:.12f}' for x in row)}]\n" for row in T) + extra


def test_a_calibration_is_for_its_lidar_topic(client, tmp_path):
    """The upload names the LiDAR topic that the file is for (lidar_topic, one of the bag's); job.json and the
    summary keep it, and a Start with another LiDAR topic is refused with both topics named."""
    from test_webapp_lidar import POINTS, T_CAM_LIDAR

    bag = scene_bag(tmp_path / "no_tf.bag", tf=None)
    c, cfg = client
    login(c)
    job_id = c.put("/api/upload", params={"name": "no_tf.bag"}, content=bag.read_bytes()).json()["id"]
    J = cfg.runs / job_id
    job = wait(c, job_id, "ready", "failed")
    assert job["status"] == "ready", job["error"]
    url, yaml = f"/api/jobs/{job_id}/calibration", calibration_yaml(T_CAM_LIDAR)
    for topic, why in (("/nope", "/nope is not a LiDAR topic of the bag; its LiDAR topics are /lidar/points"),
                       ("/cam/image_rect", "/cam/image_rect is not a LiDAR topic of the bag; its LiDAR topics are /lidar/points")):
        r = c.put(url, params={"name": "calib.yaml", "lidar_topic": topic}, content=yaml)
        assert (r.status_code, r.json()["detail"]) == (400, why), topic
    assert jobs.load(J)["calibration"] is None and not (J / "lidar").exists()
    r = c.put(url, params={"name": "calib.yaml", "lidar_topic": POINTS}, content=yaml)
    assert r.status_code == 200 and r.json()["lidar_topic"] == POINTS
    assert jobs.load(J)["calibration"]["lidar_topic"] == POINTS and c.get(f"/api/jobs/{job_id}").json()["calibration"] == r.json()
    assert c.put(url, params={"name": "calib.yaml", "lidar_topic": ""}, content=yaml).json()["lidar_topic"] is None  # none chosen

    # A second LiDAR topic, such as a depth camera's points: the calibration for /lidar/points is not for them.
    found = json.loads((J / "frames" / "topics.json").read_text())
    found["lidar"].append({**found["lidar"][0], "topic": "/depth/points", "frame_id": "depth_optical_frame"})
    (J / "frames" / "topics.json").write_text(json.dumps(found))
    assert c.put(url, params={"name": "calib.yaml", "lidar_topic": POINTS}, content=yaml).status_code == 200
    start = {"mode": "auto", "prompt": "Wooden box", "n_views": 4, "auto_continue": False, "agent": "none"}
    r = c.post(f"/api/jobs/{job_id}/start", json={**start, "lidar": {"topic": "/depth/points"}})
    assert (r.status_code, r.json()["detail"]) == (400, "the calibration calib.yaml is for /lidar/points, not /depth/points; "
                                                        "choose /lidar/points, or upload a calibration for /depth/points")
    assert jobs.load(J)["status"] == "ready" and jobs.load(J)["lidar"] is None


def test_the_calibration_summary_gives_the_stored_offset():
    """A 0 is a given offset, also in a record without lidar_topic."""
    T = np.eye(4).tolist()
    assert server.calibration_summary({"T_cam_lidar": T, "time_offset_s": 0.0})["time_offset_s"] == 0.0
    assert server.calibration_summary({"T_cam_lidar": T, "time_offset_s": None, "lidar_topic": None})["time_offset_s"] is None


def test_the_calibration_cannot_change_while_the_part_list_waits(client, tmp_path):
    """While a job with LiDAR waits for its part list, a calibration upload or delete is refused (409, with
    what to do): the new scale would block the list's confirmation. Confirm then goes on. A job without LiDAR
    has no scale to block it, so its calibration can change while its list waits (for a build with LiDAR)."""
    from test_webapp_lidar import POINTS, T_CAM_LIDAR

    bag = scene_bag(tmp_path / "scene.bag")
    c, cfg = client
    login(c)
    job_id = c.put("/api/upload", params={"name": "scene.bag"}, content=bag.read_bytes()).json()["id"]
    J = cfg.runs / job_id
    assert wait(c, job_id, "ready", "failed")["status"] == "ready"
    url = f"/api/jobs/{job_id}/calibration"
    assert c.put(url, params={"name": "calib.yaml", "lidar_topic": POINTS}, content=calibration_yaml(T_CAM_LIDAR)).status_code == 200
    opus = "sk-ant-test-" + secrets.token_hex(16)
    r = c.post(f"/api/jobs/{job_id}/start", json={**ARTICULATED, "agent": "opus", "api_key": opus, "parts_mode": "vlm_edit",
                                                  "lidar": {"topic": POINTS}})
    assert r.status_code == 200, r.text
    job = wait(c, job_id, "review", "failed")
    assert job["status"] == "review" and job["parts_review"] is True, job["error"]
    assert job["stages"]["lidar"]["status"] == "done" and job["lidar_current"] is True
    why = "the job waits for its part list; confirm the list or cancel the job first, then change the calibration"
    for refused in (c.put(url, params={"name": "other.yaml", "lidar_topic": POINTS}, content=calibration_yaml(np.eye(4))),
                    c.delete(url)):
        assert (refused.status_code, refused.json()["detail"]) == (409, why)
    assert jobs.load(J)["calibration"]["name"] == "calib.yaml" and [p.name for p in (J / "lidar" / "calibration").iterdir()] == \
        ["calib.yaml"]
    proposal = c.get(f"/api/jobs/{job_id}/parts").json()["proposal"]["parts"]
    assert c.post(f"/api/jobs/{job_id}/parts", json={"parts": proposal}).json() == {"status": "queued", "from": "parts"}
    job = wait(c, job_id, "done", "failed")
    assert job["status"] == "done" and job["lidar_current"] is True, job["error"]
    # Once the list is confirmed (here: the job is done), the calibration can change again.
    assert c.delete(url).json() == {"calibration": None}

    job_id = c.put("/api/upload", params={"name": "scene.bag"}, content=bag.read_bytes()).json()["id"]
    assert wait(c, job_id, "ready", "failed")["status"] == "ready"
    r = c.post(f"/api/jobs/{job_id}/start", json={**ARTICULATED, "agent": "opus", "api_key": opus, "parts_mode": "vlm_edit"})
    assert r.status_code == 200, r.text
    job = wait(c, job_id, "review", "failed")
    assert job["parts_review"] is True and job["lidar"] is None, job["error"]
    url = f"/api/jobs/{job_id}/calibration"
    assert c.put(url, params={"name": "calib.yaml", "lidar_topic": POINTS}, content=calibration_yaml(T_CAM_LIDAR)).status_code == 200
    assert c.delete(url).status_code == 200
    proposal = c.get(f"/api/jobs/{job_id}/parts").json()["proposal"]["parts"]
    assert c.post(f"/api/jobs/{job_id}/parts", json={"parts": proposal}).json() == {"status": "queued", "from": "parts"}
    assert wait(c, job_id, "done", "failed")["status"] == "done"


def test_a_done_job_says_when_its_scale_is_stale(client, tmp_path):
    """GET /api/jobs/{id} "lidar_current": null without LiDAR or before the lidar stage is done; true while the
    scale was found with the job's calibration; false after a delete or an upload, also of a file of the same
    name with new content (steps.done compares the sha256). A continue from lidar on the done job finds the scale
    again with the new file, and the job is done again."""
    from test_webapp_lidar import POINTS, T_CAM_LIDAR

    bag = scene_bag(tmp_path / "no_tf.bag", tf=None)
    c, cfg = client
    login(c)
    job_id = c.put("/api/upload", params={"name": "no_tf.bag"}, content=bag.read_bytes()).json()["id"]
    job = wait(c, job_id, "ready", "failed")
    assert job["status"] == "ready" and job["lidar_current"] is None, job["error"]
    url, first = f"/api/jobs/{job_id}/calibration", calibration_yaml(T_CAM_LIDAR)
    assert c.put(url, params={"name": "calib.yaml", "lidar_topic": POINTS}, content=first).status_code == 200
    r = c.post(f"/api/jobs/{job_id}/start", json={"mode": "auto", "prompt": "Wooden box", "n_views": 4, "auto_continue": True,
                                                  "agent": "none", "lidar": {"topic": POINTS}})
    assert r.status_code == 200, r.text
    job = wait(c, job_id, "done", "failed")
    assert job["status"] == "done" and job["lidar_current"] is True, job["error"]

    # The same name with new content: here an offset of 0, which is a given value (the header clocks are one).
    second = calibration_yaml(T_CAM_LIDAR, "time_offset_s: 0\n")
    r = c.put(url, params={"name": "calib.yaml", "lidar_topic": POINTS}, content=second)
    assert r.json()["time_offset_s"] == 0
    job = c.get(f"/api/jobs/{job_id}").json()
    assert (job["calibration"]["name"], job["lidar_current"]) == ("calib.yaml", False)
    assert c.delete(url).status_code == 200 and c.get(f"/api/jobs/{job_id}").json()["lidar_current"] is False
    # Without the calibration the bag has no TF link: a run from lidar would delete the results, then fail there.
    r = c.post(f"/api/jobs/{job_id}/continue", json={"from": "lidar"})
    assert (r.status_code, r.json()["detail"]) == (400, "the bag's TF does not link camera_optical_frame to lidar; upload "
                                                        "the calibration")
    job = c.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "done" and "results.zip" in job["files"]
    assert c.put(url, params={"name": "calib.yaml", "lidar_topic": POINTS}, content=first).status_code == 200
    assert c.get(f"/api/jobs/{job_id}").json()["lidar_current"] is True  # the file that the scale was found with
    assert c.put(url, params={"name": "calib.yaml", "lidar_topic": POINTS}, content=second).status_code == 200

    # The page's button: lidar and every stage after it run again (auto_continue: no stop in review).
    assert c.post(f"/api/jobs/{job_id}/continue", json={"from": "lidar"}).json() == {"status": "queued", "from": "lidar"}
    job = wait(c, job_id, "done", "failed", "review")
    assert job["status"] == "done" and job["lidar_current"] is True, job["error"]
    assert job["stages"]["sam3d"]["status"] == "done" and "results.zip" in job["files"]
    rec = c.get(f"/api/jobs/{job_id}/file/pool/lidar.json").json()
    assert rec["extrinsic"]["calibration_sha256"] == hashlib.sha256(second.encode()).hexdigest()
    assert c.get(f"/api/jobs/{job_id}/file/logs/lidar.log").text.count("== lidar ") == 2

    # A job without LiDAR has no lidar_current, nor has one whose lidar stage failed (no scan within the read span).
    vid = uploaded(c, tmp_path)
    assert c.get(f"/api/jobs/{vid}").json()["lidar_current"] is None
    late = scene_bag(tmp_path / "late.bag", lidar_late_s=30.0)
    job_id = c.put("/api/upload", params={"name": "late.bag"}, content=late.read_bytes()).json()["id"]
    assert wait(c, job_id, "ready", "failed")["status"] == "ready"
    r = c.post(f"/api/jobs/{job_id}/start", json={"mode": "auto", "prompt": "Wooden box", "n_views": 4, "auto_continue": True,
                                                  "agent": "none", "lidar": {"topic": POINTS}})
    assert r.status_code == 200, r.text
    job = wait(c, job_id, "failed", "done")
    assert job["stages"]["lidar"]["status"] == "failed" and job["lidar"] and job["lidar_current"] is None


def test_a_run_again_from_lidar_needs_the_key(client, tmp_path):
    """A done job no longer holds its key, and a run that starts at or before the agent stage runs that stage
    again. A continue from lidar, or from a later stage while the calibration changed after the scale was found
    (Runner.resume then starts from lidar), needs the key before any stage runs. Before, it went on and the job
    stopped at the agent stage, saying that the server had restarted."""
    from test_webapp_lidar import POINTS, T_CAM_LIDAR

    bag = scene_bag(tmp_path / "scene.bag")
    c, cfg = client
    login(c)
    job_id = c.put("/api/upload", params={"name": "scene.bag"}, content=bag.read_bytes()).json()["id"]
    assert wait(c, job_id, "ready", "failed")["status"] == "ready"
    r = c.post(f"/api/jobs/{job_id}/start", json={"mode": "auto", "prompt": "Wooden box", "n_views": 4, "auto_continue": True,
                                                  "agent": "astra", "api_key": KEY, "lidar": {"topic": POINTS}})
    assert r.status_code == 200, r.text
    job = wait(c, job_id, "done", "failed")
    assert job["status"] == "done" and job["lidar_current"] is True and not job["key_ok"], job["error"]
    # The scale is current. A run from select or before runs every stage again, the agent's too: it needs the
    # key. From later on, Runner.resume skips the stages that are done, so no key: nothing runs again.
    for first in ("lidar", "select"):
        r = c.post(f"/api/jobs/{job_id}/continue", json={"from": first})
        assert r.status_code == 400 and "holds no OpenAI API key" in r.json()["detail"], (first, r.text)
        assert "the job finished, which drops the key" in r.json()["detail"]
    for first in ("sam3d", "agent", "turntable"):
        r = c.post(f"/api/jobs/{job_id}/continue", json={"from": first})
        assert r.json() == {"status": "queued", "from": first}, (first, r.text)
        job = wait(c, job_id, "done", "failed", "review")
        assert job["status"] == "done" and job["warnings"] == [], (first, job)
    url = f"/api/jobs/{job_id}/calibration"
    assert c.put(url, params={"name": "calib.yaml", "lidar_topic": POINTS},
                 content=calibration_yaml(T_CAM_LIDAR)).status_code == 200
    assert c.get(f"/api/jobs/{job_id}").json()["lidar_current"] is False
    for first in ("lidar", "urdf"):  # urdf: the stale scale moves the start back to lidar
        r = c.post(f"/api/jobs/{job_id}/continue", json={"from": first})
        assert r.status_code == 400 and "holds no OpenAI API key" in r.json()["detail"], (first, r.text)
    assert c.get(f"/api/jobs/{job_id}").json()["status"] == "done"  # nothing ran
    r = c.post(f"/api/jobs/{job_id}/continue", json={"from": "lidar", "api_key": KEY})
    assert r.json() == {"status": "queued", "from": "lidar"}, r.text
    job = wait(c, job_id, "done", "failed", "review")
    assert job["status"] == "done" and job["lidar_current"] is True and job["warnings"] == [], job


def test_a_run_from_lidar_checks_the_intrinsics_first(client, tmp_path):
    """A bag without camera_info, whose calibration gave K. When the calibration is replaced by one without K,
    the review's Build and a run again from lidar answer 400 with the reason, before anything is deleted. Before,
    they skipped the camera_info check: the run deleted the results, and then the lidar stage failed on it. A bag
    that is gone from the server is refused too, also when the calibration gives K."""
    import fake_steps as F
    from test_webapp_lidar import POINTS, T_CAM_LIDAR

    bag = scene_bag(tmp_path / "no_info.bag", info=False)
    c, cfg = client
    login(c)
    job_id = c.put("/api/upload", params={"name": "no_info.bag"}, content=bag.read_bytes()).json()["id"]
    J = cfg.runs / job_id
    assert wait(c, job_id, "ready", "failed")["status"] == "ready"
    url, K = f"/api/jobs/{job_id}/calibration", F.intrinsics(192, 128)
    with_k = calibration_yaml(T_CAM_LIDAR, f"K: {K.tolist()}\nimage_size: [192, 128]\n")
    without_k = calibration_yaml(T_CAM_LIDAR)
    assert c.put(url, params={"name": "calib.yaml", "lidar_topic": POINTS}, content=with_k).status_code == 200
    r = c.post(f"/api/jobs/{job_id}/start", json={"mode": "auto", "prompt": "Wooden box", "n_views": 4, "auto_continue": False,
                                                  "agent": "none", "lidar": {"topic": POINTS}})
    assert r.status_code == 200, r.text
    job = wait(c, job_id, "review", "failed")
    assert job["status"] == "review" and job["lidar_current"] is True, job["error"]
    why = "/cam/image_rect has no usable camera_info (/cam/camera_info); give K (and D) in the calibration"

    # The review's Build: the stale scale moves the start back to lidar (Runner.resume).
    assert c.put(url, params={"name": "calib.yaml", "lidar_topic": POINTS}, content=without_k).status_code == 200
    r = c.post(f"/api/jobs/{job_id}/continue", json={})
    assert r.status_code == 400 and r.json()["detail"] == why, r.text
    job = c.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "review" and job["stages"]["select"]["status"] == "done" and (J / "selection.json").is_file()
    assert c.put(url, params={"name": "calib.yaml", "lidar_topic": POINTS}, content=with_k).status_code == 200
    assert c.post(f"/api/jobs/{job_id}/continue", json={}).json() == {"status": "queued", "from": "sam3d"}
    job = wait(c, job_id, "done", "failed")
    assert job["status"] == "done" and job["lidar_current"] is True, job["error"]

    # A done job's run again from lidar (Retry): the results stay.
    assert c.put(url, params={"name": "calib.yaml", "lidar_topic": POINTS}, content=without_k).status_code == 200
    r = c.post(f"/api/jobs/{job_id}/continue", json={"from": "lidar"})
    assert r.status_code == 400 and r.json()["detail"] == why, r.text
    job = c.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "done" and "results.zip" in job["files"]
    assert (J / "results.zip").is_file() and (J / "run").is_dir()

    # The bag is gone: the lidar stage's own reason, without the server's path.
    assert c.put(url, params={"name": "calib.yaml", "lidar_topic": POINTS}, content=with_k).status_code == 200
    Path(jobs.load(J)["video"]["path"]).unlink()
    r = c.post(f"/api/jobs/{job_id}/continue", json={"from": "lidar"})
    assert (r.status_code, r.json()["detail"]) == (400, "the uploaded bag no_info.bag is not on the server; upload it again")
    assert (J / "results.zip").is_file() and (J / "run").is_dir()


def test_select_again_checks_the_lidar_first(client, tmp_path):
    """Select again (POST /overrides) in review, after the calibration of a bag without a TF link was removed. The
    scale is stale, so Runner.resume would start from lidar and delete the selection, and the lidar stage would fail.
    Select again answers 400 with the reason first, as Build does, and the review stays; also from masks (an
    instance changed), whose run includes the lidar stage. Before, the job went from review to failed. With the
    calibration back, it selects again."""
    from test_webapp_lidar import POINTS, T_CAM_LIDAR

    bag = scene_bag(tmp_path / "no_tf.bag", tf=None)
    c, cfg = client
    login(c)
    job_id = c.put("/api/upload", params={"name": "no_tf.bag"}, content=bag.read_bytes()).json()["id"]
    J = cfg.runs / job_id
    assert wait(c, job_id, "ready", "failed")["status"] == "ready"
    url, yaml = f"/api/jobs/{job_id}/calibration", calibration_yaml(T_CAM_LIDAR)
    assert c.put(url, params={"name": "calib.yaml", "lidar_topic": POINTS}, content=yaml).status_code == 200
    r = c.post(f"/api/jobs/{job_id}/start", json={"mode": "auto", "prompt": "Wooden box", "n_views": 4, "auto_continue": False,
                                                  "agent": "none", "lidar": {"topic": POINTS}})
    assert r.status_code == 200, r.text
    job = wait(c, job_id, "review", "failed")
    assert job["status"] == "review" and job["lidar_current"] is True, job["error"]
    out = c.get(f"/api/jobs/{job_id}/selection").json()["candidates"][-1]["name"]

    assert c.delete(url).status_code == 200
    why = "the bag's TF does not link camera_optical_frame to lidar; upload the calibration"
    for ov in ({"excluded": [out]}, {"instance": {out: "none"}}):  # from select, and from masks
        r = c.post(f"/api/jobs/{job_id}/overrides", json={"locked": [], "excluded": [], "instance": {}, **ov})
        assert (r.status_code, r.json()["detail"]) == (400, why), ov
    job = c.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "review" and job["stages"]["select"]["status"] == "done", job["error"]
    assert (J / "selection.json").is_file() and (J / "run").is_dir()

    assert c.put(url, params={"name": "calib.yaml", "lidar_topic": POINTS}, content=yaml).status_code == 200
    r = c.post(f"/api/jobs/{job_id}/overrides", json={"locked": [], "excluded": [out], "instance": {}})
    assert r.json() == {"status": "queued", "from": "select"}, r.text
    job = wait(c, job_id, "review", "failed")
    assert job["status"] == "review" and job["lidar_current"] is True, job["error"]
    assert c.get(f"/api/jobs/{job_id}/selection").json()["overrides"]["excluded"] == [out]


def test_calibration_changes_take_turns(client, tmp_path, monkeypatch):
    """A calibration upload checks that the job is idle again and writes the file and job.json in one step, under the
    runner's lock; the file is parsed before. Two uploads at once (two tabs) then leave job.json naming the file that
    the folder holds, and an upload whose first check passed before a Start answers 409. Before, job.json could name
    a file that the other upload deleted, and the upload changed the calibration of the queued job."""
    import threading

    from harpor.stage_a import lidar_scale
    from test_webapp_lidar import POINTS, T_CAM_LIDAR

    bag = scene_bag(tmp_path / "scene.bag")
    c, cfg = client
    login(c)
    job_id = c.put("/api/upload", params={"name": "scene.bag"}, content=bag.read_bytes()).json()["id"]
    J = cfg.runs / job_id
    assert wait(c, job_id, "ready", "failed")["status"] == "ready"
    url, yaml, out = f"/api/jobs/{job_id}/calibration", calibration_yaml(T_CAM_LIDAR), {}

    def put(name, extra=""):
        out[name] = c.put(url, params={"name": name, "lidar_topic": POINTS}, content=yaml + extra)

    def held():
        return jobs.load(J)["calibration"]["name"], sorted(p.name for p in (J / "lidar" / "calibration").iterdir())

    # a.yaml's job.json write waits; b.yaml comes meanwhile.
    update, inside, go = jobs.update, threading.Event(), threading.Event()

    def slow_update(job_dir, **fields):
        if (fields.get("calibration") or {}).get("name") == "a.yaml":
            inside.set()
            go.wait(10)
        return update(job_dir, **fields)

    monkeypatch.setattr(jobs, "update", slow_update)
    a = threading.Thread(target=put, args=("a.yaml",))
    a.start()
    assert inside.wait(10)
    b = threading.Thread(target=put, args=("b.yaml", "# b\n"))
    b.start()
    b.join(1.0)  # without the lock, b is done by now
    go.set()
    a.join(10)
    b.join(10)
    monkeypatch.setattr(jobs, "update", update)
    assert (out["a.yaml"].status_code, out["b.yaml"].status_code) == (200, 200)
    assert held() == ("b.yaml", ["b.yaml"])

    # An upload whose first check passed, then a Start in another tab: the check under the lock refuses the upload.
    read, parsing, started = lidar_scale.read_calibration, threading.Event(), threading.Event()

    def slow_read(data, name):
        if name == "late.yaml":
            parsing.set()
            started.wait(10)
        return read(data, name)

    monkeypatch.setattr(lidar_scale, "read_calibration", slow_read)
    late = threading.Thread(target=put, args=("late.yaml", "# late\n"))
    late.start()
    assert parsing.wait(10)
    r = c.post(f"/api/jobs/{job_id}/start", json={"mode": "auto", "prompt": "Wooden box", "n_views": 4, "auto_continue": False,
                                                  "agent": "none", "lidar": {"topic": POINTS}})
    assert r.status_code == 200, r.text
    started.set()
    late.join(10)
    assert out["late.yaml"].status_code == 409 and out["late.yaml"].json()["detail"] in ("the job is queued",
                                                                                         "the job is running")
    assert held() == ("b.yaml", ["b.yaml"])
    job = wait(c, job_id, "review", "failed")
    assert job["status"] == "review" and job["calibration"]["name"] == "b.yaml", job["error"]


def test_fake_rosbag_job_with_lidar_end_to_end(client, tmp_path):
    """A bag whose /tf_static links the camera to the LiDAR and whose camera_info is rectified: the
    LiDAR scale undoes the fake DA3's 0.8, and every later stage works in meters."""
    from test_webapp_lidar import POINTS, read_diff, read_glb

    bag = scene_bag(tmp_path / "scene.bag")
    c, cfg = client
    login(c)
    job_id = c.put("/api/upload", params={"name": "scene.bag"}, content=bag.read_bytes()).json()["id"]
    J = cfg.runs / job_id
    job = wait(c, job_id, "ready", "failed")
    assert job["status"] == "ready", job["error"]
    found = c.get(f"/api/jobs/{job_id}/file/frames/topics.json").json()
    assert found["topics"][0]["tf_links"] == {POINTS: "static"} and found["topics"][0]["camera_axes"] == "optical"
    r = c.post(f"/api/jobs/{job_id}/start", json={"mode": "auto", "prompt": "Wooden box", "n_views": 4,
                                                  "auto_continue": True, "agent": "none", "lidar": {"topic": POINTS}})
    assert r.status_code == 200, r.text
    job = wait(c, job_id, "done", "failed")
    assert job["status"] == "done", job["error"]
    assert job["stages"]["lidar"]["status"] == "done" and job["stages"]["lidar"]["note"] is None
    rec = c.get(f"/api/jobs/{job_id}/file/pool/lidar.json").json()
    assert rec["ok"] and rec["s"] == pytest.approx(1.25, rel=0.02), rec["reason"]
    assert (rec["extrinsic"]["source"], rec["extrinsic"]["link"], rec["time"]["base"]) == ("tf", "static", "header")
    assert rec["intrinsics"]["source"] == "/cam/camera_info: P (rectified)" and rec["source"]["type"] == "PointCloud2"
    with np.load(J / "run" / "da3_output.npz") as run:
        assert float(run["metric_scale"]) == pytest.approx(rec["s"])
    placement = c.get(f"/api/jobs/{job_id}/file/run/placement.json").json()
    assert placement["metric_scale"] == pytest.approx(rec["s"]) and placement["units"].startswith("meters")
    # The fake box is posed in the run's DA3 units (fake_steps.run_scale): with the scale applied it has
    # the scene's size, as does the aligned GLB (without it, both would be 0.8 x).
    assert placement["posed_extent_W"] == pytest.approx([0.3, 0.2, 0.2], rel=0.02)
    aligned = json.loads((J / "physics" / "aligned.json").read_text())
    assert sorted(aligned["extent_s"]) == pytest.approx([0.2, 0.2, 0.3], rel=0.02)
    assert c.get(f"/api/jobs/{job_id}/selection").json()["lidar"]["s"] == rec["s"]
    with zipfile.ZipFile(io.BytesIO(c.get(f"/api/jobs/{job_id}/file/results.zip").content)) as z:
        assert json.loads(z.read("lidar.json")) == rec
    assert "pool/lidar.json" in job["files"]
    # The LiDAR points of the views used (select's copies of lidar/cloud/<view>.glb), after the outputs.
    clouds = lidar_clouds(c, job_id, job)
    assert job["files"][-len(clouds):] == clouds and all(job["sizes"][f] > 0 for f in clouds)
    for method in (c.get, c.head):
        r = method(f"/api/jobs/{job_id}/file/run/lidar/0.glb")
        assert r.status_code == 200 and r.headers["content-type"] == "model/gltf-binary"
        assert r.headers["cache-control"] == "no-cache"
    assert c.get(f"/api/jobs/{job_id}/file/run/lidar/0.glb").content[:4] == b"glTF"
    # The real stage compares each view with the fake DA3, which is the scene times 0.8, so the gap is near 0.
    data = c.get(f"/api/jobs/{job_id}/file/run/lidar/0.glb").content
    assert sorted(read_glb(data)) == ["da3", "diff", "scan", "used"]
    assert read_diff(data)[1]["s"] == rec["s"] and abs(read_diff(data)[1]["median_m"]) < 0.02
    # The pool's own points only while the scale has failed (server.pool_clouds).
    pool = sorted((J / "lidar" / "cloud").glob("f*.glb"))
    assert pool and not any(f.startswith("lidar/") for f in job["files"])
    assert c.get(f"/api/jobs/{job_id}/file/lidar/cloud/{pool[0].name}").status_code == 404
    for p in ("run/lidar/1234.glb", "run/lidar/x.glb", "run/lidar/0.png", "lidar/cloud/f00001.glb", "lidar/clouds.npz"):
        assert c.get(f"/api/jobs/{job_id}/file/{p}").status_code == 404, p


def lidar_clouds(c, job_id, job):
    """job["files"]'s run/lidar/<i>.glb, which must be one per view used, by i."""
    picks = c.get(f"/api/jobs/{job_id}/file/selection.json").json()["picks"]
    clouds = [f for f in job["files"] if f.startswith("run/lidar/")]
    assert clouds == [f"run/lidar/{i}.glb" for i in range(len(picks))] and len(picks) >= 3
    return clouds


def test_lidar_points_are_listed_by_view(tmp_path):
    """server.lidar_points lists run/lidar/<i>.glb by the view's number, and only the files that ALLOW serves."""
    J = tmp_path / "job"
    (J / "run" / "lidar").mkdir(parents=True)
    for name in ("10.glb", "2.glb", "0.glb", "1234.glb", "x.glb", "3.png"):
        (J / "run" / "lidar" / name).write_bytes(b"glTF")
    (J / "run" / "lidar" / "4.glb").mkdir()  # not a file
    assert list(server.lidar_points(J)) == ["run/lidar/0.glb", "run/lidar/2.glb", "run/lidar/10.glb"]
    assert server.lidar_points(tmp_path / "other") == {}


def test_pool_points_only_while_the_scale_has_failed(tmp_path):
    """server.pool_clouds lists lidar/cloud/<view>.glb by name while pool/lidar.json says ok false; resolve and the
    file access follow it, so a scale found later takes them away."""

    J = tmp_path / "job"
    (J / "lidar" / "cloud").mkdir(parents=True)
    (J / "pool").mkdir()
    for name in ("f00040.glb", "f00012.glb", ".tmp_f00050.glb", "x.glb"):
        (J / "lidar" / "cloud" / name).write_bytes(b"glTF")
    for rec, listed in ((None, []), ({"ok": True}, []), ({"ok": False}, ["lidar/cloud/f00012.glb", "lidar/cloud/f00040.glb"])):
        if rec is not None:
            (J / "pool" / "lidar.json").write_text(json.dumps(rec))
        assert list(server.pool_clouds(J)) == listed, rec
        assert (server.resolve(J, "lidar/cloud/f00012.glb") is not None) == bool(listed)


def test_fake_stray_job_end_to_end(client, tmp_path):
    """A zipped Stray Scanner export of a portrait capture: the upload is a Stray job with LiDAR on, the
    analysis skips the topics and turns the frames upright, a calibration is refused, and the build's
    LiDAR scale from the depth images undoes the fake DA3's 0.8."""
    from test_webapp_stray import write_stray

    zipped = write_stray(tmp_path / "src", n=48, portrait=3)
    c, cfg = client
    login(c)
    r = c.put("/api/upload", params={"name": "my capture.zip"}, content=zipped.read_bytes())
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]
    J = cfg.runs / job_id
    job = wait(c, job_id, "ready", "failed")
    assert job["status"] == "ready", job["error"]
    assert (job["kind"], job["source"], job["lidar"], job["calibration"]) == ("stray", {"kind": "stray", "topic": None},
                                                                             {"kind": "stray"}, None)
    assert job["video"]["name"] == "my_capture.zip" and job["stages"]["topics"]["status"] == "skipped"
    assert job["stages"]["topics"]["note"] == "Stray Scanner capture"
    assert next(j for j in c.get("/api/jobs").json() if j["id"] == job_id)["kind"] == "stray"
    v = c.get(f"/api/jobs/{job_id}/frames").json()["video"]
    assert (v["kind"], v["rotation"], v["width"], v["height"], v["n_depth"]) == ("stray", 270, 240, 320, 48)
    assert (J / "video" / "stray" / "export" / "rgb.mp4").is_file()

    # No calibration and no topic for a capture; LiDAR is on or off, nothing else.
    url = f"/api/jobs/{job_id}/calibration"
    for r in (c.put(url, params={"name": "calib.json"}, content=json.dumps({"T_cam_lidar": np.eye(4).tolist()})),
              c.delete(url)):
        assert (r.status_code, r.json()["detail"]) == (
            409, "a Stray Scanner capture needs no calibration: its depth is aligned with the color camera")
    r = c.post(f"/api/jobs/{job_id}/topic", json={"topic": "/cam"})
    assert (r.status_code, r.json()["detail"]) == (409, "this job's input is a Stray Scanner capture, not a rosbag")
    start = {"mode": "auto", "prompt": "Wooden box", "n_views": 4, "auto_continue": True, "agent": "none"}
    r = c.post(f"/api/jobs/{job_id}/start", json={**start, "lidar": {"topic": "/lidar/points"}})
    assert (r.status_code, r.json()["detail"]) == (400, 'lidar is null or {"kind": "stray"} for a Stray Scanner capture')

    # Start without "lidar": the capture keeps its LiDAR on.
    r = c.post(f"/api/jobs/{job_id}/start", json=start)
    assert r.status_code == 200, r.text
    job = wait(c, job_id, "done", "failed")
    assert job["status"] == "done", job["error"]
    assert job["lidar"] == {"kind": "stray"} and job["stages"]["lidar"]["status"] == "done"
    rec = c.get(f"/api/jobs/{job_id}/file/pool/lidar.json").json()
    assert rec["ok"] and rec["s"] == pytest.approx(1.25, rel=0.02), rec["reason"]
    assert (rec["source"]["kind"], rec["extrinsic"]["source"], rec["time"]["base"], rec["intrinsics"]["turns"]) == \
        ("stray", "stray (depth aligned with the color camera)", "frame index", 3)
    with np.load(J / "run" / "da3_output.npz") as run:
        assert float(run["metric_scale"]) == pytest.approx(rec["s"])
    placement = c.get(f"/api/jobs/{job_id}/file/run/placement.json").json()
    assert placement["metric_scale"] == pytest.approx(rec["s"]) and placement["units"].startswith("meters")
    assert placement["posed_extent_W"] == pytest.approx([0.3, 0.2, 0.2], rel=0.02)
    sel = c.get(f"/api/jobs/{job_id}/selection").json()
    assert sel["lidar"]["s"] == rec["s"] and sel["lidar"]["extrinsic"] == "stray (depth aligned with the color camera)"
    with zipfile.ZipFile(io.BytesIO(c.get(f"/api/jobs/{job_id}/file/results.zip").content)) as z:
        assert json.loads(z.read("lidar.json")) == rec
    assert "pool/lidar.json" in job["files"]
    assert lidar_clouds(c, job_id, job)  # the capture's depth, with the intrinsics of odometry.csv

    # Built again with LiDAR off: the lidar stage is skipped and select applies no scale.
    r = c.post(f"/api/jobs/{job_id}/start", json={**start, "lidar": None})
    assert r.status_code == 200, r.text
    job = wait(c, job_id, "done", "failed")
    assert job["status"] == "done", job["error"]
    assert job["lidar"] is None and job["stages"]["lidar"]["note"] == "turned off: no LiDAR"
    assert "pool/lidar.json" not in job["files"] and c.get(f"/api/jobs/{job_id}/selection").json()["lidar"] is None
    assert not [f for f in job["files"] if f.startswith("run/lidar/")]
    placement = c.get(f"/api/jobs/{job_id}/file/run/placement.json").json()
    assert placement["metric_scale"] is None and placement["units"] == "DA3 units (meters)"


def test_upload_refuses_other_files_and_too_large_uploads(client, monkeypatch, tmp_path):
    c, cfg = client
    login(c)
    for name in ("notes.txt", "bag", "clip.mp4.tar", "clip.mKv"):  # the Kelvin sign lowercases to k, but is saved as _
        r = c.put("/api/upload", params={"name": name}, content=b"x")
        assert r.status_code == 415 and r.json()["detail"] == (
            "upload a video (.mp4 .mov .m4v .avi .mkv .webm), a rosbag (.bag .mcap .db3) or a Stray Scanner export "
            f"(.zip); {Path(name).suffix or 'a file without a suffix'} is none of these"), name
    r = c.put("/api/upload", params={"name": "big.bag"}, content=b"x", headers={"Content-Length": str(server.MAX_UPLOAD + 1)})
    assert r.status_code == 413 and "larger than 20 GB" in r.json()["detail"]
    r = c.put("/api/upload", params={"name": "a.bag"}, content=iter([b"x" * 1000]))  # chunked: no Content-Length
    assert r.status_code == 411 and "Content-Length" in r.json()["detail"] and "content-length" not in r.request.headers
    with monkeypatch.context() as m:  # the stand-in stages of the client stay
        m.setattr(server.shutil, "disk_usage", lambda p: shutil._ntuple_diskusage(100 << 30, 99 << 30, 1 << 30))
        r = c.put("/api/upload", params={"name": "a.mcap"}, content=b"x" * 1000)
        assert r.status_code == 413 and "1.0 GB free" in r.json()["detail"]
        # A zip is unpacked next to itself, so it needs 2.2 x its size free (+ 2 GB) where a bag needs 1.2 x.
        for name, need in (("a.bag", 3.2), ("capture.zip", 4.2)):
            r = c.put("/api/upload", params={"name": name}, content=b"x", headers={"Content-Length": str(1 << 30)})
            assert (r.status_code, r.json()["detail"]) == (413, f"the server has 1.0 GB free and this upload needs {need} GB")
        assert c.get("/api/jobs").json() == [] and not list(cfg.runs.glob("*/video"))
    # The kind is that of the name the file is saved under: "..bag" is kept as video.bag, a rosbag.
    pytest.importorskip("rosbags")
    bag = write_bag(tmp_path / "src.bag", ["/cam/image_rect/compressed"], n=20)
    job = wait(c, c.put("/api/upload", params={"name": "..bag"}, content=bag.read_bytes()).json()["id"], "ready", "failed")
    assert job["video"]["name"] == "video.bag" and job["source"] == {"kind": "rosbag", "topic": "/cam/image_rect/compressed"}


def test_start_without_the_agent_needs_no_key(client, tmp_path):
    c, cfg = client
    login(c)
    job_id = uploaded(c, tmp_path)
    r = c.post(f"/api/jobs/{job_id}/start", json={"mode": "auto", "prompt": "box", "agent": "none", "api_key": "not checked"})
    assert r.status_code == 200 and r.json()["key"] == "no key: the agent is turned off"
    job = wait(c, job_id, "review", "failed")
    assert job["status"] == "review" and job["agent"] == "none" and job["key_ok"] is False
    assert not list((tmp_path / "keys").iterdir())


def test_opus_build_and_a_change_of_agent(client, tmp_path):
    """A build with Claude Opus 5.5 takes an Anthropic key and shows the model, tokens and cost; back to
    GPT-6-Astra, the agent and the stages after it run again, with an OpenAI key."""
    c, cfg = client
    login(c)
    job_id = uploaded(c, tmp_path)
    opus = "sk-ant-test-" + secrets.token_hex(16)
    start = {"mode": "auto", "prompt": "Wooden box", "n_views": 4, "auto_continue": True, "agent": "opus"}
    r = c.post(f"/api/jobs/{job_id}/start", json={**start, "api_key": KEY})  # an OpenAI key for Opus
    assert r.status_code == 400 and r.json()["detail"] == "Anthropic keys start with sk-ant-" and KEY not in r.text
    for wrong in ("gpt", None, 1):
        r = c.post(f"/api/jobs/{job_id}/start", json={**start, "agent": wrong, "api_key": opus})
        assert r.status_code == 400 and r.json()["detail"] == "agent is astra, opus, fable, none", wrong
    r = c.post(f"/api/jobs/{job_id}/start", json={**start, "api_key": opus})
    assert r.status_code == 200 and r.json()["key"] == "key accepted (test)"
    job = wait(c, job_id, "done", "failed")
    assert job["status"] == "done", job["error"]
    a = job["agent_result"]
    assert job["agent"] == "opus" and (a["agent"], a["model"], a["verdict"]) == ("claude", "claude-opus-5-5", "complete")
    assert a["tokens"]["cache_read_input_tokens"] > 0 and a["cost_usd"] > 0 and a["turns"] == 12
    assert "agent (fake claude-opus-5-5)" in c.get(f"/api/jobs/{job_id}/file/logs/agent.log").text

    r = c.post(f"/api/jobs/{job_id}/continue", json={"from": "package", "agent": "astra"})
    assert r.status_code == 400 and "no OpenAI API key" in r.json()["detail"]
    assert c.post(f"/api/jobs/{job_id}/continue", json={"agent": "maybe"}).status_code == 400
    r = c.post(f"/api/jobs/{job_id}/continue", json={"from": "package", "agent": "astra", "api_key": KEY})
    assert r.json() == {"status": "queued", "from": "agent"}  # the other agent's stage runs first
    job = wait(c, job_id, "done", "failed")
    assert job["status"] == "done", job["error"]
    assert job["agent"] == "astra" and job["agent_result"]["model"] == "gpt-6-astra" and "tokens" not in job["agent_result"]
    assert c.get(f"/api/jobs/{job_id}/file/logs/agent.log").text.count("== agent ") == 2
    assert not list((tmp_path / "keys").iterdir())
    assert not any(k.encode() in p.read_bytes() for p in all_files(tmp_path) for k in (KEY, opus))


ARTICULATED = {"mode": "auto", "prompt": "Wooden box", "n_views": 4, "auto_continue": True, "articulated": True}
TYPED = [{"name": "box body"}, {"name": "lid", "parent": "box body", "joint": "revolute"},
         {"name": "drawer", "parent": "box body", "joint": "prismatic", "count": 2}]


def test_start_takes_the_parts_of_an_articulated_object(client, tmp_path):
    """articulated, parts_mode and parts of the start request, in each mode: the VLM modes keep no typed
    list, agent none means "user", and a list that breaks a rule gives 400 with harpor.parts.propose's
    reason before the key is checked."""
    c, cfg = client
    login(c)
    job_id = uploaded(c, tmp_path)
    J = cfg.runs / job_id
    start = {**ARTICULATED, "auto_continue": False, "agent": "astra", "api_key": KEY}
    no_list = "type the parts of the object: without an agent there is no VLM to list them"
    for wrong, why in [({"articulated": "yes"}, "articulated is true or false"),
                       ({"parts_mode": "auto"}, "the parts mode is one of vlm, vlm_edit, user"),
                       ({"parts_mode": "user"}, "type the parts of the object"),
                       ({"agent": "none"}, no_list), ({"agent": "none", "parts_mode": "vlm_edit", "parts": []}, no_list),
                       ({"parts_mode": "user", "parts": "box body"}, "parts is a list of parts"),
                       ({"parts_mode": "user", "parts": [TYPED[0], {**TYPED[1], "parent": "box"}]},
                        "part 'lid': its parent 'box' is not in the list"),
                       ({"parts_mode": "user", "parts": [{"name": "body"}, {"name": "lid"}]},
                        "exactly one part, the base, has no parent; here 2 have none: ['body', 'lid']"),
                       ({"parts_mode": "user", "parts": [{"name": "wooden box"}]},
                        "part 'wooden box' has the name of the whole object; call its main body 'Wooden box body'"),
                       ({"parts_mode": "user", "parts": [TYPED[0], {**TYPED[1], "joint": "hinge"}], "api_key": "bad"},
                        "part 'lid': joint is one of revolute, prismatic, fixed, soft, unknown, not 'hinge'")]:
        r = c.post(f"/api/jobs/{job_id}/start", json={**start, **wrong})
        assert r.status_code == 400 and r.json()["detail"] == why, (wrong, r.text)
    assert c.get(f"/api/jobs/{job_id}").json()["status"] == "ready"

    normal = [{"name": "box body", "parent": None, "joint": "fixed", "count": 1, "axis_hint": None},
              {"name": "lid", "parent": "box body", "joint": "revolute", "count": 1, "axis_hint": None},
              {"name": "drawer", "parent": "box body", "joint": "prismatic", "count": 2, "axis_hint": None}]
    for extra, mode, typed in [({}, "vlm", []), ({"parts_mode": "vlm_edit", "parts": TYPED}, "vlm_edit", []),
                               ({"parts_mode": "user", "parts": TYPED}, "user", normal),
                               ({"agent": "none", "parts_mode": "vlm", "parts": TYPED}, "user", normal)]:
        r = c.post(f"/api/jobs/{job_id}/start", json={**start, **extra})
        assert r.status_code == 200, (extra, r.text)
        job = c.get(f"/api/jobs/{job_id}").json()
        assert (job["articulated"], job["parts_mode"], job["parts_review"]) == (True, mode, False)
        assert job["parts"] == {"mode": mode, "list": typed, "confirmed": False}, extra
        wait(c, job_id, "running", "review")  # a queued job could still be taken by the worker
        assert c.post(f"/api/jobs/{job_id}/cancel").status_code == 200
        wait(c, job_id, "cancelled", timeout=20)
    assert r.json()["key"] == "no key: the agent is turned off"
    r = c.post(f"/api/jobs/{job_id}/start", json={**start, "articulated": False, "parts_mode": "user", "parts": TYPED})
    assert r.status_code == 200 and jobs.load(J)["parts"] == {"mode": "user", "list": [], "confirmed": False}
    assert jobs.load(J)["articulated"] is False  # a rigid object keeps no part list
    wait(c, job_id, "running", "review")
    assert c.post(f"/api/jobs/{job_id}/cancel").status_code == 200
    wait(c, job_id, "cancelled", timeout=20)
    assert not any(KEY.encode() in p.read_bytes() for p in all_files(tmp_path))


def test_part_list_review_confirms_or_edits_the_list(client, tmp_path):
    """"The VLM proposes, I edit the list": the job waits after propose, whatever auto_continue says. Build
    (continue) takes the VLM's list as it is; POST /parts takes an edited one and refuses a bad one (400)
    or a job that does not wait for its list (409)."""
    c, cfg = client
    login(c)
    job_id = uploaded(c, tmp_path)
    J, runner = cfg.runs / job_id, c.app.app.state.runner
    opus = "sk-ant-test-" + secrets.token_hex(16)
    r = c.post(f"/api/jobs/{job_id}/start", json={**ARTICULATED, "agent": "opus", "api_key": opus, "parts_mode": "vlm_edit"})
    assert r.status_code == 200, r.text
    job = wait(c, job_id, "review", "failed")
    assert job["status"] == "review", job["error"]
    assert (job["parts_review"], job["parts_mode"], job["stages"]["parts"]["status"]) == (True, "vlm_edit", "pending")
    assert server.state(J, jobs.load(J))["parts_review"] is True  # the event stream's state
    got = c.get(f"/api/jobs/{job_id}/parts").json()
    assert got["waiting"] is True and got["parts"] == {"mode": "vlm_edit", "list": [], "confirmed": False}
    assert [p["name"] for p in got["proposal"]["parts"]] == ["body", "lid"] and got["proposal"]["model"] == "claude-opus-5-5"
    assert got["noun"] == "Wooden box" and got["checks"] == []  # the parts stage has not run
    anchor = json.loads((J / "selection.json").read_text())["anchor"]
    assert got["photo"] == f"scene/sam3_wooden_box/check/{anchor}.jpg"
    assert c.get(f"/api/jobs/{job_id}/file/{got['photo']}").headers["content-type"] == "image/jpeg"

    r = c.post(f"/api/jobs/{job_id}/continue", json={"from": "agent"})
    assert r.status_code == 409 and r.json()["detail"] == "the job waits for its part list; confirm or edit it first"
    for wrong, why in [([{"name": "body"}, {"name": "lid", "parent": "box"}], "part 'lid': its parent 'box' is not in the list"),
                       ([], "a part list has 1 to 12 parts"), ({"parts": [{"name": "body"}]}, "parts is a list of parts")]:
        r = c.post(f"/api/jobs/{job_id}/parts", json={"parts": wrong})
        assert r.status_code == 400 and r.json()["detail"] == why, wrong
    r = c.post(f"/api/jobs/{job_id}/parts", json={"parts": [{"name": "a" * 5000}]})  # the reason quotes the name
    from harpor.parts.propose import MAX_NAME
    assert r.status_code == 400 and r.json()["detail"] == f"part name '{'a' * 37}...' is longer than {MAX_NAME} characters"
    assert server.part_reason(ValueError("x" * 500)) == "x" * (server.PART_REASON - 3) + "..."
    assert jobs.load(J)["status"] == "review" and "confirmed" not in json.loads((J / "run" / "proposal.json").read_text())

    # Cancel while the list waits, then Retry (from parts, as the page sends it): the job stops at the list again.
    assert c.post(f"/api/jobs/{job_id}/cancel").status_code == 200
    assert wait(c, job_id, "cancelled", timeout=20)["parts_review"] is False
    r = c.post(f"/api/jobs/{job_id}/continue", json={"from": "parts", "api_key": opus})
    assert r.json() == {"status": "queued", "from": "propose"}
    job = wait(c, job_id, "review", "failed")
    assert job["parts_review"] is True and job["stages"]["parts"]["status"] == "pending"
    assert "confirmed" not in json.loads((J / "run" / "proposal.json").read_text())

    # Build: the VLM's list as it is.
    assert c.post(f"/api/jobs/{job_id}/continue", json={}).json() == {"status": "queued", "from": "parts"}
    job = wait(c, job_id, "done", "failed")
    assert job["status"] == "done", job["error"]
    proposal = json.loads((J / "run" / "proposal.json").read_text())
    assert (proposal["confirmed"], proposal["edited"], job["parts"]["confirmed"], job["parts_review"]) == (True, False, True, False)
    assert set(server.OUTPUTS) - {"pool/lidar.json", "turntable/turntable.mp4", "softvideo/softvideo.mp4"} <= set(
        job["files"])  # but LiDAR and the videos
    a = job["agent_result"]
    assert (a["mode"], a["verdict"], a["joints_delivered"]) == ("articulation", "articulated", 1)
    assert set(a["counts"]) == {"exports", "reviews", "checkpoints", "joints", "parts_added", "articulation_reviews"}
    # The VLM's lid has a requirement, so the fake agent's result has the keys of a session in rounds, and its evidence.
    assert ([x["outcome"] for x in a["rounds"]], a["requirements"]["pass"], a["candidate"]["round"], a["simulations"]) == (
        ["pass"], 1, 1, 0)
    assert a["surface"]["shares"]["present"] == 1.0
    ev = c.get(f"/api/jobs/{job_id}/file/agent/evidence.json")
    assert ev.headers["content-type"] == "application/json" and ev.json()["selected"]["verdict"] == "articulated"
    assert c.get(f"/api/jobs/{job_id}/file/refined.glb").content == (J / "agent/s/work/exports/001.glb").read_bytes()
    r = c.post(f"/api/jobs/{job_id}/parts", json={"parts": [{"name": "body"}]})
    assert r.status_code == 409 and r.json()["detail"] == "the job does not wait for its part list"

    # From select again: the VLM lists the parts anew (the server forgot the key when the job was done), and an
    # edited list goes on from the parts stage. Without a key held for the job, POST /parts asks for it.
    r = c.post(f"/api/jobs/{job_id}/continue", json={"from": "select", "api_key": opus})
    assert r.json() == {"status": "queued", "from": "select"}
    job = wait(c, job_id, "review", "failed")
    assert job["parts_review"] is True and job["parts"]["confirmed"] is False
    runner._keys.pop(job_id)  # as after a restart of the server
    edited = [{"name": "body"}, {"name": "top lid", "parent": "body", "joint": "revolute", "axis_hint": "the back top edge"}]
    r = c.post(f"/api/jobs/{job_id}/parts", json={"parts": edited})
    assert r.status_code == 400 and "holds no Anthropic API key" in r.json()["detail"]
    r = c.post(f"/api/jobs/{job_id}/parts", json={"parts": edited, "api_key": opus})
    assert r.json() == {"status": "queued", "from": "parts"}
    job = wait(c, job_id, "done", "failed")
    assert job["status"] == "done", job["error"]
    proposal = json.loads((J / "run" / "proposal.json").read_text())
    assert [p["name"] for p in proposal["parts"]] == ["body", "top lid"] and proposal["edited"] is True
    assert proposal["parts"][1]["axis_hint"] == "the back top edge" and job["parts"]["list"] == proposal["parts"]
    found = [p["name"] for p in c.get(f"/api/jobs/{job_id}/file/run/parts.json").json()["parts"] if p["status"] == "found"]
    mj = c.get(f"/api/jobs/{job_id}/file/physics/mjcf/mjcf.json").json()
    assert found == ["body", "top_lid"] and [(j["child"], j["type"]) for j in mj["joints"]] == [("top_lid", "revolute")]
    checks = c.get(f"/api/jobs/{job_id}/parts").json()["checks"]
    assert checks and all(c.get(f"/api/jobs/{job_id}/file/{p}").headers["content-type"] == "image/jpeg" for p in checks)
    urdf = c.get(f"/api/jobs/{job_id}/file/physics/urdf/model.urdf")
    assert urdf.headers["content-type"] == "application/xml" and b'<joint name="top_lid_joint"' in urdf.content
    assert not list((tmp_path / "keys").iterdir())
    assert not any(opus.encode() in p.read_bytes() for p in all_files(tmp_path))


def test_articulated_job_without_an_agent_and_the_new_files(client, tmp_path):
    """The user's list without an agent: the parts are split but get no joints, so refined.glb is not served
    (physics uses run/parts.glb). The new outputs are served, and nothing else of the parts or the URDF. The job asks
    for the 360 video: it is served as video/mp4, with 206 for a Range request, and results.zip holds it; its
    records are not served."""
    c, cfg = client
    login(c)
    job_id = uploaded(c, tmp_path)
    J = cfg.runs / job_id
    r = c.post(f"/api/jobs/{job_id}/start", json={**ARTICULATED, "agent": "none", "parts": TYPED[:2], "turntable": True})
    assert r.status_code == 200, r.text
    job = wait(c, job_id, "done", "failed")
    assert job["status"] == "done", job["error"]
    assert (job["articulated"], job["parts_mode"], job["agent_result"]) == (True, "user", None)
    assert job["stages"]["agent"]["note"].startswith("turned off: without the agent the parts get no joints")
    assert set(job["files"]) == set(server.OUTPUTS) - {"refined.glb", "pool/lidar.json", "agent/evidence.json",
                                                      "softvideo/softvideo.mp4"}  # the lid is not soft
    assert (job["turntable"], job["stages"]["turntable"]["status"], job["warnings"]) == (True, "done", [])
    mp4 = (J / "turntable" / "turntable.mp4").read_bytes()
    r = c.get(f"/api/jobs/{job_id}/file/turntable/turntable.mp4")
    assert r.status_code == 200 and r.headers["content-type"] == "video/mp4" and r.content == mp4
    assert job["sizes"]["turntable/turntable.mp4"] == len(mp4)
    r = c.get(f"/api/jobs/{job_id}/file/turntable/turntable.mp4", headers={"Range": "bytes=0-99"})
    assert r.status_code == 206 and r.content == mp4[:100] and r.headers["content-range"] == f"bytes 0-99/{len(mp4)}"
    rec = json.loads((J / "turntable" / "turntable.json").read_text())
    assert rec["glb"] == "run/parts.glb" and rec["frames"] == 240  # the GLB that physics used, every part
    with zipfile.ZipFile(io.BytesIO(c.get(f"/api/jobs/{job_id}/file/results.zip").content)) as z:
        assert z.read("turntable.mp4") == mp4
    for p in ("turntable/turntable.json", "turntable/render.json", "turntable/args.json", "turntable/blender.log",
              "turntable/../job.json"):
        assert c.get(f"/api/jobs/{job_id}/file/{p}").status_code == 404, p
    assert steps.next_glb(J) == J / "run" / "parts.glb"
    assert c.get(f"/api/jobs/{job_id}/file/refined.glb").status_code == 404
    assert "links" not in c.get(f"/api/jobs/{job_id}/file/physics/mjcf/mjcf.json").json()  # one body
    got = c.get(f"/api/jobs/{job_id}/parts").json()
    assert got["waiting"] is False and got["proposal"]["source"] == "user" and got["parts"]["mode"] == "user"
    assert c.post(f"/api/jobs/{job_id}/parts", json={"parts": TYPED[:2]}).status_code == 409

    for p, kind in [("run/parts.glb", "model/gltf-binary"), ("run/parts.json", "application/json"),
                    ("physics/mjcf/mjcf.json", "application/json"), ("physics/urdf/model.urdf", "application/xml"),
                    ("run/parts/sam3/check/0.jpg", "image/jpeg")]:
        r = c.get(f"/api/jobs/{job_id}/file/{p}")
        assert r.status_code == 200 and r.headers["content-type"] == kind, p
    assert "run/parts/sam3/check/0.jpg" in got["checks"]
    (tmp_path / "outside.jpg").write_bytes(b"secret")
    (J / "run" / "parts" / "sam3" / "check" / "99.jpg").symlink_to(tmp_path / "outside.jpg")
    assert "run/parts/sam3/check/99.jpg" not in c.get(f"/api/jobs/{job_id}/parts").json()["checks"]
    for p in ("run/proposal.json", "run/parts/faces.npz", "run/parts/vote.json", "run/parts/split_report.json",
              "run/parts/sam3/part_masks.json", "run/parts/sam3/lid/0_0.png", "run/parts/sam3/check/lid.jpg",
              "run/parts/sam3/check/99.jpg", "run/parts/sam3/check/../../../job.json", "run/parts/sam3/check/..%2F..%2F..%2Fjob.json",
              "physics/urdf/urdf.json", "physics/urdf/meshes/link_0.obj", "physics/urdf/../../job.json",
              "physics/mjcf/scene.xml", "physics/links/lid.glb", "physics/mjcf/../../job.json"):
        assert c.get(f"/api/jobs/{job_id}/file/{p}").status_code == 404, p


def test_manual_start_cancel_and_delete(client, tmp_path):
    c, cfg = client
    login(c)
    job_id = uploaded(c, tmp_path)
    r = c.post(f"/api/jobs/{job_id}/start", json={"mode": "manual", "prompt": "box", "picks": [5, 30, 50], "view0": 30,
                                                  "api_key": KEY})
    assert r.status_code == 200
    wait(c, job_id, "running")
    assert c.delete(f"/api/jobs/{job_id}").status_code == 409
    assert c.post(f"/api/jobs/{job_id}/start", json={"mode": "auto", "prompt": "box", "api_key": KEY}).status_code == 409
    assert c.post(f"/api/jobs/{job_id}/cancel").status_code == 200
    job = wait(c, job_id, "cancelled", timeout=20)
    assert job["key_ok"] is False and job["picks"] == [5, 30, 50] and job["view0"] == 30
    r = c.post(f"/api/jobs/{job_id}/continue", json={"from": "extract"})
    assert r.status_code == 400 and "paste the key" in r.json()["detail"]
    assert c.delete(f"/api/jobs/{job_id}").status_code == 200 and not (cfg.runs / job_id).exists()
    assert c.get(f"/api/jobs/{job_id}").status_code == 404
    assert not any(KEY.encode() in p.read_bytes() for p in all_files(tmp_path))


def test_main_writes_the_login_file(tmp_path, monkeypatch, capsys):
    import uvicorn

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HARPOR_ROOT", str(tmp_path / "G"))
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    monkeypatch.setenv("SLURMD_NODENAME", "atl1-1-02-003-19-2")
    monkeypatch.setattr(jobs, "default_key_dir", lambda: tmp_path / "keys")
    info = tmp_path / "home" / ".config" / "harpor" / "webapp.json"
    info.parent.mkdir(parents=True, mode=0o755)  # an existing folder is used as it is, whatever its mode
    ran = []  # (port, webapp.json, its mode) while the server runs
    monkeypatch.setattr(uvicorn.Server, "run", lambda self, sockets: ran.append(
        (sockets[0].getsockname()[1], json.loads(info.read_text()), oct(info.stat().st_mode & 0o777))))
    args = ["--runs", str(tmp_path / "runs"), "--host", "127.0.0.1"]
    server.main(args)
    port, d, mode = ran[0]
    assert mode == "0o600" and d["port"] == port > 0
    assert d["url_path"] == f"/?token={d['token']}" and len(d["token"]) >= 40
    assert d["slurm_job"] == "4242" and d["slurm_node"] == "atl1-1-02-003-19-2"
    out = capsys.readouterr().out
    assert f"ssh -L {d['port']}:127.0.0.1:{d['port']} <this machine>" in out and d["token"] in out
    assert (tmp_path / "runs").is_dir()
    assert not info.exists()  # the server stopped, so start_slurm.sh must not find it
    # A later start replaced the file meanwhile: it stays.
    monkeypatch.setattr(uvicorn.Server, "run", lambda self, sockets: info.write_text(json.dumps({"token": "later"})))
    server.main(args)
    assert json.loads(info.read_text()) == {"token": "later"}


def test_sigterm_deletes_the_login_file(tmp_path):
    """The real server process: SIGTERM (scancel, the end of the Slurm job) stops it through main's
    finally, so webapp.json is gone; uvicorn raises the signal again after its shutdown."""
    import signal

    home = tmp_path / "home"
    info = home / ".config" / "harpor" / "webapp.json"
    env = {**os.environ, "HOME": str(home), "TMPDIR": str(tmp_path), "PYTHONPATH": str(ROOT)}
    p = subprocess.Popen([sys.executable, "-m", "harpor.webapp", "--runs", str(tmp_path / "runs"),
                          "--host", "127.0.0.1"], cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        lines = _lines_until(p.stdout, b"Application startup complete")
        assert b"Application startup complete" in lines[-1], b"".join(lines)
        assert info.is_file()
        p.send_signal(signal.SIGTERM)
        assert p.wait(30) == 128 + signal.SIGTERM
        assert not info.exists()
    finally:
        p.kill()
        p.wait()


def _lines_until(f, text, limit=200):
    """The lines of f up to the first that holds text."""
    out = []
    for line in iter(f.readline, b""):
        out.append(line)
        if text in line or len(out) >= limit:
            break
    return out


def test_start_slurm_reuses_only_a_server_of_the_same_job(tmp_path):
    """start_slurm.sh with a stand-in squeue: a webapp.json left by another job, or naming another node, is
    not trusted, even when something on its node and port answers 401 as the server does."""
    import http.server
    import threading

    if shutil.which("curl") is None:
        pytest.skip("no curl")
    home, bin_ = tmp_path / "home", tmp_path / "bin"
    web = home / "harpor" / "envs" / "web" / "bin"  # the default root
    web.mkdir(parents=True)
    (web / "python").symlink_to(sys.executable)
    bin_.mkdir()
    (bin_ / "squeue").write_text("#!/bin/bash\necho 'RUNNING atl1-node-07'\n")
    (bin_ / "setsid").write_text("#!/bin/bash\nexit 1\n")  # no new server starts here
    for f in bin_.iterdir():
        f.chmod(0o755)

    class Gate(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(401)
            self.end_headers()

        def log_message(self, *a):
            pass

    other = http.server.HTTPServer(("127.0.0.1", 0), Gate)
    threading.Thread(target=other.serve_forever, daemon=True).start()
    port = other.server_address[1]
    info = home / ".config" / "harpor" / "webapp.json"
    info.parent.mkdir(parents=True)

    def start(**recorded):
        info.write_text(json.dumps({"node": "127.0.0.1", "port": port, "token": "tok", **recorded}))
        return subprocess.run(["bash", str(ROOT / "harpor" / "webapp" / "start_slurm.sh"), "222"], capture_output=True,
                              text=True, timeout=60, env={**os.environ, "HOME": str(home), "PATH": f"{bin_}:{os.environ['PATH']}"})

    try:
        p = start(slurm_job="222", slurm_node="atl1-node-07")
        assert p.returncode == 0 and "reusing it" in p.stderr, p.stderr
        assert p.stdout.splitlines()[-3:] == ["NODE=127.0.0.1", f"PORT={port}", "TOKEN=tok"]
        for stale in ({"slurm_job": "111", "slurm_node": "atl1-node-07"}, {"slurm_job": "222", "slurm_node": "atl1-node-08"}):
            p = start(**stale)
            assert p.returncode == 1 and "reusing" not in p.stderr and "starting the server in job 222" in p.stderr, p.stderr
            assert "TOKEN=" not in p.stdout and not info.exists()
    finally:
        other.shutdown()


def test_start_slurm_starts_a_server_that_the_login_node_can_reach(tmp_path):
    """start_slurm.sh with stand-ins for squeue and setsid: its alive check calls http://$node:$port from the login
    node, so the server it starts listens on every address, not on the 127.0.0.1 default. The server runs from
    this script's repo and gets the script's root by name: HARPOR_ROOT, else ~/harpor."""
    home, bin_, argv = tmp_path / "home", tmp_path / "bin", tmp_path / "argv"
    home.mkdir()
    bin_.mkdir()
    (bin_ / "squeue").write_text("#!/bin/bash\necho 'RUNNING atl1-node-07'\n")
    (bin_ / "setsid").write_text(f"#!/bin/bash\necho \"$*\" > {argv}\nexit 1\n")  # records the command, starts nothing
    (bin_ / "sleep").write_text("#!/bin/bash\nexec /bin/sleep 0.1\n")
    for f in bin_.iterdir():
        f.chmod(0o755)

    def start(root, **roots):
        web = root / "envs" / "web" / "bin"
        web.mkdir(parents=True)
        (web / "python").symlink_to(sys.executable)
        p = subprocess.run(["bash", str(ROOT / "harpor" / "webapp" / "start_slurm.sh"), "222"], capture_output=True, text=True,
                           timeout=60, env={"HOME": str(home), "PATH": f"{bin_}:{os.environ['PATH']}", **roots})
        assert p.returncode == 1 and "the server did not start" in p.stderr, p.stderr
        assert (f"cd {ROOT} && HARPOR_ROOT={root} exec {web}/python -m harpor.webapp --host 0.0.0.0 --port 0"
                in argv.read_text())

    start(home / "harpor", GATOR_ROOT=str(tmp_path / "old"))  # the old name is not read
    start(tmp_path / "new", HARPOR_ROOT=str(tmp_path / "new"))


def test_open_slurm_reads_its_settings(tmp_path):
    """open_slurm.sh with stand-ins for ssh and the browser openers, and only those on PATH: the ssh host and the
    remote folder have no default, the socket does, and the remote folder's ~ stays for the remote shell."""
    home, bin_, calls = tmp_path / "home", tmp_path / "bin", tmp_path / "calls"
    home.mkdir()
    bin_.mkdir()
    for tool in ("cat", "mkdir", "sed"):
        (bin_ / tool).symlink_to(shutil.which(tool))
    record = f'#!/bin/bash\necho "${{0##*/}} $*" >> {calls}\n'
    (bin_ / "ssh").write_text(record + "case $* in *start_slurm.sh*) printf 'NODE=n\\nPORT=7\\nTOKEN=tok\\n';; esac\n")
    (bin_ / "open").write_text(record)
    (bin_ / "ssh").chmod(0o755)
    (bin_ / "open").chmod(0o755)

    def run(**env):
        calls.write_text("")
        p = subprocess.run([shutil.which("bash"), str(ROOT / "harpor" / "webapp" / "open_slurm.sh"), "222"], text=True,
                           capture_output=True, timeout=60, env={"HOME": str(home), "PATH": str(bin_), **env})
        return p, calls.read_text().splitlines()

    p, ran = run()
    assert p.returncode == 1 and "set HARPOR_SSH_HOST=user@login-node" in p.stderr and ran == []
    p, ran = run(HARPOR_SSH_HOST="me@login")
    assert p.returncode == 1 and "set HARPOR_REMOTE_CODE" in p.stderr and ran == []
    p, ran = run(HARPOR_SSH_HOST="me@login", HARPOR_REMOTE_CODE="~/code/HARPOR")
    assert p.returncode == 0, p.stderr
    sock = f"-S {home}/.ssh/cm-harpor"
    assert ran == [f"ssh {sock} -O check me@login",
                   f"ssh {sock} me@login bash ~/code/HARPOR/harpor/webapp/start_slurm.sh 222",
                   f"ssh {sock} -O forward -L 8765:n:7 me@login", "open http://localhost:8765/?token=tok"]
    assert (home / ".config" / "harpor").stat().st_mode & 0o777 == 0o700  # as fileio.private_folder asks
    (bin_ / "xdg-open").write_text(record)  # some Linux systems have an unrelated open
    (bin_ / "xdg-open").chmod(0o755)
    p, ran = run(HARPOR_SSH_HOST="me@login", HARPOR_SSH_SOCKET="/s/cm", HARPOR_REMOTE_CODE="/code")
    assert p.returncode == 0, p.stderr
    assert ran == ["ssh -S /s/cm -O check me@login", "ssh -S /s/cm me@login bash /code/harpor/webapp/start_slurm.sh 222",
                   "ssh -S /s/cm -O forward -L 8765:n:7 me@login", "xdg-open http://localhost:8765/?token=tok"]


def test_job_api_leaves_out_agent_text_that_held_the_key(client):
    c, cfg = client
    login(c)
    job_id, _ = jobs.create_job(cfg.runs, "v.mp4")
    (cfg.runs / job_id / "agent").mkdir()
    r = {"source": "finish", "verdict": "complete", "reason": "the cup is done", "limitations": ["no handle"],
         "exit": "finished", "key_found_in": []}
    (cfg.runs / job_id / "agent" / "result.json").write_text(json.dumps(r))
    assert c.get(f"/api/jobs/{job_id}").json()["agent_result"]["limitations"] == ["no handle"]
    (cfg.runs / job_id / "agent" / "result.json").write_text(json.dumps({**r, "key_found_in": ["result.json"]}))
    assert c.get(f"/api/jobs/{job_id}").json()["agent_result"] == {
        "source": "finish", "verdict": "complete", "reason": "left out: the agent's result held the API key",
        "limitations": [], "exit": "finished"}


def test_the_job_api_serves_the_evidence_and_the_round_keys(client):
    """agent/evidence.json (agent_step's scrubbed copy of the session's control/evidence.json) is an output while its
    source does not hold the key; the session folder stays unserved. The view takes the result's rounds,
    requirements, candidate, surface and simulations, and a result without them (an older job) gives none."""
    c, cfg = client
    login(c)
    job_id, _ = jobs.create_job(cfg.runs, "v.mp4")
    J = cfg.runs / job_id
    (J / "agent" / "s" / "control").mkdir(parents=True)
    (J / "agent" / "s" / "control" / "evidence.json").write_text('{"version": 1, "reason": "raw"}')
    (J / "agent" / "evidence.json").write_text('{"version": 1, "reason": "scrubbed"}')
    rounds = [{"round": 1, "kind": "build", "budget_s": 900.0, "exit": "exited", "wall_s": 3.0, "candidate": None,
               "outcome": "fail", "scope": "local", "findings": [{"code": "joint_missing", "requirement": "r1",
                                                                   "link": "lid", "measured": None, "expected": "revolute"}]}]
    new = {"rounds": rounds, "requirements": {"pass": 0, "fail": 1, "inconclusive": 0, "unassessed": 0}, "candidate": None,
           "surface": {"shares": {"present": 1.0}, "findings": [], "blocks": False}, "simulations": 2}
    r = {"source": "fallback_original", "verdict": "fallback_original", "reason": "x", "limitations": [], "exit": "exited",
         "mode": "articulation", "key_found_in": [], **new}
    (J / "agent" / "result.json").write_text(json.dumps(r))
    job = c.get(f"/api/jobs/{job_id}").json()
    assert {k: job["agent_result"][k] for k in new} == new and "agent/evidence.json" in job["files"]
    got = c.get(f"/api/jobs/{job_id}/file/agent/evidence.json")
    assert got.status_code == 200 and got.headers["content-type"] == "application/json" and got.json()["reason"] == "scrubbed"
    for p in ("agent/s/control/evidence.json", "agent/s/result.json", "agent/evidence.json/", "agent/Evidence.json"):
        assert c.get(f"/api/jobs/{job_id}/file/{p}").status_code == 404, p
    # The key was in the session's evidence: its copy is left out too, as agent/result.json is.
    (J / "agent" / "result.json").write_text(json.dumps({**r, "key_found_in": ["control/evidence.json"]}))
    assert "agent/evidence.json" not in c.get(f"/api/jobs/{job_id}").json()["files"]
    assert c.get(f"/api/jobs/{job_id}/file/agent/evidence.json").status_code == 404
    assert server.resolve(J, "agent/evidence.json") is None and server.key_in(J, "control/evidence.json")
    (J / "agent" / "result.json").write_text(json.dumps({k: v for k, v in r.items() if k not in new}))
    assert not set(new) & set(c.get(f"/api/jobs/{job_id}").json()["agent_result"])


def test_the_soft_part_video_is_an_output_and_its_records_are_not(client):
    """softvideo/softvideo.mp4 is served as video/mp4, with 206 for a Range request, and listed with its size; the
    stage's records and its motion stay on the server."""
    c, cfg = client
    login(c)
    job_id, _ = jobs.create_job(cfg.runs, "v.mp4")
    J = cfg.runs / job_id
    (J / "softvideo").mkdir()
    mp4 = b"\0\0\0\x18ftypmp42" + bytes(range(256))
    (J / "softvideo" / "softvideo.mp4").write_bytes(mp4)
    records = ("softvideo.json", "render.json", "motion.json", "motion.npz", "args.json", "blender.log")
    for f in records:
        (J / "softvideo" / f).write_text("{}")
    job = c.get(f"/api/jobs/{job_id}").json()
    assert "softvideo/softvideo.mp4" in job["files"] and job["sizes"]["softvideo/softvideo.mp4"] == len(mp4)
    r = c.get(f"/api/jobs/{job_id}/file/softvideo/softvideo.mp4")
    assert r.status_code == 200 and r.headers["content-type"] == "video/mp4" and r.content == mp4
    r = c.get(f"/api/jobs/{job_id}/file/softvideo/softvideo.mp4", headers={"Range": "bytes=0-99"})
    assert r.status_code == 206 and r.content == mp4[:100]
    for p in [f"softvideo/{f}" for f in records] + ["softvideo/../job.json", "softvideo/turntable.mp4"]:
        assert c.get(f"/api/jobs/{job_id}/file/{p}").status_code == 404, p


def test_events_send_a_rerun_stage_once(tmp_path):
    """A client that joins while a stage's earlier run is its newest (a retry waits in the queue) gets
    the lines of the new run once, also after the cursor moves on to the new run."""
    job_id, _ = jobs.create_job(tmp_path, "v.mp4")
    J = tmp_path / job_id
    t = time.time() - 100
    for k, s in enumerate(steps.NAMES[:6]):
        jobs.update_stage(J, s, status="done", started=t - 50 + k, ended=t - 40)
    jobs.update_stage(J, "sam3d", status="failed", started=t, ended=t + 5, note="cancelled")
    jobs.update(J, status="queued")
    log = J / "logs" / "sam3d.log"
    log.write_text("== sam3d earlier\nold line\n")
    sent, cursor, last = [], None, None

    def look():
        nonlocal cursor, last
        out, cursor, last, _ = server.tick(J, cursor, last)
        sent.extend(json.loads(e.split("\ndata: ", 1)[1])["text"] for e in out if e.startswith("id: "))

    look()
    look()
    with open(log, "a") as f:
        f.write("== sam3d now\nline A\n")
    jobs.update_stage(J, "sam3d", status="running", started=time.time(), ended=None)
    jobs.update(J, status="running", current="sam3d")
    look()
    with open(log, "a") as f:
        f.write("line B\n")
    for _ in range(3):  # line B, a quiet look (the cursor moves to the new run), another look
        look()
    assert "".join(sent) == "== sam3d earlier\nold line\n== sam3d now\nline A\nline B\n"


def test_frames_retry_analyzes_the_video_again(client, tmp_path):
    c, cfg = client
    login(c)
    job_id = uploaded(c, tmp_path)
    J = cfg.runs / job_id
    # As after a cancel during the analysis: frames failed, and the video is still there.
    (J / "frames" / "meta.json").unlink()
    jobs.update_stage(J, "frames", status="failed", note="cancelled", returncode=-15)
    jobs.update(J, status="cancelled")
    assert c.post(f"/api/jobs/{job_id}/start", json={"mode": "auto", "prompt": "box", "api_key": KEY}).status_code == 409
    assert c.post(f"/api/jobs/{job_id}/continue", json={"from": "frames"}).json() == {"status": "analyzing", "from": "frames"}
    job = wait(c, job_id, "ready", "failed")
    assert job["status"] == "ready" and job["stages"]["frames"]["status"] == "done" and (J / "frames" / "meta.json").is_file()
    r = c.post(f"/api/jobs/{job_id}/continue", json={"from": "frames"})
    assert r.status_code == 409 and "already analyzed" in r.json()["detail"]
    jobs.update_stage(J, "frames", status="failed", note="cancelled", returncode=-15)
    Path(job["video"]["path"]).unlink()
    r = c.post(f"/api/jobs/{job_id}/continue", json={"from": "frames"})
    assert r.status_code == 409 and "upload it again" in r.json()["detail"]


def test_preview_video_and_view_images_are_served(client, tmp_path):
    """The input's preview video is served as video/mp4 with byte ranges, so the page can seek in it: 206 with
    Content-Range for a range, 416 for one past the end; HEAD gets the headers alone. The job names it ("preview")
    with where each view is in it ("seek"), says why a job has none or that its file is gone, or gives null before
    frames/meta.json is written. The run's view images and place's overlays are served, and nothing else around
    them."""
    c, cfg = client
    login(c)
    job_id = uploaded(c, tmp_path)  # 64 frames at 24 fps, 192x128
    J = cfg.runs / job_id
    url = f"/api/jobs/{job_id}/file/frames/preview.mp4"
    data = (J / "frames" / "preview.mp4").read_bytes()
    meta = json.loads((J / "frames" / "meta.json").read_text())
    job = c.get(f"/api/jobs/{job_id}").json()
    assert meta["preview"]["moved"] == {}  # no frames come in bursts
    assert job["preview"] == {**{k: v for k, v in meta["preview"].items() if k != "moved"}, "seek": {}}  # no views yet
    assert job["preview"]["bytes"] == len(data) > 1000
    assert {k: job["preview"][k] for k in ("width", "height", "stride", "frames", "t0")} == \
        {"width": 192, "height": 128, "stride": 2, "frames": 32, "t0": 0.0}
    r = c.get(url)
    assert r.status_code == 200 and r.headers["content-type"] == "video/mp4" and r.content == data
    assert r.headers["accept-ranges"] == "bytes" and int(r.headers["content-length"]) == len(data)
    r = c.head(url)
    assert r.status_code == 200 and r.headers["content-type"] == "video/mp4" and r.content == b""
    assert int(r.headers["content-length"]) == len(data) and r.headers["accept-ranges"] == "bytes"
    assert c.head(f"/api/jobs/{job_id}/file/frames/meta.json").status_code == 404
    n = len(data)
    for rng, a, b in (("bytes=0-99", 0, 99), ("bytes=100-", 100, n - 1), ("bytes=-10", n - 10, n - 1),
                      (f"bytes=500-{n + 50}", 500, n - 1)):
        r = c.get(url, headers={"Range": rng})
        assert r.status_code == 206 and r.headers["content-range"] == f"bytes {a}-{b}/{n}", rng
        assert r.content == data[a:b + 1] and r.headers["content-type"] == "video/mp4", rng
    r = c.get(url, headers={"Range": f"bytes={n}-"})
    assert r.status_code == 416 and r.headers["content-range"] == f"bytes */{n}" and not r.content
    assert c.get(url, headers={"Range": "bytes=abc"}).status_code in (400, 416)  # never a part of the file
    r = c.get(url, headers={"Range": "bytes=0-99", "If-Range": '"another version"'})
    assert r.status_code == 200 and r.content == data  # the file changed: all of it
    r = c.get(url, headers={"Range": "bytes=0-99", "If-Range": c.get(url).headers["etag"]})
    assert r.status_code == 206 and r.content == data[:100]

    # The views of a run, and place's overlays of the posed mesh on them.
    from PIL import Image

    (J / "run" / "input" / "images").mkdir(parents=True)
    for rel in ("run/input/images/0.png", "run/input/images/11.png", "run/overlay_0.png"):
        Image.new("RGB", (8, 6), (200, 10, 10)).save(J / rel)
        r = c.get(f"/api/jobs/{job_id}/file/{rel}")
        assert r.status_code == 200 and r.headers["content-type"] == "image/png" and r.content == (J / rel).read_bytes()
    (J / "run" / "input" / "object").mkdir()
    for rel in ("run/input/object/0.png", "run/masks/0.png", "run/input/images/1000.png", "run/input/images/x.png"):
        (J / rel).parent.mkdir(parents=True, exist_ok=True)
        (J / rel).write_bytes(b"not served")
    (J / "frames" / ".tmp_preview.mp4").write_bytes(b"half written")
    for p in ("run/input/object/0.png", "run/masks/0.png", "run/input/images/1000.png", "run/input/images/x.png",
              "run/input/images/0.jpg", "run/input/images/../../job.json", "run/input/images/0.png/..",
              "run/input/images/%2e%2e/%2e%2e/job.json", "frames/preview.mp4/../meta.json", "frames/../job.json",
              "frames/.tmp_preview.mp4", "frames/preview.mp4.part", "frames/meta.json", "video/clip_one.mp4",
              "run/overlay_0.png/../../job.json", "run//input/images/0.png"):
        assert c.get(f"/api/jobs/{job_id}/file/{p}").status_code == 404, p
    (J / "frames" / ".tmp_preview.mp4").unlink()

    # Where the views are in the preview: the kept frame at or before each pick (every second frame is kept), at
    # its time less t0, or where the frames stage moved it.
    times = meta["times"]
    picks = [{"name": "f00020", "frame": 20, "time_s": times[20]}, {"name": "f00041", "frame": 41, "time_s": times[41]},
             {"name": "f00063", "frame": 63, "time_s": times[63]}, {"name": "f00099", "frame": 99, "time_s": 9.9}]
    (J / "selection.json").write_text(json.dumps({"ok": True, "picks": picks}))
    seek = c.get(f"/api/jobs/{job_id}").json()["preview"]["seek"]
    assert seek == {"20": {"frame": 20, "at": pytest.approx(times[20])}, "41": {"frame": 40, "at": pytest.approx(times[40])},
                    "63": {"frame": 62, "at": pytest.approx(times[62])}}  # frame 99 is not in the input
    (J / "frames" / "meta.json").write_text(json.dumps({**meta, "preview": {**meta["preview"], "t0": 0.5,
                                                                            "moved": {"40": 1.7}}}))
    seek = c.get(f"/api/jobs/{job_id}").json()["preview"]["seek"]
    assert seek["41"] == {"frame": 40, "at": 1.7} and seek["20"] == {"frame": 20, "at": pytest.approx(times[20] - 0.5)}
    (J / "selection.json").unlink()

    # A preview that the frames stage could not write says why; one whose file is gone since says so; without
    # meta.json there is none.
    (J / "frames" / "meta.json").write_text(json.dumps({**meta, "preview": {"error": "ValueError: no libx264"}}))
    assert c.get(f"/api/jobs/{job_id}").json()["preview"] == {"error": "ValueError: no libx264"}
    (J / "frames" / "meta.json").write_text(json.dumps(meta))
    (J / "frames" / "preview.mp4").unlink()
    assert c.get(f"/api/jobs/{job_id}").json()["preview"] == {"missing": True}
    assert c.get(url).status_code == 404
    (J / "frames" / "meta.json").unlink()
    assert c.get(f"/api/jobs/{job_id}").json()["preview"] is None

    # Analyzing the input again deletes a half-written preview that a killed run left, and writes a new one.
    (J / "frames" / ".tmp_preview.mp4").write_bytes(b"cut off")
    jobs.update_stage(J, "frames", status="failed", note="cancelled", returncode=-15)
    jobs.update(J, status="cancelled")
    assert c.post(f"/api/jobs/{job_id}/continue", json={"from": "frames"}).json()["from"] == "frames"
    job = wait(c, job_id, "ready", "failed")
    assert job["status"] == "ready" and job["preview"]["frames"] == 32 and (J / "frames" / "preview.mp4").read_bytes()
    assert not (J / "frames" / ".tmp_preview.mp4").exists()


def test_no_delete_while_uploading(tmp_path, fake_stages):
    """The jobs list shows a job as soon as its upload starts; deleting it must wait for the upload."""
    import anyio
    import httpx

    cfg = steps.Config(G=tmp_path / "G", runs=tmp_path / "runs")
    app = server.make_app(cfg, TOKEN, key_dir=tmp_path / "keys")
    video = write_video(tmp_path / "src.mp4").read_bytes()
    seen = {}

    async def main():
        first, rest = anyio.Event(), anyio.Event()

        async def body():
            yield video[:1000]
            first.set()  # the server has the first chunk and waits for more
            await rest.wait()
            yield video[1000:]

        async def put():
            seen["put"] = await c.put("/api/upload", params={"name": "slow.mp4"}, content=body(),
                                      headers={"Content-Length": str(len(video))})  # as the page's XHR sends

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://app",
                                     headers={"Cookie": f"{server.COOKIE}={TOKEN}"}) as c:
            async with anyio.create_task_group() as tg:
                tg.start_soon(put)
                await first.wait()
                seen["list"] = (await c.get("/api/jobs")).json()
                seen["delete"] = await c.delete(f"/api/jobs/{seen['list'][0]['id']}")
                rest.set()

    try:
        anyio.run(main)
        [row] = seen["list"]
        assert row["status"] == "uploading" and seen["delete"].status_code == 409
        assert "still uploading" in seen["delete"].json()["detail"]
        assert seen["put"].status_code == 200 and seen["put"].json() == {"id": row["id"], "bytes": len(video)}
        t0 = time.time()
        while jobs.load(cfg.runs / row["id"])["status"] != "ready" and time.time() - t0 < 60:
            time.sleep(0.1)
        assert (cfg.runs / row["id"] / "video" / "slow.mp4").read_bytes() == video
    finally:
        app.app.state.runner.shutdown(timeout=5)


def test_agent_of_and_key_later_at_module_level():
    """server.agent_of, now at module level so that a worker can ask it, answers as the routes always did;
    key_later is true only without a key."""
    for b, job, want in [({}, None, "astra"), ({"agent": "opus"}, None, "opus"), ({"agent": "none"}, None, "none"),
                         ({}, {"agent": "opus"}, "opus"), ({"agent": None}, {"agent": "opus"}, "opus"),
                         ({"agent": "astra"}, {"agent": "opus"}, "astra"), ({"agent": "none"}, {"agent": "opus"}, "none"),
                         ({}, {"agent": "none"}, "none")]:
        assert server.agent_of(b, job) == want, (b, job)
    for b, job in [({"agent": None}, None), ({"agent": 1}, None), ({"agent": 0}, None), ({"agent": []}, None),
                   ({"agent": True}, None), ({"agent": False}, {"agent": "opus"}),
                   ({"agent": {}}, None), ({"agent": "Opus"}, None), ({"agent": "gpt"}, {"agent": "opus"})]:
        with pytest.raises(server.Bad) as e:
            server.agent_of(b, job)
        assert e.value.args == (400, "agent is astra, opus, fable, none"), b
    assert [server.key_later(b) for b in ({}, {"key_later": False}, {"key_later": True}, {"key_later": True, "api_key": KEY},
                                          {"key_later": True, "api_key": ""})] == [False, False, True, False, True]
    for v in (1, 0, "true", None, []):
        with pytest.raises(server.Bad) as e:
            server.key_later({"key_later": v})
        assert e.value.args == (400, "key_later is true or false"), v


def test_the_tunnel_page_stays_as_it_was(client):
    """The tunnel page loads no Supabase script and still offers the agent "none"; the server serves none of the public
    site's scripts."""
    c, _ = client
    login(c)
    page = c.get("/").text
    assert "supabase" not in page.lower() and "supa.js" not in page and "config.js" not in page
    assert 'name="agent" value="none"' in page
    for name in ("supa.js", "account.js", "config.js"):
        assert c.get(f"/static/{name}").status_code == 404, name


def test_help_page_is_served_with_or_without_the_login(client):
    """GET /help.html gives static/help.html as it is, with no-cache like the page, also without the cookie (the
    token gate lets it and its stylesheet pass), so its "Not logged in" entry reads when the cookie is stale."""
    c, _ = client
    for logged_in in (False, True):
        if logged_in:
            login(c)
        r = c.get("/help.html")
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
        assert r.headers["cache-control"] == "no-cache" and r.text == (server.STATIC / "help.html").read_text()
        assert 'href="/static/style.css"' in r.text
        assert "Not logged in. Open the ?token= link the server printed." in r.text



def test_key_later_builds_without_the_key(client, tmp_path):
    """"key_later": true (a request sent while no worker could take the key) queues Start, Build and an
    edited part list without the key; the job asks for it at the first stage that needs it. A key in the
    body wins, and without key_later the old 400 stays."""
    c, cfg = client
    login(c)
    job_id = uploaded(c, tmp_path)
    runner = c.app.app.state.runner
    opus = "sk-ant-test-" + secrets.token_hex(16)
    start = {"mode": "auto", "prompt": "Wooden box", "n_views": 4, "auto_continue": True, "agent": "opus"}
    for bad in (1, "yes", None, [True]):
        r = c.post(f"/api/jobs/{job_id}/start", json={**start, "key_later": bad})
        assert r.status_code == 400 and r.json()["detail"] == "key_later is true or false", bad
    for body in (start, {**start, "key_later": False}, {**start, "key_later": True, "api_key": KEY}):  # KEY is OpenAI's
        assert c.post(f"/api/jobs/{job_id}/start", json=body).status_code == 400, body
    assert jobs.load(cfg.runs / job_id)["status"] == "ready"
    r = c.post(f"/api/jobs/{job_id}/start", json={**start, "key_later": True})
    assert r.status_code == 200 and r.json() == {"status": "queued", "key": server.KEY_LATER}
    job = wait(c, job_id, "review", "failed")
    assert job["status"] == "review" and job["warnings"] == [jobs.NO_KEY_LATER["opus"]] and job["key_ok"] is False
    assert job["stages"]["place"]["status"] == "done" and job["stages"]["agent"]["status"] == "pending"
    assert "key_later" not in job

    r = c.post(f"/api/jobs/{job_id}/continue", json={})
    assert r.status_code == 400 and "holds no Anthropic API key" in r.json()["detail"]
    r = c.post(f"/api/jobs/{job_id}/continue", json={"key_later": True})
    assert r.json() == {"status": "queued", "from": "sam3d", "key": server.KEY_LATER}
    assert wait(c, job_id, "review", "failed")["warnings"] == [jobs.NO_KEY_LATER["opus"]]
    r = c.post(f"/api/jobs/{job_id}/continue", json={"key_later": True, "api_key": opus})  # the key wins
    assert r.json() == {"status": "queued", "from": "sam3d"}
    job = wait(c, job_id, "done", "failed")
    assert job["status"] == "done", job["error"]
    r = c.post(f"/api/jobs/{job_id}/continue", json={"from": "package", "key_later": True})  # no stage needs a key
    assert r.json() == {"status": "queued", "from": "package"}
    wait(c, job_id, "done", "failed")

    # An edited part list with key_later, once the server holds no key for the job.
    art = uploaded(c, tmp_path, name="lid box.mp4")
    r = c.post(f"/api/jobs/{art}/start", json={**ARTICULATED, "agent": "opus", "api_key": opus, "parts_mode": "vlm_edit"})
    assert r.status_code == 200, r.text
    assert wait(c, art, "review", "failed")["parts_review"] is True
    assert runner.forget_key(art) is True
    edited = [{"name": "body"}, {"name": "top lid", "parent": "body", "joint": "revolute"}]
    r = c.post(f"/api/jobs/{art}/parts", json={"parts": edited})
    assert r.status_code == 400 and "holds no Anthropic API key" in r.json()["detail"]
    r = c.post(f"/api/jobs/{art}/parts", json={"parts": edited, "key_later": "yes"})
    assert r.status_code == 400 and r.json()["detail"] == "key_later is true or false"
    r = c.post(f"/api/jobs/{art}/parts", json={"parts": edited, "key_later": True})
    assert r.json() == {"status": "queued", "from": "parts", "key": server.KEY_LATER}
    job = wait(c, art, "review", "failed")
    assert job["warnings"] == [jobs.NO_KEY_LATER["opus"]] and job["stages"]["parts"]["status"] == "done"
    assert job["parts"]["confirmed"] is True and job["stages"]["agent"]["status"] == "pending"
    assert not list((tmp_path / "keys").iterdir())
    assert not any(opus.encode() in p.read_bytes() for p in all_files(tmp_path))


# Bodies the page never sends, as a public user could relay them: the hostile test below, and the contract test
# (test_webapp_contract.py), which sends them through the public site too. n: the job's frame count.
def hostile_start(n, key):
    user = {"articulated": True, "parts_mode": "user"}
    return [{"mode": ["auto"]}, {"mode": None}, {"prompt": 7}, {"prompt": ["box"]}, {"prompt": "x" * 100000},
            {"prompt": "!!!"}, {"n_views": True}, {"n_views": "4"}, {"n_views": 4.0}, {"n_views": 10**30},
            {"mode": "manual", "picks": [True, False]}, {"mode": "manual", "picks": "1,2"}, {"picks": {"0": 1}},
            {"mode": "manual", "picks": list(range(100000))}, {"mode": "manual", "picks": [1] * 100000},
            {"mode": "manual", "picks": [0.0, 1.0]}, {"mode": "manual", "picks": [-1, 2]},
            {"view0": True}, {"view0": "3"}, {"view0": n}, {"view0": 10**6}, {"auto_continue": 1},
            {"auto_continue": "false"}, {"agent": 1}, {"agent": []}, {"agent": {"name": "opus"}}, {"agent": None},
            {"agent": "gpt"}, {"key_later": 1}, {"key_later": "yes"}, {"articulated": 1}, {"articulated": "yes"},
            {**user, "parts_mode": ["user"]}, {**user, "parts": "lid"}, {**user, "parts": [7]},
            {**user, "parts": [{"name": 7}]}, {**user, "parts": [{"name": "body"}] * 10000},
            {**user, "parts": [{"name": "body"}, {"name": "lid", "parent": "body", "count": True}]},
            {"lidar": True}, {"lidar": []}, {"lidar": "on"}, {"lidar": {"topic": 3}},
            {"agent": "astra", "api_key": 7}, {"agent": "astra", "api_key": ["sk-x"]},
            {"agent": "astra", "api_key": {"k": key}}, {"agent": "astra", "api_key": True},
            {"agent": "astra", "api_key": "sk-" + "a" * 600}, {"agent": "astra", "api_key": "sk-a\nb"},
            {"turntable": 1}, {"turntable": "yes"}, {"turntable": None}]


def hostile_overrides(name):
    """name: a pool view of the job."""
    return [{"locked": name}, {"locked": [[name]]}, {"locked": [7]}, {"excluded": ["f99999"]},
            {"locked": ["../" + name]}, {"locked": [name] * 100000}, {"excluded": [name, name]},
            {"locked": [name], "excluded": [name]}, {"instance": ["1"]}, {"instance": {name: 1}},
            {"instance": {name: True}}, {"instance": {name: "1,"}}, {"instance": {name: "9999"}},
            {"instance": {name: "999"}}, {"instance": {"f99999": "1"}}, {"view0": 0}, {"view0": True},
            {"view0": "f99999"}, {"view0": [name]}, {"agent": "gpt"}, {"agent": 1}, {"agent": "astra", "api_key": 7}]


HOSTILE_CONTINUE = [{"from": 1}, {"from": True}, {"from": ["frames"]}, {"from": {"stage": "agent"}}, {"from": "nope"},
                    {"from": "../agent"}, {"agent": []}, {"agent": "gpt"}, {"key_later": "yes"},
                    {"agent": "astra", "api_key": 7}]


def test_hostile_bodies_get_a_reason_never_a_crash(client, tmp_path):
    """Requests the page never sends, as a public user could relay them: bodies that are not JSON objects or
    nest too deeply, wrong types, true for 1, huge lists, names outside the pool, odd headers. Each gets a
    4xx with a reason (never a 500), and nothing starts."""
    c, cfg = client
    login(c)
    job_id = uploaded(c, tmp_path)
    J = cfg.runs / job_id

    def refused(route, body=None, status=400, **kw):
        r = c.post(f"/api/jobs/{job_id}/{route}", **({"json": body} if body is not None else {}), **kw)
        assert r.status_code == status, (route, str(body)[:200], r.status_code, r.text[:300])
        return r.json()["detail"]

    deep = b"[" * 100000 + b"]" * 100000
    for raw in (b"[1, 2]", b'"start"', b"null", b"\xff\xfe\x00", b"{", deep, b'{"a": ' * 50000 + b"1" + b"}" * 50000,
                b"1" * 5000):
        for route in ("start", "overrides", "parts", "continue", "topic"):
            assert refused(route, content=raw) in ("the request body is not JSON", "the request body is not a JSON object")

    start = {"mode": "auto", "prompt": "box", "n_views": 4, "agent": "none"}
    n = len(c.get(f"/api/jobs/{job_id}/frames").json()["times"])
    for wrong in hostile_start(n, KEY):
        refused("start", {**start, **wrong})
    assert jobs.load(J)["status"] == "ready"

    assert c.post(f"/api/jobs/{job_id}/start", json={**start, "auto_continue": False}).status_code == 200
    assert wait(c, job_id, "review", "failed")["status"] == "review"
    name = sorted(c.get(f"/api/jobs/{job_id}/selection").json()["sam3"])[0]
    for wrong in hostile_overrides(name):
        refused("overrides", wrong)
    for wrong in HOSTILE_CONTINUE:
        refused("continue", wrong)
    refused("topic", {"topic": ["/cam"]}, status=409)  # a video has no topic
    refused("parts", {"parts": [{"name": "body"}] * 10000}, status=409)  # the job does not wait for a list
    assert jobs.load(J)["status"] == "review"

    # Headers as raw bytes, through httpx.ASGITransport as a worker sends requests (TestClient re-encodes them):
    # "²" passes str.isdigit but not int(), and 5000 digits pass neither int() nor a seek.
    import anyio
    import httpx

    async def send(method, path, header, content=b""):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=c.app), base_url="http://app",
                                     headers={"Cookie": f"{server.COOKIE}={TOKEN}"}) as a:
            r = await a.request(method, path, content=content, headers=[header])
            return r.status_code, r.text

    for value, status in ((b"\xb2", 411), (b"9" * 5000, 413), (b"-1", 411), (b"1e3", 411)):
        assert anyio.run(send, "PUT", "/api/upload?name=a.mp4", (b"content-length", value), b"x")[0] == status, value
    status, _ = anyio.run(send, "PUT", f"/api/jobs/{job_id}/calibration?name=c.yaml", (b"content-length", b"9" * 5000), b"x")
    assert status == 413
    for last in (b"frames:\xb2", b"frames:" + b"9" * 30, b"frames:-1", b"nope:1", b"frames"):
        status, text = anyio.run(send, "GET", f"/api/jobs/{job_id}/events", (b"last-event-id", last))
        assert status == 200 and "event: end" in text, last  # as a new client
    assert len(c.get("/api/jobs").json()) == 1


def test_the_routes_read_job_json_as_it_is(client, tmp_path):
    """create_job writes every field, so GET /api/jobs/{id} fills none in, and it and /parts take the job's own parts."""
    c, cfg = client
    login(c)
    job_id = uploaded(c, tmp_path)
    J = cfg.runs / job_id
    job = jobs.load(J)
    jobs.save(J, {k: v for k, v in job.items() if k != "articulated"})
    assert "articulated" not in c.get(f"/api/jobs/{job_id}").json()
    jobs.save(J, {k: v for k, v in job.items() if k != "parts"})
    for route in ("", "/parts"):
        with pytest.raises(KeyError, match="parts"):
            c.get(f"/api/jobs/{job_id}{route}")


def test_hostile_calibration_files_are_refused_fast(client, tmp_path):
    """A calibration of nested YAML aliases (2^30 numbers once expanded) or of deep brackets is refused
    within a second, before the server parses it into lists."""
    pytest.importorskip("rosbags")
    from test_lidar_scale import billion_laughs

    c, cfg = client
    login(c)
    bag = write_bag(tmp_path / "src.bag", ["/cam/image_rect/compressed"], n=20)
    job_id = c.put("/api/upload", params={"name": "one.bag"}, content=bag.read_bytes()).json()["id"]
    assert wait(c, job_id, "ready", "failed", "choose_topic")["status"] == "ready"
    deep = "[" * 30000 + "]" * 30000
    t0 = time.time()
    for data, name, why in [(billion_laughs(), "c.yaml", "c.yaml uses YAML anchors or aliases"),
                            (deep, "c.yaml", "c.yaml nests lists and maps"), (deep, "c.json", "c.json nests lists and maps"),
                            ("- " * 20000 + "1", "c.yml", "c.yml nests lists and maps"),
                            (b"\xff\xfe" * 100, "c.yaml", "c.yaml is not a text file"),
                            ("T_cam_lidar: [" + "0.0, " * 12000 + "0.0]", "c.yaml", "T_cam_lidar has 12001 numbers"),
                            ("T_cam_lidar: " + "1" * 5000, "c.yaml", ""), ('{"T_cam_lidar": ' + "1" * 5000 + "}", "c.json", "")]:
        r = c.put(f"/api/jobs/{job_id}/calibration", params={"name": name}, content=data)
        assert r.status_code == 400 and r.json()["detail"].startswith(why), (name, r.status_code, r.text[:200])
    assert time.time() - t0 < 5
    job = c.get(f"/api/jobs/{job_id}").json()
    assert job["calibration"] is None and not (cfg.runs / job_id / "lidar" / "calibration").exists()


# static/app.js in node, with a stand-in DOM in which every element accepts anything and every call
# records its string arguments (so the text that the page would show is in `texts`).
NODE_HARNESS = r"""
const fs = require("fs"), vm = require("vm");
const texts = [], replies = JSON.parse(process.argv[3]);
const el = () => new Proxy(function () {}, {
  get: (t, k) => k === Symbol.iterator ? [][Symbol.iterator] : k in t ? t[k] : (t[k] = el()),
  apply: (t, self, args) => { texts.push(...args.filter((a) => typeof a === "string")); return el(); },
});
const location = { hash: "" }, history = { state: null, replaceState(state, title, url) { this.state = state; location.hash = url; } };
const ctx = vm.createContext({ console, texts, document: el(), window: el(), location, history,
  setInterval() {}, EventSource: el(), fetch: async (url) => ({ ok: true, status: 200, json: async () => replies[url] }) });
Promise.resolve(vm.runInContext(fs.readFileSync(process.argv[2], "utf8") + "\n" + process.argv[4], ctx))
  .catch((e) => { console.error(e.stack); process.exitCode = 1; });
"""


def run_app_js(replies, code):
    """Load app.js, then run code (an expression that may be a promise) in its scope; returns the JSON
    that code printed last."""
    if shutil.which("node") is None:
        pytest.skip("no node")
    p = subprocess.run(["node", "-e", NODE_HARNESS, "-", str(server.STATIC / "app.js"), json.dumps(replies),
                        textwrap.dedent(code)], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout.splitlines()[-1])


# The start form of static/app.js in node: every selector gives one stand-in element that keeps what
# app.js sets on it (so the form's handlers stay), the agent, mode and parts mode radios are plain
# objects, calls record their string arguments in `texts`, and fetch records every request's URL and
# method in `requests` and the POST bodies in `posts`, and answers with the replies (by URL). A reply
# with "__status" answers that HTTP error, with the reply as its JSON body. querySelectorAll answers
# from `lists` (by selector). `history` keeps its state; its replaceState and pushState calls go to
# `historyCalls` ([kind, url]), and assignments to location.hash go to `hashSets`. TextDecoder is node's (readCloud).
FORM_HARNESS = r"""
const fs = require("fs"), vm = require("vm");
const texts = [], replies = JSON.parse(process.argv[4] || "{}");
const el = () => new Proxy(function () {}, {
  get: (t, k) => k === Symbol.iterator ? [][Symbol.iterator] : k in t ? t[k] : (t[k] = el()),
  apply: (t, self, args) => { texts.push(...args.filter((a) => typeof a === "string")); return el(); },
});
const radios = (values) => values.map((value, i) => ({ value, checked: i === 0 }));
const agents = radios(["astra", "opus", "none"]), modes = radios(["auto", "manual"]);
const partsModes = radios(["vlm", "vlm_edit", "user"]), byId = {}, posts = [], requests = [];
const lists = { "input[name=agent]": agents, "input[name=mode]": modes, "input[name=partsmode]": partsModes };
const document = new Proxy({}, { get: (t, k) => k === "querySelector"
  ? (sel) => sel === "input[name=mode]:checked" ? modes.find((r) => r.checked) : (byId[sel] = byId[sel] || el())
  : k === "querySelectorAll" ? (sel) => lists[sel] || []
  : (t[k] = t[k] || el()) });
const fetch = async (url, opt) => {
  const method = (opt && opt.method) || "GET";
  requests.push({ url, method });
  if (method === "POST") posts.push({ url, body: JSON.parse(opt.body) });
  const reply = url in replies ? replies[url] : url === "/api/jobs" ? [] : {}, status = (reply && reply.__status) || 200;
  return { ok: status < 400, status, statusText: "", json: async () => reply };
};
const hashSets = [], historyCalls = [];
const location = { _hash: "", get hash() { return this._hash; }, set hash(v) { hashSets.push(v); this._hash = v; } };
const history = { state: null,
  replaceState(state, title, url) { historyCalls.push(["replace", url]); this.state = state; location._hash = url; },
  pushState(state, title, url) { historyCalls.push(["push", url]); this.state = state; location._hash = url; } };
const ctx = vm.createContext({ console, document, window: el(), location, history, setInterval() {}, EventSource: el(),
  fetch, agents, partsModes, posts, requests, texts, lists, hashSets, historyCalls, TextDecoder });
Promise.resolve(vm.runInContext(fs.readFileSync(process.argv[2], "utf8") + "\n" + process.argv[3], ctx))
  .catch((e) => { console.error(e.stack); process.exitCode = 1; });
"""


def run_form_js(code, replies=None):
    """Load app.js in FORM_HARNESS, then run code in its scope; returns the JSON that code printed last."""
    if shutil.which("node") is None:
        pytest.skip("no node")
    p = subprocess.run(["node", "-e", FORM_HARNESS, "-", str(server.STATIC / "app.js"), textwrap.dedent(code),
                        json.dumps(replies or {})], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout.splitlines()[-1])


def test_app_js_start_form_follows_the_agent():
    """The key field names the chosen agent's provider and its key prefix; with no agent it is hidden,
    not required, and no key is sent."""
    code = """
        (async () => {
          const form = [], sent = [];
          for (const [i, a] of agents.entries()) {
            agents.forEach((r, j) => { r.checked = j === i; });
            a.onchange();
            form.push([a.value, $("#keyname").textContent, $("#key").placeholder, $("#keylabel").hidden, $("#key").required]);
            V = { id: "j", picks: [], view0: null };
            Object.assign($("#key"), { value: "sk-typed" });
            Object.assign($("#prompt"), { value: "box" });
            Object.assign($("#nviews"), { value: "4" });
            Object.assign($("#pause"), { checked: true });
            posts.length = 0;
            await $("#startform").onsubmit({ preventDefault() {} });
            sent.push(posts[0]);
          }
          console.log(JSON.stringify({ form, sent }));
        })()
        """
    out = run_form_js(code)
    assert out["form"] == [["astra", "OpenAI API key", "sk-...", False, True],
                           ["opus", "Anthropic API key", "sk-ant-...", False, True],
                           ["none", "", "", True, False]]
    assert [s["url"] for s in out["sent"]] == ["/api/jobs/j/start"] * 3
    assert [(s["body"]["agent"], s["body"]["api_key"]) for s in out["sent"]] == \
        [("astra", "sk-typed"), ("opus", "sk-typed"), ("none", None)]
    assert all(s["body"]["mode"] == "auto" and s["body"]["prompt"] == "box" and s["body"]["auto_continue"] is False
               for s in out["sent"])


def test_app_js_view0_and_delete():
    times = [round(0.1 * i, 1) for i in range(50)]
    meta = {"times": times, "rel": [2.0 if i == 26 else 1.0 for i in range(50)], "candidates": [], "thumbs": [],
            "thumb_size": [32, 18], "video": {"n_frames": 50, "duration_s": 5.0, "fps": 10, "width": 64, "height": 36,
                                              "codec": "h264", "hdr": None}}
    rows = [{"id": "a", "created": "2026-10-15T10:00:00", "status": "uploading", "prompt": None, "video": "a.mp4"},
            {"id": "b", "created": "2026-10-15T09:00:00", "status": "done", "prompt": "cup", "video": "b.mp4"}]
    out = run_app_js({"/api/jobs": rows, "/api/jobs/j/frames": meta}, """
        (async () => {
          const view0 = [];
          for (const v of [25, "f00026", "f00045", "f00099", null]) {  // a pick, two pool names, a non-pick, none
            V = { id: "j", picks: [], view0: null, job: { mode: "manual", picks: [5, 25, 45], view0: v, n_views: 6, parts: { mode: "vlm", list: [] } } };
            await loadFrames();
            view0.push(V.view0);
          }
          texts.length = 0;
          await showList();
          console.log(JSON.stringify({ view0, deletes: texts.filter((t) => t === "Delete").length }));
        })()
        """)
    assert out["view0"] == [25, 25, 45, None, None]  # the pool name of a snapped pick is that pick (25 -> 26)
    assert out["deletes"] == 1  # none for the job that is still uploading


def test_app_js_topics_and_the_agent_off():
    topics = {"topics": [
        {"topic": "/cam_a/image_rect/compressed", "type": "CompressedImage", "format": "jpeg", "count": 300, "rate_hz": 10.0,
         "width": 1280, "height": 1024, "rectified": True, "preview": "frames/topic_previews/0.jpg"},
        {"topic": "/cam_b/image/compressed", "type": "CompressedImage", "format": "jpeg", "count": 300, "rate_hz": 10.0,
         "width": 1280, "height": 1024, "rectified": False, "preview": "frames/topic_previews/1.jpg"},
        {"topic": "/cam_c/image", "type": "Image", "format": "rgb8", "count": 3, "rate_hz": None, "width": 8, "height": 6,
         "rectified": None, "preview": "frames/topic_previews/2.jpg"}],
        "skipped": [{"topic": "/depth", "type": "Image", "why": "16UC1"}]}
    stages = [{"name": n, "status": "done"} for n in ("topics", "frames", "extract")] + [
        {"name": "agent", "status": "skipped", "note": "turned off: physics uses the Stage A mesh"}]
    job = {"source": {"kind": "rosbag", "topic": None}, "agent": "none", "agent_result": None, "files": ["run/posed.glb"],
           "stages": {"frames": {"status": "pending"}, "agent": stages[-1]}}
    out = run_app_js({"/api/jobs": [], "/api/jobs/j/file/frames/topics.json": topics}, """
        (async () => {
          V = { id: "j" };
          await loadTopics(""" + json.dumps(job) + """);
          const table = [...texts];
          texts.length = 0;
          renderState({ status: "done", stages: """ + json.dumps(stages) + """, agent: "none" });
          const state = [...texts];
          texts.length = 0;
          await renderResults(""" + json.dumps(job) + """);
          console.log(JSON.stringify({ table, state, results: [...texts] }));
        })()
        """)
    assert {"rectified", "raw", "unknown", "/cam_a", "/image_rect", "/compressed", "CompressedImage jpeg", "1280x1024",
            "10.0 Hz", "Use", "Use /cam_b/image/compressed"} <= set(out["table"])
    assert not any("MV-SAM3D" in t or "camera_info" in t for t in out["table"])  # why a rectified topic is better: help.html
    assert out["table"].count("Use") == 3 and "Not usable: /depth (16UC1)" not in out["table"]  # textContent
    # A stage that is turned off shows the plain "skipped" badge; its note says why (a longer badge broke the row on a phone).
    assert "skipped" in out["state"] and "turned off: physics uses the Stage A mesh" in out["state"]
    assert "skipped (turned off)" not in out["state"]
    # Once, in the facts' Agent row: the viewers have no placeholder that says it again.
    assert out["results"].count("off") == 1


def test_app_js_shows_the_opus_agent_and_its_cost():
    astra = {"agent": "codex", "model": "gpt-6-astra", "source": "finish", "verdict": "complete", "reason": "fixed the lid",
             "limitations": [], "exit": "exited"}
    job = {"source": {"kind": "video"}, "agent": "opus", "files": ["run/posed.glb", "refined.glb"],
           "stages": {"frames": {"status": "done"}, "agent": {"status": "done"}},
           "agent_result": {"source": "finish", "verdict": "complete", "reason": "fixed the lid", "limitations": [],
                            "exit": "exited", "agent": "claude", "model": "claude-opus-5-5", "turns": 12, "cost_usd": 0.672,
                            "tokens": {"input_tokens": 41000, "output_tokens": 9000, "cache_read_input_tokens": 380000,
                                       "cache_creation_input_tokens": 52000}}}
    out = run_app_js({"/api/jobs": []}, """
        (async () => {
          V = { id: "j" };
          await renderResults(""" + json.dumps(job) + """);
          const opus = [...texts];
          texts.length = 0;
          await renderResults(""" + json.dumps({**job, "agent": "astra", "agent_result": astra}) + """);
          console.log(JSON.stringify({ opus, astra: [...texts] }));
        })()
        """)
    assert "Refined (Claude Opus 5.5)" in out["opus"] and "Claude Opus 5.5 (claude-opus-5-5)" in out["opus"]
    assert "41,000 input, 9,000 output, 380,000 cache read, 52,000 cache write in 12 turns" in out["opus"]
    assert "about $0.67" in out["opus"] and "fixed the lid" in out["opus"]
    assert "GPT-6-Astra (gpt-6-astra)" in out["astra"] and "Agent tokens" not in out["astra"]


def test_app_js_parse_parts():
    """parseParts reads "name > parent : joint" lines (and "name x3" for a count); formatParts writes them back."""
    out = run_app_js({"/api/jobs": []}, r"""
        (() => {
          const parts = parseParts("cabinet body\n\n  door > cabinet body : Revolute \r\ndrawer x3 > cabinet body : prismatic\nknob × 2 > door");
          const errors = ["door > : revolute", "door > body : hinge", "> body", "a > b > c", "body\nlid > body :"].map((t) => {
            try { parseParts(t); return null; } catch (e) { return e.message; }
          });
          console.log(JSON.stringify({ parts, text: formatParts(parts), again: parseParts(formatParts(parts)), errors,
            empty: parseParts(" \n"), box: parseParts("box x") }));
        })()
        """)
    assert out["parts"] == [{"name": "cabinet body", "parent": None, "joint": None, "count": 1},
                            {"name": "door", "parent": "cabinet body", "joint": "revolute", "count": 1},
                            {"name": "drawer", "parent": "cabinet body", "joint": "prismatic", "count": 3},
                            {"name": "knob", "parent": "door", "joint": None, "count": 2}]
    assert out["text"] == ("cabinet body\ndoor > cabinet body : revolute\ndrawer x3 > cabinet body : prismatic\n"
                           "knob x2 > door : unknown")
    assert out["again"][:3] == out["parts"][:3] and out["again"][3]["joint"] == "unknown"
    assert out["errors"] == ["line 1: write the parent's name after >",
                             "line 1: the joint is one of revolute, prismatic, fixed, soft, unknown",
                             "line 1: the part needs a name", "line 1: write the part as name > parent : joint",
                             "line 2: the joint is one of revolute, prismatic, fixed, soft, unknown"]
    assert out["empty"] == [] and out["box"] == [{"name": "box x", "parent": None, "joint": None, "count": 1}]


def test_app_js_start_form_sends_the_parts():
    """The default start body is a rigid object (articulated false, no parts). The toggle shows who lists
    the parts; "I type the parts" sends the parsed list; without an agent only "I type the parts" is left,
    and the user's own choice comes back with the agent."""
    out = run_form_js("""
        (async () => {
          const sent = [], form = [];
          const state = () => ({ box: $("#partsbox").hidden, list: $("#partlistbox").hidden,
            noagent: $("#partsnoagent").hidden, required: $("#partlist").required,
            checked: partsModes.filter((r) => r.checked).map((r) => r.value), disabled: partsModes.map((r) => r.disabled) });
          const pick = (i) => { agents.forEach((r, j) => { r.checked = j === i; }); agents[i].onchange(); };
          const submit = async () => {
            V = { id: "j", picks: [], view0: null };
            Object.assign($("#key"), { value: "sk-typed" });
            Object.assign($("#prompt"), { value: "box" });
            Object.assign($("#nviews"), { value: "4" });
            Object.assign($("#pause"), { checked: true });
            posts.length = 0;
            await $("#startform").onsubmit({ preventDefault() {} });
            sent.push(posts.length ? posts[0].body : $("#startmsg").textContent);
          };
          pick(0);
          await submit();
          $("#articulated").checked = true;
          $("#articulated").onchange();
          form.push(state());
          partsModes[1].onchange();
          form.push(state());
          await submit();
          pick(2);
          form.push(state());
          pick(0);
          form.push(state());
          partsModes[2].onchange();
          $("#partlist").value = "box body\\nlid > box body : revolute\\ndrawer x2 > box body : prismatic";
          await submit();
          $("#partlist").value = "box body\\nlid > box body : hinge";
          await submit();
          pick(2);
          $("#partlist").value = "box body\\nlid > box body";
          await submit();
          console.log(JSON.stringify({ form, sent }));
        })()
        """)
    rigid, vlm_edit, user, bad, none = out["sent"]
    assert rigid["articulated"] is False and "parts_mode" not in rigid and "parts" not in rigid
    assert out["form"] == [
        {"box": False, "list": True, "noagent": True, "required": False, "checked": ["vlm"], "disabled": [False, False, False]},
        {"box": False, "list": True, "noagent": True, "required": False, "checked": ["vlm_edit"],
         "disabled": [False, False, False]},
        {"box": False, "list": False, "noagent": False, "required": True, "checked": ["user"], "disabled": [True, True, False]},
        {"box": False, "list": True, "noagent": True, "required": False, "checked": ["vlm_edit"],
         "disabled": [False, False, False]}]
    assert (vlm_edit["articulated"], vlm_edit["parts_mode"], vlm_edit["parts"], vlm_edit["api_key"]) == (True, "vlm_edit", [], "sk-typed")
    assert (user["parts_mode"], user["agent"]) == ("user", "astra") and user["parts"] == [
        {"name": "box body", "parent": None, "joint": None, "count": 1},
        {"name": "lid", "parent": "box body", "joint": "revolute", "count": 1},
        {"name": "drawer", "parent": "box body", "joint": "prismatic", "count": 2}]
    assert bad == "line 2: the joint is one of revolute, prismatic, fixed, soft, unknown"
    assert (none["agent"], none["api_key"], none["parts_mode"]) == ("none", None, "user")
    assert none["parts"] == [{"name": "box body", "parent": None, "joint": None, "count": 1},
                             {"name": "lid", "parent": "box body", "joint": None, "count": 1}]


def test_app_js_360_video_setting():
    """The Settings box "Make a 360 video" shows the job's choice (job.turntable; off unless it is true), and
    Start sends it as true or false. The object name is the job's, and empty for a job without one (not the name of the
    job opened before)."""
    meta = frames_meta()
    out = run_form_js("""
        (async () => {
          const ticked = [], sent = [], names = [];
          for (const [t, prompt] of [[true, "Wooden box"], [false, null]]) {
            V = { id: "j", picks: [], view0: null, job: { id: "j", kind: "video", status: "ready", mode: "auto", picks: [],
              n_views: 4, agent: "none", lidar: null, calibration: null, turntable: t, prompt, parts: { mode: "vlm", list: [] } } };
            $("#turntable").checked = "stale";
            await loadFrames();
            ticked.push($("#turntable").checked);
            names.push($("#prompt").value);
          }
          agents.forEach((r, j) => { r.checked = j === 2; });
          for (const on of [true, false]) {
            V = { id: "j", picks: [], view0: null };
            Object.assign($("#prompt"), { value: "box" });
            Object.assign($("#nviews"), { value: "4" });
            $("#turntable").checked = on;
            posts.length = 0;
            await $("#startform").onsubmit({ preventDefault() {} });
            sent.push(posts[0].body.turntable);
          }
          console.log(JSON.stringify({ ticked, sent, names }));
        })()
        """, {"/api/jobs/j/frames": meta})
    assert out == {"ticked": [True, False], "sent": [True, False], "names": ["Wooden box", ""]}


def test_app_js_part_list_panel():
    """The panel of a job that waits for its part list: the VLM's list as text, its notes and view 0's
    photo. Build sends the list as it is through continue, an edited list through POST /parts (a kept part
    keeps the VLM's axis hint), and a typing error stays on the page."""
    proposal = {"source": "vlm", "agent": "opus", "model": "claude-opus-5-5", "base": "body", "warnings": ["a warning"],
                "parts": [{"name": "body", "parent": None, "joint": "fixed", "count": 1, "axis_hint": None},
                          {"name": "lid", "parent": "body", "joint": "revolute", "count": 1,
                           "axis_hint": "a hinge along the back top edge"}]}
    replies = {"/api/jobs/j/parts": {"articulated": True, "noun": "wooden box", "waiting": True, "proposal": proposal,
                                     "parts": {"mode": "vlm_edit", "list": [], "confirmed": False}, "checks": [],
                                     "photo": "scene/sam3_wooden_box/check/f00012.jpg"}}
    out = run_form_js("""
        (async () => {
          V = { id: "j" };
          await loadParts();
          const shown = { texts: [...texts], text: $("#partsedit").value, fig: $("#partsfig").hidden, photo: $("#partsphoto").src,
            reset: $("#partsreset").hidden };
          const stages = ["frames", ...BUILD].map((name) => ({ name, status: name === "propose" || BUILD.indexOf(name) < 7 ? "done" : "pending" }));
          renderState({ status: "review", stages, agent: "opus", key_ok: true, warnings: [], articulated: true, parts_mode: "vlm_edit",
            parts_review: true });
          const next = $("#jnext").textContent;
          $("#key2").value = "";
          posts.length = 0;
          await $("#partsbuild").onclick();
          $("#partsedit").value = shown.text + "\\nflap > body : prismatic";
          $("#partsedit").oninput();
          const reset = $("#partsreset").hidden;
          await $("#partsbuild").onclick();
          $("#partsedit").value = "body\\nlid > body : hinge";
          await $("#partsbuild").onclick();
          const error = $("#partsmsg").textContent, sent = [...posts];
          // A VLM's name may hold ":" or ">": the list as shown is confirmed without parsing it here.
          V.parts = { proposal: { parts: [{ name: "body", parent: null, joint: "fixed", count: 1 },
            { name: "door: left", parent: "body", joint: "revolute", count: 1 }] } };
          V.partsText = formatParts(V.parts.proposal.parts);
          $("#partsedit").value = V.partsText + "\\n";
          posts.length = 0;
          await $("#partsbuild").onclick();
          $("#partsedit").value = "body\\nleft door > body : revolute";
          await $("#partsbuild").onclick();
          const labels = [stageLabel("agent", true), stageLabel("agent", false), stageLabel("propose"), stageLabel("parts"),
            stageLabel("urdf")];
          V = { id: "j" };  // act's refresh left a stand-in job
          renderState({ status: "running", current: "agent", stages, agent: "opus", key_ok: true, warnings: [], articulated: true });
          renderState({ status: "running", current: "sam3d", stages, agent: "opus", key_ok: true, warnings: [], articulated: true });
          console.log(JSON.stringify({ shown, next, reset, posts: sent, colon: posts, error, labels, build: BUILD,
            running: texts.filter((t) => t.startsWith("Running")) }));
        })()
        """, replies)
    shown = out["shown"]
    assert "Claude Opus 5.5: 1 moving part. Base: body." in shown["texts"]
    assert {"a warning", "lid", ": a hinge along the back top edge"} <= set(shown["texts"])
    assert shown["text"] == "body\nlid > body : revolute" and shown["reset"] is True and shown["fig"] is False
    assert shown["photo"] == "/api/jobs/j/file/scene/sam3_wooden_box/check/f00012.jpg"
    assert out["next"] == "Check the part list, then Build."
    assert out["reset"] is False
    assert out["posts"] == [
        {"url": "/api/jobs/j/continue", "body": {"api_key": None}},
        {"url": "/api/jobs/j/parts", "body": {"api_key": None, "parts": [
            {"name": "body", "parent": None, "joint": None, "count": 1},
            {"name": "lid", "parent": "body", "joint": "revolute", "count": 1, "axis_hint": "a hinge along the back top edge"},
            {"name": "flap", "parent": "body", "joint": "prismatic", "count": 1}]}}]
    assert out["error"] == "line 2: the joint is one of revolute, prismatic, fixed, soft, unknown"
    assert out["colon"] == [{"url": "/api/jobs/j/continue", "body": {"api_key": None}},
                            {"url": "/api/jobs/j/parts", "body": {"api_key": None, "parts": [
                                {"name": "body", "parent": None, "joint": None, "count": 1},
                                {"name": "left door", "parent": "body", "joint": "revolute", "count": 1}]}}]
    assert out["labels"] == ["agent sets the joints", "agent refinement", "part list", "split into parts", "URDF"]
    assert out["running"] == ["Running: agent sets the joints", "Running: mesh (MV-SAM3D)"]
    assert out["build"] == steps.NAMES[steps.NAMES.index("extract"):] == jobs.BUILD  # lidar after da3


def test_app_js_part_list_says_when_the_scale_is_stale():
    """A job that waits for its part list while its scale is stale (lidar_current false, here because an older
    estimator made the record): the server refuses to confirm the list (409), so the panel's first warning says so
    and what to do, by the cause (lidar_stale; a Stray capture's is the estimator), and so does the line under the
    status. Before, the page said nothing until Build was refused. A current scale, or a job without LiDAR, adds no
    line."""
    proposal = {"source": "vlm", "agent": "opus", "base": "body", "warnings": ["a warning"],
                "parts": [{"name": "body", "parent": None, "joint": "fixed", "count": 1, "axis_hint": None},
                          {"name": "lid", "parent": "body", "joint": "revolute", "count": 1, "axis_hint": None}]}
    replies = {"/api/jobs/j/parts": {"articulated": True, "noun": "wooden box", "waiting": True, "proposal": proposal,
                                     "parts": {"mode": "vlm_edit", "list": [], "confirmed": False}, "checks": [], "photo": None}}
    bag = {"kind": "rosbag", "lidar": {"kind": "rosbag", "topic": "/lidar/points"}}
    jobs_ = [{**bag, "lidar_current": False, "lidar_stale": "estimator"},
             {"kind": "stray", "lidar": {"kind": "stray"}, "lidar_current": False, "lidar_stale": "estimator"},
             {**bag, "lidar_current": False, "lidar_stale": "calibration"},
             {**bag, "lidar_current": True, "lidar_stale": None},
             {"kind": "video", "lidar": None, "lidar_current": None, "lidar_stale": None}]
    out = run_form_js(f"""
        (async () => {{
          const lines = Object.values(SCALE_STALE).map((x) => x.parts), shown = [];
          for (const job of {json.dumps(jobs_)}) {{
            V = {{ id: "j", job: {{ ...job, video: {{ name: "box" }} }} }};
            texts.length = 0;
            await loadParts();
            const stages = ["frames", ...BUILD].map((name) => ({{ name, status: BUILD.indexOf(name) < 7 ? "done" : "pending" }}));
            renderState({{ status: "review", stages, agent: "opus", key_ok: true, warnings: [], articulated: true,
              parts_mode: "vlm_edit", parts_review: true }});
            shown.push([texts.filter((t) => lines.includes(t)), texts.includes("a warning"), $("#jnext").textContent]);
          }}
          console.log(JSON.stringify(shown));
        }})()
        """, replies)
    todo = " so Build cannot confirm this list. Cancel the job, then press Retry from lidar. The job finds the scale " \
           "again, and the VLM lists the parts again."
    new_way = "The app now finds the scale in a new way," + todo
    stale, build = "The scale is out of date: cancel the job, then Retry from lidar.", "Check the part list, then Build."
    assert out == [[[new_way], True, stale], [[new_way], True, stale],
                   [["The calibration changed after the scale was found," + todo], True, stale], [[], True, build], [[], True, build]]


def test_app_js_results_of_an_articulated_job():
    """The split parts and the agent's parts with joints side by side, the links, base and joints of the MJCF,
    the agent's joint count, the mass of every link and the parts the split missed; the new files are in the
    Downloads tab, not in Results. Without the agent the parts are welded into one body. A rigid job's part
    stages show as turned off."""
    mjcf = {"name": "wooden_box", "root": "body", "fixed_base": False, "welded": {},
            "inertial": {"mass": 2.5, "com": [-1.51e-18, 0.01, 0.1], "fullinertia": [0.01] * 6},  # float noise in x
            "links": {"body": {"source": "vomp", "inertial": {"mass": 2.0, "com": [0, 0, 0.1], "fullinertia": [0.01] * 6}},
                      "lid": {"source": "hull", "inertial": {"mass": 0.5, "com": [0, 0, 0.2], "fullinertia": [0.001] * 6}}},
            "joints": [{"child": "lid", "parent": "body", "type": "revolute", "pos": [0, 0.1, 0.2], "axis": [-1, 0, 0],
                        "range": [0.0, 1.5]}]}
    summary = {"rigid_body": {"mass": 2.0, "com": [0, 0, 0.1]}, "median": {"density_kg_m3": 600.0},
               "links_skipped": {"lid": "no occupied voxel"}, "rigid_body_covers": ["body"]}
    parts = {"parts": [{"name": "body", "status": "found"}, {"name": "lid", "status": "found"},
                       {"name": None, "prompt": "drawer", "status": "not_found", "why": "no SAM 3 mask in any view"}]}
    files = ["run/posed.glb", "run/parts.glb", "run/parts.json", "refined.glb", "physics/summary.json",
             "physics/mjcf/mjcf.json", "physics/urdf/model.urdf", "results.zip"]
    job = {"id": "j", "articulated": True, "agent": "opus", "files": files, "stages": {"agent": {"status": "done"}},
           "agent_result": {"source": "finish", "verdict": "articulated", "reason": "the lid hinges", "limitations": [],
                            "exit": "exited", "agent": "claude", "model": "claude-opus-5-5", "mode": "articulation",
                            "counts": {"exports": 1, "reviews": 0, "checkpoints": 0, "joints": 2, "articulation_reviews": 1},
                            "joints_delivered": 1}}
    off = {"id": "k", "articulated": True, "agent": "none", "files": ["run/posed.glb", "run/parts.glb", "physics/summary.json",
                                                                       "physics/mjcf/mjcf.json"], "agent_result": None,
           "stages": {"agent": {"status": "skipped", "note": "turned off: without the agent the parts get no joints, "
                                                              "so physics welds them into one body"}}}
    replies = {"/api/jobs": [], "/api/jobs/j/file/physics/mjcf/mjcf.json": mjcf, "/api/jobs/j/file/physics/summary.json": summary,
               "/api/jobs/j/file/run/parts.json": parts,
               "/api/jobs/j/parts": {"checks": ["run/parts/sam3/check/0.jpg", "run/parts/sam3/check/1.jpg"]},
               "/api/jobs/k/file/physics/mjcf/mjcf.json": {"name": "wooden_box", "inertial": mjcf["inertial"]},
               "/api/jobs/k/file/physics/summary.json": summary}
    unchanged = {**job, "agent_result": {**job["agent_result"], "verdict": "unchanged", "joints_delivered": 0}}
    no_counts = {**job, "agent_result": {k: v for k, v in job["agent_result"].items() if k != "counts"}}
    stages = [{"name": "frames", "status": "done"}] + [
        {"name": n, "status": "skipped", "note": "turned off: rigid object"} for n in ("propose", "parts")] + [
        {"name": "agent", "status": "done"}]
    out = run_app_js(replies, """
        (async () => {
          V = { id: "j" };
          await renderResults(""" + json.dumps(job) + """);
          const art = [...texts];
          texts.length = 0;
          renderDownloads(""" + json.dumps(job) + """);
          const dl = [...texts];
          texts.length = 0;
          V = { id: "k" };
          await renderResults(""" + json.dumps(off) + """);
          const none = [...texts];
          texts.length = 0;
          renderState({ status: "done", stages: """ + json.dumps(stages) + """, agent: "astra" });
          const state = [...texts];
          texts.length = 0;
          V = { id: "j" };  // the agent kept the split parts as they were: no joints
          await renderResults(""" + json.dumps(unchanged) + """);
          const same = [...texts];
          const read = await renderResults(""" + json.dumps(no_counts) + """).then(() => "read", (e) => e.name);
          console.log(JSON.stringify({ art, dl, none, state, unchanged: same, no_counts: read }));
        })()
        """)
    assert out["no_counts"] == "TypeError"  # controller.run_session always writes counts
    art = out["art"]
    for t in ("Split parts", "Parts with joints (Claude Opus 5.5)", "articulated (finish)",
              "Joints", "1", "Joint changes", "2", "Links", "2: body, lid", "Base", "body, free to move",
              "Joint of lid", "turns on body, 0° to 86°", "Mass", "2.50 kg", "Center of mass", "0.000, 0.010, 0.100 m",
              "Density (median)", "600 kg/m³", "Link masses",
              "body 2.00 kg, lid 0.500 kg (from hulls)", "Parts not split off", "drawer (no SAM 3 mask in any view)",
              "Part masks (2)", "One color per part."):
        assert t in art, t
    assert "Mass note" not in art  # the MJCF's mass covers the link that VoMP skipped
    assert not {"parts.glb", "model.urdf", "results.zip"} & set(art)  # no downloads in Results
    assert {"parts.glb", "parts.json", "model.urdf", "results.zip", "Meshes", "Simulation", "Records"} <= set(out["dl"])
    assert out["none"].count("off (parts welded into one body)") == 1 and "1 (no joints, one rigid body)" in out["none"]
    assert "Mass note" in out["none"] and "Leaves out lid (no VoMP result)" in out["none"]
    assert out["state"].count("turned off: rigid object") == 2 and out["state"].count("skipped") == 2
    assert "agent refinement" in out["state"]
    assert "Agent result, no joints (Claude Opus 5.5)" in out["unchanged"]
    assert not any(t.startswith("Parts with joints") for t in out["unchanged"])


# ---- the LiDAR UI (build_plan.md note 12): upload kinds, the LiDAR box, the review, the failure, the results ----

CAM = "/cam/image_rect/compressed"
LIVOX = {"topic": "/livox/lidar", "type": "CustomMsg", "msgtype": "livox_ros_driver2/msg/CustomMsg", "frame_id": "livox_frame",
         "points": 19968, "fields": ["x", "y", "z", "reflectivity", "tag", "line", "offset_time"], "point_time": True,
         "count": 120, "rate_hz": 10.0}
CLOUD = {"topic": "/lidar/points", "type": "PointCloud2", "msgtype": "sensor_msgs/msg/PointCloud2", "frame_id": "lidar",
         "points": 4000, "fields": ["x", "y", "z", "intensity"], "point_time": False, "count": 118, "rate_hz": 9.87}
EYE = np.eye(4).tolist()


def camera_entry(links):
    return {"topic": CAM, "type": "CompressedImage", "format": "jpeg", "count": 48, "rate_hz": 10.0, "width": 192,
            "height": 128, "rectified": True, "frame_id": "camera_optical_frame", "camera_axes": "optical",
            "camera_axes_from": "name", "tf_links": links}


def frames_meta(**video):
    """frames/meta.json of 20 frames; video adds or replaces fields of "video"."""
    return {"times": [round(0.1 * i, 1) for i in range(20)], "rel": [1.0] * 20, "candidates": [3, 9, 15], "thumbs": [3, 9, 15],
            "thumb_size": [32, 18], "video": {"n_frames": 20, "duration_s": 2.0, "fps": 10, "width": 192, "height": 128,
                                              "codec": "h264", "bit_depth": 8, "hdr": None, **video}}


def test_app_js_takes_no_older_form():
    """A listing without "lidar" is a bag without LiDAR topics, and its topic in use says so. A camera topic without
    camera_axes_from and without a link needs a calibration. An agent of false is not "none"."""
    found = {"topics": [{k: v for k, v in camera_entry({"/lidar/points": None}).items() if k != "camera_axes_from"}],
             "skipped": []}
    job = {"id": "j", "kind": "rosbag", "status": "ready", "source": {"kind": "rosbag", "topic": CAM}, "agent": False,
           "agent_result": None, "files": ["run/posed.glb"],
           "stages": {"frames": {"status": "done"}, "agent": {"status": "skipped", "note": "the mesh was poor"}}}
    out = run_form_js("""
        (async () => {
          const found = """ + json.dumps(found) + """, job = """ + json.dumps(job) + """, cam = """ + json.dumps(CAM) + """;
          V = { id: "j", job, topics: found };
          lidarToggle();
          const hint = $("#lidarhint").textContent, link = lidarLink(found, cam, "/lidar/points", null);
          await loadTopics(job);
          const table = texts.filter((t) => t === "Use" || t === "in use");
          texts.length = 0;
          await renderResults(job);
          console.log(JSON.stringify({ hint, link, table, agent: texts.includes("skipped: the mesh was poor") }));
        })()
        """, {"/api/jobs/j/file/frames/topics.json": found})
    assert out == {"hint": "No LiDAR topic in this bag.", "link": ["No TF link: upload a calibration", True],
                   "table": ["in use"], "agent": True}


def test_app_js_calibration_summary_says_where_k_comes_from():
    """The summary names the source of K: the file, or the camera topic's camera_info. When the bag has no
    camera_info for the camera (topics.json camera_info null), a calibration without K is not enough, and the
    summary says so. Without a camera topic in the listing (undefined), the summary keeps the first text."""
    cal = {"name": "c.yaml", "translation_m": 0.1, "rotation_deg": 90, "has_intrinsics": False, "camera_axes": "optical",
           "time_offset_s": None}
    out = run_form_js("""
        (async () => {
          const cal = """ + json.dumps(cal) + """;
          console.log(JSON.stringify([calText(cal, "/cam/camera_info"), calText(cal, null), calText(cal, undefined),
                                      calText({ ...cal, has_intrinsics: true }, null)]));
        })()
        """)
    assert out == ["c.yaml: 10.0 cm, 90.0°. K from camera_info.",
                   "c.yaml: 10.0 cm, 90.0°. No K: the bag has no camera_info, so give K in the file.",
                   "c.yaml: 10.0 cm, 90.0°. K from camera_info.", "c.yaml: 10.0 cm, 90.0°. K from file."]

def test_app_js_upload_kinds_and_the_jobs_list():
    """A .zip is a Stray Scanner upload; another suffix is refused (help.html names the kinds). The jobs list names
    each job's kind, and the stage list has the lidar stage after da3."""
    rows = [{"id": "a", "created": "2026-10-21T09:00:00", "status": "ready", "prompt": None, "video": "a.mp4", "kind": "video"},
            {"id": "b", "created": "2026-10-21T08:00:00", "status": "ready", "prompt": None, "video": "b.bag", "kind": "rosbag"},
            {"id": "c", "created": "2026-10-21T07:00:00", "status": "done", "prompt": "box", "video": "c.zip", "kind": "stray"}]
    out = run_form_js("""
        (async () => {
          const kinds = ["clip.MP4", "scan.bag", "run.db3", "my capture.zip", "Capture.ZIP", "notes.txt", "capture.zip.txt"]
            .map((n) => KINDS.test(n));
          const opened = [];
          globalThis.XMLHttpRequest = class { constructor() { this.upload = {}; } open(m, u) { opened.push([m, u]); } send(f) { opened.push(f.name); } };
          upload({ name: "notes.txt", size: 3 });
          const refused = $("#upmsg").textContent;
          upload({ name: "my capture.zip", size: 3e6 });
          texts.length = 0;
          renderJobs(""" + json.dumps(rows) + """);
          console.log(JSON.stringify({ kinds, refused, opened, list: [...texts], label: stageLabel("lidar"), build: BUILD }));
        })()
        """)
    assert out["kinds"] == [True, True, True, True, True, False, False]
    assert out["refused"] == "notes.txt: file type not supported."
    assert out["opened"] == [["PUT", "/api/upload?name=my%20capture.zip"], "my capture.zip"]
    assert {"video", "rosbag", "Stray capture"} <= set(out["list"]) and "stray" not in out["list"]
    assert out["label"] == "metric scale (LiDAR)"
    assert out["build"][out["build"].index("da3") + 1] == "lidar" and out["build"].index("lidar") + 1 == out["build"].index("select")


def test_app_js_lidar_box_per_kind():
    """The LiDAR box of the start form and the start body's "lidar" for each kind of input: hidden for a
    video (null); a bag with one LiDAR topic preselects it, and its TF chain shows; a bag with two asks for
    one and says when TF does not link it; a bag without LiDAR topics has the box turned off with the reason
    (the box's hint says nothing else); a Stray capture starts with LiDAR on, and
    its turn and depth show. A calibration links the topics in place of TF, and the line names it and the TF link
    it replaces; a calibration made for another LiDAR topic is a warning (one without lidar_topic is not checked)."""
    static_tf = [{"parent": "base_link", "child": "camera_link", "static": True, "T_parent_child": EYE},
                 {"parent": "camera_link", "child": "camera_optical_frame", "static": True, "T_parent_child": EYE},
                 {"parent": "base_link", "child": "livox_frame", "static": True, "T_parent_child": EYE},
                 {"parent": "odom", "child": "base_link", "static": False, "T_parent_child": EYE}]
    dynamic_tf = [{"parent": "base_link", "child": "camera_optical_frame", "static": True, "T_parent_child": EYE},
                  {"parent": "/base_link", "child": "livox_frame", "static": False, "T_parent_child": EYE}]
    skipped = [{"topic": "/ouster/points", "type": "PointCloud2", "why": "no messages"}]
    topics = {
        "b1": {"topics": [camera_entry({"/livox/lidar": "static"})], "skipped": [], "lidar": [LIVOX], "lidar_skipped": skipped,
               "tf": static_tf},
        "b2": {"topics": [camera_entry({"/lidar/points": None, "/livox/lidar": "dynamic"})], "skipped": [],
               "lidar": [LIVOX, CLOUD], "lidar_skipped": [], "tf": dynamic_tf},
        "b4": {"topics": [camera_entry({})], "skipped": [], "lidar": [], "lidar_skipped": skipped, "tf": []},
        "b5": {"topics": [camera_entry({})], "skipped": [], "lidar": [], "lidar_skipped": [], "tf": []}}
    stray_video = {"kind": "stray", "rotation": 270, "rotation_source": "gravity", "container_rotation": 0, "n_depth": 19,
                   "n_confidence": 19, "n_odometry": 20, "depth_size": [256, 192], "upright_agreement": 0.9583,
                   "warnings": ["1 of the 20 frames have no depth image, so they have no LiDAR samples"]}
    replies = {f"/api/jobs/{i}/frames": frames_meta() for i in ("v", "b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8")}
    replies.update({f"/api/jobs/{i}/file/frames/topics.json": t for i, t in topics.items()})
    replies.update({"/api/jobs/b3/file/frames/topics.json": topics["b2"], "/api/jobs/b6/file/frames/topics.json": topics["b5"],
                    "/api/jobs/b7/file/frames/topics.json": topics["b1"], "/api/jobs/b8/file/frames/topics.json": topics["b2"],
                    "/api/jobs/s1/frames": frames_meta(**stray_video),
                    "/api/jobs/s2/frames": frames_meta(**{**stray_video, "rotation": 0, "upright_agreement": None, "warnings": []})})
    job = {"status": "ready", "mode": "auto", "picks": [], "n_views": 4, "auto_continue": True, "agent": "none",
           "prompt": None, "lidar": None, "calibration": None, "parts": {"mode": "vlm", "list": []}}
    bag = {**job, "kind": "rosbag", "source": {"kind": "rosbag", "topic": CAM}}
    cal = {"name": "c.yaml", "bytes": 300, "translation_m": 0.1, "rotation_deg": 90.0, "has_intrinsics": False,
           "camera_axes": "optical", "time_offset_s": None, "lidar_topic": "/livox/lidar"}
    jobs_ = {"v": {**job, "kind": "video", "source": {"kind": "video", "topic": None}}, "b1": bag, "b2": bag,
             "b3": {**bag, "lidar": {"kind": "rosbag", "topic": "/livox/lidar"}}, "b4": bag, "b5": bag,
             "b6": {**bag, "status": "done"},
             # A calibration made for the Livox (server.calibration_summary), where TF links it too.
             "b7": {**bag, "lidar": {"kind": "rosbag", "topic": "/livox/lidar"}, "calibration": cal},
             "b8": {**bag, "lidar": {"kind": "rosbag", "topic": "/livox/lidar"}, "calibration": cal},
             "s1": {**job, "kind": "stray", "source": {"kind": "stray", "topic": None}, "lidar": {"kind": "stray"}},
             "s2": {**job, "kind": "stray", "source": {"kind": "stray", "topic": None}}}
    out = run_form_js("""
        (async () => {
          const JOBS = """ + json.dumps(jobs_) + """, out = {};
          // The link and the skipped topics only while the topic box shows (a stand-in keeps an old text).
          const shown = (x) => $("#lidaropts").hidden === true ? null : x;
          const box = () => ({ hidden: $("#lidarbox").hidden, on: $("#uselidar").checked, off: $("#uselidar").disabled,
            hint: $("#lidarhint").textContent, opts: $("#lidaropts").hidden, topic: $("#lidartopic").value,
            link: shown($("#lidarlink").textContent),
            warn: shown($("#lidarlink").className === "warn"), skipped: shown($("#lidarskipped").textContent) });
          const submit = async () => {  // the refresh after a Start gets the same job back on the page
            const job = V.job;
            Object.assign($("#prompt"), { value: "box" });
            posts.length = 0;
            await $("#startform").onsubmit({ preventDefault() {} });
            V.job = job;
            return posts.length ? posts[0].body.lidar : $("#startmsg").textContent;
          };
          const open = async (id) => {
            V = { id, picks: [], view0: null, job: JOBS[id] };
            texts.length = 0;
            await loadFrames();
            return [...texts];
          };
          const check = (on) => { $("#uselidar").checked = on; $("#uselidar").onchange(); };
          const label = (t) => t.startsWith("/") && t.endsWith(")");  // an option's text, not its value or a thumbnail
          const choose = (t) => { $("#lidartopic").value = t; $("#lidartopic").onchange(); };
          await open("v");
          out.v = { box: box(), sent: await submit() };
          const b1 = await open("b1");
          out.b1 = { labels: b1.filter(label), box: box(), sent: await submit() };
          check(true);
          out.b1.on = box();
          out.b1.sent_on = await submit();
          const b2 = await open("b2");
          out.b2 = { labels: b2.filter((t) => label(t) || t.startsWith("Choose")), box: box() };
          check(true);
          out.b2.none = box();
          $("#lidartopic").focus = () => { out.b2.focused = true; };
          out.b2.error = await submit();
          choose("/lidar/points");
          out.b2.cloud = box();
          choose("/livox/lidar");
          out.b2.livox = box();
          out.b2.sent = await submit();
          await open("b3");
          out.b3 = box();
          for (const id of ["b4", "b5", "b6"]) {
            await open(id);
            out[id] = { box: box(), sent: await submit() };
          }
          // A turned-off box (b6, open now) sends null even with its checkbox checked.
          $("#uselidar").checked = true;
          out.b6.checked_off = [$("#uselidar").disabled, lidarChoice()];
          const s1 = await open("s1");
          out.s1 = { box: box(), sent: await submit(), facts: $("#strayfacts").textContent, info: $("#strayinfo").hidden,
            warn: s1.filter((t) => t.includes("depth image")), vsum: $("#vsum").textContent };
          check(false);
          out.s1.off = box();
          out.s1.sent_off = await submit();
          await open("s2");
          out.s2 = { box: box(), facts: $("#strayfacts").textContent };
          await open("b1");
          out.b1.stray_hidden = $("#strayinfo").hidden;
          out.topics = requests.filter((r) => r.url.endsWith("topics.json")).map((r) => r.url.split("/")[3]);
          // The camera topic table: the topic in use says so.
          V = { id: "b1" };
          texts.length = 0;
          await loadTopics({ source: JOBS.b1.source, stages: { frames: { status: "done" } } });
          out.table_b1 = texts.filter((t) => t === "Use" || t === "in use");
          await open("b7");
          out.b7 = box();
          await open("b8");
          out.b8 = { livox: box() };
          choose("/lidar/points");
          out.b8.cloud = box();
          V.job.calibration = { ...V.job.calibration, lidar_topic: null };  // one uploaded before a topic
          lidarToggle();
          out.b8.any = box();
          console.log(JSON.stringify(out));
        })()
        """, replies)
    v = out["v"]
    assert v["box"]["hidden"] is True and v["sent"] is None

    b1 = out["b1"]
    assert b1["labels"] == ["/livox/lidar (Livox, 10 Hz, livox_frame)"]
    assert b1["box"] == {"hidden": False, "on": False, "off": False, "hint": "", "opts": True, "topic": "/livox/lidar",
                         "link": None, "warn": None, "skipped": None}
    assert b1["sent"] is None  # off by default for a bag
    assert b1["on"] == {"hidden": False, "on": True, "off": False, "hint": "", "opts": False, "topic": "/livox/lidar", "warn": False,
                        "link": "Linked by /tf_static: camera_optical_frame <- camera_link <- base_link <- livox_frame",
                        "skipped": "Not usable: /ouster/points (no messages)"}
    assert b1["sent_on"] == {"topic": "/livox/lidar"} and b1["stray_hidden"] is True

    b2 = out["b2"]
    assert b2["labels"] == ["Choose a LiDAR topic", "/livox/lidar (Livox, 10 Hz, livox_frame)",
                            "/lidar/points (PointCloud2, 9.9 Hz, lidar)"]
    assert (b2["box"]["topic"], b2["box"]["on"]) == ("", False)  # two topics: none preselected
    assert b2["none"]["link"] == ""  # no topic chosen yet
    assert b2["error"] == "choose a LiDAR topic, or turn LiDAR off" and b2["focused"] is True  # the page's text, not a bubble
    assert (b2["cloud"]["link"], b2["cloud"]["warn"]) == ("No TF link: upload a calibration", True)
    assert (b2["livox"]["link"], b2["livox"]["warn"]) == ("Linked by /tf: camera_optical_frame <- base_link <- livox_frame", False)
    assert b2["sent"] == {"topic": "/livox/lidar"}
    assert (out["b3"]["on"], out["b3"]["topic"], out["b3"]["opts"]) == (True, "/livox/lidar", False)  # the job's own setting

    no_lidar, none, none_done = out["b4"], out["b5"], out["b6"]
    assert no_lidar["box"]["off"] is True and no_lidar["box"]["on"] is False and no_lidar["box"]["opts"] is True
    assert no_lidar["box"]["hint"] == "No LiDAR topic in this bag. Not usable: /ouster/points (no messages)."
    assert none["box"]["off"] is True and none["box"]["hint"] == "No LiDAR topic in this bag."
    assert none_done["box"]["hint"] == "No LiDAR topic in this bag."
    assert no_lidar["sent"] is None and none["sent"] is None and none_done["sent"] is None
    assert none_done["checked_off"] == [True, None]

    s1 = out["s1"]
    assert s1["box"] == {"hidden": False, "on": True, "off": False, "hint": "", "opts": True, "topic": "", "link": None,
                         "warn": None, "skipped": None}
    assert s1["sent"] == {"kind": "stray"}
    assert (s1["off"]["on"], s1["off"]["hint"], s1["sent_off"]) == (False, "", None)
    assert s1["info"] is False and s1["vsum"] == "Stray capture: 20 frames, 2.0 s, 3 candidate views"
    assert s1["facts"] == "Turned 270° to upright (gravity agrees: 96%). Depth: 19/20 frames (256x192). Confidence: 19."
    assert s1["warn"] == ["1 of the 20 frames have no depth image, so they have no LiDAR samples"]
    s2 = out["s2"]
    assert s2["box"]["on"] is False and s2["box"]["hint"] == ""  # a Start turned it off
    assert s2["facts"].startswith("No gravity data: frames as stored. Depth: 19/20 frames")
    # Only a bag's frames load its topics.json, once per load.
    assert out["topics"] == ["b1", "b2", "b3", "b4", "b5", "b6", "b1"]
    assert out["table_b1"] == ["in use"]
    # The calibration links the topics, and it replaces the bag's TF link.
    assert (out["b7"]["on"], out["b7"]["warn"]) == (True, False)
    assert out["b7"]["link"] == ("Linked by the calibration c.yaml (it replaces /tf_static: camera_optical_frame <- camera_link <- "
                                 "base_link <- livox_frame)")
    b8 = out["b8"]
    assert (b8["livox"]["link"], b8["livox"]["warn"]) == (
        "Linked by the calibration c.yaml (it replaces /tf: camera_optical_frame <- base_link <- livox_frame)", False)
    assert (b8["cloud"]["link"], b8["cloud"]["warn"]) == (
        "The calibration c.yaml is for /livox/lidar: choose that topic, or upload a calibration for /lidar/points", True)
    assert (b8["any"]["link"], b8["any"]["warn"]) == ("Linked by the calibration c.yaml", False)


def test_app_js_calibration_upload():
    """The calibration uploads as soon as it is chosen, with the LiDAR topic chosen (empty when none is): the box
    shows the server's summary, or its reason for refusing the file (also while the part list waits); Remove
    deletes it. The status line says when it worked. The upload, Start, Retry and Build wait for the request, and
    the focus comes back (to the upload after a Remove). A time offset shows when the file gives one, 0 too. In
    review, the scale's summary says when it is stale."""
    found = {"topics": [camera_entry({"/lidar/points": None})], "skipped": [], "lidar": [CLOUD], "lidar_skipped": [], "tf": []}
    # my calib.json gives no time offset.
    summary = {"name": "my_calib.json", "bytes": 412, "translation_m": 0.1234, "rotation_deg": 120.0, "has_intrinsics": False,
               "camera_axes": "optical", "time_offset_s": None, "lidar_topic": "/lidar/points"}
    lid = {"ok": True, "s": 1.25, "reason": None, "spread_rel": 0.004, "warnings": [], "views": [], "calibration_changed": False,
           "counts": {"views": 6, "views_used": 6}, "check_dir": "lidar/check"}
    put = "/api/jobs/b/calibration?name="
    waits = "the job waits for its part list; confirm the list or cancel the job first, then change the calibration"
    replies = {put + "my%20calib.json&lidar_topic=%2Flidar%2Fpoints": summary,
               put + "bad.yaml&lidar_topic=%2Flidar%2Fpoints": {"__status": 400,
                                                                "detail": "T_cam_lidar has 3 numbers; give a 4x4 matrix or 16 numbers"},
               put + "late.yaml&lidar_topic=%2Flidar%2Fpoints": {"__status": 409, "detail": waits},
               "/api/jobs/b/calibration": {"calibration": None},
               put + "k.yaml&lidar_topic=%2Flidar%2Fpoints": {**summary, "name": "k.yaml", "has_intrinsics": True,
                                                              "camera_axes": "body", "time_offset_s": 0.02},
               put + "zero.yaml&lidar_topic=%2Flidar%2Fpoints": {**summary, "name": "zero.yaml", "time_offset_s": 0},
               put + "any.yaml&lidar_topic=": {**summary, "name": "any.yaml", "lidar_topic": None},
               "/api/jobs/b/selection": {"lidar": {**lid, "calibration_changed": True}}}
    out = run_form_js("""
        (async () => {
          V = { id: "b", picks: [], view0: null, topics: """ + json.dumps(found) + """,
            job: { kind: "rosbag", status: "failed", source: { kind: "rosbag", topic: \"""" + CAM + """\" },
              lidar: { kind: "rosbag", topic: "/lidar/points" }, calibration: null } };
          lidarSetup(V.job);
          lidarToggle();
          const str = (x) => typeof x === "string" ? x : "";  // a text never set is "" on the page
          const box = () => ({ link: $("#lidarlink").textContent, warn: $("#lidarlink").className === "warn",
            sum: $("#calsum").hidden ? null : $("#calsum").textContent, btn: $("#calbtn").textContent,
            remove: $("#calremove").hidden, msg: str($("#calmsg").textContent), bad: $("#calmsg").className === "msg" });
          const pick = (name) => { const target = { files: [{ name }], value: "C:/fakepath/" + name };
            return $("#calfile").onchange({ target }).then(() => target.value); };
          // The controls' disabled state while each request runs, and the focus calls.
          const ids = ["#calfile", "#calremove", "#start", "#retry", "#build"], during = [], focused = [];
          ids.forEach((id) => { $(id).disabled = false; $(id).focus = () => focused.push(id); });
          const fetch0 = fetch;
          globalThis.fetch = (url, opt) => { during.push(ids.map((id) => $(id).disabled)); return fetch0(url, opt); };
          const out = { before: box() };
          document.activeElement = $("#calfile");
          out.cleared = await pick("my calib.json");
          out.good = box();
          await pick("bad.yaml");
          out.bad = box();
          await pick("late.yaml");
          out.late = box();
          out.job = V.job.calibration.name;
          document.activeElement = $("#calremove");
          await $("#calremove").onclick();
          out.removed = box();
          out.during = during.splice(0);
          out.after = ids.map((id) => $(id).disabled);
          out.focused = focused.splice(0);
          await pick("k.yaml");
          out.k = box().sum;
          await pick("zero.yaml");
          out.zero = box().sum;
          $("#lidartopic").value = "";  // no topic chosen
          await pick("any.yaml");
          out.any = [box().link, V.job.calibration.lidar_topic];
          $("#lidartopic").value = "/lidar/points";
          Object.assign($("#prompt"), { value: "box" });
          Object.assign($("#nviews"), { value: "4" });
          await $("#startform").onsubmit({ preventDefault() {} });
          out.sent = posts[0].body.lidar;
          // In review the scale's summary is fetched again: an upload makes it stale.
          V.sel = { lidar: """ + json.dumps(lid) + """ };
          $("#review").hidden = false;
          texts.length = 0;
          await pick("my calib.json");
          out.review = [...texts];
          out.requests = requests.filter((r) => r.url.includes("/calibration") || r.url.endsWith("/selection"))
            .map((r) => r.method + " " + r.url);
          console.log(JSON.stringify(out));
        })()
        """, replies)
    assert out["before"] == {"link": "No TF link: upload a calibration", "warn": True, "sum": None, "btn": "Upload calibration",
                             "remove": True, "msg": "", "bad": False}
    assert out["cleared"] == ""  # the same file can be chosen again
    assert out["good"] == {"link": "Linked by the calibration my_calib.json", "warn": False, "remove": False,
                           "msg": "Saved my_calib.json.", "bad": False, "btn": "Replace calibration",
                           "sum": "my_calib.json: 12.3 cm, 120.0°. K from camera_info."}
    # A refused file changes nothing on the server: the summary stays, and the reason shows.
    assert out["bad"] == {**out["good"], "msg": "T_cam_lidar has 3 numbers; give a 4x4 matrix or 16 numbers", "bad": True}
    assert out["late"] == {**out["good"], "msg": waits, "bad": True}
    assert out["job"] == "my_calib.json"
    assert out["removed"] == {**out["before"], "msg": "Removed."}
    # The requests: the uploads, the delete, and after each change of a failed job its lidar_current (GET /api/jobs/b).
    assert out["during"] == [[True] * 5] * 6 and out["after"] == [False] * 5
    # The upload keeps the focus; Remove hides after a delete, so the upload button takes it.
    assert out["focused"] == ["#calfile"] * 4
    assert out["k"] == "k.yaml: 12.3 cm, 120.0°. K from file. Body axes. Offset 0.02 s."
    assert out["zero"] == "zero.yaml: 12.3 cm, 120.0°. K from camera_info. Offset 0 s."  # the two header clocks are one
    assert out["sent"] == {"topic": "/lidar/points"}
    assert "Calibration changed. Build finds the scale again, then stops here." in out["review"]
    assert out["any"] == ["", None]  # no topic chosen: no link line, and the calibration names no topic
    topic = "&lidar_topic=%2Flidar%2Fpoints"
    assert out["requests"] == ["PUT " + put + "my%20calib.json" + topic, "PUT " + put + "bad.yaml" + topic,
                               "PUT " + put + "late.yaml" + topic, "DELETE /api/jobs/b/calibration",
                               "PUT " + put + "k.yaml" + topic, "PUT " + put + "zero.yaml" + topic,
                               "PUT " + put + "any.yaml&lidar_topic=", "PUT " + put + "my%20calib.json" + topic,
                               "GET /api/jobs/b/selection"]


LIDAR_VIEWS = [{"name": "f00003", "s": 1.0498, "points_used": 280, "status": "used"},
               {"name": "f00009", "s": 1.2104, "points_used": 1200, "status": "outlier"},
               {"name": "f00015", "s": None, "points_used": 12, "status": "few_points"}]


def test_app_js_review_shows_the_lidar_scale():
    """The review header gives the LiDAR scale with its views and spread (or DA3's own scale without LiDAR),
    the record's warnings and the folded check images with each view's s and points; each card has its view's
    LiDAR scale. A bag's review offers "Change the calibration": it opens the Settings tab (not in manual
    mode) at the calibration upload, and the tab stays after a refresh."""
    counts = {"views": 16, "views_with_points": 15, "views_used": 14, "outliers": 1, "scattered": 0, "few_points": 1,
              "no_mask": 0, "no_samples": 0}
    lid = {"ok": True, "s": 1.04321, "reason": None, "counts": counts, "spread_rel": 0.0213, "source": {"kind": "rosbag"},
           "extrinsic": "tf", "calibration_changed": False, "warnings": ["a LiDAR warning"], "views": LIDAR_VIEWS,
           "check_dir": "lidar/check"}
    cand = [{"name": n, "frame": int(n[1:]), "time_s": int(n[1:]) / 10, "status": st, "q": 0.9, "consistency": 0.8,
             "angle_to_nearest_pick": 40} for n, st in (("f00003", "picked"), ("f00009", "picked"), ("f00015", "candidate"))]
    sel = {"ok": True, "picks": [{"name": "f00003"}, {"name": "f00009"}], "n_requested": 2, "coverage_deg": 40,
           "min_spacing_deg": 90, "warnings": [], "overrides": {"locked": [], "excluded": [], "instance": {}}, "view0": None,
           "anchor": "f00003", "ring": {"center": None, "points": []}, "candidates": cand, "sam3": {},
           "check_dir": "scene/sam3_box/check"}
    stages = [{"name": n, "status": "done"} for n in ("topics", "frames", "extract", "masks", "prep", "da3", "lidar", "select")]
    job = {"id": "j", "kind": "rosbag", "status": "review", "video": {"name": "scan.bag"}, "prompt": "box", "current": None,
           "error": None, "warnings": [], "key_ok": False, "agent": "none", "articulated": False, "parts_mode": "vlm",
           "parts_review": False, "mode": "auto", "picks": [], "view0": None, "n_views": 2, "auto_continue": False,
           "parts": {}, "calibration": None, "lidar": {"kind": "rosbag", "topic": "/livox/lidar"}, "preview": {"missing": True},
           "source": {"kind": "rosbag", "topic": CAM}, "files": ["selection.json", "pool/lidar.json"], "progress": stages,
           "stages": {x["name"]: {"status": "done"} for x in stages}}
    found = {"topics": [camera_entry({"/livox/lidar": "static"})], "skipped": [], "lidar": [LIVOX], "lidar_skipped": [], "tf": []}
    replies = {"/api/jobs/j/selection": {**sel, "lidar": lid}, "/api/jobs/k/selection": {**sel, "lidar": None},
               "/api/jobs/m/selection": {**sel, "lidar": None}, "/api/jobs/j": job, "/api/jobs/j/frames": frames_meta(),
               "/api/jobs/j/file/frames/topics.json": found, "/api/jobs/j/continue": {"status": "queued", "from": "lidar"}}
    out = run_form_js("""
        (async () => {
          const out = {}, JOB = """ + json.dumps(job) + """;
          for (const [id, kind, lidar] of [["j", "rosbag", JOB.lidar], ["k", "rosbag", null], ["m", "stray", { kind: "stray" }]]) {
            V = { id, state: { warnings: [] }, job: { kind, lidar } };
            texts.length = 0;
            await loadReview();
            out[id] = [...texts];
          }
          // In review the Settings tab holds the start-over form; "Change the calibration" opens that tab (a pick, so
          // the hash keeps it) and focuses the upload.
          globalThis.matchMedia = () => ({ matches: true });
          let focused = null;
          $("#calfile").focus = () => { focused = "#calfile"; };
          $("#log").textContent = "a log";
          location._hash = "#/job/j";
          V = { id: "j", picks: [], view0: null };
          await refresh();
          out.before = [V.tab, $("#frames").hidden, $("#framesbody").hidden];
          toLidarSettings();
          out.edit = [V.tab, V.pick, $("#frames").hidden, focused, $("#start").textContent, mode(), $("#lidaropts").hidden,
            location.hash];
          await refresh();  // the same status: the tab stays
          out.kept = [V.tab, $("#frames").hidden, $("#review").hidden];
          console.log(JSON.stringify(out));
        })()
        """, replies)
    j = out["j"]
    for t in ("LiDAR scale ×1.043 (14/16 views, spread 2.1%)", "a LiDAR warning", "LiDAR check images (3)",
              "f00003: ×1.05 (280 pts)", "f00009: ×1.21 (1,200 pts, outlier)", "f00015: few points (12 pts)",
              "LiDAR ×1.05 (280 pts)", "LiDAR ×1.21 (1,200 pts, outlier)", "LiDAR: few points (12 pts)",
              "/api/jobs/j/file/lidar/check/f00009.jpg", "LiDAR points in f00009"):
        assert t in j, t
    assert not any("calibration changed" in t for t in j)
    assert "Green: mask outline. Blue: nearer. Red: farther. White: agrees." in j
    assert "Scale: Depth Anything 3's estimate (no LiDAR)" in out["k"] and not any("LiDAR x" in t for t in out["k"])
    # LiDAR is on, but its record is not there yet (Build runs the lidar stage first).
    assert "LiDAR scale: not found yet. Build finds it, then stops here." in out["m"]
    # Only a bag with LiDAR on can change its calibration in review.
    assert "Change the calibration" in j and "Change the calibration" not in out["k"] + out["m"]
    assert out["before"] == ["inputs", True, False]  # the review sits above the Inputs tab; the form waits in Settings
    assert out["edit"] == ["settings", "settings", False, "#calfile", "Start over", "auto", False, "#/job/j/settings"]
    assert out["kept"] == ["settings", False, False]


def test_app_js_review_marks_edits_and_draws_the_ring():
    """The review cards and the ring. A locked pick says "locked", with a badge of its own (not the picked one), and is
    a square on the ring; an available view is a hollow dot, a small or inconsistent one a cross, and an excluded one
    is not on the ring. Each pick's view number sits outward from the object, clear of its mark and of the cameras
    beside it, and no mark has a color of its own (style.css gives them). An edit says what Select again does to its
    card: a pick set to "never use" is left out, a view set to "always use" is added. While edits wait, Select again is
    the primary button and Build is off; undoing them brings Build back. View 0 is offered only on picks and on views
    set to "always use", and a view 0 that no card offers any more goes out as null. An added view taken as view 0 and
    set back to "auto" gives view 0 back to the saved choice, so its card shows no edit. Build and Select again bring
    the job's head back into view."""
    names = [f"f{4 * i:05d}" for i in range(12)]
    status = {"f00000": "picked", "f00012": "picked", "f00024": "picked", "f00036": "picked", "f00008": "small",
              "f00032": "inconsistent", "f00044": "excluded"}
    cand = [{"name": n, "frame": int(n[1:]), "time_s": int(n[1:]) / 10, "status": status.get(n, "available"), "q": 0.9,
             "consistency": 0.9, "angle_to_nearest_pick": 20} for n in names]
    points = [None if n == "f00044" else [math.cos(math.pi * i / 6), math.sin(math.pi * i / 6)] for i, n in enumerate(names)]
    sel = {"ok": True, "picks": [{"name": n} for n in ("f00000", "f00012", "f00024", "f00036")], "n_requested": 4,
           "coverage_deg": 30, "min_spacing_deg": 90, "warnings": [], "anchor": "f00000", "view0": None,
           "overrides": {"locked": ["f00024"], "excluded": ["f00044"], "instance": {}}, "candidates": cand,
           "ring": {"center": [0, 0], "points": points}, "sam3": {}, "check_dir": "scene/sam3_box/check", "lidar": None}
    replies = {"/api/jobs/j/selection": sel, "/api/jobs/j/overrides": {"status": "queued", "from": "select"},
               "/api/jobs/j/continue": {"status": "queued", "from": "sam3d"}}
    out = run_form_js(ELEMENTS + """
        document.createElementNS = (ns, tag) => document.createElement(tag);
        (async () => {
          const marks = [], cls = { "#build": new Set(), "#reselect": new Set() }, scrolled = [];
          $("#ring").replaceChildren = () => { marks.length = 0; };
          $("#ring").append = (...k) => marks.push(...k);
          for (const id of Object.keys(cls)) $(id).classList = { toggle: (c, on) => on ? cls[id].add(c) : cls[id].delete(c) };
          $("#jtitle").scrollIntoView = (o) => scrolled.push(o);
          $("#key2").value = "";
          V = { id: "j", state: { warnings: [] }, job: { kind: "video", lidar: null } };
          await loadReview();
          const out = {};
          out.ring = marks.filter((m) => m.kids.length).map((m) => [m.kids[0].textContent, m.tag, m.attrs.class || null]);
          out.colors = marks.flatMap((m) => ["fill", "stroke"].map((k) => m.attrs[k]).filter((v) => v != null));
          // Each view number's distance from its own camera and from the nearest other camera, in units of d.
          const d = 1.3 / 30, cams = V.sel.ring.points.map((p, i) => p && [p[0], -p[1], V.sel.candidates[i].name]).filter(Boolean);
          const own = [], other = [];
          for (const m of marks.filter((m) => m.tag === "text")) {
            const name = V.sel.picks[Number(m.textContent)].name;
            for (const [x, y, n] of cams) (n === name ? own : other).push(Math.hypot(m.attrs.x - x, m.attrs.y - y) / d);
          }
          out.gaps = [Math.min(...own), Math.min(...other)];
          const entry = (n) => [...V.cards].find(([c]) => c.name === n)[1];
          const ctl = (n) => made.find((e) => e.tag === "select" && e.attrs["aria-label"] === n + ": use");
          const card = (n) => ({ cls: [...entry(n).el.cls], note: entry(n).note.textContent, view0: !entry(n).zero.hidden });
          const bar = () => ({ build: $("#build").disabled, reselect: $("#reselect").disabled,
            primary: cls["#build"].has("secondary") ? "reselect" : "build", discard: $("#discard").hidden, msg: $("#revmsg").textContent });
          const badgeOf = (n) => entry(n).el.kids[2].kids[0];
          out.badges = ["f00000", "f00024", "f00004", "f00044"].map((n) => [text(badgeOf(n)), badgeOf(n).className]);
          out.before = [bar(), card("f00012"), card("f00004"), card("f00044")];
          ctl("f00012").onchange({ target: { value: "out" } });
          ctl("f00004").onchange({ target: { value: "in" } });
          out.edited = [bar(), card("f00012"), card("f00004")];
          ctl("f00000").onchange({ target: { value: "out" } });  // view 0 itself
          await $("#reselect").onclick();
          out.sent = posts.map((p) => p.body);
          out.reselected = scrolled.splice(0);
          for (const n of ["f00012", "f00004", "f00000"]) ctl(n).onchange({ target: { value: "auto" } });
          out.undone = [bar(), card("f00012"), card("f00004")];
          await $("#build").onclick();
          out.built = scrolled.splice(0);
          // An added view taken as view 0, then set back to auto: its choice hides, view 0 goes back to the saved choice
          // (the anchor), and the radios follow.
          ctl("f00004").onchange({ target: { value: "in" } });
          entry("f00004").radio.onchange();
          const chosen = [card("f00004"), entry("f00004").radio.checked, entry("f00000").radio.checked];
          ctl("f00004").onchange({ target: { value: "auto" } });
          out.unadded = [chosen, card("f00004"), V.ov.view0, entry("f00004").radio.checked, entry("f00000").radio.checked, bar()];
          await $("#reselect").onclick();
          out.unadded.push(posts.at(-1).body.view0);
          console.log(JSON.stringify(out));
        })()
        """, replies)
    assert out["ring"] == [["f00000 (picked)", "polygon", None], ["f00004 (available)", "circle", "cand"],
                           ["f00008 (small)", "path", "out"], ["f00012 (picked)", "circle", "pick"],
                           ["f00016 (available)", "circle", "cand"], ["f00020 (available)", "circle", "cand"],
                           ["f00024 (locked)", "rect", "lock"], ["f00028 (available)", "circle", "cand"],
                           ["f00032 (inconsistent)", "path", "out"], ["f00036 (picked)", "circle", "pick"],
                           ["f00040 (available)", "circle", "cand"]]
    assert set(out["colors"]) <= {"currentColor", "none"}
    assert out["gaps"][0] >= 3.4 - 1e-9 and out["gaps"][1] > 4  # the font is 2.4 d high
    assert out["badges"] == [["picked", "badge picked"], ["locked", "badge locked"], ["available", "badge available"],
                             ["excluded", "badge excluded"]]
    plain = {"cls": [], "note": "", "view0": False}
    assert out["before"] == [{"build": False, "reselect": True, "primary": "build", "discard": True, "msg": ""},
                             {**plain, "view0": True}, plain, plain]
    assert out["edited"] == [{"build": True, "reselect": False, "primary": "reselect", "discard": False,
                              "msg": "Select again to apply changes."},
                             {"cls": ["drops"], "note": "Left out when you select again.", "view0": False},
                             {"cls": ["adds"], "note": "Added when you select again.", "view0": True}]
    assert out["sent"] == [{"locked": ["f00024", "f00004"], "excluded": ["f00044", "f00012", "f00000"], "instance": {},
                            "view0": None, "api_key": None}]
    assert out["undone"] == out["before"][:3]
    assert out["reselected"] == out["built"] == [{"block": "nearest"}]
    assert out["unadded"] == [[{"cls": ["adds"], "note": "Added when you select again.", "view0": True}, True, False],
                              plain, None, False, True, out["before"][0], None]
    # A locked pick's badge mark is hollow, so it differs from a pick's filled one without the word.
    import re
    css = (server.STATIC / "style.css").read_text()
    locked = re.search(r"\.badge\.locked::before \{([^}]*)\}", css).group(1)
    assert "background: transparent" in locked and "box-shadow: inset" in locked


def test_app_js_lidar_failure_panel():
    """A job that failed at the lidar stage: a panel with the warnings and the check images from pool/lidar.json, and
    the next step (a calibration for a bag, another one when it has one; not for a Stray capture). The status line
    gives the reason, so neither the panel nor the stage's note repeats it. Retry starts at lidar. refresh() shows the
    panel only for a job that failed there, and only once the panel is drawn."""
    reason = ("LiDAR points cover the object in only 2 of 6 views (at least 3 are needed); the object may be outside the "
              "LiDAR's view, or the time window or T_cam_lidar is wrong (see the check images)")
    rec = {"version": 1, "ok": False, "s": None, "reason": reason, "warnings": ["a stage warning"],
           "views": [{**v, "frame": int(v["name"][1:]), "points": 400, "iqr_rel": 0.1} for v in LIDAR_VIEWS[:2]],
           "combine": {"views": 6, "views_with_points": 2, "views_used": 0, "spread_rel": None, "outliers": 0},
           "extrinsic": {"source": "calibration"}, "source": {"kind": "rosbag", "topic": "/lidar/points"}}
    stages = [{"name": n, "status": "done"} for n in ("topics", "frames", "extract", "masks", "prep", "da3")] + [
        {"name": "lidar", "status": "failed", "note": "LiDAR scale: " + reason}] + [
        {"name": n, "status": "pending"} for n in ("select", "sam3d", "place", "propose", "parts", "agent")]
    masks = [{**s, "status": "failed"} if s["name"] == "masks" else s for s in stages[:4]] + [
        {"name": n, "status": "pending"} for n in ("prep", "da3", "lidar", "select")]
    job = {"id": "j", "kind": "rosbag", "status": "failed", "video": {"name": "scan.bag"}, "files": ["pool/lidar.json"],
           "progress": stages}
    # The same job as GET /api/jobs/j gives it, and a done job (d) with the same record.
    full = {**job, "prompt": "box", "current": None, "error": "lidar: " + reason, "warnings": [], "key_ok": False,
            "agent": "none", "articulated": False, "parts_mode": "vlm", "parts_review": False, "mode": "auto", "picks": [],
            "view0": None, "n_views": 4, "auto_continue": True, "parts": {}, "calibration": None,
            "lidar": {"kind": "rosbag", "topic": "/lidar/points"}, "source": {"kind": "rosbag", "topic": CAM},
            "preview": {"missing": True}, "stages": {x["name"]: {"status": x["status"]} for x in stages}}
    done = {**full, "id": "d", "status": "done", "error": None, "progress": [{**x, "status": "done"} for x in stages],
            "stages": {x["name"]: {"status": "done"} for x in stages}}
    found = {"topics": [camera_entry({"/lidar/points": None})], "skipped": [], "lidar": [CLOUD], "lidar_skipped": [], "tf": []}
    replies = {"/api/jobs/j/file/pool/lidar.json": rec, "/api/jobs/d/file/pool/lidar.json": rec, "/api/jobs/j": full,
               "/api/jobs/d": done, "/api/jobs/j/frames": frames_meta(), "/api/jobs/d/frames": frames_meta(),
               "/api/jobs/j/file/frames/topics.json": found, "/api/jobs/d/file/frames/topics.json": found}
    out = run_form_js("""
        (async () => {
          const job = """ + json.dumps(job) + """, out = {};
          const state = (stages) => ({ status: "failed", stages, agent: "none", key_ok: false, warnings: [], error: "lidar: x" });
          V = { id: "j", job };
          texts.length = 0;
          await renderLidarFail(job);
          out.panel = [...texts];
          renderState(state(job.progress));
          out.next = $("#jnext").textContent;
          out.retry = [$("#retry").hidden, $("#retry").textContent];
          V.job = { ...job, calibration: { name: "c.yaml" } };
          renderState(state(job.progress));
          out.cal = $("#jnext").textContent;
          V.job = { ...job, kind: "stray" };
          renderState(state(job.progress));
          out.stray = $("#jnext").textContent;
          renderState(state(""" + json.dumps(masks) + """));
          out.masks = $("#jnext").textContent;
          texts.length = 0;
          await renderLidarFail({ ...job, files: [] });
          out.norecord = [...texts];
          out.failed = [lidarFailed("failed", job.progress), lidarFailed("cancelled", job.progress), lidarFailed("failed", [])];
          // The stage's note: left out when the status line's error gives it, kept otherwise.
          const note = job.progress[6].note;
          texts.length = 0;
          renderState({ ...state(job.progress), error: "lidar: " + note });
          out.note = texts.includes(note);
          texts.length = 0;
          renderState(state(job.progress));
          out.note2 = texts.includes(note);
          // The page as refresh() draws it: the panel for the job that failed at lidar, not for a done job.
          const draw0 = renderLidarFail;
          renderLidarFail = async (j) => { out.hiddenWhileDrawn = $("#lidarfail").hidden; return draw0(j); };
          out.shown = [];
          for (const id of ["j", "d"]) {
            V = { id, picks: [], view0: null };
            $("#log").textContent = "a log";  // not fetched again
            $("#lidarfail").hidden = true;
            texts.length = 0;
            await refresh();
            out.shown.push([$("#lidarfail").hidden, texts.includes("LiDAR check images (2)"),
              texts.some((x) => x.startsWith("No metric scale from LiDAR"))]);
          }
          console.log(JSON.stringify(out));
        })()
        """, replies)
    for t in ("a stage warning", "LiDAR check images (2)", "f00003: ×1.05 (280 pts)", "LiDAR settings",
              "Green: mask outline. Blue: nearer. Red: farther. White: agrees."):
        assert t in out["panel"], t
    assert not any(t.startswith("No metric scale from LiDAR") for t in out["panel"])  # the status line gives it
    # The next step names the tab where the calibration and the LiDAR box are.
    assert out["next"] == "In Settings, upload a calibration and Retry, or turn LiDAR off and build again."
    assert out["cal"] == "In Settings, replace the calibration and Retry, or turn LiDAR off and build again."
    assert out["retry"] == [False, "Retry from lidar"]
    assert out["stray"] == "In Settings, turn LiDAR off or change the views, then build again."
    assert out["masks"] == "Check the log, then Retry."
    assert "No LiDAR record was written. See the log." in out["norecord"]
    assert out["failed"] == [True, False, False]
    assert out["note"] is False and out["note2"] is True
    assert out["shown"] == [[False, True, False], [True, False, False]] and out["hiddenWhileDrawn"] is True


def test_app_js_changed_calibration_retries_from_lidar():
    """A failed or cancelled bag job whose scale was found with another calibration: the server retries from
    lidar (Runner.resume), so Retry says so and sends from lidar. The page learns it from the job's lidar_current
    (false: stale) after a load and after an upload or a delete, which fetch the job again. A reply that starts
    earlier than asked is shown."""
    stages = [{"name": n, "status": "done"} for n in ("topics", "frames", "extract", "masks", "prep", "da3", "lidar", "select")] + [
        {"name": "sam3d", "status": "failed"}] + [{"name": n, "status": "pending"} for n in ("place", "propose", "parts", "agent")]
    job = {"id": "c", "kind": "rosbag", "status": "failed", "video": {"name": "scan.bag"}, "prompt": "box", "current": None,
           "error": "sam3d: out of memory", "warnings": [], "key_ok": False, "agent": "none", "articulated": False,
           "parts_mode": "vlm", "parts_review": False, "mode": "auto", "picks": [], "view0": None, "n_views": 4,
           "auto_continue": True, "parts": {}, "calibration": None, "lidar": {"kind": "rosbag", "topic": "/lidar/points"},
           "source": {"kind": "rosbag", "topic": CAM}, "files": ["selection.json", "pool/lidar.json"], "progress": stages,
           "stages": {x["name"]: {"status": x["status"]} for x in stages}, "lidar_current": True, "preview": {"missing": True}}
    found = {"topics": [camera_entry({"/lidar/points": None})], "skipped": [], "lidar": [CLOUD], "lidar_skipped": [], "tf": []}
    summary = {"name": "c.yaml", "bytes": 300, "translation_m": 0.1, "rotation_deg": 90.0, "has_intrinsics": False,
               "camera_axes": "optical", "time_offset_s": None, "lidar_topic": "/lidar/points"}
    replies = {"/api/jobs/c/frames": frames_meta(), "/api/jobs/c/file/frames/topics.json": found,
               "/api/jobs/c/calibration?name=c.yaml&lidar_topic=%2Flidar%2Fpoints": summary,
               "/api/jobs/c/calibration": {"calibration": None}, "/api/jobs/c/continue": {"status": "queued", "from": "lidar"}}
    out = run_form_js("""
        (async () => {
          // The job as the server has it now: lidar_current is false while the scale is stale.
          let stale = false;
          const J = """ + json.dumps(job) + """;
          const fetch0 = fetch, reply = (x) => Promise.resolve({ ok: true, status: 200, json: async () => x });
          globalThis.fetch = (url, opt) => url === "/api/jobs/c" ? reply({ ...J, lidar_current: !stale })
            : url.includes("/file/logs/") ? Promise.resolve({ ok: false, status: 404, text: async () => "" }) : fetch0(url, opt);
          const page = () => [$("#retry").textContent, $("#jnext").textContent];
          const out = {};
          $("#log").textContent = "a log";
          V = { id: "c", picks: [], view0: null };
          await refresh();
          out.current = page();
          stale = true;  // what an upload does on the server
          const target = { files: [{ name: "c.yaml" }], value: "c.yaml" };
          await $("#calfile").onchange({ target });
          out.upload = [...page(), $("#calmsg").textContent];
          posts.length = 0;
          await $("#retry").onclick();
          out.sent = posts[0].body.from;
          out.runmsg = $("#runmsg").textContent;
          stale = false;  // removed again: the scale is current
          await $("#calremove").onclick();
          out.removed = [...page(), $("#calmsg").textContent];
          stale = true;  // a reload of the page finds it stale
          V = { id: "c", picks: [], view0: null };
          await refresh();
          out.reload = page();
          // The page asked for sam3d, but the server starts from lidar (it knew better): the reply says so, without a
          // cause, which the page does not know.
          stale = false;
          await refresh();
          posts.length = 0;
          await $("#retry").onclick();
          out.asked = [posts[0].body.from, $("#runmsg").textContent];
          // Only a job with LiDAR whose lidar_current is false is stale; null means that the stage is not done. A Stray
          // capture's scale goes stale too, when an older estimator found it (lidar_stale "estimator").
          out.bystate = [true, false, null, undefined].map((current) => scaleStale({ ...J, lidar_current: current }));
          out.never = [scaleStale({ ...J, lidar: null, lidar_current: false })];
          out.stray = scaleStale({ ...J, kind: "stray", lidar: { kind: "stray" }, lidar_current: false, lidar_stale: "estimator" });
          console.log(JSON.stringify(out));
        })()
        """, replies)
    assert out["current"] == ["Retry from sam3d", "Check the log, then Retry."]
    assert out["upload"] == ["Retry from lidar", "Check the log, then Retry.", "Saved c.yaml."]  # the button says it
    assert out["sent"] == "lidar" and out["runmsg"] == ""  # the server started where the page said
    assert out["removed"] == [*out["current"], "Removed."]
    assert out["reload"] == out["upload"][:2]
    assert out["asked"] == ["sam3d", "Starting from lidar (the scale is out of date)."]
    assert out["bystate"] == [False, True, False, False] and out["never"] == [False] and out["stray"] is True


def test_app_js_done_job_with_a_stale_scale_runs_again_from_lidar():
    """A done bag job whose calibration changed after the build (lidar_current false; here a file of the same name
    with new content) warns that the scale and the masses are out of date. Its button runs lidar and every stage
    after it again, and the line under the status says so, and that the job pauses for review when it does; the
    key goes with the click, since the agent runs again. The click asks first: it replaces the results. A done job
    whose scale is current offers no button, and one whose 360 video failed runs again from lidar too."""
    names = ["topics", "frames", *jobs.BUILD]
    stages = [{"name": n, "status": "done"} for n in names]
    job = {"id": "d", "kind": "rosbag", "status": "done", "video": {"name": "scan.bag"}, "prompt": "box", "current": None,
           "error": None, "warnings": [], "key_ok": False, "agent": "astra", "articulated": False, "parts_mode": "vlm",
           "parts_review": False, "mode": "auto", "picks": [], "view0": None, "n_views": 4, "auto_continue": False,
           "parts": {}, "lidar": {"kind": "rosbag", "topic": "/lidar/points"}, "source": {"kind": "rosbag", "topic": CAM},
           "calibration": {"name": "c.yaml", "bytes": 300, "translation_m": 0.1, "rotation_deg": 90.0, "has_intrinsics": False,
                           "camera_axes": "optical", "time_offset_s": None, "lidar_topic": "/lidar/points"},
           "files": ["selection.json", "pool/lidar.json", "results.zip"], "progress": stages,
           "stages": {x["name"]: {"status": x["status"]} for x in stages}, "lidar_current": True, "preview": {"missing": True}}
    found = {"topics": [camera_entry({"/lidar/points": None})], "skipped": [], "lidar": [CLOUD], "lidar_skipped": [], "tf": []}
    replies = {"/api/jobs/d/frames": frames_meta(), "/api/jobs/d/file/frames/topics.json": found,
               "/api/jobs/d/calibration?name=c.yaml&lidar_topic=%2Flidar%2Fpoints": {**job["calibration"], "bytes": 320},
               "/api/jobs/d/continue": {"status": "queued", "from": "lidar"}}
    out = run_form_js("""
        (async () => {
          // The job as the server has it now: lidar_current is false while the scale is stale, and lidar_stale says why.
          let stale = false, answer = false, cause = "calibration";
          const J = """ + json.dumps(job) + """, asked = [];
          const fetch0 = fetch, reply = (x) => Promise.resolve({ ok: true, status: 200, json: async () => x });
          globalThis.fetch = (url, opt) => url === "/api/jobs/d"
            ? reply({ ...J, lidar_current: !stale, lidar_stale: stale ? cause : null }) : fetch0(url, opt);
          globalThis.confirm = (text) => { asked.push(text); return answer; };
          const page = () => ({ retry: $("#retry").hidden ? null : $("#retry").textContent, next: $("#jnext").textContent,
            key: $("#keyrow").hidden, warn: texts.filter((t) => t.includes("out of date")) });
          const out = {};
          V = { id: "d", picks: [], view0: null };
          texts.length = 0;
          await refresh();
          out.current = page();
          stale = true;  // the same file name with new content: steps.done compares the sha256
          texts.length = 0;
          await $("#calfile").onchange({ target: { files: [{ name: "c.yaml" }], value: "c.yaml" } });
          out.upload = { ...page(), msg: $("#calmsg").textContent };
          posts.length = 0;
          await $("#retry").onclick();  // asked, and not confirmed
          out.refused = [posts.length, [...asked]];
          answer = true;
          $("#key2").value = "sk-test-key";
          await $("#retry").onclick();
          out.sent = posts.map((p) => [p.url, p.body]);
          // A reload finds it stale too; with auto_continue the run does not pause.
          J.auto_continue = true;
          V = { id: "d", picks: [], view0: null };
          texts.length = 0;
          await refresh();
          out.reload = page();
          // No agent: no key to give. A failed 360 video: the run starts from lidar all the same.
          V.job.agent = "none";
          const st = (over) => ({ ...V.state, ...over });
          renderState(st({ agent: "none" }));
          out.none = page().key;
          renderState(st({ agent: "astra", stages: J.progress.map((x) => x.name === "turntable" ? { ...x, status: "failed" } : x) }));
          out.turntable = page().retry;
          // An upload can change the cause: the build's calibration again, while an older estimator found the scale.
          renderState(st({ agent: "astra" }));
          cause = "estimator";
          texts.length = 0;
          await $("#calfile").onchange({ target: { files: [{ name: "c.yaml" }], value: "c.yaml" } });
          out.cause = page().warn;
          // A done Stray capture whose scale an older estimator found: the same button, and the estimator's warning.
          const S = { ...J, kind: "stray", video: { name: "cap.zip" }, lidar: { kind: "stray" }, calibration: null,
            source: { kind: "stray", topic: null } };
          globalThis.fetch = (url, opt) => url === "/api/jobs/d" ? reply({ ...S, lidar_current: false, lidar_stale: "estimator" })
            : fetch0(url, opt);
          V = { id: "d", picks: [], view0: null };
          texts.length = 0;
          await refresh();
          out.stray = page();
          console.log(JSON.stringify(out));
        })()
        """, replies)
    assert out["current"] == {"retry": None, "next": "", "key": True, "warn": []}  # current: no button, no warning
    warn = "The calibration changed after this build, so the scale and the masses are out of date."
    hint = "Run lidar and later stages again to update them. It runs the lidar stage and every stage after it, GPU stages included."
    assert out["upload"] == {"retry": "Run lidar and later stages again", "next": hint + " It pauses for your review.", "key": False,
                             "warn": [warn], "msg": "Saved c.yaml."}
    assert out["refused"] == [0, ["Run lidar and every stage after it again? This replaces the current results."]]
    assert out["sent"] == [["/api/jobs/d/continue", {"from": "lidar", "api_key": "sk-test-key"}]]
    assert out["reload"] == {"retry": "Run lidar and later stages again", "next": hint, "key": False, "warn": [warn]}
    assert out["none"] is True and out["turntable"] == "Run lidar and later stages again"
    new_way = "The app now finds the LiDAR scale in a new way, so this build's scale and masses are out of date."
    assert out["cause"] == [new_way]
    assert out["stray"] == {"retry": "Run lidar and later stages again", "next": hint, "key": False, "warn": [new_way]}


def test_app_js_results_show_the_metric_scale():
    """The results give the metric scale that select applied (placement.json) with its LiDAR source, views
    and spread, or DA3's own estimate."""
    place = {"metric_scale": 1.04321, "units": "meters (LiDAR scale applied in select)", "check": {"mean_mask_iou": 0.91}}
    rec = {"ok": True, "s": 1.04321, "source": {"kind": "rosbag", "topic": "/livox/lidar"},
           "combine": {"views": 16, "views_used": 14, "spread_rel": 0.0213}}
    files = ["run/posed.glb", "run/placement.json", "pool/lidar.json", "results.zip"]
    base = {"id": "j", "agent": "none", "agent_result": None, "kind": "rosbag", "lidar": {"kind": "rosbag", "topic": "/livox/lidar"},
            "stages": {"agent": {"status": "skipped", "note": "turned off: physics uses the Stage A mesh"}}}
    replies = {"/api/jobs": [], "/api/jobs/j/file/run/placement.json": place, "/api/jobs/j/file/pool/lidar.json": rec,
               "/api/jobs/s/file/run/placement.json": {**place, "metric_scale": 1.25},
               "/api/jobs/s/file/pool/lidar.json": {"ok": True, "s": 1.25, "source": {"kind": "stray", "file": "c.zip"},
                                                    "combine": {"views_used": 6, "spread_rel": 0.004}},
               "/api/jobs/n/file/run/placement.json": {**place, "metric_scale": None, "units": "DA3 units (meters)"}}
    cases = {"j": {**base, "files": files}, "s": {**base, "id": "s", "kind": "stray", "lidar": {"kind": "stray"}, "files": files},
             "n": {**base, "id": "n", "lidar": None, "files": [f for f in files if f != "pool/lidar.json"]}}
    out = run_app_js(replies, """
        (async () => {
          const out = {}, cases = """ + json.dumps(cases) + """;
          for (const [id, job] of Object.entries(cases)) {
            V = { id };
            texts.length = 0;
            await renderResults(job);
            const k = texts.indexOf("Metric scale");
            out[id] = k >= 0 ? texts[k + 2] : null;
          }
          console.log(JSON.stringify(out));
        })()
        """)
    assert out["j"] == "×1.043 from LiDAR (/livox/lidar, 14 views, spread 2.1%)"
    assert out["s"] == "×1.250 from LiDAR (iPhone LiDAR depth, 6 views, spread 0.4%)"
    assert out["n"] == "Depth Anything 3's estimate (no LiDAR)"


# ---- the input panel: the preview video and the views used ----

# A stand-in for document.createElement that keeps what app.js puts in each element (tag, attributes, children,
# class names, the video's calls), so a test can find the elements and click them; text(e) is e's text.
ELEMENTS = """
const made = [];
document.createElement = (tag) => {
  const e = { tag, kids: [], attrs: {}, cls: new Set(), calls: [],
    setAttribute(k, v) { e.attrs[k] = String(v); }, append(...k) { e.kids.push(...k); }, replaceChildren(...k) { e.kids = k; },
    pause() { e.calls.push("pause"); }, play() { e.calls.push("play"); }, load() { e.calls.push("load"); },
    scrollIntoView() { e.calls.push("scroll"); }, focus() {} };
  e.classList = { toggle: (c, on) => { if (on) e.cls.add(c); else e.cls.delete(c); }, add: (c) => e.cls.add(c) };
  made.push(e);
  return e;
};
const text = (e) => typeof e === "string" ? e : (e && e.kids || []).map(text).join("");
const views = () => made.filter((e) => e.tag === "figure" && e.className !== "inputfig");
"""
PICKS = [{"name": "f00152", "frame": 152, "time_s": 15.2, "q": 0.9}, {"name": "f00040", "frame": 40, "time_s": 4.0, "q": 0.8},
         {"name": "f00301", "frame": 301, "time_s": 30.1333, "q": 0.7}]
PREVIEW = {"file": "frames/preview.mp4", "width": 640, "height": 360, "fps": 14.985, "stride": 2, "frames": 240, "t0": 0.2,
           "bytes": 2_400_000}


def input_job(job_id, select="done", sam3d="failed", **kw):
    """GET /api/jobs/<job_id> of a video job whose views are selected and that failed at sam3d; select and sam3d
    set those stages' status."""
    names = ["frames", "extract", "masks", "prep", "da3", "lidar", "select", "sam3d", "place", "agent"]
    status = {n: "done" for n in names[:5]} | {"lidar": "skipped", "select": select, "sam3d": sam3d, "place": "pending",
                                               "agent": "pending"}
    progress = [{"name": n, "status": status[n]} for n in names]
    return {"id": job_id, "kind": "video", "status": "failed", "video": {"name": "table.MOV"}, "source": {"kind": "video"},
            "prompt": "table", "current": None, "error": "sam3d: out of memory", "warnings": [], "key_ok": False,
            "agent": "none", "articulated": False, "parts_mode": "vlm", "parts_review": False, "mode": "auto", "picks": [],
            "view0": None, "n_views": 3, "auto_continue": True, "parts": {}, "calibration": None, "lidar": None,
            "files": ["selection.json"], "preview": PREVIEW, "progress": progress,
            "stages": {n: {"status": status[n], "ended": 1000.0 + k} for k, n in enumerate(names)}, **kw}


def test_app_js_input_panel_and_views():
    """The input panel: the preview video (muted, controls, metadata only, sized by the preview's shape) with the
    input's name and kind, and one figure per view in run order with its image and "view i, frame f, t s"; view 0
    has the star. A click on a view puts the video at the frame that the preview shows for it (1 ms in), paused,
    marks that view as current and says which frame it shows. Place's later end adds the mesh fit toggle without
    drawing the video again; the toggle swaps every image for place's overlay, and the hint names the overlay's
    colors while it shows. A job without a preview says so in a few words, and its views are no buttons. The panel
    is drawn once per selection, so a refresh does not stop the video, and a selection.json that comes after another
    job's page draws nothing."""
    seek = {"152": {"frame": 152, "at": 15.0}, "40": {"frame": 40, "at": 3.8}, "301": {"frame": 300, "at": 29.9}}
    first = input_job("j", preview={**PREVIEW, "seek": seek})
    run = {**first, "files": ["selection.json", "run/placement.json"]}  # place has ended since
    old = input_job("m")  # a preview without "seek": the page seeks by the pick's time
    bag = input_job("k", kind="rosbag", video={"name": "scan.bag"}, source={"kind": "rosbag", "topic": "/cam/image_rect"},
                    preview={"error": "OSError: disk full"})
    stray = input_job("s", kind="stray", video={"name": "capture.zip"}, source={"kind": "stray"},
                      preview={"error": "ValueError: no libx264"})
    gone = input_job("g", preview={"missing": True})
    replies = {f"/api/jobs/{i}/file/selection.json": {"ok": True, "picks": PICKS, "anchor": "f00152"} for i in "jmksg"}
    out = run_form_js(ELEMENTS + """
        (async () => {
          const out = {};
          V = { id: "j" };
          await renderInput(""" + json.dumps(first) + """);
          const video = made.find((e) => e.tag === "video"), figs = views(), btns = figs.map((f) => f.kids[0]);
          out.video = { ...video.attrs, muted: video.muted };
          out.caption = text(made.find((e) => e.className === "inputfig"));
          out.caps = figs.map((f) => text(f.kids[1]));
          out.buttons = btns.map((b) => [b.tag, b.attrs["aria-label"], text(b)]);
          out.imgs = btns.map((b) => b.kids[0].attrs);
          const head = made.find((e) => e.className === "viewshead"), hint = () => made.filter((e) => e.className === "hint").map(text);
          out.head0 = text(head);
          out.hint0 = hint();
          const seek = (i) => { btns[i].onclick(); return [video.currentTime, $("#inputmsg").textContent]; };
          out.seek1 = [...seek(1), [...video.calls], [...figs[1].cls], [...figs[0].cls], btns.map((b) => b.attrs["aria-current"])];
          out.seek2 = seek(2);
          out.seek0 = [...seek(0), [...figs[0].cls], [...figs[1].cls], btns.map((b) => b.attrs["aria-current"])];
          const n = made.length;
          await renderInput(""" + json.dumps(run) + """);  // place ended: only the heading changes
          out.placed = [made.filter((e) => e.tag === "video").length, made.filter((e) => e.tag === "img").length, made.length - n,
            [...figs[0].cls], btns[0].attrs["aria-current"], text(head), $("#inputmsg").textContent];
          out.hint1 = hint();
          const toggle = made.find((e) => e.tag === "button" && text(e) === "Mesh fit");
          toggle.onclick({ currentTarget: toggle });
          out.overlay = [toggle.attrs["aria-pressed"], btns.map((b) => b.kids[0].attrs.src), hint()];
          toggle.onclick({ currentTarget: toggle });
          out.photos = [toggle.attrs["aria-pressed"], btns[2].kids[0].attrs.src, hint()];
          const m = made.length;
          await renderInput(""" + json.dumps(run) + """);  // a refresh of the same job: nothing drawn again
          out.again = made.length === m && !video.calls.includes("play") && !video.calls.includes("load");
          made.length = 0;
          V = { id: "m" };
          await renderInput(""" + json.dumps(old) + """);
          views()[2].kids[0].onclick();
          out.m = [made.find((e) => e.tag === "video").currentTime, $("#inputmsg").textContent];
          for (const job of [""" + json.dumps(bag) + ", " + json.dumps(stray) + ", " + json.dumps(gone) + """]) {
            made.length = 0;
            V = { id: job.id };
            await renderInput(job);
            out[job.id] = { video: made.some((e) => e.tag === "video"), p: made.filter((e) => e.tag === "p").map(text),
              tags: views().map((f) => f.kids[0].tag), alt: views()[0].kids[0].kids[0].attrs.alt,
              toggle: made.some((e) => e.tag === "button") };
          }
          // Job j's selection.json comes after job k's page is drawn: it draws nothing there.
          let release;
          const fetch0 = fetch;
          globalThis.fetch = (url, opt) => url === "/api/jobs/j/file/selection.json" ? new Promise((r) => {
            release = () => r({ ok: true, status: 200, json: async () => ({ picks: [""" + json.dumps(PICKS[0]) + """] }) }); })
            : fetch0(url, opt);
          V = { id: "j" };
          const slow = renderInput(""" + json.dumps(run) + """);
          V = { id: "k" };
          await renderInput(""" + json.dumps(bag) + """);
          const drawn = made.length, key = V.inputKey;
          release();
          await slow;
          out.slow = [made.length === drawn, V.input.picks.length, V.inputKey === key];
          console.log(JSON.stringify(out));
        })()
        """, replies)
    assert out["video"] == {"src": "/api/jobs/j/file/frames/preview.mp4", "controls": "", "preload": "metadata", "playsinline": "",
                            "muted": True, "width": "640", "height": "360", "style": "--ar: 640 / 360",
                            "aria-label": "Preview of table.MOV"}
    assert out["caption"] == "table.MOV (video)"
    caps = ["view 0, frame 152, 15.2 s", "view 1, frame 40, 4.0 s", "view 2, frame 301, 30.1 s"]
    assert out["caps"] == caps
    assert out["buttons"] == [["button", c + ": show it in the video", t] for c, t in zip(caps, ["★", "", ""])]
    assert [i["src"] for i in out["imgs"]] == [f"/api/jobs/j/file/run/input/images/{i}.png" for i in range(3)]
    assert all(i["loading"] == "lazy" and i["alt"] == "" and i["style"] == "aspect-ratio: 640 / 360" for i in out["imgs"])
    assert out["head0"] == "Views used (3)"  # no toggle before place
    assert out["hint0"] == ["★ view 0"]
    time1, msg1, calls, at1, at0, current = out["seek1"]
    assert time1 == pytest.approx(3.801) and calls == ["pause", "scroll"] and at1 == ["at"] and at0 == []
    assert msg1 == "View 1, frame 40, at 3.8 s" and current == ["false", "true", "false"]
    assert out["seek2"][0] == pytest.approx(29.901)
    assert out["seek2"][1] == "View 2, frame 300, at 29.9 s"  # the frame that the preview kept
    assert out["seek0"][0] == pytest.approx(15.001)
    assert out["seek0"][1:] == ["View 0, frame 152, at 15.0 s", ["at"], [], ["true", "false", "false"]]
    # The same video and images, still at view 0; the heading gains the toggle.
    assert out["placed"][:6] == [1, 3, 2, ["at"], "true", "Views used (3)Mesh fit"]
    assert out["placed"][6] == "View 0, frame 152, at 15.0 s"
    assert out["hint1"] == ["★ view 0"]  # the overlay's colors only while it shows
    assert out["overlay"] == ["true", [f"/api/jobs/j/file/run/overlay_{i}.png" for i in range(3)],
                              ["★ view 0. Red: mesh. Green: mask outline."]]
    assert out["photos"] == ["false", "/api/jobs/j/file/run/input/images/2.png", ["★ view 0"]]
    assert out["again"] is True
    assert out["m"][0] == pytest.approx(30.1333 - 0.2 + 0.001) and out["m"][1] == "View 2, at 29.9 s"
    assert out["k"] == {"video": False, "tags": ["div"] * 3, "alt": "", "toggle": False,
                        "p": ["scan.bag (rosbag, camera topic /cam/image_rect)", "No preview video: OSError: disk full", "★ view 0"]}
    assert out["s"]["p"][:2] == ["capture.zip (Stray capture)", "No preview video: ValueError: no libx264"]
    assert out["s"]["video"] is False and out["s"]["tags"] == ["div"] * 3
    assert out["g"]["p"][:2] == ["table.MOV (video)", "Preview video missing."]
    assert out["slow"] == [True, 3, True]


def test_app_js_input_panel_follows_the_selection():
    """refresh() draws the Inputs tab's video once the frames are read, and its views once they are selected (here
    after a failure at sam3d). While a new selection runs the old views go, with a sentence in their place, and the
    video stays. On a running job's page, the event that says select is done draws the views (by one refresh), and
    later events do not draw them again. The event that says place is done adds the mesh fit toggle (by one
    refresh) and keeps the video."""
    failed = input_job("j")
    pending = input_job("j", select="running", sam3d="pending", status="running", current="select", error=None)
    running = input_job("j", sam3d="running", status="running", current="sam3d", error=None)
    placed = input_job("j", sam3d="done", status="running", current="agent", error=None,
                       files=["selection.json", "run/placement.json"])
    placed["progress"] = [{**x, "status": "done", "ended": 1500.0} if x["name"] == "place" else x for x in placed["progress"]]
    out = run_form_js(ELEMENTS + """
        (async () => {
          const out = {}, jobs = { failed: """ + json.dumps(failed) + """, pending: """ + json.dumps(pending) + """,
            running: """ + json.dumps(running) + """, placed: """ + json.dumps(placed) + """ };
          let now = "failed", selections = 0;
          const fetch0 = fetch, reply = (x) => Promise.resolve({ ok: true, status: 200, json: async () => x });
          globalThis.fetch = (url, opt) => url === "/api/jobs/j" ? reply(jobs[now])
            : url === "/api/jobs/j/file/selection.json" ? (selections++, reply({ ok: true, picks: """ + json.dumps(PICKS) + """ }))
              : fetch0(url, opt);
          const streams = [];
          globalThis.EventSource = class { constructor(url) { this.on = {}; streams.push(this); }
            addEventListener(k, f) { this.on[k] = f; } close() {} };
          const refresh0 = refresh;
          let pending = null, refreshes = 0;
          refresh = () => { refreshes++; return (pending = refresh0()); };
          $("#log").textContent = "a log";
          V = { id: "j", picks: [], view0: null };
          await refresh();
          const videos = () => made.filter((e) => e.tag === "video").length;
          const noViews = () => made.filter((e) => e.tag === "p" && text(e) === NO_VIEWS).length;
          out.failed = [$("#inputbody").hidden, $("#inputnone").hidden, views().map((f) => text(f.kids[1])), videos(), noViews()];
          now = "pending";  // Build again: select runs
          await refresh();
          out.pending = [$("#inputbody").hidden, V.input, V.inputKey, videos(), noViews()];
          made.length = streams.length = 0;
          selections = 0;
          refreshes = 0;
          V = { id: "j", picks: [], view0: null };
          await refresh();
          out.live = [V.selected, V.input, videos(), noViews(), streams.length];
          now = "running";
          const state = (job) => ({ data: JSON.stringify({ status: job.status, current: job.current, error: null, warnings: [],
            key_ok: false, agent: "none", articulated: false, parts_mode: "vlm", parts_review: false, stages: job.progress }) });
          streams[0].on.state(state(jobs.pending));
          out.before = refreshes;
          streams[0].on.state(state(jobs.running));
          await pending;
          out.after = [refreshes, V.selected, selections, videos()];
          streams[0].on.state(state(jobs.running));
          out.again = refreshes;
          now = "placed";
          streams[0].on.state(state(jobs.placed));
          await pending;
          const toggles = () => made.filter((e) => e.tag === "button" && text(e) === "Mesh fit").length;
          out.placed = [refreshes, made.filter((e) => e.tag === "video").length, toggles(), selections];
          streams[0].on.state(state(jobs.placed));
          out.placedAgain = [refreshes, toggles()];
          console.log(JSON.stringify(out));
        })()
        """, {"/api/jobs/j/frames": frames_meta()})
    assert out["failed"] == [False, True, ["view 0, frame 152, 15.2 s", "view 1, frame 40, 4.0 s", "view 2, frame 301, 30.1 s"],
                             1, 0]
    assert out["live"] == [False, None, 1, 1, 1] and out["before"] == 1
    assert out["after"] == [2, True, 1, 1] and out["again"] == 2
    assert out["placed"] == [3, 1, 1, 1] and out["placedAgain"] == [3, 1]


# ---- the job page's tabs: Results, Downloads, Inputs, Settings, Log ----

def glb_bytes(data, rename=None, edit=None, **accessor):
    """A lidar.cloud_glb file as a list of bytes (for a JS Uint8Array), with its JSON chunk changed: rename
    {old node name: new name}, accessor 0's fields set, and edit(doc)."""
    import struct

    n = struct.unpack_from("<I", data, 12)[0]
    doc = json.loads(data[20:20 + n])
    for node in doc["nodes"]:
        node["name"] = (rename or {}).get(node["name"], node["name"])
    doc["accessors"][0].update(accessor)
    if edit:
        edit(doc)
    js = json.dumps(doc).encode()
    js += b" " * (-len(js) % 4)
    rest = data[20 + n:]
    return list(struct.pack("<III", 0x46546C67, 2, 20 + len(js) + len(rest)) + struct.pack("<II", len(js), 0x4E4F534A)
                + js + rest)


def test_app_js_reads_the_lidar_points_file():
    """readCloud reads lidar.cloud_glb's file into Float32Arrays of x, y, z in glTF axes. A set that the file lacks
    is empty, a node of another name (also "constructor") is not read, and a file that is not one, or whose
    accessor runs past its data, throws. orbitStart puts the eye at the origin (the view's camera), looking at the
    median of the used points, else of the scan's points in front of the camera, else 1 m ahead."""
    from harpor.webapp import lidar

    scan, used = [[0.0, 0.0, 1.0], [1.0, 2.0, 3.0]], [[0.5, 0.25, 2.0]]
    both = lidar.cloud_glb({"scan": scan, "used": used})
    files = {"both": list(both), "scan": list(lidar.cloud_glb({"scan": scan})),
             "odd": glb_bytes(both, rename={"used": "constructor"}), "magic": [0] + list(both)[1:],
             "short": list(both)[:-12], "past": glb_bytes(both, count=3)}
    out = run_form_js("""
        const files = """ + json.dumps(files) + """, out = {};
        const read = (k) => { try { const r = readCloud(new Uint8Array(files[k]).buffer); return [Array.from(r.scan), Array.from(r.used),
          r.scan instanceof Float32Array]; } catch (e) { return e.message; } };
        for (const k of Object.keys(files)) out[k] = read(k);
        const eye = (o) => { const cp = Math.cos(o.pitch); return [o.target[0] + o.dist * cp * Math.sin(o.yaw),
          o.target[1] + o.dist * Math.sin(o.pitch), o.target[2] + o.dist * cp * Math.cos(o.yaw)]; };
        const f = (a) => new Float32Array(a);
        const used = orbitStart(f([9, 9, -9]), f([1, 2, -4, 3, -1, -6, 2, 0, -5])), scan = orbitStart(f([0, 0, 5, 1, 1, -2]), f([]));
        out.orbit = [used.target, eye(used), used.dist, scan.target, eye(scan), orbitStart(f([0, 0, 5]), f([])).target];
        console.log(JSON.stringify(out));
        """)
    flip = lambda pts: [c for x, y, z in pts for c in (x, -y, -z)]  # noqa: E731  OpenCV axes to glTF's
    assert out["both"] == [flip(scan), flip(used), True]
    assert out["scan"] == [flip(scan), [], True] and out["odd"] == [flip(scan), [], True]
    assert out["magic"] == out["short"] == out["past"] == "not a LiDAR points file"
    target, at, dist, target2, at2, ahead = out["orbit"]
    assert target == [2, 0, -5] and max(map(abs, at)) < 1e-6 and dist == pytest.approx(math.hypot(2, 5))
    assert target2 == [1, 1, -2] and max(map(abs, at2)) < 1e-6 and ahead == [0, 0, -1]


def compare_glb(n=5, used=2, median=-0.0043, s=1.25, da3=True, gap=0.01):
    """A lidar.cloud_glb file with the comparison: n diff points with gaps of gap m, and a DA3 surface of 7 points."""
    from harpor.webapp import lidar

    a = np.arange(3.0 * n).reshape(n, 3) + [0, 0, 1]
    return lidar.cloud_glb({"scan": a, "used": a[:used], "da3": a[:1].repeat(7, 0) if da3 else np.zeros((0, 3)), "diff": a,
                            "diff_m": np.full(n, gap), "diff_extras": {"s": s, "median_m": median, "used": used}})


def test_app_js_reads_the_comparison():
    """readCloud reads the da3 and diff nodes, row for row, and checks them as it checks the rest: a _DIFF of another
    count or kind, or extras that are missing or odd, throw. A file without them reads as before, and a diff node of
    no used points may have a null median."""
    from harpor.webapp import lidar

    good = compare_glb()
    last = lambda doc: doc["accessors"][-1]  # noqa: E731  _DIFF

    def extras(**kw):
        return lambda doc: next(n for n in doc["nodes"] if n["name"] == "diff")["extras"].update(kw)

    def no_attr(doc):
        del doc["meshes"][-1]["primitives"][0]["attributes"]["_DIFF"]

    def no_extras(doc):
        del next(n for n in doc["nodes"] if n["name"] == "diff")["extras"]

    files = {"good": list(good), "old": list(lidar.cloud_glb({"scan": [[0.0, 0.0, 1.0]]})),
             "null": list(compare_glb(used=0, median=None)),
             "count": glb_bytes(good, edit=lambda d: last(d).update(count=4)),
             "kind": glb_bytes(good, edit=lambda d: last(d).update(type="VEC3", count=1)),
             "type": glb_bytes(good, edit=lambda d: last(d).update(componentType=5125)),
             "past": glb_bytes(good, edit=lambda d: last(d).update(count=500)),
             "no_attr": glb_bytes(good, edit=no_attr), "no_extras": glb_bytes(good, edit=no_extras),
             "s0": glb_bytes(good, edit=extras(s=0)), "s_text": glb_bytes(good, edit=extras(s="1.2")),
             "many": glb_bytes(good, edit=extras(used=6)), "half": glb_bytes(good, edit=extras(used=1.5)),
             "no_median": glb_bytes(good, edit=extras(median_m=None)),
             "text": glb_bytes(good, edit=extras(median_m="0.1")),
             "median_of_none": list(compare_glb(used=0, median=0.1)),
             "lines": glb_bytes(good, edit=lambda d: next(m for m in d["meshes"] if m["name"] == "da3")["primitives"][0].update(mode=1))}
    out = run_form_js("""
        const files = """ + json.dumps(files) + """, out = {};
        for (const k of Object.keys(files)) {
          try {
            const r = readCloud(new Uint8Array(files[k]).buffer);
            out[k] = [r.da3.length / 3, r.diff.length / 3, Array.from(r.delta), r.compare];
          } catch (e) { out[k] = e.message; }
        }
        console.log(JSON.stringify(out));
        """)
    assert out.pop("good") == [7, 5, [pytest.approx(0.01)] * 5, {"s": 1.25, "median": -0.0043, "used": 2}]
    assert out.pop("old") == [0, 0, [], None]
    assert out.pop("null")[3] == {"s": 1.25, "median": None, "used": 0}
    assert set(out.values()) == {"not a LiDAR points file"}, out


def test_app_js_compare_with_da3():
    """The box shows once a view read has a diff set, and stays on across views, also over a view with none. On, the
    viewer gets it for each view, and the legend says what the set shows, with the fixed scale; a Stray view has no
    surface to name. A view without the data says so and how to get it. Off, the legend is as before. While a view
    loads, the box leaves "loading" alone."""
    from harpor.webapp import lidar

    old = lidar.cloud_glb({"scan": [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], "used": [[0.0, 0.0, 1.0]]})
    glbs = {"/api/jobs/j/file/run/lidar/0.glb": list(old), "/api/jobs/j/file/run/lidar/1.glb": list(compare_glb()),
            "/api/jobs/j/file/run/lidar/2.glb": list(compare_glb(median=-0.00001, s=1.2, da3=False)),
            "/api/jobs/j/file/run/lidar/3.glb": list(compare_glb(used=0, median=None))}
    files = ["selection.json", *(k.split("/file/")[1] for k in glbs)]
    lid = {"kind": "rosbag", "topic": "/lidar/points"}
    picks = [*PICKS, {"name": "f00400", "frame": 400, "time_s": 40.0, "q": 0.6}]
    out = run_form_js(ELEMENTS + """
        (async () => {
          const out = {}, job = """ + json.dumps(input_job("j", lidar=lid, files=files)) + """, glbs = """ + json.dumps(glbs) + """;
          const fetch0 = fetch;
          let hold = null;
          globalThis.fetch = async (url, opt) => {
            if (hold) await hold;
            return url in glbs ? { ok: true, status: 200, arrayBuffer: async () => new Uint8Array(glbs[url]).buffer } : fetch0(url, opt);
          };
          const g = { shows: [], toggles: [], show(d) { this.shows.push([d.da3.length / 3, d.diff.length / 3]); },
            compare(on) { this.toggles.push(on); }, reset() {}, draw() {}, free() {} };
          cloudGL = (canvas, on) => { g.on = on; return g; };
          const settle = async () => { for (let k = 0; k < 40; k++) await null; };
          const one = (pred) => made.filter(pred).pop();
          V = { id: "j" };
          await renderInput(job);
          const legend = one((e) => e.className === "legend"), select = one((e) => e.tag === "select");
          const label = one((e) => e.tag === "label" && text(e) === "Compare with DA3"), box = one((e) => e.tag === "input");
          const pick = async (i) => { select.onchange({ target: { value: i } }); await settle(); return text(legend); };
          const tick = (on) => { box.onchange({ target: { checked: on } }); return text(legend); };
          out.box = [box.attrs.type, label.hidden];
          g.on.sized();
          await settle();
          out.old = [text(legend), label.hidden];
          out.plain = [await pick("1"), label.hidden];
          out.on = tick(true);
          out.toggles = [...g.toggles];
          out.stray = await pick("2");
          out.none = await pick("0");
          out.zero = await pick("3");
          out.back = await pick("1");
          out.shows = [...g.shows];
          out.off = tick(false);
          out.toggles2 = [...g.toggles];
          out.hidden = label.hidden;
          out.shown = made.some((e) => e.className === "sq diff");
          tick(true);
          let go;
          hold = new Promise((r) => { go = r; });
          cloudNow.data.clear();
          select.onchange({ target: { value: "2" } });
          await settle();
          const late = tick(false);
          go();
          hold = null;
          await settle();
          out.loading = [late, text(legend)];
          console.log(JSON.stringify(out));
        })()
        """, {"/api/jobs/j/file/selection.json": {"ok": True, "picks": picks}})
    assert out["box"] == ["checkbox", True]
    plain = "Whole scan (5) Used for the scale (2)"
    assert out["old"] == ["Whole scan (2) Used for the scale (1)", True]  # no diff set yet: no box
    assert out["plain"] == [plain, False] and out["hidden"] is False
    colors = "Blue: LiDAR nearer than DA3. Red: LiDAR farther. Full color at ±15 mm"
    assert out["on"] == f"DA3 surface (7) {colors} Scale ×1.250 Median of the 2 used points: -4.3 mm"
    assert out["toggles"] == [True] and out["toggles2"] == [True, False]
    assert out["stray"] == f" {colors} Scale ×1.200 Median of the 2 used points: +0.0 mm"  # no surface, never "-0.0"
    assert out["none"] == "Whole scan (2) Used for the scale (1) No comparison for this view."
    assert out["zero"] == f"DA3 surface (7) {colors} Scale ×1.250 No used points"
    assert out["back"] == out["on"] and out["off"] == plain
    assert out["shows"] == [[0, 0], [7, 5], [0, 5], [0, 0], [7, 5], [7, 5]]
    assert out["shown"] is True
    assert out["loading"] == ["loading", plain]


def test_cloud_gl_colors_the_gaps_on_a_fixed_scale():
    """The real cloudGL on a stand-in context. Off, the scan and the used points draw as before. On, the da3 set takes
    the scan's place, with the depth test, and the diff points the used points', without it, round and larger, with
    their gaps as an attribute that is off again for the next frame. The span is 15 mm in every view, whatever its
    gaps, and the colors are the tokens of the moment. A gap's sign picks the color, and the color is full at the span
    and past it. A view without a diff set keeps its scan and used points. Every buffer goes with free()."""
    out = run_form_js("""
        const tokens = { "--viewer-bg": "#efeeea", "--line-strong": "#8d8a83", "--fg": "#1c1c1a", "--accent": "#1e4f85",
          "--near": "#2166ac", "--far": "#b2182b" };
        globalThis.getComputedStyle = () => ({ getPropertyValue: (k) => tokens[k] });
        globalThis.ResizeObserver = class { observe() {} disconnect() {} };
        window.devicePixelRatio = 1;
        const calls = [], shaders = [], base = { createProgram: () => "program", createShader: (type) => type,
          getProgramParameter: () => true, getUniformLocation: (p, u) => u, getAttribLocation: (p, a) => a,
          getParameter: () => [1, 64], getExtension: () => null, createBuffer: () => ({}),
          shaderSource: (type, src) => shaders.push(src) };
        const gl = new Proxy(base, { get: (t, k) => k in t ? t[k] : k === k.toUpperCase() ? k
          : (...a) => { calls.push([k, ...a]); } });
        const canvas = { getContext: () => gl, addEventListener() {}, clientWidth: 400, clientHeight: 300, width: 400, height: 300 };
        const C = cloudGL(canvas, { sized() {}, lost() {} });
        // The draws since the last call, each with the state that it saw.
        const frame = () => {
          const out = [], counts = [], on = new Set(), u = {};
          let depth = false;
          for (const [k, a, b, c] of calls.splice(0)) {
            if (k === "uniform1f" || k === "uniform3fv") u[a] = k === "uniform1f" ? b : Array.from(b).map((x) => Math.round(255 * x));
            else if (k === "enable" || k === "disable") depth = k === "enable";
            else if (k === "enableVertexAttribArray") on.add(a);
            else if (k === "disableVertexAttribArray") on.delete(a);
            else if (k === "drawArrays") {
              out.push([a, depth, u.size, u.span || 0, u.color, on.has("d"), u.lo, u.hi]);
              counts.push(c);
            }
          }
          return { draws: out, counts, attrs: [...on] };
        };
        const f = (a) => new Float32Array(a), none = f([]);
        const view = (gap) => ({ scan: f([0, 0, -1, 1, 0, -1, 0, 1, -1]), used: f([0, 0, -1]), da3: f(Array(15).fill(-2)),
          diff: f([0, 0, -1, 1, 1, -1]), delta: f([gap, -gap]), compare: { s: 1.2, median: 0, used: 1 } });
        const old = { scan: f([0, 0, -1]), used: f([0, 0, -1]), da3: none, diff: none, delta: none, compare: null };
        const out = { shaders };
        C.show(view(0.001), 4 / 3);
        out.off = frame();
        C.compare(true);
        out.on = frame();
        C.show(view(0.5), 4 / 3);  // gaps 500 times as large: the same span, and the box stays on across views
        out.wide = frame();
        tokens["--fg"] = "#e7e5e0";
        C.draw();
        out.dark = frame();
        C.show(old, 4 / 3);
        out.old = frame();
        C.free();
        out.buffers = calls.filter((c) => c[0] === "deleteBuffer").length;
        // The vertex shader's color lines, run as JS: GLSL's mix and clamp, on the colors of the tokens.
        const lines = /float t = [^}]*/.exec(shaders[0])[0].replace("float t", "const t").replace("c = mix", "return mix");
        const shade = new Function("d", "span", "color", "lo", "hi", "clamp", "mix", "abs", lines);
        const clamp = (x, a, b) => Math.min(Math.max(x, a), b), mix = (a, b, t) => a.map((x, i) => x + (b[i] - x) * t);
        const paint = (d, span) => shade(d, span, [28, 28, 26], [33, 102, 172], [178, 24, 43], clamp, mix, Math.abs);
        out.shades = [-0.06, -0.03, -0.015, 0, 0.015, 0.03, 0.06].map((d) => paint(d, 0.03));
        console.log(JSON.stringify(out));
        """)
    grey, ink, accent, near, far = [141, 138, 131], [28, 28, 26], [30, 79, 133], [33, 102, 172], [178, 24, 43]
    scan = ["POINTS", True, 2, 0.015, grey, False]
    cam = ["LINES", True, 1, 0.015, ink, False]
    assert out["off"]["draws"] == [[*scan, None, None], [*cam, None, None], ["POINTS", False, 4, 0.015, accent, False, None, None]]
    both = [[*scan, None, None], [*cam, None, None], ["POINTS", False, 4, 0.015, ink, True, near, far]]
    assert out["on"]["draws"] == both and out["wide"]["draws"] == both
    assert out["off"]["counts"] == [3, 16, 1]  # the scan, the camera's pyramid and the used points
    assert out["on"]["counts"] == out["wide"]["counts"] == [5, 16, 2]  # da3 and diff in their place
    assert out["old"]["counts"] == [1, 16, 1]
    assert out["on"]["attrs"] == out["wide"]["attrs"] == ["p"]  # d is off again
    assert out["dark"]["draws"][-1][4] == [231, 229, 224]
    assert out["old"]["draws"][0][4] == grey and out["old"]["draws"][-1][:6] == ["POINTS", False, 4, 0.015, accent, False]
    assert out["buffers"] == 6
    vertex, fragment = out["shaders"]
    assert "uniform vec3 color" not in fragment and "color" in vertex  # a uniform in both needs one precision
    assert "varying vec3 c" in vertex and "varying vec3 c" in fragment
    half = [[(a + b) / 2 for a, b in zip(ink, end)] for end in (near, far)]
    assert out["shades"] == [near, near, half[0], ink, half[1], far, far]


def test_app_js_lidar_start_view_shows_the_whole_frame():
    """From the view's camera, the pyramid's frame fits the viewer's field on a wide canvas and on a phone's tall one
    (fieldOf widens the field there), for a 3:2 and a 16:9 frame; on a wide canvas the field stays 50°."""
    out = run_form_js("""
        const out = [];
        for (const [w, h] of [[760, 420], [343, 365], [343, 600]]) for (const fa of [1.5, 16 / 9]) {
          const M = perspective(fieldOf(fa, w / h), w / h, 0.01, 100), c = pyramid(1, fa);
          let far = 0;
          for (let i = 0; i < c.length; i += 3) {
            if (c[i + 2] === 0) continue;  // the apex, at the eye
            const x = M[0] * c[i], y = M[5] * c[i + 1], wc = M[11] * c[i + 2];
            far = Math.max(far, Math.abs(x / wc), Math.abs(y / wc));
          }
          out.push([w, fa, far, fieldOf(fa, w / h) * 180 / Math.PI]);
        }
        console.log(JSON.stringify(out));
        """)
    for w, fa, far, fovy in out:
        assert 0.5 < far <= 0.76, (w, fa, far)
        if w == 760:
            assert fovy == pytest.approx(50)


def test_app_js_lidar_points_in_the_inputs_tab():
    """A LiDAR job whose views are selected shows "LiDAR points" under the views: a picker of the views with points,
    Reset, the canvas (a 3D view that takes the keys, described by its legend) and a legend that says "loading" until
    the file is read, then the counts and the view's scale as the review gives it (lidarViewText). Nothing is fetched
    before the canvas has a size; a view's file is fetched once. A job without LiDAR draws no section; one without
    points says so; without WebGL a line offers the file. clearInput frees the viewer, and new point files draw it
    again. A job whose scale failed has no views used: the picker holds the pool's views (server.pool_clouds)."""
    from harpor.webapp import lidar

    rng = np.random.default_rng(0)
    glbs = {f"/api/jobs/j/file/run/lidar/{i}.glb": list(lidar.cloud_glb({"scan": rng.normal(size=(n, 3)) + [0, 0, 2],
                                                                        "used": rng.normal(size=(k, 3)) + [0, 0, 2]}))
            for i, n, k in ((0, 1234, 280), (2, 50, 9))}
    rec = {"version": 1, "ok": True, "s": 1.07, "reason": None, "combine": {"views": 3, "views_used": 2},
           "views": [{"name": "f00152", "s": 1.0498, "points_used": 280, "status": "used"},
                     {"name": "f00040", "s": None, "points_used": 0, "status": "no_samples"},
                     {"name": "f00301", "s": 1.2101, "points_used": 9, "status": "outlier"}]}
    files = ["selection.json", "pool/lidar.json", "run/lidar/0.glb", "run/lidar/2.glb"]
    lid = {"kind": "rosbag", "topic": "/lidar/points"}
    jobs_ = {"j": input_job("j", lidar=lid, files=files), "more": input_job("j", lidar=lid, files=files + ["run/lidar/1.glb"]),
             "n": input_job("n", files=files), "e": input_job("e", lidar=lid, files=files[:2]),
             "g": input_job("g", lidar=lid, files=files),
             "p": input_job("p", lidar=lid, select="pending", sam3d="pending",
                            files=["pool/lidar.json", "lidar/cloud/f00152.glb", "lidar/cloud/f00040.glb"])}
    glbs["/api/jobs/p/file/lidar/cloud/f00040.glb"] = glbs["/api/jobs/j/file/run/lidar/2.glb"]
    replies = {f"/api/jobs/{i}/file/selection.json": {"ok": True, "picks": PICKS} for i in "jneg"}
    replies.update({f"/api/jobs/{i}/file/pool/lidar.json": rec for i in "jegp"})
    out = run_form_js(ELEMENTS + """
        (async () => {
          const out = {}, jobs = """ + json.dumps(jobs_) + """, glbs = """ + json.dumps(glbs) + """, gls = [];
          const fetch0 = fetch;
          globalThis.fetch = async (url, opt) => url in glbs ? (requests.push({ url, method: "GET" }),
            { ok: true, status: 200, arrayBuffer: async () => new Uint8Array(glbs[url]).buffer }) : fetch0(url, opt);
          const gl0 = cloudGL = (canvas, on) => { const g = { on, canvas, shows: [], resets: 0, freed: 0, show(d, a) {
            this.shows.push([d.scan.length / 3, d.used.length / 3, a]); }, reset() { this.resets++; }, draw() {},
            free() { this.freed++; } }; gls.push(g); return g; };
          const settle = async () => { for (let k = 0; k < 30; k++) await null; };
          const one = (pred) => made.filter(pred).pop();
          const glbGets = () => requests.filter((r) => r.url.endsWith(".glb")).map((r) => r.url.split("/file/")[1]);
          V = { id: "j" };
          await renderInput(jobs.j);
          const legend = one((e) => e.className === "legend"), canvas = one((e) => e.tag === "canvas");
          const select = one((e) => e.tag === "select"), bar = one((e) => e.className === "cloudbar");
          out.bar = [text(bar), select.kids.map((o) => [o.attrs.value, text(o), o.attrs.selected])];
          out.canvas = [canvas.className, { ...canvas.attrs }];
          out.before = [text(legend), glbGets()];
          gls[0].on.sized();
          await settle();
          out.view0 = [text(legend), glbGets(), [...gls[0].shows]];
          select.onchange({ target: { value: "2" } });
          await settle();
          out.view2 = [text(legend), canvas.attrs["aria-label"], glbGets(), gls[0].shows.length];
          select.onchange({ target: { value: "0" } });
          await settle();
          select.onchange({ target: { value: "2" } });
          await settle();
          out.again = [glbGets(), gls[0].shows.length, text(legend)];
          one((e) => e.tag === "button" && text(e) === "Reset").onclick();
          out.reset = gls[0].resets;
          await renderInput(jobs.more);  // a new point file draws the section again, with a new viewer
          out.more = [gls.length, gls[0].freed, one((e) => e.tag === "select").kids.map(text)];
          clearInput();
          out.cleared = gls[1].freed;
          for (const id of ["n", "e"]) {
            made.length = 0;
            V = { id };
            await renderInput(jobs[id]);
            out[id] = [made.some((e) => e.tag === "canvas"), made.filter((e) => e.tag === "p").map(text)];
          }
          cloudGL = () => null;
          made.length = 0;
          V = { id: "g" };
          await renderInput(jobs.g);
          const msg = one((e) => e.className === "msg"), link = msg.kids.find((e) => e.tag === "a");
          out.g = [text(msg), { ...link.attrs }, made.some((e) => e.tag === "canvas" && e.kids.length)];
          cloudGL = gl0;
          made.length = 0;
          V = { id: "p" };
          await renderInput(jobs.p);
          const pool = one((e) => e.tag === "select");
          gls[gls.length - 1].on.sized();
          await settle();
          out.p = [made.filter((e) => e.tag === "p" && e.className === "hint").map(text), pool.kids.map((o) => [o.attrs.value, text(o)]),
            one((e) => e.tag === "canvas").attrs["aria-label"], text(one((e) => e.className === "legend")), glbGets().pop()];
          console.log(JSON.stringify(out));
        })()
        """, replies)
    assert out["bar"] == ["LiDAR pointsView view 0view 2ResetCompare with DA3", [["0", "view 0", ""], ["2", "view 2", None]]]
    assert out["canvas"] == ["cloud", {"role": "application", "aria-roledescription": "3D view", "tabindex": "0",
                                       "aria-describedby": "cloudlegend", "aria-label": "LiDAR points of view 0"}]
    assert out["before"] == ["loading", []]  # no size yet: nothing fetched
    assert out["view0"] == ["Whole scan (1,234) Used for the scale: ×1.05 (280 pts)", ["run/lidar/0.glb"],
                            [[1234, 280, 640 / 360]]]
    assert out["view2"] == ["Whole scan (50) Used for the scale: ×1.21 (9 pts, outlier)", "LiDAR points of view 2",
                            ["run/lidar/0.glb", "run/lidar/2.glb"], 2]
    assert out["again"] == [["run/lidar/0.glb", "run/lidar/2.glb"], 4, out["view2"][0]]  # fetched once per view
    assert out["reset"] == 1
    assert out["more"] == [2, 1, ["view 0", "view 1", "view 2"]] and out["cleared"] == 1
    assert out["n"] == [False, ["★ view 0"]]  # no LiDAR: no section
    assert out["e"] == [False, ["★ view 0", "No LiDAR points to show."]]
    assert out["g"][0] == "Cannot draw the points here: download them."
    assert out["g"][1] == {"href": "/api/jobs/g/file/run/lidar/0.glb", "download": "g_lidar_view0.glb"}
    assert out["p"] == [["No views selected yet."], [["f00040", "f00040"], ["f00152", "f00152"]], "LiDAR points of f00040",
                        "Whole scan (50) Used for the scale: no points", "lidar/cloud/f00040.glb"]


# For FORM_HARNESS: the tab buttons keep their attributes in tabAttrs and their data-tab, and record focus calls in
# focusedTabs; tabs() is the tab shown, the picked tab, the panels that show, the tabs with aria-selected="true",
# each tab's tabindex (in tab order) and the hash.
TAB_SETUP = """
const tabAttrs = {}, focusedTabs = [];
for (const n of TAB_NAMES) {
  const t = $("#tab-" + n);
  tabAttrs[n] = {};
  t.setAttribute = (k, v) => { tabAttrs[n][k] = v; };
  t.dataset = { tab: n };
  t.focus = () => focusedTabs.push(n);
}
const tabs = () => ({ tab: V.tab, pick: V.pick, shown: TAB_NAMES.filter((n) => $("#" + TABS[n]).hidden === false),
  selected: TAB_NAMES.filter((n) => tabAttrs[n]["aria-selected"] === "true"), roving: TAB_NAMES.map((n) => $("#tab-" + n).tabIndex),
  hash: location.hash });
// A state event of a video job with that status: topics and frames done, the rest pending (or as given).
const st = (status, done) => ({ status, warnings: [], agent: "none", key_ok: true,
  stages: ["topics", "frames", ...BUILD].map((name) => ({ name, status: (done || ["topics", "frames"]).includes(name) ? "done" : "pending" })) });
const clickTab = (n) => $("#tabs").onclick({ target: { closest: (sel) => sel === "[role=tab]" ? $("#tab-" + n) : null } });
"""
TAB_ORDER = ["results", "downloads", "inputs", "settings", "log"]


def tab_job(job_id, status, files=("selection.json",), **stage_status):
    """GET /api/jobs/<job_id> of a video job (input_job's) with that status; stage_status gives the stages' statuses
    (the others are pending)."""
    job = input_job(job_id, status=status, error=None, current=None, files=list(files))
    names = [x["name"] for x in job["progress"]]
    job["progress"] = [{"name": n, "status": stage_status.get(n, "pending")} for n in names]
    job["stages"] = {n: {"status": stage_status.get(n, "pending")} for n in names}
    return job


def page_ids():
    """index.html's elements that have an id, in page order: [(tag, attributes)]."""
    from html.parser import HTMLParser

    class Ids(HTMLParser):
        found = []

        def handle_starttag(self, tag, attrs):
            if dict(attrs).get("id"):
                self.found.append((tag, dict(attrs)))

    p = Ids()
    p.found = []
    p.feed((server.STATIC / "index.html").read_text())
    return p.found


def test_index_html_tabs_and_their_panels():
    """The job page: the waiting panels, then the tab bar (Results, Downloads, Inputs, Settings, Log) with the
    WAI-ARIA roles and links, then the five panels in tab order, hidden until app.js picks one. Each panel has the
    few words it shows while it has nothing (the Settings one is set by app.js). The Results panel holds the viewers
    and the facts, with no download link; the Settings heading is for screen readers only."""
    els = page_ids()
    order = [a["id"] for _, a in els]
    at = {a["id"]: (tag, a) for tag, a in els}
    panels = ["results", "downloads", "input", "frames", "run"]
    for waiting in ("lidarfail", "topics", "review", "partsreview"):
        assert order.index(waiting) < order.index("tabs"), waiting
    assert [order.index(p) for p in panels] == sorted(order.index(p) for p in panels) and order.index("tabs") < order.index("results")
    assert at["tabs"][1]["role"] == "tablist" and at["tabs"][1]["aria-label"] == "Job sections" and "hidden" in at["tabs"][1]
    html = (server.STATIC / "index.html").read_text()
    import re
    assert re.findall(r'role="tab" id="tab-(\w+)"[^>]*>([^<]+)</button>', html) == [
        ("results", "Results"), ("downloads", "Downloads"), ("inputs", "Inputs"), ("settings", "Settings"),
        ("log", "Log")]
    for name, panel in zip(TAB_ORDER, panels):
        tag, tab = at["tab-" + name]
        assert (tag, tab["type"], tab["role"], tab["data-tab"], tab["aria-controls"], tab["aria-selected"], tab["tabindex"]) == \
            ("button", "button", "tab", name, panel, "false", "-1")
        p = at[panel][1]
        assert (p["role"], p["aria-labelledby"], p["tabindex"], "hidden" in p) == ("tabpanel", "tab-" + name, "0", True)
    for none, text in (("resultsnone", "No results yet."), ("downloadsnone", "No files yet."),
                       ("inputnone", "Input not read yet."), ("framesnone", ""), ("runnone", "Not started yet.")):
        assert re.search(rf'<p id="{none}" class="emptytab">([^<]*)</p>', html).group(1) == text, none
    # The input's facts sit in the Inputs panel, the start form in the Settings panel, the log in Log.
    def inside(child, parent):
        start = html.index(f'id="{parent}"')
        return start < html.index(f'id="{child}"') < html.index(f'id="{panels[panels.index(parent) + 1]}"') \
            if parent != "run" else start < html.index(f'id="{child}"')
    for child in ("topicrow", "hdr", "strayinfo", "videoinfo", "inputvideo", "inputviews", "inputlidar"):
        assert inside(child, "input"), child
    assert order.index("inputviews") + 1 == order.index("inputlidar")  # the LiDAR points under the views
    for child in ("framesum", "inputwarn", "towarn", "startform", "grid", "lidarbox", "start"):
        assert inside(child, "frames"), child
    assert at["hdr"][1]["tabindex"] == at["straywarn"][1]["tabindex"] == "-1"  # "Show" focuses them
    assert at["framesum"][1] == {"id": "framesum", "class": "sr-only"}  # a heading for a screen reader, not a focus target
    assert inside("gridsum", "frames") and at["gridsum"][0] == "summary"  # "Pick frames manually" focuses it
    for child in ("runsum", "stages", "log"):
        assert inside(child, "run"), child
    assert order.index("retry") < order.index("tabs") and order.index("cancel") < order.index("tabs")  # in the job's header
    # Results: the 360 video, the viewers, then the facts and the part masks, and no download links.
    assert [order.index(x) for x in ("resultsbody", "spin", "viewers", "facts2", "partchecks")] == \
        sorted(order.index(x) for x in ("resultsbody", "spin", "viewers", "facts2", "partchecks"))
    assert order.index("spin") == order.index("resultsbody") + 1
    # The 360 video is asked for in the settings, off at first, before the review box.
    assert inside("turntable", "frames") and order.index("lidarbox") < order.index("turntable") < order.index("pause")
    assert at["turntable"][0] == "input" and at["turntable"][1]["type"] == "checkbox" and "checked" not in at["turntable"][1]
    assert '<label class="check"><input type="checkbox" id="turntable"> Make a 360 video</label>' in html
    assert inside("viewers", "results") and inside("downloadsbody", "downloads") and "todl" not in html
    # The part list's Undo edits is a full-height button, like the review bar's.
    assert at["partsreset"][1]["class"] == "secondary"
    # The ring's legend: short marks on screen, the same in words for a screen reader, with the views it leaves out.
    legend = re.search(r'<figure class="ringbox">.*?<figcaption class="hint">(.*?)</figcaption>', html, re.S).group(1)
    assert ('<span aria-hidden="true">Top view: ● picked, ■ always use, ○ available, × left out, ★ view 0, + object'
            '</span>') in legend
    words = " ".join(re.search(r'<span class="sr-only">(.*?)</span>', legend, re.S).group(1).split())
    assert all(w in words for w in ("A filled dot is a picked view", "a square a pick set to always use",
                                    "a hollow dot a view that is not picked", "a cross a view left out",
                                    "a star view 0", "a plus the object",
                                    "Views set to never use and views without a mask are not drawn."))


def test_app_js_tabs_follow_the_status():
    """The tab that shows first follows the job's status. Only the shown tab has aria-selected and tabindex 0
    (a roving tabindex), and only its panel shows. A tab that the user picks stays while the status stays (a
    refresh draws the same status again) or moves within a run (queued to running), goes into the hash, and goes
    when the job stops or starts. When the panel that hides holds the focus, or the focus is on a tab that is no
    longer selected, the new tab takes it. The preview video stops when the Inputs tab hides. Every hash change is
    a replaceState."""
    out = run_form_js(TAB_SETUP + """
        (() => {
          const out = { first: {} };
          for (const status of ["ready", "analyzing", "queued", "running", "review", "choose_topic", "done", "failed", "cancelled",
            "uploading"]) {
            V = { id: "j" };
            location._hash = "#/job/j";
            history.state = null;
            renderState(status === "uploading" ? { ...st(status), stages: st(status).stages.map((x) => ({ ...x, status: "pending" })) }
              : st(status));
            out.first[status] = { ...tabs(), bar: $("#tabs").hidden, none: $("#runnone").hidden, body: $("#runbody").hidden };
          }
          // One page through a build.
          let pauses = 0;
          V = { id: "j", video: { pause: () => { pauses++; } } };
          location._hash = "#/job/j";
          historyCalls.length = 0;
          renderState(st("ready"));
          clickTab("inputs");
          out.picked = [tabs(), history.state];
          renderState(st("ready"));  // a refresh: the same status
          out.again = [tabs(), pauses];
          clickTab("settings");  // the Inputs tab hides: its video stops
          out.paused = pauses;
          document.activeElement = $("#start");  // the form had the focus: its panel hides, so the new tab takes it
          $("#frames").contains = (x) => x === $("#start");
          renderState(st("queued"));
          out.queued = [tabs(), history.state, focusedTabs.splice(0)];
          clickTab("results");
          out.clicked = [tabs(), focusedTabs.splice(0)];  // a click moves no focus
          renderState(st("running"));  // the job starts: the same phase, so the pick stays
          out.running = [tabs(), history.state, focusedTabs.splice(0)];
          clickTab("log");
          document.activeElement = $("#tab-log");  // the focus is on the picked tab when the job ends
          $("#tabs").contains = (x) => x === $("#tab-log");
          renderState(st("done", ["topics", "frames", ...BUILD]));
          out.done = [tabs(), focusedTabs.splice(0), pauses];
          out.history = historyCalls;
          out.hashSets = hashSets;
          console.log(JSON.stringify(out));
        })()
        """)
    expect = {"ready": "settings", "analyzing": "log", "queued": "log", "running": "log", "review": "inputs",
              "choose_topic": "inputs", "done": "results", "failed": "log", "cancelled": "log", "uploading": "log"}
    for status, name in expect.items():
        k = TAB_ORDER.index(name)
        assert out["first"][status] == {"tab": name, "pick": None, "shown": [name], "selected": [name],
                                         "roving": [0 if i == k else -1 for i in range(5)], "hash": "#/job/j", "bar": False,
                                         "none": status != "uploading", "body": status == "uploading"}, status
    picked, state = out["picked"]
    assert (picked["tab"], picked["pick"], picked["shown"], picked["hash"]) == ("inputs", "inputs", ["inputs"], "#/job/j/inputs")
    assert state == {"tab": "inputs", "status": "ready"} and out["again"] == [picked, 0] and out["paused"] == 1
    queued, state, focused = out["queued"]
    assert (queued["tab"], queued["pick"], queued["hash"], state, focused) == ("log", None, "#/job/j", None, ["log"])
    assert out["clicked"][0]["tab"] == "results" and out["clicked"][1] == []
    running, state, focused = out["running"]
    assert (running["tab"], running["pick"], running["shown"], running["hash"], state, focused) == \
        ("results", "results", ["results"], "#/job/j/results", {"tab": "results", "status": "queued"}, [])
    done, focused, pauses = out["done"]
    assert (done["tab"], done["pick"], done["selected"], done["roving"]) == ("results", None, ["results"], [0, -1, -1, -1, -1])
    assert focused == ["results"] and pauses == 1  # the selected tab took the focus from the tab that is no longer
    # The first draw, the pick, (nothing for the same status), the pick, the change, the pick, (nothing for the
    # same phase), the pick, the change.
    assert out["history"] == [["replace", u] for u in ("#/job/j", "#/job/j/inputs", "#/job/j/settings", "#/job/j",
                                                      "#/job/j/results", "#/job/j/log", "#/job/j")]
    assert out["hashSets"] == []  # no history step per tab


def test_app_js_tabs_keep_the_hash():
    """The hash names the picked tab: #/job/<id>/<tab>, and the old #/job/<id> opens the status's own tab. A reload
    keeps the pick unless the job's status changed since (history.state says which status it was picked in), where
    queued, running and analyzing count as one; a pasted link with a tab shows it. A typed hash on the open page
    changes only the tab. A refresh keeps the pick."""
    done = tab_job("d", "done", files=["selection.json", "run/posed.glb"],
                   **{n: "done" for n in ("frames", "extract", "masks", "prep", "da3", "select", "sam3d", "place", "agent")},
                   lidar="skipped")
    replies = {"/api/jobs/d": done, "/api/jobs/d/frames": frames_meta(),
               "/api/jobs/d/file/selection.json": {"ok": True, "picks": PICKS, "anchor": "f00152"}}
    out = run_form_js(ELEMENTS + TAB_SETUP + """
        (async () => {
          const out = {};
          out.parse = ["#/job/abc", "#/job/abc/log", "#/job/abc/inputs", "#/job/abc/bogus", "#/job/abc/constructor", "#/", "",
            "#/job/a/b/c", "#/job/abc/downloads"].map(jobHash);
          const open = async (hash, state) => {  // a reload of the page at this hash, with this history.state
            location._hash = hash;
            history.state = state;
            await openJob(...jobHash(location.hash));
            return [V.tab, V.pick, location.hash, history.state, $("#tabs").hidden];
          };
          out.reload = await open("#/job/d/settings", { tab: "settings", status: "done" });
          out.stale = await open("#/job/d/settings", { tab: "settings", status: "running" });
          out.link = await open("#/job/d/settings", null);
          out.old = await open("#/job/d", null);
          out.bogus = await open("#/job/d/bogus", null);
          // A typed hash on the open page (hashchange): only the tab changes, and the job does not load again.
          const n = requests.length;
          location._hash = "#/job/d/log";
          route();
          out.typed = [V.tab, V.pick, requests.length - n];
          location._hash = "#/job/d";
          route();
          out.typedOld = [V.tab, V.pick, requests.length - n];
          clickTab("inputs");
          await refresh();
          out.refreshed = [V.tab, V.pick, location.hash];
          // A tab picked while the job waited for the GPU, and a reload once it runs, or once it failed.
          const reload = (status) => {
            V = { id: "q", hashTab: "inputs" };
            location._hash = "#/job/q/inputs";
            history.state = { tab: "inputs", status: "queued" };
            renderState(st(status));
            return [V.tab, V.pick, location.hash];
          };
          out.phase = [reload("running"), reload("failed")];
          out.history = historyCalls.map((x) => x[0]);
          out.hashSets = hashSets;
          console.log(JSON.stringify(out));
        })()
        """, replies)
    assert out["parse"] == [["abc", None], ["abc", "log"], ["abc", "inputs"], ["abc", None], ["abc", None], None, None, None,
                            ["abc", "downloads"]]
    assert out["reload"] == ["settings", "settings", "#/job/d/settings", {"tab": "settings", "status": "done"}, False]
    assert out["stale"] == ["results", None, "#/job/d", None, False]  # picked while it ran; it is done now
    assert out["link"] == ["settings", "settings", "#/job/d/settings", {"tab": "settings", "status": "done"}, False]
    assert out["old"] == ["results", None, "#/job/d", None, False]
    assert out["bogus"] == ["results", None, "#/job/d", None, False]
    assert out["typed"] == ["log", "log", 0] and out["typedOld"] == ["results", None, 0]
    assert out["refreshed"] == ["inputs", "inputs", "#/job/d/inputs"]
    assert out["phase"] == [["inputs", "inputs", "#/job/q/inputs"], ["log", None, "#/job/q"]]
    assert set(out["history"]) == {"replace"} and out["hashSets"] == []


def test_app_js_tabs_keyboard():
    """The tab bar follows the WAI-ARIA tabs pattern: Left and Right go to the tab before or after (and wrap),
    Home and End to the first and last; each key shows its tab at once, focuses it and moves the tabindex 0.
    Other keys, keys with Ctrl, Alt or Meta, and keys outside a tab do nothing. A click shows a tab without
    moving the focus."""
    out = run_form_js(TAB_SETUP + """
        (() => {
          V = { id: "j" };
          location._hash = "#/job/j";
          renderState(st("done", ["topics", "frames", ...BUILD]));
          const key = (from, key, mods) => {
            let prevented = false;
            $("#tabs").onkeydown({ key, target: from ? $("#tab-" + from) : { dataset: {} }, preventDefault: () => { prevented = true; },
              ...mods });
            return [V.tab, focusedTabs.splice(0).join(), prevented];
          };
          const out = { keys: [key("results", "ArrowRight"), key("downloads", "ArrowRight"), key("inputs", "ArrowRight"),
            key("settings", "ArrowRight"), key("log", "ArrowRight"), key("results", "ArrowLeft"), key("downloads", "ArrowLeft"),
            key("log", "Home"), key("results", "End"),
            key("log", "ArrowDown"), key("log", "ArrowLeft", { ctrlKey: true }), key("log", "Home", { altKey: true }),
            key("log", "ArrowRight", { metaKey: true }), key("log", "Enter"), key(null, "ArrowRight")] };
          out.end = tabs();
          clickTab("settings");
          out.click = [tabs(), focusedTabs.splice(0)];
          $("#tabs").onclick({ target: { closest: () => null } });  // the bar around the tabs
          out.between = tabs().tab;
          console.log(JSON.stringify(out));
        })()
        """)
    assert out["keys"] == [["downloads", "downloads", True], ["inputs", "inputs", True], ["settings", "settings", True],
                           ["log", "log", True], ["results", "results", True], ["log", "log", True],
                           ["results", "results", True], ["results", "results", True], ["log", "log", True],
                           ["log", "", False], ["log", "", False], ["log", "", False], ["log", "", False],
                           ["log", "", False], ["log", "", False]]
    assert out["end"] == {"tab": "log", "pick": "log", "shown": ["log"], "selected": ["log"], "roving": [-1, -1, -1, -1, 0],
                          "hash": "#/job/j/log"}
    assert out["click"][0]["tab"] == "settings" and out["click"][0]["roving"] == [-1, -1, -1, 0, -1] and out["click"][1] == []
    assert out["between"] == "settings"


def test_app_js_tabs_say_when_they_are_empty():
    """Each tab shows a few words while it has nothing: Results until a build made them (and while a new build
    runs), Downloads until a file exists (and while a build runs), Inputs until the frames are read, Settings
    until the frames are read and while the job runs, Log until a stage started. After a failed or cancelled job,
    Results and Downloads say that the job stopped. The Inputs tab's views strip says when the views are not
    selected yet."""
    all_done = {n: "done" for n in ("frames", "extract", "masks", "prep", "da3", "select", "sam3d", "place", "agent")}
    cases = {"a": tab_job("a", "analyzing", files=(), frames="running"), "u": tab_job("u", "uploading", files=()),
             "r": tab_job("r", "running", frames="done", extract="done", masks="running"),
             "y": tab_job("y", "ready", files=(), frames="done"),
             "d": tab_job("d", "done", files=["selection.json", "run/posed.glb"], lidar="skipped", **all_done),
             "b": tab_job("b", "running", files=["selection.json", "run/posed.glb"], lidar="skipped",
                          **{**all_done, "sam3d": "running", "place": "pending", "agent": "pending"}),
             "f": tab_job("f", "failed", files=(), frames="done", extract="done", masks="failed"),
             "c": tab_job("c", "cancelled", files=(), frames="done", extract="done", masks="done")}
    replies = {f"/api/jobs/{i}": j for i, j in cases.items()}
    replies.update({f"/api/jobs/{i}/frames": frames_meta() for i in cases})
    replies.update({f"/api/jobs/{i}/file/selection.json": {"ok": True, "picks": PICKS} for i in cases})
    out = run_form_js(ELEMENTS + TAB_SETUP + """
        (async () => {
          globalThis.EventSource = class { addEventListener() {} close() {} };
          const out = { texts: [SETTINGS_LATER, SETTINGS_BUSY, NO_VIEWS, FILES_LATER, FILES_BUSY, RESULTS_LATER,
            RESULTS_STOPPED, FILES_NONE_STOPPED] };
          for (const id of """ + json.dumps(list(cases)) + """) {
            made.length = 0;
            V = { id, picks: [], view0: null };
            await refresh();
            out[id] = { tab: V.tab, results: [$("#resultsnone").hidden, $("#resultsbody").hidden, $("#resultsnone").textContent],
              downloads: [$("#downloadsnone").hidden, $("#downloadsbody").hidden, $("#downloadsnone").textContent],
              inputs: [$("#inputnone").hidden, $("#inputbody").hidden],
              settings: [$("#framesnone").hidden, $("#framesbody").hidden, $("#framesnone").textContent],
              log: [$("#runnone").hidden, $("#runbody").hidden],
              noViews: made.filter((e) => e.tag === "p" && text(e) === NO_VIEWS).length };
            closeStream();
          }
          console.log(JSON.stringify(out));
        })()
        """, replies)
    later, busy, no_views, files_later, files_busy, results_later, results_stopped, files_stopped = out["texts"]
    assert later == "Available after the input is read." and busy == "Locked while the job runs."
    assert no_views == "No views selected yet."
    assert files_later == "No files yet." and files_busy == "Files show when the build stops."
    assert results_later == "No results yet."
    assert results_stopped == "Job stopped before any results." and files_stopped == "Job stopped before any files."
    none, shown = [False, True], [True, False]  # [the sentence hidden, the content hidden]
    assert out["a"] == {"tab": "log", "results": [*none, results_later], "downloads": [*none, files_later], "inputs": none,
                        "settings": [*none, later], "log": shown, "noViews": 0}
    assert out["u"] == {"tab": "log", "results": [*none, results_later], "downloads": [*none, files_later], "inputs": none,
                        "settings": [*none, later], "log": none, "noViews": 0}
    assert out["r"] == {"tab": "log", "results": [*none, results_later], "downloads": [*none, files_busy], "inputs": shown,
                        "settings": [*none, busy], "log": shown, "noViews": 1}
    assert out["y"] == {"tab": "settings", "results": [*none, results_later], "downloads": [*none, files_later],
                        "inputs": shown, "settings": [*shown, busy], "log": shown, "noViews": 1}
    assert out["d"] == {"tab": "results", "results": [*shown, results_later], "downloads": [*shown, files_later],
                        "inputs": shown, "settings": [*shown, busy], "log": shown, "noViews": 0}
    # A new build hides the old results and files.
    assert out["b"]["results"] == [*none, results_later] and out["b"]["downloads"] == [*none, files_busy]
    assert out["b"]["noViews"] == 0
    # A failed or cancelled job says that it stopped: nothing more comes until Retry or a new build.
    for job_id in ("f", "c"):
        assert out[job_id]["results"] == [*none, results_stopped] and out[job_id]["downloads"] == [*none, files_stopped], job_id


def test_app_js_hidden_log_keeps_up_without_layout():
    """While the Log tab is hidden, the live log takes its text and the stage clocks tick without
    reading any layout. When the tab opens, the log is at its end, or where the user left it."""
    out = run_form_js(TAB_SETUP + """
        (() => {
          // The log's layout reads are recorded; each new text makes it 50 px taller.
          const log = $("#log"), reads = [];
          Object.assign(log, { _scrollTop: 0, _clientHeight: 0, _scrollHeight: 0, _text: "" });  // plain values, not stand-ins
          for (const k of ["scrollTop", "clientHeight", "scrollHeight"]) Object.defineProperty(log, k, { configurable: true,
            get() { reads.push(k); return this["_" + k] || 0; }, set(v) { this["_" + k] = v; } });
          Object.defineProperty(log, "textContent", { configurable: true, get() { return this._text; },
            set(v) { this._text = v; this._scrollHeight = (this._scrollHeight || 0) + 50; } });
          log.textContent = "";
          Object.assign(log, { _scrollHeight: 900, _clientHeight: 300 });
          V = { id: "j" };
          location._hash = "#/job/j";
          renderState(st("running", ["topics", "frames", "extract"]));
          clickTab("inputs");
          const clock = { dataset: { started: String(Date.now() / 1000 - 75) }, textContent: "" };
          lists["#stages .secs[data-started]:not([data-ended])"] = [clock];
          reads.length = 0;
          appendLog("masks: frame 1\\nPROGRESS 0.5\\n");
          appendLog("masks: frame 2\\n");
          tick();
          const out = { hidden: [log.textContent, [...reads], clock.textContent] };
          clickTab("log");
          out.opened = [log._scrollTop, reads.includes("scrollHeight")];
          log._scrollTop = 100;  // the user scrolls up
          log.onscroll();
          clickTab("results");
          reads.length = 0;
          appendLog("masks: frame 3\\n");
          out.hidden2 = [reads.length, log.textContent.endsWith("frame 3\\n")];
          clickTab("log");
          out.left = [log._scrollTop, V.logAtEnd];
          log._scrollTop = 750;  // back at the end: new lines keep it there
          log.onscroll();
          appendLog("masks: frame 4\\n");
          out.follow = log._scrollTop;
          console.log(JSON.stringify(out));
        })()
        """)
    assert out["hidden"] == ["masks: frame 1\nmasks: frame 2\n", [], "1 min 15 s"]
    assert out["opened"] == [1000, True]  # two lines came while it was hidden
    assert out["hidden2"] == [0, True]
    assert out["left"] == [100, False]
    assert out["follow"] == 1100


def test_app_js_tabs_wait_for_the_end_of_a_run():
    """A running job's state event that says done changes the header but not the tab: the Log tab stays
    while the stream sends the last lines of the log. The refresh after the stream's end draws the results, then
    shows the Results tab with them. The log's heading says the log stopped."""
    built = {n: "done" for n in ("frames", "extract", "masks", "prep", "da3", "select", "sam3d", "place")}
    files = ["selection.json", "run/placement.json"]
    running = tab_job("r", "running", files=files, lidar="skipped", agent="running", **built)
    done = tab_job("r", "done", files=[*files, "run/posed.glb"], lidar="skipped", agent="done", **built)
    out = run_form_js(ELEMENTS + TAB_SETUP + """
        (async () => {
          const jobs = { running: """ + json.dumps(running) + """, done: """ + json.dumps(done) + """ };
          let now = "running";
          const fetch0 = fetch;
          globalThis.fetch = (url, opt) => url === "/api/jobs/r"
            ? Promise.resolve({ ok: true, status: 200, json: async () => jobs[now] }) : fetch0(url, opt);
          const streams = [];
          globalThis.EventSource = class { constructor() { this.on = {}; streams.push(this); }
            addEventListener(k, f) { this.on[k] = f; } close() {} };
          const refresh0 = refresh;
          let pending = null, refreshes = 0;
          refresh = () => { refreshes++; return (pending = refresh0()); };
          V = { id: "r", picks: [], view0: null };
          location._hash = "#/job/r";
          await refresh();
          const out = { running: [V.tab, streams.length, $("#resultsbody").hidden] };
          Object.assign($("#log"), { textContent: "", scrollTop: 0, clientHeight: 0, scrollHeight: 0 });
          streams[0].on.log({ data: JSON.stringify({ stage: "agent", text: "agent: done\\n" }) });
          out.live = $("#logname").textContent;
          const event = (job) => ({ data: JSON.stringify({ status: job.status, current: null, error: null, warnings: [],
            key_ok: true, agent: "none", articulated: false, parts_mode: "vlm", parts_review: false, stages: job.progress }) });
          streams[0].on.state(event(jobs.done));
          out.state = [V.tab, $("#jnext").textContent, $("#tab-log").tabIndex, refreshes, $("#resultsbody").hidden];
          now = "done";
          streams[0].on.end();
          await pending;
          out.end = [V.tab, V.pick, refreshes, $("#resultsbody").hidden, $("#resultsnone").hidden, location.hash];
          out.logname = $("#logname").textContent;
          console.log(JSON.stringify(out));
        })()
        """, {"/api/jobs/r/frames": frames_meta(), "/api/jobs/r/file/selection.json": {"ok": True, "picks": PICKS}})
    assert out["running"] == ["log", 1, True]
    # A done job has no next step: the Results tab shows itself.
    assert out["state"] == ["log", "", 0, 1, True]
    assert out["end"] == ["results", None, 2, False, True, "#/job/r"]
    # The log heading no longer says live once the stream ends.
    assert out["live"] == "live log: agent" and out["logname"] == "log until the job stopped"


def test_app_js_cancelled_stage_is_not_a_failure():
    """The stage that Cancel stopped (the runner records it as failed with the note "cancelled") shows as cancelled,
    with no note, and the summary counts it as cancelled. A failed stage leaves out its note when the status line's
    error gives it, and keeps any other note."""
    out = run_form_js(ELEMENTS + TAB_SETUP + """
        (() => {
          let lis = [];
          $("#stages").replaceChildren = (...k) => { lis = k; };
          const stages = (note) => [{ name: "frames", status: "done" }, { name: "extract", status: "done" },
            { name: "masks", status: "failed", note }, { name: "prep", status: "pending" }];
          const out = {};
          V = { id: "j" };
          for (const [status, note, error] of [["cancelled", "cancelled", null],
            ["failed", "exit 1: out of memory", "masks: exit 1: out of memory"],
            ["failed", "the server stopped", "the server stopped while this job ran; resume it"]]) {
            renderState({ ...st(status), error, stages: stages(note) }, false);
            const li = lis[2];
            out[note] = [li.className, li.kids.find((k) => /^badge/.test(k.className || "")).className, text(li),
              $("#runsum").textContent];
          }
          console.log(JSON.stringify(out));
        })()
        """)
    assert out["cancelled"] == ["cancelled", "badge cancelled", "masksSAM 3 maskscancelled", "1/3 done · 1 cancelled"]
    assert out["exit 1: out of memory"] == ["failed", "badge failed", "masksSAM 3 masksfailed", "1/3 done · 1 failed"]
    assert out["the server stopped"] == ["failed", "badge failed", "masksSAM 3 masksfailedthe server stopped",
                                         "1/3 done · 1 failed"]


def test_app_js_total_time_under_the_stage_clocks():
    """A finished job shows its total under the stage clocks: the sum of the stages' times, so the wait for the review
    between two stages does not count and a stage that did not run adds nothing. A job that has not finished shows no
    total, and the page has a place for it."""
    out = run_form_js(ELEMENTS + TAB_SETUP + """
        (() => {
          const stages = [{ name: "frames", status: "done", started: 100, ended: 130 },
            { name: "extract", status: "done", started: 1000, ended: 1090 },
            { name: "masks", status: "skipped" },
            { name: "prep", status: "done", started: 1090, ended: 4690 }];
          V = { id: "j" };
          const out = {};
          for (const status of ["queued", "running", "review", "failed", "cancelled", "done"]) {
            renderState({ ...st(status), stages }, false);
            out[status] = $("#runtotal").hidden;
          }
          out.total = $("#runtotalsecs").textContent;
          renderState({ ...st("done"), stages: [{ name: "frames", status: "done", started: 100, ended: 112.5 },
            { name: "extract", status: "skipped" }, { name: "masks", status: "skipped" }, { name: "prep", status: "skipped" }] }, false);
          out.short = $("#runtotalsecs").textContent;
          console.log(JSON.stringify(out));
        })()
        """)
    assert [out[k] for k in ("queued", "running", "review", "failed", "cancelled")] == [True] * 5
    assert out["done"] is False
    assert out["total"] == "1 h 2 min" and out["short"] == "12.5 s"
    index = (server.STATIC / "index.html").read_text()
    assert 'id="runtotal"' in index and 'id="runtotalsecs"' in index


def test_app_js_tabs_take_the_focus_after_actions():
    """When an action hides or disables the focused button, the focus goes where the page goes. After Build, the
    tab that the new status shows takes it; after a refused Start, Start takes it back. "Pick frames manually"
    opens the Settings tab (a pick) with the frames open and focuses its heading. "Change topic" focuses the topic
    table's heading, and "Keep the current topic" gives the focus back to the button, or to the tab shown when the
    Inputs tab does not show. The Settings tab counts the input's warnings, and "Show" opens the Inputs tab at the
    first one."""
    analyzed = {n: "done" for n in ("frames", "extract", "masks", "prep", "da3", "select")}
    review = tab_job("v", "review", lidar="skipped", **analyzed)
    queued = tab_job("v", "queued", lidar="skipped", **analyzed)
    bag = {**tab_job("t", "ready", frames="done"), "kind": "rosbag", "video": {"name": "scan.bag"},
           "source": {"kind": "rosbag", "topic": CAM}}
    bag["stages"]["topics"] = {"status": "done"}
    stray = {**tab_job("s", "ready", frames="done"), "kind": "stray", "video": {"name": "capture.zip"},
             "source": {"kind": "stray"}, "lidar": {"kind": "stray"}}
    stray_video = {"kind": "stray", "hdr": True, "rotation": 0, "n_depth": 19, "n_confidence": 0, "depth_size": [256, 192],
                   "upright_agreement": 1.0, "warnings": ["1 of the 20 frames have no depth image", "no confidence/ images"]}
    replies = {"/api/jobs/v/frames": frames_meta(), "/api/jobs/t/frames": frames_meta(), "/api/jobs/t": bag,
               "/api/jobs/t/file/frames/topics.json": {"topics": [camera_entry({})], "skipped": [], "lidar": [],
                                                       "lidar_skipped": [], "tf": []},
               "/api/jobs/t/start": {"__status": 400, "detail": "the OpenAI key was refused"},
               "/api/jobs/s": stray, "/api/jobs/s/frames": frames_meta(**stray_video)}
    out = run_form_js(ELEMENTS + TAB_SETUP + """
        (async () => {
          globalThis.EventSource = class { addEventListener() {} close() {} };
          globalThis.matchMedia = () => ({ matches: true });
          const jobs = { review: """ + json.dumps(review) + """, queued: """ + json.dumps(queued) + """ };
          let now = "review";
          const fetch0 = fetch;
          globalThis.fetch = (url, opt) => url === "/api/jobs/v"
            ? Promise.resolve({ ok: true, status: 200, json: async () => jobs[now] }) : fetch0(url, opt);
          const focused = [], spy = (sel) => { $(sel).focus = () => focused.push(sel); };
          ["#gridsum", "#start", "#topicshead", "#chtopic", "#hdr", "#straywarn"].forEach(spy);
          const out = {};
          const open = async (id) => {
            V = { id, picks: [], view0: null };
            location._hash = "#/job/" + id;
            $("#log").textContent = "a log";
            await refresh();
          };
          // Build in review: the button is disabled while it runs, so the focus falls to the page.
          await open("v");
          out.review = V.tab;
          document.activeElement = document.body;
          now = "queued";
          await $("#build").onclick();
          out.build = [V.tab, focusedTabs.splice(0), posts.map((x) => x.url)];
          closeStream();
          // Pick frames manually.
          now = "review";
          await open("v");
          $("#gridbox").open = false;
          focused.length = 0;
          await $("#manualstart").onclick();
          out.manual = [V.tab, V.pick, location.hash, $("#gridbox").open, focused.splice(0), V.manual];
          // A refused Start on a ready bag job.
          await open("t");
          Object.assign($("#prompt"), { value: "box" });
          Object.assign($("#nviews"), { value: "4" });
          Object.assign($("#key"), { value: "sk-x" });
          document.activeElement = document.body;
          focused.length = focusedTabs.length = 0;
          await $("#startform").onsubmit({ preventDefault() {} });
          out.refused = [V.tab, focused.splice(0), focusedTabs.splice(0), $("#startmsg").textContent];
          // The camera topic: the table's heading, then back to the button in the Inputs tab.
          clickTab("inputs");
          await $("#chtopic").onclick();
          out.change = [$("#topics").hidden, $("#tkeep").hidden, focused.splice(0)];
          await $("#tkeep").onclick();
          out.keep = [$("#topics").hidden, $("#topicrow").hidden, focused.splice(0), focusedTabs.splice(0)];
          await $("#chtopic").onclick();
          focused.length = 0;
          clickTab("settings");  // the user went to another tab, then back up to the table
          document.activeElement = document.body;
          await $("#tkeep").onclick();
          out.keepElsewhere = [V.tab, focused.splice(0), focusedTabs.splice(0)];
          out.bagWarn = $("#inputwarn").hidden;
          // A Stray capture with HDR and two export warnings.
          await open("s");
          out.warn = [V.tab, $("#inputwarn").hidden, $("#inputwarntext").textContent];
          $("#towarn").onclick();
          out.towarn = [V.tab, V.pick, location.hash, focused.splice(0)];
          console.log(JSON.stringify(out));
        })()
        """, replies)
    assert out["review"] == "inputs"
    assert out["build"] == ["log", ["log"], ["/api/jobs/v/continue"]]
    assert out["manual"] == ["settings", "settings", "#/job/v/settings", True, ["#gridsum"], True]
    assert out["refused"] == ["settings", ["#start"], [], "the OpenAI key was refused"]
    assert out["change"] == [False, False, ["#topicshead"]]
    assert out["keep"] == [True, False, ["#chtopic"], []]
    assert out["keepElsewhere"] == ["settings", [], ["settings"]]
    assert out["bagWarn"] is True
    assert out["warn"] == ["settings", False, "3 input warnings"]
    assert out["towarn"] == ["inputs", "inputs", "#/job/s/inputs", ["#hdr"]]


def test_app_js_help_link_follows_the_screen():
    """The header has one Help link, after the job's name and the Demos link and before the connection line, to
    help.html in a new tab (it stays there until app.js routes), and no subtitle. Its section follows the screen: the jobs list goes to
    adding a job, a waiting panel to its own section (the first one shown, in page order), else the tab shown. A
    refresh that shows a waiting panel moves it there; a job that does not load leaves it at the top of the page. The link is quiet and sits at the right of the header."""
    import re
    html = (server.STATIC / "index.html").read_text()
    header = html[html.index("<header>"):html.index("</header>")]
    link = '<a id="helplink" href="help.html" target="_blank" rel="noopener">Help</a>'
    demos = '<a id="demolink" href="demos.html" target="_blank" rel="noopener">Demos</a>'
    assert '<a href="#/" class="brand">HARPOR</a>' in header and "brandsub" not in html and "title=" not in header
    assert header.index('id="where"') < header.index(demos) < header.index(link) < header.index('id="conn"')
    assert header.count("<a ") == 3
    review = tab_job("v", "review", lidar="skipped", **{n: "done" for n in ("frames", "extract", "masks", "prep", "da3", "select")})
    out = run_form_js(ELEMENTS + TAB_SETUP + """
        (async () => {
          globalThis.EventSource = class { addEventListener() {} close() {} };
          let href = null;
          $("#helplink").setAttribute = (k, v) => { if (k === "href") href = v; };
          const out = {};
          show("list");
          out.list = href;
          V = { id: "j" };
          location._hash = "#/job/j";
          show("job");
          const panels = ["lidarfail", "topics", "review", "partsreview"];
          for (const x of panels) $("#" + x).hidden = true;
          renderState(st("done", ["topics", "frames", ...BUILD]));
          out.tabs = [href];
          for (const n of ["downloads", "inputs", "settings", "log"]) { clickTab(n); out.tabs.push(href); }
          // The waiting panels shown from the last on the page to the first: the first one shown wins.
          out.panels = [...panels].reverse().map((x) => { $("#" + x).hidden = false; clickTab("results"); return href; });
          for (const x of panels) $("#" + x).hidden = true;
          clickTab("inputs");
          out.back = href;
          // A refresh of a job in review shows the review panel, and the link follows it.
          V = { id: "v", picks: [], view0: null };
          location._hash = "#/job/v";
          $("#log").textContent = "a log";
          await refresh();
          out.refresh = [$("#review").hidden, V.tab, href];
          // Another job that does not load: the link does not keep the last job's section.
          out.gone = await openJob("gone").then(() => "loaded", (e) => e.message);
          out.goneHref = href;
          console.log(JSON.stringify({ ...out, at: HELP_AT }));
        })()
        """, {"/api/jobs/v": review, "/api/jobs/v/frames": frames_meta(),
              "/api/jobs/gone": {"__status": 404, "detail": "no such job"}})
    assert out["list"] == "help.html#add-job"
    assert out["tabs"] == ["help.html#" + x for x in ("results", "downloads", "inputs", "settings", "status")]
    assert out["panels"] == ["help.html#" + x for x in ("parts", "review", "topics", "lidar")]
    assert out["back"] == "help.html#inputs"
    assert out["refresh"] == [False, "inputs", "help.html#review"]
    assert out["gone"] == "no such job" and out["goneHref"] == "help.html"
    # tests/test_help_page.py checks that help.html has each of these sections.
    assert sorted(out["at"].values()) == sorted(["add-job", "lidar", "topics", "review", "parts", "results", "downloads",
                                                 "inputs", "settings", "status"])
    css = (server.STATIC / "style.css").read_text()
    rule = lambda sel: re.search(r"(?:^|\n)" + re.escape(sel) + r" \{([^}]*)\}", css).group(1)  # noqa: E731
    assert all(x in rule("#demolink, #helplink") for x in ("margin-left: auto", "var(--muted)", "var(--t-s)"))
    assert "margin-left: 0" in rule("#demolink + #helplink")  # Help sits right after Demos
    assert "margin-left" not in rule(".conn")


def test_app_js_error_box_links_to_troubleshooting():
    """A failed job's error ends with a Help link to help.html's troubleshooting section, in a new tab. A job
    without an error has no error box."""
    out = run_form_js(ELEMENTS + TAB_SETUP + """
        (() => {
          V = { id: "j" };
          location._hash = "#/job/j";
          renderState({ ...st("failed", ["topics", "frames"]), error: "masks: exit 1: out of memory" });
          const box = made.find((e) => e.className === "msg errbox"), link = box.kids.find((k) => k.tag === "a");
          made.length = 0;
          renderState(st("done", ["topics", "frames", ...BUILD]));
          console.log(JSON.stringify({ kids: box.kids.map((k) => typeof k === "string" ? k : k.tag), link: link.attrs,
            label: text(link), again: made.some((e) => e.className === "msg errbox") }));
        })()
        """)
    assert out["kids"] == ["masks: exit 1: out of memory", " ", "a"]
    assert out["link"] == {"href": "help.html#troubleshooting", "target": "_blank", "rel": "noopener"}
    assert out["label"] == "Help" and out["again"] is False


# ---- the Results and Downloads tabs in a small DOM ----

# static/app.js in node with a small DOM: every element keeps its tag, attributes, children and parent, its class
# names and a style; `hidden` is the attribute. Every selector gives one such element (byId). setTimeout's callbacks
# go to `timers` ({f, ms}). window.customElements says that <model-viewer> is defined, unless a test replaces it.
# fetch answers with the replies, by URL.
DOM_HARNESS = r"""
const fs = require("fs"), vm = require("vm");
const replies = JSON.parse(process.argv[4] || "{}");
const made = [], timers = [], byId = {};
class Elem {
  constructor(tag) {
    Object.assign(this, { tag, attrs: {}, kids: [], parent: null, style: {}, cls: new Set() });
    made.push(this);
  }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
  removeAttribute(k) { delete this.attrs[k]; }
  get hidden() { return "hidden" in this.attrs; }
  set hidden(v) { if (v) this.attrs.hidden = ""; else delete this.attrs.hidden; }
  get className() { return [...this.cls].join(" "); }
  set className(v) { this.cls = new Set(String(v).split(/\s+/).filter(Boolean)); }
  get classList() {
    const c = this.cls;
    return { add: (x) => c.add(x), remove: (x) => c.delete(x), contains: (x) => c.has(x),
      toggle: (x, on) => { on = on === undefined ? !c.has(x) : !!on; if (on) c.add(x); else c.delete(x); return on; } };
  }
  append(...k) { for (const x of k) { if (x && typeof x === "object") x.parent = this; this.kids.push(x); } }
  replaceChildren(...k) { this.kids = []; this.append(...k); }
  contains(x) { for (let e = x; e; e = e.parent) if (e === this) return true; return false; }
  get parentElement() { return this.parent; }
  get textContent() { return this.kids.map((k) => typeof k === "string" ? k : k.textContent).join(""); }
  set textContent(v) { this.kids = [String(v)]; }
  addEventListener() {}
  focus() {}
  scrollIntoView() {}
}
const document = { title: "", activeElement: null, body: {},
  querySelector: (sel) => byId[sel] || (byId[sel] = new Elem("div")), querySelectorAll: () => [],
  createElement: (tag) => new Elem(tag), createElementNS: (ns, tag) => new Elem(tag) };
const window = { addEventListener() {}, customElements: { get: () => class {}, whenDefined: () => new Promise(() => {}) } };
const location = { hash: "" }, history = { state: null, replaceState() {} };
const setTimeout = (f, ms) => timers.push({ f, ms });
const fetch = async (url) => ({ ok: true, status: 200, json: async () => url in replies ? replies[url] : url === "/api/jobs" ? [] : {} });
const ctx = vm.createContext({ console, document, window, location, history, setInterval() {}, setTimeout,
  EventSource: class { addEventListener() {} close() {} }, fetch, matchMedia: () => ({ matches: false }), confirm: () => true,
  Elem, made, timers, byId });
Promise.resolve(vm.runInContext(fs.readFileSync(process.argv[2], "utf8") + "\n" + process.argv[3], ctx))
  .catch((e) => { console.error(e.stack); process.exitCode = 1; });
"""


def run_dom_js(code, replies=None):
    """Load app.js in DOM_HARNESS, then run code in its scope; returns the JSON that code printed last."""
    if shutil.which("node") is None:
        pytest.skip("no node")
    p = subprocess.run(["node", "-e", DOM_HARNESS, "-", str(server.STATIC / "app.js"), textwrap.dedent(code),
                        json.dumps(replies or {})], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout.splitlines()[-1])


# For DOM_HARNESS: walk(e) is e and every element under it, in order.
WALK = """
const walk = (e, out = []) => { if (e && typeof e === "object") { out.push(e); e.kids.forEach((k) => walk(k, out)); } return out; };
"""

ART_MJCF = {"name": "cabinet", "root": "body", "fixed_base": True,
            "inertial": {"mass": 12.5, "com": [0.0, 0.01, 0.4], "fullinertia": [0.1] * 6},
            "links": {"body": {"source": "vomp", "inertial": {"mass": 10.0, "com": [0, 0, 0.4], "fullinertia": [0.1] * 6}},
                      "door": {"source": "vomp", "inertial": {"mass": 1.5, "com": [0, 0.2, 0.4], "fullinertia": [0.01] * 6}},
                      "drawer": {"source": "vomp", "inertial": {"mass": 0.9, "com": [0, 0, 0.2], "fullinertia": [0.01] * 6}},
                      "knob": {"source": "hull", "inertial": {"mass": 0.1, "com": [0, 0.3, 0.4], "fullinertia": [1e-4] * 6}}},
            "joints": [{"child": "door", "parent": "body", "type": "revolute", "range": [0.0, 1.5]},
                       {"child": "drawer", "parent": "body", "type": "prismatic", "range": [0.0, 0.3]},
                       {"child": "body", "parent": "world", "type": "fixed"}],
            "welded": {"knob": "door"}}
ART_RESULT = {"source": "finish", "verdict": "articulated", "reason": "the door hinges and the drawer slides",
              "limitations": ["the door's range is a guess", "the drawer has no tray"], "exit": "exited", "agent": "claude",
              "model": "claude-opus-5-5", "mode": "articulation", "counts": {"joints": 4}, "joints_delivered": 3,
              "turns": 9, "cost_usd": 0.31, "tokens": {"input_tokens": 1000, "output_tokens": 500,
                                                       "cache_read_input_tokens": 9000, "cache_creation_input_tokens": 800}}


def results_job():
    """(job, replies) for DOM_HARNESS: an articulated Opus job with every result file, so every row of the facts
    and every viewer shows."""
    place = {"metric_scale": 1.25, "check": {"mean_mask_iou": 0.82}}
    lidar = {"ok": True, "s": 1.25, "source": {"kind": "rosbag", "topic": "/lidar/points"},
             "combine": {"views": 8, "views_used": 8, "spread_rel": 0.01}}
    summary = {"rigid_body": {"mass": 12.4, "com": [0, 0.01, 0.4]}, "rigid_body_covers": ["body", "door", "drawer"],
               "median": {"youngs_modulus_pa": 4.1e9, "poisson_ratio": 0.3, "density_kg_m3": 990.0},
               "links_skipped": {"knob": "no occupied voxel"}}
    parts = {"parts": [{"name": "body", "status": "found"}, {"name": "door", "status": "found"},
                       {"name": None, "prompt": "handle", "status": "dropped_small", "why": "under 0.3% of the surface"}]}
    every = ["run/posed.glb", "run/parts.glb", "run/parts.json", "refined.glb", "run/placement.json", "pool/lidar.json",
             "physics/aligned.glb", "physics/summary.json", "physics/mjcf/mjcf.json", "physics/urdf/model.urdf", "results.zip",
             "selection.json"]
    job = {"id": "a", "articulated": True, "agent": "opus", "files": every, "agent_result": ART_RESULT,
           "stages": {"agent": {"status": "done"}, "turntable": {"status": "done"}}}
    replies = {"/api/jobs/a/file/run/placement.json": place, "/api/jobs/a/file/pool/lidar.json": lidar,
               "/api/jobs/a/file/physics/summary.json": summary, "/api/jobs/a/file/run/parts.json": parts,
               "/api/jobs/a/parts": {"checks": ["run/parts/sam3/check/0.jpg"]},
               "/api/jobs/a/file/physics/mjcf/mjcf.json": ART_MJCF}
    return job, replies


def test_app_js_results_facts_rows():
    """The Results tab's facts are plain dt/dd rows, in a fixed order: for an articulated Opus job every row, with the
    welded part's row; for a rigid job whose agent was skipped for its mask IoU, the scale, the IoU and the Agent row
    with the skip note. help.html (#results, #agent, #joints, #physics) explains each label."""
    job, replies = results_job()
    rigid = {"id": "s", "articulated": False, "agent": "astra", "agent_result": None,
             "files": ["run/posed.glb", "run/placement.json"],
             "stages": {"agent": {"status": "skipped", "note": "Stage A mean mask IoU 0.41 is below 0.5; physics uses "
                                                              "run/posed.glb"}}}
    replies["/api/jobs/s/file/run/placement.json"] = {"metric_scale": None, "check": {"mean_mask_iou": 0.41}}
    out = run_dom_js("""
        (async () => {
          const out = {};
          for (const job of """ + json.dumps([job, rigid]) + """) {
            V = { id: job.id };
            await renderResults(job);
            const [dl, ...more] = $("#facts2").kids;
            out[job.id] = { more: more.length, tag: dl.tag, rows: dl.kids.map((k) => [k.tag, k.textContent, k.kids.length]) };
          }
          console.log(JSON.stringify(out));
        })()
        """, replies)
    for found in out.values():
        assert found["tag"] == "dl" and found["more"] == 0
        assert [t for t, _, _ in found["rows"]] == ["dt", "dd"] * (len(found["rows"]) // 2)
        assert all(n == 1 for _, _, n in found["rows"])  # the text alone: no button, link or note
    rows = lambda job_id: [tuple(x[1] for x in out[job_id]["rows"][i:i + 2]) for i in range(0, len(out[job_id]["rows"]), 2)]  # noqa: E731
    assert rows("a") == [
        ("Metric scale", "×1.250 from LiDAR (/lidar/points, 8 views, spread 1.0%)"), ("Mask IoU", "0.820"),
        ("Parts not split off", "handle (under 0.3% of the surface)"), ("Agent", "Claude Opus 5.5 (claude-opus-5-5)"),
        ("Agent verdict", "articulated (finish)"), ("Agent reason", "the door hinges and the drawer slides"),
        ("Limitation", "the door's range is a guess"), ("Limitation", "the drawer has no tray"), ("Joints", "3"),
        ("Joint changes", "4"), ("Agent tokens", "1,000 input, 500 output, 9,000 cache read, 800 cache write in 9 turns"),
        ("Agent cost (estimate)", "about $0.31"), ("Links", "4: body, door, drawer, knob"),
        ("Base", "body, fixed to the world"), ("Joint of door", "turns on body, 0° to 86°"),
        ("Joint of drawer", "slides on body, 0.0 to 30.0 cm"), ("Joint of body", "fixed to world"),
        ("Joint of knob", "none (welded to door)"), ("Mass", "12.5 kg"), ("Center of mass", "0.000, 0.010, 0.400 m"),
        ("Link masses", "body 10.0 kg, door 1.50 kg, drawer 0.900 kg, knob 0.100 kg (from hulls)"),
        ("Young's modulus (median)", "4.10e+9 Pa"), ("Poisson ratio (median)", "0.300"), ("Density (median)", "990 kg/m³")]
    assert rows("s") == [("Metric scale", "Depth Anything 3's estimate (no LiDAR)"), ("Mask IoU", "0.410"),
                         ("Agent", "skipped: Stage A mean mask IoU 0.41 is below 0.5; physics uses run/posed.glb")]
    # The skip note says "Stage A" (steps.py), and the help page says what that is.
    assert "call it the Stage A mesh" in (server.STATIC / "help.html").read_text()


def test_app_js_results_names_a_soft_joint():
    """mjcf.json keeps a soft joint's type, so its row says so and not that the part is fixed."""
    job, replies = results_job()
    key = "/api/jobs/a/file/physics/mjcf/mjcf.json"
    soft = {"child": "pad", "parent": "door", "type": "soft", "pos": [0, 0, 0], "axis": [1, 0, 0], "range": [0, 0]}
    replies[key] = {**replies[key], "joints": [*replies[key]["joints"], soft]}
    out = run_dom_js("""
        (async () => {
          V = { id: "a" };
          await renderResults(""" + json.dumps(job) + """);
          console.log(JSON.stringify($("#facts2").kids[0].kids.map((k) => k.textContent)));
        })()
        """, replies)
    assert out[out.index("Joint of pad") + 1] == "soft on door"


ROUNDS_RESULT = {
    **ART_RESULT,
    "rounds": [{"round": 1, "kind": "build", "outcome": "fail", "scope": "structural", "candidate": None, "findings": []},
               {"round": 2, "kind": "replan", "outcome": None, "scope": None, "candidate": None, "findings": []},
               {"round": 3, "kind": "repair", "outcome": "pass", "scope": "none", "findings": [],
                "candidate": {"version": 2, "source": "finish", "verdict": "articulated"}}],
    "candidate": {"version": 2, "round": 3, "kind": "repair", "outcome": "pass", "why": "the only eligible candidate"},
    "requirements": {"pass": 2, "fail": 0, "inconclusive": 1, "unassessed": 2}, "simulations": 4,
    "surface": {"shares": {"present": 0.995, "retained": 0.98, "modified": 0.015, "unresolved": 0.0},
                "findings": [{"code": "surface_uv"}, {"code": "surface_uv"}], "blocks": False}}
EVIDENCE = {"version": 1, "requirements": [{"id": f"r{k}", "prompt": p} for k, p in enumerate(
    ["door", "drawer", "pad", "knob", "door"], 1)],
    "candidates": [{"version": 1, "records": [{"requirement": "r1", "target": ["door"], "joint": "revolute",
                                                "outcome": "fail", "measured": {}}]},
                   {"version": 2, "records": [
                       {"requirement": "r1", "target": ["door"], "kind": "joint", "joint": "revolute", "required": True,
                        "outcome": "pass", "measured": {"type": "revolute", "achieved": [0.0, 1.5], "states": 3}},
                       {"requirement": "r2", "target": ["drawer"], "kind": "joint", "joint": "prismatic", "required": True,
                        "outcome": "inconclusive", "measured": {"type": "prismatic", "achieved": [0.0, 0.3]}},
                       {"requirement": "r3", "target": ["<img src=x onerror=alert(1)>"], "kind": "soft", "joint": "soft",
                        "required": True, "outcome": "pass", "measured": {"type": "soft", "dmax_m": 0.004}},
                       {"requirement": "r4", "target": [], "kind": "joint", "joint": "unknown", "required": True,
                        "outcome": "unassessed", "measured": {}},
                       {"requirement": "r5", "target": ["door_2"], "kind": "joint", "joint": "revolute", "required": False,
                        "outcome": "unassessed", "measured": {}}, "not a record"]}],
    "selected": {"version": 2}}


def test_app_js_names_a_part_that_was_not_split_off():
    """A part of the list that the split did not find has the name reserved for its link as its record's target.
    The row says whether the agent created that link: cut out by the agent, or not split off (the check's
    part_missing). A found part and a record from before the reserves read as they did."""
    ev = {"requirements": [{"id": "r1", "prompt": "lid", "status": "found"},
                           {"id": "r2", "prompt": "flap", "status": "not_found", "reserve": "flap_3"},
                           {"id": "r3", "prompt": "knob", "status": "not_found"}]}
    rec = lambda rid, target, outcome, **more: {"requirement": rid, "target": target, "joint": "revolute",  # noqa: E731
                                                "outcome": outcome, "measured": {}, **more}
    records = [rec("r1", ["lid"], "pass"), rec("r2", ["flap_3"], "pass", measured={"achieved": [0, 1.5]}),
               rec("r2", ["flap_3"], "fail", findings=[{"code": "part_missing"}]),
               rec("r2", ["flap_3"], "fail", findings=[{"code": "joint_missing"}]),
               rec("r2", ["flap_3"], "unassessed"), rec("r3", [], "unassessed")]
    out = run_dom_js("console.log(JSON.stringify(" + json.dumps(records) + ".map((r) => requirementText(r, "
                     + json.dumps(ev) + "))))", {})
    assert out == ["lid, revolute: pass", "flap_3 (cut out by the agent), revolute: pass, clear from 0° to 86°",
                   "flap (not split off), revolute: fail", "flap_3 (cut out by the agent), revolute: fail",
                   "flap (not split off), revolute: unassessed", "knob (not split off), revolute: unassessed"]


def test_app_js_results_of_a_session_in_rounds():
    """An articulation that ran in rounds adds Rounds, Requirements (met of total), one row per requirement of
    agent/evidence.json (the delivered candidate's records, with the outcome and what the check measured) and Surface
    retained after the agent's joint changes, and each soft link of mjcf.json a Soft part row after the joints. Agent
    and VLM text stays text. The rows read every field defensively: a refinement's surface alone, a result whose keys
    are not of the expected kinds (a view stored on the public site) and an evidence file that does not parse leave
    out what they cannot give, never the other rows."""
    job, replies = results_job()
    mj = "/api/jobs/a/file/physics/mjcf/mjcf.json"
    replies[mj] = {**replies[mj], "soft": {"<b>pad</b>": {"material": "foam", "young": 3e4}, "grip": {"material": "rubber",
                                                                                                    "young": 1e6}},
                   "soft_rigid": {"cushion": {"code": "soft_untestable", "reason": "no usable hex mesh"}}}
    replies["/api/jobs/a/file/agent/evidence.json"] = EVIDENCE
    rounds = {**job, "files": [*job["files"], "agent/evidence.json"], "agent_result": ROUNDS_RESULT}
    odd = {**rounds, "agent_result": {**ROUNDS_RESULT, "rounds": "3", "requirements": None, "candidate": 5,
                                      "surface": {"error": "TypeError: x"}}}
    refined = {"id": "s", "articulated": False, "agent": "astra", "files": ["run/posed.glb", "refined.glb"],
               "stages": {"agent": {"status": "done"}},
               "agent_result": {"source": "finish", "verdict": "complete", "reason": "ok", "limitations": [], "agent": "codex",
                                "model": "gpt-6-astra", "mode": "refinement", "surface": {
                                    "shares": {"present": 1.0, "retained": 0.0, "modified": 0.0, "unresolved": 1.0},
                                    "findings": [], "blocks": True}}}
    out = run_dom_js("""
        (async () => {
          const out = {}, rows = () => $("#facts2").kids[0].kids.map((k) => [k.textContent, k.kids.every((x) => typeof x === "string")]);
          for (const [name, job] of Object.entries(""" + json.dumps({"rounds": rounds, "odd": odd, "refined": refined}) + """)) {
            V = { id: job.id };
            await renderResults(job);
            out[name] = rows();
          }
          const fetch0 = T.fetch;  // an evidence file that does not parse, as JSON.parse refuses NaN
          T.fetch = async (url, opts) => url.endsWith("/agent/evidence.json")
            ? { json: async () => { throw new SyntaxError("Unexpected token N in JSON"); } } : fetch0(url, opts);
          V = { id: "a" };
          await renderResults(""" + json.dumps(rounds) + """);
          out.unread = rows();
          console.log(JSON.stringify(out));
        })()
        """, replies)
    assert all(text for name in out for _, text in out[name])  # each dt and dd holds only text
    pairs = lambda name: [(out[name][i][0], out[name][i + 1][0]) for i in range(0, len(out[name]), 2)]  # noqa: E731
    got = pairs("rounds")
    at = got.index(("Joint changes", "4"))
    assert got[at + 1:at + 10] == [
        ("Rounds", "3 (build fail, replan, repair pass); delivered round 3's candidate"),
        ("Requirements", "2 of 5 met"),
        ("Requirement r1", "door, revolute: pass, clear from 0° to 86°"),
        ("Requirement r2", "drawer, prismatic: inconclusive, clear from 0.0 to 30.0 cm"),
        ("Requirement r3", "<img src=x onerror=alert(1)>, soft: pass, dents 4.0 mm"),
        ("Requirement r4", "knob (not split off), unknown: unassessed"),
        ("Requirement r5", "door_2, revolute: unassessed, not required"),
        ("Surface retained", "99.5% kept: 98.0% as it was, 1.5% changed; surface_uv (report only)"),
        ("Agent tokens", "1,000 input, 500 output, 9,000 cache read, 800 cache write in 9 turns")]
    at = got.index(("Joint of knob", "none (welded to door)"))
    assert got[at + 1:at + 5] == [("Soft part", "<b>pad</b>: foam, about 30 kPa; rigid in URDF and USD"),
                                  ("Soft part", "grip: rubber, about 1,000 kPa; rigid in URDF and USD"),
                                  ("Soft part", "cushion: rigid in MuJoCo too (no usable hex mesh)"), ("Mass", "12.5 kg")]
    odd = pairs("odd")
    assert ("Surface retained", "not compared (the check failed)") in odd
    assert not [k for k, _ in odd if k in ("Rounds", "Requirements") or k.startswith("Requirement ")]
    assert ("Surface retained", "100.0% kept: 0.0% as it was, 0.0% changed, 100.0% without UVs to check") in pairs("refined")
    unread = pairs("unread")
    assert ("Requirements", "2 of 5 met") in unread and not [k for k, _ in unread if k.startswith("Requirement ")]
    assert [k for k, _ in unread] == [k for k, _ in got if not k.startswith("Requirement ")]
    # The evidence downloads with the records, and help.html explains the rows (tests/test_help_page.py).
    assert '["agent/evidence.json", "evidence.json"]' in (server.STATIC / "app.js").read_text()


def test_app_js_viewer_says_while_its_mesh_loads():
    """Until its mesh shows, a viewer's box says "Loading the mesh" in model-viewer's poster slot, hidden from a
    screen reader, and its caption says "loading" where the size goes. Once the mesh loads, the caption gives its
    size in cm (only the name when model-viewer gives no size); when it cannot load, the caption is the name again
    and a line offers the file. When model-viewer's own script does not load, each viewer waits for it MV_WAIT_MS,
    then offers its file the same way; when the element is defined within that time, the viewer keeps waiting for
    its mesh. Captions are plain text."""
    out = run_dom_js(WALK + """
        (async () => {
          V = { id: "j" };
          const job = { id: "j", agent: "astra", articulated: false, files: ["run/posed.glb", "refined.glb"],
            stages: { agent: { status: "done" } }, agent_result: { agent: "codex", model: "gpt-6-astra", source: "finish",
              verdict: "complete", reason: "fixed the lid", limitations: [] } };
          const figs = async () => {
            made.length = 0;
            await renderResults(job);
            return walk($("#viewers")).filter((e) => e.tag === "figure");
          };
          const settle = async () => { for (let i = 0; i < 5; i++) await null; };
          const cap = (f) => f.kids[0].kids, mv = (f) => f.kids.find((k) => k.tag === "model-viewer");
          let [a, b] = await figs();
          const poster = mv(a).kids[0];
          const out = { loading: [cap(a), cap(b)], poster: [poster.tag, poster.attrs.slot, poster.className,
            poster.attrs["aria-hidden"], poster.textContent], timers: timers.length };
          mv(a).onload({ target: { getDimensions: () => ({ x: 0.604, y: 0.4, z: 0.295 }) } });
          mv(b).onload({ target: {} });
          out.loaded = [cap(a), cap(b)];
          [a] = await figs();
          mv(a).onerror();
          out.failed = [cap(a), a.kids.length, a.kids[1].textContent];
          // model-viewer's script did not load: the viewers wait, then offer their files.
          let define = null;
          const defined = new Promise((r) => { define = r; });
          window.customElements = { get: () => undefined, whenDefined: () => defined };
          [a, b] = await figs();
          out.waiting = [cap(a), cap(b), timers.map((t) => t.ms)];
          timers.splice(0).forEach((t) => t.f());
          await settle();
          out.gone = [cap(a), a.kids.length, a.kids[1].textContent, cap(b)];
          // It is defined within the wait: the viewer keeps waiting for its mesh.
          [a] = await figs();
          define();
          await settle();
          timers.splice(0).forEach((t) => t.f());
          await settle();
          out.defined = [cap(a), a.kids.length, MV_WAIT_MS];
          console.log(JSON.stringify(out));
        })()
        """)
    assert out["loading"] == [["Posed mesh: loading"], ["Refined (GPT-6-Astra): loading"]]
    assert out["poster"] == ["p", "poster", "mvload", "true", "Loading the mesh"] and out["timers"] == 0
    assert out["loaded"] == [["Posed mesh: 60 x 40 x 30 cm"], ["Refined (GPT-6-Astra)"]]
    assert out["failed"] == [["Posed mesh"], 2, "Could not show posed.glb: download it."]
    wait = out["defined"][2]
    assert out["waiting"] == [["Posed mesh: loading"], ["Refined (GPT-6-Astra): loading"], [wait, wait]] and wait >= 5000
    assert out["gone"] == [["Posed mesh"], 2, "Could not show posed.glb: download it.", ["Refined (GPT-6-Astra)"]]
    assert out["defined"][:2] == [["Posed mesh: loading"], 2]
    import re
    css = (server.STATIC / "style.css").read_text()
    mvload = re.search(r"\.mvload \{([^}]*)\}", css).group(1)
    assert all(x in mvload for x in ("height: 100%", "align-items: center", "var(--muted)"))  # "Loading the mesh"


def test_app_js_results_play_the_360_video():
    """A job with the 360 video shows it first in Results: muted, looping, inline, with controls and a name, starting by
    itself unless the system asks for reduced motion. A refresh, also with a new signed URL, does not draw it again; a
    new file (Build again) does. It stops when the Results tab hides and plays again when the tab shows, unless the user
    paused it. When it cannot play, its caption stays and a line offers the file. Without the file, nothing shows, and
    another job's page or a new build takes the old video away."""
    out = run_form_js(ELEMENTS + """
        (async () => {
          const out = {}, mp4 = "turntable/turntable.mp4";
          const job = { id: "j", prompt: "office chair", files: ["run/posed.glb", mp4], sizes: { [mp4]: 2100000 },
            stages: { turntable: { status: "done", ended: 1000.5 } } };
          const vids = () => made.filter((e) => e.tag === "video"), figs = () => made.filter((e) => e.className === "spinfig");
          V = { id: "j" };
          renderSpin(job);
          const video = vids()[0], fig = figs()[0];
          out.video = { ...video.attrs, muted: video.muted, calls: [...video.calls] };
          out.fig = [fig.tag, fig.kids.map((k) => k.tag), text(fig.kids[0])];
          renderSpin(job);
          renderSpin({ ...job, sizes: { ...job.sizes } });
          out.again = [vids().length, V.spin === video];
          // The tabs: the video stops when Results hides, and plays again when it shows.
          V.tab = "results";
          selectTab("downloads");
          out.leave = [...video.calls];
          selectTab("inputs");
          selectTab("results");
          out.back = [...video.calls];
          video.paused = true;  // the user paused it: it stays paused
          selectTab("downloads");
          selectTab("results");
          out.paused = [...video.calls];
          video.onerror();
          out.failed = [text(fig), fig.kids[1].kids[1].attrs, V.spin];
          // A new file: drawn again, and not started while another tab shows.
          V.tab = "log";
          renderSpin({ ...job, stages: { turntable: { status: "done", ended: 2000.5 } } });
          out.newfile = [vids().length, vids()[1].calls];
          // Reduced motion: no autoplay, and the Results tab does not start it.
          made.length = 0;
          globalThis.matchMedia = (q) => ({ matches: q === "(prefers-reduced-motion: reduce)" });
          V = { id: "k" };
          renderSpin({ ...job, id: "k", prompt: null });
          selectTab("results");
          out.still = [vids()[0].attrs.autoplay, vids()[0].attrs["aria-label"], vids()[0].calls];
          // No file: nothing, and the old video goes.
          renderSpin({ ...job, files: ["run/posed.glb"] });
          out.gone = [V.spin, V.spinKey];
          console.log(JSON.stringify(out));
        })()
        """)
    assert out["video"] == {"src": "/api/jobs/j/file/turntable/turntable.mp4", "autoplay": "", "muted": True, "loop": "",
                            "playsinline": "", "controls": "", "preload": "auto", "aria-label": "360 video of the office chair",
                            "calls": []}
    assert out["fig"] == ["figure", ["figcaption", "video"], "360 video"]
    assert out["again"] == [1, True]
    assert out["leave"] == ["pause"] and out["back"] == ["pause", "pause", "play"]
    assert out["paused"] == ["pause", "pause", "play", "pause"]
    assert out["failed"][0] == "360 videoCould not play the video: download it."
    assert out["failed"][1] == {"href": "/api/jobs/j/file/turntable/turntable.mp4", "download": "j_turntable.mp4"}
    assert out["failed"][2] is None
    assert out["newfile"] == [2, ["pause"]]
    assert out["still"] == [None, "360 video of the object", []]
    assert out["gone"] == [None, None]


def test_app_js_results_play_the_soft_part_video_beside_the_360_video():
    """The softvideo stage's video sits beside the 360 video, after it, and plays as it does: muted, looping, with a
    name, stopping when the Results tab hides and playing again when it shows. A job stored on the site before the stage
    has no record of it, and its video still draws. Each video is drawn once per file: a new soft-part video leaves the
    360 video as it is. Without the 360 video, the soft-part video shows alone; when it cannot play, a line offers its
    file, and without the file it goes."""
    out = run_dom_js(WALK + """
        (async () => {
          Elem.prototype.pause = function () { (this.calls = this.calls || []).push("pause"); };
          Elem.prototype.play = function () { (this.calls = this.calls || []).push("play"); };
          const out = {}, mp4 = "turntable/turntable.mp4", soft = "softvideo/softvideo.mp4";
          const job = { id: "j", prompt: "office chair", files: ["run/posed.glb", mp4, soft],
            sizes: { [mp4]: 2100000, [soft]: 640000 }, stages: { turntable: { status: "done", ended: 1000.5 } } };
          const vids = () => made.filter((e) => e.tag === "video");
          V = { id: "j", tab: "results" };
          renderSpin(job);
          const [spin, video] = vids();
          out.spin = $("#spin").kids.map((f) => [f.className, f.kids[0].textContent, f.kids[1].attrs.src]);
          out.video = { ...video.attrs, muted: video.muted, calls: video.calls || [] };
          renderSpin({ ...job, sizes: { ...job.sizes } });
          out.again = vids().length;
          renderSpin({ ...job, stages: { ...job.stages, softvideo: { status: "done", ended: 1001.5 } } });
          const [a, b] = $("#spin").kids;
          out.newsoft = [vids().length, V.spin === spin, V.soft === vids()[2], a.kids[1] === spin, b.kids[1] === V.soft];
          spinTab("downloads");
          V.tab = "downloads";
          out.leave = [[...spin.calls], [...vids()[2].calls]];
          spinTab("results");
          out.back = [spin.calls, vids()[2].calls];
          // No 360 video: the soft-part video alone, and its file when it cannot play.
          made.length = 0;
          V = { id: "k" };
          clearSpin();
          renderSpin({ ...job, id: "k", files: [soft] });
          out.alone = [$("#spin").kids.length, V.spin, $("#spin").kids[0].kids[0].textContent];
          vids()[0].onerror();
          const fig = $("#spin").kids[0];
          out.failed = [fig.textContent, fig.kids[1].kids[1].attrs, V.soft];
          renderSpin({ ...job, id: "k", files: [] });
          out.gone = [$("#spin").kids.length, V.softKey];
          console.log(JSON.stringify(out));
        })()
        """)
    assert out["spin"] == [["spinfig", "360 video", "/api/jobs/j/file/turntable/turntable.mp4"],
                           ["spinfig", "Soft-part video", "/api/jobs/j/file/softvideo/softvideo.mp4"]]
    assert out["video"] == {"src": "/api/jobs/j/file/softvideo/softvideo.mp4", "autoplay": "", "muted": True, "loop": "",
                            "playsinline": "", "controls": "", "preload": "auto",
                            "aria-label": "The soft part of the office chair, pressed and let go", "calls": []}
    assert out["again"] == 2
    assert out["newsoft"] == [3, True, True, True, True]
    assert out["leave"] == [["pause"], ["pause"]] and out["back"] == [["pause", "play"], ["pause", "play"]]
    assert out["alone"] == [1, None, "Soft-part video"]
    assert out["failed"][0] == "Soft-part videoCould not play the video: download it."
    assert out["failed"][1] == {"href": "/api/jobs/k/file/softvideo/softvideo.mp4", "download": "k_softvideo.mp4"}
    assert out["failed"][2] is None and out["gone"] == [0, None]


def test_app_js_retry_on_a_done_job_is_only_for_the_360_video():
    """A done job whose 360 video failed (steps.OPTIONAL) offers Retry from turntable, and the click asks the server for
    that stage; a done job with every stage done offers no Retry."""
    names = ["topics", "frames", *jobs.BUILD]
    stages = lambda **o: [{"name": n, "status": o.get(n, "done")} for n in names]  # noqa: E731
    out = run_form_js("""
        (async () => {
          const out = {}, state = (stages) => ({ status: "done", stages, agent: "none", key_ok: false, warnings: [], error: null });
          const job = { id: "j", video: { name: "chair.mov" } };
          V = { id: "j", job };
          renderState(state(""" + json.dumps(stages(turntable="failed")) + """));
          out.failed = [$("#retry").hidden, $("#retry").textContent];
          posts.length = 0;
          await $("#retry").onclick();
          out.sent = posts.map((p) => [p.url, p.body.from]);
          V = { id: "j", job };  // the click's refresh took the reply
          renderState(state(""" + json.dumps(stages()) + """));
          out.done = $("#retry").hidden;
          console.log(JSON.stringify(out));
        })()
        """)
    assert out["failed"] == [False, "Retry from turntable"]
    assert out["sent"] == [["/api/jobs/j/continue", "turntable"]]
    assert out["done"] is True


def test_app_js_new_build_and_another_job_take_the_360_video_away():
    """drawJob hides the results of a job that builds again, and the old 360 video goes with them; openJob takes away the
    last job's video before it draws the next one."""
    out = run_dom_js(WALK + """
        (async () => {
          const out = {};
          V = { id: "j", spin: new Elem("video"), spinKey: "k" };
          $("#spin").replaceChildren(new Elem("figure"));
          await drawJob({ id: "j", status: "running", files: [], sizes: {}, video: { name: "x.mp4" }, prompt: "box",
            source: { kind: "video" }, progress: [], agent: "none", agent_result: null, articulated: false,
            parts_review: false, stages: { frames: { status: "pending" }, select: { status: "pending" } } });
          out.build = [V.spin, V.spinKey, $("#spin").kids.length, $("#resultsbody").hidden];
          $("#spin").replaceChildren(new Elem("figure"));
          await openJob("m").catch(() => {});
          out.open = [V.id, V.spin || null, $("#spin").kids.length];
          console.log(JSON.stringify(out));
        })()
        """, {"/api/jobs/m": {"__unused": True}})
    assert out["build"] == [None, None, 0, True]
    assert out["open"] == ["m", None, 0]


# ---- the Downloads tab (app.js renderDownloads) ----

DL_SIZES = {"results.zip": 12_345_678, "run/posed.glb": 1_234_567, "run/parts.glb": 987_654, "refined.glb": 2_500_000,
            "physics/aligned.glb": 1_300_000, "turntable/turntable.mp4": 2_100_000, "softvideo/softvideo.mp4": 640_000,
            "physics/urdf/model.urdf": 4_321, "run/placement.json": 5_678,
            "physics/summary.json": 3_210, "pool/lidar.json": 45_678, "selection.json": 23_456, "run/parts.json": 812,
            "physics/mjcf/mjcf.json": 6_789, "agent/evidence.json": 4_235}


def test_app_js_downloads_tab():
    """The Downloads tab lists the job's files that exist, in five groups, each under an h3: Everything (results.zip),
    Meshes, Video (the 360 video and the soft-part video), Simulation (model.urdf, which says that it needs the meshes in results.zip) and
    Records. Each row is the
    file name as a download link that saves as <job id>_<file>, and its size from job.sizes (none when the server
    gave none), and nothing else. Groups without files are left out, a file that the tab does not know is ignored,
    and a running build lists none. Without results.zip, a line above the groups says why: the build has not made it
    yet, or it stopped first. An empty tab says a few words. The tab offers exactly the server's outputs, all of
    which the file route serves, and the Results tab has no download link."""
    job, replies = results_job()
    every = list(DL_SIZES)
    cases = {
        "a": {**job, "status": "done", "files": every, "sizes": DL_SIZES},
        "r": {"id": "r", "status": "done", "agent": "none", "agent_result": None, "articulated": False,
              "files": [f for f in every if f not in ("run/parts.glb", "run/parts.json", "refined.glb", "pool/lidar.json",
                                                      "turntable/turntable.mp4", "softvideo/softvideo.mp4",
                                                      "agent/evidence.json")]
              + ["frames/preview.mp4"], "sizes": {k: v for k, v in DL_SIZES.items() if k != "physics/aligned.glb"}},
        "v": {"id": "v", "status": "review", "files": ["selection.json", "pool/lidar.json"], "sizes": DL_SIZES},
        "f": {"id": "f", "status": "failed", "files": ["selection.json", "run/posed.glb", "run/placement.json"]},
        "c": {"id": "c", "status": "cancelled", "files": ["selection.json"], "sizes": {}},
        # No model.urdf in files: results.zip has no urdf/ (jobs.package zips what exists).
        "u": {"id": "u", "status": "done", "agent": "none", "agent_result": None,
              "files": [f for f in every if f not in ("physics/urdf/model.urdf", "run/parts.glb", "run/parts.json",
                                                      "turntable/turntable.mp4", "softvideo/softvideo.mp4",
                                                      "agent/evidence.json")]},
        "y": {"id": "y", "status": "ready", "files": [], "sizes": {}},
        "z": {"id": "z", "status": "analyzing", "files": [], "sizes": {}},
        "n": {"id": "n", "status": "cancelled", "files": [], "sizes": {}},
        "q": {"id": "q", "status": "running", "files": ["selection.json", "run/posed.glb"], "sizes": DL_SIZES}}
    out = run_dom_js(WALK + """
        (async () => {
          const out = { paths: DOWNLOADS.flatMap(([, rows]) => rows.map(([p]) => p)), sim: SIM_TEXT,
            partial: FILES_PARTIAL, stopped: FILES_STOPPED,
            sizes: [0, 999, 1000, 9_949, 9_950, 10_000, 999_499, 999_500, 99_949_999, 99_960_000, 100_000_000,
              999_499_999, 999_600_000, 1_000_000_000, 2_500_000_000].map(fileSize) };
          for (const [id, job] of Object.entries(""" + json.dumps(cases) + """)) {
            V = { id };
            made.length = 0;
            renderDownloads(job);
            const kids = $("#downloadsbody").kids, note = kids.find((k) => k.cls.has("dlnote"));
            const groups = kids.filter((k) => k.cls.has("dlgroup")).map((g) => {
              const [head, ...rest] = g.kids, list = rest[rest.length - 1], hint = rest.length > 1 ? rest[0] : null;
              return { title: head.textContent, tag: head.tag,
                labelled: list.attrs["aria-labelledby"] === head.attrs.id && !!head.attrs.id,
                cls: [g.className, list.tag, list.className], text: hint ? hint.textContent : null,
                rows: list.kids.map((li) => {
                  const a = li.kids[0], size = li.kids.find((k) => k.cls && k.cls.has("dlsize"));
                  return { name: a.textContent, href: a.attrs.href, download: a.attrs.download, cls: a.className,
                    size: size ? size.textContent : null, kids: li.kids.length, li: li.className };
                }) };
            });
            out[id] = { groups, none: [$("#downloadsnone").hidden, $("#downloadsnone").textContent], body: $("#downloadsbody").hidden,
              note: note ? [kids.indexOf(note), note.tag, note.textContent] : null };
          }
          // The Results tab of the full job: the viewers and the facts, with no download link.
          V = { id: "a" };
          made.length = 0;
          await renderResults(""" + json.dumps(cases["a"]) + """);
          out.resultLinks = [$("#viewers"), $("#facts2"), $("#partchecks")].flatMap((e) => walk(e))
            .filter((e) => e.tag === "a" && (e.cls.has("dl") || "download" in e.attrs)).length;
          console.log(JSON.stringify(out));
        })()
        """, replies)
    # Exactly the server's outputs, each one a path that the file route serves.
    assert sorted(out["paths"]) == sorted(server.OUTPUTS) and all(server.ALLOW.fullmatch(p) for p in out["paths"])
    # The bounds go by the rounded number: 99.96 MB is "100 MB", not "100.0 MB", and 999.6 MB is "1.00 GB".
    assert out["sizes"] == ["0 B", "999 B", "1.0 KB", "9.9 KB", "10 KB", "10 KB", "999 KB", "1.0 MB", "99.9 MB", "100 MB",
                            "100 MB", "999 MB", "1.00 GB", "1.00 GB", "2.50 GB"]
    a = out["a"]
    assert a["none"][0] is True and a["body"] is False and a["note"] is None
    assert [g["title"] for g in a["groups"]] == ["Everything", "Meshes", "Video", "Simulation", "Records"]
    assert all(g["labelled"] and g["tag"] == "h3" and g["cls"] == ["dlgroup", "ul", "dlist"] for g in a["groups"])
    rows = [r for g in a["groups"] for r in g["rows"]]
    assert [(r["name"], r["size"]) for r in rows] == [
        ("results.zip", "12.3 MB"), ("posed.glb", "1.2 MB"), ("parts.glb", "988 KB"), ("refined.glb", "2.5 MB"),
        ("aligned.glb", "1.3 MB"), ("turntable.mp4", "2.1 MB"), ("softvideo.mp4", "640 KB"), ("model.urdf", "4.3 KB"), ("placement.json", "5.7 KB"), ("summary.json", "3.2 KB"),
        ("lidar.json", "46 KB"), ("selection.json", "23 KB"), ("parts.json", "812 B"), ("mjcf.json", "6.8 KB"),
        ("evidence.json", "4.2 KB")]
    by_name = {r["name"]: r for r in rows}
    assert sorted(r["href"] for r in rows) == sorted(f"/api/jobs/a/file/{p}" for p in every)
    assert all(r["href"].endswith("/" + r["name"]) for r in rows)  # each file downloads as its own name
    # Every file saves under the job id, so files of two jobs do not clash; the label stays the short name.
    assert all(r["download"] == "a_" + r["name"] for r in rows)
    assert by_name["results.zip"]["cls"] == "dl primary" and all(r["cls"] == "dl" for r in rows if r["name"] != "results.zip")
    assert all(r["kids"] == 2 and r["li"] == "dlrow" for r in rows)  # the link and its size, nothing else
    texts = {g["title"]: g["text"] for g in a["groups"]}
    assert texts == {"Everything": None, "Meshes": None, "Video": None, "Simulation": out["sim"], "Records": None}
    assert out["sim"] == "Needs the meshes in results.zip."
    # A rigid job without the agent or LiDAR: only its files, a file with no size shows none, an unknown file is left out.
    r = out["r"]
    assert {g["title"]: [x["name"] for x in g["rows"]] for g in r["groups"]} == {
        "Everything": ["results.zip"], "Meshes": ["posed.glb", "aligned.glb"], "Simulation": ["model.urdf"],
        "Records": ["placement.json", "summary.json", "selection.json", "mjcf.json"]}
    assert [(x["size"], x["kids"]) for x in r["groups"][1]["rows"]] == [("1.2 MB", 2), (None, 1)]
    # Without model.urdf in files there is no Simulation group.
    assert [g["title"] for g in out["u"]["groups"]] == ["Everything", "Meshes", "Records"] and out["u"]["note"] is None
    # In review: the records alone; after a failure: what exists, without sizes when the server gave none. Without
    # results.zip, the first line says why the tab holds only these.
    assert [(g["title"], [x["name"] for x in g["rows"]]) for g in out["v"]["groups"]] == [("Records", ["lidar.json", "selection.json"])]
    assert out["v"]["note"] == [0, "p", out["partial"]] and out["partial"] == "More files after the build."
    assert [(g["title"], [(x["name"], x["size"]) for x in g["rows"]]) for g in out["f"]["groups"]] == [
        ("Meshes", [("posed.glb", None)]), ("Records", [("placement.json", None), ("selection.json", None)])]
    assert out["f"]["note"] == [0, "p", out["stopped"]] and out["c"]["note"] == [0, "p", out["stopped"]]
    assert out["stopped"] == "Build stopped before results.zip."
    assert all(out[j]["note"] is None for j in ("r", "y", "z", "n", "q"))  # results.zip is there, or no file is
    # Nothing to list: a few words.
    for job_id, text in (("y", "No files yet."), ("z", "No files yet."), ("n", "Job stopped before any files."),
                         ("q", "Files show when the build stops.")):
        assert out[job_id] == {"groups": [], "none": [False, text], "body": True, "note": None}, job_id
    assert out["resultLinks"] == 0


def test_app_js_downloads_survive_a_results_file_that_does_not_parse():
    """A result file that does not parse (a cut write after a kill or a full disk) does not stop drawJob: the
    Downloads tab still lists the files, with the sentence that the build stopped first, the facts say what could
    not be read, the check images of an earlier draw go, and the Inputs tab still draws."""
    job = {"id": "j", "status": "failed", "video": {"name": "x.mp4"}, "prompt": "box", "source": {"kind": "video"},
           "stages": {"frames": {"status": "pending"}, "agent": {"status": "pending"}, "select": {"status": "done"}},
           "progress": [], "files": ["run/posed.glb", "run/placement.json", "selection.json"],
           "sizes": {"run/posed.glb": 1_200_000, "run/placement.json": 5_700, "selection.json": 23_000},
           "agent": "none", "agent_result": None, "articulated": False, "parts_review": False}
    out = run_dom_js("""
        (async () => {
          const fetch0 = fetch;
          fetch = async (url) => url.endsWith("/run/placement.json")
            ? { ok: true, status: 200, json: async () => { throw new SyntaxError("Unexpected end of JSON input"); } }
            : fetch0(url);
          V = { id: "j" };
          $("#partchecks").replaceChildren(new Elem("details"));  // the check images of an earlier draw
          let err = null;
          try { await drawJob(""" + json.dumps(job) + """); } catch (e) { err = String(e); }
          const kids = $("#downloadsbody").kids;
          console.log(JSON.stringify({ err, body: $("#downloadsbody").hidden, none: $("#downloadsnone").hidden,
            groups: kids.filter((k) => k.cls.has("dlgroup")).map((g) => [g.kids[0].textContent,
              g.kids[g.kids.length - 1].kids.map((li) => li.kids[0].textContent)]),
            note: kids.filter((k) => k.cls.has("dlnote")).map((k) => k.textContent), facts: $("#facts2").textContent,
            checks: $("#partchecks").kids.length, results: [$("#resultsbody").hidden, $("#resultsnone").hidden],
            inputs: [$("#inputbody").hidden, $("#inputnone").hidden], unread: RESULTS_UNREAD, stopped: FILES_STOPPED }));
        })()
        """)
    assert out["err"] is None
    assert (out["body"], out["none"]) == (False, True)
    assert out["groups"] == [["Meshes", ["posed.glb"]], ["Records", ["placement.json", "selection.json"]]]
    assert out["note"] == [out["stopped"]]
    assert out["facts"] == out["unread"] + "Unexpected end of JSON input" and out["checks"] == 0
    assert out["results"] == [False, True]  # the viewer still shows
    assert out["inputs"] == [True, False]  # drawJob went on past the results


def test_style_css_360_video():
    """The 360 video is square and as tall as a viewer at most, never wider than the panel, on the viewers' background
    with their hairline; its caption is a viewer's; on a wide screen with one viewer it sits beside the viewer, 260 px
    at most, above the facts, so the facts show on the first screen. #spin is a flex box, in which margins do not
    collapse: a figure keeps only its top margin there, so the gap above the viewers, or between the 360 video and the
    soft-part video below it, is one margin (16 px), and beside the viewer only its bottom margin, above the facts."""
    import re
    css = (server.STATIC / "style.css").read_text()
    video = re.search(r"\n\.spinfig video \{([^}]*)\}", css).group(1)
    assert all(x in video for x in ("width: min(100%, clamp(260px, 45vh, 420px))", "aspect-ratio: 1", "height: auto",
                                    "var(--viewer-bg)", "1px solid var(--line)"))
    assert ".viewers figcaption, .spinfig figcaption {" in css
    assert "display: flex" in re.search(r"\n#spin \{([^}]*)\}", css).group(1)
    assert "margin-bottom: 0" in re.search(r"\n#spin > \.spinfig \{([^}]*)\}", css).group(1)
    assert "margin: var(--s4) 0" in re.search(r"\n\.spinfig \{([^}]*)\}", css).group(1)
    wide = css[css.index("@media (min-width: 1100px)"):]
    wide = wide[:wide.index("\n}")]
    assert "#resultsbody.onefig > #spin { grid-column: 2; grid-row: 1; }" in wide
    assert "#resultsbody.onefig > #facts2 { grid-column: 2; grid-row: 2; margin-top: 0; }" in wide
    assert "grid-row: 1 / span 2" in wide and "grid-template-rows: auto 1fr" in wide
    assert "#resultsbody.onefig .spinfig video { width: min(100%, 260px); }" in wide
    assert "#resultsbody.onefig .spinfig { margin: 0 0 var(--s4); }" in wide


def test_style_css_downloads():
    """A Downloads row wraps on a narrow phone, its link keeps a 32 px height (44 px on a touch screen), and the
    rows use the color tokens: a link in the accent color and mono, results.zip filled in the primary color. The
    link's arrow has empty alternative text, the group headings are h3, and one Results viewer has the facts beside
    it on a wide screen."""
    import re
    css = (server.STATIC / "style.css").read_text()

    def rule(selector, text=css):
        m = re.search(r"(?:^|\n|\}\s*)" + re.escape(selector) + r"\s*\{([^}]*)\}", text)
        assert m, selector
        return m.group(1)
    assert "flex-wrap: wrap" in rule(".dlrow")
    link = rule("a.dl")
    assert all(x in link for x in ("min-height: 32px", "var(--accent)", "var(--mono)"))
    assert all(x in rule("a.dl.primary") for x in ("var(--primary)", "var(--primary-fg)"))
    assert "var(--muted)" in rule(".dlsize") and "list-style: none" in rule(".dlist")
    coarse = css[css.index("@media (pointer: coarse)"):]
    assert "min-height: 44px" in rule("a.dl", coarse)
    assert ".downloads" not in css and ".dlitem" not in css  # the old row of downloads in Results
    # The arrow before each link has empty alternative text, so a screen reader reads only the file name; the plain
    # content line before it is the fallback for a browser without that syntax.
    arrow = rule("a.dl::before")
    assert 'content: "\\2193" / "";' in arrow and arrow.index('content: "\\2193";') < arrow.index('content: "\\2193" / "";')
    # The group headings are h3 in the text color (not the muted h4), and the note above them is muted.
    assert "font-size: var(--t-m)" in rule(".dlgroup h3") and ".dlgroup h4" not in css
    assert "var(--muted)" in rule(".dlnote")
    # One viewer: on a wide screen the facts sit beside it.
    wide = css[css.index("@media (min-width: 1100px)"):]
    wide = wide[:wide.index("\n}\n")]
    assert re.search(r"#resultsbody\.onefig \{[^}]*display: grid", wide)
    assert re.search(r"#resultsbody\.onefig > #partchecks \{[^}]*grid-column: 1 / -1", wide)
