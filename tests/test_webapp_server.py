import hashlib
import io
import json
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

from gator.webapp import jobs, server, steps  # noqa: E402

TOKEN = "tok-" + secrets.token_hex(16)
KEY = "sk-test-" + secrets.token_hex(16)
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def client(tmp_path):
    cfg = steps.Config(G=tmp_path / "G", runs=tmp_path / "runs", fake=True)
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
    for path in ("/", "/static/app.js", "/api/jobs", "/api/jobs/x/file/selection.json"):
        assert c.get(path).status_code == 401
    assert c.put("/api/upload?name=a.mp4", content=b"x").status_code == 401
    assert c.get("/?token=wrong").status_code == 401 and c.get("/?token=").status_code == 401
    cookie = login(c).headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie
    assert "/static/app.js" in c.get("/").text and c.get("/static/app.js").status_code == 200
    assert c.get("/api/jobs").json() == [] and c.get("/api/jobs/20260101-000000-abcdef").status_code == 404
    c.cookies.set(server.COOKIE, "wrong")
    assert c.get("/api/jobs").status_code == 401


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
    # A rigid object without LiDAR has every output but the split parts and the LiDAR record.
    assert set(job["files"]) == set(server.OUTPUTS) - {"run/parts.glb", "run/parts.json", "pool/lidar.json"}
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
    assert c.post(f"/api/jobs/{job_id}/start", json={"mode": "auto", "prompt": "box", "agent": False}).status_code == 409

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

    start = {"mode": "auto", "prompt": "Wooden box", "n_views": 4, "auto_continue": False, "agent": False}
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
                       "time_offset_s": 0.0}
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


def test_fake_rosbag_job_with_lidar_end_to_end(client, tmp_path):
    """A bag whose /tf_static links the camera to the LiDAR and whose camera_info is rectified: the
    LiDAR scale undoes the fake DA3's 0.8, and every later stage works in meters."""
    from test_webapp_lidar import POINTS

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

    # Built again with LiDAR off: the lidar stage is skipped and select applies no scale.
    r = c.post(f"/api/jobs/{job_id}/start", json={**start, "lidar": None})
    assert r.status_code == 200, r.text
    job = wait(c, job_id, "done", "failed")
    assert job["status"] == "done", job["error"]
    assert job["lidar"] is None and job["stages"]["lidar"]["note"] == "turned off: no LiDAR"
    assert "pool/lidar.json" not in job["files"] and c.get(f"/api/jobs/{job_id}/selection").json()["lidar"] is None
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
    monkeypatch.setattr(server.shutil, "disk_usage", lambda p: shutil._ntuple_diskusage(100 << 30, 99 << 30, 1 << 30))
    r = c.put("/api/upload", params={"name": "a.mcap"}, content=b"x" * 1000)
    assert r.status_code == 413 and "1.0 GB free" in r.json()["detail"]
    # A zip is unpacked next to itself, so it needs 2.2 x its size free (+ 2 GB) where a bag needs 1.2 x.
    for name, need in (("a.bag", 3.2), ("capture.zip", 4.2)):
        r = c.put("/api/upload", params={"name": name}, content=b"x", headers={"Content-Length": str(1 << 30)})
        assert (r.status_code, r.json()["detail"]) == (413, f"the server has 1.0 GB free and this upload needs {need} GB")
    assert c.get("/api/jobs").json() == [] and not list(cfg.runs.glob("*/video"))
    monkeypatch.undo()
    # The kind is that of the name the file is saved under: "..bag" is kept as video.bag, a rosbag.
    pytest.importorskip("rosbags")
    bag = write_bag(tmp_path / "src.bag", ["/cam/image_rect/compressed"], n=20)
    job = wait(c, c.put("/api/upload", params={"name": "..bag"}, content=bag.read_bytes()).json()["id"], "ready", "failed")
    assert job["video"]["name"] == "video.bag" and job["source"] == {"kind": "rosbag", "topic": "/cam/image_rect/compressed"}


def test_start_without_the_agent_needs_no_key(client, tmp_path):
    c, cfg = client
    login(c)
    job_id = uploaded(c, tmp_path)
    r = c.post(f"/api/jobs/{job_id}/start", json={"mode": "auto", "prompt": "box", "agent": False, "api_key": "not checked"})
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
    assert r.status_code == 400 and r.json()["detail"] == "fake mode: Anthropic keys start with sk-ant-" and KEY not in r.text
    for wrong in ("gpt", None, 1):
        r = c.post(f"/api/jobs/{job_id}/start", json={**start, "agent": wrong, "api_key": opus})
        assert r.status_code == 400 and r.json()["detail"] == "agent is astra, opus, none", wrong
    r = c.post(f"/api/jobs/{job_id}/start", json={**start, "api_key": opus})
    assert r.status_code == 200 and r.json()["key"] == "fake mode: key accepted"
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
    list, agent none means "user", and a list that breaks a rule gives 400 with gator.parts.propose's
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
                        "part 'lid': joint is one of revolute, prismatic, fixed, unknown, not 'hinge'")]:
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
    from gator.parts.propose import MAX_NAME
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
    assert set(server.OUTPUTS) - {"pool/lidar.json"} <= set(job["files"])  # every output but the LiDAR record
    a = job["agent_result"]
    assert (a["mode"], a["verdict"], a["joints_delivered"]) == ("articulation", "articulated", 1)
    assert set(a["counts"]) == {"exports", "reviews", "checkpoints", "joints", "articulation_reviews"}
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
    (physics uses run/parts.glb). The new outputs are served, and nothing else of the parts or the URDF."""
    c, cfg = client
    login(c)
    job_id = uploaded(c, tmp_path)
    J = cfg.runs / job_id
    r = c.post(f"/api/jobs/{job_id}/start", json={**ARTICULATED, "agent": "none", "parts": TYPED[:2]})
    assert r.status_code == 200, r.text
    job = wait(c, job_id, "done", "failed")
    assert job["status"] == "done", job["error"]
    assert (job["articulated"], job["parts_mode"], job["agent_result"]) == (True, "user", None)
    assert job["stages"]["agent"]["note"].startswith("turned off: without the agent the parts get no joints")
    assert set(job["files"]) == set(server.OUTPUTS) - {"refined.glb", "pool/lidar.json"}
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
    monkeypatch.setenv("GATOR_ROOT", str(tmp_path / "G"))
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    monkeypatch.setenv("SLURMD_NODENAME", "atl1-1-02-003-19-2")
    monkeypatch.setattr(jobs, "default_key_dir", lambda: tmp_path / "keys")
    info = tmp_path / "home" / ".config" / "gator" / "webapp.json"
    ran = []  # (port, webapp.json, its mode) while the server runs
    monkeypatch.setattr(uvicorn.Server, "run", lambda self, sockets: ran.append(
        (sockets[0].getsockname()[1], json.loads(info.read_text()), oct(info.stat().st_mode & 0o777))))
    args = ["--fake", "--runs", str(tmp_path / "runs"), "--host", "127.0.0.1"]
    server.main(args)
    port, d, mode = ran[0]
    assert mode == "0o600" and d["port"] == port > 0
    assert d["url_path"] == f"/?token={d['token']}" and len(d["token"]) >= 40
    assert d["slurm_job"] == "4242" and d["slurm_node"] == "atl1-1-02-003-19-2"
    out = capsys.readouterr().out
    assert f"8765:{d['node']}:{d['port']} ali497@login-ice.pace.gatech.edu" in out and d["token"] in out
    assert (tmp_path / "runs").is_dir()
    assert not info.exists()  # the server stopped, so start_ice.sh must not find it
    # A later start replaced the file meanwhile: it stays.
    monkeypatch.setattr(uvicorn.Server, "run", lambda self, sockets: info.write_text(json.dumps({"token": "later"})))
    server.main(args)
    assert json.loads(info.read_text()) == {"token": "later"}


def test_sigterm_deletes_the_login_file(tmp_path):
    """The real server process: SIGTERM (scancel, the end of the Slurm job) stops it through main's
    finally, so webapp.json is gone; uvicorn raises the signal again after its shutdown."""
    import signal

    home = tmp_path / "home"
    info = home / ".config" / "gator" / "webapp.json"
    env = {**os.environ, "HOME": str(home), "TMPDIR": str(tmp_path), "PYTHONPATH": str(ROOT)}
    p = subprocess.Popen([sys.executable, "-m", "gator.webapp", "--fake", "--runs", str(tmp_path / "runs"),
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


def test_start_ice_reuses_only_a_server_of_the_same_job(tmp_path):
    """start_ice.sh with a stand-in squeue: a webapp.json left by another job, or naming another node, is
    not trusted, even when something on its node and port answers 401 as the server does."""
    import http.server
    import threading

    if shutil.which("curl") is None:
        pytest.skip("no curl")
    home, bin_ = tmp_path / "home", tmp_path / "bin"
    web = home / "scratch" / "gator" / "envs" / "web" / "bin"
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
    info = home / ".config" / "gator" / "webapp.json"
    info.parent.mkdir(parents=True)

    def start(**recorded):
        info.write_text(json.dumps({"node": "127.0.0.1", "port": port, "token": "tok", **recorded}))
        return subprocess.run(["bash", str(ROOT / "gator" / "webapp" / "start_ice.sh"), "222"], capture_output=True,
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
    with where each view is in it ("seek"), says why a job has none or that its file is gone, or gives null for a
    job read before previews. The run's view images and place's overlays are served, and nothing else around
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

    # A preview that the frames stage could not write says why; one whose file is gone since says so; a job read
    # before previews has none.
    (J / "frames" / "meta.json").write_text(json.dumps({**meta, "preview": {"error": "ValueError: no libx264"}}))
    assert c.get(f"/api/jobs/{job_id}").json()["preview"] == {"error": "ValueError: no libx264"}
    (J / "frames" / "meta.json").write_text(json.dumps(meta))
    (J / "frames" / "preview.mp4").unlink()
    assert c.get(f"/api/jobs/{job_id}").json()["preview"] == {"missing": True}
    assert c.get(url).status_code == 404
    (J / "frames" / "meta.json").write_text(json.dumps({k: v for k, v in meta.items() if k != "preview"}))
    assert c.get(f"/api/jobs/{job_id}").json()["preview"] is None

    # Analyzing the input again deletes a half-written preview that a killed run left, and writes a new one.
    (J / "frames" / ".tmp_preview.mp4").write_bytes(b"cut off")
    jobs.update_stage(J, "frames", status="failed", note="cancelled", returncode=-15)
    jobs.update(J, status="cancelled")
    assert c.post(f"/api/jobs/{job_id}/continue", json={"from": "frames"}).json()["from"] == "frames"
    job = wait(c, job_id, "ready", "failed")
    assert job["status"] == "ready" and job["preview"]["frames"] == 32 and (J / "frames" / "preview.mp4").read_bytes()
    assert not (J / "frames" / ".tmp_preview.mp4").exists()


def test_no_delete_while_uploading(tmp_path):
    """The jobs list shows a job as soon as its upload starts; deleting it must wait for the upload."""
    import anyio
    import httpx

    cfg = steps.Config(G=tmp_path / "G", runs=tmp_path / "runs", fake=True)
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
# `historyCalls` ([kind, url]), and assignments to location.hash go to `hashSets`.
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
  fetch, agents, partsModes, posts, requests, texts, lists, hashSets, historyCalls });
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


def test_app_js_view0_retry_and_delete():
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
            V = { id: "j", picks: [], view0: null, job: { mode: "manual", picks: [5, 25, 45], view0: v, n_views: 6 } };
            await loadFrames();
            view0.push(V.view0);
          }
          const stages = (o) => ["frames", ...BUILD].map((name) => ({ name, status: o[name] || "pending" }));
          const retry = [{ frames: "failed" }, { frames: "pending" }, { frames: "done", extract: "done", masks: "failed" },
            { frames: "done" }].map((o) => retryFrom({ stages: stages(o) }));
          // Stopped at the part list: back to the list until it is confirmed, then from parts. A job without
          // LiDAR has its lidar stage skipped; a failed lidar stage is retried from lidar.
          const atList = { frames: "done", extract: "done", masks: "done", prep: "done", da3: "done", lidar: "skipped",
            select: "done", sam3d: "done", place: "done", propose: "done", parts: "failed" };
          const st = { articulated: true, parts_mode: "vlm_edit", stages: stages(atList) };
          V = { job: { parts: { mode: "vlm_edit", confirmed: false } } };
          retry.push(retryFrom(st));
          V = { job: { parts: { mode: "vlm_edit", confirmed: true } } };
          retry.push(retryFrom(st), retryFrom({ ...st, parts_mode: "vlm" }));
          retry.push(retryFrom({ stages: stages({ frames: "done", extract: "done", masks: "done", prep: "done", da3: "done",
            lidar: "failed" }) }));
          texts.length = 0;
          await showList();
          console.log(JSON.stringify({ view0, retry, deletes: texts.filter((t) => t === "Delete").length }));
        })()
        """)
    assert out["view0"] == [25, 25, 45, None, None]  # the pool name of a snapped pick is that pick (25 -> 26)
    assert out["retry"] == ["frames", "frames", "masks", None, "propose", "parts", "parts", "lidar"]
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
    job = {"source": {"kind": "rosbag", "topic": None}, "agent": False, "agent_result": None, "files": ["run/posed.glb"],
           "stages": {"frames": {"status": "pending"}, "agent": stages[-1]}}
    out = run_app_js({"/api/jobs": [], "/api/jobs/j/file/frames/topics.json": topics}, """
        (async () => {
          V = { id: "j" };
          await loadTopics(""" + json.dumps(job) + """);
          const table = [...texts];
          texts.length = 0;
          renderState({ status: "done", stages: """ + json.dumps(stages) + """, agent: false });
          const state = [...texts];
          texts.length = 0;
          await renderResults(""" + json.dumps(job) + """);
          console.log(JSON.stringify({ table, state, results: [...texts] }));
        })()
        """)
    assert {"rectified", "raw lens distortion", "unknown", "/cam_a", "/image_rect", "/compressed", "CompressedImage jpeg", "1280x1024",
            "10.0 Hz", "Use this topic"} <= set(out["table"])
    assert "MV-SAM3D expects undistorted images; pick a rectified topic if the bag has one" in out["table"]
    assert "no camera_info next to it, so the lens distortion is not known. MV-SAM3D expects undistorted images; " \
           "pick a rectified topic if the bag has one" in out["table"]  # a raw bag without camera_info gets the hint too
    assert out["table"].count("Use this topic") == 3 and "Not usable: /depth (16UC1)" not in out["table"]  # textContent
    # A stage that is turned off shows the plain "skipped" badge; its note says why (a longer badge broke the row on a phone).
    assert "skipped" in out["state"] and "turned off: physics uses the Stage A mesh" in out["state"]
    assert "skipped (turned off)" not in out["state"]
    # Once, in the facts' Agent row: the viewers have no placeholder that says it again.
    assert out["results"].count("not refined (agent turned off)") == 1


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
    assert "Refined by Claude Opus 5.5" in out["opus"] and "Claude Opus 5.5 (claude-opus-5-5)" in out["opus"]
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
                             "line 1: the joint is one of revolute, prismatic, fixed, unknown",
                             "line 1: the part needs a name", "line 1: write the part as name > parent : joint",
                             "line 2: the joint is one of revolute, prismatic, fixed, unknown"]
    assert out["empty"] == [] and out["box"] == [{"name": "box x", "parent": None, "joint": None, "count": 1}]


def test_app_js_start_form_sends_the_parts():
    """The default start body is a rigid object (articulated false, no parts). The toggle shows who lists
    the parts; "I type the parts" sends the parsed list; without an agent only "I type the parts" is left,
    and the user's own choice comes back with the agent."""
    out = run_form_js("""
        (async () => {
          const sent = [], form = [];
          const state = () => ({ box: $("#partsbox").hidden, list: $("#partlistbox").hidden, pause: $("#partspause").hidden,
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
        {"box": False, "list": True, "pause": True, "noagent": True, "required": False, "checked": ["vlm"],
         "disabled": [False, False, False]},
        {"box": False, "list": True, "pause": False, "noagent": True, "required": False, "checked": ["vlm_edit"],
         "disabled": [False, False, False]},
        {"box": False, "list": False, "pause": True, "noagent": False, "required": True, "checked": ["user"],
         "disabled": [True, True, False]},
        {"box": False, "list": True, "pause": False, "noagent": True, "required": False, "checked": ["vlm_edit"],
         "disabled": [False, False, False]}]
    assert (vlm_edit["articulated"], vlm_edit["parts_mode"], vlm_edit["parts"], vlm_edit["api_key"]) == (True, "vlm_edit", [], "sk-typed")
    assert (user["parts_mode"], user["agent"]) == ("user", "astra") and user["parts"] == [
        {"name": "box body", "parent": None, "joint": None, "count": 1},
        {"name": "lid", "parent": "box body", "joint": "revolute", "count": 1},
        {"name": "drawer", "parent": "box body", "joint": "prismatic", "count": 2}]
    assert bad == "line 2: the joint is one of revolute, prismatic, fixed, unknown"
    assert (none["agent"], none["api_key"], none["parts_mode"]) == ("none", None, "user")
    assert none["parts"] == [{"name": "box body", "parent": None, "joint": None, "count": 1},
                             {"name": "lid", "parent": "box body", "joint": None, "count": 1}]


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
          const labels = [stageLabel("agent", true), stageLabel("agent", false), stageLabel("agent", true, true),
            stageLabel("propose"), stageLabel("parts"), stageLabel("urdf")];
          V = { id: "j" };  // act's refresh left a stand-in job
          renderState({ status: "running", current: "agent", stages, agent: "opus", key_ok: true, warnings: [], articulated: true });
          console.log(JSON.stringify({ shown, next, reset, posts: sent, colon: posts, error, labels, build: BUILD,
            running: texts.filter((t) => t.startsWith("Running ")) }));
        })()
        """, replies)
    shown = out["shown"]
    assert "Claude Opus 5.5 listed 1 moving part of the wooden box. The base, which does not move, is body." in shown["texts"]
    assert {"a warning", "lid", ": a hinge along the back top edge"} <= set(shown["texts"])
    assert shown["text"] == "body\nlid > body : revolute" and shown["reset"] is True and shown["fig"] is False
    assert shown["photo"] == "/api/jobs/j/file/scene/sam3_wooden_box/check/f00012.jpg"
    assert out["next"] == "Check the part list, edit it if it is wrong, then Build with these parts."
    assert out["reset"] is False
    assert out["posts"] == [
        {"url": "/api/jobs/j/continue", "body": {"api_key": None}},
        {"url": "/api/jobs/j/parts", "body": {"api_key": None, "parts": [
            {"name": "body", "parent": None, "joint": None, "count": 1},
            {"name": "lid", "parent": "body", "joint": "revolute", "count": 1, "axis_hint": "a hinge along the back top edge"},
            {"name": "flap", "parent": "body", "joint": "prismatic", "count": 1}]}}]
    assert out["error"] == "line 2: the joint is one of revolute, prismatic, fixed, unknown"
    assert out["colon"] == [{"url": "/api/jobs/j/continue", "body": {"api_key": None}},
                            {"url": "/api/jobs/j/parts", "body": {"api_key": None, "parts": [
                                {"name": "body", "parent": None, "joint": None, "count": 1},
                                {"name": "left door", "parent": "body", "joint": "revolute", "count": 1}]}}]
    assert out["labels"] == ["agent sets the joints, 15 min budget", "agent refinement, 10 min budget",
                             "agent sets the joints", "part list", "split into parts", "URDF"]
    assert out["running"] == ["Running agent (agent sets the joints, 15 min budget)"]
    assert out["build"] == steps.NAMES[steps.NAMES.index("extract"):] == jobs.BUILD  # lidar after da3


def test_app_js_results_of_an_articulated_job():
    """The split parts and the agent's parts with joints side by side, the links, base and joints of the MJCF,
    the agent's joint count, the mass of every link and the parts the split missed; the new files are in the
    Downloads tab, not in Results. Without the agent the parts are welded into one body. A rigid job's part
    stages show as turned off."""
    mjcf = {"name": "wooden_box", "root": "body", "fixed_base": False, "welded": {},
            "inertial": {"mass": 2.5, "com": [0.0, 0.01, 0.1], "fullinertia": [0.01] * 6},
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
          console.log(JSON.stringify({ art, dl, none, state, unchanged: [...texts] }));
        })()
        """)
    art = out["art"]
    for t in ("Parts (split, before rigging)", "Parts with joints, by Claude Opus 5.5", "articulated: the parts have joints (finish)",
              "Joints", "1", "Joint changes by the agent", "2", "Links", "2: body, lid", "Base", "body, free to move",
              "Joint of lid", "turns on body, 0 to 86 deg", "Mass", "2.50 kg", "Link masses",
              "body 2.00 kg, lid 0.500 kg (from its collision hulls)", "Parts not split off", "drawer (no SAM 3 mask in any view)",
              "Part masks in the views (2)"):
        assert t in art, t
    assert "Mass note" not in art  # the MJCF's mass covers the link that VoMP skipped
    assert not {"parts.glb", "model.urdf", "results.zip"} & set(art)  # no downloads in Results
    assert {"parts.glb", "parts.json", "model.urdf", "results.zip", "Meshes", "Simulation", "Records"} <= set(out["dl"])
    assert any("the agent's result and the split parts (parts.glb, parts.json)." in t for t in out["dl"])
    welded = "no joints (agent turned off), so physics welds the parts into one body"
    assert out["none"].count(welded) == 1 and "1: the parts have no joints, so physics welds them into one body" in out["none"]
    assert "Mass note" in out["none"] and "VoMP could not voxelize lid, so the mass covers only body." in out["none"]
    assert out["state"].count("turned off: rigid object") == 2 and out["state"].count("skipped") == 2
    assert "agent refinement, 10 min budget" in out["state"]
    assert "Agent result without joints, by Claude Opus 5.5" in out["unchanged"]
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
            "height": 128, "rectified": True, "frame_id": "camera_optical_frame", "camera_axes": "optical", "tf_links": links}


def frames_meta(**video):
    """frames/meta.json of 20 frames; video adds or replaces fields of "video"."""
    return {"times": [round(0.1 * i, 1) for i in range(20)], "rel": [1.0] * 20, "candidates": [3, 9, 15], "thumbs": [3, 9, 15],
            "thumb_size": [32, 18], "video": {"n_frames": 20, "duration_s": 2.0, "fps": 10, "width": 192, "height": 128,
                                              "codec": "h264", "bit_depth": 8, "hdr": None, **video}}


def test_app_js_upload_kinds_and_the_jobs_list():
    """A .zip is a Stray Scanner upload; another suffix is refused with the three kinds named. The jobs list
    names each job's kind, and the stage list has the lidar stage after da3."""
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
    assert out["refused"] == ("notes.txt is not a video (.mp4 .mov .m4v .avi .mkv .webm), a rosbag (.bag .mcap .db3) or a "
                              "Stray Scanner export (.zip). For a ROS 2 bag folder, upload the .mcap or .db3 file in it. "
                              "For a Stray Scanner export folder, zip the folder first.")
    assert out["opened"] == [["PUT", "/api/upload?name=my%20capture.zip"], "my capture.zip"]
    assert {"video", "rosbag", "Stray capture"} <= set(out["list"]) and "stray" not in out["list"]
    assert out["label"] == "metric scale (LiDAR)"
    assert out["build"][out["build"].index("da3") + 1] == "lidar" and out["build"].index("lidar") + 1 == out["build"].index("select")


def test_app_js_lidar_box_per_kind():
    """The LiDAR box of the start form and the start body's "lidar" for each kind of input: hidden for a
    video (null); a bag with one LiDAR topic preselects it, and its TF chain shows; a bag with two asks for
    one and says when TF does not link it; a bag without LiDAR topics, or listed before they were, has the
    box turned off with the reason; a Stray capture starts with LiDAR on, and its turn and depth show."""
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
        "b5": {"topics": [camera_entry({})], "skipped": []}}  # listed before LiDAR topics were
    stray_video = {"kind": "stray", "rotation": 270, "rotation_source": "gravity", "container_rotation": 0, "n_depth": 19,
                   "n_confidence": 19, "n_odometry": 20, "depth_size": [256, 192], "upright_agreement": 0.9583,
                   "warnings": ["1 of the 20 frames have no depth image, so they have no LiDAR samples"]}
    replies = {f"/api/jobs/{i}/frames": frames_meta() for i in ("v", "b1", "b2", "b3", "b4", "b5", "b6")}
    replies.update({f"/api/jobs/{i}/file/frames/topics.json": t for i, t in topics.items()})
    replies.update({"/api/jobs/b3/file/frames/topics.json": topics["b2"], "/api/jobs/b6/file/frames/topics.json": topics["b5"],
                    "/api/jobs/s1/frames": frames_meta(**stray_video),
                    "/api/jobs/s2/frames": frames_meta(**{**stray_video, "rotation": 0, "upright_agreement": None, "warnings": []})})
    job = {"status": "ready", "mode": "auto", "picks": [], "n_views": 4, "auto_continue": True, "agent": "none",
           "prompt": None, "lidar": None, "calibration": None}
    bag = {**job, "kind": "rosbag", "source": {"kind": "rosbag", "topic": CAM}}
    jobs_ = {"v": {**job, "kind": "video", "source": {"kind": "video", "topic": None}}, "b1": bag, "b2": bag,
             "b3": {**bag, "lidar": {"kind": "rosbag", "topic": "/livox/lidar"}}, "b4": bag, "b5": bag,
             "b6": {**bag, "status": "done"},
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
          // The camera topic table: an old listing offers the topic in use again, so that its LiDAR topics get listed.
          const read = { source: JOBS.b1.source, stages: { frames: { status: "done" } } };
          for (const id of ["b5", "b1"]) {
            V = { id };
            texts.length = 0;
            await loadTopics(read);
            out["table_" + id] = texts.filter((t) => t === "Use this topic" || t === "in use");
          }
          console.log(JSON.stringify(out));
        })()
        """, replies)
    off = "Off: Depth Anything 3 estimates the scale."
    v = out["v"]
    assert v["box"]["hidden"] is True and v["sent"] is None

    b1 = out["b1"]
    assert b1["labels"] == ["/livox/lidar (Livox, 10 Hz, livox_frame)"]
    assert b1["box"] == {"hidden": False, "on": False, "off": False, "hint": off, "opts": True, "topic": "/livox/lidar",
                         "link": None, "warn": None, "skipped": None}
    assert b1["sent"] is None  # off by default for a bag
    assert b1["on"] == {"hidden": False, "on": True, "off": False, "hint": "The LiDAR's points set the metric scale.",
                        "opts": False, "topic": "/livox/lidar", "warn": False,
                        "link": "linked by /tf_static: camera_optical_frame <- camera_link <- base_link <- livox_frame",
                        "skipped": "Not usable: /ouster/points (no messages)"}
    assert b1["sent_on"] == {"topic": "/livox/lidar"} and b1["stray_hidden"] is True

    b2 = out["b2"]
    assert b2["labels"] == ["Choose a LiDAR topic", "/livox/lidar (Livox, 10 Hz, livox_frame)",
                            "/lidar/points (PointCloud2, 9.9 Hz, lidar)"]
    assert (b2["box"]["topic"], b2["box"]["on"]) == ("", False)  # two topics: none preselected
    assert b2["none"]["link"] == "choose a LiDAR topic to see its link to the camera"
    assert b2["error"] == "choose a LiDAR topic, or turn LiDAR off" and b2["focused"] is True  # the page's text, not a bubble
    assert (b2["cloud"]["link"], b2["cloud"]["warn"]) == ("no TF link: upload the calibration", True)
    assert (b2["livox"]["link"], b2["livox"]["warn"]) == ("linked by /tf: camera_optical_frame <- base_link <- livox_frame", False)
    assert b2["sent"] == {"topic": "/livox/lidar"}
    assert (out["b3"]["on"], out["b3"]["topic"], out["b3"]["opts"]) == (True, "/livox/lidar", False)  # the job's own setting

    no_lidar, old, old_done = out["b4"], out["b5"], out["b6"]
    assert no_lidar["box"]["off"] is True and no_lidar["box"]["on"] is False and no_lidar["box"]["opts"] is True
    assert no_lidar["box"]["hint"] == ("This bag has no LiDAR topic (PointCloud2 or Livox). Not usable: /ouster/points "
                                       "(no messages).")
    assert old["box"]["off"] is True and old["box"]["hint"] == (
        "This bag was read before the app listed LiDAR topics. To list them, use Change the camera topic in the Inputs tab, "
        "and choose the same topic again.")
    assert old_done["box"]["hint"] == ("This bag was read before the app listed LiDAR topics. To use its LiDAR, upload the "
                                       "bag again.")
    assert no_lidar["sent"] is None and old["sent"] is None and old_done["sent"] is None
    assert old_done["checked_off"] == [True, None]

    s1 = out["s1"]
    assert s1["box"] == {"hidden": False, "on": True, "off": False, "hint": "The iPhone's LiDAR depth sets the metric scale.",
                         "opts": True, "topic": "", "link": None, "warn": None, "skipped": None}
    assert s1["sent"] == {"kind": "stray"}
    assert (s1["off"]["on"], s1["off"]["hint"], s1["sent_off"]) == (False, off, None)
    assert s1["info"] is False and s1["vsum"] == "Stray capture: 20 frames, 2.0 s, 3 candidate views"
    assert s1["facts"] == ("The frames are turned 270 deg counterclockwise to stand upright. Gravity gives this turn in 96% "
                           "of the frames that show it. 19 of 20 frames have a depth image (256x192). 19 have a confidence image.")
    assert s1["warn"] == ["1 of the 20 frames have no depth image, so they have no LiDAR samples"]
    s2 = out["s2"]
    assert s2["box"]["on"] is False and s2["box"]["hint"] == off  # a Start turned it off
    assert s2["facts"].startswith("No frame shows gravity, so the frames stay as stored. 19 of 20 frames")
    # Only a bag's frames load its topics.json, once per load.
    assert out["topics"] == ["b1", "b2", "b3", "b4", "b5", "b6", "b1"]
    assert out["table_b5"] == ["Use this topic"] and out["table_b1"] == ["in use"]


def test_app_js_calibration_upload():
    """The calibration uploads as soon as it is chosen: the box shows the server's summary, or its reason
    for refusing the file; Remove deletes it. The status line says when it worked. The upload, Start, Retry
    and Build wait for the request, and the focus comes back (to the upload after a Remove). In review, the
    scale's summary says when it is stale."""
    found = {"topics": [camera_entry({"/lidar/points": None})], "skipped": [], "lidar": [CLOUD], "lidar_skipped": [], "tf": []}
    summary = {"name": "my_calib.json", "bytes": 412, "translation_m": 0.1234, "rotation_deg": 120.0, "has_intrinsics": False,
               "camera_axes": "optical", "time_offset_s": 0.0}
    lid = {"ok": True, "s": 1.25, "reason": None, "spread_rel": 0.004, "warnings": [], "views": [], "calibration_changed": False,
           "counts": {"views": 6, "views_used": 6}, "check_dir": "lidar/check"}
    replies = {"/api/jobs/b/calibration?name=my%20calib.json": summary,
               "/api/jobs/b/calibration?name=bad.yaml": {"__status": 400,
                                                         "detail": "T_cam_lidar has 3 numbers; give a 4x4 matrix or 16 numbers"},
               "/api/jobs/b/calibration": {"calibration": None},
               "/api/jobs/b/calibration?name=k.yaml": {**summary, "name": "k.yaml", "has_intrinsics": True,
                                                       "camera_axes": "body", "time_offset_s": 0.02},
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
          out.job = V.job.calibration.name;
          document.activeElement = $("#calremove");
          await $("#calremove").onclick();
          out.removed = box();
          out.during = during.splice(0);
          out.after = ids.map((id) => $(id).disabled);
          out.focused = focused.splice(0);
          await pick("k.yaml");
          out.k = box().sum;
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
    assert out["before"] == {"link": "no TF link: upload the calibration", "warn": True, "sum": None, "btn": "Upload a calibration",
                             "remove": True, "msg": "", "bad": False}
    assert out["cleared"] == ""  # the same file can be chosen again
    assert out["good"] == {"link": "no TF link: the calibration links them", "warn": False, "remove": False,
                           "msg": "The calibration my_calib.json is saved.", "bad": False, "btn": "Replace the calibration",
                           "sum": "Calibration my_calib.json: the LiDAR is 12.3 cm from the camera, turned 120.0 deg. "
                                  "K comes from the camera_info."}
    # A refused file changes nothing on the server: the summary stays, and the reason shows.
    assert out["bad"] == {**out["good"], "msg": "T_cam_lidar has 3 numbers; give a 4x4 matrix or 16 numbers", "bad": True}
    assert out["job"] == "my_calib.json"
    assert out["removed"] == {**out["before"], "msg": "The calibration is removed."}
    assert out["during"] == [[True] * 5] * 3 and out["after"] == [False] * 5
    # The upload keeps the focus; Remove hides after a delete, so the upload button takes it.
    assert out["focused"] == ["#calfile", "#calfile", "#calfile"]
    assert out["k"] == ("Calibration k.yaml: the LiDAR is 12.3 cm from the camera, turned 120.0 deg. It gives K. Its camera "
                        "axes are body axes (x forward, y left, z up). Its time offset is 0.02 s.")
    assert out["sent"] == {"topic": "/lidar/points"}
    assert "The calibration changed after this scale was found. Build finds the scale again first, then stops here." \
        in out["review"]
    assert out["requests"] == ["PUT /api/jobs/b/calibration?name=my%20calib.json", "PUT /api/jobs/b/calibration?name=bad.yaml",
                               "DELETE /api/jobs/b/calibration", "PUT /api/jobs/b/calibration?name=k.yaml",
                               "PUT /api/jobs/b/calibration?name=my%20calib.json", "GET /api/jobs/b/selection"]


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
           "parts": {}, "calibration": None, "lidar": {"kind": "rosbag", "topic": "/livox/lidar"},
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
          out.edit = [V.tab, V.pick, $("#frames").hidden, focused, $("#framesum").textContent, $("#start").textContent,
            mode(), $("#lidaropts").hidden, location.hash];
          await refresh();  // the same status: the tab stays
          out.kept = [V.tab, $("#frames").hidden, $("#review").hidden];
          console.log(JSON.stringify(out));
        })()
        """, replies)
    j = out["j"]
    for t in ("Metric scale from LiDAR: x1.043 (14 of 16 views, spread 2.1%)", "a LiDAR warning", "LiDAR check images (3)",
              "f00003: x1.05 (280 pts)", "f00009: x1.21 (1,200 pts, outlier)", "f00015: few points (12 pts)",
              "LiDAR x1.05 (280 pts)", "LiDAR x1.21 (1,200 pts, outlier)", "LiDAR: few points (12 pts)",
              "/api/jobs/j/file/lidar/check/f00009.jpg", "LiDAR points in f00009"):
        assert t in j, t
    assert not any("calibration changed" in t for t in j)
    assert ("The green line is the mask outline. Blue points are nearer than the scaled depth, and red points are farther. "
            "White points agree with it. Open an image to see it larger.") in j
    assert "Scale: Depth Anything 3's estimate (no LiDAR)" in out["k"] and not any("LiDAR x" in t for t in out["k"])
    # LiDAR is on, but its record is not there yet (Build runs the lidar stage first).
    assert "Metric scale from LiDAR: not found yet. Build finds it first, then stops here." in out["m"]
    # Only a bag with LiDAR on can change its calibration in review.
    assert "Change the calibration" in j and "Change the calibration" not in out["k"] + out["m"]
    assert out["before"] == ["inputs", True, False]  # the review sits above the Inputs tab; the form waits in Settings
    assert out["edit"] == ["settings", "settings", False, "#calfile", "Start over with other settings", "Start over", "auto",
                           False, "#/job/j/settings"]
    assert out["kept"] == ["settings", False, False]


def test_app_js_lidar_failure_panel():
    """A job that failed at the lidar stage: a panel with the reason from pool/lidar.json and the check
    images, and the next step (a calibration for a bag, another one when it has one; not for a Stray capture).
    Retry starts at lidar. refresh() shows the panel only for a job that failed there."""
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
            "stages": {x["name"]: {"status": x["status"]} for x in stages}}
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
          // The page as refresh() draws it: the panel for the job that failed at lidar, not for a done job.
          out.shown = [];
          for (const id of ["j", "d"]) {
            V = { id, picks: [], view0: null };
            $("#log").textContent = "a log";  // not fetched again
            $("#lidarfail").hidden = true;
            texts.length = 0;
            await refresh();
            out.shown.push([$("#lidarfail").hidden, texts.some((x) => x.startsWith("No metric scale from LiDAR: "))]);
          }
          console.log(JSON.stringify(out));
        })()
        """, replies)
    for t in ("No metric scale from LiDAR: " + reason, "a stage warning", "LiDAR check images (2)", "f00003: x1.05 (280 pts)",
              "Go to the LiDAR settings"):
        assert t in out["panel"], t
    # The scale failed, so each view's points are coloured by its own scale (white without one).
    assert ("The green line is the mask outline. Blue points are nearer than the scaled depth, and red points are farther. "
            "White points agree with it. Each view uses its own scale here. A view without one shows all its points white. "
            "Open an image to see it larger.") in out["panel"]
    # The next step names the tab where the calibration and the LiDAR box are.
    assert out["next"] == "Upload a calibration in the Settings tab, then Retry. Or turn LiDAR off there and build again."
    assert out["cal"] == "Replace the calibration in the Settings tab, then Retry. Or turn LiDAR off there and build again."
    assert out["retry"] == [False, "Retry from lidar"]
    assert out["stray"] == ("Turn LiDAR off or change the views in the Settings tab, then build again. Retry helps only "
                            "after a file or disk error.")
    assert out["masks"] == "Read the log of the failed stage, then Retry."
    assert ("The stage stopped before it wrote its record. The reason is above, and its log is in the Stages and log "
            "tab.") in out["norecord"]
    assert out["failed"] == [True, False, False]
    assert out["shown"] == [[False, True], [True, False]]


def test_app_js_changed_calibration_retries_from_lidar():
    """A failed or cancelled bag job whose scale was found with another calibration: the server retries from
    lidar (Runner.resume), so Retry says so, sends from lidar, and the next step and the Build again hint say
    why. The page learns it from /selection's calibration_changed after a load or an upload, or, without a
    selection, from pool/lidar.json's calibration file. A reply that starts earlier than asked is shown."""
    stages = [{"name": n, "status": "done"} for n in ("topics", "frames", "extract", "masks", "prep", "da3", "lidar", "select")] + [
        {"name": "sam3d", "status": "failed"}] + [{"name": n, "status": "pending"} for n in ("place", "propose", "parts", "agent")]
    job = {"id": "c", "kind": "rosbag", "status": "failed", "video": {"name": "scan.bag"}, "prompt": "box", "current": None,
           "error": "sam3d: out of memory", "warnings": [], "key_ok": False, "agent": "none", "articulated": False,
           "parts_mode": "vlm", "parts_review": False, "mode": "auto", "picks": [], "view0": None, "n_views": 4,
           "auto_continue": True, "parts": {}, "calibration": None, "lidar": {"kind": "rosbag", "topic": "/lidar/points"},
           "source": {"kind": "rosbag", "topic": CAM}, "files": ["selection.json", "pool/lidar.json"], "progress": stages,
           "stages": {x["name"]: {"status": x["status"]} for x in stages}}
    found = {"topics": [camera_entry({"/lidar/points": None})], "skipped": [], "lidar": [CLOUD], "lidar_skipped": [], "tf": []}
    summary = {"name": "c.yaml", "bytes": 300, "translation_m": 0.1, "rotation_deg": 90.0, "has_intrinsics": False,
               "camera_axes": "optical", "time_offset_s": 0.0}
    replies = {"/api/jobs/c": job, "/api/jobs/c/frames": frames_meta(), "/api/jobs/c/file/frames/topics.json": found,
               "/api/jobs/c/calibration?name=c.yaml": summary, "/api/jobs/c/calibration": {"calibration": None},
               "/api/jobs/c/continue": {"status": "queued", "from": "lidar"}}
    out = run_form_js("""
        (async () => {
          // /selection and pool/lidar.json as the server has them now.
          let stale = false, ext = null;
          const fetch0 = fetch, reply = (x) => Promise.resolve({ ok: true, status: 200, json: async () => x });
          globalThis.fetch = (url, opt) => url === "/api/jobs/c/selection" ? reply({ lidar: { ok: true, calibration_changed: stale } })
            : url === "/api/jobs/c/file/pool/lidar.json" ? reply({ ok: true, extrinsic: ext })
              : url.includes("/file/logs/") ? Promise.resolve({ ok: false, status: 404, text: async () => "" }) : fetch0(url, opt);
          const page = () => [$("#retry").textContent, $("#jnext").textContent, $("#rebuildhint").textContent];
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
          // The page asked for sam3d, but the server starts from lidar (it knew better): the reply says so.
          stale = false;
          await refresh();
          posts.length = 0;
          await $("#retry").onclick();
          out.asked = [posts[0].body.from, $("#runmsg").textContent];
          // Without a selection: pool/lidar.json's calibration file against the job's.
          const J = """ + json.dumps(job) + """, nosel = { ...J, files: ["pool/lidar.json"] }, tf = { source: "tf" };
          const cal = (name) => ({ source: "calibration", file: name });
          out.bylidar = [];
          for (const [e, c] of [[tf, null], [cal("a.yaml"), null], [cal("a.yaml"), "a.yaml"], [cal("a.yaml"), "b.yaml"], [tf, "a.yaml"]]) {
            ext = e;
            out.bylidar.push(await scaleStale({ ...nosel, calibration: c && { name: c } }));
          }
          ext = tf;
          out.never = [await scaleStale({ ...nosel, kind: "stray", lidar: { kind: "stray" } }), await scaleStale({ ...nosel, lidar: null }),
            await scaleStale({ ...nosel, calibration: { name: "a.yaml" }, stages: { ...J.stages, lidar: { status: "failed" } } }),
            await scaleStale({ ...nosel, calibration: { name: "a.yaml" }, files: [] })];
          console.log(JSON.stringify(out));
        })()
        """, replies)
    stopped = " Retry, at the top, continues from the stage that stopped instead."
    changed = (" Retry, at the top, starts from the lidar stage instead, because the calibration changed since the scale was "
               "found.")
    again = "This runs every stage again with these settings."
    stale_next = "The calibration changed since the scale was found. Retry starts again from the lidar stage."
    assert out["current"] == ["Retry from sam3d", "Read the log of the failed stage, then Retry.", again + stopped]
    assert out["upload"] == ["Retry from lidar", stale_next, again + changed,
                             "The calibration c.yaml is saved. The scale was found with the old calibration, so Retry starts "
                             "from the lidar stage."]
    assert out["sent"] == "lidar" and out["runmsg"] == ""  # the server started where the page said
    assert out["removed"] == [*out["current"], "The calibration is removed."]
    assert out["reload"] == out["upload"][:3]
    assert out["asked"] == ["sam3d", "The calibration changed since the scale was found, so the run starts from lidar."]
    assert out["bylidar"] == [False, True, False, True, True]
    assert out["never"] == [False, False, False, False]


def test_app_js_results_show_the_metric_scale():
    """The results give the metric scale that select applied (placement.json) with its LiDAR source, views
    and spread, or DA3's own estimate; the Downloads tab's sentence on results.zip names lidar.json when the job
    has it, and the agent's result only when there is one."""
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
               "/api/jobs/n/file/run/placement.json": {**place, "metric_scale": None, "units": "DA3 units (meters)"},
               "/api/jobs/o/file/run/placement.json": {"check": {"mean_mask_iou": 0.9}}}  # made before LiDAR
    cases = {"j": {**base, "files": files}, "s": {**base, "id": "s", "kind": "stray", "lidar": {"kind": "stray"}, "files": files},
             "n": {**base, "id": "n", "lidar": None, "files": [f for f in files if f != "pool/lidar.json"]},
             "o": {**base, "id": "o", "lidar": None, "files": ["run/posed.glb", "run/placement.json"]}}
    out = run_app_js(replies, """
        (async () => {
          const out = {}, cases = """ + json.dumps(cases) + """;
          for (const [id, job] of Object.entries(cases)) {
            V = { id };
            texts.length = 0;
            await renderResults(job);
            const k = texts.indexOf("Metric scale");
            out[id] = { row: k >= 0 ? texts[k + 2] : null };
            renderDownloads(job);
            out[id].zip = texts.find((t) => t.startsWith("results.zip holds")) || null;
          }
          console.log(JSON.stringify(out));
        })()
        """)
    assert out["j"]["row"] == "x1.043 from LiDAR (/livox/lidar, 14 views, spread 2.1%)"
    assert out["j"]["zip"].endswith(", the view selection and the LiDAR scale (lidar.json). "  # the agent was off
                                    "The MJCF and USD models come only in it.")
    assert out["s"]["row"] == "x1.250 from LiDAR (iPhone LiDAR depth, 6 views, spread 0.4%)"
    assert out["n"]["row"] == out["o"]["row"] == "Depth Anything 3's estimate (no LiDAR)"
    assert "lidar.json" not in out["n"]["zip"] and out["o"]["zip"] is None


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
    drawing the video again; the toggle swaps every image for place's overlay. A job without a preview says why in
    one sentence, and its views are no buttons. The panel is drawn once per selection, so a refresh does not stop
    the video, and a selection.json that comes after another job's page draws nothing."""
    seek = {"152": {"frame": 152, "at": 15.0}, "40": {"frame": 40, "at": 3.8}, "301": {"frame": 300, "at": 29.9}}
    first = input_job("j", preview={**PREVIEW, "seek": seek})
    run = {**first, "files": ["selection.json", "run/placement.json"]}  # place has ended since
    old = input_job("m")  # a preview without "seek": the page seeks by the pick's time
    bag = input_job("k", kind="rosbag", video={"name": "scan.bag"}, source={"kind": "rosbag", "topic": "/cam/image_rect"},
                    preview=None)
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
          const toggle = made.find((e) => e.tag === "button" && text(e) === "Show the mesh fit");
          toggle.onclick({ currentTarget: toggle });
          out.overlay = [toggle.attrs["aria-pressed"], btns.map((b) => b.kids[0].attrs.src)];
          toggle.onclick({ currentTarget: toggle });
          out.photos = [toggle.attrs["aria-pressed"], btns[2].kids[0].attrs.src];
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
    assert out["caption"] == "table.MOV (video) Preview at 640x360, 15 fps."
    caps = ["view 0, frame 152, 15.2 s", "view 1, frame 40, 4.0 s", "view 2, frame 301, 30.1 s"]
    assert out["caps"] == caps
    assert out["buttons"] == [["button", c + ": show it in the video", t] for c, t in zip(caps, ["★", "", ""])]
    assert [i["src"] for i in out["imgs"]] == [f"/api/jobs/j/file/run/input/images/{i}.png" for i in range(3)]
    assert all(i["loading"] == "lazy" and i["alt"] == "" and i["style"] == "aspect-ratio: 640 / 360" for i in out["imgs"])
    assert out["head0"] == "Views used (3)"  # no toggle before place
    assert "Click a view to see its frame in the video. ★ marks view 0." in out["hint0"]
    time1, msg1, calls, at1, at0, current = out["seek1"]
    assert time1 == pytest.approx(3.801) and calls == ["pause", "scroll"] and at1 == ["at"] and at0 == []
    assert msg1 == "The video shows view 1 at 3.8 s: frame 40." and current == ["false", "true", "false"]
    assert out["seek2"][0] == pytest.approx(29.901)
    assert out["seek2"][1] == "The video shows view 2 at 29.9 s: frame 300, since the preview has no frame 301."
    assert out["seek0"][0] == pytest.approx(15.001)
    assert out["seek0"][1:] == ["The video shows view 0 at 15.0 s: frame 152.", ["at"], [], ["true", "false", "false"]]
    # The same video and images, still at view 0; the heading gains the toggle.
    assert out["placed"][:6] == [1, 3, 2, ["at"], "true", "Views used (3)Show the mesh fit"]
    assert out["placed"][6] == "The video shows view 0 at 15.0 s: frame 152."
    assert ("Click a view to see its frame in the video. ★ marks view 0. The mesh fit shows the posed mesh in red and the "
            "outline of the SAM 3 mask in green.") in out["hint1"]
    assert out["overlay"] == ["true", [f"/api/jobs/j/file/run/overlay_{i}.png" for i in range(3)]]
    assert out["photos"] == ["false", "/api/jobs/j/file/run/input/images/2.png"]
    assert out["again"] is True
    assert out["m"][0] == pytest.approx(30.1333 - 0.2 + 0.001) and out["m"][1] == "The video shows view 2 at 29.9 s."
    assert out["k"] == {"video": False, "tags": ["div"] * 3, "alt": "", "toggle": False,
                        "p": ["scan.bag (rosbag, camera topic /cam/image_rect)",
                              "This job has no preview video, because its input was read before the app made previews.",
                              "★ marks view 0."]}
    assert out["s"]["p"][:2] == ["capture.zip (Stray capture)", "The app could not make a preview video: ValueError: no libx264"]
    assert out["s"]["video"] is False and out["s"]["tags"] == ["div"] * 3
    assert out["g"]["p"][:2] == ["table.MOV (video)", "The preview video of this job is missing on the server."]
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
          const toggles = () => made.filter((e) => e.tag === "button" && text(e) === "Show the mesh fit").length;
          out.placed = [refreshes, made.filter((e) => e.tag === "video").length, toggles(), selections];
          streams[0].on.state(state(jobs.placed));
          out.placedAgain = [refreshes, toggles()];
          console.log(JSON.stringify(out));
        })()
        """, {"/api/jobs/j/frames": frames_meta()})
    assert out["failed"] == [False, True, ["view 0, frame 152, 15.2 s", "view 1, frame 40, 4.0 s", "view 2, frame 301, 30.1 s"],
                             1, 0]
    assert out["pending"] == [False, None, "none", 1, 1]  # the same video; the views' sentence
    assert out["live"] == [False, None, 1, 1, 1] and out["before"] == 1
    assert out["after"] == [2, True, 1, 1] and out["again"] == 2
    assert out["placed"] == [3, 1, 1, 1] and out["placedAgain"] == [3, 1]


# ---- the job page's tabs: Results, Inputs, Settings, Stages and log ----

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
    """The job page: the waiting panels, then the tab bar (Results, Downloads, Inputs, Settings, Stages and log)
    with the WAI-ARIA roles and links, then the five panels in tab order, hidden until app.js picks one. Each panel
    has the sentence it shows while it has nothing (the Settings one is set by app.js). The Results panel starts with
    the line that points to the Downloads tab, whose tab name is a button."""
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
        ("log", "Stages and log")]
    for name, panel in zip(TAB_ORDER, panels):
        tag, tab = at["tab-" + name]
        assert (tag, tab["type"], tab["role"], tab["data-tab"], tab["aria-controls"], tab["aria-selected"], tab["tabindex"]) == \
            ("button", "button", "tab", name, panel, "false", "-1")
        p = at[panel][1]
        assert (p["role"], p["aria-labelledby"], p["tabindex"], "hidden" in p) == ("tabpanel", "tab-" + name, "0", True)
    for none, text in (("resultsnone", "The results appear here when the build has made them."),
                       ("downloadsnone", "The files appear here when the build has made them."),
                       ("inputnone", "The input's facts appear here after the app reads its frames."),
                       ("framesnone", ""),
                       ("runnone", "The stages and the log appear here when the job starts.")):
        assert re.search(rf'<p id="{none}" class="emptytab">([^<]*)</p>', html).group(1) == text, none
    # The input's facts sit in the Inputs panel, the start form in the Settings panel, the log in Stages and log.
    def inside(child, parent):
        start = html.index(f'id="{parent}"')
        return start < html.index(f'id="{child}"') < html.index(f'id="{panels[panels.index(parent) + 1]}"') \
            if parent != "run" else start < html.index(f'id="{child}"')
    for child in ("topicrow", "hdr", "strayinfo", "videoinfo", "inputvideo", "inputviews"):
        assert inside(child, "input"), child
    for child in ("framesum", "inputwarn", "towarn", "startform", "grid", "lidarbox", "start"):
        assert inside(child, "frames"), child
    assert at["hdr"][1]["tabindex"] == at["straywarn"][1]["tabindex"] == "-1"  # "Show the warnings" focuses them
    for child in ("runsum", "stages", "log"):
        assert inside(child, "run"), child
    assert order.index("retry") < order.index("tabs") and order.index("cancel") < order.index("tabs")  # in the job's header
    # Results: the pointer to the Downloads tab first, then the viewers and the facts, and no download links.
    assert [order.index(x) for x in ("resultsbody", "todl", "todlbtn", "viewers", "facts2")] == \
        sorted(order.index(x) for x in ("resultsbody", "todl", "todlbtn", "viewers", "facts2"))
    assert inside("todl", "results") and inside("downloadsbody", "downloads")
    assert re.search(r'<p id="todl" class="todl">(.*?)</p>', html).group(1) == \
        'Download the files from the <button type="button" id="todlbtn" class="link" aria-controls="downloads">Downloads' \
        '</button> tab.'


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
    """Each tab shows one sentence while it has nothing: Results until a build made them (and while a new build
    runs), Downloads until a file exists (and while a build runs), Inputs until the frames are read, Settings
    until the frames are read and while the job runs, Stages and log until a stage started. The Inputs tab's
    views strip says when the views are not selected yet."""
    all_done = {n: "done" for n in ("frames", "extract", "masks", "prep", "da3", "select", "sam3d", "place", "agent")}
    cases = {"a": tab_job("a", "analyzing", files=(), frames="running"), "u": tab_job("u", "uploading", files=()),
             "r": tab_job("r", "running", frames="done", extract="done", masks="running"),
             "y": tab_job("y", "ready", files=(), frames="done"),
             "d": tab_job("d", "done", files=["selection.json", "run/posed.glb"], lidar="skipped", **all_done),
             "b": tab_job("b", "running", files=["selection.json", "run/posed.glb"], lidar="skipped",
                          **{**all_done, "sam3d": "running", "place": "pending", "agent": "pending"})}
    replies = {f"/api/jobs/{i}": j for i, j in cases.items()}
    replies.update({f"/api/jobs/{i}/frames": frames_meta() for i in cases})
    replies.update({f"/api/jobs/{i}/file/selection.json": {"ok": True, "picks": PICKS} for i in cases})
    out = run_form_js(ELEMENTS + TAB_SETUP + """
        (async () => {
          globalThis.EventSource = class { addEventListener() {} close() {} };
          const out = { texts: [SETTINGS_LATER, SETTINGS_BUSY, NO_VIEWS, FILES_LATER, FILES_BUSY] };
          for (const id of """ + json.dumps(list(cases)) + """) {
            made.length = 0;
            V = { id, picks: [], view0: null };
            await refresh();
            out[id] = { tab: V.tab, results: [$("#resultsnone").hidden, $("#resultsbody").hidden],
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
    later, busy, no_views, files_later, files_busy = out["texts"]
    assert later == "The settings appear here after the app reads the frames."
    assert busy == "You can change the settings when the job stops."
    assert no_views == "The views used appear here after the app selects them."
    assert files_later == "The files appear here when the build has made them."
    assert files_busy == "The files appear here when the build stops."
    none, shown = [False, True], [True, False]  # [the sentence hidden, the content hidden]
    assert out["a"] == {"tab": "log", "results": none, "downloads": [*none, files_later], "inputs": none,
                        "settings": [*none, later], "log": shown, "noViews": 0}
    assert out["u"] == {"tab": "log", "results": none, "downloads": [*none, files_later], "inputs": none,
                        "settings": [*none, later], "log": none, "noViews": 0}
    assert out["r"] == {"tab": "log", "results": none, "downloads": [*none, files_busy], "inputs": shown,
                        "settings": [*none, busy], "log": shown, "noViews": 1}
    assert out["y"] == {"tab": "settings", "results": none, "downloads": [*none, files_later], "inputs": shown,
                        "settings": [*shown, busy], "log": shown, "noViews": 1}
    assert out["d"] == {"tab": "results", "results": shown, "downloads": [*shown, files_later], "inputs": shown,
                        "settings": [*shown, busy], "log": shown, "noViews": 0}
    # A new build hides the old results and files.
    assert out["b"]["results"] == none and out["b"]["downloads"] == [*none, files_busy] and out["b"]["noViews"] == 0


def test_app_js_hidden_log_keeps_up_without_layout():
    """While the Stages and log tab is hidden, the live log takes its text and the stage clocks tick without
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
    """A running job's state event that says done changes the header but not the tab: the Stages and log tab stays
    while the stream sends the last lines of the log. The refresh after the stream's end draws the results, then
    shows the Results tab with them."""
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
          const event = (job) => ({ data: JSON.stringify({ status: job.status, current: null, error: null, warnings: [],
            key_ok: true, agent: "none", articulated: false, parts_mode: "vlm", parts_review: false, stages: job.progress }) });
          streams[0].on.state(event(jobs.done));
          out.state = [V.tab, $("#jnext").textContent, $("#tab-log").tabIndex, refreshes, $("#resultsbody").hidden];
          now = "done";
          streams[0].on.end();
          await pending;
          out.end = [V.tab, V.pick, refreshes, $("#resultsbody").hidden, $("#resultsnone").hidden, location.hash];
          console.log(JSON.stringify(out));
        })()
        """, {"/api/jobs/r/frames": frames_meta(), "/api/jobs/r/file/selection.json": {"ok": True, "picks": PICKS}})
    assert out["running"] == ["log", 1, True]
    # The next step names Results only: the Results tab's first line names the Downloads tab.
    assert out["state"] == ["log", "Look at the meshes in the Results tab.", 0, 1, True]
    assert out["end"] == ["results", None, 2, False, True, "#/job/r"]


def test_app_js_tabs_take_the_focus_after_actions():
    """When an action hides or disables the focused button, the focus goes where the page goes. After Build, the
    tab that the new status shows takes it; after a refused Start, Start takes it back. "Start over in manual
    mode" opens the Settings tab (a pick) with the frames open and focuses its heading. "Change the camera topic"
    focuses the topic table's heading, and "Keep the current topic" gives the focus back to the button, or to the
    tab shown when the Inputs tab does not show. The Settings tab counts the input's warnings, and "Show the
    warnings" opens the Inputs tab at the first one."""
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
          ["#framesum", "#start", "#topicshead", "#chtopic", "#hdr", "#straywarn"].forEach(spy);
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
          // Start over in manual mode.
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
    assert out["manual"] == ["settings", "settings", "#/job/v/settings", True, ["#framesum"], True]
    assert out["refused"] == ["settings", ["#start"], [], "the OpenAI key was refused"]
    assert out["change"] == [False, False, ["#topicshead"]]
    assert out["keep"] == [True, False, ["#chtopic"], []]
    assert out["keepElsewhere"] == ["settings", [], ["settings"]]
    assert out["bagWarn"] is True
    assert out["warn"] == ["settings", False, "The input has 3 warnings."]
    assert out["towarn"] == ["inputs", "inputs", "#/job/s/inputs", ["#hdr"]]


# ---- the Results tab's info buttons (app.js info()) ----

# static/app.js in node with a small DOM: every element keeps its tag, attributes, children and parent, its class
# names, a style and a box (rect, which the test sets) for getBoundingClientRect; `hidden` is the attribute. Every
# selector gives one such element (byId). document's and window's addEventListener calls go to `listeners` as
# [target, type, handler, options], and setTimeout's callbacks to `timers` ({f, ms}; clearTimeout empties f). The
# window is 375 x 700 px. fetch answers with the replies, by URL.
INFO_HARNESS = r"""
const fs = require("fs"), vm = require("vm");
const replies = JSON.parse(process.argv[4] || "{}");
const made = [], listeners = [], timers = [], byId = {};
class Elem {
  constructor(tag) {
    Object.assign(this, { tag, attrs: {}, kids: [], parent: null, style: {}, cls: new Set(),
      rect: { left: 0, top: 0, width: 0, height: 0 } });
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
  getBoundingClientRect() { const r = this.rect; return { ...r, right: r.left + r.width, bottom: r.top + r.height }; }
  get textContent() { return this.kids.map((k) => typeof k === "string" ? k : k.textContent).join(""); }
  set textContent(v) { this.kids = [String(v)]; }
  addEventListener() {}
  focus() {}
  scrollIntoView() {}
}
const document = { title: "", activeElement: null, body: {}, documentElement: { clientWidth: 375 },
  querySelector: (sel) => byId[sel] || (byId[sel] = new Elem("div")), querySelectorAll: () => [],
  createElement: (tag) => new Elem(tag), createElementNS: (ns, tag) => new Elem(tag),
  addEventListener: (type, fn, opt) => listeners.push(["document", type, fn, opt]) };
const window = { innerWidth: 375, innerHeight: 700,
  addEventListener: (type, fn, opt) => listeners.push(["window", type, fn, opt]) };
const location = { hash: "" }, history = { state: null, replaceState() {} };
const setTimeout = (f, ms) => timers.push({ f, ms }), clearTimeout = (i) => { if (timers[i - 1]) timers[i - 1].f = null; };
const fetch = async (url) => ({ ok: true, status: 200, json: async () => url in replies ? replies[url] : url === "/api/jobs" ? [] : {} });
const ctx = vm.createContext({ console, document, window, location, history, setInterval() {}, setTimeout, clearTimeout,
  EventSource: class { addEventListener() {} close() {} }, fetch, matchMedia: () => ({ matches: false }), confirm: () => true,
  Elem, made, listeners, timers, byId });
Promise.resolve(vm.runInContext(fs.readFileSync(process.argv[2], "utf8") + "\n" + process.argv[3], ctx))
  .catch((e) => { console.error(e.stack); process.exitCode = 1; });
"""


def run_info_js(code, replies=None):
    """Load app.js in INFO_HARNESS, then run code in its scope; returns the JSON that code printed last."""
    if shutil.which("node") is None:
        pytest.skip("no node")
    p = subprocess.run(["node", "-e", INFO_HARNESS, "-", str(server.STATIC / "app.js"), textwrap.dedent(code),
                        json.dumps(replies or {})], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout.splitlines()[-1])


# Helpers for INFO_HARNESS: the rendered Results and Downloads tabs as [kind, label, the info wrapper, its parent] for
# every row (dt), viewer (figcaption), any hint among the viewers (p.hint: there should be none), the part masks' hint
# and every file of the Downloads tab (li.dlrow); the info wrapper's button and note; a row's dd; the key of a note's
# text in INFO; and whether a placed note (its style's left and top, its rect's size) meets a box.
INFO_ITEMS = """
const walk = (e, out = []) => { if (e && typeof e === "object") { out.push(e); e.kids.forEach((k) => walk(k, out)); } return out; };
const isInfo = (e) => e && typeof e === "object" && e.cls.has("info");
const items = () => [
  ...walk($("#facts2")).filter((e) => e.tag === "dt").map((e) => ["row", e.kids[0], e.kids.find(isInfo), e]),
  ...walk($("#viewers")).filter((e) => e.tag === "figcaption").map((e) => ["viewer", e.kids[0].textContent, e.kids.find(isInfo), e]),
  ...walk($("#viewers")).filter((e) => e.tag === "p" && e.cls.has("hint")).map((e) => ["placeholder", "Agent result", e.kids.find(isInfo), e]),
  ...walk($("#partchecks")).filter((e) => e.tag === "p" && e.cls.has("hint"))  // the item is the folded check images
    .map((e) => ["masks", "Part masks in the views", e.kids.find(isInfo), e]),
  ...walk($("#downloadsbody")).filter((e) => e.cls.has("dlrow")).map((e) => ["download", e.kids[0].textContent, e.kids.find(isInfo), e])];
const parts = (wrap) => ({ btn: wrap.kids[0], note: wrap.kids[1] });
const ddOf = (dt) => dt.parent.kids[dt.parent.kids.indexOf(dt) + 1];
const meets = (note, box) => {
  const n = { left: parseFloat(note.style.left), top: parseFloat(note.style.top), width: note.rect.width, height: note.rect.height };
  const b = box.rect;
  return n.left < b.left + b.width && b.left < n.left + n.width && n.top < b.top + b.height && b.top < n.top + n.height;
};
const keyOf = (text) => Object.keys(INFO).find((k) => INFO[k] === text) || null;
const shown = () => made.filter((e) => e.cls.has("infonote") && !e.hidden).map((e) => e.attrs.id);
const describe = (kind, label, wrap, owner) => {
  if (!wrap) return { kind, label, missing: true };
  const { btn, note } = parts(wrap);
  return { kind, label, last: owner.kids[owner.kids.length - 1] === wrap, wrap: wrap.tag, btn: btn.tag, type: btn.attrs.type,
    cls: btn.className, aria: btn.attrs["aria-label"], text: btn.textContent, describedby: btn.attrs["aria-describedby"],
    id: note.attrs.id, role: note.attrs.role, hidden: note.hidden, noteCls: note.className, key: keyOf(note.textContent),
    note: note.textContent };
};
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


def info_jobs():
    """(jobs, replies) for INFO_HARNESS: an articulated Opus job with every row, the same job without an agent (its
    parts welded, a link VoMP skipped), a rigid Astra job with a refined mesh, and a rigid job whose agent was skipped
    for its mask IoU."""
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
    jobs = {
        "a": {"id": "a", "articulated": True, "agent": "opus", "files": every, "stages": {"agent": {"status": "done"}},
              "agent_result": ART_RESULT},
        "w": {"id": "w", "articulated": True, "agent": "none", "agent_result": None,
              "files": [f for f in every if f not in ("refined.glb", "pool/lidar.json")],
              "stages": {"agent": {"status": "skipped", "note": "turned off"}}},
        "r": {"id": "r", "articulated": False, "agent": "astra", "files": ["run/posed.glb", "refined.glb", "run/placement.json",
                                                                             "physics/summary.json", "results.zip"],
              "stages": {"agent": {"status": "done"}},
              "agent_result": {"source": "checkpoint", "verdict": "needs_review", "reason": "seams on the lid", "limitations": [],
                               "agent": "codex", "model": "gpt-6-astra", "mode": "refinement"}},
        "s": {"id": "s", "articulated": False, "agent": "astra", "agent_result": None, "files": ["run/posed.glb", "run/placement.json"],
              "stages": {"agent": {"status": "skipped", "note": "Stage A mean mask IoU 0.41 is below 0.5; physics uses "
                                                               "run/posed.glb"}}}}
    replies = {}
    for i in jobs:
        replies.update({f"/api/jobs/{i}/file/run/placement.json": place, f"/api/jobs/{i}/file/pool/lidar.json": lidar,
                        f"/api/jobs/{i}/file/physics/summary.json": summary, f"/api/jobs/{i}/file/run/parts.json": parts,
                        f"/api/jobs/{i}/parts": {"checks": ["run/parts/sam3/check/0.jpg"]}})
    replies["/api/jobs/a/file/physics/mjcf/mjcf.json"] = ART_MJCF
    replies["/api/jobs/w/file/physics/mjcf/mjcf.json"] = {"name": "cabinet", "inertial": ART_MJCF["inertial"]}
    return jobs, replies


def test_app_js_results_have_an_info_button_per_item():
    """Every row of the Results tab's facts list, every viewer, the part masks' hint and every file of the
    Downloads tab end with an info button: a real button with the text
    "i", the label "About <item>" and aria-describedby on its note, a hidden tooltip with the item's text from INFO.
    Every note of INFO shows for some item, each id is unique, and the notes hold the numbers of the code they
    explain."""
    jobs, replies = info_jobs()
    out = run_info_js(INFO_ITEMS + """
        (async () => {
          const out = { info: INFO }, jobs = """ + json.dumps(jobs) + """;
          for (const [id, job] of Object.entries(jobs)) {
            V = { id };
            made.length = 0;
            await renderResults(job);
            renderDownloads(job);
            out[id] = items().map(([kind, label, wrap, owner]) => describe(kind, label, wrap, owner));
          }
          console.log(JSON.stringify(out));
        })()
        """, replies)
    notes = out.pop("info")
    assert all(isinstance(v, str) and v.strip() for v in notes.values())
    used, ids = set(), []
    for job_id, found in out.items():
        assert found, job_id
        for it in found:
            assert not it.get("missing"), (job_id, it)
            assert (it["wrap"], it["btn"], it["type"], it["cls"], it["text"]) == ("span", "button", "button", "infobtn", "i"), it
            assert it["aria"] == "About " + it["label"] and it["describedby"] == it["id"], it
            assert (it["role"], it["hidden"], it["noteCls"], it["last"]) == ("tooltip", True, "infonote", True), it
            assert it["key"] in notes and it["note"].strip(), it
            used.add(it["key"])
            ids.append(it["id"])
    assert used == set(notes), set(notes) - used
    assert len(ids) == len(set(ids))
    keys = lambda found, kind: [(it["label"], it["key"]) for it in found if it["kind"] == kind]  # noqa: E731
    assert keys(out["a"], "row") == [
        ("Metric scale", "metric_scale"), ("Stage A mean mask IoU", "mask_iou"), ("Parts not split off", "parts_lost"),
        ("Agent", "agent"), ("Agent verdict", "agent_verdict"), ("Agent reason", "agent_reason"),
        ("Limitation", "limitation"), ("Limitation", "limitation"), ("Joints", "joints"),
        ("Joint changes by the agent", "joint_changes"), ("Agent tokens", "agent_tokens"),
        ("Agent cost (estimate)", "agent_cost"), ("Links", "links"), ("Base", "base"), ("Joint of door", "joint"),
        ("Joint of drawer", "joint"), ("Joint of body", "joint"), ("Joint of knob", "joint_welded"), ("Mass", "mass"),
        ("Center of mass", "com"), ("Link masses", "link_masses"), ("Young's modulus (median)", "youngs"),
        ("Poisson ratio (median)", "poisson"), ("Density (median)", "density")]
    assert keys(out["a"], "viewer") == [("Stage A (MV-SAM3D, posed)", "view_posed"),
                                        ("Parts (split, before rigging)", "view_parts"),
                                        ("Parts with joints, by Claude Opus 5.5", "view_rigged")]
    assert keys(out["a"], "download") == [("results.zip", "dl_zip"), ("posed.glb", "dl_posed"), ("parts.glb", "dl_parts"),
                                          ("refined.glb", "dl_refined"), ("aligned.glb", "dl_aligned"),
                                          ("model.urdf", "dl_urdf"), ("placement.json", "dl_placement"),
                                          ("summary.json", "dl_summary"), ("lidar.json", "dl_lidar"),
                                          ("selection.json", "dl_selection"), ("parts.json", "dl_parts_json"),
                                          ("mjcf.json", "dl_mjcf")]
    assert keys(out["s"], "download") == [("posed.glb", "dl_posed"), ("placement.json", "dl_placement")]
    assert keys(out["a"], "masks") == [("Part masks in the views", "part_masks")]
    assert ("Links", "links_welded") in keys(out["w"], "row") and ("Mass note", "mass_note") in keys(out["w"], "row")
    assert ("Agent", "agent") in keys(out["w"], "row")  # the agent was off: its row says so, and no hint repeats it
    assert all(not keys(out[j], "placeholder") for j in ("a", "w", "r", "s"))
    assert keys(out["r"], "viewer")[1] == ("Refined by GPT-6-Astra", "view_refined")
    assert keys(out["s"], "row") == [("Metric scale", "metric_scale"), ("Stage A mean mask IoU", "mask_iou"),
                                     ("Agent", "agent")]
    # The notes give the rules and numbers of the code they explain (read from the code, so a changed constant fails
    # here), and what a missing value means.
    import inspect

    from gator.agent import controller, tools
    from gator.parts import geometry, vote
    from gator.stage_a import part_masks
    from gator.webapp import agent_step
    num = lambda x: f"{x:g}"  # noqa: E731
    seam_lines = inspect.signature(geometry.seams).parameters["max_lines"].default
    usd = lambda x: f"${x:.2f}".removesuffix(".00")  # noqa: E731  4.0 -> $4, 0.2 -> $0.20
    iou, small = num(steps.MIN_IOU), f"{vote.MIN_AREA:.1%}"
    score = num(inspect.signature(part_masks.run).parameters["threshold"].default)
    prices = [f"{usd(controller.PRICES[k])} {w}" for k, w in (("input_tokens", "input"), ("output_tokens", "output"),
                                                                ("cache_read_input_tokens", "cache read"),
                                                                ("cache_creation_input_tokens", "cache write"))]
    facts = {"mask_iou": [f"Below {iou}"],
             "agent": [f"{num(agent_step.BUDGET_S['refinement'])} s to edit",
                       f"{num(agent_step.BUDGET_S['articulation'])} s to set", f"up to {num(tools.MAX_EXCLUDED_S)} s",
                       f"{controller.CLAUDE_EFFORT} effort", f"below {iou}", "skipped"],
             "mass": ["64^3", "density", "16%", "collision hulls", "left out"],
             "metric_scale": ["median", "LiDAR", "before MV-SAM3D", "Without LiDAR"],
             "agent_cost": [*prices, "A ? means", "refusal fallback"], "joints": ["A ? means", "With 0"],
             "part_masks": [f"at least {score}", "outside the object", "union"],
             "parts_lost": ["not found", f"at least {score}", f"less than {small}", "still have a part"],
             "limitation": [f"at most {tools.MAX_LIMITATIONS}"], "link_masses": ["64^3", "median density"],
             "agent_verdict": [*tools.VERDICTS["refinement"], "fallback_original", "checkpoint", "export"],
             "agent_reason": ["API key"], "youngs": ["Pa"], "density": ["kg/m^3", "by its volume"],
             "poisson": ["by its volume"], "view_posed": ["view 0's camera frame", "aligned.glb", "larger than"],
             "view_rigged": ["without joints"], "dl_aligned": ["cameras' mean up"],
             "dl_urdf": ["fixed one to the link that physics welds it to"],
             "mass_note": ["MuJoCo, USD and URDF"], "com": ["bottom of the object"],
             "dl_zip": ["(package)", "API key", "the agent's result when the key was found in it"],
             "dl_refined": ["each joint that the agent set", "a part without one has none"],
             "dl_placement": ["view 0's camera frame", "mask IoU"],
             "dl_summary": ["5th to 95th percentile", "could not voxelize"],
             "dl_lidar": ["ok is false", "uploaded calibration", "Stray capture", "aligned with the color camera"],
             "dl_selection": ["view 0 first", "null without LiDAR", "ok is false", "excluded"],
             "dl_parts_json": [f"up to {seam_lines} fitted lines", "not split off"], "dl_mjcf": ["only in results.zip"]}
    for key, want in facts.items():
        assert all(f in notes[key] for f in want), (key, [f for f in want if f not in notes[key]])
    # A file's note does not repeat its group's sentence (zipText, SIM_TEXT): what results.zip holds, and that
    # model.urdf does not load alone.
    assert "MJCF" not in notes["dl_zip"] and "does not load" not in notes["dl_urdf"]
    # One to three short sentences each (a "?" that stands for a missing value ends none).
    import re
    sentences = {k: len(re.findall(r"[\w)%'\"][.!?](?:\s|$)", v)) for k, v in notes.items()}
    assert all(1 <= n <= 3 for n in sentences.values()), sentences
    assert max(len(v) for v in notes.values()) <= 450


def test_app_js_info_notes_open_and_close():
    """A note shows on hover (a mouse, with a short delay before it closes so the pointer can reach it; a touch is
    no hover) and on focus. A click, a tap, Enter or Space toggles it: a note that showed before the press hides, and
    one that did not shows and stays open (a tap that focuses the button first counts as not shown). Escape, a
    pointer down outside the button and its note, and focus that moves on close it. One note shows at a time. A
    viewer's or a download's note sits under its button, or above it when only that fits, inside the 375 px window; a
    facts row's note sits under the row's value or above its label, so it covers neither the value nor another row's
    info button. It follows a scroll, never goes above over the sticky header, and closes when its button goes under
    the header. A new rendering of the results, the jobs list and another tab close it."""
    jobs, replies = info_jobs()
    out = run_info_js(INFO_ITEMS + """
        (async () => {
          const out = {};
          V = { id: "s" };
          await renderResults(""" + json.dumps(jobs["s"]) + """);
          const all = items(), row = (x) => ({ wrap: x[2], ...parts(x[2]), dt: x[3], dd: ddOf(x[3]) });
          const [A, B] = all.filter((x) => x[0] === "row").map(row);
          const [W] = all.filter((x) => x[0] === "viewer").map((x) => ({ wrap: x[2], ...parts(x[2]) }));
          const at = (x, left, top, width, height) => { x.rect = { left, top, width, height }; };
          // A phone: one column, each value under its label.
          at(A.dt, 12, 100, 351, 28); at(A.btn, 110, 106, 18, 18); at(A.dd, 12, 128, 351, 28);
          at(B.dt, 12, 156, 351, 28); at(B.btn, 140, 162, 18, 18); at(B.dd, 12, 184, 351, 28);
          at(W.btn, 300, 100, 18, 18);
          for (const x of [A, B, W]) at(x.note, 0, 0, 320, 80);
          const on = (x) => !x.note.hidden && x.wrap.cls.has("on");
          const fire = (target, type, e) => listeners.filter((l) => l[0] === target && l[1] === type).forEach((l) => l[2](e));
          const flush = () => timers.splice(0).forEach((t) => t.f && t.f());
          const mouse = { pointerType: "mouse" }, touch = { pointerType: "touch" };
          const press = (x, e) => { fire("document", "pointerdown", { target: x.btn }); x.btn.onpointerdown(e); };
          const click = (x, detail) => x.btn.onclick({ detail });  // detail 0: Enter or Space
          const pos = (x) => [x.note.style.left, x.note.style.top];
          out.listeners = listeners.filter((l) => ["keydown", "pointerdown", "scroll"].includes(l[1]))
            .map((l) => [l[0], l[1], l[2].name, l[3] === true || !!(l[3] && l[3].capture)]);
          out.start = shown();
          // Hover: open at once, closed TIP_DELAY ms after the pointer leaves, kept when it comes back (to the note).
          A.wrap.onpointerenter(mouse);
          out.hover = [on(A), ...pos(A), shown().length];
          A.wrap.onpointerleave(mouse);
          out.leave = [on(A), timers.map((t) => t.ms)];
          flush();
          out.left = on(A);
          A.wrap.onpointerenter(mouse);
          A.wrap.onpointerleave(mouse);
          A.wrap.onpointerenter(mouse);
          flush();
          out.back = on(A);
          A.wrap.onpointerleave(mouse);
          flush();
          A.wrap.onpointerenter(touch);
          out.touch = on(A);
          // Focus: open; another button's focus opens its note alone; blur closes.
          A.btn.onfocus();
          out.focus = [on(A), on(B)];
          A.btn.onblur();
          B.btn.onfocus();
          out.other = [on(A), on(B), shown().length];
          B.btn.onblur();
          out.blur = on(B);
          // A tap that does not focus the button (Safari) opens the note and keeps it; a second tap closes it.
          press(A, touch);
          click(A, 1);
          A.wrap.onpointerleave(mouse);
          flush();
          out.tap = on(A);
          press(A, touch);
          click(A, 1);
          out.tap2 = on(A);
          // A tap that focuses the button first (Chrome): focus opens, the click keeps it; a second tap closes it.
          press(A, touch);
          A.btn.onfocus();
          click(A, 1);
          out.chromeTap = on(A);
          press(A, touch);
          click(A, 1);
          out.chromeTap2 = on(A);
          A.btn.onblur();
          // A mouse: a click on a hovered note closes it; a second click opens it and keeps it after the pointer leaves.
          A.wrap.onpointerenter(mouse);
          press(A, mouse);
          A.btn.onfocus();
          click(A, 1);
          out.mouse = on(A);
          press(A, mouse);
          click(A, 1);
          A.wrap.onpointerleave(mouse);
          flush();
          out.mouse2 = on(A);
          // The focus moves on from a clicked note: it closes.
          A.btn.onblur();
          out.moveOn = on(A);
          // The keyboard: Tab opens the note, Enter closes it, Enter again opens it.
          A.btn.onfocus();
          click(A, 0);
          out.key = on(A);
          click(A, 0);
          out.key2 = on(A);
          A.btn.onblur();
          // A press with no click (the pointer slid off) does not change what a later key does.
          press(A, mouse);
          A.btn.onfocus();
          click(A, 0);
          out.stale = on(A);
          A.btn.onblur();
          // Escape closes; another key does not.
          press(A, touch);
          click(A, 1);
          fire("document", "keydown", { key: "Enter" });
          out.enter = on(A);
          fire("document", "keydown", { key: "Escape" });
          out.escape = on(A);
          // A pointer down on the button or the note keeps it; anywhere else closes it.
          press(A, touch);
          click(A, 1);
          fire("document", "pointerdown", { target: A.note });
          fire("document", "pointerdown", { target: A.btn });
          out.inside = on(A);
          fire("document", "pointerdown", { target: B.btn });
          out.outside = on(A);
          // One at a time: a clicked note closes when another one opens on hover.
          press(A, touch);
          click(A, 1);
          B.wrap.onpointerenter(mouse);
          out.one = [on(A), on(B), shown().length];
          B.wrap.onpointerleave(mouse);
          flush();
          // A row on a phone: under its value (12 px in), and above its label near the bottom of the window.
          press(A, touch);
          click(A, 1);
          out.phone = [...pos(A), meets(A.note, A.dd), meets(A.note, A.btn), meets(A.note, A.dt)];
          at(A.dt, 12, 560, 351, 28); at(A.btn, 110, 566, 18, 18); at(A.dd, 12, 588, 351, 28);
          fire("window", "scroll", {});
          out.phoneAbove = [...pos(A), meets(A.note, A.dd), meets(A.note, A.btn), meets(A.note, A.dt)];
          fire("document", "keydown", { key: "Escape" });
          // A row at desktop width: from the value's left edge, under the row, clear of the column of info buttons, so
          // the pointer goes straight down from one button to the next.
          window.innerWidth = document.documentElement.clientWidth = 1280;
          window.innerHeight = 800;
          at(A.dt, 16, 100, 190, 34); at(A.btn, 182, 108, 18, 18); at(A.dd, 222, 100, 500, 34);
          at(B.dt, 16, 134, 190, 34); at(B.btn, 160, 142, 18, 18); at(B.dd, 222, 134, 500, 34);
          at(A.note, 0, 0, 320, 120);
          A.wrap.onpointerenter(mouse);
          out.desk = [...pos(A), meets(A.note, A.dd), meets(A.note, B.btn), meets(A.note, B.dt)];
          A.wrap.onpointerleave(mouse);
          B.wrap.onpointerenter(mouse);
          out.deskNext = [on(A), on(B)];
          B.wrap.onpointerleave(mouse);
          flush();
          window.innerWidth = document.documentElement.clientWidth = 375;
          window.innerHeight = 700;
          // A viewer's note sits under its button, or above it near the bottom of the window, 12 px in near the left edge.
          W.wrap.onpointerenter(mouse);
          out.button = pos(W);  // 375 - 12 - 320 = 43 px from the left, 100 + 18 + 6 = 124 px from the top
          W.wrap.onpointerleave(mouse);
          flush();
          at(W.btn, 4, 650, 18, 18);
          press(W, touch);
          click(W, 1);
          out.above = pos(W);
          at(W.btn, 300, 100, 18, 18);
          fire("window", "scroll", {});
          out.scrolled = pos(W);
          // A narrow note sits centred under its button.
          fire("document", "keydown", { key: "Escape" });
          at(W.note, 0, 0, 100, 40);
          at(W.btn, 150, 100, 18, 18);
          press(W, touch);
          click(W, 1);
          out.centred = pos(W);
          fire("document", "keydown", { key: "Escape" });
          at(W.note, 0, 0, 320, 80);
          // The sticky header (48 px): a note does not go above over it, and it closes when its button goes under it.
          at($("header"), 0, 0, 375, 48);
          window.innerHeight = 220;
          at(W.btn, 300, 120, 18, 18);
          press(W, touch);
          click(W, 1);
          out.underHeader = pos(W);
          window.innerHeight = 700;
          at(W.btn, 300, 60, 18, 18);
          fire("window", "scroll", {});
          out.belowHeader = on(W);
          at(W.btn, 300, 20, 18, 18);
          fire("window", "scroll", {});
          out.gone = [on(W), tip === null];
          at($("header"), 0, 0, 0, 0);
          await renderResults(""" + json.dumps(jobs["s"]) + """);
          out.redrawn = [on(A), tip === null, shown()];
          // The jobs list (show) and another tab close an open note too.
          const [C, D] = items().filter((x) => x[0] === "row").map((x) => ({ wrap: x[2], ...parts(x[2]) }));
          for (const x of [C, D]) { at(x.btn, 300, 100, 18, 18); at(x.note, 0, 0, 320, 80); }
          press(C, touch);
          click(C, 1);
          out.page = [on(C)];
          show("list");
          out.page.push(on(C), tip === null);
          V.tab = "results";
          press(D, touch);
          click(D, 1);
          out.tab = [on(D)];
          selectTab("log");
          out.tab.push(on(D), tip === null);
          console.log(JSON.stringify(out));
        })()
        """, replies)
    assert out["listeners"] == [["document", "keydown", "tipKey", False], ["document", "pointerdown", "tipOutside", True],
                                ["window", "scroll", "tipPlace", True]]
    assert out["start"] == []
    # A row's note: 12 px from the left (its value's edge), 6 px under its value (128 + 28 + 6).
    assert out["hover"] == [True, "12px", "162px", 1]
    assert out["leave"] == [True, [150]] and out["left"] is False and out["back"] is True and out["touch"] is False
    assert out["focus"] == [True, False] and out["other"] == [False, True, 1] and out["blur"] is False
    assert out["tap"] is True and out["tap2"] is False
    assert out["chromeTap"] is True and out["chromeTap2"] is False
    assert out["mouse"] is False and out["mouse2"] is True and out["moveOn"] is False
    assert out["key"] is False and out["key2"] is True and out["stale"] is False
    assert out["enter"] is True and out["escape"] is False
    assert out["inside"] is True and out["outside"] is False
    assert out["one"] == [False, True, 1]
    assert out["phone"] == ["12px", "162px", False, False, False]
    assert out["phoneAbove"] == ["12px", f"{560 - 6 - 80}px", False, False, False]  # above the whole label (dt)
    assert out["desk"] == ["222px", f"{134 + 6}px", False, False, False]
    assert out["deskNext"] == [False, True]
    assert out["button"] == ["43px", "124px"]
    assert out["above"] == ["12px", f"{650 - 6 - 80}px"] and out["scrolled"] == ["43px", "124px"]
    assert out["centred"] == [f"{150 + 9 - 50}px", "124px"]
    assert out["underHeader"] == ["43px", f"{120 + 18 + 6}px"]  # above would be 34 px, under the 48 px header
    assert out["belowHeader"] is True and out["gone"] == [False, True]
    assert out["redrawn"] == [False, True, []]
    assert out["page"] == [True, False, True] and out["tab"] == [True, False, True]


# ---- the Downloads tab (app.js renderDownloads) ----

DL_SIZES = {"results.zip": 12_345_678, "run/posed.glb": 1_234_567, "run/parts.glb": 987_654, "refined.glb": 2_500_000,
            "physics/aligned.glb": 1_300_000, "physics/urdf/model.urdf": 4_321, "run/placement.json": 5_678,
            "physics/summary.json": 3_210, "pool/lidar.json": 45_678, "selection.json": 23_456, "run/parts.json": 812,
            "physics/mjcf/mjcf.json": 6_789}


def test_app_js_downloads_tab():
    """The Downloads tab lists the job's files that exist, in four groups, each under an h3: Everything (results.zip,
    with what it holds, the URDF only when the job has one, and that the MJCF and USD models come only in it),
    Meshes, Simulation (model.urdf, which says that it does not load alone) and Records. Each row is the file name
    as a download link that saves as <job id>_<file>, its size from job.sizes (none when the server gave none) and
    its info button with its note from INFO. Groups without files are left out, a file that the tab does not know is
    ignored, and a running build lists none. Without results.zip, one sentence above the groups says why: the build
    has not made it yet, or it stopped first. An empty tab says one sentence. The tab offers exactly the server's
    outputs, all of which the file route serves, and the Results tab has no download link left."""
    jobs, replies = info_jobs()
    every = list(DL_SIZES)
    cases = {
        "a": {**jobs["a"], "status": "done", "files": every, "sizes": DL_SIZES},
        "r": {"id": "r", "status": "done", "agent": "none", "agent_result": None, "articulated": False,
              "files": [f for f in every if f not in ("run/parts.glb", "run/parts.json", "refined.glb", "pool/lidar.json")]
              + ["frames/preview.mp4"], "sizes": {k: v for k, v in DL_SIZES.items() if k != "physics/aligned.glb"}},
        "v": {"id": "v", "status": "review", "files": ["selection.json", "pool/lidar.json"], "sizes": DL_SIZES},
        "f": {"id": "f", "status": "failed", "files": ["selection.json", "run/posed.glb", "run/placement.json"]},
        "c": {"id": "c", "status": "cancelled", "files": ["selection.json"], "sizes": {}},
        # A job from before the URDF stage: results.zip has no urdf/ (jobs.package zips what exists).
        "u": {"id": "u", "status": "done", "agent": "none", "agent_result": None,
              "files": [f for f in every if f not in ("physics/urdf/model.urdf", "run/parts.glb", "run/parts.json")]},
        "y": {"id": "y", "status": "ready", "files": [], "sizes": {}},
        "z": {"id": "z", "status": "analyzing", "files": [], "sizes": {}},
        "q": {"id": "q", "status": "running", "files": ["selection.json", "run/posed.glb"], "sizes": DL_SIZES},
        # The agent's result held the API key, so jobs.package left it out of results.zip.
        "k": {"id": "k", "status": "done", "agent": "astra", "files": ["results.zip", "run/posed.glb"],
              "agent_result": {"source": "finish", "verdict": "complete", "limitations": [],
                               "reason": "left out: the agent's result held the API key"}}}
    out = run_info_js(INFO_ITEMS + """
        (async () => {
          const out = { paths: DOWNLOADS.flatMap(([, rows]) => rows.map(([p]) => p)), sim: SIM_TEXT, keyed: KEYED_RESULT,
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
                  const info = describe("download", a.textContent, li.kids.find(isInfo), li);
                  return { name: a.textContent, href: a.attrs.href, download: a.attrs.download, cls: a.className,
                    size: size ? size.textContent : null, key: info.key, aria: info.aria, last: info.last, li: li.className };
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
    assert [g["title"] for g in a["groups"]] == ["Everything", "Meshes", "Simulation", "Records"]
    assert all(g["labelled"] and g["tag"] == "h3" and g["cls"] == ["dlgroup", "ul", "dlist"] for g in a["groups"])
    rows = [r for g in a["groups"] for r in g["rows"]]
    assert [(r["name"], r["size"], r["key"]) for r in rows] == [
        ("results.zip", "12.3 MB", "dl_zip"), ("posed.glb", "1.2 MB", "dl_posed"), ("parts.glb", "988 KB", "dl_parts"),
        ("refined.glb", "2.5 MB", "dl_refined"), ("aligned.glb", "1.3 MB", "dl_aligned"), ("model.urdf", "4.3 KB", "dl_urdf"),
        ("placement.json", "5.7 KB", "dl_placement"), ("summary.json", "3.2 KB", "dl_summary"), ("lidar.json", "46 KB", "dl_lidar"),
        ("selection.json", "23 KB", "dl_selection"), ("parts.json", "812 B", "dl_parts_json"), ("mjcf.json", "6.8 KB", "dl_mjcf")]
    by_name = {r["name"]: r for r in rows}
    assert sorted(r["href"] for r in rows) == sorted(f"/api/jobs/a/file/{p}" for p in every)
    assert all(r["href"].endswith("/" + r["name"]) for r in rows)  # each file downloads as its own name
    # Every file saves under the job id, so files of two jobs do not clash; the label stays the short name.
    assert all(r["download"] == "a_" + r["name"] for r in rows)
    assert by_name["results.zip"]["cls"] == "dl primary" and all(r["cls"] == "dl" for r in rows if r["name"] != "results.zip")
    assert all(r["aria"] == "About " + r["name"] and r["last"] and r["li"] == "dlrow" for r in rows)
    texts = {g["title"]: g["text"] for g in a["groups"]}
    assert texts["Everything"] == ("results.zip holds the MuJoCo (MJCF), USD and URDF models with their meshes, the GLB "
                                   "meshes, the physics summary, the placement, the view selection, the LiDAR scale "
                                   "(lidar.json), the agent's result and the split parts (parts.glb, parts.json). "
                                   "The MJCF and USD models come only in it.")
    assert texts["Simulation"] == out["sim"] and "does not load alone" in out["sim"] and "MJCF" not in out["sim"]
    assert texts["Meshes"] is None and texts["Records"] is None
    # A rigid job without the agent or LiDAR: only its files, a file with no size shows none, an unknown file is left out.
    r = out["r"]
    assert {g["title"]: [x["name"] for x in g["rows"]] for g in r["groups"]} == {
        "Everything": ["results.zip"], "Meshes": ["posed.glb", "aligned.glb"], "Simulation": ["model.urdf"],
        "Records": ["placement.json", "summary.json", "selection.json", "mjcf.json"]}
    assert [x["size"] for x in r["groups"][1]["rows"]] == ["1.2 MB", None]
    assert r["groups"][0]["text"].endswith("the placement and the view selection. The MJCF and USD models come only in it.")
    # A job without a URDF (from before the URDF stage): results.zip does not claim one, no Simulation group, and the
    # Everything sentence still says that the MJCF and USD models come only in results.zip.
    u = out["u"]
    assert [g["title"] for g in u["groups"]] == ["Everything", "Meshes", "Records"] and u["note"] is None
    assert u["groups"][0]["text"].startswith("results.zip holds the MuJoCo (MJCF) and USD models with their meshes, ")
    assert "URDF" not in u["groups"][0]["text"] and u["groups"][0]["text"].endswith("The MJCF and USD models come only in it.")
    # A result that held the API key is not in results.zip (jobs.package), and the server says so in its reason.
    import inspect
    assert out["keyed"] in inspect.getsource(server.make_app)
    assert out["k"]["groups"][0]["text"].endswith("the placement and the view selection. The MJCF and USD models come only in it.")
    # In review: the records alone; after a failure: what exists, without sizes when the server gave none. Without
    # results.zip, the first line says why the tab holds only these.
    assert [(g["title"], [x["name"] for x in g["rows"]]) for g in out["v"]["groups"]] == [("Records", ["lidar.json", "selection.json"])]
    assert out["v"]["note"] == [0, "p", out["partial"]] and out["partial"].startswith("results.zip and the other files")
    assert [(g["title"], [(x["name"], x["size"]) for x in g["rows"]]) for g in out["f"]["groups"]] == [
        ("Meshes", [("posed.glb", None)]), ("Records", [("placement.json", None), ("selection.json", None)])]
    assert out["f"]["note"] == [0, "p", out["stopped"]] and out["c"]["note"] == [0, "p", out["stopped"]]
    assert "stopped before it made results.zip" in out["stopped"]
    assert all(out[j]["note"] is None for j in ("r", "k", "y", "z", "q"))  # results.zip is there, or no file is
    # Nothing to list: one sentence.
    later, busy = "The files appear here when the build has made them.", "The files appear here when the build stops."
    for job_id, text in (("y", later), ("z", later), ("q", busy)):
        assert out[job_id] == {"groups": [], "none": [False, text], "body": True, "note": None}, job_id
    assert out["resultLinks"] == 0


def test_app_js_downloads_tab_opens_from_results():
    """The Results tab's line "Download the files from the Downloads tab." opens the Downloads tab as a pick (in
    the hash), and the tab takes the focus from the button, which hides with its panel, and scrolls into view. A
    link with the tab opens it, and the Downloads tab sits second in the keyboard order."""
    out = run_form_js(TAB_SETUP + """
        (() => {
          V = { id: "j" };
          location._hash = "#/job/j";
          const all = ["topics", "frames", ...BUILD];
          renderState(st("done", all));
          const out = { first: tabs() };
          document.activeElement = $("#todlbtn");
          $("#results").contains = (x) => x === $("#todlbtn");
          const scrolls = [];
          $("#tab-downloads").scrollIntoView = (opt) => scrolls.push([V.tab, opt]);
          $("#todlbtn").onclick();
          out.opened = [tabs(), focusedTabs.splice(0), history.state];
          out.scrolls = scrolls;
          V = { id: "d", hashTab: "downloads" };  // a reload of #/job/d/downloads
          location._hash = "#/job/d/downloads";
          history.state = null;
          renderState(st("done", all));
          out.link = tabs();
          console.log(JSON.stringify(out));
        })()
        """)
    assert out["first"]["tab"] == "results"  # a done job opens on Results
    opened, focused, state = out["opened"]
    assert opened == {"tab": "downloads", "pick": "downloads", "shown": ["downloads"], "selected": ["downloads"],
                      "roving": [-1, 0, -1, -1, -1], "hash": "#/job/j/downloads"}
    assert focused == ["downloads"] and state == {"tab": "downloads", "status": "done"}
    # selectTab focuses without scrolling, so the link brings the tab into view itself (below the sticky header, by
    # scroll-padding-top), once the tab shows.
    assert out["scrolls"] == [["downloads", {"block": "nearest"}]]
    assert (out["link"]["tab"], out["link"]["pick"], out["link"]["hash"]) == ("downloads", "downloads", "#/job/d/downloads")


def test_app_js_downloads_survive_a_results_file_that_does_not_parse():
    """A result file that does not parse (a cut write after a kill or a full disk) does not stop drawJob: the
    Downloads tab still lists the files, with the sentence that the build stopped first, the facts say what could
    not be read, the check images of an earlier draw go, and the Inputs tab still draws."""
    job = {"id": "j", "status": "failed", "video": {"name": "x.mp4"}, "prompt": "box", "source": {"kind": "video"},
           "stages": {"frames": {"status": "pending"}, "agent": {"status": "pending"}, "select": {"status": "done"}},
           "progress": [], "files": ["run/posed.glb", "run/placement.json", "selection.json"],
           "sizes": {"run/posed.glb": 1_200_000, "run/placement.json": 5_700, "selection.json": 23_000},
           "agent": "none", "agent_result": None, "articulated": False, "parts_review": False}
    out = run_info_js(INFO_ITEMS + """
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


def test_style_css_downloads():
    """A Downloads row wraps on a narrow phone, its link keeps a 36 px height (44 px on a touch screen), and the
    rows use the color tokens. The link's arrow has empty alternative text, the group headings are h3, and one
    Results viewer has the facts beside it on a wide screen."""
    import re
    css = (server.STATIC / "style.css").read_text()

    def rule(selector, text=css):
        m = re.search(r"(?:^|\n|\}\s*)" + re.escape(selector) + r"\s*\{([^}]*)\}", text)
        assert m, selector
        return m.group(1)
    assert "flex-wrap: wrap" in rule(".dlrow")
    link = rule("a.dl")
    assert all(x in link for x in ("min-height: 36px", "var(--line-strong)", "var(--surface)", "var(--fg)"))
    assert all(x in rule("a.dl.primary") for x in ("var(--accent)", "var(--accent-fg)"))
    assert "var(--muted)" in rule(".dlsize") and "list-style: none" in rule(".dlist")
    coarse = css[css.index("@media (pointer: coarse)"):]
    assert "min-height: 44px" in rule("a.dl", coarse)
    assert ".downloads" not in css and ".dlitem" not in css  # the old row of downloads in Results
    # The arrow before each link has empty alternative text, so a screen reader reads only the file name; the plain
    # content line before it is the fallback for a browser without that syntax.
    arrow = rule("a.dl::before")
    assert 'content: "\\2193" / "";' in arrow and arrow.index('content: "\\2193";') < arrow.index('content: "\\2193" / "";')
    # The group headings are h3 in the text color (not the muted h4), and the note above them is muted.
    assert "font-size: 16px" in rule(".dlgroup h3") and ".dlgroup h4" not in css
    assert "var(--muted)" in rule(".dlnote")
    # One viewer: on a wide screen the facts sit beside it.
    wide = css[css.index("@media (min-width: 1100px)"):]
    wide = wide[:wide.index("\n}\n")]
    assert re.search(r"#resultsbody\.onefig \{[^}]*display: grid", wide)
    assert re.search(r"#resultsbody\.onefig > #partchecks \{[^}]*grid-column: 1 / -1", wide)


def test_style_css_info_buttons():
    """The info button is a small circle whose larger target takes no room; on a touch screen it stays small (no
    44 px min-height). Its note is fixed to the window above the header, at most the window less 24 px wide, and
    uses the color tokens, with a shadow token for each scheme."""
    import re
    css = (server.STATIC / "style.css").read_text()

    def rule(selector, text=css):
        m = re.search(re.escape(selector) + r"\s*\{([^}]*)\}", text)
        assert m, selector
        return m.group(1)
    btn = rule("button.infobtn")
    assert all(x in btn for x in ("width: 18px", "height: 18px", "min-height: 0", "border-radius: 50%", "position: relative"))
    assert "position: absolute" in rule("button.infobtn::before")
    note = rule(".infonote")
    assert all(x in note for x in ("position: fixed", "max-width: min(320px, calc(100vw - 24px))", "z-index: 30",
                                   "var(--surface)", "var(--fg)", "var(--line-strong)", "var(--shadow)"))
    coarse = css[css.index("@media (pointer: coarse)"):]
    assert "min-height: 0" in rule("button.infobtn", coarse)
    assert "bottom: -6px" in rule("dt button.infobtn::before", coarse)  # a row's target stops above its value
    assert "display: flex" in rule(".viewers figcaption")  # a long viewer name wraps beside its button
    dark = css[css.index("@media (prefers-color-scheme: dark)"):css.index("* { box-sizing")]
    assert "--shadow:" in css[:css.index("@media (prefers-color-scheme: dark)")] and "--shadow:" in dark
