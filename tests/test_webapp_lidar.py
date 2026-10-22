"""The lidar stage (gator/webapp/lidar.py) on synthetic bags of fake_steps' scene: a box filmed by a
camera that circles it, with a LiDAR fixed to the camera. All data is made here."""

import contextlib
import hashlib
import json
import os
import re

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("rosbags")
from test_webapp_rosbag import CI, IMG, PC2, TFM, matrix, write_bag  # noqa: E402

from gator.stage_a import lidar_scale, mvsam3d  # noqa: E402
from gator.webapp import fake_steps as F  # noqa: E402
from gator.webapp import lidar as L  # noqa: E402
from gator.webapp import rosbag, steps  # noqa: E402

S0 = 1_749_907_120 * 10**9
LAG = 50_000_000  # every message is logged 50 ms after its stamp
CAM, LIDAR = "camera_optical_frame", "lidar"
CAMERA, POINTS, LIVOX = "/cam/image_rect", "/lidar/points", "/livox/lidar"
T_BASE_CAM = ((0.05, 0.0, 0.1), (-0.5, 0.5, -0.5, 0.5))  # optical axes from base_link's body axes
T_BASE_LIDAR = ((0.0, 0.02, 0.15), (0.0, 0.0, np.sin(np.pi / 12), np.cos(np.pi / 12)))  # turned 30 deg left
T_CAM_LIDAR = np.linalg.inv(matrix(*T_BASE_CAM)) @ matrix(*T_BASE_LIDAR)
SIZE = (96, 64)
TURN_DEG = 5.0  # a dynamic link turns the LiDAR this much further left at every frame


def lidar_link(i=0, turn=0.0):
    """base_link to the LiDAR at frame i: T_BASE_LIDAR turned i * turn degrees further left."""
    a = np.radians(30.0 + i * turn) / 2
    return T_BASE_LIDAR[0], (0.0, 0.0, np.sin(a), np.cos(a))


def cam_lidar(i=0, turn=0.0):
    """T_cam_lidar at frame i (lidar_link)."""
    return np.linalg.inv(matrix(*T_BASE_CAM)) @ matrix(*lidar_link(i, turn))


def picture(i, size):
    """A texture of 4x4 cells with a bright bar that moves right, so every frame is sharp."""
    w, h = size
    rng = np.random.default_rng(0)
    a = (rng.random((h // 4, w // 4, 3)) * 180 + 30).astype(np.uint8).repeat(4, 0).repeat(4, 1)
    a[h * 5 // 16:h * 10 // 16, (10 + 2 * i) % (w // 2):(10 + 2 * i) % (w // 2) + w * 5 // 24] = 240
    return a


def scene_points(i, n, size, step=1, k=1.0, T=T_CAM_LIDAR):
    """The scene's points that the camera of frame i (of n) sees, at every step-th pixel, in the LiDAR's
    frame (T: T_cam_lidar). k scales them in the camera's frame (k < 1: wrong points nearer than the
    scene)."""
    w, h = size
    depth, _, K, _ = F.render(w, h, F.SWEEP * i / max(n - 1, 1))
    v, u = np.mgrid[0:h:step, 0:w:step]
    d = depth[v, u]
    keep = d < F.FAR - 1e-6
    u, v, d = u[keep], v[keep], d[keep]
    X = k * np.stack([(u - K[0, 2]) / K[0, 0] * d, (v - K[1, 2]) / K[1, 1] * d, d, np.ones_like(d) / k], 1)
    return (X @ np.linalg.inv(T).T)[:, :3]


def write_scene_bag(path, n=10, size=SIZE, dt=0.5, kind="points", tf="static", info=True, topic=CAMERA,
                    model="plumb_bob", lidar_clock_s=0.0, lidar_late_s=0.0, between=False, step=1, info_r=None):
    """A ROS1 bag of fake_steps' scene: n rgb8 frames of topic at 1/dt Hz (frame i at the angle
    fake_steps.frame_angle gives it), the camera_info of the topic (K and P of fake_steps.intrinsics;
    D all zero for plumb_bob, else one coefficient of model; R info_r, else the identity) unless info
    is False, TF from base_link to the camera's and the LiDAR's frames (tf "static": /tf_static;
    "dynamic": the LiDAR's link on /tf at every frame, turned TURN_DEG further at each; None: none),
    and one LiDAR scan per frame, stamped as the frame (plus lidar_clock_s on the LiDAR's clock) and
    logged as it (plus lidar_late_s): the points the camera sees, in the LiDAR's frame. kind "points":
    a PointCloud2 without per-point times (between: another scan halfway to the next frame, of points
    at half the distance); "livox": a Livox CustomMsg whose timebase is 50 ms before the stamp and
    whose points' times spread over 100 ms, plus points at half the distance 150 to 190 ms after the
    timebase, which the +-50 ms window must drop."""
    w, h = size
    K = F.intrinsics(w, h)
    turn = TURN_DEG if tf == "dynamic" else 0.0

    def cloud(m, ns, X):
        cols = {"x": X[:, 0], "y": X[:, 1], "z": X[:, 2]}
        return m.cloud(ns, cols, [("x", "f4", 0), ("y", "f4", 4), ("z", "f4", 8)], 12, frame=LIDAR)

    def make(m):
        out = []
        if info:
            d = [0.0] * 5 if model == "plumb_bob" else [0.1, 0.0, 0.0, 0.0, 0.0]
            P = np.hstack([K, np.zeros((3, 1))])
            r = np.eye(3) if info_r is None else info_r
            out.append(("/cam/camera_info", CI, S0, m.info(S0, d, K, P, model=model, r=r, size=size)))
        links = [("base_link", CAM, *T_BASE_CAM), ("base_link", LIDAR, *T_BASE_LIDAR)]
        if tf == "static":
            out.append(("/tf_static", TFM, S0, m.tf(S0, links)))
        elif tf == "dynamic":
            out.append(("/tf_static", TFM, S0, m.tf(S0, links[:1])))
        for i in range(n):
            ns = S0 + round(i * dt * 1e9)
            out.append((topic, IMG, ns + LAG, m.image(ns, picture(i, size), "rgb8", frame=CAM)))
            if tf == "dynamic":
                out.append(("/tf", TFM, ns + LAG, m.tf(ns, [("base_link", LIDAR, *lidar_link(i, turn))])))
            head, log = ns + round(lidar_clock_s * 1e9), ns + LAG + round(lidar_late_s * 1e9)
            T = cam_lidar(i, turn)
            X = scene_points(i, n, size, step, T=T)
            if kind == "points":
                out.append((POINTS, PC2, log, cloud(m, head, X)))
                if between and i < n - 1:
                    half = round(dt * 5e8)
                    out.append((POINTS, PC2, log + half, cloud(m, head + half, scene_points(i, n, size, step, 0.5, T))))
            else:
                bad = scene_points(i, n, size, 2 * step, 0.5, T)
                pts = [(int(o), *x, 0x10) for o, x in zip(np.linspace(0, 99_000_000, len(X)), X)]
                pts += [(int(o), *x, 0x10) for o, x in zip(np.linspace(150_000_000, 190_000_000, len(bad)), bad)]
                out.append((LIVOX, m.livox, log, m.livox_msg(head, head - 50_000_000, pts, frame=LIDAR)))
        return sorted(out, key=lambda x: x[2])

    return write_bag(path, make=make)


def set_job(J, **fields):
    job = json.loads((J / "job.json").read_text())
    job.update(fields)
    (J / "job.json").write_text(json.dumps(job))
    return job


def make_job(root, bag, lidar_topic=POINTS, camera=CAMERA, frames=None):
    """A rosbag job after da3, as the stages leave it: job.json, frames/topics.json
    (rosbag.image_topics), meta.json, stamps.json and pool.json, the frames in scene/images, fake_steps'
    masks, prep (mvsam3d.prep) and the fake DA3 (FAKE_DA3_SCALE, since the job has LiDAR)."""
    J = root / "job"
    (J / "frames").mkdir(parents=True)
    (J / "frames" / "topics.json").write_text(json.dumps(rosbag.image_topics(bag)))
    st = {}
    info = rosbag.probe(bag, camera, stamps_out=st)
    (J / "frames" / "stamps.json").write_text(json.dumps({"topic": camera, "clock": info["clock"], **st}))
    n = info["n_frames"]
    (J / "frames" / "meta.json").write_text(json.dumps({"times": [round(0.5 * i, 3) for i in range(n)]}))
    frames = list(range(n)) if frames is None else frames
    names = {str(i): f"f{i:05d}" for i in frames}
    (J / "frames" / "pool.json").write_text(json.dumps({"pool": frames, "names": names, "picks": {}}))
    (J / "scene" / "images").mkdir(parents=True)
    (J / "scene" / "sam3_box").mkdir()
    for i, _, rgb in rosbag.frames(bag, camera, keep=set(frames)):
        Image.fromarray(rgb).save(J / "scene" / "images" / f"{names[str(i)]}.png")
    (J / "job.json").write_text(json.dumps({
        "id": "20261021-120000-abcdef", "video": {"name": bag.name, "path": str(bag)},
        "source": {"kind": "rosbag", "topic": camera}, "lidar": {"kind": "rosbag", "topic": lidar_topic},
        "calibration": None, "prompt": "box", "slug": "box"}))
    for name in names.values():
        mask = F.render(*info_size(info), F.frame_angle(J, name))[1]
        Image.fromarray(mask.astype(np.uint8) * 255).save(J / "scene" / "sam3_box" / f"{name}.png")
    mvsam3d.prep(J / "scene", "sam3_box", J / "pool", 1024)
    F.da3(J)
    return J


def info_size(info):
    return info["width"], info["height"]


def upload(J, data, name="calib.json"):
    """The calibration as the server's PUT stores it: lidar/calibration/<name> and job.json "calibration"."""
    data = data if isinstance(data, str) else json.dumps(data)
    d = J / "lidar" / "calibration"
    d.mkdir(parents=True, exist_ok=True)
    for old in d.iterdir():
        old.unlink()
    (d / name).write_text(data)
    cal = {"name": name, "bytes": len(data), "sha256": hashlib.sha256(data.encode()).hexdigest(),
           **lidar_scale.read_calibration(data, name)}
    set_job(J, calibration=cal)


def scaled(J):
    rec = L.scale(J)
    assert rec == json.loads((J / "pool" / "lidar.json").read_text())
    return rec


def failed(J):
    """The reason the stage gives; it also writes lidar.json with ok false and that reason."""
    with pytest.raises(SystemExit) as e:
        L.scale(J)
    rec = json.loads((J / "pool" / "lidar.json").read_text())
    assert rec["ok"] is False and str(e.value) == f"no metric scale: {rec['reason']}"
    assert steps.guard("lidar", J) == f"LiDAR scale: {rec['reason']}" and not steps.done("lidar", J)
    return rec["reason"]


@pytest.fixture(scope="module")
def points_bag(tmp_path_factory):
    return write_scene_bag(tmp_path_factory.mktemp("bag") / "points.bag", between=True)


def test_tf_gives_the_scale_and_the_record(points_bag, tmp_path, capsys):
    """The bag's /tf_static links the camera to the LiDAR; the stage finds the scale that the fake DA3
    lacks (1 / 0.8) and records how it got there."""
    J = make_job(tmp_path, points_bag)
    assert F.run_scale(J / "pool", "f00000") == F.FAKE_DA3_SCALE  # the fake DA3's units per meter
    rec = scaled(J)
    assert rec["ok"] and rec["s"] == pytest.approx(1 / F.FAKE_DA3_SCALE, rel=0.01), rec["reason"]
    assert steps.done("lidar", J) and steps.guard("lidar", J) is None
    names = mvsam3d.view_names(J / "pool")
    assert [v["name"] for v in rec["views"]] == names and all(v["status"] == "used" for v in rec["views"])
    assert rec["combine"]["views_used"] == len(names) and rec["reason"] is None
    assert sorted(p.stem for p in (J / "lidar" / "check").glob("*.jpg")) == names
    assert rec["source"] == {"kind": "rosbag", "file": "points.bag", "camera_topic": CAMERA, "topic": POINTS,
                             "type": "PointCloud2", "frame_id": LIDAR, "scans": 19, "cache": "miss"}
    ext = rec["extrinsic"]
    assert (ext["source"], ext["file"], ext["link"], ext["camera_axes"], ext["calibration_sha256"]) == \
        ("tf", None, "static", "optical", None)
    assert ext["chain"] == [CAM, "base_link", LIDAR] and np.allclose(ext["T_cam_lidar"], T_CAM_LIDAR, atol=1e-6)
    intr = rec["intrinsics"]
    assert intr["source"] == "/cam/camera_info: P (rectified)" and np.allclose(intr["K"], F.intrinsics(*SIZE))
    assert (intr["D"], intr["model"], intr["size"], intr["frame_size"]) == ([], None, list(SIZE), list(SIZE))
    assert np.allclose(intr["R"], np.eye(3))
    # Both clocks log 50 ms after the stamp: the header base. The scans have no per-point times, so each
    # frame takes the nearest scan (the one with its stamp, not the wrong one 0.25 s later); the period
    # is the step between the scans read.
    assert rec["time"] == {"base": "header", "window_ms": 50.0, "offset_ms": 0.0,
                           "log_minus_header_ms": {"camera": 50.0, "lidar": 50.0}, "scan_period_ms": 250.0,
                           "scan_period_from": "scans", "point_times": False}
    assert rec["warnings"] == [] and rec["version"] == 1 and rec["seconds"] > 0
    out = capsys.readouterr().out
    assert "PROGRESS 0.000" in out and "PROGRESS 1.000" in out and f"metric scale {rec['s']:.4f}" in out
    assert f"{names[0]}: used, " in out and "extrinsic from tf (camera_optical_frame <- base_link <- lidar, static)" in out

    # The samples are the frames' pixels: every true point lands where the camera sees it.
    with np.load(J / "lidar" / "samples.npz") as f:
        uvz = f[names[3]]
    depth = F.render(*SIZE, F.frame_angle(J, names[3]))[0]
    assert len(uvz) > 1000 and np.allclose(uvz[:, 2], depth[lidar_scale.pixel(uvz[:, 1]), lidar_scale.pixel(uvz[:, 0])],
                                           rtol=1e-4)

    # The same transform from an uploaded calibration (as T_lidar_cam): the cache key changes.
    upload(J, {"T_lidar_cam": np.linalg.inv(T_CAM_LIDAR).tolist()})
    cal = scaled(J)
    assert cal["s"] == pytest.approx(rec["s"], rel=1e-6) and cal["source"]["cache"] == "miss"
    ext = cal["extrinsic"]
    sha = json.loads((J / "job.json").read_text())["calibration"]["sha256"]
    assert (ext["source"], ext["file"], ext["chain"], ext["link"], ext["camera_axes"], ext["calibration_sha256"]) == \
        ("calibration", "calib.json", None, None, "optical", sha)
    assert np.allclose(ext["T_cam_lidar"], T_CAM_LIDAR, atol=1e-6) and steps.done("lidar", J)
    # Another calibration (or none) makes the record stale; the stage then makes a current one.
    upload(J, {"T_cam_lidar": T_CAM_LIDAR.tolist(), "time_offset_s": 0.0})
    assert not steps.done("lidar", J) and steps.guard("lidar", J) is None
    set_job(J, calibration=None)
    assert not steps.done("lidar", J)
    assert scaled(J)["extrinsic"]["source"] == "tf" and steps.done("lidar", J)


def test_a_dynamic_tf_link_and_a_livox_window(tmp_path):
    """The LiDAR's link only on /tf, turned TURN_DEG further at every frame: it is taken at each frame's
    time, so every frame's samples land on the scene. A Livox scan holds points at half the distance
    100 to 140 ms after the stamp; the +-50 ms window leaves them out."""
    bag = write_scene_bag(tmp_path / "livox.bag", n=8, kind="livox", tf="dynamic", step=2)
    J = make_job(tmp_path, bag, lidar_topic=LIVOX)
    rec = scaled(J)
    assert rec["ok"] and rec["s"] == pytest.approx(1.25, rel=0.01), rec["reason"]
    assert (rec["extrinsic"]["link"], rec["extrinsic"]["chain"]) == ("dynamic", [CAM, "base_link", LIDAR])
    assert np.allclose(rec["extrinsic"]["T_cam_lidar"], cam_lidar(0), atol=1e-6)  # the first frame's
    assert rec["time"]["point_times"] is True and rec["time"]["base"] == "header" and rec["warnings"] == []
    assert rec["source"]["type"] == "CustomMsg"
    with np.load(J / "lidar" / "samples.npz") as f:  # only the true points: as many as the camera sees
        assert sorted(f.files) == [f"f{i:05d}" for i in range(8)]
        for n in f.files:
            i, uvz = int(n[1:]), f[n]
            assert len(uvz) == len(scene_points(i, 8, SIZE, 2, T=cam_lidar(i, TURN_DEG))), n
            depth = F.render(*SIZE, F.frame_angle(J, n))[0]  # each frame's own T puts them on the scene
            seen = depth[lidar_scale.pixel(uvz[:, 1]), lidar_scale.pixel(uvz[:, 0])]
            assert np.allclose(uvz[:, 2], seen, rtol=1e-4), n


def test_a_dynamic_link_far_from_a_frame(monkeypatch):
    """A dynamic link is taken from the /tf values logged within each frame's read windows, nearest to its
    stamp; a frame whose windows hold none takes the nearest of all. A frame whose value is more than
    READ_S from its stamp gets a warning."""
    r = int(L.READ_S * 1e9)
    static = [("base_link", CAM, matrix(*T_BASE_CAM), True, 0)]

    def dynamic(k, stamp):
        return "base_link", LIDAR, matrix(*lidar_link(k, TURN_DEG)), False, stamp

    def run(logged, times, monkeypatch):
        """extrinsic over /tf values [(log_ns, edge)], with each frame's window around its time."""
        def transforms(reader, windows):  # rosbag.transforms: the static edges and /tf within the windows
            return static + [e for log, e in logged if any(a <= log < b for a, b in windows)]

        monkeypatch.setattr(rosbag, "open_bag", lambda path: contextlib.nullcontext())
        monkeypatch.setattr(rosbag, "transforms", transforms)
        warnings = []
        wins = {n: [(t - r, t + r)] for n, t in times.items()}
        ext, T = L.extrinsic("x.bag", CAM, LIDAR, POINTS, None, None, None, wins, times, warnings)
        assert ext["link"] == "dynamic" and ext["chain"] == [CAM, "base_link", LIDAR]
        return T, warnings

    # Values at S0 and S0 + 2 s: f00001's windows hold none, so it takes the nearest of all, 1 s away.
    times = {"f00000": S0 + 10**8, "f00001": S0 + 10**9, "f00002": S0 + 2 * 10**9}
    logged = [(S0 + k * 2 * 10**9, dynamic(k, S0 + k * 2 * 10**9)) for k in (0, 1)]
    T, warnings = run(logged, times, monkeypatch)
    assert np.allclose(T["f00000"], cam_lidar(0)) and np.allclose(T["f00001"], cam_lidar(0))
    assert np.allclose(T["f00002"], cam_lidar(1, TURN_DEG))
    assert warnings == ["TF links camera_optical_frame to lidar on /tf, and 1 of the 3 frames have no /tf value of "
                        "that link within 0.3 s of their stamp (up to 1.00 s away), so their T_cam_lidar may be "
                        "wrong; check their check images, or upload the calibration"]
    edges = static + [e for _, e in logged]
    assert L.link_gaps(edges, [CAM, "base_link", LIDAR], times) == {"f00000": 10**8, "f00001": 10**9, "f00002": 0}
    assert L.link_gaps(edges, [CAM, "base_link"], times) == dict.fromkeys(times, 0)  # a static link only

    # A TF publisher whose clock is 5 s ahead of the camera's: each frame still takes the value logged with
    # it (the nearest of all by stamp would be another frame's), and the warning shows the clocks disagree.
    times = {f"f{k:05d}": S0 + k * 10**9 for k in range(3)}
    logged = [(S0 + k * 10**9, dynamic(k, S0 + k * 10**9 + 5 * 10**9)) for k in range(3)]
    T, warnings = run(logged, times, monkeypatch)
    assert all(np.allclose(T[f"f{k:05d}"], cam_lidar(k, TURN_DEG)) for k in range(3))
    assert len(warnings) == 1 and "3 of the 3 frames" in warnings[0] and "(up to 5.00 s away)" in warnings[0]


def test_the_time_base(tmp_path):
    """A LiDAR on its own clock (5 s ahead): frames and scans are matched by log time, with a warning;
    time_offset_s in the calibration brings the header stamps back. The offset relates the header
    clocks, so it is not added to log times; a bag whose log times copy the header stamps needs it."""
    bag = write_scene_bag(tmp_path / "clock.bag", n=8, kind="livox", lidar_clock_s=5.0, step=2)
    J = make_job(tmp_path, bag, lidar_topic=LIVOX)
    rec = scaled(J)
    assert rec["ok"] and rec["s"] == pytest.approx(1.25, rel=0.01), rec["reason"]
    assert rec["time"]["base"] == "log" and rec["time"]["log_minus_header_ms"] == {"camera": 50.0, "lidar": -4950.0}
    assert rec["warnings"] == ["the camera's and the LiDAR's header clocks differ by +5.00 s (log time minus header "
                               "stamp: camera 0.050 s, LiDAR -4.950 s), so frames and scans are matched by log time, "
                               "which is less exact; give time_offset_s in the calibration"]
    upload(J, {"T_cam_lidar": T_CAM_LIDAR.tolist(), "time_offset_s": 5.0})
    rec = scaled(J)
    assert rec["ok"] and rec["s"] == pytest.approx(1.25, rel=0.01) and rec["warnings"] == []
    assert rec["time"]["base"] == "header" and rec["time"]["offset_ms"] == 5000.0

    # A wrong offset still takes the header base, but the log times disagree with it: a warning, and no
    # frame finds its points.
    upload(J, {"T_cam_lidar": T_CAM_LIDAR.tolist(), "time_offset_s": 4.0})
    assert failed(J).startswith("LiDAR points cover the object in only 0 of 8 views")
    rec = json.loads((J / "pool" / "lidar.json").read_text())
    assert rec["time"]["base"] == "header" and rec["warnings"][0] == (
        "time_offset_s is 4 s, but the log times put the LiDAR's header clock +5.00 s from the camera's (log time "
        "minus header stamp: camera 0.050 s, LiDAR -4.950 s); check time_offset_s")

    # The camera's header stamps cannot be used: the log base, where the offset is not added (log times
    # come from the recorder's one clock), so every frame still finds its own scan.
    upload(J, {"T_cam_lidar": T_CAM_LIDAR.tolist(), "time_offset_s": 5.0})
    st = json.loads((J / "frames" / "stamps.json").read_text())
    (J / "frames" / "stamps.json").write_text(json.dumps({**st, "clock": "log"}))
    rec = scaled(J)
    assert rec["ok"] and rec["s"] == pytest.approx(1.25, rel=0.01), rec["reason"]
    assert (rec["time"]["base"], rec["time"]["offset_ms"]) == ("log", 0.0)
    assert rec["warnings"] == ["the camera's header stamps are zero, go backwards or repeat, so frames and scans are "
                               "matched by log time",
                               "time_offset_s (5 s) is not used: it relates the header stamps, and frames and scans "
                               "are matched by log time"]

    # A merged bag: each sensor's log times copy its own header stamps (plus the same delay), so the
    # LiDAR's scans are logged 5 s after the frames. The offset finds them; the log times agree with it.
    merged = write_scene_bag(tmp_path / "merged.bag", n=8, kind="livox", lidar_clock_s=5.0, lidar_late_s=5.0, step=2)
    J = make_job(tmp_path / "m", merged, lidar_topic=LIVOX)
    upload(J, {"T_cam_lidar": T_CAM_LIDAR.tolist(), "time_offset_s": 5.0})
    rec = scaled(J)
    assert rec["ok"] and rec["s"] == pytest.approx(1.25, rel=0.01) and rec["warnings"] == [], rec["reason"]
    assert rec["time"]["base"] == "header" and rec["time"]["log_minus_header_ms"] == {"camera": 50.0, "lidar": 50.0}


def test_time_base_and_matching_rules(monkeypatch):
    stamps = {"clock": "header", "header_ns": [S0, S0 + 10**8], "log_ns": [S0 + LAG, S0 + 10**8 + LAG]}
    frames = ["f00000", "f00001"]

    def scans(lag, head=0):
        return [(S0 + k * 10**8 + lag, S0 + k * 10**8 + head, np.zeros((1, 3)), None) for k in range(2)]

    assert L.time_base(stamps, frames, scans(LAG + 150_000_000), 0)[0] == "header"  # within 0.2 s
    base, rec, warn = L.time_base(stamps, frames, scans(LAG + 250_000_000), 0)
    assert base == "log" and "differ by -0.25 s" in warn[0] and rec["log_minus_header_ms"]["lidar"] == 300.0
    base, rec, warn = L.time_base(stamps, frames, scans(LAG, head=2 * 10**9), 2 * 10**9)  # the offset explains it
    assert (base, rec["offset_ms"], warn) == ("header", 2000.0, [])
    # With an offset the base is the header's; the log times only check it. A bag whose log times copy the
    # header stamps (scans logged 2 s later too) agrees with it; log times that fit neither give a warning.
    assert L.time_base(stamps, frames, scans(LAG + 2 * 10**9, head=2 * 10**9), 2 * 10**9)[1:] == (
        {"base": "header", "window_ms": 50.0, "offset_ms": 2000.0,
         "log_minus_header_ms": {"camera": 50.0, "lidar": 50.0}},
        [])
    base, rec, warn = L.time_base(stamps, frames, scans(LAG, head=2 * 10**9), 10**9)
    assert (base, rec["offset_ms"]) == ("header", 1000.0) and warn == [
        "time_offset_s is 1 s, but the log times put the LiDAR's header clock +2.00 s from the camera's (log time "
        "minus header stamp: camera 0.050 s, LiDAR -1.950 s); check time_offset_s"]
    no_head = [(S0 + LAG, 0, np.zeros((1, 3)), None)] + scans(LAG)[1:]
    base, _, warn = L.time_base(stamps, frames, no_head, 0)
    assert base == "log" and warn == ["some scans have no header stamp, so frames and scans are matched by log time"]
    base, rec, warn = L.time_base({**stamps, "clock": "log"}, frames, scans(LAG), 0)
    assert base == "log" and rec["log_minus_header_ms"]["camera"] is None and "camera's header stamps" in warn[0]
    base, rec, warn = L.time_base({**stamps, "clock": "log"}, frames, scans(LAG), 10**9)  # not used in the log base
    assert (base, rec["offset_ms"], len(warn)) == ("log", 0.0, 2) and warn[1] == (
        "time_offset_s (1 s) is not used: it relates the header stamps, and frames and scans are matched by log time")

    # The scan period: the median step between the scans read (steps between read windows do not count),
    # else topics.json's rate, else 100 ms.
    def at(*ms):
        return [(S0 + t * 10**6, S0 + t * 10**6 - 7, np.zeros((1, 3)), None) for t in ms]

    assert L.scan_period(at(0, 100, 200, 300, 5000, 5100, 5200), "header", 1000.0) == (1e8, "scans")
    assert L.scan_period(at(0, 5000), "log", 20.0) == (5e7, "rate_hz")
    assert L.scan_period(at(0), "header", None) == (1e8, "default")

    # Per-point times: the points within +-50 ms, in the header's or the log's clock.
    t = S0 + np.arange(0, 100_000_000, 10_000_000)
    xyz = np.arange(30.0).reshape(10, 3)
    scan = [(S0 + 120_000_000, S0, xyz, t)]
    assert len(L.frame_points(scan, "header", S0 + 50_000_000, 1e8)) == 10
    assert L.frame_points(scan, "header", S0 + 120_000_000, 1e8).tolist() == xyz[7:].tolist()  # 70 to 90 ms
    assert L.frame_points(scan, "log", S0 + 240_000_000, 1e8).tolist() == xyz[7:].tolist()  # shifted by 120 ms
    # Without them: the nearest scan, within one scan period.
    plain = [(S0 + k * 10**8, S0 + k * 10**8, xyz + k, None) for k in range(3)]
    assert L.frame_points(plain, "header", S0 + 290_000_000, 1e8).tolist() == (xyz + 2).tolist()
    assert len(L.frame_points(plain, "header", S0 + 310_000_000, 1e8)) == 0
    assert L.frame_points(plain, "log", S0 + 40_000_000, 1e8).tolist() == xyz.tolist()

    # While reading, a scan keeps only the points that keep chooses (with their times); a scan that keeps
    # none still counts for the clocks, and the window skips it.
    def fake_scans(path, topic, windows):
        yield S0 + LAG, S0, np.array([[0, 0, 2.0], [0, 0, -2.0]], np.float32), np.array([S0, S0 + 1])
        yield S0 + LAG + 10**8, S0 + 10**8, np.array([[0, 0, -1.0]], np.float32), np.array([S0 + 10**8])

    monkeypatch.setattr(rosbag, "scans", fake_scans)
    got = L.read_scans("x.bag", POINTS, [(S0, S0 + 10**9)], keep=lambda xyz, log: xyz[:, 2] > 0)
    assert [(log, head, xyz.tolist(), t.tolist()) for log, head, xyz, t in got] == [
        (S0 + LAG, S0, [[0, 0, 2.0]], [S0]), (S0 + LAG + 10**8, S0 + 10**8, [], [])]
    assert L.frame_points(got, "header", S0, 1e8).tolist() == [[0, 0, 2.0]]
    assert L.frame_points(got, "log", S0 + LAG + 10**8, 1e8).shape == (0, 3)
    assert L.frame_points([], "header", S0, 1e8).shape == (0, 3)  # a frame whose windows hold no scan

    # in_view: a scan keeps the points in the image of a frame whose read window holds its log time, with
    # that frame's transform (f00001's puts the point behind the camera).
    intr = {"source": "test", "K": F.intrinsics(*SIZE), "D": [], "model": None, "R": None, "size": list(SIZE)}
    back = np.diag([1.0, 1.0, -1.0, 1.0])
    keep = L.in_view(["f00000", "f00001"], {"f00000": [(0, 10)], "f00001": [(5, 30)]},
                     {"f00000": np.eye(4), "f00001": back}, intr, {"f00000": SIZE, "f00001": SIZE})
    ahead = np.array([[0.0, 0.0, 2.0], [5.0, 0.0, 1.0]])  # on the axis, and far left of the image
    assert keep(ahead, 2).tolist() == [True, False] and keep(ahead, 7).tolist() == [True, False]
    assert keep(ahead, 20).tolist() == [False, False] and keep(ahead, 40).tolist() == [False, False]
    assert keep(-ahead, 20).tolist() == [True, False]


def test_the_samples_cache(points_bag, tmp_path, monkeypatch):
    """Every extracted frame is sampled, also one that prep left out, so a rerun with the same key reads
    no scans, also after a mask change brings that view into the pool or takes one out; another bag
    file (mtime) or another calibration is a miss."""
    J = make_job(tmp_path, points_bag, frames=[1, 3, 5, 7, 9])
    out = tmp_path / "out"
    for d in ("input/images", "input/object", "masks"):  # prep left f00009 out (its mask was missing)
        (out / d).mkdir(parents=True)
        (J / "pool" / d / "f00009.png").rename(out / d / "f00009.png")
    first = scaled(J)
    assert first["source"]["cache"] == "miss" and len(first["views"]) == 4
    meta = json.loads((J / "lidar" / "samples.json").read_text())
    assert meta["key"] == {"version": 2, "kind": "rosbag",
                           "input": {"name": "points.bag", "bytes": points_bag.stat().st_size,
                                     "mtime_ns": points_bag.stat().st_mtime_ns},
                           "topics": {"camera": CAMERA, "lidar": POINTS}, "calibration_sha256": None, "window_ms": 50.0}
    assert meta["frames"] == ["f00001", "f00003", "f00005", "f00007", "f00009"]
    reads = []
    scans = rosbag.scans
    monkeypatch.setattr(rosbag, "scans", lambda *a: reads.append(a) or scans(*a))
    again = scaled(J)
    assert again["source"]["cache"] == "hit" and reads == [] and again["s"] == first["s"]
    same = ("extrinsic", "intrinsics", "time")
    assert {k: again[k] for k in same} == {k: first[k] for k in same}
    for d in ("input/images", "input/object", "masks"):  # a mask change: prep now has f00009
        (out / d / "f00009.png").rename(J / "pool" / d / "f00009.png")
    more = scaled(J)
    assert more["source"]["cache"] == "hit" and reads == [] and len(more["views"]) == 5
    assert more["views"][-1]["name"] == "f00009" and more["views"][-1]["status"] == "used"
    for d in ("input/images", "input/object", "masks"):  # and another: prep left f00009 out again
        (J / "pool" / d / "f00009.png").unlink()
    fewer = scaled(J)
    assert fewer["source"]["cache"] == "hit" and reads == [] and len(fewer["views"]) == 4
    assert sorted(p.stem for p in (J / "lidar" / "check").glob("*.jpg")) == ["f00001", "f00003", "f00005", "f00007"]
    os.utime(points_bag, ns=(points_bag.stat().st_atime_ns, points_bag.stat().st_mtime_ns + 10**9))
    assert scaled(J)["source"]["cache"] == "miss" and len(reads) == 1
    upload(J, {"T_cam_lidar": T_CAM_LIDAR.tolist()})
    assert scaled(J)["source"]["cache"] == "miss" and len(reads) == 2
    assert scaled(J)["source"]["cache"] == "hit" and len(reads) == 2
    # The stamps of the frames come from the camera topic again when stamps.json is missing.
    (J / "lidar" / "samples.json").unlink()
    (J / "frames" / "stamps.json").unlink()
    assert scaled(J)["s"] == pytest.approx(first["s"], rel=1e-6)


def test_each_failure_says_why(points_bag, tmp_path):
    J = make_job(tmp_path, points_bag, frames=[0, 2, 4, 6, 8])
    job = json.loads((J / "job.json").read_text())
    root = tmp_path / "bags"
    root.mkdir()

    # No TF link and no calibration; the calibration then fixes it.
    set_job(J, video={**job["video"], "path": str(write_scene_bag(root / "no_tf.bag", tf=None))})
    assert failed(J) == "the bag's TF does not link camera_optical_frame to lidar; upload the calibration"
    assert json.loads((J / "pool" / "lidar.json").read_text())["source"]["topic"] == POINTS
    upload(J, {"T_cam_lidar": T_CAM_LIDAR.tolist()})
    assert scaled(J)["ok"]
    set_job(J, calibration=None)

    # No camera_info, then a calibration with K; K for another aspect ratio fails.
    set_job(J, video={**job["video"], "path": str(write_scene_bag(root / "no_info.bag", info=False))})
    assert failed(J) == "/cam/image_rect has no usable camera_info (/cam/camera_info); give K (and D) in the calibration"
    upload(J, {"T_cam_lidar": T_CAM_LIDAR.tolist(), "K": F.intrinsics(*SIZE).tolist()})
    assert scaled(J)["intrinsics"]["source"] == "calibration calib.json: K"
    upload(J, {"T_cam_lidar": T_CAM_LIDAR.tolist(), "K": F.intrinsics(*SIZE).tolist(), "image_size": [100, 100]})
    assert failed(J) == ("the intrinsics (calibration calib.json: K) are for 100x100 images but the frames are 96x64; "
                         "give K for the frames' size in the calibration")
    upload(J, {"T_cam_lidar": T_CAM_LIDAR.tolist(), "K": F.intrinsics(192, 128).tolist(), "image_size": [192, 128]})
    assert scaled(J)["ok"]  # K for twice the size, scaled to the frames
    (J / "lidar" / "calibration" / "calib.json").unlink()
    assert failed(J) == "the calibration file calib.json is missing; upload it again"
    set_job(J, calibration=None)

    # A raw topic whose camera_info has an unknown distortion model (its stamps are read again).
    raw = write_scene_bag(root / "raw.bag", topic="/cam/image_raw", model="weird_model")
    set_job(J, video={**job["video"], "path": str(raw)}, source={"kind": "rosbag", "topic": "/cam/image_raw"})
    assert failed(J) == ("/cam/camera_info: unknown distortion model 'weird_model'; use plumb_bob, rational_polynomial or "
                         "equidistant; use a rectified topic or give K and D in the calibration")
    set_job(J, video=job["video"], source=job["source"])

    # The scans were logged long after the frames.
    set_job(J, video={**job["video"], "path": str(write_scene_bag(root / "late.bag", lidar_late_s=30.0))})
    assert failed(J) == ("/lidar/points has no scans within 0.3 s of the frames' times; was the LiDAR recording "
                         "during the camera's frames?")
    # A bag without the listed topic: rosbag.py's own reason.
    set_job(J, video={**job["video"], "path": str(write_scene_bag(root / "other.bag", kind="livox", step=4))})
    assert failed(J) == "the bag has no topic /lidar/points"
    set_job(J, video=job["video"])

    # Another topic, and a bag listed before the LiDAR listing.
    set_job(J, lidar={"kind": "rosbag", "topic": "/nope"})
    assert failed(J) == "/nope is not a LiDAR topic of the bag"
    set_job(J, lidar=job["lidar"])
    topics = json.loads((J / "frames" / "topics.json").read_text())
    (J / "frames" / "topics.json").write_text(json.dumps({k: v for k, v in topics.items() if k != "lidar"}))
    assert failed(J) == L.OLD_TOPICS == ("the bag was listed before LiDAR topics were; choose its camera topic again "
                                         "to list them (for a job in review or done: upload the bag again)")
    (J / "frames" / "topics.json").write_text(json.dumps(topics))

    # T_lidar_cam given as T_cam_lidar: the points miss the object; the estimate says so, with check images.
    upload(J, {"T_cam_lidar": np.linalg.inv(T_CAM_LIDAR).tolist()})
    reason = failed(J)
    rec = json.loads((J / "pool" / "lidar.json").read_text())
    assert reason.startswith("LiDAR points cover the object in only") and len(rec["views"]) == 5
    assert rec["extrinsic"]["source"] == "calibration" and len(list((J / "lidar" / "check").glob("*.jpg"))) == 5
    with pytest.raises(SystemExit, match="LiDAR points cover the object in only"):
        L.main(["scale", "--job", str(J)])
    upload(J, {"T_cam_lidar": T_CAM_LIDAR.tolist()})
    assert L.main(["scale", "--job", str(J)]) is None and steps.done("lidar", J)

    # A pool view that prep.json does not know: the estimate's reason.
    prep = json.loads((J / "pool" / "prep.json").read_text())
    (J / "pool" / "prep.json").write_text(json.dumps({**prep, "views": prep["views"][1:]}))
    assert failed(J) == f"{prep['views'][0]['name']} is not in {J / 'pool' / 'prep.json'}; rerun prep"
    (J / "pool" / "prep.json").write_text(json.dumps(prep))

    # A job without LiDAR has no such stage; a malformed setting says so.
    set_job(J, lidar=None)
    with pytest.raises(SystemExit, match="this job has no LiDAR"):
        L.scale(J)
    set_job(J, lidar={"kind": "rosbag"})
    F.da3(J)  # the fake DA3 takes it as no LiDAR; the lidar stage says why
    assert F.run_scale(J / "pool", "f00000") == 1.0
    with pytest.raises(SystemExit, match=re.escape('job.json: lidar is null, {"kind": "rosbag", "topic": T} or '
                                                   '{"kind": "stray"}')):
        L.scale(J)
    # Stray's LiDAR on a bag job says why.
    set_job(J, lidar={"kind": "stray"})
    assert failed(J) == "LiDAR of kind stray needs a Stray Scanner capture; this job's input is a rosbag"


def test_a_calibration_with_a_rectified_camera_info(tmp_path):
    """A rectified topic's camera_info R turns the points after the calibration's T_cam_lidar, so T must be
    to the unrectified optical frame: the record says so. K in the calibration leaves R out."""
    a = np.radians(2.0)
    R = np.array([[np.cos(a), 0.0, np.sin(a)], [0.0, 1.0, 0.0], [-np.sin(a), 0.0, np.cos(a)]])
    J = make_job(tmp_path, write_scene_bag(tmp_path / "rect.bag", n=6, tf=None, info_r=R, step=2))
    upload(J, {"T_cam_lidar": T_CAM_LIDAR.tolist()})
    with contextlib.suppress(SystemExit):  # 2 degrees off: the scale may or may not pass
        L.scale(J)
    rec = json.loads((J / "pool" / "lidar.json").read_text())
    assert np.allclose(rec["intrinsics"]["R"], R) and rec["warnings"][0] == (
        "the camera_info of /cam/image_rect has a rectification R of 2.00 deg, which turns the points after the "
        "calibration's T_cam_lidar: that must be to the unrectified optical frame (camera_optical_frame); if it was "
        "calibrated on the rectified images, give K in the calibration too")
    upload(J, {"T_cam_lidar": T_CAM_LIDAR.tolist(), "K": F.intrinsics(*SIZE).tolist()})
    rec = scaled(J)
    assert rec["ok"] and rec["s"] == pytest.approx(1.25, rel=0.01) and rec["warnings"] == []
    assert rec["intrinsics"]["R"] is None


def test_tf_reason():
    assert L.tf_reason("cam", "lidar", "/points") == "the bag's TF does not link cam to lidar; upload the calibration"
    assert L.tf_reason("", "lidar", "/points") == ("the camera's images have no frame_id, so the bag's TF cannot link "
                                                   "them to lidar; upload the calibration")
    assert L.tf_reason("cam", "", "/points") == ("the scans of /points have no frame_id, so the bag's TF cannot link "
                                                 "them; upload the calibration")


# ---- a Stray Scanner capture ----

def make_stray_job(root, portrait=0, n=24, **kw):
    """A Stray Scanner job after da3, as the stages leave it: the zip in video/ (write_stray of
    fake_steps' scene; kw go to it), the real analyze and extract (which unpack it in video/stray),
    fake_steps' masks, prep and the fake DA3 (FAKE_DA3_SCALE, since the job has LiDAR)."""
    from test_webapp_stray import write_stray

    from gator.webapp import video

    J = root / "job"
    (J / "video").mkdir(parents=True)
    zipped = write_stray(root / "src", n=n, portrait=portrait, **kw).rename(J / "video" / "capture.zip")
    (J / "job.json").write_text(json.dumps({
        "id": "20261021-120000-abcdef", "video": {"name": zipped.name, "path": str(zipped)},
        "source": {"kind": "stray", "topic": None}, "lidar": {"kind": "stray"}, "calibration": None, "mode": "auto",
        "prompt": "box", "slug": "box"}))
    video.analyze(J)
    video.extract(J)
    (J / "scene" / "sam3_box").mkdir()
    for p in sorted((J / "scene" / "images").glob("*.png")):
        size = Image.open(p).size
        mask = F.render(*size, F.frame_angle(J, p.stem))[1]
        Image.fromarray(mask.astype(np.uint8) * 255).save(J / "scene" / "sam3_box" / p.name)
    mvsam3d.prep(J / "scene", "sam3_box", J / "pool", 1024)
    F.da3(J)
    return J


def test_a_stray_capture_gives_the_scale_and_its_cache(tmp_path, capsys, monkeypatch):
    """A portrait capture: its depth images, turned upright as the frames were, give the scale that the
    fake DA3 lacks; a rerun reads no depth image, and another zip does."""
    from gator.webapp import stray

    J = make_stray_job(tmp_path, portrait=1)
    capsys.readouterr()
    names = mvsam3d.view_names(J / "pool")
    extracted = sorted(p.stem for p in (J / "scene" / "images").glob("*.png"))
    assert F.run_scale(J / "pool", names[0]) == F.FAKE_DA3_SCALE
    rec = scaled(J)
    assert rec["ok"] and rec["s"] == pytest.approx(1 / F.FAKE_DA3_SCALE, rel=0.01), rec["reason"]
    assert steps.done("lidar", J) and steps.guard("lidar", J) is None
    assert [v["name"] for v in rec["views"]] == names and all(v["status"] == "used" for v in rec["views"])
    assert rec["source"] == {"kind": "stray", "file": "capture.zip", "depth_size": [256, 192],
                             "frames": len(extracted), "cache": "miss"}
    assert rec["extrinsic"] == {"source": "stray (depth aligned with the color camera)", "calibration_sha256": None}
    assert rec["intrinsics"] == {"source": "stray: depth scaled to the frame size", "depth_size": [256, 192],
                                 "frame_size": [240, 320], "turns": 1, "min_confidence": 2, "z_range_m": [0.1, 5.0]}
    assert rec["time"] == {"base": "frame index"} and rec["warnings"] == []
    out = capsys.readouterr().out
    assert ("LiDAR stray capture.zip, cache miss; extrinsic from stray (depth aligned with the color camera); "
            "intrinsics from stray: depth scaled to the frame size") in out
    assert "time base frame index" in out and f"metric scale {rec['s']:.4f}" in out
    # The samples are the frames' pixels: they land where the upright camera sees that depth.
    with np.load(J / "lidar" / "samples.npz") as f:
        uvz = f[names[2]]
    depth = F.render(240, 320, F.frame_angle(J, names[2]))[0]
    near = uvz[:, 2] < F.FAR - 0.01
    err = np.abs(depth[lidar_scale.pixel(uvz[near, 1]), lidar_scale.pixel(uvz[near, 0])] / uvz[near, 2] - 1)
    assert len(uvz) == 256 * 192 and np.median(err) < 2e-3

    meta = json.loads((J / "lidar" / "samples.json").read_text())
    zipped = J / "video" / "capture.zip"
    assert meta["key"] == {"version": 2, "kind": "stray",
                           "input": {"name": "capture.zip", "bytes": zipped.stat().st_size,
                                     "mtime_ns": zipped.stat().st_mtime_ns},
                           "turns": 1, "frame_sizes": [[240, 320]], "min_confidence": 2, "z_range_m": [0.1, 5.0]}
    assert meta["frames"] == extracted
    reads, depth_samples = [], stray.depth_samples
    monkeypatch.setattr(stray, "depth_samples", lambda *a, **kw: reads.append(a[1]) or depth_samples(*a, **kw))
    again = scaled(J)
    assert again["source"]["cache"] == "hit" and reads == [] and again["s"] == rec["s"]
    assert {k: again[k] for k in ("extrinsic", "intrinsics", "time")} == {k: rec[k] for k in ("extrinsic", "intrinsics",
                                                                                              "time")}
    os.utime(zipped, ns=(zipped.stat().st_atime_ns, zipped.stat().st_mtime_ns + 10**9))  # another upload
    assert scaled(J)["source"]["cache"] == "miss" and sorted(reads) == [int(n[1:]) for n in extracted]


def test_stray_samples_without_depth_and_each_failure(tmp_path, monkeypatch):
    """Frames without a depth image, or with one but without its confidence image, get no samples, with
    a warning; a missing zip, a frames analysis that is not a capture's, or a bag's LiDAR setting on a
    capture fail with the reason, as does a disk too full to unpack the capture again."""
    from gator.webapp import stray

    J = make_stray_job(tmp_path, n=24, n_depth=22, without_confidence=range(1, 22, 4))
    job = json.loads((J / "job.json").read_text())
    names = mvsam3d.view_names(J / "pool")
    meta = json.loads((J / "frames" / "meta.json").read_text())
    assert meta["video"]["n_depth"] == 22 and meta["video"]["warnings"] == [
        "2 of the 24 frames have no depth image, so they have no LiDAR samples",
        "6 of the 22 depth images have no confidence image, so those frames have no LiDAR samples"]
    frames = sorted(p.stem for p in (J / "scene" / "images").glob("*.png"))
    late = [n for n in frames if int(n[1:]) >= 22]
    unsure = [n for n in frames if int(n[1:]) < 22 and int(n[1:]) % 4 == 1]
    assert late and unsure  # the last frames are candidates of this capture, and some without confidence
    samples, info = L.stray_samples(J, job, {"kind": "stray"}, names)
    assert sorted(samples) == frames
    assert all(len(samples[n]) == (0 if n in late + unsure else 256 * 192) for n in frames)
    assert info["warnings"] == [f"frames {', '.join(late)} have no depth image, so they have no samples",
                                f"frames {', '.join(unsure)} have a depth image but no confidence image, so they "
                                "have no samples"]
    rec = scaled(J)
    assert rec["ok"] and rec["s"] == pytest.approx(1.25, rel=0.01)
    empty = sorted(n for n in names if n in late + unsure)
    assert rec["warnings"] == info["warnings"] + [f"no LiDAR samples in {len(empty)} of {len(names)} views: "
                                                  f"{', '.join(empty)}"]
    assert {v["name"]: v["status"] for v in rec["views"] if v["name"] in empty} == dict.fromkeys(empty, "no_samples")

    (J / "lidar").rename(tmp_path / "cache")  # the stage reads the capture again
    zipped = J / "video" / "capture.zip"
    zipped.rename(tmp_path / "capture.zip")
    assert failed(J) == "the uploaded export capture.zip is not on the server; upload it again"
    rec = json.loads((J / "pool" / "lidar.json").read_text())
    assert rec["source"] == {"kind": "stray", "file": "capture.zip"} and rec["views"] == []
    (tmp_path / "capture.zip").rename(zipped)
    (J / "frames" / "meta.json").write_text(json.dumps({**meta, "video": {**meta["video"], "kind": "video"}}))
    assert failed(J) == "frames/meta.json is not of a Stray Scanner capture; analyze the input again"
    (J / "frames" / "meta.json").write_text(json.dumps(meta))
    (J / "video" / "stray" / ".unpacked.json").unlink()  # the stage unpacks the zip again, onto a full disk

    def full(*a):
        raise OSError(28, "No space left on device")

    with monkeypatch.context() as m:
        m.setattr(stray, "_extract", full)
        assert failed(J) == "cannot unpack capture.zip: [Errno 28] No space left on device"
    assert sorted(p.name for p in (J / "video" / "stray").iterdir()) == []
    set_job(J, lidar={"kind": "rosbag", "topic": POINTS})
    assert failed(J) == "LiDAR of kind rosbag needs a rosbag input; this job's input is a Stray Scanner capture"
    set_job(J, lidar={"kind": "stray"})
    assert scaled(J)["ok"]
