import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy.ndimage import gaussian_filter

from gator.webapp import video as V

H, W, N, BLUR = 96, 160, 64, 30
TABLE = Path("~/Downloads/table.MOV").expanduser()


def picture(i):
    """A textured background with a white square moving right."""
    rng = np.random.default_rng(0)
    a = (rng.random((H // 4, W // 4, 3)) * 200).astype(np.uint8).repeat(4, 0).repeat(4, 1)
    a[30:60, 10 + 2 * i:40 + 2 * i] = 255
    return a


def shot(i):
    """Frame i of the test inputs: picture(i), blurred at BLUR."""
    a = picture(i)
    return gaussian_filter(a.astype(float), (3, 3, 0)).astype(np.uint8) if i == BLUR else a


def write_video(path, rotation=0):
    import av

    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=30, options={"crf": "12"})
        s.width, s.height, s.pix_fmt = W, H, "yuv420p"
        if rotation:
            s.set_display_rotation(rotation)
        for i in range(N):
            a = shot(i)
            for p in s.encode(av.VideoFrame.from_ndarray(a, format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    return path


def write_bag(path, topics=("/side/image_raw", "/cam/image_rect"), stamp=None):
    """A ROS1 bag with an rgb8 Image topic per name at 30 Hz: shot(i), flipped upside down on the first topic.
    stamp(i, ns): the header stamp of message i (None = ns, 40 ms before the log time)."""
    from rosbags.rosbag1 import Writer
    from rosbags.typesys import Stores, get_typestore

    ts = get_typestore(Stores.ROS1_NOETIC)
    T = ts.types
    with Writer(path) as w:
        conns = [w.add_connection(t, "sensor_msgs/msg/Image", typestore=ts) for t in topics]
        for i in range(N):
            ns = 1_749_907_120 * 10**9 + round(i * 1e9 / 30)
            st = stamp(i, ns) if stamp else ns
            head = T["std_msgs/msg/Header"](seq=i, stamp=T["builtin_interfaces/msg/Time"](sec=st // 10**9, nanosec=st % 10**9),
                                            frame_id="cam")
            for k, c in enumerate(conns):
                a = np.ascontiguousarray(shot(i)[::-1] if k == 0 else shot(i))
                msg = T["sensor_msgs/msg/Image"](header=head, height=H, width=W, encoding="rgb8", is_bigendian=0,
                                                 step=W * 3, data=a.reshape(-1))
                w.write(c, ns + 40_000_000, ts.serialize_ros1(msg, "sensor_msgs/msg/Image"))
    return path


def job(tmp_path, video, **kw):
    d = tmp_path / "job"
    d.mkdir()
    (d / "job.json").write_text(json.dumps({"video": {"name": video.name, "path": str(video)}, "mode": "auto", **kw}))
    return d


def test_probe_and_frames(tmp_path):
    path = write_video(tmp_path / "v.mp4")
    info = V.probe(path)
    assert info["n_frames"] == N and (info["width"], info["height"]) == (W, H) and info["rotation"] == 0
    assert info["codec"] == "h264" and info["hdr"] is None and info["bit_depth"] == 8
    assert info["duration_s"] == pytest.approx(N / 30, abs=0.05) and info["fps"] == pytest.approx(30, abs=1)
    got = list(V.frames(path, keep={0, 5}))
    assert [i for i, _, _ in got] == [0, 5] and got[1][1] == pytest.approx(5 / 30, abs=1e-3)
    assert got[0][2].shape == (H, W, 3) and np.abs(got[0][2].astype(float) - picture(0)).mean() < 8
    assert next(V.frames(path, width=80))[2].shape == (48, 80, 3)


def test_rotated_video_comes_out_upright(tmp_path):
    for rot, k in ((90, 1), (-90, 3), (180, 2)):
        path = write_video(tmp_path / f"r{rot}.mp4", rotation=rot)
        info = V.probe(path)
        assert info["rotation"] in (rot, rot - 360) and (info["width"], info["height"]) == ((H, W) if k % 2 else (W, H))
        _, _, a = next(V.frames(path))
        assert np.abs(a.astype(float) - np.rot90(picture(0), k)).mean() < 8
        _, _, small = next(V.frames(path, width=48))
        assert small.shape == ((80, 48, 3) if k % 2 else (29, 48, 3))


def test_metrics_find_the_blurred_frame(tmp_path):
    path = write_video(tmp_path / "v.mp4")
    got = list(V.frames(path))
    m = V.metrics((V.gray(a) for _, _, a in got), [t for _, t, _ in got])
    assert m["rel"][BLUR] < 0.6 and np.median(m["rel"]) == pytest.approx(1, abs=0.1)
    assert m["lv"][BLUR] < 0.5 * np.median(m["lv"])
    assert 60 < m["luma"].mean() < 160 and m["clip_hi"].max() > 0.03 and m["clip_lo"].max() < 0.05
    flat = np.full((10, 10), 255.0)
    assert V.frame_stats(flat) == (0.0, 255.0, 1.0, 0.0)


def test_candidates_and_snap():
    t = np.arange(300) / 30  # 10 s -> 20 bins of 15 frames; the central 60% of bin b is 15b+3 to 15b+12
    rel, luma = np.ones(300), np.full(300, 120.0)
    rel[[16, 20, 21, 35]] = (2.0, 1.5, 1.4, 1.2)  # 16 is the sharpest but outside the central part
    rel[45:60] = 0.3  # bin 3 is blurred
    luma[60:75] = 10  # bin 4 is too dark
    c = V.candidates(t, rel, luma, 10.0)
    assert c[:3] == [3, 20, 35] and c == sorted(c) and len(c) == 18
    assert not any(45 <= i < 75 for i in c)
    assert all(b - a >= 6 for a, b in zip(c[:-1], c[1:]))
    assert len(V.candidates(np.arange(30) / 30, np.ones(30), np.full(30, 100.0), 1.0)) <= 16  # K is at least 16
    assert len(V.candidates(np.arange(3000) / 30, np.ones(3000), np.full(3000, 100.0), 100.0)) == 32
    assert V.candidates([0.0, 0.0], [1.0, 2.0], [100.0, 100.0], 0.0) == [1]  # a zero duration: one frame, once
    assert V.snap(25, t, rel) == 20 and V.snap(12, t, rel) == 16 and V.snap(40, t, rel) == 35
    assert V.snap(0, t, np.ones(300)) == 0 and V.snap(100, t, np.ones(300)) == 100


def test_analyze_and_extract(tmp_path, capsys):
    path = write_video(tmp_path / "v.mp4")
    j = job(tmp_path, path, mode="manual", picks=[BLUR, 2])
    V.main(["analyze", "--job", str(j)])
    out = capsys.readouterr().out
    assert "PROGRESS 0" in out and "PROGRESS 1" in out
    meta = json.loads((j / "frames" / "meta.json").read_text())
    assert meta["video"]["n_frames"] == N and len(meta["times"]) == len(meta["rel"]) == N
    assert meta["video"]["kind"] == "video" and "topic" not in meta["video"]
    assert not (j / "frames" / "stamps.json").exists()  # only a rosbag has absolute stamps
    assert BLUR not in meta["candidates"] and meta["candidates"]
    assert meta["thumbs"] == list(range(N)) and meta["thumb_size"] == [W, H]
    assert Image.open(j / "frames" / "thumbs" / "5.jpg").size == (W, H)  # never enlarged
    V.main(["extract", "--job", str(j)])
    pool = json.loads((j / "frames" / "pool.json").read_text())
    snapped = pool["picks"][str(BLUR)]
    assert snapped != BLUR and abs(snapped - BLUR) <= 8 and pool["picks"]["2"] in range(10)
    assert set(pool["pool"]) == set(meta["candidates"]) | {snapped, pool["picks"]["2"]}
    files = sorted(p.name for p in (j / "scene" / "images").iterdir())
    assert files == sorted(f"{pool['names'][str(i)]}.png" for i in pool["pool"])
    a = np.asarray(Image.open(j / "scene" / "images" / f"f{snapped:05d}.png"))
    assert a.shape == (H, W, 3) and np.abs(a.astype(float) - picture(snapped)).mean() < 8
    # A rerun with other picks removes the frames that left the pool.
    (j / "job.json").write_text(json.dumps({"video": {"path": str(path)}, "mode": "auto"}))
    V.main(["extract", "--job", str(j)])
    assert len(list((j / "scene" / "images").iterdir())) == len(meta["candidates"])


def test_rosbag_job(tmp_path, capsys):
    pytest.importorskip("rosbags")
    bag = write_bag(tmp_path / "b.bag")
    j = job(tmp_path, bag, mode="manual", picks=[BLUR], source={"kind": "rosbag", "topic": None})
    with pytest.raises(SystemExit, match="choose the bag's camera topic"):
        V.analyze(j)
    (j / "frames" / "topic_previews").mkdir(parents=True)
    (j / "frames" / "topic_previews" / "7.jpg").write_bytes(b"from an earlier listing")
    V.main(["topics", "--job", str(j)])
    assert "PROGRESS 1" in capsys.readouterr().out
    t = json.loads((j / "frames" / "topics.json").read_text())
    assert [x["topic"] for x in t["topics"]] == ["/cam/image_rect", "/side/image_raw"] and t["skipped"] == []
    assert [x["rectified"] for x in t["topics"]] == [True, None] and t["topics"][0]["rate_hz"] == 30.0
    assert [x["preview"] for x in t["topics"]] == ["frames/topic_previews/0.jpg", "frames/topic_previews/1.jpg"]
    assert sorted(p.name for p in (j / "frames" / "topic_previews").iterdir()) == ["0.jpg", "1.jpg"]
    a = np.asarray(Image.open(j / "frames" / "topic_previews" / "1.jpg"))  # never enlarged; the middle message
    assert a.shape == (H, W, 3) and np.abs(V.gray(a) - V.gray(shot(N // 2)[::-1])).mean() < 3

    spec = json.loads((j / "job.json").read_text())
    (j / "job.json").write_text(json.dumps({**spec, "source": {"kind": "rosbag", "topic": "/cam/image_rect"}}))
    (j / "frames" / "thumbs").mkdir()
    (j / "frames" / "thumbs" / f"{N + 5}.jpg").write_bytes(b"from another topic")
    V.main(["analyze", "--job", str(j)])
    assert not (j / "frames" / "thumbs" / f"{N + 5}.jpg").exists()
    meta = json.loads((j / "frames" / "meta.json").read_text())
    v = meta["video"]
    assert (v["kind"], v["topic"], v["n_frames"], v["codec"], v["width"], v["height"]) == \
        ("rosbag", "/cam/image_rect", N, "Image rgb8", W, H)
    ns = [1_749_907_120 * 10**9 + round(i * 1e9 / 30) for i in range(N)]
    assert json.loads((j / "frames" / "stamps.json").read_text()) == {
        "topic": "/cam/image_rect", "clock": "header", "header_ns": ns, "log_ns": [x + 40_000_000 for x in ns]}
    assert "header_ns" not in json.dumps(meta)
    assert v["fps"] == pytest.approx(30) and meta["times"][:3] == pytest.approx([0, 1 / 30, 2 / 30], abs=1e-4)
    assert BLUR not in meta["candidates"] and meta["candidates"] and meta["thumbs"] == list(range(N))
    V.main(["extract", "--job", str(j)])
    pool = json.loads((j / "frames" / "pool.json").read_text())
    assert set(pool["pool"]) == set(meta["candidates"]) | {pool["picks"][str(BLUR)]} and pool["picks"][str(BLUR)] != BLUR
    for i in pool["pool"]:
        assert np.array_equal(np.asarray(Image.open(j / "scene" / "images" / f"f{i:05d}.png")), shot(i))
    (j / "job.json").write_text(json.dumps({**spec, "source": {"kind": "rosbag", "topic": "/gone"}}))
    with pytest.raises(SystemExit, match="no topic /gone"):
        V.analyze(j)


def test_rosbag_job_with_bad_stamps(tmp_path, capsys, monkeypatch):
    """A bag whose header stamps jump back is timed by log time; extract reads with the clock in meta.json."""
    pytest.importorskip("rosbags")
    from gator.webapp import rosbag

    bag = write_bag(tmp_path / "b.bag", ("/cam/image_rect",), stamp=lambda i, ns: ns if i < N // 2 else ns - 10**9)
    j = job(tmp_path, bag, source={"kind": "rosbag", "topic": "/cam/image_rect"})
    written, write_json = [], V.write_json
    monkeypatch.setattr(V, "write_json", lambda path, data: written.append(path.name) or write_json(path, data))
    V.main(["analyze", "--job", str(j)])
    assert written == ["stamps.json", "meta.json"]  # meta.json is the done marker
    st = json.loads((j / "frames" / "stamps.json").read_text())
    ns = [1_749_907_120 * 10**9 + round(i * 1e9 / 30) for i in range(N)]
    assert (st["topic"], st["clock"]) == ("/cam/image_rect", "log")  # both lists are kept whatever the clock
    assert st["header_ns"] == [x if i < N // 2 else x - 10**9 for i, x in enumerate(ns)]
    assert st["log_ns"] == [x + 40_000_000 for x in ns]
    out = capsys.readouterr().out
    assert "frame times from the log time" in out and "warning: 1 header stamps" in out
    meta = json.loads((j / "frames" / "meta.json").read_text())
    assert (meta["video"]["clock"], meta["video"]["stamp_warnings"]) == ("log", 1)
    assert meta["times"] == pytest.approx([i / 30 for i in range(N)], abs=1e-4) and meta["candidates"]
    clocks, frames = [], rosbag.frames
    monkeypatch.setattr(rosbag, "frames", lambda *a, clock=None, **kw: clocks.append(clock) or frames(*a, clock=clock, **kw))
    V.main(["extract", "--job", str(j)])
    assert clocks == ["log"] and len(list((j / "scene" / "images").iterdir())) == len(meta["candidates"])


def test_stray_job(tmp_path, capsys):
    """A zipped Stray Scanner export of a portrait capture: analyze unpacks it next to the zip and turns the
    frames upright by gravity; extract writes them upright."""
    from test_webapp_stray import DEPTH, picture, write_stray

    J = tmp_path / "job"
    (J / "video").mkdir(parents=True)
    zipped = write_stray(tmp_path / "src", n=24, portrait=1).rename(J / "video" / "capture.zip")
    (J / "job.json").write_text(json.dumps({"video": {"name": "capture.zip", "path": str(zipped)}, "mode": "auto",
                                            "source": {"kind": "stray", "topic": None}}))
    V.main(["analyze", "--job", str(J)])
    out = capsys.readouterr().out
    assert "Stray Scanner capture: 24 depth images (256x192), 24 poses; turned 90 degrees upright by gravity " \
        "(100% of the frames agree)" in out and "PROGRESS 1" in out
    assert (J / "video" / "stray" / "export" / "rgb.mp4").is_file() and (J / "video" / "stray" / ".unpacked.json").is_file()
    meta = json.loads((J / "frames" / "meta.json").read_text())
    v = meta["video"]
    assert (v["kind"], v["rotation"], v["rotation_source"], v["width"], v["height"], v["n_frames"]) == \
        ("stray", 90, "gravity", 240, 320, 24)
    assert (v["n_depth"], v["n_odometry"], v["depth_size"], v["warnings"]) == (24, 24, list(DEPTH), [])
    assert meta["thumb_size"] == [240, 320] and meta["candidates"] and len(meta["times"]) == 24
    assert not (J / "frames" / "stamps.json").exists()
    a = V.gray(np.asarray(Image.open(J / "frames" / "thumbs" / "3.jpg")))  # JPEG: compared in gray
    up = V.gray(picture(3, (240, 320)))
    assert a.shape == (320, 240) and np.abs(a - up).mean() < 10 < np.abs(a - up[::-1, ::-1]).mean()
    V.main(["extract", "--job", str(J)])
    pool = json.loads((J / "frames" / "pool.json").read_text())
    assert pool["pool"] == meta["candidates"]
    for i in pool["pool"]:
        a = np.asarray(Image.open(J / "scene" / "images" / f"f{i:05d}.png"))
        assert a.shape == (320, 240, 3) and np.abs(a.astype(float) - picture(i, (240, 320))).mean() < 8
    # A zip that is not an export fails the analysis with the reason.
    (J / "video" / "capture.zip").write_bytes(b"not a zip")
    with pytest.raises(SystemExit, match="capture.zip is not a readable zip"):
        V.analyze(J)


def test_bad_inputs(tmp_path):
    (tmp_path / "x.mp4").write_bytes(b"not a video")
    with pytest.raises(SystemExit, match="cannot read"):
        V.probe(tmp_path / "x.mp4")
    with pytest.raises(SystemExit, match="missing"):
        V.analyze(job(tmp_path, tmp_path / "gone.mp4"))
    (tmp_path / "x").mkdir()
    with pytest.raises(SystemExit, match="for a rosbag job; this job's input is a video"):
        V.topics(job(tmp_path / "x", tmp_path / "x.mp4"))
    (tmp_path / "y").mkdir()
    (tmp_path / "x.zip").write_bytes(b"PK")
    with pytest.raises(SystemExit, match="for a rosbag job; this job's input is a Stray Scanner capture"):
        V.topics(job(tmp_path / "y", tmp_path / "x.zip", source={"kind": "stray", "topic": None}))


@pytest.mark.skipif(not TABLE.exists(), reason="the user's video is not on this machine")
def test_probe_table_video():
    info = V.probe(TABLE)
    assert info["n_frames"] == 478 and info["hdr"] == "HLG" and info["bit_depth"] == 10
    assert (info["width"], info["height"]) == (1920, 1080) and info["codec"] == "hevc"
