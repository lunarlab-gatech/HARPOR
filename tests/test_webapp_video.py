import json
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy.ndimage import gaussian_filter

from harpor.webapp import video as V

H, W, N, BLUR = 96, 160, 64, 30


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


def write_video(path, rotation=0, options=None):
    import av

    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=30, options={"crf": "12", **(options or {})})
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
    (d / "job.json").write_text(json.dumps({"video": {"name": video.name, "path": str(video)}, "mode": "auto",
                                            "source": {"kind": "video", "topic": None}, **kw}))
    return d


def read_preview(path):
    """(facts, [(time_s, RGB)]) of a preview video: codec, pix_fmt, width, height, the first frame's rotation and
    the container's duration_s."""
    import av

    with av.open(str(path)) as c:
        s = c.streams.video[0]
        got = [(float(f.time), f.rotation, f.to_ndarray(format="rgb24")) for f in c.decode(s)]
        facts = {"codec": s.codec_context.name, "pix_fmt": s.codec_context.pix_fmt, "width": s.codec_context.width,
                 "height": s.codec_context.height, "rotation": got[0][1], "duration_s": c.duration / 1e6}
    return facts, [(t, a) for t, _, a in got]


def square_x(a):
    """The column centre of the white square (picture()) in an upright frame."""
    cols = np.flatnonzero((V.gray(a[35:55].astype(float)) > 230).mean(0) > 0.5)
    return cols.mean()


def check_preview(J, stride, size, upright=None, wrong=None):
    """J/frames/preview.mp4 against J/frames/meta.json: H.264 yuv420p in an MP4 with its index first, no rotation
    tag, the size (long side at most 640 px, both even), every stride-th frame at its meta.json time less t0
    (the time of frame 0), the duration up to the last kept frame plus one, and meta.json's "preview" record.
    upright(i), wrong(i): frame i upright and turned wrong; each kept frame is much nearer the upright one, in
    gray. Returns (meta, the kept frames, the preview's frames)."""
    meta = json.loads((J / "frames" / "meta.json").read_text())
    rec, times = meta["preview"], meta["times"]
    kept = list(range(0, len(times), stride))
    data = (J / "frames" / "preview.mp4").read_bytes()
    assert data[4:8] == b"ftyp" and data.index(b"moov") < data.index(b"mdat")  # it can play before it is all loaded
    assert not (J / "frames" / ".tmp_preview.mp4").exists()
    facts, got = read_preview(J / "frames" / "preview.mp4")
    assert (facts["codec"], facts["pix_fmt"], facts["rotation"]) == ("h264", "yuv420p", 0)
    assert (facts["width"], facts["height"]) == tuple(size) and max(size) <= 640 and size[0] % 2 == size[1] % 2 == 0
    assert rec == {"file": "frames/preview.mp4", "width": size[0], "height": size[1],
                   "fps": round(meta["video"]["fps"] / stride, 3), "stride": stride, "frames": len(kept), "t0": times[0],
                   "moved": {}, "bytes": len(data)}
    assert rec["fps"] <= V.PREVIEW_FPS * 1.01
    assert len(got) == len(kept)
    assert [t for t, _ in got] == pytest.approx([times[i] - times[0] for i in kept], abs=1e-3)
    last = times[kept[-1]] - times[0]
    assert last < facts["duration_s"] <= last + 1.5 / rec["fps"]
    if upright:
        for (_, a), i in zip(got, kept):
            right = np.abs(V.gray(a) - V.gray(upright(i))).mean()
            assert right < 12 and right < 0.5 * np.abs(V.gray(a) - V.gray(wrong(i))).mean(), i
    return meta, kept, got


def test_preview_plan():
    """The preview's size (long side at most 640 px, both sides even, never enlarged) and stride (at most 15 fps,
    with 1% leeway for a measured rate)."""
    plan = lambda w, h, fps: V.preview_plan({"width": w, "height": h, "fps": fps})  # noqa: E731
    assert plan(1920, 1080, 29.97) == (640, 360, 2) and plan(1080, 1920, 30.0012) == (360, 640, 2)
    assert plan(1440, 1920, 60) == (480, 640, 4) and plan(1920, 1440, 240) == (640, 480, 16)
    assert plan(160, 96, 30) == (160, 96, 2) and plan(193, 129, 10) == (192, 128, 1) and plan(1, 1, None) == (2, 2, 1)
    assert plan(640, 480, 24) == (640, 480, 2) and plan(640, 480, 15.1) == (640, 480, 1) and plan(640, 480, 15.5)[2] == 2


def test_preview_of_a_video(tmp_path, capsys, monkeypatch):
    """analyze writes the preview in its decode pass, before meta.json: every second frame of the 30 fps video
    (15 fps), at its own size, each frame where the video has it."""
    path = write_video(tmp_path / "v.mp4")
    j = job(tmp_path, path)
    replaced, replace = [], V.os.replace
    monkeypatch.setattr(V.os, "replace", lambda a, b: replaced.append(Path(b).name) or replace(a, b))
    V.main(["analyze", "--job", str(j)])
    assert replaced == ["preview.mp4", "meta.json"]  # meta.json is the done marker
    assert re.search(r"^preview video: 32 frames, 160x96, 15\.\d* fps, 0\.\d MB$", capsys.readouterr().out, re.M)
    _, kept, got = check_preview(j, 2, (W, H), lambda i: shot(i), lambda i: shot(i)[::-1, ::-1])
    assert [square_x(a) for _, a in got] == pytest.approx([square_x(shot(i)) for i in kept], abs=1.5)


def test_preview_of_a_rotated_video_is_upright(tmp_path):
    """The preview has no rotation tag: its frames are turned upright as the frames stage turns them."""
    path = write_video(tmp_path / "r.mp4", rotation=90)
    j = job(tmp_path, path)
    V.analyze(j)
    check_preview(j, 2, (H, W), lambda i: np.rot90(shot(i), 1), lambda i: np.rot90(shot(i), 3))


def test_preview_of_a_large_video(tmp_path):
    """A 1024x576 video at 60 fps: the preview's long side is 640 px and it keeps every fourth frame."""
    import av

    rng = np.random.default_rng(2)
    bg = (rng.random((576 // 16, 1024 // 16, 3)) * 160 + 20).astype(np.uint8).repeat(16, 0).repeat(16, 1)
    path = tmp_path / "big.mp4"
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=60, options={"crf": "18", "preset": "ultrafast"})
        s.width, s.height, s.pix_fmt = 1024, 576, "yuv420p"
        for i in range(24):
            a = bg.copy()
            a[200:380, 40 * i:40 * i + 160] = 255
            for p in s.encode(av.VideoFrame.from_ndarray(a, format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    j = job(tmp_path, path)
    V.analyze(j)
    _, kept, got = check_preview(j, 4, (640, 360))
    k = 640 / 1024
    for (_, a), i in zip(got, kept):  # the bar where the frame has it
        cols = np.flatnonzero((V.gray(a[140:220].astype(float)) > 235).mean(0) > 0.5)
        assert cols.mean() == pytest.approx(k * (40 * i + 80), abs=2)


def test_preview_moves_the_frames_of_a_burst(tmp_path):
    """Frames that come in bursts (a bag timed by log time: 5 messages 0.2 ms apart every 150 ms, 33 fps): a kept
    frame less than 2 ms after the kept frame before it moves to 2 ms after that frame, and "moved" says where it
    is. So a seek 1 ms after each kept frame's time in the preview shows that frame and no later one."""
    times = [0.15 * (i // 5) + 0.0002 * (i % 5) for i in range(60)]
    pv = V.Preview(tmp_path / "preview.mp4", {"width": 64, "height": 48, "fps": 33.3})
    with pv:
        for i, t in enumerate(times):
            a = np.zeros((48, 64, 3), np.uint8)
            a[:, i:i + 4] = 255  # a bar at column i
            pv.add(i, t, a)
        rec = pv.close()
    kept = list(range(0, 60, 3))
    assert (rec["stride"], rec["frames"], rec["t0"]) == (3, 20, 0.0)
    at = {k: rec["moved"].get(str(k), times[k]) for k in kept}  # each kept frame's time in the preview
    assert rec["moved"] == pytest.approx({"3": 0.002, "9": 0.1522, "18": 0.452, "24": 0.6022, "33": 0.902, "39": 1.0522,
                                          "48": 1.352, "54": 1.5022}, abs=1e-4)
    _, got = read_preview(tmp_path / "preview.mp4")
    pts = [t for t, _ in got]
    assert pts == pytest.approx([at[k] for k in kept], abs=1e-4) and min(np.diff(pts)) >= 0.002 - 1e-6
    for k in kept:
        j = max(n for n, t in enumerate(pts) if t <= at[k] + 0.001)  # the frame that a browser shows there
        bar = np.flatnonzero(V.gray(got[j][1][20:28].astype(float)).mean(0) > 128)
        assert kept[j] == k and bar.mean() == pytest.approx(k + 1.5, abs=1), k


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


def test_frame_stats_find_the_blurred_frame(tmp_path):
    path = write_video(tmp_path / "v.mp4")
    got = list(V.frames(path))
    lv, luma, clip_hi, clip_lo = np.array([V.frame_stats(V.gray(a)) for _, _, a in got]).T  # as analyze() does
    rel = V.relative(lv, [t for _, t, _ in got])
    assert rel[BLUR] < 0.6 and np.median(rel) == pytest.approx(1, abs=0.1)
    assert lv[BLUR] < 0.5 * np.median(lv)
    assert 60 < luma.mean() < 160 and clip_hi.max() > 0.03 and clip_lo.max() < 0.05
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
    (j / "job.json").write_text(json.dumps({"video": {"path": str(path)}, "mode": "auto",
                                            "source": {"kind": "video", "topic": None}}))
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
    # The preview shows the chosen topic (the other one is upside down), every second message.
    check_preview(j, 2, (W, H), shot, lambda i: shot(i)[::-1])
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
    from harpor.webapp import rosbag

    bag = write_bag(tmp_path / "b.bag", ("/cam/image_rect",), stamp=lambda i, ns: ns if i < N // 2 else ns - 10**9)
    j = job(tmp_path, bag, source={"kind": "rosbag", "topic": "/cam/image_rect"})
    written, write_json = [], V.write_json
    monkeypatch.setattr(V, "write_json", lambda path, data, **kw: written.append(path.name) or write_json(path, data, **kw))
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
    # The preview is turned upright by gravity too. It starts at frame 0, which rgb.mp4 read without its edit
    # list does not time at 0.
    check_preview(J, 1, (240, 320), lambda i: picture(i, (240, 320)), lambda i: picture(i, (240, 320))[::-1, ::-1])
    assert meta["preview"]["t0"] == meta["times"][0] != 0
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


def test_preview_failures_leave_no_file(tmp_path, capsys, monkeypatch):
    """Without the encoder, or when it refuses its options, the analysis goes on without a preview: meta.json says
    why, and no preview (also none from an earlier analysis) or half-written file stays. A decode pass that stops
    deletes the half-written file."""
    path = write_video(tmp_path / "v.mp4")
    j = job(tmp_path, path)
    for name, value, why in (("PREVIEW_CODEC", "no_such_codec", "UnknownCodecError"),
                             ("PREVIEW_OPTIONS", {"preset": "nope"}, "ArgumentError")):
        (j / "frames").mkdir(exist_ok=True)
        (j / "frames" / "preview.mp4").write_bytes(b"an earlier analysis's preview")
        with monkeypatch.context() as m:
            m.setattr(V, name, value)
            meta = V.analyze(j)
        assert meta["preview"]["error"].startswith(why + ": ") and set(meta["preview"]) == {"error"}
        assert json.loads((j / "frames" / "meta.json").read_text())["preview"] == meta["preview"]
        assert f"warning: no preview video ({why}: " in capsys.readouterr().out
        assert sorted(p.name for p in (j / "frames").iterdir()) == ["meta.json", "thumbs"]
        assert meta["candidates"] and len(meta["times"]) == N

    def stops(job, video, extra=None):  # the input ends badly after 5 frames
        probe_input, read = source(job, video, extra)

        def frames(info, **kw):
            for i, t, a in read(info, **kw):
                if i == 5:
                    raise SystemExit("message 5 is broken")
                yield i, t, a
        return probe_input, frames

    source = V.source
    monkeypatch.setattr(V, "source", stops)
    (j / "frames" / "meta.json").unlink()
    with pytest.raises(SystemExit, match="message 5 is broken"):
        V.analyze(j)
    assert sorted(p.name for p in (j / "frames").iterdir()) == ["thumbs"]


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


def test_probe_finds_hlg(tmp_path):
    """A file tagged with the HLG transfer function, as a phone writes it, gives hdr "HLG"."""
    path = write_video(tmp_path / "hlg.mp4", options={
        "color_trc": "arib-std-b67", "x264-params": "transfer=arib-std-b67:colorprim=bt2020:colormatrix=bt2020nc"})
    assert V.probe(path)["hdr"] == "HLG"
