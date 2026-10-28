import dataclasses
import io
import json
import shutil
import sqlite3
import struct
import time
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from scipy.spatial.transform import Rotation

pytest.importorskip("rosbags")
from rosbags.highlevel import AnyReader  # noqa: E402
from rosbags.typesys import Stores, get_types_from_msg, get_typestore  # noqa: E402

from gator.stage_a import lidar_scale  # noqa: E402
from gator.webapp import rosbag as R  # noqa: E402

H, W, N = 48, 64, 6
S0 = 1_749_907_120 * 10**9  # header stamps start here, as in the user's bags
LAG = 50_000_000  # and the log time is 50 ms later
CI, IMG, CIMG = "sensor_msgs/msg/CameraInfo", "sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage"
PC2, TFM = "sensor_msgs/msg/PointCloud2", "tf2_msgs/msg/TFMessage"
LIVOX1, LIVOX2 = "livox_ros_driver/msg/CustomMsg", "livox_ros_driver2/msg/CustomMsg"
PF = {"i1": 1, "u1": 2, "i2": 3, "u2": 4, "i4": 5, "u4": 6, "f4": 7, "f8": 8}  # PointField datatype
MONO16 = [0, 1, 1, 3, 4, 5]  # header stamps of /mono16/image in tenths of a second; one repeats

# Synthetic intrinsics and extrinsics. /tf_static: base_link is the parent of camera_optical_frame and
# livox_frame. Q_OPT turns base_link's body axes (x forward, y left, z up) into optical axes.
K_FRONT, P_FRONT = [[80, 0, 32], [0, 80, 24], [0, 0, 1]], [[70, 0, 31, 0], [0, 72, 23, 0], [0, 0, 1, 0]]
K_RAW, P_RECT = [[90, 0, 32], [0, 91, 24], [0, 0, 1]], [[60, 0, 30, 0], [0, 61, 22, 0], [0, 0, 1, 0]]
R_RECT = Rotation.from_euler("y", 2, degrees=True).as_matrix()  # a stereo rectification
Q_OPT = (-0.5, 0.5, -0.5, 0.5)
T_BASE_CAM = ((0.1, 0.0, 0.2), Q_OPT)
T_BASE_LIVOX = ((0.2, 0.0, 0.3), (0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)))  # turned 90 deg left
LIVOX_BASE = 1234  # a Livox message's timebase is its stamp plus this (ns)
POINTS_LAYOUT = [("x", "f4", 0), ("y", "f4", 4), ("z", "f4", 8), ("intensity", "f4", 16), ("t", "u4", 20)]  # 32 B
RADAR_LAYOUT = [("x", "f8", 0), ("y", "f8", 8), ("z", "f8", 16), ("time", "f4", 24)]  # 28 B


def matrix(t, q):
    """4x4 of a translation and a quaternion (x, y, z, w)."""
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = Rotation.from_quat(q).as_matrix(), t
    return T


def points_cols(i):
    """/points message i: 10 points in 2 rows, 1 ms apart; point 3 is zero and point 7 has a NaN."""
    k = np.arange(10)
    x, y, z = 1.0 + k + i, 0.5 * k, 0.1 * k
    x[3] = y[3] = z[3] = 0
    x[7] = np.nan
    return {"x": x, "y": y, "z": z, "intensity": 100.0 + k, "t": k * 1_000_000}


def radar_cols(i):
    """/radar/points message i: 4 points, big endian, with time offsets in s from -0.05."""
    k = np.arange(4)
    return {"x": 3.0 + k + i, "y": -1.0 * k, "z": 0.25 * k, "time": -0.05 + 0.02 * k}


def livox_points(i):
    """(offset_time, x, y, z, tag) of Livox message i: 3 points kept (0, 1, 5), two by tag, one zero."""
    return [(0, 2.0 + i, 0, 0, 0x00), (100_000, 2, 1, 0, 0x10), (200_000, 2, 2, 0, 0x20), (300_000, 2, 3, 0, 0x30),
            (400_000, 0, 0, 0, 0x00), (500_000, 2, 5, 1, 0x11)]


def picture(i):
    """A texture of 2x2 cells with a white square that moves right; every 2x2 cell has one color, so a
    Bayer mosaic of it debayers exactly."""
    rng = np.random.default_rng(0)
    a = (rng.random((H // 2, W // 2, 3)) * 200).astype(np.uint8).repeat(2, 0).repeat(2, 1)
    a[10:30, 4 + 2 * i:24 + 2 * i] = 255
    return a


def luma(a):
    return np.round(a @ np.array([0.299, 0.587, 0.114])).astype(np.uint8)


class Msgs:
    """Messages of one typestore: ROS1 (Noetic, D and K, livox_ros_driver) or ROS2 (Humble, d and k,
    livox_ros_driver2). The Livox types come from rosbag.LIVOX_MSGS."""

    def __init__(self, ros1):
        self.ros1, self.ts = ros1, get_typestore(Stores.ROS1_NOETIC if ros1 else Stores.ROS2_HUMBLE)
        self.livox = LIVOX1 if ros1 else LIVOX2
        types = {}
        for name, text in R.LIVOX_MSGS.items():
            if name.split("/")[0] == self.livox.split("/")[0]:
                types.update(get_types_from_msg(text, name))
        if ros1:  # Noetic's store has no tf2_msgs
            types.update(get_types_from_msg("geometry_msgs/TransformStamped[] transforms", TFM))
        self.ts.register(types)
        self.T = self.ts.types

    def header(self, ns, frame="cam"):
        stamp = self.T["builtin_interfaces/msg/Time"](sec=ns // 10**9, nanosec=ns % 10**9)
        return self.T["std_msgs/msg/Header"](**({"seq": 0} if self.ros1 else {}), stamp=stamp, frame_id=frame)

    def image(self, ns, a, encoding, pad=0, big=False, frame="cam"):
        """An Image of a (uint8, or uint16 written big or little endian) with pad bytes after each row."""
        h, w = a.shape[:2]
        rows = np.ascontiguousarray(a.astype(">u2" if big else "<u2") if a.dtype == np.uint16 else a)
        rows = rows.view(np.uint8).reshape(h, -1)
        rows = np.concatenate([rows, np.full((h, pad), 7, np.uint8)], 1) if pad else rows
        return self.T[IMG](header=self.header(ns, frame), height=h, width=w, encoding=encoding, is_bigendian=int(big),
                           step=rows.shape[1], data=np.ascontiguousarray(rows).reshape(-1))

    def cloud(self, ns, cols, layout, step, height=1, big=False, row_pad=0, frame="livox_frame"):
        """A PointCloud2 of the columns cols {name: values}, placed by layout [(name, numpy type, offset)] in
        points of step bytes, in height rows with row_pad bytes after each, big or little endian. Every
        padding byte is 0xAB."""
        n = len(next(iter(cols.values())))
        end = ">" if big else "<"
        dt = np.dtype({"names": [x[0] for x in layout], "formats": [end + x[1] for x in layout],
                       "offsets": [x[2] for x in layout], "itemsize": step})
        pts = np.frombuffer(b"\xab" * (n * step), dt).copy()
        for name, _, _ in layout:
            pts[name] = cols[name]
        rows = pts.view(np.uint8).reshape(height, -1)
        rows = np.concatenate([rows, np.full((height, row_pad), 0xAB, np.uint8)], 1)
        fields = [self.T["sensor_msgs/msg/PointField"](name=name, offset=off, datatype=PF[typ], count=1)
                  for name, typ, off in layout]
        return self.T[PC2](header=self.header(ns, frame), height=height, width=n // height, fields=fields,
                           is_bigendian=big, point_step=step, row_step=rows.shape[1], data=rows.reshape(-1),
                           is_dense=False)

    def livox_msg(self, ns, timebase, points, frame="livox_frame"):
        """A Livox CustomMsg of points [(offset_time, x, y, z, tag)]."""
        P = self.T[self.livox.replace("CustomMsg", "CustomPoint")]
        pts = [P(offset_time=o, x=x, y=y, z=z, reflectivity=10, tag=tag, line=1) for o, x, y, z, tag in points]
        return self.T[self.livox](header=self.header(ns, frame), timebase=timebase, point_num=len(pts), lidar_id=0,
                                  rsvd=np.zeros(3, np.uint8), points=pts)

    def tf(self, ns, links):
        """A TFMessage of links [(parent, child, translation, quaternion x, y, z, w)], all stamped ns."""
        T = self.T

        def one(parent, child, t, q):
            tf = T["geometry_msgs/msg/Transform"](translation=T["geometry_msgs/msg/Vector3"](x=t[0], y=t[1], z=t[2]),
                                                  rotation=T["geometry_msgs/msg/Quaternion"](x=q[0], y=q[1], z=q[2], w=q[3]))
            return T["geometry_msgs/msg/TransformStamped"](header=self.header(ns, parent), child_frame_id=child, transform=tf)

        return T[TFM](transforms=[one(*x) for x in links])

    def compressed(self, ns, a, fmt, codec):
        buf = io.BytesIO()
        Image.fromarray(a).save(buf, codec, **({"quality": 95, "subsampling": 0} if codec == "JPEG" else {}))
        return self.T[CIMG](header=self.header(ns), format=fmt, data=np.frombuffer(buf.getvalue(), np.uint8))

    def info(self, ns, d, k=np.eye(3), p=np.eye(3, 4), model="plumb_bob", r=np.eye(3), size=(W, H), binning=(0, 0),
             roi=(0, 0, 0, 0)):
        """A CameraInfo; roi is (x_offset, y_offset, width, height)."""
        T, d, k, p, r = (self.T, np.array(d, float), np.ravel(k).astype(float), np.ravel(p).astype(float),
                         np.ravel(r).astype(float))
        kw = {"D": d, "K": k, "R": r, "P": p} if self.ros1 else {"d": d, "k": k, "r": r, "p": p}
        roi = T["sensor_msgs/msg/RegionOfInterest"](x_offset=roi[0], y_offset=roi[1], width=roi[2], height=roi[3],
                                                    do_rectify=False)
        return T[CI](header=self.header(ns), height=size[1], width=size[0], distortion_model=model,
                     binning_x=binning[0], binning_y=binning[1], roi=roi, **kw)


def contents(m):
    """[(topic, msgtype, log time ns, message)] of the test bag: camera topics, LiDAR topics (PointCloud2
    and Livox), /tf_static and /tf."""
    out = [("/raw/camera_info", CI, S0, m.info(S0, [0.1, -0.02, 0, 0, 0.001], K_RAW)),
           ("/png/camera_info", CI, S0, m.info(S0, [0, 0, 0, 0, 0])),
           ("/cam_rect/camera_info", CI, S0, m.info(S0, [0.3, 0, 0, 0, 0], p=P_RECT, r=R_RECT)),  # the name says rect
           ("/front/camera_info", CI, S0, m.info(S0, [0, 0, 0, 0, 0], K_FRONT, P_FRONT, r=R_RECT)),  # all-zero D
           ("/note", "std_msgs/msg/String", S0, m.T["std_msgs/msg/String"](data="not an image")),
           ("/tf_static", TFM, S0, m.tf(S0, [("base_link", "camera_optical_frame", *T_BASE_CAM),
                                             ("/base_link", "livox_frame", *T_BASE_LIVOX)]))]
    for i in range(N + 2):
        ns, a = S0 + i * 10**8, picture(i)
        out += [("/png/image/compressed", CIMG, ns + LAG, m.compressed(ns, a, "png", "PNG")),
                ("/points", PC2, ns + LAG, m.cloud(ns, points_cols(i), POINTS_LAYOUT, 32, height=2, row_pad=8)),
                ("/livox/lidar", m.livox, ns + LAG, m.livox_msg(ns, ns + LIVOX_BASE, livox_points(i))),
                ("/radar/points", PC2, ns + LAG, m.cloud(ns, radar_cols(i), RADAR_LAYOUT, 28, big=True, frame="radar")),
                ("/flat", PC2, ns + LAG, m.cloud(ns, {"intensity": np.ones(3)}, [("intensity", "f4", 0)], 4))]
        if i % 2 == 0:  # the radar is linked by /tf only, and it moves
            out.append(("/tf", TFM, ns + LAG, m.tf(ns, [("base_link", "radar", (0, 0.5 + 0.1 * i, 0), (0, 0, 0, 1))])))
        if i >= N:
            continue
        out.append(("/front/image_raw", IMG, ns + LAG, m.image(ns, a, "rgb8", frame="camera_optical_frame")))
        g = luma(a)
        mosaic = a[..., 1].copy()  # rggb: green, then red and blue in their places
        mosaic[0::2, 0::2], mosaic[1::2, 1::2] = a[0::2, 0::2, 0], a[1::2, 1::2, 2]
        out += [("/cam_rect/image_rect/compressed", CIMG, ns + LAG, m.compressed(ns, a, "rgb8; jpeg compressed bgr8", "JPEG")),
                ("/raw/image_raw", IMG, ns + LAG, m.image(ns, a, "rgb8", pad=5)),
                ("/bgr/image", IMG, ns + LAG, m.image(ns, a[..., ::-1], "bgr8")),
                ("/mono/image", IMG, S0 + 2 * i * 10**8, m.image(0, g, "mono8")),  # no stamps: log time counts
                ("/mono16/image", IMG, ns + LAG,
                 m.image(S0 + MONO16[i] * 10**8, g.astype(np.uint16) * 257, "mono16", pad=2, big=True)),
                ("/bayer/image", IMG, ns + LAG, m.image(ns, mosaic, "bayer_rggb8", pad=3)),
                ("/depth/image", IMG, ns + LAG, m.image(ns, g.astype(np.uint16) * 4, "16UC1")),
                ("/yuv/image", IMG, ns + LAG, m.image(ns, np.zeros((H, W, 2), np.uint8), "yuv422"))]
    return sorted(out, key=lambda x: x[2])


def write_bag(path, storage="mcap", make=contents):
    """The test bag (make(Msgs) gives its messages, see contents) as a ROS1 .bag (path ends in .bag) or
    a ROS2 folder (storage mcap or sqlite3). A message given as bytes is written as it is; None only
    adds the topic."""
    ros1 = path.suffix == ".bag"
    m = Msgs(ros1)
    if ros1:
        from rosbags.rosbag1 import Writer

        writer = Writer(path)
    else:
        from rosbags.rosbag2 import StoragePlugin, Writer

        writer = Writer(path, version=9, storage_plugin=StoragePlugin[storage.upper()])
    with writer as w:
        conns = {}
        for topic, typ, t, msg in make(m):
            if topic not in conns:
                conns[topic] = w.add_connection(topic, typ, typestore=m.ts)
            if msg is None:
                continue
            raw = msg if isinstance(msg, bytes) else m.ts.serialize_ros1(msg, typ) if ros1 else m.ts.serialize_cdr(msg, typ)
            w.write(conns[topic], t, raw)
    return path


def strip_definitions(db3):
    """Remove a .db3's message definitions, as a Humble-era bag has none."""
    with sqlite3.connect(db3) as c:
        c.execute("DELETE FROM message_definitions")
    c.close()


@pytest.fixture(scope="module", params=["ros1", "mcap_dir", "mcap_file", "db3_dir", "db3_file"])
def bag(request, tmp_path_factory):
    """The test bag in every form. The single .db3 file has no message definitions (a Humble-era bag),
    so it is read with rosbag.typestore()."""
    d = tmp_path_factory.mktemp(request.param)
    if request.param == "ros1":
        return write_bag(d / "test.bag")
    kind, form = request.param.split("_")
    p = write_bag(d / "b", "mcap" if kind == "mcap" else "sqlite3")
    if form == "file" and kind == "db3":
        strip_definitions(p / "b.db3")
    return p if form == "dir" else p / f"b.{kind}"


def test_info_topic():
    assert R.info_topic("/cam_b/image_rect/compressed") == "/cam_b/camera_info"
    assert R.info_topic("/a/b/image_raw") == "/a/b/camera_info" and R.info_topic("cam/image") == "/cam/camera_info"


def test_image_topics(bag):
    previews = {}
    out = R.image_topics(bag, previews)
    got = {t["topic"]: t for t in out["topics"]}
    assert [t["topic"] for t in out["topics"]] == [
        "/png/image/compressed", "/cam_rect/image_rect/compressed", "/front/image_raw",  # rectified first, then by count
        "/bayer/image", "/bgr/image", "/mono/image", "/mono16/image", "/raw/image_raw"]
    assert [(s["topic"], s["type"]) for s in out["skipped"]] == [("/depth/image", "Image"), ("/yuv/image", "Image")]
    assert "16UC1" in out["skipped"][0]["why"] and "yuv422 is not supported" in out["skipped"][1]["why"]
    png, rect, raw, mono = (got[t] for t in ("/png/image/compressed", "/cam_rect/image_rect/compressed",
                                              "/raw/image_raw", "/mono/image"))
    assert png == {"topic": "/png/image/compressed", "type": "CompressedImage", "count": N + 2, "duration_s": 0.8,
                   "rate_hz": 10.0, "width": W, "height": H, "format": "png", "rectified": True,
                   "camera_info": "/png/camera_info", "distortion": {"model": "plumb_bob", "max_abs_d": 0.0},
                   "frame_id": "cam", "camera_axes": "body", "camera_axes_from": "default",
                   "tf_links": {"/livox/lidar": None, "/points": None, "/radar/points": None}}
    assert rect["format"] == "jpeg" and rect["rectified"] is True and rect["distortion"]["max_abs_d"] == 0.3
    assert raw["rectified"] is False and raw["distortion"]["max_abs_d"] == 0.1 and raw["camera_info"] == "/raw/camera_info"
    assert raw["format"] == "rgb8" and raw["type"] == "Image" and (raw["duration_s"], raw["rate_hz"]) == (0.6, 10.0)
    assert mono["rectified"] is None and mono["camera_info"] is None and mono["distortion"] is None
    assert (mono["duration_s"], mono["rate_hz"]) == (1.2, 5.0)  # log time
    assert got["/mono16/image"]["format"] == "mono16" and got["/bayer/image"]["format"] == "bayer_rggb8"
    assert set(previews) == set(got) and all(a.shape == (H, W, 3) and a.dtype == np.uint8 for a in previews.values())
    assert np.array_equal(previews["/png/image/compressed"], picture((N + 2) // 2))  # the middle message


def test_frames_decode_every_encoding(bag):
    exact = {"/png/image/compressed": picture, "/raw/image_raw": picture, "/bgr/image": picture,
             "/bayer/image": picture, "/mono/image": lambda i: np.repeat(luma(picture(i))[..., None], 3, 2),
             "/mono16/image": lambda i: np.repeat(luma(picture(i))[..., None], 3, 2)}
    for topic, want in exact.items():
        got = list(R.frames(bag, topic))
        assert [i for i, _, _ in got] == list(range(len(got))), topic
        for i, _, a in got:
            assert a.dtype == np.uint8 and np.array_equal(a, want(i)), (topic, i)
    got = list(R.frames(bag, "/cam_rect/image_rect/compressed"))
    assert len(got) == N and all(np.abs(a.astype(float) - picture(i)).mean() < 4 for i, _, a in got)
    # Times: header stamps from the first message (the 50 ms log lag drops out), else log time.
    assert [t for _, t, _ in got] == pytest.approx([0.1 * i for i in range(N)])
    assert [t for _, t, _ in R.frames(bag, "/mono/image")] == pytest.approx([0.2 * i for i in range(N)])
    # mono16 repeats a stamp, so the whole topic is timed by log time.
    assert [t for _, t, _ in R.frames(bag, "/mono16/image")] == pytest.approx([0.1 * i for i in range(N)])
    for topic in ("/cam_rect/image_rect/compressed", "/raw/image_raw", "/bayer/image"):
        _, _, small = next(R.frames(bag, topic, width=32))
        assert small.shape == (24, 32, 3)


def test_frames_keep_stops_early(bag, monkeypatch):
    pulled = []
    messages = AnyReader.messages

    def counting(self, *a, **kw):
        for x in messages(self, *a, **kw):
            pulled.append(x[0].topic)
            yield x

    monkeypatch.setattr(AnyReader, "messages", counting)
    got = list(R.frames(bag, "/raw/image_raw", keep={1, 3}, clock="header"))
    assert [i for i, _, _ in got] == [1, 3] and [t for _, t, _ in got] == pytest.approx([0.1, 0.3])
    assert np.array_equal(got[1][2], picture(3)) and len(pulled) == 5  # message 4 ends it
    assert list(R.frames(bag, "/raw/image_raw", keep=set(), clock="header")) == []
    pulled.clear()  # without the clock, frames reads the topic's stamps first
    assert [t for _, t, _ in R.frames(bag, "/raw/image_raw", keep={1, 3})] == pytest.approx([0.1, 0.3])
    assert len(pulled) == N + 5


def test_probe(bag):
    info = R.probe(bag, "/cam_rect/image_rect/compressed")
    assert info == {"n_frames": N, "duration_s": 0.6, "fps": 10.0, "width": W, "height": H, "rotation": 0,
                    "codec": "CompressedImage jpeg", "hdr": None, "bit_depth": 8, "clock": "header",
                    "stamp_warnings": 0}
    m16 = R.probe(bag, "/mono16/image")
    assert (m16["codec"], m16["bit_depth"], m16["clock"], m16["stamp_warnings"], m16["duration_s"]) == \
        ("Image mono16", 16, "log", 1, 0.6)
    mono = R.probe(bag, "/mono/image")  # no stamps at all: log time, and nothing to warn about
    assert (mono["fps"], mono["clock"], mono["stamp_warnings"]) == (5.0, "log", 0)
    assert R.probe(bag, "/png/image/compressed")["n_frames"] == N + 2


def test_bad_topics_and_files(bag, tmp_path):
    for topic, msg in (("/nope", "no topic /nope"), ("/note", "not a camera image topic")):
        with pytest.raises(SystemExit, match=msg):
            R.probe(bag, topic)
    with pytest.raises(SystemExit, match="yuv422"):
        next(R.frames(bag, "/yuv/image"))
    with pytest.raises(SystemExit, match="depth"):
        R.probe(bag, "/depth/image")
    for name in ("x.bag", "x.mcap", "x.db3"):
        (tmp_path / name).write_bytes(b"not a bag" * 100)
        with pytest.raises(SystemExit, match="cannot read"):
            R.image_topics(tmp_path / name)
    with pytest.raises(SystemExit, match="not a LiDAR topic"):
        next(R.scans(bag, "/png/image/compressed", [(S0, S0 + 10**9)]))
    with pytest.raises(SystemExit, match="no topic /nope"):
        next(R.scans(bag, "/nope", []))
    with pytest.raises(SystemExit, match="not a camera image topic"):
        R.camera_model(bag, "/points")


def test_unpack_layouts():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 256, (5, 7, 4), dtype=np.uint8)

    def msg(data, encoding, w=7, step=None, big=0):
        data = np.ascontiguousarray(data).view(np.uint8).reshape(5, -1)
        return SimpleNamespace(height=5, width=w, encoding=encoding, step=step or data.shape[1], is_bigendian=big,
                               data=data.reshape(-1))

    assert np.array_equal(R.unpack(msg(a, "rgba8")), a[..., :3])
    assert np.array_equal(R.unpack(msg(a[..., [2, 1, 0, 3]], "bgra8")), a[..., :3])
    g16 = rng.integers(0, 1000, (5, 7)).astype("<u2")
    assert np.array_equal(R.unpack(msg(g16, "mono16"))[..., 0], np.round(g16.astype(float) * 255 / g16.max()).astype(np.uint8))
    m = a[..., 0]  # odd size: the last row and column copy their neighbours
    for pattern in ("bggr", "gbrg", "grbg"):
        rgb = R.unpack(msg(m, f"bayer_{pattern}8"))
        cell = m[:2, :2].ravel()
        want = [cell[pattern.index("r")], (int(cell[pattern.index("g")]) + int(cell[pattern.rindex("g")]) + 1) // 2,
                cell[pattern.index("b")]]
        assert rgb.shape == (5, 7, 3) and list(rgb[0, 0]) == list(rgb[1, 1]) == want
        assert np.array_equal(rgb[4], rgb[3]) and np.array_equal(rgb[:, 6], rgb[:, 5])
    with pytest.raises(SystemExit, match="does not fit"):
        R.unpack(msg(a, "rgba8", step=40))


def test_upper_case_suffix(tmp_path):
    p = shutil.copy(write_bag(tmp_path / "t.bag"), tmp_path / "UP.BAG")
    assert [t["topic"] for t in R.image_topics(p)["topics"]][:2] == ["/png/image/compressed", "/cam_rect/image_rect/compressed"]
    assert R.probe(p, "/raw/image_raw")["n_frames"] == N


def raw_cimg(m, ns, data):
    return m.T[CIMG](header=m.header(ns), format="jpeg", data=np.frombuffer(data, np.uint8))


def bomb():
    """A JPEG whose header says 65000x65000 pixels, which pillow refuses as a decompression bomb."""
    buf = io.BytesIO()
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(buf, "JPEG")
    a = buf.getvalue()
    i = a.find(b"\xff\xc0")  # SOF0: marker, length, precision, height, width
    return a[:i + 5] + struct.pack(">HH", 65000, 65000) + a[i + 9:]


def test_broken_messages_and_topics(tmp_path, capsys):
    """A 0x0 Image topic and topics whose messages pillow cannot read are skipped and do not stop the
    listing; one bad message inside a good topic gives a black frame, and the preview takes another."""
    garbage = b"\xff\xd8garbage" * 10

    def make(m):
        out = []
        for i in range(7):  # 0 and 3 (the middle) are corrupt, 5 is a bomb
            ns = S0 + i * 10**8
            msg = raw_cimg(m, ns, garbage) if i in (0, 3) else raw_cimg(m, ns, bomb()) if i == 5 else \
                m.compressed(ns, picture(i), "png", "PNG")
            out.append(("/good/image/compressed", CIMG, ns, msg))
        for i in range(2):
            ns = S0 + i * 10**8
            out += [("/bomb/image/compressed", CIMG, ns, raw_cimg(m, ns, bomb())),
                    ("/dead/image/compressed", CIMG, ns, raw_cimg(m, ns, garbage)),
                    ("/empty/image", IMG, ns, m.T[IMG](header=m.header(ns), height=0, width=0, encoding="rgb8",
                                                      is_bigendian=0, step=0, data=np.zeros(0, np.uint8)))]
        return sorted(out, key=lambda x: x[2])

    bag = write_bag(tmp_path / "broken.bag", make=make)
    previews = {}
    out = R.image_topics(bag, previews)
    assert [t["topic"] for t in out["topics"]] == ["/good/image/compressed"]
    assert (out["topics"][0]["width"], out["topics"][0]["height"], out["topics"][0]["count"]) == (W, H, 7)
    assert np.array_equal(previews["/good/image/compressed"], picture(6))  # the middle and the first are corrupt
    why = {s["topic"]: s["why"] for s in out["skipped"]}
    assert list(why) == ["/bomb/image/compressed", "/dead/image/compressed", "/empty/image"]
    assert "decompression bomb" in why["/bomb/image/compressed"] and why["/dead/image/compressed"].startswith("cannot decode")
    assert why["/empty/image"] == "empty image (0x0)"
    info = R.probe(bag, "/good/image/compressed")
    assert (info["n_frames"], info["width"], info["height"]) == (7, W, H)
    capsys.readouterr()
    got = list(R.frames(bag, "/good/image/compressed"))
    assert [i for i, _, _ in got] == list(range(7)) and [t for _, t, _ in got] == pytest.approx([0.1 * i for i in range(7)])
    for i, _, a in got:
        assert np.array_equal(a, np.zeros((H, W, 3), np.uint8) if i in (0, 3, 5) else picture(i)), i
    assert capsys.readouterr().out.count("it counts as a black frame") == 3
    assert [i for i, _, _ in R.frames(bag, "/good/image/compressed", keep={3, 4})] == [3, 4]
    with pytest.raises(SystemExit, match="cannot decode message 0 of /dead"):
        list(R.frames(bag, "/dead/image/compressed"))
    with pytest.raises(SystemExit, match="cannot decode any message of /bomb"):
        R.probe(bag, "/bomb/image/compressed")
    with pytest.raises(SystemExit, match="empty image"):
        R.probe(bag, "/empty/image")


@pytest.mark.parametrize("stamps,warnings", [
    (lambda i: S0, 7), (lambda i: 0 if i == 7 else S0 + i * 10**8, 1),
    (lambda i: S0 + i * 10**8 if i < 4 else 10**9 + i * 10**8, 1)], ids=["constant", "late zero", "clock reset"])
def test_bad_stamps_use_log_time(tmp_path, stamps, warnings):
    """Header stamps that repeat, drop to zero or jump back time the whole topic by log time."""
    def make(m):
        return [("/cam/image_rect/compressed", CIMG, S0 + i * 10**8 + LAG, m.compressed(stamps(i), picture(i), "png", "PNG"))
                for i in range(8)]

    bag = write_bag(tmp_path / "b.bag", make=make)
    [t] = R.image_topics(bag)["topics"]
    assert (t["duration_s"], t["rate_hz"]) == (0.8, 10.0)
    info = R.probe(bag, "/cam/image_rect/compressed")
    assert (info["clock"], info["stamp_warnings"], info["duration_s"], info["fps"]) == ("log", warnings, 0.8, 10.0)
    for clock in (None, "log"):
        times = [x for _, x, _ in R.frames(bag, "/cam/image_rect/compressed", clock=clock)]
        assert times == pytest.approx([0.1 * i for i in range(8)]), clock


def test_uncalibrated_camera_info(tmp_path):
    """camera_info_manager without a calibration sends an empty D and a zero K: that says nothing about
    the lens, so the topic is not called rectified."""
    def make(m):
        roi = m.T["sensor_msgs/msg/RegionOfInterest"](x_offset=0, y_offset=0, height=0, width=0, do_rectify=False)
        out = []
        for ns_, d in (("/usb_cam", []), ("/zero_k", [0, 0, 0, 0, 0])):
            info = m.T[CI](header=m.header(S0), height=0, width=0, distortion_model="", binning_x=0, binning_y=0, roi=roi,
                           D=np.array(d, float), K=np.zeros(9), R=np.zeros(9), P=np.zeros(12))
            out += [(f"{ns_}/camera_info", CI, S0, info)] + [
                (f"{ns_}/image_raw", IMG, S0 + i * 10**8, m.image(S0 + i * 10**8, picture(i), "rgb8")) for i in range(3)]
        return sorted(out, key=lambda x: x[2])

    bag = write_bag(tmp_path / "u.bag", make=make)
    got = {t["topic"]: t for t in R.image_topics(bag)["topics"]}
    for topic in ("/usb_cam/image_raw", "/zero_k/image_raw"):
        t = got[topic]
        assert t["rectified"] is None and t["camera_info"] == topic.replace("image_raw", "camera_info"), topic
        assert t["distortion"] == {"model": "", "max_abs_d": 0.0}, topic
        assert R.camera_model(bag, topic) is None, topic  # no intrinsics in it


def test_lidar_and_tf_listing(bag):
    """The topics pass lists the LiDAR topics from one message each, the TF links and each camera
    topic's frame; the result stays JSON."""
    out = R.image_topics(bag)
    json.dumps(out)
    livox = LIVOX1 if bag.suffix == ".bag" else LIVOX2
    common = {"point_time": True, "count": N + 2, "rate_hz": 10.0}
    assert out["lidar"] == [
        {"topic": "/livox/lidar", "type": "CustomMsg", "msgtype": livox, "frame_id": "livox_frame", "points": 3,
         "fields": list(R.LIVOX_FIELDS), **common},
        {"topic": "/points", "type": "PointCloud2", "msgtype": PC2, "frame_id": "livox_frame", "points": 8,
         "fields": ["x", "y", "z", "intensity", "t"], **common},
        {"topic": "/radar/points", "type": "PointCloud2", "msgtype": PC2, "frame_id": "radar", "points": 4,
         "fields": ["x", "y", "z", "time"], **common}]
    assert out["lidar_skipped"] == [{"topic": "/flat", "type": "PointCloud2",
                                     "why": "no float x, y and z fields (fields: intensity)"}]
    assert [(e["parent"], e["child"], e["static"]) for e in out["tf"]] == [
        ("base_link", "camera_optical_frame", True), ("base_link", "livox_frame", True), ("base_link", "radar", False)]
    assert np.allclose(out["tf"][0]["T_parent_child"], matrix(*T_BASE_CAM), atol=1e-6)
    assert np.allclose(out["tf"][2]["T_parent_child"], matrix((0, 0.5, 0), (0, 0, 0, 1)))  # the first /tf value
    got = {t["topic"]: t for t in out["topics"]}
    front = got["/front/image_raw"]
    assert (front["frame_id"], front["camera_axes"], front["camera_axes_from"], front["rectified"]) == (
        "camera_optical_frame", "optical", "name", True)
    assert front["tf_links"] == {"/livox/lidar": "static", "/points": "static", "/radar/points": "dynamic"}
    assert all(t["tf_links"] == dict.fromkeys(front["tf_links"]) for k, t in got.items() if k != "/front/image_raw")

    with R.open_bag(bag) as r:
        edges = R.transforms(r)
        got = R.first_messages(r, R.lidar_connections(r, "/points"), 3, keep=1)  # as the rate is timed
    assert [x[1] for x in got] == [S0 + i * 10**8 + LAG for i in range(3)]
    assert [x[2] is None for x in got] == [False, True, True]  # only the first message's data is kept
    assert len(edges) == 2 + 4 and sum(e[3] for e in edges) == 2
    ext = R.camera_to_lidar("camera_optical_frame", "livox_frame", edges)
    assert ext["chain"] == ["camera_optical_frame", "base_link", "livox_frame"]
    assert (ext["link"], ext["camera_axes"]) == ("static", "optical")
    assert np.allclose(ext["T_cam_lidar"], np.linalg.inv(matrix(*T_BASE_CAM)) @ matrix(*T_BASE_LIVOX))
    # A point 2 m ahead of the camera (along base_link's x) is 2 m along the optical axis.
    ahead = np.linalg.inv(matrix(*T_BASE_LIVOX)) @ [2.1, 0, 0.2, 1]
    assert np.allclose(ext["T_cam_lidar"] @ ahead, [0, 0, 2, 1])
    T, frames, link = R.tf_chain(edges, "base_link", "radar", t_ns=S0 + 4.2 * 10**8)  # the /tf value at 0.4 s
    assert (frames, link) == (["base_link", "radar"], "dynamic") and np.allclose(T[:3, 3], [0, 0.9, 0])


def test_tf_listing_reads_all_of_tf(tmp_path, monkeypatch):
    """The topics pass reads every /tf message, in its one pass over the camera topics, and lists each link
    once; transforms reads only the given windows. A zero quaternion and a link to its own frame are left
    out; a zero stamp takes the log time."""
    def make(m):
        out = [("/cam/image_rect", IMG, S0, m.image(S0, picture(0), "rgb8", frame="cam_optical"))]
        out += [("/tf", TFM, S0 + k * 10**9, m.tf(S0 + k * 10**9, [("base", f"f{k}", (k, 0, 0), (0, 0, 0, 1))]))
                for k in range(8)]
        out.append(("/tf_static", TFM, S0 + 7 * 10**9, m.tf(0, [
            ("base", "cam_optical", (0, 0, 1), (0, 0, 0, 1)), ("base", "bad", (0, 0, 0), (0, 0, 0, 0)),
            ("base", "/base", (0, 0, 0), (0, 0, 0, 1))])))
        out.append(("/robot/tf_static", TFM, S0, m.tf(S0, [("base", "ns_lidar", (0, 1, 0), (0, 0, 0, 1))])))  # namespaced
        return sorted(out, key=lambda x: x[2])

    bag = write_bag(tmp_path / "tf.bag", make=make)
    passes, messages = [], AnyReader.messages
    monkeypatch.setattr(AnyReader, "messages", lambda self, connections=(), **kw: passes.append(
        sorted({c.topic for c in connections})) or messages(self, connections=connections, **kw))
    out = R.image_topics(bag)
    assert [(e["child"], e["static"]) for e in out["tf"]] == [("cam_optical", True), ("ns_lidar", True)] + [
        (f"f{k}", False) for k in range(8)]
    assert out["topics"][0]["tf_links"] == {} and out["lidar"] == out["lidar_skipped"] == []
    assert passes == [["/cam/image_rect", "/robot/tf_static", "/tf", "/tf_static"]]  # the bag is read once
    with R.open_bag(bag) as r:
        edges = R.transforms(r, [(S0 + 6 * 10**9, S0 + 7 * 10**9 + 1)])
        assert [(e[1], e[3], e[4]) for e in edges] == [("ns_lidar", True, S0), ("cam_optical", True, S0 + 7 * 10**9),
                                                       ("f6", False, S0 + 6 * 10**9), ("f7", False, S0 + 7 * 10**9)]
        assert len(R.transforms(r)) == 10 and [e[1] for e in R.transforms(r, [])] == ["ns_lidar", "cam_optical"]


def test_lidar_problems_never_fail_the_listing(tmp_path, capsys):
    """A cloud that does not fit its data, integer coordinates, a topic without messages and TF messages
    that cannot be decoded are reported; the camera topics are listed as before."""
    def make(m):
        good = m.cloud(S0, points_cols(0), POINTS_LAYOUT, 32, height=2, row_pad=8)
        return [("/cam/image_rect", IMG, S0, m.image(S0, picture(0), "rgb8", frame="")),
                ("/short", PC2, S0, dataclasses.replace(good, data=good.data[:40])),
                ("/ints", PC2, S0, m.cloud(S0, {n: np.ones(2) for n in "xyz"},
                                           [("x", "i2", 0), ("y", "i2", 2), ("z", "i2", 4)], 6)),
                ("/empty", m.livox, S0, None),
                ("/tf_static", TFM, S0, b"\x05\x00\x00\x00garbage"), ("/tf", TFM, S0, b"\xff\xff\xff\x7f")]

    for path, storage in ((tmp_path / "p.bag", "mcap"), (tmp_path / "m", "mcap"), (tmp_path / "s", "sqlite3")):
        out = R.image_topics(write_bag(path, storage, make=make))
        assert [t["topic"] for t in out["topics"]] == ["/cam/image_rect"] and out["topics"][0]["tf_links"] == {}
        assert (out["topics"][0]["frame_id"], out["topics"][0]["camera_axes"]) == ("", None)  # no frame: no axes
        assert out["lidar"] == [] and out["tf"] == []
        assert out["lidar_skipped"] == [
            {"topic": "/empty", "type": "CustomMsg", "why": "no messages"},
            {"topic": "/ints", "type": "PointCloud2", "why": "no float x, y and z fields (fields: x, y, z)"},
            {"topic": "/short", "type": "PointCloud2",
             "why": "cannot decode: a 5x2 cloud of 32-byte points with row_step 168 does not fit in 40 bytes"}]
        assert capsys.readouterr().out.count("warning: cannot decode the /tf") == 2


def test_decode_pointcloud2():
    m, ns = Msgs(False), S0 + 5
    cols, keep = points_cols(0), [0, 1, 2, 4, 5, 6, 8, 9]  # 3 is zero, 7 has a NaN
    xyz, t, head, frame = R.decode_pointcloud2(m.cloud(ns, cols, POINTS_LAYOUT, 32, height=2, row_pad=8,
                                                       frame="/livox_frame"))
    assert xyz.dtype == np.float32 and t.dtype == np.int64 and (head, frame) == (ns, "livox_frame")
    assert np.array_equal(xyz, np.stack([cols[k] for k in "xyz"], 1)[keep].astype(np.float32))
    assert t.tolist() == [ns + k * 1_000_000 for k in keep]  # "t": ns after the stamp

    rc = radar_cols(2)  # float64 coordinates and a float32 "time" in s, big and little endian
    for big in (True, False):
        xyz, t, _, frame = R.decode_pointcloud2(m.cloud(ns, rc, RADAR_LAYOUT, 28, big=big, frame="radar"))
        assert frame == "radar" and np.array_equal(xyz, np.stack([rc[k] for k in "xyz"], 1).astype(np.float32))
        want = ns + np.round(rc["time"].astype(np.float32).astype(float) * 1e9).astype(np.int64)
        assert t.tolist() == want.tolist()

    layout = [("x", "f4", 0), ("y", "f4", 4), ("z", "f4", 8), ("timestamp", "f8", 16)]
    for head in (ns, 120 * 10**9 + 5):  # wall time, and sim time (small in ns too)
        want = head + np.arange(4) * 10**7
        for scale in (1e-9, 1):  # "timestamp": absolute s or ns
            c = {"x": np.ones(4), "y": np.arange(4.0), "z": np.ones(4), "timestamp": want * scale}
            _, t, _, _ = R.decode_pointcloud2(m.cloud(head, c, layout, 24))
            assert np.abs(t - want).max() < 1000, (head, scale)

    def times(name, typ, values):
        c = {"x": np.arange(1.0, 4.0), "y": np.ones(3), "z": np.ones(3), name: values}
        layout = [("x", "f4", 0), ("y", "f4", 4), ("z", "f4", 8), (name, typ, 12)]
        xyz, t, _, _ = R.decode_pointcloud2(m.cloud(ns, c, layout, 20))
        assert t is None or len(t) == len(xyz)
        return None if t is None else (xyz[:, 0].tolist(), (t - ns).tolist())

    # A 5 Hz scan stamped at its end: its times reach past 0.2 s before the stamp, and they are all kept.
    assert times("time", "f4", [-0.21875, -0.125, 0.0]) == ([1, 2, 3], [-218_750_000, -125_000_000, 0])
    # A time further than 0.5 s from the stamp, or NaN, drops only its own point; more than half of them, the times.
    assert times("t", "u4", [0, 10**8, 6 * 10**8]) == ([1, 2], [0, 10**8])
    assert times("time", "f4", [0, np.nan, 0.1]) == ([1, 3], [0, 100_000_001])
    assert times("t", "u4", [0, 6 * 10**8, 7 * 10**8]) is None
    assert times("t", "f4", [0, 0.01, 0.02]) is None  # "t" must be uint32 ns
    assert times("timestamp", "f8", [1.0, 2.0, 3.0]) is None  # not near the stamp at all
    assert times("intensity", "f4", [0, 1, 2]) is None

    empty = m.cloud(ns, {k: np.zeros(0) for k in "xyz"}, POINTS_LAYOUT[:3], 12)
    xyz, t, _, _ = R.decode_pointcloud2(empty)
    assert xyz.shape == (0, 3) and t is None
    ints = m.cloud(ns, {k: np.ones(2) for k in "xyz"}, [("x", "i2", 0), ("y", "i2", 2), ("z", "i2", 4)], 6)
    with pytest.raises(ValueError, match="no float x, y and z fields"):
        R.decode_pointcloud2(ints)
    good = m.cloud(ns, cols, POINTS_LAYOUT, 32, height=2, row_pad=8)
    with pytest.raises(ValueError, match="does not fit in 200 bytes"):
        R.decode_pointcloud2(dataclasses.replace(good, data=good.data[:200]))
    nopad = R.decode_pointcloud2(dataclasses.replace(good, data=good.data[:-8]))  # the last row's padding is missing
    assert nopad[0].shape == (8, 3)
    with pytest.raises(ValueError, match="does not fit in a point of 16 bytes"):
        R.decode_pointcloud2(dataclasses.replace(good, point_step=16))
    # row_step 0 (some converters): one row, or rows with no room for padding, read as unpadded.
    for c in (m.cloud(ns, cols, POINTS_LAYOUT, 32), m.cloud(ns, cols, POINTS_LAYOUT, 32, height=2)):
        assert R.decode_pointcloud2(dataclasses.replace(c, row_step=0))[0].shape == (8, 3)
    with pytest.raises(ValueError, match="row_step 0 does not fit"):  # padded rows: the padding is unknown
        R.decode_pointcloud2(dataclasses.replace(good, row_step=0))


def test_decode_livox():
    for ros1 in (True, False):
        m, ns = Msgs(ros1), S0 + 7
        pts = livox_points(1) + [(600_000, np.nan, 1.0, 1.0, 0x00)]
        xyz, t, head, frame = R.decode_livox(m.livox_msg(ns, ns + LIVOX_BASE, pts, frame="/livox_frame"))
        assert xyz.dtype == np.float32 and xyz.tolist() == [[3, 0, 0], [2, 1, 0], [2, 5, 1]]  # tags 0x00, 0x10, 0x11
        assert t.dtype == np.int64 and t.tolist() == [ns + LIVOX_BASE + o for o in (0, 100_000, 500_000)]
        assert (head, frame) == (ns, "livox_frame")
        xyz, t, _, _ = R.decode_livox(m.livox_msg(ns, ns, []))
        assert xyz.shape == (0, 3) and t.shape == (0,)
        kept = (0, 100_000, 500_000)
        _, t, head, _ = R.decode_livox(m.livox_msg(ns, 123 * 10**9, livox_points(1)))  # timebase on the LiDAR's clock
        assert head == ns and t.tolist() == [ns + o for o in kept]  # the header's clock
        _, t, head, _ = R.decode_livox(m.livox_msg(0, ns, livox_points(1)))  # no header stamp: timebase counts
        assert head == 0 and t.tolist() == [ns + o for o in kept]


def test_typestore(tmp_path):
    """typestore() is Humble with both Livox drivers' types; a .db3 without message definitions (as
    Humble wrote them) decodes Livox only with it."""
    ts = R.typestore()
    assert ts is R.typestore() and "sensor_msgs/msg/PointCloud2" in ts.types
    for pkg in ("livox_ros_driver", "livox_ros_driver2", "livox_interfaces"):  # the last: livox_ros2_driver
        assert R.LIDAR[f"{pkg}/msg/CustomMsg"] == "CustomMsg"
        assert [f[0] for f in ts.fielddefs[f"{pkg}/msg/CustomMsg"][1]] == [
            "header", "timebase", "point_num", "lidar_id", "rsvd", "points"]
        assert [f[0] for f in ts.fielddefs[f"{pkg}/msg/CustomPoint"][1]] == list(R.LIVOX_FIELDS)
    db3 = write_bag(tmp_path / "b", "sqlite3", make=lambda m: [
        ("/livox/lidar", m.livox, S0, m.livox_msg(S0, S0, livox_points(0)))]) / "b.db3"
    strip_definitions(db3)
    with AnyReader([db3], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as r:
        [c] = r.connections
        with pytest.raises(Exception, match="livox_ros_driver2/msg/CustomMsg"):  # an unknown type
            r.deserialize(next(r.messages(connections=[c]))[2], c.msgtype)
    [(log, head, xyz, t)] = R.scans(db3, "/livox/lidar", [(S0, S0 + 1)])
    assert (log, head) == (S0, S0) and xyz.shape == (3, 3)

    # A bag that defines its camera topic's type but not Livox (recorded without the Livox .msg):
    # rosbags then ignores the default typestore, so open_bag adds the missing types.
    mixed = write_bag(tmp_path / "m", "sqlite3", make=lambda m: [
        ("/cam/image_rect", IMG, S0, m.image(S0, picture(0), "rgb8")),
        ("/livox/lidar", m.livox, S0, m.livox_msg(S0, S0, livox_points(0)))]) / "m.db3"
    with sqlite3.connect(mixed) as c:
        c.execute("DELETE FROM message_definitions WHERE topic_type LIKE 'livox%'")
    c.close()
    with AnyReader([mixed], default_typestore=R.typestore()) as r:
        assert LIVOX2 not in r.typestore.fielddefs and IMG in r.typestore.fielddefs
    out = R.image_topics(mixed)
    assert [t["topic"] for t in out["topics"]] == ["/cam/image_rect"] and out["lidar_skipped"] == []
    assert [(x["topic"], x["msgtype"], x["points"]) for x in out["lidar"]] == [("/livox/lidar", LIVOX2, 3)]
    assert [xyz.shape for _, _, xyz, _ in R.scans(mixed, "/livox/lidar", [(S0, S0 + 1)])] == [(3, 3)]


def test_tf_chain():
    Tc, Tl = matrix(*T_BASE_CAM), matrix(*T_BASE_LIVOX)
    edges = [("base_link", "camera_optical_frame", Tc, True, 0), ("base_link", "livox_frame", Tl, True, 0)]
    T, frames, link = R.tf_chain(edges, "camera_optical_frame", "/livox_frame")
    assert np.allclose(T, np.linalg.inv(Tc) @ Tl) and link == "static"
    assert frames == ["camera_optical_frame", "base_link", "livox_frame"]
    back, frames2, _ = R.tf_chain(edges, "livox_frame", "camera_optical_frame")  # the reverse way
    assert np.allclose(back @ T, np.eye(4)) and frames2 == frames[::-1]
    assert R.tf_chain(edges, "camera_optical_frame", "nowhere") is None and R.tf_chain(edges, "", "livox_frame") is None
    same = R.tf_chain(edges, "base_link", "/base_link")
    assert np.array_equal(same[0], np.eye(4)) and same[1:] == (["base_link"], "static")

    def shift(y):
        return matrix((0, y, 0), (0, 0, 0, 1))

    # A static path wins over a shorter dynamic one; a dynamic link takes its value nearest the time.
    dyn = [("camera_optical_frame", "livox_frame", shift(9), False, 5)] + \
        [("base_link", "radar", shift(y), False, t) for t, y in ((20, 0.2), (10, 0.1), (30, 0.3))]
    assert np.allclose(R.tf_chain(edges + dyn, "camera_optical_frame", "livox_frame")[0], T)
    T3, frames3, link3 = R.tf_chain(edges + dyn, "camera_optical_frame", "radar", t_ns=22)
    assert (frames3, link3) == (["camera_optical_frame", "base_link", "radar"], "dynamic")
    assert np.allclose(T3, np.linalg.inv(Tc) @ shift(0.2))
    assert np.allclose(R.tf_chain(edges + dyn, "base_link", "radar")[0], shift(0.1))  # no time: the earliest
    assert np.allclose(R.tf_chain(edges + [("base_link", "livox_frame", shift(1), True, 9)], "base_link",
                                  "livox_frame")[0], shift(1))  # a later static value replaces the earlier one

    # Body axes: camera_link has x forward, so R_OPT_LINK turns the chain into optical axes.
    body = [("base_link", "camera_link", matrix((0.1, 0, 0.2), (0, 0, 0, 1)), True, 0), edges[1]]
    ext = R.camera_to_lidar("camera_link", "livox_frame", body)
    assert (ext["camera_axes"], ext["link"], ext["chain"]) == ("body", "static", ["camera_link", "base_link", "livox_frame"])
    assert np.allclose(ext["T_cam_lidar"][:3], lidar_scale.R_OPT_LINK @ R.tf_chain(body, "camera_link", "livox_frame")[0][:3])
    ahead = np.linalg.inv(Tl) @ [2.1, 0, 0.2, 1]  # 2 m ahead of both cameras
    assert np.allclose(ext["T_cam_lidar"] @ ahead, [0, 0, 2, 1])
    opt = R.camera_to_lidar("camera_optical_frame", "livox_frame", edges)
    assert opt["camera_axes"] == "optical" and np.allclose(opt["T_cam_lidar"] @ ahead, [0, 0, 2, 1])
    assert R.camera_to_lidar("camera_link", "nowhere", body) is None
    assert (R.camera_axes("/cam_optical"), R.camera_axes("camera_link"), R.camera_axes(" ")) == ("optical", "body", None)
    assert (ext["camera_axes_from"], opt["camera_axes_from"]) == ("default", "name")

    # A frame without "optical" in its name has optical axes when one of its TF links is the standard
    # body-to-optical rotation, as a Kalibr or EuRoC cam0 under base_link: from either end of the link,
    # and also with the camera tilted by less than AXES_TOL_DEG. A name with "optical" decides alone.
    cam0 = [("base_link", "cam0", Tc, True, 0), edges[1]]
    kal = R.camera_to_lidar("cam0", "livox_frame", cam0)
    assert (kal["camera_axes"], kal["camera_axes_from"], kal["link"]) == ("optical", "tf", "static")
    assert np.allclose(kal["T_cam_lidar"], opt["T_cam_lidar"]) and R.camera_axes("cam0", cam0) == "optical"
    assert R.camera_axes_from("cam0", [("cam0", "imu", R.invert(Tc), False, 0)]) == ("optical", "tf")  # cam0 the parent

    def tilted(deg):  # the optical camera turned down by deg about its x axis
        return [("base_link", "cam0", Tc @ matrix((0, 0, 0), Rotation.from_euler("x", deg, degrees=True).as_quat()),
                 True, 0)]

    assert R.camera_axes_from("cam0", tilted(R.AXES_TOL_DEG - 5)) == ("optical", "tf")
    assert R.camera_axes_from("cam0", tilted(R.AXES_TOL_DEG + 5)) == ("body", "default")
    # camera_link's links go to base_link (the same axes) and to its optical frame camera_lens (the rotation the
    # other way), so it keeps body axes; TF shows camera_lens to be optical.
    tree = body + [("camera_link", "camera_lens", Tc, True, 0)]
    assert R.camera_axes_from("camera_link", tree) == ("body", "default")
    assert R.camera_axes_from("camera_lens", tree) == ("optical", "tf")
    assert R.camera_axes_from("camera_optical_frame", body) == ("optical", "name")
    # A stereo pair as Kalibr gives it: cam1 hangs from cam0 by a near-identity link, so it has cam0's axes, from
    # cam0's TF link or from its name. Under a body camera it keeps body axes; one step only.
    near = matrix((0.1, 0, 0), Rotation.from_euler("y", 2, degrees=True).as_quat())
    assert R.camera_axes_from("cam1", cam0 + [("cam0", "cam1", near, True, 0)]) == ("optical", "tf")
    assert R.camera_axes_from("cam1", [("cam0_optical", "cam1", near, True, 0)]) == ("optical", "tf")
    assert R.camera_axes_from("cam1", body + [("camera_link", "cam1", near, True, 0)]) == ("body", "default")
    assert R.camera_axes_from("cam2", cam0 + [("cam0", "cam1", near, True, 0), ("cam1", "cam2", near, True, 0)]) == (
        "body", "default")
    # Only static links count in that step: a moving odom link of the parent, turned near R_OPT_LINK at some
    # moment, says nothing about the camera's mount.
    odom = [("odom", "base_link", Tc, False, 0)]
    assert R.camera_axes_from("camera_link", body + odom) == ("body", "default")
    assert R.camera_axes_from("camera_link", body + [(p, c, T, True, t) for p, c, T, _, t in odom]) == ("optical", "tf")
    # Many values of one link: each neighbour is looked at once.
    many = [("base_link", "camera_link", body[0][2], True, k) for k in range(3000)]
    many += [("base_link", f"s{i}", body[0][2], True, k) for i in range(4) for k in range(3000)]
    t0 = time.perf_counter()
    assert R.camera_axes_from("camera_link", many) == ("body", "default") and time.perf_counter() - t0 < 2.0
    assert R.camera_axes_from("cam0", []) == ("body", "default") and R.camera_axes_from("", cam0) == (None, None)
    # The axes given to camera_to_lidar are used: the record's axes hold for every frame.
    forced = R.camera_to_lidar("cam0", "livox_frame", cam0, axes=("body", "default"))
    assert (forced["camera_axes"], forced["camera_axes_from"]) == ("body", "default")
    assert np.allclose(forced["T_cam_lidar"][:3], lidar_scale.R_OPT_LINK @ R.tf_chain(cam0, "cam0", "livox_frame")[0][:3])


def test_scans_reads_only_the_windows(bag, monkeypatch):
    calls, pulled = [], []
    messages = AnyReader.messages

    def counting(self, *a, **kw):
        calls.append((kw.get("start"), kw.get("stop")))
        for x in messages(self, *a, **kw):
            pulled.append(x[1])
            yield x

    monkeypatch.setattr(AnyReader, "messages", counting)
    log = [S0 + i * 10**8 + LAG for i in range(N + 2)]
    windows = [(log[2], log[3] + 1), (log[1], log[2] + 1), (log[5] - 10, log[5] + 10), (log[7], log[7])]
    got = list(R.scans(bag, "/livox/lidar", windows))
    assert calls == [(log[1], log[3] + 1), (log[5] - 10, log[5] + 10)]  # merged; the empty window is dropped
    assert pulled == [g[0] for g in got] == [log[i] for i in (1, 2, 3, 5)]
    for (_, head, xyz, t), i in zip(got, (1, 2, 3, 5)):
        assert head == S0 + i * 10**8 and xyz[0].tolist() == [2 + i, 0, 0]
        assert t.tolist() == [head + LIVOX_BASE + o for o in (0, 100_000, 500_000)]
    [(_, _, xyz, t)] = R.scans(bag, "/points", [(log[4], log[4] + 1)])
    assert xyz.shape == (8, 3) and xyz[0, 0] == 5 and len(t) == 8
    assert list(R.scans(bag, "/radar/points", [])) == []


def test_scans_skip_empty_scans(tmp_path):
    """An empty Livox message (zero stamp and timebase) and a cloud whose points are all NaN give no scan."""
    def make(m):
        nan = {k: np.full(2, np.nan) for k in "xyz"}
        return [("/livox/lidar", m.livox, S0, m.livox_msg(0, 0, [])), ("/cloud", PC2, S0, m.cloud(S0, nan, POINTS_LAYOUT[:3], 12)),
                ("/livox/lidar", m.livox, S0 + 10**8, m.livox_msg(S0 + 10**8, S0 + 10**8, livox_points(0)))]

    bag = write_bag(tmp_path / "e.bag", make=make)
    got = list(R.scans(bag, "/livox/lidar", [(S0, S0 + 10**9)]))
    assert [(log, head, len(xyz)) for log, head, xyz, _ in got] == [(S0 + 10**8, S0 + 10**8, 3)]
    assert list(R.scans(bag, "/cloud", [(S0, S0 + 1)])) == []


def test_probe_stamps_match_frames(bag):
    for topic, clock in (("/cam_rect/image_rect/compressed", "header"), ("/mono16/image", "log")):
        st = {}
        info = R.probe(bag, topic, stamps_out=st)
        assert info["clock"] == clock and sorted(st) == ["header_ns", "log_ns"]
        assert len(st["header_ns"]) == len(st["log_ns"]) == info["n_frames"]
        ns = st["header_ns" if clock == "header" else "log_ns"]
        times = [t for _, t, _ in R.frames(bag, topic)]
        assert times == pytest.approx([(x - ns[0]) / 1e9 for x in ns])  # index i is frame i
    assert st["header_ns"] == [S0 + s * 10**8 for s in MONO16] and st["log_ns"] == [S0 + i * 10**8 + LAG for i in range(N)]


def test_camera_model(bag):
    """P with its R for a "rect" topic, K for an all-zero D, K and D for a raw topic."""
    eye = np.eye(3).tolist()
    rect = R.camera_model(bag, "/cam_rect/image_rect/compressed")  # the name says rect: P, though D is not zero
    assert np.allclose(rect.pop("R"), R_RECT) and rect == {
        "K": [r[:3] for r in P_RECT], "D": [], "model": None, "size": [W, H], "source": "/cam_rect/camera_info: P (rectified)"}
    assert R.camera_model(bag, "/front/image_raw") == {  # all-zero D on a raw topic: K, the raw image's; no R
        "K": K_FRONT, "D": [], "model": None, "R": eye, "size": [W, H], "source": "/front/camera_info: K (no distortion)"}
    assert R.camera_model(bag, "/raw/image_raw") == {
        "K": K_RAW, "D": [0.1, -0.02, 0, 0, 0.001], "model": "plumb_bob", "R": eye, "size": [W, H],
        "source": "/raw/camera_info: K and D (plumb_bob)"}
    assert R.camera_model(bag, "/png/image/compressed") is None  # K and P are the identity: placeholders
    assert R.camera_model(bag, "/mono/image") is None  # no camera_info


def test_camera_model_sizes(tmp_path):
    """roi and binning move and divide the matrix; a camera_info for another size of the same aspect
    ratio is scaled to the images; another aspect ratio is an error; 0x0 means the images' size."""
    k_big = [[160, 0, 63.5], [0, 160, 47.5], [0, 0, 1]]  # for 128x96, the images' 64x48 doubled
    cases = {"/bin": {"k": k_big, "size": (128, 96), "binning": (2, 2)}, "/half": {"k": k_big, "size": (128, 96)},
             "/roi": {"k": [[100, 0, 64], [0, 100, 48], [0, 0, 1]], "size": (128, 96), "roi": (32, 16, 64, 48)},
             "/wide": {"k": k_big, "size": (128, 48)}, "/nosize": {"k": K_RAW, "size": (0, 0)},
             "/off": {"k": [[90, 0, 500], [0, 90, 24], [0, 0, 1]]}}

    def make(m):
        out = []
        for ns_, kw in cases.items():
            out += [(f"{ns_}/camera_info", CI, S0, m.info(S0, [0.1, 0, 0, 0, 0], **kw)),
                    (f"{ns_}/image_raw", IMG, S0, m.image(S0, picture(0), "rgb8"))]
        return out

    bag = write_bag(tmp_path / "s.bag", make=make)
    half = [[80, 0, 31.5], [0, 80, 23.5], [0, 0, 1]]  # pixel centres at integers: (63.5 + 0.5) / 2 - 0.5
    for ns_, K, how in (("/bin", half, ""), ("/half", half, ", scaled from 128x96"),
                        ("/roi", [[100, 0, 32], [0, 100, 32], [0, 0, 1]], ""), ("/nosize", K_RAW, "")):
        got = R.camera_model(bag, f"{ns_}/image_raw")
        assert np.allclose(got["K"], K) and got["size"] == [W, H], ns_
        assert got["source"] == f"{ns_}/camera_info: K and D (plumb_bob){how}", ns_
    # A point lands on the binned pixel that covers its full-size pixel.
    X = [[0.3, -0.2, 2.0]]
    u, v, _, _ = lidar_scale.project(X, k_big, [0.1], "plumb_bob", (128, 96))
    ub, vb, _, ok = lidar_scale.project(X, half, [0.1], "plumb_bob", (W, H))
    assert ok.all() and np.allclose([ub[0], vb[0]], [(u[0] + 0.5) / 2 - 0.5, (v[0] + 0.5) / 2 - 0.5])
    with pytest.raises(ValueError, match="/wide/camera_info is for 128x48 images but /wide/image_raw has 64x48"):
        R.camera_model(bag, "/wide/image_raw")
    assert R.camera_model(bag, "/off/image_raw") is None  # the principal point is outside the image


def test_camera_model_distortion_models(tmp_path):
    cases = {"/fish": ("equidistant", [0.1, 0.01, 0, 0]), "/kb": ("Fisheye", [0.2, 0, 0, 0]),
             "/rat": ("rational_polynomial", [0.1, 0.01, 0, 0, 0.001, 0.05, 0, 0]),
             "/odd": ("weird_model", [0.1, 0, 0, 0, 0]), "/long": ("equidistant", [0.1, 0, 0, 0, 0.3]),
             "/nok": ("plumb_bob", [0.1, 0, 0, 0, 0])}

    def make(m):
        out = []
        for ns_, (model, d) in cases.items():
            k = np.zeros((3, 3)) if ns_ == "/nok" else K_RAW
            out += [(f"{ns_}/camera_info", CI, S0, m.info(S0, d, k, model=model)),
                    (f"{ns_}/image_raw", IMG, S0, m.image(S0, picture(0), "rgb8"))]
        return out

    bag = write_bag(tmp_path / "m.bag", make=make)
    for ns_, model in (("/fish", "equidistant"), ("/kb", "equidistant"), ("/rat", "rational_polynomial")):
        got = R.camera_model(bag, f"{ns_}/image_raw")
        assert (got["model"], got["D"], got["K"]) == (model, cases[ns_][1], K_RAW), ns_
        u, v, _, ok = lidar_scale.project([[0.1, 0.05, 2.0]], got["K"], got["D"], got["model"], got["size"])
        assert ok.all(), ns_  # lidar_scale.project takes the result as it is
    for ns_ in ("/odd", "/long"):
        with pytest.raises(ValueError, match=f"{ns_}/camera_info: .*; use a rectified topic or give K and D in the calibration"):
            R.camera_model(bag, f"{ns_}/image_raw")
    assert R.camera_model(bag, "/nok/image_raw") is None  # a raw topic needs a K
