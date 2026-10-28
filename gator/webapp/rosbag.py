"""Camera images, LiDAR scans and TF in a rosbag: its topics, the facts of one topic, its frames and scans.

Runs in envs/web (rosbags 0.11.8, pure Python, with numpy and pillow). video.py calls it for a job
whose job.json has "source": {"kind": "rosbag", "topic": T}. rosbags reads a ROS1 .bag, a ROS2 bag
folder (metadata.yaml next to the .db3 or .mcap files) and a single ROS2 .mcap or .db3 file, without
ROS. The message types (typestore), the LiDAR decoders, TF and the intrinsics (camera_model): build_plan.md
note 12, Rosbag source 1-6.

image_topics lists the topics of sensor_msgs/msg/Image and sensor_msgs/msg/CompressedImage in one pass
over the bag. Depth and float images (16UC1, 32FC1, 16SC1, 64FC1, compressedDepth), encodings that
frames() cannot decode, empty (0x0) images and topics without messages go under "skipped". A topic is
rectified when its name has a "rect" segment (image_rect, rect_color, ...) or when the camera_info of
its namespace has all-zero distortion; it is not when that camera_info has a non-zero coefficient, and
unknown (None) without one or when the camera_info is uncalibrated (empty D or all-zero K, as
camera_info_manager sends without a calibration file). The namespace is the topic without its trailing
image, image_raw, image_rect or compressed segments: /cam/image_rect/compressed -> /cam/camera_info.
MV-SAM3D and Depth Anything 3 expect pinhole images, so the UI warns about raw lens distortion; nothing
here undistorts.

Frames are numbered in log-time order (the order of recording) and timed from the first message by one
clock per topic: header.stamp when every stamp is set and comes after the one before, else log time.
probe reports the clock and counts the stamps that are zero, go backwards or repeat (stamp_warnings; 0
for a topic without stamps). A message that cannot be decoded gives a black frame, with a warning, so
that the frame indices stay those of the messages. CompressedImage is decoded by pillow (jpeg or png; a
format string such as "rgb8; jpeg compressed bgr8" names the codec). Image takes rgb8, bgr8, rgba8,
bgra8, mono8, mono16 (scaled to 8 bit by each frame's max) and bayer_rggb8, bayer_bggr8, bayer_gbrg8,
bayer_grbg8 (each 2x2 cell gives one RGB value for its 4 pixels), with row padding (step) and
is_bigendian. Bag images carry no rotation. probe(stamps_out=) also gives every frame's header stamp
and log time in ns (index i = message i, the order of frames()).
"""

import functools
import io
import re
import tempfile
from collections import Counter, deque
from contextlib import ExitStack, contextmanager
from pathlib import Path

import numpy as np

from gator.stage_a import lidar_scale

TYPES = {"sensor_msgs/msg/Image": "Image", "sensor_msgs/msg/CompressedImage": "CompressedImage"}
INFO = "sensor_msgs/msg/CameraInfo"
DEPTH = ("16UC1", "32FC1", "16SC1", "64FC1")
BYTES = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1, "mono16": 2,
         "bayer_rggb8": 1, "bayer_bggr8": 1, "bayer_gbrg8": 1, "bayer_grbg8": 1}  # per pixel
IMAGE_SEGMENTS = {"image", "image_raw", "image_rect", "image_color", "image_mono", "image_rect_color",
                  "image_rect_mono", "compressed"}
LIVOX_PACKAGES = ("livox_ros_driver", "livox_ros_driver2", "livox_interfaces")  # the last: livox_ros2_driver's
LIDAR = {"sensor_msgs/msg/PointCloud2": "PointCloud2",
         **{f"{pkg}/msg/CustomMsg": "CustomMsg" for pkg in LIVOX_PACKAGES}}
TF = ("tf2_msgs/msg/TFMessage", "tf/msg/tfMessage")
AXES_TOL_DEG = 20  # camera_axes_from: a TF link this close to the body-to-optical rotation shows optical axes, so a
# tilted camera passes; every other turn between axis conventions is 90 deg or more from it
RATE_MESSAGES = 11  # the topics pass times a LiDAR topic's rate by the log times of its first 11 messages
POINT_TIME_S = 0.5  # per-point times further than this from the stamp are not used (a scan of a 2 Hz LiDAR fits)
TIMEBASE_S = 0.2  # a Livox timebase further than this from a set header stamp is on another clock
MIN_FOCAL = 0.1  # camera_model: fx or fy below this share of the image's long side is a placeholder
ASPECT_TOL = 0.01  # intrinsics for another image size are scaled when the aspect ratios agree this well
POINT_TYPES = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}  # PointField datatype
POINT_TIMES = (("t", 6, "offset_ns"), ("time", 7, "offset_s"), ("timestamp", 8, "absolute"))  # name, datatype, meaning
LIVOX_TAGS = (0x00, 0x10)  # kept values of tag & 0x30
LIVOX_FIELDS = ("offset_time", "x", "y", "z", "reflectivity", "tag", "line")
LIVOX_POINT = """\
uint32 offset_time
float32 x
float32 y
float32 z
uint8 reflectivity
uint8 tag
uint8 line
"""
LIVOX_CUSTOM = """\
std_msgs/Header header
uint64 timebase
uint32 point_num
uint8 lidar_id
uint8[3] rsvd
CustomPoint[] points
"""
LIVOX_MSGS = {f"{pkg}/msg/{name}": text for pkg in LIVOX_PACKAGES
              for name, text in (("CustomPoint", LIVOX_POINT), ("CustomMsg", LIVOX_CUSTOM))}  # .msg texts


@functools.cache
def typestore():
    """The ROS2 Humble typestore with the Livox types of LIVOX_MSGS, made once."""
    from rosbags.typesys import Stores, get_types_from_msg, get_typestore

    ts = get_typestore(Stores.ROS2_HUMBLE)
    types = {}
    for name, text in LIVOX_MSGS.items():
        types.update(get_types_from_msg(text, name))
    ts.register(types)
    return ts


@contextmanager
def open_bag(path):
    """The open rosbags AnyReader of the bag at path. typestore() decodes a bag without its own
    message definitions. It also gives the types that a bag with some definitions lacks (add_types)."""
    from rosbags.highlevel import AnyReader

    path = Path(path)
    with ExitStack() as stack:
        p = path
        if path.is_file() and path.suffix != path.suffix.lower():  # rosbags reads only ".bag" as ROS1
            p = Path(stack.enter_context(tempfile.TemporaryDirectory())) / ("bag" + path.suffix.lower())
            p.symlink_to(path.resolve())
        try:
            r = AnyReader([p], default_typestore=typestore())
            r.open()
        except Exception as e:  # rosbags raises its own errors, OSError, and struct, yaml and sqlite errors
            raise SystemExit(f"cannot read {path}: {e}")
        stack.callback(r.close)
        add_types(r)
        yield r


def add_types(r):
    """Register in an open reader's typestore the types of its topics that the bag does not define but
    typestore() has, with the types they use. rosbags uses its default typestore only when no topic has
    a definition, so a bag with definitions for its camera topics but none for Livox needs this. The
    bag's own definitions stay."""
    have, known = r.typestore.fielddefs, typestore().fielddefs
    want, todo = {}, [c.msgtype for c in r.connections]
    while todo:
        name = todo.pop()
        if name in have or name in want or name not in known:
            continue
        want[name] = known[name]
        todo += [x for x in _strings(known[name][1]) if x in known]  # the field types it uses
    if want:
        try:
            r.typestore.register(want)
        except Exception as e:
            print(f"warning: cannot add the message types {', '.join(sorted(want))} ({e})", flush=True)


def _strings(x):
    """Every string in nested tuples and lists."""
    if isinstance(x, str):
        return [x]
    return [s for y in x for s in _strings(y)] if isinstance(x, (tuple, list)) else []


def connections(r, topic):
    cs = [c for c in r.connections if c.topic == topic]
    if not cs:
        raise SystemExit(f"the bag has no topic {topic}")
    if cs[0].msgtype not in TYPES:
        raise SystemExit(f"{topic} is {cs[0].msgtype}, not a camera image topic")
    return cs


def stamp(msg):
    """header.stamp in ns."""
    s = msg.header.stamp
    return s.sec * 10**9 + s.nanosec


def stamp_clock(stamps):
    """("header" or "log", warnings) of a topic's header stamps (ns, log order): the stamps are the
    clock when every one is set and comes after the one before; warnings counts the stamps that are
    not (0 for a topic without stamps)."""
    s = np.asarray(stamps, np.int64)
    if not s.any():
        return "log", 0
    bad = int(s[0] == 0) + int(((s[1:] == 0) | (s[1:] <= s[:-1])).sum())
    return ("log" if bad else "header"), bad


def span(t0, t1, n):
    """(duration_s, rate_hz) of n messages from t0 to t1 (ns); the duration counts the last message's period."""
    d = (t1 - t0) / 1e9 * n / max(n - 1, 1)
    return round(d, 4), (round(n / d, 4) if d > 0 else None)


def info_topic(topic):
    """The camera_info topic of an image topic's namespace (see the module docstring)."""
    parts = topic.strip("/").split("/")
    while parts and parts[-1] in IMAGE_SEGMENTS:
        parts.pop()
    return "/" + "/".join(parts + ["camera_info"])


def named_rect(topic):
    """True when the topic's name has a "rect" segment (image_rect, rect_color, ...)."""
    return re.search(r"(^|[/_])rect", topic, re.I) is not None


def info_array(ci, name):
    """A camera_info array as floats by its ROS2 name (d, k, p), also under its ROS1 name (D, K, P)."""
    return np.asarray(getattr(ci, name) if hasattr(ci, name) else getattr(ci, name.upper()), float).reshape(-1)


def frame_name(frame_id):
    """A TF frame name without spaces and a leading "/" (ROS1 tf allowed one)."""
    return (frame_id or "").strip().lstrip("/")


def codec(msg):
    """jpeg or png of a CompressedImage, from its format string, else from the data's first bytes."""
    f = msg.format.lower()
    if "png" in f:
        return "png"
    if "jpeg" in f or "jpg" in f:
        return "jpeg"
    head = bytes(msg.data[:4])
    return "jpeg" if head[:2] == b"\xff\xd8" else "png" if head == b"\x89PNG" else msg.format


def unusable(msg):
    """Why frames() cannot use the topic of this message, else None."""
    if hasattr(msg, "format"):  # CompressedImage
        f = msg.format
        if "compresseddepth" in f.lower() or f.split(";")[0].strip().upper() in DEPTH:
            return f"depth image ({f})"
        return None
    if not (msg.width and msg.height):
        return f"empty image ({msg.width}x{msg.height})"
    if msg.encoding.upper() in DEPTH:
        return f"depth or float image ({msg.encoding})"
    if msg.encoding.lower() not in BYTES:
        return f"encoding {msg.encoding} is not supported"
    return None


def describe(msg):
    """(width, height, format) of a message: format is jpeg/png or the Image encoding."""
    if hasattr(msg, "format"):
        from PIL import Image

        with Image.open(io.BytesIO(msg.data)) as im:
            return im.width, im.height, codec(msg)
    return msg.width, msg.height, msg.encoding


def gray8(a):
    """RGB uint8 of a 16-bit gray image, scaled by its max."""
    g = (a.astype(np.float32) * (255 / max(int(a.max()), 1))).round().astype(np.uint8)
    return np.repeat(g[..., None], 3, 2)


def debayer(a, pattern):
    """RGB of a Bayer mosaic; pattern names the 2x2 cell row by row ("rggb"). Each cell gives one RGB
    value (the two greens averaged) for its 4 pixels; an odd last row or column copies its neighbour."""
    h, w = a.shape
    H, W = h // 2 * 2, w // 2 * 2
    cells = [a[0:H:2, 0:W:2], a[0:H:2, 1:W:2], a[1:H:2, 0:W:2], a[1:H:2, 1:W:2]]
    g = (cells[pattern.index("g")].astype(np.uint16) + cells[pattern.rindex("g")] + 1) // 2
    rgb = np.stack([cells[pattern.index("r")], g.astype(np.uint8), cells[pattern.index("b")]], -1)
    rgb = rgb.repeat(2, 0).repeat(2, 1)
    return np.pad(rgb, ((0, h - H), (0, w - W), (0, 0)), mode="edge") if (H, W) != (h, w) else rgb


def unpack(msg):
    """RGB uint8 HxWx3 of a sensor_msgs/Image."""
    enc = msg.encoding.lower()
    if enc not in BYTES:
        raise SystemExit(f"image encoding {msg.encoding!r} is not supported "
                         "(rgb8, bgr8, rgba8, bgra8, mono8, mono16, bayer_rggb8/bggr8/gbrg8/grbg8)")
    h, w, b, step = msg.height, msg.width, BYTES[enc], msg.step
    data = np.asarray(msg.data, np.uint8)
    if not (h and w):
        raise SystemExit(f"an empty {w}x{h} image")
    if step < w * b or data.size < h * step:
        raise SystemExit(f"a {w}x{h} {msg.encoding} image with step {step} does not fit in {data.size} bytes")
    rows = data[:h * step].reshape(h, step)[:, :w * b]
    if enc == "mono16":
        return gray8(np.ascontiguousarray(rows).view(">u2" if msg.is_bigendian else "<u2"))
    if enc.startswith("bayer_"):
        return debayer(rows, enc[6:10])
    a = rows.reshape(h, w, b)
    if enc == "mono8":
        return np.repeat(a, 3, 2)
    return np.ascontiguousarray(a[..., 2::-1] if enc.startswith("bgr") else a[..., :3])


def decode(msg, width=None):
    """RGB uint8 HxWx3 of an Image or CompressedImage message; width: output width (None = as stored)."""
    from PIL import Image

    if hasattr(msg, "format"):
        im = Image.open(io.BytesIO(msg.data))
        w, h = im.size
        if width:
            im.draft(None, (width, round(width * h / w)))  # JPEG decodes at 1/2, 1/4 or 1/8 when that is enough
        if im.mode.startswith("I"):  # 16-bit png
            im = Image.fromarray(gray8(np.asarray(im)))
        elif im.mode != "RGB":
            im = im.convert("RGB")
    else:
        rgb = unpack(msg)
        if not width or rgb.shape[1] == width:
            return rgb
        im, (h, w) = Image.fromarray(rgb), rgb.shape[:2]
    if width and im.width != width:
        im = im.resize((width, round(width * h / w)), Image.BILINEAR)
    return np.asarray(im)


def preview(r, msgs, side):
    """(width, height, format, RGB uint8 with long side `side`, never enlarged) of the first of msgs
    [(connection, raw)] that decodes."""
    for c, raw in msgs:
        try:
            msg = r.deserialize(raw, c.msgtype)
            w, h, fmt = describe(msg)
            return w, h, fmt, decode(msg, max(1, round(w * min(1, side / max(w, h)))))
        except (SystemExit, Exception) as e:  # also pillow's DecompressionBombError
            err = e
    raise SystemExit(f"cannot decode: {err}")


def image_topics(path, previews=None, side=320, progress=None):
    """{"topics": [...], "skipped": [...], "lidar": [...], "lidar_skipped": [...], "tf": [...]} of the
    bag (topics.json in video.py without "preview"), from one pass over the camera topics and all of
    /tf_static and /tf, and the first message of each LiDAR topic (build_plan.md note 12, Rosbag source 2).
    The stage reads /tf near any frame, so the listing reads all of it, and keeps one value of each link.
    Topics are sorted rectified first, then by message count, highest first; LiDAR topics by message count.
    previews: a dict that gets {topic: the middle message (else the first or the last that decodes) as
    RGB uint8 with long side `side`, never enlarged}. progress: called with the share of the bag read,
    0 to 1."""
    with open_bag(path) as r:
        conns = [c for c in r.connections if c.msgtype in TYPES or c.msgtype == INFO or tf_static(c) is not None]
        kinds = {c.topic: TYPES[c.msgtype] for c in conns if c.msgtype in TYPES}
        count = Counter()
        for c in conns:
            count[c.topic] += c.msgcount
        first, mid, last, infos, seen, stamps, frame_ids = {}, {}, {}, {}, Counter(), {}, {}
        links = {}  # (parent, child, static): the edge that tf_listing lists, so a long bag's /tf takes little memory
        start, total, shown = r.start_time, max(r.end_time - r.start_time, 1), 0.0
        for c, t, raw in (r.messages(connections=conns) if conns else ()):  # no connections would mean all
            if c.msgtype == INFO:
                infos.setdefault(c.topic, (c, raw))
                continue
            if c.msgtype in TF:  # a message that cannot be decoded is skipped: TF problems never fail the listing
                for e in tf_edges(r, c, t, raw):
                    if e[3] or (e[0], e[1], False) not in links:  # the last static value, the first dynamic one
                        links[(e[0], e[1], e[3])] = e
                continue
            k = seen[c.topic]
            seen[c.topic] += 1
            if not k:
                first[c.topic] = (c, t, raw)
            if k == count[c.topic] // 2:
                mid[c.topic] = (c, raw)
            last[c.topic] = (c, t, raw)
            msg = r.deserialize(raw, c.msgtype)
            stamps.setdefault(c.topic, []).append(stamp(msg))
            if not k:
                frame_ids[c.topic] = frame_name(msg.header.frame_id)
            if progress and (t - start) / total >= shown + 0.02:
                shown = (t - start) / total
                progress(shown)

        lidar, lidar_skipped = lidar_topics(r)
        edges = list(links.values())
        topics, skipped = [], []
        for topic in sorted(kinds):
            if not seen[topic]:
                skipped.append({"topic": topic, "type": kinds[topic], "why": "no messages"})
                continue
            c, t0, raw = first[topic]
            cl, t1, rl = last[topic]
            why = unusable(r.deserialize(raw, c.msgtype))
            if why is None:
                try:  # the middle message, else the first or the last
                    w, h, fmt, small = preview(r, [mid.get(topic, (c, raw)), (c, raw), (cl, rl)], side)
                except SystemExit as e:
                    why = str(e)
            if why:
                skipped.append({"topic": topic, "type": kinds[topic], "why": why})
                continue
            if stamp_clock(stamps[topic])[0] == "header":
                t0, t1 = stamps[topic][0], stamps[topic][-1]
            duration, rate = span(t0, t1, seen[topic])
            rect = True if named_rect(topic) else None
            itopic, dist = info_topic(topic), None
            if itopic in infos:
                ci = r.deserialize(infos[itopic][1], infos[itopic][0].msgtype)
                d, k = np.abs(info_array(ci, "d")), info_array(ci, "k")
                dist = {"model": ci.distortion_model, "max_abs_d": round(float(d.max(initial=0)), 6)}
                if rect is None and d.size and k.any():  # an uncalibrated camera_info has no D or a zero K
                    rect = dist["max_abs_d"] == 0
            axes, axes_from = camera_axes_from(frame_ids[topic], edges)
            topics.append({"topic": topic, "type": kinds[topic], "count": seen[topic], "duration_s": duration,
                           "rate_hz": rate, "width": w, "height": h, "format": fmt, "rectified": rect,
                           "camera_info": itopic if itopic in infos else None, "distortion": dist,
                           "frame_id": frame_ids[topic], "camera_axes": axes, "camera_axes_from": axes_from,
                           "tf_links": tf_links(edges, frame_ids[topic], lidar)})
            if previews is not None:
                previews[topic] = small
    topics.sort(key=lambda t: (t["rectified"] is not True, -t["count"], t["topic"]))
    return {"topics": topics, "skipped": skipped, "lidar": lidar, "lidar_skipped": lidar_skipped,
            "tf": tf_listing(edges)}


def probe(path, topic, stamps_out=None):
    """Facts of one image topic, with video.probe's keys: n_frames is the message count, duration_s and
    fps follow the frame times, width and height are those of the first message that pillow can read.
    clock is "header" or "log" (see the module docstring); stamp_warnings counts the header stamps that
    are zero, go backwards or repeat. stamps_out: a dict that gets "header_ns" and "log_ns", the lists
    of every message's header stamp and log time in ns, in frame order (index i = message i)."""
    with open_bag(path) as r:
        cs = connections(r, topic)
        stamps, logs, size = [], [], None
        for c, t, raw in r.messages(connections=cs):
            msg = r.deserialize(raw, c.msgtype)
            if not logs:
                why = unusable(msg)
                if why:
                    raise SystemExit(f"{topic}: {why}")
            if size is None:
                try:
                    size = describe(msg)
                except Exception:  # a corrupt message (frames() gives it a black frame)
                    pass
            stamps.append(stamp(msg))
            logs.append(t)
    if not logs:
        raise SystemExit(f"{topic} has no messages")
    if size is None:
        raise SystemExit(f"cannot decode any message of {topic}")
    if stamps_out is not None:
        stamps_out.update(header_ns=stamps, log_ns=logs)
    (w, h, fmt), (clock, warnings) = size, stamp_clock(stamps)
    t = stamps if clock == "header" else logs
    duration, fps = span(min(t), max(t), len(t))
    return {"n_frames": len(t), "duration_s": duration, "fps": fps, "width": w, "height": h, "rotation": 0,
            "codec": f"{TYPES[cs[0].msgtype]} {fmt}", "hdr": None, "bit_depth": 16 if fmt == "mono16" else 8,
            "clock": clock, "stamp_warnings": warnings}


def frames(path, topic, keep=None, width=None, clock=None):
    """Yield (index, time_s, RGB uint8 HxWx3) of topic in log-time order, as video.frames does.
    keep: set of message indices (None = all); stops after the last one. width: output width (None =
    as stored). clock: probe()'s "header" or "log" (None = find it, which reads the topic once more).
    A message that cannot be decoded gives a black frame of the size of the first one that can."""
    last = max(keep, default=-1) if keep is not None else None
    with open_bag(path) as r:
        cs = connections(r, topic)
        if clock is None:
            clock = stamp_clock([stamp(r.deserialize(raw, c.msgtype)) for c, _, raw in r.messages(connections=cs)])[0]
        base, shape, pending = None, None, []  # pending: undecodable frames before the first that decodes
        for i, (c, t, raw) in enumerate(r.messages(connections=cs)):
            if last is not None and i > last:
                break
            if base is not None and keep is not None and i not in keep:
                continue
            msg = r.deserialize(raw, c.msgtype)
            if base is None:  # the first message sets the time origin, kept or not
                why = unusable(msg)
                if why:
                    raise SystemExit(f"{topic}: {why}")
                base = stamp(msg) if clock == "header" else t
            if keep is not None and i not in keep:
                continue
            ti = ((stamp(msg) if clock == "header" else t) - base) / 1e9
            try:
                rgb = decode(msg, width)
            except (SystemExit, Exception) as e:  # a corrupt message, also pillow's DecompressionBombError
                print(f"warning: cannot decode message {i} of {topic} ({e}); it counts as a black frame", flush=True)
                if shape is None:
                    pending.append((i, ti, e))
                    continue
                rgb = np.zeros(shape, np.uint8)
            if shape is None:
                shape = rgb.shape
                for j, tj, _ in pending:
                    yield j, tj, np.zeros(shape, np.uint8)
            yield i, ti, rgb
        if pending and shape is None:
            raise SystemExit(f"cannot decode message {pending[0][0]} of {topic}: {pending[0][2]}")


def camera_model(path, topic):
    """The intrinsics of a camera topic's images from the first message of its camera_info
    (info_topic), or None without a usable one: {"K": 3x3 list, "D": list, "model": None or the
    distortion model, "R": 3x3 list, "size": [width, height] of the topic's first image, "source":
    where they come from}. Project R @ X with lidar_scale.project(X, K, D, model, size).

    The matrix. A topic with a "rect" segment in its name uses P's left 3x3 and no distortion (K when
    P is unusable and D is all zero). Another topic with an all-zero D uses K and no distortion (P when
    K is unusable). Any other topic uses K, D and the model, named as lidar_scale.distortion() names
    them (plumb_bob, rational_polynomial, equidistant and their aliases). An unknown model raises
    ValueError. A matrix is unusable when fx or fy is under MIN_FOCAL of the camera_info's long side
    (a zero or identity placeholder), the principal point is outside the image or the last row is not
    [0, 0, 1].

    R turns points from the camera's TF frame into the frame that K projects. It is camera_info's R (a
    stereo pair's rectification) when K comes from P, else the identity.

    The size. The matrix is moved to the camera_info's roi and divided by its binning. When the
    camera_info is for another image size with the same aspect ratio (a resized republish), the
    matrix is scaled to the topic's images; another aspect ratio raises ValueError. Pixel centres are
    at integers, as in mvsam3d.cameras(). A camera_info of size 0x0 is taken to be for the images'
    size."""
    with open_bag(path) as r:
        cs = connections(r, topic)
        itopic = info_topic(topic)
        got = first_messages(r, [c for c in r.connections if c.topic == itopic and c.msgtype == INFO])
        if not got:
            return None
        ci = r.deserialize(got[0][2], got[0][0].msgtype)
        size = image_size(r, cs, topic)
    K, P, D = info_array(ci, "k"), info_array(ci, "p"), info_array(ci, "d")
    K = K.reshape(3, 3) if K.size == 9 else None
    P = P.reshape(3, 4)[:, :3] if P.size == 12 else None
    bx, by = max(int(ci.binning_x), 1), max(int(ci.binning_y), 1)
    x0, y0 = int(ci.roi.x_offset), int(ci.roi.y_offset)
    full = (int(ci.width), int(ci.height)) if ci.width and ci.height else (size[0] * bx, size[1] * by)
    expect = ((int(ci.roi.width) or full[0] - x0) // bx, (int(ci.roi.height) or full[1] - y0) // by) \
        if ci.width and ci.height else tuple(size)

    def fit(M):
        """M for the topic's images, or None when it is unusable."""
        if M is None or not np.isfinite(M).all() or not np.allclose(M[2], [0, 0, 1]):
            return None
        if min(M[0, 0], M[1, 1]) < MIN_FOCAL * max(full):  # a zero or identity placeholder
            return None
        if not (0 <= M[0, 2] <= full[0] and 0 <= M[1, 2] <= full[1]):
            return None
        M = M.copy()
        M[0, 2] -= x0
        M[1, 2] -= y0
        return rescale(M, 1 / bx, 1 / by)

    rect, plain = named_rect(topic), not D.any()
    model, coeffs = None, []
    if rect or plain:
        tries = [(P, "P (rectified)"), (K, "K (no distortion)")]
        if not rect:
            tries.reverse()
        if not plain:  # a rect topic whose D is not zero: P only
            tries = tries[:1]
        M, how = None, None
        for m, h in tries:
            M, how = fit(m), h
            if M is not None:
                break
        if M is None:
            return None
    else:
        M, how = fit(K), None
        if M is None:
            return None
        try:
            model, coeffs = lidar_scale.distortion(D, ci.distortion_model)
        except ValueError as e:
            raise ValueError(f"{itopic}: {e}; use a rectified topic or give K and D in the calibration") from None
        how, coeffs = f"K and D ({model})", coeffs.tolist()
    if tuple(expect) != tuple(size):
        sx, sy = size[0] / max(expect[0], 1), size[1] / max(expect[1], 1)
        if not (expect[0] > 0 and expect[1] > 0) or abs(sx / sy - 1) > ASPECT_TOL:
            raise ValueError(f"{itopic} is for {expect[0]}x{expect[1]} images but {topic} has {size[0]}x{size[1]} "
                             "images; give K and D in the calibration")
        M = rescale(M, sx, sy)
        how += f", scaled from {expect[0]}x{expect[1]}"
    rot = rotation(info_array(ci, "r")) if how.startswith("P") else np.eye(3)
    return {"K": M.tolist(), "D": coeffs, "model": model, "R": rot.tolist(), "size": list(size),
            "source": f"{itopic}: {how}"}


def image_size(r, cs, topic):
    """[width, height] of the first message of an image topic that describe() reads."""
    gen = r.messages(connections=cs)
    try:
        for c, _, raw in gen:
            try:
                w, h, _ = describe(r.deserialize(raw, c.msgtype))
            except Exception:  # a corrupt message
                continue
            return [int(w), int(h)]
    finally:
        gen.close()
    raise SystemExit(f"cannot decode any message of {topic}")


def rescale(M, sx, sy):
    """A 3x3 camera matrix for an image scaled by sx and sy, with pixel centres at integers:
    u' = (u + 0.5) sx - 0.5."""
    M = np.array(M, float)
    M[0] *= sx
    M[1] *= sy
    M[0, 2] += 0.5 * sx - 0.5
    M[1, 2] += 0.5 * sy - 0.5
    return M


def rotation(a):
    """A camera_info's R as a 3x3 rotation, or the identity when it is not one (an unset R is zero)."""
    if a.size == 9 and np.isfinite(a).all():
        R = a.reshape(3, 3)
        if np.linalg.norm(R @ R.T - np.eye(3)) < lidar_scale.ORTHONORMAL_TOL and np.linalg.det(R) > 0:
            return R
    return np.eye(3)


def first_messages(r, cs, n=1, keep=None):
    """[(connection, log_ns, raw)] of the first n messages of the connections cs. Reads no more. keep:
    how many of them keep their data (None = all); the others have None for raw, so that large
    messages read only for their log times are not held."""
    if not cs:  # no connections would mean all
        return []
    gen = r.messages(connections=cs)
    try:
        return [(c, t, raw if keep is None or i < keep else None) for i, (c, t, raw) in zip(range(n), gen)]
    finally:
        gen.close()


def lidar_connections(r, topic):
    cs = [c for c in r.connections if c.topic == topic]
    if not cs:
        raise SystemExit(f"the bag has no topic {topic}")
    if cs[0].msgtype not in LIDAR:
        raise SystemExit(f"{topic} is {cs[0].msgtype}, not a LiDAR topic (PointCloud2 or Livox CustomMsg)")
    return [c for c in cs if c.msgtype == cs[0].msgtype]


def lidar_topics(r):
    """(lidar, lidar_skipped) of an open bag, from the first message of each LiDAR topic (see the module
    docstring). Nothing here raises: a problem puts the topic under lidar_skipped."""
    lidar, skipped = [], []
    for topic in sorted({c.topic for c in r.connections if c.msgtype in LIDAR}):
        typ = None
        try:
            cs = lidar_connections(r, topic)
            typ = LIDAR[cs[0].msgtype]
            got = first_messages(r, cs, RATE_MESSAGES, keep=1)  # only the first is decoded
            if not got:
                skipped.append({"topic": topic, "type": typ, "why": "no messages"})
                continue
            c, t, raw = got[0]
            msg = r.deserialize(raw, c.msgtype)
            why = xyz_problem(msg.fields) if typ == "PointCloud2" else None
            if why:
                skipped.append({"topic": topic, "type": typ, "why": why})
                continue
            xyz, times, _, frame = decode_scan(msg, c.msgtype)
        except (SystemExit, Exception) as e:  # LiDAR problems never fail the listing
            skipped.append({"topic": topic, "type": typ, "why": f"cannot decode: {e}"})
            continue
        count = max(sum(x.msgcount for x in cs), len(got))
        lidar.append({"topic": topic, "type": typ, "msgtype": c.msgtype, "frame_id": frame, "points": len(xyz),
                      "fields": [f.name for f in msg.fields] if typ == "PointCloud2" else list(LIVOX_FIELDS),
                      "point_time": times is not None, "count": count, "rate_hz": span(t, got[-1][1], len(got))[1]})
    lidar.sort(key=lambda x: (-x["count"], x["topic"]))
    return lidar, skipped


def camera_axes(frame_id, edges=()):
    """The axes of camera_axes_from alone: "optical", "body" or None."""
    return camera_axes_from(frame_id, edges)[0]


def camera_axes_from(frame_id, edges=()):
    """(axes, rule) of a camera's frame: "optical" (x right, y down, z forward) or "body" (x forward, y
    left, z up), and the rule that decided. "name": the frame's name has "optical". "tf": the name does
    not, but one of the frame's TF links (edges, see transforms) is the standard body-to-optical rotation
    (lidar_scale.R_OPT_LINK) within AXES_TOL_DEG, as a Kalibr or EuRoC cam0 under base_link is; or the
    frame hangs by a static link within AXES_TOL_DEG of no turn from a frame that has optical axes by its
    name or by such a static link, as Kalibr's cam1 under cam0 does. Only that one step is taken, on
    static links, so the values of a moving link (odom to base_link) say nothing about a mount. A sibling
    sensor mounted at exactly R_OPT_LINK from the camera's parent still makes a body camera optical.
    "default": neither, so body axes. (None, None) for an empty frame name. edges may hold many values of
    one link: each neighbour is looked at once."""
    name = frame_name(frame_id)
    if not name:
        return None, None
    if "optical" in name.lower():
        return "optical", "name"
    least = 1 + 2 * np.cos(np.radians(AXES_TOL_DEG))  # the trace of a turn by AXES_TOL_DEG

    def turns(f, static_only=False):  # (the other frame, R that maps its points into f's) of f's links
        for parent, child, T, static, _ in edges:
            if f in (parent, child) and (static or not static_only):
                R = np.asarray(T, float)[:3, :3]
                yield (child, R) if f == parent else (parent, R.T)

    def by_link(f, static_only=False):
        return any(np.trace(lidar_scale.R_OPT_LINK.T @ R) >= least for _, R in turns(f, static_only))

    if by_link(name):
        return "optical", "tf"
    near = {other for other, R in turns(name, True) if np.trace(R) >= least}
    if any("optical" in other.lower() or by_link(other, True) for other in near):
        return "optical", "tf"
    return "body", "default"


def tf_links(edges, frame_id, lidar):
    """{LiDAR topic: "static", "dynamic" or None} of a camera frame: how TF links it to each LiDAR's frame."""
    out = {}
    for x in lidar:
        got = tf_chain(edges, frame_id, x["frame_id"]) if frame_id and x["frame_id"] else None
        out[x["topic"]] = got[2] if got else None
    return out


def tf_listing(edges):
    """topics.json's "tf": one {parent, child, static, T_parent_child} per link, the last value of a
    static link and the first of a dynamic one, static links first."""
    seen = {}
    for parent, child, T, static, _ in edges:
        if static or (parent, child, static) not in seen:
            seen[(parent, child, static)] = T
    return [{"parent": p, "child": c, "static": s, "T_parent_child": np.round(T, 6).tolist()}
            for (p, c, s), T in sorted(seen.items(), key=lambda x: (not x[0][2], x[0][0], x[0][1]))]


def xyz_problem(fields):
    """Why a PointCloud2 with these fields has no points to use (no float x, y and z), else None."""
    have = {}
    for f in fields:
        have.setdefault(f.name, f)
    if all(n in have and have[n].datatype in (7, 8) and have[n].count <= 1 for n in "xyz"):
        return None
    return f"no float x, y and z fields (fields: {', '.join(f.name for f in fields) or 'none'})"


def decode_pointcloud2(msg):
    """(xyz float32 (N, 3), t_ns int64 (N,) or None, header_ns, frame_id) of a PointCloud2. The fields'
    offsets, point_step, row_step (row padding), height and is_bigendian set the layout. A row_step
    shorter than a row (0 from some converters) counts as no padding when there is one row or the data
    has no room for padding. Points with a coordinate that is not finite and points at (0, 0, 0) are
    dropped. t_ns: each point's time, from the first of "t" (uint32, ns after the stamp), "time"
    (float32, s after the stamp) and "timestamp" (float64, absolute s or ns) whose times lie within
    POINT_TIME_S of the stamp for more than half of the points, else None (point_times). With times,
    the points whose time is not finite or lies further from the stamp are dropped too."""
    why = xyz_problem(msg.fields)
    if why:
        raise ValueError(why)
    have = {}
    for f in msg.fields:
        have.setdefault(f.name, f)
    h, w, step, row = int(msg.height), int(msg.width), int(msg.point_step), int(msg.row_step)
    names = ["x", "y", "z"] + [n for n, kind, _ in POINT_TIMES
                               if n in have and have[n].datatype == kind and have[n].count <= 1]
    for n in names:
        f = have[n]
        if f.offset + np.dtype(POINT_TYPES[f.datatype]).itemsize > step:
            raise ValueError(f"field {n} at offset {f.offset} does not fit in a point of {step} bytes")
    head, frame = stamp(msg), frame_name(msg.header.frame_id)
    if not (h and w):
        return np.zeros((0, 3), np.float32), None, head, frame
    data = np.asarray(msg.data, np.uint8).reshape(-1)
    if row < w * step and (h == 1 or data.size == h * w * step):  # some converters leave row_step 0
        row = w * step
    if row < w * step or data.size < (h - 1) * row + w * step:
        raise ValueError(f"a {w}x{h} cloud of {step}-byte points with row_step {row} does not fit in "
                         f"{data.size} bytes")
    if data.size < h * row:  # the last row's padding may be missing
        data = np.concatenate([data, np.zeros(h * row - data.size, np.uint8)])
    end = ">" if msg.is_bigendian else "<"
    dt = np.dtype({"names": names, "formats": [end + POINT_TYPES[have[n].datatype] for n in names],
                   "offsets": [have[n].offset for n in names], "itemsize": step})
    pts = np.ascontiguousarray(data[:h * row].reshape(h, row)[:, :w * step]).view(dt).reshape(-1)
    with np.errstate(over="ignore", invalid="ignore"):  # float64 beyond float32's range becomes inf
        xyz = np.stack([pts["x"], pts["y"], pts["z"]], 1).astype(np.float32)
    ok = np.isfinite(xyz).all(1) & (xyz != 0).any(1)
    t, timed = point_times(pts[ok], names, head)
    return xyz[ok][timed], None if t is None else t[timed], head, frame


def point_times(pts, names, head):
    """(per-point times in ns, which points have one) of a PointCloud2's points (see decode_pointcloud2),
    or (None, all true). A field gives times when more than half of its points lie within POINT_TIME_S
    of the stamp; one bad time (NaN, or a slow LiDAR's jitter) then drops only its own point, not the
    times of the whole scan."""
    for name, _, kind in POINT_TIMES:
        if name not in names:
            continue
        v = pts[name].astype(np.float64)
        if kind == "offset_ns":
            offsets = [v * 1e-9]
        elif kind == "offset_s":
            offsets = [v]
        else:  # absolute in s or in ns: the unit that puts the times near the stamp (sim time is small in ns too)
            offsets = [v - head / 1e9, (v - head) * 1e-9]
        for off in offsets:
            with np.errstate(invalid="ignore"):
                near = np.abs(off) <= POINT_TIME_S
            if 2 * near.sum() <= len(near):
                continue
            if kind == "offset_ns":
                return head + pts[name].astype(np.int64), near
            return head + np.round(np.where(near, off, 0.0) * 1e9).astype(np.int64), near
    return None, np.ones(len(pts), bool)


def decode_livox(msg):
    """(xyz float32 (N, 3), t_ns int64 (N,), header_ns, frame_id) of a Livox CustomMsg. Points whose
    tag has (tag & 0x30) other than 0x00 or 0x10, points at (0, 0, 0) and points with a coordinate that
    is not finite are dropped. A point's time is timebase + offset_time (ns). Some drivers stamp the
    header by the ROS clock and leave timebase on the LiDAR's own clock: when timebase is further than
    TIMEBASE_S from a set header stamp, the times are header stamp + offset_time, in the header's
    clock like every other stamp."""
    a = np.array([(p.offset_time, p.x, p.y, p.z, p.tag) for p in msg.points], np.float64).reshape(-1, 5)
    xyz = a[:, 1:4].astype(np.float32)
    ok = np.isin(a[:, 4].astype(np.int64) & 0x30, LIVOX_TAGS) & np.isfinite(xyz).all(1) & (xyz != 0).any(1)
    head, base = stamp(msg), int(msg.timebase)
    if head and abs(base - head) > TIMEBASE_S * 1e9:
        base = head
    return xyz[ok], base + a[ok, 0].astype(np.int64), head, frame_name(msg.header.frame_id)


def decode_scan(msg, msgtype):
    """decode_pointcloud2 or decode_livox, by the message type."""
    return decode_pointcloud2(msg) if LIDAR[msgtype] == "PointCloud2" else decode_livox(msg)


def merge_windows(windows):
    """Sorted log-time windows [(start_ns, stop_ns)] with the overlapping ones merged; empty ones are dropped."""
    out = []
    for a, b in sorted((int(a), int(b)) for a, b in windows if b > a):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def scans(path, topic, windows):
    """Yield (log_ns, header_ns, xyz, t_ns) of a LiDAR topic's messages within the log-time windows
    [(start_ns, stop_ns)], stop excluded, in log-time order (decode_pointcloud2, decode_livox). The
    windows are merged where they overlap and each one is one indexed read, so the rest of the bag is
    not read. A message that cannot be decoded is skipped with a warning. A scan without points is
    skipped too: it gives no samples, and an empty Livox message has a zero stamp."""
    spans = merge_windows(windows)
    with open_bag(path) as r:
        cs = lidar_connections(r, topic)
        for a, b in spans:
            for c, t, raw in r.messages(connections=cs, start=a, stop=b):
                try:
                    xyz, times, head, _ = decode_scan(r.deserialize(raw, c.msgtype), c.msgtype)
                except Exception as e:
                    print(f"warning: cannot decode the {topic} message at log time {t} ns ({e}); it is skipped",
                          flush=True)
                    continue
                if len(xyz):
                    yield t, head, xyz, times


def transform_matrix(tf):
    """The 4x4 matrix of a geometry_msgs/Transform, or None for a zero or non-finite quaternion."""
    q, p = tf.rotation, tf.translation
    q = np.array([q.w, q.x, q.y, q.z], float)
    n = np.linalg.norm(q)
    if not (np.isfinite(n) and n > 1e-9) or not np.isfinite([p.x, p.y, p.z]).all():
        return None
    w, x, y, z = q / n
    T = np.eye(4)
    T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    T[:3, 3] = p.x, p.y, p.z
    return T


def invert(T):
    """The inverse of a rigid 4x4 transform."""
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def transforms(r, windows=None):
    """TF edges [(parent, child, T_parent_child 4x4, static, t_ns)] of an open bag: every /tf_static
    message, and the /tf messages within the log-time windows [(start_ns, stop_ns)] (None = all of
    /tf). Namespaced TF topics (/robot/tf_static, /robot/tf) count too. T_parent_child maps child
    points into the parent frame. Frame names lose a leading "/"; t_ns is the transform's header stamp,
    else the log time. Transforms with a zero quaternion and those to their own frame are left out; a
    message that cannot be decoded is skipped with a warning."""
    out = []
    for static in (True, False):
        cs = [c for c in r.connections if tf_static(c) is static]
        if not cs:
            continue
        for a, b in [(None, None)] if static or windows is None else merge_windows(windows):
            for c, t, raw in r.messages(connections=cs, start=a, stop=b):
                out += tf_edges(r, c, t, raw)
    return out


def tf_static(c):
    """True for a connection of /tf_static, False for one of /tf (namespaced ones too: /robot/tf), else None."""
    if c.msgtype not in TF:
        return None
    return {"tf_static": True, "tf": False}.get(c.topic.rstrip("/").rsplit("/", 1)[-1])


def tf_edges(r, c, t, raw):
    """The TF edges (see transforms) of one /tf or /tf_static message logged at t (ns); none, with a
    warning, when it cannot be decoded."""
    try:
        msg = r.deserialize(raw, c.msgtype)
        got = [(frame_name(x.header.frame_id), frame_name(x.child_frame_id), transform_matrix(x.transform), stamp(x))
               for x in msg.transforms]
    except Exception as e:  # also a TF type of another definition
        print(f"warning: cannot decode the {c.topic} message at log time {t} ns ({e})", flush=True)
        return []
    static = tf_static(c)
    return [(parent, child, T, static, s or t) for parent, child, T, s in got
            if T is not None and parent and child and parent != child]


def tf_chain(edges, target, source, t_ns=None):
    """(T_target_source 4x4, [target, ..., source], "static" or "dynamic") of the shortest TF path from
    the target frame to the source frame, or None. T_target_source maps source points into the target
    frame. The search goes both ways along the edges: from a child to its parent uses the inverted
    transform. First it tries the static edges alone ("static"). If they give no path, it adds the
    dynamic edges ("dynamic"). A dynamic link uses its value nearest t_ns (None: the earliest). A static
    value replaces a dynamic one of the same link. A static link given twice uses its last value."""
    target, source = frame_name(target), frame_name(source)
    if not (target and source):
        return None
    if target == source:
        return np.eye(4), [target], "static"
    static, dynamic = {}, {}
    for parent, child, T, is_static, t in edges:
        if is_static:
            static[(parent, child)] = np.asarray(T, float)  # the last one counts, as in tf2
        else:
            dynamic.setdefault((parent, child), []).append((t, np.asarray(T, float)))
    nearest = {k: min(v, key=lambda x: x[0] if t_ns is None else abs(x[0] - t_ns))[1] for k, v in dynamic.items()}
    for kind, links in (("static", static), ("dynamic", {**nearest, **static} if nearest else None)):
        found = links and _path(links, target, source)
        if found:
            return found[0], found[1], kind
    return None


def _path(links, target, source):
    """(T_target_source, frames) of the shortest path over links {(parent, child): T}, or None."""
    adj = {}
    for (p, c), T in links.items():
        adj.setdefault(p, []).append((c, T))
        adj.setdefault(c, []).append((p, invert(T)))
    prev, queue = {target: None}, deque([target])
    while queue and source not in prev:
        f = queue.popleft()
        for g, T in adj.get(f, ()):
            if g not in prev:
                prev[g] = (f, T)  # T maps g's points into f
                queue.append(g)
    if source not in prev:
        return None
    T, frames, f = np.eye(4), [source], source
    while prev[f] is not None:
        f, Tf = prev[f]
        T = Tf @ T
        frames.append(f)
    return T, frames[::-1]


def camera_to_lidar(cam_frame, lidar_frame, edges, t_ns=None, axes=None):
    """{"T_cam_lidar": 4x4, "chain": frames, "link": "static" or "dynamic", "camera_axes": "optical" or
    "body", "camera_axes_from": its rule} from TF, or None without a path (tf_chain). T_cam_lidar maps
    LiDAR points to the camera's optical axes (x right, y down, z forward): a camera frame with body axes
    gets lidar_scale.R_OPT_LINK on the left. axes: (camera_axes, its rule) to use; None decides them by
    camera_axes_from on the edges."""
    got = tf_chain(edges, cam_frame, lidar_frame, t_ns)
    if got is None:
        return None
    T, frames, link = got
    axes, rule = axes or camera_axes_from(cam_frame, edges)
    if axes == "body":
        T = T.copy()
        T[:3, :] = lidar_scale.R_OPT_LINK @ T[:3, :]
    return {"T_cam_lidar": T, "chain": frames, "link": link, "camera_axes": axes, "camera_axes_from": rule}
