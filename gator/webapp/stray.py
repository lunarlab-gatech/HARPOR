"""An iPhone LiDAR capture exported by Stray Scanner: unpacking the zip, its facts, its frames and its depth.

Runs in envs/web (PyAV, numpy, pillow). video.py calls it for a job whose job.json has "source":
{"kind": "stray"} (an uploaded .zip), and the lidar stage (lidar.py) reads the depth.

The export's format: https://github.com/strayrobots/scanner/blob/main/docs/format.md. Its layout, the unpack
limits, the pairing, the edit-list trap, the turn from gravity and the depth samples: build_plan.md note 12,
Stray Scanner source.

How Stray writes them (its DatasetEncoder, VideoEncoder and OdometryEncoder): every saved frame gets
a row of odometry.csv and a frame of rgb.mp4, numbered by one counter. depth/ and confidence/ are
named by that counter too, but a frame gets none when ARKit gave it no depth (and no confidence image
when ARKit gave no confidence map). The pose is q_WA q_AC with q_AC a half turn about x, so it is
camera-to-world in OpenCV camera axes. rgb.mp4 has no rotation tag. Its first frame is written one
tick before the writer's start time, so the mp4's edit list hides it; rgb.mp4 is read without the
edit list (OPEN), else every frame would be paired with the next frame's depth and pose.

  unpack          the zip, checked as untrusted, into dest/export; dest/.unpacked.json makes a second call
                  with the same zip return at once
  probe           the export's facts; refuses an export that cannot be paired frame by frame
  upright_turns   k, the counterclockwise quarter turns from gravity: the circular mean of up in the image
                  over the frames, and the k that brings it closest to (0, -1), the top of the image
  frames          the frames of rgb.mp4, turned upright by k
  intrinsics      each frame's K, from odometry.csv, else camera_matrix.csv; only the LiDAR points of the
                  Inputs tab use it, never the scale
  depth_points    a frame's LiDAR samples [u, v, z] in pixels of the upright frame, and with K their camera
                  points and the frame's whole scan
"""

import csv
import os
import re
import shutil
import stat
import struct
import zipfile
import zlib
from pathlib import Path, PurePosixPath

import numpy as np

from gator.fileio import read_json, write_json

FILES = ("rgb.mp4", "odometry.csv", "camera_matrix.csv", "imu.csv")  # extracted from the export's root
FRAME_FILE = re.compile(r"(depth|confidence)/[0-9]{6}\.png")  # and these under it (ASCII digits only)
ODOMETRY = ("timestamp", "frame", "x", "y", "z", "qx", "qy", "qz", "qw", "fx", "fy", "cx", "cy")  # columns read
ROOT = "export"  # dest/export is the unpacked export
MAX_ENTRIES = 200_000
DIR_MAX = 64 << 20  # bytes of the zip's central directory (a real export's entries take about 130 bytes each)
METHODS = (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)  # zipfile limits the output of these as it reads
MAX_RATIO = 200  # of a member's bytes to its compressed bytes
TOTAL_FACTOR, TOTAL_SLACK, TOTAL_MAX = 4, 1 << 30, 40 << 30  # unpacked total <= min(4 x zip + 1 GB, 40 GB)
CHUNK = 1 << 20
OPEN = {"ignore_editlist": "1"}  # rgb.mp4 is read with every frame it holds (see the module docstring)
MIN_DEPTH_SHARE = 0.5  # an export with depth images for fewer of its frames is refused
MAX_DISAGREE = 0.2  # a warning when more frames than this point up another way
MIN_TILT = 0.1  # a frame whose up lies less than this in the image plane does not vote (camera straight down)
DEPTH_UNITS = 1000.0  # depth PNGs hold millimetres
MAX_DEPTH_PIXELS = 1 << 20  # a phone's depth images are 256x192; larger ones are refused before they are decoded
MIN_CONF = 2  # ARKit's high confidence
Z_RANGE = (0.1, 5.0)  # m; the LiDAR's useful range
MAX_CAMERA_MATRIX = 4096  # bytes of camera_matrix.csv; more is not one


def _size(n):
    return f"{n / 2**30:.1f} GB" if n >= 2**30 else f"{n / 2**20:.1f} MB" if n >= 2**20 else f"{n} bytes"


def _unsafe(name):
    """Why an entry's name could write outside the folder, else None."""
    if "\\" in name:
        return "a backslash"
    if name.startswith("/"):
        return "an absolute path"
    if re.match(r"[A-Za-z]:", name):
        return "a drive letter"
    if ".." in name.split("/"):
        return '".."'
    if "\0" in name:
        return "a NUL byte"
    return None


def _junk(name):
    return name.startswith("__MACOSX/") or PurePosixPath(name).name.startswith("._")


def _too_many():
    return f"the zip has more than {MAX_ENTRIES} entries; zip one Stray Scanner export"


def _entries(f):
    """The number of entries in the central directory of the zip open as f, counted up to MAX_ENTRIES + 1
    the way zipfile walks it, from zipfile's own end record. zipfile builds an object for every entry
    before anyone can count them, so unpack() calls this first. SystemExit for a directory of more than
    DIR_MAX bytes; zipfile.BadZipFile for a file without an end record."""
    try:
        end = zipfile._EndRecData(f)
    except OSError:
        end = None
    if not end:
        raise zipfile.BadZipFile("File is not a zip file")
    size = end[zipfile._ECD_SIZE]
    if size > DIR_MAX:
        raise SystemExit(f"the zip's list of entries takes {_size(size)}, more than {_size(DIR_MAX)}; zip one Stray "
                         "Scanner export")
    start = end[zipfile._ECD_LOCATION] - size  # as zipfile finds it
    if end[zipfile._ECD_SIGNATURE] == zipfile.stringEndArchive64:
        start -= zipfile.sizeEndCentDir64 + zipfile.sizeEndCentDir64Locator
    if start < 0:
        raise zipfile.BadZipFile("Bad offset for central directory")
    f.seek(start)
    data = f.read(size)
    n = pos = 0
    while pos + zipfile.sizeCentralDir <= len(data) and n <= MAX_ENTRIES:
        pos += zipfile.sizeCentralDir + sum(struct.unpack_from("<3H", data, pos + 28))  # name, extra, comment
        n += 1
    return n


def members(infos, zip_bytes):
    """[(ZipInfo, path under the root)] to extract, after every check (build_plan.md note 12, Stray Scanner
    source 2). SystemExit with the reason for a zip that is not a Stray Scanner export or not safe."""
    if len(infos) > MAX_ENTRIES:
        raise SystemExit(_too_many())
    for i in infos:
        why = _unsafe(i.filename)
        if why:
            raise SystemExit(f"the zip has an entry with {why} ({i.filename[:200]!r}); zip the export folder again")
        if stat.S_ISLNK(i.external_attr >> 16):
            raise SystemExit(f"the zip has a symlink ({i.filename[:200]!r}); zip the export folder again")
    kept = [i for i in infos if not i.is_dir() and not _junk(i.filename)]
    videos = [i.filename for i in kept if PurePosixPath(i.filename).name == "rgb.mp4"]
    if not videos:
        raise SystemExit("the zip has no rgb.mp4; zip the Stray Scanner export folder (with rgb.mp4, depth/ and "
                         "odometry.csv)")
    if len(videos) > 1:
        raise SystemExit(f"the zip has {len(videos)} rgb.mp4 files ({', '.join(videos[:3])}); zip one Stray Scanner "
                         "export")
    prefix = videos[0][:-len("rgb.mp4")]
    out, seen = [], set()
    for i in kept:
        rel = i.filename[len(prefix):]
        if not i.filename.startswith(prefix) or not (rel in FILES or FRAME_FILE.fullmatch(rel)):
            continue
        if rel in seen:
            raise SystemExit(f"the zip holds {i.filename} twice; zip the export folder again")
        seen.add(rel)
        if i.flag_bits & 0x1:
            raise SystemExit(f"{i.filename} is encrypted in the zip; zip the export without a password")
        if i.compress_type not in METHODS:
            raise SystemExit(f"{i.filename} uses compression method {i.compress_type}, not deflate; zip the export "
                             "again with the usual (deflate) compression")
        if i.file_size > MAX_RATIO * i.compress_size:
            raise SystemExit(f"{i.filename} unpacks to more than {MAX_RATIO} times its compressed size; this is not a "
                             "Stray Scanner export")
        out.append((i, rel))
    total, cap = sum(i.file_size for i, _ in out), limit(zip_bytes)
    if total > cap:
        raise SystemExit(f"unpacked, the export would take {_size(total)}, more than {_size(cap)} for a "
                         f"{_size(zip_bytes)} zip; this is not a Stray Scanner export")
    return out


def limit(zip_bytes):
    """The most bytes that a zip of zip_bytes may unpack to."""
    return min(TOTAL_FACTOR * zip_bytes + TOTAL_SLACK, TOTAL_MAX)


def _extract(z, info, out, left):
    """Stream one member into a new file (0600), counting its real bytes. Returns them."""
    out.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    n = 0
    with z.open(info) as src, os.fdopen(os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as f:
        while chunk := src.read(CHUNK):
            n += len(chunk)
            if n > info.file_size:
                raise SystemExit(f"{info.filename} unpacks to more bytes than the zip says; this is not a Stray "
                                 "Scanner export")
            if n > left:
                raise SystemExit("the export unpacks to more bytes than the zip allows; this is not a Stray Scanner "
                                 "export")
            f.write(chunk)
    return n


def unpack(zip_path, dest):
    """The unpacked export's root, dest/export (build_plan.md note 12, Stray Scanner source 2). Unpacks the
    zip only when dest/.unpacked.json is for another zip (bytes, mtime_ns) or a member is missing. SystemExit
    with the reason for a zip that is not a safe Stray Scanner export, or when the disk cannot take it."""
    zip_path, dest = Path(zip_path), Path(dest)
    root, tmp = dest / ROOT, dest / ".tmp"
    try:
        st = zip_path.stat()
        done = read_json(dest / ".unpacked.json")
        if isinstance(done, dict) and done.get("bytes") == st.st_size and done.get("mtime_ns") == st.st_mtime_ns \
                and isinstance(done.get("members"), list) and all((root / m).is_file() for m in done["members"]):
            return root
        dest.mkdir(parents=True, exist_ok=True, mode=0o700)
        (dest / ".unpacked.json").unlink(missing_ok=True)
        for old in (tmp, root):  # an earlier zip's export, or one cut short
            shutil.rmtree(old, ignore_errors=True)
        with open(zip_path, "rb") as f:
            if _entries(f) > MAX_ENTRIES:
                raise SystemExit(_too_many())
            with zipfile.ZipFile(f) as z:
                todo = members(z.infolist(), st.st_size)
                tmp.mkdir(mode=0o700)
                total, cap = 0, limit(st.st_size)
                for info, rel in todo:
                    total += _extract(z, info, tmp / rel, cap - total)
        os.replace(tmp, root)
        rec = {"bytes": st.st_size, "mtime_ns": st.st_mtime_ns, "root": ROOT, "members": sorted(rel for _, rel in todo)}
        write_json(dest / ".unpacked.json", rec, indent=None)  # last: the done marker
        return root
    except (zipfile.BadZipFile, zlib.error, EOFError, NotImplementedError, UnicodeDecodeError) as e:
        shutil.rmtree(tmp, ignore_errors=True)
        raise SystemExit(f"{zip_path.name} is not a readable zip: {e}") from None
    except OSError as e:  # a full disk, say
        shutil.rmtree(tmp, ignore_errors=True)
        raise SystemExit(f"cannot unpack {zip_path.name}: {e}") from None
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def read_odometry(root):
    """odometry.csv as arrays, its rows in frame order: frame (int), timestamp, position (N, 3), quat (N, 4;
    qx, qy, qz, qw) and intrinsics (N, 4; fx, fy, cx, cy) or None. SystemExit when it is missing or cannot
    be read: it comes from the user's zip, so it may not be UTF-8 text or may hold a field longer than the
    csv module reads."""
    f = Path(root) / "odometry.csv"
    if not f.is_file():
        raise SystemExit("the export has no odometry.csv, so its frames cannot be turned upright; this export is "
                         "not supported")
    try:
        with open(f, newline="", encoding="utf-8-sig") as fh:
            rows = [r for r in csv.reader(fh) if any(x.strip() for x in r)]
    except UnicodeDecodeError:
        raise SystemExit("odometry.csv is not UTF-8 text; this export is not supported") from None
    except csv.Error as e:
        raise SystemExit(f"odometry.csv cannot be read: {e}; this export is not supported") from None
    head = [h.strip() for h in rows[0]] if rows else []
    missing = [c for c in ("frame", "qx", "qy", "qz", "qw") if c not in head]
    if missing:
        raise SystemExit(f"odometry.csv has no {', '.join(missing)} column; this export is not supported")
    used = [h for h in ODOMETRY if h in head]
    try:
        data = np.array([[float(r[head.index(h)]) for h in used] for r in rows[1:]], float).reshape(-1, len(used))
    except (ValueError, IndexError):
        raise SystemExit("odometry.csv has a row that is not all numbers; this export is not supported") from None
    data = data[np.argsort(data[:, used.index("frame")], kind="stable")]

    def cols(*names):
        return data[:, [used.index(n) for n in names]] if all(n in used for n in names) else None

    t = cols("timestamp")
    return {"frame": data[:, used.index("frame")].astype(int), "timestamp": None if t is None else t[:, 0],
            "position": cols("x", "y", "z"), "quat": cols("qx", "qy", "qz", "qw"),
            "intrinsics": cols("fx", "fy", "cx", "cy")}


def _turns(d):
    """The counterclockwise quarter turns (0 to 3) that bring each image direction d (..., 2) closest to
    (0, -1); a turn maps (dx, dy) to (dy, -dx)."""
    d = np.asarray(d, float)
    return np.argmax(np.stack([-d[..., 1], d[..., 0], d[..., 1], -d[..., 0]], -1), -1)


def upright_turns(quats):
    """(k, agreement): the counterclockwise quarter turns that bring the frames upright, from gravity
    (see the module docstring), and the share of the frames whose own up gives the same k. quats are
    camera-to-world rotations (qx, qy, qz, qw) in OpenCV camera axes, in a world with +y up (Stray's
    OdometryEncoder turns ARKit's camera axes by q_AC, a half turn about x). Frames that look straight
    up or down, and empty poses, do not vote; when no frame votes, k is 0 and the agreement None."""
    q = np.asarray(quats, float).reshape(-1, 4)
    size = np.linalg.norm(q, axis=1)
    q = q / np.maximum(size, 1e-12)[:, None]
    x, y, z, w = q.T
    up = np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z)], 1)  # R_wc^T (0, 1, 0), its x and y
    norm = np.linalg.norm(up, axis=1)
    use = (norm > MIN_TILT) & (size > 1e-6)  # an all-zero (or NaN) row is no pose
    if not use.any():
        return 0, None
    d = up[use] / norm[use, None]
    k = int(_turns(d.mean(0)))
    return k, float(np.mean(_turns(d) == k))


def turns(info):
    """The counterclockwise quarter turns of probe()'s rotation."""
    return (int(info["rotation"]) // 90) % 4


def _indices(root, folder):
    """The frame numbers of the images in root/folder (depth or confidence)."""
    d = Path(root) / folder
    return {int(p.name[:6]) for p in d.iterdir() if FRAME_FILE.fullmatch(f"{folder}/{p.name}")} if d.is_dir() else set()


def probe(root):
    """video.probe of rgb.mp4 (read with OPEN) with kind "stray", rotation 90 k from gravity
    (rotation_source "gravity", container_rotation the mp4's own), the width and height after that turn,
    n_depth, n_confidence, n_odometry, depth_size, upright_agreement (None when no frame shows gravity
    in the image) and warnings. SystemExit when the export cannot be paired frame by frame: no
    odometry.csv, another number of rows than rgb.mp4 has frames, frame numbers that are not 0 to N - 1,
    no depth image, a depth image past the last frame, depth images for fewer than MIN_DEPTH_SHARE of
    the frames, or a depth image larger than a phone's."""
    from PIL import Image

    from gator.webapp import video

    root = Path(root)
    odo = read_odometry(root)
    info = video.probe(root / "rgb.mp4", options=OPEN)
    n, n_odo = info["n_frames"], len(odo["frame"])
    if n_odo != n:
        raise SystemExit(f"rgb.mp4 has {n} frames and odometry.csv has {n_odo} rows; this export is not supported")
    if not np.array_equal(odo["frame"], np.arange(n)):
        raise SystemExit(f"odometry.csv's frame column is not 0 to {n - 1}, one row for each frame; this export is not "
                         "supported")
    depth = _indices(root, "depth")
    if not depth:
        raise SystemExit("the export has no depth images (depth/000000.png, ...); this export is not supported")
    if max(depth) >= n:
        raise SystemExit(f"depth/{max(depth):06d}.png is past the last of rgb.mp4's {n} frames; this export is not "
                         "supported")
    if len(depth) < MIN_DEPTH_SHARE * n:
        raise SystemExit(f"only {len(depth)} of rgb.mp4's {n} frames have a depth image; this export is not supported")
    first = f"depth/{min(depth):06d}.png"
    try:
        with Image.open(root / first) as im:
            depth_size = list(im.size)
    except Image.UnidentifiedImageError:  # its message holds the path on the server
        raise SystemExit(f"{first} cannot be read (it is not an image, or its header is damaged); this export is "
                         "not supported") from None
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as e:
        raise SystemExit(f"{first} cannot be read ({e}); this export is not supported") from None
    if depth_size[0] * depth_size[1] > MAX_DEPTH_PIXELS:
        raise SystemExit(f"{first} is {depth_size[0]}x{depth_size[1]}, larger than a phone's depth images; this "
                         "export is not supported")
    k, agree = upright_turns(odo["quat"])
    kc = (info["rotation"] // 90) % 4
    w, h = (info["width"], info["height"]) if kc % 2 == 0 else (info["height"], info["width"])  # as stored
    warnings = []
    if len(depth) < n:
        warnings.append(f"{n - len(depth)} of the {n} frames have no depth image, so they have no LiDAR samples")
    conf = _indices(root, "confidence")
    if not conf:
        warnings.append("the export has no confidence/ images, so the LiDAR scale uses every depth pixel")
    elif depth - conf:
        warnings.append(f"{len(depth - conf)} of the {len(depth)} depth images have no confidence image, so those "
                        "frames have no LiDAR samples")
    if agree is None:
        warnings.append("the camera looked straight up or down in every frame, so gravity gives no turn; the frames "
                        "are left as stored")
    elif agree < 1 - MAX_DISAGREE:
        warnings.append(f"only {agree:.0%} of the frames are upright when turned {90 * k} degrees (gravity in the "
                        "image); the phone turned during the capture, so the other frames come out sideways")
    if abs(depth_size[0] / depth_size[1] - w / h) > 0.01 * w / h:
        warnings.append(f"the depth images are {depth_size[0]}x{depth_size[1]} but the video is {w}x{h}, another "
                        "aspect ratio; the depth is scaled to the video in each direction")
    return {**info, "kind": "stray", "rotation": 90 * k, "rotation_source": "gravity",
            "container_rotation": info["rotation"], "width": w if k % 2 == 0 else h, "height": h if k % 2 == 0 else w,
            "n_depth": len(depth), "n_confidence": len(conf), "n_odometry": n_odo, "depth_size": depth_size,
            "upright_agreement": None if agree is None else round(agree, 4), "warnings": warnings}


def frames(root, info, keep=None, width=None):
    """video.frames of rgb.mp4 (read with OPEN), turned upright by probe()'s rotation instead of the
    container's."""
    from gator.webapp import video

    return video.frames(Path(root) / "rgb.mp4", keep=keep, width=width, turns=turns(info), options=OPEN)


def _pixels(path, rel):
    """(mode, pixels) of a depth or confidence image. ValueError for one of more than MAX_DEPTH_PIXELS,
    before it is decoded, and for one that cannot be decoded. Pillow raises OSError for a damaged image,
    SyntaxError for a PNG with a broken chunk, ValueError for some (a truncated pHYs chunk), and
    UnidentifiedImageError for a file that is not an image or a PNG whose header is damaged. The message
    names the image by rel, not by its path on the server."""
    from PIL import Image

    try:
        with Image.open(path) as im:
            size = im.size
            if size[0] * size[1] <= MAX_DEPTH_PIXELS:
                return im.mode, np.asarray(im)
    except Image.DecompressionBombError as e:
        raise ValueError(f"{rel}: {e}") from None
    except Image.UnidentifiedImageError:  # its message holds the path
        raise ValueError(f"{rel} cannot be read: it is not an image, or its header is damaged") from None
    except (OSError, SyntaxError, ValueError) as e:
        raise ValueError(f"{rel} cannot be read: {e}") from None
    raise ValueError(f"{rel} is {size[0]}x{size[1]}, larger than a phone's depth images")


def _camera_matrix(root):
    """camera_matrix.csv as a 3x3 matrix, or None when it is missing or not 3 rows of 3 finite numbers with
    the last row 0 0 1 and fx, fy > 0. It comes from the user's zip, so nothing else is trusted."""
    f = Path(root) / "camera_matrix.csv"
    try:
        if not f.is_file() or f.stat().st_size > MAX_CAMERA_MATRIX:
            return None
        rows = [r.split(",") for r in f.read_text(encoding="utf-8-sig").splitlines() if r.strip()]
        K = np.array([[float(x) for x in r] for r in rows], float)
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    if K.shape != (3, 3) or not np.isfinite(K).all() or not np.allclose(K[2], [0, 0, 1]):
        return None
    return K if K[0, 0] > 0 and K[1, 1] > 0 else None


def intrinsics(root, n_frames):
    """(K of each frame (n_frames, 3, 3), at the video's stored size, and its source: "odometry.csv" or
    "camera_matrix.csv"), or (None, None). odometry.csv's fx, fy, cx, cy are per frame; camera_matrix.csv
    (the last frame's) is the fallback for every frame."""
    if n_frames < 1:
        return None, None
    try:
        odo = read_odometry(root)
    except SystemExit:
        odo = None
    c = None if odo is None else odo["intrinsics"]
    if c is not None and len(c) == n_frames and np.array_equal(odo["frame"], np.arange(n_frames)) \
            and np.isfinite(c).all() and (c[:, :2] > 0).all():
        K = np.zeros((n_frames, 3, 3))
        K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = c[:, 0], c[:, 1], c[:, 2], c[:, 3], 1.0
        return K, "odometry.csv"
    K = _camera_matrix(root)
    return (np.repeat(K[None], n_frames, 0), "camera_matrix.csv") if K is not None else (None, None)


def _depth(root, i, min_conf):
    """(z in meters, confidence or None) of frame i's depth image: confidence only with min_conf (0: not read).
    ValueError when the images cannot be read or are not what the format says: depth in one 16-bit channel,
    confidence in one channel (8-bit as Stray writes it, or 16-bit) of the depth image's size."""
    root = Path(root)
    mode, z = _pixels(root / "depth" / f"{i:06d}.png", f"depth/{i:06d}.png")
    if mode not in ("I;16", "I;16B", "I"):
        raise ValueError(f"depth/{i:06d}.png is a {mode} image, not 16-bit depth")
    z = z.astype(np.float64) / DEPTH_UNITS
    if not min_conf:
        return z, None
    c = root / "confidence" / f"{i:06d}.png"
    if not c.is_file():
        raise ValueError(f"depth/{i:06d}.png has no confidence/{i:06d}.png")
    mode, conf = _pixels(c, f"confidence/{i:06d}.png")
    if mode not in ("L", "I;16", "I;16B", "I"):  # else the shapes below differ only in channels
        raise ValueError(f"confidence/{i:06d}.png is a {mode} image, not one-channel confidence")
    if conf.shape != z.shape:
        raise ValueError(f"confidence/{i:06d}.png is {conf.shape[1]}x{conf.shape[0]} but depth/{i:06d}.png is "
                         f"{z.shape[1]}x{z.shape[0]}")
    return z, conf


def _stored(u_d, v_d, depth_hw, rgb_wh, k):
    """Depth pixels to pixels of the frame as stored (size W x H, the upright frame of size rgb_wh turned back by
    k), and that size."""
    h_d, w_d = depth_hw
    W, H = (rgb_wh[0], rgb_wh[1]) if k % 2 == 0 else (rgb_wh[1], rgb_wh[0])
    return (u_d + 0.5) * (W / w_d) - 0.5, (v_d + 0.5) * (H / h_d) - 0.5, (W, H)


def _camera(u, v, z, K, k):
    """Stored pixels with depth z to camera points (N, 3) of the upright frame: ((u - cx) z / fx, (v - cy) z / fy, z),
    then each counterclockwise quarter turn maps (x, y) to (y, -x), as rot90_points maps the pixels."""
    x, y = (u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1]
    for _ in range(k % 4):
        x, y = y, -x
    return np.stack([x, y, z], 1).astype(np.float32)


def depth_points(root, i, rgb_wh, k, K, min_conf=MIN_CONF, z_range=Z_RANGE):
    """(samples, xyz, scan) of frame i's depth image, read once. samples: (N, 3) float32 [u, v, z], u, v in
    pixels of the upright frame of size rgb_wh (pixel centres at integers), z in meters; only the pixels with
    confidence >= min_conf (0: every pixel, without reading confidence/) and z within z_range. Depth pixel
    (u_d, v_d) is at (u_d + 0.5) W / w_d - 0.5 (and the same for v) in the stored frame of size W x H, then
    turned by k, the frame's counterclockwise quarter turns; the depth is not resampled. xyz (N, 3): the
    samples' pixels as camera points, row for row; scan (M, 3): every depth pixel with a depth (z > 0) at any
    confidence as camera points. K: the frame's intrinsics at the stored frame's size; without it (None), xyz
    and scan are None. Camera points are float32, in meters, in the upright frame's optical axes. ValueError
    when the images cannot be read or are not what the format says (_depth)."""
    from gator.stage_a import lidar_scale

    z, conf = _depth(root, i, min_conf)
    keep = (z >= z_range[0]) & (z <= z_range[1])
    if conf is not None:
        keep &= conf >= min_conf
    v_d, u_d = np.nonzero(keep)
    us, vs, size = _stored(u_d, v_d, z.shape, rgb_wh, k)
    u, v = lidar_scale.rot90_points(us, vs, size, k)
    samples = np.stack([u, v, z[keep]], 1).astype(np.float32)
    if K is None:
        return samples, None, None
    K = np.asarray(K, float)
    vw, uw = np.nonzero(z > 0)
    uw, vw, _ = _stored(uw, vw, z.shape, rgb_wh, k)
    return samples, _camera(us, vs, z[keep], K, k), _camera(uw, vw, z[z > 0], K, k)
