"""The metric scale from LiDAR: the web app's lidar stage (build_plan.md note 12).

Runs in envs/web (numpy, scipy, pillow, rosbags; CPU only), from the gator_clone root, after da3:

    python -m gator.webapp.lidar scale --job J

J/job.json "lidar" (steps.lidar) names the source. SOURCES has one function per kind that gives the
LiDAR samples of the pool frames: "rosbag" (rosbag_samples), a LiDAR topic of the job's bag, and
"stray" (stray_samples), the depth images of a Stray Scanner capture.

Samples. Every pool frame ("f00012", frame 12 of the input: message 12 of the camera topic, or frame
12 of a capture's rgb.mp4) gets an (N, 3) array [u, v, z]: the LiDAR points at the frame's time as
pixels of scene/images/<name>.png (pixel centres at integers, as lidar_scale takes them) and their
depth z in meters along the camera's optical axis. The frames that extract wrote but prep left out are
sampled too, so that a mask change finds them in the cache. The cache is J/lidar/samples.npz with
J/lidar/samples.json, keyed by the input file (name, bytes, mtime_ns) and what else the samples
depend on (a bag's topics, the calibration file's sha256 and the time window; a capture's turn, the
frames' size and the depth filters); a rerun with the same key reads no scans or depth images.

A rosbag's samples:
  extrinsic   T_cam_lidar from the uploaded calibration (job.json "calibration": the file
              J/lidar/calibration/<name>, read again with lidar_scale.read_calibration), else from the
              bag's TF from the camera topic's frame_id to the LiDAR's (rosbag.camera_to_lidar; a
              dynamic link is taken at each frame's camera stamp from the /tf values logged within
              that frame's read windows, with a warning when a frame has no /tf value of it within
              READ_S of its stamp), else the stage fails with "upload the calibration".
              T_cam_lidar is to the camera topic's frame, the unrectified optical frame, as TF's is:
              a rectified topic's camera_info R turns the points after it (a warning says so when the
              calibration gives T_cam_lidar but not K).
  intrinsics  the calibration's K, D and distortion_model (for its image_size, else for the frames'
              size), else the camera topic's camera_info (rosbag.camera_model, whose R turns the points
              first), else the stage fails. extract writes the frames at their full size, so the matrix
              fits them as it is; for another size of the same aspect ratio it is scaled.
  time        the frames' stamps come from frames/stamps.json (rosbag.probe again when it is missing or
              is for another topic). Scans are read only within +-READ_S of each frame's log time (and
              of that time plus time_offset_s, for a bag whose log times copy each sensor's header
              stamps). time_offset_s relates the header clocks (t_lidar = t_cam + time_offset_s). The
              time base is the header stamps when the camera's header stamps are usable, every scan has
              a stamp, and either time_offset_s is given (the log times then only check it) or the two
              clocks' log-minus-header offsets agree within CLOCK_TOL_S. Else frames and scans are
              matched by log time, the recorder's one clock, without time_offset_s and with a warning.
              A frame's stamp (plus time_offset_s in the header base) is matched to the points of the
              scans read for that frame: with per-point times, the points within +-WINDOW_S of it;
              without them (a PointCloud2 without a time field), every point of the nearest scan
              within one scan period (the median step between the scans read).
  projection  x_cam = T_cam_lidar x_lidar, then lidar_scale.project. While the scans are read, each
              keeps only the points that land in the image of a frame whose read window holds it, so a
              360 degree LiDAR does not fill the memory.

A Stray Scanner capture's samples (stray.depth_samples): the depth image depth/NNNNNN.png of each
frame, in millimetres, aligned with the color camera, so there is no extrinsic, no intrinsics and no
time matching. Each depth pixel with confidence 2 and z within 0.1 to 5 m is scaled from the depth
image's size to the frame's as stored, then turned upright by the turn that the frames got
(frames/meta.json "video" "rotation", from gravity). The zip is unpacked in J/video/stray (stray.unpack,
which the frames stage did already). A frame without a depth image, or with one but without its
confidence image, gets no samples, with a warning.
The record's source is {kind "stray", file, depth_size, frames (the frames sampled)}, its extrinsic
{source "stray (depth aligned with the color camera)"}, its intrinsics {source "stray: depth scaled to
the frame size", depth_size, frame_size, turns, min_confidence, z_range_m} and its time {base "frame
index"}.

The estimate is lidar_scale.estimate, with the check images in J/lidar/check. The record,
pool/lidar.json, gains source, extrinsic {source, file, calibration_sha256, chain, link, camera_axes,
T_cam_lidar}, intrinsics {source, K, D, model, R, size, frame_size}, time {base, window_ms, offset_ms,
log_minus_header_ms {camera, lidar}, scan_period_ms, scan_period_from, point_times} and the warnings of
both steps, and is written last. extrinsic.calibration_sha256 (null without a calibration) lets
steps.done see a record that an upload or a delete of the calibration made stale. When the samples
cannot be made (no TF link, no intrinsics, ...), the record has ok false and the reason. The stage
prints PROGRESS lines while it reads scans, then one line per view
(lidar_scale.summary), and exits 1 with the reason when the record is not ok.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path

import numpy as np

from gator.stage_a import lidar_scale

WINDOW_S = 0.05  # points within +-50 ms of a camera stamp
READ_S = 0.3  # scans are read within +-0.3 s of each frame's log time
CLOCK_TOL_S = 0.2  # the camera's and the LiDAR's log-minus-header offsets agree this well: the header base
ASPECT_TOL = 0.01  # intrinsics for another size are scaled to the frames when the aspect ratios agree this well
# A bag whose topics.json is older than the LiDAR listing (Runner.submit_analyze lists it again on a topic choice).
OLD_TOPICS = ("the bag was listed before LiDAR topics were; choose its camera topic again to list them (for a job in "
              "review or done: upload the bag again)")
CACHE_VERSION = 2  # of lidar/samples.json; another version is a cache miss (2: time_offset_s only in the header base)


class Failed(Exception):
    """The samples cannot be made: args[0] is the reason. info holds the record's fields known so far."""

    def __init__(self, reason, info=None):
        super().__init__(reason)
        self.info = info or {}


def tf_reason(camera_frame, lidar_frame, lidar_topic):
    """Why the bag's TF cannot bring the LiDAR's points to the camera; the fix is the calibration file."""
    if not camera_frame:
        return (f"the camera's images have no frame_id, so the bag's TF cannot link them to "
                f"{lidar_frame or lidar_topic}; upload the calibration")
    if not lidar_frame:
        return f"the scans of {lidar_topic} have no frame_id, so the bag's TF cannot link them; upload the calibration"
    return f"the bag's TF does not link {camera_frame} to {lidar_frame}; upload the calibration"


def write_json(path, data):
    tmp = path.with_name(".tmp_" + path.name)
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def progress(x):
    print(f"PROGRESS {x:.3f}", flush=True)


# ---- the samples cache ----

def cache_read(J, key, names):
    """{name: samples} and the info of lidar/samples.*, when its key is key and it has every name; else None."""
    d = Path(J) / "lidar"
    meta = read_json(d / "samples.json")
    if not isinstance(meta, dict) or meta.get("key") != key or not set(names) <= set(meta.get("frames") or []):
        return None
    try:
        with np.load(d / "samples.npz") as f:
            samples = {n: np.asarray(f[n], float) for n in meta["frames"]}
    except (OSError, ValueError, KeyError):
        return None
    return samples, meta.get("info") or {}


def cache_write(J, key, samples, info):
    """lidar/samples.npz, then lidar/samples.json (the key) last; the old key goes first."""
    d = Path(J) / "lidar"
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    (d / "samples.json").unlink(missing_ok=True)
    tmp = d / ".tmp_samples.npz"
    with open(tmp, "wb") as f:
        np.savez(f, **{n: np.asarray(a, np.float32).reshape(-1, 3) for n, a in samples.items()})
    os.replace(tmp, d / "samples.npz")
    write_json(d / "samples.json", {"key": key, "frames": sorted(samples), "info": info,
                                    "points": {n: int(len(a)) for n, a in sorted(samples.items())}})


# ---- the parts of a rosbag's samples ----

def frame_names(J, names):
    """The pool views and the other frames that extract wrote (frames/pool.json), sorted."""
    pool = read_json(Path(J) / "frames" / "pool.json") or {}
    extracted = pool.get("names") or {}
    extracted = extracted.values() if isinstance(extracted, dict) else extracted
    more = {n for n in extracted if isinstance(n, str) and (Path(J) / "scene" / "images" / f"{n}.png").is_file()}
    bad = [n for n in names if not re.fullmatch(r"f\d+", n)]
    if bad:
        raise Failed(f"pool views {', '.join(bad[:5])} are not named like f00012 (a frame number), so they are not "
                     "frames of the input")
    return sorted(set(names) | {n for n in more if re.fullmatch(r"f\d+", n)})


def calibration(J, job):
    """(the parsed calibration, its file's sha256, its name), or (None, None, None) without one."""
    cal = job.get("calibration")
    if not cal:
        return None, None, None
    name = str(cal.get("name") or "")
    f = Path(J) / "lidar" / "calibration" / name
    if not name or Path(name).name != name or not f.is_file():
        raise Failed(f"the calibration file {name or '(no name)'} is missing; upload it again")
    data = f.read_bytes()
    try:
        parsed = lidar_scale.read_calibration(data, name)
    except ValueError as e:
        raise Failed(f"the calibration: {e}") from None
    return parsed, hashlib.sha256(data).hexdigest(), name


def camera_stamps(J, path, topic):
    """{"clock", "header_ns", "log_ns"} of the camera topic's frames: frames/stamps.json, or rosbag.probe."""
    from gator.webapp import rosbag

    st = read_json(Path(J) / "frames" / "stamps.json")
    if isinstance(st, dict) and st.get("topic") == topic and isinstance(st.get("header_ns"), list) \
            and isinstance(st.get("log_ns"), list) and st.get("clock") in ("header", "log"):
        return st
    print("frames/stamps.json is missing or for another topic; reading the camera's stamps", flush=True)
    out = {}
    info = rosbag.probe(path, topic, stamps_out=out)
    return {"topic": topic, "clock": info["clock"], **out}


def fit_matrix(K, size, frame_size, source):
    """K for an image of size size, made to fit frames of frame_size (rosbag.rescale); Failed for
    another aspect ratio."""
    from gator.webapp import rosbag

    if tuple(size) == tuple(frame_size):
        return np.asarray(K, float)
    sx, sy = frame_size[0] / size[0], frame_size[1] / size[1]
    if abs(sx / sy - 1) > ASPECT_TOL:
        raise Failed(f"the intrinsics ({source}) are for {size[0]}x{size[1]} images but the frames are "
                     f"{frame_size[0]}x{frame_size[1]}; give K for the frames' size in the calibration")
    return rosbag.rescale(K, sx, sy)


def calibration_source(cal, name):
    """The intrinsics' source text for a calibration that gives K (also jobs.lidar_settings')."""
    model = cal.get("distortion_model")
    return f"calibration {name}: K" + (f" and D ({model})" if model else "")


def intrinsics(path, topic, info_topic, cal, cal_name, frame_size, warnings):
    """The intrinsics record {source, K, D, model, R, size, frame_size}: the calibration's K, else the
    camera topic's camera_info. Failed without either."""
    from gator.webapp import rosbag

    if cal is not None and cal.get("K") is not None:
        model, D = cal.get("distortion_model"), cal.get("D") or []
        size = cal.get("image_size") or list(frame_size)
        rec = {"source": calibration_source(cal, cal_name), "K": cal["K"], "D": D, "model": model, "R": None,
               "size": list(size)}
    else:
        if cal is not None and cal.get("D"):
            warnings.append("the calibration has D but no K, so the camera_info's intrinsics are used and its D is not")
        try:
            m = rosbag.camera_model(path, topic)
        except ValueError as e:
            raise Failed(str(e)) from None
        if m is None:
            raise Failed(f"{topic} has no usable camera_info ({info_topic or rosbag.info_topic(topic)}); give K "
                         "(and D) in the calibration")
        rec = {"source": m["source"], "K": m["K"], "D": m["D"], "model": m["model"], "R": m["R"], "size": m["size"]}
    fit_matrix(rec["K"], rec["size"], frame_size, rec["source"])  # fails early for another aspect ratio
    return {**rec, "frame_size": list(frame_size)}


def extrinsic(path, cam_frame, lidar_frame, lidar_topic, cal, cal_name, cal_sha, wins, times, warnings):
    """(the extrinsic record, {frame name: T_cam_lidar 4x4}). From the calibration, else from the bag's
    TF: the static edges and /tf within the frames' read windows (wins: {name: [(start_ns, stop_ns)]}).
    A dynamic link is taken at each frame's camera stamp (times: {name: ns}) from the /tf values logged
    within that frame's own windows, so a TF publisher on another clock than the camera's cannot give a
    frame another frame's value; a frame whose windows hold none takes the nearest of all. A warning
    names the frames whose value is more than READ_S from their stamp (link_gaps). Failed when TF has
    no path."""
    from gator.webapp import rosbag

    if cal is not None:
        T = np.asarray(cal["T_cam_lidar"], float)
        return ({"source": "calibration", "file": cal_name, "calibration_sha256": cal_sha, "chain": None,
                 "link": None, "camera_axes": cal["camera_axes"], "T_cam_lidar": np.round(T, 6).tolist()},
                dict.fromkeys(times, T))
    edges, first, own = [], None, {}
    if cam_frame and lidar_frame:
        with rosbag.open_bag(path) as r:
            edges = rosbag.transforms(r, [w for n in times for w in wins[n]])
            first = rosbag.camera_to_lidar(cam_frame, lidar_frame, edges) if edges else None
            if first is not None and first["link"] == "dynamic":
                static = [e for e in edges if e[3]]
                own = {n: static + [e for e in rosbag.transforms(r, wins[n]) if not e[3]] for n in times}
    if first is None:
        raise Failed(tf_reason(cam_frame, lidar_frame, lidar_topic))
    per, gaps = {}, {}
    for n, t in times.items():
        got, used = first, edges
        if first["link"] == "dynamic":
            got = rosbag.camera_to_lidar(cam_frame, lidar_frame, own[n], t)
            if got is None or got["link"] != "dynamic" or got["chain"] != first["chain"]:
                got = rosbag.camera_to_lidar(cam_frame, lidar_frame, edges, t)  # none in its windows
            else:
                used = own[n]
            gaps[n] = link_gaps(used, first["chain"], {n: t})[n]
        per[n] = np.asarray(got["T_cam_lidar"], float)
    far = {n: g for n, g in gaps.items() if g > READ_S * 1e9}
    if far:
        warnings.append(f"TF links {first['chain'][0]} to {first['chain'][-1]} on /tf, and {len(far)} of the "
                        f"{len(times)} frames have no /tf value of that link within {READ_S:g} s of their stamp "
                        f"(up to {max(far.values()) / 1e9:.2f} s away), so their T_cam_lidar may be wrong; check "
                        "their check images, or upload the calibration")
    T0 = per[min(per)] if per else np.asarray(first["T_cam_lidar"], float)
    return ({"source": "tf", "file": None, "calibration_sha256": None, "chain": first["chain"], "link": first["link"],
             "camera_axes": first["camera_axes"], "T_cam_lidar": np.round(T0, 6).tolist()}, per)


def link_gaps(edges, chain, times):
    """{frame name: ns} of a TF chain: for each frame (times: {name: ns}), how far in time the /tf value
    of its dynamic links that rosbag.tf_chain takes (the nearest) is, the largest over those links. A
    link that /tf_static gives is not dynamic."""
    static = {(p, c) for p, c, _, s, _ in edges if s}
    stamps = {}
    for p, c, _, s, t in edges:
        if not s:
            stamps.setdefault(frozenset((p, c)), []).append(t)
    links = [frozenset(x) for x in zip(chain, chain[1:]) if x not in static and x[::-1] not in static]
    out = {}
    for n, t in times.items():
        gaps = [min(abs(int(x) - t) for x in stamps[k]) for k in links if k in stamps]
        out[n] = max(gaps, default=0)
    return out


def read_scans(path, topic, windows, keep=None, lo=0.1, hi=0.85):
    """[(log_ns, header_ns, xyz, t_ns or None)] of the LiDAR topic within the log-time windows
    (rosbag.scans), with PROGRESS lines from lo to hi as the windows are read. keep: a function of
    (xyz, log_ns) that gives the points to hold (None: all), so that a 360 degree scan keeps only what
    the camera can see (in_view)."""
    from gator.webapp import rosbag

    spans = rosbag.merge_windows(windows)
    total = max(sum(b - a for a, b in spans), 1)
    out, shown = [], lo
    for log, head, xyz, t in rosbag.scans(path, topic, windows):
        t = None if t is None else np.asarray(t, np.int64)
        if keep is not None:
            m = keep(xyz, int(log))
            xyz, t = xyz[m], None if t is None else t[m]
        out.append((int(log), int(head), xyz, t))
        passed = sum(min(b, log) - a for a, b in spans if a <= log)
        x = lo + (hi - lo) * passed / total
        if x >= shown + 0.02:
            shown = x
            progress(x)
    return out


def time_base(stamps, frames, scans, offset_ns):
    """(base, the time record, warnings). offset_ns is the calibration's time_offset_s, which relates the
    header clocks (t_lidar = t_cam + offset). "header" when the camera's header clock is usable, every
    scan has a stamp, and either the offset is given or the log-minus-header offsets of the two clocks
    agree within CLOCK_TOL_S; else "log". With an offset the log times only check it: in a live
    recording their offsets differ by it, in a bag whose log times copy the header stamps they agree, and
    otherwise a warning says so. In the log base the offset is not used, since log times come from the
    recorder's one clock. The record's offset_ms is the offset added to the camera stamps (0 in the log
    base)."""
    idx = [int(n[1:]) for n in frames]
    head = np.asarray(stamps["header_ns"], np.int64)[idx]
    log = np.asarray(stamps["log_ns"], np.int64)[idx]
    cam = float(np.median(log - head)) if stamps["clock"] == "header" else None
    heads = np.array([s[1] for s in scans], np.int64)
    lid = float(np.median(np.array([s[0] for s in scans], np.int64) - heads)) if len(heads) and heads.all() else None
    tol, warnings, base = CLOCK_TOL_S * 1e9, [], "header"
    seen = None if cam is None or lid is None else (f"(log time minus header stamp: camera {cam / 1e9:.3f} s, "
                                                    f"LiDAR {lid / 1e9:.3f} s)")
    if cam is None:
        base = "log"
        warnings.append("the camera's header stamps are zero, go backwards or repeat, so frames and scans are "
                        "matched by log time")
    elif lid is None:
        base = "log"
        warnings.append("some scans have no header stamp, so frames and scans are matched by log time")
    elif offset_ns:
        if abs(cam - lid - offset_ns) > tol and abs(cam - lid) > tol:
            warnings.append(f"time_offset_s is {offset_ns / 1e9:g} s, but the log times put the LiDAR's header clock "
                            f"{(cam - lid) / 1e9:+.2f} s from the camera's {seen}; check time_offset_s")
    elif abs(cam - lid) > tol:
        base = "log"
        warnings.append(f"the camera's and the LiDAR's header clocks differ by {(cam - lid) / 1e9:+.2f} s {seen}, so "
                        "frames and scans are matched by log time, which is less exact; give time_offset_s in the "
                        "calibration")
    if base == "log" and offset_ns:
        warnings.append(f"time_offset_s ({offset_ns / 1e9:g} s) is not used: it relates the header stamps, and "
                        "frames and scans are matched by log time")
    rec = {"base": base, "window_ms": round(WINDOW_S * 1000, 3),
           "offset_ms": round((offset_ns if base == "header" else 0) / 1e6, 3),
           "log_minus_header_ms": {"camera": None if cam is None else round(cam / 1e6, 3),
                                   "lidar": None if lid is None else round(lid / 1e6, 3)}}
    return base, rec, warnings


def scan_period(scans, base, rate_hz):
    """(the LiDAR's scan period in ns, where it comes from): "scans", the median step between the scans
    that were read (in the base's clock; steps over 2 READ_S lie between read windows and do not
    count); else "rate_hz", topics.json's rate (from the bag's first messages only); else "default",
    100 ms."""
    t = np.unique([s[1] if base == "header" else s[0] for s in scans])
    d = np.diff(t)
    d = d[(d > 0) & (d <= 2 * READ_S * 1e9)]
    if len(d):
        return float(np.median(d)), "scans"
    if rate_hz and rate_hz > 0:
        return 1e9 / rate_hz, "rate_hz"
    return 1e8, "default"


def in_view(frames, wins, T, intr, sizes):
    """read_scans' keep: the points of a scan logged at log_ns that land in the image of a frame whose
    read windows (wins: {name: [(start_ns, stop_ns)]}) hold that time, with the frame's T_cam_lidar
    and size. Frames that share both are projected once."""
    def keep(xyz, log):
        views = {}
        for n in frames:
            if any(a <= log < b for a, b in wins[n]):
                views.setdefault((T[n].tobytes(), sizes[n]), (T[n], sizes[n]))
        m = np.zeros(len(xyz), bool)
        for Tn, size in views.values():
            m |= project(xyz, Tn, intr, size)[3]
        return m

    return keep


def frame_points(scans, base, t_cam, period):
    """The LiDAR points (M, 3) matched to a camera stamp t_cam (ns, in the base): with per-point times the
    points within +-WINDOW_S, else every point of the nearest scan within one period."""
    w = WINDOW_S * 1e9
    if all(s[3] is not None for s in scans):
        got = []
        for log, head, xyz, t in scans:
            if not len(t):
                continue
            tb = t if base == "header" else log + (t - (head or int(t.min())))
            keep = np.abs(tb - t_cam) <= w
            if keep.any():
                got.append(xyz[keep])
        return np.concatenate(got) if got else np.zeros((0, 3), np.float32)
    stamps = np.array([s[1] if base == "header" else s[0] for s in scans], np.int64)
    i = int(np.argmin(np.abs(stamps - t_cam)))
    return scans[i][2] if abs(stamps[i] - t_cam) <= period else np.zeros((0, 3), np.float32)


def project(xyz, T, intr, frame_size):
    """(u, v, z, ok) of LiDAR points in the frame's pixels (lidar_scale.project): T_cam_lidar, then the
    camera_info's R, then the intrinsics fitted to the frame (fit_matrix)."""
    X = np.asarray(xyz, float).reshape(-1, 3) @ T[:3, :3].T + T[:3, 3]
    if intr.get("R") is not None:
        X = X @ np.asarray(intr["R"], float).T
    K = fit_matrix(intr["K"], intr["size"], frame_size, intr["source"])
    return lidar_scale.project(X, K, intr["D"], intr["model"], frame_size)


def to_samples(xyz, T, intr, frame_size):
    """[u, v, z] (float32) of the LiDAR points that land in the frame (project)."""
    u, v, z, ok = project(xyz, T, intr, frame_size)
    return np.stack([u[ok], v[ok], z[ok]], 1).astype(np.float32)


def rosbag_samples(J, job, spec, names):
    """({frame name: (N, 3) [u, v, z]}, info) of a rosbag job's LiDAR topic (see the module docstring).
    info: source, extrinsic, intrinsics, time and warnings. Failed with the reason."""
    from gator.webapp import steps
    from gator.webapp.jobs import INPUTS

    path, topic = Path(job["video"]["path"]), spec["topic"]
    kind = steps.source(job)["kind"]
    if kind != "rosbag":
        raise Failed(f"LiDAR of kind rosbag needs a rosbag input; this job's input is {INPUTS.get(kind, kind)}",
                     {"source": {"kind": "rosbag", "file": path.name, "topic": topic}})
    cam = steps.source(job).get("topic")
    if not cam:
        raise Failed("choose the bag's camera topic first")
    if not path.is_file():
        raise Failed(f"the uploaded bag {path.name} is not on the server; upload it again")
    found = read_json(J / "frames" / "topics.json") or {}
    if "lidar" not in found:
        raise Failed(OLD_TOPICS)
    lid = next((x for x in found["lidar"] if x["topic"] == topic), None)
    if lid is None:
        why = {x["topic"]: x["why"] for x in found.get("lidar_skipped") or []}
        raise Failed(f"the LiDAR topic {topic} cannot be used: {why[topic]}" if topic in why
                     else f"{topic} is not a LiDAR topic of the bag")
    camera = next((t for t in found.get("topics") or [] if t["topic"] == cam), {})
    source = {"kind": "rosbag", "file": path.name, "camera_topic": cam, "topic": topic, "type": lid.get("type"),
              "frame_id": lid.get("frame_id")}
    try:
        return _rosbag_samples(J, names, path, topic, cam, camera, lid, source, job)
    except Failed as e:
        e.info = {"source": source, **e.info}
        raise
    except SystemExit as e:  # rosbag.py's own reasons: a bag it cannot read, a topic that is not there
        raise Failed(str(e), {"source": source}) from None


def _rosbag_samples(J, names, path, topic, cam, camera, lid, source, job):
    """rosbag_samples once the topics are known."""
    from PIL import Image

    cal, cal_sha, cal_name = calibration(J, job)
    frames = frame_names(J, names)
    st = path.stat()
    key = {"version": CACHE_VERSION, "kind": "rosbag",
           "input": {"name": path.name, "bytes": st.st_size, "mtime_ns": st.st_mtime_ns},
           "topics": {"camera": cam, "lidar": topic}, "calibration_sha256": cal_sha, "window_ms": WINDOW_S * 1000}
    hit = cache_read(J, key, names)
    if hit is not None:
        samples, info = hit
        print(f"LiDAR samples of {len(samples)} frames from the cache (lidar/samples.npz)", flush=True)
        return samples, {**info, "source": {**info.get("source", source), "cache": "hit"}}

    warnings = []
    sizes = {}
    for n in frames:
        with Image.open(J / "scene" / "images" / f"{n}.png") as im:
            sizes[n] = im.size
    stamps = camera_stamps(J, path, cam)
    count = len(stamps["log_ns"])
    late = [n for n in frames if int(n[1:]) >= count]
    if late:
        raise Failed(f"frames {', '.join(late[:5])} are not messages of {cam}, which has {count}; choose the camera "
                     "topic again to analyze the bag again")
    offset = int(round((cal or {}).get("time_offset_s", 0.0) * 1e9))
    r = int(READ_S * 1e9)
    wins = {}  # each frame's read windows: around its log time, and around that plus the offset
    for n in frames:
        t = int(stamps["log_ns"][int(n[1:])])
        wins[n] = [(t - r, t + r)] + ([(t + offset - r, t + offset + r)] if offset else [])
    windows = [w for n in frames for w in wins[n]]
    intr = intrinsics(path, cam, camera.get("camera_info"), cal, cal_name, sizes[frames[0]], warnings)
    tf_time = {n: int(stamps["header_ns" if stamps["clock"] == "header" else "log_ns"][int(n[1:])]) for n in frames}
    ext, T = extrinsic(path, camera.get("frame_id"), lid.get("frame_id"), topic, cal, cal_name, cal_sha, wins,
                       tf_time, warnings)
    R = intr.get("R")
    if ext["source"] == "calibration" and R is not None and not np.allclose(R, np.eye(3), atol=1e-6):
        angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        warnings.append(f"the camera_info of {cam} has a rectification R of {angle:.2f} deg, which turns the points "
                        f"after the calibration's T_cam_lidar: that must be to the unrectified optical frame "
                        f"({camera.get('frame_id') or 'the camera frame'}); if it was calibrated on the rectified "
                        "images, give K in the calibration too")
    progress(0.1)
    print(f"reading {topic} within {READ_S} s of {len(frames)} frames", flush=True)
    scans = read_scans(path, topic, windows, in_view(frames, wins, T, intr, sizes))
    info = {"source": {**source, "scans": len(scans)}, "extrinsic": ext, "intrinsics": intr}
    if not scans:
        raise Failed(f"{topic} has no scans within {READ_S} s of the frames' times; was the LiDAR recording "
                     "during the camera's frames?", info)
    base, trec, more = time_base(stamps, frames, scans, offset)
    warnings += more
    period, period_from = scan_period(scans, base, lid.get("rate_hz"))
    point_times = all(s[3] is not None for s in scans)
    info["time"] = {**trec, "scan_period_ms": round(period / 1e6, 3), "scan_period_from": period_from,
                    "point_times": point_times}
    add = offset if base == "header" else 0  # the offset relates the header clocks, not the log times
    samples = {}
    for n in frames:  # only the scans read for this frame
        near = [s for s in scans if any(a <= s[0] < b for a, b in wins[n])]
        t_cam = int(stamps["header_ns" if base == "header" else "log_ns"][int(n[1:])]) + add
        samples[n] = to_samples(frame_points(near, base, t_cam, period), T[n], intr, sizes[n])
    info["warnings"] = warnings
    cache_write(J, key, samples, info)
    return samples, {**info, "source": {**info["source"], "cache": "miss"}}


def stray_samples(J, job, spec, names):
    """({frame name: (N, 3) [u, v, z]}, info) of a Stray Scanner capture's depth images (see the module
    docstring). Failed with the reason."""
    from PIL import Image

    from gator.webapp import steps, stray
    from gator.webapp.jobs import INPUTS

    path = Path(job["video"]["path"])
    path = path if path.is_absolute() else J / path
    source = {"kind": "stray", "file": path.name}
    kind = steps.source(job)["kind"]
    if kind != "stray":
        raise Failed(f"LiDAR of kind stray needs a Stray Scanner capture; this job's input is {INPUTS.get(kind, kind)}",
                     {"source": source})
    if not path.is_file():
        raise Failed(f"the uploaded export {path.name} is not on the server; upload it again", {"source": source})
    video = (read_json(J / "frames" / "meta.json") or {}).get("video") or {}
    if video.get("kind") != "stray" or "rotation" not in video:
        raise Failed("frames/meta.json is not of a Stray Scanner capture; analyze the input again", {"source": source})
    k = stray.turns(video)
    frames = frame_names(J, names)
    sizes = {}
    for n in frames:
        with Image.open(J / "scene" / "images" / f"{n}.png") as im:
            sizes[n] = im.size
    upright = (video["width"], video["height"])
    aspect = upright[0] / upright[1]
    off = [n for n in frames if abs(sizes[n][0] / sizes[n][1] - aspect) > 0.01 * aspect]
    if off:  # extract writes the frames at the video's size; another aspect ratio is another turn
        w, h = sizes[off[0]]
        raise Failed(f"frames {', '.join(off[:5])} are {w}x{h}, but the capture's upright frames are "
                     f"{upright[0]}x{upright[1]}; analyze the input again", {"source": source})
    st = path.stat()
    key = {"version": CACHE_VERSION, "kind": "stray",
           "input": {"name": path.name, "bytes": st.st_size, "mtime_ns": st.st_mtime_ns}, "turns": k,
           "frame_sizes": sorted(map(list, set(sizes.values()))),
           "min_confidence": stray.MIN_CONF, "z_range_m": list(stray.Z_RANGE)}
    hit = cache_read(J, key, names)
    if hit is not None:
        samples, info = hit
        print(f"LiDAR samples of {len(samples)} frames from the cache (lidar/samples.npz)", flush=True)
        return samples, {**info, "source": {**info.get("source", source), "cache": "hit"}}

    try:
        root = stray.unpack(path, path.parent / "stray")
    except SystemExit as e:
        raise Failed(str(e), {"source": source}) from None
    except OSError as e:
        raise Failed(f"cannot unpack {path.name}: {e}", {"source": source}) from None
    warnings = []
    min_conf = stray.MIN_CONF
    if not any(stray.FRAME_FILE.fullmatch(f"confidence/{p.name}") for p in (root / "confidence").glob("*.png")):
        min_conf = 0
        warnings.append("the export has no confidence/ images, so every depth pixel is used, also the doubtful ones")
    progress(0.1)
    print(f"reading the depth images of {len(frames)} frames, turned {90 * k} degrees", flush=True)
    samples, missing, unsure = {}, [], []  # frames without a depth image; with one but without its confidence
    for j, n in enumerate(frames):
        i = int(n[1:])
        if not (root / "depth" / f"{i:06d}.png").is_file():
            missing.append(n)
        elif min_conf and not (root / "confidence" / f"{i:06d}.png").is_file():
            unsure.append(n)
        else:
            try:
                samples[n] = stray.depth_samples(root, i, sizes[n], k, min_conf=min_conf)
            except (ValueError, OSError) as e:
                raise Failed(f"{n}: {e}", {"source": source}) from None
        samples.setdefault(n, np.zeros((0, 3), np.float32))
        progress(0.1 + 0.75 * (j + 1) / len(frames))
    for left, what in ((missing, "no depth image"), (unsure, "a depth image but no confidence image")):
        if left:
            more = f" and {len(left) - 5} more" if len(left) > 5 else ""
            warnings.append(f"frames {', '.join(left[:5])}{more} have {what}, so they have no samples")
    info = {"source": {**source, "depth_size": video.get("depth_size"), "frames": len(frames)},
            "extrinsic": {"source": "stray (depth aligned with the color camera)", "calibration_sha256": None},
            "intrinsics": {"source": "stray: depth scaled to the frame size", "depth_size": video.get("depth_size"),
                           "frame_size": list(sizes[frames[0]]), "turns": k, "min_confidence": min_conf,
                           "z_range_m": list(stray.Z_RANGE)},
            "time": {"base": "frame index"}, "warnings": warnings}
    cache_write(J, key, samples, info)
    return samples, {**info, "source": {**info["source"], "cache": "miss"}}


SOURCES = {"rosbag": rosbag_samples, "stray": stray_samples}  # job.json "lidar" "kind": its samples


# ---- the stage ----

def record(rec, info, t0):
    """pool/lidar.json: lidar_scale.estimate's record (or a failed one) with the source's fields."""
    out = {"version": lidar_scale.VERSION, "ok": rec["ok"], "s": rec["s"], "reason": rec["reason"],
           "source": info.get("source"), "extrinsic": info.get("extrinsic"), "intrinsics": info.get("intrinsics"),
           "time": info.get("time")}
    out.update({k: v for k, v in rec.items() if k not in out and k not in ("warnings", "seconds")})
    out["warnings"] = list(info.get("warnings") or []) + list(rec.get("warnings") or [])
    out["seconds"] = round(time.time() - t0, 2)
    return out


def scale(job_dir):
    """Write J/pool/lidar.json (see the module docstring); SystemExit with the reason when it is not ok."""
    from gator.stage_a.mvsam3d import view_names
    from gator.webapp import steps

    J = Path(job_dir)
    t0 = time.time()
    job = json.loads((J / "job.json").read_text())
    try:
        spec = steps.lidar(job)
    except ValueError as e:
        raise SystemExit(f"job.json: {e}") from None
    if spec is None:
        raise SystemExit("this job has no LiDAR (job.json lidar is null), so it has no metric scale stage")
    if spec["kind"] not in SOURCES:
        raise SystemExit(f"no LiDAR source of kind {spec['kind']!r}")
    names = view_names(J / "pool")
    if not names or not (J / "pool" / "da3_output.npz").is_file():
        raise SystemExit("the pool has no views with DA3 depth; run prep and da3 first")
    progress(0)
    check = J / "lidar" / "check"
    shutil.rmtree(check, ignore_errors=True)  # the check images of views that left the pool
    out = J / "pool" / "lidar.json"
    out.unlink(missing_ok=True)
    try:
        samples, info = SOURCES[spec["kind"]](J, job, spec, names)
    except Failed as e:
        rec = record({"ok": False, "s": None, "reason": str(e), "views": [], "combine": None}, e.info, t0)
        write_json(out, rec)
        raise SystemExit(f"no metric scale: {e}") from None
    progress(0.9)
    src, ext, tm = info.get("source") or {}, info.get("extrinsic") or {}, info.get("time") or {}
    what = f"{src.get('topic')} ({src.get('type')})" if src.get("kind") == "rosbag" else \
        f"{src.get('kind')} {src.get('file')}"
    print(f"LiDAR {what}, cache {src.get('cache')}; extrinsic from {ext.get('source')}"
          + (f" ({' <- '.join(ext['chain'])}, {ext.get('link')})" if ext.get("chain") else
             f" {ext['file']}" if ext.get("file") else "")
          + f"; intrinsics from {(info.get('intrinsics') or {}).get('source')}", flush=True)
    if "window_ms" in tm:
        print(f"time base {tm['base']}, +-{tm['window_ms']:g} ms, offset {tm['offset_ms']:g} ms, "
              f"per-point times {tm.get('point_times')}", flush=True)
    elif tm:
        print(f"time base {tm['base']}", flush=True)
    try:
        est = lidar_scale.estimate(J / "pool", {n: samples[n] for n in names if n in samples}, check_dir=check)
    except ValueError as e:  # a pool that prep and da3 left inconsistent
        write_json(out, record({"ok": False, "s": None, "reason": str(e), "views": [], "combine": None}, info, t0))
        raise SystemExit(f"no metric scale: {e}") from None
    rec = record(est, info, t0)
    write_json(out, rec)  # last: with ok, the stage's done marker
    lines = lidar_scale.summary(rec)
    print("\n".join(lines[:-1]))
    for w in rec["warnings"]:
        print("warning:", w)
    if not rec["ok"]:
        raise SystemExit(lines[-1])
    print(lines[-1])
    progress(1)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="step", required=True)
    sub.add_parser("scale", help="the metric scale of a job's pool from its LiDAR").add_argument(
        "--job", required=True, help="job folder with job.json")
    a = ap.parse_args(argv)
    scale(a.job)


if __name__ == "__main__":
    main()
