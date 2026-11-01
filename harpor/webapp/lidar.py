"""The metric scale from LiDAR: the web app's lidar stage (build_plan.md note 12).

Runs in envs/web (numpy, scipy, pillow, rosbags; CPU only), from the repo root, after da3:

    python -m harpor.webapp.lidar scale --job J

J/job.json "lidar" (steps.lidar) names the source. SOURCES has one function per kind that gives the
LiDAR samples of the pool frames: "rosbag" (rosbag_samples), a LiDAR topic of the job's bag, and
"stray" (stray_samples), the depth images of a Stray Scanner capture.

Every pool frame ("f00012") gets an (N, 3) array [u, v, z] of samples: the LiDAR points at the frame's
time as pixels of scene/images/<name>.png and their depth in meters. The samples and the frames' clouds for
the Inputs tab are cached in J/lidar/samples.npz and J/lidar/clouds.npz, keyed in J/lidar/samples.json. The
stage writes J/lidar/cloud/<view>.glb (cloud_glb) with the check images, also when the record is not ok,
before pool/lidar.json. How the samples, the time matching, the intrinsics and the clouds work:
build_plan.md note 12 (Rosbag source, Stray Scanner source, Points for the Inputs tab).

Each cloud also compares the LiDAR with DA3, at the job's s (references: a view's own s only when the job has none):
"diff" holds the LiDAR points that have a ratio r, also those outside the mask, which do not affect s, with their
signed gap in meters, z (1 - s / r): z - s x DA3's depth without a baseline, positive when the LiDAR reads farther. For
a rosbag job, "da3" holds DA3's surface at s (da3_surface). A Stray capture's K stays inside stray_samples, so it has
"diff" only.

The estimate is lidar_scale.estimate, with the check images in J/lidar/check. For a rosbag job it also follows
the LiDAR's rays (rays: the camera points of the samples and the LiDAR's origin in the camera frame). The record,
pool/lidar.json, gains source, extrinsic {source, file, calibration_sha256, chain, link, camera_axes,
camera_axes_from, T_cam_lidar}, intrinsics {source, K, D, model, R, size, frame_size}, time {base, window_ms,
offset_ms, log_minus_header_ms {camera, lidar}, log_minus_last_point_ms, scan_period_ms, scan_period_from,
point_times} and the warnings of both steps (a Stray capture's fields: build_plan.md note 12, pool/lidar.json),
and is written last.
extrinsic.calibration_sha256 (null without a calibration) lets steps.done see a record that an upload or a
delete of the calibration made stale. When the samples cannot be made (no TF link, no intrinsics, ...), the
record has ok false and the reason. The stage prints PROGRESS lines while it reads scans, then one line per
view (lidar_scale.summary), and exits 1 with the reason when the record is not ok.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import time
from pathlib import Path

import numpy as np

from harpor.fileio import read_json, replace, write_json
from harpor.stage_a import lidar_scale

WINDOW_S = 0.05  # points within +-50 ms of a camera stamp, or within half a scan period when that is longer,
WINDOW_MAX_S = 0.1  # up to +-100 ms (a 5 Hz LiDAR): see window()
READ_S = 0.4  # scans are read at least +-0.4 s from each frame's log time, and further for a slow LiDAR (read_span)
CLOCK_TOL_S = 0.2  # the camera's and the LiDAR's log-minus-stamp offsets agree this well (time_base): the header base
CACHE_VERSIONS = {"rosbag": 7, "stray": 6}  # of lidar/samples.json, by kind; another version is a miss
SCAN_POINTS = 60_000  # the most points of a view's whole scan in its cloud (lidar/cloud/<view>.glb)
SCAN_KEEP = 20_000  # while reading, each scan also keeps a random subset of its points, at most this many, for the clouds
DA3_POINTS = 60_000  # the most points of DA3's surface in a view's cloud
NO_K = ("the export has no intrinsics (fx, fy, cx, cy in odometry.csv, or camera_matrix.csv), so the Inputs tab has "
        "no LiDAR points")


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


def progress(x):
    print(f"PROGRESS {x:.3f}", flush=True)


# ---- the samples cache ----

def cache_read(J, key, names):
    """({name: samples}, the info, {name: (xyz, scan)}) of lidar/samples.* and lidar/clouds.npz, when the key is
    key and the files have every name (the clouds: every frame that samples.json lists with one); else None."""
    d = Path(J) / "lidar"
    meta = read_json(d / "samples.json")
    if not isinstance(meta, dict) or meta.get("key") != key or not set(names) <= set(meta.get("frames") or []):
        return None
    try:
        with np.load(d / "samples.npz") as f:
            samples = {n: np.asarray(f[n], float) for n in meta["frames"]}
        with np.load(d / "clouds.npz") as f:
            clouds = {n: (f[f"{n}_xyz"], f[f"{n}_scan"]) for n in meta.get("clouds") or []}
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return samples, meta.get("info") or {}, clouds


def cache_write(J, key, samples, info, clouds):
    """lidar/samples.npz, lidar/clouds.npz, then lidar/samples.json (the key) last; the old key goes first."""
    d = Path(J) / "lidar"
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    (d / "samples.json").unlink(missing_ok=True)
    replace(d / "samples.npz", np.savez, **{n: np.asarray(a, np.float32).reshape(-1, 3) for n, a in samples.items()})
    replace(d / "clouds.npz", np.savez, **{f"{n}_{k}": np.asarray(a, np.float32).reshape(-1, 3)
                                           for n, c in clouds.items() for k, a in zip(("xyz", "scan"), c)})
    write_json(d / "samples.json", {"key": key, "frames": sorted(samples), "clouds": sorted(clouds), "info": info,
                                    "points": {n: int(len(a)) for n, a in sorted(samples.items())}}, indent=1)


# ---- the parts of a rosbag's samples ----

def frame_names(J, names):
    """The pool views and the other frames that extract wrote (frames/pool.json), sorted."""
    pool = read_json(Path(J) / "frames" / "pool.json") or {}
    extracted = (pool.get("names") or {}).values()
    more = {n for n in extracted if isinstance(n, str) and (Path(J) / "scene" / "images" / f"{n}.png").is_file()}
    bad = [n for n in names if not re.fullmatch(r"f\d+", n)]
    if bad:
        raise Failed(f"pool views {', '.join(bad[:5])} are not named like f00012 (a frame number), so they are not "
                     "frames of the input")
    return sorted(set(names) | {n for n in more if re.fullmatch(r"f\d+", n)})


def calibration(J, job, topic):
    """(the parsed calibration, its file's sha256, its name), or (None, None, None) without one. Failed when
    it was uploaded for another LiDAR topic than topic (job.json calibration.lidar_topic; one without
    lidar_topic is not checked), as jobs.lidar_settings refuses at the start: a retry or a run from lidar
    after a later upload does not pass there."""
    cal = job.get("calibration")
    if not cal:
        return None, None, None
    made_for = cal.get("lidar_topic")
    if made_for and made_for != topic:
        raise Failed(f"the calibration {cal.get('name')} is for {made_for}, not {topic}; choose {made_for}, or upload "
                     f"a calibration for {topic}")
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


def camera_stamps(J, topic):
    """{"clock", "header_ns", "log_ns"} of the camera topic's frames: frames/stamps.json. Failed when it is missing or
    for another topic."""
    st = read_json(Path(J) / "frames" / "stamps.json")
    if not isinstance(st, dict) or st.get("topic") != topic:
        raise Failed("frames/stamps.json is missing or for another topic; choose the camera topic again to analyze the "
                     "bag again")
    return st


def fit_matrix(K, size, frame_size, source):
    """K for an image of size size, made to fit frames of frame_size (rosbag.rescale); Failed for
    another aspect ratio."""
    from harpor.webapp import rosbag

    if tuple(size) == tuple(frame_size):
        return np.asarray(K, float)
    sx, sy = frame_size[0] / size[0], frame_size[1] / size[1]
    if abs(sx / sy - 1) > rosbag.ASPECT_TOL:
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
    from harpor.webapp import rosbag

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
    TF: the static edges and /tf within the frames' read windows (wins: {name: [(start_ns, stop_ns)]}),
    or all of /tf when those hold no path, or for one frame whose name does not give its axes, as the
    topic listing reads it all (each frame then takes the nearest value, with the warning below). A dynamic link is taken at each frame's camera stamp (times:
    {name: ns}) from the /tf values logged within that frame's own windows, so a TF publisher on another
    clock than the camera's cannot give a frame another frame's value; a frame whose windows hold none
    takes the nearest of all. The camera's
    axes are decided once, from all the edges (rosbag.camera_axes_from), and hold for every frame. A
    camera and a LiDAR in one frame need no TF: the link is the identity, as in the listing. A
    warning names the frames whose value is more than READ_S from their stamp (link_gaps). Failed when
    TF has no path."""
    from harpor.webapp import rosbag

    if cal is not None:
        T = np.asarray(cal["T_cam_lidar"], float)
        return ({"source": "calibration", "file": cal_name, "calibration_sha256": cal_sha, "chain": None,
                 "link": None, "camera_axes": cal["camera_axes"], "camera_axes_from": "calibration",
                 "T_cam_lidar": np.round(T, 6).tolist()}, dict.fromkeys(times, T))
    edges, first, own = [], None, {}
    if cam_frame and lidar_frame:
        with rosbag.open_bag(path) as r:
            edges = rosbag.transforms(r, [w for n in times for w in wins[n]])
            first = rosbag.camera_to_lidar(cam_frame, lidar_frame, edges)  # one frame: the identity, also without TF
            # No path near the frames (the link may be on /tf only away from them), or one frame for the camera and
            # the LiDAR whose name does not give its axes: all of /tf, as the listing reads it.
            lone = first is not None and len(first["chain"]) == 1 and first["camera_axes_from"] != "name"
            if first is None or lone:
                whole = rosbag.transforms(r)
                got = rosbag.camera_to_lidar(cam_frame, lidar_frame, whole) if whole else None
                if got is not None:  # the frames need only the values of the chain's links
                    first, pairs = got, set(zip(got["chain"], got["chain"][1:]))
                    edges = [e for e in whole if (e[0], e[1]) in pairs or (e[1], e[0]) in pairs]
            if first is not None and first["link"] == "dynamic":
                static = [e for e in edges if e[3]]
                own = {n: static + [e for e in rosbag.transforms(r, wins[n]) if not e[3]] for n in times}
    if first is None:
        raise Failed(tf_reason(cam_frame, lidar_frame, lidar_topic))
    per, gaps, axes = {}, {}, (first["camera_axes"], first["camera_axes_from"])
    for n, t in times.items():
        got, used = first, edges
        if first["link"] == "dynamic":
            got = rosbag.camera_to_lidar(cam_frame, lidar_frame, own[n], t, axes)
            if got is None or got["link"] != "dynamic" or got["chain"] != first["chain"]:
                got = rosbag.camera_to_lidar(cam_frame, lidar_frame, edges, t, axes)  # none in its windows
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
             "camera_axes": axes[0], "camera_axes_from": axes[1], "T_cam_lidar": np.round(T0, 6).tolist()}, per)


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


def at_most(a, k, seed):
    """At most k rows of a: a random choice fixed by seed, kept in their order (a stride would draw stripes on
    an organized cloud)."""
    if len(a) <= k:
        return a
    return a[np.sort(np.random.default_rng(seed).choice(len(a), k, replace=False))]


def read_scans(path, topic, windows, keep=None, lo=0.1, hi=0.85, whole=0):
    """[(log_ns, header_ns, xyz, t_ns or None, whole, end_ns or None)] of the LiDAR topic within the
    log-time windows (rosbag.scans), with PROGRESS lines from lo to hi as the windows are read. keep: a
    function of (xyz, log_ns) that gives the points to hold (None: all), so that a 360 degree scan keeps
    only what the camera can see (in_view). whole: (xyz, t_ns or None) of at most that many of the scan's
    points, taken before keep (at_most, seeded by the log time), for the clouds; None when whole is 0.
    end_ns: the time of the scan's last point (last_point), also taken before keep; None without per-point
    times."""
    from harpor.webapp import rosbag

    spans = rosbag.merge_windows(windows)
    total = max(sum(b - a for a, b in spans), 1)
    out, shown = [], lo
    for log, head, xyz, t in rosbag.scans(path, topic, windows):
        t = None if t is None else np.asarray(t, np.int64)
        end = None if t is None or not len(t) else last_point(t)
        part = None
        if whole:
            rows = at_most(np.arange(len(xyz)), whole, int(log))
            part = (np.array(xyz[rows], np.float32), None if t is None else t[rows])
        if keep is not None:
            m = keep(xyz, int(log))
            xyz, t = xyz[m], None if t is None else t[m]
        out.append((int(log), int(head), xyz, t, part, end))
        passed = sum(min(b, log) - a for a, b in spans if a <= log)
        x = lo + (hi - lo) * passed / total
        if x >= shown + 0.02:
            shown = x
            progress(x)
    return out


def last_point(t):
    """The time (ns) of a scan's last point, from its points' times t (int64): the latest time that lies no
    further past the 99th percentile than the 1st percentile lies before it. That is the latest time of a
    normal sweep. point_times keeps a bad time up to POINT_TIME_S from the stamp, and in the log base the
    last point moves every point of the scan (frame_points), so one bad time far past the sweep is left out."""
    d = t - t.min()
    lo, hi = np.percentile(d, [1, 99])
    return int(t[d <= 2 * hi - lo].max())


def unread(windows, done):
    """The parts of the log-time windows [(start_ns, stop_ns)] that the windows done do not cover, merged."""
    from harpor.webapp import rosbag

    out, done = [], rosbag.merge_windows(done)
    for a, b in rosbag.merge_windows(windows):
        for c, d in done:
            if c < b and a < d:  # they overlap
                if a < c:
                    out.append((a, c))
                a = max(a, d)
        if a < b:
            out.append((a, b))
    return out


def time_base(stamps, frames, scans, offset_ns):
    """(base, the time record, warnings). offset_ns is the calibration's time_offset_s (None when it gives
    none; 0 is given), which relates the header clocks (t_lidar = t_cam + offset). "header" when the
    camera's header clock is usable, every scan has a stamp, and either the offset is given or the two
    clocks agree within CLOCK_TOL_S; else "log". The clocks are compared by how long after its stamp each
    sensor's messages are logged: the camera's frames after their header stamps, and the LiDAR's scans
    after their header stamps and, when the points have times, also after their last point. A live
    recorder logs a scan once it is swept, so a driver that stamps the sweep's start (livox_ros_driver2)
    fits the last point; a bag whose log times copy the header stamps fits the header stamp. The clocks
    agree when either fits. Without per-point times a stamp at the sweep's start puts the LiDAR's up to a
    scan period later, which a clock offset would too, so then the log base, which takes each such scan
    at the middle of its sweep (frame_points). With an offset the log times only check it: in a live
    recording the two differ by it, in a bag whose log times copy the header stamps they agree, and
    otherwise a warning says so. A live recording on one clock with a wrong offset has the log times of
    such a bag, so when only the zero shift fits a nonzero offset, a warning names both readings. In
    the log base the offset is not used, since log times come from the
    recorder's one clock. The record's offset_ms is the offset added to the camera stamps (0 in the log
    base); log_minus_last_point_ms is the LiDAR's from its last points (None without per-point times)."""
    idx = [int(n[1:]) for n in frames]
    head = np.asarray(stamps["header_ns"], np.int64)[idx]
    log = np.asarray(stamps["log_ns"], np.int64)[idx]
    cam = float(np.median(log - head)) if stamps["clock"] == "header" else None
    heads = np.array([s[1] for s in scans], np.int64)
    lids = {}  # the LiDAR's log-minus-stamp offsets: from the header stamps, and from the last points
    if len(heads) and heads.all():
        logs = np.array([s[0] for s in scans], np.int64)
        lids["header"] = float(np.median(logs - heads))
        if all(s[5] is not None for s in scans):
            lids["last_point"] = float(np.median(logs - np.array([s[5] for s in scans], np.int64)))
    tol, warnings, base = CLOCK_TOL_S * 1e9, [], "header"
    shifts = (0.0,) if offset_ns is None else (float(offset_ns), 0.0)
    # How far the LiDAR's offset nearest the camera's is from it (less the offset in a live recording). A warning
    # names the clocks' gap by the last points when there are some: a live recorder logs a scan once it is swept.
    gap = min((abs(v - cam + d) for v in lids.values() for d in shifts), default=None) if cam is not None else None
    near = lids.get("last_point", lids.get("header"))
    seen = None if cam is None or near is None else (
        f"(log time minus header stamp: camera {cam / 1e9:.3f} s, LiDAR {lids['header'] / 1e9:.3f} s"
        + (f"; log time minus each scan's last point: LiDAR {lids['last_point'] / 1e9:.3f} s" if "last_point" in lids
           else "") + ")")
    if cam is None:
        base = "log"
        warnings.append("the camera's header stamps are zero, go backwards or repeat, so frames and scans are "
                        "matched by log time")
    elif not lids:
        base = "log"
        warnings.append("some scans have no header stamp, so frames and scans are matched by log time")
    elif offset_ns is not None:
        if gap > tol:
            warnings.append(f"time_offset_s is {offset_ns / 1e9:g} s, but the log times put the LiDAR's header clock "
                            f"{(cam - near) / 1e9:+.2f} s from the camera's {seen}; check time_offset_s")
        elif offset_ns and min(abs(v - cam + offset_ns) for v in lids.values()) > tol:  # only the zero shift fits
            warnings.append(f"time_offset_s is {offset_ns / 1e9:g} s, but the log times put the LiDAR's header clock "
                            f"within {CLOCK_TOL_S:g} s of the camera's {seen}; either the bag's log times copy the "
                            "header stamps, or time_offset_s is wrong (0 when both use one clock); check time_offset_s")
    elif gap > tol:
        base = "log"
        warnings.append(f"the camera's and the LiDAR's header clocks differ by {(cam - near) / 1e9:+.2f} s {seen}, so "
                        "frames and scans are matched by log time, which is less exact; give time_offset_s in the "
                        "calibration (0 when both use one clock)")
    if base == "log" and offset_ns is not None:
        warnings.append(f"time_offset_s ({offset_ns / 1e9:g} s) is not used: it relates the header stamps, and "
                        "frames and scans are matched by log time")
    ms = {k: round(v / 1e6, 3) for k, v in lids.items()}
    rec = {"base": base, "offset_ms": round(((offset_ns or 0) if base == "header" else 0) / 1e6, 3),
           "log_minus_header_ms": {"camera": None if cam is None else round(cam / 1e6, 3), "lidar": ms.get("header")},
           "log_minus_last_point_ms": ms.get("last_point")}
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


def window(period):
    """The point window's half width in ns: half the scan period (ns), at least WINDOW_S and at most
    WINDOW_MAX_S. A spinning LiDAR sweeps the camera's view once a period, so +-half a period holds one
    sweep wherever the camera's stamp falls; a fixed +-50 ms would miss it for 1 frame in 5 at 5 Hz."""
    return min(max(WINDOW_S * 1e9, period / 2), WINDOW_MAX_S * 1e9)


def read_span(period):
    """The half width in ns of each frame's read windows, for a scan period in ns. A scan with points in
    the frame's window can be logged as far as the window, one sweep and CLOCK_TOL_S from the frame's log
    time: a live recorder logs a scan once it is swept, a bag whose log times copy the header stamps logs
    it at its stamp, which may be its first point, and the clock check (time_base) accepts CLOCK_TOL_S
    more delay than the camera's. At least READ_S: 0.4 s at 10 Hz, 0.5 s at 5 Hz, 0.8 s at 2 Hz. A longer
    period counts as 0.5 s, the slowest sweep whose point times rosbag keeps (POINT_TIME_S), so a wrong
    rate in the listing cannot make every read long."""
    from harpor.webapp.rosbag import POINT_TIME_S

    period = min(period, POINT_TIME_S * 1e9)
    return max(READ_S * 1e9, window(period) + period + CLOCK_TOL_S * 1e9)


def frame_points(scans, base, t_cam, period):
    """The LiDAR points (M, 3) matched to a camera stamp t_cam (ns, in the base): with per-point times the
    points within +-window(period), else every point of the nearest scan within one period. scans:
    [(log_ns, header_ns, xyz, t_ns or None, ..., end_ns or None)] (read_scans). In the log base a point's
    time goes to the log clock by its scan's last point (end_ns), since a scan is logged once it is swept
    (only the transport delay is left), and a scan without per-point times counts at its log time less
    half a period, the middle of its sweep."""
    w = window(period)
    if all(s[3] is not None for s in scans):
        got = []
        for log, head, xyz, t, *_, end in scans:
            if not len(t):
                continue
            tb = t if base == "header" else log + (t - end)
            keep = np.abs(tb - t_cam) <= w
            if keep.any():
                got.append(xyz[keep])
        return np.concatenate(got) if got else np.zeros((0, 3), np.float32)
    stamps = np.array([s[1] if base == "header" else s[0] - round(period / 2) for s in scans], np.int64)
    i = int(np.argmin(np.abs(stamps - t_cam)))
    return scans[i][2] if abs(stamps[i] - t_cam) <= period else np.zeros((0, 3), np.float32)


def to_camera(xyz, T, intr):
    """LiDAR points in the frame where a sample's z lives: T_cam_lidar, then the camera_info's R."""
    X = np.asarray(xyz, float).reshape(-1, 3) @ T[:3, :3].T + T[:3, 3]
    if intr.get("R") is not None:
        X = X @ np.asarray(intr["R"], float).T
    return X


def project(xyz, T, intr, frame_size):
    """(u, v, z, ok, X) of LiDAR points in the frame's pixels (lidar_scale.project): to_camera, then the
    intrinsics fitted to the frame (fit_matrix). X: the camera points."""
    X = to_camera(xyz, T, intr)
    K = fit_matrix(intr["K"], intr["size"], frame_size, intr["source"])
    return (*lidar_scale.project(X, K, intr["D"], intr["model"], frame_size), X)


def rays(info, clouds):
    """lidar_scale.estimate's rays of a rosbag job: each frame's camera points (the clouds' xyz, row for row with the
    samples), the LiDAR's origin in that frame (info's rays) and the intrinsics that made the samples' pixels."""
    intr = info["intrinsics"]
    return {n: {"points": clouds[n][0], "origin": r["origin"], "size": r["size"], "D": intr["D"], "model": intr["model"],
                "K": fit_matrix(intr["K"], intr["size"], r["size"], intr["source"])} for n, r in info["rays"].items()}


def to_samples(xyz, T, intr, frame_size):
    """([u, v, z], their camera points), float32 and row for row, of the LiDAR points that land in the frame
    (project)."""
    u, v, z, ok, X = project(xyz, T, intr, frame_size)
    return np.stack([u[ok], v[ok], z[ok]], 1).astype(np.float32), X[ok].astype(np.float32)


def rosbag_samples(J, job, spec, names):
    """({frame name: (N, 3) [u, v, z]}, info, clouds) of a rosbag job's LiDAR topic (build_plan.md note 12,
    Rosbag source). info: source, extrinsic, intrinsics, time and warnings; clouds: {frame name: (xyz, scan)}.
    Failed with the reason."""
    from harpor.webapp import steps
    from harpor.webapp.jobs import INPUTS

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
    lid = next((x for x in found.get("lidar") or [] if x["topic"] == topic), None)
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

    cal, cal_sha, cal_name = calibration(J, job, topic)
    frames = frame_names(J, names)
    st = path.stat()
    key = {"version": CACHE_VERSIONS["rosbag"], "kind": "rosbag",
           "input": {"name": path.name, "bytes": st.st_size, "mtime_ns": st.st_mtime_ns},
           "topics": {"camera": cam, "lidar": topic}, "calibration_sha256": cal_sha,
           "window_ms": [WINDOW_S * 1000, WINDOW_MAX_S * 1000]}
    hit = cache_read(J, key, names)
    if hit is not None:
        samples, info, clouds = hit
        print(f"LiDAR samples of {len(samples)} frames from the cache (lidar/samples.npz)", flush=True)
        return samples, {**info, "source": {**info.get("source", source), "cache": "hit"}}, clouds

    warnings = []
    sizes = {}
    for n in frames:
        with Image.open(J / "scene" / "images" / f"{n}.png") as im:
            sizes[n] = im.size
    stamps = camera_stamps(J, cam)
    count = len(stamps["log_ns"])
    late = [n for n in frames if int(n[1:]) >= count]
    if late:
        raise Failed(f"frames {', '.join(late[:5])} are not messages of {cam}, which has {count}; choose the camera "
                     "topic again to analyze the bag again")
    given = (cal or {}).get("time_offset_s")
    offset = None if given is None else int(round(given * 1e9))  # 0 is given: the header clocks are one

    def read_windows(r):  # each frame's read windows: around its log time, and around that plus the offset
        out = {}
        for n in frames:
            t = int(stamps["log_ns"][int(n[1:])])
            out[n] = [(t - r, t + r)] + ([(t + offset - r, t + offset + r)] if offset else [])
        return out

    listed = scan_period([], "log", lid.get("rate_hz"))[0]  # no scans yet: the listing's rate, else 100 ms
    r = int(read_span(listed))
    wins = read_windows(r)
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
    print(f"reading {topic} within {r / 1e9:.3g} s of {len(frames)} frames", flush=True)
    scans = read_scans(path, topic, windows, in_view(frames, wins, T, intr, sizes), whole=SCAN_KEEP)
    info = {"source": {**source, "scans": len(scans)}, "extrinsic": ext, "intrinsics": intr}
    if not scans:
        raise Failed(f"{topic} has no scans within {r / 1e9:.3g} s of the frames' times; was the LiDAR recording "
                     "during the camera's frames?", info)
    base, trec, more = time_base(stamps, frames, scans, offset)
    period, period_from = scan_period(scans, base, lid.get("rate_hz"))
    if period > 1.1 * listed and read_span(period) > r:  # the listing's rate was too high, not jitter: read the rest
        r = int(read_span(period))
        print(f"reading {topic} within {r / 1e9:.3g} s of the frames, for its scan period of {period / 1e6:.0f} ms",
              flush=True)
        wider = read_windows(r)
        rest = unread([w for n in frames for w in wider[n]], windows)
        scans = sorted(scans + read_scans(path, topic, rest, in_view(frames, wider, T, intr, sizes), 0.85, 0.9,
                                          SCAN_KEEP), key=lambda s: s[0])
        wins, info["source"]["scans"] = wider, len(scans)
        base, trec, more = time_base(stamps, frames, scans, offset)
        period, period_from = scan_period(scans, base, lid.get("rate_hz"))
    warnings += more
    point_times = all(s[3] is not None for s in scans)
    info["time"] = {**trec, "window_ms": round(window(period) / 1e6, 3), "scan_period_ms": round(period / 1e6, 3),
                    "scan_period_from": period_from, "point_times": point_times}
    add = (offset or 0) if base == "header" else 0  # the offset relates the header clocks, not the log times
    samples, clouds = {}, {}
    for n in frames:  # only the scans read for this frame
        near = [s for s in scans if any(a <= s[0] < b for a, b in wins[n])]
        t_cam = int(stamps["header_ns" if base == "header" else "log_ns"][int(n[1:])]) + add
        samples[n], xyz = to_samples(frame_points(near, base, t_cam, period), T[n], intr, sizes[n])
        scan = to_camera(frame_points([(s[0], s[1], *s[4], s[5]) for s in near], base, t_cam, period), T[n], intr)
        scan = scan[np.isfinite(scan).all(1)]
        clouds[n] = (xyz, at_most(scan, SCAN_POINTS, int(n[1:])).astype(np.float32))
    info["rays"] = {n: {"origin": np.round(to_camera(np.zeros(3), T[n], intr)[0], 6).tolist(), "size": list(sizes[n])}
                    for n in frames}  # the LiDAR's origin where the samples' z lives
    info["warnings"] = warnings
    cache_write(J, key, samples, info, clouds)
    return samples, {**info, "source": {**info["source"], "cache": "miss"}}, clouds


def stray_samples(J, job, spec, names):
    """({frame name: (N, 3) [u, v, z]}, info, clouds) of a Stray Scanner capture's depth images (see the module
    docstring); clouds: {frame name: (xyz, scan)}, none without intrinsics. Failed with the reason. A frame whose
    depth or confidence image cannot be used (stray.depth_points' ValueError) gets no samples, with a warning, and
    the samples are then not cached, so the next run reads it again. Failed when such frames leave fewer than
    lidar_scale.MIN_VIEWS pool views."""
    from PIL import Image

    from harpor.webapp import rosbag, steps, stray
    from harpor.webapp.jobs import INPUTS

    path = Path(job["video"]["path"])
    source = {"kind": "stray", "file": path.name}
    kind = steps.source(job)["kind"]
    if kind != "stray":
        raise Failed(f"LiDAR of kind stray needs a Stray Scanner capture; this job's input is {INPUTS.get(kind, kind)}",
                     {"source": source})
    if not path.is_file():
        raise Failed(f"the uploaded export {path.name} is not on the server; upload it again", {"source": source})
    video = (read_json(J / "frames" / "meta.json") or {}).get("video") or {}
    if video.get("kind") != "stray":
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
    key = {"version": CACHE_VERSIONS["stray"], "kind": "stray",
           "input": {"name": path.name, "bytes": st.st_size, "mtime_ns": st.st_mtime_ns}, "turns": k,
           "frame_sizes": sorted(map(list, set(sizes.values()))),
           "min_confidence": stray.MIN_CONF, "z_range_m": list(stray.Z_RANGE)}
    hit = cache_read(J, key, names)
    if hit is not None:
        samples, info, clouds = hit
        print(f"LiDAR samples of {len(samples)} frames from the cache (lidar/samples.npz)", flush=True)
        return samples, {**info, "source": {**info.get("source", source), "cache": "hit"}}, clouds

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
    K, K_from = stray.intrinsics(root, int(video.get("n_frames") or 0))
    if K is None:
        warnings.append(NO_K)
    stored = upright if k % 2 == 0 else upright[::-1]  # the video as stored, the size K is for
    progress(0.1)
    print(f"reading the depth images of {len(frames)} frames, turned {90 * k} degrees", flush=True)
    samples, clouds = {}, {}
    missing, unsure = [], []  # frames without a depth image; with one but without its confidence
    bad = {}  # {frame: why its depth or confidence image cannot be used}
    for j, n in enumerate(frames):
        i = int(n[1:])
        if not (root / "depth" / f"{i:06d}.png").is_file():
            missing.append(n)
        elif min_conf and not (root / "confidence" / f"{i:06d}.png").is_file():
            unsure.append(n)
        else:
            w, h = sizes[n] if k % 2 == 0 else sizes[n][::-1]
            Ki = None if K is None or i >= len(K) else rosbag.rescale(K[i], w / stored[0], h / stored[1])
            try:
                samples[n], xyz, scan = stray.depth_points(root, i, sizes[n], k, Ki, min_conf=min_conf)
            except (ValueError, OSError) as e:  # a damaged image costs only its frame
                bad[n] = str(e)
            else:
                if Ki is not None:
                    clouds[n] = (xyz, at_most(scan, SCAN_POINTS, i))
        samples.setdefault(n, np.zeros((0, 3), np.float32))
        progress(0.1 + 0.75 * (j + 1) / len(frames))
    for left, what in ((missing, "no depth image"), (unsure, "a depth image but no confidence image")):
        if left:
            more = f" and {len(left) - 5} more" if len(left) > 5 else ""
            warnings.append(f"frames {', '.join(left[:5])}{more} have {what}, so they have no samples")
    if bad:
        first, more = next(iter(bad.values())), f" and {len(bad) - 5} more" if len(bad) > 5 else ""
        warnings.append(f"frames {', '.join(list(bad)[:5])}{more} have depth or confidence images that cannot be used, "
                        f"so they have no samples ({first})")
        lost = [n for n in names if n in bad]
        read = [n for n in names if n not in bad and n not in missing and n not in unsure]
        if lost and len(read) < lidar_scale.MIN_VIEWS:
            images = "depth and confidence images" if min_conf else "depth images"  # an export may have no confidence
            raise Failed(f"only {len(read)} of the {len(names)} views have {images} that can be used (at least "
                         f"{lidar_scale.MIN_VIEWS} are needed); {bad[lost[0]]}",
                         {"source": source, "warnings": warnings})
    info = {"source": {**source, "depth_size": video.get("depth_size"), "frames": len(frames)},
            "extrinsic": {"source": "stray (depth aligned with the color camera)", "calibration_sha256": None},
            "intrinsics": {"source": "stray: depth scaled to the frame size", "depth_size": video.get("depth_size"),
                           "frame_size": list(sizes[frames[0]]), "turns": k, "min_confidence": min_conf,
                           "z_range_m": list(stray.Z_RANGE), "K_from": K_from},
            "time": {"base": "frame index"}, "warnings": warnings}
    if not bad:  # else the next run reads the damaged frames again (a disk error may have passed)
        cache_write(J, key, samples, info, clouds)
    return samples, {**info, "source": {**info["source"], "cache": "miss"}}, clouds


SOURCES = {"rosbag": rosbag_samples, "stray": stray_samples}  # job.json "lidar" "kind": its samples


# ---- the clouds of the Inputs tab ----

def cloud_glb(parts):
    """A GLB (bytes) of point sets, or None when every set is empty. parts: {"scan": (N, 3), "used": (M, 3), and
    optionally "da3": (K, 3), "diff": (L, 3), with "diff_m": (L,) and "diff_extras"} camera points in OpenCV axes. Each
    set becomes (x, -y, -z), glTF's axes: x right, y up, z back toward the viewer, with the camera at the origin looking
    down -z as a glTF camera does. One node per set that has points, named as the set, with one POINTS primitive (mode
    0) whose POSITION accessor has min and max, as glTF requires. "diff" also has diff_m as a float SCALAR attribute,
    _DIFF, and diff_extras as the node's extras. Points that are not finite are left out, and a row of "diff" with a
    value that is not finite. Hand-written, so envs/web needs no trimesh."""
    nodes, meshes, accessors, views, chunks, offset = [], [], [], [], [], 0

    def accessor(a, kind, **bounds):  # a as a new accessor of float rows; returns its index
        nonlocal offset
        data = a.tobytes()  # 4 bytes a number, so every view starts 4-byte aligned
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(data), "target": 34962})
        accessors.append({"bufferView": len(views) - 1, "componentType": 5126, "type": kind, "count": len(a), **bounds})
        chunks.append(data)
        offset += len(data)
        return len(accessors) - 1

    for name in ("scan", "used", "da3", "diff"):
        a = np.asarray(parts.get(name, np.zeros((0, 3))), float).reshape(-1, 3)
        keep = np.isfinite(a).all(1)
        if name == "diff" and len(a):
            m = np.asarray(parts["diff_m"], float).reshape(-1)
            keep &= np.isfinite(m)
        a = (a[keep] * [1.0, -1.0, -1.0]).astype("<f4")
        if not len(a):
            continue  # glTF has no accessor of count 0
        attributes = {"POSITION": accessor(a, "VEC3", min=a.min(0).tolist(), max=a.max(0).tolist())}
        node = {"name": name, "mesh": len(meshes)}
        if name == "diff":
            attributes["_DIFF"] = accessor(m[keep].astype("<f4"), "SCALAR")
            node["extras"] = parts["diff_extras"]
        meshes.append({"name": name, "primitives": [{"attributes": attributes, "mode": 0}]})
        nodes.append(node)
    if not nodes:
        return None
    doc = {"asset": {"version": "2.0", "generator": "HARPOR lidar"}, "scene": 0,
           "scenes": [{"nodes": list(range(len(nodes)))}], "nodes": nodes, "meshes": meshes, "accessors": accessors,
           "bufferViews": views, "buffers": [{"byteLength": offset}]}
    js = json.dumps(doc, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    binary = b"".join(chunks)
    binary += b"\0" * (-len(binary) % 4)
    return (struct.pack("<III", 0x46546C67, 2, 28 + len(js) + len(binary)) + struct.pack("<II", len(js), 0x4E4F534A)
            + js + struct.pack("<II", len(binary), 0x004E4942) + binary)


def references(est):
    """{view: the s that compares it with DA3} of an estimate: the job's combined s when it has one, else the view's
    own, as the check images pick it, so a failed scale can still be looked at. A view with neither is left out. The
    combined s, because the differences between views are what the comparison shows."""
    out = {}
    for v in est["views"]:
        s = v.get("s") if est["s"] is None else est["s"]
        if s is not None:
            out[v["name"]] = s
    return out


def view_diff(xyz, p, s):
    """cloud_glb's "diff" parts of a view: the LiDAR points with a ratio r (p: the view's points_out, row by row in
    xyz) and z (1 - s / r), meters, signed. Its extras: s, the median over the used points (None without any) and
    their count."""
    r = np.asarray(p["r"], float)
    X = np.asarray(xyz, float)[np.asarray(p["index"], int)]
    keep = np.isfinite(X).all(1) & np.isfinite(r) & (r > 0)
    m = X[keep, 2] * (1 - s / r[keep])
    used = m[np.asarray(p["used"], bool)[keep]]
    used = used[np.isfinite(used)]
    return {"diff": X[keep], "diff_m": m, "diff_extras": {
        "s": float(s), "median_m": float(np.median(used)) if len(used) else None, "used": len(used)}}


def da3_surface(depth, crop, camera, s):
    """DA3's surface at scale s as camera points (at most DA3_POINTS, float32): each DA3 pixel with a depth, at a
    fixed stride, goes through crop to the source frame, then along its ray (camera: K, D, model and size, as in rays)
    to s x its depth along the optical axis. A pixel that the camera model cannot invert is left out."""
    depth = np.asarray(depth, float)
    h, w = depth.shape
    step = 1
    while depth[::step, ::step].size > DA3_POINTS:
        step += 1
    d = depth[::step, ::step]
    gv, gu = np.nonzero(np.isfinite(d) & (d > 0))
    u, v = lidar_scale.grid_to_source(gu * step, gv * step, crop, (w, h))
    x, y, ok = lidar_scale.unproject(u, v, camera["K"], camera["D"], camera["model"], camera["size"])
    z = s * d[gv, gu]
    return np.stack([x * z, y * z, z], 1)[ok].astype(np.float32)


def compared(clouds, points, refs, cameras):
    """{view: the parts of its comparison with DA3 for cloud_glb}: "diff" for each view with a reference s (refs), and
    "da3" too for a view with a camera (cameras: rays' output, None for a Stray capture) and diff rows, since the viewer
    draws it only with them."""
    out = {}
    for n, (xyz, _) in clouds.items():
        if n in refs and n in points:
            out[n] = view_diff(xyz, points[n], refs[n])
            if n in (cameras or {}) and len(out[n]["diff"]):
                out[n]["da3"] = da3_surface(points[n]["depth"], points[n]["crop"], cameras[n], refs[n])
    return out


def write_clouds(folder, clouds, points, compare=None):
    """folder/<view>.glb for each view of clouds ({view: (xyz, scan)}) that has points: its whole scan, the points
    that the scale used (points: lidar_scale.estimate's points_out; the used rows of xyz) and its comparison with DA3
    (compare: compared's). Returns the views written."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    out = []
    for n, (xyz, scan) in sorted(clouds.items()):
        p = points.get(n)
        used = np.asarray(xyz)[np.asarray(p["index"])[np.asarray(p["used"], bool)]] if p else np.zeros((0, 3))
        data = cloud_glb({"scan": scan, "used": used, **(compare or {}).get(n, {})})
        if data is None:
            continue
        tmp = folder / f".tmp_{n}.glb"
        tmp.write_bytes(data)
        os.replace(tmp, folder / f"{n}.glb")
        out.append(n)
    return out


# ---- the stage ----

def record(rec, info, t0, kind):
    """pool/lidar.json: lidar_scale.estimate's record (or a failed one) with the source's fields; kind: the source's
    kind, which picks the version (lidar_scale.VERSIONS)."""
    out = {"version": lidar_scale.VERSIONS[kind], "ok": rec["ok"], "s": rec["s"], "reason": rec["reason"],
           "source": info.get("source"), "extrinsic": info.get("extrinsic"), "intrinsics": info.get("intrinsics"),
           "time": info.get("time")}
    out.update({k: v for k, v in rec.items() if k not in out and k not in ("warnings", "seconds")})
    out["warnings"] = list(info.get("warnings") or []) + list(rec.get("warnings") or [])
    out["seconds"] = round(time.time() - t0, 2)
    return out


def scale(job_dir):
    """Write J/pool/lidar.json (see the module docstring); SystemExit with the reason when it is not ok."""
    from harpor.stage_a.mvsam3d import view_names
    from harpor.webapp import steps

    J = Path(job_dir)
    t0 = time.time()
    job = steps.job(J)
    spec = steps.lidar(job)
    if spec is None:
        raise SystemExit("this job has no LiDAR (job.json lidar is null), so it has no metric scale stage")
    if spec["kind"] not in SOURCES:
        raise SystemExit(f"no LiDAR source of kind {spec['kind']!r}")
    names = view_names(J / "pool")
    if not names or not (J / "pool" / "da3_output.npz").is_file():
        raise SystemExit("the pool has no views with DA3 depth; run prep and da3 first")
    progress(0)
    check, cloud = J / "lidar" / "check", J / "lidar" / "cloud"
    for old in (check, cloud):  # the check images and clouds of views that left the pool
        shutil.rmtree(old, ignore_errors=True)
    out = J / "pool" / "lidar.json"
    out.unlink(missing_ok=True)
    try:
        samples, info, clouds = SOURCES[spec["kind"]](J, job, spec, names)
    except Failed as e:
        rec = record({"ok": False, "s": None, "reason": str(e), "views": [], "combine": None}, e.info, t0, spec["kind"])
        write_json(out, rec, indent=1)
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
    pts = {}
    try:
        cameras = rays(info, clouds) if spec["kind"] == "rosbag" else None
        est = lidar_scale.estimate(J / "pool", {n: samples[n] for n in names if n in samples}, check_dir=check,
                                   points_out=pts, source=spec["kind"],  # the reason names the source's causes
                                   rays=cameras)
    except ValueError as e:  # a pool that prep and da3 left inconsistent
        write_json(out, record({"ok": False, "s": None, "reason": str(e), "views": [], "combine": None}, info, t0,
                               spec["kind"]),
                   indent=1)
        raise SystemExit(f"no metric scale: {e}") from None
    shown = {n: clouds[n] for n in names if n in clouds}
    # With the check images, also when not ok.
    write_clouds(cloud, shown, pts, compared(shown, pts, references(est), cameras))
    rec = record(est, info, t0, spec["kind"])
    write_json(out, rec, indent=1)  # last: with ok, the stage's done marker
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
