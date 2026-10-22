"""Frames of the uploaded video, rosbag or Stray Scanner capture: facts, sharpness per frame, candidate
views and the frame pool.

Runs in envs/web (PyAV, rosbags, numpy, pillow), from the gator_clone root:

    python -m gator.webapp.video topics --job J
    python -m gator.webapp.video analyze --job J
    python -m gator.webapp.video extract --job J

The input is the file named by J/job.json "video", read as job.json "source" says: {"kind": "video"}
(also when "source" is missing), {"kind": "rosbag", "topic": T}, the camera topic T of a bag
(rosbag.py), or {"kind": "stray"}, a zipped Stray Scanner export (stray.py), unpacked into
J/video/stray/export the first time it is read. Everything below is the same for every kind.

topics (rosbags only) lists the bag's camera image topics in J/frames/topics.json {"topics": [{topic,
type, count, duration_s, rate_hz, width, height, format, rectified, camera_info, distortion,
preview}], "skipped": [{topic, type, why}]} (rosbag.image_topics) and writes each topic's middle
message, long side 320 px, as J/frames/topic_previews/<i>.jpg, i its place in the list.

analyze decodes the input once. For every frame it measures the sharpness lv (variance of a 3x3
Laplacian of the gray frame at width 960), rel = lv / the median lv of the frames within 0.5 s
(motion blur shows as a dip, whatever the scene's texture), the mean luma and the shares of clipped
pixels (>= 250, <= 5). It writes J/frames/meta.json ("video" is the probe, with "kind" and, for a
rosbag, "topic") and J/frames/thumbs/<idx>.jpg (long side 320 px; every frame up to 600 frames, else
an even stride plus the candidates). For a rosbag it also writes J/frames/stamps.json {topic, clock,
header_ns, log_ns}, every frame's header stamp and log time in ns (index i = frame i), before
meta.json; meta.json stays small because the server sends it whole. Every step prints `PROGRESS x`
lines (0 to 1) for the server.

Candidates: the video is cut into K = 2 per second (16 to 32) equal time bins, and each bin gives
its sharpest frame by rel within the central 60% of the bin, so two candidates are never next to
each other. A bin gives none when that frame is blurred (rel < 0.6) or badly exposed (mean luma
outside 30 to 225).

extract writes the pool, the candidates plus the user's picks in manual mode, each pick snapped to
the sharpest frame within 0.25 s, as J/scene/images/f<idx:05d>.png (full size, upright), and
J/frames/pool.json {pool, names, picks: {user frame: snapped frame}}.

Frames are numbered in display order and timed by their own timestamps, because phone video has a
variable frame rate. PyAV leaves frames as stored and reports frame.rotation; frames() turns them
upright. HDR video (HLG or PQ) is converted to 8-bit RGB as it is, which gives flat colors; probe
reports it so the UI can warn. A rosbag's frames are its messages in log-time order (rosbag.py). A
Stray Scanner capture's rgb.mp4 is stored as the sensor sees it; its turn comes from gravity in the
odometry (stray.probe's rotation), which replaces the container's (frames(turns=)). It is read
without its edit list (stray.OPEN), which hides Stray's first frame. meta.json's "video" gains the
depth and odometry counts and the capture's warnings.
"""

import argparse
import io
import json
import os
from pathlib import Path

import numpy as np

HDR = {18: "HLG", 16: "PQ"}  # AVCOL_TRC_ARIB_STD_B67, AVCOL_TRC_SMPTE2084
METRIC_WIDTH = 960
THUMB_SIDE = 320
MAX_THUMBS = 600


def open_video(path, options=None):
    """(container, its first video stream). options: FFmpeg's demuxer options, such as stray.OPEN."""
    import av

    try:
        c = av.open(str(path), options=options or {})
    except (av.FFmpegError, OSError) as e:
        raise SystemExit(f"cannot read {path}: {e}")
    if not c.streams.video:
        c.close()
        raise SystemExit(f"{path} has no video stream")
    s = c.streams.video[0]
    s.thread_type = "AUTO"
    return c, s


def frames(path, keep=None, width=None, turns=None, options=None):
    """Yield (index, time_s, upright RGB uint8 HxWx3) in display order.
    keep: set of frame indices (None = all). width: upright output width (None = full size).
    turns: counterclockwise quarter turns that replace the container's rotation (None: the container's),
    so that a frame is never turned twice. options: as open_video takes them."""
    c, s = open_video(path, options)
    last = max(keep, default=-1) if keep is not None else None
    with c:
        for i, f in enumerate(c.decode(s)):
            if last is not None and i > last:
                break
            if keep is not None and i not in keep:
                continue
            k = (f.rotation // 90) % 4 if turns is None else turns % 4  # counterclockwise quarter turns to upright
            size = {}
            if width:
                w, h = (f.width, f.height) if k % 2 == 0 else (f.height, f.width)
                tw, th = width, round(width * h / w)
                size = {"width": tw, "height": th} if k % 2 == 0 else {"width": th, "height": tw}
            rgb = f.to_ndarray(format="rgb24", **size)
            if k:
                rgb = np.ascontiguousarray(np.rot90(rgb, k))
            yield i, float(f.time), rgb


def probe(path, options=None):
    """Facts of the video; n_frames is counted by decoding (headers can be wrong). options: as open_video
    takes them."""
    c, s = open_video(path, options)
    with c:
        cc = s.codec_context
        n, first = 0, None
        for f in c.decode(s):
            if first is None:
                first = {"w": f.width, "h": f.height, "rotation": int(f.rotation), "trc": f.color_trc, "t": float(f.time)}
            last_t = float(f.time)
            n += 1
        if not n:
            raise SystemExit(f"no frames decoded from {path}")
        if s.duration:
            duration = float(s.duration * s.time_base)
        elif c.duration:
            duration = c.duration / 1e6
        else:
            duration = (last_t - first["t"]) * n / max(n - 1, 1)
        k = (first["rotation"] // 90) % 4
        w, h = (first["w"], first["h"]) if k % 2 == 0 else (first["h"], first["w"])
        return {"n_frames": n, "duration_s": round(duration, 4), "fps": round(n / duration, 4) if duration else None,
                "width": w, "height": h, "rotation": first["rotation"], "codec": cc.name,
                "hdr": HDR.get(cc.color_trc) or HDR.get(first["trc"]),
                "bit_depth": cc.format.components[0].bits if cc.format else None}


def gray(rgb):
    return rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)


def frame_stats(g):
    """(lv, mean luma, share >= 250, share <= 5) of one gray frame."""
    lap = g[1:-1, :-2] + g[1:-1, 2:] + g[:-2, 1:-1] + g[2:, 1:-1] - 4 * g[1:-1, 1:-1]
    return float(lap.var()), float(g.mean()), float((g >= 250).mean()), float((g <= 5).mean())


def relative(lv, times, window_s=0.5):
    """lv over the median lv of the frames within window_s (times ascending)."""
    lv, times = np.asarray(lv, float), np.asarray(times, float)
    lo = np.searchsorted(times, times - window_s - 1e-9, "left")
    hi = np.searchsorted(times, times + window_s + 1e-9, "right")
    med = np.array([np.median(lv[a:b]) for a, b in zip(lo, hi)])
    return np.divide(lv, med, out=np.zeros_like(lv), where=med > 0)


def metrics(gray_frames, times):
    """Per-frame arrays lv, rel, luma, clip_hi, clip_lo of gray frames (pass them at width 960)."""
    s = np.array([frame_stats(g) for g in gray_frames], float).reshape(-1, 4)
    return {"lv": s[:, 0], "rel": relative(s[:, 0], times), "luma": s[:, 1], "clip_hi": s[:, 2], "clip_lo": s[:, 3]}


def candidates(times, rel, luma, duration):
    """Sorted frame indices: the sharpest frame of each time bin (see the module docstring)."""
    times, rel, luma = np.asarray(times, float), np.asarray(rel, float), np.asarray(luma, float)
    K = int(np.clip(round(2 * duration), 16, 32))
    edges = times[0] + np.linspace(0, duration, K + 1)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        idx = np.flatnonzero((times >= a + 0.2 * (b - a)) & (times <= b - 0.2 * (b - a)))
        if not len(idx):
            continue
        i = idx[np.argmax(rel[idx])]
        if rel[i] >= 0.6 and 30 <= luma[i] <= 225:
            out.append(int(i))
    return sorted(set(out))  # bins of a zero duration all give the same frame


def snap(index, times, rel, window_s=0.25):
    """The sharpest frame (by rel) within window_s of frame index; ties go to the nearest."""
    times, rel = np.asarray(times, float), np.asarray(rel, float)
    idx = np.flatnonzero(np.abs(times - times[index]) <= window_s + 1e-9)
    return int(max(idx, key=lambda j: (rel[j], -abs(j - index))))


def write_json(path, data):
    tmp = path.with_name(".tmp_" + path.name)
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


def load_job(job_dir):
    f = Path(job_dir) / "job.json"
    if not f.exists():
        raise SystemExit(f"{f} not found")
    job = json.loads(f.read_text())
    video = Path(job["video"]["path"])
    video = video if video.is_absolute() else Path(job_dir) / video
    if not video.exists():
        raise SystemExit(f"the uploaded file {video} is missing")
    return job, video


def source(job, path, extra=None):
    """(probe(), frames(info, keep=None, width=None)) of the job's input, as job["source"] says; info is
    probe()'s dict, whose clock times a rosbag's frames and whose rotation turns a Stray Scanner
    capture's frames. extra: a dict that a rosbag's probe() fills with the "header_ns" and "log_ns"
    lists of its frames (rosbag.probe's stamps_out); the other kinds leave it alone. A Stray Scanner
    zip is unpacked into path.parent / "stray" first (stray.unpack, which does nothing the second
    time)."""
    src = job.get("source") or {"kind": "video"}
    if src.get("kind") == "video":
        return (lambda: {**probe(path), "kind": "video"}), (lambda info, **kw: frames(path, **kw))
    if src.get("kind") == "stray":
        from gator.webapp import stray

        root = stray.unpack(path, Path(path).parent / "stray")
        return (lambda: stray.probe(root)), (lambda info, **kw: stray.frames(root, info, **kw))
    if src.get("kind") != "rosbag":
        raise SystemExit(f"unknown input kind {src.get('kind')!r} in job.json")
    from gator.webapp import rosbag

    topic = src.get("topic")
    if not topic:
        raise SystemExit("choose the bag's camera topic first")
    return ((lambda: {**rosbag.probe(path, topic, stamps_out=extra), "kind": "rosbag", "topic": topic}),
            (lambda info, **kw: rosbag.frames(path, topic, clock=info.get("clock"), **kw)))


def topics(job_dir):
    from PIL import Image

    from gator.webapp import rosbag

    job_dir = Path(job_dir)
    job, path = load_job(job_dir)
    kind = (job.get("source") or {}).get("kind") or "video"
    if kind != "rosbag":
        raise SystemExit(f"topics is for a rosbag job; this job's input is "
                         f"{'a Stray Scanner capture' if kind == 'stray' else 'a video'}")
    print("PROGRESS 0", flush=True)
    previews = {}
    out = rosbag.image_topics(path, previews, THUMB_SIDE, lambda x: print(f"PROGRESS {0.95 * x:.3f}", flush=True))
    pdir = job_dir / "frames" / "topic_previews"
    pdir.mkdir(parents=True, exist_ok=True, mode=0o700)
    for old in pdir.glob("*.jpg"):  # an earlier listing's previews
        old.unlink()
    for i, t in enumerate(out["topics"]):
        tmp = pdir / f".tmp_{i}.jpg"
        Image.fromarray(previews[t["topic"]]).save(tmp, "JPEG", quality=85)
        os.replace(tmp, pdir / f"{i}.jpg")
        t["preview"] = f"frames/topic_previews/{i}.jpg"
        print(f"{t['topic']}: {t['type']} {t['format']} {t['width']}x{t['height']}, {t['count']} messages, "
              f"{t['rate_hz']} Hz, rectified {t['rectified']}, camera_info {t['camera_info']}, distortion {t['distortion']}")
    for t in out["skipped"]:
        print(f"skipped {t['topic']} ({t['type']}): {t['why']}")
    write_json(job_dir / "frames" / "topics.json", out)  # last: the done marker
    print(f"{len(out['topics'])} camera image topics in {path.name}")
    print("PROGRESS 1", flush=True)
    return out


def analyze(job_dir):
    from PIL import Image

    job_dir = Path(job_dir)
    job, video = load_job(job_dir)
    stamps = {}
    probe_input, read = source(job, video, stamps)
    print("PROGRESS 0", flush=True)
    info = probe_input()
    n, w, h = info["n_frames"], info["width"], info["height"]
    print(f"{video.name}{' ' + info['topic'] if info['kind'] == 'rosbag' else ''}: {n} frames, "
          f"{info['duration_s']:.2f} s, {info['fps']} fps, {w}x{h}, {info['codec']}, "
          f"rotation {info['rotation']}, HDR {info['hdr']}, {info['bit_depth']} bit")
    if info["kind"] == "rosbag":
        print(f"frame times from the {'header stamps' if info['clock'] == 'header' else 'log time'}")
    if info["kind"] == "stray":
        d = info["depth_size"]
        agree = info["upright_agreement"]
        print(f"Stray Scanner capture: {info['n_depth']} depth images ({d[0]}x{d[1]}), {info['n_odometry']} poses; "
              f"turned {info['rotation']} degrees upright by gravity "
              + ("(no frame shows it)" if agree is None else f"({agree:.0%} of the frames agree)"))
        for msg in info["warnings"]:
            print(f"warning: {msg}")
    if info.get("stamp_warnings"):
        print(f"warning: {info['stamp_warnings']} header stamps are zero, go backwards or repeat; the frames are "
              "timed by log time")
    print("PROGRESS 0.05", flush=True)
    k = min(1.0, THUMB_SIDE / max(w, h))
    tsize = (max(1, round(w * k)), max(1, round(h * k)))
    # Every thumbnail is kept in memory as JPEG bytes (~15 kB each) until the candidates are known.
    jpgs, stats, times = {}, [], []
    every = max(1, n // 50)
    for i, t, rgb in read(info, width=min(METRIC_WIDTH, w)):
        stats.append(frame_stats(gray(rgb)))
        times.append(t)
        buf = io.BytesIO()
        Image.fromarray(rgb).resize(tsize, Image.BILINEAR).save(buf, "JPEG", quality=80)
        jpgs[i] = buf.getvalue()
        if i % every == 0:
            print(f"PROGRESS {0.05 + 0.9 * i / n:.3f}", flush=True)
    s = np.array(stats).reshape(-1, 4)
    rel = relative(s[:, 0], times)
    cand = candidates(times, rel, s[:, 1], info["duration_s"])
    thumbs = sorted(set(range(0, len(times), -(-len(times) // MAX_THUMBS))) | set(cand))
    tdir = job_dir / "frames" / "thumbs"
    tdir.mkdir(parents=True, exist_ok=True, mode=0o700)
    names = {f"{i}.jpg" for i in thumbs}
    for old in tdir.glob("*.jpg"):  # thumbnails of an earlier analysis (another topic of the bag)
        if old.name not in names:
            old.unlink()
    for i in thumbs:
        (tdir / f"{i}.jpg").write_bytes(jpgs[i])
    meta = {"video": info, "times": [round(t, 4) for t in times], "lv": np.round(s[:, 0], 2).tolist(),
            "rel": np.round(rel, 4).tolist(), "luma": np.round(s[:, 1], 2).tolist(),
            "clip_hi": np.round(s[:, 2], 4).tolist(), "clip_lo": np.round(s[:, 3], 4).tolist(),
            "candidates": cand, "thumbs": thumbs, "thumb_size": list(tsize)}
    if info["kind"] == "rosbag":  # the absolute frame times, for matching LiDAR scans to frames
        write_json(job_dir / "frames" / "stamps.json", {"topic": info["topic"], "clock": info["clock"], **stamps})
    write_json(job_dir / "frames" / "meta.json", meta)  # last: the server takes it as the done marker
    print(f"{len(cand)} candidates: {cand}")
    print(f"rel sharpness: median {np.median(rel):.2f}, {(rel < 0.6).mean():.0%} of frames below 0.6")
    print("PROGRESS 1", flush=True)
    return meta


def extract(job_dir):
    from PIL import Image

    job_dir = Path(job_dir)
    job, video = load_job(job_dir)
    _, read = source(job, video)
    meta = json.loads((job_dir / "frames" / "meta.json").read_text())
    picks = {}
    if job.get("mode") == "manual":
        n = len(meta["times"])
        for p in job.get("picks") or []:
            if not 0 <= int(p) < n:
                raise SystemExit(f"pick {p} is not a frame of the video (0 to {n - 1})")
            picks[str(p)] = snap(int(p), meta["times"], meta["rel"])
    pool = sorted(set(meta["candidates"]) | set(picks.values()))
    if not pool:
        raise SystemExit("no usable frames: every candidate is blurred or badly exposed; pick frames by hand")
    names = {str(i): f"f{i:05d}" for i in pool}
    out = job_dir / "scene" / "images"
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    for old in out.glob("*.png"):  # frames of an earlier pool would otherwise get masks too
        if old.stem not in names.values():
            old.unlink()
    print("PROGRESS 0", flush=True)
    done = 0
    for i, _, rgb in read(meta["video"], keep=set(pool)):
        tmp = out / f".tmp_{names[str(i)]}.png"
        Image.fromarray(rgb).save(tmp, "PNG", compress_level=1)
        os.replace(tmp, out / f"{names[str(i)]}.png")
        done += 1
        print(f"PROGRESS {done / len(pool):.3f}", flush=True)
    missing = [nm for nm in names.values() if not (out / f"{nm}.png").exists()]
    if missing:
        raise SystemExit(f"the input ended before frames {missing}")
    write_json(job_dir / "frames" / "pool.json", {"pool": pool, "names": names, "picks": picks})
    moved = {p: s for p, s in picks.items() if int(p) != s}
    print(f"{len(pool)} frames -> {out}" + (f"; picks snapped {moved}" if moved else ""))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="step", required=True)
    for name in ("topics", "analyze", "extract"):
        sub.add_parser(name).add_argument("--job", required=True, help="job folder with job.json")
    a = ap.parse_args(argv)
    {"topics": topics, "analyze": analyze, "extract": extract}[a.step](a.job)


if __name__ == "__main__":
    main()
