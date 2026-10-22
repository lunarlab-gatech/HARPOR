"""Choose the views for MV-SAM3D from the pool of video frames (web app stage "select").

Runs in envs/web, from the gator_clone root, after masks, prep and da3 ran on the pool:

    python -m gator.webapp.views select --job J

Reads J/pool (mvsam3d prep and da3 output), J/scene/sam3_<slug>/sam3.json, J/frames/meta.json and
pool.json, and job.json; writes J/selection.json and the run folder J/run. A new selection makes the
later stages' outputs stale, so J/run, J/agent, J/physics and J/results.zip are removed first.

Every pool frame with a mask gets a quality q (quality(): SAM 3 score, mask area, truncation at
the image border, sharpness, clipped pixels in the mask). DA3 gives every view's camera. A view whose
mask disagrees with the others (consistency() < 0.5, checked both ways) is "inconsistent": SAM 3
found something else there or more than the object, or DA3 placed that camera wrong. The object
center c is the median of the confident masked points of the consistent views, and each view's
direction is the unit vector from c to its camera center; views are compared by the angle between
directions.

Auto mode leaves out views with no mask, a mask under 1% of the image or an inconsistent mask, takes
as view 0 (the anchor) the best view whose mask stays off the image border, and fills the n views by
k-center selection on the directions (kcenter()), so the picks spread around the object. While two
picks are closer than 15 deg (and not both view 0 or locked), k-center runs again with one view
fewer, down to 3 views. When over a third of the masked views are inconsistent, selection.json says
"ok": false and the step exits with code 2, so the user can pick by hand. Manual mode takes the
user's frames that have masks. Both modes honour job.json's view0 and its overrides: "locked" views
are always in, "excluded" views never (pool names such as "f00012", or frame numbers). Picks that
end up closer than 15 deg get a warning.

build_run() copies the picks into J/run as views 0..N-1 (view 0 = the anchor) with DA3's arrays
sliced to them, so the mvsam3d sam3d and place steps run on it as on any run folder.

A job with LiDAR (job.json "lidar") has J/pool/lidar.json from the lidar stage (lidar_scale.py).
select stops first, before it removes anything, unless that record is ok; build_run then multiplies
DA3's depth, pointmaps and camera translations by its scale s, so every later stage works in meters.
The pool stays raw and build_run rebuilds J/run from it, so s is applied once however often select
runs. run/prep.json and selection.json record metric_scale (null without LiDAR).

selection.json "ring" places every camera center on the best-fit plane of the camera centers, seen
from above (u right, v up; flip v for SVG), with the object center as "center".
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

from gator.stage_a import lidar_scale
from gator.webapp import steps

MIN_AREA = 0.01
MIN_CONSISTENCY = 0.5
MIN_SPACING_DEG = 15.0


def pool_name(x):
    """'f00012' for frame 12; a pool name stays as it is."""
    return x if isinstance(x, str) and not x.isdigit() else f"f{int(x):05d}"


def frame_of(name):
    return int(name[1:])


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def to_world(points_c, E):
    """Camera points (N, 3) to DA3's world; E is OpenCV world-to-camera 3x4."""
    E = np.asarray(E, float)
    return (points_c - E[:3, 3]) @ E[:3, :3]


def at_size(mask, hw):
    from gator.stage_a.place import resize_mask

    return mask if mask.shape == tuple(hw) else resize_mask(mask, (hw[1], hw[0]))


def valid(P):
    return np.isfinite(P).all(-1) & (P[..., 2] > 1e-6)


def camera_centers(E):
    E = np.asarray(E, float)
    return -np.einsum("nij,ni->nj", E[:, :3, :3], E[:, :3, 3])


def object_center(pointmaps, conf, masks, E):
    """Median of the masked points whose conf is at or above their view's median, in DA3's world."""
    pts = []
    for P, cf, m, e in zip(pointmaps, conf, masks, E):
        m = at_size(m, P.shape[:2]) & valid(P)
        if m.any():
            c = cf[m]
            pts.append(to_world(P[m][c >= np.median(c)], e))
    if not pts:
        raise SystemExit("no masked DA3 points in any view")
    return np.median(np.concatenate(pts), axis=0)


def consistency(pointmaps, masks, E, K, dilate_frac=0.02, budget=100000):
    """Per view, the smaller of two shares, with every mask grown by dilate_frac of the image diagonal:
    the share of the other views' masked points that project into its mask (low when the mask misses
    the object or DA3 placed the camera wrong), and the share of its own masked points that project
    into the other views' masks (low when the mask holds more than the object, a table, say). All
    views are thinned with one stride to about budget points."""
    from scipy.ndimage import distance_transform_edt

    h, w = pointmaps[0].shape[:2]
    masks = [at_size(m, (h, w)) for m in masks]
    sel = [m & valid(P) for P, m in zip(pointmaps, masks)]
    step = max(1, sum(int(s.sum()) for s in sel) // budget)
    world = [to_world(P[s][::step], e) for P, s, e in zip(pointmaps, sel, E)]
    grown = [distance_transform_edt(~m) <= dilate_frac * np.hypot(h, w) for m in masks]
    hits = np.zeros((len(masks), len(masks)))  # hits[i, j]: view i's points inside view j's grown mask
    for j, (e, k) in enumerate(zip(E, K)):
        for i, p in enumerate(world):
            if i == j or not len(p):
                continue
            x = p @ np.asarray(e)[:3, :3].T + np.asarray(e)[:3, 3]
            z = np.maximum(x[:, 2], 1e-9)
            u = np.round(k[0, 0] * x[:, 0] / z + k[0, 2]).astype(int)
            v = np.round(k[1, 1] * x[:, 1] / z + k[1, 2]).astype(int)
            ok = (x[:, 2] > 1e-6) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            hits[i, j] = grown[j][v[ok], u[ok]].sum()
    count = np.array([len(p) for p in world], float)
    into = hits.sum(0) / np.maximum(count.sum() - count, 1)  # the other views' points in this mask
    out = hits.sum(1) / np.maximum(count * (len(masks) - 1), 1)  # this view's points in the other masks
    return np.minimum(into, out)


def truncated(mask, px=2):
    return bool(mask[:px].any() or mask[-px:].any() or mask[:, :px].any() or mask[:, -px:].any())


def quality(score, area_frac, truncated, rel, clipped_frac):
    """(q, terms): q is the product of the terms."""
    terms = {"score": float(score), "area": float(min(1.0, np.sqrt(area_frac / 0.05))),
             "truncated": 0.5 if truncated else 1.0, "sharp": float(min(1.0, rel / 0.9)),
             "clipped": 0.5 if clipped_frac > 0.02 else 1.0}
    return float(np.prod(list(terms.values()))), terms


def angles(dirs):
    d = unit(dirs)
    return np.degrees(np.arccos(np.clip(d @ d.T, -1.0, 1.0)))


def kcenter(dirs, q, n, anchor, locked=()):
    """n indices into dirs, anchor first, then the locked ones: greedy k-center on the angles, then
    1-swap local search with anchor and locked fixed. A swap is taken when the coverage radius (the
    largest angle from any view to its nearest pick) drops by 1 deg or more. Within 1 deg of the best
    radius so far, it is taken when fewer views sit at that radius (this gets past plateaus where two
    gaps are equally wide), else when the smallest angle between picks grows, else the sum of q."""
    D, q = angles(dirs), np.asarray(q, float)
    fixed = list(dict.fromkeys([anchor, *locked]))
    S = list(fixed)
    while len(S) < min(n, len(D)):
        d = D[:, S].min(1)
        S.append(max((i for i in range(len(D)) if i not in S), key=lambda i: (round(d[i], 6), q[i], -i)))

    def obj(S, r_min):
        near = D[:, S].min(1)
        pair = D[np.ix_(S, S)][np.triu_indices(len(S), 1)]
        return near.max(), -int((near >= r_min - 1e-6).sum()), pair.min() if len(pair) else 180.0, q[S].sum()

    r_min = obj(S, 0)[0]
    best = obj(S, r_min)
    improved = True
    while improved:  # ends: r_min only falls, and with r_min fixed each swap raises (count, spacing, q)
        improved = False
        for k in range(len(fixed), len(S)):
            for c in range(len(D)):
                if c in S:
                    continue
                T = S[:k] + [c] + S[k + 1:]
                o = obj(T, r_min)
                if o[0] <= best[0] - 1 or (o[0] < r_min + 1 and o[1:] > best[1:]):
                    S, r_min = T, min(r_min, o[0])
                    best, improved = obj(S, r_min), True
    return S


def thin(S, D, q, fixed=(), min_deg=MIN_SPACING_DEG, keep=3):
    """Drop the lower-q free view of the closest pair with a free view (one not in fixed) while that
    pair is closer than min_deg and more than keep views are left. Two fixed views may stay close."""
    S = list(S)
    while len(S) > keep:
        free = np.array([i not in fixed for i in S])
        sub = np.where(free[:, None] | free[None, :], D[np.ix_(S, S)], np.inf)
        np.fill_diagonal(sub, np.inf)
        a, b = np.unravel_index(np.argmin(sub), sub.shape)
        if sub[a, b] >= min_deg:
            break
        S.remove(min((S[i] for i in (a, b) if free[i]), key=lambda i: q[i]))
    return S


def ring(centers, E, c):
    """Camera centers and the object center on the best-fit plane of the camera centers, normal up."""
    mu = centers.mean(0)
    vt = np.linalg.svd(centers - mu)[2] if len(centers) > 1 else np.eye(3)
    e1, n = vt[0], np.cross(vt[0], vt[1])
    up = -np.asarray(E, float)[:, 1, :3].mean(0)  # OpenCV y is down
    n = n if n @ up >= 0 else -n
    B = np.stack([e1, np.cross(n, e1)], 1)
    return np.round((c - mu) @ B, 4).tolist(), np.round((centers - mu) @ B, 4).tolist()


def build_run(pool_dir, names, run_dir, metric=None):
    """Copy the pool views names (anchor first) into run_dir as views 0..N-1, with da3_output.npz
    sliced to them, and write run_dir/prep.json. metric: the LiDAR scale s, applied to the sliced
    arrays (lidar_scale.apply_scale), or None."""
    pool_dir, run_dir = Path(pool_dir), Path(run_dir)
    if run_dir.exists():
        shutil.rmtree(run_dir)
    for sub in ("input/images", "input/object", "masks"):
        (run_dir / sub).mkdir(parents=True)
        for i, n in enumerate(names):
            shutil.copyfile(pool_dir / sub / f"{n}.png", run_dir / sub / f"{i}.png")
    with np.load(pool_dir / "da3_output.npz") as da:
        stem = {Path(str(f)).stem: k for k, f in enumerate(da["image_files"])}
        order, count, arrays = [stem[n] for n in names], len(da["image_files"]), {}
        for k in da.files:  # every per-view array; process_res stays
            a = da[k]
            arrays[k] = a[order] if a.ndim and a.shape[0] == count else a
    if metric is not None:
        arrays = lidar_scale.apply_scale(arrays, metric)
    arrays["image_files"] = np.array([str(run_dir / "input" / "images" / f"{i}.png") for i in range(len(names))])
    (run_dir / "prep.json").write_text(json.dumps(
        {"source": "webapp pool", "metric_scale": None if metric is None else float(metric),
         "views": [{"name": str(i), "pool_name": n, "frame": frame_of(n)} for i, n in enumerate(names)]},
        indent=1))
    tmp = run_dir / ".tmp_da3_output.npz"
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, run_dir / "da3_output.npz")  # last: with selection.json, the stage's done marker


def write_json(path, data):
    tmp = path.with_name(".tmp_" + path.name)
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


def metric_scale(job_dir, job):
    """The LiDAR scale s from J/pool/lidar.json, or None for a job without LiDAR (steps.lidar). Stops with
    the reason when the lidar stage found no usable scale."""
    try:
        if steps.lidar(job) is None:
            return None
    except ValueError as e:
        raise SystemExit(f"job.json: {e}") from None
    path = Path(job_dir) / "pool" / "lidar.json"
    if not path.is_file():
        raise SystemExit("LiDAR is on but pool/lidar.json is missing; run the metric scale (LiDAR) stage first")
    try:
        rec = json.loads(path.read_text())
    except ValueError:
        raise SystemExit("pool/lidar.json is not valid JSON; run the metric scale (LiDAR) stage again") from None
    if not isinstance(rec, dict):
        raise SystemExit("pool/lidar.json is not a LiDAR record; run the metric scale (LiDAR) stage again")
    if rec.get("ok") is not True:
        raise SystemExit(f"the LiDAR metric scale failed: {rec.get('reason') or 'no reason given'}. "
                         "Fix it and retry, or build again with LiDAR off")
    s = rec.get("s")
    if isinstance(s, bool) or not isinstance(s, (int, float)) or not np.isfinite(s) or s <= 0:
        raise SystemExit(f"pool/lidar.json has no usable scale ({s!r}); run the metric scale (LiDAR) stage again")
    return float(s)


def metric_record(job_dir, s):
    """selection.json's metric_scale: s, the views used and their spread (from J/pool/lidar.json), or None."""
    if s is None:
        return None
    c = json.loads((Path(job_dir) / "pool" / "lidar.json").read_text()).get("combine") or {}
    return {"s": s, "views_used": c.get("views_used"), "spread_rel": c.get("spread_rel")}


def measure(job_dir, job, meta):
    """{pool name: view facts} of the pool views that have a mask and a DA3 camera."""
    from PIL import Image

    from gator.stage_a import mvsam3d
    from gator.stage_a.sam3_masks import slug

    sam = json.loads((job_dir / "scene" / f"sam3_{job.get('slug') or slug(job['prompt'])}" / "sam3.json").read_text())
    scores = {v["name"]: max([v["instances"][k]["score"] for k in v.get("picked") or []], default=0.0)
              for v in sam["views"]}
    pool_dir = job_dir / "pool"
    with np.load(pool_dir / "da3_output.npz") as da:
        da = {k: da[k] for k in ("image_files", "pointmaps", "conf", "extrinsics", "intrinsics")}
    stem = {Path(str(f)).stem: k for k, f in enumerate(da["image_files"])}
    view = {}
    for n in mvsam3d.view_names(pool_dir):
        if n not in stem:
            continue
        i = stem[n]
        m = np.array(Image.open(pool_dir / "masks" / f"{n}.png")) > 0
        g = np.asarray(Image.open(pool_dir / "input" / "images" / f"{n}.png").convert("L"))
        clipped = float(((g[m] >= 250) | (g[m] <= 5)).mean()) if m.any() else 0.0
        q, terms = quality(scores.get(n, 0.0), m.mean(), truncated(m), meta["rel"][frame_of(n)], clipped)
        view[n] = {"q": q, "terms": terms, "area": float(m.mean()), "truncated": truncated(m),
                   "mask": at_size(m, da["pointmaps"].shape[1:3]), "P": da["pointmaps"][i], "conf": da["conf"][i],
                   "E": da["extrinsics"][i][:3], "K": da["intrinsics"][i]}
    return view


def select(job_dir):
    """Write J/selection.json and J/run (see the module docstring). Returns the selection dict."""
    job_dir = Path(job_dir)
    job = json.loads((job_dir / "job.json").read_text())
    metric = metric_scale(job_dir, job)  # first: a missing or failed LiDAR scale leaves the earlier outputs alone
    for stale in ("run", "agent", "physics"):
        shutil.rmtree(job_dir / stale, ignore_errors=True)
    (job_dir / "results.zip").unlink(missing_ok=True)
    meta = json.loads((job_dir / "frames" / "meta.json").read_text())
    pool = json.loads((job_dir / "frames" / "pool.json").read_text())
    mode, prompt = job.get("mode", "auto"), job["prompt"]
    over = job.get("overrides") or {}
    excluded = {pool_name(x) for x in over.get("excluded") or []}
    locked = [pool_name(x) for x in over.get("locked") or [] if pool_name(x) not in excluded]
    view = measure(job_dir, job, meta)
    names = [pool["names"][str(i)] for i in pool["pool"]]
    used = [n for n in names if n in view and n not in excluded]
    out = {"job": job, "meta": meta, "names": names, "view": view, "warnings": [],
           "metric_scale": metric_record(job_dir, metric)}
    warn = out["warnings"].append
    status = {n: "no_mask" if n not in view else "excluded" if n not in used else "available" for n in names}
    if len(used) < 2:
        fail(job_dir, out, status, f"SAM 3 found {prompt!r} in only {len(used)} usable pool frames; "
                                   "try another noun or other frames")

    vs = [view[n] for n in used]
    E = np.array([v["E"] for v in vs])
    agree = consistency([v["P"] for v in vs], [v["mask"] for v in vs], E, [v["K"] for v in vs])
    good = [v for v, s in zip(vs, agree) if s >= MIN_CONSISTENCY] or vs  # a wrong mask's points stay out of c
    out["center"] = c = object_center([v["P"] for v in good], [v["conf"] for v in good], [v["mask"] for v in good],
                                      np.array([v["E"] for v in good]))
    for v, s, C in zip(vs, agree, camera_centers(E)):
        v.update(consistency=float(s), center=C, dir=unit(C - c))
    for n in used:
        status[n] = ("small" if view[n]["area"] < MIN_AREA
                     else "inconsistent" if view[n]["consistency"] < MIN_CONSISTENCY else "available")
    for n in locked:
        if n not in view:
            warn(f"locked view {n} has no mask; left out")
        elif status[n] != "available":
            warn(f"locked view {n} is {status[n]}; kept because it is locked")
    locked = [n for n in locked if n in view]
    view0 = job.get("view0")
    if view0 is not None:
        view0 = pool_name(pool.get("picks", {}).get(str(view0), view0) if mode == "manual" else view0)

    if mode == "manual":
        user = [pool_name(pool.get("picks", {}).get(str(p), p)) for p in job.get("picks") or []]
        for n in user:
            if n not in view:
                warn(f"frame {frame_of(n)} has no mask of {prompt!r}; left out")
            elif n in excluded:
                warn(f"frame {frame_of(n)} is excluded; left out")
            elif status[n] != "available":
                warn(f"frame {frame_of(n)} is {status[n]}; kept because you picked it")
        picks = list(dict.fromkeys([n for n in user if n in used] + locked))
        if len(picks) < 2:
            fail(job_dir, out, status, f"only {len(picks)} of your frames have a mask of {prompt!r}; "
                                       "MV-SAM3D needs at least 2")
        if view0 not in picks and view0 is not None:
            warn(f"view 0 ({view0}) is not among the picks; took the best view")
        anchor = view0 if view0 in picks else max(picks, key=lambda n: view[n]["q"])
    else:
        bad = [n for n in used if status[n] == "inconsistent"]
        if len(bad) > len(used) / 3:
            fail(job_dir, out, status, f"{len(bad)} of {len(used)} masked views disagree with the others "
                                       "(SAM 3 or DA3 went wrong); pick the frames by hand")
        avail = list(dict.fromkeys([n for n in used if status[n] == "available"] + locked))
        if len(avail) < 2:
            fail(job_dir, out, status, f"only {len(avail)} usable views of {prompt!r}")
        if view0 not in avail and view0 is not None:
            warn(f"view 0 ({view0}) is not usable; took the best view")
        whole = [n for n in avail if not view[n]["truncated"]]
        if not whole:
            warn("every usable mask touches the image border; the object may be cut off")
        anchor = view0 if view0 in avail else max(whole or avail, key=lambda n: view[n]["q"])
        n_req = int(job.get("n_views") or 6)
        fixed = [avail.index(n) for n in dict.fromkeys([anchor, *locked])]
        if len(fixed) > n_req:
            warn(f"view 0 and the locked views are {len(fixed)}, more than the {n_req} requested")
        q = np.array([view[n]["q"] for n in avail])
        dirs = [view[n]["dir"] for n in avail]
        D = angles(dirs)
        first = S = kcenter(dirs, q, n_req, fixed[0], fixed[1:])
        while len(thin(S, D, q, fixed)) < len(S):  # two picks closer than 15 deg: k-center with one fewer
            S = kcenter(dirs, q, len(S) - 1, fixed[0], fixed[1:])
        if len(S) < len(first):
            warn(f"the video covers only {D.max():.0f} deg around the object; kept {len(S)} views "
                 f"instead of {len(first)}")
        picks = [avail[i] for i in S]
    picks = [anchor] + [n for n in picks if n != anchor]

    build_run(job_dir / "pool", picks, job_dir / "run", metric=metric)
    sel = record(out, status, picks, anchor)
    if sel["min_spacing_deg"] is not None and sel["min_spacing_deg"] < MIN_SPACING_DEG:
        sel["warnings"].append(f"two picks are only {sel['min_spacing_deg']:.0f} deg apart")
    write_json(job_dir / "selection.json", sel)
    for i, p in enumerate(sel["picks"]):
        print(f"view {i}: {p['name']} at {p['time_s']} s, q {p['q']:.2f}")
    print(f"{mode}: {len(picks)} views, coverage radius {sel['coverage_deg']} deg, "
          f"smallest spacing {sel['min_spacing_deg']} deg -> {job_dir / 'run'}")
    if metric is not None:
        print(f"metric scale {metric:.4f} from LiDAR applied")
    for w in sel["warnings"]:
        print("warning:", w)
    return sel


def record(out, status, picks=(), anchor=None):
    """The selection.json dict; ok when there are picks. metric_scale is {s, views_used, spread_rel}
    for a job with LiDAR, else None."""
    job, view, names, picks = out["job"], out["view"], out["names"], list(picks)
    times = out["meta"]["times"]
    with_dir = [n for n in names if "dir" in view.get(n, {})]
    near, pair = {}, np.zeros(0)
    if picks:
        D = angles([view[n]["dir"] for n in with_dir + picks])
        near = dict(zip(with_dir, D[:len(with_dir), len(with_dir):].min(1)))
        pair = D[len(with_dir):, len(with_dir):][np.triu_indices(len(picks), 1)]
    center, uv = None, {}
    if len(with_dir) >= 2:
        center, pts = ring(np.array([view[n]["center"] for n in with_dir]), np.array([view[n]["E"] for n in with_dir]),
                           out["center"])
        uv = dict(zip(with_dir, pts))
    covered = [near[n] for n in near if n in picks or status.get(n) == "available"]
    return {"ok": bool(picks), "mode": job.get("mode", "auto"),
            "n_requested": int(job.get("n_views") or 6) if job.get("mode", "auto") == "auto" else len(picks),
            "picks": [{"name": n, "frame": frame_of(n), "time_s": times[frame_of(n)], "q": round(view[n]["q"], 4),
                       "q_terms": {k: round(x, 4) for k, x in view[n]["terms"].items()},
                       "consistency": round(view[n]["consistency"], 3),
                       "dir": np.round(view[n]["dir"], 4).tolist()} for n in picks],
            "anchor": anchor,
            "candidates": [{"name": n, "frame": frame_of(n), "time_s": times[frame_of(n)],
                            "status": "picked" if n in picks else status.get(n, "no_mask"),
                            "q": round(view[n]["q"], 4) if n in view else None,
                            "consistency": round(view[n]["consistency"], 3) if "consistency" in view.get(n, {}) else None,
                            "angle_to_nearest_pick": round(float(near[n]), 1) if n in near else None} for n in names],
            "coverage_deg": round(float(max(covered)), 1) if covered else None,
            "min_spacing_deg": round(float(pair.min()), 1) if len(pair) else None,
            "ring": {"center": center, "points": [uv.get(n) for n in names]},
            "metric_scale": out.get("metric_scale"), "warnings": list(out["warnings"])}


def fail(job_dir, out, status, msg):
    out["warnings"].append(msg)
    write_json(Path(job_dir) / "selection.json", record(out, status))
    print(msg)
    raise SystemExit(2)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="step", required=True)
    sub.add_parser("select").add_argument("--job", required=True, help="job folder with job.json")
    a = ap.parse_args(argv)
    select(a.job)


if __name__ == "__main__":
    main()
