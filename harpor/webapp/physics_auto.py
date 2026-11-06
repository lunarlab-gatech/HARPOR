"""Physics inputs for the web app without a person in the loop (stage "align").

Runs in envs/sam3d (numpy, scipy, trimesh, pillow), from the repo root, after place (and the
agent, unless it was skipped):

    python -m harpor.webapp.physics_auto align --job J

align takes the GLB for physics (steps.next_glb: the agent's delivery, or run/posed.glb when the
agent was skipped or delivered nothing usable, run/parts.glb for an articulated job; pick_glb also
refuses a delivery whose size is not that of the GLB the agent started from) and runs
harpor.physics.align on it with --up-near at the up hint and --x footprint, plus --opaque when the GLB
has transparent or transmissive materials, which VoMP rejects. It writes J/physics/up.json {hint,
info, glb, glb_rule, opaque} and J/physics/aligned.glb with aligned.json. For an articulated job
(job.json "articulated") it adds --links-out J/physics/links: one GLB per link, for VoMP.

The up hint (up_hint()) is the normal of the support plane under the object in view 0's DA3
pointmap: a RANSAC plane fit to the confident points in a band around the mask and below its
center (a table top or the floor). Frame W is view 0's OpenCV camera. The reference up is the mean
up of all the run's DA3 cameras in W (cameras_up()); around an orbit it is close to gravity, while
view 0's own up (0, -1, 0) tilts by the camera's pitch. The plane is taken when it holds at least
30% of the points, at least half of the object's masked points are not below it, and its normal is
within 45 deg of the reference, or within 75 deg when it holds at least half of the points (a
camera that looks down steeply at a table). Else the hint is the reference up, and info says why.
Both the posed GLB and the agent's GLB are in frame W.

sim_name() makes the object noun a MuJoCo and USD name.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

CAMERA_UP = np.array([0.0, -1.0, 0.0])
MAX_TILT_DEG, MAX_TILT_WIDE_DEG = 45.0, 75.0  # plane normal to the reference up; past 45, half the points
SEE_THROUGH = {"KHR_materials_transmission", "KHR_materials_volume", "KHR_materials_ior"}


def plane_ransac(pts, threshold, iters=3000, seed=0, max_points=20000):
    """(unit normal, inlier mask over pts) of the plane with the most points within threshold, refined
    by SVD on its inliers."""
    rng = np.random.default_rng(seed)
    sample = pts if len(pts) <= max_points else pts[rng.choice(len(pts), max_points, replace=False)]
    best, best_count = None, -1
    for a, b, c in sample[rng.integers(0, len(sample), (iters, 3))]:
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue
        n = n / norm
        count = np.count_nonzero(np.abs((sample - a) @ n) < threshold)
        if count > best_count:
            best, best_count = (n, a), count
    if best is None:
        return None, np.zeros(len(pts), bool)
    inl = np.abs((pts - best[1]) @ best[0]) < threshold
    mu = pts[inl].mean(0)
    n = np.linalg.svd(pts[inl] - mu, full_matrices=False)[2][2]
    return n, np.abs((pts - mu) @ n) < threshold


def cameras_up(E, i):
    """The mean up of the DA3 cameras E (N, 3, 4), OpenCV world-to-camera, in view i's camera frame
    (view i's own up (0, -1, 0) if the mean is zero). The mean is short when the cameras look down
    steeply around an orbit, but it still points up."""
    R = np.asarray(E, float)[:, :3, :3]
    up = R[i] @ -R[:, 1].mean(0)  # row 1 of R is the camera's y axis (down) in DA3's world
    norm = np.linalg.norm(up)
    return up / norm if norm > 1e-6 else CAMERA_UP.copy()


def up_hint(run_dir, seed=0):
    """(up in frame W, info): the support plane's normal under the object in view 0, or the cameras'
    mean up (cameras_up()) with the reason in info."""
    from scipy.ndimage import distance_transform_edt

    from harpor.stage_a.place import resize_mask
    from harpor.stage_a.sam3d import load_mask

    run_dir = Path(run_dir)
    with np.load(run_dir / "da3_output.npz") as da:
        i = [Path(str(f)).stem for f in da["image_files"]].index("0")
        P, conf, ref = da["pointmaps"][i], da["conf"][i], cameras_up(da["extrinsics"], i)
    h, w = P.shape[:2]
    mask = resize_mask(load_mask(run_dir / "masks" / "0.png"), (w, h))
    ref_info = {"reference_up": np.round(ref, 5).tolist()}

    def fallback(reason, **kw):
        return ref.copy(), {"source": "cameras", "reason": reason, **ref_info, **kw}

    if not mask.any():
        return fallback("view 0's mask is empty")
    diag = np.hypot(h, w)
    d = distance_transform_edt(~mask)
    band = (d > 0.02 * diag) & (d <= 0.12 * diag)
    band[:int(np.nonzero(mask)[0].mean())] = False  # only rows at or below the mask's center (y is down)
    finite = np.isfinite(P).all(-1) & (P[..., 2] > 1e-6)
    pts = P[band & finite & (conf >= np.percentile(conf, 30))].astype(float)
    if len(pts) < 50:
        return fallback(f"only {len(pts)} confident points around and below the object", points=int(len(pts)))
    threshold = 0.01 * float(np.median(pts[:, 2]))
    n, inl = plane_ransac(pts, threshold, seed=seed)
    if n is None:
        return fallback("the points around the object are degenerate", points=int(len(pts)))
    n = n if n @ ref >= 0 else -n
    height = (P[mask & finite] - pts[inl].mean(0)) @ n  # of the object's points above the plane
    above = float(np.mean(height > -threshold)) if len(height) else 0.0
    share, angle = float(inl.mean()), float(np.degrees(np.arccos(np.clip(n @ ref, -1, 1))))
    info = {"points": int(len(pts)), "threshold": round(threshold, 5), "inlier_share": round(share, 3),
            "object_above_share": round(above, 3), "angle_to_reference_deg": round(angle, 1),
            "plane_normal": np.round(n, 5).tolist(), **ref_info}
    if share < 0.3:
        return fallback(f"the best plane holds only {share:.0%} of the points around the object", **info)
    if above < 0.5:
        return fallback(f"only {above:.0%} of the object's points are on or above the plane, so the object "
                        "does not stand on it (a wall behind it?)", **info)
    if angle > MAX_TILT_WIDE_DEG or (angle > MAX_TILT_DEG and share < 0.5):
        return fallback(f"the plane is {angle:.0f} deg from the cameras' up and holds {share:.0%} of the points "
                        "(a wall?)", **info)
    return n, {"source": "plane", **info}


def needs_opaque(glb_path):
    """True when a material of the GLB is transmissive, has volume or IOR, blends or is partly transparent."""
    from harpor.physics.align import read_glb

    doc, _ = read_glb(glb_path)
    for m in doc.get("materials", []):
        alpha = (m.get("pbrMetallicRoughness", {}).get("baseColorFactor") or [1, 1, 1, 1])[3]
        if SEE_THROUGH & set(m.get("extensions", {})) or m.get("alphaMode") == "BLEND" or alpha < 1:
            return True
    return False


def sim_name(prompt):
    """A name for MuJoCo and USD: lower case, [a-z0-9_], not starting with a digit, not scene or world."""
    s = re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")
    if not s or s[0].isdigit() or s in ("scene", "world"):
        s = f"obj_{s}" if s else "obj"
    return s


def pick_glb(job_dir):
    """(path, why) of the GLB for physics, by steps.next_glb: the agent's delivery, else run/parts.glb (an
    articulated job) or run/posed.glb.

    The agent started from that other GLB (agent_step.mode). Its delivery is used only when it keeps that
    GLB's size (controller.check_size); else physics takes that GLB, and why gives the reason.
    """
    from harpor.agent import controller
    from harpor.webapp import steps

    job_dir = Path(job_dir)
    glb = steps.next_glb(job_dir)
    f = job_dir / "agent" / "result.json"
    r = json.loads(f.read_text()) if f.is_file() else {}
    if glb == steps.agent_glb(job_dir):
        jf, parts = job_dir / "job.json", job_dir / "run" / "parts.glb"
        job = json.loads(jf.read_text()) if jf.is_file() else {}
        start = parts if steps.articulated(job) and parts.is_file() else job_dir / "run" / "posed.glb"
        bad = controller.check_size(glb, start)
        if bad:
            return start, f"the agent's {r.get('source')} GLB is not used: {bad}"
        return glb, f"the agent's {r.get('source')} GLB, verdict {r.get('verdict')}"
    return glb, steps.skip("agent", job_dir) or (f"the agent gave no usable GLB (verdict {r.get('verdict')})"
                                                 if r else "no agent result")


def align_job(job_dir, prepare_parts=False):
    from harpor.physics import align
    from harpor.webapp import steps

    job_dir = Path(job_dir)
    f = job_dir / "job.json"
    links = steps.articulated(json.loads(f.read_text()) if f.is_file() else {})
    glb, rule = (job_dir / "run/parts.glb", "Stage B split parts") if prepare_parts else pick_glb(job_dir)
    if not glb.exists():
        raise SystemExit(f"{glb} is missing; run the sam3d and place stages first")
    hint, info = up_hint(job_dir / "run")
    opaque = needs_opaque(glb)
    out = job_dir / ("run/vomp_parts" if prepare_parts else "physics")
    out.mkdir(exist_ok=True, mode=0o700)
    up = {"hint": np.round(hint, 6).tolist(), "info": info, "glb": str(glb), "glb_rule": rule, "opaque": opaque}
    (out / "up.json").write_text(json.dumps(up, indent=1))
    print(f"GLB: {glb} ({rule}); up hint {up['hint']} from the {info['source']}"
          + (f" ({info['reason']})" if "reason" in info else "") + ("; made opaque for VoMP" if opaque else ""))
    # --up-near=... in one argument: argparse takes "-0.1,..." after a space for an option.
    align.main(["--glb", str(glb), "--out", str(out / "aligned.glb"), "--up-near=" + ",".join(map(str, up["hint"])),
                "--x", "footprint", *(["--opaque"] if opaque else []), *(["--links-out", str(out / "links")] if links else [])])


def main(argv=None):
    from harpor.physics.material_transfer import transfer_job

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="step", required=True)
    sub.add_parser("align").add_argument("--job", required=True, help="job folder")
    sub.add_parser("prepare-parts").add_argument("--job", required=True, help="job folder")
    a = ap.parse_args(argv)
    if a.step == "prepare-parts":
        align_job(a.job, prepare_parts=True)
        return
    (Path(a.job) / "physics/summary.json").unlink(missing_ok=True)
    align_job(a.job)
    transfer_job(a.job)


if __name__ == "__main__":
    main()
