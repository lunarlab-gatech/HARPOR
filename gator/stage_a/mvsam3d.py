"""Stage A, multi-view: MV-SAM3D with Depth Anything 3 cameras and depth (build_plan.md note 4b).

Four steps write into one run folder, each in its own process:

  prep   images and masks -> input/images/<name>.png, input/object/<name>.png (RGBA, the object in
         alpha, as MV-SAM3D's loader needs) and masks/<name>.png (L, for the agent's review), all at
         one size: center-cropped to the median aspect ratio, resized to the median size, long side
         at most --max-side
  da3    Depth Anything 3 (envs/da3) -> da3_output.npz: depth, conf, pointmaps (each view's OpenCV
         camera, in DA3 units, meters for the Nested model), pointmaps_sam3d, extrinsics (OpenCV
         world-to-camera in DA3's world), intrinsics (at the processed size), image_files, process_res
  sam3d  MV-SAM3D fusion of all views (envs/sam3d, MV-SAM3D first on sys.path) with texture baking
         -> canonical.glb, splat.ply, pose.json (canonical -> view 0's camera, PyTorch3D axes)
  place  fits the mesh to the masks of all views: first its scale (MV-SAM3D's own scale was 11% too
         large on its example), then shift, size and a small rotation (its position was 17 cm off on
         the round table); --fit scale or --fit none limit this. Then -> posed.glb in frame W (view 0's OpenCV camera, DA3
         units), cameras.json (every view's K at the image size and w2c relative to view 0, for the
         agent's manifest), placement.json (the fit, mask IoU per view, view 0 depth check) and
         overlay_<name>.png

    python -m gator.stage_a.mvsam3d prep --scene DIR --mask-prompt NAME --out RUN [--max-side 1024]
    envs/da3/bin/python -m gator.stage_a.mvsam3d da3 --da3-root Depth-Anything-3 --out RUN
    python -m gator.stage_a.mvsam3d sam3d --mv-root MV-SAM3D --sam3d-root sam-3d-objects --out RUN
    python -m gator.stage_a.mvsam3d place --out RUN [--fit pose|scale|none]

DA3's world frame is its chosen reference view, not view 0, so cameras are always taken relative to
view 0. View 0 is the first name in natural order, as in MV-SAM3D's loader.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

from gator.stage_a.place import P3D_TO_CV

DA3_MODEL = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"


def natural(names):
    return sorted(names, key=lambda n: (0, int(n)) if re.fullmatch(r"\d+", n) else (1, n))


def view_names(out):
    return natural(p.stem for p in (Path(out) / "input" / "images").glob("*.png"))


def scene_images(scene):
    """{name: path} of the photos (PNG or JPG) in scene/images, in natural order. Hidden files, such as
    macOS's ._ copies, are skipped."""
    images = {p.stem: p for p in (Path(scene) / "images").iterdir()
              if p.suffix.lower() in (".png", ".jpg", ".jpeg") and p.is_file() and not p.name.startswith(".")}
    return {n: images[n] for n in natural(images)}


def prep(scene, mask_prompt, out, max_side):
    """Copy views that have both an image and a mask, all at one size.

    Depth Anything 3 center-crops a batch of mixed sizes to the smallest one, which cameras() does
    not model. So every view is center-cropped to the median aspect ratio and resized to the median
    size, with the long side at most max_side. Views that already share a size are only resized.
    """
    from PIL import Image

    from gator.stage_a.sam3d import load_mask

    scene, out = Path(scene), Path(out)
    for d in ("input/images", "input/object", "masks"):
        (out / d).mkdir(parents=True, exist_ok=True)
    images = scene_images(scene)
    pairs = []
    for name, path in images.items():
        cands = [scene / mask_prompt / f"{name}{s}" for s in ("_mask.png", ".png")]
        mask_path = next((c for c in cands if c.exists()), None)
        if mask_path is not None:
            pairs.append((name, Image.open(path).convert("RGB"), mask_path))
    if len(pairs) < 2:
        raise SystemExit(f"{len(pairs)} views with a mask found; MV-SAM3D needs at least 2")
    aspect = float(np.median([img.width / img.height for _, img, _ in pairs]))
    boxes = []
    for _, img, _ in pairs:
        w, h = img.size
        cw, ch = (round(h * aspect), h) if w / h > aspect else (w, round(w / aspect))
        boxes.append(((w - cw) // 2, (h - ch) // 2, (w - cw) // 2 + cw, (h - ch) // 2 + ch))
    cw = float(np.median([bx[2] - bx[0] for bx in boxes]))
    k = min(1.0, max_side / max(cw, cw / aspect))
    size = (round(cw * k), round(cw * k / aspect))
    kept = []
    for (name, img, mask_path), box in zip(pairs, boxes):
        m = load_mask(mask_path)
        if m.shape != (img.height, img.width):
            raise SystemExit(f"mask {mask_path} is {m.shape[1]}x{m.shape[0]}, image {name} is {img.width}x{img.height}")
        crop = img.crop(box).resize(size, Image.LANCZOS)
        mask = Image.fromarray(m.astype(np.uint8) * 255).crop(box).resize(size, Image.NEAREST)
        crop.save(out / "input" / "images" / f"{name}.png")
        rgba = Image.new("RGBA", size, (255, 255, 255, 0))
        rgba.putalpha(mask)
        rgba.save(out / "input" / "object" / f"{name}.png")
        mask.save(out / "masks" / f"{name}.png")
        kept.append({"name": name, "size": list(size), "source": str(images[name]), "source_size": list(img.size),
                     "crop": list(box), "mask_px": int(np.array(mask).astype(bool).sum())})
    (out / "prep.json").write_text(json.dumps({"scene": str(scene), "mask_prompt": mask_prompt, "max_side": max_side,
                                               "aspect": aspect, "views": kept}, indent=1))
    print(f"{len(kept)} views at {size[0]}x{size[1]}: {', '.join(v['name'] for v in kept)}")


def da3(da3_root, out, model, process_res):
    """Depth Anything 3 on all views, saved in the layout of MV-SAM3D's scripts/run_da3.py."""
    import torch

    sys.path.insert(0, str(Path(da3_root) / "src"))
    from depth_anything_3.api import DepthAnything3

    from PIL import Image

    out = Path(out)
    names = view_names(out)
    paths = [str(out / "input" / "images" / f"{n}.png") for n in names]
    sizes = {Image.open(p).size for p in paths}
    if len(sizes) > 1:  # DA3 would center-crop them, which cameras() does not model; prep makes one size
        raise SystemExit(f"views have different sizes {sorted(sizes)}; rerun prep")
    t0 = time.time()
    m = DepthAnything3.from_pretrained(model).to("cuda").eval()
    with torch.no_grad():
        pred = m.inference(image=paths, process_res=process_res)

    def npf(x):
        x = x.detach().float().cpu().numpy() if hasattr(x, "detach") else x
        return None if x is None else np.asarray(x, dtype=np.float32)

    depth, K, E, conf = npf(pred.depth), npf(pred.intrinsics), npf(pred.extrinsics), npf(getattr(pred, "conf", None))
    n, h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    pts = np.stack([(u - K[:, 0, 2, None, None]) * depth / K[:, 0, 0, None, None],
                    (v - K[:, 1, 2, None, None]) * depth / K[:, 1, 1, None, None], depth], -1)
    np.savez(out / "da3_output.npz", depth=depth, conf=conf if conf is not None else np.ones_like(depth), pointmaps=pts,
             pointmaps_sam3d=pts.transpose(0, 3, 1, 2).copy(), extrinsics=E, intrinsics=K,
             image_files=np.array(paths), process_res=process_res)
    print(f"DA3 {model}: {n} views at {w}x{h} in {time.time() - t0:.1f} s, "
          f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


def sam3d(mv_root, sam3d_root, out, seed):
    """MV-SAM3D multi-view fusion with texture baking, called the way run_inference_weighted.py does."""
    out, mv_root = Path(out), Path(mv_root).resolve()
    os.environ.setdefault("CONDA_PREFIX", sys.prefix)  # notebook/inference.py reads it at import
    sys.path[:0] = [str(mv_root), str(mv_root / "notebook")]  # MV-SAM3D's sam3d_objects before the editable one
    import torch
    from inference import Inference
    from load_images_and_masks import load_images_and_masks_from_path
    from sam3d_objects.model.backbone.tdfy_dit.utils import render_utils
    from sam3d_objects.utils.latent_weighting import WeightingConfig

    import sam3d_objects
    assert Path(sam3d_objects.__file__).resolve().is_relative_to(mv_root), sam3d_objects.__file__
    t0 = time.time()
    inf = Inference(str(Path(sam3d_root) / "checkpoints" / "hf" / "pipeline.yaml"), compile=False)
    pipe = inf._pipeline
    # As in sam3d.py: bake a texture with nvdiffrast, and render the Gaussians for the bake with gsplat,
    # because the default inria backend is not installed.
    pipe.rendering_engine = "nvdiffrast"
    render_frames = render_utils.render_frames
    render_utils.render_frames = lambda s, e, k, options={}, **kw: render_frames(
        s, e, k, {"backend": "gsplat", **options}, **kw)

    images, masks, names = load_images_and_masks_from_path(out / "input", mask_prompt="object")
    da = np.load(out / "da3_output.npz")
    stem = {Path(str(f)).stem: i for i, f in enumerate(da["image_files"])}
    if names != view_names(out) or any(n not in stem for n in names):
        raise SystemExit(f"views {names} do not match prep and DA3 ({sorted(stem)})")
    pointmaps = [da["pointmaps_sam3d"][stem[n]] for n in names]
    # The settings of run_inference_weighted.py's default (entropy weighting in both stages).
    weighting = WeightingConfig(use_entropy=True, entropy_alpha=30.0, attention_layer=6, attention_step=0,
                                min_weight=0.001, weight_source="entropy")
    res = pipe.run_multi_view(view_images=images, view_masks=masks, view_pointmaps=pointmaps, seed=seed,
                              mode="multidiffusion", stage1_inference_steps=50, stage2_inference_steps=25,
                              decode_formats=["gaussian", "mesh"], with_mesh_postprocess=True,
                              with_texture_baking=True, use_vertex_color=False, weighting_config=weighting,
                              ss_weighting=True, ss_entropy_layer=9, ss_entropy_alpha=30.0, ss_warmup_steps=1)
    if res.get("glb") is None:
        raise SystemExit("MV-SAM3D returned no mesh")
    res["glb"].export(out / "canonical.glb")
    res["gs"].save_ply(str(out / "splat.ply"))

    def arr(x):
        return np.asarray(x.detach().float().cpu() if hasattr(x, "detach") else x).reshape(-1).tolist()

    pose = {"rotation_wxyz": arr(res["rotation"]), "translation": arr(res["translation"]), "scale": arr(res["scale"]),
            "frame": "canonical -> view 0 camera, PyTorch3D axes, row vectors (note 2), DA3 units",
            "views": names, "view0": names[0], "seed": seed, "glb_extents": np.asarray(res["glb"].extents).tolist(),
            "glb_faces": int(len(res["glb"].faces)), "seconds": round(time.time() - t0, 1),
            "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
    (out / "pose.json").write_text(json.dumps(pose, indent=1))
    print(json.dumps(pose, indent=1))


def cameras(out):
    """Every view's K at its image size and OpenCV world-to-camera relative to view 0's camera."""
    from PIL import Image

    out = Path(out)
    da = np.load(out / "da3_output.npz")
    stem = {Path(str(f)).stem: i for i, f in enumerate(da["image_files"])}
    names = view_names(out)
    h, w = da["depth"].shape[1:]
    E = {n: np.vstack([da["extrinsics"][stem[n]], [0.0, 0.0, 0.0, 1.0]]) for n in names}
    E0_inv = np.linalg.inv(E[names[0]])
    cams = []
    for n in names:
        W, H = Image.open(out / "input" / "images" / f"{n}.png").size
        K = da["intrinsics"][stem[n]].astype(float).copy()
        sx, sy = W / w, H / h  # DA3 resized the whole image (no crop when all views share a size)
        K[0, 0], K[0, 2] = K[0, 0] * sx, (K[0, 2] + 0.5) * sx - 0.5
        K[1, 1], K[1, 2] = K[1, 1] * sy, (K[1, 2] + 0.5) * sy - 0.5
        cams.append({"name": n, "K": K.tolist(), "w2c": (E[n] @ E0_inv).tolist(), "width": W, "height": H,
                     "image": str(out / "input" / "images" / f"{n}.png"), "mask": str(out / "masks" / f"{n}.png")})
    return cams


def view_ious(x_w, faces, cams, masks, size=512):
    """Mask IoU of the mesh's silhouette in every view, drawn with the long side at `size` px."""
    from gator.stage_a.place import resize_mask, silhouette

    ious = []
    for c, m in zip(cams, masks):
        W, H = c["width"], c["height"]
        small = (round(W * size / max(W, H)), round(H * size / max(W, H)))
        T = np.array(c["w2c"])
        sil = silhouette(x_w @ T[:3, :3].T + T[:3, 3], faces, np.array(c["K"]), (W, H), small)
        m = resize_mask(m, small)
        ious.append(float((sil & m).sum() / max((sil | m).sum(), 1)))
    return np.array(ious)


def fit_scale(x_w, faces, cams, masks, lo=0.7, hi=1.3, size=384, limits=(0.3, 3.0)):
    """Scale about the mesh's box center that maximizes the mean mask IoU over all views (note 4b).

    MV-SAM3D's predicted scale can be off (11% too large on its own example) while the cameras and
    the depth are right, so only the size is fitted; the rotation and the center stay. The search
    is coarse to fine: steps of 0.05 over [lo, hi], grown by 0.05 steps while the best value sits at
    an edge (up to `limits`), then steps of 0.01 and 0.0025 around the best value. Ties go to the
    scale closest to 1. silhouette() fills boundary pixels, so the fit runs about 0.5% small.
    Returns the scale, the center, the per-view IoUs at scale 1 and at the scale, and whether the
    scale stopped at a limit.
    """
    c = (x_w.min(0) + x_w.max(0)) / 2
    before = view_ious(x_w, faces, cams, masks, size)
    seen = {1.0: before}

    def score(k):
        k = round(float(min(max(k, limits[0]), limits[1])), 4)
        if k not in seen:
            seen[k] = view_ious(c + k * (x_w - c), faces, cams, masks, size)
        return seen[k].mean(), -abs(k - 1), k

    best = max(score(k) for k in np.arange(lo, hi + 1e-9, 0.05))[2]
    while best <= lo and lo > limits[0]:
        lo = round(lo - 0.05, 4)
        best = max(score(lo), score(best))[2]
    while best >= hi and hi < limits[1]:
        hi = round(hi + 0.05, 4)
        best = max(score(hi), score(best))[2]
    for step, span in ((0.01, 0.05), (0.0025, 0.01)):
        best = max([score(k) for k in np.arange(best - span, best + span + 1e-9, step)] + [score(best)])[2]
    return best, c, before, seen[best], bool(best <= limits[0] or best >= limits[1])


def scaled_pose(pose, k, center_w):
    """The pose whose canonical_glb_to_w gives center_w + k * (x_w - center_w): with
    x_w = ((v * s) @ R + t) * P3D_TO_CV and P3D_TO_CV its own inverse, the scale becomes k * s and
    the translation k * t + (1 - k) * center_w * P3D_TO_CV."""
    s = np.asarray(pose["scale"], dtype=float).reshape(-1)
    t = np.asarray(pose["translation"], dtype=float).reshape(3)
    return {"rotation_wxyz": list(pose["rotation_wxyz"]), "scale": (k * s).tolist(),
            "translation": (k * t + (1 - k) * np.asarray(center_w) * P3D_TO_CV).tolist()}


def rotation_from_vector(rv_deg):
    """Rotation matrix (column vectors) for an axis-angle vector in degrees."""
    th = np.radians(np.linalg.norm(rv_deg))
    if th < 1e-12:
        return np.eye(3)
    u = np.radians(np.asarray(rv_deg, dtype=float)) / th
    U = np.array([[0.0, -u[2], u[1]], [u[2], 0.0, -u[0]], [-u[1], u[0], 0.0]])
    return np.eye(3) + np.sin(th) * U + (1 - np.cos(th)) * U @ U


def matrix_to_quat_wxyz(R):
    """Inverse of place.quat_wxyz_to_matrix for a proper rotation (Shepperd's method)."""
    t = np.trace(R)
    if t > 0:
        q = np.sqrt(t + 1) * 2
        w, x, y, z = q / 4, (R[2, 1] - R[1, 2]) / q, (R[0, 2] - R[2, 0]) / q, (R[1, 0] - R[0, 1]) / q
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        q = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / q, q / 4, (R[0, 1] + R[1, 0]) / q, (R[0, 2] + R[2, 0]) / q
    elif R[1, 1] > R[2, 2]:
        q = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / q, (R[0, 1] + R[1, 0]) / q, q / 4, (R[1, 2] + R[2, 1]) / q
    else:
        q = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / q, (R[0, 2] + R[2, 0]) / q, (R[1, 2] + R[2, 1]) / q, q / 4
    v = np.array([w, x, y, z])
    return (v / np.linalg.norm(v) * (1 if w >= 0 else -1)).tolist()


def similar_pose(pose, k, R_w, shift_w, center_w):
    """The pose whose canonical_glb_to_w gives center_w + shift_w + k * R_w (x_w - center_w).

    With x_w = ((v * s) @ Rp + t) * P (row vectors, P = P3D_TO_CV, its own inverse, D = diag(P)):
    the scale becomes k * s, the rotation Rp @ D @ R_w.T @ D, and the translation
    (k * (t * P) @ R_w.T + center_w + shift_w - k * center_w @ R_w.T) * P.
    """
    from gator.stage_a.place import quat_wxyz_to_matrix

    s = np.asarray(pose["scale"], dtype=float).reshape(-1)
    t = np.asarray(pose["translation"], dtype=float).reshape(3)
    c, D = np.asarray(center_w, dtype=float), np.diag(P3D_TO_CV)
    Rp = quat_wxyz_to_matrix(pose["rotation_wxyz"]) @ D @ np.asarray(R_w).T @ D
    t_new = (k * (t * P3D_TO_CV) @ np.asarray(R_w).T + c + np.asarray(shift_w) - k * c @ np.asarray(R_w).T) * P3D_TO_CV
    return {"rotation_wxyz": matrix_to_quat_wxyz(Rp), "scale": (k * s).tolist(), "translation": t_new.tolist()}


def fit_pose(x_w, faces, cams, masks, size=384, max_evals=400, shift_limit=0.25, rot_limit=20.0,
             scale_limits=(0.5, 2.0), rot_gain=0.005):
    """Second stage after fit_scale: shift, size and a small rotation about the box center that
    maximize the mean mask IoU over all views (note 4b). fit_scale alone cannot move the mesh, and
    MV-SAM3D's position can be off (the round table: 17 cm).

    Coordinate descent over log scale, the shift (in box diagonals) and an axis-angle rotation
    (degrees). Steps start at 0.05, 0.05 and 5 and halve when no single move helps, while they stay
    at or above 0.003, 0.003 and 0.3. One more move, tried first in each sweep, scales the mesh
    about view 0's camera, with the same steps in log scale as the scale. It keeps view 0's
    silhouette and changes size and distance together, and it is taken only within the bounds.
    With narrow views the IoU is nearly flat along it and drops to both sides, so moves of one
    coordinate cannot follow it: without it, a box 5% too large and 5% too far (views at 0 and +-10
    degrees) stopped 3.4% too large and 1.2 cm too far. Shift and rotation are coupled, so a descent over all seven coordinates at once can trade a
    missing shift for a false rotation. The search therefore runs twice: first with the rotation
    held at 0, then with it free from that result. The second run leaves out the camera move, which
    costs evaluations there and gained little in tests. The rotation is kept only when it raises
    the mean IoU by rot_gain. Bounds: shift_limit diagonals and rot_limit degrees per axis, scale
    within scale_limits of the start; at most max_evals evaluations in all. Returns (k, R_w,
    shift_w, rotation vector in degrees), the center, the per-view IoUs at the start and at the
    result, the number of evaluations, and whether the result sits on a bound.
    """
    c = (x_w.min(0) + x_w.max(0)) / 2
    diag = float(np.linalg.norm(np.ptp(x_w, 0)))
    seen = {}

    def ious(p):
        key = tuple(np.round(p, 6))
        if key not in seen:
            x = c + p[1:4] * diag + np.exp(p[0]) * (x_w - c) @ rotation_from_vector(p[4:]).T
            seen[key] = view_ious(x, faces, cams, masks, size)
        return seen[key]

    lo = np.array([np.log(scale_limits[0]), -shift_limit, -shift_limit, -shift_limit, -rot_limit, -rot_limit, -rot_limit])
    hi = -lo
    hi[0] = np.log(scale_limits[1])
    T0 = np.asarray(cams[0]["w2c"], dtype=float)
    o = (-T0[:3, :3].T @ T0[:3, 3] - c) / diag  # view 0's camera, as a shift in box diagonals

    def moved(p, i, h):
        """p moved by h along axis i, or for i = 7 scaled by exp(h) about view 0's camera: that
        multiplies the scale by exp(h) and gives the shift o + exp(h) * (shift - o). A camera move
        that leaves the bounds gives p: clipped, it would not keep view 0's silhouette. Its shift
        part is about h times the camera's distance in box diagonals, so a far object takes it at
        the smaller steps only."""
        q = p.copy()
        if i < 7:
            q[i] += h
        else:
            q[0], q[1:4] = q[0] + h, o + np.exp(h) * (q[1:4] - o)
            if (q < lo).any() or (q > hi).any():
                return p
        return np.clip(q, lo, hi)

    def descend(p, free):
        best = ious(p).mean()
        steps = np.array([0.05, 0.05, 0.05, 0.05, 5.0, 5.0, 5.0, 0.05]) * free
        floor = np.array([0.003, 0.003, 0.003, 0.003, 0.3, 0.3, 0.3, 0.003])
        while (steps >= floor).any() and len(seen) < max_evals:
            improved = False
            for i in [i for i in (7, 0, 1, 2, 3, 4, 5, 6) if steps[i] >= floor[i]]:
                for sign in (1.0, -1.0):
                    q = moved(p, i, sign * steps[i])
                    if (q == p).all() or len(seen) >= max_evals:
                        continue
                    v = ious(q).mean()
                    if v > best + 1e-6:
                        p, best, improved = q, v, True
                        break
            if not improved:
                steps = steps / 2
        return p

    p0 = np.zeros(7)
    p_shift = descend(p0, np.array([1, 1, 1, 1, 0, 0, 0, 1]))
    p_rot = descend(p_shift, np.array([1, 1, 1, 1, 1, 1, 1, 0]))
    p = p_rot if ious(p_rot).mean() - ious(p_shift).mean() >= rot_gain else p_shift
    at_bound = bool(np.any(np.isclose(p, lo) | np.isclose(p, hi)))
    return (float(np.exp(p[0])), rotation_from_vector(p[4:]), p[1:4] * diag, p[4:].copy()), c, ious(p0), ious(p), len(seen), at_bound


def place(out, samples=300000, fit="pose", min_gain=0.005):
    """fit: "pose" (scale, then shift, size and a small rotation), "scale" or "none" (True and False
    mean "pose" and "none")."""
    fit = {True: "pose", False: "none"}.get(fit, fit)
    if fit not in ("pose", "scale", "none"):
        raise ValueError(f"fit must be pose, scale or none, not {fit!r}")
    import trimesh
    from PIL import Image
    from scipy.ndimage import binary_erosion
    from scipy.spatial import cKDTree

    from gator import frames
    from gator.stage_a.place import canonical_glb_to_w, resize_mask, silhouette

    out = Path(out)
    pose_raw = json.loads((out / "pose.json").read_text())
    mesh = trimesh.load(out / "canonical.glb", force="mesh", process=False)
    cams = cameras(out)
    (out / "cameras.json").write_text(json.dumps(cams, indent=1))
    masks = [np.array(Image.open(c["mask"])) > 0 for c in cams]

    # Fit the size to the masks of all views; keep MV-SAM3D's pose when the fit gains too little.
    pose = {k: pose_raw[k] for k in ("rotation_wxyz", "translation", "scale")}
    x_w = canonical_glb_to_w(mesh.vertices, pose)
    scale_fit = {"applied": False, "scale": 1.0}
    if fit in ("scale", "pose"):
        k, center, before, after, at_limit = fit_scale(x_w, mesh.faces, cams, masks)
        applied = bool(after.mean() - before.mean() >= min_gain)
        scale_fit = {"applied": applied, "scale": k if applied else 1.0, "candidate_scale": k, "center_W": center.tolist(),
                     "mean_iou_before": round(float(before.mean()), 4), "candidate_mean_iou": round(float(after.mean()), 4),
                     "min_gain": min_gain, "at_limit": at_limit, "fit_px": 384,
                     "search": "0.05 steps over 0.7 to 1.3, grown while the best is at an edge (0.3 to 3.0), then 0.01 and 0.0025"}
        if before.max() == 0:
            scale_fit["warning"] = "the mesh overlaps no mask at scale 1"
        elif at_limit:
            scale_fit["warning"] = "the fitted scale stopped at a search limit"
        if applied:
            pose = scaled_pose(pose, k, center)
            x_w = canonical_glb_to_w(mesh.vertices, pose)
        print("scale fit:", json.dumps(scale_fit))
    pose_fit = {"applied": False}
    if fit == "pose":
        (k, R_w, shift, rvec), center, before, after, evals, at_bound = fit_pose(x_w, mesh.faces, cams, masks)
        applied = bool(after.mean() - before.mean() >= min_gain)
        cand = {"scale": round(k, 4), "shift_W": np.round(shift, 4).tolist(), "rotation_deg": np.round(rvec, 2).tolist()}
        pose_fit = {"applied": applied, **(cand if applied else {"scale": 1.0, "shift_W": [0.0] * 3, "rotation_deg": [0.0] * 3}),
                    **{"candidate_" + key: v for key, v in cand.items()}, "center_W": center.tolist(),
                    "mean_iou_before": round(float(before.mean()), 4), "candidate_mean_iou": round(float(after.mean()), 4),
                    "min_gain": min_gain, "evaluations": evals, "at_bound": at_bound, "fit_px": 384,
                    "search": "coordinate descent on log scale, shift (box diagonals) and rotation (degrees) about the box "
                              "center, first with the rotation held at 0 and a scale about view 0's camera tried first in "
                              "each sweep, then with the rotation free (kept if it gains 0.005); steps 0.05/0.05/5 halving "
                              "while at or above 0.003/0.003/0.3; bounds 0.5 to 2 x, 0.25 diag, 20 deg"}
        if at_bound:
            pose_fit["warning"] = "the fitted pose sits on a search bound"
        if applied:
            pose = similar_pose(pose, k, R_w, shift, center)
            x_w = canonical_glb_to_w(mesh.vertices, pose)
        print("pose fit:", json.dumps(pose_fit))

    trimesh.Trimesh(frames.w_to_gltf(x_w), mesh.faces, visual=mesh.visual, process=False).export(out / "posed.glb")
    diag = float(np.linalg.norm(np.ptp(x_w, 0)))
    ious = view_ious(x_w, mesh.faces, cams, masks)
    check = {"views": [{"name": c["name"], "mask_iou": round(float(v), 4)} for c, v in zip(cams, ious)]}
    for c, m in zip(cams, masks):
        W, H = c["width"], c["height"]
        x_c = x_w @ np.array(c["w2c"])[:3, :3].T + np.array(c["w2c"])[:3, 3]
        small = (round(W * 512 / max(W, H)), round(H * 512 / max(W, H)))
        sil = silhouette(x_c, mesh.faces, np.array(c["K"]), (W, H), small)
        m = resize_mask(m, small)
        img = np.asarray(Image.open(c["image"]).convert("RGB").resize(small)).astype(float)
        img[sil] = 0.55 * img[sil] + 0.45 * np.array([255.0, 0.0, 0.0])
        img[m & ~binary_erosion(m, iterations=2)] = [0, 255, 0]
        Image.fromarray(img.astype(np.uint8)).save(out / f"overlay_{c['name']}.png")
    # View 0: DA3's masked points must lie on the posed mesh (both are in view 0's camera).
    da = np.load(out / "da3_output.npz")
    stem = {Path(str(f)).stem: i for i, f in enumerate(da["image_files"])}
    P = da["pointmaps"][stem[cams[0]["name"]]]
    m0 = resize_mask(masks[0], (P.shape[1], P.shape[0]))
    P = P[m0 & np.isfinite(P).all(-1) & (P[..., 2] > 1e-6)]
    d, _ = cKDTree(trimesh.Trimesh(x_w, mesh.faces, process=False).sample(samples)).query(P)
    metric = float(da["metric_scale"]) if "metric_scale" in da.files else None  # select applied a LiDAR scale
    check.update(mean_mask_iou=round(float(ious.mean()), 4),
                 view0_depth_to_mesh_median_over_diag=float(np.median(d) / diag),
                 view0_depth_within_2pct_diag=float((d < 0.02 * diag).mean()))
    units = "meters (LiDAR scale applied in select)" if metric is not None else "DA3 units (meters)"
    placement = {"frame_W": f"OpenCV camera of view 0: x right, y down, z forward, in {units}",
                 "glb_axes": "posed.glb vertices are glTF: (x, y, z)_W = (x, -z, y)_glTF (Prompt 2)",
                 "view0": cams[0]["name"], "metric_scale": metric, "units": units,
                 "pose": pose, "pose_note": "pose gives posed.glb through canonical_glb_to_w; pose_mvsam3d is MV-SAM3D's own",
                 "pose_mvsam3d": {k: pose_raw[k] for k in ("rotation_wxyz", "translation", "scale")},
                 "scale_fit": scale_fit, "pose_fit": pose_fit,
                 "posed_extent_W": np.ptp(x_w, 0).tolist(), "posed_center_W": ((x_w.min(0) + x_w.max(0)) / 2).tolist(),
                 "check": check}
    (out / "placement.json").write_text(json.dumps(placement, indent=1))
    print(json.dumps(placement, indent=1))


def session_views(out):
    """The views for controller.prepare_session (cameras.json written by place)."""
    return [{k: c[k] for k in ("K", "w2c", "width", "height", "image", "mask")}
            for c in json.loads((Path(out) / "cameras.json").read_text())]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="step", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--scene", required=True, help="folder with images/ and a mask folder")
    p.add_argument("--mask-prompt", required=True, help="mask folder name; masks are <name>_mask.png or <name>.png")
    p.add_argument("--out", required=True)
    p.add_argument("--max-side", type=int, default=1024)
    d = sub.add_parser("da3")
    d.add_argument("--da3-root", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--model", default=DA3_MODEL)
    d.add_argument("--process-res", type=int, default=504)
    s = sub.add_parser("sam3d")
    s.add_argument("--mv-root", required=True)
    s.add_argument("--sam3d-root", required=True, help="sam-3d-objects clone with checkpoints/hf")
    s.add_argument("--out", required=True)
    s.add_argument("--seed", type=int, default=42)
    pl = sub.add_parser("place")
    pl.add_argument("--out", required=True)
    pl.add_argument("--fit", choices=("pose", "scale", "none"), default="pose",
                    help="pose: scale, then shift, size and a small rotation (default); scale: size only; none")
    pl.add_argument("--no-fit-scale", action="store_true", help="same as --fit none")
    a = ap.parse_args(argv)
    if a.step == "prep":
        prep(a.scene, a.mask_prompt, a.out, a.max_side)
    elif a.step == "da3":
        da3(a.da3_root, a.out, a.model, a.process_res)
    elif a.step == "sam3d":
        sam3d(a.mv_root, a.sam3d_root, a.out, a.seed)
    else:
        place(a.out, fit="none" if a.no_fit_scale else a.fit)


if __name__ == "__main__":
    main()
