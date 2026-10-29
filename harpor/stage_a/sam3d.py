"""Stage A, single view: SAM 3D Objects on one photo and its mask (build_plan.md note 4).

Runs inside the sam3d micromamba env, from the gator_clone root:

    python -m harpor.stage_a.sam3d --sam3d-root <sam-3d-objects clone> \
        --image view0.png --mask view0_mask.png --out <dir>

Writes to <dir>:
  canonical.glb  SAM 3D canonical unit box (y up) with a baked 1024 px texture
  splat.ply      the Gaussians, canonical frame
  pose.json      rotation (w, x, y, z), translation and scale from the canonical frame to the
                 photo's camera in PyTorch3D axes (x left, y up, z forward), plus run facts
  pointmap.npy   MoGe pointmap of the photo (H', W', 3), PyTorch3D camera axes, same units as
                 the pose; the posed mesh must land on it inside the mask (note 2)
"""

import argparse
import json
import os
import sys
import time

import numpy as np


def load_mask(path):
    """Object pixels of a mask PNG. When the alpha channel varies, it is the mask (alpha > 0, as in
    SAM 3D's notebook loader and MV-SAM3D's RGBA masks). Otherwise the mask is drawn in the color
    channels, as in an opaque white-on-black export (the SAM 3 web demo's): brightest channel above
    127, or above 0 for a 0/1 mask. Anti-aliased edges count when at least half covered."""
    from PIL import Image

    a = np.array(Image.open(path))
    if a.ndim == 3 and a.shape[2] in (2, 4) and a[..., -1].min() < a[..., -1].max():
        return a[..., -1] > 0
    v = (a[..., :-1] if a.shape[2] in (2, 4) else a).max(-1) if a.ndim == 3 else a
    return v > (127 if v.max() > 1 else 0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sam3d-root", required=True, help="sam-3d-objects clone with checkpoints/hf")
    ap.add_argument("--image", required=True)
    ap.add_argument("--mask", required=True, help="PNG at the photo's size, read by load_mask")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)

    from PIL import Image

    sys.path.insert(0, os.path.join(a.sam3d_root, "notebook"))
    from inference import Inference  # reads CONDA_PREFIX at import
    from sam3d_objects.model.backbone.tdfy_dit.utils import render_utils

    t0 = time.time()
    inf = Inference(os.path.join(a.sam3d_root, "checkpoints", "hf", "pipeline.yaml"), compile=False)
    # notebook/inference.py forces pytorch3d and turns texture baking off, so its GLB has vertex
    # colours only (issues #75, #146). Bake a texture with nvdiffrast instead.
    inf._pipeline.rendering_engine = "nvdiffrast"
    # The bake renders the Gaussians with backend "inria" (diff_gaussian_rasterization), which
    # SAM 3D does not install. Use its gsplat backend.
    render_frames = render_utils.render_frames
    render_utils.render_frames = lambda s, e, k, options={}, **kw: render_frames(
        s, e, k, {"backend": "gsplat", **options}, **kw)

    image = np.array(Image.open(a.image).convert("RGB"))
    mask = load_mask(a.mask)
    if mask.shape != image.shape[:2] or not mask.any():
        raise SystemExit(f"mask {mask.shape} with {int(mask.sum())} px does not fit image {image.shape[:2]}")
    out = inf._pipeline.run(inf.merge_mask_to_rgba(image, mask), None, a.seed, with_mesh_postprocess=True,
                            with_texture_baking=True, use_vertex_color=False)
    if out.get("glb") is None:
        raise SystemExit("SAM 3D returned no mesh")

    os.makedirs(a.out, exist_ok=True)
    out["glb"].export(os.path.join(a.out, "canonical.glb"))
    out["gs"].save_ply(os.path.join(a.out, "splat.ply"))
    np.save(os.path.join(a.out, "pointmap.npy"), np.asarray(out["pointmap"], dtype=np.float32))

    def arr(v):
        return np.asarray(v.detach().float().cpu() if hasattr(v, "detach") else v).reshape(-1).tolist()

    pose = {
        "rotation_wxyz": arr(out["rotation"]),
        "translation": arr(out["translation"]),
        "scale": arr(out["scale"]),
        "frame": "canonical -> photo camera, PyTorch3D axes, row vectors (note 2)",
        "image": os.path.abspath(a.image),
        "mask": os.path.abspath(a.mask),
        "image_size_wh": [image.shape[1], image.shape[0]],
        "mask_px": int(mask.sum()),
        "seed": a.seed,
        "glb_extents": np.asarray(out["glb"].extents).tolist(),
        "glb_faces": int(len(out["glb"].faces)),
        "seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(a.out, "pose.json"), "w") as f:
        json.dump(pose, f, indent=1)
    print(json.dumps(pose, indent=1))


if __name__ == "__main__":
    main()
