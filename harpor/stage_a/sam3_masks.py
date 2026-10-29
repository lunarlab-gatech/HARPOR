"""Object masks from a text prompt with SAM 3 (build_plan.md note 3).

Runs in envs/sam3seg (transformers 5.20.0, weights facebook/sam3), from the gator_clone root:

    python -m harpor.stage_a.sam3_masks --scene DIR --prompt chair [--prompt "coffee cup" ...]
    python -m harpor.stage_a.sam3_masks --scene DIR --prompt chair --pick img2=1 --overwrite

For every photo in DIR/images (PNG or JPG), each prompt writes into DIR/sam3_<slug>, where the slug
is the prompt in lower case with every run of other characters made one _ ("Coffee cup" ->
sam3_coffee_cup):
  <name>.png        the mask, L mode 0/255 at the photo's size. `mvsam3d prep --mask-prompt
                    sam3_<slug>` and `sam3d.py --mask` read it as it is.
  sam3.json         every instance SAM 3 found in each photo (score, xyxy box, pixels), the pick,
                    warnings and the weights' revision
  check/<name>.jpg  the photo with every instance outlined and numbered; the mask in green, the
                    other instances in red

Instances are numbered by score, highest first, and the mask is instance 0. --pick NAME=K takes
instance K of photo NAME instead, --pick NAME=K,L the union of K and L (one object that SAM 3 split
in two), and --pick NAME=none no mask (the object is not in that photo). NAME=0 goes back to the top
score. A rerun keeps the earlier picks of the photos that its --pick does not name; an earlier pick
that names an instance SAM 3 no longer finds falls back to the top score, with a warning. A photo
with no instance above --threshold gets no mask file, so prep skips it.

A rerun replaces an earlier run's files only with --overwrite, and only once every photo is done.
Each file is written under a temporary name and then renamed. A hand-made <name>_mask.png in the
folder stays, and prep prefers it.

Photos are read as stored, without the EXIF rotation, like every other step.
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from harpor.fileio import replace

MODEL = "facebook/sam3"


class Segmenter:
    """SAM 3's image model (transformers Sam3Model). Each photo is encoded once for all prompts."""

    def __init__(self, model, processor, device):
        import torch
        import transformers

        self.model, self.processor, self.device = model, processor, device
        self.info = {"transformers": transformers.__version__, "torch": torch.__version__, "device": device}

    @classmethod
    def load(cls, model_id=MODEL):
        import torch
        from transformers import Sam3Model, Sam3Processor
        from transformers.utils import cached_file

        # Not AutoModel: facebook/sam3's config is the video model's, so AutoModel gives Sam3VideoModel.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        t0 = time.time()
        model = Sam3Model.from_pretrained(model_id).to(device).eval()
        seg = cls(model, Sam3Processor.from_pretrained(model_id), device)
        seg.info.update(model=model_id, revision=Path(cached_file(model_id, "config.json")).parent.name,
                        load_seconds=round(time.time() - t0, 1))
        return seg

    def __call__(self, image, prompts, threshold=0.5, mask_threshold=0.5, cls_token=True):
        """Photo (H, W, 3) uint8 -> {prompt: (scores (N,), xyxy boxes in px (N, 4), masks (N, H, W) bool)},
        sorted by score, highest first.

        With text alone, transformers skips the geometry encoder and so drops the CLS token that
        Meta's code always appends (transformers issue #49592). cls_token adds one padding box
        (label -10), which runs the encoder and appends the CLS token while the box itself stays
        masked out.
        """
        import torch

        H, W = image.shape[:2]
        out = {}
        with torch.inference_mode():
            pixels = self.processor(images=image, return_tensors="pt")["pixel_values"].to(self.device)
            vision = self.model.get_vision_features(pixel_values=pixels)
            for prompt in prompts:
                text = self.processor(text=prompt, return_tensors="pt").to(self.device)
                boxes = {}
                if cls_token:
                    boxes = {"input_boxes": torch.tensor([[[0.5, 0.5, 1.0, 1.0]]], device=self.device),
                             "input_boxes_labels": torch.full((1, 1), -10, dtype=torch.long, device=self.device)}
                o = self.model(vision_embeds=vision, input_ids=text["input_ids"],
                               attention_mask=text["attention_mask"], **boxes)
                r = self.processor.post_process_instance_segmentation(
                    o, threshold=threshold, mask_threshold=mask_threshold, target_sizes=[(H, W)])[0]
                scores = r["scores"].float().cpu().numpy()
                order = np.argsort(-scores, kind="stable")
                masks = r["masks"].bool().cpu().numpy() if len(scores) else np.zeros((0, H, W), bool)
                out[prompt] = (scores[order], r["boxes"].float().cpu().numpy()[order].reshape(-1, 4), masks[order])
        return out

    def peak_gib(self):
        import torch

        return round(torch.cuda.max_memory_allocated() / 2**30, 1) if self.device == "cuda" else None


def slug(prompt):
    s = re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")
    if not s:
        raise SystemExit(f"prompt {prompt!r} has no letters or digits")
    return s


def parse_picks(items):
    """['img2=1', 'img3=0,2', 'img4=none'] -> {'img2': [1], 'img3': [0, 2], 'img4': []}."""
    picks = {}
    for item in items:
        m = re.fullmatch(r"(.+)=(none|\d+(?:,\d+)*)", item)
        if m is None:
            raise SystemExit(f"--pick {item!r} is not NAME=K, NAME=K,L or NAME=none")
        picks[m[1]] = [] if m[2] == "none" else sorted({int(k) for k in m[2].split(",")})
    return picks


def outputs(folder, names):
    """The files an earlier run wrote into folder for these photos. A hand-made <name>_mask.png stays."""
    files = [folder / "sam3.json"] + [f for n in names for f in (folder / f"{n}.png", folder / "check" / f"{n}.jpg")]
    return [f for f in files if f.exists()]


def earlier_picks(folder):
    """{name: indices} of the photos whose mask an earlier run took from --pick."""
    f = folder / "sam3.json"
    views = json.loads(f.read_text())["views"] if f.exists() else []
    return {v["name"]: v["picked"] for v in views if v["pick"] == "user"}


def check_image(image, items, max_side, font_div, empty):
    """The photo scaled to a long side of at most max_side, with items drawn in order: each (mask, colour, tint,
    text, box) is tinted by tint (0 or None: not tinted), outlined in its colour and labelled with text above its
    box; then the labels. empty is written when there are no items."""
    from PIL import Image, ImageDraw, ImageFont
    from scipy.ndimage import binary_erosion

    from harpor.stage_a.place import resize_mask

    H, W = image.shape[:2]
    k = min(1.0, max_side / max(H, W))
    size = (round(W * k), round(H * k))
    ov = np.asarray(Image.fromarray(image).resize(size, Image.LANCZOS)).astype(float)
    for mask, color, tint, _, _ in items:
        m = resize_mask(mask, size)
        if tint:
            ov[m] = (1 - tint) * ov[m] + tint * np.array(color)
        ov[m & ~binary_erosion(m, iterations=2)] = color
    out = Image.fromarray(ov.astype(np.uint8))
    draw = ImageDraw.Draw(out)
    px = max(12, size[0] // font_div)
    font = ImageFont.load_default(size=px)
    for _, color, _, text, box in items:
        draw.text((box[0] * k, max(0.0, box[1] * k - px)), text, fill=color, font=font, stroke_width=2,
                  stroke_fill=(0, 0, 0))
    if not items:
        draw.text((px, px), empty, fill=(255, 40, 40), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    return out


def overlay(image, scores, boxes, masks, picked, max_side=1600):
    """The photo with every instance outlined and numbered. The picked ones, which form the mask, are
    tinted green and drawn last; the others are outlined in red."""
    green, red = (0, 255, 0), (255, 40, 40)
    order = [i for i in range(len(masks)) if i not in picked] + list(picked)
    return check_image(image, [(masks[i], green if i in picked else red, 0.4 if i in picked else None,
                                f"{i}: {scores[i]:.2f}", boxes[i]) for i in order], max_side, 50, "no instance")


def run(scene, prompts, picks=None, threshold=0.5, mask_threshold=0.5, overwrite=False, segmenter=None):
    """Masks of every prompt in every photo of scene/images; see the module docstring. Every photo is
    segmented and every pick checked before an earlier run's files are replaced."""
    from PIL import Image

    from harpor.stage_a.mvsam3d import scene_images

    scene, picks = Path(scene), picks or {}
    images = scene_images(scene)
    if not images:
        raise SystemExit(f"no PNG or JPG photos in {scene / 'images'}")
    if len({slug(p) for p in prompts}) < len(prompts):
        raise SystemExit(f"prompts {prompts} give the same folder twice")
    if picks and len(prompts) > 1:
        raise SystemExit("--pick works with one --prompt; rerun that prompt alone with --overwrite")
    if set(picks) - set(images):
        raise SystemExit(f"--pick names {sorted(set(picks) - set(images))}, which are not in {scene / 'images'}")
    folders = {p: scene / f"sam3_{slug(p)}" for p in prompts}
    for d in folders.values():
        if outputs(d, images) and not overwrite:
            raise SystemExit(f"{d} has masks from an earlier run; pass --overwrite to replace them")
    # A photo keeps its earlier --pick unless this --pick names it again; NAME=0 is the top score, not a pick.
    rules = {p: {n: k for n, k in {**earlier_picks(d), **picks}.items() if n in images and k != [0]}
             for p, d in folders.items()}

    seg = segmenter or Segmenter.load()
    t0 = time.time()
    views, masks_out, checks = {p: [] for p in prompts}, {}, {}
    for name, path in images.items():
        image = np.array(Image.open(path).convert("RGB"))  # writable, so torch.from_numpy does not warn
        found = seg(image, prompts, threshold, mask_threshold)
        picked = {}
        for p in prompts:
            scores, boxes, masks = found[p]
            idx, warnings = rules[p].get(name), []
            if idx is not None and any(k >= len(scores) for k in idx):
                pick = f"{name}={','.join(map(str, idx))}"
                if name in picks:
                    raise SystemExit(f"--pick {pick}: SAM 3 found {len(scores)} instances")
                warnings.append(f"the earlier pick {pick} is gone ({len(scores)} instances now); took the top score")
                idx = None
            rule, idx = ("user", idx) if idx is not None else ("top_score", [0] if len(scores) else [])
            rec = {"name": name, "image": str(path), "size_wh": [image.shape[1], image.shape[0]],
                   "instances": [{"index": i, "score": round(float(s), 4), "box_xyxy": [round(float(v), 1) for v in b],
                                  "mask_px": int(m.sum())} for i, (s, b, m) in enumerate(zip(scores, boxes, masks))],
                   "picked": idx, "pick": rule, "mask": None, "mask_px": 0, "warnings": warnings}
            if idx:
                picked[p] = masks[idx].any(0)
                rec.update(mask=f"{name}.png", mask_px=int(picked[p].sum()))
            elif rule == "user":
                warnings.append(f"left out by --pick {name}=none")
            else:
                warnings.append(f"no instance above score {threshold}; no mask written")
            checks[p, name] = overlay(image, scores, boxes, masks, idx)
            views[p].append(rec)
        masks_out.update({(p, name): m for p, m in picked.items()})
        # Two prompts on one object (say "cup" and "glass") would give two meshes of it.
        ps = list(picked)
        for i, a in enumerate(ps):
            for b in ps[i + 1:]:
                iou = (picked[a] & picked[b]).sum() / max((picked[a] | picked[b]).sum(), 1)
                if iou > 0.5:
                    for p, q in ((a, b), (b, a)):
                        views[p][-1]["warnings"].append(f"mask IoU {iou:.2f} with the mask of prompt {q!r}")

    seconds = round(time.time() - t0, 1)
    empty = []
    for p, d in folders.items():
        (d / "check").mkdir(parents=True, exist_ok=True)
        for name in images:
            if (p, name) in masks_out:
                replace(d / f"{name}.png", Image.fromarray(np.where(masks_out[p, name], 255, 0).astype(np.uint8)).save)
            replace(d / "check" / f"{name}.jpg", checks[p, name].save, quality=90)
        record = {**seg.info, "prompt": p, "threshold": threshold, "mask_threshold": mask_threshold,
                  "seconds": seconds, "peak_gib": seg.peak_gib(), "views": views[p]}
        replace(d / "sam3.json", Path.write_text, json.dumps(record, indent=1))
        for name in images:  # an old mask of a photo that has none now, so prep cannot use it
            if (p, name) not in masks_out and (d / f"{name}.png").exists():
                (d / f"{name}.png").unlink()
        n = sum(v["mask"] is not None for v in views[p])
        top = [v["instances"][0]["score"] for v in views[p] if v["instances"]]
        print(f"{p!r}: {n} of {len(images)} photos masked -> {d}"
              + (f" (top scores {min(top):.2f} to {max(top):.2f})" if top else ""))
        for v in views[p]:
            if v["pick"] == "user":
                print(f"  {v['name']}: picked {','.join(map(str, v['picked'])) or 'none'}")
            for w in v["warnings"]:
                print(f"  {v['name']}: {w}")
        if n == 0:
            empty.append(p)
    if empty:
        raise SystemExit(f"no mask for {empty}; check the prompt or lower --threshold")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True, help="folder with images/")
    ap.add_argument("--prompt", required=True, action="append", help="a short noun phrase; repeat for more objects")
    ap.add_argument("--pick", action="append", default=[], help="NAME=K, NAME=K,L or NAME=none (one --prompt only)")
    ap.add_argument("--threshold", type=float, default=0.5, help="lowest instance score (detection x presence)")
    ap.add_argument("--overwrite", action="store_true", help="replace the outputs of an earlier run")
    a = ap.parse_args(argv)
    run(a.scene, a.prompt, parse_picks(a.pick), a.threshold, overwrite=a.overwrite)


if __name__ == "__main__":
    main()
