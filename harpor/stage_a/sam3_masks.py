"""Object masks from a text prompt with SAM 3 (build_plan.md note 3).

Runs in envs/sam3seg (transformers 5.20.0, weights facebook/sam3), from the repo root:

    python -m harpor.stage_a.sam3_masks --scene DIR --prompt NOUN [--pick img2=1 ...]

For every photo in DIR/images (PNG or JPG), the prompt writes into DIR/sam3_<slug>, where the slug
is the prompt in lower case with every run of other characters made one _ ("Coffee cup" ->
sam3_coffee_cup):
  <name>.png        the mask, L mode 0/255 at the photo's size. `mvsam3d prep --mask-prompt
                    sam3_<slug>` reads it as it is.
  sam3.json         every instance SAM 3 found in each photo (score, xyxy box, pixels), the pick,
                    warnings and the weights' revision
  check/<name>.jpg  the photo with every instance outlined and numbered; the mask in green, the
                    other instances in red

Instances are numbered by score, highest first, and the mask is instance 0. --pick NAME=K takes
instance K of photo NAME instead, --pick NAME=K,L the union of K and L (one object that SAM 3 split
in two), and --pick NAME=none no mask (the object is not in that photo). NAME=0 goes back to the top
score. A photo with no instance above the threshold gets no mask file, so prep skips it.

A rerun replaces an earlier run's files once every photo is done. Each file is written under a temporary
name and then renamed.

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

    def __call__(self, image, prompt, threshold=0.5, mask_threshold=0.5):
        """Photo (H, W, 3) uint8 -> (scores (N,), xyxy boxes in px (N, 4), masks (N, H, W) bool) of the prompt,
        sorted by score, highest first.

        With text alone, transformers skips the geometry encoder and so drops the CLS token that
        Meta's code always appends (transformers issue #49592). One padding box (label -10) runs the
        encoder and appends the CLS token while the box itself stays masked out.
        """
        import torch

        H, W = image.shape[:2]
        with torch.inference_mode():
            pixels = self.processor(images=image, return_tensors="pt")["pixel_values"].to(self.device)
            vision = self.model.get_vision_features(pixel_values=pixels)
            text = self.processor(text=prompt, return_tensors="pt").to(self.device)
            boxes = {"input_boxes": torch.tensor([[[0.5, 0.5, 1.0, 1.0]]], device=self.device),
                     "input_boxes_labels": torch.full((1, 1), -10, dtype=torch.long, device=self.device)}
            o = self.model(vision_embeds=vision, input_ids=text["input_ids"],
                           attention_mask=text["attention_mask"], **boxes)
            r = self.processor.post_process_instance_segmentation(
                o, threshold=threshold, mask_threshold=mask_threshold, target_sizes=[(H, W)])[0]
        scores = r["scores"].float().cpu().numpy()
        order = np.argsort(-scores, kind="stable")
        masks = r["masks"].bool().cpu().numpy() if len(scores) else np.zeros((0, H, W), bool)
        return scores[order], r["boxes"].float().cpu().numpy()[order].reshape(-1, 4), masks[order]

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


def run(scene, prompt, picks=None, threshold=0.5, mask_threshold=0.5, segmenter=None):
    """Masks of the prompt in every photo of scene/images; see the module docstring. Every photo is
    segmented and every pick checked before an earlier run's files are replaced."""
    from PIL import Image

    from harpor.stage_a.mvsam3d import scene_images

    scene, picks = Path(scene), picks or {}
    images = scene_images(scene)
    if not images:
        raise SystemExit(f"no PNG or JPG photos in {scene / 'images'}")
    if set(picks) - set(images):
        raise SystemExit(f"--pick names {sorted(set(picks) - set(images))}, which are not in {scene / 'images'}")
    folder = scene / f"sam3_{slug(prompt)}"
    rules = {n: k for n, k in picks.items() if k != [0]}  # NAME=0 is the top score, not a pick

    seg = segmenter or Segmenter.load()
    t0 = time.time()
    views, masks_out, checks = [], {}, {}
    for name, path in images.items():
        image = np.array(Image.open(path).convert("RGB"))  # writable, so torch.from_numpy does not warn
        scores, boxes, masks = seg(image, prompt, threshold, mask_threshold)
        idx, warnings = rules.get(name), []
        if idx is not None and any(k >= len(scores) for k in idx):
            raise SystemExit(f"--pick {name}={','.join(map(str, idx))}: SAM 3 found {len(scores)} instances")
        rule, idx = ("user", idx) if idx is not None else ("top_score", [0] if len(scores) else [])
        rec = {"name": name, "image": str(path), "size_wh": [image.shape[1], image.shape[0]],
               "instances": [{"index": i, "score": round(float(s), 4), "box_xyxy": [round(float(v), 1) for v in b],
                              "mask_px": int(m.sum())} for i, (s, b, m) in enumerate(zip(scores, boxes, masks))],
               "picked": idx, "pick": rule, "mask": None, "mask_px": 0, "warnings": warnings}
        if idx:
            masks_out[name] = masks[idx].any(0)
            rec.update(mask=f"{name}.png", mask_px=int(masks_out[name].sum()))
        elif rule == "user":
            warnings.append(f"left out by --pick {name}=none")
        else:
            warnings.append(f"no instance above score {threshold}; no mask written")
        checks[name] = overlay(image, scores, boxes, masks, idx)
        views.append(rec)

    seconds = round(time.time() - t0, 1)
    (folder / "check").mkdir(parents=True, exist_ok=True)
    for name in images:
        if name in masks_out:
            replace(folder / f"{name}.png", Image.fromarray(np.where(masks_out[name], 255, 0).astype(np.uint8)).save)
        replace(folder / "check" / f"{name}.jpg", checks[name].save, quality=90)
    record = {**seg.info, "prompt": prompt, "threshold": threshold, "mask_threshold": mask_threshold,
              "seconds": seconds, "peak_gib": seg.peak_gib(), "views": views}
    replace(folder / "sam3.json", Path.write_text, json.dumps(record, indent=1))
    for name in images:  # an old mask of a photo that has none now, so prep cannot use it
        if name not in masks_out and (folder / f"{name}.png").exists():
            (folder / f"{name}.png").unlink()
    n = len(masks_out)
    top = [v["instances"][0]["score"] for v in views if v["instances"]]
    print(f"{prompt!r}: {n} of {len(images)} photos masked -> {folder}"
          + (f" (top scores {min(top):.2f} to {max(top):.2f})" if top else ""))
    for v in views:
        if v["pick"] == "user":
            print(f"  {v['name']}: picked {','.join(map(str, v['picked'])) or 'none'}")
        for w in v["warnings"]:
            print(f"  {v['name']}: {w}")
    if n == 0:
        raise SystemExit(f"no mask for {prompt!r}; check the prompt")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True, help="folder with images/")
    ap.add_argument("--prompt", required=True, help="a short noun phrase")
    ap.add_argument("--pick", action="append", default=[], help="NAME=K, NAME=K,L or NAME=none")
    a = ap.parse_args(argv)
    run(a.scene, a.prompt, parse_picks(a.pick))


if __name__ == "__main__":
    main()
