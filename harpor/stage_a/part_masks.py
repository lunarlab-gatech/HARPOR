"""Part masks with SAM 3: every instance of every part of the proposal in every view (build_plan.md note 11).

Runs in envs/sam3seg (as sam3_masks), from the repo root:

    python -m harpor.stage_a.part_masks --run J/run --proposal J/run/proposal.json --out J/run/parts/sam3

The photos are the selected views in RUN/input/images (at most 12), not the 32 candidates. Each photo
is encoded once for all part prompts (sam3_masks.Segmenter). The prompt of a part is its name in the
proposal. The base is segmented too, first, when the list has another part: its masks show the vote
which faces do not move and which part masks hold the whole object. Writes into OUT:
  <slug>/<view>_<k>.png  instance k of the part in that view (L, 0/255 at the photo's size), numbered
                         by score, highest first
  part_masks.json        every instance (score, xyxy box, pixels, file) per part and view, the parts
                         with no instance in any view (not_found) and the weights' revision
  check/<view>.jpg       the photo with every instance tinted and outlined in its part's colour and
                         labelled "<slug> <k>: <score>"

Unlike sam3_masks there is no pick: the face vote (harpor.parts.vote) uses every instance, and a part
with no instance anywhere is listed in not_found, which is not an error. A rerun replaces OUT as a
whole, and only once every photo is done: the files go into a temporary folder that then takes OUT's
place.
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

from harpor.fileio import replace
from harpor.stage_a.sam3_masks import Segmenter, check_image, slug

COLORS = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240),
          (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (170, 110, 40), (255, 225, 25)]


def photos(images_dir):
    """{name: path} of the PNG and JPG photos in images_dir, in natural order (as mvsam3d.scene_images)."""
    from harpor.stage_a.mvsam3d import natural

    files = {p.stem: p for p in Path(images_dir).iterdir()
             if p.suffix.lower() in (".png", ".jpg", ".jpeg") and p.is_file() and not p.name.startswith(".")}
    return {n: files[n] for n in natural(files)}


def part_prompts(proposal):
    """The names of the proposal's parts to segment: the base, then the others; none when the base is alone."""
    others = [p["name"] for p in proposal["parts"] if p["name"] != proposal["base"]]
    return [proposal["base"], *others] if others else []


def overlay_parts(image, found, max_side=1600):
    """The photo with every instance tinted and outlined in its part's colour and labelled.
    found: [(label, colour, scores, boxes, masks)], one entry per part."""
    items = [(m, color, 0.35, f"{label} {i}: {sc:.2f}", b) for label, color, scores, boxes, masks in found
             for i, (sc, b, m) in enumerate(zip(scores, boxes, masks))]
    return check_image(image, items, max_side, 60, "no part instance")


def run(images_dir, parts, out, threshold=0.4, mask_threshold=0.5, segmenter=None):
    """Every instance of every part name in every photo of images_dir; see the module docstring."""
    from PIL import Image

    out = Path(out)
    images = photos(images_dir)
    if not images:
        raise SystemExit(f"no PNG or JPG photos in {images_dir}")
    slugs = [slug(p) for p in parts]
    if len(set(slugs)) < len(slugs):
        raise SystemExit(f"parts {parts} give the same folder twice")
    tmp = out.with_name(".tmp_" + out.name)
    shutil.rmtree(tmp, ignore_errors=True)
    for d in ("check", *slugs):
        (tmp / d).mkdir(parents=True)

    seg = (segmenter or Segmenter.load()) if parts else None
    t0 = time.time()
    records = {p: {"name": p, "slug": s, "instances": 0, "views": {}} for p, s in zip(parts, slugs)}
    sizes = {}
    for n, (name, path) in enumerate(images.items()):
        image = np.array(Image.open(path).convert("RGB"))  # writable, so torch.from_numpy does not warn
        sizes[name] = [image.shape[1], image.shape[0]]
        found = seg(image, parts, threshold, mask_threshold) if parts else {}
        shown = []
        for i, (p, s) in enumerate(zip(parts, slugs)):
            scores, boxes, masks = found[p]
            recs = []
            for k, (sc, b, m) in enumerate(zip(scores, boxes, masks)):
                f = f"{s}/{name}_{k}.png"
                Image.fromarray(np.where(m, 255, 0).astype(np.uint8)).save(tmp / f)
                recs.append({"index": k, "score": round(float(sc), 4), "box_xyxy": [round(float(v), 1) for v in b],
                             "mask_px": int(m.sum()), "file": f})
            records[p]["views"][name] = recs
            records[p]["instances"] += len(recs)
            shown.append((s, COLORS[i % len(COLORS)], scores, boxes, masks))
        overlay_parts(image, shown).save(tmp / "check" / f"{name}.jpg", quality=90)
        print(f"PROGRESS {(n + 1) / len(images):.2f}", flush=True)

    not_found = [p for p in parts if records[p]["instances"] == 0]
    info = seg.info if seg is not None else {"model": None}
    record = {**info, "threshold": threshold, "mask_threshold": mask_threshold, "seconds": round(time.time() - t0, 1),
              "peak_gib": seg.peak_gib() if seg is not None else None, "views": list(images), "size_wh": sizes,
              "parts": [records[p] for p in parts], "not_found": not_found}
    replace(tmp / "part_masks.json", Path.write_text, json.dumps(record, indent=1))
    shutil.rmtree(out, ignore_errors=True)
    os.replace(tmp, out)
    for p in parts:
        r = records[p]
        seen = sum(bool(v) for v in r["views"].values())
        print(f"{p!r}: {r['instances']} instances in {seen} of {len(images)} photos -> {out / r['slug']}")
    if not_found:
        print(f"not found in any photo: {not_found}")
    return record


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="the run folder, with input/images")
    ap.add_argument("--proposal", required=True, help="proposal.json of harpor.parts.stage propose")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.4, help="lowest instance score (detection x presence)")
    a = ap.parse_args(argv)
    proposal = json.loads(Path(a.proposal).read_text())
    run(Path(a.run) / "input" / "images", part_prompts(proposal), a.out, a.threshold)


if __name__ == "__main__":
    main()
