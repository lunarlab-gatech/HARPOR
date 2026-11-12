"""Render the original and the candidate with the same cameras and lights (build_plan.md note 5).

blender -b --factory-startup --python-exit-code 1 --python review.py -- ARGS.json

ARGS.json comes from tools.review: root, out, clay, views (from the manifest), candidate, original,
and detail (close-up cameras: center x y z, radius, view direction x y z, all in W). The controller
also uses this script for the original's previews, with candidate None.
The stock glTF importer maps glTF to W, so Blender world coordinates are W and the input cameras
apply as they are. Input views are cropped to the object (the mask's box and the projected assets)
and rendered at INPUT_LONG pixels on the long side. Writes one PNG per view and asset, sheet.png (one
row per view: photo, original, candidate, overlay) and review.json with the anchor evidence and the
crop windows. The render code is in render_common.py.
"""
# Untested: needs Blender 5.2, which the test environment does not have.

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_common import (box, detail_views, import_glb, input_camera, on_white, orbit_views, place_camera,  # noqa: E402
                           points, render, setup, write_sheet)

A = json.loads(Path(sys.argv[sys.argv.index("--") + 1]).read_text())
ROOT, OUT = Path(A["root"]), Path(A["out"])

scene, cam = setup(A["clay"])
assets = {n: import_glb(A[n]) for n in ("original", "candidate") if A.get(n)}
names = list(assets)
main = "candidate" if "candidate" in assets else "original"
pts = {n: points(assets[n]) for n in names}
allp = np.concatenate(list(pts.values()))
center, diag = (allp.min(0) + allp.max(0)) / 2, float(np.linalg.norm(allp.max(0) - allp.min(0)))
rows, records = [], []
views = ([{**v, "name": f"view{v['index']:02d}"} for v in A["views"]] + orbit_views(center, diag)
         + detail_views(A.get("detail") or []))
for v in views:
    rec, photo, mask = {"name": v["name"]}, None, None
    K, width, height = v["K"], v["width"], v["height"]
    if v.get("image"):
        K, width, height, photo, mask, crop = input_camera(v, ROOT, allp)
        rec.update(crop)
    place_camera(scene, cam, K, v["w2c"], width, height, (1e-3 * diag, 1e3 * diag))
    renders = {n: render(scene, assets, n, OUT / f"{v['name']}_{n}.png") for n in names}
    rec.update({n: f"{v['name']}_{n}.png" for n in names})
    row = [on_white(renders[n]) for n in names]
    if photo is not None:
        m = renders[main][..., 3:] * 0.5
        overlay = photo.copy()
        overlay[..., :3] = photo[..., :3] * (1 - m) + renders[main][..., :3] * m
        row = [photo] + row + [overlay]
        if mask is not None:
            for n in names:
                sil = renders[n][..., 3] > 0.5
                rec[f"iou_{n}"] = float((sil & mask).sum() / max((sil | mask).sum(), 1))
    rows.append((row, photo is not None))
    records.append(rec)
write_sheet(rows, OUT / "sheet.png")
review = {"clay": A["clay"], "views": records,
          "bbox": {n: box(p) for n, p in pts.items()}}
if len(names) == 2:
    o, c = review["bbox"]["original"], review["bbox"]["candidate"]
    review["center_shift"] = float(np.linalg.norm(np.subtract(c["center"], o["center"])) / np.linalg.norm(o["extent"]))
    review["extent_ratio"] = np.divide(c["extent"], o["extent"]).tolist()
(OUT / "review.json").write_text(json.dumps(review, indent=2))
