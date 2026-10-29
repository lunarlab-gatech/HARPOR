"""How well an agent session's delivered asset matches the input photos (build_plan.md note 5).

    python -m harpor.eval.photo_match ROOT [ROOT ...] [--dino] [--json OUT]

ROOT is a session folder (controller.prepare_session) after the agent ran. Nothing is rendered: the
scores reuse the review renders of the session, ROOT/work/reviews/<receipt>/viewNN_original.png and
viewNN_candidate.png, which review.py rendered under the input cameras, cropped to the windows in
review.json. The review is the one whose receipt ends in the first 12 hex digits of result.json's
sha256, so its candidate is the delivered GLB; else the latest review, with a warning. The photo and
its mask are cropped and resized as review.py does, so the IoU here equals the review's.

Per input view, for the original and for the candidate render:
  iou       photo mask vs render alpha > 0.5 (higher is better)
  color_de  CIE76 delta E between the mean Lab colour of the photo in its mask and that of the
            render in its alpha: the overall colour (lower is better)
  pix_de    mean per-pixel delta E where mask and alpha overlap (lower is better)
  edge      Pearson correlation of the Sobel gradient magnitudes of the photo and the render in
            grayscale, where mask or alpha holds. The render is composited on mid gray, and the photo
            keeps its own background. Both get a light Gaussian blur first, so resampling steps and
            render noise count less. Measures fine structure such as thin bars and openings (higher is
            better); 0 when one image has no gradient there
  bg_de     mean per-pixel delta E between the photo and the render composited over the photo's own
            background, where mask or alpha holds. The background under the mask is filled in from
            the nearest pixels outside it and blurred. So a correct opening in the render shows
            background as the photo does, and missing geometry shows background where the photo
            shows the object (lower is better). Approximate: the fill is a guess
  bg_edge   as edge, but for that composite against the photo (higher is better)
  dino      only with --dino: cosine of the DINOv2 ViT-S/14 CLS features of the photo and the render,
            both on mid gray (higher is better). The model loads only from the torch hub cache
            (TORCH_HOME), and nothing is downloaded; without the cache, dino is skipped with a note
Per session: the means over the views, and candidate minus original (the mean over the views where
both are valid). A view's value is NaN in the arrays and null in the JSON when it has no pixels to
measure, for example a render that is empty in that view.

The photos and the renders are lit differently, so the absolute colour numbers mean less than the
deltas and the comparison of agents under the same renderer. A clay review (opaque gray) makes the
colour scores meaningless; the session then gets a warning.
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ASSETS = ("original", "candidate")
METRICS = ("iou", "color_de", "pix_de", "edge", "bg_de", "bg_edge", "dino")
RECEIPT_DIR = re.compile(r"r(\d{3,})-([0-9a-f]{12})")  # tools.review: r<NNN>-<first 12 hex of the GLB's sha256>
GRAY = 0.5  # the background that the renders (and, for dino, the photos) are composited on
EDGE_SIGMA = 1.0  # px, the Gaussian blur before the Sobel filter
FILL_GROW, FILL_SIGMA = 3, 8.0  # px: the mask is grown before the background fill, which is then blurred
LUMA = np.array([0.299, 0.587, 0.114])  # ITU-R 601, as PIL's "L"
# sRGB (D65) to CIE XYZ, and the D65 white point.
SRGB_TO_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                        [0.2126729, 0.7151522, 0.0721750],
                        [0.0193339, 0.1191920, 0.9503041]])
WHITE_D65 = np.array([0.95047, 1.0, 1.08883])
DINO_MODEL, DINO_REPO, DINO_SIDE = "dinov2_vits14", "facebookresearch_dinov2_main", 224
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def fit(a, h, w):
    """Nearest-pixel resize to h x w, exactly as review.py's fit (which cannot be imported outside Blender)."""
    return a[np.arange(h) * a.shape[0] // h][:, np.arange(w) * a.shape[1] // w]


def srgb_to_lab(rgb):
    """CIE L*a*b* (D65) of sRGB values in [0, 1], shape (..., 3)."""
    c = np.asarray(rgb, np.float64)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    t = lin @ SRGB_TO_XYZ.T / WHITE_D65
    f = np.where(t > (6 / 29) ** 3, np.cbrt(t), t / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def gray(rgb):
    return np.asarray(rgb, np.float64) @ LUMA


def on_gray(rgba):
    """RGB of an RGBA render (straight alpha) composited on mid gray."""
    a = rgba[..., 3:]
    return rgba[..., :3] * a + GRAY * (1 - a)


def gradient(g, sigma=EDGE_SIGMA):
    """Sobel gradient magnitude of a grayscale image, after a Gaussian blur of sigma px."""
    if sigma:
        g = ndimage.gaussian_filter(g, sigma)
    return np.hypot(ndimage.sobel(g, 0), ndimage.sobel(g, 1))


def pearson(a, b):
    """Pearson correlation of two samples; 0 when one of them is constant."""
    a, b = a - a.mean(), b - b.mean()
    d = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum() / d) if d > 0 else 0.0


def read_rgb(path):
    """Photo as float RGB in [0, 1], read as stored (no EXIF rotation, like every other step)."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), np.float64) / 255


def read_rgba(path):
    """Review render as float RGBA in [0, 1], straight alpha (the PNG as Blender saved it)."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGBA"), np.float64) / 255


def read_mask(path):
    """Mask as review.py reads it: the first channel above 0.5."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGBA"))[..., 0] > 127


def photo_crop(work, view, window, h, w):
    """The photo and mask of a manifest view, cropped to the review window [u0, v0, u1, v1] and resized
    to the render's h x w as review.py does: first to the manifest's size, then the crop, then fit."""
    u0, v0, u1, v1 = window
    photo = fit(read_rgb(work / view["image"]), view["height"], view["width"])
    mask = fit(read_mask(work / view["mask"]), view["height"], view["width"])
    return fit(photo[v0:v1, u0:u1], h, w), fit(mask[v0:v1, u0:u1], h, w)


def background(photo, mask):
    """The photo with the (grown) mask filled in from the nearest pixels outside it, then blurred there:
    a guess of what lies behind the object."""
    hole = ndimage.binary_dilation(mask, iterations=FILL_GROW)
    if hole.all() or not hole.any():
        return photo.copy()
    idx = ndimage.distance_transform_edt(hole, return_distances=False, return_indices=True)
    fill = photo[idx[0], idx[1]]
    blur = np.stack([ndimage.gaussian_filter(fill[..., c], FILL_SIGMA) for c in range(3)], -1)
    return np.where(hole[..., None], blur, photo)


def view_scores(photo, mask, render, bg=None):
    """iou, color_de, pix_de and edge of one render (h, w, 4) against the photo (h, w, 3) and its mask."""
    sil = render[..., 3] > 0.5
    union, both = sil | mask, sil & mask
    nan = float("nan")
    out = {"iou": float(both.sum() / union.sum()) if union.any() else nan}
    lab_p, lab_r = srgb_to_lab(photo), srgb_to_lab(render[..., :3])
    out["color_de"] = (float(np.linalg.norm(lab_p[mask].mean(0) - lab_r[sil].mean(0)))
                       if mask.any() and sil.any() else nan)
    out["pix_de"] = float(np.linalg.norm(lab_p[both] - lab_r[both], axis=1).mean()) if both.any() else nan
    out["edge"] = (pearson(gradient(gray(photo))[union], gradient(gray(on_gray(render)))[union])
                   if union.any() else nan)
    bg = background(photo, mask) if bg is None else bg
    a = render[..., 3:]
    comp = render[..., :3] * a + bg * (1 - a)
    out["bg_de"] = (float(np.linalg.norm(lab_p[union] - srgb_to_lab(comp)[union], axis=1).mean())
                    if union.any() else nan)
    out["bg_edge"] = pearson(gradient(gray(photo))[union], gradient(gray(comp))[union]) if union.any() else nan
    return out


def dino_embedder(model_name=DINO_MODEL):
    """(embed, note): embed(list of (h, w, 3) RGB arrays) -> (n, d) unit CLS features, or None when
    torch or the cached model is missing. Loads only from torch.hub.get_dir() (TORCH_HOME): the repo
    folder facebookresearch_dinov2_main and checkpoints/<model>_pretrain.pth. Nothing is downloaded."""
    try:
        import torch
    except ImportError as e:
        return None, f"dino skipped: torch is not available ({e})"
    hub = Path(torch.hub.get_dir())
    repo, ckpt = hub / DINO_REPO, hub / "checkpoints" / f"{model_name}_pretrain.pth"
    missing = [str(p) for p in (repo, ckpt) if not p.exists()]
    if missing:
        return None, f"dino skipped: no {' or '.join(missing)} (set TORCH_HOME to a torch hub cache that has DINOv2)"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load(str(repo), model_name, source="local", pretrained=False)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model = model.to(device).eval()
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

    def embed(images):
        batch = []
        for im in images:
            h, w = im.shape[:2]
            s = max(h, w)
            sq = np.full((s, s, 3), GRAY)  # pad to a square on the same gray, so the aspect stays
            sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = im
            x = torch.from_numpy(sq).permute(2, 0, 1)[None].float()
            batch.append(torch.nn.functional.interpolate(x, size=(DINO_SIDE, DINO_SIDE), mode="bilinear",
                                                         antialias=True, align_corners=False))
        x = (torch.cat(batch) - mean) / std
        with torch.inference_mode():
            f = model(x.to(device))  # the normalized CLS token
        return torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy()

    return embed, f"dino: {model_name} from {repo}"


def pick_review(work, sha256):
    """(review folder, warnings): the latest review whose receipt matches sha256's first 12 hex digits,
    else the latest review. Exits when the session has no review."""
    found = []
    for d in (work / "reviews").glob("r*-*"):
        m = RECEIPT_DIR.fullmatch(d.name)
        if m and (d / "review.json").is_file():
            found.append((int(m[1]), m[2], d))
    if not found:
        sys.exit(f"no review with a review.json in {work / 'reviews'}")
    found.sort()
    match = [d for _, sha12, d in found if sha256 and sha12 == sha256[:12]]
    if match:
        return match[-1], []
    why = "result.json has no sha256" if not sha256 else f"no review of the delivered GLB (sha256 {sha256[:12]})"
    return found[-1][2], [f"{why}; scored the latest review, {found[-1][2].name}"]


def _mean(values):
    v = np.asarray(values, np.float64)
    ok = np.isfinite(v)
    return float(v[ok].mean()) if ok.any() else float("nan")


def score_session(root, embed=None):
    """Every score of one session; see the module docstring. embed: from dino_embedder, or None."""
    root = Path(root)
    work = root / "work"
    result_file = root / "result.json"
    result = json.loads(result_file.read_text()) if result_file.is_file() else {}
    warnings = [] if result else [f"no {result_file}; verdict unknown"]
    folder, picked = pick_review(work, result.get("sha256"))
    warnings += picked
    review = json.loads((folder / "review.json").read_text())
    if review.get("clay"):
        warnings.append(f"{folder.name} is a clay review (opaque gray), so the colour scores are meaningless")
    manifest = {f"view{v['index']:02d}": v for v in json.loads((work / "input" / "manifest.json").read_text())["views"]}
    metrics = METRICS if embed else METRICS[:-1]
    views = []
    for rec in review["views"]:
        if "crop" not in rec:  # orbit and detail views have no photo
            continue
        v = manifest.get(rec["name"])
        if v is None or not v.get("image") or not v.get("mask"):
            warnings.append(f"{rec['name']}: no photo or no mask in the manifest; skipped")
            continue
        names = [n for n in ASSETS if rec.get(n)]
        renders = {n: read_rgba(folder / rec[n]) for n in names}
        h, w = renders[names[0]].shape[:2]
        photo, mask = photo_crop(work, v, rec["crop"], h, w)
        out = {"name": rec["name"], "crop": rec["crop"], "size_wh": [w, h],
               "review_iou": {n: rec.get(f"iou_{n}") for n in names}}
        bg = background(photo, mask)
        out.update({n: view_scores(photo, mask, renders[n], bg) for n in names})
        if embed:
            f = embed([np.where(mask[..., None], photo, GRAY)] + [on_gray(renders[n]) for n in names])
            for n, fn in zip(names, f[1:]):
                out[n]["dino"] = float(f[0] @ fn)
        if len(names) == 2:
            out["delta"] = {k: out["candidate"][k] - out["original"][k] for k in metrics}
        views.append(out)
    if not views:
        sys.exit(f"{root}: review {folder.name} has no input view with a photo and a mask")
    names = [n for n in ASSETS if all(n in v for v in views)]
    name = root.parent.name if root.name == "s" else root.name  # controller sessions live in <run>/s
    s = {"name": name, "root": str(root), "verdict": result.get("verdict"),
         "agent": result.get("agent") or result.get("cli"), "model": result.get("model"),
         "glb": result.get("glb"), "sha256": result.get("sha256"), "review": folder.name,
         "clay": bool(review.get("clay")), "warnings": warnings, "n_views": len(views), "views": views,
         "mean": {n: {k: _mean([v[n][k] for v in views]) for k in metrics} for n in names}}
    if len(names) == 2:
        s["delta"] = {k: _mean([v["delta"][k] for v in views]) for k in metrics}
    return s


def _finite(x):
    """NaN and infinity as None, so the JSON is valid."""
    if isinstance(x, dict):
        return {k: _finite(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_finite(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def _num(x, digits, signed=False):
    if x is None or not math.isfinite(x):
        return "-"
    return f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"


def table(sessions, dino=False):
    """One row per session: the candidate's means, each followed by candidate minus original."""
    metrics = METRICS if dino else METRICS[:-1]
    digits = {"iou": 3, "color_de": 1, "pix_de": 1, "edge": 3, "bg_de": 1, "bg_edge": 3, "dino": 3}
    width = max([7] + [len(s["name"]) for s in sessions])
    head = f"{'session':<{width}}  {'verdict':<12} {'views':>5}  {'review':<6}" + "".join(
        f"  {k:>15}" for k in metrics)
    rows = [head]
    for s in sessions:
        cand = s["mean"].get("candidate", {})
        delta = s.get("delta", {})
        cells = [f"{_num(cand.get(k), digits[k]):>7} {_num(delta.get(k), digits[k], True):>7}" for k in metrics]
        rows.append(f"{s['name']:<{width}}  {str(s['verdict']):<12} {s['n_views']:>5}  {s['review'].split('-')[0]:<6}"
                    + "".join(f"  {c:>15}" for c in cells))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", metavar="ROOT", help="a session folder (with result.json and work/)")
    ap.add_argument("--dino", action="store_true", help="add the DINOv2 score (from the torch hub cache only)")
    ap.add_argument("--json", type=Path, metavar="OUT", help="write every number to this JSON file")
    a = ap.parse_args(argv)
    embed, note = dino_embedder() if a.dino else (None, None)
    sessions = [score_session(r, embed) for r in a.roots]
    print("each metric: the candidate's mean over views, then candidate minus original; "
          "iou, edge, bg_edge, dino higher is better; color_de, pix_de, bg_de lower is better")
    for line in table(sessions, dino=embed is not None):
        print(line)
    if note:
        print(note)
    for s in sessions:
        for w in s["warnings"]:
            print(f"{s['name']}: {w}")
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(_finite({"dino": note, "sessions": sessions}), indent=1))
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
