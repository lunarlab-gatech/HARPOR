"""Stage A placement: put the SAM 3D mesh into the photo's camera frame (build_plan.md note 2).

Reads a run folder written by gator.stage_a.sam3d and writes into it:
  posed.glb        the textured mesh in frame W (OpenCV camera of the photo), stored with the
                   Prompt 2 glTF axes: (x, y, z)_W = (x, -z, y)_glTF
  placement.json   pose, the camera K fitted to SAM 3D's pointmap, and the alignment check
  overlay.png      the posed mesh's silhouette (red) over the photo, with the mask outline (green)

    python -m gator.stage_a.place --run <dir>

The check needs no outside data: the posed mesh must lie on SAM 3D's own pointmap inside the mask,
and its silhouette under the fitted K must match the mask.
"""

import argparse
import json
import os

import numpy as np

from gator import frames

# to_glb turns SAM 3D's z-up canonical frame to glTF y-up with V = v @ GLB_TURN; undo with v = V @ GLB_TURN.T.
GLB_TURN = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
P3D_TO_CV = np.array([-1.0, -1.0, 1.0])  # PyTorch3D camera (x left, y up) -> OpenCV (x right, y down)


def quat_wxyz_to_matrix(q):
    """Same matrix as pytorch3d.transforms.quaternion_to_matrix (column-vector convention)."""
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def canonical_glb_to_w(V, pose):
    """GLB vertices (SAM 3D canonical, y up) -> frame W.

    SAM 3D applies Transform3d().scale(s).rotate(R).translate(t) to row vectors, so a point maps
    to (v * s) @ R + t in PyTorch3D camera axes (research.md section 2.5).
    """
    v = np.asarray(V, dtype=float) @ GLB_TURN.T
    R = quat_wxyz_to_matrix(pose["rotation_wxyz"])
    s = np.asarray(pose["scale"], dtype=float).reshape(-1)
    s = s if s.size == 3 else np.repeat(s[:1], 3)
    t = np.asarray(pose["translation"], dtype=float).reshape(3)
    return ((v * s) @ R + t) * P3D_TO_CV


def fit_intrinsics(pointmap_cv, image_wh):
    """Fit fx, fy, cx, cy (full-image pixels) to the pointmap, whose grid is the photo downsampled."""
    h, w = pointmap_cv.shape[:2]
    W, H = image_wh
    ii, jj = np.mgrid[0:h, 0:w]
    u = (jj + 0.5) * (W / w) - 0.5
    v = (ii + 0.5) * (H / h) - 0.5
    P = pointmap_cv.reshape(-1, 3)
    ok = np.isfinite(P).all(1) & (P[:, 2] > 1e-6)
    x, y = P[ok, 0] / P[ok, 2], P[ok, 1] / P[ok, 2]
    fx, cx = np.polyfit(x, u.reshape(-1)[ok], 1)
    fy, cy = np.polyfit(y, v.reshape(-1)[ok], 1)
    res = np.hypot(fx * x + cx - u.reshape(-1)[ok], fy * y + cy - v.reshape(-1)[ok])
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    return K, float(np.sqrt(np.mean(res ** 2)))


def silhouette(vertices_w, faces, K, image_wh, out_wh):
    """Filled silhouette of the mesh under K, drawn at out_wh (a downscaled copy of the photo)."""
    from PIL import Image, ImageDraw

    W, H = image_wh
    sx, sy = out_wh[0] / W, out_wh[1] / H
    z = vertices_w[:, 2]
    uv = vertices_w @ K.T
    uv = uv[:, :2] / np.maximum(z, 1e-9)[:, None]
    uv = np.stack([(uv[:, 0] + 0.5) * sx - 0.5, (uv[:, 1] + 0.5) * sy - 0.5], 1)
    img = Image.new("L", out_wh, 0)
    draw = ImageDraw.Draw(img)
    for f in faces[(z[faces] > 1e-6).all(1)]:
        draw.polygon([tuple(p) for p in uv[f]], fill=255)
    return np.asarray(img) > 0


def resize_mask(mask, out_wh):
    from PIL import Image

    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(out_wh, Image.NEAREST)) > 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="folder with canonical.glb, pose.json, pointmap.npy")
    ap.add_argument("--surface-samples", type=int, default=300000)
    a = ap.parse_args(argv)

    import trimesh
    from PIL import Image
    from scipy.spatial import cKDTree

    from gator.stage_a.sam3d import load_mask

    pose = json.load(open(os.path.join(a.run, "pose.json")))
    mesh = trimesh.load(os.path.join(a.run, "canonical.glb"), force="mesh", process=False)
    image = Image.open(pose["image"]).convert("RGB")
    W, H = image.size
    mask = load_mask(pose["mask"])
    pm = np.load(os.path.join(a.run, "pointmap.npy")) * P3D_TO_CV
    h, w = pm.shape[:2]

    # 1. Pose the mesh and write the GLB in the Prompt 2 axes. Every step is a proper rotation or
    #    a positive scale, so face winding is kept.
    x_w = canonical_glb_to_w(mesh.vertices, pose)
    posed = trimesh.Trimesh(frames.w_to_gltf(x_w), mesh.faces, visual=mesh.visual, process=False)
    posed.export(os.path.join(a.run, "posed.glb"))

    # 2. Camera: fit K to the pointmap (it shares SAM 3D's camera).
    K, k_rms = fit_intrinsics(pm, (W, H))

    # 3. Check: the masked pointmap must lie on the posed mesh, and the silhouette must match the mask.
    m_small = resize_mask(mask, (w, h))
    P = pm.reshape(-1, 3)
    vis = m_small.reshape(-1) & np.isfinite(P).all(1) & (P[:, 2] > 1e-6)
    surf = trimesh.Trimesh(x_w, mesh.faces, process=False).sample(a.surface_samples)
    d, _ = cKDTree(surf).query(P[vis])
    diag = float(np.linalg.norm(np.ptp(x_w, 0)))
    sil = silhouette(x_w, mesh.faces, K, (W, H), (w, h))
    iou = float((sil & m_small).sum() / max((sil | m_small).sum(), 1))
    check = {
        "mask_iou": round(iou, 4),
        "pointmap_to_mesh_median": float(np.median(d)),
        "pointmap_to_mesh_p90": float(np.percentile(d, 90)),
        "pointmap_to_mesh_median_over_diag": float(np.median(d) / diag),
        "pointmap_within_2pct_diag": float((d < 0.02 * diag).mean()),
        "masked_pointmap_points": int(vis.sum()),
        "K_fit_rms_px": round(k_rms, 3),
    }

    # 4. Overlay at pointmap resolution: mesh silhouette in red, mask outline in green.
    from scipy.ndimage import binary_erosion

    ov = np.asarray(image.resize((w, h))).astype(float)
    ov[sil] = 0.55 * ov[sil] + 0.45 * np.array([255.0, 0.0, 0.0])
    ov[m_small & ~binary_erosion(m_small, iterations=2)] = [0, 255, 0]
    Image.fromarray(ov.astype(np.uint8)).save(os.path.join(a.run, "overlay.png"))

    placement = {
        "frame_W": "OpenCV camera of the photo: x right, y down, z forward, in pointmap units",
        "glb_axes": "posed.glb vertices are glTF: (x, y, z)_W = (x, -z, y)_glTF (Prompt 2)",
        "units": "MoGe pointmap units, not guaranteed metric",
        "pose": {k: pose[k] for k in ("rotation_wxyz", "translation", "scale")},
        "K": K.tolist(),
        "image_size_wh": [W, H],
        "posed_extent_W": np.ptp(x_w, 0).tolist(),
        "posed_center_W": ((x_w.min(0) + x_w.max(0)) / 2).tolist(),
        "check": check,
    }
    with open(os.path.join(a.run, "placement.json"), "w") as f:
        json.dump(placement, f, indent=1)
    print(json.dumps(placement, indent=1))


if __name__ == "__main__":
    main()
