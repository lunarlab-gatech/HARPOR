"""Pose and mask helpers for Stage A (build_plan.md note 2).

canonical_glb_to_w moves a SAM 3D canonical GLB into frame W (OpenCV camera axes) with the pose
that SAM 3D returns. harpor.stage_a.mvsam3d uses these helpers to place its mesh, and silhouette and
resize_mask also serve the mask code.
"""

import numpy as np

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

