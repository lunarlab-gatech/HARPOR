"""Face-id maps of a mesh under the selected cameras, for the face vote (build_plan.md note 11).

face_ids draws every face of the mesh with its index (painter's algorithm: faces sorted by mean depth,
the farthest first, each drawn with ImageDraw.polygon), so every pixel holds the nearest face that
covers it, or -1. Cameras are OpenCV K and world-to-camera matrices, as everywhere (frames.py), and
the map is drawn at most 768 px on the long side. A face counts as visible in a view when it holds at
least one pixel there; faces smaller than a pixel may hold none.
"""

import numpy as np

from harpor import frames

MAX_SIDE = 768


def scaled(K, image_wh, max_side=MAX_SIDE):
    """K and size (w, h) of the photo scaled so that its long side is at most max_side px."""
    W, H = image_wh
    k = min(1.0, max_side / max(W, H))
    size = (max(1, round(W * k)), max(1, round(H * k)))
    return frames.crop_intrinsics(np.asarray(K, dtype=float), (0, 0, W, H), size), size


def face_ids(vertices_w, faces, K, w2c, size, near=1e-6):
    """(h, w) int map of the face seen at each pixel (its index in faces), -1 where no face is.

    Faces with a vertex behind the camera, faces outside the image and projections far larger than
    the image are skipped. Ties in depth keep the order of faces.
    """
    from PIL import Image, ImageDraw

    w, h = size
    faces = np.asarray(faces)
    uv, z = frames.project(np.asarray(vertices_w, dtype=float), np.asarray(K, dtype=float),
                           np.asarray(w2c, dtype=float))
    tri, tz = uv[faces], z[faces]
    with np.errstate(invalid="ignore"):
        lo, hi = tri.min(1), tri.max(1)
        ok = (tz > near).all(1) & np.isfinite(tri).all((1, 2))
        ok &= (hi[:, 0] >= -0.5) & (lo[:, 0] <= w - 0.5) & (hi[:, 1] >= -0.5) & (lo[:, 1] <= h - 0.5)
        ok &= (hi - lo).max(1) < 8 * max(w, h)
    idx = np.flatnonzero(ok)
    order = idx[np.argsort(-tz[idx].mean(1), kind="stable")]
    img = Image.new("I", (w, h), 0)
    draw = ImageDraw.Draw(img)
    for i, p in zip(order.tolist(), tri[order].reshape(-1, 6).tolist()):
        draw.polygon(p, fill=i + 1)
    return np.asarray(img, dtype=np.int64) - 1


def visible_pixels(ids, n_faces):
    """(n_faces,) pixels that each face holds in the map."""
    return np.bincount(ids[ids >= 0], minlength=n_faces)


def inside_pixels(ids, mask, n_faces):
    """(n_faces,) pixels that each face holds inside the mask (same size as the map)."""
    return np.bincount(ids[(ids >= 0) & np.asarray(mask, dtype=bool)], minlength=n_faces)
