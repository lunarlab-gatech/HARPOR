"""Metric scale from LiDAR (build_plan.md note 12).

Depth Anything 3 Nested only estimates metric depth. A job with LiDAR compares LiDAR depth with DA3's
depth inside the object masks, and select multiplies DA3's depth, pointmaps and camera translations by
the scale s that it finds, so that MV-SAM3D, place, the agent and physics all work in meters.

Samples. Every pool frame ("f00012") gets an (N, 3) float array [u, v, z]: the LiDAR points that
fall on the frame, as pixels of the upright full-size source frame (scene/images/<name>.png) at
integer pixel centres, and z, the depth in meters along the camera's optical axis. project() turns
camera points into pixels; rot90_points() follows np.rot90 when the frames were turned upright.

The estimate (estimate()) runs on the pool after da3:
  view_ratio  per view: the samples go to DA3's grid through prep's crop (source_to_grid), a z-buffer
              drops points hidden behind nearer ones (more than 3% plus 9 cm behind, so that range
              noise passes), the pool mask is eroded by about 1% of the grid diagonal, DA3's depth must
              be valid and its conf at least the 30th percentile in the mask. r = z / depth; points
              beyond max(3 x 1.4826 MAD, ln 1.02) of the median ln r are trimmed, and the view's scale
              s_i is the median r of the rest (the used points). Status few_points (under 30 valid
              points), scattered (the IQR of the used points over 25% of s_i, or the trim drops more
              than 40% of the valid points, or more than 10% below the median r: hidden points are
              never nearer), no_mask, no_samples, else a candidate.
  rays        with estimate(rays=...) (a LiDAR beside the camera: each point's camera point and the LiDAR's
              origin), view_ratio runs again along the LiDAR's rays, from the s_i above (follow_rays). Each
              ray is traced through DA3's depth times s_i (ray_hits), which takes the z-buffer's place. A
              ray that meets the surface only at an edge comes from behind the object as the camera sees
              it, so its point is dropped. The mask, conf and depth are read where the ray meets the
              surface, not at the point's own pixel, which range noise moves along the epipolar line. r is
              the scale that would put that crossing at the measured range, and the trim is on r, not
              ln r. A point stays only when its twin, mirrored about the crossing, also lands in the image
              and on DA3's grid. The view is also scattered when over 10% of its mask's rays stay in front
              of the surface. Each pass starts from the s_i of the pass before, at most 3 passes.
  combine     the weighted median of ln s_i (weights sqrt(min(points, 2000)), and no view weighs more
              than the median view); views beyond max(3 x 1.4826 MAD, ln 1.05) of it are outliers, the
              others are used; s is the weighted median of the used views. It fails when fewer than 3
              views have points, fewer than 3 or under half of them agree, the spread is over 8%, or s
              is outside 0.5 to 2.
The record is pool/lidar.json (VERSIONS: 3 for a bag, 2 for a Stray capture); check images show the
points coloured by r / s on the
prep image, to spot calibration and timing errors. estimate(points_out=...) also gives each view's
points with their rows in the samples ("index"), so the lidar stage can show which points the scale
used. read_calibration() reads the LiDAR-camera calibration file (YAML or JSON).

    python -m gator.stage_a.lidar_scale estimate --pool P --samples S.npz --out F [--check DIR]
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from gator.fileio import write_json

SCALED = ("depth", "pointmaps", "pointmaps_sam3d")  # scaled whole; the extrinsics' translation column too
# Of pool/lidar.json, by the samples' source (job.json "lidar" "kind"). steps.done("lidar") takes only a record of its
# source's version, so the scale is found again when the estimator changes for that source. Version 2 came with the
# samples' time window and time base of lidar.CACHE_VERSION 4, ZBUFFER_ABS, the trim, the scatter rule on the used
# points, and the weights capped at the median view's. Version 3 (a bag only) follows the points along the LiDAR's rays
# (follow_rays): at 0.4 m with a LiDAR 15 cm beside the camera, s came out 0.8% high on average and single views up to
# 2.6% high (synthetic). A Stray capture has no rays and gets the same scale as before, so its jobs stay current.
VERSIONS = {"rosbag": 3, "stray": 2}
VERSION = max(VERSIONS.values())  # the newest

# Projection.
MIN_Z = 0.1  # m; nearer points are dropped
FOLD_MARGIN = 0.2  # plumb_bob, rational: the pinhole pre-filter keeps points inside the image grown by this share
FOLD_STEPS = 4096  # samples of the distortion curve when finding where it stops increasing (the fold)
DISTORTION_MODELS = ("plumb_bob", "rational_polynomial", "equidistant")
MODEL_ALIASES = {"radtan": "plumb_bob", "radial-tangential": "plumb_bob", "fisheye": "equidistant",
                 "kannala_brandt": "equidistant", "kannala-brandt": "equidistant"}

# Per view.
ZBUFFER_TOL = 0.03  # a point is kept when z <= (1 + tol) x the nearest z in its 3x3 DA3 pixels + ZBUFFER_ABS
ZBUFFER_ABS = 0.09  # m: 3 sigma of a spinning LiDAR's range noise, which a relative tolerance alone would cut
ERODE_FRAC = 0.01  # the mask is eroded by this share of the DA3 grid's diagonal (at least 1 px)
ERODE_KEEP = 0.3  # ... less while the eroded mask keeps under this share of the mask
CONF_PERCENTILE = 30  # DA3 conf must be at least this percentile of conf in the mask
MIN_POINTS = 30  # fewer used points: few_points
MAX_IQR_REL = 0.25  # a wider IQR of the used points' r over s_i: scattered
TRIM_MADS = 3.0  # used points lie within max(TRIM_MADS x 1.4826 MAD, ln(1 + TRIM_MIN_REL)) of the median ln r
TRIM_MIN_REL = 0.02
MAX_TRIM_REL = 0.4  # the trim drops more of the valid points: scattered (two groups of about the same size)
MAX_TRIM_NEAR_REL = 0.1  # ... more of them below the median r, where no hidden point lies: scattered

# Rays (ray_hits, follow_rays).
RAY_SPAN = 0.25  # a ray is traced where the true range can be: within +-25% of the measured range
RAY_STEP_PX = 0.5  # its samples lie this many DA3 px apart on average (more at the span's near end) ...
RAY_STEPS = (64, 512)  # ... with at least and at most this many steps along it
RAY_EDGE_DEG = 84.0  # DA3's depth that changes faster along a ray than a surface seen this far from face-on,
RAY_EDGE_REL = 0.005  # plus this share of the depth (DA3's noise), is an edge
RAY_FIT_PX = 3.0  # the crossing is where a line fit to the gap over this many DA3 px either side of it is zero
# A larger share of the eroded mask's rays in front of the surface over their span: scattered. In synthetic scenes,
# correct views of a 30 cm box stay at or below 0.1 with DA3 edges blurred by up to 8 px (Gaussian sigma) at 3,500
# beams per sr, and up to 6 px at 40,000 (8 px gives 0.102). Views locked on a wall close behind a small box lie at
# 0.34 to 0.5. Thin objects, softer edges and the image's edge can put correct views over it, and a dark object's lock
# can stay under it (build_plan.md note 12, open issues).
RAY_MAX_FRONT = 0.1
RAY_PASSES = 3  # at most; a pass traces the rays at the s_i of the pass before ...
RAY_TOL = 1e-4  # ... until s_i changes by less than this share
RAY_CHUNK = 250_000  # ray samples at a time

# Combine.
WEIGHT_CAP = 2000  # a view's weight is sqrt(min(points, WEIGHT_CAP)), and at most the median view's
OUTLIER_MADS = 3.0  # outliers lie beyond max(OUTLIER_MADS x 1.4826 MAD, ln(1 + OUTLIER_MIN_REL))
OUTLIER_MIN_REL = 0.05
MIN_VIEWS = 3
MAX_SPREAD_REL = 0.08  # the weighted RMS of the used views' ln s_i about ln s, as a share
S_MIN, S_MAX = 0.5, 2.0  # outside: likely an inverted transform or wrong units
# What the reason of each failed check points to, by the samples' source (job.json "lidar" "kind"). A Stray capture's
# depth is aligned with the color camera and paired with its frame by name, so it has no time window, no time offset
# and no T_cam_lidar.
CAUSES = {
    "rosbag": {"coverage": "the object may be outside the LiDAR's view, or the time window or T_cam_lidar is wrong "
                           "(see the check images)",
               "agreement": "look for wrong masks, a time offset or a wrong T_cam_lidar in the check images",
               "spread": "look for a time offset or wrong masks in the check images",
               "range": "check T_cam_lidar (is it T_lidar_cam?) and the units"},
    "stray": {"coverage": "the phone's depth may have low confidence on the object, the object may be small or far "
                          "away, or the masks may miss it (see the check images)",
              "agreement": "look for wrong masks, or a shiny or dark object whose depth the phone gets wrong, in the "
                           "check images",
              "spread": "look for wrong masks, or a shiny or dark object whose depth the phone gets wrong, in the "
                        "check images",
              "range": "check that the depth images hold millimeters, as Stray Scanner writes them"}}

# Check images.
CHECK_SIDE = 640  # long side in px
CHECK_RANGE = 1.25  # the colours saturate at r / s = 1 / 1.25 (blue) and 1.25 (red)

# Calibration files.
ORTHONORMAL_TOL = 1e-3  # |R R^T - I| below this is re-orthonormalised; above it is an error
MAX_TRANSLATION = 2.0  # m between LiDAR and camera; more is likely millimeters
MAX_YAML_DEPTH = 16  # lists and maps in lists and maps; real calibration files use at most 6
# The camera's optical axes (x right, y down, z forward) from body axes (x forward, y left, z up).
R_OPT_LINK = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])

VIEW_KEYS = ("zbuffer_tol", "zbuffer_abs", "erode_frac", "erode_keep", "conf_percentile", "min_points", "max_iqr_rel",
             "trim_mads", "trim_min_rel", "max_trim_rel", "max_trim_near_rel")
RAY_KEYS = ("ray_span", "ray_step_px", "ray_edge_deg", "ray_edge_rel", "ray_fit_px", "ray_max_front")
COMBINE_KEYS = ("weight_cap", "outlier_mads", "outlier_min_rel", "min_views", "max_spread_rel", "s_min", "s_max")


def apply_scale(arrays, s):
    """A copy of da3_output.npz's arrays with depth, pointmaps, pointmaps_sam3d and the extrinsics'
    translations times s, and metric_scale (the product of every scale applied) set. Intrinsics,
    confidences and image files stay."""
    s = float(s)
    if not np.isfinite(s) or s <= 0:
        raise ValueError(f"the scale is a positive number, not {s}")
    out = {k: np.array(v, copy=True) for k, v in arrays.items()}
    for k in SCALED:
        if k in out:
            out[k] = out[k] * np.asarray(s, dtype=out[k].dtype)
    if "extrinsics" in out:
        out["extrinsics"][..., :3, 3] *= s
    out["metric_scale"] = np.float64(s * float(arrays.get("metric_scale", 1.0)))
    return out


def thresholds(**overrides):
    """Every threshold of the estimate by name (the module constants in lower case), with overrides."""
    t = {"zbuffer_tol": ZBUFFER_TOL, "zbuffer_abs": ZBUFFER_ABS, "erode_frac": ERODE_FRAC, "erode_keep": ERODE_KEEP,
         "conf_percentile": CONF_PERCENTILE, "min_points": MIN_POINTS, "max_iqr_rel": MAX_IQR_REL,
         "trim_mads": TRIM_MADS, "trim_min_rel": TRIM_MIN_REL, "max_trim_rel": MAX_TRIM_REL,
         "max_trim_near_rel": MAX_TRIM_NEAR_REL,
         "ray_span": RAY_SPAN, "ray_step_px": RAY_STEP_PX, "ray_edge_deg": RAY_EDGE_DEG, "ray_edge_rel": RAY_EDGE_REL,
         "ray_fit_px": RAY_FIT_PX, "ray_max_front": RAY_MAX_FRONT, "ray_passes": RAY_PASSES,
         "weight_cap": WEIGHT_CAP, "outlier_mads": OUTLIER_MADS, "outlier_min_rel": OUTLIER_MIN_REL,
         "min_views": MIN_VIEWS, "max_spread_rel": MAX_SPREAD_REL, "s_min": S_MIN, "s_max": S_MAX}
    unknown = sorted(set(overrides) - set(t))
    if unknown:
        raise TypeError(f"unknown thresholds: {', '.join(unknown)}")
    return {**t, **overrides}


def source_to_grid(u, v, crop, grid_wh):
    """Source pixels to a grid that covers prep's crop box (x0, y0, x1, y1), pixel centres at integers,
    as in mvsam3d.cameras(): (u + 0.5 - x0) * w / (x1 - x0) - 0.5. The grid is DA3's or the prep image."""
    x0, y0, x1, y1 = (float(c) for c in crop)
    w, h = grid_wh
    return ((np.asarray(u, float) + 0.5 - x0) * w / (x1 - x0) - 0.5,
            (np.asarray(v, float) + 0.5 - y0) * h / (y1 - y0) - 0.5)


def rot90_points(u, v, size_wh, k):
    """Pixels of an image of size size_wh to the same pixels in np.rot90(image, k) (k counterclockwise
    quarter turns). The turned image's size is size_wh swapped when k is odd."""
    u, v = np.asarray(u, float), np.asarray(v, float)
    w, h = size_wh
    for _ in range(k % 4):
        u, v, w, h = v, w - 1 - u, h, w
    return u, v


def pixel(x):
    """The pixel index of a coordinate (pixel centres at integers)."""
    return np.floor(np.asarray(x, float) + 0.5).astype(int)


def distortion(D, model):
    """(model, coefficients): the model's canonical name (None for no distortion) and D as floats."""
    D = np.zeros(0) if D is None else np.asarray(D, float).reshape(-1)
    if model is not None and not isinstance(model, str):
        raise ValueError(f"the distortion model is a name, not {model!r}")
    model = model if model is None else model.strip().lower()
    model = MODEL_ALIASES.get(model, model)
    if model in (None, "", "none", "pinhole"):
        if D.any():
            raise ValueError("D has nonzero coefficients but no distortion_model; "
                             "give plumb_bob, rational_polynomial or equidistant")
        return None, D
    if model not in DISTORTION_MODELS:
        raise ValueError(f"unknown distortion model {model!r}; use plumb_bob, rational_polynomial or equidistant")
    if model == "equidistant" and np.any(D[4:]):
        raise ValueError(f"equidistant takes 4 coefficients (k1..k4), not {len(D)}")
    if np.any(D[8:]):
        raise ValueError(f"D has {len(D)} coefficients; the thin prism and tilt terms after k6 are not supported")
    return model, D


def distort(x, y, D, model):
    """Normalised pinhole coordinates to distorted ones, as OpenCV and ROS do it. plumb_bob and
    rational_polynomial take D = k1, k2, p1, p2, k3, k4, k5, k6 (missing ones are 0); equidistant
    (Kannala-Brandt, OpenCV's fisheye) takes k1..k4 on the angle from the axis."""
    model, D = distortion(D, model)
    x, y = np.asarray(x, float), np.asarray(y, float)
    if model is None:
        return x, y
    if model == "equidistant":
        r = np.hypot(x, y)
        f = np.where(r > 1e-12, _theta_d(np.arctan(r), D) / np.maximum(r, 1e-12), 1.0)
        return x * f, y * f
    _, _, p1, p2 = np.pad(D[:4], (0, max(0, 4 - len(D))))
    r2 = x * x + y * y
    radial = _radial(r2, D)
    return (x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
            y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)


def _theta_d(th, D):
    """Kannala-Brandt: the distorted angle of the angle th from the axis."""
    k1, k2, k3, k4 = np.pad(D[:4], (0, max(0, 4 - len(D))))
    t2 = th * th
    return th * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))


def _radial(r2, D):
    """plumb_bob and rational_polynomial: the radial factor at r^2."""
    k1, k2, _, _, k3, k4, k5, k6 = np.pad(D[:8], (0, max(0, 8 - len(D))))
    return (1 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (1 + r2 * (k4 + r2 * (k5 + r2 * k6)))


def fold_limit(D, model, hi):
    """Where the distortion curve on [0, hi] first stops increasing, or inf: the radius r (plumb_bob,
    rational_polynomial: r times the radial factor) or the angle from the axis (equidistant: theta_d).
    Points beyond it fold back towards the centre and land on the wrong pixels."""
    model, D = distortion(D, model)
    if model is None or not np.isfinite(hi) or hi <= 0:
        return np.inf
    x = np.linspace(0.0, hi, FOLD_STEPS + 1)
    with np.errstate(all="ignore"):
        y = _theta_d(x, D) if model == "equidistant" else x * _radial(x * x, D)
        bad = ~(np.diff(y) > 0)
    return float(x[np.argmax(bad)]) if bad.any() else np.inf


def project(X_c, K, D, model, size_wh):
    """Camera points (N, 3), OpenCV optical axes in meters, to pixels: (u, v, z, ok). ok marks the
    points in front (z > MIN_Z) that land inside the image of size size_wh. Far outside the calibrated
    field the distortion polynomial can fold points back into the image, so two guards come first.
    plumb_bob and rational_polynomial: a pinhole pre-filter keeps only points whose undistorted pixel
    lies inside the image grown by FOLD_MARGIN of its size on every side, and their radius must be
    below fold_limit(). equidistant: the angle from the axis must be below fold_limit() on [0, 90 deg]
    (a pinhole bound would drop a wide fisheye's outer ring)."""
    X = np.asarray(X_c, float).reshape(-1, 3)
    K = np.asarray(K, float)
    W, H = size_wh
    model, D = distortion(D, model)
    z = X[:, 2]
    front = np.isfinite(X).all(1) & (z > MIN_Z)
    zs = np.where(front, z, 1.0)
    x, y = np.where(front, X[:, 0] / zs, 0.0), np.where(front, X[:, 1] / zs, 0.0)
    r = np.hypot(x, y)
    if model == "equidistant":
        near = front & (np.arctan(r) < fold_limit(D, model, np.pi / 2))
    else:
        up, vp = K[0, 0] * x + K[0, 1] * y + K[0, 2], K[1, 1] * y + K[1, 2]
        mw, mh = FOLD_MARGIN * W, FOLD_MARGIN * H
        lo_u, hi_u, lo_v, hi_v = -0.5 - mw, W - 0.5 + mw, -0.5 - mh, H - 0.5 + mh
        near = front & (up >= lo_u) & (up < hi_u) & (vp >= lo_v) & (vp < hi_v)
        if model is not None:  # the radius of the grown image's farthest corner bounds the search
            corners = np.linalg.solve(K, np.array([[lo_u, hi_u, lo_u, hi_u], [lo_v, lo_v, hi_v, hi_v], [1.0] * 4]))
            near &= r < fold_limit(D, model, float(np.hypot(*corners[:2]).max()))
    xd, yd = distort(np.where(near, x, 0.0), np.where(near, y, 0.0), D, model)
    u, v = K[0, 0] * xd + K[0, 1] * yd + K[0, 2], K[1, 1] * yd + K[1, 2]
    ok = near & (u >= -0.5) & (u < W - 0.5) & (v >= -0.5) & (v < H - 0.5)
    return u, v, z, ok


def zbuffer(iu, iv, z, hw, tol=ZBUFFER_TOL, abs_tol=ZBUFFER_ABS):
    """Which points (pixel indices iu, iv on a grid of size hw, depth z) to keep: z at most (1 + tol)
    times the nearest z in their 3x3 neighbourhood, plus abs_tol meters. A LiDAR beside the camera sees
    surfaces that the camera cannot, and their points land among the foreground's. abs_tol lets range
    noise pass: without it the nearest of the noisy points in a window would decide, and the points
    kept would lie nearer than the surface."""
    from scipy.ndimage import minimum_filter

    iu, iv, z = np.asarray(iu, int), np.asarray(iv, int), np.asarray(z, float)
    near = np.full(tuple(hw), np.inf)
    np.minimum.at(near, (iv, iu), z)
    near = minimum_filter(near, size=3, mode="constant", cval=np.inf)
    return z <= (1 + tol) * near[iv, iu] + abs_tol


def trim(r, mads=TRIM_MADS, min_rel=TRIM_MIN_REL, log=True):
    """Which ratios r (all finite and positive) to keep: ln r within max(mads x 1.4826 MAD,
    ln(1 + min_rel)) of the median ln r; mads inf keeps them all. Hidden points that the z-buffer
    missed and points at the mask's edge sit on one side of the median and would move it. At least half
    of the points stay, since half lie within one MAD. log False trims r itself, within max(mads x
    1.4826 MAD, min_rel x the median): r that is linear in the range noise (follow_rays) stays even
    about its median, and a cut even in ln r would drop more of its low side."""
    x = np.log(np.asarray(r, float)) if log else np.asarray(r, float)
    if not len(x) or np.isinf(mads):  # inf x a MAD of 0 would be NaN, and would drop every point
        return np.ones(len(x), bool)
    m = np.median(x)
    least = float(np.log1p(min_rel)) if log else min_rel * abs(float(m))
    return np.abs(x - m) <= max(mads * 1.4826 * float(np.median(np.abs(x - m))), least)


def bilinear(a, x, y):
    """a (h, w) at grid coordinates x, y (pixel centres at integers), bilinear. NaN outside the pixel
    centres and next to a pixel that is not finite and positive."""
    h, w = a.shape
    a = np.where(np.isfinite(a) & (a > 0), a, np.nan)
    x, y = np.asarray(x, float), np.asarray(y, float)
    inside = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)  # NaN is outside
    x, y = np.where(inside, x, 0.0), np.where(inside, y, 0.0)
    x0, y0 = np.minimum(np.floor(x), max(w - 2, 0)).astype(int), np.minimum(np.floor(y), max(h - 2, 0)).astype(int)
    fx, fy = x - x0, y - y0
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    v = (1 - fy) * ((1 - fx) * a[y0, x0] + fx * a[y0, x1]) + fy * ((1 - fx) * a[y1, x0] + fx * a[y1, x1])
    return np.where(inside, v, np.nan)


def ray_hits(points, origin, camera, crop, depth, s, span=RAY_SPAN, step_px=RAY_STEP_PX, edge_deg=RAY_EDGE_DEG,
             edge_rel=RAY_EDGE_REL, fit_px=RAY_FIT_PX):
    """Where each LiDAR ray meets s x DA3's depth, a surface as the camera sees it: (hit, rho, gu, gv,
    slope, front), one row per point.

    points (N, 3): the points in the camera frame (meters); origin (3,): the LiDAR's origin in that
    frame; camera: the K, D, distortion model ("model") and image size ("size", w x h) that made the
    samples' pixels (project); crop, depth: as view_ratio's. A ray runs from origin through its point. It
    is sampled where its range lies within rho / (1 + span) to rho / (1 - span) of the point's range rho
    (so the true range is within +-span of the measured one), evenly in range, with as many steps as make
    the span's pinhole length on the grid step_px DA3 px per step (at most RAY_STEPS[1]). The pixels move
    faster at the near end, by the ratio of the span's far z to its near z in the camera frame times the
    mean step: (1 + span) / (1 - span) for an origin in the camera's z = 0 plane, more for one behind the
    camera or with pincushion distortion. Each ray has its own steps, so its result does not depend on
    the other rays in the call. Its gap
    g = z - s x depth (bilinear; unknown outside the image or the grid, or without valid depth) is below
    0 in front of the surface. The ray meets the surface at its first sample with g >= 0, unless the depth
    changes there faster than on a surface that the camera sees edge_deg from face-on, plus edge_rel of
    the depth. That is an edge: the ray came from behind the object as the camera sees it (a face that
    only the LiDAR sees, or what lies behind the object). A ray that starts behind the surface, or meets
    none in its span, is not a hit either. front marks the rays that stay in front of the surface over
    their whole span (with at least one known sample): points in free space as the camera sees it, which
    no occlusion explains.

    rho (the crossing's range) and slope (dg / drho there) come from a line fit to g over the samples
    within fit_px DA3 px of the first crossing, with no edge between them. With noisy depth, the first
    sample past the surface comes early. gu, gv: the crossing on DA3's grid. Rows that are not hits have
    NaN."""
    P0 = np.asarray(points, float).reshape(-1, 3)
    O = np.asarray(origin, float).reshape(3)
    depth = np.where(np.isfinite(depth) & (depth > 0), np.asarray(depth, float), np.nan)
    h, w = depth.shape
    K, D, model = np.asarray(camera["K"], float), camera.get("D"), camera.get("model")
    size = tuple(int(x) for x in camera["size"])
    x0, y0, x1, y1 = (float(c) for c in crop[:4])
    slope_px = np.tan(np.radians(edge_deg)) * (x1 - x0) / (K[0, 0] * w)  # relative depth change per DA3 px
    n = len(P0)
    hit, front, rho, gu, gv, slope = np.zeros(n, bool), np.zeros(n, bool), *(np.full(n, np.nan) for _ in range(4))
    ray = P0 - O
    rng = np.linalg.norm(ray, axis=1)
    good = np.isfinite(rng) & (rng > 0)
    dirs = ray / np.where(good, rng, 1.0)[:, None]
    lo, hi = rng / (1 + span), rng / (1 - span)

    def grid(X):  # (gu, gv, z) of points on DA3's grid; gu is NaN outside the image
        u, v, z, ok = project(X.reshape(-1, 3), K, D, model, size)
        a, b = source_to_grid(u, v, crop, (w, h))
        shape = X.shape[:-1]
        return np.where(ok, a, np.nan).reshape(shape), b.reshape(shape), z.reshape(shape)

    rows = np.flatnonzero(good)
    ends = [O + r[rows, None] * dirs[rows] for r in (lo, hi)]
    with np.errstate(invalid="ignore", divide="ignore"):  # the span's length on the grid, pinhole: also off the image
        a, b = ((e[:, :2] / e[:, 2:]) @ K[:2, :2].T * [w / (x1 - x0), h / (y1 - y0)] for e in ends)
        far = np.where((ends[0][:, 2] > MIN_Z) & (ends[1][:, 2] > MIN_Z), np.hypot(*(a - b).T), np.inf)
    steps = np.clip(np.ceil(np.where(np.isfinite(far), far, np.inf) / step_px), *RAY_STEPS).astype(int)
    order = np.argsort(steps, kind="stable")
    rows, steps = rows[order], steps[order]  # rays with a similar number of steps go together
    per = max(1, RAY_CHUNK // (RAY_STEPS[1] + 1))
    with np.errstate(invalid="ignore", divide="ignore"):
        for i in range(0, len(rows), per):
            c, n_ = rows[i:i + per], steps[i:i + per, None]
            m = int(n_.max())
            k = np.arange(len(c))
            q = np.arange(m + 1) / n_  # each ray at its own steps, and NaN past its end
            R = np.where(q <= 1, lo[c, None] + (hi - lo)[c, None] * q, np.nan)
            u, v, z = grid(O + R[..., None] * dirs[c, None, :])
            hgt = s * bilinear(depth, u, v)
            g = z - hgt
            lim = z[:, 1:] * (slope_px * np.hypot(np.diff(u, axis=1), np.diff(v, axis=1)) + edge_rel)
            edge = np.c_[np.zeros(len(c), bool), ~(np.abs(np.diff(hgt, axis=1)) <= lim)]  # unknown depth too
            behind = g >= 0  # unknown is not behind
            front[c] = ~behind.any(axis=1) & np.isfinite(g).any(axis=1)
            j = np.maximum(np.argmax(behind, axis=1), 1)
            ok = behind[k, j] & (g[k, j - 1] < 0) & ~edge[k, j]  # the ray starts in front and crosses no edge
            f = -g[k, j - 1] / (g[k, j] - g[k, j - 1])  # the crossing between the two samples
            cu, cv = u[k, j - 1] + f * (u[k, j] - u[k, j - 1]), v[k, j - 1] + f * (v[k, j] - v[k, j - 1])
            seg = np.cumsum(edge, axis=1)
            win = (seg == seg[k, j][:, None]) & (np.hypot(u - cu[:, None], v - cv[:, None]) <= fit_px) & np.isfinite(g)
            cnt = win.sum(1)
            mr = np.where(win, R, 0.0).sum(1) / np.maximum(cnt, 1)
            mg = np.where(win, g, 0.0).sum(1) / np.maximum(cnt, 1)
            dr = np.where(win, R - mr[:, None], 0.0)
            b_ = (dr * np.where(win, g - mg[:, None], 0.0)).sum(1) / (dr * dr).sum(1)
            r_ = mr - mg / b_
            ok &= (cnt >= 2) & (b_ > 0) & (r_ >= lo[c]) & (r_ <= hi[c])
            q = np.clip(np.where(ok, (r_ - lo[c]) / (hi - lo)[c] * n_[:, 0], 0.0), 0, n_[:, 0] - 1e-9)
            q0 = np.floor(q).astype(int)
            fq = q - q0
            cu, cv = u[k, q0] + fq * (u[k, q0 + 1] - u[k, q0]), v[k, q0] + fq * (v[k, q0 + 1] - v[k, q0])
            ok &= np.isfinite(cu) & np.isfinite(cv)
            hit[c], rho[c], slope[c] = ok, np.where(ok, r_, np.nan), np.where(ok, b_, np.nan)
            gu[c], gv[c] = np.where(ok, cu, np.nan), np.where(ok, cv, np.nan)
    return hit, rho, gu, gv, slope, front


def view_ratio(uvz, crop, depth, conf, mask, zbuffer_tol=ZBUFFER_TOL, zbuffer_abs=ZBUFFER_ABS, erode_frac=ERODE_FRAC,
               erode_keep=ERODE_KEEP, conf_percentile=CONF_PERCENTILE, min_points=MIN_POINTS, max_iqr_rel=MAX_IQR_REL,
               trim_mads=TRIM_MADS, trim_min_rel=TRIM_MIN_REL, max_trim_rel=MAX_TRIM_REL,
               max_trim_near_rel=MAX_TRIM_NEAR_REL, ray=None, s_ref=None, ray_span=RAY_SPAN, ray_step_px=RAY_STEP_PX,
               ray_edge_deg=RAY_EDGE_DEG, ray_edge_rel=RAY_EDGE_REL, ray_fit_px=RAY_FIT_PX,
               ray_max_front=RAY_MAX_FRONT):
    """One view's ratio of LiDAR depth to DA3 depth. uvz: the frame's samples; crop: prep's crop box
    of the frame; depth, conf: DA3's (h, w) arrays; mask: the pool mask at any size.

    Returns (record, points). The record counts the points at each step (points: the samples,
    points_grid: inside DA3's grid, points_visible: after the z-buffer, or along rays the hits whose
    twins land on the grid, points_mask:
    inside the eroded mask, points_valid: with valid DA3 depth and conf, points_used: also within the
    trim of ln r) and
    gives erode_px, conf_min, s (the median of the used points' r = z / depth), iqr_rel (the IQR of
    the used points' r over s) and status: no_mask, no_samples, few_points (under min_points valid
    points: the trim must not turn a scattered view into one that combine leaves out of its counts),
    scattered or candidate (combine makes it used or outlier). The view is scattered when iqr_rel is over
    max_iqr_rel, or the trim drops more than max_trim_rel of the valid points, or more than
    max_trim_near_rel of them below their median r. Hidden points that the z-buffer missed always lie
    farther. The trim may drop them, up to max_trim_rel of the valid points, and the view keeps its s,
    since the IQR is that of the points that give s. points holds the visible points for the check
    image: u, v (source pixels), r (NaN without valid depth), used, and index: each point's row in uvz.

    ray (follow_rays): the view's LiDAR rays, {points: (N, 3) camera points row for row with uvz,
    origin, and the camera of ray_hits}, traced at the scale s_ref. ray_hits then takes the z-buffer's
    place: at the noisy pixel, the points measured short hid the ones measured long, and on a dense scan s
    came out about 0.6% low. A visible point must be a ray hit; the record's points_hidden counts the
    others, and points_twin the hits that the twin rule below drops. The mask, conf and depth are read
    where its ray meets the surface. Its r is the scale that puts that crossing at the measured range:
    s_ref + (range - crossing) x slope / depth. That is linear in the range noise, so the trim is on r, not
    ln r (trim with log False). Without a baseline (origin 0), r is z over DA3's depth at the point.
    A hit stays only when its twin (the point mirrored about its crossing along its ray) also lands in the
    image and on DA3's grid. The samples hold only the points whose noisy pixel is in the image, so near
    the image's edge on the LiDAR's side the points measured short are gone; without their twins' rule the
    others made s high there. front_rel is the share of the rays from the eroded mask (by their own pixel)
    that stay in front of the surface over their whole span (ray_hits' front), and over ray_max_front the
    view is scattered. No occlusion explains such points, so s_ref is likely the scale of another surface,
    such as a wall close behind a small object, which the rays then confirm. front_rel counts only the
    points that the z-buffer keeps (rays_mask of them): the wall's points that the LiDAR sees past the
    object's edge are hits at the wall's scale, and on a dense scan they made a lock look clean. This rule
    comes before few_points when rays_mask is at least min_points: such a view has few hits because its
    rays miss the surface, and few_points would leave it out of combine's counts."""
    from scipy.ndimage import distance_transform_edt

    from gator.webapp.views import at_size

    depth = np.asarray(depth, float)
    h, w = depth.shape
    conf = np.ones_like(depth) if conf is None else np.asarray(conf, float)
    m = at_size(np.asarray(mask) > 0, (h, w))
    uvz = np.asarray(uvz, float).reshape(-1, 3)
    if ray is not None and len(ray["points"]) != len(uvz):
        raise ValueError(f"the rays have {len(ray['points'])} points and the samples {len(uvz)}; they go row for row")
    rows = np.flatnonzero(np.isfinite(uvz).all(1) & (uvz[:, 2] > 0))
    uvz = uvz[rows]
    gu, gv = source_to_grid(uvz[:, 0], uvz[:, 1], crop, (w, h))
    iu, iv = pixel(gu), pixel(gv)
    g = (iu >= 0) & (iu < w) & (iv >= 0) & (iv < h)
    iu, iv, z, src, rows = iu[g], iv[g], uvz[g, 2], uvz[g, :2], rows[g]
    vis = zbuffer(iu, iv, z, (h, w), zbuffer_tol, zbuffer_abs)
    if ray is None:  # along the rays, ray_hits drops what the camera cannot see, and vis counts only in front_rel
        iu, iv, z, src, rows = iu[vis], iv[vis], z[vis], src[vis], rows[vis]
    dist = distance_transform_edt(m)
    e = max(1, round(erode_frac * np.hypot(h, w)))
    while e > 0 and (dist > e).sum() < erode_keep * m.sum():
        e -= 1
    inner = dist > e
    hidden = front_rel = None
    own_n = 0
    if ray is not None:
        O = np.asarray(ray["origin"], float).reshape(3)
        X = np.asarray(ray["points"], float).reshape(-1, 3)[rows]
        hit, rho, hu, hv, slope, fr = ray_hits(X, O, ray, crop, depth, s_ref, ray_span, ray_step_px, ray_edge_deg,
                                               ray_edge_rel, ray_fit_px)
        own = inner[iv, iu] & vis  # the points' own pixels in the eroded mask, of the points the z-buffer keeps
        own_n = int(own.sum())
        hidden, front_rel = int((~hit).sum()), float((fr & own).sum() / own_n) if own_n else None
        with np.errstate(invalid="ignore"):  # the twin: the point mirrored about its crossing, along its ray
            rng = np.linalg.norm(X - O, axis=1)
            tu, tv, _, tok = project(O + (2 * rho - rng)[:, None] * (X - O) / rng[:, None], ray["K"], ray.get("D"),
                                     ray.get("model"), tuple(int(x) for x in ray["size"]))
            tu, tv = source_to_grid(np.where(tok, tu, -1.0), np.where(tok, tv, -1.0), crop, (w, h))
        twin = tok & (pixel(tu) >= 0) & (pixel(tu) < w) & (pixel(tv) >= 0) & (pixel(tv) < h)
        keep = hit & twin
        twins = int((hit & ~twin).sum())
        X, rho, hu, hv, slope, src, rows = X[keep], rho[keep], hu[keep], hv[keep], slope[keep], src[keep], rows[keep]
        iu, iv = np.clip(pixel(hu), 0, w - 1), np.clip(pixel(hv), 0, h - 1)

    in_mask = m & np.isfinite(conf)
    conf_min = float(np.percentile(conf[in_mask], conf_percentile)) if in_mask.any() else None
    c = conf[iv, iu]
    with np.errstate(invalid="ignore", divide="ignore"):
        if ray is None:
            d = depth[iv, iu]
            valid = np.isfinite(d) & (d > 0)
            r = np.where(valid, z / np.where(valid, d, 1.0), np.nan)
        else:
            d = bilinear(depth, hu, hv)
            r = s_ref + (np.linalg.norm(X - O, axis=1) - rho) * slope / d
            valid = np.isfinite(r) & (r > 0)
            r = np.where(valid, r, np.nan)
    fit = inner[iv, iu] & valid & np.isfinite(c) & (c >= (conf_min if conf_min is not None else np.inf))
    used = fit.copy()
    used[fit] = trim(r[fit], trim_mads, trim_min_rel, log=ray is None)
    ru = r[used]
    s = float(np.median(ru)) if len(ru) else None
    iqr = float(np.subtract(*np.percentile(ru, [75, 25])) / s) if len(ru) else None
    rf, dropped = r[fit], ~used[fit]  # the valid points' r, and which of them the trim dropped
    near = int((dropped & (rf < np.median(rf))).sum()) if len(rf) else 0
    status = ("no_mask" if not m.any() else "no_samples" if not len(uvz)
              else "scattered" if own_n >= max(1, min_points) and front_rel > ray_max_front  # also without hits
              else "few_points" if fit.sum() < max(1, min_points) or not len(ru)
              else "scattered" if iqr > max_iqr_rel or dropped.sum() > max_trim_rel * len(rf)
              or near > max_trim_near_rel * len(rf) else "candidate")
    rec = {"points": int(len(uvz)), "points_grid": int(g.sum()), "points_visible": int(len(rows)),
           "points_mask": int(inner[iv, iu].sum()), "points_valid": int(fit.sum()), "points_used": int(used.sum()),
           "erode_px": int(e),
           "conf_min": None if conf_min is None else round(conf_min, 4),
           "s": None if s is None else round(s, 6), "iqr_rel": None if iqr is None else round(iqr, 4), "status": status}
    if hidden is not None:
        rec["points_hidden"], rec["points_twin"], rec["rays_mask"] = hidden, twins, own_n
        rec["front_rel"] = None if front_rel is None else round(front_rel, 4)
    return rec, {"u": src[:, 0], "v": src[:, 1], "r": r, "used": used, "index": rows}


def follow_rays(uvz, crop, depth, conf, mask, ray, s, t):
    """view_ratio along the view's LiDAR rays (ray, as view_ratio's), from s, the view's scale without
    them. Each pass traces the rays at the s of the pass before, up to t["ray_passes"] times, until s
    changes by less than RAY_TOL. The crossings move with s, so the first pass, at the scale without
    rays, can still be off by a little of that scale's bias. t: thresholds(). The record gains passes."""
    kw = {k: t[k] for k in VIEW_KEYS + RAY_KEYS}
    passes = 0
    while True:
        passes += 1
        rec, points = view_ratio(uvz, crop, depth, conf, mask, ray=ray, s_ref=s, **kw)
        done = rec["s"] is None or abs(rec["s"] / s - 1) < RAY_TOL or passes >= t["ray_passes"]
        s = rec["s"]
        if done:
            return {**rec, "passes": passes}, points


def weighted_median(x, w):
    """The value where the cumulative weight of the sorted x reaches half the total (the mean of the
    two middle values when it reaches exactly half)."""
    x, w = np.asarray(x, float), np.asarray(w, float)
    o = np.argsort(x)
    x, c = x[o], np.cumsum(w[o])
    half, tol = c[-1] / 2, 1e-9 * c[-1]  # sums of the same weights in another order differ by rounding
    i = int(np.searchsorted(c, half - tol))
    if abs(c[i] - half) <= tol and i + 1 < len(x):
        return float((x[i] + x[i + 1]) / 2)
    return float(x[i])


def combine(views, weight_cap=WEIGHT_CAP, outlier_mads=OUTLIER_MADS, outlier_min_rel=OUTLIER_MIN_REL,
            min_views=MIN_VIEWS, max_spread_rel=MAX_SPREAD_REL, s_min=S_MIN, s_max=S_MAX, source="rosbag"):
    """The scale from view_ratio's records (see the module docstring). Every candidate's status
    becomes used or outlier, in place. Returns ok, s (None when no view is a candidate), reason
    (None when ok; its causes are CAUSES[source]), spread_rel, the outlier cut and the counts. A view's
    weight is at most the median view's. Else a view with more than half the weight is the median alone
    and its MAD is 0, so every other view becomes an outlier, and one point more or less in the other
    views can flip the result."""
    causes = CAUSES[source]
    cand = [v for v in views if v["status"] == "candidate"]
    with_points = sum(v["status"] in ("candidate", "scattered") for v in views)
    s = spread = mad = cut = None
    kept = 0
    if cand:
        x = np.log([v["s"] for v in cand])
        wt = np.sqrt(np.minimum([v["points_used"] for v in cand], weight_cap))
        wt = np.minimum(wt, np.median(wt))
        m = weighted_median(x, wt)
        mad = weighted_median(np.abs(x - m), wt)
        cut = max(outlier_mads * 1.4826 * mad, float(np.log1p(outlier_min_rel)))
        keep = np.abs(x - m) <= cut
        for v, k in zip(cand, keep):
            v["status"] = "used" if k else "outlier"
        ls = weighted_median(x[keep], wt[keep])
        s, kept = float(np.exp(ls)), int(keep.sum())
        spread = float(np.expm1(np.sqrt(np.average((x[keep] - ls) ** 2, weights=wt[keep]))))
    count = {k: sum(v["status"] == k for v in views) for k in ("outlier", "scattered", "few_points", "no_mask", "no_samples")}
    need = max(min_views, int(np.ceil(with_points / 2)))
    if with_points < min_views:
        reason = (f"LiDAR points cover the object in only {with_points} of {len(views)} views (at least {min_views} "
                  f"are needed); {causes['coverage']}")
    elif kept < need:
        reason = (f"the per-view scales disagree: {kept} of the {with_points} views with LiDAR points agree "
                  f"({count['scattered']} scattered, {count['outlier']} outliers) and at least {need} must; "
                  f"{causes['agreement']}")
    elif spread > max_spread_rel:
        reason = f"the per-view scales spread by {spread:.1%}, more than {max_spread_rel:.0%}; {causes['spread']}"
    elif not s_min <= s <= s_max:
        reason = f"the LiDAR scale {s:.2f} is far from Depth Anything 3's metric estimate; {causes['range']}"
    else:
        reason = None
    return {"ok": reason is None, "s": s, "reason": reason, "spread_rel": None if spread is None else round(spread, 5),
            "views": len(views), "views_with_points": int(with_points), "candidates": len(cand), "views_used": kept,
            "outliers": count["outlier"], "scattered": count["scattered"], "few_points": count["few_points"],
            "no_mask": count["no_mask"], "no_samples": count["no_samples"],
            "mad_rel": None if mad is None else round(float(np.expm1(mad)), 5),
            "cut_rel": None if cut is None else round(float(np.expm1(cut)), 5)}


def load_samples(samples):
    """{name: (N, 3) float array} from a mapping or an .npz path; any other shape is an error."""
    if isinstance(samples, (str, Path)):
        with np.load(samples) as f:
            samples = {k: f[k] for k in f.files}
    out = {}
    for name, a in samples.items():
        a = np.asarray(a, float)
        if a.ndim != 2 or a.shape[1] != 3:
            raise ValueError(f"the samples of {name} have shape {a.shape}; they are (N, 3) [u, v, z]")
        out[str(name)] = a
    return out


def estimate(pool_dir, samples, check_dir=None, points_out=None, source="rosbag", rays=None, **overrides):
    """The lidar.json record of a pool (mvsam3d prep and da3 ran on it) and the LiDAR samples
    ({pool name: (N, 3) [u, v, z]} or an .npz). overrides change thresholds by name (thresholds()).
    With check_dir, writes check_dir/<name>.jpg for every view. A dict points_out gets {view name:
    view_ratio's points}. source, the samples' kind, picks the causes that a failed check names (CAUSES).
    rays: {pool name: view_ratio's ray}. A view with rays and a scale goes on along them (follow_rays);
    its record has passes (0 when the view had no scale to start from). The record's version is that of
    the estimator that ran: a bag's with rays, else a Stray capture's (VERSIONS)."""
    from PIL import Image

    from gator.stage_a.mvsam3d import view_names

    t0 = time.time()
    t = thresholds(**overrides)
    pool_dir = Path(pool_dir)
    samples = load_samples(samples)
    crops = {v["name"]: v["crop"] for v in json.loads((pool_dir / "prep.json").read_text())["views"]}
    with np.load(pool_dir / "da3_output.npz") as da:
        depth = da["depth"]
        conf = da["conf"] if "conf" in da.files else np.ones_like(depth)
        stem = {Path(str(f)).stem: i for i, f in enumerate(da["image_files"])}
        process_res = int(da["process_res"]) if "process_res" in da.files else None
    views, points = [], {}
    for n in view_names(pool_dir):
        if n not in stem:
            continue
        if n not in crops:
            raise ValueError(f"{n} is not in {pool_dir / 'prep.json'}; rerun prep")
        mask = np.array(Image.open(pool_dir / "masks" / f"{n}.png")) > 0
        i = stem[n]
        args = (samples.get(n, np.zeros((0, 3))), crops[n], depth[i], conf[i], mask)
        rec, points[n] = view_ratio(*args, **{k: t[k] for k in VIEW_KEYS})
        ray = (rays or {}).get(n)
        if ray is not None and rec["s"] is not None:
            rec, points[n] = follow_rays(*args, ray, rec["s"], t)
        elif ray is not None:
            rec["passes"] = 0
        views.append({"name": n, "frame": int(n[1:]) if re.fullmatch(r"f\d+", n) else None, **rec})
    comb = combine(views, source=source, **{k: t[k] for k in COMBINE_KEYS})
    ok, s, reason = comb.pop("ok"), comb.pop("s"), comb.pop("reason")
    if points_out is not None:
        points_out.update(points)
    warnings = []
    unknown = [k for k in samples if k not in points]
    if unknown:
        listed = ", ".join(unknown[:5]) + (f" and {len(unknown) - 5} more" if len(unknown) > 5 else "")
        warnings.append(f"LiDAR samples for frames that are not pool views: {listed} (pool views are named like f00012)")
        if len(unknown) == len(samples):
            reason = (f"none of the LiDAR samples' frame names ({listed}) is a pool view; pool views are named "
                      "like f00012, so the samples were made for another pool or with other names")
    if s is not None:
        warnings += [f"{v['name']}: its scale {v['s']:.3f} is {v['s'] / s - 1:+.0%} from the others; left out"
                     for v in views if v["status"] == "outlier"]
    missing = [v["name"] for v in views if v["status"] == "no_samples"]
    if missing:
        warnings.append(f"no LiDAR samples in {len(missing)} of {len(views)} views: {', '.join(missing)}")
    if check_dir is not None:
        check_dir = Path(check_dir)
        check_dir.mkdir(parents=True, exist_ok=True)
        for v in views:
            ref = s if s is not None else v["s"]
            check_image(pool_dir / "input" / "images" / f"{v['name']}.png", pool_dir / "masks" / f"{v['name']}.png",
                        crops[v["name"]], points[v["name"]], ref, check_dir / f"{v['name']}.jpg")
    return {"version": VERSIONS["rosbag" if rays else "stray"], "ok": ok, "s": s, "reason": reason, "views": views,
            "combine": {**comb, "thresholds": t},
            "da3": {"process_res": process_res, "grid": [int(depth.shape[2]), int(depth.shape[1])]},
            "warnings": warnings, "seconds": round(time.time() - t0, 2)}


def check_image(image, mask, crop, points, s, out):
    """Write out (a JPEG): the prep image, dimmed, long side CHECK_SIDE, with the mask outline in green
    and the points (view_ratio's) coloured by r / s: blue where the LiDAR is nearer than DA3 times s,
    white where they agree, red where it is farther, saturating at CHECK_RANGE. Used points are 3x3
    dots, the others 2x2. Without s (None) the points are white. The JPEG keeps full colour resolution:
    chroma subsampling would wash the red and blue dots into the background."""
    from PIL import Image
    from scipy.ndimage import binary_erosion

    from gator.webapp.views import at_size

    img = Image.open(image).convert("RGB")
    k = CHECK_SIDE / max(img.size)
    size = (round(img.width * k), round(img.height * k))
    a = np.asarray(img.resize(size, Image.BILINEAR), float) * 0.55
    m = at_size(np.array(Image.open(mask)) > 0 if isinstance(mask, (str, Path)) else np.asarray(mask) > 0,
                (size[1], size[0]))
    a[m & ~binary_erosion(m, iterations=2)] = (0, 255, 0)
    r = np.asarray(points["r"], float)
    keep = np.isfinite(r)
    u, v = source_to_grid(np.asarray(points["u"])[keep], np.asarray(points["v"])[keep], crop, size)
    iu, iv, r, used = pixel(u), pixel(v), r[keep], np.asarray(points["used"], bool)[keep]
    t = np.clip(np.log(r / s) / np.log(CHECK_RANGE), -1, 1) if s else np.zeros(len(r))
    white, blue, red = np.array([247.0, 247, 247]), np.array([33.0, 102, 172]), np.array([178.0, 24, 43])
    col = white + np.where(t[:, None] < 0, -t[:, None] * (blue - white), t[:, None] * (red - white))
    for sel, offsets in ((~used, (0, 1)), (used, (-1, 0, 1))):
        for dy in offsets:
            for dx in offsets:
                x, y = iu[sel] + dx, iv[sel] + dy
                ok = (x >= 0) & (x < size[0]) & (y >= 0) & (y < size[1])
                a[y[ok], x[ok]] = col[sel][ok]
    Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(out, quality=90, subsampling=0)
    return out


def summary(rec):
    """One line per view of a lidar.json record, then the result."""
    lines = []
    for v in rec["views"]:
        line = f"{v['name']}: {v['status']}, {v['points_used']} of {v['points']} points"
        if v["s"] is not None:
            line += f", s {v['s']:.4f}, IQR {v['iqr_rel']:.1%}"
        lines.append(line)
    c = rec["combine"]
    lines.append(f"metric scale {rec['s']:.4f} from {c['views_used']} views, spread {c['spread_rel']:.1%}" if rec["ok"]
                 else f"no metric scale: {rec['reason']}")
    return lines


def _floats(x, key):
    try:
        a = np.asarray(x, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"{key} is not a list of numbers") from None
    if not np.isfinite(a).all():
        raise ValueError(f"{key} has numbers that are not finite")
    return a


def _number(x, key):
    try:
        f = float(x)
    except (TypeError, ValueError):
        raise ValueError(f"{key} is not a number") from None
    if isinstance(x, bool) or not np.isfinite(f):
        raise ValueError(f"{key} is not a number")
    return f


def _matrix(x, key, shape=None, forms=""):
    """A float array of shape (flat, any size, without one) from a nested list, a flat list or
    OpenCV's {rows, cols, data}."""
    if isinstance(x, dict):
        if "data" not in x:
            raise ValueError(f"{key} has no data")
        a = _floats(x["data"], key).reshape(-1)
        rows, cols = x.get("rows"), x.get("cols")
        if rows is not None and cols is not None and a.size != int(_number(rows, key + " rows")) * int(_number(cols, key + " cols")):
            raise ValueError(f"{key} has {a.size} numbers in data for {rows}x{cols}")
    else:
        a = _floats(x, key)
    if shape is None:
        return a.reshape(-1)
    if a.size != int(np.prod(shape)):
        raise ValueError(f"{key} has {a.size} numbers; give {forms}")
    return a.reshape(shape)


def _transform(data, key):
    T = _matrix(data[key], key, (4, 4), "a 4x4 matrix or 16 numbers")
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError(f"the last row of {key} is {T[3].tolist()}, not [0, 0, 0, 1]")
    return T


def _yaml(text, name="calibration.yaml"):
    """The YAML text loaded with a safe loader (ruamel.yaml, else PyYAML), after _plain_yaml has walked
    its parser events."""
    try:
        from ruamel.yaml import YAML, events
    except ImportError:
        try:
            import yaml
        except ImportError:
            raise ValueError("reading YAML needs ruamel.yaml or PyYAML; give the calibration as JSON") from None
        _plain_yaml(yaml.parse(text, Loader=yaml.SafeLoader), yaml.events, name)
        return yaml.safe_load(text)
    _plain_yaml(YAML(typ="safe", pure=True).parse(text), events, name)
    return YAML(typ="safe", pure=True).load(text)


def _plain_yaml(stream, events, name):
    """ValueError at the first anchor or alias (& and *) in a YAML parser event stream, or at nesting
    deeper than MAX_YAML_DEPTH; it stops there, before the parser reads on. A safe loader keeps an
    alias as a second reference to the same list, so a few hundred bytes of nested aliases can stand
    for billions of numbers once np.asarray walks them. And the pure-Python parser takes minutes over
    a 64 KB file of nested brackets. Calibration files need neither."""
    depth = 0
    for e in stream:
        if getattr(e, "anchor", None) is not None:
            raise ValueError(f"{name} uses YAML anchors or aliases (& and *); write the values out")
        if isinstance(e, events.CollectionStartEvent):
            depth += 1
            if depth > MAX_YAML_DEPTH:
                raise ValueError(f"{name} nests lists and maps deeper than {MAX_YAML_DEPTH} levels")
        elif isinstance(e, events.CollectionEndEvent):
            depth -= 1


def read_calibration(text, name="calibration.yaml"):
    """The LiDAR-camera calibration in a YAML or JSON file's text (name gives the format: .json is
    JSON, anything else YAML) as a dict, or ValueError with the reason.

    Keys: T_cam_lidar (4x4, meters, x_cam = T x_lidar; a nested list, 16 numbers or OpenCV's
    {rows, cols, data}), camera_axes (optical, the default, or body: x forward, y left, z up), and
    optional time_offset_s, K (3x3, 9 numbers or fx, fy, cx, cy), D, distortion_model (plumb_bob,
    rational_polynomial, equidistant) and image_size [w, h]. time_offset_s is the LiDAR clock minus the
    camera clock: t_lidar = t_cam + time_offset_s, so it is added to a camera stamp before the scans
    are matched to it. It is None when the file has no time_offset_s (or a null one): 0 is an offset
    that the file gives, which says the two header clocks are one. In place of T_cam_lidar: T_lidar_cam
    (inverted) or FAST-LIVO2's Rcl (9 numbers) and Pcl (3), at the top level or under extrin_calib. A
    ROS 2 parameter file (one node with ros__parameters) is read from inside. ROS camera_info's
    camera_matrix, distortion_coefficients, image_width and image_height also work. OpenCV FileStorage
    files (a %YAML:1.0 header, !!opencv-matrix tags) are read as YAML; their distortion_coefficients
    without a distortion_model are plumb_bob (4 or 5 numbers) or rational_polynomial (8, 12 or 14).

    The rotation must be orthonormal within ORTHONORMAL_TOL (it is then re-orthonormalised by SVD)
    with det > 0, the translation at most MAX_TRANSLATION (2 m) and K positive. The result's T_cam_lidar maps LiDAR points to the
    camera's optical axes (x right, y down, z forward): body axes get R_OPT_LINK on the left. YAML anchors
    and aliases, and lists and maps nested deeper than MAX_YAML_DEPTH, are refused (_plain_yaml)."""
    if isinstance(text, bytes):
        try:
            text = text.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise ValueError(f"{name} is not a text file") from None
    text = text[1:] if text.startswith("\ufeff") else text  # a byte order mark, as Windows editors write
    opencv = re.search(r"(?m)^%YAML", text) is not None
    if Path(name).suffix.lower() == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"{name} is not valid JSON: {e}") from None
        except RecursionError:
            raise ValueError(f"{name} nests lists and maps too deeply") from None
    else:
        text = re.sub(r"(?m)^%YAML.*$", "", text).replace("!!opencv-matrix", "")
        try:
            data = _yaml(text, name)
        except ValueError:
            raise
        except Exception as e:
            problem, mark = getattr(e, "problem", None), getattr(e, "problem_mark", None)
            if problem:  # the parser's own words and the line, not its context ("while parsing ...")
                msg = problem + (f" (line {mark.line + 1})" if getattr(mark, "line", None) is not None else "")
            else:
                lines = str(e).strip().splitlines()
                msg = lines[0] if lines else type(e).__name__
            raise ValueError(f"{name} is not valid YAML: {msg}") from None
    if not isinstance(data, dict):
        raise ValueError(f"{name} has no keys; give T_cam_lidar (a 4x4 matrix, x_cam = T x_lidar)")
    if len(data) == 1:  # a ROS 2 parameter file: {node: {ros__parameters: {...}}}
        (node,) = data.values()
        if isinstance(node, dict) and isinstance(node.get("ros__parameters"), dict):
            data = node["ros__parameters"]
    kalibr = [k for k in ("intrinsics", "distortion_coeffs") if k in data]
    if kalibr:
        raise ValueError(f"{name} has {' and '.join(kalibr)} (Kalibr's names); give K (fx, fy, cx, cy) and D "
                         "with distortion_model instead")
    ext = data.get("extrin_calib") if isinstance(data.get("extrin_calib"), dict) else {}
    livo = data if "Rcl" in data or "Pcl" in data else ext if "Rcl" in ext or "Pcl" in ext else None
    given = [k for k in ("T_cam_lidar", "T_lidar_cam") if k in data] + (["Rcl, Pcl"] if livo is not None else [])
    if not given:
        raise ValueError(f"{name} has no T_cam_lidar, T_lidar_cam or Rcl and Pcl")
    if len(given) > 1:
        raise ValueError(f"{name} gives {' and '.join(given)}; give one of them")
    if given[0] == "Rcl, Pcl":
        if "Rcl" not in livo or "Pcl" not in livo:
            raise ValueError("FAST-LIVO2's extrinsics need both Rcl and Pcl")
        T = np.eye(4)
        T[:3, :3] = _matrix(livo["Rcl"], "Rcl", (3, 3), "9 numbers")
        T[:3, 3] = _matrix(livo["Pcl"], "Pcl", (3,), "3 numbers")
        key = "Rcl"
    else:
        key = given[0]
        T = _transform(data, key)
    R = T[:3, :3]
    err = float(np.linalg.norm(R @ R.T - np.eye(3)))
    if err >= ORTHONORMAL_TOL:
        raise ValueError(f"the rotation in {key} is not orthonormal (|R R^T - I| = {err:.2g}); check the numbers")
    if np.linalg.det(R) <= 0:
        raise ValueError(f"the rotation in {key} is a reflection (det < 0); check the axes")
    U, _, Vt = np.linalg.svd(R)
    T[:3, :3] = U @ Vt
    if key == "T_lidar_cam":
        R, t = T[:3, :3].copy(), T[:3, 3].copy()
        T[:3, :3], T[:3, 3] = R.T, -R.T @ t
    n = float(np.linalg.norm(T[:3, 3]))
    if n > MAX_TRANSLATION:
        raise ValueError(f"the translation is {n:.3g} m: is it in millimeters?")
    axes = data.get("camera_axes", "optical")
    if axes not in ("optical", "body"):
        raise ValueError(f"camera_axes is {axes!r}; use optical (x right, y down, z forward) "
                         "or body (x forward, y left, z up)")
    if axes == "body":
        T[:3, :] = R_OPT_LINK @ T[:3, :]

    offset = data.get("time_offset_s")
    offset = None if offset is None else _number(offset, "time_offset_s")
    K = data.get("K", data.get("camera_matrix"))
    if K is not None:
        k = _matrix(K, "K")
        if k.size == 4:
            k = np.array([k[0], 0, k[2], 0, k[1], k[3], 0, 0, 1])
        if k.size != 9:
            raise ValueError(f"K has {k.size} numbers; give a 3x3 matrix, 9 numbers or fx, fy, cx, cy")
        K = k.reshape(3, 3)
        if not (K[0, 0] > 0 and K[1, 1] > 0 and K[0, 2] > 0 and K[1, 2] > 0) or not np.allclose(K[2], [0, 0, 1]):
            raise ValueError(f"K must have positive fx, fy, cx and cy and a last row [0, 0, 1], not {K.tolist()}")
        K = K.tolist()
    D = data.get("D", data.get("distortion_coefficients"))
    if D is not None:
        D = _matrix(D, "D").tolist()
    model = data.get("distortion_model")
    if model is None and opencv and D is not None and "D" not in data:  # OpenCV's calibrateCamera
        model = {4: "plumb_bob", 5: "plumb_bob", 8: "rational_polynomial", 12: "rational_polynomial",
                 14: "rational_polynomial"}.get(len(D))
    model = distortion(D, model)[0]
    if model is not None and D is None:
        raise ValueError(f"distortion_model is {model} but D is missing; give its coefficients as D")
    size = data.get("image_size")
    if size is None and "image_width" in data and "image_height" in data:
        size = [data["image_width"], data["image_height"]]
    if size is not None:
        a = _floats(size, "image_size").reshape(-1)
        if a.size != 2 or (a <= 0).any() or (a != np.round(a)).any():
            raise ValueError(f"image_size is [width, height] in pixels, not {size}")
        size = [int(a[0]), int(a[1])]
    return {"T_cam_lidar": T.tolist(), "camera_axes": axes, "time_offset_s": offset, "K": K, "D": D,
            "distortion_model": model, "image_size": size, "given_as": key}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("estimate", help="the metric scale of a pool from LiDAR samples")
    e.add_argument("--pool", required=True, help="pool folder (mvsam3d prep and da3 ran on it)")
    e.add_argument("--samples", required=True, help=".npz with an (N, 3) [u, v, z] array per pool frame name")
    e.add_argument("--out", required=True, help="the lidar.json record")
    e.add_argument("--check", help="folder for the check images")
    a = ap.parse_args(argv)
    rec = estimate(a.pool, a.samples, a.check)
    write_json(a.out, rec, indent=1)
    lines = summary(rec)
    print("\n".join(lines[:-1]))
    if not rec["ok"]:
        raise SystemExit(lines[-1])
    print(lines[-1])


if __name__ == "__main__":
    main()
