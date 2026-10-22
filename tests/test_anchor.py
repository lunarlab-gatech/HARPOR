import numpy as np
import pytest

from gator import frames as F
from gator.inference import anchor as A

BOX_C = np.array([0.35, 2.5, -0.4])  # G frame: in front of view 0 and below it
BOX_SIZE = np.array([0.6, 0.4, 0.3])
LO, HI = BOX_C - BOX_SIZE / 2, BOX_C + BOX_SIZE / 2
K = np.array([[160.0, 0, 63.5], [0, 160.0, 47.5], [0, 0, 1]])
SHAPE = (96, 128)


def look_at(eye_w, target_w):
    f = (target_w - eye_w) / np.linalg.norm(target_w - eye_w)
    up = np.array([0.0, -1.0, 0.0]) if abs(f[1]) < 0.9 else np.array([0.0, 0.0, 1.0])
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    R = np.stack([r, np.cross(f, r), f])
    return F.rigid(R, -R @ eye_w)


def rays_g(K, w2c, shape):
    """Camera center and per-pixel ray directions in G, scaled so the ray parameter is the depth."""
    v, u = np.mgrid[:shape[0], :shape[1]].astype(float)
    d_cam = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    R = w2c[:3, :3]
    return F.w_to_g(-R.T @ w2c[:3, 3]), F.w_to_g(d_cam @ R)


def render_box(K, w2c, shape):
    o, d = rays_g(K, w2c, shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        a, b = (LO - o) / d, (HI - o) / d
    near, far = np.minimum(a, b).max(-1), np.maximum(a, b).min(-1)
    return np.where((far >= near) & (near > 0), near, 0.0)


def render_sphere(c, r, K, w2c, shape):
    o, d = rays_g(K, w2c, shape)
    b = (d * (o - c)).sum(-1)
    a = (d * d).sum(-1)
    disc = b * b - a * ((o - c) @ (o - c) - r * r)
    return np.where(disc > 0, (-b - np.sqrt(np.maximum(disc, 0))) / a, 0.0)


def box_views():
    c_w = F.g_to_w(BOX_C)
    eyes = [c_w + 1.8 * np.array(e) / np.sqrt(3) for e in [(1, -1, -1), (-1, 1, 1)]]
    w2cs = [np.eye(4)] + [look_at(e, c_w) for e in eyes]
    depths = [render_box(K, T, SHAPE) for T in w2cs]
    return [d > 0 for d in depths], depths, [K] * 3, w2cs


def test_erode_strips_5px_but_not_at_image_border():
    m = np.zeros((40, 40), bool)
    m[10:30, 10:30] = True
    expected = np.zeros_like(m)
    expected[15:25, 15:25] = True
    assert np.array_equal(A.erode(m), expected)
    m = np.zeros((40, 40), bool)
    m[:, :20] = True
    expected = np.zeros_like(m)
    expected[:, :15] = True
    assert np.array_equal(A.erode(m), expected)


def test_box_anchor_is_exact():
    masks, depths, Ks, w2cs = box_views()
    assert np.allclose(w2cs[0], np.eye(4))
    t, s = A.anchor(masks, depths, Ks, w2cs, margin=1.25)
    assert np.allclose(t, BOX_C, atol=1e-9)
    assert np.isclose(s, 1.25 * BOX_SIZE.max())
    q = F.normalize(A.observed_points(masks, depths, Ks, w2cs), t, s)
    assert np.abs(q).max() <= 0.5 / 1.25 + 1e-9


def test_erosion_drops_boundary_pixels():
    masks, depths, Ks, w2cs = box_views()
    bad = [np.where(m & ~A.erode(m), 1.3 * d, d) for m, d in zip(masks, depths)]
    raw = np.concatenate([F.unproject_depth(d, k, T, m)[0] for m, d, k, T in zip(masks, bad, Ks, w2cs)])
    t_raw, _ = A.percentile_frame(F.w_to_g(raw), 1.0)
    assert not np.allclose(t_raw, BOX_C, atol=1e-3)  # the ring would move the anchor without erosion
    t, s = A.anchor(masks, bad, Ks, w2cs, margin=1.0)
    assert np.allclose(t, BOX_C, atol=1e-9) and np.isclose(s, BOX_SIZE.max())


def test_outliers_beyond_percentiles_are_ignored():
    masks, depths, Ks, w2cs = box_views()
    noisy = []
    for m, d in zip(masks, depths):
        v, u = np.nonzero(A.erode(m))
        d = d.copy()
        d[v[::200], u[::200]] *= 10.0  # flying pixels far behind the object
        noisy.append(d)
    p = A.observed_points(masks, noisy, Ks, w2cs)
    assert p[:, 1].max() > 10.0
    t, s = A.anchor(masks, noisy, Ks, w2cs, margin=1.0)
    assert np.allclose(t, BOX_C, atol=1e-9) and np.isclose(s, BOX_SIZE.max())


def test_sphere_points_and_symmetric_anchor():
    c, r, D = np.array([0.2, 3.0, -0.1]), 0.4, 2.0
    Kc = np.array([[100.0, 0, 31.5], [0, 100.0, 31.5], [0, 0, 1]])
    eyes = [c + sign * D * e for e in np.eye(3) for sign in (1, -1)]
    w2cs = [look_at(F.g_to_w(e), F.g_to_w(c)) for e in eyes]
    depths = [render_sphere(c, r, Kc, T, (64, 64)) for T in w2cs]
    masks = [d > 0 for d in depths]
    p = A.observed_points(masks, depths, [Kc] * 6, w2cs)
    assert np.allclose(np.linalg.norm(p - c, axis=1), r, atol=1e-9)
    t, s = A.anchor(masks, depths, [Kc] * 6, w2cs, margin=1.5)
    assert np.allclose(t, c, atol=1e-9)
    # Pixel samples crowd the caps that face each camera, so the box sits between the
    # uniform-surface value 0.96 r (Archimedes, next test) and r.
    assert 1.5 * 0.96 * 2 * r < s < 1.5 * 2 * r


def test_sphere_surface_percentile_box():
    # Archimedes: each coordinate of a uniform sphere-surface sample is uniform on [-r, r].
    n, r, c = 200000, 0.7, np.array([1.0, -2.0, 0.5])
    i = np.arange(n)
    z = 1 - (2 * i + 1) / n
    phi = i * np.pi * (3 - np.sqrt(5))
    p = c + r * np.stack([np.sqrt(1 - z * z) * np.cos(phi), np.sqrt(1 - z * z) * np.sin(phi), z], -1)
    t, s = A.percentile_frame(p, margin=1.1)
    assert np.allclose(t, c, atol=2e-3 * r)
    assert np.isclose(s, 1.1 * 2 * 0.96 * r, rtol=2e-3)


def test_no_valid_points_raises():
    m = np.zeros((40, 40), bool)
    m[10:30, 17:23] = True  # 6 px wide, gone after a 5 px erosion
    with pytest.raises(ValueError, match="no valid depth"):
        A.anchor([m], [np.ones((40, 40))], [K], [np.eye(4)], margin=1.0)
    m[10:30, 10:30] = True
    with pytest.raises(ValueError, match="no valid depth"):
        A.anchor([m], [np.full((40, 40), np.nan)], [K], [np.eye(4)], margin=1.0)


def test_anchor_without_extent_raises():
    y, x = np.mgrid[-5:6, -5:6]
    m = np.zeros((40, 40), bool)
    m[15:26, 15:26] = x * x + y * y <= 25  # erodes to its center pixel, so s would be 0
    assert A.erode(m).sum() == 1
    with pytest.raises(ValueError, match="no extent"):
        A.anchor([m], [np.ones((40, 40))], [K], [np.eye(4)], margin=1.0)


def test_pointmap_q():
    depth = np.full((4, 6), 2.0)
    depth[1, 2], depth[3, 0] = np.nan, 0.0
    Kp = np.array([[10.0, 0, 2.5], [0, 20.0, 1.5], [0, 0, 1]])
    w2c = F.rigid(np.eye(3), [0.0, 0.0, -1.0])  # camera center at (0, 0, 1) in W
    q = A.pointmap_q(depth, Kp, w2c, t=np.array([0.1, 1.0, 0.2]), s=0.5)
    assert q.shape == (4, 6, 3)
    assert np.isnan(q[1, 2]).all() and np.isnan(q[3, 0]).all()
    assert np.isfinite(q).all(-1).sum() == 22
    # pixel (u, v) = (4, 0): p_W = (0.3, -0.15, 3), p_G = (0.3, 3, 0.15)
    assert np.allclose(q[0, 4], [0.4, 4.0, -0.1])
