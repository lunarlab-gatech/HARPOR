import itertools

import numpy as np
import pytest
from scipy import ndimage

from gator import frames as F
from gator.inference import realign as R

rng = np.random.default_rng(0)
T_BOX, S_BOX = np.array([0.3, 2.0, -0.2]), 0.8
# Box faces pass through cell centers of the 64 grid in the frame (T_BOX, S_BOX), so the boundary
# voxel centers lie on the faces and that frame is the exact optimum of the fit.
BOX_LO = T_BOX + S_BOX * (np.array([12.5, 16.5, 20.5]) / 64 - 0.5)
BOX_HI = T_BOX + S_BOX * (np.array([52.5, 40.5, 40.5]) / 64 - 0.5)
SPHERE_C, SPHERE_R = np.array([0.31, 2.03, -0.22]), 0.37


def cell_edges(t, s, res):
    """Per-axis G-frame lower and upper edges (3, res) of the cells of a res grid in frame (t, s)."""
    e = np.asarray(t)[:, None] + s * (np.arange(res + 1) / res - 0.5)
    return e[:, :-1], e[:, 1:]


def outer_and(a):
    return a[0][:, None, None] & a[1][None, :, None] & a[2][None, None, :]


def outer_sum(a):
    return a[0][:, None, None] + a[1][None, :, None] + a[2][None, None, :]


def box_shell(t, s, res):
    """Cells crossed by the surface of the box [BOX_LO, BOX_HI]."""
    lo, hi = cell_edges(t, s, res)
    overlap = outer_and((lo <= BOX_HI[:, None]) & (hi >= BOX_LO[:, None]))
    inside = outer_and((lo > BOX_LO[:, None]) & (hi < BOX_HI[:, None]))
    return overlap & ~inside


def sphere_shell(t, s, res):
    """Cells crossed by the sphere surface: nearest point inside it, farthest corner outside it."""
    lo, hi = cell_edges(t, s, res)
    c = SPHERE_C[:, None]
    near = outer_sum(np.maximum(0, np.maximum(lo - c, c - hi)) ** 2)
    far = outer_sum(np.maximum(np.abs(lo - c), np.abs(hi - c)) ** 2)
    return (near <= SPHERE_R ** 2) & (far >= SPHERE_R ** 2)


def box_face(axis, value, n=60):
    a, b = [k for k in range(3) if k != axis]
    u, v = np.meshgrid(np.linspace(BOX_LO[a], BOX_HI[a], n), np.linspace(BOX_LO[b], BOX_HI[b], n))
    p = np.empty((n * n, 3))
    p[:, axis], p[:, a], p[:, b] = value, u.ravel(), v.ravel()
    return p


def front_hemisphere(n=20000):
    """Sphere-surface samples on the half that faces view 0 (G -y)."""
    i = np.arange(n)
    z = 1 - (2 * i + 1) / n
    phi = i * np.pi * (3 - np.sqrt(5))
    d = np.stack([np.sqrt(1 - z * z) * np.cos(phi), np.sqrt(1 - z * z) * np.sin(phi), z], -1)
    return SPHERE_C + SPHERE_R * d[d[:, 1] < 0]


def with_outliers(p, frac=0.05):
    r = np.random.default_rng(1)
    far = p[r.choice(len(p), int(frac * len(p)))] + r.uniform(0.3, 1.0, (int(frac * len(p)), 3))
    return np.concatenate([p, far])


def test_clean_keeps_large_parts_and_drops_floaters():
    occ = np.zeros((64,) * 3, bool)
    occ[10:30, 10:30, 10:30] = True
    occ[11:29, 11:29, 11:29] = False  # hollow box, 2168 voxels
    occ[30, 30, 30] = True  # touches the box only at a corner, so it is 26-connected to it
    occ[40:46, 40:46, 40:46] = True  # 216 voxels, more than 5% of the largest part
    expected = occ.copy()
    occ[55:57, 5:7, 50:52] = True  # 8-voxel floater
    occ[60, 60, 2] = True
    assert np.array_equal(R.clean_occupancy(occ), expected)


def test_clean_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        R.clean_occupancy(np.zeros((64,) * 3, bool))


def test_normalization_is_the_tight_box():
    lo, hi = np.array([-1.0, 0.5, 2.0]), np.array([0.2, 1.0, 2.3])
    p = np.concatenate([np.array(list(itertools.product(*zip(lo, hi)))), rng.uniform(lo, hi, (100, 3))])
    t, s = R.normalization(p)
    assert np.allclose(t, [-0.4, 0.75, 2.15])
    assert np.isclose(s, 1.2 / 0.99999)
    q = F.normalize(p, t, s)
    assert np.allclose([q[:, 0].min(), q[:, 0].max()], [-0.499995, 0.499995])
    assert np.allclose(q.max(0) - q.min(0), np.array([1.2, 0.5, 0.3]) * 0.99999 / 1.2)
    with pytest.raises(ValueError, match="no extent"):
        R.normalization(np.ones((3, 3)))


def test_revoxelize_in_same_frame_is_plain_max_pool():
    t, s = np.array([0.3, 2.0, -0.2]), 0.9
    occ = sphere_shell(t, s, 64)
    assert np.array_equal(R.revoxelize(occ, t, s, t, s), occ)
    assert np.array_equal(R.max_pool(occ), sphere_shell(t, s, 32))


def test_revoxelize_magnified_cell_fills_every_new_cell():
    # One 64 cell seen 4 times larger covers new cells 30 to 33 on each axis. Two samples per axis
    # would mark only 8 of the 64.
    occ = np.zeros((64,) * 3, bool)
    occ[32, 32, 32] = True
    expected = np.zeros_like(occ)
    expected[30:34, 30:34, 30:34] = True
    assert np.array_equal(R.revoxelize(occ, np.zeros(3), 1.0, np.full(3, 0.5 / 64), 0.25), expected)


def test_trilinear_value_and_gradient():
    f = rng.normal(size=(5, 6, 7))
    g = rng.uniform(0, 1, (50, 3)) * [4, 5, 6]
    val, grad = R.trilinear(f, g)
    assert np.allclose(val, ndimage.map_coordinates(f, g.T, order=1))
    h = 1e-6 * np.eye(3)
    num = np.stack([(R.trilinear(f, g + e)[0] - R.trilinear(f, g - e)[0]) / 2e-6 for e in h], 1)
    assert np.allclose(grad, num, atol=1e-6)
    # a plane of zeros at x = 2 in a distance field: the interpolant is |x - 2| with slope +-1
    d = np.abs(np.arange(5.0) - 2)[:, None, None] * np.ones((5, 6, 7))
    val, grad = R.trilinear(d, np.array([[2.4, 1.3, 5.5], [0.5, 4.0, 6.0]]))
    assert np.allclose(val, [0.4, 1.5]) and np.allclose(grad, [[1, 0, 0], [-1, 0, 0]])


SEEN3 = [(1, 0), (2, 1), (0, 0)]  # (axis, low or high) faces seen from front, top and left: G -y, +z, -x
SEEN5 = SEEN3 + [(0, 1), (1, 1)]  # both x faces and both y faces, so the points fix the scale


def box_faces(faces):
    return np.concatenate([box_face(ax, (BOX_LO, BOX_HI)[hi][ax]) for ax, hi in faces])


@pytest.mark.parametrize("faces, shift, scale", [
    (SEEN3, (3, -2, 2), 0.9),  # the structure is too small, so the seen faces stick out of it
    (SEEN3, (0, 0, 0), 0.85),
    (SEEN5, (3, -2, 2), 1.1),
    (SEEN5, (-4, 1, 3), 1.2)])
@pytest.mark.parametrize("outliers", [0.0, 0.15])
def test_fit_recovers_shift_and_scale(faces, shift, scale, outliers):
    cell = S_BOX / 64
    pts = with_outliers(box_faces(faces), outliers)
    t, s = R.fit(box_shell(T_BOX, S_BOX, 64), T_BOX + np.array(shift) * cell, scale * S_BOX, pts)
    assert np.allclose(t, T_BOX, atol=0.02 * cell)
    assert np.isclose(s, S_BOX, rtol=1e-3)


@pytest.mark.parametrize("shift", [(2, 0, 0), (4, 0, 0), (0, 2, 0), (2, 2, -2), (5, 5, -5)])
def test_fit_corrects_shift_without_changing_scale(shift):
    # Only the -x face constrains a shift along x, so most points already fit. A shift toward the
    # seen corner fits the points as well as a larger box would, and the fit must pick the shift.
    cell = S_BOX / 64
    t, s = R.fit(box_shell(T_BOX, S_BOX, 64), T_BOX + np.array(shift) * cell, S_BOX, box_faces(SEEN3))
    assert np.allclose(t, T_BOX, atol=0.02 * cell)
    assert np.isclose(s, S_BOX, rtol=1e-3)


def test_fit_keeps_scale_the_points_cannot_see():
    # Three faces seen and a structure 10% too large: any larger box that keeps the seen corner has
    # zero one-sided distance, so the fit aligns the corner and keeps the generated scale.
    cell = S_BOX / 64
    occ, t0 = box_shell(T_BOX, S_BOX, 64), T_BOX + np.array([3, -2, 2]) * cell
    t, s = R.fit(occ, t0, 1.1 * S_BOX, box_faces(SEEN3))
    corner = np.array([BOX_LO[0], BOX_LO[1], BOX_HI[2]])
    assert np.isclose(s, 1.1 * S_BOX, rtol=1e-3)
    assert np.allclose(F.denormalize(F.normalize(corner, T_BOX, S_BOX), t, s), corner, atol=0.02 * cell)


def test_fit_recovers_sphere_from_front_hemisphere():
    t_gen, s_gen = SPHERE_C, 1.25 * 2 * SPHERE_R
    cell = s_gen / 64
    pts = with_outliers(front_hemisphere(), 0.15)
    t, s = R.fit(sphere_shell(t_gen, s_gen, 64), t_gen + np.array([3, -2, 2]) * cell, 1.1 * s_gen, pts)
    assert np.allclose(t, t_gen, atol=0.2 * cell)
    assert np.isclose(s, s_gen, rtol=3e-3)


def test_fit_keeps_anchor_that_already_fits():
    # Depth noise of 0.3 cells along each face normal: fitting it gains far less than MIN_GAIN.
    pts = box_faces(SEEN3)
    n = len(pts) // 3
    for i, (ax, _) in enumerate(SEEN3):
        pts[i * n:(i + 1) * n, ax] += rng.normal(0, 0.3 * S_BOX / 64, n)
    t, s = R.fit(box_shell(T_BOX, S_BOX, 64), T_BOX, S_BOX, pts)
    assert t is T_BOX and s == S_BOX


def test_realign_cube_is_exact():
    # A one-cell hollow cube on cells 16 to 47, seen on all six faces through its boundary cell
    # centers, generated under an anchor 1.3 cells and 5% off. The new box spans the cells, so the
    # cube fills the new grid and its 32^3 shell is the outer layer of cells.
    t0, s0 = np.array([0.1, 1.5, -0.3]), 0.6
    occ = np.zeros((64,) * 3, bool)
    occ[16:48, 16:48, 16:48] = True
    occ[17:47, 17:47, 17:47] = False
    lo, hi = t0 + s0 * (16.5 / 64 - 0.5), t0 + s0 * (47.5 / 64 - 0.5)
    pts = []
    for ax in range(3):
        a, b = [k for k in range(3) if k != ax]
        u, v = np.meshgrid(np.linspace(lo[a], hi[a], 40), np.linspace(lo[b], hi[b], 40))
        for value in (lo[ax], hi[ax]):
            p = np.empty((u.size, 3))
            p[:, ax], p[:, a], p[:, b] = value, u.ravel(), v.ravel()
            pts.append(p)
    t, s, occ32 = R.realign(occ, t0 + 1.3 * s0 / 64, 0.95 * s0, np.concatenate(pts))
    assert np.allclose(t, t0, atol=1e-3 * s0 / 64)
    assert np.isclose(s, 0.5 * s0 / 0.99999, rtol=1e-4)
    expected = np.ones((32,) * 3, bool)
    expected[1:31, 1:31, 1:31] = False
    assert np.array_equal(occ32, expected)


def test_realign_matches_direct_voxelization():
    # The generator drew the sphere in the frame (t_gen, s_gen), but the anchor it was given differs.
    t_gen, s_gen = SPHERE_C + np.array([0.02, -0.03, 0.01]), 1.25 * 2 * SPHERE_R
    occ = sphere_shell(t_gen, s_gen, 64)
    occ[2:4, 2:4, 60:62] = True  # floater
    t_anchor, s_anchor = t_gen + np.array([0.03, 0.05, -0.02]), 0.9 * s_gen
    t, s, occ32 = R.realign(occ, t_anchor, s_anchor, with_outliers(front_hemisphere()))

    # The new box spans the occupied cells, so each side lies 0 to 1 old cell outside the sphere.
    old_cell = s_gen / 64
    assert np.allclose(t, SPHERE_C, atol=0.6 * old_cell)
    assert -0.2 * old_cell < s / R.BOX_MARGIN - 2 * SPHERE_R < 2.2 * old_cell
    direct = sphere_shell(t, s, 32)
    near = np.ones((3, 3, 3), bool)
    assert occ32.shape == (32, 32, 32)
    assert not (occ32 & ~ndimage.binary_dilation(direct, near)).any()
    assert not (direct & ~ndimage.binary_dilation(occ32, near)).any()
    assert (direct & occ32).sum() >= 0.98 * direct.sum()
