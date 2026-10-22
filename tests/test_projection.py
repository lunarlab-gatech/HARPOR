import numpy as np
import torch

from gator import frames as F
from gator.models import projection as P

rng = np.random.default_rng(0)
CROP = (96, 64)  # (w, h), not square, so swapped axes show up
K = np.array([[80.0, 0, 46.0], [0, 76.0, 33.2], [0, 0, 1]])
T_W = np.array([0.2, -0.1, 4.0])  # anchor center in W, in front of view 0
S = 0.8


def tt(x):
    return torch.tensor(np.array(x, dtype=np.float64))


def look_at(eye, target):
    """OpenCV world-to-camera matrix in W, with W's +y (down) as image down."""
    z = (target - eye) / np.linalg.norm(target - eye)
    x = np.cross([0.0, 1.0, 0.0], z)
    x /= np.linalg.norm(x)
    R = np.stack([x, np.cross(z, x), z])
    return F.rigid(R, -R @ eye)


def scene(n_views=3, n=50):
    """View 0 is the identity (W is view 0). The other views look at the anchor from around it."""
    dirs = [(1.0, -0.3, -1.0), (-1.0, -0.5, -0.4), (0.3, -0.8, 1.0)]
    w2c_w = [np.eye(4)] + [look_at(T_W + 4 * np.array(d) / np.linalg.norm(d), T_W) for d in dirs]
    w2c_g = F.w2c_in_g(np.stack(w2c_w[:n_views]))
    q = rng.uniform(-0.5, 0.5, size=(n, 3))
    return q, F.w_to_g(T_W), w2c_g


def coord_map(h, w):
    """(2, h, w) map whose value at each map pixel is the crop pixel coordinate (x, y) of its center."""
    x = (np.arange(w) + 0.5) * CROP[0] / w - 0.5
    y = (np.arange(h) + 0.5) * CROP[1] / h - 0.5
    return np.stack(np.meshgrid(x, y, indexing="xy"))


def const_maps(values, c=2, lr=(4, 6), hr=(64, 96)):
    v = tt(values)[:, None, None, None]
    return v * torch.ones(len(values), c, *lr, dtype=torch.float64), -v * torch.ones(len(values), c, *hr, dtype=torch.float64)


def run(q, t, w2c, feat_lr, feat_hr, view_mask=None, s=S):
    Ks = tt(np.broadcast_to(K, (len(w2c), 3, 3)))
    return P.gator_proj(tt(q), tt(t), s, Ks, tt(w2c), CROP, feat_lr, feat_hr, view_mask)


def test_project_matches_frames():
    q, t, w2c = scene()
    p_g = F.denormalize(q, t, S)
    uv, z = P.project(tt(p_g), tt(np.broadcast_to(K, (3, 3, 3))), tt(w2c))
    for v in range(3):
        uv_ref, z_ref = F.project(p_g, K, w2c[v])
        assert np.allclose(uv[v].numpy(), uv_ref) and np.allclose(z[v].numpy(), z_ref)
    # G-frame cameras on G points give the same pixels as W cameras on W points.
    uv_w, _ = F.project(F.g_to_w(p_g), K, np.eye(4))
    assert np.allclose(uv[0].numpy(), uv_w)


def test_samples_equal_projected_uv():
    q, t, w2c = scene()
    p_g = F.denormalize(q, t, S)
    uv_ref = np.stack([F.project(p_g, K, w2c[v])[0] for v in range(3)])
    # Bilinear samples of a linear map are exact between the outermost patch centers (7.5 px in).
    assert (uv_ref >= 7.5).all() and (uv_ref[..., 0] <= CROP[0] - 8.5).all() and (uv_ref[..., 1] <= CROP[1] - 8.5).all()
    feat_lr = tt(np.stack([coord_map(4, 6)] * 3))
    feat_hr = tt(np.stack([coord_map(64, 96)] * 3))
    f, count = run(q, t, w2c, feat_lr, feat_hr)
    assert (count == 3).all()
    mean_uv = uv_ref.mean(0)
    assert np.allclose(f[:, :2].numpy(), mean_uv) and np.allclose(f[:, 2:].numpy(), mean_uv)
    f0, _ = run(q, t, w2c[:1], feat_lr[:1], feat_hr[:1])
    assert np.allclose(f0[:, :2].numpy(), uv_ref[0]) and np.allclose(f0[:, 2:].numpy(), uv_ref[0])


def opposite_cameras():
    """View 0 looks along +z_W from the W origin, view 1 along -z_W from the same point."""
    back = F.rigid(np.diag([-1.0, 1.0, -1.0]), np.zeros(3))
    return F.w2c_in_g(np.stack([np.eye(4), back]))


def to_q(p_w, t=np.zeros(3), s=1.0):
    return F.normalize(F.w_to_g(np.asarray(p_w, dtype=float)), t, s)


def test_behind_and_outside_are_excluded():
    p_w = [[0, 0, 5.0],      # in front of view 0, behind view 1
           [0, 0, -5.0],     # behind view 0 but lands on its principal point; in front of view 1
           [100.0, 0, 5.0]]  # in front of view 0 but far outside its crop; behind view 1
    f, count = run(to_q(p_w), np.zeros(3), opposite_cameras(), *const_maps([1.0, 2.0]), s=1.0)
    assert count.tolist() == [1, 1, 0]
    assert np.allclose(f.numpy(), [[1, 1, -1, -1], [2, 2, -2, -2], [0, 0, 0, 0]])


def test_crop_edges_follow_half_pixel_convention():
    w, h = CROP
    u = np.array([-0.45, -0.55, w - 0.55, w - 0.45, 10.0, 10.0, 10.0, 10.0])
    v = np.array([10.0, 10.0, 10.0, 10.0, -0.45, -0.55, h - 0.55, h - 0.45])
    p_w = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    f, count = run(to_q(p_w), np.zeros(3), opposite_cameras()[:1], *const_maps([1.0]), s=1.0)
    assert count.tolist() == [1, 0, 1, 0, 1, 0, 1, 0]
    # Border padding: a sample in the outer half-pixel band keeps the edge value and does not fade.
    assert np.allclose(f.numpy(), np.array([[1, 1, -1, -1], [0] * 4] * 4))


def test_samples_agree_with_crop_intrinsics():
    """Maps hold the full-image coordinate of each map pixel center, so samples give full-image uv."""
    K_full = np.array([[500.0, 0, 319.5], [0, 510.0, 239.5], [0, 0, 1]])
    box = (200.25, 150.5, 344.25, 246.5)  # 144 x 96 px, resized to CROP = 96 x 64
    K_crop = F.crop_intrinsics(K_full, box, CROP)
    uv_full = rng.uniform([box[0] + 13, box[1] + 13], [box[2] - 13, box[3] - 13], size=(20, 2))
    ray = np.concatenate([(uv_full - K_full[:2, 2]) / K_full[[0, 1], [0, 1]], np.ones((20, 1))], -1)
    p_w = ray * rng.uniform(1, 5, size=(20, 1))

    def full_coord_map(h, w):
        x = box[0] + (np.arange(w) + 0.5) * (box[2] - box[0]) / w - 0.5
        y = box[1] + (np.arange(h) + 0.5) * (box[3] - box[1]) / h - 0.5
        return np.stack(np.meshgrid(x, y, indexing="xy"))[None]

    f, count = P.gator_proj(tt(to_q(p_w)), tt(np.zeros(3)), 1.0, tt(K_crop[None]),
                            tt(F.w2c_in_g(np.eye(4)[None])), CROP, tt(full_coord_map(4, 6)),
                            tt(full_coord_map(64, 96)))
    assert (count == 1).all()
    assert np.allclose(f[:, :2].numpy(), uv_full) and np.allclose(f[:, 2:].numpy(), uv_full)


def test_voxel_seen_by_no_view_gets_zeros():
    p_w = [[0, 0, 5.0], [0, 50.0, 1.0], [-30.0, 0, 0.5]]  # only the first is seen by view 0
    w2c = opposite_cameras()[:1]
    w2c = np.concatenate([w2c, w2c])
    feat_lr, feat_hr = const_maps([3.0, 3.0])
    f, count = run(to_q(p_w), np.zeros(3), w2c, feat_lr, feat_hr, s=1.0)
    assert count.tolist() == [2, 0, 0]
    assert torch.isfinite(f).all()
    assert np.allclose(f.numpy(), [[3, 3, -3, -3], [0] * 4, [0] * 4])


def test_average_uses_only_valid_views():
    # Views 0 and 2 see the point, view 1 has it behind. The mean is (1 + 100) / 2, not 111 / 3.
    w2c = opposite_cameras()
    w2c = np.stack([w2c[0], w2c[1], w2c[0]])
    f, count = run(to_q([[0.1, 0.2, 3.0]]), np.zeros(3), w2c, *const_maps([1.0, 10.0, 100.0]), s=1.0)
    assert count.tolist() == [2]
    assert np.allclose(f.numpy(), [[50.5, 50.5, -50.5, -50.5]])


def test_padded_views_are_ignored_and_objects_batch():
    q, t, w2c = scene(n_views=3, n=20)
    feat_lr = tt(rng.normal(size=(3, 5, 4, 6)))
    feat_hr = tt(rng.normal(size=(3, 5, 64, 96)))
    f2, c2 = run(q, t, w2c[:2], feat_lr[:2], feat_hr[:2])
    assert (c2 == 2).all()
    # The padded view sees every voxel, and its maps are NaN, so any leak would show.
    pad_lr, pad_hr = feat_lr.clone(), feat_hr.clone()
    pad_lr[2], pad_hr[2] = float("nan"), float("nan")
    mask = torch.tensor([True, True, False])
    f, c = run(q, t, w2c, pad_lr, pad_hr, mask)
    assert torch.equal(c, c2) and torch.allclose(f, f2)

    # Object 0 uses all three views, object 1 has its third view padded, with its own anchor.
    q1, t1, s1 = rng.uniform(-0.5, 0.5, size=(20, 3)), F.w_to_g(T_W + [0.1, 0.05, -0.2]), 0.6
    f1, c1 = run(q1, t1, w2c[:2], feat_lr[:2], feat_hr[:2], s=s1)
    f3, c3 = run(q, t, w2c, feat_lr, feat_hr)
    fb, cb = P.gator_proj(tt(np.stack([q, q1])), tt(np.stack([t, t1])), tt([S, s1]),
                          tt(np.broadcast_to(K, (2, 3, 3, 3))), tt(np.stack([w2c, w2c])), CROP,
                          torch.stack([feat_lr, pad_lr]), torch.stack([feat_hr, pad_hr]),
                          torch.tensor([[True, True, True], [True, True, False]]))
    assert (c3 == 3).all() and torch.isnan(run(q, t, w2c, pad_lr, pad_hr)[0]).all()
    assert torch.equal(cb[0], c3) and torch.equal(cb[1], c1)
    assert torch.allclose(fb[0], f3) and torch.allclose(fb[1], f1)


def test_injection_matches_pixal3d_proj_linear():
    layer = P.ProjInjection()
    assert layer.proj_linear.weight.shape == (1536, 2048)  # Pixal3D proj_linear, [1536, 2048]
    small = P.ProjInjection(4, 3)
    h, f = torch.randn(5, 3), torch.randn(5, 4)
    assert torch.equal(small(h, f), h)  # zero start is an identity
    W, b = torch.randn(3, 4), torch.randn(3)
    pixal3d = {"blocks.7.cross_attn.proj_linear.weight": W, "blocks.7.cross_attn.proj_linear.bias": b}
    prefix = "blocks.7.cross_attn."
    small.load_state_dict({k[len(prefix):]: v for k, v in pixal3d.items()})
    assert torch.allclose(small(h, f), h + f @ W.T + b)
