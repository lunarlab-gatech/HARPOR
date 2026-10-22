import numpy as np
import torch

from gator import frames as F
from gator.models.correspondence import CorrespondenceEmbed, fourier, patch_centers, patch_rays
from gator.models.modality_mixer import ModalityMixer
from gator.models.pointmap_encoder import PointmapEncoder
from gator.models.text_adapters import SiglipAdapter, T5GemmaAdapter

torch.manual_seed(0)
rng = np.random.default_rng(0)

# Small crops keep the tests fast: 32 px with 8 px patches gives a 4 x 4 patch grid.
SIZE, PATCH, WIDTH = 32, 8, 64
P = (SIZE // PATCH) ** 2
nan, inf = float("nan"), float("inf")


def random_rotation():
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def cameras(B, V):
    """Float32 crop intrinsics, G-frame world-to-camera matrices and anchors."""
    K = torch.tensor([[40.0, 0, 15.5], [0, 42.0, 16.0], [0, 0, 1]]).expand(B, V, 3, 3)
    w2c = [[F.w2c_in_g(F.rigid(random_rotation(), rng.normal(size=3))) for _ in range(V)] for _ in range(B)]
    return K, torch.tensor(np.array(w2c), dtype=torch.float32), torch.randn(B, 3), torch.rand(B) + 0.5


def small_modules(gates=False):
    enc = PointmapEncoder(out_dim=WIDTH, dim=32, image_size=SIZE, patch_size=PATCH)
    corr = CorrespondenceEmbed(WIDTH, SIZE, PATCH)
    mix = ModalityMixer(WIDTH)
    if gates:
        with torch.no_grad():
            corr.gate.normal_()
            mix.gate.normal_()
    return enc, corr, mix


def observe(enc, corr, mix, rgb, mask, xyz, cams, view_mask):
    return mix(torch.stack([rgb, mask, enc(xyz)], -2), corr(*cams, view_mask), view_mask)


def test_default_grid_matches_dinov3():
    enc = PointmapEncoder()
    assert enc.grid == 24 and enc.proj.out_features <= 256
    assert enc.pos_window.shape == (1, 257, 256) and enc.pos_grid.shape == (1, 576, 256)
    xyz = torch.randn(1, 300, 400, 3)
    xyz[0, :50] = nan
    with torch.no_grad():
        out = enc(xyz)
    assert out.shape == (1, 576, 1024) and torch.isfinite(out).all()
    assert CorrespondenceEmbed().gamma.shape == (576, 24)


def test_shapes_for_1_to_4_views_with_padding():
    enc, corr, mix = small_modules(gates=True)
    B, V = 2, 4
    for n in range(1, 5):
        view_mask = torch.arange(V) < torch.tensor([[n], [V]])
        K, w2c, t, s = cameras(B, V)
        K, w2c = K.clone(), w2c.clone()
        K[~view_mask], w2c[~view_mask] = nan, nan
        xyz = torch.randn(B, V, 40, 40, 3)
        xyz[~view_mask] = nan
        rgb, mask = torch.randn(B, V, P, WIDTH), torch.randn(B, V, P, WIDTH)
        rgb[~view_mask] = nan
        out = observe(enc, corr, mix, rgb, mask, xyz, (K, w2c, t, s), view_mask)
        assert out.shape == (B, V, P, 3, WIDTH)
        assert torch.isfinite(out).all()
        assert (out[~view_mask] == 0).all() and (out[view_mask].abs().sum(-1) > 0).all()
        out.square().sum().backward()
        for m in (enc, corr, mix):
            assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_invalid_pixels_use_invalid_token():
    enc = small_modules()[0]
    xyz = torch.rand(1, SIZE, SIZE, 3) - 0.5
    valid = torch.ones(1, SIZE, SIZE, dtype=torch.bool)
    xyz[0, 0, 0] = nan
    xyz[0, 3, 5, 2] = inf
    xyz[0, 9, 9, 1] = -inf
    valid[0, 20, 7] = False
    bad = ~valid
    bad[0, 0, 0] = bad[0, 3, 5] = bad[0, 9, 9] = True

    feat = enc.pixels(xyz, valid)
    assert torch.equal(feat[bad], enc.invalid_token.expand(4, -1))
    assert torch.allclose(feat[~bad], enc.proj(xyz[~bad] / enc.clamp))
    out = enc(xyz, valid)
    assert torch.isfinite(out).all()

    # Values under invalid pixels do not matter, and a fully invalid view is still finite.
    xyz2 = xyz.clone()
    xyz2[bad] = 123.0
    assert torch.equal(enc(xyz2, ~bad), out)
    assert torch.isfinite(enc(torch.full((1, SIZE, SIZE, 3), nan))).all()


def test_z_at_or_below_minus_one_gives_no_nan():
    enc = small_modules()[0]
    xyz = torch.rand(1, SIZE, SIZE, 3) - 0.5
    xyz[..., 2] = -torch.linspace(1.0, 6.0, SIZE)
    feat = enc.pixels(xyz)
    x, y = xyz[0, 0, 0, :2] / 4  # the linear remap divides by clamp = 4, so z = -1 maps to -0.25
    assert torch.allclose(feat[0, 0, 0], enc.proj(torch.stack([x, y, torch.tensor(-0.25)])))
    assert torch.allclose(feat, enc.proj(xyz.clamp(-4, 4) / 4))
    assert torch.isfinite(enc(xyz)).all()


def test_nearest_resize():
    enc = small_modules()[0]
    xyz = torch.randn(1, SIZE, SIZE, 3)
    big = xyz.repeat_interleave(3, 1).repeat_interleave(3, 2)
    assert torch.equal(enc(big), enc(xyz))
    valid = torch.ones(1, 3 * SIZE, 3 * SIZE, dtype=torch.bool)
    valid[0, 15:18, 18:21] = False
    feat = enc.pixels(big, valid)
    assert torch.equal(feat[0, 5, 6], enc.invalid_token) and torch.equal(feat[0, 5, 7], enc.pixels(xyz)[0, 5, 7])

    # Output pixel j takes the input pixel nearest to its center (j + 0.5) n / SIZE - 0.5 (no ties at n = 20, 48).
    for n in (20, 48):
        idx = 0.05 * torch.arange(n, dtype=torch.float32)
        xyz = torch.stack(torch.meshgrid(idx, idx, indexing="ij") + (torch.zeros(n, n),), -1)[None]
        src = torch.tensor(np.rint((np.arange(SIZE) + 0.5) * n / SIZE - 0.5), dtype=torch.long)
        assert torch.allclose(enc.pixels(xyz), enc.proj(xyz[:, src][:, :, src] / enc.clamp), atol=1e-6)


def test_pointmap_windows_are_local():
    enc = small_modules()[0]
    xyz = 0.3 * torch.randn(1, SIZE, SIZE, 3)
    out = enc(xyz)
    xyz[0, PATCH:2 * PATCH, 2 * PATCH:3 * PATCH] += 0.5
    changed = (enc(xyz) - out).abs().amax(-1)[0] > 0
    assert changed.nonzero().flatten().tolist() == [1 * (SIZE // PATCH) + 2]


def test_fourier_and_patch_centers():
    g = fourier(torch.tensor([[0.5, -0.25]], dtype=torch.float64), 2)
    r = np.sqrt(0.5)
    assert torch.allclose(g, torch.tensor([[1, 0, 0, -1, -r, -1, r, 0]], dtype=torch.float64), atol=1e-12)
    c = patch_centers(384, 16)
    assert torch.equal(c[[0, 1, 24, 575]], torch.tensor([[7.5, 7.5], [23.5, 7.5], [7.5, 23.5], [375.5, 375.5]]))
    u0 = 8 / 384 * 2 - 1
    assert torch.allclose(CorrespondenceEmbed().gamma[0], fourier(torch.tensor([u0, u0]), 6))


def test_principal_ray_is_camera_forward_in_g():
    K = np.array([[300.0, 0, 199.5], [0, 320.0, 199.5], [0, 0, 1]])  # center of patch (12, 12)
    t, s = rng.normal(size=3), 0.4
    for w2c_w in (np.eye(4), F.rigid(random_rotation(), rng.normal(size=3))):
        rays = patch_rays(torch.tensor(K)[None], torch.tensor(F.w2c_in_g(w2c_w))[None], torch.tensor(t), s)
        assert rays.shape == (1, 576, 6)
        d = rays[0, 12 * 24 + 12, :3].numpy()
        assert np.allclose(d, F.w_to_g(w2c_w[2, :3]))  # camera +z axis, from W to G
        if np.allclose(w2c_w, np.eye(4)):
            assert np.allclose(d, [0, 1, 0])


def test_rays_pass_through_patch_centers():
    K0 = np.array([[500.0, 0, 320], [0, 510.0, 240], [0, 0, 1]])
    K = F.crop_intrinsics(K0, (200, 100, 456, 356), (384, 384))
    w2c = F.w2c_in_g(F.rigid(random_rotation(), rng.normal(size=3)))
    t, s = rng.normal(size=3), 0.7
    rays = patch_rays(torch.tensor(K)[None], torch.tensor(w2c)[None], torch.tensor(t), s)[0].numpy()
    d, m = rays[:, :3], rays[:, 3:]
    p = F.invert_rigid(w2c)[:3, 3] + 2.5 * d
    uv, z = F.project(p, K, w2c)
    assert np.allclose(uv, patch_centers().numpy()) and (z > 0).all()
    assert np.allclose(np.linalg.norm(d, axis=-1), 1)
    assert np.allclose(m, np.cross(F.normalize(p, t, s), d))  # the moment of any point on the ray

    # The ray direction is the average over the patch of the per-pixel rays at depth 1.
    pix, _ = F.unproject_depth(np.ones((384, 384)), K, w2c)
    mean = (pix - F.invert_rigid(w2c)[:3, 3]).reshape(24, 16, 24, 16, 3).mean((1, 3)).reshape(-1, 3)
    assert np.allclose(mean / np.linalg.norm(mean, axis=-1, keepdims=True), d)


def test_zero_gates_are_identity_at_init():
    _, corr, mix = small_modules()
    view_mask = torch.tensor([[True, True, False], [True, True, True]])
    a = corr(*cameras(2, 3), view_mask)
    assert a.shape == (2, 3, P, WIDTH) and torch.equal(a, torch.zeros_like(a))
    h = torch.randn(2, 3, P, 3, WIDTH)
    out = mix(h, a, view_mask)
    assert torch.equal(out[view_mask], h[view_mask]) and (out[~view_mask] == 0).all()
    out.square().sum().backward()
    assert corr.gate.grad.abs().sum() > 0 and mix.gate.grad.abs().sum() > 0


def test_padded_views_may_hold_any_camera():
    """Zero, singular or NaN cameras on padded views change nothing and keep the gradients finite."""
    _, corr, mix = small_modules(gates=True)
    K, w2c, t, s = cameras(2, 3)
    view_mask = torch.tensor([[True, False, False], [True, True, False]])
    h = torch.randn(2, 3, P, 3, WIDTH)
    ref = mix(h, corr(K, w2c, t, s, view_mask), view_mask)
    for pad_K, pad_w2c in ((0.0, 0.0), (torch.eye(3), 0.0), (nan, nan)):
        K2, w2 = K.clone(), w2c.clone()
        K2[~view_mask], w2[~view_mask] = pad_K, pad_w2c
        corr.zero_grad()
        out = mix(h, corr(K2, w2, t, s, view_mask), view_mask)
        out.square().sum().backward()
        assert torch.allclose(out, ref, atol=1e-6)
        assert all(torch.isfinite(p.grad).all() for p in corr.parameters())


def test_mixer_mixes_within_one_patch():
    mix = small_modules(gates=True)[2]
    h, a = torch.randn(1, 2, P, 3, WIDTH), torch.randn(1, 2, P, WIDTH)
    both = torch.ones(1, 2, dtype=torch.bool)
    out = mix(h, a, both)
    h2 = h.clone()
    h2[0, 0, 5, 1] += 1.0  # mask token of patch 5 in view 0
    diff = (mix(h2, a, both) - out).abs().amax(-1)
    assert (diff[0, 0, 5] > 0).all()
    diff[0, 0, 5] = 0
    assert (diff == 0).all()

    h3 = h.clone()
    h3[0, 1] = nan
    out3 = mix(h3, a, torch.tensor([[True, False]]))
    assert torch.allclose(out3[0, 0], out[0, 0], atol=1e-6) and (out3[0, 1] == 0).all()


def test_permuting_views_permutes_outputs():
    enc, corr, mix = small_modules(gates=True)
    B, V = 2, 4
    rgb, mask = torch.randn(B, V, P, WIDTH), torch.randn(B, V, P, WIDTH)
    xyz = torch.randn(B, V, SIZE, SIZE, 3)
    K, w2c, t, s = cameras(B, V)
    view_mask = torch.tensor([[True, True, True, False], [True, True, True, True]])
    out = observe(enc, corr, mix, rgb, mask, xyz, (K, w2c, t, s), view_mask)
    perm = torch.tensor([2, 0, 3, 1])
    out_p = observe(enc, corr, mix, rgb[:, perm], mask[:, perm], xyz[:, perm], (K[:, perm], w2c[:, perm], t, s),
                    view_mask[:, perm])
    assert torch.allclose(out_p, out[:, perm], atol=1e-5)
    assert not torch.allclose(out[:, 0], out[:, 1])


def test_siglip_adapter_adds_zero_at_init_and_for_missing_prompts():
    ad = SiglipAdapter()
    emb, present = torch.randn(3, 1152), torch.tensor([True, False, True])
    out = ad(emb, present)
    assert out.shape == (3, 1536) and torch.equal(out, torch.zeros_like(out))
    with torch.no_grad():
        ad.net[-1].weight.normal_(std=0.02)
    out = ad(emb, present)
    assert (out[1] == 0).all() and (out[[0, 2]].abs().sum(-1) > 0).all()


def test_t5gemma_adapter_ignores_padding():
    ad = T5GemmaAdapter()
    tokens = torch.randn(2, 8, 1152)
    mask = torch.arange(8) < torch.tensor([[5], [8]])
    out = ad(tokens, mask)
    assert out.shape == (2, 8, 1024) and (out[0, 5:] == 0).all()
    alone = ad(tokens[:1, :5], torch.ones(1, 5, dtype=torch.bool))
    assert torch.allclose(out[0, :5], alone[0], atol=1e-5)
    tokens[0, 5:] = 1e3
    assert torch.allclose(ad(tokens, mask)[0], out[0], atol=1e-5)
