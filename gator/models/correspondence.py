"""Correspondence embedding a_vp = A(gamma(u_p), r_vp) (build_plan.md note 4, research.md 2.8).

gamma(u_p) is a Fourier encoding of the patch center, normalized to [-1, 1] over the crop. r_vp is
the Plücker ray (d, o x d) of the patch in the anchor-normalized G frame. Cameras are crop
intrinsics and G-frame world-to-camera matrices (frames.w2c_in_g). There is no view-index
embedding, so a view enters only through its camera.
"""

import torch
from torch import nn


def patch_centers(image_size=384, patch_size=16):
    """Pixel coordinates (n*n, 2) of the patch centers, row-major. Pixel centers sit at integers."""
    c = torch.arange(image_size // patch_size) * patch_size + (patch_size - 1) / 2
    v, u = torch.meshgrid(c, c, indexing="ij")
    return torch.stack([u, v], -1).reshape(-1, 2)


def patch_rays(K, w2c, t, s, image_size=384, patch_size=16):
    """Plücker rays (*B, V, n*n, 6) = (d, o x d) of the patches, in anchor-normalized G.

    K (*B, V, 3, 3), w2c (*B, V, 4, 4) in G, anchor t (*B, 3) and s (*B). Each ray passes through
    its patch center. Its direction is the mean of the per-pixel K^-1 [u, v, 1] over the patch,
    because that map is linear. Normalization has no rotation, so only the origin moves.
    """
    uv = patch_centers(image_size, patch_size).to(K)
    R_t = w2c[..., :3, :3].mT
    d = torch.cat([uv, torch.ones_like(uv[:, :1])], -1) @ (R_t @ torch.linalg.inv(K)).mT
    d = d / d.norm(dim=-1, keepdim=True)
    s = torch.as_tensor(s, dtype=K.dtype, device=K.device)
    o = (-(R_t @ w2c[..., :3, 3:])[..., 0] - t[..., None, :]) / s[..., None, None]
    return torch.cat([d, torch.cross(o[..., None, :].expand_as(d), d, dim=-1)], -1)


def fourier(u, octaves):
    """[sin(2^k pi u), cos(2^k pi u)] for k < octaves, per coordinate: (..., C) -> (..., 2 C octaves)."""
    x = u[..., None] * (torch.pi * 2.0 ** torch.arange(octaves, dtype=u.dtype, device=u.device))
    return torch.cat([x.sin(), x.cos()], -1).flatten(-2)


class CorrespondenceEmbed(nn.Module):
    """A: small MLP on [gamma(u_p), r_vp], scaled by a zero-initialized gate, so a_vp starts at zero."""

    def __init__(self, width=1024, image_size=384, patch_size=16, octaves=6):
        super().__init__()
        self.image_size, self.patch_size = image_size, patch_size
        u = (patch_centers(image_size, patch_size) + 0.5) / image_size * 2 - 1
        self.register_buffer("gamma", fourier(u, octaves), persistent=False)
        self.mlp = nn.Sequential(nn.Linear(4 * octaves + 6, width), nn.SiLU(), nn.Linear(width, width))
        self.gate = nn.Parameter(torch.zeros(width))

    def forward(self, K, w2c, t, s, view_mask):
        """Cameras as in patch_rays and view_mask (*B, V), False on padded views -> embeddings (*B, V, n*n, width).

        Padded views may hold any camera. They get the identity camera, so no NaN reaches the gradients.
        """
        m = view_mask[..., None, None]
        K = torch.where(m, K, torch.eye(3, dtype=K.dtype, device=K.device))
        w2c = torch.where(m, w2c, torch.eye(4, dtype=w2c.dtype, device=w2c.device))
        r = patch_rays(K, w2c, t, s, self.image_size, self.patch_size).to(self.gamma.dtype)
        return self.gate * self.mlp(torch.cat([self.gamma.expand(*r.shape[:-1], -1), r], -1))
