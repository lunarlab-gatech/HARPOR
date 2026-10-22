"""Pointmap encoder for the structure flow (build_plan.md note 4).

Written from the description of SAM 3D's PointPatchEmbed in the build plan, not from SAM 3D code.
Pointmaps are anchor-normalized G-frame points q with shape (..., H, W, 3). The output has one
token per DINOv3 patch, in row-major patch order (384 px with 16 px patches gives 24 x 24).
"""

import torch
import torch.nn.functional as F
from torch import nn


class PointmapEncoder(nn.Module):
    def __init__(self, out_dim=1024, dim=256, image_size=384, patch_size=16, heads=16, mlp_ratio=2.0,
                 clamp=4.0):
        super().__init__()
        self.image_size, self.patch_size, self.grid = image_size, patch_size, image_size // patch_size
        self.clamp = clamp
        self.proj = nn.Linear(3, dim)
        self.invalid_token = nn.Parameter(0.02 * torch.randn(dim))
        self.cls_token = nn.Parameter(0.02 * torch.randn(1, 1, dim))
        self.pos_window = nn.Parameter(0.02 * torch.randn(1, 1 + patch_size**2, dim))
        self.pos_grid = nn.Parameter(0.02 * torch.randn(1, self.grid**2, dim))
        self.block = nn.TransformerEncoderLayer(dim, heads, int(dim * mlp_ratio), dropout=0.0, activation="gelu",
                                                layer_norm_eps=1e-6, batch_first=True, norm_first=True)
        self.out = nn.Sequential(nn.LayerNorm(dim, eps=1e-6), nn.Linear(dim, out_dim))

    def pixels(self, xyz, valid=None):
        """Per-pixel features (N, S, S, dim): nearest resize, invalid mask, linear remap, Linear(3, dim).

        A pixel is invalid if any coordinate is non-finite or valid (..., H, W) flags it. Invalid
        pixels get the learned invalid token instead of their projection.
        """
        x = xyz.reshape(-1, *xyz.shape[-3:]).permute(0, 3, 1, 2)
        ok = torch.isfinite(x).all(1, keepdim=True)
        if valid is not None:
            ok &= valid.reshape(-1, 1, *valid.shape[-2:])
        size = (self.image_size, self.image_size)
        x = F.interpolate(torch.where(ok, x, 0.0), size=size, mode="nearest-exact")
        ok = F.interpolate(ok.to(x.dtype), size=size, mode="nearest-exact") > 0.5
        x = x.clamp(-self.clamp, self.clamp) / self.clamp
        return torch.where(ok.permute(0, 2, 3, 1), self.proj(x.permute(0, 2, 3, 1)), self.invalid_token)

    def forward(self, xyz, valid=None):
        """Pointmaps (..., H, W, 3), optional valid flags (..., H, W) -> tokens (..., grid^2, out_dim)."""
        h = self.pixels(xyz, valid)
        N, D, n, p = h.shape[0], h.shape[-1], self.grid, self.patch_size
        h = h.reshape(N, n, p, n, p, D).transpose(2, 3).reshape(N * n * n, p * p, D)
        h = torch.cat([self.cls_token.expand(len(h), -1, -1), h], 1) + self.pos_window
        h = self.block(h)[:, 0].reshape(N, n * n, D) + self.pos_grid
        return self.out(h).reshape(*xyz.shape[:-3], n * n, -1)
