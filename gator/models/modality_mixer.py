"""Modality mixer, paper Eq. 1 (build_plan.md note 4).

Residual attention over the three tokens (RGB, mask, pointmap) of one patch. No attention runs
across patches or views. The gate starts at zero, so the mixer starts as the identity on h + a.
"""

import torch
from torch import nn


class ModalityMixer(nn.Module):
    def __init__(self, width=1024, heads=16):
        super().__init__()
        self.modality = nn.Parameter(0.02 * torch.randn(3, width))
        self.norm = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.gate = nn.Parameter(torch.zeros(width))

    def forward(self, h, a, view_mask):
        """Tokens h (B, V, P, 3, D) in order RGB, mask, pointmap, correspondence a (B, V, P, D) and
        view_mask (B, V), False on padded views. Returns mixed tokens (B, V, P, 3, D), zero on padded views.
        """
        x = (h + a[..., None, :])[view_mask]
        z = (self.norm(x) + self.modality).flatten(0, 1)
        y = x + self.gate * self.attn(z, z, z, need_weights=False)[0].view_as(x)
        out = h.new_zeros(h.shape)
        out[view_mask] = y
        return out
