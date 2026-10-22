"""Text adapters (build_plan.md notes 4 and 7).

SiglipAdapter: pooled SigLIP2 text embedding (1152) -> term added to the structure-flow timestep
embedding, which drives AdaLN in every block. Its output layer starts at zero.
T5GemmaAdapter: T5Gemma 2 encoder tokens (1152) -> cross-attention tokens (1024) for the geometry
and appearance flows.
"""

import torch
from torch import nn


class SiglipAdapter(nn.Module):
    def __init__(self, in_dim=1152, width=1536):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, width), nn.SiLU(),
                                 nn.Linear(width, width))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, emb, present):
        """Pooled embeddings (B, in_dim) and prompt flags (B,) -> (B, width). A missing prompt adds zero."""
        return torch.where(present[:, None], self.net(emb), 0.0)


class T5GemmaAdapter(nn.Module):
    def __init__(self, in_dim=1152, width=1024, heads=16):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, width)
        self.layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.0, activation="gelu",
                                                batch_first=True, norm_first=True)

    def forward(self, tokens, mask):
        """Tokens (B, L, in_dim) and mask (B, L), True on real tokens -> (B, L, width), zero on padding.

        A dropped prompt should pass the encoding of the empty prompt, not an all-False mask.
        """
        x = self.layer(self.proj(self.norm(tokens)), src_key_padding_mask=~mask)
        return x * mask[..., None]
