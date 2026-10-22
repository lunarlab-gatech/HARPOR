"""GatorProj, paper Eq. 2, and the Eq. 6 feature injection (build_plan.md note 6).

Voxel coordinates q live in the anchor-normalized frame, so p_G = s q + t (frames.denormalize).
Cameras are OpenCV world-to-camera matrices that act on G-frame points (frames.w2c_in_g), with the
intrinsics of the crop. Pixel centers sit at integer coordinates, so a crop of size (w, h) covers
[-0.5, w - 0.5) x [-0.5, h - 0.5).

Every input may carry the same leading batch dimensions (*B), one entry per object.
"""

import torch
import torch.nn.functional as F
from torch import nn


def project(p, K, w2c):
    """Project points p (*B, N, 3) into views K (*B, V, 3, 3), w2c (*B, V, 4, 4).

    Returns pixel coordinates (*B, V, N, 2) and depth (*B, V, N), as frames.project does.
    """
    p_cam = torch.einsum("...vij,...nj->...vni", w2c[..., :3, :3], p) + w2c[..., None, :3, 3]
    uvw = torch.einsum("...vij,...vnj->...vni", K, p_cam)
    z = p_cam[..., 2]
    return uvw[..., :2] / z[..., None], z


def to_grid(uv, crop_size):
    """Crop pixels to grid_sample coordinates, ((p + 0.5) / size) * 2 - 1, with crop_size = (w, h).

    Every feature map spans the whole crop, so one grid serves all map resolutions.
    """
    return (uv + 0.5) / uv.new_tensor(crop_size) * 2 - 1


def sample(feat, grid):
    """Bilinear samples (*B, V, N, C) of maps feat (*B, V, C, h, w) at grid (*B, V, N, 2)."""
    out = F.grid_sample(feat.flatten(0, -4), grid.flatten(0, -3)[:, None], mode="bilinear",
                        padding_mode="border", align_corners=False)
    return out[:, :, 0].transpose(1, 2).reshape(*grid.shape[:-1], feat.shape[-3])


def gator_proj(q, t, s, K, w2c, crop_size, feat_lr, feat_hr, view_mask=None):
    """Paper Eq. 2: mean of [F^D; F^N] over the views where a voxel is in front and inside the crop.

    q (*B, N, 3), t (*B, 3), s (*B), K (*B, V, 3, 3), w2c (*B, V, 4, 4) in G, crop_size (w, h),
    feat_lr (*B, V, C, h, w) DINOv3 patches, feat_hr (*B, V, C, H, W) NAF output, and view_mask
    (*B, V), False on padded views. Returns features (*B, N, 2C), zero where no view is valid,
    and the valid-view count (*B, N).
    """
    p = q * torch.as_tensor(s, dtype=q.dtype, device=q.device)[..., None, None] + t[..., None, :]
    uv, z = project(p, K, w2c)
    grid = to_grid(uv, crop_size)
    valid = (z > 0) & ((grid >= -1) & (grid < 1)).all(-1)
    if view_mask is not None:
        valid &= view_mask[..., None]
    grid = grid.masked_fill(~valid[..., None], 0)
    f = torch.cat([sample(feat_lr, grid), sample(feat_hr, grid)], -1)
    f = f.masked_fill_(~valid[..., None], 0).sum(-3)
    count = valid.sum(-2)
    return f / count.clamp(min=1)[..., None], count


class ProjInjection(nn.Module):
    """Eq. 6 projection term, h + W f + b. Named like Pixal3D's `cross_attn.proj_linear`.

    Starts at zero, so a new layer is an identity until Pixal3D weights are loaded.
    """

    def __init__(self, in_channels=2048, width=1536):
        super().__init__()
        self.proj_linear = nn.Linear(in_channels, width)
        nn.init.zeros_(self.proj_linear.weight)
        nn.init.zeros_(self.proj_linear.bias)

    def forward(self, h, f):
        return h + self.proj_linear(f)
