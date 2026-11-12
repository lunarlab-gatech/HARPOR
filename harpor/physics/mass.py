"""Rigid-body mass properties from VoMP's per-voxel density (build_plan.md note 9).

A rigid MuJoCo body holds one mass, one centre of mass and one inertia tensor, so the voxels are
summed. Each occupied voxel is a solid box of edge h (per axis) and density rho:
m_i = rho_i * hx * hy * hz, COM = sum m_i x_i / M, and
I = sum m_i [(r_i . r_i) I3 - r_i r_i^T] + the boxes' own inertia M / 12 * (hy^2 + hz^2, ...).
`fullinertia` is about the COM, in MuJoCo's order Ixx Iyy Izz Ixy Ixz Iyz.
"""

import numpy as np


def mass_properties(centers, rho, h):
    """centers (N, 3) in meters, rho (N,) in kg/m^3, h the voxel edge (scalar or per axis) in meters."""
    centers = np.asarray(centers, float)
    h = np.broadcast_to(np.asarray(h, float), (3,))
    m = np.asarray(rho, float) * h.prod()
    M = m.sum()
    com = (m[:, None] * centers).sum(0) / M
    r = centers - com
    I = np.eye(3) * (m * (r * r).sum(1)).sum() - np.einsum("i,ij,ik->jk", m, r, r)
    I += np.diag(M / 12 * (h[[1, 0, 0]] ** 2 + h[[2, 2, 1]] ** 2))
    return {
        "mass": float(M),
        "volume": float(len(m) * h.prod()),
        "com": com.tolist(),
        "inertia": I.tolist(),
        "fullinertia": [I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]],
    }
