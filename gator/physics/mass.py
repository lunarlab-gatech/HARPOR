"""Rigid-body mass properties from VoMP's per-voxel density (build_plan.md note 9).

A rigid MuJoCo body holds one mass, one centre of mass and one inertia tensor, so the voxels are
summed. Each occupied voxel is a solid box of edge h (per axis) and density rho:
m_i = rho_i * hx * hy * hz, COM = sum m_i x_i / M, and
I = sum m_i [(r_i . r_i) I3 - r_i r_i^T] + the boxes' own inertia M / 12 * (hy^2 + hz^2, ...).
`fullinertia` is about the COM, in MuJoCo's order Ixx Iyy Izz Ixy Ixz Iyz.
An articulated object has one such body per link (build_plan.md note 11); combine() gives the whole object.
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


def scaled_mass_properties(centers, rho, h, s):
    """The same voxels after a per-axis scale s about the origin (for a known true size)."""
    s = np.broadcast_to(np.asarray(s, float), (3,))
    return mass_properties(np.asarray(centers, float) * s, rho, np.asarray(h, float) * s)


def combine(bodies):
    """One rigid body from several (dicts of mass_properties, or with fullinertia only): the masses add,
    the COM is mass-weighted, and each inertia moves to the common COM (parallel axis theorem)."""
    m = np.array([b["mass"] for b in bodies], float)
    c = np.array([b["com"] for b in bodies], float)
    M = m.sum()
    com = (m[:, None] * c).sum(0) / M
    I = np.zeros((3, 3))
    for b, mi, ci in zip(bodies, m, c):
        if "inertia" in b:
            Ib = np.asarray(b["inertia"], float)
        else:
            xx, yy, zz, xy, xz, yz = b["fullinertia"]
            Ib = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]], float)
        d = ci - com
        I += Ib + mi * (d @ d * np.eye(3) - np.outer(d, d))
    return {
        "mass": float(M),
        "volume": float(sum(b.get("volume", 0.0) for b in bodies)),
        "com": com.tolist(),
        "inertia": I.tolist(),
        "fullinertia": [I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]],
    }
