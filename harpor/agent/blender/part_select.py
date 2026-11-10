"""The face selection of add_part.py, in plain numpy, so that it also runs without Blender (tests/test_part_select.py).

A plane is [px, py, pz, nx, ny, nz]: a point on it and its normal, which points to the side of the part. A
region is the points on that side of every plane, so one plane is a half space, two a wedge, six a box. Nothing
here knows what the part is: the same code selects a panel, a block or a disc.

  face_indices(values, n)          the checked polygon indices of a --faces file
  unit_planes(planes)              planes with unit normals
  inside(points, planes, tol)      which points are in the region
  touching(vertex_in, faces)       which faces have a vertex in the region: the faces a plane cut can change
  components(n, pairs, mask)       the connected pieces of the masked faces, by shared edges
  nearest_piece(labels, centers, seed)   the piece with the face centre nearest a point
"""

import numpy as np


def face_indices(values, n):
    """The sorted, distinct polygon indices of a --faces array for a mesh of n polygons. ValueError when it is
    not a 1-D array of integers in [0, n), or is empty."""
    a = np.asarray(values)
    if a.ndim != 1 or a.size == 0:
        raise ValueError("the faces file must hold a 1-D array of polygon indices, and at least one")
    if a.dtype == bool:
        if a.size != n:
            raise ValueError(f"a mask of faces must have one value per polygon ({n}), not {a.size}")
        a = np.flatnonzero(a)
        if a.size == 0:
            raise ValueError("the mask selects no face")
    elif not np.issubdtype(a.dtype, np.integer):
        raise ValueError(f"the faces file must hold integers (or one bool per polygon), not {a.dtype}")
    a = np.unique(a.astype(np.int64))
    if a[0] < 0 or a[-1] >= n:
        raise ValueError(f"polygon indices must be 0 to {n - 1}; the file holds {int(a[0])} to {int(a[-1])}")
    return a


def unit_planes(planes):
    """planes as an (k, 6) float array with unit normals. ValueError for a plane with a zero or non-finite normal."""
    p = np.asarray(planes, float).reshape(-1, 6)
    if not len(p) or not np.isfinite(p).all():
        raise ValueError("a plane is six finite numbers: a point and a normal")
    norm = np.linalg.norm(p[:, 3:], axis=1)
    if (norm < 1e-12).any():
        raise ValueError("a plane's normal cannot be zero")
    return np.concatenate([p[:, :3], p[:, 3:] / norm[:, None]], 1)


def inside(points, planes, tol=0.0):
    """One bool per point: on the normal's side of every plane, or within tol of it."""
    x, p = np.asarray(points, float).reshape(-1, 3), unit_planes(planes)
    d = (x[:, None, :] - p[None, :, :3]) * p[None, :, 3:]
    return (d.sum(-1) >= -tol).all(1)


def touching(vertex_in, faces):
    """One bool per face: any of its vertices is in the region. faces: the vertex indices of each face."""
    v = np.asarray(vertex_in, bool)
    return np.array([bool(v[list(f)].any()) for f in faces], bool)


def components(n, pairs, mask):
    """labels, pieces: a piece number per face (-1 outside mask) and the number of pieces. Two masked faces are in
    one piece when a chain of masked faces that share edges joins them. pairs: (a, b) face indices per shared edge."""
    mask = np.asarray(mask, bool)
    parent = np.arange(n)

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in pairs:
        if mask[a] and mask[b]:
            ra, rb = root(a), root(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    labels, names = np.full(n, -1), {}
    for i in np.flatnonzero(mask):
        labels[i] = names.setdefault(root(int(i)), len(names))
    return labels, len(names)


def nearest_piece(labels, centers, seed):
    """The piece (a label of components) that has the face centre nearest seed; -1 when no face has a piece."""
    labels = np.asarray(labels)
    idx = np.flatnonzero(labels >= 0)
    if not len(idx):
        return -1
    d = np.linalg.norm(np.asarray(centers, float).reshape(-1, 3)[idx] - np.asarray(seed, float), axis=1)
    return int(labels[idx[int(d.argmin())]])
