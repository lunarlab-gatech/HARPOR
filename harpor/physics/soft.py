"""The hex mesh of a soft link and the map from its nodes to the visible mesh (build_plan.md note 11).

harpor.physics.mjcf turns a soft link into a MuJoCo flex: tetrahedra over a coarse voxel mesh, pinned to the
parent, with the link's own mesh as a skin on the nodes.

Voxels: the link is voxelized in its own oriented box and the nodes are mapped back, so a tilted pad gets clean
layers and the mesh keeps its size and layers under any rigid pose. Its nodes can differ by a mirror image or
a fraction of a degree, because oriented_bounds signs its axes freely and rounds its search angles. The pitch
starts at half the box's thinnest side and grows by 1.25 while the mesh has over 800 nodes and the next pitch
still gives 2 layers. Over 2000 nodes or under 2 layers is no usable mesh, and a part whose surface alone
fills over 2000 voxels at the widest pitch is refused before it is voxelized, which would take minutes. Each
voxel splits into 5 tetrahedra, mirrored by (i + j + k) % 2 on the global indices so that neighbours share
their face diagonals. fill() cannot close an open mesh, which stays hollow. A voxel corner can stand up to a
pitch proud of the skin, and the hex volume overshoots the mesh, so a soft link's mass never comes from it.

Seat: the samples of the parent's surface where the part sits on it, within 1 pitch of the part's surface and
with normals that oppose its normals there.

Pins: the nodes within 1 pitch of the parent's surface and on its side of the seat (n . (x - s) <= 0 for the
nearest seat sample s and its outward normal n), because the distance alone pins a layer inside the part.
The seat judges the side because the surface nearest a node can be the far side of a parent thinner than a
pitch, whose normal points away from the part. Without a seat, every sample judges it; when that gives
none, the distance alone. A part with no pin, with over 70% of its nodes pinned, or with a 6-connected
voxel island that has no pin has no usable support: an island would float free under gravity.

Binding: each visible vertex takes the barycentric weights of a tetrahedron, W in paper Eq. 4, the one holding
it or else the least far outside.

Press: the direction points at the parent, minus the mean outward normal of the seat, or without a seat (a
part open on the side it sits on) of the parent's surface nearest the pins. Under a length of 0.3 (a
sleeve) there is no direction. The press nodes lie under the middle of the surface that faces away from it,
from samples of that surface, so that a box with vertices only at its corners has them too.

numpy, scipy and trimesh only, with cKDTree for the searches: envs/mujoco has no rtree.
"""

import numpy as np

from harpor import frames

TARGET_NODES, MAX_NODES = 800, 2000
GROWTH = 1.25  # of the pitch
WIDEST = 3  # growths: 1.25^4 > 2, so a fourth leaves 1 layer
MIN_LAYERS = 2
MAX_PINNED = 0.7  # share of the nodes
SAMPLES = (1000, 200000)  # of a surface, about 4 per pitch along it
NEAREST_TETS = 32  # searched per vertex
BATCH = 20000  # vertices at a time, because each one needs NEAREST_TETS solves
MIN_DIRECTION = 0.3  # length of the mean parent normal: a sleeve has no press direction
FACING = 0.5  # cos 60 degrees: press nodes lie under a surface facing away from the press direction

CORNERS = np.array(list(np.ndindex(2, 2, 2)))  # of a voxel, as offsets: corner 6 is (1, 1, 0)
EVEN = np.array([[0, 3, 5, 6], [4, 0, 5, 6], [2, 0, 3, 6], [1, 0, 3, 5], [7, 3, 5, 6]])  # centre, then 4 corner tets
ODD = np.array([[1, 2, 4, 7], [0, 1, 2, 4], [6, 2, 4, 7], [5, 1, 4, 7], [3, 1, 2, 7]])


class SoftError(ValueError):
    """A soft link with no usable mesh. code is the finding: soft_no_support or soft_untestable."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def hexes(matrix):
    """(cells, corners, cell_nodes) of the True voxels of matrix: their indices (M, 3), the integer corners of
    the mesh (N, 3), and the 8 nodes of each voxel, in CORNERS order (M, 8)."""
    cells = np.argwhere(matrix)
    corners, inverse = np.unique((cells[:, None] + CORNERS).reshape(-1, 3), axis=0, return_inverse=True)
    return cells, corners, inverse.reshape(-1, 8)


def _voxelized(boxed, pitch):
    """The filled voxel grid of a mesh in its box. trimesh raises ValueError when it cannot subdivide that fine."""
    try:
        return boxed.voxelized(pitch).fill()
    except ValueError as e:
        raise SoftError("soft_untestable", f"it cannot be voxelized at {pitch:g} m: {e}") from None


def surface_voxels(mesh, pitch):
    """A lower bound of the voxels that the surface fills at pitch, from samples. It costs nothing next to
    voxelizing, which first subdivides the mesh to half the pitch: millions of faces for a thin mat."""
    import trimesh

    points = trimesh.sample.sample_surface(mesh, 10 * MAX_NODES, seed=0)[0]
    return len(np.unique(np.round(points / pitch), axis=0))


def voxels(mesh):
    """(matrix, pitch, S_from_voxel): the filled voxels of a mesh by the pitch rule, and the 4x4 that takes a
    voxel index to a voxel centre in the mesh's frame. Raises SoftError when there is no usable mesh, early
    when even the widest pitch has too many voxels."""
    import trimesh

    box_from_s, extents = trimesh.bounds.oriented_bounds(mesh)  # thinnest side first
    if not extents.min() > 0:
        raise SoftError("soft_untestable", "it is flat")
    boxed = mesh.copy()
    boxed.apply_transform(box_from_s)
    pitch = extents.min() / 2
    if surface_voxels(boxed, pitch * GROWTH ** WIDEST) > MAX_NODES:
        raise SoftError("soft_untestable", f"it is too thin for its size: {extents.min():g} m across")
    grid = _voxelized(boxed, pitch)
    while len(hexes(grid.matrix)[1]) > TARGET_NODES:
        wider = _voxelized(boxed, pitch * GROWTH)
        if min(wider.matrix.shape) < MIN_LAYERS:
            break
        pitch, grid = pitch * GROWTH, wider
    n, layers = len(hexes(grid.matrix)[1]), min(grid.matrix.shape)
    if n > MAX_NODES or layers < MIN_LAYERS:
        raise SoftError("soft_untestable", f"its voxels give {n} nodes and {layers} layers at a pitch of {pitch:g} m")
    return grid.matrix, pitch, np.linalg.inv(box_from_s) @ grid.transform


def five_tets(cells, cell_nodes, nodes):
    """The 5 tetrahedra of each voxel (5M, 4), oriented positive at nodes."""
    pattern = np.where((cells.sum(1) % 2 == 0)[:, None, None], EVEN[None], ODD[None]).reshape(-1, 20)
    tets = np.take_along_axis(cell_nodes, pattern, axis=1).reshape(-1, 4)
    a, b, c, d = np.moveaxis(nodes[tets], 1, 0)
    flip = ((np.cross(b - a, c - a)) * (d - a)).sum(1) < 0
    tets[flip, 2:] = tets[flip, 2:][:, ::-1]
    return tets


def surface(mesh, pitch):
    """(points, outward normals) of samples of a mesh's surface, about 4 per pitch along it."""
    import trimesh

    points, faces = trimesh.sample.sample_surface(mesh, int(np.clip(mesh.area / (pitch / 4) ** 2, *SAMPLES)), seed=0)
    return points, mesh.face_normals[faces]


def seated(points, normals, skin_points, skin_normals, pitch):
    """The mask of the parent's surface samples where the part sits on it, from samples of both (see the module
    docstring)."""
    from scipy.spatial import cKDTree

    dist, near = cKDTree(skin_points).query(points)
    return (dist <= pitch) & ((normals * skin_normals[near]).sum(1) < 0)


def pins(nodes, points, normals, pitch, seat=None):
    """Ids of the nodes pinned to the surface samples of the parent (see the module docstring). seat: the mask of
    the samples that judge the side; every sample when it is None or empty."""
    from scipy.spatial import cKDTree

    close = cKDTree(points).query(nodes)[0] <= pitch
    if seat is not None and seat.any():
        points, normals = points[seat], normals[seat]
    near = cKDTree(points).query(nodes)[1]
    ids = np.flatnonzero(close & (((nodes - points[near]) * normals[near]).sum(1) <= 0))
    return ids if len(ids) else np.flatnonzero(close)


def support_problem(matrix, cells, cell_nodes, pinned):
    """Why the pinned nodes do not hold the voxels, or None."""
    from scipy import ndimage

    n = len(np.unique(cell_nodes))
    if not len(pinned):
        return "no node lies within a pitch of its parent"
    if len(pinned) > MAX_PINNED * n:
        return f"{len(pinned)} of its {n} nodes are pinned"
    labels, islands = ndimage.label(matrix)
    held = labels[tuple(cells[np.isin(cell_nodes, pinned).any(1)].T)]
    if len(np.unique(held)) < islands:
        return f"{islands - len(np.unique(held))} of its {islands} voxel islands have no pin"
    return None


def press_direction(normals):
    """The unit direction of a press toward the parent: minus the mean of the parent's outward normals where the
    part meets it (see the module docstring). Zeros for none, or for a sleeve, where they cancel out."""
    mean = -normals.sum(0) / max(len(normals), 1)
    length = np.linalg.norm(mean)
    return mean / length if length >= MIN_DIRECTION else np.zeros(3)


def hex_mesh(mesh, parent):
    """The flex of a mesh on its parent mesh, both in one frame: {nodes, tets, pinned, pitch, direction, press}.
    Raises SoftError when there is no usable mesh or support."""
    from scipy.spatial import cKDTree

    matrix, pitch, s_from_voxel = voxels(mesh)
    cells, corners, cell_nodes = hexes(matrix)
    nodes = frames.transform_points(s_from_voxel, corners - 0.5)
    points, normals = surface(parent, pitch)
    skin_points, skin_normals = surface(mesh, pitch)
    seat = seated(points, normals, skin_points, skin_normals, pitch)
    pinned = pins(nodes, points, normals, pitch, seat)
    problem = support_problem(matrix, cells, cell_nodes, pinned)
    if problem:
        raise SoftError("soft_no_support", problem)
    tets = five_tets(cells, cell_nodes, nodes)
    direction = press_direction(normals[seat] if seat.any() else normals[cKDTree(points).query(nodes[pinned])[1]])
    return {"nodes": nodes, "tets": tets, "pinned": pinned, "pitch": float(pitch), "direction": direction,
            "press": press_nodes(skin_points, skin_normals, direction, nodes, tets, pinned)}


def bind(vertices, nodes, tets):
    """(tet_of_vertex, weights): each vertex's tetrahedron (K,) and its 4 barycentric weights there (K, 4)."""
    from scipy.spatial import cKDTree

    corner = nodes[tets[:, 0]]
    solve = np.linalg.inv(np.moveaxis(nodes[tets[:, 1:]] - corner[:, None], 1, 2))
    tree = cKDTree(nodes[tets].mean(1))
    tet_of_vertex, weights = np.zeros(len(vertices), int), np.zeros((len(vertices), 4))
    for start in range(0, len(vertices), BATCH):
        x = vertices[start:start + BATCH]
        near = tree.query(x, k=min(NEAREST_TETS, len(tets)))[1].reshape(len(x), -1)
        w = np.einsum("nkij,nkj->nki", solve[near], x[:, None] - corner[near])
        w = np.concatenate([1 - w.sum(2, keepdims=True), w], axis=2)
        best = w.min(2).argmax(1)
        rows = np.arange(len(x))
        tet_of_vertex[start:start + BATCH], weights[start:start + BATCH] = near[rows, best], w[rows, best]
    return tet_of_vertex, weights


def press_nodes(points, normals, direction, nodes, tets, pinned):
    """The nodes to press, sorted: of the samples of the part's surface (points, normals) facing away from
    direction, each one in the central half of that patch's radius gives its largest-weight free node. No nodes
    without a direction or a patch."""
    facing = np.flatnonzero(normals @ -direction >= FACING)
    if not len(facing):
        return np.zeros(0, int)
    across = points[facing] - points[facing].mean(0)
    across -= (across @ direction)[:, None] * direction
    radius = np.linalg.norm(across, axis=1)
    central = facing[radius <= radius.max() / 2]
    tet, weights = bind(points[central], nodes, tets)
    near = tets[tet]
    free = np.where(np.isin(near, pinned), -np.inf, weights)
    return np.setdiff1d(np.take_along_axis(near, free.argmax(1)[:, None], axis=1), pinned)


def binding(flex, meshes):
    """{vertices, tet_of_vertex, weights} of a hex_mesh and the link's primitives (trimeshes)."""
    vertices = np.concatenate([m.vertices for m in meshes])
    tet_of_vertex, weights = bind(vertices, flex["nodes"], flex["tets"])
    return {"vertices": vertices, "tet_of_vertex": tet_of_vertex, "weights": weights}


def bones(tets, pinned, tet_of_vertex, weights):
    """[(node, vertex ids, weights)] of a skin, one per node that moves a vertex. Node -1 is all the pinned
    nodes merged: they hang from one body, so they are one bone."""
    node = np.where(np.isin(tets[tet_of_vertex], pinned), -1, tets[tet_of_vertex]).ravel()
    keys, inverse = np.unique(np.c_[node, np.repeat(np.arange(len(tet_of_vertex)), 4)], axis=0, return_inverse=True)
    total = np.bincount(inverse, weights.ravel())
    keys, total = keys[total != 0], total[total != 0]
    cuts = np.flatnonzero(np.diff(keys[:, 0])) + 1
    return [(int(k[0, 0]), k[:, 1], t) for k, t in zip(np.split(keys, cuts), np.split(total, cuts))]
