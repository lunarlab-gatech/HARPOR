import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from harpor import frames
from harpor.physics import soft

trimesh = pytest.importorskip("trimesh")


def pad(size, center=(0, 0, 0)):
    """A closed box with vertices across its faces."""
    box = trimesh.creation.box(size).subdivide().subdivide()
    box.apply_translation(center)
    return box


def lens(size, center=(0, 0, 0)):
    """A closed flattened sphere with the extents of size."""
    m = trimesh.creation.icosphere(subdivisions=4)
    m.apply_scale(np.array(size) / 2)
    m.apply_translation(center)
    return m


def code(mesh, parent):
    with pytest.raises(soft.SoftError) as e:
        soft.hex_mesh(mesh, parent)
    return e.value.code


def block():
    """A 0.6 x 0.6 x 0.2 m block with its top at z = 0."""
    return pad([0.6, 0.6, 0.2], [0, 0, -0.1])


def test_oriented_voxels_keep_the_mesh_under_any_rigid_pose():
    # Voxels on the world axes would change with the tilt; the link's own box keeps 3 clean layers.
    counts = set()
    for i, r in enumerate(Rotation.random(5, random_state=0).as_matrix()):
        m = lens([0.46, 0.44, 0.07])
        m.apply_transform(frames.rigid(r, [0.1 * i, 0.3, -0.2]))
        matrix, pitch, s_from_voxel = soft.voxels(m)
        thin = s_from_voxel[:3, 0] / np.linalg.norm(s_from_voxel[:3, 0])
        assert abs(thin @ r[:, 2]) > np.cos(np.radians(1)) and np.isclose(pitch, 0.035, rtol=0.01)
        counts.add((len(soft.hexes(matrix)[1]), min(matrix.shape)))
    assert counts == {(628, 3)}


@pytest.mark.parametrize("thickness", [0.04, 0.05])
def test_pitch_grows_to_the_target_and_keeps_the_layers(thickness):
    matrix, pitch, _ = soft.voxels(pad([0.46, 0.44, thickness]))
    assert np.isclose(pitch, thickness / 2 * 1.25 ** (3 if thickness == 0.04 else 2))
    assert len(soft.hexes(matrix)[1]) == 784 and min(matrix.shape) == 3


def test_a_thin_or_flat_part_has_no_usable_mesh():
    # A 5 mm mat has 28224 nodes at the last 2-layer pitch; it is refused before it is voxelized.
    for thickness in (0.005, 0.001):
        with pytest.raises(soft.SoftError, match="too thin") as e:
            soft.hex_mesh(pad([0.4, 0.4, thickness]), block())
        assert e.value.code == "soft_untestable"
    assert soft.surface_voxels(pad([0.4, 0.4, 0.005]), 0.005 * 1.25 ** 3 / 2) > soft.MAX_NODES
    # Over the node limit after the pitch rule: refused after it.
    with pytest.raises(soft.SoftError, match="nodes and 3 layers"):
        soft.hex_mesh(pad([0.4, 0.4, 0.018]), block())
    flat = trimesh.Trimesh([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], [[0, 1, 2], [0, 2, 3]], process=False)
    assert code(flat, block()) == "soft_untestable"


def test_five_tets_fill_the_voxels_and_share_their_faces():
    matrix = np.zeros((3, 4, 2), bool)
    matrix[:, :3, :] = True
    matrix[1, 3, 0] = True  # not a plain box, so that the parity is on global indices
    cells, corners, cell_nodes = soft.hexes(matrix)
    R = Rotation.from_euler("xyz", [20, 30, 40], degrees=True).as_matrix()
    nodes = corners * 0.1 @ R.T
    tets = soft.five_tets(cells, cell_nodes, nodes)
    a, b, c, d = np.moveaxis(nodes[tets], 1, 0)
    volume = (np.cross(b - a, c - a) * (d - a)).sum(1) / 6
    assert len(tets) == 5 * len(cells) and (volume > 0).all() and np.isclose(volume.sum(), len(cells) * 0.001)
    faces = np.sort(np.concatenate([tets[:, list(f)] for f in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))]), axis=1)
    count = np.unique(faces, axis=0, return_counts=True)[1]
    # Every inner face is shared, so the two patterns meet without a crack: only the voxels' outer faces are
    # left, 2 triangles each.
    padded = np.pad(matrix, 1).astype(int)
    outer = sum(np.abs(np.diff(padded, axis=k)).sum() for k in range(3))
    assert set(count) == {1, 2} and (count == 1).sum() == 2 * outer


def test_pins_stay_on_the_parents_side():
    # A pad on a block: the hex stands proud below the pad, inside the block, and only that layer is pinned.
    # The node levels are -0.015, 0.015, 0.045 and 0.075 m, and 0.015 is within a pitch too.
    m = soft.hex_mesh(pad([0.4, 0.3, 0.06], [0, 0, 0.03]), block())
    z = m["nodes"][:, 2].round(6)
    assert m["pitch"] == 0.03 and np.unique(z).tolist() == [-0.015, 0.015, 0.045, 0.075]
    assert np.flatnonzero(z < 0).tolist() == m["pinned"].tolist()
    # Within a pitch of the block but not on its side: nodes above the top are left free.
    d = np.array([[0, 0, 0.01], [0, 0, -0.01], [0, 0, 0.5]])
    points, normals = np.array([[0.0, 0, 0]]), np.array([[0.0, 0, 1]])
    assert soft.pins(d, points, normals, 0.03).tolist() == [1]


def test_pins_fall_back_to_the_distance():
    # All the hex above the parent's surface: no node is on its side.
    nodes = np.array([[0, 0, 0.01], [0, 0, 0.02], [0, 0, 0.5]])
    assert soft.pins(nodes, np.array([[0.0, 0, 0]]), np.array([[0.0, 0, 1]]), 0.03).tolist() == [0, 1]


@pytest.mark.parametrize("seat", [0.04, 0.015, 0.005])
def test_pins_on_a_parent_thinner_than_the_hex_are_the_layer_under_its_top(seat):
    # The hex stands about half a pitch (1.75 cm) below the lens, so under a thin seat the bottom layer is below the
    # seat's underside too. That underside is the surface nearest those nodes, and its normal points away from the
    # lens; the seat under the lens judges the side, so they stay pinned and the layer above them stays free.
    top = 0.4 + seat
    m = soft.hex_mesh(lens([0.46, 0.44, 0.07], [0, 0, top + 0.035]), pad([0.5, 0.48, seat], [0, 0, 0.4 + seat / 2]))
    z = m["nodes"][:, 2]
    assert len(m["pinned"]) == 140 and (z[m["pinned"]] < top).all()
    assert np.flatnonzero(z < top).tolist() == m["pinned"].tolist() and np.allclose(m["direction"], [0, 0, -1])
    # One node under a plate 1 cm thick: the nearest sample is the underside, so only the seat keeps it.
    node, points, normals = np.array([[0, 0, -0.012]]), np.array([[0.0, 0, 0], [0, 0, -0.01]]), np.array([[0.0, 0, 1], [0, 0, -1]])
    assert soft.pins(node, points, normals, 0.03).tolist() == [0]  # by the distance alone
    assert soft.pins(np.r_[node, [[0, 0, 0.012]]], points, normals, 0.03, np.array([True, False])).tolist() == [0]


def test_a_part_needs_a_pin_on_every_island():
    assert code(pad([0.4, 0.3, 0.06], [0, 0, 1.0]), block()) == "soft_no_support"  # far from its parent
    matrix = np.ones((2, 2, 2), bool)
    cells, _, cell_nodes = soft.hexes(matrix)
    assert "pinned" in soft.support_problem(matrix, cells, cell_nodes, np.arange(19))  # of 27 nodes
    assert soft.support_problem(matrix, cells, cell_nodes, np.arange(18)) is None
    # One slab on the block and another floating high above it: the second would fall free.
    two = trimesh.util.concatenate([pad([0.4, 0.3, 0.06], [0, 0, 0.03]), pad([0.4, 0.3, 0.06], [0, 0, 0.6])])
    with pytest.raises(soft.SoftError, match="voxel islands"):
        soft.hex_mesh(two, block())


def test_press_direction_points_at_the_part_seated_on_the_parent():
    # The lid is thinner than the pitch, so the surface nearest a pin is its underside, with the wrong normal.
    lid = pad([0.6, 0.4, 0.04], [0, 0, 0.02])
    m = soft.hex_mesh(pad([0.5, 0.3, 0.06], [0, 0, 0.07]), lid)
    assert m["pitch"] > 0.02 and np.allclose(m["direction"], [0, 0, -1])
    tilted = frames.rigid(Rotation.from_euler("x", 35, degrees=True).as_matrix(), [0.2, 0.1, 0])
    a, b = pad([0.5, 0.3, 0.06], [0, 0, 0.07]), lid.copy()
    a.apply_transform(tilted)
    b.apply_transform(tilted)
    assert np.allclose(soft.hex_mesh(a, b)["direction"], tilted[:3, :3] @ [0, 0, -1], atol=1e-6)


def test_a_sleeve_has_no_press_direction():
    bar = trimesh.creation.cylinder(radius=0.03, height=0.6, sections=32)
    sleeve = trimesh.creation.annulus(r_min=0.03, r_max=0.06, height=0.2, sections=32)
    assert np.allclose(soft.hex_mesh(sleeve, bar)["direction"], 0)


def test_a_part_open_where_it_sits_takes_its_direction_from_the_pins():
    # Without an underside only the cushion's side walls meet the lid, and they do not face it: there is no seat.
    # The lid's surface nearest the pins, its top, gives the direction.
    lid = pad([0.6, 0.4, 0.04], [0, 0, 0.02])
    cushion = pad([0.5, 0.3, 0.06], [0, 0, 0.07])
    cushion = trimesh.Trimesh(cushion.vertices, cushion.faces[cushion.triangles_center[:, 2] > 0.0401], process=False)
    points, normals = soft.surface(lid, 0.03)
    assert not soft.seated(points, normals, *soft.surface(cushion, 0.03), 0.03).any()
    m = soft.hex_mesh(cushion, lid)
    assert np.allclose(m["direction"], [0, 0, -1]) and len(m["press"]) and (m["nodes"][m["pinned"], 2] < 0.04).all()


def test_weights_rebuild_every_vertex_from_a_tet_that_holds_it(monkeypatch):
    mesh = lens([0.46, 0.44, 0.07])
    m = soft.hex_mesh(mesh, pad([0.6, 0.6, 0.2], [0, 0, -0.13]))
    monkeypatch.setattr(soft, "BATCH", 97)
    tet, w = soft.bind(mesh.vertices, m["nodes"], m["tets"])
    assert tet.shape == (len(mesh.vertices),) and w.shape == (len(mesh.vertices), 4)
    assert np.allclose(w.sum(1), 1) and w.min() > -1e-9
    assert np.allclose((m["nodes"][m["tets"][tet]] * w[..., None]).sum(1), mesh.vertices, atol=1e-12)
    # A point outside the mesh takes the nearest fit, with a negative weight that still rebuilds it.
    out = np.array([[0.0, 0.0, m["nodes"][:, 2].max() + 0.01]])
    tet, w = soft.bind(out, m["nodes"], m["tets"])
    assert w.min() < 0 and np.allclose((m["nodes"][m["tets"][tet]] * w[..., None]).sum(1), out)


def test_press_nodes_are_free_and_under_the_middle_of_the_far_surface():
    mesh = lens([0.46, 0.44, 0.07], [0, 0, 0.035])
    m = soft.hex_mesh(mesh, block())
    assert np.allclose(m["direction"], [0, 0, -1]) and len(m["press"]) > 3
    p = m["nodes"][m["press"]]
    assert not set(m["press"]) & set(m["pinned"].tolist())
    assert np.linalg.norm(p[:, :2], axis=1).max() < 0.2 and p[:, 2].min() > 0.02
    # Without a direction nothing faces away from it.
    assert soft.press_nodes(*soft.surface(mesh, m["pitch"]), np.zeros(3), m["nodes"], m["tets"], m["pinned"]).size == 0


def test_a_box_with_vertices_only_at_its_corners_has_press_nodes():
    # Its 4 top vertices lie at the rim of the top's patch, none in its middle; samples of the top fill it. Half the
    # patch's radius, 0.146 m, spans the box's width but only the middle of its length.
    box = trimesh.creation.box([0.5, 0.3, 0.06])
    box.apply_translation([0, 0, 0.03])
    assert len(box.vertices) == 8
    m = soft.hex_mesh(box, block())
    p = m["nodes"][m["press"]]
    assert len(p) > 9 and not set(m["press"]) & set(m["pinned"].tolist())
    assert np.abs(p[:, 0]).max() < 0.146 + m["pitch"] and p[:, 2].min() > 0.03


def test_bones_merge_the_pinned_nodes():
    tets = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    tet_of_vertex = np.array([0, 0, 1])
    weights = np.array([[0.5, 0.25, 0.25, 0.0], [-0.5, 0.5, 0.5, 0.5], [0.25, 0.25, 0.25, 0.25]])
    bones = {n: (ids.tolist(), w.tolist()) for n, ids, w in soft.bones(tets, np.array([0, 1]), tet_of_vertex, weights)}
    # Vertex 1's pinned weights cancel out, and a zero weight binds nothing.
    assert bones == {-1: ([0, 2], [0.75, 0.25]), 2: ([0, 1, 2], [0.25, 0.5, 0.25]), 3: ([1, 2], [0.5, 0.25]),
                     4: ([2], [0.25])}


def test_the_mesh_needs_no_rtree(monkeypatch):
    # envs/mujoco has no rtree.
    monkeypatch.setitem(sys.modules, "rtree", None)
    mesh = lens([0.46, 0.44, 0.07], [0, 0, 0.035])
    m = soft.hex_mesh(mesh, block())
    assert len(soft.binding(m, [mesh])["weights"]) == len(mesh.vertices)
