"""Face selection of add_part (harpor/agent/blender/part_select.py), without Blender.

The selection must work for any part, so the three shapes here differ on purpose: a flat panel that stands on a
box, a cylinder on a plate, and a block inside a larger block. The steps are those of blender/add_part.py with
its plane selection, without the bisect: the faces that touch the region, the piece of them nearest the seed,
and of that piece the faces whose centre is in the region.
"""

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from harpor.agent.blender import part_select as S  # noqa: E402


def joined(*meshes):
    """One mesh of several, with the face ranges of each: [(start, stop), ...]."""
    spans, start = [], 0
    for m in meshes:
        spans.append((start, start + len(m.faces)))
        start += len(m.faces)
    return trimesh.util.concatenate(list(meshes)), spans


def arrays(mesh):
    return mesh.vertices, mesh.faces.tolist(), mesh.triangles_center, mesh.face_adjacency.tolist()


def by_planes(mesh, planes, seed):
    """The faces that add_part's plane selection takes (see the module docstring)."""
    v, faces, centers, pairs = arrays(mesh)
    labels, _ = S.components(len(faces), pairs, S.touching(S.inside(v, planes, 1e-9), faces))
    near = labels == S.nearest_piece(labels, centers, seed)
    labels, _ = S.components(len(faces), pairs, near & S.inside(centers, planes, 1e-9))
    return np.flatnonzero(labels == S.nearest_piece(labels, centers, seed))


def box(lo, hi, cuts=0):
    b = trimesh.creation.box(bounds=np.array([lo, hi], float))
    for _ in range(cuts):
        b = b.subdivide()
    return b


def test_a_flat_panel_on_a_box_is_one_half_space():
    """Two panels stand on the box's top, at x = 0 and x = 1. The plane just under z = 1 holds both, and the box's
    top too; the seed picks one panel."""
    mesh, (body, left, right) = joined(box([0, 0, 0], [1, 1, 1], 2), box([0, 0, 1], [0.02, 1, 1.4], 2),
                                       box([0.98, 0, 1], [1, 1, 1.4], 2))
    up = [[0, 0, 0.999, 0, 0, 1]]
    assert by_planes(mesh, up, [0.0, 0.5, 1.2]).tolist() == list(range(*left))
    assert by_planes(mesh, up, [1.0, 0.5, 1.2]).tolist() == list(range(*right))
    # a normal that is not a unit vector means the same plane
    assert by_planes(mesh, [[0, 0, 0.999, 0, 0, 7.5]], [1.0, 0.5, 1.2]).tolist() == list(range(*right))
    # a second plane makes a wedge: only the upper half of the right panel
    upper = by_planes(mesh, up + [[0, 0, 1.2, 0, 0, 1]], [1.0, 0.5, 1.3])
    assert 0 < len(upper) < right[1] - right[0] and set(upper) <= set(range(*right))
    assert (mesh.triangles_center[upper][:, 2] >= 1.2).all()


def test_a_cylinder_on_a_plate_is_chosen_by_distance_to_its_axis():
    """A knob: the agent's own rule (distance to an axis) gives the faces, and face_indices checks them."""
    knob = trimesh.creation.cylinder(radius=0.05, height=0.04, sections=24)
    knob.apply_translation([0.3, 0.2, 0.031])  # just above the plate's top, at z = 0.01
    mesh, (plate, disc) = joined(box([0, 0, 0], [1, 1, 0.01], 3), knob)
    _, faces, centers, pairs = arrays(mesh)
    rule = (np.linalg.norm(centers[:, :2] - [0.3, 0.2], axis=1) < 0.06) & (centers[:, 2] > 0.0105)
    idx = S.face_indices(np.flatnonzero(rule), len(faces))
    assert idx.tolist() == list(range(*disc))
    assert S.face_indices(rule, len(faces)).tolist() == idx.tolist()  # one bool per polygon means the same
    labels, pieces = S.components(len(faces), pairs, rule)
    assert pieces == 1 and (labels[idx] == 0).all() and (labels[:plate[1]] == -1).all()


def test_a_block_inside_a_larger_block_is_a_box_of_six_planes():
    """A drawer-like block inside a shell: six planes, normals inward, hold the block and none of the shell."""
    mesh, (shell, block) = joined(box([0, 0, 0], [1, 1, 1], 2), box([0.3, 0.3, 0.3], [0.6, 0.6, 0.6], 1))
    lo, hi = np.array([0.25, 0.25, 0.25]), np.array([0.65, 0.65, 0.65])
    planes = [[*lo, *n] for n in np.eye(3)] + [[*hi, *(-n)] for n in np.eye(3)]
    assert by_planes(mesh, planes, [0.45, 0.45, 0.45]).tolist() == list(range(*block))
    # a region with no face in it selects nothing
    v, faces, centers, pairs = arrays(mesh)
    far = [[5, 5, 5, 1, 0, 0]]
    labels, pieces = S.components(len(faces), pairs, S.touching(S.inside(v, far), faces))
    assert pieces == 0 and S.nearest_piece(labels, centers, [0, 0, 0]) == -1


def test_inside_and_components():
    pts = [[0, 0, 0], [0, 0, 1], [0, 0, -1e-7]]
    assert S.inside(pts, [[0, 0, 0, 0, 0, 1]]).tolist() == [True, True, False]
    assert S.inside(pts, [[0, 0, 0, 0, 0, 1]], tol=1e-6).tolist() == [True, True, True]
    assert S.inside(pts, [[0, 0, 0, 0, 0, 1], [0, 0, 0.5, 0, 0, -1]]).tolist() == [True, False, False]
    # a chain 0-1-2-3 with face 2 outside the mask: two pieces
    labels, pieces = S.components(4, [(0, 1), (1, 2), (2, 3)], [True, True, False, True])
    assert labels.tolist() == [0, 0, -1, 1] and pieces == 2
    assert S.nearest_piece(labels, [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], [2.9, 0, 0]) == 1
    assert S.touching([True, False, False], [[0, 1], [1, 2]]).tolist() == [True, False]


@pytest.mark.parametrize("values, why", [
    ([], "at least one"), ([[1, 2]], "1-D"), ([0.5, 1.0], "integers"), ([0, 9], "0 to 8"), ([-1, 2], "0 to 8"),
    (np.zeros(9, bool), "selects no face"), (np.ones(4, bool), "one value per polygon")])
def test_bad_face_files_are_refused(values, why):
    with pytest.raises(ValueError, match=why):
        S.face_indices(np.asarray(values), 9)


def test_face_indices_are_sorted_and_distinct():
    assert S.face_indices(np.array([5, 2, 5, 0], np.int32), 9).tolist() == [0, 2, 5]


@pytest.mark.parametrize("planes", [[[0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, float("nan")]], []])
def test_bad_planes_are_refused(planes):
    with pytest.raises(ValueError):
        S.unit_planes(planes)
