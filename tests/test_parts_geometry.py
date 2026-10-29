import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from harpor.parts import geometry as G


def split_box(n=2):
    """A unit box subdivided n times, with every face's vertices of its own (as a textured GLB has
    them along its texture seams), and the lid: the faces above 75% of the height."""
    b = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    for _ in range(n):
        b = b.subdivide()
    v = b.vertices[b.faces].reshape(-1, 3)
    f = np.arange(len(v)).reshape(-1, 3)
    return v, f, (v[f].mean(1)[:, 2] > 0.25).astype(int)


def test_topology_merges_texture_seams_and_keeps_the_face_order():
    v, f, _ = split_box()
    assert len(trimesh.Trimesh(v, f, process=False).face_adjacency) == 0
    t = G.topology(v, f)
    assert len(t.faces) == len(f) and np.allclose(t.vertices[t.faces], v[f])
    assert len(t.face_adjacency) == len(f) * 3 // 2  # closed: every edge has two faces


def test_a_box_cut_at_75_percent_gives_one_seam_with_four_lines():
    v, f, ids = split_box()
    seams = G.seams(G.topology(v, f), ids, ["body", "lid"])
    assert len(seams) == 1
    s = seams[0]
    assert s["parts"] == ["body", "lid"] and s["edges"] == 16 and np.isclose(s["length"], 4.0)
    assert np.allclose(s["center"], [0, 0, 0.25])
    assert len(s["lines"]) == 4
    sides = set()
    for line in s["lines"]:
        d = np.array(line["direction"])
        assert np.isclose(line["length"], 1.0) and np.isclose(line["inlier_share"], 0.25)
        assert np.allclose(np.abs(d), [1, 0, 0]) or np.allclose(np.abs(d), [0, 1, 0])
        p0, p1 = np.array(line["p0"]), np.array(line["p1"])
        assert np.allclose(np.abs(p0[:2]), 0.5) and np.allclose(np.abs(p1[:2]), 0.5) and np.isclose(p0[2], 0.25)
        assert np.isclose(np.linalg.norm(p1 - p0), 1.0) and np.allclose((p1 - p0) / 1.0, d)
        n = np.array(line["normals"]["lid"])
        assert np.allclose(n, line["normals"]["body"]) and np.isclose(n[2], 0) and np.isclose(np.abs(n).max(), 1)
        sides.add(tuple(np.round(n).astype(int)))
    assert sides == {(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)}
    assert np.allclose(s["normals"]["lid"], 0) and np.allclose(s["normals"]["body"], 0)  # the four sides cancel


def test_no_seam_without_a_cut_and_three_parts_give_three_pairs():
    v, f, ids = split_box()
    t = G.topology(v, f)
    assert G.seams(t, np.zeros(len(f), int)) == []
    ids3 = ids.copy()
    c = v[f].mean(1)
    ids3[(ids == 0) & (c[:, 2] < -0.25)] = 2  # a bottom ring below the body
    pairs = [s["parts"] for s in G.seams(t, ids3, ["body", "lid", "base"])]
    assert pairs == [["body", "lid"], ["body", "base"]]


def test_a_jagged_seam_still_gives_its_line():
    rng = np.random.default_rng(3)
    x = np.linspace(0, 1, 41)
    a = np.stack([x[:-1], rng.normal(0, 0.003, 40), np.zeros(40)], 1)
    b = np.stack([x[1:], rng.normal(0, 0.003, 40), np.zeros(40)], 1)
    lines = G.fit_lines(a, b)
    assert len(lines) == 1 and lines[0]["inlier_share"] == 1.0 and abs(lines[0]["length"] - 1) < 1e-3
    assert np.allclose(lines[0]["direction"], [1, 0, 0], atol=0.02)
    assert G.fit_lines(a, b) == lines  # deterministic


def test_boxes_of_a_turned_block():
    b = trimesh.creation.box(extents=(0.6, 0.2, 0.4))
    R = Rotation.from_euler("zyx", [30, 20, 10], degrees=True).as_matrix()
    p = b.vertices @ R.T + [1.0, 2.0, 3.0]
    box = G.aabb(p)
    assert np.allclose(box["center"], (p.min(0) + p.max(0)) / 2) and np.allclose(box["extent"], np.ptp(p, 0))
    o = G.obb(p)
    axes = np.array(o["axes"])
    assert np.allclose(o["extents"], [0.6, 0.4, 0.2]) and np.allclose(o["center"], [1, 2, 3])
    assert np.allclose(axes @ axes.T, np.eye(3)) and np.isclose(np.linalg.det(axes), 1)
    assert np.allclose(np.abs(axes), np.abs(R.T[[0, 2, 1]]), atol=1e-6)
    assert np.allclose(np.sort(np.array(o["corners"]), 0), np.sort(p, 0), atol=1e-9)
    flat = G.obb(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]))  # three points: no 3-D box
    assert np.isfinite(flat["extents"]).all() and len(flat["corners"]) == 8
