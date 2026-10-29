import time

import numpy as np
import trimesh

from harpor.parts import render
from harpor.stage_a.place import silhouette

K = np.array([[200.0, 0, 79.5], [0, 200.0, 59.5], [0, 0, 1]])
SIZE = (160, 120)


def box(extents, center):
    b = trimesh.creation.box(extents=extents)
    return b.vertices + center, b.faces


def test_a_box_shows_its_near_side_and_its_silhouette():
    v, f = box((0.4, 0.3, 0.2), (0.0, 0.0, 2.0))
    ids = render.face_ids(v, f, K, np.eye(4), SIZE)
    assert ids.shape == (120, 160) and ids.dtype == np.int64
    seen = np.unique(ids[ids >= 0])
    assert np.allclose(v[f[seen]].mean(1)[:, 2], 1.9)  # only the side facing the camera
    sil = silhouette(v, f, K, SIZE, SIZE)
    assert ((ids >= 0) & sil).sum() / ((ids >= 0) | sil).sum() > 0.97
    vis = render.visible_pixels(ids, len(f))
    assert vis.sum() == (ids >= 0).sum() and set(np.flatnonzero(vis)) == set(seen)
    left = np.zeros(ids.shape, bool)
    left[:, :80] = True
    inside = render.inside_pixels(ids, left, len(f))
    assert inside.sum() == ((ids >= 0) & left).sum() and (inside <= vis).all()


def test_a_near_box_hides_the_far_one():
    v1, f1 = box((0.4, 0.3, 0.2), (0.0, 0.0, 2.0))
    v2, f2 = box((0.1, 0.1, 0.1), (0.0, 0.0, 1.5))
    v, f = np.concatenate([v2, v1]), np.concatenate([f2, f1 + len(v2)])  # the near box's faces come first
    ids = render.face_ids(v, f, K, np.eye(4), SIZE)
    near = (ids >= 0) & (ids < len(f2))
    assert ids[59, 79] < len(f2) and ids[59, 64] >= len(f2) and near.sum() > 100
    alone = render.face_ids(v1, f1, K, np.eye(4), SIZE)
    assert (alone[near] >= 0).all()  # the far box is behind every pixel of the near one
    far = render.visible_pixels(ids, len(f))[len(f2):].sum()
    assert abs(far - (render.visible_pixels(alone, len(f1)).sum() - near.sum())) <= 2


def test_faces_behind_the_camera_or_outside_are_skipped():
    v, f = box((0.4, 0.3, 0.2), (0.0, 0.0, -2.0))
    assert (render.face_ids(v, f, K, np.eye(4), SIZE) == -1).all()
    v, f = box((0.4, 0.3, 0.2), (5.0, 0.0, 2.0))
    assert (render.face_ids(v, f, K, np.eye(4), SIZE) == -1).all()


def test_a_camera_that_turns_sees_another_side():
    v, f = box((0.4, 0.3, 0.2), (0.0, 0.0, 0.0))
    T = np.eye(4)
    T[:3, :3] = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])[:3, :3]  # camera at x = 2, looking along -x
    T[:3, 3] = [0.0, 0.0, 2.0]
    ids = render.face_ids(v, f, K, T, SIZE)
    seen = np.unique(ids[ids >= 0])
    assert len(seen) and np.allclose(v[f[seen]].mean(1)[:, 0], 0.2)


def test_scaled_keeps_the_projection():
    Kbig = np.array([[900.0, 0, 799.5], [0, 900.0, 599.5], [0, 0, 1]])
    Ks, size = render.scaled(Kbig, (1600, 1200))
    assert size == (768, 576) and np.isclose(Ks[0, 0], 900 * 0.48) and np.isclose(Ks[0, 2], 383.5)
    assert render.scaled(Kbig, (640, 480))[1] == (640, 480)


def test_two_hundred_thousand_faces_in_six_views_take_seconds():
    m = trimesh.creation.icosphere(subdivisions=7, radius=0.3)  # 327680 faces
    v = m.vertices + [0.0, 0.0, 1.2]
    Ks, size = render.scaled(np.array([[900.0, 0, 511.5], [0, 900.0, 383.5], [0, 0, 1]]), (1024, 768))
    t = time.time()
    for a in np.linspace(0, 2 * np.pi, 6, endpoint=False):
        ids = render.face_ids(v, m.faces, Ks, trimesh.transformations.rotation_matrix(a, [0, 1, 0], [0, 0, 1.2]), size)
    assert time.time() - t < 30 and (ids >= 0).mean() > 0.15
