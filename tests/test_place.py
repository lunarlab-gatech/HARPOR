import numpy as np
from scipy.spatial.transform import Rotation

from gator.stage_a.place import GLB_TURN, P3D_TO_CV, canonical_glb_to_w, fit_intrinsics, quat_wxyz_to_matrix


def test_quaternion_matches_scipy():
    q_xyzw = Rotation.random(random_state=0).as_quat()
    q_wxyz = np.r_[q_xyzw[3], q_xyzw[:3]]
    assert np.allclose(quat_wxyz_to_matrix(q_wxyz), Rotation.from_quat(q_xyzw).as_matrix())


def test_identity_pose_only_undoes_glb_turn_and_flips_axes():
    V = np.random.default_rng(1).normal(size=(10, 3))
    pose = {"rotation_wxyz": [1, 0, 0, 0], "translation": [0, 0, 2], "scale": [0.5]}
    expected = ((V @ GLB_TURN.T) * 0.5 + [0, 0, 2]) * P3D_TO_CV
    assert np.allclose(canonical_glb_to_w(V, pose), expected)


def test_row_vector_rotation_convention():
    # SAM 3D rotates row vectors as p @ R, so a +90 deg turn about z sends x to -y.
    R = Rotation.from_euler("z", 90, degrees=True)
    q = R.as_quat()
    pose = {"rotation_wxyz": [q[3], q[0], q[1], q[2]], "translation": [0, 0, 0], "scale": [1]}
    V = np.array([[1.0, 0.0, 0.0]]) @ np.linalg.inv(GLB_TURN.T)  # canonical (1, 0, 0) after the undo
    assert np.allclose(canonical_glb_to_w(V, pose), np.array([[0.0, -1.0, 0.0]]) * P3D_TO_CV)


def test_fit_intrinsics_recovers_k():
    W, H, h, w = 640, 480, 120, 160
    fx, fy, cx, cy = 500.0, 510.0, 321.0, 238.0
    ii, jj = np.mgrid[0:h, 0:w]
    u = (jj + 0.5) * (W / w) - 0.5
    v = (ii + 0.5) * (H / h) - 0.5
    z = 2.0 + 0.1 * np.sin(ii)
    pm = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], -1)
    K, rms = fit_intrinsics(pm, (W, H))
    assert np.allclose(K, [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], atol=1e-6) and rms < 1e-6
