import json

import numpy as np
import pytest
import trimesh

from harpor import frames
from harpor.physics import material_transfer as M
from harpor.webapp import jobs, steps


def box(path, extent=(1, 1, 1), center=(0, 0, 0)):
    mesh = trimesh.creation.box(extent)
    mesh.vertices = frames.w_to_gltf(mesh.vertices + center)
    mesh.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial())
    mesh.export(path)
    return path


def prediction(center=(0, 0, 0), density=600):
    return {"centers": np.array([center], float), "density": np.array([density]),
            "youngs_modulus": np.array([2e9]), "poisson_ratio": np.array([0.3])}


def test_prediction_coordinates_include_placement_and_alignment():
    pose = {"rotation_wxyz": [0.5, 0.5, 0.5, 0.5], "translation": [1, 2, 3], "scale": [2, 3, 4]}
    T = frames.rigid(np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]), [2, 5, 7])
    points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    from harpor.stage_a.place import canonical_glb_to_w
    expected = frames.transform_points(T, canonical_glb_to_w(frames.w_to_gltf(points), pose))
    assert np.allclose(M.prediction_centers(points, pose, T), expected)
    assert np.allclose(np.linalg.norm(M.prediction_centers(points, pose, T) -
                                     M.prediction_centers([[0, 0, 0]], pose, T), axis=1), [2, 3, 4])


def test_final_geometry_recomputes_mass_and_inertia(tmp_path):
    small = box(tmp_path / "small.glb")
    big = box(tmp_path / "big.glb", (2, 2, 2), (3, 0, 0))
    original, predictions = {"config": "test"}, {"": prediction()}
    a = M.transfer(small, tmp_path / "a", original, predictions, np.eye(4))["rigid_body"]
    b = M.transfer(big, tmp_path / "b", original, predictions, np.eye(4))["rigid_body"]
    assert b["mass"] == pytest.approx(a["mass"] * 8)
    assert b["com"] == pytest.approx([3, 0, 0])
    assert np.array(b["inertia"]) == pytest.approx(np.array(a["inertia"]) * 32)


def test_touching_links_keep_their_own_materials(tmp_path):
    body = box(tmp_path / "body.glb", center=(0, 0, 0))
    lid = box(tmp_path / "lid.glb", center=(1, 0, 0))
    # Deliberately put both predictions at the same location: link identity wins.
    predictions = {"body": prediction(density=600), "lid": prediction(density=7800)}
    summary = M.transfer(body, tmp_path / "out", {"config": "test"}, predictions, np.eye(4), [body, lid])
    assert summary["links"]["lid"]["median"]["density_kg_m3"] == 7800
    assert summary["links"]["body"]["median"]["density_kg_m3"] == 600
    assert summary["links"]["lid"]["rigid_body"]["mass"] / summary["links"]["body"]["rigid_body"]["mass"] == pytest.approx(13)


def test_material_stage_branch_and_retry_preservation(tmp_path):
    assert steps.NAMES.index("vomp") == steps.NAMES.index("sam3d") + 1
    assert steps.NAMES.index("vomp_parts") == steps.NAMES.index("parts") + 1
    assert steps.NAMES.index("vomp_parts") < steps.NAMES.index("agent")
    for articulated in (False, True):
        (tmp_path / "job.json").write_text(json.dumps({"articulated": articulated}))
        assert bool(steps.skip("vomp", tmp_path)) is articulated
        assert bool(steps.skip("vomp_parts", tmp_path)) is not articulated
    for folder in ("run/vomp", "run/vomp_parts", "physics"):
        (tmp_path / folder).mkdir(parents=True)
        (tmp_path / folder / "summary.json").write_text("{}")
    jobs._clear(tmp_path, steps.NAMES[steps.NAMES.index("agent"):])
    assert steps.done("vomp", tmp_path) and steps.done("vomp_parts", tmp_path)
    assert not (tmp_path / "physics").exists()
    jobs._clear(tmp_path, steps.NAMES[steps.NAMES.index("parts"):])
    assert steps.done("vomp", tmp_path) and not steps.done("vomp_parts", tmp_path)
