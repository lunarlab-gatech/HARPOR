import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from test_physics import body_and_lid, hull_parts

from harpor import frames


def xyz(element):
    return [float(c) for c in element.find("origin").get("xyz").split()]


def test_urdf_of_a_textured_two_part_object(tmp_path):
    """The body and lid of one rigid GLB become one link in frame S: the summary's inertial, one OBJ with an object
    and material per MJCF visual (S coordinates, the lid's texture kept), and the MJCF's collision hulls."""
    trimesh = pytest.importorskip("trimesh")
    from harpor.physics.mjcf import build as build_mjcf
    from harpor.physics.urdf import build, check

    glb, summary = body_and_lid(tmp_path)
    rb = json.loads(summary.read_text())["rigid_body"]
    mj = build_mjcf(glb, summary, tmp_path / "mjcf", "chest", decompose=hull_parts)
    report = build(glb, summary, tmp_path / "mjcf", tmp_path / "urdf", "chest")
    c = check(tmp_path / "urdf")
    assert (c["links"], c["joints"], c["joint_types"], c["meshes"], c["missing"]) == (1, 0, {}, 2, [])
    (link,) = report["links"]
    assert (report["name"], link["name"], link["mesh"], link["collisions"]) == (
        "chest", "chest", "meshes/link_0.obj", ["meshes/collision_0.obj"])
    assert link["inertial"] == {k: rb[k] for k in ("mass", "com", "fullinertia")}
    assert json.loads((tmp_path / "urdf" / "urdf.json").read_text()) == report

    robot = ET.parse(tmp_path / "urdf" / "model.urdf").getroot()
    chest = robot.find("link[@name='chest']")
    assert xyz(chest.find("visual")) == [0, 0, 0] and xyz(chest.find("collision")) == [0, 0, 0]
    assert np.allclose(xyz(chest.find("inertial")), rb["com"], atol=1e-6)
    assert np.isclose(float(chest.find("inertial/mass").get("value")), rb["mass"])
    inertia = chest.find("inertial/inertia")
    assert np.allclose([float(inertia.get(k)) for k in ("ixx", "iyy", "izz")], rb["fullinertia"][:3], rtol=1e-5)

    # One object per visual, in S coordinates; only the lid's material maps its texture.
    lid = next(v["mesh"] for v in mj["visuals"] if v["name"] == "lid")
    scene = trimesh.load(tmp_path / "urdf" / "meshes" / "link_0.obj", force="scene")
    assert sorted(scene.geometry) == sorted(v["mesh"] for v in mj["visuals"])
    assert np.allclose(scene.bounds, [[-0.3, -0.2, 0], [0.3, 0.2, 0.34]], atol=1e-6)
    assert np.allclose(scene.geometry[lid].bounds, [[-0.3, -0.2, 0.3], [0.3, 0.2, 0.34]], atol=1e-6)
    assert scene.geometry[lid].visual.material.image.size == (16, 16)
    mtl = (tmp_path / "urdf" / "meshes" / "link_0.mtl").read_text()
    tex = next(v["texture"] for v in mj["visuals"] if v["name"] == "lid")
    assert mtl.count("map_Kd ") == 1 and f"map_Kd {tex}.png" in mtl and (tmp_path / "urdf" / "meshes" / f"{tex}.png").is_file()


def test_urdf_of_a_rigid_box(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    from harpor.physics.mjcf import build as build_mjcf
    from harpor.physics.urdf import build, check

    box = trimesh.creation.box([1.2, 0.8, 0.5])
    box.apply_translation([0, 0, 0.25])
    trimesh.Scene(trimesh.Trimesh(frames.w_to_gltf(box.vertices), box.faces)).export(tmp_path / "box.glb")
    rb = {"mass": 12.0, "com": [0.01, -0.02, 0.25], "fullinertia": [0.9, 1.5, 2.0, 0.05, -0.02, 0.01]}
    (tmp_path / "summary.json").write_text(json.dumps({"rigid_body": rb}))
    build_mjcf(tmp_path / "box.glb", tmp_path / "summary.json", tmp_path / "mjcf", "box", decompose=hull_parts)
    build(tmp_path / "box.glb", tmp_path / "summary.json", tmp_path / "mjcf", tmp_path / "urdf", "box")
    c = check(tmp_path / "urdf")
    assert (c["links"], c["joints"], c["meshes"], c["missing"]) == (1, 0, 2, [])
    link = ET.parse(tmp_path / "urdf" / "model.urdf").getroot().find("link[@name='box']")
    assert link.find("inertial/inertia").get("ixy") == "0.05" and link.find("inertial/origin").get("xyz") == "0.01 -0.02 0.25"
    assert link.find("collision/geometry/mesh").get("filename") == "meshes/collision_0.obj"
