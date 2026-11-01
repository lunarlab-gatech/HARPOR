import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from test_physics import fix_to_world, hull_parts, inflated_hull, two_box_hinge, two_lid_chest

from harpor import frames


def xyz(element):
    return [float(c) for c in element.find("origin").get("xyz").split()]


def test_urdf_of_a_hinged_box(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    from harpor.physics.mjcf import build as build_mjcf
    from harpor.physics.urdf import build, check

    glb, summary, hinge, _ = two_box_hinge(tmp_path)
    s = json.loads(summary.read_text())
    build_mjcf(glb, summary, tmp_path / "mjcf", "chest", decompose=hull_parts)
    report = build(glb, summary, tmp_path / "mjcf", tmp_path / "urdf", "chest")
    c = check(tmp_path / "urdf")
    assert (c["links"], c["joints"], c["joint_types"], c["missing"]) == (2, 1, {"lid_joint": "revolute"}, [])
    assert report["root"] == "body" and not report["fixed_base"]

    robot = ET.parse(tmp_path / "urdf" / "model.urdf").getroot()
    lid = robot.find("link[@name='lid']")
    origin = np.array(hinge["origin"])
    assert np.allclose([float(c) for c in lid.find("visual/origin").get("xyz").split()], -origin)
    assert np.allclose([float(c) for c in lid.find("collision/origin").get("xyz").split()], -origin)
    com = np.array(s["links"]["lid"]["rigid_body"]["com"])
    assert np.allclose([float(c) for c in lid.find("inertial/origin").get("xyz").split()], com - origin, atol=1e-6)
    assert np.isclose(float(lid.find("inertial/mass").get("value")), s["links"]["lid"]["rigid_body"]["mass"])
    joint = robot.find("joint[@name='lid_joint']")
    assert (joint.find("parent").get("link"), joint.find("child").get("link")) == ("body", "lid")
    assert np.allclose([float(c) for c in joint.find("origin").get("xyz").split()], origin)
    assert np.allclose([float(c) for c in joint.find("axis").get("xyz").split()], hinge["axis"])
    assert (joint.find("limit").get("lower"), joint.find("limit").get("upper")) == ("0", "1.5")

    # The lid's OBJ holds S coordinates (its visual origin and its link frame cancel), and its material
    # keeps the texture.
    mesh = lid.find("visual/geometry/mesh").get("filename")
    m = trimesh.load(tmp_path / "urdf" / mesh, force="mesh")
    assert np.allclose(m.bounds, [[-0.3, -0.2, 0.3], [0.3, 0.2, 0.34]], atol=1e-6)
    mtl = (tmp_path / "urdf" / mesh).with_suffix(".mtl").read_text()
    assert "map_Kd texture_" in mtl and m.visual.material.image is not None


def test_urdf_of_a_rigid_box_and_of_a_fixed_base(tmp_path):
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

    glb, summary, _, _ = two_box_hinge(tmp_path / "chest")
    fix_to_world(glb)
    mj = build_mjcf(glb, summary, tmp_path / "chest" / "mjcf", "chest", decompose=hull_parts)
    report = build(glb, summary, tmp_path / "chest" / "mjcf", tmp_path / "chest" / "urdf", "chest")
    c = check(tmp_path / "chest" / "urdf")
    assert report["fixed_base"] and c["links"] == 3 and c["joint_types"] == {"world_joint": "fixed", "lid_joint": "revolute"}
    # The base stands at the MJCF's start height, as in MJCF and USD.
    world = ET.parse(tmp_path / "chest" / "urdf" / "model.urdf").getroot().find("joint[@name='world_joint']")
    assert np.allclose(xyz(world), [0, 0, mj["start_z"]]) and mj["start_z"] > 0


def test_urdf_welds_a_loose_part_to_its_parent_node(tmp_path):
    pytest.importorskip("trimesh")
    from harpor.physics.mjcf import build as build_mjcf
    from harpor.physics.urdf import build, check

    glb, summary = two_lid_chest(tmp_path)
    mj = build_mjcf(glb, summary, tmp_path / "mjcf", "chest", decompose=inflated_hull)
    report = build(glb, summary, tmp_path / "mjcf", tmp_path / "urdf", "chest")
    c = check(tmp_path / "urdf")
    assert c["joint_types"] == {"world_joint": "fixed", "lid_left_joint": "revolute", "lid_right_joint": "revolute",
                                "knob_joint": "fixed"} and c["links"] == 5 and c["missing"] == []
    knob = ET.parse(tmp_path / "urdf" / "model.urdf").getroot().find("joint[@name='knob_joint']")
    assert knob.find("parent").get("link") == "lid_left"
    assert np.allclose(xyz(knob), [0.15, -0.2, -0.3])  # the knob's frame (S's origin) seen from lid_left's hinge
    assert report["excluded_pairs"] == mj["excluded_pairs"]
