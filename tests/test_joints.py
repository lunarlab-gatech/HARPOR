import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from harpor import frames
from harpor.parts import joints as J
from harpor.physics.align import read_glb, write_glb

HINGE = {"version": 1, "frame": "W", "parent": "body", "type": "revolute", "origin": [0.0, -0.3, 0.2],
         "axis": [-1.0, 0.0, 0.0], "lower": 0.0, "upper": 1.5}


def doc_with(*names):
    return {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": list(range(len(names)))}],
            "nodes": [{"name": n, "mesh": i} for i, n in enumerate(names)]}


def test_joints_round_trip_through_a_glb(tmp_path):
    doc = doc_with("body", "lid", "drawer")
    doc["nodes"][1]["extras"] = {"note": "kept"}
    slide = {"version": 1, "frame": "W", "parent": "body", "type": "prismatic", "origin": [0, 0, 0],
             "axis": [0, 0, -1], "lower": 0, "upper": 0.3}
    out = J.write_joints(doc, {"lid": HINGE, "drawer": slide})
    assert "extras" not in doc["nodes"][2] and isinstance(out["nodes"][1]["extras"][J.KEY], str)
    write_glb(tmp_path / "a.glb", out, b"")
    joints, links = J.load(tmp_path / "a.glb")
    assert joints == {"lid": HINGE, "drawer": slide} and links == ["body", "lid", "drawer"]
    assert read_glb(tmp_path / "a.glb")[0]["nodes"][1]["extras"]["note"] == "kept"
    # A reader accepts a dict too, and writing an empty set removes every joint.
    doc["nodes"][2]["extras"] = {J.KEY: slide}
    assert J.read_joints(doc) == {"drawer": slide}
    assert J.read_joints(J.write_joints(out, {})) == {} and J.write_joints(out, {})["nodes"][1]["extras"] == {"note": "kept"}
    with pytest.raises(ValueError, match="no node is named 'door'"):
        J.write_joints(doc, {"door": HINGE})
    doc["nodes"][2]["extras"] = {J.KEY: "{not json"}
    with pytest.raises(ValueError, match="'drawer' is not valid JSON"):
        J.read_joints(doc)


def test_validate_accepts_a_tree_and_catches_each_fault():
    links = ["body", "lid", "drawer", "handle"]
    base = {"version": 1, "frame": "W", "parent": "world", "type": "fixed"}
    handle = {"version": 1, "frame": "W", "parent": "drawer", "type": "fixed"}
    good = {"lid": HINGE, "drawer": {**HINGE, "type": "prismatic", "axis": [0, 0, -1], "upper": 0.3},
            "body": base, "handle": handle}
    assert J.validate(good, links) == [] and J.validate({}, links) == []

    def problems(joints, names=links):
        return " | ".join(J.validate(joints, names))

    assert "no mesh node is named 'door'" in problems({"door": HINGE})
    assert "the parent 'base' is not a mesh node" in problems({"lid": {**HINGE, "parent": "base"}})
    assert "its own parent" in problems({"lid": {**HINGE, "parent": "lid"}})
    assert "2 roots (body, drawer)" in problems({"lid": HINGE, "handle": handle})
    assert "2 roots (body, lid)" in problems({"lid": {**HINGE, "parent": "world", "type": "fixed"}, "drawer": HINGE})
    assert "cycle: lid -> drawer -> lid" in problems({"lid": {**HINGE, "parent": "drawer"}, "drawer": {**HINGE, "parent": "lid"}})
    assert "unit vector" in problems({"lid": {**HINGE, "axis": [1, 1, 0]}})
    assert "finite numbers" in problems({"lid": {**HINGE, "origin": [0, float("nan"), 0]}})
    assert "finite numbers" in problems({"lid": {**HINGE, "upper": float("inf")}})
    assert "must hold 0" in problems({"lid": {**HINGE, "lower": 0.2}})
    assert "needs lower < upper" in problems({"lid": {**HINGE, "upper": 0.0}})  # MuJoCo refuses a zero range
    assert J.validate({"lid": {**HINGE, "type": "fixed", "upper": 0.0}}, links) == []
    assert "only a fixed joint" in problems({"body": {**HINGE, "parent": "world"}})
    assert "type must be" in problems({"lid": {**HINGE, "type": "ball"}})
    assert "version must be 1" in problems({"lid": {**HINGE, "version": 2}})
    assert "mix frames" in problems({"lid": HINGE, "drawer": {**HINGE, "frame": "S"}})
    assert "share a name: lid" in problems({}, ["body", "lid", "lid"])
    assert "reserved" in problems({}, ["world", "lid"])


def test_tree_finds_the_root_children_and_loose_links():
    links = ["lid", "body", "drawer", "knob"]
    joints = {"lid": HINGE, "drawer": {**HINGE, "type": "prismatic"}}
    root, children, fixed = J.tree(joints, links)
    assert (root, fixed) == ("body", False) and children["body"] == ["lid", "drawer"] and children["lid"] == []
    assert J.loose(joints, links, root) == ["knob"]
    joints["body"] = {"version": 1, "frame": "W", "parent": "world", "type": "fixed"}
    assert J.tree(joints, links)[0::2] == ("body", True)
    assert J.tree({}, links) == ("lid", {k: [] for k in links}, False)
    with pytest.raises(ValueError, match="2 roots"):
        J.tree({"lid": HINGE, "knob": {**HINGE, "parent": "drawer"}}, links)


def test_a_loose_link_is_welded_to_its_nearest_mesh_ancestor():
    # tray sits below drawer through an empty node; shelf is below no mesh node.
    doc = doc_with("body", "drawer", "tray", "shelf")
    doc["nodes"].append({"name": "group", "children": [2]})
    doc["nodes"][1]["children"] = [4]
    doc["scenes"][0]["nodes"] = [0, 1, 3]
    assert J.holders(doc) == {"tray": "drawer"}
    joints = {"drawer": {**HINGE, "type": "prismatic"}}
    assert J.welds(joints, J.mesh_nodes(doc), "body", J.holders(doc)) == {"tray": "drawer", "shelf": "body"}
    assert J.welds(joints, J.mesh_nodes(doc), "body") == {"tray": "body", "shelf": "body"}


def test_transform_moves_origin_and_axis_into_s():
    T = frames.rigid(Rotation.from_euler("xyz", [30, -20, 75], degrees=True).as_matrix(), [0.1, 0.2, 0.3])
    fixed = {"version": 1, "frame": "W", "parent": "body", "type": "fixed"}
    out = J.transform({"lid": HINGE, "knob": fixed}, T)
    assert out["lid"]["frame"] == "S" and list(out["lid"]) == list(HINGE)
    assert np.allclose(out["lid"]["origin"], frames.transform_points(T, HINGE["origin"]))
    assert np.allclose(out["lid"]["axis"], T[:3, :3] @ HINGE["axis"])
    assert (out["lid"]["lower"], out["lid"]["upper"]) == (0.0, 1.5) and HINGE["frame"] == "W"
    assert np.allclose(out["knob"]["origin"], T[:3, 3]) and out["knob"]["lower"] == out["knob"]["upper"] == 0
    assert J.validate(out, ["body", "lid", "knob"]) == []


def test_joint_transform_moves_a_point():
    # The lid's front corner, 0.4 m in front of a hinge along -x (W: y down, z away from the camera): a
    # quarter turn lifts it straight up.
    p = np.array([0.0, -0.3, -0.2])
    T = J.joint_transform(HINGE, np.pi / 2)
    assert np.allclose(frames.transform_points(T, p), [0.0, -0.7, 0.2])
    assert np.allclose(frames.transform_points(T, HINGE["origin"]), HINGE["origin"])  # the hinge stays
    slide = {**HINGE, "type": "prismatic", "axis": [0.0, 0.0, -1.0]}
    assert np.allclose(frames.transform_points(J.joint_transform(slide, 0.25), p), p + [0, 0, -0.25])
    assert np.allclose(J.joint_transform({**HINGE, "type": "fixed"}, 1.0), np.eye(4))


def test_axis_rotation_takes_x_onto_the_axis():
    for axis in ([1, 0, 0], [-1, 0, 0], [0, 0, 1], [0.6, -0.8, 0], [1e-12 - 1, 0, 0], [0.3, 0.4, -np.sqrt(0.75)]):
        w, x, y, z = J.axis_rotation(axis)
        R = Rotation.from_quat([x, y, z, w]).as_matrix()
        a = np.asarray(axis, float)
        assert np.isclose(np.linalg.norm([w, x, y, z]), 1) and np.allclose(R @ [1, 0, 0], a / np.linalg.norm(a))
    assert np.allclose(J.axis_rotation([1, 0, 0]), [1, 0, 0, 0])


def test_joint_strings_are_plain_json():
    out = J.write_joints(doc_with("body", "lid"), {"lid": HINGE})
    assert json.loads(out["nodes"][1]["extras"][J.KEY]) == HINGE
