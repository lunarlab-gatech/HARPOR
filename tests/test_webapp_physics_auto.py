import json
import re

import numpy as np
import pytest
import trimesh
from PIL import Image

from harpor import frames
from harpor.physics.align import read_glb, write_glb
from harpor.webapp import physics_auto as A

H, W = 120, 160
K = np.array([[100.0, 0, 79.5], [0, 100.0, 59.5], [0, 0, 1]])


def rot(axis, deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    i, j = [(1, 2), (2, 0), (0, 1)][axis]
    R = np.eye(3)
    R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
    return R


TILT = rot(2, 10) @ rot(0, 20)
UP = TILT @ np.array([0.0, -1.0, 0.0])
DA3_WORLD = rot(1, 30) @ rot(2, -40)  # W from DA3's world: view 0's DA3 camera is not at the identity


def half_turn(axis):
    """The 180 deg rotation about the unit axis: a camera on the far side of an orbit around it."""
    return 2 * np.outer(axis, axis) - np.eye(3)


def plane_view(up=UP, height=0.6):
    """View 0's pointmap of a plane `height` below the camera with normal up; a box-shaped mask in the
    lower middle where the object stands 10% nearer than the plane."""
    v, u = np.mgrid[0:H, 0:W]
    r = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones((H, W))], -1)
    t = -height / (r @ up)
    P = np.where((t > 0)[..., None], r * t[..., None], r * 50.0)  # above the horizon: far away
    mask = np.zeros((H, W), bool)
    mask[70:95, 60:100] = True
    P[mask] *= 0.9
    return P.astype(np.float32), mask


def write_run(run, P, mask, view1=np.eye(3)):
    """A run of views 0 and 1; view1 is view 1's camera-to-W rotation (view 0's is the identity)."""
    (run / "masks").mkdir(parents=True)
    Image.fromarray((mask * 255).astype(np.uint8)).resize((2 * W, 2 * H), Image.NEAREST).save(run / "masks" / "0.png")
    files = [str(run / "input" / "images" / f"{i}.png") for i in (1, 0)]  # view 0 second in DA3's order
    cams = ((view1, np.array([1.0, 0, 3])), (np.eye(3), np.zeros(3)))  # camera-to-W rotation and center
    E = [np.hstack([C.T @ DA3_WORLD, -C.T @ c[:, None]]) for C, c in cams]
    np.savez(run / "da3_output.npz", pointmaps=np.stack([P * 2, P]), conf=np.ones((2, H, W), np.float32),
             depth=np.stack([P[..., 2] * 2, P[..., 2]]), extrinsics=np.array(E, np.float32),
             image_files=np.array(files), process_res=504)


def box_glb(path, center_w=(0.0, 0.4, 2.0), alpha=255, scale=1.0):
    """A 0.3 x 0.1 x 0.2 box (times scale) whose thin side lies along UP, in frame W, saved as glTF."""
    b = trimesh.creation.box(np.multiply((0.3, 0.1, 0.2), scale))
    x_w = b.vertices @ TILT.T + center_w
    mat = trimesh.visual.material.PBRMaterial(baseColorFactor=[200, 100, 50, alpha])
    trimesh.Trimesh(frames.w_to_gltf(x_w), b.faces, visual=trimesh.visual.TextureVisuals(material=mat)).export(path)
    return path


def test_up_hint_finds_a_tilted_plane(tmp_path):
    write_run(tmp_path / "run", *plane_view())
    up, info = A.up_hint(tmp_path / "run")
    assert info["source"] == "plane" and info["inlier_share"] > 0.9 and info["object_above_share"] > 0.9
    assert np.degrees(np.arccos(up @ UP)) < 0.5 and up[1] < 0
    assert np.allclose(info["reference_up"], [0, -1, 0], atol=1e-5)
    assert info["angle_to_reference_deg"] == pytest.approx(np.degrees(np.arccos(-UP[1])), abs=0.5)


def test_up_hint_takes_a_table_seen_steeply_from_above(tmp_path):
    steep = rot(0, 60) @ np.array([0.0, -1.0, 0.0])  # view 0 looks 60 deg down at the table
    write_run(tmp_path / "steep", *plane_view(up=steep))
    up, info = A.up_hint(tmp_path / "steep")
    assert info["source"] == "plane" and np.degrees(np.arccos(up @ steep)) < 0.5
    assert info["angle_to_reference_deg"] == pytest.approx(60, abs=0.5)
    # Beyond 45 deg the plane must hold half of the points: here clutter covers 60% of the band.
    for normal, source in ((steep, "cameras"), (UP, "plane")):  # within 45 deg, 30% is enough
        P, mask = plane_view(up=normal)
        P[:, :90] *= np.random.default_rng(0).uniform(0.5, 0.8, (H, 90, 1))
        write_run(tmp_path / source, P, mask)
        info = A.up_hint(tmp_path / source)[1]
        assert 0.3 <= info["inlier_share"] < 0.5 and info["source"] == source
    assert "60 deg from the cameras' up and holds" in A.up_hint(tmp_path / "cameras")[1]["reason"]


def test_up_hint_uses_the_cameras_mean_up(tmp_path):
    g = rot(0, 55) @ np.array([0.0, -1.0, 0.0])  # gravity's up in W; the cameras orbit it, 55 deg down
    write_run(tmp_path / "noise", np.random.default_rng(0).uniform(0.5, 3, (H, W, 3)), plane_view()[1], half_turn(g))
    up, info = A.up_hint(tmp_path / "noise")
    assert info["source"] == "cameras" and "holds only" in info["reason"] and np.allclose(up, g, atol=1e-5)
    assert np.allclose(info["reference_up"], g, atol=1e-4)
    g = rot(0, 80) @ np.array([0.0, -1.0, 0.0])  # 80 deg from view 0's up, but the orbit's mean up
    write_run(tmp_path / "orbit", *plane_view(up=g, height=1.0), view1=half_turn(g))
    up, info = A.up_hint(tmp_path / "orbit")
    assert info["source"] == "plane" and info["angle_to_reference_deg"] < 1


def test_up_hint_falls_back_to_the_cameras_up(tmp_path):
    write_run(tmp_path / "noise", np.random.default_rng(0).uniform(0.5, 3, (H, W, 3)), plane_view()[1])
    up, info = A.up_hint(tmp_path / "noise")
    assert np.allclose(up, [0, -1, 0], atol=1e-6) and info["source"] == "cameras" and "holds only" in info["reason"]
    wall = rot(0, 80) @ np.array([0.0, -1.0, 0.0])  # nearly facing the camera
    write_run(tmp_path / "wall", *plane_view(up=wall, height=1.0))
    up, info = A.up_hint(tmp_path / "wall")
    assert np.allclose(up, [0, -1, 0], atol=1e-6) and "80 deg from the cameras' up" in info["reason"]
    P, mask = plane_view()
    P[mask] *= 1.2  # the object is behind the plane: a wall in front of it, not its support
    write_run(tmp_path / "behind", P, mask)
    up, info = A.up_hint(tmp_path / "behind")
    assert info["source"] == "cameras" and info["object_above_share"] < 0.1 and "does not stand on it" in info["reason"]
    P, mask = plane_view()
    write_run(tmp_path / "empty", P, np.zeros_like(mask))
    assert "empty" in A.up_hint(tmp_path / "empty")[1]["reason"]


def test_needs_opaque(tmp_path):
    g = box_glb(tmp_path / "box.glb")
    assert not A.needs_opaque(g)
    assert A.needs_opaque(box_glb(tmp_path / "half.glb", alpha=128))
    doc, binary = read_glb(g)
    for patch in ({"extensions": {"KHR_materials_transmission": {"transmissionFactor": 1.0}}},
                  {"extensions": {"KHR_materials_ior": {"ior": 1.5}}}, {"alphaMode": "BLEND"}):
        d = json.loads(json.dumps(doc))
        d["materials"][0].update(patch)
        write_glb(tmp_path / "p.glb", d, binary)
        assert A.needs_opaque(tmp_path / "p.glb"), patch
    d = json.loads(json.dumps(doc))
    d["materials"][0]["alphaMode"] = "MASK"
    write_glb(tmp_path / "p.glb", d, binary)
    assert not A.needs_opaque(tmp_path / "p.glb")


def test_sim_name():
    cases = {"coffee cup": "coffee_cup", " Coffee  Cup! ": "coffee_cup", "3d printer": "obj_3d_printer",
             "scene": "obj_scene", "World": "obj_world", "  ": "obj", "café mug": "caf_mug", "x-ray": "x_ray"}
    for prompt, name in cases.items():
        assert A.sim_name(prompt) == name and re.fullmatch(r"[A-Za-z_]\w*", name, re.ASCII)


def test_align_job_picks_the_glb_and_aligns_it(tmp_path, capsys):
    j = tmp_path / "job"
    write_run(j / "run", *plane_view())
    box_glb(j / "run" / "posed.glb")
    (j / "run" / "placement.json").write_text(json.dumps({"check": {"mean_mask_iou": 0.8}}))
    A.main(["align", "--job", str(j)])
    up = json.loads((j / "physics" / "up.json").read_text())
    assert up["glb"] == str(j / "run" / "posed.glb") and up["glb_rule"] == "no agent result"
    assert up["info"]["source"] == "plane" and not up["opaque"]
    rep = json.loads((j / "physics" / "aligned.json").read_text())
    assert rep["up_near"]["axis"] in ("pca-thin", "normals") and rep["up_near"]["angle_deg"] < 1 and rep["x_rule"] == "footprint"
    assert np.allclose(rep["extent_s"], [0.3, 0.2, 0.1], atol=1e-3) and not rep["opaque"]

    # The agent's delivery wins, unless placement was too poor for the agent to run.
    (j / "agent" / "s" / "work" / "exports").mkdir(parents=True)
    box_glb(j / "agent" / "s" / "work" / "exports" / "001.glb", alpha=128)
    result = {"source": "finish", "verdict": "complete", "glb": "work/exports/001.glb"}
    (j / "agent" / "result.json").write_text(json.dumps(result))
    A.main(["align", "--job", str(j)])
    up = json.loads((j / "physics" / "up.json").read_text())
    assert up["glb"] == str(j / "agent" / "s" / "work" / "exports" / "001.glb") and up["opaque"]
    assert "finish GLB, verdict complete" in up["glb_rule"]
    assert json.loads((j / "physics" / "aligned.json").read_text())["opaque"]
    for r in ({**result, "verdict": "fallback_original"}, {**result, "glb": None}, {**result, "glb": "../../run/posed.glb"},
              {"verdict": "needs_review", "glb": None}):
        (j / "agent" / "result.json").write_text(json.dumps(r))
        g, rule = A.pick_glb(j)
        assert g == j / "run" / "posed.glb" and "no usable GLB" in rule
    (j / "agent" / "result.json").unlink()
    (j / "run" / "placement.json").write_text(json.dumps({"check": {"mean_mask_iou": 0.3}}))
    g, rule = A.pick_glb(j)
    assert g == j / "run" / "posed.glb" and "below 0.5" in rule


def test_align_job_refuses_an_agent_glb_of_another_size(tmp_path, capsys):
    """The agent's GLB goes to physics only when it keeps the size of the GLB the agent started from (posed.glb,
    or parts.glb for an articulated job; controller.check_size). Else physics uses that GLB."""
    from harpor.agent.fake_agent import write_glb

    j = tmp_path / "job"
    write_run(j / "run", *plane_view())
    box_glb(j / "run" / "posed.glb")
    (j / "run" / "placement.json").write_text(json.dumps({"metric_scale": 1.25, "check": {"mean_mask_iou": 0.9}}))
    exp = j / "agent" / "s" / "work" / "exports"
    exp.mkdir(parents=True)
    (j / "agent" / "result.json").write_text(json.dumps({"source": "finish", "verdict": "complete",
                                                         "glb": "work/exports/001.glb"}))
    box_glb(exp / "001.glb", scale=10)  # a unit slip in Blender
    A.align_job(j)
    up = json.loads((j / "physics" / "up.json").read_text())
    assert up["glb"] == str(j / "run" / "posed.glb")
    assert up["glb_rule"] == ("the agent's finish GLB is not used: wrong size: every side of its box is at least 10 "
                              "times the original's (over 2.0)")
    assert "is not used: wrong size" in capsys.readouterr().out
    assert np.allclose(json.loads((j / "physics" / "aligned.json").read_text())["extent_s"], [0.3, 0.2, 0.1], atol=1e-3)
    for scale, used in ((0.45, False), (0.55, True), (1.9, True), (2.1, False)):
        box_glb(exp / "001.glb", scale=scale)
        assert (A.pick_glb(j)[0] == exp / "001.glb") == used, scale

    # An articulated job compares with the split parts, and falls back to them.
    (j / "job.json").write_text(json.dumps({"articulated": True}))
    write_glb(j / "run" / "parts.glb", "parts", 0.2)
    write_glb(exp / "001.glb", "parts", 2.0)
    (j / "agent" / "result.json").write_text(json.dumps({"source": "finish", "verdict": "articulated",
                                                         "glb": "work/exports/001.glb"}))
    g, rule = A.pick_glb(j)
    assert g == j / "run" / "parts.glb" and "wrong size: every side of its box is at least 10 times" in rule
    write_glb(exp / "001.glb", "parts", 0.2)
    assert A.pick_glb(j)[0] == exp / "001.glb"


def test_align_job_needs_a_glb(tmp_path):
    write_run(tmp_path / "job" / "run", *plane_view())
    with pytest.raises(SystemExit, match="posed.glb is missing"):
        A.main(["align", "--job", str(tmp_path / "job")])


def test_align_job_writes_the_links_of_an_articulated_job(tmp_path):
    """An articulated job aligns its split parts (no agent result) or the agent's GLB with joints, and
    writes one GLB per link for VoMP; the joints go to frame S. A rigid job writes no links."""
    from harpor.agent.fake_agent import write_glb
    from harpor.parts.joints import mesh_nodes

    j = tmp_path / "job"
    write_run(j / "run", *plane_view())
    box_glb(j / "run" / "posed.glb")
    (j / "job.json").write_text(json.dumps({"articulated": True}))
    write_glb(j / "run" / "parts.glb", "parts", 0.2)
    A.main(["align", "--job", str(j)])
    up = json.loads((j / "physics" / "up.json").read_text())
    assert up["glb"] == str(j / "run" / "parts.glb") and up["glb_rule"] == "no agent result"
    rep = json.loads((j / "physics" / "aligned.json").read_text())
    assert (rep["joints"], rep["links"], rep["fixed_base"]) == ({}, [], False)
    links = sorted((j / "physics" / "links").glob("*.glb"))
    assert [p.stem for p in links] == ["body", "lid"]
    assert [mesh_nodes(read_glb(p)[0]) for p in links] == [["body"], ["lid"]]

    hinge = {"version": 1, "frame": "W", "parent": "body", "type": "revolute", "origin": [0.0, -0.05, 0.1],
             "axis": [1.0, 0.0, 0.0], "lower": 0.0, "upper": 1.0}
    (j / "agent" / "s" / "work" / "exports").mkdir(parents=True)
    write_glb(j / "agent" / "s" / "work" / "exports" / "001.glb", "parts", 0.2, joints={"lid": hinge})
    (j / "agent" / "result.json").write_text(json.dumps({"source": "finish", "verdict": "articulated",
                                                         "glb": "work/exports/001.glb"}))
    A.main(["align", "--job", str(j)])
    up = json.loads((j / "physics" / "up.json").read_text())
    assert up["glb"].endswith("work/exports/001.glb") and "verdict articulated" in up["glb_rule"]
    rep = json.loads((j / "physics" / "aligned.json").read_text())
    T = np.array(rep["T_s_from_w"])
    assert set(rep["joints"]) == {"lid"} and rep["joints"]["lid"]["frame"] == "S" and rep["links"] == ["body", "lid"]
    assert np.allclose(rep["joints"]["lid"]["origin"], T[:3, :3] @ hinge["origin"] + T[:3, 3])
    assert sorted(p.stem for p in (j / "physics" / "links").glob("*.glb")) == ["body", "lid"]

    # The agent stage skipped (no result): the split parts again. A rigid job writes no links.
    (j / "agent" / "result.json").unlink()
    assert A.pick_glb(j) == (j / "run" / "parts.glb", "no agent result")
    (j / "job.json").write_text(json.dumps({"articulated": False}))
    (j / "physics" / "links").rename(tmp_path / "old_links")
    A.main(["align", "--job", str(j)])
    assert json.loads((j / "physics" / "up.json").read_text())["glb"] == str(j / "run" / "posed.glb")
    assert not (j / "physics" / "links").exists()
