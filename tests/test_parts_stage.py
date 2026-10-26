"""The parts stage on a hand-built run folder: propose (the user's list), part_masks with a stand-in for
SAM 3, vote, split with the fake Blender of gator.agent.fake_agent, and finish."""

import json
import shlex
import sys

import numpy as np
import pytest
import trimesh
from PIL import Image

from gator import frames
from gator.agent import tools
from gator.parts import render
from gator.parts import stage as S
from gator.parts import vote as V
from gator.physics.align import read_glb, write_glb
from gator.stage_a import part_masks
from test_parts_vote import CENTER, look_at

FAKE_BLENDER = [sys.executable, "-m", "gator.agent.fake_agent", "--blender"]
W, H = 200, 150
K = np.array([[220.0, 0, 99.5], [0, 220.0, 74.5], [0, 0, 1]])


class LidSegmenter:
    """Stands in for SAM 3: finds the lid (the faces above 75% of the box's height) in every photo,
    which it knows by the photo's colour."""

    info = {"model": "fake", "device": "cpu"}

    def __init__(self, ids, lid):
        self.ids, self.lid = ids, lid

    def __call__(self, image, prompts, threshold=0.5, mask_threshold=0.5):
        m = np.isin(self.ids[int(image[0, 0, 0]) - 90], self.lid)
        r, c = np.nonzero(m)
        return {p: (np.array([0.9]), np.array([[c.min(), r.min(), c.max() + 1, r.max() + 1]], float), m[None])
                for p in prompts}

    def peak_gib(self):
        return None


@pytest.fixture
def job(tmp_path):
    """A job of a 40 x 30 x 30 cm textured box seen from six cameras above it, with the user's list
    "box body" and "lid", through part_masks."""
    J = tmp_path / "job"
    run = J / "run"
    for d in ("masks", "input/images"):
        (run / d).mkdir(parents=True)
    box = trimesh.creation.box(extents=(0.4, 0.3, 0.3))
    for _ in range(3):
        box = box.subdivide()
    uv = (box.vertices[:, :2] - box.vertices[:, :2].min(0)) / np.ptp(box.vertices[:, :2], 0)
    tex = Image.fromarray(np.full((8, 8, 3), (180, 120, 70), np.uint8))
    visual = trimesh.visual.TextureVisuals(uv=uv, material=trimesh.visual.material.PBRMaterial(baseColorTexture=tex))
    trimesh.Trimesh(frames.w_to_gltf(box.vertices + CENTER), box.faces, visual=visual,
                    process=False).export(run / "posed.glb")
    x_w, faces = V.load_mesh(run / "posed.glb")
    lid = np.flatnonzero(x_w[faces].mean(1)[:, 1] < -0.075)
    cams, maps = [], []
    for k, az in enumerate(range(0, 360, 60)):
        a, e = np.radians(az), np.radians(35)
        T = look_at(CENTER + 1.3 * np.array([np.sin(a) * np.cos(e), -np.sin(e), -np.cos(a) * np.cos(e)]))
        ids = render.face_ids(x_w, faces, K, T, (W, H))
        maps.append(ids)
        Image.fromarray(np.where(ids >= 0, 255, 0).astype(np.uint8)).save(run / "masks" / f"{k}.png")
        Image.fromarray(np.full((H, W, 3), (90 + k, 100, 120), np.uint8)).save(run / "input" / "images" / f"{k}.png")
        cams.append({"name": str(k), "K": K.tolist(), "w2c": T.tolist(), "width": W, "height": H,
                     "image": str(run / "input" / "images" / f"{k}.png"), "mask": str(run / "masks" / f"{k}.png")})
    (run / "cameras.json").write_text(json.dumps(cams))
    (J / "job.json").write_text(json.dumps({"id": "x", "prompt": "box", "agent": "none", "parts": {
        "mode": "user", "list": [{"name": "box body"}, {"name": "lid", "parent": "box body", "joint": "revolute",
                                                         "axis_hint": "hinge on the back edge"}], "confirmed": True}}))
    S.main(["propose", "--job", str(J)])
    part_masks.run(run / "input" / "images", ["lid"], run / "parts" / "sam3", segmenter=LidSegmenter(maps, lid))
    return J, lid


def test_vote_split_finish(job, capsys):
    J, lid = job
    run = J / "run"
    S.main(["vote", "--job", str(J)])
    S.main(["split", "--job", str(J), "--blender", shlex.join(FAKE_BLENDER)])
    assert (run / "parts" / "split.log").exists() and not (run / "parts.json").exists()
    S.main(["finish", "--job", str(J)])
    out = capsys.readouterr().out
    assert "vote: 768 faces, 2 parts (box_body, lid)" in out and "lid > box_body : revolute" in out

    doc = json.loads((run / "parts.json").read_text())
    assert doc["version"] == 1 and doc["frame"] == "W" and doc["base"] == "box_body" and doc["noun"] == "box"
    assert doc["glb_sha256"] == tools.sha256(run / "parts.glb") and doc["source"] == "user"
    body, top = doc["parts"]
    assert (body["name"], body["role"], body["parent"], body["joint_guess"]) == ("box_body", "base", None, "fixed")
    assert (top["name"], top["prompt"], top["parent"], top["joint_guess"]) == ("lid", "lid", "box_body", "revolute")
    assert top["axis_hint"] == "hinge on the back edge" and top["status"] == "found" and top["cap_faces"] == 0
    assert top["faces"] == len(lid) and body["faces"] == 768 - len(lid) and top["confidence"] > 0.9
    assert np.allclose(top["aabb"]["min"], [-0.2, -0.15, 1.85]) and np.allclose(top["aabb"]["max"], [0.2, -0.075, 2.15])
    assert np.allclose(sorted(top["obb"]["extents"]), [0.075, 0.3, 0.4]) and np.allclose(top["obb"]["center"], [0, -0.1125, 2])
    assert set(top["visibility"]) == {str(k) for k in range(6)} and all(v > 0 for v in top["visibility"].values())
    (seam,) = doc["seams"]
    assert seam["parts"] == ["box_body", "lid"] and np.isclose(seam["length"], 1.4) and len(seam["lines"]) == 4
    assert sorted(round(line["length"], 3) for line in seam["lines"]) == [0.3, 0.3, 0.4, 0.4]
    assert all(np.isclose(line["p0"][1], -0.075) for line in seam["lines"])
    nodes, _ = read_glb(run / "parts.glb")
    assert sorted(n["name"] for n in nodes["nodes"] if "mesh" in n) == ["box_body", "lid"]


def test_finish_refuses_a_glb_without_the_part_nodes(job):
    J, _ = job
    run = J / "run"
    S.main(["vote", "--job", str(J)])
    S.split(J, blender=FAKE_BLENDER)
    good = run.joinpath("parts.glb").read_bytes()
    m = trimesh.load(run / "parts.glb", force="mesh", process=False)
    m.export(run / "parts.glb")  # one unnamed mesh
    with pytest.raises(SystemExit, match="not the parts"):
        S.finish(J)
    assert not (run / "parts.json").exists()
    run.joinpath("parts.glb").write_bytes(good)
    doc, binary = read_glb(run / "parts.glb")
    del next(n for n in doc["nodes"] if n.get("name") == "lid")["name"]
    write_glb(run / "parts.glb", doc, binary)
    with pytest.raises(SystemExit, match=r"nodes \['None', 'box_body'\], not the parts"):
        S.finish(J)
    assert not (run / "parts.json").exists()


def test_a_failed_split_names_its_log(job):
    J, _ = job
    S.main(["vote", "--job", str(J)])
    with pytest.raises(SystemExit, match="split.log"):
        S.split(J, blender=[sys.executable, "-c", "import sys; sys.exit(3)", "--"])
    with pytest.raises(SystemExit, match="--sif or --blender"):
        S.split(J)


def test_split_runs_blender_in_a_container_that_sees_only_the_run(tmp_path):
    cmd = S.split_command(tmp_path / "job", "/scratch/gator/sif/gator-agent.sif")
    run = str((tmp_path / "job" / "run").resolve())
    assert cmd[:2] == ["apptainer", "exec"] and cmd[-2:] == ["/scratch/gator/sif/gator-agent.sif", "blender"]
    binds = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--bind"]
    assert binds == [f"{run}:{run}", f"{tools.BLENDER_SCRIPTS}:{tools.BLENDER_SCRIPTS}:ro"]


def test_parents_follow_the_longest_seam_and_skip_parts_not_found():
    proposal = {"base": "body", "parts": [{"name": "body", "parent": None}, {"name": "door", "parent": "body"},
                                          {"name": "shelf", "parent": "body"}, {"name": "knob", "parent": "door"},
                                          {"name": "handle", "parent": "shelf"}]}
    clusters = {"body": {"role": "base", "prompt": "body"}, "door_1": {"role": "part", "prompt": "door"},
                "door_2": {"role": "part", "prompt": "door"}, "knob": {"role": "part", "prompt": "knob"},
                "handle": {"role": "part", "prompt": "handle"}}
    names = ["body", "door_1", "door_2", "knob", "handle"]
    seams = [{"parts": ["door_1", "knob"], "length": 0.1}, {"parts": ["door_2", "knob"], "length": 0.3}]
    assert S.parents(clusters, proposal, names, seams) == {"body": None, "door_1": "body", "door_2": "body",
                                                           "knob": "door_2", "handle": "body"}
