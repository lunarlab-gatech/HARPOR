"""The parts stage on a hand-built run folder: propose (the user's list), part_masks with a stand-in for
SAM 3, vote, split with the fake Blender of harpor.agent.fake_agent, and finish."""

import json
import shlex
import shutil
import sys

import numpy as np
import pytest
import trimesh
from PIL import Image

from harpor import frames
from harpor.agent import tools
from harpor.parts import render
from harpor.parts import stage as S
from harpor.parts import vote as V
from harpor.physics import sim_check
from harpor.physics.align import read_glb, write_glb
from harpor.stage_a import part_masks
from test_parts_vote import CENTER, look_at

FAKE_BLENDER = [sys.executable, "-m", "harpor.agent.fake_agent", "--blender"]
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


def test_whole_makes_one_part_and_reserves_the_others(job, capsys):
    """A job that skips the split: whole puts every face in the base and reads no part mask, the split and finish
    run as they are, and parts.json has one part object and, for every other part of the list, a requirement with
    a reserve, which the agent cuts out. No seam, since nothing was cut."""
    J, _ = job
    run = J / "run"
    shutil.rmtree(run / "parts" / "sam3")  # whole needs no part masks
    S.main(["whole", "--job", str(J)])
    S.main(["split", "--job", str(J), "--blender", shlex.join(FAKE_BLENDER)])
    S.main(["finish", "--job", str(J)])
    out = capsys.readouterr().out
    assert "whole: 768 faces, all in box_body; the split is skipped, and the agent cuts lid" in out
    vote = json.loads((run / "parts" / "vote.json").read_text())
    assert (vote["names"], vote["settings"], vote["counts"]["instances"]) == (["box_body"], {"split": False}, 0)
    with np.load(run / "parts" / "faces.npz") as data:
        assert not data["part"].any() and len(data["part"]) == 768 and list(data["names"]) == ["box_body"]
    doc = json.loads((run / "parts.json").read_text())
    body, lost = doc["parts"]
    assert (body["name"], body["role"], body["faces"], body["status"], body["pieces"]) == ("box_body", "base", 768,
                                                                                         "found", 1)
    assert (lost["name"], lost["prompt"], lost["status"], lost["why"]) == (
        None, "lid", "not_found", "the split was skipped: the agent cuts the parts")
    (r,) = doc["requirements"]
    assert (r["part"], r["status"], r["reserve"], r["parent"], r["required"], r["joint"]) == (
        None, "not_found", "lid", "box_body", True, "revolute")
    assert doc["seams"] == [] and doc["warnings"][0] == "the split was skipped: the agent cuts the parts"
    nodes, _ = read_glb(run / "parts.glb")
    assert [n["name"] for n in nodes["nodes"] if "mesh" in n] == ["box_body"]


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
    assert doc["requirements"] == [  # the user's list: the default requirement, with no span target
        {"id": "r1", "part": "lid", "prompt": "lid", "parent": "box_body", "kind": "joint", "joint": "revolute",
         "text": "turns about its hinge or axle", "action": {"type": "sweep", "directions": ["lower", "upper"]},
         "observe": {"type": "joint_range", "relative_to": "parent"}, "test": {},
         "scope": {"start": "mesh pose, every joint at 0", "base": "fixed", "gravity": False},
         "source": "default", "status": "found", "required": True}]
    assert "r1 'lid': joint revolute, found\n" in out


def test_finish_takes_the_proposal_requirements_or_the_defaults(job):
    J, _ = job
    run = J / "run"
    S.main(["vote", "--job", str(J)])
    S.split(J, blender=FAKE_BLENDER)
    proposal = json.loads((run / "proposal.json").read_text())
    proposal["requirements"] = [{"part": "lid", "joint": "revolute", "text": "opens a quarter turn", "range": 90.0,
                                 "source": "vlm"}]
    (run / "proposal.json").write_text(json.dumps(proposal))
    (req,) = S.finish(J)["requirements"]
    assert (req["text"], req["test"], req["source"]) == ("opens a quarter turn", {"min_span": 1.571}, "vlm")
    del proposal["requirements"]  # a proposal older than requirements
    (run / "proposal.json").write_text(json.dumps(proposal))
    (req,) = S.finish(J)["requirements"]
    assert (req["part"], req["text"], req["test"], req["source"]) == ("lid", "turns about its hinge or axle", {}, "default")


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
    cmd = S.split_command(tmp_path / "job", "/scratch/gator/sif/harpor-agent.sif")
    run = str((tmp_path / "job" / "run").resolve())
    assert cmd[:2] == ["apptainer", "exec"] and cmd[-2:] == ["/scratch/gator/sif/harpor-agent.sif", "blender"]
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


def test_requirements_make_max_count_found_records():
    """3 drawers with 2 found: 2 found and 1 not_found, all required. 1 door with 2 found: the second is not
    required. A knob that was only dropped_small (not in clusters): not_found. Revolute range is a span target in
    radians, prismatic range only text; a soft part gets the press. A not_found record has a reserve, the name of
    the link an agent may create for it, and a link as its parent (the first found link of its proposal parent)."""
    proposal = {"base": "cabinet body", "parts": [
        {"name": "cabinet body", "parent": None, "joint": "fixed", "count": 1},
        {"name": "drawer", "parent": "cabinet body", "joint": "prismatic", "count": 3},
        {"name": "door", "parent": "cabinet body", "joint": "revolute", "count": 1},
        {"name": "knob", "parent": "door", "joint": "unknown", "count": 1},
        {"name": "pad", "parent": "door", "joint": "soft", "count": 1}],
        "requirements": [
            {"part": "drawer", "joint": "prismatic", "text": "pulls out", "range": 40.0, "source": "vlm"},
            {"part": "door", "joint": "revolute", "text": "swings open", "range": 360.0, "source": "vlm"},
            {"part": "knob", "joint": "unknown", "text": "turns", "range": 90.0, "source": "vlm"},
            {"part": "pad", "joint": "soft", "text": "dents", "material": "gel", "source": "vlm"}]}
    clusters = {"cabinet_body": {"prompt": "cabinet body"}, "drawer_1": {"prompt": "drawer"},
                "drawer_2": {"prompt": "drawer"}, "door_1": {"prompt": "door"}, "door_2": {"prompt": "door"},
                "pad": {"prompt": "pad"}}
    names = list(clusters)
    parent = {"cabinet_body": None, "drawer_1": "cabinet_body", "drawer_2": "cabinet_body", "door_1": "cabinet_body",
              "door_2": "cabinet_body", "pad": "door_1"}
    reqs = S.requirements(proposal, clusters, names, parent)
    assert [(r["id"], r["part"], r["prompt"], r["parent"], r["status"], r["required"]) for r in reqs] == [
        ("r1", "drawer_1", "drawer", "cabinet_body", "found", True),
        ("r2", "drawer_2", "drawer", "cabinet_body", "found", True),
        ("r3", None, "drawer", "cabinet_body", "not_found", True),
        ("r4", "door_1", "door", "cabinet_body", "found", True),
        ("r5", "door_2", "door", "cabinet_body", "found", False),
        ("r6", None, "knob", "door_1", "not_found", True),
        ("r7", "pad", "pad", "door_1", "found", True)]
    assert [r.get("reserve") for r in reqs] == [None, None, "drawer_3", None, None, "knob", None]
    assert all("reserve" not in r for r in reqs if r["status"] == "found")
    drawer, door, knob, pad = reqs[0], reqs[3], reqs[5], reqs[6]
    assert (drawer["kind"], drawer["text"], drawer["test"]) == ("joint", "pulls out (full travel about 40 cm)", {})
    assert (door["text"], door["test"]) == ("swings open", {"min_span": 6.283})
    assert (knob["kind"], knob["joint"], knob["text"], knob["test"]) == ("joint", "unknown", "turns", {})
    assert pad == {"id": "r7", "part": "pad", "prompt": "pad", "parent": "door_1", "kind": "soft", "joint": "soft",
                   "material": "gel", "text": "dents", "action": {"type": "press", "toward": "support"},
                   "observe": {"type": "node_and_visible_residual", "after": "rigid alignment"},
                   "test": {"min_response_diag": sim_check.RESPONSE, "recovery": True},
                   "scope": {"start": "mesh pose", "base": "fixed", "support": "parent", "gravity": False},
                   "source": "vlm", "status": "found", "required": True, "parents": ["door_1", "door_2"]}
    # The knob hangs on a door too, and the door part has two links: either carries it.
    assert (knob["parent"], knob["parents"]) == ("door_1", ["door_1", "door_2"])
    assert all("parents" not in r for r in reqs if r["prompt"] in ("drawer", "door"))  # their parent is the base
    assert S.MIN_RESPONSE == sim_check.RESPONSE


def test_pieces_count_the_connected_pieces_of_each_part():
    """geometry.pieces: per label the connected pieces of its faces, by shared edges, and the share of its area in
    the largest. Two boxes under one label are two pieces; a label over half a box is one. This is the hint, in
    parts.json, that a found part may hold the surface of two things."""
    import trimesh

    from harpor.parts import geometry

    big = trimesh.creation.box(bounds=[[0, 0, 0], [1, 1, 1]])
    small = trimesh.creation.box(bounds=[[2, 0, 0], [2.5, 0.5, 0.5]])
    mesh = geometry.topology(*(lambda m: (m.vertices, m.faces))(trimesh.util.concatenate([big, small])))
    one = np.zeros(24, int)
    assert geometry.pieces(mesh, one, 2) == [(2, pytest.approx(6 / 7.5)), (0, 0.0)]
    two = np.r_[np.zeros(12, int), np.ones(12, int)]
    assert geometry.pieces(mesh, two, 2) == [(1, 1.0), (1, 1.0)]
    # the top face of the big box as its own label: both labels stay in one piece each
    top = np.isclose(mesh.triangles_center[:, 2], 1.0)
    lab = np.where(top, 1, 0)
    lab[12:] = 0
    assert [n for n, _ in geometry.pieces(mesh, lab, 2)] == [2, 1]


def _proposal(parts):
    return {"base": parts[0][0], "parts": [{"name": n, "parent": p, "joint": j, "count": c} for n, p, j, c in parts]}


def test_reserves_are_free_names_with_link_parents():
    """The reserve of a missing part continues the vote's names and is free among the links and the other reserves.
    A missing part on a missing part gets that part's reserve as its parent; with nothing between, the base. This
    holds for any part list: nothing here is about one object."""
    # a wheel hub that was not found, and the 2 wheels that hang on it: created in order, hub first
    proposal = _proposal([("cart", None, "fixed", 1), ("hub", "cart", "revolute", 1), ("wheel", "hub", "revolute", 2),
                          ("lid", "cart", "revolute", 1)])
    clusters, names = {"cart": {"prompt": "cart"}, "lid": {"prompt": "lid"}}, ["cart", "lid"]
    reqs = S.requirements(proposal, clusters, names, {"cart": None, "lid": "cart"})
    assert [(r["prompt"], r["status"], r.get("reserve"), r["parent"]) for r in reqs] == [
        ("hub", "not_found", "hub", "cart"), ("wheel", "not_found", "wheel_1", "hub"),
        ("wheel", "not_found", "wheel_2", "hub"), ("lid", "found", None, "cart")]
    # one of 2 found is named <slug>, so the missing one is <slug>_2; a link that has that name already moves it on
    proposal = _proposal([("desk", None, "fixed", 1), ("drawer", "desk", "prismatic", 2),
                          ("drawer 2", "desk", "prismatic", 1)])
    clusters = {"desk": {"prompt": "desk"}, "drawer": {"prompt": "drawer"}, "drawer_2": {"prompt": "drawer 2"}}
    reqs = S.requirements(proposal, clusters, list(clusters), {"desk": None, "drawer": "desk", "drawer_2": "desk"})
    assert [(r["part"], r.get("reserve")) for r in reqs] == [("drawer", None), (None, "drawer_2_2"), ("drawer_2", None)]
    reserves = [r["reserve"] for r in reqs if "reserve" in r]
    assert not set(reserves) & set(clusters) and len(set(reserves)) == len(reserves)


def test_copies_of_a_part_pair_with_copies_of_their_parent():
    """Five caster wheels on five caster forks. Missing wheels take the fork of their own index, so wheel 3's reserve
    hangs on fork 3, and parents lists all five forks, any of which the host's check accepts. A found wheel whose
    fork was not found hangs on a fork's reserve, not on the base. A part with fewer copies than its parent part
    takes the first link."""
    proposal = _proposal([("chair base", None, "fixed", 1), ("caster fork", "chair base", "revolute", 5),
                          ("caster wheel", "caster fork", "revolute", 5), ("lever", "caster fork", "revolute", 1)])
    clusters = {"chair_base": {"prompt": "chair base"}}
    reqs = S.requirements(proposal, clusters, ["chair_base"], {"chair_base": None})
    forks = [f"caster_fork_{k}" for k in range(1, 6)]
    assert [(r.get("reserve"), r["parent"], r.get("parents")) for r in reqs[:5]] == [(f, "chair_base", None) for f in forks]
    wheels = reqs[5:10]
    assert [(r["reserve"], r["parent"]) for r in wheels] == [(f"caster_wheel_{k}", forks[k - 1]) for k in range(1, 6)]
    assert all(r["parents"] == forks for r in wheels)
    assert (reqs[10]["reserve"], reqs[10]["parent"], reqs[10]["parents"]) == ("lever", "caster_fork_1", forks)
    # one wheel found, its forks not: it hangs on a fork's reserve, and the other four wheels are reserves
    clusters = {"chair_base": {"prompt": "chair base"}, "caster_wheel": {"prompt": "caster wheel"}}
    reqs = S.requirements(proposal, clusters, ["chair_base", "caster_wheel"],
                          {"chair_base": None, "caster_wheel": "chair_base"})
    found = next(r for r in reqs if r["part"] == "caster_wheel")
    assert (found["parent"], found["parents"]) == ("caster_fork_1", forks)


def test_a_split_that_found_every_part_reserves_nothing():
    proposal = _proposal([("box", None, "fixed", 1), ("flap", "box", "revolute", 2), ("pad", "box", "soft", 1)])
    clusters = {"box": {"prompt": "box"}, "flap_1": {"prompt": "flap"}, "flap_2": {"prompt": "flap"},
                "pad": {"prompt": "pad"}}
    reqs = S.requirements(proposal, clusters, list(clusters), {"box": None, "flap_1": "box", "flap_2": "box",
                                                               "pad": "box"})
    assert [r["status"] for r in reqs] == ["found"] * 3 and not any("reserve" in r for r in reqs)
