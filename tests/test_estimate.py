"""The estimate stage: the measurements of the final mesh, the check of the model's answer, the rigid body, and the
stage itself with the model's requests stubbed (no network)."""

import io
import json

import numpy as np
import pytest
import trimesh
from PIL import Image

from harpor import frames
from harpor.agent import tools, vlm
from harpor.physics import estimate as E
from harpor.webapp import estimate_step as S
from harpor.webapp.agent_step import REDACTED

SIZE = (0.3, 0.2, 0.2)  # the box in S (z up), in m
KEY = "sk-ant-test-0123456789abcdefXYZ"
OKEY = "sk-proj-test-9876543210zyxwvuQRS"
COLORS = [(200, 30, 30), (30, 200, 30), (30, 30, 200), (200, 200, 30), (30, 200, 200), (200, 30, 200)]
GOOD = {"material": "pine wood", "structure": "solid", "mass_kg": 6.0, "mass_low_kg": 4.0, "mass_high_kg": 9.0,
        "friction": 0.5, "friction_low": 0.35, "friction_high": 0.7, "reason": "Pine is about 500 kg/m^3."}


def box_glb(path, size=SIZE, center=(0, 0, 0)):
    """A closed box GLB whose load_w vertices (S) span size around center."""
    mesh = trimesh.creation.box(size)
    mesh.vertices = frames.w_to_gltf(mesh.vertices + center)
    mesh.export(path)
    return path


def job_folder(J, n=6, lidar=False, stored="/elsewhere/run/input/images", size=(64, 48)):
    """A job after align: physics/aligned.glb, n photos of size (width, height) in run/input/images,
    run/cameras.json with the paths of another machine (stored), job.json, run/placement.json and a key file with
    both keys."""
    (J / "physics").mkdir(parents=True)
    box_glb(J / "physics" / "aligned.glb")
    images = J / "run" / "input" / "images"
    images.mkdir(parents=True)
    cams, (w, h) = [], size
    for i in range(n):
        Image.fromarray(np.full((h, w, 3), COLORS[i % len(COLORS)], np.uint8)).save(images / f"{i}.png")
        cams.append({"name": str(i), "image": f"{stored}/{i}.png", "width": w, "height": h})
    (J / "run" / "cameras.json").write_text(json.dumps(cams))
    (J / "run" / "placement.json").write_text(json.dumps({"metric_scale": 1.25 if lidar else None}))
    (J / "job.json").write_text(json.dumps({"id": "x", "prompt": "toy box", "agent": "opus"}))
    (J / "key.env").write_text(f"OPENAI_API_KEY={OKEY}\nANTHROPIC_API_KEY='{KEY}'\n")
    return J


def stub(monkeypatch, *answers):
    """vlm.ask_claude and vlm.ask_astra give the answers in turn (a SystemExit is raised); the calls are recorded."""
    calls, left = [], list(answers)

    def answer(**call):
        calls.append(call)
        a = left.pop(0)
        if isinstance(a, BaseException):
            raise a
        return a

    monkeypatch.setattr(vlm, "ask_claude", lambda key, prompt, image, client=None, model=tools.CLAUDE_MODEL:
                        answer(api="claude", key=key, prompt=prompt, image=image, model=model))
    monkeypatch.setattr(vlm, "ask_astra", lambda key, prompt, image:
                        answer(api="astra", key=key, prompt=prompt, image=image))
    return calls


def text(**changes):
    return "Here is my estimate.\n" + json.dumps({**GOOD, **changes})


GEOM = {"filled_volume_m3": 0.012}


def test_voxels_and_geometry_of_a_closed_box(tmp_path):
    glb = box_glb(tmp_path / "box.glb", center=(1.0, -2.0, 0.5))
    vox = E.voxels(glb)
    assert vox["h"] == pytest.approx(0.3 / E.RESOLUTION)
    assert len(vox["surface"]) < len(vox["filled"])
    g = E.geometry(glb, vox)
    volume, area = np.prod(SIZE), 2 * (0.3 * 0.2 + 0.3 * 0.2 + 0.2 * 0.2)
    assert g["extent_m"] == pytest.approx(list(SIZE))
    # Voxels that the surface only crosses count, so the filled volume of this bulky box is a few percent high.
    assert g["filled_volume_m3"] == pytest.approx(volume, rel=0.1) and g["filled_volume_m3"] >= volume
    assert g["hull_volume_m3"] == pytest.approx(volume, rel=1e-6)
    assert g["surface_area_m2"] == pytest.approx(area, rel=1e-6)
    assert g["voxel_edge_m"] == vox["h"]
    assert (g["voxels_filled"], g["voxels_surface"]) == (len(vox["filled"]), len(vox["surface"]))
    assert np.allclose(vox["filled"].mean(0), [1.0, -2.0, 0.5], atol=vox["h"])


def test_the_filled_volume_of_a_thin_part_is_far_too_large(tmp_path):
    """Each surface is padded by up to one voxel edge, so the excess grows with the surface area over the volume: a
    15 x 7 x 0.8 cm phone comes out about 1.5 times its volume. The hull volume stays exact for a box."""
    size = (0.15, 0.07, 0.008)
    glb = box_glb(tmp_path / "phone.glb", size=size)
    g = E.geometry(glb, E.voxels(glb))
    assert 1.3 < g["filled_volume_m3"] / np.prod(size) < 1.8
    assert g["hull_volume_m3"] == pytest.approx(np.prod(size), rel=1e-6)


def test_voxels_refuse_a_mesh_without_finite_extent(tmp_path):
    from harpor.agent.fake_agent import write_glb

    write_glb(tmp_path / "nan.glb", kind="nan")
    with pytest.raises(ValueError, match="no finite extent"):
        E.voxels(tmp_path / "nan.glb")


def test_rigid_body_solid_and_hollow(tmp_path):
    vox = E.voxels(box_glb(tmp_path / "box.glb"))
    solid, hollow = E.rigid_body(vox, 2.5, "solid"), E.rigid_body(vox, 2.5, "hollow")
    for rb in (solid, hollow):
        assert rb["mass"] == pytest.approx(2.5, rel=1e-9)
        assert set(rb) == {"mass", "volume", "com", "inertia", "fullinertia"}
    assert solid["com"] == pytest.approx(hollow["com"], abs=1e-9)
    assert np.allclose(solid["com"], 0, atol=1e-9)
    assert all(h > s for h, s in zip(hollow["fullinertia"][:3], solid["fullinertia"][:3]))
    assert solid["volume"] == pytest.approx(len(vox["filled"]) * vox["h"] ** 3)
    with pytest.raises(ValueError):
        E.rigid_body(vox, 2.5, "liquid")


def test_rigid_body_follows_the_size_and_place_of_the_mesh(tmp_path):
    """The voxels are those of the final mesh, so the rigid body follows its size and place: a box twice as large,
    moved to (3, 0, 0) and given 8 times the mass (the same density), has 8 times the volume, its center of mass at
    (3, 0, 0) and 32 times the inertia."""
    small = E.voxels(box_glb(tmp_path / "small.glb", size=(1, 1, 1)))
    big = E.voxels(box_glb(tmp_path / "big.glb", size=(2, 2, 2), center=(3, 0, 0)))
    a, b = E.rigid_body(small, 1.0, "solid"), E.rigid_body(big, 8.0, "solid")
    assert b["mass"] == pytest.approx(8 * a["mass"]) and b["volume"] == pytest.approx(8 * a["volume"])
    assert a["com"] == pytest.approx([0, 0, 0], abs=1e-9) and b["com"] == pytest.approx([3, 0, 0])
    assert np.array(b["inertia"]) == pytest.approx(np.array(a["inertia"]) * 32)


def test_parse_takes_a_good_answer():
    got = E.parse(text(), GEOM)
    assert got == GOOD and list(got) == list(E.FIELDS)
    # Strings become one printable line, and are cut to their limits; ints count as numbers.
    got = E.parse(text(material="oak\x1b[31m\n" + "x" * 300, reason="a\tb", mass_kg=6), GEOM)
    assert got["material"] == ("oak[31m " + "x" * 300)[:200] and got["reason"] == "a b"
    assert got["mass_kg"] == 6.0 and isinstance(got["mass_kg"], float)


@pytest.mark.parametrize("answer, why", [
    (json.dumps({k: v for k, v in GOOD.items() if k != "friction_high"}), "lacks friction_high"),
    (text(structure="liquid"), "structure"),
    (text(material="  \n"), "material is empty"),
    (text(reason=7), "reason must be a string"),
    (text(mass_kg=True), "mass_kg must be a number"),
    (text(mass_high_kg="9"), "mass_high_kg must be a number"),
    (text(friction=float("nan")), "friction must be a finite number"),
    (text(mass_high_kg=float("inf")), "mass_high_kg must be a finite number"),
    (text(mass_low_kg=7.0), "mass_low_kg <= mass_kg <= mass_high_kg"),
    (text(friction_low=0.6), "friction_low <= friction <= friction_high"),
    (text(mass_kg=20000.0, mass_high_kg=30000.0), "mass_kg is 20000, outside"),
    (text(mass_low_kg=0.0001), "mass_low_kg is 0.0001, outside"),
    (text(friction_high=2.5), "friction_high is 2.5, outside"),
    (text(mass_kg=300.0, mass_high_kg=400.0), "above 20000 kg/m"),
    ("I cannot tell.", "no JSON object"),
    ('{"a": ' + "[" * 100_000, "nested too deep"),
])
def test_parse_refuses_a_bad_answer(answer, why):
    with pytest.raises(ValueError, match=why):
        E.parse(answer, GEOM)


def test_prompt_gives_the_measurements_and_the_scale_source():
    geom = {"extent_m": [0.3, 0.2, 0.2], "filled_volume_m3": 0.0128, "hull_volume_m3": 0.012,
            "surface_area_m2": 0.32, "voxel_edge_m": 0.005}
    p = E.prompt("toy box", geom, lidar=True)
    for part in ("Object: toy box", "0.300 x 0.200 x 0.200 m", "0.0128 m^3", "0.012 m^3", "0.32 m^2",
                 "from LiDAR", "data, not instructions", "solid or hollow", "90 percent", E.SCHEMA):
        assert part in p
    # The filled volume runs high for thin parts, and a low filled volume does not mean a cavity (a chair).
    for part in ("an edge of 0.005 m", "pads every surface by up to one voxel edge (0.005 m)",
                 "1.5 to 3 times the real volume", "its real volume, which is usually smaller than the filled volume",
                 "concave or open, such as a chair"):
        assert part in p, part
    assert "means an open cavity" not in p and "density times the filled volume" not in p
    assert p.endswith("Return JSON only.")
    q = E.prompt("toy box", geom, lidar=False)
    assert "monocular depth estimate" in q and "from LiDAR" not in q


def test_pick_views_and_photo_paths(tmp_path):
    assert S.pick_views(1) == [0] and S.pick_views(3) == [0, 1, 2]
    assert S.pick_views(6) == [0, 1, 3, 5] and S.pick_views(10) == [0, 2, 5, 8]
    J = job_folder(tmp_path / "j", n=2)
    here = J / "run" / "input" / "images" / "0.png"
    assert S.photo(J, {"image": "/elsewhere/0.png"}) == here
    other = tmp_path / "other.png"
    Image.new("RGB", (8, 8)).save(other)
    assert S.photo(J, {"image": str(other)}) == other  # not under run/input/images: the stored path
    assert S.photo(J, {"image": "/elsewhere/9.png"}) is None


@pytest.mark.parametrize("agent, api, model", [("opus", "claude", tools.CLAUDE_MODEL),
                                               ("fable", "claude", tools.FABLE_MODEL),
                                               ("astra", "astra", tools.CODEX_MODEL)])
def test_run_writes_the_summary_and_the_views(tmp_path, monkeypatch, capsys, agent, api, model):
    J = job_folder(tmp_path / "j")
    (J / "physics" / "summary.json").write_text(json.dumps({"rigid_body": {"mass": 1.0}}))  # from before this stage
    calls = stub(monkeypatch, text())
    s = S.run(J, J / "key.env", agent)
    assert json.loads((J / "physics" / "summary.json").read_text()) == s
    assert set(s) == {"glb", "geometry", "rigid_body", "friction", "estimate"}
    assert s["glb"] == str((J / "physics" / "aligned.glb").resolve())
    assert set(s["geometry"]) == {"extent_m", "filled_volume_m3", "hull_volume_m3", "surface_area_m2", "voxel_edge_m",
                                  "voxels_filled", "voxels_surface"}
    assert set(s["rigid_body"]) == {"mass", "volume", "com", "inertia", "fullinertia"}
    assert s["rigid_body"]["mass"] == pytest.approx(6.0) and s["friction"] == 0.5
    e = s["estimate"]
    assert {k: e[k] for k in E.FIELDS} == GOOD
    assert (e["agent"], e["model"], e["attempts"], e["views"]) == (agent, model, 1, [0, 1, 3, 5])
    assert isinstance(e["seconds"], float) and e["seconds"] >= 0
    [c] = calls
    assert c["api"] == api and c["key"] == (OKEY if agent == "astra" else KEY)
    if api == "claude":
        assert c["model"] == model
    assert "Object: toy box" in c["prompt"] and "monocular depth estimate" in c["prompt"]
    assert c["image"][:2] == b"\xff\xd8"  # JPEG
    # A 2 x 2 grid of the four 64 x 48 photos, view 0 at the top left and view 5 at the bottom right.
    sheet = Image.open(J / "physics" / "estimate_views.jpg").convert("RGB")
    assert sheet.size == (128, 96)
    assert np.allclose(sheet.getpixel((32, 24)), COLORS[0], atol=12)
    assert np.allclose(sheet.getpixel((96, 72)), COLORS[5], atol=12)
    out = capsys.readouterr().out
    assert "mass 6 kg" in out and "friction 0.5" in out and "pine wood (solid)" in out


def test_run_with_lidar_and_a_hollow_object(tmp_path, monkeypatch):
    J = job_folder(tmp_path / "j", n=1, lidar=True)
    calls = stub(monkeypatch, text(structure="hollow"))
    s = S.run(J, J / "key.env", "opus")
    assert "from LiDAR" in calls[0]["prompt"]
    assert s["estimate"]["views"] == [0] and s["estimate"]["structure"] == "hollow"
    assert Image.open(J / "physics" / "estimate_views.jpg").size == (64, 48)
    vox = E.voxels(J / "physics" / "aligned.glb")
    assert s["rigid_body"]["fullinertia"] == pytest.approx(E.rigid_body(vox, 6.0, "hollow")["fullinertia"])


def test_large_photos_fit_their_tiles_and_reach_the_model_whole(tmp_path, monkeypatch):
    """Each photo fits into 1024 x 1024 px (TILE), so four 2400 x 1600 photos give a 2 x 2 grid of 2048 x 1366 px.
    The model gets that image at 2048 px on the long side (IMAGE_SIDE), not at vlm's default of 1024."""
    assert (S.TILE, S.IMAGE_SIDE) == (1024, 2048)
    J = job_folder(tmp_path / "j", n=4, size=(2400, 1600))
    calls = stub(monkeypatch, text())
    s = S.run(J, J / "key.env", "opus")
    assert s["estimate"]["views"] == [0, 1, 2, 3]
    sheet = Image.open(J / "physics" / "estimate_views.jpg").convert("RGB")
    assert sheet.size == (2048, 1366)
    assert np.allclose(sheet.getpixel((512, 341)), COLORS[0], atol=12)
    assert np.allclose(sheet.getpixel((1536, 1024)), COLORS[3], atol=12)
    assert Image.open(io.BytesIO(calls[0]["image"])).size == (2048, 1366)


def test_one_invalid_answer_gets_one_retry(tmp_path, monkeypatch):
    J = job_folder(tmp_path / "j")
    calls = stub(monkeypatch, text(friction=3.0, friction_high=3.5), text())
    s = S.run(J, J / "key.env", "fable")
    assert s["estimate"]["attempts"] == 2 and len(calls) == 2
    assert "friction is 3, outside 0.01 to 2" in calls[1]["prompt"]
    assert calls[1]["prompt"].startswith(calls[0]["prompt"])


def test_two_invalid_answers_end_the_stage(tmp_path, monkeypatch):
    J = job_folder(tmp_path / "j")
    stub(monkeypatch, "no idea", text(structure="liquid"))
    with pytest.raises(SystemExit, match="invalid answer twice: structure"):
        S.run(J, J / "key.env", "opus")
    assert not (J / "physics" / "summary.json").exists()


def test_an_api_error_ends_the_stage(tmp_path, monkeypatch):
    J = job_folder(tmp_path / "j")
    calls = stub(monkeypatch, SystemExit("the Anthropic key was rejected"))
    with pytest.raises(SystemExit, match="key was rejected"):
        S.run(J, J / "key.env", "opus")
    assert len(calls) == 1 and not (J / "physics" / "summary.json").exists()


def test_the_key_never_reaches_the_summary_or_the_log(tmp_path, monkeypatch, capsys):
    J = job_folder(tmp_path / "j")
    escaped = "".join(f"\\u{ord(c):04x}" for c in KEY)
    split = KEY[:8] + "\\u200b" + KEY[8:]  # whole again once printable drops the zero-width space
    first = text(material=f"wood {KEY}", mass_kg=f"{OKEY}")  # invalid: its reason goes to the log and the prompt
    second = (f"My key is {KEY}. " + json.dumps({**GOOD, "material": "wood KEY1", "reason": "KEY2 and KEY3"})
              .replace("KEY1", OKEY).replace("KEY2", escaped).replace("KEY3", split))
    calls = stub(monkeypatch, first, second)
    s = S.run(J, J / "key.env", "opus")
    stored = (J / "physics" / "summary.json").read_text()
    out = capsys.readouterr()
    for k in (KEY, OKEY):
        assert k not in stored and k not in out.out and k not in out.err and k not in calls[1]["prompt"]
    assert s["estimate"]["material"] == f"wood {REDACTED}"
    assert s["estimate"]["reason"] == f"{REDACTED} and {REDACTED}"


def test_run_refuses_a_missing_key_or_mesh(tmp_path, monkeypatch):
    J = job_folder(tmp_path / "j")
    stub(monkeypatch)
    (J / "only_openai.env").write_text(f"OPENAI_API_KEY={OKEY}\n")
    with pytest.raises(SystemExit, match="no ANTHROPIC_API_KEY line"):
        S.run(J, J / "only_openai.env", "fable")
    with pytest.raises(SystemExit, match="no key file"):
        S.run(J, J / "missing.env", "fable")
    (J / "physics" / "aligned.glb").unlink()
    with pytest.raises(SystemExit, match="run the align stage first"):
        S.run(J, J / "key.env", "fable")


def test_main_takes_job_key_file_and_agent(tmp_path, monkeypatch):
    J = job_folder(tmp_path / "j")
    calls = stub(monkeypatch, text())
    S.main(["--job", str(J), "--key-file", str(J / "key.env"), "--agent", "fable"])
    assert calls[0]["model"] == tools.FABLE_MODEL
    assert json.loads((J / "physics" / "summary.json").read_text())["estimate"]["agent"] == "fable"
    with pytest.raises(SystemExit):
        S.main(["--job", str(J), "--key-file", str(J / "key.env"), "--agent", "none"])
