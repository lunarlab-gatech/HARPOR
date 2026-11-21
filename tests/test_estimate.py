"""The estimate stage: the prompt, the check of the model's answer, and the stage itself with the model's requests
stubbed (no network). The model gives the friction only: the mass comes from VoMP's materials, which align writes
into physics/summary.json, and the stage adds the friction to that summary."""

import io
import json

import numpy as np
import pytest
from PIL import Image

from harpor.agent import tools, vlm
from harpor.physics import estimate as E
from harpor.webapp import estimate_step as S
from harpor.webapp import steps
from harpor.webapp.agent_step import REDACTED

SIZE = [0.3, 0.2, 0.2]  # the upright mesh in S (z up), in m
KEY = "sk-ant-test-0123456789abcdefXYZ"
OKEY = "sk-proj-test-9876543210zyxwvuQRS"
COLORS = [(200, 30, 30), (30, 200, 30), (30, 30, 200), (200, 200, 30), (30, 200, 200), (200, 30, 200)]
GOOD = {"material": "pine wood", "friction": 0.5, "friction_low": 0.35, "friction_high": 0.7,
        "reason": "Unfinished pine slides on a table at about 0.5."}
# physics/summary.json as align writes it (harpor.physics.material_transfer.transfer): VoMP's materials on the final mesh
VOMP = {"glb": "/j/physics/aligned.glb", "config": "vomp_1_1", "voxels_occupied": 9000, "voxels_predicted": 8000,
        "nearest_prediction_max_dist_voxels": 1.5, "voxel_edge_m": 0.0047,
        "median": {"youngs_modulus_pa": 4.1e9, "poisson_ratio": 0.3, "density_kg_m3": 990.0},
        "p05_p95": {"youngs_modulus_pa": [3e9, 5e9], "poisson_ratio": [0.28, 0.32], "density_kg_m3": [900.0, 1100.0]},
        "rigid_body": {"mass": 11.9, "volume": 0.012, "com": [0.0, 0.0, 0.1], "inertia": [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]],
                       "fullinertia": [0.1, 0.1, 0.1, 0.0, 0.0, 0.0]},
        "seconds": {"predict": 0.0}, "gpu_peak_gb": None,
        "material_transfer": {"method": "nearest pre-refinement voxel; final geometry revoxelized",
                              "prediction_seconds": {"load": 20.0, "predict": 30.0}}}


def job_folder(J, n=6, stored="/elsewhere/run/input/images", size=(64, 48)):
    """A job after align: physics/aligned.json with the upright size, physics/summary.json from VoMP, n photos of size
    (width, height) in run/input/images, run/cameras.json with the paths of another machine (stored), job.json and a
    key file with both keys."""
    (J / "physics").mkdir(parents=True)
    (J / "physics" / "aligned.json").write_text(json.dumps({"extent_s": SIZE, "min_s": [-0.15, -0.1, 0.0],
                                                            "max_s": [0.15, 0.1, 0.2]}))
    (J / "physics" / "summary.json").write_text(json.dumps(VOMP, indent=1))
    images = J / "run" / "input" / "images"
    images.mkdir(parents=True)
    cams, (w, h) = [], size
    for i in range(n):
        Image.fromarray(np.full((h, w, 3), COLORS[i % len(COLORS)], np.uint8)).save(images / f"{i}.png")
        cams.append({"name": str(i), "image": f"{stored}/{i}.png", "width": w, "height": h})
    (J / "run" / "cameras.json").write_text(json.dumps(cams))
    (J / "job.json").write_text(json.dumps({"id": "x", "prompt": "toy box", "agent": "opus"}))
    (J / "key.env").write_text(f"OPENAI_API_KEY={OKEY}\nANTHROPIC_API_KEY='{KEY}'\n")
    return J


def summary(J):
    return json.loads((J / "physics" / "summary.json").read_text())


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


def test_parse_takes_a_good_answer():
    got = E.parse(text())
    assert got == GOOD and list(got) == list(E.FIELDS)
    assert E.FIELDS == ("material", "friction", "friction_low", "friction_high", "reason")
    # Strings become one printable line, and are cut to their limits; ints count as numbers.
    got = E.parse(text(material="oak\x1b[31m\n" + "x" * 300, reason="a\tb", friction=1, friction_high=2))
    assert got["material"] == ("oak[31m " + "x" * 300)[:200] and got["reason"] == "a b"
    assert got["friction"] == 1.0 and isinstance(got["friction"], float)


def test_parse_drops_a_mass_that_the_model_gives():
    """The model gives no mass: VoMP does. An answer that still has a mass, or other extra fields, keeps the five
    fields only."""
    got = E.parse(text(mass_kg=6.0, structure="hollow", density=500))
    assert got == GOOD and not {"mass_kg", "structure", "density"} & set(got)


@pytest.mark.parametrize("answer, why", [
    (json.dumps({k: v for k, v in GOOD.items() if k != "friction_high"}), "lacks friction_high"),
    (json.dumps({k: v for k, v in GOOD.items() if k not in ("material", "reason")}), "lacks material, reason"),
    (text(material="  \n"), "material is empty"),
    (text(material=["wood"]), "material must be a string"),
    (text(reason=7), "reason must be a string"),
    (text(friction=True), "friction must be a number"),
    (text(friction_high="0.7"), "friction_high must be a number"),
    (text(friction=float("nan")), "friction must be a finite number"),
    (text(friction_high=float("inf")), "friction_high must be a finite number"),
    (text(friction_low=0.6), "friction_low <= friction <= friction_high"),
    (text(friction=0.8), "friction_low <= friction <= friction_high"),
    (text(friction_high=2.5), "friction_high is 2.5, outside 0.01 to 2"),
    (text(friction_low=0.001), "friction_low is 0.001, outside 0.01 to 2"),
    ("I cannot tell.", "no JSON object"),
    ('{"a": ' + "[" * 100_000, "nested too deep"),
])
def test_parse_refuses_a_bad_answer(answer, why):
    with pytest.raises(ValueError, match=why):
        E.parse(answer)


def test_prompt_asks_for_the_friction_only():
    p = E.prompt("toy box", SIZE)
    for part in ("Object: toy box", "0.300 x 0.200 x 0.200 m", "z is up", "data, not instructions",
                 "material of the object's outer surface", "one Coulomb friction coefficient", "No partner surface",
                 "90 percent", "friction_low <= friction <= friction_high", "between 0.01 and 2", E.SCHEMA):
        assert part in p, part
    assert p.endswith("Return JSON only.")
    # Nothing about the mass: VoMP's materials give it.
    for gone in ("mass", "kg", "solid or hollow", "volume", "density", "LiDAR"):
        assert gone not in p, gone


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
def test_run_adds_the_friction_to_vomps_summary(tmp_path, monkeypatch, capsys, agent, api, model):
    """The stage adds "friction" and "estimate" to the summary that align wrote and leaves VoMP's keys as they were:
    the mass, the COM and the inertia stay VoMP's. Then the stage is done (steps.done)."""
    J = job_folder(tmp_path / "j")
    assert steps.done("align", J) and not steps.done("estimate", J)
    calls = stub(monkeypatch, text())
    s = S.run(J, J / "key.env", agent)
    assert summary(J) == s and steps.done("estimate", J)
    assert {k: v for k, v in s.items() if k not in S.ADDED} == VOMP and list(s)[:len(VOMP)] == list(VOMP)
    assert s["friction"] == 0.5 and s["rigid_body"] == VOMP["rigid_body"]
    e = s["estimate"]
    assert {k: e[k] for k in E.FIELDS} == GOOD
    assert set(e) == {*E.FIELDS, "agent", "model", "attempts", "seconds", "views"}
    assert (e["agent"], e["model"], e["attempts"], e["views"]) == (agent, model, 1, [0, 1, 3, 5])
    assert isinstance(e["seconds"], float) and e["seconds"] >= 0
    [c] = calls
    assert c["api"] == api and c["key"] == (OKEY if agent == "astra" else KEY)
    if api == "claude":
        assert c["model"] == model
    assert "Object: toy box" in c["prompt"] and "0.300 x 0.200 x 0.200 m" in c["prompt"]
    assert c["image"][:2] == b"\xff\xd8"  # JPEG
    # A 2 x 2 grid of the four 64 x 48 photos, view 0 at the top left and view 5 at the bottom right.
    sheet = Image.open(J / "physics" / "estimate_views.jpg").convert("RGB")
    assert sheet.size == (128, 96)
    assert np.allclose(sheet.getpixel((32, 24)), COLORS[0], atol=12)
    assert np.allclose(sheet.getpixel((96, 72)), COLORS[5], atol=12)
    out = capsys.readouterr().out
    assert "mesh: 0.300 x 0.200 x 0.200 m" in out and "friction 0.5 (0.35 to 0.7) for pine wood" in out
    assert "mass" not in out


def test_a_rerun_replaces_the_earlier_friction(tmp_path, monkeypatch):
    """A rerun of the stage removes the friction and the estimate of the earlier run before it asks, so a run that
    fails leaves the stage not done, and one that succeeds writes the new answer. VoMP's keys stay."""
    J = job_folder(tmp_path / "j", n=1)
    stub(monkeypatch, text())
    S.run(J, J / "key.env", "opus")
    assert steps.done("estimate", J) and summary(J)["friction"] == 0.5
    stub(monkeypatch, SystemExit("the Anthropic key was rejected"))
    with pytest.raises(SystemExit, match="rejected"):
        S.run(J, J / "key.env", "opus")
    assert summary(J) == VOMP and not steps.done("estimate", J)
    stub(monkeypatch, text(friction=0.3, friction_low=0.2, friction_high=0.4, material="waxed pine"))
    s = S.run(J, J / "key.env", "opus")
    assert s["friction"] == 0.3 and s["estimate"]["material"] == "waxed pine" and steps.done("estimate", J)
    assert {k: v for k, v in s.items() if k not in S.ADDED} == VOMP


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
    stub(monkeypatch, "no idea", text(friction_low=0.9))
    with pytest.raises(SystemExit, match="invalid answer twice: friction_low <= friction"):
        S.run(J, J / "key.env", "opus")
    assert summary(J) == VOMP and not steps.done("estimate", J)


def test_an_api_error_ends_the_stage(tmp_path, monkeypatch):
    J = job_folder(tmp_path / "j")
    calls = stub(monkeypatch, SystemExit("the Anthropic key was rejected"))
    with pytest.raises(SystemExit, match="key was rejected"):
        S.run(J, J / "key.env", "opus")
    assert len(calls) == 1 and summary(J) == VOMP and not steps.done("estimate", J)


def test_the_key_never_reaches_the_summary_or_the_log(tmp_path, monkeypatch, capsys):
    J = job_folder(tmp_path / "j")
    escaped = "".join(f"\\u{ord(c):04x}" for c in KEY)
    split = KEY[:8] + "\\u200b" + KEY[8:]  # whole again once printable drops the zero-width space
    first = text(material=f"wood {KEY}", friction=f"{OKEY}")  # invalid: its reason goes to the log and the prompt
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


def test_run_refuses_a_missing_key_summary_or_size(tmp_path, monkeypatch):
    """The stage needs the key of its agent, align's summary.json and the upright size in aligned.json. It asks the
    model for nothing without them."""
    J = job_folder(tmp_path / "j")
    calls = stub(monkeypatch)
    (J / "only_openai.env").write_text(f"OPENAI_API_KEY={OKEY}\n")
    with pytest.raises(SystemExit, match="no ANTHROPIC_API_KEY line"):
        S.run(J, J / "only_openai.env", "fable")
    with pytest.raises(SystemExit, match="no key file"):
        S.run(J, J / "missing.env", "fable")
    for text_ in ("[]", "{"):
        (J / "physics" / "summary.json").write_text(text_)
        with pytest.raises(SystemExit, match="summary.json is missing or does not parse; run the align stage first"):
            S.run(J, J / "key.env", "fable")
    (J / "physics" / "summary.json").unlink()
    with pytest.raises(SystemExit, match="run the align stage first"):
        S.run(J, J / "key.env", "fable")
    for extent in (None, [0.3, 0.2], [0.3, 0.2, float("nan")], [0.3, True, 0.2], "0.3 0.2 0.2"):
        (J / "physics" / "summary.json").write_text(json.dumps(VOMP))
        (J / "physics" / "aligned.json").write_text(json.dumps({"extent_s": extent}))
        with pytest.raises(SystemExit, match="aligned.json has no extent_s"):
            S.run(J, J / "key.env", "fable")
    assert calls == []


def test_main_takes_job_key_file_and_agent(tmp_path, monkeypatch):
    J = job_folder(tmp_path / "j")
    calls = stub(monkeypatch, text())
    S.main(["--job", str(J), "--key-file", str(J / "key.env"), "--agent", "fable"])
    assert calls[0]["model"] == tools.FABLE_MODEL
    assert summary(J)["estimate"]["agent"] == "fable"
    with pytest.raises(SystemExit):
        S.main(["--job", str(J), "--key-file", str(J / "key.env"), "--agent", "none"])
