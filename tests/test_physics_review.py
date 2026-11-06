import json

import numpy as np
import pytest

from harpor.agent import controller, physics_review as P, validate
from harpor.fileio import write_json


def test_judgment_schema_and_failures_feed_repair_loop():
    report = {"outcome": "pass", "scope": "none", "findings": []}
    failed = P.parse(json.dumps({"outcome": "fail", "reason": "The lid detaches",
                                "findings": [{"link": "lid", "why": "It separates during motion", "repair": "Fix its hinge anchor"}]}))
    result = P.apply(report, failed)
    assert result["outcome"] == "fail" and result["scope"] == "local"
    assert "Fix its hinge anchor" in result["findings"][0]["why"]
    assert controller.next_round([{"kind": "build", "outcome": result["outcome"], "scope": result["scope"]}]) == "repair"
    missing = P.apply(report, {"outcome": "inconclusive", "reason": "No frames", "findings": []})
    assert missing["outcome"] == "fail" and missing["scope"] == "none"
    for value in ({"outcome": "pass", "reason": "ok", "findings": [{}]},
                  {"outcome": "fail", "reason": "bad", "findings": []},
                  {"outcome": "unknown", "reason": "bad", "findings": []}):
        with pytest.raises(ValueError):
            P.parse(json.dumps(value))


def test_visual_pass_does_not_override_mechanical_failure():
    report = {"outcome": "fail", "scope": "structural", "findings": [validate.finding("rest_collision")]}
    assert P.apply(report, {"outcome": "pass", "reason": "Looks plausible", "findings": []})["outcome"] == "fail"


def test_refinement_rounds_accept_complete_and_unchanged(tmp_path):
    control = tmp_path / "control"
    (control / "physics").mkdir(parents=True)
    write_json(control / "physics/config.json", {})
    write_json(control / "manifest.json", {"mode": "refinement"})
    assert controller.has_rounds(tmp_path, "refinement")
    report = {"outcome": "pass", "scope": "none", "findings": []}
    for verdict in ("complete", "unchanged"):
        assert validate.round_report(control, report, {"source": "finish", "verdict": verdict})["outcome"] == "pass"


def test_controller_repairs_a_visual_failure_and_selects_verified_candidate(tmp_path, monkeypatch):
    from test_material_transfer import box
    from harpor.agent import tools

    (tmp_path / "control/physics").mkdir(parents=True)
    (tmp_path / "work/exports").mkdir(parents=True)
    write_json(tmp_path / "control/physics/config.json", {})
    write_json(tmp_path / "control/manifest.json", {"mode": "refinement"})
    original = box(tmp_path / "original.glb", (0.2, 0.2, 0.2))
    mesh = tmp_path / "work/exports/001.glb"
    mesh.write_bytes(original.read_bytes())
    pick = {"source": "finish", "verdict": "complete", "glb": mesh, "reason": "repaired", "limitations": []}
    monkeypatch.setattr(controller, "select_deliverable", lambda *a: (pick, []))
    def mechanical(glb, *a, **k):
        return {"sha256": tools.sha256(glb), "outcome": "pass", "scope": "none", "findings": [], "records": [],
                "surface": {}, "errors": []}
    monkeypatch.setattr(validate, "check", mechanical)
    verdicts = iter([
        {"outcome": "fail", "reason": "Unstable", "findings": [{"link": None, "why": "It tips", "repair": "Fix the base"}]},
        {"outcome": "pass", "reason": "Stable", "findings": []}])
    monkeypatch.setattr(P, "check", lambda *a, **k: next(verdicts))
    first = controller.check_round(tmp_path, 1, "build")
    assert first["report"]["outcome"] == "fail"
    assert "Fix the base" in validate.feedback(first["report"])
    second = controller.check_round(tmp_path, 2, "repair")
    assert second["report"]["outcome"] == "pass"
    selected, candidate, _ = controller.select_candidate(tmp_path)
    assert selected["verdict"] == "complete" and candidate["version"] == 2


def test_judge_uses_reference_and_simulation_and_redacts_answer(tmp_path, monkeypatch):
    from PIL import Image
    from harpor.parts import propose

    (tmp_path / "control").mkdir()
    (tmp_path / "work/input").mkdir(parents=True)
    Image.new("RGB", (100, 100), "red").save(tmp_path / "work/input/photo.png")
    Image.new("RGB", (400, 300), "blue").save(tmp_path / "sheet.jpg")
    write_json(tmp_path / "control/manifest.json", {"source": "a box", "views": [{"image": "input/photo.png"}]})
    key = "sk-test-secret"
    (tmp_path / "key.env").write_text("OPENAI_API_KEY=" + key)
    calls = []
    def ask(k, prompt, image):
        calls.append((k, prompt, image))
        return json.dumps({"outcome": "pass", "reason": "Stable " + key[:7] + "\x1b" + key[7:], "findings": []})
    monkeypatch.setattr(propose, "ask_astra", ask)
    sim = {k: {} for k in ("materials", "dynamics", "settling", "soft_materials", "fixed_base")}
    result = P.judge(tmp_path, tmp_path, sim, {"agent": "astra", "key_file": str(tmp_path / "key.env")})
    assert result["outcome"] == "pass" and key not in json.dumps(result)
    assert "[redacted API key]" in result["reason"]
    assert calls[0][0] == key and "Measurements:" in calls[0][1] and len(calls[0][2]) > 1000


def test_session_preparation_uses_placed_mass_and_preserves_predictions(tmp_path, monkeypatch):
    from test_material_transfer import box, prediction

    job, root = tmp_path / "job", tmp_path / "session"
    (job / "run/vomp").mkdir(parents=True)
    for folder in ("control", "work/input", "session"):
        (root / folder).mkdir(parents=True)
    write_json(job / "job.json", {"articulated": False})
    write_json(job / "run/vomp/summary.json", {"config": "test", "rigid_body": {"mass": 600}})
    np.savez(job / "run/vomp/voxels.npz", **prediction())
    pose = {"rotation_wxyz": [1, 0, 0, 0], "translation": [1, 2, 3], "scale": [2]}
    write_json(job / "run/placement.json", {"pose": pose})
    box(root / "original.glb", (2, 2, 2), (-1, -2, 3))
    (root / "session/prompt.txt").write_text("Refine this mesh.")
    monkeypatch.setattr("harpor.webapp.physics_auto.up_hint", lambda run: (np.array([0, 0, 1]), {}))
    P.prepare(root, job, tmp_path / "key.env", "astra")
    summary = json.loads((root / "work/input/physics.json").read_text())
    assert summary["rigid_body"]["mass"] == pytest.approx(4800, rel=0.06)
    assert summary["rigid_body"]["com"] == pytest.approx([-1, -2, 3])
    assert json.loads((job / "run/vomp/summary.json").read_text())["rigid_body"]["mass"] == 600
    assert P.enabled(root / "control") and "Physics supervision" in (root / "session/prompt.txt").read_text()


@pytest.mark.parametrize("kind", ["rigid", "articulated", "soft"])
def test_real_settling_and_motion_evidence(tmp_path, monkeypatch, kind):
    pytest.importorskip("mujoco")
    from harpor.physics import supervise, mjcf
    from test_sim_check import chest, hull_parts, pad_on_block
    from test_material_transfer import box, prediction

    glb = (chest(tmp_path) if kind == "articulated" else pad_on_block(tmp_path) if kind == "soft"
           else box(tmp_path / "box.glb", (0.3, 0.2, 0.2)))
    from harpor.parts import joints
    from harpor.physics import align
    doc, binary = align.read_glb(glb)
    align.write_glb(glb, joints.write_joints(doc, joints.transform(joints.read_joints(doc), np.eye(4), "W")), binary)
    # chest is authored in S. The supervisor accepts W, so choose its existing +Z as up.
    source = tmp_path / "materials"
    source.mkdir()
    names = {"rigid": [""], "articulated": ["body", "lid", "drawer"], "soft": ["block", "pad"]}[kind]
    files = {name: f"part_{i}.npz" for i, name in enumerate(names)}
    write_json(source / "config.json", {"up": [0, 0, 1], "articulated": kind != "rigid", "predictions": files})
    write_json(source / "summary.json", {"config": "test"})
    for name, filename in files.items():
        np.savez(source / filename, **prediction(density=1000 if name == "lid" else 600))
    build = mjcf.build
    monkeypatch.setattr(mjcf, "build", lambda *a, **k: build(*a, decompose=hull_parts, **k))
    # Exercise real dynamics, frames and clips. This needs a working local GL context.
    result = supervise.run(glb, source, tmp_path / "out", "a" * 64)
    assert result["asset_sha256"] == "a" * 64
    assert len(result["settling"]["states"]) == 13
    assert result["settling"]["gravity"] == [0, 0, -9.81]
    assert len(result["clips"]) >= (1 if kind == "rigid" else 2)
    assert (tmp_path / "out" / result["sheet"]).is_file()
    assert all((tmp_path / "out" / name).stat().st_size > 1000 for name in result["clips"].values())
    if kind == "articulated":
        assert result["dynamics"]["joints"]
        assert result["materials"]["links"]["lid"]["median"]["density_kg_m3"] == 1000
    elif kind == "soft":
        assert result["dynamics"]["soft"] and result["soft_materials"]["pad"]["material"] == "foam"
