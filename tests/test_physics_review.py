import json
from pathlib import Path

import numpy as np
import pytest

from harpor.agent import controller, physics_review as P, validate
from harpor.fileio import write_json


def test_judgment_schema_and_failures_feed_repair_loop():
    report = {"outcome": "pass", "scope": "none", "findings": []}
    failed = P.parse(json.dumps({"outcome": "fail", "reason": "The box tips over",
                                "findings": [{"why": "It rolls onto its side", "repair": "Flatten its base"}]}))
    assert failed["findings"] == [{"why": "It rolls onto its side", "repair": "Flatten its base"}]
    result = P.apply(report, failed)
    assert result["outcome"] == "fail" and result["scope"] == "local"
    assert result["findings"][0]["code"] == "physics_implausible"
    assert "Flatten its base" in result["findings"][0]["why"]
    assert controller.next_round([{"kind": "build", "outcome": result["outcome"], "scope": result["scope"]}]) == "repair"
    missing = P.apply(report, {"outcome": "inconclusive", "reason": "No frames", "findings": []})
    assert missing["outcome"] == "fail" and missing["scope"] == "none"
    for value in ({"outcome": "pass", "reason": "ok", "findings": [{}]},
                  {"outcome": "fail", "reason": "bad", "findings": []},
                  {"outcome": "unknown", "reason": "bad", "findings": []}):
        with pytest.raises(ValueError):
            P.parse(json.dumps(value))


def test_visual_pass_does_not_override_mechanical_failure():
    """A failed host check stays failed when the VLM passes the frames."""
    report = {"outcome": "fail", "scope": "structural", "findings": [validate.finding("surface_identity")]}
    result = P.apply(report, {"outcome": "pass", "reason": "Looks plausible", "findings": []})
    assert result["outcome"] == "fail" and result["scope"] == "structural"
    assert [f["code"] for f in result["findings"]] == ["surface_identity"]


def checked(tmp_path, monkeypatch, report=None, judged=None, files=("sheet.jpg", "test_00.gif")):
    """P.check of a candidate with the supervisor subprocess replaced: it writes report (with the candidate's
    sha256 unless report sets one) and files into its --out folder and exits 0. judged: the VLM's judgment, or
    None for a judge that must not be asked."""
    from harpor.agent import tools

    root = tmp_path / "session"
    (root / "control/physics").mkdir(parents=True)
    (root / "work").mkdir()
    write_json(root / "control/physics/config.json", {"agent": "astra", "key_file": str(tmp_path / "key.env")})
    glb = root / "candidate.glb"
    glb.write_bytes(b"candidate")
    sha = tools.sha256(glb)
    monkeypatch.setenv("GATOR_MUJOCO", "python")
    def run(cmd, log, timeout, **popen):
        out = Path(cmd[cmd.index("--out") + 1])
        for name in files:
            (out / name).write_bytes(b"frame")
        if report is not None:
            write_json(out / "report.json", {"asset_sha256": sha, **report})
        return 0
    monkeypatch.setattr(tools, "_run", run)
    def judge(*a):
        assert judged is not None, "the judge was asked without simulation evidence"
        return {**judged, "findings": list(judged["findings"])}
    monkeypatch.setattr(P, "judge", judge)
    return root, P.check(root, glb)


def simulation(warnings=()):
    return {"materials": {}, "settling": {"warnings": list(warnings), "states": []},
            "sheet": "sheet.jpg", "clips": {"settling": "test_00.gif"}}


def test_a_mujoco_warning_fails_a_visual_pass(tmp_path, monkeypatch):
    """A MuJoCo warning while settling fails the candidate even when the VLM passes its frames; without a warning
    the same pass stands. Either way the frames are published as evidence for the next round."""
    passed = {"outcome": "pass", "reason": "Rests on its base", "findings": []}
    root, judgment = checked(tmp_path / "warned", monkeypatch, simulation(["BADQACC"]), passed)
    assert judgment["outcome"] == "fail"
    assert "BADQACC" in judgment["findings"][0]["why"] and judgment["findings"][0]["repair"]
    applied = P.apply({"outcome": "pass", "scope": "none", "findings": []}, judgment)
    assert applied["outcome"] == "fail" and applied["findings"][0]["code"] == "physics_implausible"
    root, judgment = checked(tmp_path / "quiet", monkeypatch, simulation(), passed)
    assert judgment["outcome"] == "pass" and judgment["findings"] == []
    assert (root / "work" / judgment["evidence"] / "test_00.gif").is_file()
    assert judgment["clips"] == {"settling": "test_00.gif"}


@pytest.mark.parametrize("case", ["no_mujoco", "no_report", "wrong_sha", "error"])
def test_missing_simulation_evidence_is_inconclusive(tmp_path, monkeypatch, case):
    """No configured MuJoCo, no report, a report of another asset or a supervisor error is inconclusive, never a
    pass, and the VLM is not asked. Applied to a passing check, it fails with scope none."""
    report = {"no_mujoco": simulation(), "no_report": None, "wrong_sha": {**simulation(), "asset_sha256": "b" * 64},
              "error": {"error": "ValueError: No simulation frames were captured"}}[case]
    if case == "no_mujoco":
        monkeypatch.delenv("GATOR_MUJOCO", raising=False)
        from harpor.agent import tools
        monkeypatch.setattr(tools, "_run", lambda *a, **k: pytest.fail("the supervisor ran without GATOR_MUJOCO"))
        root = tmp_path / "session"
        (root / "control/physics").mkdir(parents=True)
        write_json(root / "control/physics/config.json", {"agent": "astra", "key_file": str(tmp_path / "key.env")})
        (root / "candidate.glb").write_bytes(b"candidate")
        monkeypatch.setattr(P, "judge", lambda *a: pytest.fail("the judge was asked without evidence"))
        judgment = P.check(root, root / "candidate.glb")
    else:
        root, judgment = checked(tmp_path, monkeypatch, report)
    assert judgment["outcome"] == "inconclusive" and judgment["findings"] == []
    assert judgment["reason"].startswith("Physics supervision unavailable")
    applied = P.apply({"outcome": "pass", "scope": "none", "findings": []}, judgment)
    assert applied["outcome"] == "fail" and applied["scope"] == "none"
    assert [f["code"] for f in applied["findings"]] == ["physics_unassessed"]


def test_refinement_rounds_accept_complete_and_unchanged(tmp_path):
    control = tmp_path / "control"
    (control / "physics").mkdir(parents=True)
    write_json(control / "physics/config.json", {})
    write_json(control / "manifest.json", {})
    assert controller.has_rounds(tmp_path)
    report = {"outcome": "pass", "scope": "none", "findings": []}
    for verdict in ("complete", "unchanged"):
        assert validate.round_report(control, report, {"source": "finish", "verdict": verdict})["outcome"] == "pass"


def test_controller_repairs_a_visual_failure_and_selects_verified_candidate(tmp_path, monkeypatch):
    from test_material_transfer import box
    from harpor.agent import tools

    (tmp_path / "control/physics").mkdir(parents=True)
    (tmp_path / "work/exports").mkdir(parents=True)
    write_json(tmp_path / "control/physics/config.json", {})
    write_json(tmp_path / "control/manifest.json", {})
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
        {"outcome": "fail", "reason": "Unstable", "findings": [{"why": "It tips", "repair": "Fix the base"}]},
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
    """The judge sees the manifest's photo above the simulation sheet and the materials and settling measurements,
    with the job's key; an API key echoed in its answer, even split by a control character, is redacted."""
    from PIL import Image
    from harpor.agent import vlm

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
    monkeypatch.setattr(vlm, "ask_astra", ask)
    sim = {"materials": {"rigid_body": {"mass": 1.5}}, "settling": {"warnings": [], "states": [{"t": 0.0}]},
           "sheet": "sheet.jpg", "clips": {}}
    result = P.judge(tmp_path, tmp_path, sim, {"agent": "astra", "key_file": str(tmp_path / "key.env")})
    assert result["outcome"] == "pass" and key not in json.dumps(result)
    assert "[redacted API key]" in result["reason"]
    assert calls[0][0] == key and "Measurements:" in calls[0][1] and len(calls[0][2]) > 1000
    assert "Object: a box" in calls[0][1]
    assert json.loads(calls[0][1].split("Measurements:\n")[1]) == {k: sim[k] for k in ("materials", "settling")}
    with Image.open(tmp_path / "judge.jpg") as panel:
        assert panel.size == (400, 100 + 300)  # the photo above the sheet, never enlarged


def test_session_preparation_uses_placed_mass_and_preserves_predictions(tmp_path, monkeypatch):
    from test_material_transfer import box, prediction

    job, root = tmp_path / "job", tmp_path / "session"
    (job / "run/vomp").mkdir(parents=True)
    for folder in ("control", "work/input", "session"):
        (root / folder).mkdir(parents=True)
    write_json(job / "job.json", {})
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


def hull(vertices, faces, threshold):
    """One convex hull in place of CoACD: exact for a box, and fast."""
    import trimesh

    h = trimesh.Trimesh(vertices, faces).convex_hull
    return [(np.asarray(h.vertices), np.asarray(h.faces))]


def test_real_settling_evidence(tmp_path, monkeypatch):
    """supervise.run settles a real box in MuJoCo for three seconds under gravity: 13 states, the predicted
    material on the candidate, no warnings for a box at rest, and a contact sheet and clip of rendered frames.
    It needs a working local GL context."""
    pytest.importorskip("mujoco")
    from harpor.physics import supervise, mjcf
    from test_material_transfer import box, prediction

    glb = box(tmp_path / "box.glb", (0.3, 0.2, 0.2))
    source = tmp_path / "materials"
    source.mkdir()
    write_json(source / "config.json", {"up": [0, 0, 1], "predictions": {"": "voxels_00.npz"}})
    write_json(source / "summary.json", {"config": "test"})
    np.savez(source / "voxels_00.npz", **prediction(density=600))
    build = mjcf.build
    monkeypatch.setattr(mjcf, "build", lambda *a, **k: build(*a, decompose=hull, **k))
    result = supervise.run(glb, source, tmp_path / "out", "a" * 64)
    assert result["asset_sha256"] == "a" * 64
    settling = result["settling"]
    assert len(settling["states"]) == 13 and settling["seconds"] == 3.0
    assert settling["gravity"] == [0, 0, -9.81] and settling["warnings"] == []
    assert settling["states"][-1]["t"] == pytest.approx(3.0, abs=0.01)
    assert settling["states"][-1]["speed"] < 0.05  # a box on its face comes to rest
    assert result["materials"]["median"]["density_kg_m3"] == 600
    assert result["materials"]["rigid_body"]["mass"] == pytest.approx(600 * 0.3 * 0.2 * 0.2, rel=0.1)
    assert result["clips"] == {"settling": "test_00.gif"}
    assert (tmp_path / "out" / result["sheet"]).is_file()
    assert all((tmp_path / "out" / name).stat().st_size > 1000 for name in result["clips"].values())
