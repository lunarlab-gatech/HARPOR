import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from harpor import frames
from harpor.parts.joints import SOFT_MATERIALS, read_joints, write_joints
from harpor.physics import mjcf, sim_check
from harpor.physics.align import read_glb, write_glb

trimesh = pytest.importorskip("trimesh")
mujoco = pytest.importorskip("mujoco")

from test_physics import hull_parts  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SHA = "c" * 64  # the candidate's sha256; model.glb has its own


@pytest.fixture
def hulls(monkeypatch):
    """One convex hull per link in place of CoACD, as the MJCF tests do."""
    build = mjcf.build
    monkeypatch.setattr(mjcf, "build", lambda *a, **k: build(*a, decompose=hull_parts, **k))


def box(size, center):
    b = trimesh.creation.box(size)
    b.apply_translation(center)
    return b


def lens(size, center):
    """A closed flattened sphere with the extents of size."""
    m = trimesh.creation.icosphere(subdivisions=4)
    m.apply_scale(np.array(size) / 2)
    m.apply_translation(center)
    return m


def write_parts(glb, meshes, joints):
    """glb with the mesh nodes {name: trimesh, its numbers those of S} and these joints, in frame S."""
    glb.parent.mkdir(parents=True, exist_ok=True)
    scene = trimesh.Scene()
    for name, m in meshes.items():
        scene.add_geometry(trimesh.Trimesh(frames.w_to_gltf(m.vertices), m.faces, process=False), node_name=name,
                           geom_name=name)
    scene.export(glb)
    doc, binary = read_glb(glb)
    write_glb(glb, write_joints(doc, joints), binary)
    return glb


def chest(tmp_path):
    """A 0.6 x 0.4 x 0.3 m body box, with a lid on top hinged on its back top edge to open upward by up to 1.5 rad
    and a drawer in its front that slides out along -y by up to 0.3 m. D is 0.797 m."""
    hinge = {"version": 1, "frame": "S", "parent": "body", "type": "revolute", "origin": [0.0, 0.2, 0.3],
             "axis": [-1.0, 0.0, 0.0], "lower": 0.0, "upper": 1.5}
    slide = {"version": 1, "frame": "S", "parent": "body", "type": "prismatic", "origin": [0.0, 0.0, 0.0],
             "axis": [0.0, -1.0, 0.0], "lower": 0.0, "upper": 0.3}
    return write_parts(tmp_path / "parts.glb", {"body": box([0.6, 0.4, 0.3], [0, 0, 0.15]),
                                                "lid": box([0.6, 0.4, 0.04], [0, 0, 0.32]),
                                                "drawer": box([0.5, 0.35, 0.1], [0, -0.02, 0.1])},
                       {"lid": hinge, "drawer": slide})


def soft(parent, material="foam"):
    return {"version": 1, "frame": "S", "parent": parent, "type": "soft", "material": material,
            **SOFT_MATERIALS[material], "support": "parent"}


def pad_on_block(tmp_path, lift=0.0):
    """A 0.6 x 0.6 x 0.2 m block with a 46 x 44 x 7 cm foam lens resting on it, or lifted by lift. D is 0.89 m."""
    block = box([0.6, 0.6, 0.2], [0, 0, 0.1]).subdivide().subdivide()
    return write_parts(tmp_path / "parts.glb", {"block": block, "pad": lens([0.46, 0.44, 0.07], [0, 0, 0.235 + lift])},
                       {"pad": soft("block")})


def report(out):
    return json.loads((out / "report.json").read_text())


def test_each_joint_follows_a_ramp_over_its_limits(tmp_path, hulls):
    # Without --intervals each joint runs over its own limits: to the upper end, to the lower end and back to rest,
    # each leg a 1 s ramp of the ctrl and a 0.25 s hold, with a state every 0.25 s.
    r = sim_check.run(chest(tmp_path), tmp_path / "sim", SHA)
    assert r["findings"] == [] and np.isclose(r["diagonal_m"], np.sqrt(0.6 ** 2 + 0.4 ** 2 + 0.34 ** 2))
    assert r["runtime"] == {"mujoco": mujoco.__version__, "dt": 0.002, "fixed_base": True, "gravity": 0,
                            "integrator": "euler", "solver": "NEWTON", "tolerance_angle_deg": 5.0, "tolerance_diag": 0.005}
    lid, drawer = r["joints"]
    assert (lid["link"], lid["unit"], lid["interval"], drawer["link"], drawer["unit"]) == ("lid", "rad", [0.0, 1.5],
                                                                                          "drawer", "m")
    ramp = np.r_[0, 0.25, 0.5, 0.75, 1, 1, 0.75, 0.5, 0.25, 0, np.zeros(6)]
    for j, top in ((lid, 1.5), (drawer, 0.3)):
        t, ctrl, q = (np.array([s[k] for s in j["states"]]) for k in ("t", "ctrl", "q"))
        assert np.allclose(t, np.arange(16) * 0.25) and np.allclose(ctrl, top * ramp)
        # Inside a ramp the joint trails the ctrl by (kv + damping) / kp times the rate, 2.4 / w s at any inertia
        # (less one time step, for the ctrl leads the state it acts on); at the end of each hold it is there.
        assert np.allclose(ctrl[2:5] - q[2:5], 2.4 / mjcf.OMEGA * top, rtol=0.05)
        assert np.abs(ctrl - q)[[5, 10, 15]].max() < 0.005 * top and max(j["errors"]) < 0.005 * top


def test_a_joint_that_ends_past_the_tolerance_fails_tracking(tmp_path, hulls):
    # The ctrl stays within the limits, so q stops at the upper limit of an interval that goes past it. 0.1 rad
    # (5.7 degrees) and 5 mm (0.0063 D) past are beyond the tolerances, 0.07 rad and 2 mm are not.
    glb = chest(tmp_path)
    D = np.sqrt(0.6 ** 2 + 0.4 ** 2 + 0.34 ** 2)
    out, intervals = tmp_path / "sim", tmp_path / "intervals.json"
    argv = ["--glb", str(glb), "--out", str(out), "--asset-sha", SHA, "--intervals", str(intervals)]
    intervals.write_text(json.dumps({"lid": [0.0, 1.6], "drawer": [0.0, 0.305]}))
    sim_check.main(argv)
    r = report(out)
    assert [(f["code"], f["link"]) for f in r["findings"]] == [("tracking", "lid"), ("tracking", "drawer")]
    lid, drawer = r["findings"]
    assert 0.1 <= lid["measured"] < 0.105 and np.isclose(lid["expected"], np.radians(5))
    assert 0.005 <= drawer["measured"] < 0.006 and np.isclose(drawer["expected"], 0.005 * D)
    assert all(max(j["errors"][1:]) < 0.01 for j in r["joints"])  # only the upper end is out of reach
    # The report names the candidate it came from, which is not model.glb.
    assert r["asset_sha256"] == SHA and r["glb_sha256"] == hashlib.sha256(glb.read_bytes()).hexdigest() != SHA
    intervals.write_text(json.dumps({"lid": [0.0, 1.57], "drawer": [0.0, 0.302]}))
    sim_check.main(argv)
    assert report(out)["findings"] == []
    # Only the links of the file run.
    intervals.write_text(json.dumps({"lid": [0.0, 0.5]}))
    sim_check.main(argv)
    (lid,) = report(out)["joints"]
    assert (lid["link"], lid["interval"], max(s["ctrl"] for s in lid["states"])) == ("lid", [0.0, 0.5], 0.5)


def test_a_glb_without_a_hinge_slide_or_soft_joint_is_not_simulated(tmp_path):
    # No joint, or only a fixed base and a fixed lid: nothing runs, and no model is built.
    rigid = write_parts(tmp_path / "rigid" / "parts.glb", {"body": box([0.6, 0.4, 0.3], [0, 0, 0.15])}, {})
    fixed = write_parts(tmp_path / "fixed" / "parts.glb", {"body": box([0.6, 0.4, 0.3], [0, 0, 0.15]),
                                                           "lid": box([0.6, 0.4, 0.04], [0, 0, 0.32])},
                        {"body": {"version": 1, "frame": "S", "parent": "world", "type": "fixed"},
                         "lid": {"version": 1, "frame": "S", "parent": "body", "type": "fixed"}})
    for glb in (rigid, fixed):
        r = sim_check.run(glb, glb.parent / "sim", SHA)
        assert r["skipped"] and (r["asset_sha256"], r["joints"], r["soft"], r["findings"]) == (SHA, [], [], [])
        assert not (glb.parent / "sim").exists()


def test_a_failure_writes_an_error_and_exits_0(tmp_path):
    # Joints still marked W fail in harpor.physics.mjcf. The command line runs in envs/mujoco, which cannot import
    # the agent's modules.
    glb = chest(tmp_path)
    doc, binary = read_glb(glb)
    write_glb(glb, write_joints(doc, {k: {**j, "frame": "W"} for k, j in read_joints(doc).items()}), binary)
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    p = subprocess.run([sys.executable, "-m", "harpor.physics.sim_check", "--glb", str(glb), "--out",
                        str(tmp_path / "sim"), "--asset-sha", SHA], cwd=REPO, env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    r = report(tmp_path / "sim")
    assert set(r) == {"asset_sha256", "error"} and r["asset_sha256"] == SHA
    assert r["error"].startswith("ValueError: ") and "must be in frame S" in r["error"]
    code = "import sys; import harpor.physics.sim_check; print(sorted(m for m in sys.modules if m.startswith('harpor.agent')))"
    assert subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True).stdout == "[]\n"


def test_the_press_dents_the_pad_and_it_springs_back(tmp_path, hulls):
    # D is 0.89 m, so the target is 2.7 mm, which the force reaches below its cap.
    r = sim_check.run(pad_on_block(tmp_path), tmp_path / "sim", SHA)
    (p,) = r["soft"]
    D, pitch = r["diagonal_m"], json.loads((tmp_path / "sim" / "mjcf" / "mjcf.json").read_text())["soft"]["pad"]["pitch"]
    assert r["joints"] == [] and r["findings"] == [] and p["press_nodes"] > 9 and p["direction"] == [0, 0, -1]
    assert np.isclose(p["cap_n"], 0.08 * 3e4 * pitch ** 2) and not p["cap_limited"] and p["force_n"] < p["cap_n"]
    assert np.isclose(p["target_m"], 0.003 * D) and p["target_m"] <= p["dmax_m"] < 1.2 * p["target_m"]
    assert min(p["visible_dmax_m"], p["pairwise_m"]) > 0.002 * D
    assert p["residual_m"] < 1e-6 and p["shift_m"] < 1e-9 and np.isclose(p["recovery_limit_m"], 0.35 * p["dmax_m"])
    assert [s["phase"] for s in p["states"]] == ["start", "ramp", "hold", "release", "settle"]
    assert np.isclose(p["states"][-1]["t"] - p["states"][1]["t"], 0.25 + 0.25 + 1.5)


def built_pad(tmp_path):
    """(model with zero gravity, binding, D) of pad_on_block, as sim_check builds it."""
    glb = pad_on_block(tmp_path)
    (tmp_path / "summary.json").write_text(json.dumps({"median": {"density_kg_m3": 1000.0}}))
    mjcf.build(glb, tmp_path / "summary.json", tmp_path / "mjcf", "object", fixed_base=True, textures=False)
    m = mujoco.MjModel.from_xml_path(str(tmp_path / "mjcf" / "object.xml"))
    m.opt.gravity[:] = 0
    return m, dict(np.load(tmp_path / "mjcf" / "soft" / "pad.npz")), sim_check.diagonal(glb)


def test_the_press_stops_rising_at_the_target_or_at_the_cap(tmp_path, hulls):
    m, b, D = built_pad(tmp_path)
    rows, target = b["tets"][b["tet_of_vertex"]], 0.003 * D
    d = mujoco.MjData(m)
    # With a cap far above what the target takes, the force stops rising at the first sample that reaches it.
    big = sim_check.press(m, d, 0, b["press"], b["direction"], target, 10.0)
    ramp, hold = big["ends"]["ramp"], big["ends"]["hold"]
    assert not big["cap_limited"] and ramp > 5 and big["residual"][ramp] >= target > big["residual"][ramp - 1]
    assert np.allclose(big["force"][:ramp + 1], 0.1 * np.arange(1, ramp + 2))
    assert (big["force"][ramp:hold + 1] == big["force"][ramp]).all() and big["force"][hold + 1] < big["force"][ramp]
    # 0.1 N a node: the force ramps to the cap and no further, and the dent stays under 0.001 D. That is a test
    # that could not be done, not a part that does not respond.
    mujoco.mj_resetData(m, d)
    small = sim_check.press(m, d, 0, b["press"], b["direction"], target, 0.1)
    assert small["cap_limited"] and small["ends"]["ramp"] == 99 and np.isclose(small["force"].max(), 0.1)
    assert np.isclose(small["force"][small["ends"]["hold"]], 0.1) and small["residual"].max() < 0.001 * D
    _, found = sim_check.soft_findings("pad", small, rows, b["weights"], D)
    assert [f["code"] for f in found] == ["soft_untestable"] and "force cap" in found[0]["why"]
    _, found = sim_check.soft_findings("pad", {**small, "cap_limited": False}, rows, b["weights"], D)
    assert [f["code"] for f in found] == ["soft_no_response"]


def test_inertia_is_no_warning_in_a_model_with_a_flex(tmp_path, hulls):
    # MuJoCo 3.16 reports a flex's M+K solve that ran out of iterations as INERTIA, as a cushion on a hinged lid and a
    # foot pad on the body raise it while both press and the hinge track. A rigid model keeps it.
    flex, _, _ = built_pad(tmp_path)
    rigid = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><joint/><geom size="0.1"/></body></worldbody></mujoco>')
    for m, want in ((flex, ["BADQACC"]), (rigid, ["INERTIA", "BADQACC"])):
        d = mujoco.MjData(m)
        assert sim_check.warnings(m, d) == []
        d.warning[mujoco.mjtWarning.mjWARN_INERTIA].number = 1
        d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number = 1
        assert sim_check.warnings(m, d) == want


def dent(P0, node, depth):
    """P0 with one node pushed down by depth."""
    P = P0.copy()
    P[node, 2] -= depth
    return P


def trace(P0, samples):
    """A press trace of these node samples from P0, as press() gives it."""
    return {"rest": P0, "P0": P0, "P": np.array(samples), "cap_limited": False,
            "residual": np.array([np.linalg.norm(sim_check.residual(P, P0), axis=1).max() for P in samples])}


def test_recovery_needs_the_residual_back_under_035_dmax_or_0001_d():
    # 20 nodes, each its own skin vertex, with one node pressed 10 mm and then let go to a part of that. D is 1 m, so
    # the floor is 1 mm. The rigid fit takes the same share of each dent, so the residuals keep their ratio.
    P0 = np.random.default_rng(0).uniform(-0.15, 0.15, (20, 3))
    rows, weights = np.repeat(np.arange(20)[:, None], 4, 1), np.tile([1.0, 0, 0, 0], (20, 1))
    for depth, left, codes in ((0.01, 0.37, ["soft_no_recovery"]), (0.01, 0.33, []),
                               (0.002, 0.45, []), (0.002, 0.6, ["soft_no_recovery"])):
        t = trace(P0, [dent(P0, 3, depth), dent(P0, 3, left * depth)])
        measures, found = sim_check.soft_findings("pad", t, rows, weights, 1.0)
        assert [f["code"] for f in found] == codes, (depth, left)
        assert np.isclose(measures["residual_m"] / measures["dmax_m"], left, rtol=1e-3)
        assert np.isclose(measures["recovery_limit_m"], max(0.001, 0.35 * measures["dmax_m"]))


def test_a_part_still_dented_at_the_end_fails_recovery(tmp_path, hulls, monkeypatch):
    # Without the release and the settle the press ends under full force: the report says the pad did not spring back.
    monkeypatch.setattr(sim_check, "RELEASE_S", 0.0)
    monkeypatch.setattr(sim_check, "RECOVER_S", 0.0)
    r = sim_check.run(pad_on_block(tmp_path), tmp_path / "sim", SHA)
    (p,) = r["soft"]
    assert [(f["code"], f["link"]) for f in r["findings"]] == [("soft_no_recovery", "pad")]
    assert p["residual_m"] > 0.9 * p["dmax_m"] and r["findings"][0]["expected"] == p["recovery_limit_m"]


def test_a_soft_part_without_support_or_a_press_direction_cannot_be_tested(tmp_path, hulls):
    # A pad 20 cm above its block has no usable mesh, so it stays rigid with the code of harpor.physics.mjcf.
    r = sim_check.run(pad_on_block(tmp_path / "far", lift=0.2), tmp_path / "far" / "sim", SHA)
    assert [(f["code"], f["link"]) for f in r["findings"]] == [("soft_no_support", "pad")] and r["soft"] == []
    # A sleeve's parent normals cancel out: there is no press direction.
    bar = trimesh.creation.cylinder(radius=0.03, height=0.6, sections=32)
    sleeve = trimesh.creation.annulus(r_min=0.03, r_max=0.06, height=0.2, sections=32)
    glb = write_parts(tmp_path / "sleeve" / "parts.glb", {"bar": bar, "sleeve": sleeve}, {"sleeve": soft("bar")})
    r = sim_check.run(glb, glb.parent / "sim", SHA)
    assert [(f["code"], f["why"]) for f in r["findings"]] == [("soft_untestable", "it has no press direction, as a sleeve")]
    assert r["soft"][0]["direction"] == [0, 0, 0]
