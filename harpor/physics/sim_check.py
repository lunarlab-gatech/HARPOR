"""Execute the joints and press the soft parts of a candidate in MuJoCo (build_plan.md note 11, paper A.5 and A.6).

    python -m harpor.physics.sim_check --glb model.glb --out DIR --asset-sha SHA [--intervals FILE]

model.glb is the candidate with its joints marked S, though they keep W's numbers (harpor.agent.validate writes
it). The model has a fixed base, zero gravity and no floor, so which way is up does not matter. DIR gets
summary.json, a stub with only a median density, so that every link's inertial is its hulls filled at 1000 kg/m^3;
mjcf/, the model of harpor.physics.mjcf without textures; and report.json. --asset-sha is the candidate's sha256,
not model.glb's: the report carries it, so that its states name the asset they came from (A.6). --intervals is a
JSON file {link: [lower, upper]} of the intervals to execute, the ones the clearance search achieved; without it
every hinge and slide runs over its own limits. Any failure writes {"error": ...} to report.json and exits 0,
since the caller reads the report. A GLB without a hinge, slide or soft joint is not simulated: the report says
skipped.

Execution, one joint at a time with the others at 0: the ctrl of its position actuator ramps linearly from rest to
the upper end of its interval, to the lower end and back to rest, each leg over 1 s and then held for 0.25 s,
because a step target shakes a soft part on the joint. A state {t, ctrl, q} is recorded every 0.25 s. The ctrl
stays within the joint's limits, so an interval past a limit is not reached. D is the asset's diagonal, and the
tolerance is 5 degrees for a hinge and 0.005 D for a slide. tracking: the end of a leg further than that from its
target. unstable: a state that is not finite, a MuJoCo warning, or q past a limit by more than that. MuJoCo 3.16
reports a flex's M+K solve that ran out of iterations as a singular inertia, which two flexes on different bodies
of a moving object raise while the motion stays sound, so in a model with a flex INERTIA is not a warning here;
the state checks and the recovery still catch a part that fails.

Press, for each soft link, on its binding soft/<link>.npz: settle for 0.25 s and take the nodes there as P0, which
must lie within 0.001 D of rest, or the link is unstable. Then push the press nodes along the press direction with
a force that ramps over 1 s to a cap of 0.08 E h^2 per node (E the Young's modulus, h the pitch), sampled every
10 ms, and stops rising once the largest residual reaches the target 3 x 0.001 D; hold it 0.25 s, release it over
0.25 s and settle for 1.5 s. A sample's residual is the nodes' displacement from P0 left after the least-squares
rigid fit onto P0 (Kabsch), so that motion of the whole part does not count, and dmax is its largest value over
the samples. The visible dmax is the residual field of that sample carried to the skin by Eq. 4.
- soft_no_response: dmax, the largest change of a distance between two nodes there, or the visible dmax is under
  0.001 D. It is soft_untestable instead when the force reached the cap without the target, because the cap
  scales with E, so no other material can help.
- soft_untestable also without a press direction (a sleeve) or press nodes. A link that has no usable mesh keeps
  the code of mjcf.json soft_rigid.
- soft_no_recovery: the residual after the settle is above max(0.001 D, 0.35 dmax).
- binding_mismatch: the problems of harpor.physics.mjcf.check_binding.
- unstable: a node that is not finite, or a MuJoCo warning.

report.json holds asset_sha256, glb_sha256, runtime, diagonal_m, joints [{link, joint, type, unit, interval,
limits, tolerance, errors, states}], soft [{link, flex, press_nodes, direction, target_m, cap_n, force_n,
cap_limited, dmax_m, visible_dmax_m, pairwise_m, residual_m, recovery_limit_m, shift_m, states}], findings [{code,
link, measured, expected, why}] and seconds. press() runs the press alone, so that other callers can reuse it, also
to another measure of the residual field (the soft-part video's visible dent, harpor.webapp.softvideo).
It runs in envs/mujoco and imports nothing from harpor.agent, so not harpor.agent.tools either.
"""

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np

from harpor.parts import joints as joint_spec
from harpor.physics import mjcf

NAME = "object"
DENSITY = 1000.0  # kg/m^3, the stub summary's median
LEG_S, HOLD_S, STATE_S = 1.0, 0.25, 0.25  # a leg's ramp, its hold, and the time between recorded states
ANGLE_TOL, DIAG_TOL = np.radians(5.0), 0.005  # endpoint tolerances: a hinge's, and a slide's as a share of D
SETTLE_S, RAMP_S, RELEASE_S, RECOVER_S, SAMPLE_S = 0.25, 1.0, 0.25, 1.5, 0.01  # the press; the hold is HOLD_S
RESPONSE = 0.001  # of D: the smallest response and the recovery floor
TARGET = 3  # times RESPONSE: the press stops rising there
CAP = 0.08  # times E h^2: the force per node
RECOVERY = 0.35  # of dmax


def finding(code, link, measured=None, expected=None, why=""):
    return {"code": code, "link": link, "measured": measured, "expected": expected, "why": why}


def diagonal(glb):
    """D: the diagonal of the box around every vertex of the GLB."""
    v = np.concatenate([m.vertices for _, m in mjcf.load_s(glb)])
    return float(np.linalg.norm(v.max(0) - v.min(0)))


def steps(m, seconds):
    return int(round(seconds / m.opt.timestep))


def warnings(m, d):
    """Names of the MuJoCo warnings raised since the last reset, but INERTIA in a model with a flex (see the module
    docstring)."""
    import mujoco

    names = [mujoco.mjtWarning(i).name.removeprefix("mjWARN_") for i, w in enumerate(d.warning) if w.number]
    return [n for n in names if not (n == "INERTIA" and m.nflex)]


def execute(m, d, name, interval):
    """Run the joint name (and its actuator, of the same name) over interval [lower, upper] from rest, with the
    other actuators at 0 (see the module docstring). Returns (states, the ends [{target, q}], q's range seen)."""
    import mujoco

    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    act, adr = m.actuator(name).id, m.jnt_qposadr[m.joint(name).id]
    ramp, hold, every = steps(m, LEG_S), steps(m, HOLD_S), steps(m, STATE_S)
    states, ends, seen = [], [], [0.0, 0.0]
    ctrl = 0.0

    def record():
        states.append({"t": float(d.time), "ctrl": float(ctrl), "q": float(d.qpos[adr])})

    record()
    for target in (interval[1], interval[0], 0.0):
        start = ctrl
        for k in range(1, ramp + hold + 1):
            ctrl = start + (target - start) * min(k, ramp) / ramp
            d.ctrl[act] = ctrl
            mujoco.mj_step(m, d)
            q = float(d.qpos[adr])
            seen = [min(seen[0], q), max(seen[1], q)]
            if k % every == 0:
                record()
        ends.append({"target": float(target), "q": q})
    return states, ends, seen


def joint_record(m, d, joint, name, interval, D):
    """(report entry, findings) of executing one hinge or slide. joint: its entry of mjcf.json joints; name: its
    joint and actuator."""
    link, hinge = joint["child"], joint["type"] == "revolute"
    tol, unit = (ANGLE_TOL, "rad") if hinge else (DIAG_TOL * D, "m")
    states, ends, seen = execute(m, d, name, interval)
    errors = [abs(e["q"] - e["target"]) for e in ends]
    out = [finding("tracking", link, err, tol, f"q ends at {e['q']:.4g} {unit}, {err:.3g} {unit} from its target "
                                               f"{e['target']:.4g} {unit}")
           for e, err in zip(ends, errors) if err > tol]
    lo, hi = joint["range"]
    if seen[0] < lo - tol or seen[1] > hi + tol:
        out.append(finding("unstable", link, seen, [lo, hi], f"q reached [{seen[0]:.4g}, {seen[1]:.4g}] {unit}, past "
                                                            f"its limits [{lo:.4g}, {hi:.4g}]"))
    if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()) or warnings(m, d):
        out.append(finding("unstable", link, warnings(m, d), [], "the state is not finite or MuJoCo warned"))
    entry = {"link": link, "joint": name, "type": joint["type"], "unit": unit, "interval": list(interval),
             "limits": [lo, hi], "tolerance": tol, "errors": errors, "states": states}
    return entry, out


def residual(P, P0):
    """(N, 3) displacements of the points P from P0 left after the least-squares rigid fit of P onto P0 (Kabsch)."""
    a, b = P - P.mean(0), P0 - P0.mean(0)
    U, _, Vt = np.linalg.svd(a.T @ b)
    R = U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt  # a @ R is a turned onto b
    return a @ R - b


def largest(field):
    """The largest displacement of a residual field (N, 3)."""
    return float(np.linalg.norm(field, axis=1).max())


def press(m, d, flex, nodes, direction, target, cap, measure=largest):
    """Press the nodes (indices into the flex's vertices) along direction from the current state: settle, a force per
    node that ramps to cap and stops rising once the largest residual reaches target, hold, release and settle (see
    the module docstring). measure: what of the residual field (N, 3) the target is for, by default its largest node
    displacement. Returns {rest, P0, t, P (samples, nodes, 3), residual, force, cap_limited, ends}: the nodes before
    the settle and after it; the time from the start, the nodes, their measure and the force per node of every 10 ms
    sample; whether the force reached the cap without the target; and the last sample of each phase."""
    import mujoco

    adr, num = int(m.flex_vertadr[flex]), int(m.flex_vertnum[flex])
    bodies = m.flex_vertbodyid[adr + np.asarray(nodes, int)]
    every = steps(m, SAMPLE_S)
    mujoco.mj_forward(m, d)
    t0, rest = d.time, d.flexvert_xpos[adr:adr + num].copy()
    mujoco.mj_step(m, d, steps(m, SETTLE_S))
    P0 = d.flexvert_xpos[adr:adr + num].copy()
    t, P, res, force, ends = [], [], [], [], {}

    def sample(f):
        d.xfrc_applied[bodies, :3] = f * np.asarray(direction, float)
        mujoco.mj_step(m, d, every)
        t.append(d.time - t0)
        P.append(d.flexvert_xpos[adr:adr + num].copy())
        res.append(measure(residual(P[-1], P0)))
        force.append(f)

    n, f = round(RAMP_S / SAMPLE_S), 0.0
    for k in range(1, n + 1):
        f = cap * k / n
        sample(f)
        if res[-1] >= target:
            break
    ends["ramp"] = len(P) - 1
    reached = res[-1] >= target
    for _ in range(round(HOLD_S / SAMPLE_S)):
        sample(f)
    ends["hold"] = len(P) - 1
    n = round(RELEASE_S / SAMPLE_S)
    for k in range(1, n + 1):
        sample(f * (1 - k / n))
    ends["release"] = len(P) - 1
    for _ in range(round(RECOVER_S / SAMPLE_S)):
        sample(0.0)
    ends["settle"] = len(P) - 1
    d.xfrc_applied[bodies] = 0
    return {"rest": rest, "P0": P0, "t": np.array(t), "P": np.array(P), "residual": np.array(res),
            "force": np.array(force), "cap_limited": not reached, "ends": ends}


def soft_findings(link, trace, rows, weights, D):
    """(measures, findings) of a press trace. rows: the 4 nodes of each skin vertex's tetrahedron, weights: W of
    Eq. 4."""
    from scipy.spatial.distance import pdist

    res, P0 = trace["residual"], trace["P0"]
    peak = int(res.argmax())
    skin = (residual(trace["P"][peak], P0)[rows] * weights[..., None]).sum(1)  # Eq. 4 on the residual field
    dmax, floor = float(res[peak]), RESPONSE * D
    measures = {"dmax_m": dmax, "visible_dmax_m": float(np.linalg.norm(skin, axis=1).max()),
                "pairwise_m": float(np.abs(pdist(trace["P"][peak]) - pdist(P0)).max()),
                "residual_m": float(res[-1]), "recovery_limit_m": max(floor, RECOVERY * dmax),
                "shift_m": float(np.linalg.norm(P0 - trace["rest"], axis=1).max())}
    out = []
    if measures["shift_m"] > floor:
        out.append(finding("unstable", link, measures["shift_m"], floor, "the nodes moved while settling at rest"))
    response = {k: measures[k] for k in ("dmax_m", "pairwise_m", "visible_dmax_m")}
    if min(response.values()) < floor:
        code = "soft_untestable" if trace["cap_limited"] else "soft_no_response"
        why = "the press reached its force cap" if trace["cap_limited"] else "the press reached its target"
        out.append(finding(code, link, response, floor, f"{why}, yet the part responds less than 0.001 D"))
    if measures["residual_m"] > measures["recovery_limit_m"]:
        out.append(finding("soft_no_recovery", link, measures["residual_m"], measures["recovery_limit_m"],
                           "the part does not spring back after the press"))
    return measures, out


def soft_record(m, d, rec, npz, D):
    """(report entry, findings) of pressing one soft link. rec: its entry of mjcf.json soft."""
    import mujoco

    with np.load(npz) as z:
        b = {k: z[k] for k in z.files}
    link, flex = str(b["link"]), str(b["flex"])
    entry = {"link": link, "flex": flex, "press_nodes": len(b["press"]), "direction": b["direction"].tolist()}
    if not b["direction"].any():
        return entry, [finding("soft_untestable", link, why="it has no press direction, as a sleeve")]
    if not len(b["press"]):
        return entry, [finding("soft_untestable", link, why="no press node lies under its far face")]
    mujoco.mj_resetData(m, d)
    target, cap = TARGET * RESPONSE * D, CAP * rec["young"] * rec["pitch"] ** 2
    trace = press(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_FLEX, flex), b["press"], b["direction"], target, cap)
    measures, out = soft_findings(link, trace, b["tets"][b["tet_of_vertex"]], b["weights"], D)
    if not np.isfinite(trace["P"]).all() or warnings(m, d):
        out.append(finding("unstable", link, warnings(m, d), [], "a node is not finite or MuJoCo warned"))
    states = [{"t": SETTLE_S, "phase": "start", "force_n": 0.0, "residual_m": 0.0}]  # P0
    states += [{"t": float(trace["t"][i]), "phase": phase, "force_n": float(trace["force"][i]),
                "residual_m": float(trace["residual"][i])} for phase, i in trace["ends"].items()]
    entry.update({"target_m": target, "cap_n": cap, "force_n": float(trace["force"].max()),
                  "cap_limited": trace["cap_limited"], **measures, "states": states})
    return entry, out


def runtime(m):
    import mujoco

    return {"mujoco": mujoco.__version__, "dt": float(m.opt.timestep), "fixed_base": True, "gravity": 0,
            "integrator": mujoco.mjtIntegrator(m.opt.integrator).name.removeprefix("mjINT_").lower(),
            "solver": mujoco.mjtSolver(m.opt.solver).name.removeprefix("mjSOL_"),
            "tolerance_angle_deg": float(np.degrees(ANGLE_TOL)), "tolerance_diag": DIAG_TOL}


def run(glb, out, asset_sha, intervals=None):
    """The report of one GLB (see the module docstring). intervals: {link: [lower, upper]}, or None for the limits."""
    import mujoco

    start, out = time.time(), Path(out)
    report = {"asset_sha256": asset_sha, "glb_sha256": hashlib.sha256(Path(glb).read_bytes()).hexdigest()}
    joints, _ = joint_spec.load(glb)
    if not any(j.get("type") != "fixed" for j in joints.values()):
        return {**report, "skipped": "the GLB has no hinge, slide or soft joint", "joints": [], "soft": [],
                "findings": []}
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"median": {"density_kg_m3": DENSITY}}))
    shutil.rmtree(out / "mjcf", ignore_errors=True)
    mj = mjcf.build(glb, out / "summary.json", out / "mjcf", NAME, fixed_base=True, textures=False)
    built = time.time()
    m = mujoco.MjModel.from_xml_path(str(out / "mjcf" / f"{NAME}.xml"))
    m.opt.gravity[:] = 0
    d = mujoco.MjData(m)
    D = diagonal(glb)
    findings = [finding(r["code"], link, why=r["reason"]) for link, r in mj["soft_rigid"].items()]
    executed = []
    for j in mj["joints"]:
        link = j["child"]
        if link not in mj["actuators"] or (intervals is not None and link not in intervals):
            continue
        entry, found = joint_record(m, d, j, mj["actuators"][link]["name"],
                                    j["range"] if intervals is None else intervals[link], D)
        executed.append(entry)
        findings += found
    pressed = []
    for link, rec in mj["soft"].items():
        npz = out / "mjcf" / rec["file"]
        problems = mjcf.check_binding(m, npz, glb)
        if problems:
            findings.append(finding("binding_mismatch", link, why="; ".join(problems)))
        entry, found = soft_record(m, d, rec, npz, D)
        pressed.append(entry)
        findings += found
    return {**report, "runtime": runtime(m), "diagonal_m": D, "joints": executed, "soft": pressed,
            "findings": findings, "seconds": {"build": built - start, "total": time.time() - start}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--asset-sha", required=True, help="the candidate's sha256, which the report carries")
    ap.add_argument("--intervals", help="JSON {link: [lower, upper]} of the intervals to execute")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        intervals = json.loads(Path(a.intervals).read_text()) if a.intervals else None
        report = run(a.glb, out, a.asset_sha, intervals)
    except Exception as e:  # noqa: BLE001 - the caller reads the report, so every failure goes there
        report = {"asset_sha256": a.asset_sha, "error": f"{type(e).__name__}: {e}"}
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report.get(k) for k in ("error", "skipped", "findings", "seconds") if k in report}))


if __name__ == "__main__":
    main()
