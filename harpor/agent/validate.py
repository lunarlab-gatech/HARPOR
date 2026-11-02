"""The host's check of an articulation candidate (build_plan.md note 11; paper A.3, A.5 and A.6).

check(glb, control, original, timeout) runs on the host in envs/sam3d, during a round through the simulate helper
(harpor.agent.tools), while the agent is alive, or after it. It reads the GLB once without following a symlink,
hashes it and copies it to control/validate/<sha16>/candidate.glb, and every later step reads only that copy:
1. Joints and links (harpor.parts.joints): J.validate's problems, and joints in a frame other than W, are
   invalid_joints, and nothing below runs on them. The link names must be file names (harpor.physics.align.
   safe_links), since harpor.physics.mjcf makes files and XML of them: else bad_link_name, and MuJoCo does not run.
2. The surface against original, root/original.glb (harpor.agent.surface.compare); its failure is surface_error.
3. Clearance in W on the link meshes (tools.link_meshes, tools.penetration), one hinge or slide at a time with every
   other joint at 0. The reference [l0, h0] (reference) is the joint's limits, each side widened to the entry
   "<link>|<type>" of control/ranges.json, since the original interval stays the reference across repairs (paper
   A.5): a repair can widen a range, or turn it around, as a side whose limit is 0 has no reference, but a narrower
   range keeps the wider reference. Only check(..., pin_ranges=True), the check at a round's end, writes entries,
   so a draft that simulate checks does not set one, and a new type gets its own. At rest the moving links (the
   joint's child and the links it carries) must not lie in another link (rest_collision), and every other link must
   be closed, or inside and outside say nothing (clearance_missing); either ends the search (paper A.5). Then for
   n = 0, 1, 2 the interval [l0, h0] (2/3)^n is swept at STEPS states each way from rest to its ends (none toward an
   end at 0) until one passes; a state fails when a sample lies deeper than the tolerance inside another link. A
   reduced interval is tried only while its span keeps MIN_SHARE of h0 - l0, and MIN_HINGE for a hinge or the
   tolerance for a slide, under which a joint that does not move would still track. None passing is axis_blocked,
   one passing after a reduction range_blocked, and an achieved span under those floors, or one that the joint's
   limits cut, range_short. The search stops with sim_timeout when the check's time is up.
4. MuJoCo: model.glb is the copy with its joints marked S but W's numbers (J.transform with the identity), which
   harpor.physics.sim_check builds with a fixed base, zero gravity and no floor, so up does not matter. It runs as
   GATOR_MUJOCO (split like a shell) -m harpor.physics.sim_check through tools._run, from the repository with
   PYTHONSAFEPATH=1, so the helper's SIGTERM kills it too, within what the other steps left of timeout. Each hinge
   and slide runs over its achieved interval within its limits (executed; without one it does not run) and each
   soft part is pressed; their findings count as sim_check gives them, the codes of mjcf.json soft_rigid among them.
   Its error is compile_error; GATOR_MUJOCO unset, not starting, exiting without a report or without its modules is
   sim_unavailable, and running out of time sim_timeout. A GLB without a hinge, slide or soft joint is not
   simulated.
5. assess(requirements, report), with the requirements of control/parts.json.

The report is cached at control/validate/<sha16>/report.json, so the check at a round's end reuses a simulate of
the same bytes, unless a reference changed since. A report with a failed step (errors, or the surface's or MuJoCo's
error, which the sim_* codes and compile_error come with) is never cached.

assess gives each requirement a record {requirement, target, kind, joint, text, required, action, initial_state,
measured, findings, outcome}. A finding is {code, requirement, link, measured, expected, why}: a link's finding is
the requirement's of that link, and a requirement adds link_missing, joint_missing, joint_type_mismatch (joint
unknown takes either moving joint), parent_mismatch (the joint's parent does not move with the requirement's
parent, report carriers), range_short (an executed span short of its test min_span by more than SPAN_TOL) and
surface_assignment (its link keeps under KEPT of its own native area). These stay in the record of a
requirement that is not required, so an agent that joins a fragment to its part is not asked for it. A part that
was not found, or one not required that is gone, is unassessed. The outcomes (paper A.3): fail with a finding not in
INCONCLUSIVE; else inconclusive with a finding, or without a trajectory of at least MIN_STATES states of this
candidate's sha256 for the link (A.6); else pass. Surface findings count only while surface.ENFORCE is on; until
then they are reported and decide nothing. scope (SCOPE, paper A.3) is structural with a structural finding, else
local with a local one, else none, and sim_unavailable makes it none, since no repair can be checked. outcome is
pass when every required found requirement passes, the surface does not block and no finding is structural; the
agent's verdict is not part of it.

round_report(control, report, pick, errors) is the report of a round of the session (harpor.agent.controller):
it adds what the round's pick says, no_candidate or not_finished, so the round's outcome is pass only when the
agent's verdict is articulated too. feedback(report) is the report as text for the agent's next prompt.
"""

import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from harpor.agent import surface, tools
from harpor.fileio import read_json, write_json
from harpor.parts import joints as J
from harpor.physics import align

REPO = Path(__file__).resolve().parents[2]
HOST_CHECK_TIMEOUT_S = 600.0  # the check at a round's end; simulate's is shorter (tools.SIMULATE_TIMEOUT_S)
SHRINK, REDUCTIONS = 2 / 3, 2  # paper A.5: [l0, h0] (2/3)^n, n up to 2
STEPS = 4  # sweep states each way from rest
MIN_SHARE, MIN_HINGE = 0.2, math.radians(10.0)  # the retained span: of h0 - l0, and at least this for a hinge
SPAN_TOL = math.radians(5.0)  # an achieved span meets a min_span within this, so +-3.14 rad meets 6.283
KEPT = 0.5  # of a requirement link's own native area
MIN_STATES = 3  # of a trajectory (paper A.6)
MISSING = ("ModuleNotFoundError", "ImportError")  # sim_check errors of an env without its modules
MAX_FEEDBACK, MAX_TEXT = 6000, 300  # characters of feedback, and of one text in it before quoting
LOCAL, STRUCTURAL, NONE = "local", "structural", "none"
SCOPE = {  # paper A.3: local repairs, structural replanning, or nothing an agent can do
    "no_candidate": LOCAL, "not_finished": LOCAL, "bad_link_name": LOCAL, "joint_missing": LOCAL,
    "range_blocked": LOCAL, "range_short": LOCAL, "clearance_missing": LOCAL, "tracking": LOCAL, "unstable": LOCAL,
    "compile_error": LOCAL, "sim_timeout": LOCAL, "binding_mismatch": LOCAL, "surface_uv": LOCAL,
    "surface_material": LOCAL, "surface_deleted": LOCAL, "surface_flipped": LOCAL, "surface_error": LOCAL,
    "surface_assignment": LOCAL, "validation_error": LOCAL,
    "link_missing": STRUCTURAL, "joint_type_mismatch": STRUCTURAL, "parent_mismatch": STRUCTURAL,
    "invalid_joints": STRUCTURAL,
    "rest_collision": STRUCTURAL, "axis_blocked": STRUCTURAL, "surface_identity": STRUCTURAL,
    "soft_no_support": STRUCTURAL, "soft_no_response": STRUCTURAL, "soft_no_recovery": STRUCTURAL,
    "soft_untestable": NONE, "sim_unavailable": NONE}
# the test could not be done: the requirement is inconclusive, not failed
INCONCLUSIVE = ("compile_error", "sim_timeout", "sim_unavailable", "clearance_missing", "soft_untestable",
                "validation_error")
LABELS = (STRUCTURAL, LOCAL, NONE, "report only")  # the order of findings in feedback


class OutOfTime(Exception):
    """The check's time ran out in a host step, which ends the check with sim_timeout (run)."""


def finding(code, link=None, measured=None, expected=None, why="", requirement=None):
    return {"code": code, "requirement": requirement, "link": link, "measured": measured, "expected": expected,
            "why": why}


def requirements(control):
    """The requirements of control/parts.json, the host's copy of the parts stage's record ([] without them)."""
    doc = read_json(Path(control) / "parts.json")
    reqs = doc.get("requirements") if isinstance(doc, dict) else None
    return [r for r in reqs if isinstance(r, dict)] if isinstance(reqs, list) else []


def read_ranges(control):
    """control/ranges.json, {"<link>|<type>": [l0, h0]} of the references; {} without it."""
    ranges = read_json(Path(control) / "ranges.json")
    return ranges if isinstance(ranges, dict) else {}


def directions(lo, hi, steps=STEPS):
    """The joint values of a sweep from rest: steps each way to hi and to lo, none toward an end at 0."""
    return [end * k / steps for end in (hi, lo) if end for k in range(1, steps + 1)]


def blocked(rep, child, qs):
    """The first state of a penetration report with samples inside another link, as {q, inside_share, max_depth,
    worst_other_link}; None when all are clear. qs: the joint values of its states. max_depth is above 0 only with
    such a sample, while inside_share, rounded, can be 0 with one among many."""
    for q, s in zip(qs, rep["joints"][child]["states"]):
        if s["max_depth"] > 0:
            return {"q": q, **{k: s[k] for k in ("inside_share", "max_depth", "worst_other_link")}}
    return None


def in_time(deadline):
    """Raises OutOfTime when the check's time is up."""
    if time.time() >= deadline:
        raise OutOfTime("the clearance search ran past the check's time")


def clearance(meshes, joints, links, held, child, reference, deadline=math.inf):
    """The clearance search of the hinge or slide of child from reference [l0, h0] (see the module docstring):
    {type, limits, reference, floor, tolerance, rest, open, tests [{n, interval, blocked}], achieved, reductions}.
    meshes: tools.link_meshes; joints: full; held: holders. Raises OutOfTime when a sweep would start after
    deadline."""
    j = joints[child]
    carrier = tools.carriers(joints, links, held)
    l0, h0 = reference
    in_time(deadline)
    rest = tools.penetration(meshes, {child: j}, tools.sweep_poses(joints, links, held, child, [0.0]), carrier)
    group = rest["joints"][child]["moving_links"]
    rec = {"type": j["type"], "limits": [j["lower"], j["upper"]], "reference": [l0, h0],
           "floor": max(MIN_SHARE * (h0 - l0), MIN_HINGE if j["type"] == "revolute" else rest["tolerance"]),
           "tolerance": rest["tolerance"], "rest": blocked(rest, child, [0.0]),
           "open": [k for k, v in rest["links"].items() if k not in group and not v["watertight"]],
           "tests": [], "achieved": None, "reductions": None}
    if rec["rest"] or rec["open"]:
        return rec
    for n in range(REDUCTIONS + 1):
        lo, hi = l0 * SHRINK ** n, h0 * SHRINK ** n
        if n and hi - lo < rec["floor"]:
            break
        qs = directions(lo, hi)
        in_time(deadline)
        rep = tools.penetration(meshes, {child: j}, tools.sweep_poses(joints, links, held, child, qs), carrier)
        rec["tests"].append({"n": n, "interval": [lo, hi], "blocked": blocked(rep, child, qs)})
        if rec["tests"][-1]["blocked"] is None:
            rec["achieved"], rec["reductions"] = [lo, hi], n
            break
    return rec


def reference(pinned, lower, upper):
    """The reference [l0, h0] of a hinge or slide with limits [lower, upper] (see the module docstring): each side
    the wider of the limit and the end of pinned, the entry of control/ranges.json (None without one), and 0 where
    the limit is 0."""
    if not pinned:
        return [lower, upper]
    return [min(pinned[0], lower) if lower else 0.0, max(pinned[1], upper) if upper else 0.0]


def executed(rec):
    """The interval that MuJoCo runs of a clearance record: its achieved interval within the joint's limits, None
    without one."""
    if not rec["achieved"]:
        return None
    (lo, hi), (a, b) = rec["limits"], rec["achieved"]
    return [max(lo, a), min(hi, b)]


def clearances(copy, doc, joints, links, ranges, deadline=math.inf):
    """{child: clearance record} of every hinge and slide of the copy, from the references of ranges; OutOfTime when
    the search runs past deadline."""
    meshes = tools.link_meshes(copy)
    if sorted(meshes) != sorted(links):
        raise ValueError(f"the mesh nodes {sorted(links)} load as the meshes {sorted(meshes)}")
    full, held = {c: J.full(j) for c, j in joints.items()}, J.holders(doc)
    return {c: clearance(meshes, full, links, held, c, reference(ranges.get(f"{c}|{j['type']}"), j["lower"],
                                                                 j["upper"]), deadline)
            for c, j in full.items() if j["type"] not in J.STATIC}


def run_sim(folder, doc, binary, joints, sha, intervals, deadline):
    """The report of harpor.physics.sim_check on the copy (see the module docstring), or {code, error} when it gave
    none. intervals: {link: [lower, upper]} to execute; deadline: when the check's time is up."""
    try:
        python = shlex.split(os.environ.get("GATOR_MUJOCO", ""))
    except ValueError as e:
        return {"code": "sim_unavailable", "error": f"GATOR_MUJOCO cannot be split like a shell: {e}"}
    if not python:
        return {"code": "sim_unavailable", "error": "GATOR_MUJOCO, the python of envs/mujoco, is not set"}
    model, out, file = folder / "model.glb", folder / "sim", folder / "intervals.json"
    align.write_glb(model, J.write_joints(doc, J.transform(joints, np.eye(4), "S")), binary)
    write_json(file, intervals)
    left = deadline - time.time()
    if left <= 0:
        return {"code": "sim_timeout", "error": "the other checks used all of the check's time"}
    cmd = [*python, "-m", "harpor.physics.sim_check", "--glb", str(model), "--out", str(out), "--asset-sha", sha,
           "--intervals", str(file)]
    with open(folder / "sim.log", "w") as log:
        try:
            code = tools._run(cmd, log, left, cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO),
                                                                 "PYTHONSAFEPATH": "1"})
        except subprocess.TimeoutExpired:
            return {"code": "sim_timeout", "error": f"MuJoCo ran past the {left:.0f} s left of the check"}
        except OSError as e:
            return {"code": "sim_unavailable", "error": f"GATOR_MUJOCO does not start: {e}"}
    rep = read_json(out / "report.json")
    if code or not isinstance(rep, dict):
        return {"code": "sim_unavailable", "error": f"GATOR_MUJOCO exited with {code} without a report"}
    if "error" in rep:
        return {**rep, "code": "sim_unavailable" if str(rep["error"]).startswith(MISSING) else "compile_error"}
    return rep


def problems(joints, links):
    """J.validate's problems of a joint set, and one for joints in a frame other than W."""
    out = J.validate(joints, links)
    if not out and any(j["frame"] != "W" for j in joints.values()):
        out.append("the joints must be in frame W, as set_joint stores them")
    return out


def _step(report, name, fn, *args):
    """fn(*args), or None after its failure went to the report's errors. OutOfTime goes on to run."""
    try:
        return fn(*args)
    except OutOfTime:
        raise
    except Exception as e:  # noqa: BLE001 - a host check on agent-made files must not stop the others
        report["errors"].append(f"{name}: {type(e).__name__}: {e}")
        return None


def run(folder, sha, original, ranges, deadline):
    """The steps of check on folder/candidate.glb (see the module docstring), before assess. carriers: {link: the
    link whose motion it has} (tools.carriers) of a valid joint set."""
    copy = folder / "candidate.glb"
    report = {"version": 1, "sha256": sha, "links": [], "joints": {}, "problems": [], "bad_names": None,
              "surface": surface.compare(copy, original), "carriers": {}, "clearance": {}, "sim": {}, "errors": []}
    try:
        doc, binary = align.read_glb(copy)
        joints, links, held = J.read_joints(doc), J.mesh_nodes(doc), J.holders(doc)
    except Exception as e:  # noqa: BLE001 - a GLB whose joints cannot be read is the agent's to fix
        report.update(problems=[f"the joints cannot be read: {type(e).__name__}: {e}"],
                      sim={"skipped": "the joints are not a valid tree"})
        return report
    report.update(links=links, joints=joints, problems=problems(joints, links))
    try:
        align.safe_links(links)
    except ValueError as e:
        report["bad_names"] = str(e)
    if report["problems"]:
        report["sim"] = {"skipped": "the joints are not a valid tree in frame W"}
        return report
    if joints:
        report["carriers"] = _step(report, "carriers", tools.carriers, joints, links, held) or {}
    try:
        report["clearance"] = _step(report, "clearance", clearances, copy, doc, joints, links, ranges, deadline) or {}
    except OutOfTime as e:
        report["sim"] = {"code": "sim_timeout", "error": str(e)}
        return report
    if report["bad_names"]:
        report["sim"] = {"skipped": "the link names are not file names"}
    elif all(j["type"] == "fixed" for j in joints.values()):
        report["sim"] = {"skipped": "no hinge, slide or soft joint"}
    else:
        intervals = {c: executed(rec) for c, rec in report["clearance"].items() if rec["achieved"]}
        report["sim"] = (_step(report, "simulation", run_sim, folder, doc, binary, joints, sha, intervals, deadline)
                         or {"skipped": "the simulation step failed"})
    return report


def failed(report):
    """Whether a step of a report failed (see the module docstring), so that it must not be cached."""
    return bool(report["errors"]) or "error" in report["surface"] or "error" in report["sim"]


def cached(path, sha, ranges):
    """The report at path when it is of sha and its references are those of ranges now, else None."""
    report = read_json(path)
    if not isinstance(report, dict) or report.get("sha256") != sha:
        return None
    clear = report["clearance"]
    now = {k: reference(ranges.get(f"{k}|{c['type']}"), *c["limits"]) for k, c in clear.items()}
    return report if now == {k: c["reference"] for k, c in clear.items()} else None


def pin(control, report, ranges):
    """Write the references of the report's hinges and slides to control/ranges.json where they differ from its
    entries: a new entry, or one that a repair widened or turned around (reference)."""
    new = {f"{k}|{c['type']}": c["reference"] for k, c in report["clearance"].items()}
    if any(ranges.get(k) != v for k, v in new.items()):
        write_json(Path(control) / "ranges.json", {**ranges, **new})


def check(glb, control, original, timeout=HOST_CHECK_TIMEOUT_S, pin_ranges=False):
    """The report of the GLB at glb (see the module docstring), with cached, whether it came from the cache.
    original: root/original.glb. timeout: seconds for the check, of which MuJoCo gets what the other steps left.
    pin_ranges: write this check's references to control/ranges.json (pin; the check at a round's end). Raises
    OSError when glb is not a regular file."""
    t0 = time.time()
    data = tools.read_file(glb)
    sha = hashlib.sha256(data).hexdigest()
    folder, ranges = Path(control) / "validate" / sha[:16], read_ranges(control)
    report = cached(folder / "report.json", sha, ranges)
    hit = report is not None
    if not hit:
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)
        (folder / "candidate.glb").write_bytes(data)
        report = run(folder, sha, original, ranges, t0 + timeout)
        report.update(assess(requirements(control), report), seconds=round(time.time() - t0, 2))
        if not failed(report):
            write_json(folder / "report.json", report)
    if pin_ranges:
        pin(control, report, ranges)
    return {**report, "cached": hit}


def clearance_finding(link, rec):
    """The finding of a clearance record (see the module docstring), None for a link that keeps its range."""
    unit = "rad" if rec["type"] == "revolute" else "m"

    def where(b):
        return (f"q = {b['q']:.4g} {unit}, where {b['inside_share']:.1%} of the samples of the moving links lie "
                f"inside {b['worst_other_link']}, {b['max_depth']:.3g} m deep at most")

    if rec["rest"]:
        return finding("rest_collision", link, rec["rest"]["inside_share"], 0.0,
                       f"at rest, {where(rec['rest'])}, beyond the tolerance of {rec['tolerance']:.3g} m")
    if rec["open"]:
        return finding("clearance_missing", link, rec["open"], [],
                       f"not closed, so inside and outside are unknown: {', '.join(rec['open'][:8])}")
    if rec["achieved"] is None:
        last = rec["tests"][-1]
        return finding("axis_blocked", link, last["interval"], rec["reference"],
                       f"no interval passed after {last['n']} reductions; the last is blocked at "
                       f"{where(last['blocked'])}")
    if rec["reductions"]:
        return finding("range_blocked", link, rec["achieved"], rec["reference"],
                       f"the reference interval is blocked at {where(rec['tests'][0]['blocked'])}; it passed after "
                       f"{rec['reductions']} reduction{'s' if rec['reductions'] > 1 else ''} by 2/3")
    if executed(rec) != rec["achieved"]:
        return finding("range_short", link, rec["limits"], rec["achieved"],
                       f"the joint's limits do not reach the ends of the interval that is clear; the reference "
                       f"{rec['reference']} stays across repairs, so a smaller range does not meet it")
    span = rec["achieved"][1] - rec["achieved"][0]
    if span < rec["floor"]:
        return finding("range_short", link, span, rec["floor"],
                       f"the span is under the retained span: {MIN_SHARE} of the reference and "
                       + (f"{math.degrees(MIN_HINGE):g} degrees for a hinge" if unit == "rad" else
                          f"the tolerance of {rec['tolerance']:.3g} m for a slide"))
    return None


def checks(report):
    """The findings of a report's steps, each without its requirement."""
    out = []
    if report["problems"]:
        out.append(finding("invalid_joints", measured=len(report["problems"]), expected=0,
                           why="; ".join(report["problems"][:3])))
    if report["bad_names"]:
        out.append(finding("bad_link_name", why=report["bad_names"]))
    if "error" in report["surface"]:
        out.append(finding("surface_error", why=report["surface"]["error"]))
    out += [finding(**f) for f in report["surface"].get("findings", [])]
    out += [f for f in (clearance_finding(k, rec) for k, rec in report["clearance"].items()) if f]
    if "error" in report["sim"]:
        out.append(finding(report["sim"].get("code", "compile_error"), why=report["sim"]["error"]))
    out += [finding(**f) for f in report["sim"].get("findings", [])]
    return out + [finding("validation_error", why=e) for e in report["errors"]]


def requirement_findings(r, report):
    """The findings that a found requirement adds (see the module docstring)."""
    link, rid, want = r.get("part"), r["id"], r.get("joint")
    if link not in report["links"]:
        return [finding("link_missing", link, None, link, "no mesh node has the name of the found part", rid)]
    j = report["joints"].get(link)
    if not isinstance(j, dict):
        return [finding("joint_missing", link, None, want, "the part has no joint; set it with set_joint", rid)]
    if j.get("type") not in (("revolute", "prismatic") if want == "unknown" else (want,)):
        return [finding("joint_type_mismatch", link, j.get("type"), want,
                        "the joint's type is not the requirement's, which cannot change", rid)]
    out, parent, carrier = [], r.get("parent"), report.get("carriers") or {}
    if parent in report["links"] and carrier.get(j.get("parent"), j.get("parent")) != carrier.get(parent, parent):
        out.append(finding("parent_mismatch", link, j.get("parent"), parent,
                           "the part hangs on a part that does not move with the requirement's parent; give it that "
                           "parent, or a part fixed to it", rid))
    clear, span = report["clearance"].get(link), (r.get("test") or {}).get("min_span")
    ran = executed(clear) if clear else None
    if ran and span and ran[1] - ran[0] + SPAN_TOL < span:
        out.append(finding("range_short", link, ran[1] - ran[0], span,
                           f"the achieved span is short of the requirement's by more than "
                           f"{math.degrees(SPAN_TOL):g} degrees; a smaller range does not meet it", rid))
    area = report["surface"].get("links", {}).get(link)
    if area and area["native"] > 0 and area["kept"] < KEPT * area["native"]:
        out.append(finding("surface_assignment", link, area["kept"] / area["native"], KEPT,
                           "share of its own native surface that the part's link keeps", rid))
    return out


def trajectory(r, report):
    """The states of a requirement's link in the MuJoCo report of this candidate ([] without one)."""
    sim = report["sim"]
    if sim.get("asset_sha256") != report["sha256"]:
        return []
    runs = sim.get("soft" if r.get("kind") == "soft" else "joints") or []
    return next((x.get("states") or [] for x in runs if x.get("link") == r.get("part")), [])


def measured(r, report):
    """What a record measured: the joint's type, its clearance, the press and the trajectory's states."""
    link = r.get("part")
    j = report["joints"].get(link)
    out = {"type": j.get("type") if isinstance(j, dict) else None}
    c = report["clearance"].get(link)
    if c:
        out.update(reference=c["reference"], achieved=c["achieved"], reductions=c["reductions"])
    press = next((x for x in report["sim"].get("soft") or [] if x.get("link") == link), {})
    out.update({k: press[k] for k in ("dmax_m", "visible_dmax_m", "residual_m", "cap_limited") if k in press})
    return {**out, "states": len(trajectory(r, report))}


def counts(f):
    """Whether a finding decides anything: surface findings only while surface.ENFORCE is on."""
    return surface.ENFORCE or not f["code"].startswith("surface_")


def outcome(findings, evidence):
    """A record's outcome (see the module docstring): fail, inconclusive or pass."""
    codes = [f["code"] for f in findings if counts(f)]
    if any(c not in INCONCLUSIVE for c in codes):
        return "fail"
    return "pass" if evidence and not codes else "inconclusive"


def scope(findings):
    """The scope of a set of findings (see the module docstring)."""
    codes = [f["code"] for f in findings if counts(f)]
    if "sim_unavailable" in codes:
        return NONE
    kinds = {SCOPE.get(c, LOCAL) for c in codes}
    return next((k for k in (STRUCTURAL, LOCAL) if k in kinds), NONE)


def record(r, measures=None):
    """A requirement's record, unassessed until assess gives it findings and an outcome."""
    found = r.get("status") == "found"
    return {"requirement": r["id"], "target": [r["part"]] if found else [], "kind": r.get("kind"),
            "joint": r.get("joint"), "text": r.get("text"), "required": bool(r.get("required", True)),
            "action": r.get("action"), "initial_state": (r.get("scope") or {}).get("start"),
            "measured": measures or {}, "findings": [], "outcome": "unassessed"}


def assess(requirements, report):
    """{records, findings, scope, outcome} of a report's steps under requirements (see the module docstring)."""
    owner = {r["part"]: r["id"] for r in requirements if r.get("status") == "found" and r.get("part")}
    findings = [{**f, "requirement": owner.get(f["link"])} for f in checks(report)]
    records, required = [], []
    for r in requirements:
        found, keep = r.get("status") == "found", bool(r.get("required", True))
        own = requirement_findings(r, report) if found else []
        rec = {**record(r, measured(r, report)), "findings": own + [f for f in findings if f["requirement"] == r["id"]]}
        if not found or (not keep and r["part"] not in report["links"]):
            rec.update(findings=[], outcome="unassessed")
        else:
            rec["outcome"] = outcome(rec["findings"], len(trajectory(r, report)) >= MIN_STATES)
        if found and keep:
            findings += own
            required.append(rec)
        records.append(rec)
    ok = (all(rec["outcome"] == "pass" for rec in required) and not report["surface"].get("blocks")
          and not any(SCOPE.get(f["code"], LOCAL) == STRUCTURAL for f in findings if counts(f)))
    return {"records": records, "findings": findings, "scope": scope(findings), "outcome": "pass" if ok else "fail"}


def _not_finished(pick, findings):
    """The why of not_finished: how the round handed off its pick, with the agent's reason or open defect, and what
    is open."""
    how = (f"the round finished as {pick['verdict']}" if pick["source"] == "finish" else
           f"the round ended without finish, so its newest {pick['source']} was taken")
    if pick["source"] != "export" and pick.get("reason"):  # an export's reason is the controller's
        how += f": {pick['reason']}"
    return how + ("; the other findings are open" if any(counts(f) for f in findings) else
                  "; the checks passed, so review it with review_articulation and finish articulated")


def round_report(control, report, pick, errors=()):
    """The report of a build or repair round (controller.check_round; paper A.3). report: check's report of the
    round's pick (select_deliverable's, in the copy of controller.store_candidate), or None when nothing was
    checked, and then every requirement of control/parts.json is unassessed. The round adds a validation_error for
    each of errors (a host step that raised), no_candidate when it left no valid GLB (pick None or
    fallback_original), and not_finished when its verdict is not articulated (needs_review or unchanged, or a
    checkpoint or export taken without finish). Its outcome is pass only when check's is and nothing was added, so
    the agent's verdict must be articulated too; its scope counts the added findings."""
    if report is None:
        report = {"version": 1, "sha256": None, "surface": {}, "errors": [], "records": [
            record(r) for r in requirements(control)], "findings": [], "outcome": "fail"}
    added = [finding("validation_error", why=e) for e in errors]
    if pick is None or pick["source"] == "fallback_original":
        added.append(finding("no_candidate", why="the round left no valid GLB: no finish, checkpoint or export that "
                                                 "passes the controller's checks"))
    elif pick["verdict"] != "articulated":
        added.append(finding("not_finished", measured=pick["verdict"], expected="articulated",
                             why=_not_finished(pick, report["findings"] + added)))
    findings = report["findings"] + added
    ok = report["outcome"] == "pass" and not added
    return {**report, "findings": findings, "scope": scope(findings), "outcome": "pass" if ok else "fail"}


def label(f):
    """A finding's scope (SCOPE), or "report only" for a surface finding while surface.ENFORCE is off."""
    return SCOPE.get(f["code"], LOCAL) if counts(f) else "report only"


def ranked(findings):
    """findings in the order of LABELS, structural first, each label in its own order."""
    return sorted(findings, key=lambda f: LABELS.index(label(f)))


def _json(x):
    """x as JSON: ASCII, floats to 4 significant digits, every string cut to MAX_TEXT characters first."""
    def short(v):
        if isinstance(v, float):
            return float(f"{v:.4g}")
        if isinstance(v, str):
            return v[:MAX_TEXT]
        if isinstance(v, (list, tuple)):
            return [short(y) for y in v]
        return {str(k)[:MAX_TEXT]: short(y) for k, y in v.items()} if isinstance(v, dict) else v

    return json.dumps(short(x), sort_keys=True)


def _cap(lines, limit=MAX_FEEDBACK):
    """lines as text of at most limit characters: the lines that fit, then how many more the report has."""
    out, size = [], 0
    for k, line in enumerate(lines):
        if size + len(line) + 1 > limit - 80:
            out.append(f"... and {len(lines) - k} more lines, all in the report")
            break
        out.append(line)
        size += len(line) + 1
    return "\n".join(out) + "\n"


def feedback(report):
    """The report as text for the agent's next prompt: its outcome and scope; every finding, structural ones first,
    with its requirement, link, measured and expected values; every requirement's outcome and measures; and the
    boxes of the surface that is gone. At most MAX_FEEDBACK characters, and the same text for the same report.
    Requirement text, which can come from the photo, and the names and reasons of the agent appear only as JSON
    (_json), which json.dumps keeps ASCII."""
    found = ranked(report["findings"])
    head = f"candidate sha256 {report['sha256'][:12]}" if report["sha256"] else "no candidate checked"
    lines = [f"{head}: outcome {report['outcome']}, scope {report['scope']}",
             f"findings ({len(found)}), structural first:" if found else "findings: none"]
    lines += [f"- {label(f)} {f['code']}, requirement {_json(f['requirement'])}, link {_json(f['link'])}: measured "
              f"{_json(f['measured'])}, expected {_json(f['expected'])}; {_json(f['why'])}" for f in found]
    lines.append("requirements:" if report["records"] else "requirements: none")
    lines += [f"- {_json(r['requirement'])} {_json(r['text'])}, link {_json(r['target'][0] if r['target'] else None)}"
              f", {_json(r['kind'])} {_json(r['joint'])}{'' if r['required'] else ', not required'}: {r['outcome']}; "
              f"measured {_json(r['measured'])}" for r in report["records"]]
    regions = report["surface"].get("lost_regions") or []
    if regions:
        lines.append("surface that is gone, boxes in W (compare with original/original.glb):")
        lines += [f"- min {_json(x['min'])}, max {_json(x['max'])}, share {_json(x['share'])}" for x in regions]
    return _cap(lines)
