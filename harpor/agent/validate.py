"""The host's check of a refinement candidate in a session with rounds (build_plan.md note 5; paper A.3 and A.6).

check(glb, control, original) runs on the host in envs/sam3d after a build or repair round (harpor.agent.controller).
It reads the GLB once without following a symlink, hashes it and copies it to control/validate/<sha16>/candidate.glb,
and compares that copy with original, root/original.glb (harpor.agent.surface.compare). A failure of the comparison
is surface_error. The report is cached at control/validate/<sha16>/report.json unless a step failed.

The physics review of the same copy (harpor.agent.physics_review) adds its own findings to the report: the
controller applies them after check. The outcome is pass when no finding counts. Surface findings count only while
surface.ENFORCE is on; until then they are reported and decide nothing. scope (SCOPE, paper A.3) is structural with a
structural finding, else local with a local one, else none: none means no agent action can help.

round_report(control, report, pick, errors) is the report of a round of the session: it adds what the round's pick
says, no_candidate or not_finished, so the round's outcome is pass only when the agent's verdict is complete or
unchanged too. feedback(report) is the report as text for the agent's next prompt.
"""

import hashlib
import json
import shutil
import time
from pathlib import Path

from harpor.agent import surface, tools
from harpor.fileio import read_json, write_json

MAX_FEEDBACK, MAX_TEXT = 6000, 300  # characters of feedback, and of one text in it before quoting
LOCAL, STRUCTURAL, NONE = "local", "structural", "none"
SCOPE = {  # paper A.3: local repairs, structural replanning, or nothing an agent can do
    "no_candidate": LOCAL, "not_finished": LOCAL, "surface_uv": LOCAL, "surface_material": LOCAL,
    "surface_deleted": LOCAL, "surface_flipped": LOCAL, "surface_error": LOCAL, "validation_error": LOCAL,
    "physics_implausible": LOCAL, "surface_identity": STRUCTURAL, "physics_unassessed": NONE}
LABELS = (STRUCTURAL, LOCAL, NONE, "report only")  # the order of findings in feedback
FINISHED = ("complete", "unchanged")  # the verdicts of a round that handed off


def finding(code, measured=None, expected=None, why=""):
    return {"code": code, "measured": measured, "expected": expected, "why": why}


def checks(report):
    """The findings of a report's steps."""
    out = []
    if "error" in report["surface"]:
        out.append(finding("surface_error", why=report["surface"]["error"]))
    out += [finding(**{k: f.get(k) for k in ("code", "measured", "expected", "why")})
            for f in report["surface"].get("findings", [])]
    return out + [finding("validation_error", why=e) for e in report["errors"]]


def counts(f):
    """Whether a finding decides anything: surface findings only while surface.ENFORCE is on."""
    return surface.ENFORCE or not f["code"].startswith("surface_")


def scope(findings):
    """The scope of a set of findings (see the module docstring)."""
    kinds = {SCOPE.get(f["code"], LOCAL) for f in findings if counts(f)}
    return next((k for k in (STRUCTURAL, LOCAL) if k in kinds), NONE)


def outcome(findings):
    """pass when no finding counts (counts), else fail."""
    return "fail" if any(counts(f) for f in findings) else "pass"


def check(glb, control, original):
    """The report of the GLB at glb (see the module docstring), with cached, whether it came from the cache:
    {version, sha256, surface, errors, findings, scope, outcome, seconds, cached}. original: root/original.glb.
    Raises OSError when glb is not a regular file."""
    t0 = time.time()
    data = tools.read_file(glb)
    sha = hashlib.sha256(data).hexdigest()
    folder = Path(control) / "validate" / sha[:16]
    report = read_json(folder / "report.json")
    hit = isinstance(report, dict) and report.get("sha256") == sha
    if not hit:
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)
        (folder / "candidate.glb").write_bytes(data)
        report = {"version": 2, "sha256": sha, "surface": surface.compare(folder / "candidate.glb", original),
                  "errors": []}
        found = checks(report)
        report.update(findings=found, scope=scope(found), outcome=outcome(found), seconds=round(time.time() - t0, 2))
        if "error" not in report["surface"]:
            write_json(folder / "report.json", report)
    return {**report, "cached": hit}


def _not_finished(pick, findings):
    """The why of not_finished: how the round handed off its pick, with the agent's reason or open defect, and what
    is open."""
    how = (f"the round finished as {pick['verdict']}" if pick["source"] == "finish" else
           f"the round ended without finish, so its newest {pick['source']} was taken")
    if pick["source"] != "export" and pick.get("reason"):  # an export's reason is the controller's
        how += f": {pick['reason']}"
    return how + ("; the other findings are open" if any(counts(f) for f in findings) else
                  "; the checks passed, so review it and finish complete")


def round_report(control, report, pick, errors=()):
    """The report of a build or repair round (controller.check_round; paper A.3). report: check's report of the
    round's pick (select_deliverable's, in the copy of controller.store_candidate), with the physics review applied,
    or None when nothing was checked. The round adds a validation_error for each of errors (a host step that raised),
    no_candidate when it left no valid GLB (pick None or fallback_original), and not_finished when its verdict is not
    complete or unchanged (needs_review, or a checkpoint or export taken without finish). Its outcome is pass only
    when check's is and nothing was added; its scope counts the added findings."""
    if report is None:
        report = {"version": 2, "sha256": None, "surface": {}, "errors": [], "findings": [], "outcome": "fail"}
    added = [finding("validation_error", why=e) for e in errors]
    if pick is None or pick["source"] == "fallback_original":
        added.append(finding("no_candidate", why="the round left no valid GLB: no finish, checkpoint or export that "
                                                 "passes the controller's checks"))
    elif pick["verdict"] not in FINISHED:
        added.append(finding("not_finished", measured=pick["verdict"], expected="complete",
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
    with its measured and expected values; and the boxes of the surface that is gone. At most MAX_FEEDBACK
    characters, and the same text for the same report. The VLM's and the agent's text appear only as JSON (_json),
    which json.dumps keeps ASCII."""
    found = ranked(report["findings"])
    head = f"candidate sha256 {report['sha256'][:12]}" if report["sha256"] else "no candidate checked"
    lines = [f"{head}: outcome {report['outcome']}, scope {report['scope']}",
             f"findings ({len(found)}), structural first:" if found else "findings: none"]
    lines += [f"- {label(f)} {f['code']}: measured {_json(f['measured'])}, expected {_json(f['expected'])}; "
              f"{_json(f['why'])}" for f in found]
    regions = report["surface"].get("lost_regions") or []
    if regions:
        lines.append("surface that is gone, boxes in W (compare with original/original.glb):")
        lines += [f"- min {_json(x['min'])}, max {_json(x['max'])}, share {_json(x['share'])}" for x in regions]
    return _cap(lines)
