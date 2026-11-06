"""The evidence of a session in rounds, root/control/evidence.json (paper A.1 and A.3; build_plan.md note 11).

The controller writes it after every round (controller.run_rounds), so a session that stops keeps what its rounds
found, and once more with the choice of its deliverable (controller.select_candidate). It holds no trajectory. Its
parts follow the paper's representation:

  requirements  S: the requirements of control/parts.json
  candidates    one entry per stored candidate (controller.store_candidate), oldest first, with the agent's verdict,
                reason and limitations, MuJoCo's runtime configuration, and
                  mechanisms  K: its joints, m1, m2, ... in the GLB's order, with origin and axis in W, limits, the
                              reference of the clearance search (control/ranges.json) and a soft part's material
                  entities    X: its links and the flexes that MuJoCo pressed
                  gamma       the relations: each found requirement's mechanism and link, with the native area
                              that the link keeps (harpor.agent.surface)
                  surface     rho: harpor.agent.surface's report
                  records     E: each requirement's record (harpor.agent.validate.record), with its outcome
                              unassessed, pass, fail or inconclusive
                and the findings, errors, scope and outcome of its round
  rounds        result.json's rounds
  selected      controller.select_candidate's choice, null until the session ends

Records belong to one candidate: each candidate's come from the check of its own bytes, so a changed build starts
at unassessed (paper A.3), and a candidate whose check never ended keeps every record unassessed.
"""

from pathlib import Path

import numpy as np

from harpor.agent import validate
from harpor.fileio import read_json, write_json
from harpor.parts import joints as J

CANDIDATE_KEYS = ("version", "round", "kind", "glb", "from", "sha256", "source", "agent_verdict", "reason",
                  "limitations")
SOFT_KEYS = ("material", "young", "poisson", "density", "support")  # a soft joint's parameter assumptions


def diagonal(original):
    """D of root/original.glb, the diagonal of its box after the node transforms; None when it cannot be read."""
    from harpor.agent.controller import box_extents  # the controller imports this module

    try:
        return float(np.linalg.norm(box_extents(original)))
    except Exception:  # noqa: BLE001 - the evidence is written without it
        return None


def mechanisms(report):
    """K of a candidate's report (see the module docstring)."""
    joints = report.get("joints") if isinstance(report.get("joints"), dict) else {}
    clearance = report.get("clearance") or {}
    rows = [(child, j) for child, j in joints.items() if isinstance(j, dict)]
    return [{"id": f"m{k}", "link": child, "parent": j.get("parent"), "type": j.get("type"),
             "origin_w": j.get("origin"), "axis_w": j.get("axis"),
             "range": None if j.get("type") in J.STATIC else [j.get("lower"), j.get("upper")],
             "reference_range": (clearance.get(child) or {}).get("reference"),
             "assumptions": {key: j.get(key) for key in SOFT_KEYS} if j.get("type") == "soft" else {}}
            for k, (child, j) in enumerate(rows, 1)]


def entities(report):
    """X of a candidate's report: its links (mesh nodes) and the flexes that MuJoCo pressed."""
    soft = (report.get("sim") or {}).get("soft") or []
    return {"links": list(report.get("links") or []),
            "flex": [{k: s.get(k) for k in ("link", "flex", "press_nodes")} for s in soft]}


def gamma(records, mechs, surface):
    """The relations of each found requirement: the mechanism of its link, the link and the native area it keeps."""
    by_link, kept = {m["link"]: m["id"] for m in mechs}, surface.get("links") or {}
    return [{"requirement": r["requirement"], "mechanism": by_link.get(r["target"][0]), "link": r["target"][0],
             "faces_area_m2": (kept.get(r["target"][0]) or {}).get("kept")} for r in records if r["target"]]


def candidate(cand, requirements):
    """A stored candidate's entry (see the module docstring). cand: its record in control/candidates."""
    report = cand.get("report") or {}
    records = report.get("records") or [validate.record(r) for r in requirements]
    mechs, surface = mechanisms(report), report.get("surface") or {}
    return {**{k: cand.get(k) for k in CANDIDATE_KEYS}, "runtime": (report.get("sim") or {}).get("runtime") or {},
            "mechanisms": mechs, "entities": entities(report), "gamma": gamma(records, mechs, surface),
            "surface": surface, "records": records, "physics_review": report.get("physics_review"),
            "findings": report.get("findings") or [],
            "errors": report.get("errors") or [], "scope": report.get("scope"), "outcome": report.get("outcome")}


def build(root, rounds, candidates, selected=None):
    """The evidence (see the module docstring). rounds: result.json's; candidates: the records of control/candidates;
    selected: controller.select_candidate's choice, or None while the rounds run."""
    root = Path(root)
    control = root / "control"
    parts, requirements = read_json(control / "parts.json"), validate.requirements(control)
    return {"version": 1, "object": parts.get("noun") if isinstance(parts, dict) else None,
            "diagonal_m": diagonal(root / "original.glb"), "requirements": requirements,
            "candidates": [candidate(c, requirements) for c in candidates], "rounds": rounds, "selected": selected}


def write(root, rounds, candidates, selected=None):
    """Write root/control/evidence.json (build)."""
    write_json(Path(root) / "control" / "evidence.json", build(root, rounds, candidates, selected))
