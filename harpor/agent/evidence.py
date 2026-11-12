"""The evidence of a session in rounds, root/control/evidence.json (paper A.1 and A.3; build_plan.md note 5).

The controller writes it after every round (controller.run_rounds), so a session that stops keeps what its rounds
found, and once more with the choice of its deliverable (controller.select_candidate). It holds:

  object        the manifest's source line, which names the object and the capture
  diagonal_m    the diagonal of root/original.glb's box
  candidates    one entry per stored candidate (controller.store_candidate), oldest first, with the agent's verdict,
                reason and limitations, the surface report against the original (harpor.agent.surface), the physics
                review (harpor.agent.physics_review), and the findings, errors, scope and outcome of its round
  rounds        result.json's rounds
  selected      controller.select_candidate's choice, null until the session ends
"""

from pathlib import Path

import numpy as np

from harpor.fileio import read_json, write_json

CANDIDATE_KEYS = ("version", "round", "kind", "glb", "from", "sha256", "source", "agent_verdict", "reason",
                  "limitations")


def diagonal(original):
    """D of root/original.glb, the diagonal of its box after the node transforms; None when it cannot be read."""
    from harpor.agent.controller import box_extents  # the controller imports this module

    try:
        return float(np.linalg.norm(box_extents(original)))
    except Exception:  # noqa: BLE001 - the evidence is written without it
        return None


def candidate(cand):
    """A stored candidate's entry (see the module docstring). cand: its record in control/candidates."""
    report = cand.get("report") or {}
    return {**{k: cand.get(k) for k in CANDIDATE_KEYS}, "surface": report.get("surface") or {},
            "physics_review": report.get("physics_review"), "findings": report.get("findings") or [],
            "errors": report.get("errors") or [], "scope": report.get("scope"), "outcome": report.get("outcome")}


def build(root, rounds, candidates, selected=None):
    """The evidence (see the module docstring). rounds: result.json's; candidates: the records of control/candidates;
    selected: controller.select_candidate's choice, or None while the rounds run."""
    root = Path(root)
    manifest = read_json(root / "control" / "manifest.json")
    return {"version": 2, "object": manifest.get("source") if isinstance(manifest, dict) else None,
            "diagonal_m": diagonal(root / "original.glb"),
            "candidates": [candidate(c) for c in candidates], "rounds": rounds, "selected": selected}


def write(root, rounds, candidates, selected=None):
    """Write root/control/evidence.json (build)."""
    write_json(Path(root) / "control" / "evidence.json", build(root, rounds, candidates, selected))
