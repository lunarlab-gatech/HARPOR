"""The evidence of a session in rounds, root/control/evidence.json (paper A.1 and A.3; README.md note 5).

The controller writes it after every round (controller.run_rounds), so a session that stops keeps what its rounds
found, and once more with the choice of its deliverable (controller.select_candidate). It holds:

  object        the manifest's source line, which names the object and the capture
  diagonal_m    the diagonal of root/original.glb's box
  candidates    one entry per stored candidate (controller.store_candidate), oldest first, with the agent's verdict,
                reason and limitations, the surface report against the original (harpor.agent.surface), and the
                findings, errors, scope and outcome of its round
  rounds        result.json's rounds
  selected      controller.select_candidate's choice, null until the session ends
  lidar         the session's LiDAR geometry (harpor.agent.lidar_geometry): requested and effective mode, status and
                reason, evidence and settings key, the settings used, the split, the baseline's regions and, once the
                deliverable is chosen, the winner's hash with its before and after values, coverage and reason; null
                for a session without it

Each candidate also holds lidar, its LiDAR check (status, regions with their gains and coverage, resolved targets).
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


LIDAR_KEYS = ("status", "mode", "key", "regions", "protected", "silhouettes", "targets", "resolved", "gain_m",
              "regressed", "error", "reason")


def candidate(cand):
    """A stored candidate's entry (see the module docstring). cand: its record in control/candidates."""
    report = cand.get("report") or {}
    lid = report.get("lidar")
    return {**{k: cand.get(k) for k in CANDIDATE_KEYS}, "surface": report.get("surface") or {},
            "findings": report.get("findings") or [],
            "errors": report.get("errors") or [], "scope": report.get("scope"), "outcome": report.get("outcome"),
            "lidar": None if lid is None else {k: lid.get(k) for k in LIDAR_KEYS if k in lid}}


def lidar(root, candidates, selected):
    """The session's LiDAR entry (see the module docstring), or None."""
    from harpor.agent import lidar_geometry

    control = Path(root) / "control"
    m = lidar_geometry.manifest(control)
    if m is None:
        return None
    base = read_json(control / "lidar" / "baseline.json") or {}
    out = {"mode": m.get("mode"), "status": m.get("status"), "reason": m.get("reason"), "key": m.get("key"),
           "settings": m.get("settings"), "split": m.get("split"), "groups": m.get("groups"), "views": m.get("views"),
           "baseline": {k: base.get(k) for k in ("regions", "unobserved", "alignment", "views") if k in base},
           "winner": None}
    version = (selected or {}).get("version")
    win = next((c for c in candidates if c.get("version") == version), None) if version else None
    if win is not None:
        lid = (win.get("report") or {}).get("lidar") or {}
        out["winner"] = {"version": version, "sha256": win.get("sha256"), "evidence_key": lid.get("key"),
                         "status": lid.get("status"), "regions": lid.get("regions"), "resolved": lid.get("resolved"),
                         "gain_m": lid.get("gain_m"), "why": (selected or {}).get("why")}
    return out


def build(root, rounds, candidates, selected=None):
    """The evidence (see the module docstring). rounds: result.json's; candidates: the records of control/candidates;
    selected: controller.select_candidate's choice, or None while the rounds run."""
    root = Path(root)
    manifest = read_json(root / "control" / "manifest.json")
    return {"version": 3, "object": manifest.get("source") if isinstance(manifest, dict) else None,
            "diagonal_m": diagonal(root / "original.glb"),
            "candidates": [candidate(c) for c in candidates], "rounds": rounds, "selected": selected,
            "lidar": lidar(root, candidates, selected)}


def write(root, rounds, candidates, selected=None):
    """Write root/control/evidence.json (build)."""
    write_json(Path(root) / "control" / "evidence.json", build(root, rounds, candidates, selected))
