"""The agent prompt, with the path edits in CHANGES applied. README.md note 5 ("The prompt's source and edits") gives
its source, license and reasons."""

from pathlib import Path

HERE = Path(__file__).resolve().parent
FILE = "refinement_prompt.md"
WORK = ("The Codex CLI runs in the work folder itself, mounted at /work, so the paper's work/ prefix "
        "would point to /work/work.")
BUDGET = ("HARPOR gives a build round 900 s (controller.BUDGET_S), not the paper's 600 s. The deadline in "
          "run_budget.json stays the authority, also for a repair with its own budget.")
CHANGES = [
    ("Read work/run_budget.json at launch.", "Read /work/run_budget.json at launch.", WORK),
    ("The full ten-minute (600-second) work", "The full fifteen-minute (900-second) work", BUDGET),
]
# Added to the first prompt of a session with rounds and to every round prompt (controller.prepare_session, run_rounds).
GUIDANCE = """
This session runs in rounds. After each build or repair the host checks the candidate against the
original and writes the check to evidence/round_NN.json. A failed check feeds a repair round. Repair
the reported issues while preserving the photographed object, and do not change the object's scale.
"""


LIDAR_INSPECT = """
LiDAR geometry (inspection only). This job has measured LiDAR returns, in meters, in frame W. The host scored the
original against them: original/previews/lidar/summary.json lists the surfaces, the regions where the mesh and the
returns disagree, and the surfaces with no returns, and original/previews/lidar/lidar_viewNN.png draws the returns on
the photos (red: the mesh sticks out toward the sensor, blue: it lies behind the measured surface, a cross: a miss).
Every review adds the same for the candidate (lidar.json and lidar_viewNN.png in its folder). Use them as evidence when
you judge the geometry. Their tolerances are not calibrated for this sensor, so the host reports LiDAR findings
without acting on them, and the LiDAR sculpt helper is off. A repair of a surface without returns is inferred from the
photos; say so in the hand-off reason.
"""
LIDAR_SCULPT = """
LiDAR geometry (sculpt). This job has measured LiDAR returns, in meters, in frame W. The host scored the original
against them: original/previews/lidar/summary.json lists the repair regions (id L1, L2, ...; status target, conflict or
observation; seed, normal and bounds in W; the median signed error in meters and a suggested operation), the surfaces
with no returns and any alignment notice. original/previews/lidar/lidar_viewNN.png draws the fitting returns on the
photos (red: the mesh sticks out toward the sensor, blue: it lies behind the measured surface, a cross: a miss).
input/lidar/ has the fitting returns themselves. The host keeps other views back and checks every hand-off on them.

Workflow for a target region:
1. Look at the overlays and the photos, and decide whether the region needs a flat or a curved repair.
2. Save the scene and run review; its folder gets lidar.json with each region's current fit and the correspondences
   the helper needs.
3. Run `lidar_sculpt --receipt <receipt> --patch <id> --op plane` for a flat surface, or `--op push` for a curved
   one (optional `--strength 0.5` for a half step). It edits /work/candidate.blend in place, keeps the rim, the seams,
   the UVs and the materials, and keeps the reviewed scene in sculpt/ for a rollback.
4. Review again. Keep the result when the region's error went down and nothing else got worse; checkpoint it. If it got
   worse, copy the scene from sculpt/<receipt>-before.blend back to /work/candidate.blend.
5. Finish. Name in the reason which repairs the LiDAR measured and which you inferred from the photos alone.

Example: `review` prints receipt r002-0123abcd4567 and "lidar L1 target (host-verifiable): ... median signed -9.8 mm;
suggested plane". Then `lidar_sculpt --receipt r002-0123abcd4567 --patch L1 --op plane`, `review` (now about 0 mm),
`checkpoint --receipt r003-... --defect "none open"`, and `finish --receipt r003-... --verdict complete --reason
"L1 flattened to the LiDAR returns (measured); texture of the lid cleaned (inferred)"`.

Do not sculpt a region of status conflict or observation, or for an alignment notice: those are registration or
coverage findings, not shape errors. A surface without returns can still get a photo-guided repair; it is inferred.
When no region is a target, do not run lidar_sculpt.
"""


def lidar_text(mode):
    """The prompt's LiDAR part for the session's effective mode (lidar_geometry.freeze): none without evidence."""
    return {"inspect": LIDAR_INSPECT, "sculpt": LIDAR_SCULPT}.get(mode, "")


def original():
    return (HERE / FILE).read_text(encoding="utf-8")


def load():
    """The refinement prompt, with CHANGES applied."""
    text = original()
    for old, new, _ in CHANGES:
        if text.count(old) != 1:
            raise ValueError(f"{FILE}: {old!r} occurs {text.count(old)} times, expected once")
        text = text.replace(old, new)
    return text


def _best(best):
    """The sentence on the best candidate so far: best is {glb, round, passed}, or None."""
    if not best:
        return "No round has handed off a valid candidate yet."
    return (f"The best candidate so far is /work/{best['glb']}, from round {best['round']}; its check "
            f"{'passed' if best['passed'] else 'did not pass'}. If a change makes the result worse, go back to it: "
            "start an empty scene, import that GLB, pack the images and save the scene as /work/candidate.blend.")


def round_prompt(budget_s, feedback, best, evidence, lidar=None):
    """The prompt of a repair round, a round after the first (controller.run_rounds). Each round is a new agent with
    no memory of the others, so the prompt carries the whole task: the refinement prompt and the host's feedback.

    feedback: harpor.agent.validate.feedback of the last build or repair; best: the best candidate so far in work/
    (_best); evidence: the work/ path of the last check's report; lidar: the session's effective LiDAR mode (lidar_text).
    """
    return (load() + lidar_text(lidar) + f"\nThis is a repair round with a {budget_s:.0f} second budget. The deadline in "
            "run_budget.json takes precedence. Continue from candidate.blend. Inspect the host's check "
            f"in {evidence}, fix the reported defects, then review and finish complete (or unchanged "
            "if no repair is supported).\n" + _best(best) + "\nHost feedback:\n" + feedback)
