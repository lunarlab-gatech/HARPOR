"""The agent prompt, with the path edits in CHANGES applied. README.md gives its source, license and reasons."""

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


def round_prompt(budget_s, feedback, best, evidence):
    """The prompt of a repair round, a round after the first (controller.run_rounds). Each round is a new agent with
    no memory of the others, so the prompt carries the whole task: the refinement prompt and the host's feedback.

    feedback: harpor.agent.validate.feedback of the last build or repair; best: the best candidate so far in work/
    (_best); evidence: the work/ path of the last check's report.
    """
    return (load() + f"\nThis is a repair round with a {budget_s:.0f} second budget. The deadline in "
            "run_budget.json takes precedence. Continue from candidate.blend. Inspect the host's check "
            f"in {evidence}, fix the reported defects, then review and finish complete (or unchanged "
            "if no repair is supported).\n" + _best(best) + "\nHost feedback:\n" + feedback)
