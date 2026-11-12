"""The agent prompt, with the path edits in CHANGES applied. README.md gives its source, license and reasons."""

from pathlib import Path

HERE = Path(__file__).resolve().parent
FILE = "refinement_prompt.md"
WORK = ("The Codex CLI runs in the work folder itself, mounted at /work, so the paper's work/ prefix "
        "would point to /work/work.")
CHANGES = [
    ("Read work/run_budget.json at launch.", "Read /work/run_budget.json at launch.", WORK),
]
# The sentence of the repair right after a planning round, and only there: that plan came from the same check.
PLAN = " The planning round before this one wrote /work/plan.md from the same check: follow it."


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


def round_prompt(kind, budget_s, feedback, best, evidence, planned=False):
    """The prompt of a round after the first, kind 'repair' or 'replan' (controller.run_rounds). Each round is a new
    agent with no memory of the others, so the prompt carries the whole task: a repair is the refinement prompt and
    the host's feedback; a replan writes a plan and stops.

    feedback: harpor.agent.validate.feedback of the last build or repair; best: the best candidate so far in work/
    (_best); evidence: the work/ path of the last check's report; planned: the round before this repair was a
    planning round, whose plan (PLAN) came from the same check. A later repair has the feedback of a newer check, so
    it is not sent to that plan.
    """
    if kind == "replan":
        return (f"This is a planning round with {budget_s:.0f} seconds. Do not edit the mesh or call helpers. "
                f"Inspect {evidence} and write a concrete physics repair plan to /work/plan.md. "
                "Preserve the object's photographed shape and scale.\n" + feedback)
    return (load() + f"\nThis is a physics repair round with a {budget_s:.0f} second budget. The deadline in "
            "run_budget.json takes precedence. Continue from candidate.blend. Inspect the simulation evidence "
            f"in {evidence}, fix the reported physical defects, then review and finish complete (or unchanged "
            "if no repair is supported)." + (PLAN if planned else "") + "\n" + _best(best) + "\nHost feedback:\n" + feedback)
