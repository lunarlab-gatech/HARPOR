"""The agent prompts, with the path edits in CHANGES applied. README.md gives their source, license and reasons."""

from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = {"refinement": "refinement_prompt.md", "articulation": "articulation_prompt.md"}
WORK = ("The Codex CLI runs in the work folder itself, mounted at /work, so the paper's work/ prefix "
        "would point to /work/work.")
CHANGES = {
    "refinement": [
        ("Read work/run_budget.json at launch.", "Read /work/run_budget.json at launch.", WORK),
    ],
    "articulation": [],
}
ROUNDS = {"repair": "repair_prompt.md", "replan": "replan_prompt.md"}  # the later rounds of controller.run_rounds
BUDGET = "The budget is 900 seconds"  # the articulation prompt's budget sentence, which a repair round sets
# The end of the repair template's first paragraph in the round right after a planning round, and only there.
PLAN = " The planning round before this one wrote /work/plan.md from the same check: follow it."


def original(mode):
    return (HERE / FILES[mode]).read_text(encoding="utf-8")


def load(mode):
    """The prompt for mode 'refinement' or 'articulation', with CHANGES applied."""
    text = original(mode)
    for old, new, _ in CHANGES[mode]:
        if text.count(old) != 1:
            raise ValueError(f"{FILES[mode]}: {old!r} occurs {text.count(old)} times, expected once")
        text = text.replace(old, new)
    return text


def _best(best):
    """The sentence on the best candidate so far: best is {glb, round, passing, required}, or None."""
    if not best:
        return "No round has handed off a valid candidate yet."
    return (f"The best candidate so far is /work/{best['glb']}, from round {best['round']}: {best['passing']} of "
            f"{best['required']} required requirements pass. If a change makes the result worse, go back to it: "
            "start an empty scene, import that GLB, pack the images and save the scene as /work/candidate.blend.")


def round_prompt(kind, budget_s, feedback, best, evidence, planned=False):
    """The prompt of a round after the first, kind 'repair' or 'replan' (controller.run_rounds). Each round is a new
    agent with no memory of the others, so the prompt carries the whole task: a repair is the articulation prompt
    with its budget sentence (BUDGET) set to budget_s, then repair_prompt.md; a replan is replan_prompt.md alone.

    feedback: harpor.agent.validate.feedback of the last build or repair; best: the best candidate so far in work/
    (_best); evidence: the work/ path of the last check's report; planned: the round before this repair was a
    planning round, whose plan (PLAN) was made from the same check.
    """
    text = (HERE / ROUNDS[kind]).read_text(encoding="utf-8").format(
        budget_s=budget_s, feedback=feedback.rstrip("\n"), best=_best(best), evidence=evidence,
        plan=PLAN if planned else "")
    if kind == "replan":
        return text
    base = load("articulation")
    if base.count(BUDGET) != 1:
        raise ValueError(f"{FILES['articulation']}: {BUDGET!r} occurs {base.count(BUDGET)} times, expected once")
    return base.replace(BUDGET, f"The budget is {budget_s:.0f} seconds") + "\n" + text
