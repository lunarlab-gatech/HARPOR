"""The paper's two agent prompts, with our changes recorded (build_plan.md note 5), and our own prompt
for the articulation mode (note 11).

refinement_prompt.md and scratch_prompt.md are the files embedded in the arXiv HTML of "GATOR:
Generative and Agentic 3D Object Reconstruction From Casual Images" (Qirui Wu, Stan Birchfield, Hesam
Rabeti, Angel X. Chang and Bowen Wen, arXiv 2610.11215v1, 2026), byte for byte. The paper is licensed
CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). load() applies CHANGES, so every
difference between what our agent reads and the paper's text is listed here with its reason.
articulation_prompt.md is ours, not the paper's, so it has no changes to record.
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = {"refinement": "refinement_prompt.md", "from_scratch": "scratch_prompt.md",
         "articulation": "articulation_prompt.md"}
WORK = ("The Codex CLI runs in the work folder itself, mounted at /work, so the paper's work/ prefix "
        "would point to /work/work.")
CHANGES = {
    "refinement": [
        ("Read work/run_budget.json at launch.", "Read /work/run_budget.json at launch.", WORK),
    ],
    "from_scratch": [
        ("Read work/run_budget.json first", "Read /work/run_budget.json first", WORK),
        ("Save a complete packed work/candidate.blend early.", "Save a complete packed /work/candidate.blend early.", WORK),
    ],
    "articulation": [],
}


def original(mode):
    return (HERE / FILES[mode]).read_text(encoding="utf-8")


def load(mode):
    """The prompt for mode 'refinement', 'from_scratch' or 'articulation', with CHANGES applied."""
    text = original(mode)
    for old, new, _ in CHANGES[mode]:
        if text.count(old) != 1:
            raise ValueError(f"{FILES[mode]}: {old!r} occurs {text.count(old)} times, expected once")
        text = text.replace(old, new)
    return text
