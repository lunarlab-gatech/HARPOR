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
