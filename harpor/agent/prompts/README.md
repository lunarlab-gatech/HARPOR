# Agent prompts

`refinement_prompt.md` is Prompt 1, the refinement prompt file embedded in the arXiv HTML of GATOR
(https://arxiv.org/html/2610.11215v1, appendix A.4), byte for byte.

Source: Qirui Wu, Stan Birchfield, Hesam Rabeti, Angel X. Chang and Bowen Wen, "GATOR: Generative
and Agentic 3D Object Reconstruction From Casual Images", arXiv 2610.11215v1, 2026. License:
CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).

The agent does not read the file directly. `load(mode)` in `__init__.py` applies `CHANGES`, which
lists every edit with its reason. The only edit is a path: the agent CLI runs inside the work folder
(`/work`), so `work/run_budget.json` becomes `/work/run_budget.json`. The prompt names no
Codex-specific tool, so no tool names change. Features that the prompt expects from the helpers (the
original's previews, `review --clay` and a shared detail rig through `review --detail`) are
implemented in the controller and `tools.py`.

`articulation_prompt.md` is our own prompt for the articulation mode (build_plan.md note 11), not
the paper's. The agent gets it when an articulated job's parts stage has split the mesh: it gives
the moving parts their joints with `set_joint`, checks them with `review_articulation` and hands
off within 900 seconds. A part of the part list that the split missed has a requirement with a
`reserve` name, and the agent cuts it out of the mesh with `add_part` first (after PhysAg3D, whose
agent selects each part's surface itself). Surface that the split put in the wrong part is moved
with `move_faces`. Since it is not copied from the paper, `CHANGES["articulation"]` is empty:
there is no original text to record edits against, and the file is edited directly.

`repair_prompt.md` and `replan_prompt.md` are our own too. When `input/parts.json` has a requirement
with a link (a found part, or the reserve of a missing one), the articulation session runs in rounds (`controller.run_rounds`, after PhysAg3D's
repair loop): the build round gets the articulation prompt, and the host's check of each build
(`harpor.agent.validate`) decides the next round, a repair or a planning round (replan). Each round
is a new agent with no memory of the others, so `round_prompt(kind, budget_s, feedback, best,
evidence)` gives it the whole task:

- repair: the articulation prompt with its budget sentence (`BUDGET`) set to the round's budget,
  then `repair_prompt.md`; only the repair right after a planning round gets `PLAN`, the sentence
  that tells it to follow `/work/plan.md`, since that plan came from the same check
- replan: `replan_prompt.md` alone; its agent writes `/work/plan.md`, and every helper is refused

Both templates say that `candidate.blend` holds the last round's work, name the check's report under
`evidence/`, and say that the requirements cannot change and that a span target is graded: a
range must be clear and reachable, and one that reaches less than its target is recorded as reduced. They take the check's feedback, the best candidate so far, the round's budget and
(repair) the plan sentence through `str.format`, so their own text has no braces. `FILES` and `load` serve only the two modes.
