# Agent prompts

`refinement_prompt.md` is Prompt 1, the refinement prompt file embedded in the arXiv HTML of GATOR
(https://arxiv.org/html/2610.11215v1, appendix A.4), byte for byte.

Source: Qirui Wu, Stan Birchfield, Hesam Rabeti, Angel X. Chang and Bowen Wen, "GATOR: Generative
and Agentic 3D Object Reconstruction From Casual Images", arXiv 2610.11215v1, 2026. License:
CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).

The agent does not read the file directly. `load()` in `__init__.py` applies `CHANGES`, which
lists every edit with its reason. The only edit is a path: the agent CLI runs inside the work folder
(`/work`), so `work/run_budget.json` becomes `/work/run_budget.json`. The prompt names no
Codex-specific tool, so no tool names change. Features that the prompt expects from the helpers (the
original's previews, `review --clay` and a shared detail rig through `review --detail`) are
implemented in the controller and `tools.py`.

The physics check runs the session in rounds (`controller.run_rounds`): after each build or repair,
the host settles the candidate in MuJoCo and the job's VLM judges the frames
(`harpor.agent.physics_review`). Each round is a new agent with no memory of the others, so
`round_prompt(kind, budget_s, feedback, best, evidence)` gives it the whole task:

- repair: the refinement prompt, then a short paragraph with the round's budget, the best candidate
  so far and the host's feedback (`harpor.agent.validate.feedback`); only the repair right after a
  planning round gets `PLAN`, the sentence that tells it to follow `/work/plan.md`
- replan: a short planning prompt alone; its agent writes `/work/plan.md`, and every helper is refused

`physics_review.GUIDANCE` is appended to the first prompt and to every round prompt.
