# Agent prompts

`refinement_prompt.md` is Prompt 1, the refinement prompt file embedded in the arXiv HTML of GATOR
(https://arxiv.org/html/2610.11215v1, appendix A.4), byte for byte.

Source: Qirui Wu, Stan Birchfield, Hesam Rabeti, Angel X. Chang and Bowen Wen, "GATOR: Generative
and Agentic 3D Object Reconstruction From Casual Images", arXiv 2610.11215v1, 2026. License:
CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).

The agent does not read the file directly. `load()` in `__init__.py` applies `CHANGES`, which
lists every edit with its reason. There are two edits. A path: the agent CLI runs inside the work folder
(`/work`), so `work/run_budget.json` becomes `/work/run_budget.json`. The budget: HARPOR gives a build
round 900 s (`controller.BUDGET_S`), so "ten-minute (600-second)" becomes "fifteen-minute
(900-second)"; the deadline in `run_budget.json` stays the authority. The prompt names no
Codex-specific tool, so no tool names change. Features that the prompt expects from the helpers (the
original's previews, `review --clay` and a shared detail rig through `review --detail`) are
implemented in the controller and `tools.py`.

A web-app job runs the session in rounds (`controller.run_rounds`): after each build or repair,
the host checks the candidate against the original (`harpor.agent.validate`). Each round is a new agent with no memory of the others, so
`round_prompt(budget_s, feedback, best, evidence)` gives a repair round the whole task: the
refinement prompt, then a short paragraph with the round's budget, the best candidate so far and
the host's feedback (`harpor.agent.validate.feedback`).

`GUIDANCE` is appended to the first prompt and to every round prompt.
