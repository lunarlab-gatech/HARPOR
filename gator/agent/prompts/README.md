# Agent prompts

`refinement_prompt.md` (Prompt 1) and `scratch_prompt.md` (Prompt 2) are the two prompt files
embedded in the arXiv HTML of GATOR (https://arxiv.org/html/2610.11215v1, appendix A.4).

Source: Qirui Wu, Stan Birchfield, Hesam Rabeti, Angel X. Chang and Bowen Wen, "GATOR: Generative
and Agentic 3D Object Reconstruction From Casual Images", arXiv 2610.11215v1, 2026. License:
CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).

The agent does not read these files directly. `load(mode)` in `__init__.py` applies `CHANGES`,
which lists every edit with its reason. The only edits are paths: the agent CLI runs inside the
work folder (`/work`), so `work/run_budget.json` and `work/candidate.blend` become
`/work/run_budget.json` and `/work/candidate.blend`. The prompts name no Codex-specific tool, so
no tool names change. Features that the prompts expect from the helpers (the original's
previews, `review --clay` and a shared detail rig through `review --detail`) are implemented in
the controller and `tools.py`.

`articulation_prompt.md` is our own prompt for the articulation mode (build_plan.md note 11), not
the paper's. The agent gets it when an articulated job's parts stage has split the mesh: it gives
the moving parts their joints with `set_joint`, checks them with `review_articulation` and hands
off within 900 seconds. Since it is not copied from the paper, `CHANGES["articulation"]` is empty:
there is no original text to record edits against, and the file is edited directly.
