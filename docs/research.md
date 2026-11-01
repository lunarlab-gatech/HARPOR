# GATOR research notes

Sources behind `build_plan.md`; where the two files disagree, `build_plan.md` wins.

The generator and benchmark notes are cut: sections 1.3, 1.5, 4.2 to 4.5 and 5, and the Generator and
Training lists of 1.10. The last commit that holds them is 91003c9: `git show 91003c9:docs/research.md`.

Tags: "(unconfirmed)" means no primary source confirmed the claim. "(our design)" marks our own choices.

## 1. The paper

### 1.1 Source and numbering

The paper is "GATOR: Generative and Agentic 3D Object Reconstruction From Casual Images" (arXiv 2610.11215v1, HTML https://arxiv.org/html/2610.11215v1, PDF https://arxiv.org/pdf/2610.11215v1). The project page is https://research.nvidia.com/labs/lpr/gator/. The GitHub repository kasvii/GATOR is not this project. It belongs to an unrelated 2023 human-mesh paper.

Page references such as (p.5) are PDF pages. Table and figure numbers follow the PDF. The equation in section 1.2 comes from the MathML of the arXiv HTML, because the PDF text layer drops glyphs inside equations.

The arXiv HTML labels the qualitative ablation grid (PDF Fig. 5) as "Table 5". From that grid on, HTML table numbers are one higher than PDF numbers, and HTML figure numbers are one lower. For example, PDF Table 6 is HTML Table 7, and PDF Fig. 12 is HTML Fig. 11. The running text of the HTML is shifted the same way.

### 1.2 Inputs and frame

Each view v supplies an RGB image I_v, a target mask M_v, intrinsics K_v, a camera pose T_v and depth D_v. The text c is optional and can be a category name or a VLM caption (p.4).
```math
\mathcal{C}=(\{I_{v},M_{v},K_{v},T_{v},D_{v}\}_{v=1}^{N},c)
```

- The ScanNet++ evaluation feeds up to ten views (p.29), although training uses at most four (p.22). The paper does not explain why this works.
- The paper calls the structure latents "pose-aware occupancy latents" (p.5). It also says that training is pose-aligned (p.2).
- On re-alignment, the paper says that the anchor from partial depth can be wrong when observations are limited. So GATOR aligns the generated structure with the observations and recomputes the normalization before projection (p.19).

### 1.4 Agent stage

- The agent is GPT-6-Astra, run non-interactively through the Codex CLI at extra-high reasoning effort (A.4).
- Sec. 3.5 names the failure modes that the agent targets. They are missing parts, fused openings and texture contamination. The agent should make targeted repairs that keep the reliable regions, the coordinate frame and the scale. It should not remodel the object (p.6).
- The ten-minute budget has no fixed limit on the number of revisions (p.19).
- The from-scratch baseline uses the same agent, harness, effort, budget and inputs, but it gets no initial mesh (p.19). Figures shorten its name to "GPT-6".
- Fig. 11 (p.31) sorts the edits into six types. They are fine detail recovery, texture correction, missing part completion, extraneous geometry removal, topology repair and structure preservation.

#### Prompt 1, refinement (A.4, pp.19–22)

- Cameras come as K plus world-to-camera matrices in column-vector convention with OpenCV axes.
- The agent reads `work/run_budget.json` at launch and after each export. The prompt calls its deadline authoritative. The full 600 s is available at once. The budget is an emergency ceiling, not a target. The prompt sets no phase fractions and no freeze window.
- Reasoning, modeling, review renders and visual inspection count against the budget. GLB serialization through the helper, its import checks and final publication do not.
- After each material improvement, the agent saves a packed .blend and calls the export helper. It must not export a GLB inline from a modeling script. It should not re-export an unchanged scene.
- The agent must use the prebuilt review tool and must not write its own validation, rendering, panel or report scripts. The tool snapshots the candidate, checks import and materials, and reports native-frame anchor evidence and anchor statistics. The prompt says anchor statistics are evidence, not proof of pose or quality.
- If a correction makes the result worse, the agent keeps the earlier reviewed candidate. If a later texture projection regresses, the agent keeps the earlier good repair. Before another revision, the agent calls `checkpoint` with `needs_review` and the outstanding defect. That snapshot must stay intact.
- The agent stops right after it selects and inspects the best candidate. It stops when the supported geometry, articulation and appearance goals are met. It also stops when the remaining uncertainty cannot be resolved, and then it hands off with `needs_review` and names the defect.
- The agent must not stop at a trivial shader change while a structural repair is feasible. It must not fill spare time with invented detail. If the original is already good, the agent delivers it unchanged after inspection. `finish` is the last action. The prompt prefers a `needs_review` handoff over spending the clock on documentation.
- If the worker is interrupted, the controller keeps a valid checkpoint. A recovered export that was never inspected is labeled `needs_review`. An unchanged-original fallback is never called a successful refinement.

#### Prompt 2, from scratch (A.4, pp.19–22)

- The agent gets no starting mesh. It must deliver one posed GLB in the supplied camera world frame. The prompt names the packed file `work/candidate.blend`.
- An input manifest lists each evidence item and its source, such as sensor depth or a Pi3X or VGGT-Ω estimate. It also names the compared model whose inputs are shared. Its metadata gives the selected-view mapping, resolution, depth convention, units, frame and confidence semantics. It also gives the conversion to camera-world units and the resize and crop conventions that match K to the depth.
- Depth and point processing, texturing and pose fitting count against the budget. Only deterministic serialization, import and final publication run outside the clock. No modeling may happen in that interval.
- The review helper adds orbit views and clay views to the per-camera panels. The controller also publishes compact machine-readable receipts.
- The prompt implies that the agent uses only local tools. It implies no network, no installers, no other models, and no access to ground truth, scores or earlier reconstructions.

### 1.6 Benchmarks, baselines and metrics

#### Benchmarks (Sec. 4.1 on p.6, B.2 on pp.23–24, D.1 to D.4 on pp.25–29)

- All methods share the same input setup for each dataset.
- Toys4K (p.23, p.25). The 525 objects were chosen before any method was run, to span the range of geometric complexity in each category. Blender renders them with studio lighting. The four input views come from casual viewpoints with irregular spacing and varied elevation, field of view and roll. The six held-out views differ from the inputs and are used only for evaluation. No method gets the rendered ground-truth depth or cameras.
- LM-O and HB use sensor depth and intrinsics. The LM-O camera poses come from registration with depth and image correspondences. The HB camera poses come from calibration markers.
- HANDAL uses four frames chosen with the visibility annotations. VGGT-Ω estimates the cameras and depth from the frames of each scene.
- ScanNet++ (pp.23–24, p.29). The benchmark extends the DP-Recon, ShapeR and JRM protocol of 69 objects in 6 scenes with 160 targets from 43 more scenes. VGGT-Ω first estimates cameras and depth from the frames of each scene. Then one to ten views per target are picked for target visibility and viewpoint diversity. Target selection, masks and category text all come from the annotations. IoU projects each aligned prediction into all reference views, and failed reconstructions score zero.

#### Inferred target counts

The paper states only the HANDAL ablation count (55, p.29) and the ScanNet++ count (229). It says that the tabletop benchmark uses "all" LM-O, HB and HANDAL objects (p.7, p.27). Every ADD-SB@0.1 value in the results tables is an exact fraction of the counts below.

- LM-O has 8 targets. The value 62.50% equals 5/8.
- HB has 68 targets. The values 63.24% = 43/68, 97.06% = 66/68 and 98.53% = 67/68 fit.
- HANDAL has 55 targets. The values 96.36% = 53/55, 63.64% = 35/55 and 69.09% = 38/55 fit. The Table 4 values 94.55% = 52/55 and 98.18% = 54/55 also fit, so the ablation probably uses the same 55 targets.
- The pooled tabletop total is 131. The values 98.47% = 129/131, 63.36% = 83/131, 85.50% = 112/131, 87.02% = 114/131 and 91.60% = 120/131 fit.
- ScanNet++ has 229 targets. The values 95.63% = 219/229 and 90.83% = 208/229 fit.
- Table 7 has 152 attempts, which is 76 LM-O and HB targets times two view selections. Its 98.03% equals 149/152.

#### Baselines (p.7, p.25, p.27, p.29)

- All baselines use their released checkpoints and four views, unless noted otherwise.
- Pixal3D-SV and Pixal3D-MV both run on Toys4K. On real data, the multi-view checkpoint gets the shared cameras.
- MV-SAM3D keeps its own pose and texture optimization.
- On ScanNet++, Pixal3D, MV-SAM3D, ShapeR, GATOR and GPT-6-Astra share the selected views, masks, cameras and depth. ReconViaGen uses its own estimator (p.29).

#### Metrics (Appendix C, p.25)

- Pose metrics are measured before any object-to-reference alignment (pp.7–8).
- The camera-only similarity is shared by all objects and all methods in a scene. C.1 names HANDAL and ScanNet++ for it. Toys4K also uses estimated Pi3X cameras, so the same registration applies there (our reading). Toys4K has no ADD-SB, so there the registration only starts the shape alignment.
- The tables label silhouette IoU as IoU2D, and Sec. 4.1 cites Pixal3D for it. IoU is averaged over the matched cameras of each object and then over objects.
- For appearance, the prediction and the reference share the cameras and the studio lighting, with no exposure or color fitting. Predictions keep their own materials. Reference CAD models get colors from real-world capture. The paper says that the appearance scores measure agreement with the rendered reference, not accurate material recovery.
- Per-dataset numbers are means over targets.

### 1.7 Results

Bold means bold in the paper. [D] marks a dark green cell (best), and [L] marks a light green cell (second best). The GPT-6-Astra rows are gray in the paper and left out of the ranking. "n/a" marks a cell the paper leaves as a dash. Tables 4 and 7 are in section 1.8.

The text claims that GATOR cuts CD by 19.7% against Pixal3D-MV on Toys4K, roughly halves Pixal3D's CD on the tabletop data, and cuts CD by 23% against RecGen on ScanNet++. The rounded table entries allow a Toys4K cut from 9.5% to 26.1% and a ScanNet++ cut from 22.2% to 26.5%. The tabletop ratio is 0.009 to 0.018, or 0.50. So all three claims are consistent. The claim that GATOR leads all nine tabletop metrics counts the SSIM tie with Pixal3D and leaves out the gray GPT-6-Astra row.

In Table 2, the ADD-SB mean, ADD-SB@0.1, CD, NC and F1 columns equal the object-weighted means of Table 6. The ReconViaGen F1 cell is an exception, because Table 2 gives 0.608 and the object-weighted mean of Table 6 is 0.612. The IoU, PSNR, SSIM and LPIPS columns do not equal those means.

#### Table 1 (p.7): four-view reconstruction on Toys4K († means MultiDiffusion)

| Method | CD↓ | NC↑ | F1↑ | IoU2D↑ | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|---|---|---|---|
| GPT-6-Astra (gray) | 0.013 | 0.754 | 0.614 | 0.774 | 17.11 | 0.865 | 0.214 |
| TRELLIS.2† | 0.014 | 0.751 | 0.628 | 0.747 | 15.73 | 0.855 | 0.231 |
| ReconViaGen | 0.014 | 0.767 | 0.579 | 0.764 | 14.39 | 0.825 | 0.228 |
| Pixal3D-SV | 0.027 | 0.681 | 0.433 | 0.629 | 14.22 | 0.844 | 0.287 |
| Pixal3D-MV | 0.011 [L] | 0.783 [L] | 0.701 [L] | 0.798 [L] | 17.18 [L] | 0.871 [L] | 0.203 [L] |
| GATOR (Ours) | **0.009** [D] | **0.801** [D] | **0.757** [D] | **0.819** [D] | **17.60** [D] | **0.873** [D] | **0.191** [D] |

#### Table 2 (p.7): real-world tabletop reconstruction, LM-O, HB and HANDAL pooled

| Method | ADD-SB Mean↓ | @0.1↑ | CD↓ | NC↑ | F1↑ | IoU2D↑ | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|---|---|---|---|---|---|
| GPT-6-Astra (gray) | 0.027 | 100.00 | 0.016 | 0.799 | 0.773 | 0.791 | 17.16 | 0.886 | 0.160 |
| SimFoundry | 0.183 | 63.36 | 0.040 | 0.716 | 0.533 | 0.626 | 14.90 | 0.875 | 0.200 |
| RecGen | 0.057 | 85.50 | 0.021 | 0.791 [L] | 0.666 | 0.757 | 16.00 | 0.885 [L] | 0.178 |
| ReconViaGen | n/a | n/a | 0.027 | 0.750 | 0.608 | 0.730 | 14.48 | 0.858 | 0.181 |
| ShapeR | 0.057 | 87.02 | 0.028 | 0.709 | 0.606 | 0.700 | n/a | n/a | n/a |
| MV-SAM3D | 0.044 [L] | 91.60 [L] | 0.019 | 0.778 | 0.687 | 0.757 | 15.51 | 0.883 | 0.168 |
| Pixal3D | n/a | n/a | 0.018 [L] | 0.784 | 0.780 [L] | 0.813 [L] | 16.94 [L] | **0.890** [D] | 0.163 [L] |
| GATOR (Ours) | **0.022** [D] | **98.47** [D] | **0.009** [D] | **0.859** [D] | **0.904** [D] | **0.841** [D] | **17.97** [D] | **0.890** [D] | **0.140** [D] |

#### Table 3 (p.8): real-world indoor scenes on ScanNet++ (229 targets, 49 scenes)

| Method | ADD-SB Mean↓ | @0.1↑ | CD↓ | NC↑ | F1↑ | IoU2D↑ |
|---|---|---|---|---|---|---|
| GPT-6-Astra (gray) | 0.049 | 90.83 | 0.033 | 0.716 | 0.581 | 0.716 |
| SimFoundry | 0.125 | 72.49 | 0.104 | 0.664 | 0.426 | 0.614 |
| RecGen | 0.067 | 90.83 [L] | 0.041 [L] | 0.703 [L] | 0.540 [L] | 0.682 [L] |
| ReconViaGen | n/a | n/a | 0.161 | 0.676 | 0.423 | 0.498 |
| ShapeR | 0.071 | 81.66 | 0.060 | 0.684 | 0.458 | 0.610 |
| MV-SAM3D | 0.057 [L] | 90.83 [L] | 0.042 | 0.699 | 0.493 | 0.648 |
| Pixal3D | n/a | n/a | 0.052 | 0.679 | 0.475 | 0.623 |
| GATOR (Ours) | **0.041** [D] | **95.63** [D] | **0.031** [D] | **0.726** [D] | **0.603** [D] | **0.733** [D] |

#### Table 6 (p.27): real-world tabletop reconstruction per dataset

The caption says that the generator alone ("GATOR w/o agent") already beats every baseline on every geometry metric. It also says that refinement improves PSNR and LPIPS on every dataset. In the LM-O block, SimFoundry's 62.50 is marked second best because every other scored method has 100.00.

LM-O (sensor depth)

| Method | ADD-SB Mean↓ | @0.1↑ | CD↓ | NC↑ | F1↑ | IoU2D↑ | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|---|---|---|---|---|---|
| GPT-6-Astra (gray) | 0.045 | 100.00 | 0.031 | 0.777 | 0.524 | 0.789 | 15.92 | 0.899 | 0.243 |
| SimFoundry | 0.375 | 62.50 [L] | 0.043 | 0.783 | 0.460 | 0.748 | 15.50 | 0.882 | 0.252 |
| RecGen | 0.043 [L] | **100.00** [D] | 0.027 | 0.796 | 0.542 | 0.811 | 16.95 | 0.911 | 0.225 |
| ReconViaGen | n/a | n/a | 0.049 | 0.723 | 0.413 | 0.728 | 12.58 | 0.837 | 0.256 |
| ShapeR | 0.060 | **100.00** [D] | 0.050 | 0.651 | 0.316 | 0.712 | n/a | n/a | n/a |
| MV-SAM3D | 0.045 | **100.00** [D] | 0.026 | 0.817 | 0.598 | 0.839 | 15.70 | 0.880 | 0.212 |
| Pixal3D | n/a | n/a | 0.052 | 0.700 | 0.368 | 0.764 | 14.69 | 0.896 | 0.258 |
| GATOR w/o agent (Ours) | **0.034** [D] | **100.00** [D] | 0.023 [L] | 0.827 [L] | 0.655 [L] | 0.872 [L] | 17.21 [L] | 0.917 [L] | 0.204 [L] |
| GATOR (Ours) | **0.034** [D] | **100.00** [D] | **0.022** [D] | **0.836** [D] | **0.665** [D] | **0.873** [D] | **17.53** [D] | **0.925** [D] | **0.189** [D] |

HB (sensor depth)

| Method | ADD-SB Mean↓ | @0.1↑ | CD↓ | NC↑ | F1↑ | IoU2D↑ | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|---|---|---|---|---|---|
| GPT-6-Astra (gray) | 0.020 | 100.00 | 0.017 | 0.812 | 0.731 | 0.873 | 15.94 | 0.864 | 0.189 |
| SimFoundry | 0.200 | 63.24 | 0.050 | 0.704 | 0.468 | 0.650 | 13.24 | 0.845 | 0.234 |
| RecGen | 0.040 | 97.06 | 0.020 | 0.796 | 0.657 | 0.844 | 14.91 | 0.861 | 0.201 |
| ReconViaGen | n/a | n/a | 0.027 | 0.766 | 0.587 | 0.810 | 14.44 | 0.836 | 0.196 |
| ShapeR | 0.030 | 98.53 [L] | 0.023 | 0.757 | 0.661 | 0.811 | n/a | n/a | n/a |
| MV-SAM3D | 0.029 [L] | **100.00** [D] | 0.020 | 0.784 | 0.642 | 0.825 | 15.15 | 0.869 [L] | 0.193 |
| Pixal3D | n/a | n/a | 0.017 [L] | 0.812 | 0.791 | 0.910 | **16.57** [D] | **0.871** [D] | 0.180 |
| GATOR w/o agent (Ours) | **0.012** [D] | **100.00** [D] | **0.009** [D] | 0.882 [L] | 0.916 [L] | **0.919** [D] | 15.83 | 0.863 | 0.174 [L] |
| GATOR (Ours) | **0.012** [D] | **100.00** [D] | **0.009** [D] | **0.883** [D] | **0.917** [D] | 0.917 [L] | 16.53 [L] | 0.866 | **0.166** [D] |

HANDAL (estimated depth)

| Method | ADD-SB Mean↓ | @0.1↑ | CD↓ | NC↑ | F1↑ | IoU2D↑ | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|---|---|---|---|---|---|
| GPT-6-Astra (gray) | 0.032 | 100.00 | 0.012 | 0.787 | 0.861 | 0.674 | 19.10 | 0.914 | 0.106 |
| SimFoundry | 0.134 | 63.64 | 0.026 | 0.723 | 0.624 | 0.572 | 17.16 | 0.915 | 0.143 |
| RecGen | 0.080 | 69.09 | 0.020 | 0.785 | 0.695 | 0.624 | 17.38 | 0.914 | 0.137 |
| ReconViaGen | n/a | n/a | 0.024 | 0.732 | 0.671 | 0.617 | 14.85 | 0.893 | 0.147 |
| ShapeR | 0.090 | 70.91 | 0.030 | 0.658 | 0.581 | 0.539 | n/a | n/a | n/a |
| MV-SAM3D | 0.063 | 80.00 [L] | 0.017 | 0.766 | 0.754 | 0.647 | 15.98 | 0.904 | 0.124 |
| Pixal3D | n/a | n/a | 0.014 | 0.761 | 0.828 | 0.686 | 17.84 | 0.917 [L] | 0.122 |
| GATOR w/o agent (Ours) | **0.032** [D] | **96.36** [D] | **0.008** [D] | 0.826 [L] | **0.936** [D] | **0.757** [D] | 19.42 [L] | **0.927** [D] | 0.097 [L] |
| GATOR (Ours) | 0.033 [L] | **96.36** [D] | 0.009 [L] | **0.832** [D] | 0.923 [L] | 0.728 [L] | **20.07** [D] | 0.917 [L] | **0.096** [D] |

### 1.8 Analyses

#### Ablation (Table 4 on p.9, D.5 on p.29)

The ablation uses 55 HANDAL targets with four views and estimated cameras and depth. Each row adds one component to the row above. Every configuration uses the same sampling settings. All variants train on the same synthetic data with the same pretrained image encoders (p.8). The base and mixer variants share the same geometry and appearance flows. The agent row refines the text-conditioned outputs with the ten-minute A.4 agent and does not rerun the generator. The ablation models are trained separately from the main model, so their scores do not compare directly with Tables 2 and 6. Table 4 uses bold marks only.

| Configuration | ADD-SB Mean↓ | @0.1↑ | CD↓ | NC↑ | F1↑ | IoU2D↑ | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|---|---|---|---|---|---|
| Base model | 0.057 | 94.55 | 0.010 | 0.813 | 0.877 | 0.688 | 18.20 | 0.922 | 0.116 |
| + Modality mixer | 0.037 | 94.55 | 0.011 | 0.811 | 0.872 | 0.700 | 18.16 | 0.922 | 0.114 |
| + Text conditioning | 0.032 | **98.18** | 0.008 | 0.828 | 0.930 | **0.739** | 18.93 | **0.925** | 0.102 |
| + Agentic refinement | **0.032** | **98.18** | **0.007** | **0.829** | **0.936** | 0.717 | **19.60** | 0.918 | **0.097** |

Sec. 4.4 (p.8) says that refinement mainly improves appearance and local topology. It says that average geometry changes little on Toys4K, LM-O and HB and gets slightly worse on HANDAL. In Table 4, refinement on 55 HANDAL targets, probably the same ones, improved CD and F1 instead. Table 1 has no "w/o agent" row, so only Fig. 6 supports the Toys4K part of the claim.

#### Efficiency (Fig. 6 on p.9, Sec. 4.4 on pp.8–9)

The study uses ten Toys4K objects. The x-axis is the mean recorded full-pipeline time. GATOR's time includes an estimated 32 s of generation per object and excludes mesh export. Each point is a mean over valid outputs only. We read the values below from the HTML figure image, so they are approximate. The GPT-6 valid counts come from the text. The paper gives no valid count for GATOR, but the project page says GATOR is valid at every budget. Because the plot covers only these ten objects, its GATOR CD/D of about 0.007 need not match the Table 1 CD of 0.009.

| Series | Budget | Plotted time (min) | CD/D | LPIPS | Valid outputs |
|---|---|---|---|---|---|
| GPT-6 | 1 min | no point | n/a | n/a | 0/10 |
| GPT-6 | 2 min | ~1.99 | ~0.0236 | ~0.322 | 5/10 |
| GPT-6 | 5 min | ~4.97 | ~0.0180 | ~0.255 | 10/10 |
| GPT-6 | 10 min | ~9.89 | ~0.0189 | ~0.258 | 10/10 |
| GPT-6 | 20 min | ~18.96 | ~0.0132 | ~0.215 | 10/10 |
| GATOR | no refinement | ~0.50 | ~0.0071 | ~0.198 | not stated |
| GATOR | 2 min agent | ~2.57 | ~0.0071 | ~0.195 | not stated |
| GATOR | 5 min agent | ~5.55 | ~0.0073 | ~0.190 | not stated |
| GATOR | 10 min agent | ~10.09 | ~0.0071 | ~0.188 | not stated |
| GATOR | 20 min agent | ~18.45 | ~0.0073 | ~0.189 | not stated |

#### Number of views (Table 7 on p.29, D.3 on p.29)

Table 7 pools LM-O and HB and uses nested sets of two, four and eight views. Each target is evaluated with two view selections, and recall counts every attempt. GATOR is reported without refinement. RecGen appears only at two views. Table 7 uses bold marks only.

| Views | Method | ADD-SB Mean↓ | @0.1↑ | CD↓ | NC↑ | F1↑ | IoU2D↑ | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | RecGen | 0.041 | **98.03** | 0.021 | 0.796 | 0.643 | 0.845 | 15.16 | **0.867** | 0.203 |
| 2 | Pixal3D | n/a | n/a | 0.027 | 0.757 | 0.648 | 0.840 | 15.32 | 0.865 | 0.209 |
| 2 | GATOR (Ours) | **0.015** | **98.03** | **0.011** | **0.865** | **0.863** | **0.897** | **15.75** | 0.866 | **0.183** |
| 4 | Pixal3D | n/a | n/a | 0.021 | 0.799 | 0.745 | 0.875 | **16.29** | **0.872** | 0.190 |
| 4 | GATOR (Ours) | **0.014** | **98.03** | **0.010** | **0.874** | **0.891** | **0.902** | 15.86 | 0.866 | **0.179** |
| 8 | Pixal3D | n/a | n/a | 0.020 | 0.813 | 0.774 | 0.886 | **16.74** | **0.875** | 0.185 |
| 8 | GATOR (Ours) | **0.013** | **98.03** | **0.010** | **0.880** | **0.901** | **0.900** | 15.85 | 0.867 | **0.174** |

D.3 gives unrounded GATOR values. From two to eight views, mean ADD-SB falls from 0.0148 to 0.0134, and CD falls from 0.0109 to 0.0096. The paper says that GATOR's two-view CD is about half of Pixal3D's eight-view CD. The ratio of 0.0109 to 0.020 is about 0.55.

#### Physics simulation (D.6 on p.31, Fig. 12 on p.32)

PyBullet simulates all 15 objects that GATOR reconstructs in one ScanNet++ meeting room with ten input views. The scene runs once with the generated outputs and once with the refined outputs. Both runs use the same upright placement slightly above a common floor, the same assumed mass and settings, and convex collision shapes. After settling, three generated chairs lean or topple by 40 to 79 degrees. Their refined versions stay within 1.3 degrees of upright. The refined conference table has straight, uniform legs instead of uneven, slanted ones.

### 1.9 Limitations

D.7 (p.31) says that GATOR depends on target masks and on camera and depth estimates. Gross input failures, such as a missing segmentation or badly wrong depth, can still harm the result. Agentic refinement depends on a proprietary model and adds up to ten minutes of agent time per object. D.7 names tighter integration with segmentation and geometry estimation, open agents, and automatic geometric verification of agent edits as future work.

### 1.10 Details the paper leaves open

Benchmarks and metrics

- For Toys4K, the paper does not list the 525 object IDs or the selection rule. It does not give the render resolution, lighting rig, view ranges, held-out view placement, mask source, Pi3X settings, MultiDiffusion settings or the definition of "robust surface extent". It does not list the ten objects of the efficiency study.
- The paper does not say which LM-O frames are used, how views are chosen or how cameras are registered. It does not say which HB scenes, frames and sensor give the 68 targets. For HANDAL, the paper does not say whether the data comes from the BOP release or the original release, which scenes give the 55 targets, or how frames are picked from the visibility annotations. It does not give the VGGT-Ω checkpoint, the frames per scene, the scale handling or the exact category strings.
- The paper does not give the rule for the "best" views of RecGen and SimFoundry, which view Pixal3D-SV gets, the ShapeR input preparation, or the MV-SAM3D and ReconViaGen settings. It does not say how the two view selections per target in Table 7 are made.
- For ScanNet++, the paper does not list the 49 scenes or the 229 targets. It does not give the rule for the 160 new targets, the image source (DSLR or iPhone) and fisheye handling, the view-selection rule, the SAM3 prompt type, how partial references are cut, or how alignment is fit only to observed surfaces.
- The paper does not say how the camera-only similarity is computed. It does not give the ICP initialization, variant, scale handling, iterations or outlier rejection. Initialization matters for methods that output a canonical frame.
- The paper does not define the CAD diameter, the normal computation, the IoU cameras and resolution, or the source of the reference silhouette. It does not give the appearance cameras, lighting, diffuse material parameters, color management, SSIM settings or LPIPS backbone.
- The paper does not say how failures enter the means. Only the ScanNet++ IoU states that failures score zero. The paper does not explain how IoU and the appearance metrics are pooled in Table 2. It reports one run per method and no run-to-run variance.

Agent and simulation

- The paper does not give the Codex CLI version, flags, sandbox, approval mode or network policy. It does not say whether a system prompt or AGENTS.md exists beyond Prompts 1 and 2, how paths and object IDs are passed, or whether the agent sees the category or caption.
- The paper does not give the Blender version, render engine, lighting rig, render resolution, matched views, comparison layout, number of original previews, `--clay` implementation or detail-rig format. It does not define "native-frame anchor evidence", "anchor statistics", the review receipt or "unchanged bytes".
- The paper does not give the `run_budget.json` schema, the wall-clock or cumulative accounting, the deadline enforcement, the export timeout, how export time is credited, or whether start-up latency counts. It does not say whether the helpers are scripts, MCP tools or Codex functions, what they take and return, or what the reimport and independent validation check.
- The paper does not give the file formats of depth, pointmaps, confidence, masks and cameras. It does not say whether images are full frames or crops, whether the refinement GLB is in the camera-world frame, or whether placement is baked into vertices or stored as a node transform.
- The paper does not report how many runs ended in `needs_review`, deadline recovery or unchanged fallback. It does not report the mean agent time used or the agent failures. It does not say which GATOR budgets were tested in Fig. 6, what hardware the 32 s estimate assumes, or what the GPT-6 cell in the 0.5-min column of Fig. 6(b) shows. It does not say whether the Table 4 agent runs used Prompt 1 unchanged.
- The paper does not give the PyBullet version, masses, friction, time step, settle time, convex decomposition method or tilt measurement. It does not list the 15 objects or say what "upright placement" means relative to the predicted pose.

## 2. Stage A: MV-SAM3D

The repos were read on 2026-10-13. In the paths below, `MV`, `S3` and `DA3` stand for these blob URLs:

- `MV`: MV-SAM3D HEAD `abb04b5` (2026-05-30), https://github.com/devinli123/MV-SAM3D/blob/abb04b5e8af5bc33b0265bdf19937e76bbb6bcdd
- `S3`: SAM 3D Objects HEAD `f91db41` (2026-06-02), https://github.com/facebookresearch/sam-3d-objects/blob/f91db411c50efee93d8db7aeb323885650f6f722
- `DA3`: Depth Anything 3 HEAD `3d835ec` (2026-07-27), https://github.com/ByteDance-Seed/Depth-Anything-3/blob/3d835ec1a5802d64a8b8b15f817a1ab54809bfe4

The MV-SAM3D paper is arXiv 2603.11633 v2 (2026-04-09), https://arxiv.org/abs/2603.11633. "(from code)" marks a statement read from source but not run.

### 2.1 What it is and licenses

- The MV-SAM3D repo is a copy of the SAM 3D Objects source with new files added. It is not a GitHub fork (`"fork": false` at https://api.github.com/repos/devinli123/MV-SAM3D). It keeps the SAM 3D weights unchanged and adds multi-view fusion, Depth Anything 3 (DA3) pointmap input and a pose refinement step (`MV/README.md`, https://arxiv.org/abs/2603.11633).
- Every SAM 3D Objects source file is present in MV-SAM3D except `demo.py`, `doc/` (setup.md and images), the four notebooks with their sample images, `checkpoints/.gitignore` and `patching/hydra` (from code, file-list diff of the two repos).
- Fusion averages or weights the shape and texture latents across views (`MV/sam3d_objects/pipeline/multi_view_utils.py#L48`). The author confirms the view 0 pose (scale, rotation, translation) in https://github.com/devinli123/MV-SAM3D/issues/1.
- The generator runs once per view at every flow step, so cost grows linearly with the number of views (from code, `MV/sam3d_objects/pipeline/multi_view_utils.py#L138`).

Sources for the licenses in the build plan's Dependencies table are `MV/LICENSE` and `S3/LICENSE` (identical files), `S3/README.md` (License section), `DA3/LICENSE`, `DA3/README.md#L216-L217` and https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE-1.1. Other license facts:

| Component | License | Source |
|---|---|---|
| SAM License file | Meta "SAM License", last updated 2025-11-19. GitHub shows it as "Other". | `MV/LICENSE` |
| DA3NESTED-GIANT-LARGE weights (the run_da3.py default) | CC BY-NC 4.0 (non-commercial only), like -1.1 | `DA3/README.md#L216-L217` |
| DA3-BASE, DA3-SMALL, DA3METRIC-LARGE, DA3MONO-LARGE | Apache 2.0 | `DA3/README.md#L223-L230` |
| MoGe code | MIT, except its bundled DINOv2 code (Apache 2.0) | https://github.com/microsoft/MoGe/blob/a8c37341bc0325ca99b9d57981cc3bb2bd3e255b/README.md (License section) |
| MoGe weights `Ruicheng/moge-vitl` | MIT (HF license tag), not gated | https://huggingface.co/Ruicheng/moge-vitl |
| DINOv2 code and weights | Apache 2.0 | https://github.com/facebookresearch/dinov2 (README, License section) |

The SAM License grants use, modification and redistribution. Redistribution must stay under the same agreement with a copy of it. It requires credit in publications, trade-control compliance, no ITAR or military end uses, and no reverse engineering of the SAM Materials (`MV/LICENSE`, section 1.b).

### 2.2 Install and checkpoints

The MV-SAM3D README only says to follow the setup of SAM 3D Objects and of DA3 (`MV/README.md`). Its requirement files and conda env file are byte-identical to those of SAM 3D Objects (from code, `diff`).

Build plan note 4 lists the main pins. Their sources are `MV/environments/default.yml` (Python, CUDA), `MV/requirements.txt` (spconv, hydra-core), `MV/requirements.p3d.txt` (pytorch3d, flash_attn) and `MV/requirements.inference.txt` (kaolin, gsplat). Other pins and gaps:

| Item | Pin | Source |
|---|---|---|
| Platform | linux-64 | `S3/doc/setup.md` |
| CUDA toolchain | cuda-toolkit 12.1.1 and nvcc 12.1.105 from conda, and gcc 12.4 | `MV/environments/default.yml` |
| torch | Not listed by name. It is fixed by `torchaudio==2.5.1+cu121` and the kaolin wheel index `torch-2.5.1_cu121`. | `MV/requirements.txt`, `S3/doc/setup.md` |
| xformers | 0.0.28.post3 | `MV/requirements.txt` |
| MoGe | git commit `a8c3734`, which pulls utils3d at `3913c65` | `MV/requirements.txt`, https://github.com/microsoft/MoGe/blob/a8c37341bc0325ca99b9d57981cc3bb2bd3e255b/pyproject.toml |
| bpy | 4.3.0 (Blender as a Python module) | `MV/requirements.txt` |
| nvdiffrast | Not listed. Needed only for texture baking (section 2.4). | from code |
| diff_gaussian_rasterization | Not listed. Optional import. Needed by the default Gaussian render backend used in texture baking. | `MV/sam3d_objects/model/backbone/tdfy_dit/renderers/gaussian_render.py#L23-L34` |

The install steps of `S3/doc/setup.md` work inside the MV-SAM3D clone. Build plan note 4 has the recipe that worked on ICE.

- `patching/hydra` downloads a patched `hydra/core/utils.py` from GitHub (`S3/patching/hydra`).
- A build off a GPU node may make pytorch3d raise "Not compiled with GPU support" (`S3/doc/setup.md`).
- `notebook/inference.py` sets `os.environ["CUDA_HOME"] = os.environ["CONDA_PREFIX"]` at import. Without `CONDA_PREFIX` this raises `KeyError` (`MV/notebook/inference.py#L5`).

#### Depth Anything 3 install

- `scripts/run_da3.py` imports from `src/` of the sibling clone `../Depth-Anything-3`. It raises an error if the folder is missing (`MV/scripts/run_da3.py#L30-L45`).
- The DA3 install is `pip install xformers torch>=2 torchvision`, then `pip install -e .` (`DA3/README.md#L88-L93`). DA3 requires `numpy<2` and Python 3.9 to 3.13 (`DA3/pyproject.toml`).
- DA3's optional `[gs]` extra pins gsplat `0b4dddf`, which differs from SAM 3D's `2323de5` (`DA3/pyproject.toml`). run_da3.py does not use the Gaussian head, so the extra is not needed.

#### Checkpoints and runtime downloads

| Model | Location | Notes | Source |
|---|---|---|---|
| SAM 3D Objects | HF `facebook/sam-3d-objects` | Access needs a request and acceptance of Meta's terms. It is refused in sanctioned jurisdictions. Files in `checkpoints/`: `pipeline.yaml`, `ss_generator.ckpt` (6.69 GB), `slat_generator.ckpt` (4.91 GB), decoders and encoders. | `S3/doc/setup.md`, https://huggingface.co/facebook/sam-3d-objects/tree/main/checkpoints |
| DA3 | HF `depth-anything/DA3NESTED-GIANT-LARGE` (run_da3.py default) | No gating notice on the HF page. The DA3 README marks this checkpoint deprecated and recommends `DA3NESTED-GIANT-LARGE-1.1`. 1.40B params. | `MV/scripts/run_da3.py#L151`, `DA3/README.md#L211` |
| Depth model inside SAM 3D | Set in the gated `pipeline.yaml` | `Ruicheng/moge-vitl` is not gated. The pipeline instantiates `moge.model.v1.MoGeModel.from_pretrained` at load time (issue #96 trace). A user log in issue #67 shows that load fetching `Ruicheng/moge-vitl/model.pt` from HF. These are user logs, not docs. | `MV/sam3d_objects/pipeline/depth_models/moge.py`, https://github.com/facebookresearch/sam-3d-objects/issues/96, https://github.com/facebookresearch/sam-3d-objects/issues/67 |
| DINOv2 | `torch.hub.load(..., source="github")` | Not gated. The code default is `dinov2_vitb14`, but user logs show the shipped config loads `dinov2_vitl14_reg`. It downloads the hub code from GitHub and `dinov2_vitl14_reg4_pretrain.pth` from dl.fbaipublicfiles.com at runtime. | `MV/sam3d_objects/model/backbone/dit/embedder/dino.py#L14-L35`, logs in https://github.com/facebookresearch/sam-3d-objects/issues/15 and https://github.com/facebookresearch/sam-3d-objects/issues/70 |
| SAM 3 | HF `facebook/sam3` (section 4.1) | Used only by `preprocessing/build_mvsam3d_dataset.py`. Paths are hard-coded to `/mnt/workspace/users/lbc/sam3`. Use `--sam3_checkpoint` or bring our own masks (we make ours with build plan note 3). | `MV/preprocessing/sam3_segmenter.py#L23-L31`, https://huggingface.co/facebook/sam3 |

The weights download is in build plan note 4, step 3.

- The HF repo `facebook/sam-3d-objects` was last modified on 2026-06-12, after the S3 HEAD commit (https://huggingface.co/api/models/facebook/sam-3d-objects). Whether `pipeline.yaml` changed then is (unconfirmed), because the file is gated.
- MV-SAM3D loads `checkpoints/{model_tag}/pipeline.yaml` relative to the working directory. The default tag is `hf` (`MV/run_inference_weighted.py#L2588`).

### 2.3 Depth Anything 3 and `da3_output.npz`

- `scripts/run_da3.py` also takes `--process_res 504`, `--no_vis` and `--device cuda`. It calls `DepthAnything3.from_pretrained(model).inference(image=paths, process_res=504, export_format="mini_npz-glb-depth_vis", show_cameras=True)`. It passes no cameras, so DA3 estimates the poses (`MV/scripts/run_da3.py`).
- It sorts images by numeric stem. It reads png, jpg, jpeg, webp and bmp.
- Without `--model_path`, the script first tries local paths, including the raw HF cache folder `models--depth-anything--DA3NESTED-GIANT-LARGE` (`MV/scripts/run_da3.py#L137-L152`). Whether `from_pretrained` loads that folder is (unconfirmed).
- DA3 resizes the long side to `process_res` (504) and rounds each side to a multiple of 14. All arrays are at that size, not the input size (`DA3/src/depth_anything_3/utils/io/input_processor.py#L316-L377`).

Arrays saved in `da3_output.npz` (`MV/scripts/run_da3.py#L234`). `harpor/stage_a/mvsam3d.py` writes the same layout. "Required for" names what MV-SAM3D reads each key for:

| Key | Shape | Content | Required for |
|---|---|---|---|
| `depth` | (N, H, W) | DA3 depth. Meters for the Nested model (`DA3/README.md#L235`). | pose optimization |
| `pointmaps` | (N, H, W, 3) | Depth back-projected with `intrinsics`, in each view's own OpenCV camera frame (x right, y down, z forward). A script comment says "PyTorch3D coords", but the code computes OpenCV coordinates. | - |
| `pointmaps_sam3d` | (N, 3, H, W) | Same data, channel-first. This is the only key MV-SAM3D requires (`MV/run_inference_weighted.py#L2652-L2660`). Any H and W work, because of the nearest resize below. | every run with `--da3_output` |
| `extrinsics` | (N, 3, 4) | OpenCV world-to-camera in DA3's own world frame (`DA3/README.md#L117`). | visibility weighting, pose optimization, merge |
| `intrinsics` | (N, 3, 3) | At the processed resolution. | pose optimization |
| `image_files` | (N,) | Input paths. Views are matched to inference images by file stem. | view matching; without it, matching falls back to index order |
| `process_res` | scalar | 504 by default. | - |

- Avoid `Inf` in invalid pixels. A SAM 3D issue shows that `Inf` in pointmaps produces a NaN scale (https://github.com/facebookresearch/sam-3d-objects/issues/174). The pipeline marks clipped points as NaN itself (`MV/sam3d_objects/pipeline/inference_pipeline_pointmap.py#L214-L238`), so NaN is likely safe for invalid pixels (unconfirmed).

- Unless `--no_vis` is set, DA3 also writes `scene.glb`. This is a point cloud in glTF axes. Its metadata key `hf_alignment` is a 4x4 matrix built from the first camera and the point-cloud center (`DA3/src/depth_anything_3/utils/export/glb.py#L136-L159`). `--merge_da3_glb` and `--run_pose_optimization` need `scene.glb`, so do not pass `--no_vis` with either flag.
- `hf_alignment` must survive the GLB write and reload. In our local test with trimesh 5.1.1, it came back as a nested list. The trimesh version in the SAM 3D env is not pinned, so this is (unconfirmed) for that env. If it is missing, the merge code falls back to the extrinsics without centering and logs a warning (`MV/run_inference_weighted.py#L128-L171`). Pose optimization raises an error instead (`MV/sam3d_objects/pose_align/pose_optimization.py#L123-L126`).
- MV-SAM3D nearest-resizes each external pointmap to the image size and converts it from OpenCV to PyTorch3D axes, (x, y, z) to (-x, -y, z) (`MV/sam3d_objects/pipeline/inference_pipeline_pointmap.py#L261-L315`).
- Without `--da3_output`, SAM 3D's own depth model runs per view (`MV/sam3d_objects/pipeline/inference_pipeline_pointmap.py#L248-L259`).

### 2.4 Inputs, run command and outputs

#### Inputs

Build plan note 3 gives the folder layout. Sources: `MV/README.md`, `MV/notebook/load_images_and_masks.py#L202-L332`.

- Images may be RGB `.png` or `.jpg`. A mask may be named `<n>.png`, as in the README tree, or `<n>_mask.png`, as in the shipped example.
- Views are matched by file stem across `images/`, the mask folder and the DA3 `image_files`. A view without a mask is skipped.
- A mask in any mode other than RGBA is accepted with a warning and treated as the whole image. This includes RGB, grayscale (`L`) and palette PNGs, and any `.jpg` mask (`MV/notebook/load_images_and_masks.py#L33-L40`). Plain binary masks must be converted to RGBA first.
- The loader reads only `.png` and `.jpg`. run_da3.py also reads `.jpeg`, `.webp` and `.bmp`. So mixed extensions can break the view matching (from code).
- There is no config file. Settings are CLI flags. The model config is `checkpoints/hf/pipeline.yaml`.
- Without `--mask_prompt`, images and `<name>_mask.png` files sit in one folder.

#### Run command and arguments

Paths to `notebook/`, `checkpoints/` and `visualization/` are relative to the working directory. Main arguments of `run_inference_weighted.py` (`MV/run_inference_weighted.py#L3871-L3970`):

| Flag | Default | Meaning |
|---|---|---|
| `--input_path` | required | Scene folder |
| `--mask_prompt` | none | Mask folder name. A comma list switches to multi-object mode. |
| `--image_names` | all | Comma list of stems |
| `--da3_output` | none | Path to `da3_output.npz` |
| `--seed`, `--stage1_steps`, `--stage2_steps` | 42, 50, 25 | Sampling |
| `--decode_formats` | `gaussian,mesh` | Outputs to decode |
| `--no_stage1_weighting`, `--no_stage2_weighting` | off | Use a simple average across views |
| `--stage2_weight_source` | `entropy` | `visibility` and `mixed` need `--da3_output` |
| `--merge_da3_glb` | off | Writes object plus DA3 scene. Needs `--da3_output`. If `scene.glb` is missing, it only logs a warning (`MV/run_inference_weighted.py#L108-L111`). |
| `--overlay_pointmap` | off | Object over the view 0 pointmap |
| `--run_pose_optimization` | off | Chamfer pose refinement against DA3 points. Also `--pose_opt_iterations 300`, `--pose_opt_lr 0.01`, `--pose_opt_optimize_scale`. |
| `--model_tag` | `hf` | Checkpoint folder |

- There is no output-path flag (https://github.com/devinli123/MV-SAM3D/issues/10). The full output folder is `./visualization/<dataset>/<mask_prompt>/<dataset>_<mask>_mv_s1a30_s2e30_<timestamp>/` (`MV/run_inference_weighted.py#L1873-L1954`).
- `run_inference.py` is a simpler simple-average script. It has no DA3 input and does not save the pose (`MV/run_inference.py`). Use `run_inference_weighted.py`.

#### Single-object and multi-object mode

- One `--mask_prompt` name runs single-object mode. A comma list runs multi-object mode (`MV/run_inference_weighted.py#L3987`).
- Multi-object mode loops over objects and reruns full single-object inference for each. It reloads the model each time and forces `merge_da3_glb=True`, so it needs `--da3_output` (`MV/run_inference_weighted.py#L2160-L2360`, `#L2299`, `#L2733`).
- Each object's pose is relative to the first view in which that object has a mask (from code).
- The paper's collision and contact terms and its layout injection are not in the code. `--run_pose_optimization` uses Chamfer distance plus regularizers only (`MV/sam3d_objects/pose_align/pose_optimization.py#L423-L434`). A user reports it makes a 4-object layout worse (https://github.com/devinli123/MV-SAM3D/issues/19).
- For GATOR, single-object mode per object is enough.

#### Outputs

| File | Format | Frame |
|---|---|---|
| `result.glb` | trimesh mesh with vertex colors. No UVs, no texture, no simplification. | Object canonical, GLB Y-up axes, about [-0.5, 0.5]^3, unscaled |
| `result.ply` | 3D Gaussian PLY (xyz, normals, f_dc, opacity, scale, rotation) | Object canonical, Z-up (no axis swap) |
| `params.npz` | `translation` (3), `rotation` (quaternion w, x, y, z), `scale` (3), `downsample_factor`, `pointmap_scale`, `pointmap_shift`, `coords` | Pose from canonical to view 0 camera |
| `inference.log`, `debug_masked_images/` | text, PNG | - |
| `result_merged_scene.glb` (with `--merge_da3_glb`) | object plus DA3 points | DA3 `scene.glb` frame |
| `pose_optimization/optimized_params.npz`, `result_pose_optimized.glb`, `result_merged_scene_optimized.glb` (with `--run_pose_optimization`) | refined pose, canonical mesh copy, merged scene | as above |

Sources: `MV/run_inference_weighted.py#L3160-L3290`, `MV/sam3d_objects/model/backbone/tdfy_dit/representations/gaussian/gaussian_model.py#L136-L161`.

#### Textured GLB

- `run_weighted_inference` hard-codes `with_mesh_postprocess=False`, `with_texture_baking=False` and `use_vertex_color=True` (`MV/run_inference_weighted.py#L2943` single view, `#L2992` multi-view). There is no CLI flag to change this.
- With `with_texture_baking=True`, `to_glb` unwraps UVs with xatlas, renders the Gaussians at 1024 px and writes a PBR `baseColorTexture` (`MV/sam3d_objects/model/backbone/tdfy_dit/utils/postprocessing_utils.py#L362-L690`).
- Texture baking needs nvdiffrast even with the pytorch3d engine. `bake_texture` builds a utils3d `RastContext` in "opt" mode for every engine (`postprocessing_utils.py#L473`), and that class imports `nvdiffrast.torch` (https://github.com/EasternJournalist/utils3d/blob/3913c65d81e05e47b9f367250cf8c0f7462a0900/utils3d/torch/rasterization.py).
- Texture baking renders Gaussians with the default backend `inria`, which needs `diff_gaussian_rasterization` (`MV/sam3d_objects/model/backbone/tdfy_dit/utils/render_utils.py#L77`). The graphdeco-inria version lacks the `kernel_size` argument this code passes. A user reports the mip-splatting fork works. A SAM 3D contributor suggests switching the default backend to `gsplat` instead (https://github.com/facebookresearch/sam-3d-objects/issues/29).
- The `Inference` wrapper forces `rendering_engine = "pytorch3d"` (`MV/notebook/inference.py#L88`, also `MV/run_inference_weighted.py#L2598`). With pytorch3d, mesh post-processing skips hole filling (`postprocessing_utils.py#L619`). A user reports that pytorch3d baking gives a washed-out texture and nvdiffrast fixes it. This is a user report, not an official statement (https://github.com/facebookresearch/sam-3d-objects/issues/75).
- SAM 3D Objects behaves the same. Its `Inference` wrapper also passes `with_texture_baking=False` and `use_vertex_color=True`, so its default GLB has vertex colors (`S3/notebook/inference.py#L114-L117`). A SAM 3D user confirms that `with_texture_baking=True` is what produces a texture map (https://github.com/facebookresearch/sam-3d-objects/issues/146).
- The edit in build plan note 4 works: the textured GLB looks right (build plan note 4). Optional extras are `with_mesh_postprocess=True`, a compatible `diff_gaussian_rasterization` in place of the `gsplat` backend, and the `nvdiffrast` rendering engine.
- The mesh frame does not change. `to_glb` applies the same Z-up to Y-up rotation with or without baking (`postprocessing_utils.py#L671`).

### 2.5 Frames and units

- The pose in `params.npz` is relative to the camera of view 0. It is not in the DA3 world frame. Sources: author reply "It's view 0." in https://github.com/devinli123/MV-SAM3D/issues/8, `MV/sam3d_objects/pipeline/inference_pipeline.py#L1708-L1716`.
- PyTorch3D camera axes are x left, y up, z forward (`MV/sam3d_objects/utils/coordinate_transforms.py#L1-L45`).
- The pose decoder uses view 0's `pointmap_scale` and `pointmap_shift`, so units are those of the view 0 pointmap (`MV/sam3d_objects/pipeline/inference_pipeline.py#L1710`). Without DA3, the units come from the depth model in `pipeline.yaml`. The pinned MoGe v1 outputs a scale-invariant point map (MoGe README at `a8c3734`, line 89). So the pose would not be metric in that case (unconfirmed for this config).

Mapping `result.glb` to the OpenCV camera frame of view 0 (from code, after `merge_glb_with_da3_aligned` in `MV/run_inference_weighted.py#L195-L235`):

```python
import numpy as np
import torch
import trimesh
from pytorch3d.transforms import Transform3d, quaternion_to_matrix

p = np.load("params.npz")
A = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
scene = trimesh.load("result.glb")
V = np.vstack([g.vertices for g in scene.geometry.values()])   # GLB Y-up canonical
v = V @ A.T                                                     # back to SAM 3D Z-up canonical
s = torch.tensor(p["scale"], dtype=torch.float32).reshape(1, -1).expand(1, 3)
R = quaternion_to_matrix(torch.tensor(p["rotation"], dtype=torch.float32).reshape(1, 4))  # w, x, y, z
t = torch.tensor(p["translation"], dtype=torch.float32).reshape(1, 3)
T = Transform3d().scale(s).rotate(R).translate(t)               # row vectors: (v * s) @ R + t
x_p3d = T.transform_points(torch.from_numpy(v).float()[None])[0].numpy()  # PyTorch3D camera of view 0
x_cv = x_p3d * np.array([-1.0, -1.0, 1.0])                      # OpenCV camera of view 0
```

- The GLB step `v = V @ A.T` exactly undoes the rotation `to_glb` applies (`postprocessing_utils.py#L671`). It gives the same vertices as the GLB-to-SAM3D step in a user's working alignment script for SAM 3D (https://github.com/facebookresearch/sam-3d-objects/issues/144).
- MV-SAM3D's own comments name these axes inconsistently (`coordinate_transforms.py` versus `to_glb`). Trust the matrices, not the labels.
- For `result.ply`, apply `T` to the Gaussian centers directly, without the `A` step (`MV/notebook/inference.py#L255-L270`).
- A point in the DA3 world frame is `X_w = inv(E0) @ X_cv`, where `E0` is the `extrinsics` of view 0 padded to 4x4. For the `scene.glb` frame, also multiply by `hf_alignment`.
- Alignment is not proven in MV-SAM3D. Two open issues ask how to put the result back onto the input images. Neither has a reply, and one user says the SAM 3D method from issue #144 does not work for them in MV-SAM3D (https://github.com/devinli123/MV-SAM3D/issues/11, https://github.com/devinli123/MV-SAM3D/issues/14). SAM 3D users also report that predictions are not always aligned with the depth or the mask, mostly for small and thin objects (https://github.com/facebookresearch/sam-3d-objects/issues/162).

### 2.6 VRAM, runtime and known issues

#### VRAM and runtime

- The MV-SAM3D paper ran on one A100 80 GB (https://arxiv.org/html/2603.11633v2).
- A user reports out-of-memory on 48 GB with visibility weighting. There is no reply (https://github.com/devinli123/MV-SAM3D/issues/7). An L40S has 48 GB (https://www.nvidia.com/en-us/data-center/l40s/). On our L40S, the default entropy mode with 8 views peaked at 18.7 GiB allocated (build plan note 4b).
- Runtime per object is not reported in the repo, the paper or the issues. On our L40S, 8 views took 294 s (build plan note 4b). The author has tested up to about 20 views (https://github.com/devinli123/MV-SAM3D/issues/3).
- The paper used 2 and 5 views on GSO. Its own scenes hold 2 to 8 objects, and each scene was captured from 15 viewpoints (https://arxiv.org/html/2603.11633v2).

#### Shipped example

- `data/example/images/1.png` to `8.png` are RGB PNGs at 4096 x 3072. `data/example/stuffed_toy/1_mask.png` to `8_mask.png` are RGBA masks. The folder is about 98 MB in total (`MV/data/example`).
- No DA3 output is shipped, so `run_da3.py` must run first.
- The multi-object scene `desk_objects0` in the README is not shipped (https://github.com/devinli123/MV-SAM3D/issues/17).
- The example names are 1 to 8. A user passed `--image_names 0,1,2,3,4,5,6,7` with visibility weighting and got the assertion "Number of pointmaps must match number of images". The issue is closed with no reply (https://github.com/devinli123/MV-SAM3D/issues/5).
- A run without DA3 should also work: `python run_inference_weighted.py --input_path ./data/example --mask_prompt stuffed_toy` (from code, not run). The pose is then not metric (section 2.5).

#### Quality reports

- Users see the merged scene object flipped or misaligned (https://github.com/devinli123/MV-SAM3D/issues/9, open). Two users in the issue #9 comments find the simple average better than the default weighting.
- One user cannot reproduce the paper's GSO30 numbers, and a second user sees worse quality on clean single-object images (https://github.com/devinli123/MV-SAM3D/issues/18, open, no author reply).
- The author says weighting helps with uneven views and symmetric objects (https://github.com/devinli123/MV-SAM3D/issues/12).

## 3. Stage B: the agent harness

### 3.1 Blender

**Version and invocation**
- The Blender 5.2.2 Linux x64 tarball needs glibc 2.28 or newer (https://www.blender.org/download/). Blender 5.3 is in beta until Nov 4, 2026, and it makes Vulkan the default GPU backend on Linux (https://developer.blender.org/docs/release_notes/5.3/). The study stays on 5.2.2 for every run.
- The only Linux wheel of `bpy` 5.2.2 is `cp313 manylinux_2_28_x86_64` (https://pypi.org/project/bpy/). The `bpy` module behaves like `blender --background --factory-startup`. It does not accept command-line arguments such as `--cycles-device` and `--threads`. It loads the default scene with a cube, so scripts call `read_factory_settings(use_empty=True)`. Other Python modules can use the GPU in a way that blocks Cycles. The module writes no crash log, cannot be reloaded and edits one .blend at a time (https://docs.blender.org/api/current/info_advanced_blender_as_bpy.html).
- The agent and the helpers therefore use the full Blender binary (our design). The agent runs `blender -b candidate.blend --python-exit-code 1 --python edit.py` (the exit-code flag must come first, or a failing script exits 0), and each script saves the packed .blend. The .blend file then holds all state, which keeps checkpoints simple.
- Useful command-line flags are `-b`, `-P/--python`, `--python-expr`, `--python-exit-code`, `-E/--engine`, `-f`, `--factory-startup` and `--gpu-backend {vulkan,metal,opengl}`. After `--`, the flag `--cycles-device {CPU,CUDA,OPTIX,HIP,ONEAPI,METAL}` selects the Cycles device, and `+CPU` adds the CPU (https://docs.blender.org/manual/en/latest/advanced/command_line/arguments.html).
- On ICE, the Blender run of the part split took 1.0 s, and the Apptainer start brought the step to 10 s (build plan note 11). If many small scripts make this cost matter, a persistent Blender worker per GPU would help, like the bpy exec socket of BlenderMCP (https://github.com/ahujasid/blender-mcp) or the per-GPU `BlenderServer` of SceneSmith (https://github.com/nepfaff/scenesmith). We do not depend on BlenderMCP itself, because its `generate_3d`, `search_assets` and `import_asset` tools would break Prompt 2's rule against external assets.

**glTF import, export and the axis mapping**
- The Blender 5.2 importer is `bpy.ops.import_scene.gltf(filepath=..., import_pack_images=True, merge_vertices=False, import_shading='NORMALS', guess_original_bind_pose=True, import_scene_as_collection=True, import_merge_material_slots=True, ...)` (https://docs.blender.org/api/current/bpy.ops.import_scene.html).
- The exporter defaults that matter are `export_yup=True`, `export_apply=False`, `export_materials='EXPORT'`, `export_image_format='AUTO'`, `export_texcoords=True`, `export_normals=True`, `export_cameras=False`, `export_lights=False`, `use_selection=False`, `export_draco_mesh_compression_enable=False` and `export_vertex_color='MATERIAL'`. Setting `export_apply=True` applies modifiers but prevents the export of shape keys. The 5.2 signature has `export_format=''` as the default, so the export helper must set `export_format='GLB'` explicitly (https://docs.blender.org/api/current/bpy.ops.export_scene.html).
- The importer converts glTF Y-up space to Blender Z-up space with `convert_loc(x) = u * Vector([x[0], -x[2], x[1]])`, where `u = 1 / scene.unit_settings.scale_length`. Cameras and lights get an extra +90 degree correction about X. A debug branch (`bpy.app.debug_value == 100`) turns the conversion off. The source is https://raw.githubusercontent.com/KhronosGroup/glTF-Blender-IO/main/addons/io_scene_gltf2/blender/imp/blender_gltf.py (lines 75 to 120).
- Because `u` scales positions on import, the scene keeps `scene.unit_settings.scale_length = 1.0`. Nobody exports with `export_yup=False` or converts the cameras a second time. A double or missing axis conversion makes the candidate bounding box equal the original box rotated 90 degrees about X. The reimport check in `export_glb` looks for this signature.
- Prompt 2 asks the agent to apply the axis conversion exactly once. With the stock importer, this means exporting with `export_yup=True` and adding no manual rotation.

**Headless rendering**
- Cycles with OptiX needs no display, and `apptainer --nv` binds the OptiX libraries through `nvliblist.conf`. `render_common.py` selects the device in the script.

**Matched cameras, lighting and clay renders**
- For an OpenCV world-to-camera matrix `w2c`, the Blender camera gets `cam.matrix_world = inv(w2c) @ diag(1, -1, -1, 1)`. OpenCV cameras look along +Z with +Y down, and Blender cameras look along -Z with +Y up. Flipping the camera's Y and Z axes maps one to the other. `render_common.py` uses this mapping, and the review's mask IoU on the teddy (0.87) matched the placement's (0.865) (build plan note 5).
- For the intrinsics, BlenderProc's `set_intrinsics_from_K_matrix` computes `pixel_aspect` from fx/fy, `f_mm = fx * sensor_mm / view_fac_px`, `shift_x = (cx - (W - 1)/2) / -view_fac_px` and `shift_y = (cy - (H - 1)/2) / view_fac_px * pixel_aspect_ratio` (https://raw.githubusercontent.com/DLR-RM/BlenderProc/main/blenderproc/python/camera/CameraUtility.py, lines 245 to 295). The `(W - 1)/2` term assumes integer pixel centers, so the K convention must be checked. The render resolution equals the input image size at 100 percent, so overlays line up pixel for pixel with the photos and the masks.
- Matched lighting uses one fixed world light, a fixed exposure and the `Standard` view transform (our design). The original and the candidate render with the same settings in the same Blender process. The 5.2 default view transform is probably AgX (unconfirmed), so the helper sets the transform explicitly.
- Clay renders use `view_layer.material_override` with an opaque gray Principled BSDF (alpha 1). Alpha-painted holes then render as solid surface, and only real geometric gaps stay open. Whether `material_override` keeps this name in 5.2 is unconfirmed. The fallback is to assign a temporary clay material to every mesh slot.

## 4. SAM 3 and physical properties

### 4.1 SAM 3

- SAM 3 accepts short noun phrases, image exemplars, points, boxes or masks as prompts. For images, it outputs instance masks, xyxy boxes, scores and a semantic map. For video, it also outputs object IDs per frame.
- SAM 3.1 ("Object Multiplex", HF `facebook/sam3.1`, gated) speeds up multi-object video tracking. Its HF license tag shows only "other".
- SAM 3 uses the SAM License of Nov 19, 2025 (section 2.1), which has no non-commercial clause.
- Meta's code (github.com/facebookresearch/sam3 at `7f473cf`, 2026-10-14; no tags) asks for Python 3.12+, PyTorch 2.7+ and CUDA 12.6+. Building the image model needs a GPU: `position_encoding.py#L54-L56` and `decoder.py#L300-L304` allocate on `"cuda"` in `__init__`, and `edt.py` imports triton. `import sam3` also needs einops, psutil, pycocotools and `setuptools<82` (for `pkg_resources`), which `pyproject.toml` does not declare (issue #227). The PyPI wheel `sam3` 0.1.4 lacks the BPE vocab file. Its `Sam3Processor` returns boolean masks at the photo's size, xyxy pixel boxes and scores; the default confidence threshold is 0.5. Clicks go through `model.predict_inst` with `enable_inst_interactivity=True`, which ignores the text prompt (`sam3/model/sam3_image.py#L628-L665`). In our ICE check (2026-10-15), the image model failed in fp32 with "mat1 and mat2 must have the same dtype" in `vitdet.py` and ran under bf16 autocast, as in the notebooks. In an editable install, `pkg_resources` cannot find the BPE file, so `build_sam3_image_model` needs `bpe_path=<clone>/sam3/assets/bpe_simple_vocab_16e6.txt.gz`.
- HF `facebook/sam3` (gated, manual approval) holds both `sam3.pt` (3.45 GB) and transformers files: `config.json` (a `sam3_video` config), `model.safetensors` (3.44 GB, F32, 859.9M parameters) and the processor and CLIP tokenizer files (https://huggingface.co/api/models/facebook/sam3/tree/main). `Sam3Model` loads only the 840.4M detector parameters and ignores the tracker ones. `AutoModel` on this repo gives `Sam3VideoModel`, so the explicit classes are needed (`modeling_sam3.py#L2184-L2199` at v5.20.0).
- Transformers added SAM 3 in v5.0.0 (first stable release 2026-01-26); no 4.x release has it. v5.20.0 (2026-10-15) needs torch 2.6 or newer (`import_utils.py#L194-L200`) and huggingface-hub 1.31 or newer. Transformers lists torchvision only as an extra, but SAM 3's processor and model import it (`image_processing_sam3.py#L30`, `modeling_sam3.py#L807`). The image path has no custom CUDA ops.
- The processor resizes every photo to 1008 x 1008 without keeping the aspect ratio. The model predicts 200 queries with masks at 288 x 288. `post_process_instance_segmentation(outputs, threshold=0.3, mask_threshold=0.5, target_sizes)` keeps queries whose score, sigmoid(query logit) x sigmoid(presence logit), is above `threshold`; it returns unsorted scores, xyxy pixel boxes and 0/1 int64 masks resized to the target size, with no NMS. The docs use `threshold=0.5`. With no instance, the masks keep the shape (0, 288, 288) (`image_processing_sam3.py#L884-L934`). The tokenizer pads text to 32 tokens with no truncation, so longer prompts fail.
- With a text prompt alone, transformers skips the geometry encoder (`has_geometry_prompts` in `Sam3Model.forward`), so the prompt lacks the learned CLS token that Meta's code always appends through a dummy geometric prompt. Issue #49592 (opened 2026-10-14) reports this; the fix, PR #49614, is open and marked breaking. One padding box with label -10 runs the encoder: its mask hides the box, and the CLS token is appended (`modeling_sam3.py#L1301-L1306`, `#L2371-L2428`). On a tiny random model, moving the padding box changes no logit, while dropping the CLS token does. On real photos, the padding box brings the scores close to those of Meta's code (build plan note 3).
- `transformers.image_utils.load_image` applies the EXIF rotation, and a numpy array passed to the processor is used as it is.

### 4.6 Physical properties: VoMP v1.1 and simulator formats

Checked on 2026-10-14 against the sources named here. Our use is in build_plan.md note 9.

**VoMP v1.1** (https://github.com/nv-tlabs/VoMP/releases/tag/v1.1, commit 641a22d, 2026-10-13)
- It is the only commit after the physprop pin ac6826f. The release notes claim better results on noisy splats, no need for a specific object orientation, and more robust results on very soft objects. No source gives v1.1 accuracy numbers. The HF model card still describes v1.0, and arXiv 2510.22975 is still v2 (2026-03-02).
- The geometry transformer has the same architecture. It was retrained with random rotations of the 64^3 grid (one of 24), Gaussian noise (0.05) on the DINO features, an L2 loss and 164k steps (`configs/materials/v1.1/geometry_encoder/train.json`). The material VAE changed from TripletVAE with a flow to a StandardVAE.
- The normalization range is wider: E 20 Pa to 1.2e12 Pa, nu -0.19 to 0.5, rho 1.2 to 22,087 kg/m^3. v1.0 used E from 1e6 Pa, nu 0 to 0.49 and rho 102.6 to 19,800 (`vomp_1_1/` and `vomp_1_0/normalization_params.json` on HF). nu can reach 0.5, where the Lamé lambda diverges, so clamp it before FEM.
- Issue #6 (https://github.com/nv-tlabs/VoMP/issues/6, open) reports v1.0 predicting E 22 to 181 times too high on soft objects.
- The mesh path renders with the `bpy==4.2.0` pip module (Python 3.11 only) instead of a Blender 3.0.1 binary. It voxelizes the geometry that Blender imported (`renders/normalized_geometry.npz`), so the old axis mismatch between renders and voxels is gone. The renders are fixed at 150 views, 512 px and a 40 degree FOV, and they use Cycles on the GPU with no CPU fallback.
- `get_mesh_materials` takes .glb, .gltf, .obj, .ply, .stl and USD. It rejects materials with transmission or partial alpha. A GLB or OBJ gets one segment label for all its voxels, and every mesh object in the file is joined into one grid.
- Voxels come from surface cells plus a libigl winding-number fill (|w| > 0.5). Features are computed for at most 72,000 voxels, and prediction keeps at most 32,768.
- `save_materials` writes `voxel_data` with the fields x, y, z, youngs_modulus, poissons_ratio, density and segment_id. It does not keep `transform_center` and `transform_scale`.
- HF `nvidia/PhysicalAI-Simulation-VoMP-Model` now has `vomp_1_0/` and `vomp_1_1/` folders (3.44 GB in total, about 345 MB needed for v1.1 inference). The old root files are gone.

**Simulators**
- MuJoCo 3.16.0 mesh assets are STL, OBJ or MSH, and textures are PNG or KTX (https://mujoco.readthedocs.io/en/latest/XMLreference.html#asset-mesh). There is no glTF decoder (https://github.com/google-deepmind/mujoco/issues/2672). The OBJ decoder ignores the MTL file and builds faces from the first object only. A mesh geom collides as its convex hull. A rigid body takes one mass, COM and inertia (`<inertial pos mass fullinertia>`). `young` and `poisson` exist only on flex elasticity, as one value per flex.
- Isaac Sim 6.x cannot open a GLB as a USD layer; it converts it with omni.kit.asset_converter (drag and drop, or Content Browser "Import and Convert"; default 1 cm per unit and MDL materials unless "Meter as World Unit" and "UsdPreviewSurface" are checked). The result is visual only. A simulated object needs UsdPhysics RigidBodyAPI, CollisionAPI with a convexHull or convexDecomposition approximation, MassAPI and a physics material. MassAPI stores inertia diagonalized: diagonalInertia plus principalAxes. Without mass or density, PhysX uses 1000 kg/m^3 times the collider volume. https://docs.omniverse.nvidia.com/extensions/latest/ext_asset-converter.html, https://openusd.org/release/api/class_usd_physics_mass_a_p_i.html
- Isaac Sim's MJCF importer (isaacsim.asset.importer.mjcf) uses mujoco-usd-converter. Isaac Sim 6.2.0 bundles 0.6.1, which keeps explicit inertials, OBJ meshes with UVs and the diffuse PNG; 6.0.x bundles 0.2.0, which predates full-inertia support and two mass bugs fixed in 0.4.1. It ignores contype/conaffinity. https://docs.isaacsim.omniverse.nvidia.com/latest/overview/release_notes.html
- Isaac Sim / PhysX deformables take one material per body. Newton takes per-tet stiffness, warp.fem per-node fields and Kaolin Simplicits per-point E, nu and rho. VoMP ships examples for warp.fem, libuipc and Newton, and none for MuJoCo, PhysX or USD physics.
