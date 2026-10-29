# GATOR research notes

Sources behind `build_plan.md`; where the two files disagree, `build_plan.md` wins.

Tags: "(unconfirmed)" means no primary source confirmed the claim. "(our inference)" and "(our
design)" mark our own reasoning or choices. "(estimate)" marks a number we computed.

## 1. The paper

### 1.1 Source and numbering

The paper is "GATOR: Generative and Agentic 3D Object Reconstruction From Casual Images" (arXiv 2610.11215v1, HTML https://arxiv.org/html/2610.11215v1, PDF https://arxiv.org/pdf/2610.11215v1). The project page is https://research.nvidia.com/labs/lpr/gator/. The GitHub repository kasvii/GATOR is not this project. It belongs to an unrelated 2023 human-mesh paper.

Page references such as (p.5) are PDF pages. Table and figure numbers follow the PDF. The equations come from the MathML of the arXiv HTML, because the PDF text layer drops glyphs inside equations. Section 5.1 writes out Eq. 3 (5.1.2), Eq. 6 (5.1.6) and Eq. 4 (5.1.10). The metric code in section 4.5 implements Eq. 7 and Eq. 8. Section 1.3 gives Eq. 1, Eq. 2 and Eq. 5.

The arXiv HTML labels the qualitative ablation grid (PDF Fig. 5) as "Table 5". From that grid on, HTML table numbers are one higher than PDF numbers, and HTML figure numbers are one lower. For example, PDF Table 6 is HTML Table 7, and PDF Fig. 12 is HTML Fig. 11. The running text of the HTML is shifted the same way.

### 1.2 Inputs and frame

Each view v supplies an RGB image I_v, a target mask M_v, intrinsics K_v, a camera pose T_v and depth D_v. The text c is optional and can be a category name or a VLM caption (p.4).
```math
\mathcal{C}=(\{I_{v},M_{v},K_{v},T_{v},D_{v}\}_{v=1}^{N},c)
```

- The ScanNet++ evaluation feeds up to ten views (p.29), although training uses at most four (p.22). The paper does not explain why this works.
- The paper calls the structure latents "pose-aware occupancy latents" (p.5). It also says that training is pose-aligned (p.2).
- On re-alignment, the paper says that the anchor from partial depth can be wrong when observations are limited. So GATOR aligns the generated structure with the observations and recomputes the normalization before projection (p.19).

### 1.3 Generator equations

For patch p in view v, h^m_vp is the token of modality m, which is RGB (I), mask (M) or pointmap (P). The correspondence embedding a_vp = A(γ(u_p), r_vp) combines the Fourier-encoded patch coordinates γ(u_p) with the Plücker rays r_vp averaged over the patch (p.5).

Eq. 1 defines the modality mixer (p.5, p.18). The mixer outputs three tokens per patch, and the set of mixed tokens from all views is the bank B.
```math
\{\widetilde{\mathbf{h}}^{m}_{vp}\}_{m\in\{I,M,P\}}=\operatorname{ModalityMixer}\!\left(\{\mathbf{h}^{m}_{vp}+\mathbf{a}_{vp}\}_{m\in\{I,M,P\}}\right).
```

Eq. 2 gives the projected feature of an active coordinate q_j (p.5).
```math
\mathbf{f}_{j}=\frac{1}{|\mathcal{V}_{j}|}\sum_{v\in\mathcal{V}_{j}}[F_{v}^{D}(\pi_{v}(s_{g}\mathbf{q}_{j}+\mathbf{t}_{g}));F_{v}^{N}(\pi_{v}(s_{g}\mathbf{q}_{j}+\mathbf{t}_{g}))],
```

F^D_v is the frozen DINOv3 feature map of the target-masked image of view v, and F^N_v is its NAF-upsampled version. π_v projects into the cropped view v. V_j holds the views where the point lies in front of the camera and projects inside the crop (p.5, p.18).

Eq. 5 is the structure-flow loss (p.18).
```math
\mathcal{L}_{\mathrm{SS}}=\mathbb{E}\!\left[\|v_{\theta}(\mathbf{z}_{t},t;\mathcal{B},\mathbf{g}(c))-\mathbf{u}_{t}\|_{2}^{2}\right]
```

Here z_t and u_t follow Eq. 4, and g(c) is the adapted SigLIP2 text modulation. The geometry and appearance flows use the same objective on normalized shape latents and normalized PBR latents (p.18).

Fig. 2 marks frozen modules with a snowflake. SigLIP2, both DINOv3 encoders and T5Gemma2 carry one. The point encoder has none, so it is trained. The "DINOv3 + NAF" box has no snowflake drawn.

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

### 1.5 Training data

- S3 (B.1, p.23) draws on about 980,000 assets from collections "including" 17 named sources, so the list may be incomplete. In the paper's order, the sources are Objaverse-XL, TexVerse, MolmoSpaces, 3D-FUTURE, HSSD, Articraft, GRScenes-100 (cited through GRUtopia), ABO, OmniObject3D, InternScenes, Artiverse, PartNet-Mobility (cited through SAPIEN), DTC, GSO, NVIDIA SimReady, RoboCasa and ArtVIP.
- Meshes and materials are converted once into a common representation. Scene composition samples assets so that the whole pool is covered evenly. Rotations mix upright rotations with unrestricted 3D rotations. Objects are dropped near one another onto the ground or onto existing piles. Some scenes are settled with rigid-body simulation, and the other scenes hold freely posed objects that are only checked for collisions.
- Cameras vary in distance, elevation, roll and field of view. The lighting uses random light strength and direction and occasional sunlight. Objects keep their authored materials, so the renders match the cached PBR targets.
- Each view stores RGB, camera parameters, object identities, depth and visibility information. An object-only pass records depth and instance masks, so transparent materials cannot corrupt them. Opacity-aware rendering keeps the openings of alpha-cutout assets consistent with their geometry. Complete objects are voxelized after scene placement and cached as latents together with their placement transforms.
- Perturbations act on camera pose, intrinsics, depth, masks and image quality. Modality dropout changes which inputs are available. The targets always stay complete.
- S3 is combined with 3D-FRONT, InternScenes, SceneSmith and SAGE through one shared input and target format. So InternScenes is both an asset source and a scene source.

#### Table 5 (p.24): corpus of the structure model

Epoch items are scene-object instances. Share is each source's fraction of the epoch items. Reuse is epoch items per unique asset. Assets are counted within each source, so the total is not deduplicated across sources. The paper gives no corpus table for the geometry and appearance flows.

| Scene source | Scenes | Epoch items | Unique assets | Share (%) | Reuse |
|---|---:|---:|---:|---:|---:|
| S3 | 199,106 | 2,287,840 | 949,212 | 68.44 | 2.41x |
| 3D-FRONT | 10,506 | 63,624 | 9,454 | 1.90 | 6.73x |
| InternScenes | 23,402 | 248,302 | 83,915 | 7.43 | 2.96x |
| SceneSmith | 541 | 19,993 | 9,947 | 0.60 | 2.01x |
| SAGE | 9,999 | 723,117 | 492,743 | 21.63 | 1.47x |
| Total | 243,554 | 3,342,876 | 1,545,271 | 100.00 | 2.16x |

### 1.6 Benchmarks, baselines and metrics

Section 4 gives the data sources, the baselines and the metric implementations.

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

Bold means bold in the paper. [D] marks a dark green cell (best), and [L] marks a light green cell (second best). The GPT-6-Astra rows are gray in the paper and left out of the ranking. "n/a" marks a cell the paper leaves as a dash. Tables 4 and 7 are in section 1.8, and Table 5 is in section 1.5.

The text claims that GATOR cuts CD by 19.7% against Pixal3D-MV on Toys4K, roughly halves Pixal3D's CD on the tabletop data, and cuts CD by 23% against RecGen on ScanNet++. The rounded table entries allow a Toys4K cut from 9.5% to 26.1% and a ScanNet++ cut from 22.2% to 26.5%. The tabletop ratio is 0.009 to 0.018, or 0.50. So all three claims are consistent. The claim that GATOR leads all nine tabletop metrics counts the SSIM tie with Pixal3D and leaves out the gray GPT-6-Astra row.

Section 4.5 describes how Table 2 pools the per-dataset values of Table 6. The ReconViaGen F1 cell is an exception, because Table 2 gives 0.608 and the object-weighted mean of Table 6 is 0.612.

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

Generator

- The paper does not give the DINOv3 variant, layer or output normalization. It does not say whether the class and register tokens enter B or how the mask is fed to DINOv3. It does not give the widths and form of the projections and modality embeddings, or the number of tokens per view.
- The paper does not give the pointmap encoder's resolution, patch size, width, heads, MLP ratio or depth validity thresholds. It does not say whether SAM 3D's position embedding and "exp" point remap are kept. It does not give the Fourier frequencies, the normalization of u_p, the frame and scaling of the Plücker rays, or the architecture of A.
- The paper does not give the mixer's layers, heads, width, FFN, LayerNorm placement or gate form. The text implies that the mixer runs once before the denoiser with no timestep input, so B could be cached across the 12 steps. The paper does not say whether every structure block cross-attends to B or whether the TRELLIS.2 cross-attention weights are reused. It does not give the SigLIP2 variant or the adapter's depth, normalization and injection point.
- The paper does not give the T5Gemma2 size, the maximum text length, the adapter output width, or how the text tokens join the five DINOv3 tokens. It does not say which VLM wrote the captions, where the category labels for about 980K assets came from, or which prompt the structure flow sees in training.
- The paper does not give the NAF output resolution or the sampling convention. It does not say whether projection uses 64³ or 32³ coordinates, how intrinsics are adjusted for crops, or what background the masked images use. Fig. 2 shows black.
- The paper does not define the anchor's "center" and "extent", the outlier handling or the margin inside the unit cube. It does not say how the structure is aligned with the observations after generation, or whether it is re-voxelized. It does not say which world frame the training targets use, or whether training randomizes the world rotation. It does not give the placement refinement's renderer, losses, weights, optimizer, iterations, scale parameterization, point-alignment metric or occlusion test.
- The paper does not say whether B, g(c), G and f are dropped together or separately for CFG. It does not give the null condition beyond "contributes nothing" for SigLIP2 and the empty-prompt encoding for T5Gemma2. It does not give the CFG formula, whether TRELLIS.2's guidance interval, rescale and timestep shift are kept, or whether guidance 3 applies to all three flows.
- The paper does not give the decoder resolution, remeshing, UV and texture size, GLB export settings, number of samples or seeds.

Training

- The paper does not give the logit-normal parameters, AdamW betas, EMA, warmup, learning-rate schedule, training steps, per-GPU batch, gradient accumulation or hardware. It does not give the training budgets of the ablation models, the view-count distribution in training, the rates of modality dropout and input perturbation, the estimator that produced the training cameras and depth, or the training data of the geometry and appearance flows.
- The paper does not say which Pixal3D checkpoint (single-view or multi-view) initialized the flows, where the appearance flow's initialization came from, or how the new text tokens in G were added.
- For S3, the paper does not give the common asset representation, per-source counts, filtering, deduplication or license rules. It does not give the even-coverage algorithm, what the [0.6, 3.0] scale multiplies, the rotation mix, the settled and free-pose split, or the physics engine.
- For S3 rendering, the paper does not give the views per scene, camera ranges, resolution, Cycles settings, environment-map and ground-texture libraries, or sunlight probability. It does not give the stored data format, the visibility thresholds, the alpha threshold for opacity-aware rendering, or how the four scene datasets were rendered and their targets chosen.

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

## 4. SAM 3, datasets, baselines and metrics

### 4.1 SAM 3

- SAM 3 accepts short noun phrases, image exemplars, points, boxes or masks as prompts. For images, it outputs instance masks, xyxy boxes, scores and a semantic map. For video, it also outputs object IDs per frame.
- SAM 3.1 ("Object Multiplex", HF `facebook/sam3.1`, gated) speeds up multi-object video tracking. Its HF license tag shows only "other".
- SAM 3 uses the SAM License of Nov 19, 2025 (section 2.1), which has no non-commercial clause.
- In our web app, the category name can serve as both the SAM 3 prompt and GATOR's text condition (unconfirmed, our inference).
- Meta's code (github.com/facebookresearch/sam3 at `7f473cf`, 2026-10-14; no tags) asks for Python 3.12+, PyTorch 2.7+ and CUDA 12.6+. Building the image model needs a GPU: `position_encoding.py#L54-L56` and `decoder.py#L300-L304` allocate on `"cuda"` in `__init__`, and `edt.py` imports triton. `import sam3` also needs einops, psutil, pycocotools and `setuptools<82` (for `pkg_resources`), which `pyproject.toml` does not declare (issue #227). The PyPI wheel `sam3` 0.1.4 lacks the BPE vocab file. Its `Sam3Processor` returns boolean masks at the photo's size, xyxy pixel boxes and scores; the default confidence threshold is 0.5. Clicks go through `model.predict_inst` with `enable_inst_interactivity=True`, which ignores the text prompt (`sam3/model/sam3_image.py#L628-L665`). In our ICE check (2026-10-15), the image model failed in fp32 with "mat1 and mat2 must have the same dtype" in `vitdet.py` and ran under bf16 autocast, as in the notebooks. In an editable install, `pkg_resources` cannot find the BPE file, so `build_sam3_image_model` needs `bpe_path=<clone>/sam3/assets/bpe_simple_vocab_16e6.txt.gz`.
- HF `facebook/sam3` (gated, manual approval) holds both `sam3.pt` (3.45 GB) and transformers files: `config.json` (a `sam3_video` config), `model.safetensors` (3.44 GB, F32, 859.9M parameters) and the processor and CLIP tokenizer files (https://huggingface.co/api/models/facebook/sam3/tree/main). `Sam3Model` loads only the 840.4M detector parameters and ignores the tracker ones. `AutoModel` on this repo gives `Sam3VideoModel`, so the explicit classes are needed (`modeling_sam3.py#L2184-L2199` at v5.20.0).
- Transformers added SAM 3 in v5.0.0 (first stable release 2026-01-26); no 4.x release has it. v5.20.0 (2026-10-15) needs torch 2.6 or newer (`import_utils.py#L194-L200`) and huggingface-hub 1.31 or newer. Transformers lists torchvision only as an extra, but SAM 3's processor and model import it (`image_processing_sam3.py#L30`, `modeling_sam3.py#L807`). The image path has no custom CUDA ops.
- The processor resizes every photo to 1008 x 1008 without keeping the aspect ratio. The model predicts 200 queries with masks at 288 x 288. `post_process_instance_segmentation(outputs, threshold=0.3, mask_threshold=0.5, target_sizes)` keeps queries whose score, sigmoid(query logit) x sigmoid(presence logit), is above `threshold`; it returns unsorted scores, xyxy pixel boxes and 0/1 int64 masks resized to the target size, with no NMS. The docs use `threshold=0.5`. With no instance, the masks keep the shape (0, 288, 288) (`image_processing_sam3.py#L884-L934`). The tokenizer pads text to 32 tokens with no truncation, so longer prompts fail.
- With a text prompt alone, transformers skips the geometry encoder (`has_geometry_prompts` in `Sam3Model.forward`), so the prompt lacks the learned CLS token that Meta's code always appends through a dummy geometric prompt. Issue #49592 (opened 2026-10-14) reports this; the fix, PR #49614, is open and marked breaking. One padding box with label -10 runs the encoder: its mask hides the box, and the CLS token is appended (`modeling_sam3.py#L1301-L1306`, `#L2371-L2428`). On a tiny random model, moving the padding box changes no logit, while dropping the CLS token does. On real photos, the padding box brings the scores close to those of Meta's code (build plan note 3).
- `transformers.image_utils.load_image` applies the EXIF rotation, and a numpy array passed to the processor is used as it is.

### 4.2 Evaluation datasets

| Benchmark | Source | License | Access | Main files | Other files |
|---|---|---|---|---|---|
| Toys4K | https://github.com/rehg-lab/lowshot-shapebias/tree/main/toys4k | none stated, per-asset attribution files | Google Form | `toys4k_blend_files.zip` (size not stated) | OBJ, point-cloud and sample-render zips |
| LM-O | HF `bop-benchmark/lmo` | CC BY-SA 4.0 | open | `lmo_base.zip` 5.68 kB, `lmo_models.zip` 5.37 MB, `lmo_test_all.zip` 720 MB or `lmo_test_bop19.zip` 118 MB | `lmo_train.zip` 355 MB |
| HB | HF `bop-benchmark/hb` | CC0 1.0 | open | `hb_base` 6.05 kB, `hb_models.zip` 200 MB, `hb_val_primesense.zip` 2.35 GB, `hb_val_kinect.zip` 12.9 GB | test Primesense 7.27 GB (BOP19 0.17 GB), test Kinect 37.2 GB, `hb_train_pbr.zip` 23.7 GB |
| HANDAL (BOP) | HF `bop-benchmark/handal` | CC BY-SA 4.0 | open | `handal_base` 7.73 kB, `handal_models.zip` 62.5 MB, `handal_val.zip` 3.48 GB | test 26.5 GB (BOP24 0.56 GB), onboarding 33.5 GB dynamic and 16.8 GB static, PBR 74.5 GB |
| ScanNet++ v2 | https://scannetpp.mlsg.cit.tum.de/scannetpp/ | ScanNet++ Terms of Use, non-commercial | signed application | 49 scenes | see below |

- Section 1.6 gives the target and view counts and how they were inferred.
- The per-dataset recall counts in Table 6 add up to the Table 2 counts for every method (83, 112, 114, 120 and 129 of 131). This confirms the split of 8 + 68 + 55 targets. GATOR's 129/131 = 98.47% matches Table 2.

#### Toys4K

- Toys4K is by Stojanov et al., CVPR 2021 (arXiv 2101.07296). The Toys4K paper gives 4,179 objects, but some were removed from the public release. The official list `toys4k/renderer/jsons/toys_dict.json` has 4,000 objects in 105 categories, with at least 15 per category. The microwave and marker categories have 18 objects each.
- The form (https://forms.gle/w7Zf82umwaKxr9L7A) gives five zip files. They include `toys4k_blend_files.zip` (textured .blend files), `toys4k_obj_files.zip` (OBJ files without materials), `toys4k_point_clouds.zip` and sample renders. Each asset has its own attribution file, for example `airplane_010_attribution.txt`.
- TRELLIS-500K's `Toys4k.csv` has 3,229 rows in 105 categories, plus captions. This is the filtered size at aesthetic threshold 4.5. It has only 4 microwaves and 4 markers, so it caps a five-per-category list at 523. GATOR's 525 objects must therefore come from the full 4,000-object list.
- The TRELLIS v1 loader `dataset_toolkits/datasets/Toys4k.py` does no aesthetic filtering itself. It reads the CSV with SHA-256 checks and expects `toys4k_blend_files.zip` to be downloaded by hand. TRELLIS.2's `data_toolkit` lists Toys4k as a test set but has no `datasets/` module, so it must reuse this v1 module.
- GATOR takes five objects per category. They were chosen before any method ran, "spanning the range of within-category geometric complexity" (p.23). The paper gives no selection rule.
- GATOR renders four input views per object in Blender under studio lighting. The viewpoints are casual, with irregular spacing and varied elevation, field of view and roll. The six held-out evaluation views differ from all inputs (p.25).

#### LM-O (Linemod-Occluded, BOP)

- LM-O annotates eight objects (IDs 1, 5, 6, 8, 9, 10, 11 and 12) in test scene 2 (`000002`). The images are 640×480 (`bop_toolkit_lib/dataset_params.py`).
- The `test_all` split has 1,214 images and 9,038 instances. The BOP19 subset has 200 images and 1,445 targets, with image IDs from 3 to 1212 (arXiv 2009.07378).
- The BOP ground truth includes object models, 6D poses, modal masks and RGB-D. The models are scanned meshes with vertex color (unconfirmed). The LM-O test ground truth is public (unconfirmed).
- GATOR's LM-O camera poses come from "registration with depth and image correspondences" (p.27).

#### HB (HomebrewedDB, BOP)

- HB has 33 objects in 13 scenes. Val and test both use scenes 1 to 13. The Primesense images are 640×480, and the Kinect 2 images are 1920×1080.
- RecGen's evaluation uses the Kinect images. The paper does not say which sensor GATOR used.
- Val has 4,420 images (13 scenes × 340 frames). These are turntable sequences with 2 × 170 frames at 30° and 45° elevation. Test has 13,000 handheld images and 67,542 instances (HB paper, arXiv 1904.03167, p.4). The BOP page says the ground truth is public "only for training and validation images, not for test images." So GATOR's 68 targets most likely come from val.
- The count of 68 matches all object instances across the 13 scenes. The 67,542 test instances over 1,000 frames per scene give about 67.5 scene-object pairs (unconfirmed inference). The unique `obj_id` values per scene in `scene_gt.json`, summed, would confirm it.
- Scenes 10 and 11 hold the same objects as scene 5, with colored light and painted textures (HB paper p.5 and p.7). So the 68 targets cover at most 33 unique models.
- Val scene 1 has 3 objects (IDs 20, 28 and 33) in 340 frames. Its `scene_camera.json` holds only `cam_K` and `depth_scale`.
- GATOR's HB cameras come from the calibration markers (p.27). HB's ground truth was built from the markerboard trajectory (HB paper, section 3.5), so T_cam_world = T_cam_obj · T_world_obj⁻¹ from any ground-truth object gives the same cameras.

#### HANDAL

- BOP-HANDAL (BOP-H3, 2024) has 40 of the original 212 objects, in 7 categories. The categories are hammers, ladles, measuring cups, power drills, spatulas, strainers and whisks. Its CAD models were made by 3D artists.
- Val is scenes 1 to 10, with 2,208 images. Test is scenes 11 to 71, with 13,261 images and 74,771 instances, which is about 5.6 objects per image. The images are 1920×1440. The ground truth is public for train and val, but not for test. The BOP version has no depth.
- The original NVlabs release (https://github.com/NVlabs/HANDAL) has 212 objects in 17 categories. Its depth comes from NeRF (Instant-NGP), and its meshes are NeRF meshes, not CAD models. Its license is CC BY-NC-SA 4.0, and its test split has 51 objects, not 55.
- So GATOR most likely used BOP-HANDAL val (an inference). The paper normalizes by a "CAD diameter" (p.25), its figures show only BOP categories, the project page shows four multi-object HANDAL scenes, and the BOP test ground truth is not public.
- 55 targets over 10 scenes is 5.5 per scene, which is close to the test density of 5.6 per image. No public `scene_gt` for HANDAL val was found, so the per-scene counts are unconfirmed. The unique (scene, obj_id) pairs in the val `scene_gt.json` would confirm them.

#### ScanNet++

- ScanNet++ v2 (Dec 2024) has 1,006 scenes. It includes laser scans, 33 MP DSLR images, iPhone RGB-D, and 360° panoramas for 956 scenes. The DSLR download is 2 MP fisheye (`OPENCV_FISHEYE`), and the iPhone frames are 1920×1440.
- The ground truth includes laser point clouds, meshes, semantic and instance meshes, COLMAP poses, iPhone depth and anonymization masks. The per-object annotations are in `scans/segments_anno.json`.
- The download sizes are about 1.5 TB for the default download, 371 GB for DSLR at 2 MP, 9 TB for DSLR with 33 MP, 132 GB for meshes and semantics, 720 GB for point clouds, and 319 GB for panocam (https://scannetpp.mlsg.cit.tum.de/scannetpp/documentation). Downloading single scenes should be possible (unconfirmed).
- Access needs an account, an application, the signed Terms of Use and a personal token for the download script. The form needs handwritten signatures from both the requester and a PI or supervisor. Clause 1 of the terms allows "non-commercial research and educational purposes" and says "Commercial use is strictly prohibited." Clause 4 limits sharing to colleagues who also signed. Clause 7 also binds a for-profit employer.
- The six DP-Recon scenes are `acd69a1746`, `07ff1c45bb`, `116456116b`, `1b75758486`, `8f82c394d6` and `9859de300f`. They are DP-Recon's scan1 to scan6, in that order (https://github.com/DP-Recon/DP-Recon/issues/9). PhyRecon uses seven scenes with a different numbering, `acd69a1746`, `67d702f2e8`, `49a82360aa`, `07ff1c45bb`, `daffc70503`, `116456116b` and `8d563fc2cc` (https://github.com/PhyRecon/PhyRecon/issues/4). Do not mix the two lists.
- In GATOR, the six DP-Recon scenes hold 69 objects, and the other 43 scenes add 160 targets (p.7, p.23, p.29). GATOR uses 1 to 10 views per target, "chosen for target visibility and viewpoint diversity". Fig. 12 shows "all ten input views" of one scene (p.32).
- DP-Recon (arXiv 2503.14830) uses 10 input views per scene and renders instance masks from the ground-truth mesh with the "official rendering engine". It trains at 384×384 (`scannetpp_grid.conf`). Its author says that the original indices of the sparse views are lost. ShapeR and JRM (arXiv 2603.25985) follow DP-Recon's six scenes, and neither publishes the scene IDs or the 69-object list.
- DP-Recon's code is Apache-2.0 (https://github.com/DP-Recon/DP-Recon). Its preprocessed data is in https://drive.google.com/drive/folders/1VUo6egav6j4aBw_ADRFP4NEE8pgB4alq, which includes `scannetpp.zip` (517 MB). The script `eval/eval_object_recon.py` reads the per-object meshes from `data/scannetpp/GTmesh/scan{id}/obj_{id}.ply`. Counting those files should give the 69 objects (unconfirmed).
- None of DP-Recon, PhyRecon, ShapeR and JRM says whether the views are DSLR or iPhone frames. PhyRecon describes ScanNet++ as having "higher-quality images and more accurate camera poses", which suggests DSLR (unconfirmed). To check, compare the aspect ratio and intrinsics of the processed images. A 3:2 ratio means DSLR, and a 4:3 ratio means iPhone.

### 4.3 Training data sources

| Source | Where | Count | Format | License and access | Size |
|---|---|---|---|---|---|
| TexVerse | HF `YiboZhang2001/TexVerse` (2K/4K/8K) and `TexVerse-1K` | 858,669 models (1,659,097 resolution variants), 158,518 with PBR | GLB, textures of 1024 px or more | ODC-By card, not gated | 23.3 TB (full), 10.9 TB (1K) |
| Objaverse 1.0 | HF `allenai/objaverse`, `pip install objaverse` | 800K+ | GLB | ODC-By with per-object licenses, not gated | 8.9 TB |
| ABO | https://amazon-berkeley-objects.s3.amazonaws.com/index.html | 7,900+ products | glTF 2.0 with 4K PBR maps | CC BY 4.0, direct download, S3 or Athena | `abo-3dmodels.tar` 154 GB |
| OmniObject3D | https://github.com/omniobject3d/OmniObject3D | 6,000 objects in 190 categories (unconfirmed) | scanned OBJ with one JPG texture, not PBR | CC BY 4.0, OpenXLab CLI after signup, or Google Drive | 1.2 TB compressed |
| MolmoSpaces | HF `allenai/molmospaces` (arXiv 2602.11337) | 130k objects, 230k+ scenes | decimated and baked Objaverse objects, MJCF and Isaac USD | ODC-BY for the Objaverse part, CC BY 4.0 for the rest, not gated | about 13.2 TB |
| RoboCasa | https://github.com/robocasa/robocasa | 3,200+ objects (Objaverse and AI-generated) | probably MJCF (unconfirmed) | assets CC BY 4.0, download script | about 10 GB |
| Articraft-10K | HF `camvsl/Articraft-10K` (arXiv 2605.15187) | about 10K articulated objects in 245 categories | URDF from LLM-written programs, simple materials (unconfirmed) | CC BY 4.0 | about 1.1 GB |
| GSO | Gazebo Fuel, https://research.google/blog/scanned-objects-by-google-research-a-dataset-of-3d-scanned-common-household-items/ | 1,030 objects | OBJ with PNG albedo, not PBR | CC BY 4.0 | about 13 GB |
| ArtVIP | HF `X-Humanoid/ArtVIP` | 476 articulated objects (v1.3) | USD from professional modelers | Apache-2.0, not gated | about 15.8 GB |
| SAGE-10k | HF `nvidia/SAGE-10k` | 10,000 scenes, 565K objects generated with TRELLIS | scene JSON plus meshes, kits export GLB and USD | Apache-2.0, not gated | about 1.03 TB |
| HSSD (HSSD-200) | HF `hssd/hssd-models` | 18k+ models, 211 scenes, 6,670 pass TRELLIS.2's PBR filter | artist-made | CC BY-NC 4.0, gated with auto-approval | 38.5 GB |
| 3D-FRONT | Tianchi application (arXiv 2011.09127) | 18,968 rooms, 13,151 furniture models | JSON layouts that use 3D-FUTURE models | research only for 3D-FUTURE, the same terms for 3D-FRONT unconfirmed | not stated |
| 3D-FUTURE | Tianchi application (arXiv 2009.09633) | 9,992 furniture models | textured, no PBR | "scientific research purpose only" | not stated |

- We use 3D-FUTURE only through 3D-FRONT.
- TRELLIS.2's set of about 977K shape assets is close to GATOR's 980K, so it is probably the core of GATOR's asset pool (unconfirmed). TRELLIS.2 left out 3D-FUTURE because it has no PBR materials.

- The paper's other sources are not planned. They are InternScenes (https://github.com/InternRobotics/InternScenes, CC BY-NC-SA 4.0, about 2.46 TB), SceneSmith (HF `nepfaff/scenesmith-example-scenes`, Apache-2.0 except the CC BY-NC 4.0 NotGenerated subset), Objaverse-XL (HF `allenai/objaverse-xl`, ODC-By with per-object licenses), GRScenes-100 (HF `InternRobotics/GRScenes`, CC BY-NC-SA 4.0), Artiverse (https://github.com/3dlg-hcvc/artiverse, gated with manual approval), PartNet-Mobility (HF `sapien-sim/PartNetMobility`, non-commercial, gated with manual approval), DTC (https://www.projectaria.com/datasets/dtc/, CC BY-SA 4.0 plus a no-sale rule) and the NVIDIA SimReady packs (https://docs.omniverse.nvidia.com/usd/latest/usd_content_samples/downloadable_packs.html, exact license unconfirmed).
- The Objaverse 1.0 per-object licenses are 721K CC-BY, 52K CC-BY-NC-SA, 25K CC-BY-NC, 16K CC-BY-SA and 3.5K CC0. The Objaverse-XL GitHub subset clones repositories and fails often, so Objaverse 1.0, TexVerse-1K and ABO are the easier sources for a small build.
- The 3D-FUTURE agreement (https://terms.aliyun.com/legal-agreement/terms/suit_bu1_ali_cloud/suit_bu1_ali_cloud202004171628_60052.html) allows only scientific research (section 3.2). It forbids commercialization (3.3.1), redistribution (3.3.2), and use "for commercialization or providing external services" of the data, derivatives and results (3.3.3). The agreement names only 3D-FUTURE.
- A public web app might count as "providing external services" under those terms (unconfirmed reading). If the app goes public, its models should be trained without 3D-FRONT and 3D-FUTURE.

#### Reusable TRELLIS and TRELLIS.2 data

- TRELLIS v1 publishes only metadata, in HF `JeffreyXiang/TRELLIS-500K`. The metadata holds sha256, file_identifier, aesthetic score and captions, in about 506 MB of CSV under MIT. It covers 500,777 training assets, and 494,770 of them have captions. Microsoft releases no renders or latents.
- The TRELLIS v1 `dataset_toolkits` scripts download the assets, render 150 views, voxelize them and encode the latents (https://github.com/microsoft/TRELLIS/blob/main/DATASET.md). The TRELLIS.2 `data_toolkit/` supports ObjaverseXL, ABO, HSSD, TexVerse, Toys4k and SketchfabPicked.
- The TRELLIS.2 `render_cond.py` still calls `importlib.import_module('datasets.<name>')`, although the `datasets/` modules are missing (https://github.com/microsoft/TRELLIS.2/issues/83). A user workaround copies the TRELLIS v1 `HSSD.py` and adds an `__init__.py`.
- HF `weikaih/swh-trellis2-400k` (about 979 GB, license "other") gives TRELLIS.2 latents for 397,797 Software Heritage GitHub meshes, with shape512, shape1024, ss64 and pbr512 (309K assets), plus 16 renders per asset at 1024 px.
- HF `true3d/Trellis_512v_SF` (about 558 GB) and `simon123905/trellis500k-sketchfab-archives` (about 622 GB) mirror the raw TRELLIS-500K archives. Neither has a license field. HF `YiboZhang2001/G-buffer-TexVerse` (about 25 TB, CC0 on the card) has object-centric G-buffer renders of 351,847 TexVerse assets at 2K or 4K.
- Object-centric render sets cannot supply the cluttered scene renders. Reuse still saves asset curation, aesthetic and PBR filtering, captions, and mesh and PBR dumps.

### 4.4 Baselines

| Baseline | Paper | Code | Weights | License |
|---|---|---|---|---|
| ReconViaGen | Chang et al., ICLR 2026, arXiv 2510.23306 | https://github.com/GAP-LAB-CUHK-SZ/ReconViaGen | HF `Stable-X/trellis-vggt-v0-2` | MIT tag |
| MV-SAM3D | Li et al., arXiv 2603.11633 | https://github.com/devinli123/MV-SAM3D | gated HF `facebook/sam-3d-objects` | SAM License |
| RecGen | Zadaianchuk et al., ECCV 2026, arXiv 2604.27106 | https://github.com/TRI-ML/recgen | `scripts/download_pretrained.py` | TRI Non-Commercial License |
| ShapeR | Siddiqui et al., CVPR 2026, arXiv 2601.11514 | https://github.com/facebookresearch/ShapeR | HF `facebook/ShapeR`, downloads automatically | CC BY-NC 4.0 |
| SimFoundry | Ranawaka et al., arXiv 2606.28276 | https://github.com/NVlabs/SimFoundry | Hunyuan3D-2.1 and FoundationPose | Apache 2.0 for NVIDIA code, some components restricted |

RecGen3D (arXiv 2604.01479) is not a baseline. GATOR cites it only in related work (p.3).

- On Toys4K, every method gets the same four images and masks. GATOR, Pixal3D and GPT-6-Astra use the Pi3X cameras and depth. ReconViaGen uses its own geometry estimator, and TRELLIS.2 conditions on the images alone.

| Baseline | How the paper ran it |
|---|---|
| TRELLIS.2 + MultiDiffusion | 4 views fused, Toys4K only |
| Pixal3D SV and MV | MV checkpoint with the shared cameras, no placement |
| ReconViaGen | masked RGB with its own geometry estimator, no placement |
| MV-SAM3D | its depth estimator replaced by the shared pointmaps and cameras |
| RecGen | internal alignment, best two of four views |
| ShapeR | grayscale images, filtered points and a caption, untextured |
| SimFoundry | best single view, Hunyuan3D-2.1 plus FoundationPose |
| GPT-6-Astra from scratch | same agent setup, no starting mesh |

- **Pixal3D**
  - Pixal3D expects square images, one shared field of view, a centered principal point, cameras that look at the object center, and view 0 as a front view at (0, -d, 0) (section 5.2.2).
- **TRELLIS.2 with MultiDiffusion**
  - The TRELLIS.2 default `pipeline.json` pulls HF `facebook/dinov3-vitl16-pretrain-lvd1689m` (gated, DINOv3 License) and HF `briaai/RMBG-2.0` (gated, CC BY-NC 4.0). Commercial use of RMBG-2.0 needs a BRIA agreement. So a TRELLIS.2-based path is not commercially clean, although the code is MIT.
  - TRELLIS v1 `run_multi_image` in `trellis/pipelines/trellis_image_to_3d.py` defaults to `stochastic` mode, which uses one image per step in round-robin order. The `multidiffusion` mode must be passed explicitly. In that mode, each step runs the model once per image and averages the predictions. CFG applies inside `cfg_interval` with one negative condition, so each step inside the interval runs N + 1 forward passes.
  - In the arXiv HTML, a dagger on the TRELLIS.2 row of Table 1 marks MultiDiffusion, so GATOR most likely ported this averaging (`inject_sampler_multi_image`) to the three TRELLIS.2 flows (unconfirmed).
  - MultiDiffusion is by Bar-Tal et al., ICML 2023, arXiv 2302.08113. Its code (https://github.com/omerbt/MultiDiffusion) targets Stable Diffusion 2 base through the diffusers `StableDiffusionPanoramaPipeline` and states no license.
- **ReconViaGen**
  - ReconViaGen uses TRELLIS v1 as the generator and a LoRA-fine-tuned VGGT as the reconstructor. The reconstruction features condition the generation. The method refines camera poses with VGGT, image matching and PnP-RANSAC.
  - It takes unposed multi-view images, and the demo removes the background with BiRefNet. It outputs a GLB mesh or 3D Gaussians. With 16 images, v0.2 needs under 18 GB without refinement and under 24 GB with refinement.
  - The weights carry an MIT tag and are not gated. They derive from VGGT-1B, which is CC BY-NC 4.0, so the MIT tag may not cover them (unconfirmed).
  - The v0.5 branch (HF Space `Stable-X/ReconViaGen-v0.5`) still loads `trellis-vggt-v0-2` for the sparse structure stage. It uses `microsoft/TRELLIS.2-4B` for the shape and texture stages, which gives high-resolution meshes with PBR. The paper does not say which version it ran, and its textured output suggests v0.5 (unconfirmed).
- **MV-SAM3D**
  - MV-SAM3D treats multi-view fusion as MultiDiffusion in the 3D latent space, weighted by attention entropy and visibility. It adds collision and contact constraints for scenes with several objects. Section 2 covers its code, install, inputs and outputs.
- **RecGen**
  - RecGen ("Reconstruction by Generation") comes from the Toyota Research Institute. It is fine-tuned from TRELLIS-image-large, and it jointly generates shape, texture and pose from RGB-D.
  - It takes RGB, depth (a PNG in mm or a float array in meters), a mask and a 3×3 intrinsics matrix. Because it needs depth, the HANDAL and ScanNet++ runs use the VGGT-Omega depth.
  - The model is natively 2-view, with one anchor view plus second views. With N > 1 second views, the default `multidiffusion` mode averages the velocity predictions of the (anchor, view_i) pairs at each step. A `stochastic` mode is also available.
  - It outputs a textured mesh in the object frame and in the camera frame, optional Gaussians, a GLB, and a 4×4 pose in the anchor view's camera frame.
  - The training data is on S3. The repository supports pixi, Docker and Compose with CUDA 12.1 or 11.8, and the README states no VRAM figure. The license allows research, evaluation and personal use. Bundled third-party code keeps its own terms, and the 3DGS part is also non-commercial.
- **ShapeR**
  - ShapeR (Meta Reality Labs) is a rectified-flow transformer on VecSet latents. It is conditioned on sparse SLAM points, posed multi-view images and per-object VLM captions. It outputs metric, untextured meshes.
  - The code expects sequences that the Aria MPS pipeline and a 3D instance detector have already processed, saved as pickles. The quality preset uses 16 views and 50 steps. The balance preset uses 16 views and 25 steps, and it is the recommended one. The speed preset uses 4 views and 10 steps.
- **SimFoundry**
  - SimFoundry (NVIDIA and partners, v4 of 5 Aug 2026) turns a real video into a sim-ready digital twin plus "digital cousins". Its pipeline has 13 stages, from video processing to USD and OmniGibson export. The policy and data-generation code is "Coming Soon".
  - The pipeline uses SAM3, DINOv3, Depth Anything 3, Hunyuan3D-2, FoundationPose, RMBG-2.0 and Gemini through Vertex AI or the API. It needs about 250 GB of disk. The mesh stage (stage 7) needs about 29 GiB VRAM by default.
  - The SimFoundry README names Hunyuan3D-2, so GATOR's use of 2.1 is a swap.
  - Hunyuan3D-2.1 (HF `tencent/Hunyuan3D-2.1`, https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) turns one image into a mesh with PBR texture. It needs 10 GB VRAM for shape, 21 GB for texture and 29 GB for both. Its license is the Tencent Hunyuan 3D 2.1 Community License. The license territory excludes the EU, the UK and South Korea. Products with more than 1M monthly active users on the release date (June 13, 2025) need a separate license.
  - FoundationPose (https://github.com/NVlabs/FoundationPose) estimates 6D pose from a CAD model or reference images, with RGB-D input. Its weights are on Google Drive. The NVIDIA Source Code License allows use for "research or evaluation purposes only".

### 4.5 Metric implementations

#### Existing implementations

- RecGen's `recgen_eval` package is the only released reference implementation of ADD-SB, and it is the closest match to GATOR. Its license is the TRI Non-Commercial License. The HB and ArtVIP data must be downloaded separately.
  - `EVALUATION.md` (https://github.com/TRI-ML/recgen/blob/main/EVALUATION.md) defines ADD-SB as the bidirectional Chamfer distance, computed without ICP and normalized by the ground-truth diameter. It reports recall at 0.1 and 0.05. Its CD_norm adds an ICP step.
  - `recgen_eval/poseval6d/core/metrics.py` samples with `trimesh.sample.sample_surface(mesh, 99999)`, which is area-weighted and unseeded. The unseeded sampling causes about 1% jitter per sample. It computes `chamfer_dist = 0.5*(dists1.mean()+dists2.mean())` with a scipy cKDTree and multiplies it by 100 to give cm.
  - Its ICP is Open3D `registration_icp` with correspondence threshold 0.02 and initial transform `np.eye(4)`. The prediction is voxel-downsampled at 0.005. `with_scaling` is False by default.
  - `recgen_eval/poseval6d/evaluation/evaluator.py` computes `addss = chamfer_dist_no_icp / diameter_cm` and `addss_10 = float(addss < 0.10)`. It also computes `addss_05`, `addss_02` and `chamfer_normalized_icp_with_scale`. The diameter is `gt_mesh_metadata['diameter']` in meters, multiplied by 100. The formula 0.5 × (mean + mean) / D equals GATOR's Eq. 7. A failed sample gets "Chamfer = 20 cm, recall 0".
- ShapeR's `evaluation/eval.py` (https://github.com/facebookresearch/ShapeR/blob/main/evaluation/eval.py) is the closest match to GATOR's C.2.
  - It samples with trimesh `mesh.sample(n_points, return_index=True)`, with `n_points=100000` and face normals. It uses a cKDTree with unsquared distances and computes `chamfer = 0.5*(completeness+accuracy)`. NC uses `np.abs` of the dot product and averages the two directions with weights of 0.5 each.
  - It rescales both meshes by 2 × max(bounds).
  - Its F-score uses absolute thresholds in its own coordinates. It reports the value at index 10 of `linspace(1/144, 2, 1000)`, which is about 0.027, and it does no ICP.
- SAM 3D (arXiv 2511.16624v2, App. D.3.2) defines ADD-S with the same formula as GATOR's Eq. 7 and Eq. 8. SAM 3D and TRELLIS.2 ship no evaluation code, and Pixal3D shows no evaluation scripts at the top level.
- PyTorch3D's `knn_points` returns squared distances, so `chamfer_distance(norm=2)` gives squared L2. Its `iterative_closest_point` and `corresponding_points_alignment` (Umeyama) default to `estimate_scale=False`. Release 0.7.8 states PyTorch 2.1 to 2.4. Kaolin's `chamfer_distance` defaults to `squared=True` and adds the two direction means without halving. So neither gives GATOR's CD as it is.
- trimesh `sample_surface` weights faces by area by default and takes a `seed` argument (https://trimesh.org/trimesh.sample.html). A scipy cKDTree is fast enough for 100k × 100k points.

#### Diameter D

- The BOP `models_info.json` file holds the CAD diameter, which is "the largest distance between any pair of model vertices". The models are in mm. `calc_pts_diameter` in `bop_toolkit_lib/misc.py` computes it. RecGen reads the ground-truth diameters from files such as `HB/hb_models/models_eval/models_info.json`.
- The farthest pair of a point set always lies on its convex hull, so the exact diameter needs only the convex-hull vertices.
- RecGen's `recgen_eval/diameter.py` computes the diameter of the predicted mesh, not the reference. It randomly subsamples 5,000 hull vertices, and it falls back to 10,000 random raw vertices if the hull fails. So its result is random and can underestimate the true diameter.

#### Similarity fits

- Open3D's `TransformationEstimation.cpp` calls `Eigen::umeyama(source_mat, target_mat, with_scaling_)`. Eigen's `Umeyama.h` sets S(m−1) = −1 when det(U) · det(V) < 0, so R = U S Vᵀ is a proper rotation. The scale is c = (1 / src_var) · singularValues.dot(S), which is never negative because the flipped value is the smallest. This matches GATOR's proper similarity.
- evo's `umeyama_alignment(x, y, with_scale)` in `evo/core/geometry.py` applies the same sign fix. Its default is `with_scale=False`.
- With estimated cameras, the paper computes one similarity per scene from the cameras alone, and all objects and all methods in that scene share it (p.25, C.1).
- On HB and LM-O, outputs go back to the dataset frame through the inverse of input view 0's pose.
- On ScanNet++, ICP is fit only from the observed surface to the prediction. CD and F1 stay symmetric, so completed surfaces that the scan missed are penalized. ScanNet++ is scored on geometry only.
- One-directional scaled ICP tends to shrink the source onto part of the target (unconfirmed). To avoid this, gather correspondences in both directions in each iteration before the Umeyama solve.

#### Silhouette IoU and appearance metrics

- GATOR averages silhouette IoU over the matched cameras of each object, then across objects (C.2). On ScanNet++, IoU projects into all reference views (p.29).
- The appearance metrics (C.3) reuse the C.2 alignment and do no exposure or color fitting. Predictions keep their native materials. On Toys4K, the shared cameras are the six held-out views.
- Pixal3D, which GATOR cites for IoU, computes IoU between the rendered normal maps and the ground-truth normal maps, only over regions where both exist. A pure mask IoU is a reasonable reading of GATOR, but it is not an exact copy. On the tabletop benchmarks, the "matched cameras" are probably the input cameras (unconfirmed).
- nvdiffrast gives a mask as `rast[..., 3] > 0`: the triangle ID is offset by one, and the background is 0. nvdiffrast v0.4.0 (https://github.com/NVlabs/nvdiffrast, the tag that TRELLIS.2 `setup.sh` clones) builds at pip install time. It removed the OpenGL backend, so it is CUDA only.
- The paper does not name its evaluation renderer. GATOR renders the Toys4K inputs and the agent's comparisons in Blender (p.6, p.22, p.25), so the evaluation renderer is probably Blender too (unconfirmed inference).
- TRELLIS v1 `dataset_toolkits/blender_script/render.py` has a studio-light rig: a point light (1000 W), a large top area light (10000 W) and a bottom area light (1000 W). It uses a transparent film, Cycles at 128 samples, and normalization to a unit cube.
- TRELLIS.2's `trellis2/renderers/pbr_mesh_renderer.py` renders on the GPU. It uses nvdiffrast, split-sum environment-map shading and ACES tonemapping, and its background is black. Whether it has a background-color option is unconfirmed.
- The `lpips` package 0.1.4 (https://github.com/richzhang/PerceptualSimilarity) defaults to the `alex` network, which its authors call "best as a forward metric". They recommend `vgg` for backpropagation. Inputs must be in [−1, 1], and the current metric version is 0.1. The 3DGS `metrics.py` uses `net_type='vgg'`. GATOR, TRELLIS, TRELLIS.2 and Pixal3D state no backbone.
- scikit-image `structural_similarity(gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0, channel_axis=-1)` matches Wang et al.'s SSIM. The paper's PSNR is computed on the full image with the white background (inferred from p.25).

#### Pooling

- In Table 2, the ADD-SB mean, ADD-SB@0.1, CD, NC and F1 equal the object-weighted means of Table 6. The IoU, PSNR, SSIM and LPIPS columns do not. A HANDAL weight of about 0.87 per object reproduces them for every method. For example, the SimFoundry PSNR is 15.02 with object weights, and a HANDAL weight of 0.87 gives the reported 14.90.
- The 4-view GATOR row of Table 7 matches the pooled LM-O and HB "w/o agent" rows of Table 6 for ADD-SB (0.0143) and CD (0.0105). It does not match for IoU (0.914 pooled against 0.902) or PSNR (15.98 pooled against 15.86). Recall at 4 views is 76/76 in Table 6 but 149/152 in Table 7. So the second view selection changes the scores and causes 3 failures.

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

## 5. Deferred: the full GATOR generator

### 5.1 Design

None of this is built. The paths in the headings below are planned homes for the code, and none of them exists in the repo.

#### 5.1.1 Pipeline, training and build order

The full generator replaces MV-SAM3D in Stage A. It runs in five steps.

1. **Preprocessing (5.1.3).** SAM 3 finds the target's mask in each photo. Pi3X (for a few photos) or VGGT-Omega (for a long image sequence) estimates each photo's camera and depth in the frame of view 0. The masked depth points give the anchor, a center and a size. The anchor moves and scales the object into a unit cube, but it never rotates it. So the generator must output the object already facing the way it faces in the photos.
2. **Structure flow (5.1.4).** It decides which cells of a 64^3 voxel grid the object fills, including its hidden parts.
3. **Re-alignment (5.1.5).** It fits the generated structure to the observed depth points, recomputes the normalization and reduces the grid to 32^3.
4. **GatorProj (5.1.6), geometry and appearance flows (5.1.7).** GatorProj projects each occupied voxel into the photos and averages the image features there. The geometry flow generates fine shape detail, and the appearance flow generates PBR materials (base color, metallic, roughness, alpha).
5. **Decode and place (5.1.8).** The frozen TRELLIS.2 decoders produce a textured mesh. Placement refinement adjusts only translation and scale. The output is a GLB with the pose baked into its vertices.

The paper does not define the training frame, the anchor, re-alignment or placement. Our choices are in 5.1.2, 5.1.3, 5.1.5 and 5.1.8. Section 1.10 lists every detail the paper leaves open.

**Backbone.** The backbone is TRELLIS.2 plus Pixal3D. We rejected Hunyuan3D, because its VecSet latents have no 3D positions for the Eq. 2 projection. No newer open model replaces TRELLIS.2 (5.2.9). Pixal3D branch `master` is the TRELLIS.2 version (DINOv3 plus NAF) that matches Eq. 2 and Eq. 6. Branch `paper` is the Direct3D-S2 version. The README says `main`, but that branch does not exist, so `git clone -b main` fails.

**Training.** Each flow starts from released weights and is fine-tuned separately. The structure flow starts from TRELLIS.2. The geometry and appearance flows start from Pixal3D, a TRELLIS.2 variant that already projects image features onto voxels. The codecs and the image and text encoders stay frozen. The training data is synthetic. Our S3 builder (5.1.9) renders cluttered Blender scenes made from open 3D asset libraries. Each object's complete shape, in its scene pose, is the training target, even when the object is mostly hidden. Training mixes clean cameras and depth with perturbed and estimated ones, so the models tolerate estimator errors. We use far fewer assets than the paper.

**Compute.** PACE Phoenix H100 nodes train the models. The two 512 flows may also train on lunarL40S (5.1.10). The assistant cannot log in to PACE, so the assistant prepares the job files and the user submits them.

**Build order.** Each step replaces one part, so we can measure each change on its own.

1. v0: the released Pixal3D-MV weights with our preprocessing, GatorProj, re-alignment, placement and export. This needs no training.
2. v1: our trained structure flow.
3. v2: our trained geometry flow.
4. v3: our trained appearance flow. This is GATOR without the agent.

**Evaluation.** The full generator targets the GATOR rows of Tables 1 to 3, and the "GATOR w/o agent" and GATOR rows of Table 6. The analyses to repeat are the ablation on the 55 HANDAL targets (Table 4), the agent time budget study on 10 Toys4K objects (Fig. 6), the 2, 4 and 8 view study on LM-O and HB (Table 7), and a PyBullet stability test of about 15 objects from one ScanNet++ meeting room (section 1.8).

#### 5.1.2 Coordinate frames

The world frame W is defined in `build_plan.md` note 2.

- Cameras are OpenCV world-to-camera matrices. We re-base every estimator output to view 0.
- The anchor normalization is q = (x - t_g) / s_g, and x = s_g q + t_g maps back (Eq. 3). Size is metric only if the input depth is metric.
- The generation frame G is (x, y, z)_G = (x, z, -y)_W. This rotation maps the OpenCV camera convention (forward +z, up -y) to the Z-up convention that TRELLIS.2 and Pixal3D were trained on. This rule is our reading of the paper. The paper does not state it.
- TRELLIS.2's `to_glb` maps vertices (x, y, z) to (x, z, -y), a -90 degree rotation about X. It also rotates normals and flips the V texture coordinate. G equals the glTF frame, so the export step must skip that rotation or undo G before it calls `to_glb`. Otherwise the axis change happens twice.
- The final pose is baked into the GLB vertices. `placement.json` stores its translation, scale and frame.
- In training, the structure target is rebuilt every step, so its frame can change per sample. The geometry and appearance targets are encoded once, so their frame is fixed. We encode two frames per scene, taken from two rendered cameras. 5.3.1 gives the reasoning.
- Before any large data build, we check two things. The TRELLIS.2 512 codecs must reconstruct rotated shapes about as well as canonical shapes. The per-sample structure rebuild must match direct voxelization. If the codecs fail on rotated shapes, we switch to gravity-aligned frames (5.3.1).

#### 5.1.3 Inputs and preprocessing (`harpor/inference/`)

- Pi3X comes from github.com/yyfz/Pi3 and HF `yyfz233/Pi3X`. Its code is BSD-3, and its weights are CC BY-NC 4.0. It returns camera-to-world poses in OpenCV convention, so preprocessing inverts them.
- VGGT-Omega comes from github.com/facebookresearch/vggt-omega and HF `facebook/VGGT-Omega`. It outputs cameras and depth at relative scale. The paper does not say which checkpoint it used, so we run both `vggt_omega_1b_512.pt` and `vggt_omega_1b_416_reproduce.pt`. The second one needs `image_resolution=416`.
- Every view is cropped around the target with a margin of 1.1. The structure flow uses 384 px crops that keep the scene context. The geometry and appearance flows use 512 px crops with the background masked out.
- The anchor is the 2nd to 98th percentile box of the depth points inside the masks, times a margin m. The masks are eroded by 5 px first. Depth is valid where it is finite and positive. We pick m from a sweep of 1.0 to 1.5, as the smallest value that clips under 5% of training targets. 5.3.1 explains why the margin is needed.

#### 5.1.4 Observation tokens and structure flow (`harpor/models/structure_flow.py`)

The structure flow starts from TRELLIS.2 `ss_flow_img_dit_1_3B_64_bf16` (width 1536, 30 blocks, 12 heads, condition width 1024). Each block runs self-attention, then cross-attention to the condition, then the FFN. Each of the 16^3 latent cells is one token, so there are 4,096 tokens. The 8x16^3 latent decodes to the 64^3 occupancy.

Each crop is cut into patches on one shared grid. A 384 px crop with 16 px patches gives 24 x 24 = 576 patches. Each patch gets three tokens, so one view gives 1,728 tokens.

- RGB token: frozen DINOv3 features of the crop. DINOv3 outputs one class token, four register tokens, then the patch tokens. TRELLIS.2 applies a layer norm without weights to all tokens.
- Mask token: frozen DINOv3 features of the mask image. The RGB and mask features pass through separate projections, and each gets a modality embedding.
- Pointmap token: the depth unprojected into the anchor-normalized frame and embedded by our pointmap encoder.

The pointmap encoder follows the design of `PointPatchEmbed` in SAM 3D (5.2.7). We write it from this description and do not copy SAM 3D code, because of the SAM license.

1. Resize the pointmap (nearest neighbor) and mark invalid pixels.
2. Remap the coordinates linearly. SAM 3D's "exp" remap applies log1p(z), which gives NaN for normalized z at or below -1.
3. Apply `Linear(3, D)` per pixel. Zero the invalid pixels and add a learned invalid token.
4. Split into windows. Prepend a window class token and add a learned position embedding.
5. Run one transformer block (16 heads, MLP ratio 2) per window, and keep the window class token.
6. Add a learned patch-grid position embedding.

The window grid must match the DINOv3 grid (input 384 with patch 16, or 192 with patch 8). Keep the width at 256 or less, so the encoder does not cost more than the DiT.

Each patch also gets a correspondence embedding from Fourier-encoded patch coordinates and the Plücker ray of the patch, averaged over its pixels (5.2.8). There is no view-index embedding, and key-padding masks hide padded views. The modality mixer (Eq. 1) is residual attention over the three tokens of one patch, with no attention across patches or views. The structure latents then cross-attend to the bank of mixed tokens from all views.

Text enters through a pooled SigLIP2 embedding (1,152 dimensions) of a short category prompt. Text must be lowercased and padded to `max_length=64`. An adapter adds the embedding to the timestep embedding, which drives AdaLN in every block. A missing prompt adds nothing. The correspondence gates, the mixer gates and the SigLIP2 adapter output all start at zero, so the new paths start as identities.

Training target: we cache a 256^3 occupancy per placed object. Each step computes the anchor from the sampled views, rebuilds the 64^3 grid on the GPU, and runs the frozen structure encoder (5.3.1).

#### 5.1.5 Re-alignment (`harpor/inference/realign.py`)

- Decode the 64^3 occupancy and drop small floating parts.
- Fit the translation and scale of the structure to the observed points.
- Recompute the normalization from the occupied bounding box. Max-pool to a 32^3 grid for the 512 flows, as the TRELLIS.2 512 pipeline does. Training targets use the same normalization function.
- 5.3.2 gives the floater threshold, the fit and the re-voxelization in the new frame.

#### 5.1.6 GatorProj, Eq. 2 (`harpor/models/projection.py`)

- Each voxel center is mapped to the world with s_g q + t_g and projected into every view, with that view's real intrinsics and pose. The features are averaged over the views where the point is in front of the camera and inside the crop. A voxel with no valid view gets a zero feature.
- There is no occlusion test. Every voxel along a camera ray gets the feature of the pixel it projects to. The masked crops keep occluders and background out.
- Each feature is a bilinear DINOv3 sample concatenated with a NAF-upsampled DINOv3 sample, 2048 channels in total. We sample as Pixal3D does, with the half-pixel shift ((p + 0.5) / res) * 2 - 1 and `grid_sample(mode='bilinear', align_corners=False)`.
- NAF upsamples to 512 at image 512 for the geometry flow, with the un-normalized [0, 1] image as its guide. The appearance flow uses the same NAF target (our inference). Both flows use the same voxel coordinates, so the projected features are computed once per object and shared.
- NAF runs one view at a time, as Pixal3D does, because a 512 x 512 x 1024 feature map is 1 GiB per view in fp32. NAF loads through `torch.hub`, so offline jobs use a vendored copy (5.2.3).
- Eq. 6 adds the projected feature into every transformer block, between self-attention and the FFN: `x = x + CrossAttn(LN(x), G) + Linear(2048 -> 1536)(f)`. The weight shapes match Pixal3D's `SparseProjectAttention`, so Pixal3D's projection weights can start this layer.
- In training, the projection uses perturbed or estimated cameras. We also jitter the ground-truth structure in translation and scale to imitate re-alignment error.

#### 5.1.7 Geometry and appearance flows (`harpor/models/slat_flow.py`)

Both flows use the TRELLIS.2 512 codecs (sparse-conv VAEs, 32 latent channels on a 32^3 sparse grid). They take only masked RGB features, with no mask tokens, pointmap tokens or mixer.

- The geometry flow starts from Pixal3D `slat_flow_img2shape_dit_1_3B_512_bf16` or its `_mv` twin. Pixal3D trained both with DINOv3 input 512 px and NAF target 512. A short zero-shot validation test picks the single-view or the multi-view checkpoint (5.2.2).
- Pixal3D released no 512 texture checkpoint. The same validation test picks one of three starting points for the appearance flow:
  - Pixal3D texture-1024 MV (the `_mv` twin of `slat_flow_imgshape2tex_dit_1_3B_1024_bf16`), run on the 512 grid. Pixal3D trained it with DINOv3 input 1024 px and NAF target 1024. Apart from its 30 projection layers, it has the same parameters as TRELLIS.2 texture-512. It uses RoPE, so only the positions rescale.
  - TRELLIS.2 texture-512 plus the Pixal3D projection layers.
  - TRELLIS.2 texture-512 with new projection layers that start at zero.
- The appearance flow takes 64 input channels: 32 noise channels plus the 32-channel shape latent. So the shape conditions the texture by channel concatenation.
- Cross-attention sees G: five view-averaged DINOv3 global tokens (the class token and the four register tokens) plus the T5Gemma 2 caption tokens.
- T5Gemma 2: load the full model, keep `get_encoder().text_model`, and load text with `AutoProcessor` (5.2.6). The 1B encoder width of 1152 comes from public fine-tune configs, because the official config is gated. The adapter is a LayerNorm, a Linear layer from 1152 to 1024, and one 16-head transformer layer. A dropped prompt uses the encoding of the empty prompt. Training captions range from short category names to long descriptions.
- Both flows train on the ground-truth structure and use the predicted structure at inference. Their targets are the 512 shape and PBR latents of the complete placed object.
- Pixal3D's "bf16" checkpoints are FP32 safetensors, and its `finetune_from` loads them with `torch.load`. We convert the files or patch the loader. Saving fine-tuned weights has its own dtype trap (5.2.1, issue #126).

#### 5.1.8 Decode, export and placement (`harpor/inference/`)

- The TRELLIS.2 decoders output an O-Voxel grid. Meshes live in the cube [-0.5, 0.5]^3. Each voxel holds base color, metallic, roughness and alpha.
- `to_glb` fills holes, simplifies, unwraps UVs, bakes PBR into textures with nvdiffrast and inpaints the atlas. The inpainting can stall on the CPU, so we use the export settings in 5.3.2.
- Placement refinement optimizes only translation and log-scale (4 numbers) with Adam for about 200 iterations. Orientation and shape stay fixed. The losses are a soft silhouette IoU, a depth term, and a one-sided distance from the observed points to the mesh. Pixels hidden behind nearer objects get zero weight. The step keeps its result only if the multi-view IoU does not drop. 5.3.2 gives the parameterization, weights and schedule.

#### 5.1.9 S3 training data (`harpor/data/s3/`)

- Asset sources: TexVerse-1K (PBR subset), Objaverse 1.0, ABO, OmniObject3D, MolmoSpaces, RoboCasa, Articraft-10K, GSO and ArtVIP. SAGE-10k scenes, HSSD and 3D-FRONT are optional. HSSD is non-commercial, and 3D-FRONT probably is too. Section 4.3 gives sizes, licenses and access.
- InternScenes is left out. Some of its textures came from Hunyuan3D, and the Hunyuan license forbids using its outputs to train other models.
- Every evaluation object is removed from the training data, by sha256 and by DINOv3 similarity of 0.95 or more.
- Each scene holds 8 to 16 assets, with log-uniform scale in [0.6, 3.0]. Most assets stand upright. PyBullet settles part of the scenes, with CoACD collision shapes.
- The camera paths are orbits, traversals, whole-cluster and sub-cluster views, and human-height scans. Each scene gets 32 views at 1024 px, rendered with Blender Cycles and OptiX, a random environment map and a textured ground.
- Object masks and object-only depth come from rasterizing each object's mesh. Extra Blender passes per object would multiply the render time.
- A visibility index picks usable views per object from visible area and truncation, and keeps a small share of hard occlusions.
- Training observations mix clean, perturbed and estimated cameras and depth. The estimated runs (VGGT-Omega and Pi3X) are registered to the ground-truth cameras with a camera-only similarity transform. Runs with more than 5 degrees or 20% residual are rejected. Pointmap tokens are dropped in 10% of samples.
- Captions come from TRELLIS-500K (MIT) where they exist, and from a local vision-language model otherwise. SigLIP2 and T5Gemma 2 embeddings are cached once per caption.
- Published TRELLIS latents are all in canonical pose, so we cannot reuse them. Pixal3D's view-aligned latent encoders (`encode_shape_latent_view.py`, `encode_pbr_latent_view.py`, `encode_ss_latent_view.py`) are a template for our posed targets (5.3.1).
- The TRELLIS.2 `data_toolkit/` dumps meshes and PBR with Blender. It was written for Blender 3.0.1, so we test it on 5.2.2 first. Its per-dataset `datasets/` module is missing (issue #83). 5.2.1 gives the workarounds.
- The data is packed into WebDataset shards, because PACE limits each user to 1M files.

#### 5.1.10 Training (`harpor/trainers/`)

- All three flows use rectified flow with sigma_min = 1e-5 (Eq. 4). The noisy sample is z_t = (1 - t) z_0 + (sigma_min + (1 - sigma_min) t) eps, and the target velocity is (1 - sigma_min) eps - z_0. t = 1 is noise and t = 0 is data.
- Shared settings: AdamW (weight decay 0.01, betas 0.9 and 0.95), bf16, adaptive gradient clipping (max norm 1.0, 95th percentile), EMA 0.9999, condition dropout 0.1 (text tokens included), effective batch 128, 1 to 4 views, constant learning rate.
- Structure flow: learning rate 1e-4, logit-normal timesteps (mean 1, std 1). Geometry and appearance flows: learning rate 2e-5, uniform timesteps.
- TRELLIS.2's multi-image trainer (`MultiImageConditionedSparseFlowMatchingCFGTrainer`, 1 to 4 views) is the starting point for the sparse flows.
- When a 512 flow trains on an L40S, use activation checkpointing, `gradient_as_bucket_view=True` and the EMA off the GPU (5.3.3).
- TRELLIS.2 multi-GPU training crashes in its snapshot step. We set `num_workers=0` in the snapshot loader, turn snapshots off and evaluate offline (5.3.3).
- In TRELLIS.2, the `sdpa` backend covers only dense attention. Sparse attention needs flash-attn or xformers.
- Sampling uses 12 steps per flow with TRELLIS.2's `FlowEulerGuidanceIntervalSampler`. We read the paper's "guidance scale 3" as `guidance_strength=3.0` in TRELLIS.2's guidance formula. We keep the TRELLIS.2 interval and rescale defaults. 5.2.1 gives the formula and the defaults. The negative condition is all zeros.

#### 5.1.11 Runtime images

The components need conflicting library versions, so they run in separate Docker images. PACE runs the same images through Apptainer (`apptainer build x.sif docker-archive://x.tar`, then run with `--nv --cleanenv`).

| Image | Contents |
|---|---|
| `gator-gen` | CUDA 12.4.1, Python 3.10, torch 2.6.0+cu124, flash-attn 2.7.3, nvdiffrast v0.4.0, CuMesh, FlexGEMM, o-voxel, TRELLIS.2, Pixal3D, NAF, DINOv3 |
| `gator-est` | Python 3.12, torch 2.7+ with CUDA 12.6 (the minimums of Meta's `sam3` package): SAM 3, Pi3X, VGGT-Omega, SigLIP2, T5Gemma 2 |
| `gator-render` | Blender 5.2.2, Python 3.11, PyBullet 3.2.7, CoACD |

- NATTEN 0.21.0 has wheels only for torch 2.7, so `gator-gen` builds it from source with `NATTEN_CUDA_ARCH="8.9;9.0"` (5.2.3 gives the command). Build every extension with `TORCH_CUDA_ARCH_LIST="8.9;9.0"` for the L40S and the H100.
- `gator-gen` pins transformers 4.57.3, as Pixal3D does. The text encoders live in `gator-est`, because T5Gemma 2 needs transformers 5.0 or later. Transformers 5.4.0 and later break the DINOv3 loop of TRELLIS.2 and Pixal3D (5.2.4). If one image must hold both, use transformers 5.0 to 5.3.x, or patch the loop as in TRELLIS.2 PR #156.
- CuMesh fails to build with CUDA 12.8 unless `NVCC_APPEND_FLAGS="--extended-lambda"` is set.
- lunarL40S runs driver 555, which may not run the CUDA 12.6 image. If `gator-est` fails there, it runs on PACE.

### 5.2 Generator stack: TRELLIS.2, Pixal3D, NAF and encoders

#### 5.2.1 TRELLIS.2

##### Release, latents and checkpoints

- The paper is "Native and Compact Structured Latents for 3D Generation" (Xiang et al.), arXiv 2512.14692 (https://arxiv.org/abs/2512.14692), a CVPR 2026 oral. No text-conditioned or multi-image TRELLIS.2 weights exist. Issue #115 asks for text weights.
- Each active O-Voxel stores 6 material channels and shape features. The shape features are a dual vertex in [0, 1]^3, three edge-intersection flags for the X, Y and Z edges, and a splitting weight that controls how each quad splits into triangles. Converting a mesh to O-Voxel takes a few seconds on one CPU (a QEF solve on Hermite data). Converting back takes tens of milliseconds. `trellis2/pipelines/trellis2_image_to_3d.py` sets voxel_size = 1/resolution.
- The SC-VAE is a fully sparse-convolution U-Net with no attention. It downsamples 16x and has 32 latent channels, which gives the config name `f16c32`. The paper gives 354M parameters for the encoder and 474M for the decoder. A 512^3 asset gives about 2.2K tokens, and a 1024^3 asset gives about 9.6K tokens.
- Shape and material use separate VAEs. The texture decoder follows the subdivision structure of the shape decoder (`guide_subs`). The decoder class is `FlexiDualGridVaeDecoder` (`ckpts/shape_dec_next_dc_f16c32_fp16.json`), and `set_resolution` sets its resolution at run time.
- The flow checkpoints are `ckpts/ss_flow_img_dit_1_3B_64_bf16` (`SparseStructureFlowModel`), `ckpts/slat_flow_img2shape_dit_1_3B_{512,1024}_bf16` and `ckpts/slat_flow_imgshape2tex_dit_1_3B_{512,1024}_bf16` (`SLatFlowModel`). Each has about 1.29B parameters. The SC-VAE checkpoints are `ckpts/{shape,tex}_{enc,dec}_next_dc_f16c32_fp16`.
- TRELLIS.2 ships no structure VAE of its own. Its `pipeline.json` points the sparse structure decoder to the TRELLIS v1 decoder in `microsoft/TRELLIS-image-large`.
- The training configs use `ElasticSLatFlowModel` instead of `SLatFlowModel`. It is a subclass with a memory-elastic training mixin (`trellis2/models/structured_latent_flow.py`, line 202).

##### Structure, shape and texture stages

- The structure encoder config (`ss_enc_conv3d_16l8_fp16.json`) has in_channels 1, latent_channels 8 and channels [32, 128, 512]. It downsamples twice, from 64 to 16.
- `SparseStructureFlowModel` (`trellis2/models/sparse_structure_flow.py`) has mlp_ratio 5.3334, which gives an FFN width of 8192. It uses RoPE, a shared adaLN ("adaLN-single", `share_mod`) and QK RMSNorm in both self-attention and cross-attention. Its call is `forward(x, t, cond)`, and it asserts that x has shape [B, 8, 16, 16, 16].
- GATOR retrains the structure flow to output objects in the observation frame instead of the canonical asset frame (GATOR p5, p22 and p23). Keeping the condition width at 1024 lets the pretrained cross-attention K and V weights still apply (our inference).
- The pipeline max-pools the 64^3 decoder output to 32 for pipeline types `512`, `1024_cascade` (the default) and `1536_cascade`. It keeps 64 for `1024`.
- `data_toolkit/encode_ss_latent.py` does not voxelize the mesh. It reads the coordinates of a 1024-resolution shape latent (1024/16 = 64), sets those cells to 1 and encodes the grid. Its flag `--shape_latent_name` has default None, so the caller must pass it (the README passes `shape_enc_next_dc_f16c32_fp16_1024`). The other defaults are `--resolution 64` and `--enc_pretrained microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16`.
- The shape flow config for 512 (`configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16.json`) sets in and out channels 32 and resolution 32. The texture flow config (`configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16.json`) sets out_channels 32. The texture flow receives the normalized shape latent through `concat_cond`.
- The sparse flow call is `forward(x, t, cond, concat_cond=None)`. A `cond` given as a list is packed into a `VarLenTensor`, so the code already supports a different number of condition tokens per sample.
- The cascade samples a 512 latent first. It upsamples the coordinates with `shape_slat_decoder.upsample` and samples again at 1024 or 1536. If the token count passes `max_num_tokens` (49152), the pipeline lowers the resolution in steps of 128, but never below 1024. The default `1024_cascade` uses only the 1024 texture model.

##### Image conditioning

- `pipeline.json` sets `image_cond_model` to `DinoV3FeatureExtractor` with `facebook/dinov3-vitl16-pretrain-lvd1689m`.
- `DinoV3FeatureExtractor` (`trellis2/modules/image_feature_extractor.py`) loads `transformers.DINOv3ViTModel`, resizes the image to `image_size` and applies the ImageNet mean and std. It runs the embeddings, the RoPE module and every layer by hand in a loop over `self.model.layer`. It skips the model's final affine norm, returns every token and calls `.cuda()` directly.
- `get_cond(image, resolution)` gives the 512 models DINOv3 at 512 px and the 1024 models DINOv3 at 1024 px. The structure flow always gets the 512 px features (`cond_512`).
- Inputs are background-removed with RMBG-2.0 (BiRefNet), cropped square around the alpha bounding box and premultiplied onto black.

##### Sampling and CFG

These are the `pipeline.json` defaults for the 12-step `FlowEulerGuidanceIntervalSampler`. Pixal3D's `pipeline.json` has the same values.

| Stage | Guidance strength | Rescale | Interval | rescale_t |
|---|---|---|---|---|
| Structure | 7.5 | 0.7 | [0.6, 1.0] | 5.0 |
| Shape | 7.5 | 0.5 | [0.6, 1.0] | 3.0 |
| Texture | 1.0 (CFG off) | 0.0 | [0.6, 0.9] | 3.0 |

- TRELLIS.2 and Pixal3D compute w·pos + (1 - w)·neg (`trellis2/pipelines/samplers/classifier_free_guidance_mixin.py` and `pixal3d/pipelines/samplers`). TRELLIS v1 computes (1 + w)·pos - w·neg (`trellis/pipelines/samplers/classifier_free_guidance_mixin.py`). So TRELLIS.2 w = 3 equals v1 w = 2, and a v1 "scale 3" equals TRELLIS.2 w = 4. Any MultiDiffusion port must state which formula it uses.
- In TRELLIS.2, w = 1 turns CFG off and skips the negative pass. Outside `guidance_interval` the sampler forces w = 1. The optional rescale is based on the standard deviation of x0. With the default intervals, CFG is active on 10, 9 and 6 of the 12 steps for structure, shape and texture (our count from `flow_euler.py`).
- GATOR p22 lists 12 steps and "guidance scale 3" for the structure flow and for the geometry and appearance flows. Under our reading (5.1.10), this turns CFG on for the texture stage. An ablation with interval [0, 1] and rescale 0 is a useful check (our inference).

##### Multi-image support

- Official inference takes one image. `run()` takes one `Image.Image`, and there is no `run_multi_image`. Issues #10, #77 and #103 ask for multi-image input, and no maintainer has replied.
- `MultiImageConditionedSparseFlowMatchingCFGTrainer` is defined in `trellis2/trainers/flow_matching/sparse_flow_matching.py` at line 288. `MultiImageConditionedMixin` in `trellis2/datasets/components.py` draws 1 to `max_image_cond_view` views per asset (default 4). `encode_images` in `trellis2/trainers/flow_matching/mixins/image_conditioned.py` joins the DINOv3 tokens of all views into one variable-length condition, with a zero negative of shape [1, C] per sample. No released checkpoint was trained this way, and there is no multi-view trainer for the dense structure flow.
- The dense attention in `full_attn.py` has no mask argument. GATOR's padded, variable-view token bank for the structure flow therefore needs new key-padding mask code.
- TRELLIS v1 `run_multi_image(mode='multidiffusion')` averages the conditional velocity over all views at each step. It then applies CFG once with one negative condition (https://raw.githubusercontent.com/microsoft/TRELLIS/main/trellis/pipelines/trellis_image_to_3d.py).
- Community ports by OpsiClear and Allen-Zhou729 are reported deleted or broken. One user found the multidiffusion mode "incorrect" while the stochastic mode worked. We write our own port for the TRELLIS.2 + MultiDiffusion baseline. The port wraps `_inference_model` of each sampler and should be about 50 lines (unconfirmed estimate).

##### Decode and GLB export

- `decode_latent` returns a `MeshWithVoxel`, which is a mesh plus sparse PBR voxels. Base color is in channels 0 to 2, metallic in 3, roughness in 4 and alpha in 5. It also calls `fill_holes`.
- Besides the steps in 5.1.8, `o_voxel.postprocess.to_glb(...)` (`o-voxel/o_voxel/postprocess.py`) can remesh with narrow-band dual contouring. Its UV unwrap uses CuMesh, which is based on xatlas. After rasterizing in UV space, it snaps texels onto the original surface with a BVH and samples the voxel PBR grid trilinearly. It inpaints with `cv2.INPAINT_TELEA` and packs the metallicRoughness texture as [0, roughness, metallic] in RGB.
- The output is a `trimesh.Trimesh`, and alpha_mode is hard-coded to `OPAQUE`. The vertex rotation is at line 313. It has determinant +1, so it is a rotation and not a mirror.
- The README example uses a 1M face target, 4096 px textures, `remesh=True` and webp textures. Issues #18 and #53 report CPU stalls in `cv2.inpaint`, webp export and simplification.
- Section 3.3 of the nvdiffrast license ("Nvidia Source Code License (1-Way Commercial)") limits use to non-commercial research or evaluation. `to_glb` imports nvdiffrast, so the stock export path is non-commercial.
- The README reports about 3 s, 17 s and 60 s on an H100 at 512^3, 1024^3 and 1536^3, and it asks for at least 24 GB of GPU memory. `low_vram=True` is the default and moves each sub-model to the GPU only while it runs. Users report peaks of 30 to 63 GB, mostly with `low_vram` off or in Gradio (issues #11, #24 and #55).

##### Training code and data toolkit

- The repo has `train.py`, SC-VAE configs (`*_f16c32_fp16` and `ft_512`) and flow configs. The flow configs cover the structure flow (`ss_flow_img_dit_1_3B_64_bf16`) and the 512 shape and texture flows, plus `_ft1024` variants.
- `trellis2/trainers/__init__.py` lists `ImageConditionedFlowMatchingCFGTrainer`, `ImageConditionedSparseFlowMatchingCFGTrainer`, `MultiImageConditionedSparseFlowMatchingCFGTrainer`, `TextConditionedFlowMatchingCFGTrainer` and `TextConditionedSparseFlowMatchingCFGTrainer`. The text trainers use CLIP ViT-L/14 text with 77 tokens. A `SparseStructureVaeTrainer` class exists, but there is no structure-VAE config.
- The structure flow config uses DINOv3 at image_size 512, logitNormal(1, 1) timesteps, lr 1e-4, and the optimizer, clipping, EMA and `p_uncond` settings of 5.1.10.
- The shape and texture configs use max_tokens 8192, per-channel latent mean and std normalization, and uniform timesteps. The paper says flow timesteps come from logitNorm(1, 1), so the paper and the released configs disagree. GATOR's geometry and appearance flows reuse the released 512 latent mean and std.
- The configs set max_steps to 1M. One user who trained from scratch on about 700K meshes reports that the loss plateaus after about 100K steps (issue #157).
- The paper trained on Objaverse-XL, ABO, HSSD and TexVerse, after dropping assets with an aesthetic score below 4.5. That left about 977K shape assets and 738K material assets. It rendered 16 Blender views per asset with a FoV range of 10 to 70 degrees. The VAEs trained at batch 128 and the flows at batch 256.
- A fine-tuned state dict cast entirely to bf16 breaks inference with empty coords. The fix is to load the official model, call `load_state_dict(strict=False)` and then save (issue #126).
- Multi-GPU training crashes with "terminate called without an active exception" or "illegal memory access" (issues #88, #90, #95 and #164). The workaround is `num_workers=0` in `snapshot_dataset` and `run_snapshot`. The bug is still open upstream.
- The `data_toolkit/` scripts are `build_metadata`, `download`, `dump_mesh`, `dump_pbr`, `asset_stats`, `dual_grid`, `voxelize_pbr`, `encode_shape_latent`, `encode_pbr_latent`, `encode_ss_latent` and `render_cond` (16 views). The missing `datasets/` module raises a `ModuleNotFoundError`. One workaround copies the dataset files from TRELLIS v1. Pixal3D ships `data_toolkit/datasets/{ABO,ObjaverseXL,TexVerse}.py`, but it has no HSSD file.

##### Build notes

- The official setup is `setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm`. It creates a conda env with Python 3.10, torch 2.6.0 + cu124 and torchvision 0.21.0. It installs flash-attn 2.7.3, nvdiffrast tag v0.4.0, nvdiffrec from JeffreyXiang (branch `renderutils`), CuMesh and FlexGEMM from JeffreyXiang at unpinned HEAD, o-voxel from the local folder and utils3d pinned to a commit. It leaves transformers unpinned and installs gradio 6.0.1, pillow-simd, kornia, timm and lpips.
- The setup does not set `TORCH_CUDA_ARCH_LIST`. CUDA 12.4 is recommended, and only Linux is supported. There is no official Dockerfile. A community image exists (camenduru TostUI, issue #31).
- Sparse convolution defaults to `flex_gemm` (`trellis2/modules/sparse/config.py`), which needs Triton 3.2 or later and torch 2.4 or later. `SPARSE_CONV_BACKEND` also accepts spconv and torchsparse, but one user got a fragmented mesh with spconv (issue #23).
- Attention defaults to flash_attn. `SPARSE_ATTN_BACKEND` accepts xformers, flash_attn and flash_attn_3. `ATTN_BACKEND` also accepts sdpa and naive. flash_attn_3 works only on Hopper GPUs.
- CuMesh fails to build with CUDA 12.8 because of cubvh lambdas (issue #106). Section 5.1.11 gives the fix.
- One report says the v1 structure decoder gives all-negative logits (issue #160). This is probably a setup error, because the default pipeline works for many users (unconfirmed).
- The README lists the A100 and the H100 (sm_90) as verified. Ada GPUs (sm_89) work in practice, and users run TRELLIS.2 on the RTX 4090 (issues #31 and #106).
- FlexGEMM's Triton autotuning adds latency on the first run, so the Triton cache should live on persistent storage (our inference).

#### 5.2.2 Pixal3D

##### Paper, repository and checkpoints

- The paper is "Pixal3D: Pixel-Aligned 3D Generation from Images" (Li et al., Tsinghua, Tencent ARC and Victoria University of Wellington), arXiv 2605.10922, SIGGRAPH 2026. GATOR cites it as "Li et al., 2026b".
- The repo branches are `master`, `paper` and `pr-12`. Code references below are to `master` commit f7cf38429b.
- The `NOTICE` file lists dinov2 (Apache-2.0) and TRELLIS.2, Direct3D-S2 and MoGe (MIT). It does not list DINOv3 or NAF. Issue #33 asks about commercial use of the weights and has no maintainer reply.
- The `paper` branch uses DINOv2 ViT-L/14 with registers at 518 px, with optional NAF upsampling to 518x518. Its weights are `TencentARC/Pixal3D-D` (folders `dense/`, `sparse512/`, `sparse1024/`), which has no LICENSE file and states MIT only in its card metadata. It adds the feature volume to the noise volume once, at the input (paper Sec. 3.2.2). The `master` version injects the projected features in every block. The `paper` branch reproduces the Pixal3D paper numbers, but it does not use the TRELLIS.2 latents that GATOR needs.
- The `master` weights are `TencentARC/Pixal3D` (the old name "Pixal3D-T" redirects there). We read HF revision b0cb2e1b. `ckpts/` holds 22 files, about 46 GB.
- In issue #19 the maintainer said that only the flow models were retrained and the VAE was kept. In issue #30 the maintainer said that the TRELLIS.2 version uses the TRELLIS.2 training datasets. `configs/gen/ss_flow_img_dit_1_3B_32_bf16_proj_finetune_ft64.json` sets `pretrained_ss_dec` to `microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16`.
- The flow files `ss_flow_img_dit_1_3B_64_bf16`, `slat_flow_img2shape_dit_1_3B_{512,1024}_bf16` and `slat_flow_imgshape2tex_dit_1_3B_1024_bf16` each have a `_mv` twin. The files are stored as F32 despite the "bf16" name (5.1.7), and their JSONs set `"dtype": "bfloat16"` for inference. The structure file also holds one C64 RoPE buffer. The structure flow has 1.34B parameters, and the others have 1.39B.
- The decoders `shape_dec_next_dc_f16c32_fp16` (0.474B), `tex_dec_next_dc_f16c32_fp16` and `ss_dec_conv3d_16l8_fp16` are F16 and have the same sha256 as the TRELLIS.2 and TRELLIS-image-large files. The SV and MV flows share them.
- The only architectural difference from TRELLIS.2 is one `proj_linear` in each of the 30 blocks. Pixal3D texture-1024 minus 30 x (1536 x 2048 + 1536) parameters gives exactly the parameter count of TRELLIS.2 texture-512. Pixal3D released no 256 checkpoints.
- The SV and MV JSONs are the same, and they add `"image_attn_mode": "proj"` to the TRELLIS.2 flow architecture. The shape-512 JSON sets resolution 32 and in_channels 32. The 1024 JSONs set resolution 64. The texture JSON sets in_channels 64 and out_channels 32. The shape and texture JSONs set `proj_in_channels: 2048`. The structure JSON sets none, so its `proj_linear` is [1536, 1024] and takes DINOv3 features only.
- In May 2026 the maintainer wrote that the released checkpoint "has not been specifically fine-tuned for multi-view inputs" (issue #9). The MV checkpoints came later, and no document describes how they were trained.

##### Loading Pixal3D weights

- Pixal3D state dict keys are `blocks.N.cross_attn.cross_attn_block.*` and `blocks.N.cross_attn.proj_linear.*`. TRELLIS.2 keys are `blocks.N.cross_attn.*`. `_remap_checkpoint_keys` maps `cross_attn.*` to `cross_attn.cross_attn_block.*` (`pixal3d/trainers/basic.py` L446).
- If `proj_linear` is missing, `_fill_denoiser_proj_linear_from_image_cond` raises a RuntimeError unless an image_cond checkpoint is supplied (`image_conditioned_proj.py` L1133 to L1248). An old [1536, 1024] `proj_linear` is zero-padded to [1536, 2048]. Starting with zero-initialized projection layers needs a small patch that fills zeros instead of raising.
- In `proj` mode, `proj_linear` gets a "scaled" normal init, not a zero init (Pixal3D `structured_latent_flow.py` L145 to L152). Only the `gated_proj` mode starts at zero. It concatenates DINOv3 features with VAE color features, and no released checkpoint uses it.

##### Feature extraction and projection

The code is in `pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py`: `DinoV3ProjFeatureExtractor` at L440, `ProjGrid` at L142 to L232, the projection math at L27 to L108 and the MV subclass `DinoV3ProjMultiViewFeatureExtractor` from L707.

1. The DINOv3 pass matches TRELLIS.2 (L546 to L558). The patch tokens become [B, S/16, S/16, 1024], and the CLS token plus 4 registers form the global context (L657).
2. `ProjGrid` builds R^3 points with `torch.linspace(-1, 1, R)`, so index i maps to -1 + 2i/(R-1), and the points are not voxel centers. The grid is rotated to the Blender Z-up frame and scaled by 1/(2·mesh_scale), so the object box is [-0.5, 0.5]^3 / mesh_scale.
3. The camera is a square pinhole with a centered principal point and one field of view, `camera_angle_x`. The focal length in pixels is (16 / tan(fov/2)) · S / 32 (the Blender 32 mm sensor). Depth is -z_cam. Pixel coordinates are x = f·x/(-z) + S/2 and y = -f·y/(-z) + S/2. In the single-view case, the camera sits at (0, -d, 0) and looks along +Y.
4. Sampling uses the half-pixel normalization at L222 and `F.grid_sample` at L130 to L133 with `padding_mode='border'`. With NAF on, the [B, 1024, S/16, S/16] map is upsampled to `naf_target_size` and sampled again (L598 to L655).
5. A `valid_mask` (inside the image and depth > 0) is computed but never used. Points outside the image get clamped border features, and points behind the camera are not zeroed.
6. For the shape and texture stages, the dense R^3 output is indexed at the sparse coordinates of the current structure (`encode_image_proj` at L1550 to L1569, and `get_proj_cond_shape` in `pixal3d/pipelines/pixal3d_image_to_3d.py`).
7. The CFG negative is zeros for both the global and the projected inputs (L1583).

The MV extractor (`multiview_fusion="average"`, L392 to L437 and L707 to L846) projects one view at a time, sums the features and divides by V, with no validity weighting. It also averages the 5 global tokens, so the denoiser input shapes are the same for SV and MV. A comment at L803 to L809 notes that a dense [R^3, 2048] volume costs 2 GiB in fp32 at R = 64.

The features enter every block through `ModulatedSparseTransformerCrossBlock` (`pixal3d/modules/sparse/transformer/modulated.py`, L186 to L198) and `SparseProjectAttention` (`pixal3d/modules/sparse/attention/proj_attention.py`).

| GATOR term | Pixal3D `master` code |
|---|---|
| F^D and F^N | `z_proj_lr` (32x32 DINOv3 patch map) and `z_proj_hr` (NAF map), joined by `torch.cat(..., dim=-1)` |
| G | `z_global` (`context['global']`), averaged over V in MV mode |
| LN_l | `self.norm2`, `LayerNorm32(elementwise_affine=True)` |
| CrossAttn_l | the `SparseMultiHeadAttention` cross block inside `SparseProjectAttention` |
| W_l f_j + b_l | `SparseProjectAttention.proj_linear` applied to `context['proj']` |
| crop margin 1.1 | `preprocess_image` uses `size * 1.1` |

- GATOR does not say whether q_j is a voxel center. Pixal3D samples a corner-inclusive grid.
- GATOR's structure flow uses no projected features. Pixal3D's structure flow uses `proj` mode.
- Our projector replaces `ProjGrid` and `ProjGridMV` with a per-view pinhole that uses the intrinsics after cropping. It samples at the active sparse coordinates instead of building the dense volume, which would cost about 0.25 GiB per view at R = 32 (our arithmetic). It reuses `SparseProjectAttention`, `ModulatedSparseTransformerCrossBlock`, `DinoV3ProjFeatureExtractor` and `ImageConditionedProjMixin`.

##### Inference settings and cameras

`pipeline.json` and `pipeline_mv.json` name `DinoV3FeatureExtractor`, but the pipeline ignores that entry and sets `image_cond_model_*` to None (`pixal3d_image_to_3d.py`, around L117 to L121). The real settings are the `IMAGE_COND_CONFIGS` dicts in `inference.py`, `app.py` and `inference_mv.py`.

| Stage (weights) | DINOv3 image size | Projection grid | NAF target |
|---|---|---|---|
| structure (`ss_flow_img_dit_1_3B_64_bf16`) | 512 | 16 | NAF off |
| shape 512 (`slat_flow_img2shape_dit_1_3B_512_bf16`) | 512 | 32 | 512 |
| shape 1024 (`slat_flow_img2shape_dit_1_3B_1024_bf16`) | 1024 | 64, or res/16 (64 to 96) in the 1536 cascade | 512 |
| texture 1024 (`slat_flow_imgshape2tex_dit_1_3B_1024_bf16`) | 1024 | 64, with the same override | 1024 |

- The NAF target 128 is only the constructor default for `naf_target_size=None` (L473 to L474) and a stale `_comment` in the texture-256 config. No released config or script uses 128.
- The MV script uses `multiview_fusion="average"` for all stages. `--num_views N` keeps the first N views.
- For a single view, `inference.py` runs MoGe-2 (`Ruicheng/moge-2-vitl`, MIT) on the crop and converts the predicted focal length to a horizontal FOV (`get_camera_params_wild_moge`). `distance_from_fov` then puts the world point (-0.5, 0, 0)/mesh_scale on pixel column 0. The default `mesh_scale` is 1.0. A manual `--fov` flag skips MoGe-2.
- The paper (Sec. 3.2.2) instead sets the distance so that the four image-corner rays pass through the four back-face vertices of the unit cube. We did not check the `paper` branch code.
- Single-view preprocessing (`Pixal3DImageTo3DPipeline.preprocess_image`) runs BiRefNet only if the image has no alpha. It crops a square around the alpha bounding box and composites the crop onto black.
- For multiple views, Pixal3D estimates no cameras (`inference_mv.py`, `load_views` and `compute_relative_calc_mat`). The user supplies `transforms.json` with a 4x4 camera-to-world matrix per view and one `camera_angle_x`, in the Blender/NeRF convention (Z-up world, camera looking along -Z). Frame 0 must be the canonical front view at (0, -d, 0). Each view is projected with calc_mat_i = F · inv(C_0) · C_i, so every view is placed relative to view 0. The distance of each view is the norm of its camera position, so the rig must look at the object center. Views are never cropped or rescaled.
- The shipped MV example has four views at azimuths 0, 90, 180 and 270 degrees, elevation 0 and a 20 degree FOV.
- The output object lives in the camera-aligned frame of view 0. Pixal3D predicts no placement. GATOR p5 says that Pixal3D "implicitly assumes that the target is fully visible from its anchor view".

##### Training recipe

- Pixal3D's data toolkit has `dual_grid_view.py` next to the view-aligned latent encoders of 5.1.9. The encoders re-encode with the frozen TRELLIS.2 VAEs.
- Each stage trains from low to high resolution. The structure flow goes from 32 to 64. The shape and texture flows go from 256 to 512 to 1024.

The eight configs in `configs/gen/` are listed below with abbreviated names. The full names follow the pattern of `slat_flow_img2shape_dit_1_3B_256_bf16_proj_finetune_ft512.json`.

| Config | image_size | grid_res | NAF target | Starts from |
|---|---|---|---|---|
| `ss_flow_img_dit_1_3B_32_bf16_proj_finetune` | 512 | 8 | off | nothing |
| `ss_..._32_..._ft64` | 512 | 16 | off | structure-32 checkpoint |
| `slat_flow_img2shape_..._256_..._proj_finetune` | 512 | 16 | 256 | nothing |
| `slat_flow_img2shape_..._256_..._ft512` | 512 | 32 | 512 | shape-256 checkpoint |
| `slat_flow_img2shape_..._512_..._ft1024` | 1024 | 64 | 512 | shape-512 checkpoint, lr 2e-5 |
| `slat_flow_imgshape2tex_..._256_..._proj_finetune` | 512 | 16 | 256 | nothing |
| `slat_flow_imgshape2tex_..._512_..._proj_finetune` | 512 | 32 | 256 | texture-256 checkpoint |
| `slat_flow_imgshape2tex_..._512_..._ft1024` | 1024 | 64 | 1024 | texture-512 checkpoint, lr 2e-5, `full_pbr` true |

- All eight configs use `DinoV3ProjFeatureExtractor` with the mirror `camenduru/dinov3-vitl16-pretrain-lvd1689m`, `image_attn_mode` proj and `num_views: 2`. The `finetune_ckpt` fields are placeholders such as `<path_to_slat_img2shape_256_checkpoint>`.
- `num_views: 2` means that each sample picks one of two pre-rendered view-aligned latents (`structured_latent_shape.py` L330). It is not multi-view conditioning.
- No multi-view training config is published. The MV commit f7cf38429b adds only `inference_mv.py`, the MV pipeline, the MV extractor and README text. The trainer's `_init_image_cond_model` builds only the single-view extractors (Proj and VaeProj). The paper version fine-tuned the single-view model with 2 to 6 random views (paper Sec. 3.4).
- The optimizer, clipping, EMA and `p_uncond` settings match 5.1.10. The learning rate is 1e-4 for the structure flow and the 256 and 512 stages, and 2e-5 for the 1024 stages. The structure flow uses logit-normal timesteps (mean 1, std 1), and the shape and texture flows use uniform timesteps. These settings match GATOR A.5 closely, which suggests that GATOR used the TRELLIS.2 and Pixal3D training code (unconfirmed).

##### Requirements and known issues

- The install order is TRELLIS.2 first (the code imports `o_voxel` and uses FlexGEMM), then `requirements.txt`, then `natten==0.21.0`, then a custom `utils3d` wheel. `requirements.txt` includes MoGe from git and `diffusers==0.37.1`.
- The README says `ATTN_BACKEND=sdpa` works without flash-attn. Whether the sparse flows run without flash-attn or xformers is (unconfirmed).
- A comment in `inference.py` (L86) says that loading all models at once takes "~18 GB". A low-VRAM mode loads one stage at a time. One user reported a peak of about 11 GB at 1024 resolution after patching the DINOv3 extractor (issue #7, unconfirmed).
- Issues to watch are #31 (NAF fails with "CUDA driver error: device not ready" on WSL2 with an RTX 3090), #25 (CUDA out of memory), #22 (an error when loading the structure checkpoint), #18 ("thin shell" double surfaces) and #23 (no Docker support).

##### Initialization for GATOR

- Pixal3D texture-1024 is the preferred appearance-flow start (our inference). It already has 5 global DINOv3 tokens and a trained 2048 to 1536 projection. TRELLIS.2's cross-attention was trained on all 1029 DINOv3 tokens and would see only 5. The 512 and 1024 texture JSONs differ only in `resolution`.
- GATOR feeds view-averaged features, so the `_mv` checkpoints are the closer match. The zero-shot test in 5.1.7 compares the SV and `_mv` checkpoints by flow-matching validation loss at step 0 on GATOR-style multi-view batches.

#### 5.2.3 NAF

- The paper is "NAF: Zero-Shot Feature Upsampling via Neighborhood Attention Filtering" (Chambon et al., Valeo.ai), arXiv 2511.18452, CVPR 2026.
- `hubconf.py` downloads https://github.com/valeoai/NAF/releases/download/model/naf_release.pth (2.66 MB). The release has no separate weight license, so the repo's Apache-2.0 probably applies (unconfirmed). NAF++ (14.8M parameters) has no released checkpoint.
- The default training config uses `vit_base_patch16_dinov3.lvd1689m` (DINOv3 ViT-B/16), 25,000 steps, 512 px images and bf16 (`config/backbone/base.yaml` and `config/base.yaml`). Which backbone produced `naf_release.pth` is (unconfirmed). NAF works with DINOv3 ViT-L/16 in practice, because Pixal3D runs it on ViT-L features.

##### How it works

The sources are `src/model/naf.py`, `src/layers/attentions.py` and `config/model/naf.yaml`.

- NAF upsamples backbone features to any output size, guided only by the RGB image. One trained model works for any backbone without retraining. The README says outputs go "up to 2K", and the paper reports a maximum ratio of 72x on an A100 40 GB.
- The queries come from an image encoder with two branches of 1x1 and 3x3 convolutions (128 channels each, 2 layers). It runs on the guide image at the output resolution and then applies 2D RoPE. The keys are the same encodings, average-pooled to the feature resolution.
- The values are the backbone features themselves, split over 4 heads, so C must be divisible by 4. With DINOv3-L, the q/k head dim is 64 and the value head dim is 256.
- Each output pixel attends to a 9x9 neighborhood of low-resolution cells. NAF calls `natten.na2d(..., backend="cutlass-fna")` with dilation equal to the integer upsampling ratio. It first resizes the keys and values to the output size (nearest-exact), so a full-size copy of the values exists next to the output.
- The output is a softmax-weighted average of the original features, so it stays in the backbone's feature space (our inference from the code and NAF paper Eq. 3).
- The default model has dim 256, 4 attention heads, 4 RoPE heads and kernel size 9. Paper Table 6 gives 0.66M parameters.
- NAF's reference notebook (`notebooks/inference.ipynb`) normalizes the guide with the ImageNet mean and std. Pixal3D feeds the un-normalized image, and the Pixal3D weights were trained on that input.

##### NATTEN

- NAF supports the old NATTEN API (`na2d_qk`, `na2d_av`) and the new one (`na2d` with `backend="cutlass-fna"`). `docs/INSTALL.md` lists tested version pairs such as `natten==0.20.1+torch270cu128`.
- NATTEN 0.20.1 is the first release whose cutlass-fna backend allows a value head dim that differs from the q/k head dim. NAF needs this for DINOv3-L.
- CUTLASS FNA runs on any GPU with compute capability 5.0 or higher (https://natten.org/backends/). NAF hard-codes `cutlass-fna`, so the H100 does not use the faster `hopper-fna` kernel.
- Pixal3D pins `natten==0.21.0`. The latest NATTEN is 0.21.7. Official wheels are at https://whl.natten.org. The Pixal3D author publishes a third-party wheel, `natten-0.21.0+torch2.6cu124-cp310`, at https://github.com/LDYang694/Storages/releases/tag/20260430, with an unknown arch list.
- The source build command is `NATTEN_CUDA_ARCH="8.9;9.0" NATTEN_N_WORKERS=64 pip install natten==0.21.0 --no-build-isolation`. It needs CUDA toolkit 12.0 or later and cmake. In an Apptainer build with no GPU, `NATTEN_CUDA_ARCH` must be set, or libnatten is skipped.
- `torch.hub.load("valeoai/NAF", "naf", pretrained=True)` fetches the repo code from GitHub at run time. Offline jobs load a vendored copy with `source="local"`. We pin NAF commit 37f2dfc (our choice).

##### Compute and memory

- Paper Table 1 reports 265 GFLOPs and 18 FPS for upsampling 384-channel 28x28 features to 448x448 (16x). For DINOv3-L at 512 px, our estimate is about 400 GFLOP per view (unconfirmed). The same formula gives about 284 GFLOP for the Table 1 setting.
- The resized value map adds another 1 GiB to the 1 GiB output at 512 px, so the peak is about 3 to 5 GiB per view in fp32 (unconfirmed). At 1024 x 1024 the output alone is 4 GiB per view.

#### 5.2.4 DINOv3

- The code is at https://github.com/facebookresearch/dinov3, and the weights are in the HF collection "facebook/dinov3".
- ViT-L/16 has 303M parameters, embed dim 1024, 16 heads, 24 layers and an MLP FFN. Like all DINOv3 ViTs, it uses patch size 16, 4 register tokens and RoPE. The layer count is confirmed only through the config of the ungated camenduru mirror, because the official config is gated.
- The transformers docs index the CLS token as `last_hidden_state[:, 0]` and the patches as `last_hidden_state[:, 1 + num_register_tokens:]`. A 384 px crop gives 581 tokens per view, and a 512 px crop gives 1,029.
- Input sides must be multiples of 16, or the model crops to the nearest smaller multiple. RoPE lets the model run at 384 and 512 px without new position tables. The TRELLIS.2 structure flow was trained on 512 px features, so its quality on 384 px crops is untested (unconfirmed).
- The cond_channels of 1024 in TRELLIS.2 and Pixal3D match the ViT-L width. A different DINOv3 size would break the pretrained cross-attention inputs (our inference, strongly supported by both configs).
- The mask stream can feed the binary mask as a 3-channel image through the same frozen DINOv3, as SAM 3D does with `DinoForMasks` (see 5.2.7).
- DINOv3 ViT-L needs about 0.6 GB in bf16, so we run it on the fly. Caching about 2 MB of tokens per view for hundreds of thousands of assets would take too much storage.

##### License and access

- The custom Meta "DINOv3 License" (https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md) allows royalty-free use, copying, modification and derivatives. Redistribution must include the agreement, and publications must acknowledge use. It forbids ITAR-covered, military, warfare, nuclear, espionage and weapons uses. It does not explicitly address commercial use. This is a summary, not legal advice.
- The HF API reports `gated: manual` for ViT-L, so approval may not be instant. The GitHub route needs a request form, after which Meta emails download URLs.
- The ungated mirror `camenduru/dinov3-vitl16-pretrain-lvd1689m` has the same `model.safetensors` sha256 (dcb2e451...) as the official file and ships `LICENSE.md`, so the DINOv3 License applies either way. We use the official repo with the team's HF account and avoid the mirror for anything we publish.

##### transformers versions

- `DINOv3ViTModel` first appears in transformers 4.56.0. Through 5.3.0, the layer signature `forward(hidden_states, position_embeddings=...)` does not change.
- From 5.4.0 on, the layers sit in a `DINOv3ViTEncoder` submodule at `model.model.layer`, so the TRELLIS.2 and Pixal3D loop raises an AttributeError (TRELLIS.2 issue #147 and PR #156). The PR #156 patch is `encoder = getattr(self.model, "model", self.model)`. After any patch, DINOv3 features should match a 4.57.3 run within 1e-4 on a few test images.
- TRELLIS.2's `setup.sh` installs transformers unpinned, so a fresh install today pulls a version that breaks.

#### 5.2.5 SigLIP2

- The collection is https://huggingface.co/collections/google/siglip2-67b5dcef38c175486e240107.
- FixRes checkpoints such as `google/siglip2-so400m-patch16-512` have `model_type: siglip`, so `AutoModel` loads `SiglipModel`. NaFlex checkpoints have `model_type: siglip2` and need `Siglip2Model` (transformers 4.50 or later).
- `siglip2-so400m-patch16-512` has 1.14B parameters, and the full model needs about 2.3 GB in bf16. Its text tower has hidden size 1152, 27 layers, projection size 1152 and a vocabulary of 256,000 Gemma tokens. The text tower alone is roughly 0.7B parameters (unconfirmed). The so400m checkpoints at other resolutions probably share the same text tower (unconfirmed).
- The pooled embedding is `last_hidden_state[:, -1, :]` passed through `self.head` (`modeling_siglip.py` and `modeling_siglip2.py`). Because it reads the last position, padding matters. Recent processors lowercase text by themselves, but older ones may not.
- In current transformers (docs for v5.19.0), `get_text_features` returns a `BaseModelOutputWithPooling` with the embedding in `.pooler_output`. Older versions returned a plain tensor.
- "Normalized" in GATOR most likely means L2 normalization (our inference).

#### 5.2.6 T5Gemma 2

- T5Gemma 2 (arXiv 2512.14856, https://huggingface.co/docs/transformers/model_doc/t5gemma2) adapts Gemma 3 decoder-only models into encoder-decoders with UL2. It also takes images through a frozen SigLIP vision encoder, but GATOR uses only text. The encoder uses full bidirectional self-attention (`create_bidirectional_mask`). EmbeddingGemma uses T5Gemma 2 checkpoints, which supports using the encoder as a text embedder.
- Three pretrained checkpoints exist (270m-270m, 1b-1b and 4b-4b), and none is instruction-tuned.
- `google/t5gemma-2-1b-1b` has 2.12B parameters in total. Its encoder has 26 layers, 4 heads, 1 KV head, head_dim 256 and max_position_embeddings 32,768. The encoder has 698M non-embedding and 302M embedding parameters, so a text-only encoder is about 1.0B parameters (2 GB in bf16).
- The official `config.json` files are gated (HTTP 401). The encoder width comes from public fine-tune configs such as `AlekseyCalvin/Lyrical_ru2en_T5Gemma2_1b_SFT`. It agrees with paper Table 2, because the embedding params divided by the 262,144-token vocabulary give 1152. The width stays (unconfirmed) until we read the official config.
- `google/t5gemma-2-4b-4b` (encoder width about 2560, unconfirmed, 3.9B text-encoder parameters, 7.8 GB in bf16) is the upgrade if text quality matters.
- The repos are gated (`gated: manual`) with the license tag `gemma`. The user must accept the Gemma Terms of Use (https://ai.google.dev/gemma/terms). A public web app counts as a "Hosted Service". It must pass the Section 3.2 use restrictions on to users and include the Gemma notice (our reading, not legal advice).

##### Loading the encoder

- The `t5gemma2` module exists from v5.0.0rc0 on and is missing in every 4.x release, including 4.57.6. transformers 5.19 needs torch 2.5 or later and Python 3.10 to 3.14.
- `modeling_t5gemma2.py` exports `T5Gemma2Encoder`, `T5Gemma2Model` and `T5Gemma2ForConditionalGeneration`, among others. The docs load the model with `AutoModelForSeq2SeqLM` and the inputs with `AutoProcessor`, not `AutoTokenizer`. `AutoTokenizer` probably works too (unconfirmed).
- `get_encoder()` returns a `T5Gemma2Encoder`, which holds `text_model` (a `T5Gemma2TextEncoder`), `vision_tower` and `multi_modal_projector`. A call with only `input_ids` and `attention_mask` skips the vision tower. `T5Gemma2TextEncoder` applies a final RMSNorm and returns `BaseModelOutput(last_hidden_state=...)`.
- We did not test whether `T5Gemma2Encoder.from_pretrained` loads the full checkpoint directly, because the key prefixes may differ.
- A max_length of 128 is a reasonable start (our inference). We precompute the empty-prompt encoding once. Only the adapter trains.

#### 5.2.7 SAM 3D point-patch embedding

- The paper is "SAM 3D: 3Dfy Anything in Images", arXiv 2511.16624.
- The "SAM License" covers both code and checkpoints. It allows royalty-free use and derivatives, allows redistribution only under the same agreement, requires attribution in publications, and forbids ITAR, military, nuclear, espionage and weapons uses. The weights at `facebook/sam-3d-objects` are gated (`gated: manual`, license `other`), and so are the YAML configs.

##### PointPatchEmbed design

The class is `PointPatchEmbed` in `sam3d_objects/model/backbone/dit/embedder/pointmap.py`. The constructor defaults are `input_size=256`, `patch_size=8`, `embed_dim=768` and `remap_output="exp"`.

1. The input is xyz with shape (B, 3, H, W) and an optional valid mask. With no mask, a pixel is valid when all three coordinates are finite.
2. Invalid pixels are set to 0, and a `PointRemapper` (`point_remapper.py`) normalizes the range. The default "exp" remap maps xy to xy / (1 + z) and z to log1p(z), and it assumes camera-frame xyz with positive z. The other options are "linear", "sinh", "sinh_exp" and "exp_disparity". Our encoder uses the linear remap, optionally clamped (5.1.4).
3. `point_proj = nn.Linear(3, embed_dim)` uses xavier_uniform init with gain 0.02. Zeroing the invalid pixels gives them no gradient, and the learned `invalid_xyz_token` is then added to them.
4. Each window gets a learned `cls_token` and a learned `pos_embed_window` of shape (1, 1 + patch_size^2, D). So `pos_embed_window` must be built for the chosen patch size at construction. At a 384 px crop with `patch_size=16`, each window holds 257 tokens.
5. One timm `Block` runs inside each window with `num_heads=16`, `mlp_ratio=2.0`, `qkv_bias=True` and LayerNorm eps 1e-6. The code cites MCC (`facebookresearch/MCC`, `mcc_model.py`).
6. A learned patch-grid `pos_embed` of shape (1, D, n, n) is bilinearly resized and added. An optional pointmap dropout replaces all window tokens with a learned `dropped_xyz_token`.

- The released `ss_generator.yaml` sets embed_dim 512, input 256, patch 8 and remap `linear`. It runs DINOv2-L/14 at 518 px on a grid that does not line up with the pointmap grid. The source is the mirror https://huggingface.co/jetjodh/sam-3d-objects/blob/main/checkpoints/ss_generator.yaml.
- `EmbedderFuser` (`embedder_fuser.py`) gives each condition embedder a projection net (a LayerNorm plus a gated feed-forward) and a learned "positional group" index embedding, then concatenates the token lists.
- `DinoForMasks` (`embedder/dino.py`) runs masks through the DINO backbone. The default `Dino` class loads `dinov2_vitb14` through torch.hub. The paper runs DINOv2 on the object crop with its mask and on the full image with its mask.
- The SAM 3D paper text does not describe PointPatchEmbed. Its Section E.5 says that sensor or monocular-depth pointmaps "minimally affects the shape performance".
- SAM 3D predicts pose and coarse shape jointly as p(O, R, t, s | I, M). `InvariantPoseTarget` (`sam3d_objects/data/dataset/tdfy/pose_target.py`) writes the object-to-scene transform as T(x) = s_scene · |t_rel| · [s_tilde · R(q) · x + t_unit]. `layout_post_optimization_utils.py` refines the pose with mask render-and-compare (`run_alignment`, `compute_loss` and `run_render_compare`), ICP against the pointmap (`run_ICP`) and occlusion checks.

#### 5.2.8 Plücker and Fourier embeddings

- A ray with origin o and unit direction d has Plücker coordinates (d, m), with moment m = o × d. The 6-D vector does not depend on which point on the ray is the origin.
- LVSM (https://github.com/Haian-Jin/LVSM) builds a per-pixel `pose_cond = cat([o x d, d])` in `model/LVSM_scene_decoder_only.py` (`get_posed_input`, option "default_plucker") and patchifies it with the image. Rays come from `compute_rays` in `utils/data_utils.py`. Its license is unknown (NOASSERTION).
- RayDiffusion (https://github.com/jasonyzhang/RayDiffusion) computes one ray per patch on a 16x16 grid inside the crop, as (direction, moment), in `ray_diffusion/utils/rays.py` (`Rays` and `cameras_to_rays`). Its diffuser (`model/diffuser.py`) concatenates DINO patch features, rays and optional NDC coordinates per patch.
- Our ray construction (our inference) expresses camera centers as o' = (o - t_g)/s_g. Directions do not change, because the normalization has no rotation. We average the per-pixel directions over each 16 x 16 patch, renormalize and compute the moment o' × d. The ray through the patch center, as in RayDiffusion, is a simpler equivalent choice.
- The usual Fourier form is γ(u) = [sin(2^k π u), cos(2^k π u)] for k = 0 to L - 1. PyTorch3D's `HarmonicEmbedding` (`pytorch3d/renderer/implicit/harmonic_embedding.py`) implements it with `n_harmonic_functions` (default 6), `omega_0`, `logspace` and `append_input`.
- Our Fourier construction (our inference) normalizes the patch center to [-1, 1] in crop coordinates and applies 6 to 8 octaves. It concatenates the result with the 6-D Plücker vector, runs a small MLP to the token width and multiplies by a zero-initialized gate.

#### 5.2.9 Other models considered

- Hunyuan3D's texture stage paints fixed canonical views in 2D and takes no camera poses. Versions 2.5, 3.0 and 3.1 are closed.
- Most strong 2026 work, including Pixal3D, builds on TRELLIS.2. We found no public TRELLIS.2 fine-tune with point-cloud or depth conditioning.
- MoGe-2 is needed only to run Pixal3D's single-view mode unchanged, because GATOR gets cameras from Pi3X, VGGT-Omega or sensors.
- RMBG-2.0 (BiRefNet) is not needed, because masks come from the datasets or SAM 3. Its card says CC BY-NC 4.0, and it fails to load on transformers 5.0 (TRELLIS.2 issue #101).

### 5.3 Design findings

#### 5.3.1 Training-target frame and latent caching

**Our reading of the training frame**
- The paper says that normalization never rotates, that outputs share the input axes, and that there is no view-index embedding (p18).
- No sentence in the paper mentions gravity alignment or a random global rotation (checked on pages 4 to 9, 18 to 19 and 22 to 31). For the simulation test, objects were placed upright by hand (p31, D.6), which weakly suggests that outputs are not gravity-aligned by default.
- Our reading is that the cached targets live in the training scene frame with the placement rotation baked in (p23). Training observations must then be in that same frame. Estimated cameras and depth on S3 are most likely registered back to it with a camera-only Sim(3), as in the evaluation (p25). The estimator always runs on at least 3 cameras. A Sim(3) is fit from estimated to ground-truth camera centers with Umeyama, and the rotation is refined with the chordal mean of R_gt,i R_est,i^T. Only runs that pass the check in 5.1.9 are cut into 1 to 4 view subsets.
- Pi3X output poses are not anchored to frame 0, so they must be re-expressed relative to camera 0 (`pi3/models/pi3x.py`). Estimated depth at fp16 and 256^2 takes 131 KB per view.

**Structure target, rebuilt per sample**
- TRELLIS.2's `data_toolkit/encode_ss_latent.py` builds the 64^3 occupancy from the 1024-resolution shape-latent coords and asserts that they are below 64. It does no pooling, runs at batch 1 with `sample_posterior=False` and saves the key `z` (https://raw.githubusercontent.com/microsoft/TRELLIS.2/main/data_toolkit/encode_ss_latent.py). Its source coords are already on a 64^3 grid (1024/16) and are stored as uint8 (`encode_shape_latent.py`). Resampling a 64^3 set under a new shift and scale aliases by about one cell, so our source must be at least 256^3.
- One source for the cached 256^3 occupancy is the 512 dual-grid coords, which the 512 shape latent needs anyway, floor-divided by 2. Block-sparse storage keeps each occupied 64^3 parent cell plus a 64-bit mask of its 4^3 children. At about 9K blocks of 11 bytes, one instance takes about 100 KB (estimate, scaled from the 2.2K average tokens of 512 latents in https://arxiv.org/html/2512.14692 Table 1).
- The online rebuild runs on the GPU in under 1 ms (estimate). It expands the blocks to about 140K cell centers and adds the 8 cell corners when the magnification exceeds about 2. It maps the cells to the world with x = s_b c + t_b, then to the anchor frame with q = (x - t_g) / s_g. It computes floor((q + 0.5) * 64), drops cells outside the cube, and scatters.
- The SS encoder config has 1 input channel, 8 latent channels, channels [32, 128, 512], 2 residual blocks per level and 2 middle blocks (https://huggingface.co/microsoft/TRELLIS-image-large/raw/main/ckpts/ss_enc_conv3d_16l8_fp16.json). Our count gives 323 GMAC (0.65 TFLOP) per sample and 59.3M parameters, which at fp16 matches the 119 MB checkpoint. On an L40S, one encode takes about 3.6 to 6 ms, and the peak activation is 16.8 MB per sample (estimates). One training sample of the 1.3B structure DiT costs about 32 TFLOP before cross-attention, so the encoder adds about 2%.
- Partial depth underestimates the extent and pulls the center toward the camera, which is why the anchor in 5.1.3 needs the margin m.

**Geometry and appearance targets, cached once**
- TRELLIS.2's `blender_script/dump_mesh.py` applies `matrix_world`, rescales so the longest bounding-box side is 1, and offsets by the box center. It does no axis swap and writes vertices and faces only. `dual_grid.py` recomputes the box center, uses scale 0.99999 / max extent with no rotation, and calls `mesh_to_flexible_dual_grid` with aabb ±0.5, `face_weight` 1.0, `boundary_weight` 0.2 and `regularization_weight` 1e-2. It writes `.vxz` files with uint8 vertex offsets and an `intersected` bitmask.
- `voxelize_pbr.py` uses the same center and scale and calls `blender_dump_to_volumetric_attr`, dropping the normal and emissive channels. The encoders `shape_enc_next_dc_f16c32_fp16` and `tex_enc_next_dc_f16c32_fp16` run at batch 1 and save float32 feats and uint8 coords. The sources are the files named above under https://raw.githubusercontent.com/microsoft/TRELLIS.2/main/data_toolkit/.
- Translation and isotropic scale never force a rerun, because the box center is subtracted and the scale is normalized. Any rotation and any non-uniform scale force a new O-Voxel conversion and encoding, because the box, the voxel assignment, the dual-vertex offsets and the edge flags all change. Permuting O-Voxels for 90 degree turns is not worth trying. The rotation R can be applied to the dumped vertices in Python before the bounding-box step, so the Blender dump runs once per asset and not once per instance.
- The normalization should use the bounding box of the complete ground-truth object after rotation. Three paper points support this. Both flows train on ground-truth structure (p5). For limited observations, the normalization is recomputed from the aligned structure before projection (p19). The flows start from Pixal3D, which uses a nearly equivalent normalization (p18). Each instance stores R, t_b and s_b, so that x = s_b q + t_b, which is the paper's placement transform (p23).
- O-Voxel conversion at 512 takes a few CPU seconds per instance (under 10 s according to the TRELLIS.2 README). The shape and PBR encoders have 354M parameters each, and their speed is not published. Our estimate is 0.05 to 0.3 s each on an L40S. Storage per instance is about 141 KB for the shape latent and 141 KB for the PBR latent in fp16 (twice that in float32), plus the occupancy and metadata, so about 0.4 MB per instance in fp16 (estimate).

**Other projects, rejected designs and open items**
- Pixal3D `master` reads `transforms.json` and treats each `transform_matrix` as camera-to-world. `sphere_normalize_torch` divides by the largest radius from the box center without subtracting the center. `transform_mesh` applies T_final = R_back · c2w_new · inv(c2w_orig) · R_ply · R_init, which moves each view's camera to one fixed canonical camera. It then scales by 0.49999 / max|coord| and writes `view{idx}_scale.json`. One full O-Voxel conversion and encoder pass runs per asset, view and resolution. The sources are `data_toolkit/{dual_grid_view.py, utils.py, voxelize_pbr_view.py, encode_ss_latent_view.py, encode_shape_latent_view.py}` on https://github.com/TencentARC/Pixal3D.
- RecGen (https://github.com/TRI-ML/recgen) and SAM 3D Objects (https://arxiv.org/html/2511.16624) keep one shape latent per object in an object-centric frame and predict the pose separately, so they never re-encode per view. GATOR bakes rotation into the occupancy latent (p5), so it must cache per placement.
- We rejected a Pixal3D-style cache of K = 8 pre-encoded anchors, because it freezes the perturbations at cache time and loses the per-sample diversity of p23. We rejected a RecGen-style box-frame SS latent plus a 4-number anchor-to-box token, because it conflicts with p5 and p18. We also rejected a fresh random rotation for every sample, because it would need about 4.6 s of CPU time on 168 cores and about 7.7 s of GPU time on 5 GPUs per step at batch 128 (estimate).
- The gravity-aligned fallback in 5.1.2 needs an up estimate at test time, from a RANSAC support plane or the mean camera up-vector. It fails on close-ups with no visible support plane, and the paper does not describe it.
- Five items remain open. They are the clip rate of the chosen anchor margin, the anchor rule when modality dropout removes the pointmap tokens, whether the PBR and shape latent coords are equal per instance, the real encoder time on an L40S at 512, and whether the 3D-FRONT, SceneSmith and SAGE frames must be unified to Z-up before the frame rotation. For the dropout case, one option is to compute the anchor from depth anyway and drop only the tokens.

#### 5.3.2 Inference chain and placement findings

**What the paper says**
- The anchor only initializes translation and scale (p4 to p5, p18 Eq. 3). A final step refines translation and isotropic scale with differentiable rendering of silhouettes and depth plus point alignment. Orientation and shape stay fixed (p5 to p6). Placement accounts for foreground occlusion (p19).

**Released pose and layout code**
- SAM 3D Objects (`sam3d_objects/pipeline/layout_post_optimization_utils.py` on https://github.com/facebookresearch/sam-3d-objects) ships its layout step turned off (`with_layout_postprocess=False`). The mesh path runs an occlusion check, a heuristic init (height-ratio scale, then centroid match), optional Open3D ICP (threshold 0.05, kept only if IoU improves), then render and compare. The loss is 200·MSE(soft silhouette, mask) + 0.1·(quaternion to identity) + 0.05·‖t‖² + 0.05·(s - 1)². The occlusion test compares depth on a 2 px inner mask edge with the minimum depth on the outer edge, and it skips the step instead of masking pixels. The result is rejected if IoU < 0.5 or if IoU does not improve.
- MV-SAM3D (`sam3d_objects/pose_align/pose_optimization.py` on https://github.com/devinli123/MV-SAM3D) uses no renderer. Its loss is a one-sided Chamfer distance from observed Depth Anything 3 points to at most 50K mesh points. It optimizes a normalized quaternion, translation and log-scale with a 0.001 regularizer toward the initial value. It has no occlusion handling and no acceptance gate.
- RecGen (`recgen_inference/preprocessing.py` on https://github.com/TRI-ML/recgen) predicts the pose with a flow and has no post-optimization. Its observation normalization is a good model for an anchor. It erodes the mask (kernel 5), takes the per-axis median of the masked points as the center, and sets the diameter to 2·(q97.5 - q2.5) of the radial distances.
- FoundationPose (`estimater.py` on https://github.com/NVlabs/FoundationPose) initializes translation at the mask-box center pixel and the median valid depth. It uses a 1.2 crop ratio and tanh-bounded translation steps. ReconViaGen (`app_fine.py` on https://github.com/GAP-LAB-CUHK-SZ/ReconViaGen) instead aligns the structure in latent space by optimizing the SS-flow initial noise for 5 Adam steps with a soft Dice loss.
- None of these projects fixes rotation. Our placement borrows SAM 3D's translation-plus-scale stage and IoU gate, MV-SAM3D's log-scale and one-sided Chamfer, RecGen's robust anchor, FoundationPose's 1.2 crop and bounded steps, and SAM 3D's depth-edge occlusion test changed to mask pixels.

**Re-alignment after the structure flow (our design)**
- Connected components (26-connectivity) smaller than 5% of the largest one count as floaters. The fit keeps rotation fixed (Umeyama with R = I, or 30 Adam steps) and uses a trimmed one-sided distance from observed points to boundary-voxel centers. The fit is kept only if the residual drops by more than 10%.
- The new center is the center of the world AABB of the occupied voxels, not the centroid, because the centroid shifts toward thick parts. The new scale is the longest side times the margin. This box uses the full generated shape, so it is robust to partial views but depends on generation quality.
- TRELLIS.2 maps `ss_res` 512 to 32 by max-pooling the decoded 64^3 by a factor of 2 (`trellis2/pipelines/trellis2_image_to_3d.py`). The frame changes after re-alignment, so this plain max-pool is not enough. The structure is re-voxelized by splatting 2^3 sub-samples per old voxel into a new 64^3 grid and is then max-pooled to 32^3.

**Placement refinement (our design)**
- The parameterization is x' = c + exp(ℓ)(x - c) + d0 τ, where c is the mesh AABB center and d0 is its diagonal, so scale and translation decouple. A suggested clamp is |ℓ| ≤ ln 1.5 and ‖τ‖ ≤ 0.5. The optimizer works on a proxy of at most 50K faces made with `cumesh.simplify`. Each view uses a square crop around the target box with ratio 1.2, rendered at 256 px.
- The occluder mask is recomputed each iteration without gradient. It covers other-instance masks (dilated by 2 px), pixels outside the target mask whose valid depth lies more than τ_occ in front of the rendered depth, and crop pixels outside the image. τ_occ = max(0.02 d0, 1 cm). The weight map is 1 minus this mask, so the optimizer does not shrink the object to fit the visible part.
- The silhouette loss is a soft IoU on the anti-aliased coverage, weighted by the map. The depth loss is a Huber loss (δ = 0.01) on |Z - D| / d0 over eroded, unoccluded, confident, covered pixels. The point loss is the mean of min(NN distance, 0.1 d0) / d0 from at most 20K observed points to 50K surface samples. It runs in the observed-to-mesh direction only, so completed unseen parts are not penalized. A regularizer of 1e-3·(‖τ‖² + ℓ²) matches MV-SAM3D.
- Suggested weights are λs = 1, λp = 10, and λd = 10 with sensor depth or 5 with estimated depth. The first 50 iterations use only points and depth, which gives a wide basin. Iterations 50 to 199 use all losses with cosine decay to 0.1 times the learning rate. Learning rates are 1e-2 for τ and 5e-3 for ℓ. The run stops early if the relative improvement stays under 1e-4 for 20 iterations, and it keeps the best result.
- With one view and no depth, ℓ is frozen, because silhouettes cannot separate scale from distance. Views whose mask covers less than 1% of the crop are dropped.
- One nvdiffrast iteration probably takes 5 to 10 ms on an L40S, so 200 iterations take 1 to 2 s (estimate). PyTorch3D soft rasterization would take about 5 to 20 s. For a public deployment without nvdiffrast (license: https://github.com/NVlabs/nvdiffrast/blob/main/LICENSE.txt), the options are PyTorch3D (BSD), Kaolin (Apache-2.0) or a license-free point-splat silhouette with a detached z-buffer.

**Latency and memory on one 48 GB L40S (estimates)**
- The L40S has about 2.7 times less dense BF16 compute and 3.9 times less memory bandwidth than an H100 SXM (5.3.3). TRELLIS.2 at 512 takes about 3 s on an H100 (https://github.com/microsoft/TRELLIS.2), so it should take about 8 to 11 s on an L40S.
- With warm models, a placed mesh takes about 11 to 22 s at N = 4 input views and 15 to 30 s at N = 10. GLB export adds 5 to 12 s with the settings below and 30 to 120 s or more with the README defaults. Resident weights take about 15 to 25 GB, and the peak is about 25 to 35 GB, so one job runs per GPU. TRELLIS.2 itself needs at least 24 GB.
- Several factors could add up to the paper's 32 s. The flows alone take 8 to 15 s on an A100 or L40S class GPU. Full CFG over all 12 steps doubles the network evaluations, while TRELLIS.2 uses guidance intervals such as [0.6, 1.0]. Cross-attention over 3 x 576 x N tokens adds cost at every structure step. The front-end models add 1 to 3 s. Placement iterations, cold loading (10 to 20 s) and the agent's preview renders may also be counted. Per-stage timing logs will show which factors apply.

**GLB export settings that avoid stalls**
- `cv2.inpaint(TELEA)` runs 4 times on the CPU over all unused atlas texels (`postprocess.py` lines 288 to 292). Issue #18 reports under 10% CPU use on a 32-thread CPU, which matches one busy thread (https://github.com/microsoft/TRELLIS.2/issues/18). Issue #53 points to `mesh.simplify` at line 185 in the remesh path (https://github.com/microsoft/TRELLIS.2/issues/53).
- The recommended settings are `decimation_target` 200K to 300K (the HF demo default is 300K), `texture_size` 2048 (the demo default, avoid 4096), `remesh=True`, `remesh_band=1` and `remesh_project=0`. The demo also keeps `mesh.simplify(16777216)` before `to_glb` (https://huggingface.co/spaces/microsoft/TRELLIS.2/blob/main/app.py).
- The inpaint step should be patched to fill only a 4 to 8 px band around the charts and to fill the rest with a constant. A GPU push-pull fill also works. The CPU tail (xatlas packing, inpaint, webp) should run in a process pool so the GPU worker stays free.

#### 5.3.3 Training memory and throughput findings

**Code facts (confirmed in the repos)**
- `SparseStructureFlowModel` has `use_checkpoint: bool = False` (`trellis2/models/sparse_structure_flow.py`). Neither TRELLIS.2 `configs/gen/ss_flow_img_dit_1_3B_64_bf16.json` nor Pixal3D `configs/gen/ss_flow_img_dit_1_3B_32_bf16_proj_finetune_ft64.json` sets it, so the dense structure flow trains without checkpointing. Block checkpointing calls `torch.utils.checkpoint.checkpoint(self._forward, x, mod, context, phases, use_reentrant=False)` in `ModulatedTransformerCrossBlock` (`trellis2/modules/transformer/modulated.py`), whose cross-attention has no gate.
- The sparse flows use `ElasticSLatFlowModel = SparseTransformerElasticMixin + SLatFlowModel`. `with_mem_ratio` checkpoints the first `min(ceil((1 - mem_ratio) * 30) + 1, 30)` blocks (`trellis2/models/sparse_elastic_mixin.py`, `trellis2/utils/elastic_utils.py`).
- `LinearMemoryController` records `torch.cuda.max_memory_allocated` and fits `mem = k·(tokens·mem_ratio) + b` with `np.polyfit` over a ring buffer of 1,000 entries every 500 steps. It then solves `mem_ratio = (available·target_ratio - b) / (k·tokens)`, clamped to [0, max_mem_ratio]. `max_mem_ratio` grows by 0.1 per fit, and `mem_ratio` is random before the first fit. The code defaults are target 0.8 and start 0.1. Both repos' 512 configs use `target_ratio` 0.75 and `max_mem_ratio_start` 0.5.
- `max_tokens: 8192` is a per-sample filter and not a per-batch budget. Micro-batches are load-balanced by voxel count (`trellis2/datasets/structured_latent.py`). The Pixal3D 512 flow configs use `batch_size_per_gpu` 8, `batch_split` 2, `num_workers` 14 and `proj_in_channels` 2048. DINOv3 and NAF run online under `torch.no_grad()` on every batch, with `naf_target_size` 512 for shape and 256 for texture (`pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py`).
- Both trainers use plain DDP through `mp.spawn`, with `bucket_cap_mb=128` and `find_unused_parameters=False`, and neither supports FSDP, DeepSpeed or ZeRO. They do not set `gradient_as_bucket_view`, and the PyTorch default is False. Turning it on saves memory equal to the gradient size (https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html). The EMA is a `deepcopy` of the master params on rank 0 only. In `amp` mode the master params are the model params, which are fp32 by default, so weights, gradients and Adam state are all fp32. `batch_split` gives gradient accumulation with `no_sync` (`trellis2/trainers/basic.py`, `pixal3d/trainers/basic.py`).

**Parameters and memory per GPU (estimates)**
- One block has about 42.5M parameters at d = 1536 (self-attention 4d², cross-attention 2d² + 2·1024·d, MLP 2·d·8192). Thirty blocks plus embedders give 1.29B for the TRELLIS.2 structure flow. The GATOR structure flow is about 1.32B, with about 30M for the mixer, pointmap encoder and SigLIP2 adapter. The sparse flows are about 1.40B, because 30 Linear(2048 to 1536) layers add 94M and the T5Gemma 2 adapter adds about 15M.
- State memory is P x (4 weights + 4 gradients + 8 Adam + 4 DDP bucket + 2 autocast bf16 cache) bytes on every rank, plus 4P for the EMA on rank 0. Frozen DINOv3 ViT-L plus the CUDA and NCCL context add about 3.1 GiB. Activations take about 78·d bytes, or about 120 KB, per token per block. Each cross-attention context token costs about 17 KB per block. Usable memory is about 45 GiB on an L40S and about 79.6 GiB on an H100 80 GB (unconfirmed).

| Item (GiB) | Structure flow (1.32B) | Sparse flow (1.40B) |
|---|---|---|
| Static total, rank 0 / other ranks | about 35 / 30 | about 37 / 32 |
| Activations per sample, no checkpointing | 14.5 (1 view) to 17.0 (4 views) | 7.6 at 2.2K tokens, 28.4 at 8,192 tokens |
| Activations per sample, full checkpointing | about 1.2 to 1.3, plus about 2.3 for the condition encoder at 4 views | 0.65 at 2.2K tokens, 2.4 at 8,192 tokens |
| NAF upsampled map, transient | none | 0.5 per view at 512² in bf16, 1 in fp32 |

- A consistency check supports these numbers. The TRELLIS.2 structure config uses 8 samples per GPU with split 4 (micro-batch 2) on H100, which we estimate at about 63 GiB. A micro-batch of 4 would need about 91 GiB.
- On a 48 GB L40S without FSDP or ZeRO, a per-GPU batch of at least 1 fits only with activation checkpointing. Rank 0 is the limit, because it holds the EMA. Rank 0 therefore checkpoints more and is the slowest rank, and DDP waits for it.

| Flow | L40S, default code | L40S with bucket view on and EMA off the GPU | H100 80 GB |
|---|---|---|---|
| Structure | Out of memory at 1 sample without checkpointing (about 50 to 53 GiB). With `use_checkpoint: true`, micro-batch 2 to 3 fits (about 40 to 42 GiB). | Micro-batch 1 without checkpointing is tight (about 43 GiB). Micro-batch 6 to 8 fits with checkpointing. | Micro-batch 2 without checkpointing (about 71 GiB at 4 views). Checkpointing allows 16 or more but adds 33% compute. |
| Geometry and appearance | The elastic budget (0.75 x 45 = 33.7 GiB) is below rank-0 static memory, so the controller always checkpoints fully. Micro-batch 4 fits for average samples, and worst-case 8K-token batches need 2 to 3. | With `target_ratio` about 0.9, micro-batch 4 fits with partial checkpointing. | Micro-batch 4, as in the shipped configs, with about 30% of blocks checkpointed. |

- ZeRO-1 (`ZeroRedundancyOptimizer` over 5 GPUs) would save about 6.6 GiB per GPU, but it needs a patch to `basic.py`. GATOR's pointmap encoder runs one transformer block over the pixels of each patch (p18). At width 1024 it would cost about 3.9 TFLOP forward per 384² view, and a width of 256 or less keeps it near 2.5% of compute.
- More changes that help on the L40S are an elastic `target_ratio` of about 0.9, fused AdamW, and NAF at a 256 target or only at projected pixels. Set `num_workers` to 8 to 14, because the TRELLIS.2 default of `ceil(cpu_count / num_gpus)` gives about 34 workers per rank on lunarL40S.
- An effective batch of 128 does not divide by 5 GPUs. On the L40S box, use 120 (24 per GPU) or 130. For short runs, an EMA of 0.9995 instead of 0.9999 may help (unconfirmed suggestion). One option for the appearance flow is to train only the cross-attention layers, the projection linears and the T5Gemma 2 adapter first. This adds text conditioning to Pixal3D's texture weights and saves optimizer memory and about a third of the backward compute (estimate).

**Hardware, FLOPs and throughput (estimates unless sourced)**
- The L40S peaks at 362.05 TFLOPS dense BF16 (733 with sparsity), with 48 GB, 864 GB/s, PCIe Gen4 and no NVLink (https://www.nvidia.com/en-us/data-center/l40s/). A measured large matmul on an L40S reaches about 270 TFLOPS, and LLM training on 4 L40S reaches 37 to 44% MFU (https://arxiv.org/html/2512.15306).
- The H100 SXM lists 1,979 TFLOPS BF16 with sparsity, so about 989.5 dense, and 3.35 TB/s (https://www.nvidia.com/en-us/data-center/h100/). The H100 PCIe 80 GB lists 1,513 with sparsity, so about 756 dense (PNY datasheet). The H100 PCIe is therefore about 1.31 times slower than the SXM part.
- FlashAttention-2 supports Ampere, Ada and Hopper, and FlashAttention-3 needs H100 or H800 (https://github.com/Dao-AILab/flash-attention). FA2 reaches about 35% utilization on H100, and FA3 reaches 740 TFLOPS (75%) in FP16 (https://arxiv.org/abs/2407.08608). Besides the TRELLIS.2 backends in 5.2.1, Pixal3D accepts `flash_attn_4` and `sdpa` (`pixal3d/modules/sparse/config.py`).
- The structure flow costs 45.4 TFLOP per sample at 1 view and 62.0 at 4 views (53.7 on average), with a context of 1,728 tokens per view. DINOv3 on RGB and mask adds 1.9, and full checkpointing adds 16.8. Going from 1 to 4 views costs only about 1.4 times more. Each sparse flow costs 21.1 TFLOP per sample at 2.2K tokens. Elastic checkpointing adds 2.1 on H100 and 6.8 on L40S, and DINOv3 at 512 adds 1.8.
- Attention is 31 to 43% of forward FLOPs in the structure flow and about 18% in the sparse flows. FA3 therefore gives about 25 to 30% on the structure flow and about 7% on the sparse flows.

| Setup | Structure flow, samples/s per GPU | Sparse flow, samples/s per GPU |
|---|---|---|
| L40S (structure fully checkpointed at micro-batch 3, sparse elastic at micro-batch 4) | 1.50 optimistic, 0.30 pessimistic | 2.43 optimistic, 0.73 pessimistic |
| H100 SXM (structure at micro-batch 2, sparse at micro-batch 4) | 5.34 optimistic (6.7 with FA3), 1.07 pessimistic | 7.92 optimistic (8.5 with FA3), 2.37 pessimistic |

- The optimistic column assumes 30% MFU for the structure flow and 20% for the sparse flows. The pessimistic column assumes 6% MFU, which matches published fine-tunes. RecGen trained TRELLIS-image-large (about 1.2B) for 55K steps at batch 512 on 64 H100 in about 48 h, which implies 2.55 samples/s per H100 and about 5% of peak (https://arxiv.org/html/2604.27106). TRELLIS.2 issue #138 trained the 1.3B shape flow for 50K steps at batch 2 and lr 2.5e-5 on 1 A100 80 GB in about 30 h, which implies 0.93 samples/s and about 6 to 8% of A100 peak (https://github.com/microsoft/TRELLIS.2/issues/138). Measuring a few hundred steps on each GPU type decides which column applies.
- Two other published runs report steps but no training time. Pixal3D's paper version trained its sparse DiT for 200K, 100K, 80K and 40K steps at 256, 384, 512 and 1024, and its dense DiT for 300K steps at 1e-4 plus 200K at 2e-5 (https://arxiv.org/html/2605.10922). ReconViaGen fine-tuned TRELLIS for 40K steps at batch 192 on 8 A800 80 GB (https://arxiv.org/html/2510.23306v1).
- Scaled from NAF's published cost (5.2.3) to 512² and 1024 channels, NAF takes about 70 to 190 ms per view on an A100 or L40S and 30 to 125 ms on an H100. At 512 for the geometry flow, it adds 0.08 to 0.30 s per sample on H100 and 0.18 to 0.48 s on L40S. In the optimistic case, this drops H100 throughput from 7.9 to between 2.3 and 4.8 samples/s, and L40S throughput from 2.4 to between 1.1 and 1.7. A 256 target cuts the cost about 4 times.
- Each step all-reduces 5.3 to 5.6 GB of fp32 gradients. On 5 L40S over PCIe Gen4 this takes about 0.4 to 0.9 s per optimizer step against about 17 s of compute, so 2 to 5%. `nvidia-smi topo -m` shows the topology.

**Encoders: online or cached**
- DINOv3 ViT-L/16 costs 0.38 TFLOP per image at 384 px and 0.73 at 512 px. On RGB and mask at 384, it costs 0.77 TFLOP per view in the structure flow, about 5% of training FLOPs. On masked RGB at 512, it costs 0.73 TFLOP per view in the sparse flows, about 12%. NAF dominates in time, not in FLOPs.
- Cached DINOv3 tokens in bf16 take 1.19 MB per image at 384 px (581 x 1024 x 2 B) and 2.11 MB per view at 512 px. A full NAF map takes 537 MB per view at 512² and 134 MB at 256², so it cannot be cached. Caching the per-voxel projected features instead takes about 9 MB per item, but it fixes the view subset and the camera perturbation.
- Camera, intrinsics and depth noise, view-subset sampling and modality dropout leave the DINOv3 features unchanged. Mask perturbation, image-quality perturbation and crop jitter change them (p23). DINOv3 and NAF therefore run online. Only the text embeddings are cached, plus one empty-prompt embedding, which keeps T5Gemma 2 and SigLIP2 off the training GPUs.

**Multi-GPU crashes**
- TRELLIS.2 #88 ("terminate called without an active exception", multi-GPU only) is open. Two users confirm the workaround: `num_workers=0` in `snapshot_dataset()` and `run_snapshot()`, with `iter(dataloader)` moved outside the loop. Our trainers also turn snapshots off by setting `i_sample` very high.
- Three more TRELLIS.2 issues are open with no workaround. #90 is an illegal memory access in the NCCL watchdog (4 RTX PRO 6000, 2 H200). #95 is a crash in flex_gemm on the first forward pass (2 H200), and xformers did not help. #164 is an illegal memory access in `visualize_sample` inside nvdiffrast `dr.rasterize` for the shape and texture DiTs, while the structure DiT ran fine. TRELLIS v1 #256 is a dataloader hang from `num_workers = cpu_count / num_gpus`, fixed with a small worker count.
- Pixal3D's `snapshot_dataset` hard-codes `num_workers=0`, and its `snapshot()` runs on the master only with `dist.barrier()` when `num_samples < 4`. A code comment says that it frees cached GPU memory before the snapshot to avoid OOM and illegal address errors. Pixal3D therefore already includes the #88 workaround. None of its visible issues concern multi-GPU or DDP (https://github.com/TencentARC/Pixal3D/issues). Pixal3D uses the same flex_gemm and flash_attn stack, so whether it avoids #95 and #164 is unconfirmed. The dense structure flow does not use flex_gemm, so #95 probably does not apply to it (estimate).
