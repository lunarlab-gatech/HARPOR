# HARPOR

HARPOR turns a short capture of one rigid object into a 3D model that a simulator can load. It started as a
re-implementation of GATOR (arXiv 2610.11215).

The input is a phone video, a ROS 1 or 2 bag or a Stray Scanner export of one rigid object. The output is a textured
mesh and MJCF, USD and URDF files with estimated mass and materials. An optional AI agent refines the geometry and the
appearance in Blender, and MuJoCo checks each result. The research focus is estimating and refining the mass and the
friction of rigid objects.

![HARPOR pipeline: a capture becomes a posed mesh through SAM 3, Depth Anything 3, view selection, MV-SAM3D and a pose fit (A), VoMP estimates the materials right after MV-SAM3D (B), an optional agent refines the geometry and appearance and a host check settles each candidate in MuJoCo with the VoMP materials (C), and the export transfers the materials and writes MJCF, USD and URDF files, an optional 360 video and results.zip (D)](docs/pipeline.svg)

HARPOR finds the object in the frames with SAM 3, gets cameras and depth from Depth Anything 3 and picks the views.
LiDAR or iPhone depth sets the metric scale when the capture has it. MV-SAM3D builds a textured mesh from the views,
and a pose fit places it in the scene (A). VoMP v1.1 estimates the density, Young's modulus and Poisson's ratio of the
mesh right after MV-SAM3D makes it (B).

The agent stage is optional (C). The agent works in Blender on the geometry and the appearance. After each build or
repair, the host compares the surface with the original, settles the candidate under gravity in MuJoCo with the VoMP
materials, and asks a VLM to judge the frames against the photo. A failed check starts a repair round, with at most 3
builds. The host then delivers the best stored candidate. The export turns the mesh upright, transfers the VoMP
materials to it and recomputes the mass and the inertia (D). Each of the MJCF, USD and URDF files holds one rigid body.
Friction is fixed at 0.6, because VoMP does not predict it. An optional 360 video of the final mesh comes next, and the
job ends with results.zip.

HARPOR used to split objects into moving parts, add joints and simulate soft parts. Those features were removed so the
work can focus on mass and friction. Commit 7f23a9a is the last commit that has them, for when they come back.

## Install

You need a Linux x86_64 machine with an NVIDIA GPU from Ampere to Hopper (RTX 30xx/40xx, A100, A6000, L40S,
H100/H200) and a driver for CUDA 13.0 or newer. Blackwell cards (RTX 50xx, B200) don't work yet. You also need Hugging
Face access to `facebook/sam3` and `facebook/sam-3d-objects`. The agent needs an OpenAI or Anthropic key.

### With docker

You need docker and the NVIDIA Container Toolkit. Build the image once from the repo root. The build takes about two
hours and 60 GB, and needs no GPU:

```bash
docker build -f containers/Dockerfile -t harpor:0.1 .
```

Then check the setup, download the weights (about 30 GB) and start the web app:

```bash
containers/harpor-docker check
HF_TOKEN=hf_... containers/harpor-docker weights
containers/harpor-docker serve
```

Open the URL it prints. The weights, caches and jobs go in `~/harpor-data` (set `HARPOR_DATA` to change it).
`weights` also keeps the Hugging Face token there, in `cache/huggingface/token`, because the pipeline needs it to load
the gated models. The app listens on localhost only. To use it from another computer, run
`ssh -L 8765:localhost:8765 <gpu machine>` there.

### Without docker

`setup/install.sh` builds the six environments (web, sam3d, da3, sam3seg, vomp11 and mujoco) under `HARPOR_ROOT`
(by default `~/harpor`) and then downloads the weights. It runs `setup/web.sh`, `setup/sam3d.sh`, `setup/mvsam3d.sh`,
`setup/sam3seg.sh`, `setup/vomp11.sh`, `setup/mujoco.sh`, `setup/agent.sh` and `setup/weights.sh` in order, and each
one also runs on its own. The agent and Blender then run in Apptainer, so the machine needs `apptainer` too:

```bash
HF_TOKEN=hf_... bash setup/install.sh
~/harpor/envs/web/bin/python -m harpor.webapp
```

On a Slurm cluster, `harpor/webapp/open_slurm.sh` starts the app inside a GPU job and opens it on your laptop. Its
header lists the settings it needs. [docs/build_plan.md](docs/build_plan.md) note 10 has the details.

### From the terminal

You can also run a whole job without the browser:

```bash
containers/harpor-docker run cup.mp4 --prompt mug --agent opus     # with docker
python -m harpor.webapp.run cup.mp4 --prompt mug --agent opus      # without docker, in envs/web
```

`--agent` is `astra` (GPT-6-Astra), `opus` (Claude Opus 5.5), `fable` (Claude Fable 5.2) or `none`, which is the
default. `--help` lists every option and explains what to do when a job stops to ask you something.

## More

- [docs/build_plan.md](docs/build_plan.md) explains how HARPOR works, and [docs/research.md](docs/research.md) has the
  research notes.
- The tests need no GPU. They run whole jobs with stand-in stages (`tests/fake_steps.py`):

  ```bash
  python3.12 -m venv .venv
  .venv/bin/pip install pytest fastapi uvicorn httpx numpy scipy pillow av rosbags pyyaml trimesh "anthropic==1.15.0"
  .venv/bin/python -m pytest -q tests
  ```

  Tests that need a package you don't have are skipped.
- A standalone static demo gallery can be built separately. [docs/public_site.md](docs/public_site.md) explains how.
- HARPOR's code is MIT licensed ([LICENSE](LICENSE)). SAM 3, SAM 3D Objects and MV-SAM3D keep their own SAM License.
