# HARPOR

HARPOR turns a short video of an object into a 3D model you can drop into a simulator. It started as a
re-implementation of GATOR (arXiv 2610.11215).

You give it a phone video, a rosbag or a Stray Scanner export. You get back a textured mesh and MJCF, USD and URDF
files with mass and materials filled in. If the object has moving parts, like a lid or a drawer, HARPOR splits them
out and an AI agent adds the joints. It then tests the joints in MuJoCo and gives the agent another try when a test
fails. A part can also be soft, like a seat cushion, so it dents in MuJoCo.

![HARPOR pipeline: a video becomes a posed mesh (A), an object with parts is split into pieces (B), an optional agent refines the mesh or adds joints and is checked in MuJoCo (C), materials are estimated after generation (A) or splitting (B), and simulation files are written (D)](docs/pipeline.svg)

HARPOR finds the object in the video and builds a mesh in the right pose (A). VoMP estimates a rigid object's
materials immediately after MV-SAM3D generates it. For an object with parts, HARPOR first splits the mesh (B),
then runs VoMP on each part. An optional agent in Blender refines the mesh or adds joints (C). MuJoCo tests the
predicted physics, and a VLM judges simulation frames against the photos; failures feed repair rounds.
Finally, HARPOR transfers the estimates to the final geometry, recomputes mass and inertia, and exports the
simulation files (D).

The soft parts and the repair rounds have only run in tests so far, not on a real object.

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
the gated models. The app
listens on localhost only. To use it from another computer, run `ssh -L 8765:localhost:8765 <gpu machine>` there.

### Without docker

`setup/install.sh` builds the same seven environments under `HARPOR_ROOT` (by default `~/harpor`) and then downloads
the weights. It runs `setup/web.sh`, `setup/sam3d.sh`, `setup/mvsam3d.sh`, `setup/sam3seg.sh`, `setup/vomp11.sh`,
`setup/mujoco.sh`, `setup/agent.sh` and `setup/weights.sh` in order, and each one also runs on its own. The agent and
Blender then run in Apptainer, so the machine needs `apptainer` too:

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

`--agent` is `astra` (GPT-6-Astra), `opus` (Claude Opus 5.5), `fable` (Claude Fable 5.2) or `none`. `--help` lists
every option and explains what to do when a job stops to ask you something.

## More

- [docs/build_plan.md](docs/build_plan.md) explains how HARPOR works, and [docs/research.md](docs/research.md) has the
  research notes.
- The tests need no GPU. They run whole jobs with stand-in stages (`tests/fake_steps.py`):

  ```bash
  python3.12 -m venv .venv
  .venv/bin/pip install pytest fastapi uvicorn httpx numpy scipy pillow av rosbags pyyaml trimesh "anthropic==1.15.0" rtree
  .venv/bin/python -m pytest -q tests
  ```

  Tests that need a package you don't have are skipped.
- A standalone static demo gallery can be built separately; see [docs/public_site.md](docs/public_site.md).
- HARPOR's code is MIT licensed ([LICENSE](LICENSE)). SAM 3, SAM 3D Objects and MV-SAM3D keep their own SAM License.
