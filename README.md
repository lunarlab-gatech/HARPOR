# HARPOR

High-fidelity Articulated sim-Ready Object Reconstruction: a short capture of one object in, simulation-ready
models out. It started as a re-implementation of GATOR (arXiv 2610.11215).

Give it a phone video, a rosbag or a zipped Stray Scanner export. It builds a textured mesh, can split it into
moving parts with joints, estimates mass and materials, and writes GLB, MJCF, USD and URDF files, with an optional
360-degree video. LiDAR in a rosbag, or an iPhone's depth, sets the real-world scale. The app's Help page explains
every setting.

## Try it

**Public site:** https://harpor.andyli3964.workers.dev/ shows finished runs on its Demos page. To run your own jobs,
sign in, wait for the owner's approval, and connect your GPU machine ([docs/public_site.md](docs/public_site.md)).

**Fake mode, no GPU:** every stage runs a quick stand-in, so you can try the interface on a laptop.

```bash
git clone https://github.com/lunarlab-gatech/HARPOR.git && cd HARPOR
python3.12 -m venv .venv && .venv/bin/pip install fastapi uvicorn httpx numpy scipy pillow av rosbags pyyaml trimesh cryptography
.venv/bin/python -m harpor.webapp --fake --runs ./runs --host 127.0.0.1 --port 8765
```

Open the "on this machine" link it prints, then add a short video.

**Real jobs:** you need Linux with an NVIDIA GPU, Apptainer, Hugging Face access to `facebook/sam3` and
`facebook/sam-3d-objects`, the environments in [docs/build_plan.md](docs/build_plan.md) notes 3, 4, 4b, 7 and 9 and
the header of `harpor/webapp/start_ice.sh`, and
an OpenAI or Anthropic key if you want the agent. Clone the repo anywhere, set `HARPOR_ROOT` to the folder with
`envs/`, `sif/` and the model clones (by default `~/scratch/gator`), and run `python -m harpor.webapp` from the repo
with the Python of `envs/web`. `setup/mvsam3d.sh`, `setup/vomp11.sh` and `setup/mujoco.sh` build the environments of
notes 4b and 9 in that folder.

## More

- Docs: [build_plan.md](docs/build_plan.md) (how it works), [public_site.md](docs/public_site.md) (the site and
  personal workers), [research.md](docs/research.md) (research notes).
- Tests: `.venv/bin/pip install pytest "anthropic==1.15.0" rtree && .venv/bin/python -m pytest -q tests`. Tests that
  need a package you do not have skip.
- License: MIT for HARPOR's code ([LICENSE](LICENSE)). The models keep their own licenses: SAM 3, SAM 3D Objects
  and MV-SAM3D use the SAM License.
