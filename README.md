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

## Run it in a terminal

`python -m harpor.webapp.run` runs a whole job with no web server and no browser. It uses the same stages and the same
checks as the page. Run it from the repo with the Python of `envs/web`, on the machine with the GPU. Add `--fake` to
try it on a laptop. `--help` lists every flag.

```bash
python -m harpor.webapp.run cup.mp4 --prompt mug
```

```bash
python -m harpor.webapp.run cup.mp4 --prompt mug --agent opus --turntable
```

```bash
python -m harpor.webapp.run chair.bag --prompt chair --articulated --parts chair.json --camera-topic /cam/image --lidar-topic /ouster/points --calibration cal.yaml
```

The command prints each stage as it runs, then the path of `results.zip`. The job folders are in `--runs DIR`, by
default `$HARPOR_ROOT/webapp_runs`.

The command never asks a question. When a job needs you, it prints what to look at and the exact next command, and it
exits with code 3.

| The job stops | What to do |
| --- | --- |
| At a bag with several camera topics | Run the printed command with one of the topics it lists. |
| After the analysis, without `--prompt` | Look at the thumbnails, then run the printed command with the noun. Add `--picks 12,80,150` to choose the frames yourself. |
| After the view selection, with `--review` or when the automatic selection fails | `--resume ID` goes on. Add `--exclude`, `--lock`, `--view0` or `--instance` to select again and look again. |
| After the part list, with `--edit-parts` | `--resume ID` accepts the list. `--resume ID --parts edited.json` replaces it. |

A failed stage prints the reason, its log file and the resume command, and the command exits with code 1. `--resume ID`
then goes on from the first stage that is not done. Ctrl-C, SIGTERM and a closed terminal cancel the running stage and
leave a job that can resume. Under `nohup` the job keeps running after the terminal closes. Bad arguments and refused
settings exit with code 2.

Settings stay as they are once the build has started, so use a new job to change them. The one exception is
`--calibration`, which starts again from the lidar stage.

No agent runs unless you ask for one. `--agent astra` and `--agent opus` read the key from `--key-file F`, or else from
`~/.config/harpor/openai.env` or `anthropic.env`. The file holds a line `OPENAI_API_KEY=...` or
`ANTHROPIC_API_KEY=...`, and it has mode 0600 in a folder of mode 0700. Without an agent there is no VLM to list the
parts, so `--articulated` needs `--parts`, a JSON list such as
`[{"name": "body"}, {"name": "lid", "parent": "body", "joint": "revolute"}]`.

One process runs the jobs of a runs folder, so the command refuses to start while a server or another command uses
that folder. Stop the server, or give another `--runs DIR`. Jobs of this command stay off the public site.

On ICE, run the command as a step of your Slurm job, the way `harpor/webapp/start_ice.sh` starts the server, so that
the stages share the job's GPU. `CODE` is the repo and `G` is `HARPOR_ROOT`.

```bash
srun --jobid=JOBID --overlap --ntasks=1 bash -lc "module purge 2>/dev/null; cd $CODE && HARPOR_ROOT=$G exec $G/envs/web/bin/python -m harpor.webapp.run /full/path/cup.mp4 --prompt mug"
```

## More

- Docs: [build_plan.md](docs/build_plan.md) (how it works), [public_site.md](docs/public_site.md) (the site and
  personal workers), [research.md](docs/research.md) (research notes).
- Tests: `.venv/bin/pip install pytest "anthropic==1.15.0" rtree && .venv/bin/python -m pytest -q tests`. Tests that
  need a package you do not have skip.
- License: MIT for HARPOR's code ([LICENSE](LICENSE)). The models keep their own licenses: SAM 3, SAM 3D Objects
  and MV-SAM3D use the SAM License.
