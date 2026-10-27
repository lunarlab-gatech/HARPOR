# HARPOR

High-fidelity Articulated sim-Ready Object Reconstruction. HARPOR started as a re-implementation of the GATOR
pipeline (arXiv 2610.11215).

HARPOR turns a short capture of one object into simulator models. It builds a textured mesh, splits the mesh into
moving parts with revolute or prismatic joints if you check Articulated, and estimates each part's mass, Young's
modulus, Poisson ratio and density. It writes GLB meshes (as captured, upright, and one per part for an articulated
object), MuJoCo (MJCF), USD and URDF models, and a `results.zip` with everything. It can also make a 360-degree
video of the result.

The input is one of these:

- A video (`.mp4`, `.mov`, `.m4v`, `.avi`, `.mkv` or `.webm`). Record with HDR off, walk around the object, and
  keep it in frame.
- A rosbag (`.bag`, or the `.mcap` or `.db3` file of a ROS 2 bag). A LiDAR topic in the bag can set the real-world
  scale.
- A zipped Stray Scanner export from an iPhone with LiDAR. Its depth sets the real-world scale.

The app's Help page (the Help link at the top right) explains every setting and result.

## Three ways to use it

**The public site.** Sign in at https://harpor.andyli3964.workers.dev/ with Google or GitHub, and wait for the
owner to approve your account. Jobs run on your own GPU machine, and the site shows them. To connect a machine, see
"Run a personal worker on your own GPU machine" in [docs/public_site.md](docs/public_site.md). The Demos page shows
finished runs, and it needs no sign-in.

**Fake mode, without a GPU.** Every stage runs a quick stand-in, so you can try the interface on a laptop. The
results are placeholders.

```bash
git clone https://github.com/lunarlab-gatech/HARPOR.git && cd HARPOR
python3.12 -m venv .venv && .venv/bin/pip install fastapi uvicorn httpx numpy scipy pillow av rosbags pyyaml trimesh cryptography
.venv/bin/python -m gator.webapp --fake --runs ./runs --host 127.0.0.1 --port 8765
```

Then open the "on this machine" link that the server prints (it holds a login token), and add a short video.

**Real jobs on your own GPU machine.** You need:

- Linux with an NVIDIA GPU (we use L40S) and Apptainer.
- The model environments, containers and checkouts in [docs/build_plan.md](docs/build_plan.md) notes 4, 4b, 7 and 9.
- Access on Hugging Face to the gated models `facebook/sam3` and `facebook/sam-3d-objects`.
- An API key for the agent that refines the mesh: an OpenAI key for GPT-6-Astra, or an Anthropic key for Claude
  Opus 5.5. Local jobs can also run without an agent.

The stages run from `$GATOR_ROOT/gator_clone` (by default `~/scratch/gator/gator_clone`), so clone the repo there.
Start the web app from that folder with the Python of `envs/web`. Add `--worker` to show the machine's jobs on the
public site (see [docs/public_site.md](docs/public_site.md)).

```bash
python -m gator.webapp
```

## Repository

| Folder | What it holds |
|---|---|
| `gator/webapp` | The web app: server, job runner, stages, personal worker, public site build |
| `gator/stage_a` | Masks (SAM 3), depth (Depth Anything 3), the mesh (MV-SAM3D) and its pose, LiDAR scale |
| `gator/parts` | The part list and the split of the mesh into parts |
| `gator/agent` | The agent that refines the mesh in Blender, and sets joints |
| `gator/physics` | Upright pose, materials (VoMP), MJCF, USD and URDF |
| `supabase` | The public site's database schema |

Docs: [build_plan.md](docs/build_plan.md) explains how HARPOR works, [public_site.md](docs/public_site.md) covers
the public site, personal workers and the owner's setup, and [research.md](docs/research.md) holds research notes.

The tests use pytest. A test skips when it needs something that you do not have, for example MuJoCo, USD,
torch, Postgres 15, or Node.js 20 for the page's script tests.

```bash
.venv/bin/pip install pytest "anthropic==1.15.0" rtree && .venv/bin/python -m pytest -q tests
```

## License

HARPOR's own code is MIT (see [LICENSE](LICENSE)). The models keep their own licenses. SAM 3, SAM 3D Objects and
MV-SAM3D use the SAM License, and docs/build_plan.md lists the rest.
