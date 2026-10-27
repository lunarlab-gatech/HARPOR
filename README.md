# HARPOR

High-fidelity Articulated sim-Ready Object Reconstruction.

HARPOR turns a short capture of one object into models you can drop into a simulator. You give it a phone video,
a rosbag or a Stray Scanner capture. It picks good views, builds a textured mesh, splits the mesh into moving parts
and sets their joints if you ask, and estimates mass and materials. Then it writes MuJoCo (MJCF), USD and URDF
models. An optional LiDAR step sets the real-world scale, and an optional 360-degree video shows the result.

It started as a re-implementation of the GATOR pipeline (arXiv 2610.11215).

## What you get

- Meshes as GLB files: the mesh as captured, an upright copy, and one mesh per part for an articulated object.
- Joints between the parts (revolute or prismatic), when you turn on articulation.
- Mass, Young's modulus, Poisson ratio and density per part.
- MJCF, USD and URDF models, and a `results.zip` with everything.
- Optional: a 360-degree video of the result.

## Three ways to use it

### 1. The public site

https://harpor.andyli3964.workers.dev/

Sign in with Google or GitHub. The site's owner approves each account by hand. Jobs run on your own GPU machine;
the site shows them. The Demos page shows finished runs, and you can see it without signing in.

To connect your GPU machine, follow "Run a personal worker on your own GPU machine" in
[docs/public_site.md](docs/public_site.md).

### 2. Try the web app without a GPU (fake mode)

Fake mode runs every stage with quick stand-ins, so you can try the whole interface on a laptop. The results are
placeholders, not real reconstructions.

```bash
git clone https://github.com/lunarlab-gatech/HARPOR.git && cd HARPOR
```

```bash
python3.12 -m venv .venv && .venv/bin/pip install fastapi uvicorn httpx numpy scipy pillow av rosbags pyyaml trimesh cryptography
```

```bash
.venv/bin/python -m gator.webapp --fake --runs ./runs --host 127.0.0.1 --port 8765
```

Open the "on this machine" link that the server prints (it holds a login token), and add a short video.

### 3. Run real jobs on your own GPU machine

You need:

- Linux with an NVIDIA GPU (we use L40S) and Apptainer.
- The model environments, containers and checkouts. [docs/build_plan.md](docs/build_plan.md) notes 4, 4b, 7 and 9
  list them.
- Access on Hugging Face to the gated models `facebook/sam3` and `facebook/sam-3d-objects`.
- An API key for the agent that refines the mesh: an OpenAI key for GPT-6-Astra, or an Anthropic key for
  Claude Opus 5.5. You can also run without an agent.

The stages run from `$GATOR_ROOT/gator_clone` (by default `~/scratch/gator/gator_clone`), so clone the repo there.
Then start the web app from that folder, with the web environment's Python:

```bash
python -m gator.webapp
```

Add `--worker` to show the machine's jobs on the public site (see docs/public_site.md).

## Inputs

- A video: `.mp4`, `.mov`, `.m4v`, `.avi`, `.mkv` or `.webm`. Walk around the object, keep it in frame, and record
  with HDR off.
- A rosbag: `.bag`, or the `.mcap` or `.db3` file of a ROS 2 bag. A LiDAR topic in the bag can set the scale.
- A Stray Scanner capture (iPhone LiDAR): zip the export folder.

The app's Help page (the Help link at the top right) explains every setting and result.

## Repository layout

| Folder | What it holds |
|---|---|
| `gator/webapp` | The web app: server, job runner, stages, personal worker, public site build |
| `gator/stage_a` | Masks (SAM 3), depth (Depth Anything 3), the mesh (MV-SAM3D) and its pose, LiDAR scale |
| `gator/parts` | The part list and the split of the mesh into parts |
| `gator/agent` | The agent that refines the mesh in Blender, and sets joints |
| `gator/physics` | Upright pose, materials (VoMP), MJCF, USD and URDF |
| `supabase` | The public site's database schema |
| `tests` | Tests (pytest) |
| `docs` | Design notes, research notes and the public site's setup |

## Tests

```bash
.venv/bin/pip install pytest "anthropic==1.15.0" rtree && .venv/bin/python -m pytest -q tests
```

Tests that need a package you do not have (MuJoCo, USD, torch, Node.js 20 for the page's script tests) skip.

## Docs

- [docs/build_plan.md](docs/build_plan.md): how HARPOR works, with design and build notes.
- [docs/public_site.md](docs/public_site.md): the public site, personal workers, and the owner's setup.
- [docs/research.md](docs/research.md): research notes and sources.

## License

MIT for HARPOR's own code: see [LICENSE](LICENSE). The models it uses keep their own licenses. SAM 3, SAM 3D Objects
and MV-SAM3D use the SAM License; docs/build_plan.md lists the rest.
