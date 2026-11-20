"""Stand-ins for the stages of a web-app job, so the tests run whole jobs without GPUs or the GPU envs.

    python tests/fake_steps.py STAGE --job J [--key-file F] [--agent astra|opus|fable]

The fake_stages fixture (conftest.py) puts command in place of steps.command and ask_provider in place of
jobs.ask_provider, and tests/fake_launch.py does the same for a test that starts the server or the terminal command
as a process of its own.

Each stage writes the files its real command writes (steps.py), in a second or two, and prints a
few log lines and PROGRESS lines. The scene is synthetic and ignores the video's pixels: a box of
30 x 20 x 20 cm on a floor, filmed by a camera that circles it at 1 m, 50 cm above the floor, over
270 degrees from the first frame to the last. The masks, the Depth Anything 3 cameras and depth
and the posed mesh all come from that one scene, so the real view selection and the real gravity
alignment work on them.

Real code runs wherever it needs no GPU and none of the GPU envs: topics, frames and extract (video.py, also
for a rosbag, since rosbags is pure Python, and for a Stray Scanner zip), prep (mvsam3d.prep), lidar
(lidar.py, with the LiDAR points of the views), select (views.py), align (physics_auto.py), estimate
(estimate_step.py, with the fake model's answer), cameras.json
(mvsam3d.cameras), turntable (harpor.webapp.turntable with the fake Blender of harpor.agent.fake_agent,
whose frames are the GLB's outline; the MP4 is a real H.264 file) and package (jobs.package).
For a job with LiDAR (steps.lidar: a bag's LiDAR topic or a Stray Scanner capture's
depth), the fake DA3 is off by FAKE_DA3_SCALE: its depth, pointmaps and camera translations are 0.8 x
the scene's, so the real lidar stage must find 1.25 from the bag's true points or the capture's true
depth images, and place records metric_scale as the real place does. place sizes the posed box in the
run's DA3 units (run_scale), as MV-SAM3D would, so a scale that select did not apply shows in the
aligned size. The agent and estimate stages
check the key file (mode 0600, an OPENAI_API_KEY line for astra, an ANTHROPIC_API_KEY line for
opus and fable) and never print the key. The agent stage writes the result record of the agent it
stands in for. The estimate stage runs the real harpor.webapp.estimate_step with ESTIMATE as the
model's answer in place of a request: a hollow cardboard box of 0.25 kg with friction 0.4.

mjcf writes a box body with the mass and the friction of physics/summary.json, usd its rigid body as text, and
urdf one box link with the real harpor.physics.urdf.
"""

import argparse
import json
import re
import runpy
import shlex
import shutil
import signal
import stat
import sys
import time
from pathlib import Path

import numpy as np

from harpor.fileio import write_json

ROOT = Path(__file__).resolve().parents[1]  # the repo, which the stand-ins import harpor from

BOX_LO, BOX_HI = np.array([-0.15, -0.2, -0.1]), np.array([0.15, 0.0, 0.1])  # world: y down, floor at y = 0
TARGET = np.array([0.0, -0.1, 0.0])
RADIUS, HEIGHT, SWEEP = 1.0, 0.5, 1.5 * np.pi
FAR = 4.0
FOCAL = 0.9  # fx = fy = FOCAL * image width
DELAY = 0.05
FAKE_DA3_SCALE = 0.8  # a job with LiDAR: DA3's depth and camera translations are this times the scene's
COLOR = [170, 120, 70, 255]
FAKE_BLENDER = [sys.executable, "-m", "harpor.agent.fake_agent", "--blender"]  # the videos' Blender
# The fake model's answer in the estimate stage
ESTIMATE = {"material": "cardboard", "structure": "hollow", "mass_kg": 0.25, "mass_low_kg": 0.15, "mass_high_kg": 0.4,
            "friction": 0.4, "friction_low": 0.3, "friction_high": 0.5,
            "reason": "fake model: an empty cardboard box of this size weighs about a quarter of a kilogram"}


def command(stage, job, cfg, key_file=None):
    """steps.command for the tests: (argv, cwd, env additions) that run this file's stand-in of stage. It refuses
    what steps.command refuses: an unknown stage, and a stage that needs the key file (steps.needs_key) without one."""
    from harpor.webapp import steps

    if stage not in steps.NAMES:
        raise ValueError(f"unknown stage {stage!r}")
    key = steps.needs_key(stage, job)
    if key and key_file is None:
        raise ValueError(f"the {stage} stage needs the key file")
    q = shlex.quote
    script = f"exec {q(sys.executable)} {q(str(Path(__file__).resolve()))} {stage} --job {q(str(Path(cfg.runs) / job['id']))}"
    if key:
        script += f" --key-file {q(str(key_file))} --agent {steps.agent(job)}"
    return ["bash", "-c", script], str(ROOT), {"PYTHONUNBUFFERED": "1", "PYTHONPATH": str(ROOT)}


def ask_provider(key, agent):
    """jobs.ask_provider for the tests: every key that passes jobs.check_key's own checks passes, with no request."""
    return True, "key accepted (test)"


def pace(n=4):
    for i in range(1, n + 1):
        time.sleep(DELAY)
        print(f"PROGRESS {i / n:.2f}", flush=True)


def real(module, *args):
    """Run a real CLI in this process, as `python -m module args` would."""
    sys.argv = [module, *map(str, args)]
    runpy.run_module(module, run_name="__main__", alter_sys=True)


def camera(angle):
    """(R, C): OpenCV world-to-camera rotation and camera center of the circling camera."""
    C = np.array([RADIUS * np.sin(angle), -HEIGHT, -RADIUS * np.cos(angle)])
    z = (TARGET - C) / np.linalg.norm(TARGET - C)
    x = np.cross([0.0, 1.0, 0.0], z)
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z]), C


def intrinsics(w, h):
    return np.array([[FOCAL * w, 0.0, (w - 1) / 2], [0.0, FOCAL * w, (h - 1) / 2], [0.0, 0.0, 1.0]])


def render(w, h, angle):
    """Ray-cast the scene: depth (optical-axis z, h x w), the box mask, K and the 3x4 extrinsics."""
    R, C = camera(angle)
    K = intrinsics(w, h)
    v, u = np.mgrid[0:h, 0:w].astype(float)
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)  # z = 1, so t is depth
    d = rays @ R  # world directions
    with np.errstate(divide="ignore", invalid="ignore"):
        t_floor = np.where(d[..., 1] > 1e-9, -C[1] / d[..., 1], np.inf)
        t1, t2 = (BOX_LO - C) / d, (BOX_HI - C) / d
        near, far = np.nanmax(np.minimum(t1, t2), -1), np.nanmin(np.maximum(t1, t2), -1)
    t_box = np.where((far >= near) & (far > 0), np.maximum(near, 0), np.inf)
    mask = t_box < t_floor
    return np.minimum(np.minimum(t_floor, t_box), FAR), mask, K, np.hstack([R, (-R @ C)[:, None]])


def frame_angle(job_dir, name):
    meta = json.loads((Path(job_dir) / "frames" / "meta.json").read_text())
    n = len(meta["times"])
    return SWEEP * int(re.sub(r"\D", "", name)) / max(n - 1, 1)


def material():
    import trimesh

    return trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
        baseColorFactor=COLOR, metallicFactor=0.0, roughnessFactor=0.8))


def box_mesh(R=np.eye(3), t=np.zeros(3), k=1.0):
    """The box with a PBR material, its world points mapped by x -> R (k x) + t (world to W), in glTF axes."""
    import trimesh

    from harpor import frames

    box = trimesh.creation.box(extents=BOX_HI - BOX_LO)
    x = k * (box.vertices + (BOX_LO + BOX_HI) / 2) @ np.asarray(R).T + t
    return trimesh.Trimesh(frames.w_to_gltf(x), box.faces, process=False, visual=material())


def view0_pose(run, name=None):
    """(R, t) that map the scene's world to frame W, x -> R x + t: view 0's DA3 extrinsics. name: view 0,
    else the first camera of run/cameras.json."""
    run = Path(run)
    name = name or json.loads((run / "cameras.json").read_text())[0]["name"]
    with np.load(run / "da3_output.npz") as da:
        stem = {Path(str(f)).stem: i for i, f in enumerate(da["image_files"])}
        E0 = da["extrinsics"][stem[name]].astype(float)
    return E0[:, :3], E0[:, 3]


def run_scale(run, name):
    """The run's DA3 units per meter of the scene, to 4 decimals: view 0's (name's) camera distance from
    the world origin in run/da3_output.npz over the scene's, which is the same for every camera of the
    circle (camera). MV-SAM3D works in DA3's units, so the fake posed box takes this size: 1 after
    select applied the LiDAR scale, FAKE_DA3_SCALE when it was lost."""
    return round(float(np.linalg.norm(view0_pose(run, name)[1]) / np.hypot(RADIUS, HEIGHT)), 4)


def check_key_file(key_file, which):
    """SystemExit unless the key file is mode 0600 and starts with the line of the agent's provider.
    The key is never read into a message."""
    from harpor.webapp.steps import KEY_VAR

    f, var = Path(key_file or ""), KEY_VAR[which]
    if not key_file or not f.is_file() or stat.S_IMODE(f.stat().st_mode) != 0o600:
        raise SystemExit(f"key file {f} is missing or not mode 0600")
    if not f.read_text().startswith(f"{var}="):
        raise SystemExit(f"key file {f} has no {var} line")


def masks(J):
    from PIL import Image

    from harpor.stage_a.mvsam3d import scene_images

    job = json.loads((J / "job.json").read_text())
    picks = (job.get("overrides") or {}).get("instance") or {}
    out = J / "scene" / f"sam3_{job['slug']}"
    (out / "check").mkdir(parents=True, exist_ok=True)
    views = []
    images = scene_images(J / "scene")
    for k, (name, path) in enumerate(images.items()):
        img = Image.open(path).convert("RGB")
        W, H = img.size
        s = min(1.0, 480 / max(W, H))  # the silhouette at most 480 px wide, then scaled up
        m = render(round(W * s), round(H * s), frame_angle(J, name))[1]
        m = np.array(Image.fromarray(m).resize((W, H), Image.NEAREST))
        inst = []
        if m.any():
            r, c = np.nonzero(m)
            box = [float(c.min()), float(r.min()), float(c.max() + 1), float(r.max() + 1)]
            inst = [{"index": 0, "score": 0.93, "box_xyxy": box, "mask_px": int(m.sum())}]
        rec = {"name": name, "image": str(path), "size_wh": [W, H], "instances": inst, "picked": [0] if inst else [],
               "pick": "top_score", "mask": None, "mask_px": 0, "warnings": []}
        if picks.get(name) == "none":
            rec.update(picked=[], pick="user", warnings=[f"left out by --pick {name}=none"])
        elif not inst:
            rec["warnings"].append("no instance above score 0.5; no mask written")
        if rec["picked"]:
            Image.fromarray(np.where(m, 255, 0).astype(np.uint8)).save(out / f"{name}.png")
            rec.update(mask=f"{name}.png", mask_px=int(m.sum()))
        else:
            (out / f"{name}.png").unlink(missing_ok=True)
        check = np.asarray(img).astype(float)
        if rec["picked"]:
            check[m] = 0.6 * check[m] + 0.4 * np.array([0.0, 255.0, 0.0])
        Image.fromarray(check.astype(np.uint8)).save(out / "check" / f"{name}.jpg", quality=85)
        views.append(rec)
        if k % 4 == 0:
            print(f"PROGRESS {(k + 1) / len(images):.2f}", flush=True)
    (out / "sam3.json").write_text(json.dumps({"model": "fake", "prompt": job["prompt"], "threshold": 0.5,
                                               "mask_threshold": 0.5, "views": views}, indent=1))
    n = sum(v["mask"] is not None for v in views)
    print(f"{job['prompt']!r}: {n} of {len(views)} photos masked -> {out} (fake SAM 3)")


def prep(J):
    from harpor.stage_a import mvsam3d

    job = json.loads((J / "job.json").read_text())
    for d in ("input", "masks"):
        shutil.rmtree(J / "pool" / d, ignore_errors=True)
    (J / "pool" / "prep.json").unlink(missing_ok=True)
    pace(2)
    mvsam3d.prep(J / "scene", f"sam3_{job['slug']}", J / "pool", 1024)


def da3(J):
    """DA3 rendered from the scene; for a job with LiDAR of either kind, off by FAKE_DA3_SCALE (depth,
    pointmaps and the extrinsics' translations)."""
    from PIL import Image

    from harpor.stage_a.mvsam3d import view_names
    from harpor.webapp import steps

    off = FAKE_DA3_SCALE if steps.lidar(json.loads((J / "job.json").read_text())) else 1.0
    names = view_names(J / "pool")
    W, H = Image.open(J / "pool" / "input" / "images" / f"{names[0]}.png").size
    s = min(504, max(W, H)) / max(W, H)
    w, h = round(W * s), round(H * s)
    out = {k: [] for k in ("depth", "mask", "K", "E")}
    for name in names:
        for k, x in zip(out, render(w, h, frame_angle(J, name))):
            out[k].append(x)
    depth, K = np.array(out["depth"], np.float32) * np.float32(off), np.array(out["K"], np.float32)
    E = np.array(out["E"], np.float32)
    E[:, :, 3] *= off
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    pts = np.stack([(u - K[:, 0, 2, None, None]) * depth / K[:, 0, 0, None, None],
                    (v - K[:, 1, 2, None, None]) * depth / K[:, 1, 1, None, None], depth], -1)
    pace(3)
    np.savez(J / "pool" / "da3_output.npz", depth=depth, conf=np.ones_like(depth), pointmaps=pts,
             pointmaps_sam3d=pts.transpose(0, 3, 1, 2).copy(), extrinsics=E, intrinsics=K,
             image_files=np.array([str(J / "pool" / "input" / "images" / f"{n}.png") for n in names]), process_res=504)
    print(f"DA3 (fake): {len(names)} views at {w}x{h}" + (f", {off} x the scene's scale" if off != 1 else ""))


def sam3d(J):
    from harpor.stage_a.mvsam3d import view_names

    pace(4)
    box_mesh().export(J / "run" / "canonical.glb")
    names = view_names(J / "run")
    pose = {"rotation_wxyz": [1.0, 0.0, 0.0, 0.0], "translation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0],
            "frame": "fake: canonical box", "views": names, "view0": names[0], "seed": 42, "seconds": 0.2, "peak_gib": None}
    (J / "run" / "pose.json").write_text(json.dumps(pose, indent=1))
    print(f"MV-SAM3D (fake): a box from {len(names)} views")


def place(J):
    from PIL import Image

    from harpor.stage_a import mvsam3d

    run = J / "run"
    cams = mvsam3d.cameras(run)
    (run / "cameras.json").write_text(json.dumps(cams, indent=1))
    R0, t0 = view0_pose(run, cams[0]["name"])
    da3_units = run_scale(run, cams[0]["name"])  # the box in DA3's units, as MV-SAM3D's would be
    box_mesh(R0, t0, da3_units).export(run / "posed.glb")
    for c in cams:
        img = Image.open(c["image"]).convert("RGB")
        k = 512 / max(img.size)
        size = (round(img.width * k), round(img.height * k))
        a = np.asarray(img.resize(size)).astype(float)
        m = np.array(Image.open(c["mask"]).resize(size, Image.NEAREST)) > 0
        a[m] = 0.55 * a[m] + 0.45 * np.array([255.0, 0.0, 0.0])
        Image.fromarray(a.astype(np.uint8)).save(run / f"overlay_{c['name']}.png")
    pace(3)
    center = R0 @ (da3_units * (BOX_LO + BOX_HI) / 2) + t0
    with np.load(run / "da3_output.npz") as da:  # select applied a LiDAR scale
        metric = float(da["metric_scale"]) if "metric_scale" in da.files else None
    units = "meters (LiDAR scale applied in select)" if metric is not None else "DA3 units (meters)"
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(R0.T @ np.diag([-1., -1., 1.])).as_quat()
    fitted_pose = {"rotation_wxyz": q[[3, 0, 1, 2]].tolist(), "translation": (t0 * [-1, -1, 1]).tolist(),
                   "scale": [da3_units] * 3}
    placement = {"frame_W": f"OpenCV camera of view 0: x right, y down, z forward, in {units}",
                 "glb_axes": "posed.glb vertices are glTF: (x, y, z)_W = (x, -z, y)_glTF (Prompt 2)",
                 "view0": cams[0]["name"], "metric_scale": metric, "units": units, "fake": True,
                 "pose": fitted_pose, "scale_fit": {"applied": False},
                 "pose_fit": {"applied": False}, "posed_extent_W": (da3_units * (BOX_HI - BOX_LO)).tolist(),
                 "posed_center_W": center.tolist(),
                 "check": {"views": [{"name": c["name"], "mask_iou": 0.8} for c in cams], "mean_mask_iou": 0.8,
                           "view0_depth_to_mesh_median_over_diag": 0.01, "view0_depth_within_2pct_diag": 0.9}}
    (run / "placement.json").write_text(json.dumps(placement, indent=1))
    print(f"placement (fake): mean mask IoU 0.8 over {len(cams)} views")


def agent(J, key_file, which="astra"):
    """A refinement that returns posed.glb. The result has the surface report of the real session
    (harpor.agent.surface); the fake runs no physics rounds."""
    from harpor.agent import controller, surface, tools
    from harpor.webapp.agent_step import AGENTS

    check_key_file(key_file, which)
    root = J / "agent" / "s"
    shutil.rmtree(root, ignore_errors=True)
    (root / "work" / "exports").mkdir(parents=True)
    (root / "control").mkdir()
    for f in ("result.json", "evidence.json"):
        (J / "agent" / f).unlink(missing_ok=True)
    glb = root / "work" / "exports" / "001.glb"
    shutil.copyfile(J / "run" / "posed.glb", root / "original.glb")
    shutil.copyfile(J / "run" / "posed.glb", glb)
    pace(4)
    kind = AGENTS[which]  # the controller's name of the agent
    info = controller.agent_record(kind)
    result = {"agent": kind, "model": info["model"], "effort": info["effort"], "source": "finish",
              "verdict": "complete", "reason": "fake agent: the posed mesh is returned as the refinement", "limitations": [],
              "glb": "work/exports/001.glb", "sha256": tools.sha256(glb), "exit": "finish_sigint", "returncode": 130,
              "cli_exit": controller.cli_exit(130, kind), "timings": {"wall_s": 4 * DELAY},
              "counts": {"exports": 1, "reviews": 1, "checkpoints": 0}, "rejected": [], "network_commands": [],
              "key_found_in": [], "surface": surface.compare(glb, root / "original.glb")}
    if kind == "claude":
        tokens = {"input_tokens": 41000, "output_tokens": 9000, "cache_read_input_tokens": 380000,
                  "cache_creation_input_tokens": 52000}
        result.update(turns=12, tokens=tokens, end_reason="interrupted",
                      cost_usd=round(sum(tokens[k] * tools.PRICES[tools.CLAUDE_MODEL][k] for k in tokens) / 1e6, 4))
    write_json(root / "result.json", result)
    shutil.copyfile(root / "result.json", J / "agent" / "result.json")
    print(f"agent (fake {info['model']}): finish {result['verdict']}, GLB {result['glb']}")


def estimate(J, key_file, which="astra"):
    """The real estimate stage (harpor.webapp.estimate_step.run) with ESTIMATE as the model's answer: vlm.ask_claude
    and vlm.ask_astra give it in this process, with no request."""
    from harpor.agent import vlm
    from harpor.webapp import estimate_step

    check_key_file(key_file, which)
    answer = "Here is my estimate.\n" + json.dumps(ESTIMATE)
    vlm.ask_claude = lambda key, prompt, image, client=None, model=None: answer
    vlm.ask_astra = lambda key, prompt, image: answer
    pace(2)
    estimate_step.run(J, key_file, which)


def mjcf(J):
    """A box body with the inertial and the friction of physics/summary.json (harpor.physics.mjcf.DEFAULT_FRICTION
    without one), as the real build takes them."""
    from harpor.physics.mjcf import DEFAULT_FRICTION
    from harpor.webapp.physics_auto import sim_name

    job = json.loads((J / "job.json").read_text())
    name, P = sim_name(job["prompt"]), J / "physics"
    summary = json.loads((P / "summary.json").read_text())
    rb = summary["rigid_body"]
    friction = DEFAULT_FRICTION if summary.get("friction") is None else float(summary["friction"])
    aligned = json.loads((P / "aligned.json").read_text())
    lo, hi = np.array(aligned["min_s"]), np.array(aligned["max_s"])
    shutil.rmtree(P / "mjcf", ignore_errors=True)
    (P / "mjcf").mkdir()
    z0 = max(0.0, -float(lo[2])) + 0.005

    def vec(x):
        return " ".join(f"{v:.6g}" for v in x)

    (P / "mjcf" / f"{name}.xml").write_text(f"""\
<mujoco model="{name}">
  <worldbody>
    <body name="{name}" pos="0 0 {z0:.4f}">
      <freejoint/>
      <inertial pos="{vec(rb['com'])}" mass="{rb['mass']:.6g}" fullinertia="{vec(rb['fullinertia'])}"/>
      <geom type="box" pos="{vec((lo + hi) / 2)}" size="{vec((hi - lo) / 2)}" friction="{friction:g} 0.005 0.0001"/>
    </body>
  </worldbody>
</mujoco>
""")
    (P / "mjcf" / "scene.xml").write_text(f"""\
<mujoco model="scene">
  <include file="{name}.xml"/>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.05" friction="{friction:g} 0.005 0.0001"/>
  </worldbody>
</mujoco>
""")
    pace(2)
    (P / "mjcf" / "mjcf.json").write_text(json.dumps({"name": name, "fake": True, "friction": friction,
                                                      "collision_hulls": 1, "start_z": z0,
                                                      "inertial": {k: rb[k] for k in ("mass", "com", "fullinertia")},
                                                      "compiled": True}, indent=1))
    print(f"MJCF (fake): {P / 'mjcf' / 'scene.xml'}")


def usd(J):
    """A rigid body with the mass of physics/summary.json, as text."""
    from harpor.webapp.physics_auto import sim_name

    job = json.loads((J / "job.json").read_text())
    name, P = sim_name(job["prompt"]), J / "physics"
    rb = json.loads((P / "summary.json").read_text())["rigid_body"]
    (P / "usd").mkdir(exist_ok=True)
    (P / "usd" / f"{name}.usda").write_text(f"""\
#usda 1.0
(
    defaultPrim = "{name}"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{name}" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"])
{{
    float physics:mass = {rb['mass']:.6g}
}}
""")
    (P / "usd" / "scene.usda").write_text(f"""\
#usda 1.0

def Xform "World"
{{
    def "{name}" (prepend references = @./{name}.usda@)
    {{
    }}
}}
""")
    pace(2)
    (P / "usd" / "usd.json").write_text(json.dumps({"name": name, "fake": True, "checked": {"mass": rb["mass"]}},
                                                   indent=1))
    print(f"USD (fake): {P / 'usd' / 'scene.usda'}")


def urdf(J):
    """One link with a box mesh, written by the real harpor.physics.urdf."""
    import trimesh

    from harpor.physics import urdf as U
    from harpor.physics.mjcf import write_obj
    from harpor.webapp.physics_auto import sim_name

    job = json.loads((J / "job.json").read_text())
    name, P = sim_name(job["prompt"]), J / "physics"
    out = P / "urdf"
    shutil.rmtree(out, ignore_errors=True)
    aligned = json.loads((P / "aligned.json").read_text())
    rb = json.loads((P / "summary.json").read_text())["rigid_body"]
    (out / "meshes").mkdir(parents=True)
    box = trimesh.creation.box(bounds=np.array([aligned["min_s"], aligned["max_s"]]))
    write_obj(out / "meshes" / "link_0.obj", box.vertices, box.faces)
    link = {"name": name, "inertial": {k: rb[k] for k in ("mass", "com", "fullinertia")},
            "mesh": "meshes/link_0.obj", "collisions": ["meshes/link_0.obj"]}
    (out / "model.urdf").write_text(U.urdf_xml(name, link))
    report = {"glb": str((P / "aligned.glb").resolve()), "name": name, "links": [link]}
    pace(2)
    report.update(fake=True, checked=U.check(out))
    (out / "urdf.json").write_text(json.dumps(report, indent=1))
    print(f"URDF (fake): {report['checked']['links']} links, {report['checked']['joints']} joints, {out / 'model.urdf'}")


def turntable(J):
    """The real stage with the fake Blender: the frames, then the MP4 (harpor.webapp.turntable). A Cancel kills the
    fake Blender's process group too, as turntable.main makes it do."""
    from harpor.agent import tools
    from harpor.webapp import turntable as T

    signal.signal(signal.SIGTERM, tools._on_sigterm)
    T.render(J, blender=FAKE_BLENDER)
    T.encode(J)


def main(argv=None):
    from harpor.webapp import jobs, steps

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=steps.NAMES)
    ap.add_argument("--job", required=True)
    ap.add_argument("--key-file", help="agent and estimate stages: the env file with the API key of the job's agent")
    ap.add_argument("--agent", choices=("astra", "opus", "fable"), default="astra",
                    help="agent and estimate stages: the job's agent")
    a = ap.parse_args(argv)
    J = Path(a.job)
    print(f"fake {a.stage}: {J.name}", flush=True)
    if a.stage == "topics":
        real("harpor.webapp.video", "topics", "--job", J)
    elif a.stage == "frames":
        real("harpor.webapp.video", "analyze", "--job", J)
    elif a.stage == "extract":
        real("harpor.webapp.video", "extract", "--job", J)
    elif a.stage == "lidar":
        real("harpor.webapp.lidar", "scale", "--job", J)
    elif a.stage == "select":
        real("harpor.webapp.views", "select", "--job", J)
    elif a.stage == "align":
        real("harpor.webapp.physics_auto", "align", "--job", J)
    elif a.stage == "package":
        jobs.package(J)
    elif a.stage in ("agent", "estimate"):
        globals()[a.stage](J, a.key_file, a.agent)
    else:
        globals()[a.stage](J)


if __name__ == "__main__":
    main()
