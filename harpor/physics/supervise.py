"""MuJoCo visual evidence for physics supervision, using pre-refinement materials.

Runs in envs/mujoco. Writes actual simulation frames as GIF clips and a contact
sheet, plus numerical settling, joint and soft-part evidence. No VLM runs here.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from harpor.fileio import write_json
from harpor.physics import align, material_transfer, mjcf, sim_check


class Capture:
    def __init__(self, out):
        self.out, self.frames, self.renderer, self.model = Path(out), {}, None, None

    def __call__(self, m, d, name):
        import mujoco
        from PIL import Image, ImageDraw

        if self.model is not m:
            self.close()
            self.model = m
            self.renderer = mujoco.Renderer(m, height=300, width=400)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.lookat[:] = m.stat.center
        camera.distance = max(0.2, 2.5 * m.stat.extent)
        camera.azimuth, camera.elevation = 135, -25
        self.renderer.update_scene(d, camera=camera)
        frame = Image.fromarray(self.renderer.render())
        ImageDraw.Draw(frame).text((8, 8), f"{name}  {d.time:.2f} s", fill="white", stroke_width=1, stroke_fill="black")
        self.frames.setdefault(name, []).append(frame)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
        self.renderer, self.model = None, None

    def save(self):
        from PIL import Image

        if not self.frames:
            raise ValueError("No simulation frames were captured")
        sheet = Image.new("RGB", (400 * 4, 300 * len(self.frames)), "white")
        clips = {}
        for row, (name, frames) in enumerate(self.frames.items()):
            path = self.out / f"test_{row:02d}.gif"
            frames[0].save(path, save_all=True, append_images=frames[1:], duration=250, loop=0)
            clips[name] = path.name
            for col, idx in enumerate(np.linspace(0, len(frames) - 1, 4).astype(int)):
                sheet.paste(frames[idx], (400 * col, 300 * row))
        sheet.save(self.out / "sheet.jpg")
        return {"sheet": "sheet.jpg", "clips": clips}


def settle(m, d, observe):
    """Three seconds under gravity on the exported floor; retain motion and warnings."""
    import mujoco

    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    states = []
    for k in range(13):
        if k:
            mujoco.mj_step(m, d, sim_check.steps(m, 0.25))
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            raise ValueError("The settling simulation became non-finite")
        states.append({"t": float(d.time), "speed": float(np.linalg.norm(d.qvel)),
                       "com": d.subtree_com[0].tolist()})
        observe(m, d, "settling")
    warnings = sim_check.warnings(m, d)
    return {"seconds": 3.0, "gravity": m.opt.gravity.tolist(), "states": states, "warnings": warnings}


def run(glb, materials, out, sha, intervals=None):
    import mujoco

    materials, out = Path(materials), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((materials / "config.json").read_text())
    original = json.loads((materials / "summary.json").read_text())
    predictions = {}
    for name, filename in config["predictions"].items():
        with np.load(materials / filename) as z:
            predictions[name] = {k: z[k] for k in z.files}
    aligned = out / "aligned.glb"
    align.main(["--glb", str(glb), "--out", str(aligned), "--up-near=" + ",".join(map(str, config["up"])),
                "--x", "footprint", *(["--links-out", str(out / "links")] if config["articulated"] else [])])
    transform = json.loads((out / "aligned.json").read_text())["T_s_from_w"]
    links = sorted((out / "links").glob("*.glb")) if config["articulated"] else None
    summary = material_transfer.transfer(aligned, out, original, predictions, transform, links)
    capture = Capture(out)
    try:
        dynamics = sim_check.run(aligned, out / "motion", sha, intervals, summary=out / "summary.json", observe=capture)
        # A second model retains gravity, the floor and the object's actual fixed/free base.
        model = mjcf.build(aligned, out / "summary.json", out / "settling", "object", textures=False)
        m = mujoco.MjModel.from_xml_path(str(out / "settling/scene.xml"))
        settling = settle(m, mujoco.MjData(m), capture)
        images = capture.save()
    finally:
        capture.close()
    return {"asset_sha256": sha, "materials": summary, "dynamics": dynamics, "settling": settling,
            "soft_materials": model.get("soft", {}), "fixed_base": model.get("fixed_base", False), **images}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("glb", "materials", "out", "asset-sha"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--intervals")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        intervals = json.loads(Path(args.intervals).read_text()) if args.intervals else None
        report = run(args.glb, args.materials, out, args.asset_sha, intervals)
    except Exception as e:  # the host treats missing evidence as inconclusive
        report = {"asset_sha256": args.asset_sha, "error": f"{type(e).__name__}: {e}"}
    write_json(out / "report.json", report)


if __name__ == "__main__":
    main()
