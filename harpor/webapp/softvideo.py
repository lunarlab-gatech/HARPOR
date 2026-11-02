"""The softvideo stage of a web-app job: a video of the soft part of the MuJoCo model, pressed and let go (build_plan.md
note 11).

    python -m harpor.webapp.softvideo motion --job J                               (envs/mujoco)
    python -m harpor.webapp.softvideo render --job J (--sif SIF | --blender CMD)   (envs/sam3d)
    python -m harpor.webapp.softvideo encode --job J                               (envs/web)

motion presses the first soft link of physics/mjcf/mjcf.json soft in the mjcf stage's model, physics/mjcf/<name>.xml
(no floor), loaded through MjSpec with the root's free joint deleted and gravity 0: the exported root is free, and
under gravity it drops and tilts by more than the dent. Every hinge and slide is welded too, since the press would
turn a hinged parent that the video shows at rest. It runs harpor.physics.sim_check.press on the press nodes and
direction of the link's binding (soft/<link>.npz) to a visible dent of min(DENT_M, DENT_SHARE t), t the part's depth
along the press direction, with a force cap of CAP E h^2 per node (E the Young's modulus, h the pitch): deeper than
the 3 mm test press, so that it shows at true scale. The visible dent is the residual field carried to the skin by
Eq. 4, as sim_check's visible dmax. FRAMES frames sample the press evenly from rest through the hold, the release and
the settle, so the video loops. J/softvideo/motion.npz holds rest, the link's vertices in S (K, 3), and disp, their
displacement in each frame by Eq. 4 (FRAMES, K, 3), from the binding's rest nodes, so that frame 0 is the rest;
motion.json the link, the target, the cap, the force, whether the cap stopped the press first and the dent reached.
The steps are a fixed count, so motion needs no timeout.

render runs blender/turntable.py with "soft" on the 360 video's GLB and matrix (harpor.webapp.turntable.source): the
object does not turn, the camera frames the soft part at 1.3 times its size, and each frame moves the part's
vertices to UPRIGHT_TO_W @ (rest + disp). Blender runs in the agent's SIF, in a container that sees only
J/softvideo, the GLB and the Blender scripts, for at most turntable.RENDER_TIMEOUT_S, and writes FRAMES PNG frames
into J/softvideo/frames; then J/softvideo/render.json. encode makes J/softvideo/softvideo.mp4 as the 360 video's
(FRAMES / FPS s) and J/softvideo/softvideo.json, the stage's done marker, last. The stage is optional
(harpor.webapp.steps.OPTIONAL): a failure warns, and the job goes on.
"""

import argparse
import shlex
import shutil
import signal
import time
from pathlib import Path

import numpy as np

from harpor.fileio import read_json, write_json
from harpor.webapp import turntable as T

FRAMES, FPS = 90, T.FPS  # 3 s; encode writes the 360 video's rate
DENT_M, DENT_SHARE = 0.012, 0.2  # the visible dent: at most 12 mm, and a fifth of the part's depth
CAP = 0.3  # times E h^2: the force per node


def model(xml):
    """The compiled model of xml with its root fixed (MjSpec: the root's free joint deleted), every hinge and slide
    welded (its joint and its position actuator deleted) and no gravity. An actuator's gains hold its link against
    the link's own inertia, not against the press, so a hinged parent would turn under it, while the video shows the
    parent at rest."""
    import mujoco

    spec = mujoco.MjSpec.from_file(str(xml))
    for body in spec.worldbody.bodies:
        for j in body.joints:
            if j.type == mujoco.mjtJoint.mjJNT_FREE:
                spec.delete(j)
    for a in list(spec.actuators):  # each hinge and slide has one (harpor.physics.mjcf); the flex's node slides do not
        spec.delete(spec.joint(a.target))
        spec.delete(a)
    m = spec.compile()
    m.opt.gravity[:] = 0
    return m


def motion(job_dir):
    """motion.npz and motion.json in a fresh J/softvideo (see the module docstring). SystemExit with the reason when the
    model has no soft part, the part has no press, or a state is not finite."""
    import mujoco

    from harpor.physics import sim_check

    J = Path(job_dir)
    mj, out, t0 = read_json(J / "physics" / "mjcf" / "mjcf.json"), J / "softvideo", time.time()
    if not isinstance(mj, dict) or not mj.get("soft"):
        raise SystemExit("physics/mjcf/mjcf.json has no soft part: run mjcf first")
    link, rec = next(iter(mj["soft"].items()))
    with np.load(J / "physics" / "mjcf" / rec["file"]) as z:
        b = {k: z[k] for k in z.files}
    if not b["direction"].any() or not len(b["press"]):
        raise SystemExit(f"the soft part {link} has no press direction or no press node")
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    m = model(J / "physics" / "mjcf" / f"{mj['name']}.xml")
    rows, weights = b["tets"][b["tet_of_vertex"]], b["weights"]

    def skin(field):  # Eq. 4: a field on the nodes, carried to the link's vertices
        return (field[rows] * weights[..., None]).sum(1)

    depth = float(np.ptp(b["vertices"] @ b["direction"]))
    target, cap = min(DENT_M, DENT_SHARE * depth), CAP * rec["young"] * rec["pitch"] ** 2
    flex = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_FLEX, str(b["flex"]))
    trace = sim_check.press(m, mujoco.MjData(m), flex, b["press"], b["direction"], target, cap,
                            lambda r: sim_check.largest(skin(r)))
    t, P = np.r_[0.0, trace["t"]], np.r_[trace["rest"][None], trace["P"]]
    at = np.abs(t[None] - np.arange(FRAMES)[:, None] * t[-1] / FRAMES).argmin(1)  # frame FRAMES would be rest again
    nodes = P[at] - [0.0, 0.0, float(b["start_z"])] - b["nodes"]  # the root's height (mjcf.check_binding)
    disp = np.stack([skin(x) for x in nodes])
    if not np.isfinite(disp).all():
        raise SystemExit(f"the press of {link} gave a state that is not finite")
    np.savez(out / "motion.npz", rest=b["vertices"], disp=disp.astype(np.float32))
    done = {"link": link, "flex": str(b["flex"]), "frames": FRAMES, "fps": FPS, "press_s": float(t[-1]),
            "depth_m": depth, "target_m": target, "cap_n": cap, "force_n": float(trace["force"].max()),
            "cap_limited": trace["cap_limited"], "dent_m": float(trace["residual"].max()),
            "mujoco": mujoco.__version__, "motion_s": round(time.time() - t0, 1)}
    write_json(out / "motion.json", done, indent=1)
    print(f"soft part {link}: a dent of {1000 * done['dent_m']:.1f} mm (target {1000 * target:.1f} mm) with "
          f"{done['force_n']:.3g} N per node" + (", stopped by the cap" if done["cap_limited"] else ""), flush=True)
    return done


def render(job_dir, sif=None, blender=None, timeout=T.RENDER_TIMEOUT_S):
    """The frames and render.json in J/softvideo, after motion (see the module docstring). blender: as for
    harpor.webapp.turntable.render. SystemExit with the reason when there is no motion, or Blender fails, passes the
    timeout or writes fewer than FRAMES frames."""
    J = Path(job_dir).resolve()
    if blender is None and sif is None:
        raise SystemExit("render needs --sif or --blender")
    out, t = J / "softvideo", time.time()
    moved = read_json(out / "motion.json")
    if not isinstance(moved, dict) or not (out / "motion.npz").is_file():
        raise SystemExit("no softvideo/motion.npz: the motion step did not finish")
    glb, matrix, rule = T.source(J)
    shutil.rmtree(out / "frames", ignore_errors=True)
    (out / "frames").mkdir()
    if isinstance(blender, str):
        blender = shlex.split(blender)
    blender = blender or T.blender_command(J, glb, sif, "softvideo")
    args = {"assets": [{"path": str(glb), "matrix": matrix.tolist()}], "frames_dir": str(out / "frames"),
            "frames": FRAMES, "fps": FPS, "panel": T.SIDE, "samples": T.SAMPLES, "elevation": T.ELEVATION_DEG,
            "soft": {"motion": str(out / "motion.npz"), "matrix": T.UPRIGHT_TO_W.tolist()}}
    T.run_blender(out, blender, args, timeout)
    rec = {**moved, "glb": glb.relative_to(J).as_posix(), "glb_rule": rule, "matrix": matrix.tolist(), "side": T.SIDE,
           "elevation_deg": T.ELEVATION_DEG, "render_s": round(time.time() - t, 1)}
    write_json(out / "render.json", rec, indent=1)
    print(f"soft-part video of {rec['link']} in {rec['glb']}: {FRAMES} frames at {T.SIDE} px in "
          f"{rec['render_s']:.0f} s", flush=True)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="step", required=True)
    sub.add_parser("motion").add_argument("--job", required=True)
    r = sub.add_parser("render")
    r.add_argument("--job", required=True)
    r.add_argument("--sif", help="the agent's SIF, with Blender")
    r.add_argument("--blender", help="a command that stands for Blender instead of the SIF (tests)")
    sub.add_parser("encode").add_argument("--job", required=True)
    a = ap.parse_args(argv)
    if a.step == "motion":
        motion(a.job)
    elif a.step == "render":
        from harpor.agent import tools

        signal.signal(signal.SIGTERM, tools._on_sigterm)  # Cancel kills Blender's process group too
        render(a.job, a.sif, a.blender)
    else:
        T.encode(a.job, "softvideo")


if __name__ == "__main__":
    main()
