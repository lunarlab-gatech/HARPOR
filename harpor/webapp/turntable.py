"""The turntable stage of a web-app job: a 360 video of the finished object (build_plan.md note 10).

    python -m harpor.webapp.turntable render --job J (--sif SIF | --blender CMD)   (envs/sam3d)
    python -m harpor.webapp.turntable encode --job J                               (envs/web)

render runs blender/turntable.py (build_plan.md note 5) on the GLB that physics used (physics/up.json "glb":
the agent's GLB, else run/parts.glb or run/posed.glb), turned upright with physics/aligned.json "T_s_from_w".
So the video shows the object and pose of aligned.glb, with every part, but with the GLB's own materials, since
align may make aligned.glb opaque for VoMP. Blender runs in the agent's SIF, in a container that sees only
J/turntable, the GLB (read-only) and the Blender scripts (read-only), for at most RENDER_TIMEOUT_S. It writes
FRAMES PNG frames of SIDE px into J/turntable/frames: one turn, seen from ELEVATION_DEG above, with the long side
toward the camera in frame 1. Then J/turntable/render.json. Blender's arguments are J/turntable/args.json, and
its log J/turntable/blender.log. --blender CMD (split like a shell) replaces the container and Blender; fake mode
and the tests pass the fake Blender of harpor.agent.fake_agent.

encode makes J/turntable/turntable.mp4 from the frames with PyAV, as video.py makes the preview: H.264 (libx264,
yuv420p, crf CRF, preset veryfast, a keyframe every 2 s) in MP4 with the index first, at FPS fps, so the video is
FRAMES / FPS s long and loops. It deletes the frames, then writes J/turntable/turntable.json, the stage's done
marker, last: render.json with the file's codec, pixel format, size, frames, length (seconds), bytes and
encode_s. envs/web has no trimesh, so encode must not import harpor.agent.controller.

The soft-part video (harpor.webapp.softvideo) takes the same GLB, matrix, Blender script (run_blender), container
(blender_command with its own folder, since the container sees only that one) and encode, in J/softvideo.
"""

import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from harpor.fileio import read_json, write_json

FRAMES, FPS, SIDE, SAMPLES = 240, 30, 720, 32  # 8 s, one turn; H.264 needs even sides
ELEVATION_DEG = 20  # the camera's angle above the horizontal
CRF = "23"  # sharper than the preview's 28; the codec and preset are video.py's
RENDER_TIMEOUT_S = 1200.0
# Frame S (z up, the long side on x) into turntable.py's W (y down): the camera sees the long side from -y of S.
UPRIGHT_TO_W = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])


def source(job_dir):
    """(the GLB that physics used, the 4x4 matrix that turns it upright into turntable.py's W, the rule that chose
    the GLB): physics/up.json "glb" and "glb_rule", and UPRIGHT_TO_W @ physics/aligned.json "T_s_from_w". SystemExit
    with the reason when align has not run, or when the GLB is gone, a symlink or outside the job folder. The GLB
    never holds the API key: steps.next_glb gives the agent's GLB only through steps.agent_glb, which refuses those."""
    J = Path(job_dir)
    up, aligned = read_json(J / "physics" / "up.json"), read_json(J / "physics" / "aligned.json")
    T = np.asarray(aligned.get("T_s_from_w") if isinstance(aligned, dict) else None, dtype=object)
    if not isinstance(up, dict) or T.shape != (4, 4):
        raise SystemExit("no physics/up.json or physics/aligned.json: run align first")
    glb = Path(str(up.get("glb") or ""))
    if not glb.is_file():
        raise SystemExit(f"the GLB that physics used is gone: {up.get('glb')}")
    if glb.is_symlink() or not glb.resolve().is_relative_to(J.resolve()):
        raise SystemExit(f"the GLB that physics used is not a file of this job: {up.get('glb')}")
    return glb.resolve(), UPRIGHT_TO_W @ T.astype(float), up.get("glb_rule")


def blender_command(job_dir, glb, sif, folder="turntable"):
    """Blender in the agent's SIF, in a container that sees only J/<folder> (read-write; the soft-part video's is
    J/softvideo), the GLB and the Blender scripts (read-only), each at its host path (controller.sif_blender)."""
    from harpor.agent import controller

    return controller.sif_blender(sif, rw=[(Path(job_dir) / folder).resolve()], ro=[glb])


def _count(frames, stop, every=2.0, total=FRAMES):
    """Print PROGRESS (up to 0.9) as Blender writes the total frames, every `every` s, until stop is set."""
    last = None
    while not stop.wait(every):
        p = round(0.9 * min(len(list(frames.glob("*.png"))), total) / total, 2)
        if p != last:
            print(f"PROGRESS {p:.2f}", flush=True)
            last = p


def run_blender(out, blender, args, timeout=RENDER_TIMEOUT_S):
    """Blender (a list) on blender/turntable.py with args, as out/args.json, and its log out/blender.log; PROGRESS
    follows the frames. SystemExit with the reason when Blender fails, passes the timeout or writes fewer than
    args["frames"] frames into args["frames_dir"]."""
    from harpor.agent import tools

    frames, total, log_rel = Path(args["frames_dir"]), args["frames"], f"{out.name}/blender.log"
    (out / "args.json").write_text(json.dumps(args, indent=1))
    cmd = tools.blender_argv(blender, "turntable.py", out / "args.json")
    stop = threading.Event()
    counter = threading.Thread(target=_count, args=(frames, stop), kwargs={"total": total}, daemon=True)
    counter.start()
    try:
        with open(out / "blender.log", "w") as log:
            try:
                code = tools._run(cmd, log, timeout)
            except subprocess.TimeoutExpired:
                raise SystemExit(f"Blender took longer than {timeout:g} s; see {log_rel}") from None
            except OSError as e:  # a Blender that does not start
                raise SystemExit(f"Blender did not start: {e}") from None
    finally:
        stop.set()
        counter.join()
    if code != 0:
        raise SystemExit(f"Blender failed (exit {code}); see {log_rel}")
    n = len(list(frames.glob("*.png")))
    if n < total:
        raise SystemExit(f"Blender wrote {n} of {total} frames; see {log_rel}")


def render(job_dir, sif=None, blender=None, timeout=RENDER_TIMEOUT_S):
    """The frames and render.json (see the module docstring), in a fresh J/turntable. blender: the command that
    stands for Blender (a list or a shell string), else blender_command(J, the GLB, sif). SystemExit with the reason
    when Blender fails, passes the timeout or writes fewer than FRAMES frames (run_blender)."""
    J = Path(job_dir).resolve()
    if blender is None and sif is None:
        raise SystemExit("render needs --sif or --blender")
    glb, matrix, rule = source(J)
    out, t = J / "turntable", time.time()
    shutil.rmtree(out, ignore_errors=True)
    (out / "frames").mkdir(parents=True)
    blender = shlex.split(blender) if isinstance(blender, str) else blender or blender_command(J, glb, sif)
    args = {"assets": [{"path": str(glb), "matrix": matrix.tolist()}], "frames_dir": str(out / "frames"),
            "frames": FRAMES, "fps": FPS, "panel": SIDE, "samples": SAMPLES, "elevation": ELEVATION_DEG}
    run_blender(out, blender, args, timeout)
    rec = {"glb": glb.relative_to(J).as_posix(), "glb_rule": rule, "matrix": matrix.tolist(), "frames": FRAMES,
           "side": SIDE, "fps": FPS, "elevation_deg": ELEVATION_DEG, "render_s": round(time.time() - t, 1)}
    write_json(out / "render.json", rec, indent=1)
    # The rule stays in render.json: an agent-off rule ("turned off: ...") would read as if the video were off.
    print(f"360 video of {rec['glb']}, the GLB that physics used: {FRAMES} frames at {SIDE} px in "
          f"{rec['render_s']:.0f} s", flush=True)
    return rec


def probe(mp4):
    """{codec, pix_fmt, width, height, frames, seconds} of an MP4; the frames are counted by decoding."""
    import av

    with av.open(str(mp4)) as c:
        s = c.streams.video[0]
        n = sum(1 for _ in c.decode(s))
        cc = s.codec_context
        seconds = float(s.duration * s.time_base) if s.duration else n / FPS
        return {"codec": cc.name, "pix_fmt": cc.pix_fmt, "width": cc.width, "height": cc.height, "frames": n,
                "seconds": round(seconds, 3)}


def encode(job_dir, folder="turntable"):
    """<folder>.mp4 from the frames in J/<folder>, then <folder>.json last (see the module docstring); the soft-part
    video's folder is softvideo (harpor.webapp.softvideo)."""
    from fractions import Fraction

    import av
    from PIL import Image

    from harpor.webapp import steps, video

    out = Path(job_dir) / folder
    rec, names = read_json(out / "render.json"), sorted((out / "frames").glob("*.png"))
    if not isinstance(rec, dict) or not names:
        raise SystemExit("no frames to encode: the render step did not finish")
    t, tmp, mp4 = time.time(), out / f".tmp_{folder}.mp4", out / f"{folder}.mp4"
    with Image.open(names[0]) as im:
        w, h = im.size[0] - im.size[0] % 2, im.size[1] - im.size[1] % 2  # yuv420p needs even sides
    with av.open(str(tmp), "w", format="mp4", options={"movflags": "+faststart"}) as c:
        s = c.add_stream(video.PREVIEW_CODEC, rate=FPS, options={**video.PREVIEW_OPTIONS, "crf": CRF, "g": str(2 * FPS)})
        s.width, s.height, s.pix_fmt = w, h, "yuv420p"
        for i, name in enumerate(names):
            with Image.open(name) as im:
                f = av.VideoFrame.from_ndarray(np.asarray(im.convert("RGB")), format="rgb24")
            f = f.reformat(width=w, height=h, format="yuv420p")
            f.pts, f.time_base = i, Fraction(1, FPS)
            for p in s.encode(f):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    facts = probe(tmp)
    if facts["frames"] != len(names):  # turntable.mp4 is there only when it holds every frame
        raise SystemExit(f"the video has {facts['frames']} of {len(names)} frames")
    os.replace(tmp, mp4)
    shutil.rmtree(out / "frames")
    done = {**rec, **facts, "bytes": mp4.stat().st_size, "encode_s": round(time.time() - t, 1)}
    write_json(out / f"{folder}.json", done, indent=1)
    print("PROGRESS 1", flush=True)
    size = f"{done['bytes'] / 1e6:.1f} MB" if done["bytes"] >= 1e6 else f"{done['bytes'] / 1e3:.0f} KB"
    print(f"{steps.OPTIONAL[folder]}: {done['frames']} frames, {w}x{h}, {done['seconds']:.1f} s, {size}", flush=True)
    return done


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="step", required=True)
    r = sub.add_parser("render")
    r.add_argument("--job", required=True)
    r.add_argument("--sif", help="the agent's SIF, with Blender")
    r.add_argument("--blender", help="a command that stands for Blender instead of the SIF (tests)")
    sub.add_parser("encode").add_argument("--job", required=True)
    a = ap.parse_args(argv)
    if a.step == "render":
        from harpor.agent import tools

        signal.signal(signal.SIGTERM, tools._on_sigterm)  # Cancel kills Blender's process group too
        render(a.job, a.sif, a.blender)
    else:
        encode(a.job)


if __name__ == "__main__":
    main()
