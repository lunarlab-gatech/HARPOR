"""Estimate stage of a web-app job: the job's model predicts the mass and the friction of the final mesh in one
request (harpor.physics.estimate).

Runs in envs/sam3d from the repo root, after align:

    python -m harpor.webapp.estimate_step --job J --key-file F --agent astra|opus|fable

It voxelizes J/physics/aligned.glb and measures it. It puts up to estimate.MAX_VIEWS of the job's input photos
(run/cameras.json: view 0 first, the others spread evenly over the rest) in a grid of 2 columns, J/physics/
estimate_views.jpg. Then it sends that image and the prompt to the job's model through harpor.agent.vlm:
GPT-6.2-Astra for astra (F holds OPENAI_API_KEY), Claude Opus 5.5 for opus and Claude Fable 5.2 for fable (F holds
ANTHROPIC_API_KEY). An answer that fails the checks of estimate.parse gets one retry, with the reason added to the
prompt. A second invalid answer or an API error ends the stage.

It writes J/physics/summary.json: the GLB, its measurements ("geometry"), the rigid body that mjcf, usd and urdf
read (the mass spread evenly over the filled voxels of a solid object, or the surface voxels of a hollow one),
the friction, and the checked answer with the model, the attempts, the seconds and the views ("estimate").

The key is read from F into memory only. Each answer has its JSON escapes of plain characters written out
(vlm.unescape) and then every key of F (controller.read_keys) replaced by "[redacted API key]" (agent_step.scrub)
before it is parsed, stored or printed. The checked strings are cleaned to one printable line, and the keys are
replaced again after that: a key split by a character that the cleaning drops is whole only then.
"""

import argparse
import time
from pathlib import Path

from harpor.agent.tools import MODELS, printable
from harpor.fileio import read_json, write_json
from harpor.physics import estimate
from harpor.webapp import steps

AGENTS = ("astra", "opus", "fable")
COLUMNS = 2
TILE = 1024  # each photo fits into TILE x TILE px
IMAGE_SIDE = 2048  # the long side of the image the model gets


def pick_views(n, most=estimate.MAX_VIEWS):
    """The indices of the views to show, out of n: view 0 first, then the middles of equal runs of the rest."""
    k, m = min(most, n) - 1, n - 1
    return [0] + [1 + (2 * j + 1) * m // (2 * k) for j in range(k)] if n else []


def photo(J, cam):
    """The path of a camera's image: by its file name under J/run/input/images first (cameras.json holds the
    paths of the machine that ran place), else the stored path. None when neither is a file."""
    stored = Path(cam["image"])
    for p in (Path(J) / "run" / "input" / "images" / stored.name, stored):
        if p.is_file():
            return p
    return None


def views_image(J, out):
    """Write the grid of the job's input photos to out. The indices of the views shown, in cameras.json."""
    from PIL import Image

    cams = read_json(Path(J) / "run" / "cameras.json")
    if not isinstance(cams, list) or not cams:
        raise SystemExit(f"{J}/run/cameras.json is missing or empty; run the place stage first")
    used, tiles = [], []
    for i in pick_views(len(cams)):
        p = photo(J, cams[i])
        if p is not None:
            im = Image.open(p).convert("RGB")
            im.thumbnail((TILE, TILE), Image.LANCZOS)
            used.append(i)
            tiles.append(im)
    if not tiles:
        raise SystemExit("none of the job's input photos is there")
    cols = min(COLUMNS, len(tiles))
    rows = -(-len(tiles) // cols)
    w, h = max(t.width for t in tiles), max(t.height for t in tiles)
    sheet = Image.new("RGB", (cols * w, rows * h), "white")
    for n, t in enumerate(tiles):
        r, c = divmod(n, cols)
        sheet.paste(t, (c * w + (w - t.width) // 2, r * h + (h - t.height) // 2))
    sheet.save(out, "JPEG", quality=90)
    return used


def metric_from_lidar(J):
    """True when select applied a LiDAR scale: run/placement.json's metric_scale is a number (the web app's
    "Metric scale" row reads it the same way)."""
    m = (read_json(Path(J) / "run" / "placement.json") or {}).get("metric_scale")
    return isinstance(m, (int, float)) and not isinstance(m, bool)


def ask(agent, key, text, image):
    """The answer text of the job's model."""
    from harpor.agent import vlm

    if agent == "astra":
        return vlm.ask_astra(key, text, image)
    return vlm.ask_claude(key, text, image, model=MODELS[agent])


def run(job_dir, key_file, agent):
    """Estimate the mass and the friction of the job's final mesh (see the module docstring). The summary dict."""
    from harpor.agent import controller, vlm
    from harpor.webapp.agent_step import finite, scrub

    J = Path(job_dir)
    if agent not in AGENTS:
        raise SystemExit(f"--agent is {', '.join(AGENTS)}, not {agent!r}")
    if not Path(key_file).is_file():
        raise SystemExit(f"no key file {key_file}")
    keys = controller.read_keys(key_file)  # every key of the file, not only the one this agent uses
    key = vlm.read_key(key_file, agent)
    if not key:
        raise SystemExit(f"the key file has no {steps.KEY_VAR[agent]} line")
    glb, out = J / "physics" / "aligned.glb", J / "physics" / "summary.json"
    sheet = J / "physics" / "estimate_views.jpg"
    out.unlink(missing_ok=True)
    sheet.unlink(missing_ok=True)
    if not glb.is_file():
        raise SystemExit(f"{glb} is missing; run the align stage first")
    try:
        vox = estimate.voxels(glb)
        geom = estimate.geometry(glb, vox)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    print(f"mesh: {' x '.join(f'{x:.3f}' for x in geom['extent_m'])} m, filled volume {geom['filled_volume_m3']:.4g} "
          f"m^3, hull volume {geom['hull_volume_m3']:.4g} m^3, surface area {geom['surface_area_m2']:.4g} m^2")
    views = views_image(J, sheet)
    lidar = metric_from_lidar(J)
    text = estimate.prompt(printable(steps.job(J)["prompt"]), geom, lidar)
    image = vlm.jpeg(sheet, max_side=IMAGE_SIDE)
    model = MODELS[agent]
    start = time.time()
    for attempt in (1, 2):
        answer = scrub(vlm.unescape(ask(agent, key, text, image)), keys)
        try:
            checked = scrub(estimate.parse(answer, geom), keys)
            break
        except ValueError as e:
            why = scrub(printable(e), keys)
            if attempt == 2:
                raise SystemExit(f"{model} gave an invalid answer twice: {why}") from None
            print(f"{model}'s answer was invalid ({why}); asking once more")
            text += (f"\n\nYour previous answer was invalid: {why}. "
                     "Answer again with one JSON object that follows the schema exactly.")
    seconds = time.time() - start
    summary = {"glb": str(glb.resolve()), "geometry": geom,
               "rigid_body": estimate.rigid_body(vox, checked["mass_kg"], checked["structure"]),
               "friction": checked["friction"],
               "estimate": {**checked, "agent": agent, "model": model, "attempts": attempt, "seconds": seconds,
                            "views": views}}
    summary = finite(summary)
    write_json(out, summary, indent=1)
    est = summary["estimate"]
    print(f"estimate: mass {est['mass_kg']:.4g} kg ({est['mass_low_kg']:.4g} to {est['mass_high_kg']:.4g}), friction "
          f"{est['friction']:.3g} ({est['friction_low']:.3g} to {est['friction_high']:.3g}), {est['material']} "
          f"({est['structure']}); {model}, {attempt} attempt{'s' if attempt > 1 else ''}, {seconds:.0f} s, "
          f"scale from {'LiDAR' if lidar else 'depth estimate'}")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", required=True, help="job folder")
    ap.add_argument("--key-file", required=True,
                    help="env file with OPENAI_API_KEY=... (astra) or ANTHROPIC_API_KEY=... (opus, fable)")
    ap.add_argument("--agent", choices=AGENTS, required=True)
    a = ap.parse_args(argv)
    run(a.job, a.key_file, a.agent)


if __name__ == "__main__":
    main()
