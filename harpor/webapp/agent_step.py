"""Agent stage of a web-app job: GPT-6.2-Astra, Claude Opus 5.5 or Claude Fable 5.2 refines the Stage A mesh
(README.md note 5).

Runs in envs/sam3d from the repo root, with apptainer on the PATH:

    python -m harpor.webapp.agent_step --job J --sif harpor-agent.sif --key-file F [--agent astra|opus|fable]

It prepares a fresh session in J/agent/s from J/run, runs it with controller.run_container_session and
writes a copy of J/agent/s/result.json, without the key, to J/agent/result.json. The session is a
refinement of posed.glb (900 s for the build round) with every view of cameras.json. --agent astra (the default) runs GPT-6.2-Astra through the Codex
CLI in Apptainer, and F holds OPENAI_API_KEY; --agent opus runs Claude Opus 5.5 through
harpor.agent.claude_agent at xhigh effort, and --agent fable runs Claude Fable 5.2 through it at high
effort (EFFORT; envs/sam3d needs anthropic==1.15.0), and F holds ANTHROPIC_API_KEY. The web app writes
F to /dev/shm for this stage only.

The Codex agent can read the key (from /proc), so its finish text or its commands may hold it. Every
string of the result has each key of F (controller.read_keys) replaced by "[redacted API key]" before
J/agent/result.json is written and before anything is printed (stdout is the stage's log, which the
web app serves). The agent's reason and limitations, and the reasons of rejected GLBs, which can name
the agent's links, are printed as one line without control characters (tools.printable), with the keys
replaced again after that cleaning: a key split by a control character that it drops is whole only
then. The helpers already store the agent's text that way.

The session runs in rounds (controller.run_rounds): the host checks each round's candidate against the original
(harpor.agent.validate), and a failed check feeds a repair round. The session's evidence, root/control/evidence.json (harpor.agent.evidence), holds the
agent's reason and limitations, so it goes to J/agent/evidence.json, which the web app serves, only scrubbed as the
result is, and as
strict JSON (finite: browsers' JSON.parse takes no NaN or Infinity). It is written
whether the session ends or stops, since the controller keeps it after every round; agent/result.json is finite too.

LiDAR geometry (--lidar-geometry off, inspect or sculpt; job.json lidar_geometry, by default
lidar_geometry.DEFAULT_MODE). geometry_request asks for the job's measured returns only when LiDAR is on for the job,
its scale record is current (steps.done), selection, prep and placement agree on the applied metric scale, and the
run's geometry index belongs to the pool's (its sha256) with the same masks. Otherwise the session runs its ordinary
photo-guided refinement, and control/lidar/manifest.json says why the geometry is unavailable. result.json gets
lidar_geometry: the requested and effective modes and the status.

SIGTERM (Cancel in the web app) ends the step as it ends a helper of harpor.agent.tools: the Blender
processes the step started are killed, and SystemExit unwinds the controller, whose cleanup kills
the agent's process group (the agent runs in a session of its own, so the signal to this step's
group does not reach it) and stops the tool server. The step then exits with 143 and writes no
J/agent/result.json.
"""

import argparse
import json
import math
import shutil
import signal
from pathlib import Path

from harpor.agent import lidar_geometry
from harpor.agent.tools import MODELS, printable
from harpor.fileio import read_json, write_json
from harpor.webapp import steps

REDACTED = "[redacted API key]"
AGENTS = {"astra": "codex", "opus": "claude", "fable": "claude"}  # the web app's name -> the controller's
# The claude agent's effort per model. Opus runs at the xhigh of the GPT-6-Astra comparison runs. Fable runs at high,
# Anthropic's advice for Fable 5.2: its xhigh turns run longer and use many more tokens, for uneven gains.
EFFORT = {"opus": "xhigh", "fable": "high"}
MODEL = {a: MODELS[a] for a, c in AGENTS.items() if c == "claude"}  # the claude agent's model (Codex has its own)
BUDGET_S = 900.0  # the build round's, = controller.BUDGET_S; a repair has controller.REPAIR_BUDGET_S


def scrub(x, key):
    """JSON data x with key replaced by REDACTED in every string (dict keys included). key: one key, or
    a list of them (controller.read_keys: an env file can hold both)."""
    keys = [k for k in ([key] if isinstance(key, str) else key or []) if k]
    if isinstance(x, str):
        for k in keys:
            x = x.replace(k, REDACTED)
        return x
    if isinstance(x, list):
        return [scrub(v, keys) for v in x]
    if isinstance(x, dict):
        return {scrub(k, keys): scrub(v, keys) for k, v in x.items()}
    return x


def finite(x):
    """JSON data x with None for every number that is not finite (NaN, Infinity), which Python's json writes and
    browsers' JSON.parse refuses."""
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, list):
        return [finite(v) for v in x]
    if isinstance(x, dict):
        return {k: finite(v) for k, v in x.items()}
    return x


def publish_evidence(root, out, keys):
    """root/control/evidence.json as out, without keys (scrub) and finite. A session without rounds has none, and
    then out stays absent."""
    ev = read_json(Path(root) / "control" / "evidence.json")
    if ev is not None:
        write_json(out, finite(scrub(ev, keys)))


def print_rounds(r):
    """Print the rounds of a result r in rounds and the stored candidate that it delivered."""
    for x in r["rounds"]:
        c = x["candidate"] or {}
        got = f"{x['outcome']}, scope {x['scope']}"
        if isinstance(c.get("version"), int):
            got += f", candidate v{c['version']:02d} ({c['verdict']})"
        found = list(dict.fromkeys(f["code"] for f in x["findings"]))
        print(f"  round {x['round']} {x['kind']}: {got}" + (f"; findings: {', '.join(found)}" if found else ""))
    c = r.get("candidate")
    print(f"  delivered candidate v{c['version']:02d} of round {c['round']}: {c['why']}" if c else
          "  delivered no stored candidate")


def source(prompt, n, kind):
    """The manifest's source line. kind: the job's input (steps.source)."""
    from harpor.webapp.jobs import INPUTS

    return (f"MV-SAM3D from {n} frames from {INPUTS[kind]} of the {prompt}, with Depth Anything 3 cameras; "
            f"pose (scale, shift, small rotation) fitted to the {n} SAM 3 masks")


def _same(a, b):
    return isinstance(a, (int, float)) and isinstance(b, (int, float)) and abs(a - b) <= 1e-9 * max(abs(a), 1.0)


def geometry_request(J, job, requested):
    """The LiDAR geometry request for controller.prepare_session: {requested, index (J/run/lidar/geometry/index.json)
    or None, reason (why it is unavailable)}. See the module docstring for the conditions."""
    import hashlib

    J = Path(J)
    req = {"requested": requested, "index": None, "reason": None}

    def no(why):
        return {**req, "reason": why}

    if requested == "off":
        return req
    if job.get("lidar") is None or steps.lidar(job) is None:
        return no("LiDAR is off for this job")
    if not steps.done("lidar", J):
        return no("the LiDAR scale record is missing, failed or stale; run the LiDAR stage again")
    rec = read_json(J / "pool" / "lidar.json") or {}
    geo = rec.get("geometry")
    if not isinstance(geo, dict) or not geo.get("sha256"):
        return no("this job's LiDAR stage kept no measured returns (a job from before LiDAR geometry); rerun the LiDAR "
                  "and selection stages to upgrade it")
    s = rec.get("s")
    sel = ((read_json(J / "selection.json") or {}).get("metric_scale") or {}).get("s")
    prep = (read_json(J / "run" / "prep.json") or {}).get("metric_scale")
    place = (read_json(J / "run" / "placement.json") or {}).get("metric_scale")
    if not (_same(s, sel) and _same(s, prep) and _same(s, place)):
        return no(f"the metric scale differs between the LiDAR record ({s}), selection ({sel}), the run ({prep}) and "
                  f"placement ({place}); run selection again")
    pool_index = J / "lidar" / "geometry" / "index.json"
    run_index = J / "run" / "lidar" / "geometry" / "index.json"
    if not pool_index.is_file() or hashlib.sha256(pool_index.read_bytes()).hexdigest() != geo["sha256"]:
        return no("the LiDAR returns changed since the scale record; run the LiDAR stage again")
    index = read_json(run_index)
    if not isinstance(index, dict) or index.get("pool_index_sha256") != geo["sha256"]:
        return no("the run's LiDAR returns are not those of the current LiDAR stage; run selection again")
    pool = read_json(pool_index) or {}
    for v in index.get("views") or []:
        want = ((pool.get("views") or {}).get(v.get("pool_name")) or {}).get("mask_sha256")
        mask = J / "run" / "masks" / f"{v.get('index')}.png"
        if want and mask.is_file() and hashlib.sha256(mask.read_bytes()).hexdigest() != want:
            return no(f"view {v.get('index')}'s mask changed since the LiDAR stage; run it and selection again")
    return {**req, "index": str(run_index)}


def run(job_dir, sif, key_file, agent="astra", lidar_mode=None):
    from harpor.agent import controller
    from harpor.stage_a import mvsam3d

    J = Path(job_dir)
    if agent not in AGENTS:
        raise SystemExit(f"--agent is {', '.join(AGENTS)}, not {agent!r}")
    if not Path(key_file).is_file():
        raise SystemExit(f"no key file {key_file}")
    keys = controller.read_keys(key_file)  # every key of the file, not only the one this agent uses
    job = steps.job(J)
    root, out, evidence = J / "agent" / "s", J / "agent" / "result.json", J / "agent" / "evidence.json"
    if root.exists():  # an earlier try that failed or was cancelled; prepare_session needs a fresh folder
        shutil.rmtree(root)
    out.unlink(missing_ok=True)
    evidence.unlink(missing_ok=True)
    views = mvsam3d.session_views(J / "run")
    mode = lidar_mode or job.get("lidar_geometry") or lidar_geometry.DEFAULT_MODE
    if mode not in lidar_geometry.MODES:
        raise SystemExit(f"--lidar-geometry is {', '.join(lidar_geometry.MODES)}, not {mode!r}")
    request = geometry_request(J, job, mode)
    controller.prepare_session(root, None, views, source=source(job["prompt"], len(views), steps.source(job)["kind"]),
                               metric=True, original=J / "run" / "posed.glb", agent=AGENTS[agent],
                               effort=EFFORT.get(agent), model=MODEL.get(agent), rounds=True, lidar=request)
    lid = lidar_geometry.manifest(root / "control") or {}
    print(f"LiDAR geometry: requested {mode}, effective {lid.get('mode', {}).get('effective')}"
          + (f" ({lid['reason']})" if lid.get("reason") else ""), flush=True)
    try:
        controller.run_container_session(root, sif, key_file, budget_s=BUDGET_S)
    finally:
        publish_evidence(root, evidence, keys)
    r = finite(scrub(json.loads((root / "result.json").read_text()), keys))
    write_json(out, r)
    print(f"agent: {r['source']}, verdict {r['verdict']}, exit {r['exit']} ({r['cli_exit']}), "
          f"{r['timings']['wall_s']:.0f} s, GLB {r['glb']}")
    if r.get("model"):
        print(f"  model: {r['model']}" + (f", effort {r['effort']}" if r.get("effort") else ""))
    if r.get("tokens"):
        print(f"  {r.get('turns')} turns, tokens {r['tokens']}, about ${r.get('cost_usd') or 0:.2f}")
    if r.get("rounds"):
        print_rounds(r)
    print(f"  reason: {scrub(printable(r.get('reason')), keys)}")
    for line in r.get("limitations") or []:
        print(f"  limitation: {scrub(printable(line), keys)}")
    for x in r.get("rejected") or []:
        print(f"  rejected {x['from']} {x['glb']}: {scrub(printable(x['why']), keys)}")
    if r.get("network_commands"):
        print(f"  network commands: {r['network_commands']}")
    if r.get("key_found_in"):
        print(f"  WARNING: the API key is in {r['key_found_in']}; these files are never served or packaged")
    return r


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", required=True)
    ap.add_argument("--sif", required=True)
    ap.add_argument("--key-file", required=True,
                    help="env file with OPENAI_API_KEY=... (astra) or ANTHROPIC_API_KEY=... (opus, fable)")
    ap.add_argument("--agent", choices=sorted(AGENTS), default="astra")
    ap.add_argument("--lidar-geometry", choices=lidar_geometry.MODES, default=None,
                    help=f"LiDAR geometry mode; default job.json lidar_geometry, else {lidar_geometry.DEFAULT_MODE}")
    a = ap.parse_args(argv)
    from harpor.agent import tools

    signal.signal(signal.SIGTERM, tools._on_sigterm)  # the helpers' own stop path; see the module docstring
    run(a.job, a.sif, a.key_file, a.agent, a.lidar_geometry)


if __name__ == "__main__":
    main()
