"""Agent stage of a web-app job: GPT-6-Astra, Claude Opus 5.5 or Claude Fable 5.2 refines the Stage A mesh
(build_plan.md note 5), or gives the parts of an articulated job their joints (note 11).

Runs in envs/sam3d from the repo root, with apptainer on the PATH:

    python -m harpor.webapp.agent_step --job J --sif harpor-agent.sif --key-file F [--agent astra|opus|fable]

It prepares a fresh session in J/agent/s from J/run, runs it with controller.run_container_session and
writes a copy of J/agent/s/result.json, without the key, to J/agent/result.json. The session is a
refinement of posed.glb (600 s) or, for an articulated job (job.json "articulated": true) whose parts
stage made run/parts.glb, an articulation of parts.glb with run/parts.json as input/parts.json (900 s).
Both use every view of cameras.json. --agent astra (the default) runs GPT-6-Astra through the Codex
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
then. The helpers already store the agent's text that way; this also cleans an older result.

An articulation whose parts.json has a found requirement runs in rounds (controller.run_rounds), and the host
checks each round's candidate with MuJoCo, the python of envs/mujoco that GATOR_MUJOCO names (steps.command
exports it). Without that python every round would be sim_unavailable, so the step stops before the session runs.
The session's evidence, root/control/evidence.json (harpor.agent.evidence), holds the agent's reason, limitations and
link names, so it goes to J/agent/evidence.json, which the web app serves, only scrubbed as the result is, and as
strict JSON (finite: browsers' JSON.parse takes no NaN or Infinity, which MuJoCo's states can give). It is written
whether the session ends or stops, since the controller keeps it after every round; agent/result.json is finite too.

SIGTERM (Cancel in the web app) ends the step as it ends a helper of harpor.agent.tools: the Blender
processes the step started are killed, and SystemExit unwinds the controller, whose cleanup kills
the agent's process group (the agent runs in a session of its own, so the signal to this step's
group does not reach it) and stops the tool server. The step then exits with 143 and writes no
J/agent/result.json.
"""

import argparse
import json
import math
import os
import shlex
import shutil
import signal
from pathlib import Path

from harpor.agent.tools import MODELS, printable
from harpor.fileio import read_json, write_json
from harpor.webapp import steps

REDACTED = "[redacted API key]"
AGENTS = {"astra": "codex", "opus": "claude", "fable": "claude"}  # the web app's name -> the controller's
# The claude agent's effort per model. Opus runs at the xhigh of the Astra comparison runs. Fable runs at high,
# Anthropic's advice for Fable 5.2: its xhigh turns run longer and use many more tokens, for uneven gains.
EFFORT = {"opus": "xhigh", "fable": "high"}
MODEL = {a: MODELS[a] for a, c in AGENTS.items() if c == "claude"}  # the claude agent's model (Codex has its own)
BUDGET_S = {"refinement": 600.0, "articulation": 900.0}


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


def check_mujoco():
    """SystemExit unless GATOR_MUJOCO (split like a shell) names a python that exists (see the module docstring)."""
    value = os.environ.get("GATOR_MUJOCO", "")
    try:
        python = shlex.split(value)
    except ValueError:
        python = []
    if not python or shutil.which(python[0]) is None:
        raise SystemExit(f"GATOR_MUJOCO is {value!r}, not the python of envs/mujoco: the host checks this "
                         "articulation's requirements with MuJoCo there. Make it with setup/mujoco.sh")


def print_rounds(r):
    """Print the rounds of a result r in rounds, the outcome counts of its requirements and the stored candidate
    that it delivered."""
    for x in r["rounds"]:
        c = x["candidate"] or {}
        got = f"{x['outcome']}, scope {x['scope']}" if x["outcome"] else "no check"
        if isinstance(c.get("version"), int):
            got += f", candidate v{c['version']:02d} ({c['verdict']})"
        found = list(dict.fromkeys(f["code"] for f in x["findings"]))
        print(f"  round {x['round']} {x['kind']}: {got}" + (f"; findings: {', '.join(found)}" if found else ""))
    n = r.get("requirements") or {}
    print("  requirements: " + ", ".join(f"{n.get(k, 0)} {k}" for k in ("pass", "fail", "inconclusive", "unassessed")))
    c = r.get("candidate")
    print(f"  delivered candidate v{c['version']:02d} of round {c['round']}: {c['why']}" if c else
          "  delivered no stored candidate")


def source(prompt, n, kind, parts=False):
    """The manifest's source line. kind: the job's input (steps.source). parts: the mesh was split into parts."""
    from harpor.webapp.jobs import INPUTS

    return (f"MV-SAM3D from {n} frames from {INPUTS[kind]} of the {prompt}, with Depth Anything 3 cameras; "
            f"pose (scale, shift, small rotation) fitted to the {n} SAM 3 masks"
            + ("; split into parts by SAM 3 part masks (input/parts.json)" if parts else ""))


def mode(job_dir, job):
    """articulation for an articulated job whose parts stage made run/parts.glb, else refinement."""
    return "articulation" if steps.articulated(job) and (Path(job_dir) / "run" / "parts.glb").is_file() \
        else "refinement"


def run(job_dir, sif, key_file, agent="astra"):
    from harpor.agent import controller, physics_review
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
    kind = mode(J, job)
    original, parts = J / "run" / "posed.glb", None
    if kind == "articulation":
        original, parts = J / "run" / "parts.glb", J / "run" / "parts.json"
        if not parts.is_file():
            raise SystemExit(f"no {parts}; the parts stage did not finish")
    views = mvsam3d.session_views(J / "run")
    controller.prepare_session(root, kind, None, views,
                               source=source(job["prompt"], len(views), steps.source(job)["kind"], parts is not None),
                               metric=True, original=original, parts=parts, agent=AGENTS[agent],
                               effort=EFFORT.get(agent), model=MODEL.get(agent))
    physics_review.prepare(root, J, key_file, agent)
    if controller.has_rounds(root, kind):
        check_mujoco()
    try:
        controller.run_container_session(root, sif, key_file, budget_s=BUDGET_S[kind])
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
    if kind == "articulation":
        c = r.get("counts") or {}
        print(f"  articulation: {r.get('joints_delivered')} joints delivered, {c.get('joints', 0)} set_joint changes, "
              f"{c.get('articulation_reviews', 0)} articulation reviews")
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
    a = ap.parse_args(argv)
    from harpor.agent import tools

    signal.signal(signal.SIGTERM, tools._on_sigterm)  # the helpers' own stop path; see the module docstring
    run(a.job, a.sif, a.key_file, a.agent)


if __name__ == "__main__":
    main()
