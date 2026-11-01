"""Agent stage of a web-app job: GPT-6-Astra or Claude Opus 5.5 refines the Stage A mesh (build_plan.md note 5),
or gives the parts of an articulated job their joints (note 11).

Runs in envs/sam3d from the repo root, with apptainer on the PATH:

    python -m harpor.webapp.agent_step --job J --sif harpor-agent.sif --key-file F [--agent astra|opus]

It prepares a fresh session in J/agent/s from J/run, runs it with controller.run_container_session and
writes a copy of J/agent/s/result.json, without the key, to J/agent/result.json. The session is a
refinement of posed.glb (600 s) or, for an articulated job (job.json "articulated": true) whose parts
stage made run/parts.glb, an articulation of parts.glb with run/parts.json as input/parts.json (900 s).
Both use every view of cameras.json. --agent astra (the default) runs GPT-6-Astra through the Codex
CLI in Apptainer, and F holds OPENAI_API_KEY; --agent opus runs Claude Opus 5.5 through
harpor.agent.claude_agent at xhigh effort (envs/sam3d needs anthropic==1.15.0), and F holds
ANTHROPIC_API_KEY. The web app writes F to /dev/shm for this stage only.

The Codex agent can read the key (from /proc), so its finish text or its commands may hold it. Every
string of the result has each key of F (controller.read_keys) replaced by "[redacted API key]" before
J/agent/result.json is written and before anything is printed (stdout is the stage's log, which the
web app serves).

SIGTERM (Cancel in the web app) ends the step as it ends a helper of harpor.agent.tools: the Blender
processes the step started are killed, and SystemExit unwinds the controller, whose cleanup kills
the agent's process group (the agent runs in a session of its own, so the signal to this step's
group does not reach it) and stops the tool server. The step then exits with 143 and writes no
J/agent/result.json.
"""

import argparse
import json
import shutil
import signal
from pathlib import Path

from harpor.fileio import write_json
from harpor.webapp import steps

REDACTED = "[redacted API key]"
AGENTS = {"astra": "codex", "opus": "claude"}  # the web app's name -> the controller's
EFFORT = {"opus": "xhigh"}
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
    from harpor.agent import controller
    from harpor.stage_a import mvsam3d

    J = Path(job_dir)
    if agent not in AGENTS:
        raise SystemExit(f"--agent is astra or opus, not {agent!r}")
    if not Path(key_file).is_file():
        raise SystemExit(f"no key file {key_file}")
    keys = controller.read_keys(key_file)  # every key of the file, not only the one this agent uses
    job = steps.job(J)
    root, out = J / "agent" / "s", J / "agent" / "result.json"
    if root.exists():  # an earlier try that failed or was cancelled; prepare_session needs a fresh folder
        shutil.rmtree(root)
    out.unlink(missing_ok=True)
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
                               effort=EFFORT.get(agent))
    controller.run_container_session(root, sif, key_file, budget_s=BUDGET_S[kind])
    r = scrub(json.loads((root / "result.json").read_text()), keys)
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
    print(f"  reason: {r.get('reason')}")
    for line in r.get("limitations") or []:
        print(f"  limitation: {line}")
    for x in r.get("rejected") or []:
        print(f"  rejected {x['from']} {x['glb']}: {x['why']}")
    if r.get("network_commands"):
        print(f"  network commands: {r['network_commands']}")
    if r.get("key_found_in"):
        print(f"  WARNING: the API key is in {r['key_found_in']}; these files are never served or packaged")
    return r


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", required=True)
    ap.add_argument("--sif", required=True)
    ap.add_argument("--key-file", required=True, help="env file with OPENAI_API_KEY=... (astra) or ANTHROPIC_API_KEY=... (opus)")
    ap.add_argument("--agent", choices=sorted(AGENTS), default="astra")
    a = ap.parse_args(argv)
    from harpor.agent import tools

    signal.signal(signal.SIGTERM, tools._on_sigterm)  # the helpers' own stop path; see the module docstring
    run(a.job, a.sif, a.key_file, a.agent)


if __name__ == "__main__":
    main()
