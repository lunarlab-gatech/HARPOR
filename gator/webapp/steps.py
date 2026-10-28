"""The stages of a web-app job, the command that runs each one on ICE, and when each one is done.

Every stage is one subprocess, `bash -c SCRIPT`, so the server itself needs only envs/web. The
scripts are those of our ICE runs: the same interpreters by full path and the same environment
lines, with G=$(readlink -f ~/scratch)/gator and the job folder J under G/webapp_runs:

  topics   envs/web     video topics            frames/topics.json (rosbag input; skipped for a video and
                                                a Stray Scanner capture)
  frames   envs/web     video analyze           frames/meta.json
  extract  envs/web     video extract           every pool frame in scene/images
  masks    envs/sam3seg sam3_masks              scene/sam3_<slug>/sam3.json, >= 2 frames masked
  prep     envs/sam3d   mvsam3d prep            pool/prep.json
  da3      envs/da3     mvsam3d da3             pool/da3_output.npz with every pool view
  lidar    envs/web     lidar scale             pool/lidar.json "ok", made with the job's calibration and
                                                its source's lidar_scale.VERSIONS (skipped without LiDAR); the
                                                metric scale from LiDAR, which select applies
  select   envs/web     views select            selection.json "ok" and run/da3_output.npz
           (the job pauses here for review unless auto_continue)
  sam3d    envs/sam3d   mvsam3d sam3d           run/pose.json
  place    envs/sam3d   mvsam3d place           run/placement.json
  propose  envs/sam3d   parts.stage propose     run/proposal.json (skipped for a rigid object); it takes
                                                the job's key unless the parts mode is "user"
           (a "vlm_edit" job pauses here for review until the user confirms the list, whatever
           auto_continue says; a retry after a cancel stops here again)
  parts    envs/sam3seg part_masks, then        run/parts.json, written last (skipped for a rigid object)
           envs/sam3d   parts.stage vote, split
                        (Blender in the SIF) and finish
  agent    envs/sam3d   agent_step --agent A    agent/result.json (skipped below 0.5 mask IoU, or when
                                                the job's agent is "none"); it refines posed.glb (600 s),
                                                or gives the parts of an articulated job their joints
                                                (900 s)
  align    envs/sam3d   physics_auto align      physics/aligned.json (and physics/links/ when articulated)
  vomp     envs/vomp11  physics/vomp_run.py     physics/summary.json (per link when articulated)
  mjcf     envs/mujoco  physics.mjcf            physics/mjcf/mjcf.json with "compiled"
  usd      envs/mujoco  physics.usd             physics/usd/usd.json with "checked"
  urdf     envs/mujoco  physics.urdf            physics/urdf/urdf.json with "checked"
  turntable                                     turntable/turntable.json, written last (skipped unless
           envs/sam3d   turntable render        job.json "turntable"): a 360 video of the GLB that
                        (Blender in the SIF),   physics used, turned upright. Its failure warns and
           envs/web     turntable encode        does not fail the job (OPTIONAL)
  package  envs/web     jobs package            results.zip

With WEBAPP_FAKE=1 every stage runs `python -m gator.webapp.fake_steps STAGE --job J` instead,
with the same outputs, so the app works on a Mac without GPUs.
"""

import json
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

NAMES = ["topics", "frames", "extract", "masks", "prep", "da3", "lidar", "select", "sam3d", "place", "propose", "parts",
         "agent", "align", "vomp", "mjcf", "usd", "urdf", "turntable", "package"]
GPU = ("masks", "da3", "sam3d", "parts", "agent", "vomp", "turntable")
# The stages whose failure leaves the job usable, each with the words that start its warning: the build goes on.
OPTIONAL = {"turntable": "360 video"}
REVIEW_AFTER = "select"
PARTS_AFTER = "propose"  # an unconfirmed "vlm_edit" part list pauses the job here (pause_after)
STAGES = [{"name": n, "gpu": n in GPU, "review_after": n == REVIEW_AFTER} for n in NAMES]
CODE = Path(__file__).resolve().parents[2]  # the gator_clone that fake mode runs from

# Copied from the ICE run scripts of Stage A and of VoMP; keep them exactly.
STAGE_A_ENV = ('HF_HOME=$G/.cache/huggingface TORCH_HOME=$G/.cache/torch TORCH_EXTENSIONS_DIR=$G/.cache/torch_extensions '
               'TORCH_CUDA_ARCH_LIST="8.9;9.0" OMP_NUM_THREADS=16 CONDA_PREFIX=$G/envs/sam3d '
               'CPATH=$G/envs/sam3d/targets/x86_64-linux/include '
               'LIBRARY_PATH=$G/envs/sam3d/targets/x86_64-linux/lib:$G/envs/sam3d/lib PATH=$G/envs/sam3d/bin:$PATH')
SAM3_ENV = "HF_HOME=$G/.cache/huggingface"
VOMP_ACTIVATE = 'eval "$($G/tools/micromamba shell hook -s bash)"; micromamba activate $G/envs/vomp11'
VOMP_ENV = ('CUDA_HOME=$CONDA_PREFIX HF_HOME=$G/.cache/huggingface TORCH_HOME=$G/.cache/torch '
            'TORCH_EXTENSIONS_DIR=$G/.cache/torch_extensions_vomp11 TORCH_CUDA_ARCH_LIST="8.9;9.0" OMP_NUM_THREADS=16 '
            'CPATH=$CONDA_PREFIX/targets/x86_64-linux/include LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/lib:$CONDA_PREFIX/lib')
MARKERS = {"topics": "frames/topics.json", "frames": "frames/meta.json", "prep": "pool/prep.json",
           "sam3d": "run/pose.json", "place": "run/placement.json", "propose": "run/proposal.json",
           "parts": "run/parts.json", "agent": "agent/result.json", "align": "physics/aligned.json",
           "vomp": "physics/summary.json", "turntable": "turntable/turntable.json", "package": "results.zip"}
MIN_IOU = 0.5
# job.json "agent": GPT-6-Astra (Codex CLI), Claude Opus 5.5 (claude_agent) or no agent stage
AGENTS = ("astra", "opus", "none")
PROVIDER = {"astra": "OpenAI", "opus": "Anthropic"}  # whose API key each agent takes
KEY_VAR = {"astra": "OPENAI_API_KEY", "opus": "ANTHROPIC_API_KEY"}  # the key file's line, per agent
# job.json "parts" "mode": the job's VLM lists the parts, the VLM proposes and the user edits, or the user types them
PARTS_MODES = ("vlm", "vlm_edit", "user")


@dataclass
class Config:
    G: Path            # $(readlink -f ~/scratch)/gator: envs/, sif/, gator_clone/, the model clones
    runs: Path         # the job folders
    fake: bool = False
    demos: Path | None = None  # the demos folder (WEBAPP_DEMOS), shown at /demos.html (demos.py)


def config_from_env():
    G = Path(os.environ.get("GATOR_ROOT") or Path("~/scratch").expanduser().resolve() / "gator")
    demos = os.environ.get("WEBAPP_DEMOS")
    return Config(G=G, runs=Path(os.environ.get("WEBAPP_RUNS") or G / "webapp_runs"),
                  fake=os.environ.get("WEBAPP_FAKE") == "1", demos=Path(demos).expanduser() if demos else None)


def job(job_dir):
    """job.json of the job folder."""
    return json.loads((Path(job_dir) / "job.json").read_text())


def _json(path):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else {}


def agent(job):
    """job.json's "agent": astra, opus or none. Jobs made before the choice hold true (astra) or false
    (none), and older ones nothing (astra). ValueError for anything else."""
    a = job.get("agent", True)
    if a is True or a is False:
        return "astra" if a else "none"
    if isinstance(a, str) and a in AGENTS:
        return a
    raise ValueError(f"agent is one of {', '.join(AGENTS)}")


def articulated(job):
    """True when the job builds an articulated object (job.json "articulated"); older jobs are rigid."""
    return job.get("articulated") is True


def parts_mode(job):
    """Who lists the parts (job.json "parts" "mode", PARTS_MODES; default "vlm"). A job without an agent
    has no VLM, so its mode is "user". ValueError for an unknown mode."""
    mode = (job.get("parts") or {}).get("mode") or "vlm"
    if mode not in PARTS_MODES:
        raise ValueError(f"the parts mode is one of {', '.join(PARTS_MODES)}")
    return "user" if agent(job) == "none" else mode


def turntable(job):
    """True when the job asks for the 360 video (job.json "turntable"); older jobs do not."""
    return job.get("turntable") is True


def needs_key(stage, job):
    """True when the stage takes the API key of the job's agent: the agent stage, and the propose stage
    of an articulated job whose VLM lists the parts."""
    if agent(job) == "none":
        return False
    return stage == "agent" or (stage == "propose" and articulated(job) and parts_mode(job) != "user")


def command(stage, job, cfg, key_file=None):
    """(argv, cwd, env additions) for stage of job (the job.json dict). key_file: the env file with the
    API key of the job's agent, for the stages that need it (needs_key); only its path goes on the
    command line."""
    if stage not in NAMES:
        raise ValueError(f"unknown stage {stage!r}")
    key = needs_key(stage, job)
    if key and key_file is None:
        raise ValueError(f"the {stage} stage needs the key file")
    q = shlex.quote
    env = {"PYTHONUNBUFFERED": "1"}  # so PROGRESS lines reach the log as they are printed
    if cfg.fake:
        script = f"exec {q(sys.executable)} -m gator.webapp.fake_steps {stage} --job {q(str(Path(cfg.runs) / job['id']))}"
        if key:
            script += f" --key-file {q(str(key_file))} --agent {agent(job)}"
        return ["bash", "-c", script], str(CODE), env

    lines = ["set -eo pipefail", f"G={q(str(cfg.G))}", f"J={q(str(Path(cfg.runs) / job['id']))}"]
    run = "cd $G/gator_clone && "
    sif = " --sif $G/sif/gator-agent.sif"  # the agent's container, which also runs Blender
    if stage in ("topics", "frames", "extract", "lidar", "select", "package"):
        module = {"topics": "gator.webapp.video topics", "frames": "gator.webapp.video analyze",
                  "extract": "gator.webapp.video extract", "lidar": "gator.webapp.lidar scale",
                  "select": "gator.webapp.views select", "package": "gator.webapp.jobs package"}[stage]
        lines.append(run + f'$G/envs/web/bin/python -m {module} --job "$J"')
    elif stage == "masks":
        # Without the earlier sam3.json, sam3_masks keeps no earlier picks: only the job's overrides pick.
        # --prompt=NOUN in one word, so that a noun such as "-cup" is not read as an option.
        picks = "".join(f" --pick {q(f'{n}={k}')}" for n, k in sorted(job["overrides"].get("instance", {}).items()))
        lines += [f"export {SAM3_ENV}", f'rm -f "$J/scene/sam3_{job["slug"]}/sam3.json"',
                  run + '$G/envs/sam3seg/bin/python -m gator.stage_a.sam3_masks --scene "$J/scene" '
                        f"--prompt={q(job['prompt'])} --overwrite{picks}"]
    elif stage == "vomp":
        # An articulated job's aligned GLB has a GLB per link in physics/links: VoMP runs on each, into physics/vomp.
        links = articulated(job)
        lines += ['rm -rf "$J/physics/work" "$J/physics/materials.npz" "$J/physics/voxels.npz" "$J/physics/summary.json"'
                  + (' "$J/physics/vomp"' if links else ""),
                  VOMP_ACTIVATE, f"export {VOMP_ENV}",
                  'cd $G/VoMP && python $G/gator_clone/gator/physics/vomp_run.py --glb "$J/physics/aligned.glb" --out "$J/physics"'
                  + (' --links "$J/physics/links"' if links else "")]
    elif stage in ("mjcf", "usd", "urdf"):
        from gator.webapp.physics_auto import sim_name

        common = f'$G/envs/mujoco/bin/python -m gator.physics.{stage} --glb "$J/physics/aligned.glb" ' \
                 f'--summary "$J/physics/summary.json" --name {q(sim_name(job["prompt"]))}'
        if stage == "mjcf":
            lines += ['rm -rf "$J/physics/mjcf"', run + common + ' --out "$J/physics/mjcf"']
        else:
            lines.append(run + common + f' --mjcf "$J/physics/mjcf" --out "$J/physics/{stage}"')
    elif stage == "turntable":
        # Blender renders the frames in the agent's SIF (envs/sam3d), and PyAV encodes them in envs/web.
        step = run + '$G/envs/{}/bin/python -m gator.webapp.turntable {} --job "$J"'
        lines += ['rm -rf "$J/turntable"', f"export {STAGE_A_ENV}",
                  step.format("sam3d", "render") + sif, step.format("web", "encode")]
    elif stage == "parts":
        # One script in two envs: SAM 3 part masks in envs/sam3seg, then the vote, the split (Blender in the
        # agent's SIF) and parts.json, the done marker, in envs/sam3d. The old marker is removed first.
        stage_cmd = run + '$G/envs/sam3d/bin/python -m gator.parts.stage {} --job "$J"'
        lines += ['rm -f "$J/run/parts.json" "$J/run/parts.glb"', f"export {SAM3_ENV}",
                  run + '$G/envs/sam3seg/bin/python -m gator.stage_a.part_masks --run "$J/run" '
                        '--proposal "$J/run/proposal.json" --out "$J/run/parts/sam3"',
                  f"export {STAGE_A_ENV}", stage_cmd.format("vote"),
                  stage_cmd.format("split") + sif, stage_cmd.format("finish")]
    else:
        lines.append(f"export {STAGE_A_ENV}")
        py = "$G/envs/da3/bin/python" if stage == "da3" else "$G/envs/sam3d/bin/python"
        keyed = f" --key-file {q(str(key_file))} --agent {agent(job)}" if key else ""
        args = {"prep": f'-m gator.stage_a.mvsam3d prep --scene "$J/scene" --mask-prompt sam3_{job["slug"]} --out "$J/pool"',
                "da3": '-m gator.stage_a.mvsam3d da3 --da3-root $G/Depth-Anything-3 --out "$J/pool"',
                "sam3d": '-m gator.stage_a.mvsam3d sam3d --mv-root $G/MV-SAM3D --sam3d-root $G/sam-3d-objects --out "$J/run"',
                "place": '-m gator.stage_a.mvsam3d place --out "$J/run"',
                "propose": '-m gator.parts.stage propose --job "$J"' + keyed,
                "agent": '-m gator.webapp.agent_step --job "$J"' + sif + keyed,
                "align": '-m gator.webapp.physics_auto align --job "$J"'}[stage]
        if stage == "prep":  # a view that lost its mask must not stay in the pool from an earlier prep
            lines.append('rm -rf "$J/pool/input" "$J/pool/masks" "$J/pool/prep.json"')
        lines.append(run + f"{py} {args}")
    return ["bash", "-c", "\n".join(lines)], str(Path(cfg.G) / "gator_clone"), env


def done(stage, job_dir):
    """True when the stage's outputs are all there (see the module docstring)."""
    J = Path(job_dir)
    if stage in MARKERS:
        return (J / MARKERS[stage]).is_file()
    if stage == "extract":
        names = _json(J / "frames" / "pool.json").get("names")
        names = names.values() if isinstance(names, dict) else names
        return bool(names) and all((J / "scene" / "images" / f"{n}.png").is_file() for n in names)
    if stage == "masks":
        return (J / "scene" / f"sam3_{job(J)['slug']}" / "sam3.json").is_file()
    if stage == "da3":  # kept across mask changes, so it must still cover every view prep made
        import numpy as np

        f = J / "pool" / "da3_output.npz"
        if not f.is_file():
            return False
        with np.load(f) as da:
            seen = {Path(str(p)).stem for p in da["image_files"]}
        return {p.stem for p in (J / "pool" / "input" / "images").glob("*.png")} <= seen
    if stage == "lidar":  # an upload or delete of the calibration since the record, or an older estimator, makes it stale
        from gator.fileio import read_json
        from gator.stage_a.lidar_scale import VERSIONS

        rec = read_json(J / "pool" / "lidar.json")  # None when it does not parse: not done
        if not isinstance(rec, dict) or rec.get("ok") is not True:
            return False
        meta = _json(J / "job.json")
        kind = meta["lidar"].get("kind") if isinstance(meta.get("lidar"), dict) else None  # each source has its version
        if not isinstance(kind, str) or kind not in VERSIONS or rec.get("version") != VERSIONS[kind]:
            return False
        cal = meta.get("calibration") or {}
        return (rec.get("extrinsic") or {}).get("calibration_sha256") == cal.get("sha256")
    if stage == "select":
        return _json(J / "selection.json").get("ok") is True and (J / "run" / "da3_output.npz").is_file()
    if stage == "mjcf":
        return "compiled" in _json(J / "physics" / "mjcf" / "mjcf.json")
    if stage in ("usd", "urdf"):
        return "checked" in _json(J / "physics" / stage / f"{stage}.json")
    raise ValueError(f"unknown stage {stage!r}")


def source(job):
    """job.json's "source" ({"kind": "video", "rosbag" or "stray", "topic"}); jobs made before rosbag input
    are videos."""
    return job.get("source") or {"kind": "video", "topic": None}


def lidar(job):
    """job.json's "lidar": None (missing, null or false: no LiDAR, as in older jobs),
    {"kind": "rosbag", "topic": T}, a LiDAR topic of the job's bag, or {"kind": "stray"}, the depth of a
    Stray Scanner capture. ValueError for anything else."""
    v = job.get("lidar")
    if v is None or v is False:
        return None
    if isinstance(v, dict) and v.get("kind") == "rosbag" and isinstance(v.get("topic"), str) and v["topic"]:
        return {"kind": "rosbag", "topic": v["topic"]}
    if isinstance(v, dict) and v.get("kind") == "stray":
        return {"kind": "stray"}
    raise ValueError('lidar is null, {"kind": "rosbag", "topic": T} or {"kind": "stray"}')


def guard(stage, job_dir):
    """The reason a stage that ran (or failed) leaves the job unusable, else None."""
    J = Path(job_dir)
    if stage == "lidar":
        from gator.fileio import read_json

        rec = read_json(J / "pool" / "lidar.json")  # None when it does not parse
        if not isinstance(rec, dict) or not rec or rec.get("ok") is True:
            return None
        return f"LiDAR scale: {rec.get('reason') or 'no reason given'}"
    if stage == "topics":
        found = _json(J / "frames" / "topics.json")
        if not found or found.get("topics"):
            return None
        left = ", ".join(f"{s['topic']} ({s['why']})" for s in found.get("skipped", []))
        return "no camera image topics in the bag" + (f"; not usable: {left}" if left else "")
    if stage != "masks":
        return None
    j = job(J)
    rec = _json(J / "scene" / f"sam3_{j['slug']}" / "sam3.json")
    if not rec:
        return None
    k = sum(v.get("mask") is not None for v in rec["views"])
    return f"SAM 3 found {j['prompt']} in only {k} frames; try another noun" if k < 2 else None


def skip(stage, job_dir):
    """The reason to skip a stage, else None: topics for a video and a Stray Scanner capture, lidar without
    LiDAR, propose and parts for a rigid object, the agent when the job has none or on a poor Stage A
    placement, and turntable unless the job asks for the 360 video. A malformed "lidar" is not skipped: the
    stage fails with the reason."""
    job = _json(Path(job_dir) / "job.json")
    if stage == "topics":
        return {"rosbag": None, "stray": "Stray Scanner capture"}.get(source(job)["kind"], "video input")
    if stage == "lidar":
        try:
            return None if lidar(job) else "turned off: no LiDAR"
        except ValueError:
            return None
    if stage in ("propose", "parts"):
        return None if articulated(job) else "turned off: rigid object"
    if stage == "turntable":
        return None if turntable(job) else "turned off"
    if stage != "agent":
        return None
    if agent(job) == "none":
        return ("turned off: without the agent the parts get no joints, so physics welds them into one body"
                if articulated(job) else "turned off: physics uses the Stage A mesh")
    iou = _json(Path(job_dir) / "run" / "placement.json").get("check", {}).get("mean_mask_iou")
    if iou is not None and iou < MIN_IOU:
        return f"Stage A mean mask IoU {iou:.2f} is below {MIN_IOU}; physics uses " + \
            ("run/parts.glb, whose parts get no joints" if articulated(job) else "run/posed.glb")
    return None


def pause_after(stage, job_dir):
    """True when the job stops for review after the stage: after propose, while run/proposal.json holds the
    VLM's list from mode "vlm_edit" (its "source" and "mode") and the user has not confirmed it yet. The
    proposal decides, not the job's agent now, so a change of agent cannot skip the check; auto_continue
    does not matter either."""
    J = Path(job_dir)
    if stage != PARTS_AFTER or not articulated(_json(J / "job.json")):
        return False
    p = _json(J / MARKERS[stage])
    return p.get("source") == "vlm" and p.get("mode") == "vlm_edit" and p.get("confirmed") is not True


def waits_for_parts(job_dir, job=None):
    """True while the job waits in review for the user to confirm or edit the VLM's part list
    (pause_after propose). job: job.json's dict, if the caller has read it."""
    job = job if job is not None else _json(Path(job_dir) / "job.json")
    return job.get("status") == "review" and (job.get("stages") or {}).get(PARTS_AFTER, {}).get("status") == "done" \
        and pause_after(PARTS_AFTER, job_dir)


def agent_glb(job_dir):
    """The agent's deliverable when it gave one, else None.

    A fallback verdict, a missing file, a symlink or a file that holds the API key (result
    key_found_in) all mean None.
    """
    J = Path(job_dir)
    r = _json(J / "agent" / "result.json")
    glb = r.get("glb")
    if r.get("source") and r.get("verdict") not in (None, "fallback_original") and glb \
            and glb not in (r.get("key_found_in") or []):
        p = J / "agent" / "s" / glb
        if p.is_file() and not p.is_symlink() and p.resolve().is_relative_to((J / "agent" / "s").resolve()):
            return p
    return None


def next_glb(job_dir):
    """The GLB for physics: the agent's deliverable (agent_glb), else run/parts.glb for an articulated
    job whose parts stage made it (its parts have no joints), else run/posed.glb."""
    J = Path(job_dir)
    glb = agent_glb(J)
    if glb is not None:
        return glb
    if articulated(_json(J / "job.json")) and (J / "run" / "parts.glb").is_file():
        return J / "run" / "parts.glb"
    return J / "run" / "posed.glb"
