import json
import os
import re
import secrets
import signal
import socket
import stat
import subprocess
import sys
import textwrap
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gator.fileio import write_json
from gator.webapp import agent_step, jobs, steps

ROOT = Path(__file__).resolve().parents[1]
KEY = "sk-fake-" + secrets.token_hex(16)
ICE = steps.Config(G=Path("/storage/ice1/5/0/ali497/gator"), runs=Path("/storage/ice1/5/0/ali497/gator/webapp_runs"))
JOB = {"id": "20261015-120000-abcdef", "prompt": "coffee cup", "slug": "coffee_cup",
       "overrides": {"locked": [], "excluded": [], "instance": {"f00012": "1", "f00020": "none"}}}
ART = {**JOB, "articulated": True, "parts": {"mode": "vlm", "list": [], "confirmed": False}}  # an articulated job
# The spec's environment lines, copied from our ICE scripts.
STAGE_A = ('HF_HOME=$G/.cache/huggingface TORCH_HOME=$G/.cache/torch TORCH_EXTENSIONS_DIR=$G/.cache/torch_extensions '
           'TORCH_CUDA_ARCH_LIST="8.9;9.0" OMP_NUM_THREADS=16 CONDA_PREFIX=$G/envs/sam3d '
           'CPATH=$G/envs/sam3d/targets/x86_64-linux/include '
           'LIBRARY_PATH=$G/envs/sam3d/targets/x86_64-linux/lib:$G/envs/sam3d/lib PATH=$G/envs/sam3d/bin:$PATH')
VOMP = ('eval "$($G/tools/micromamba shell hook -s bash)"; micromamba activate $G/envs/vomp11',
        'export CUDA_HOME=$CONDA_PREFIX HF_HOME=$G/.cache/huggingface TORCH_HOME=$G/.cache/torch '
        'TORCH_EXTENSIONS_DIR=$G/.cache/torch_extensions_vomp11 TORCH_CUDA_ARCH_LIST="8.9;9.0" OMP_NUM_THREADS=16 '
        'CPATH=$CONDA_PREFIX/targets/x86_64-linux/include LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/lib:$CONDA_PREFIX/lib')


@pytest.fixture
def make_runner(tmp_path):
    made = []

    def make(fake=True, **kw):
        cfg = steps.Config(G=tmp_path / "G", runs=tmp_path / "runs", fake=fake)
        r = jobs.Runner(cfg, key_dir=tmp_path / "keys", **kw)
        made.append(r)
        return r, cfg

    yield make
    for r in made:
        r.shutdown(timeout=5)


def wait(J, *statuses, timeout=90):
    t0 = time.time()
    while time.time() - t0 < timeout:
        job = jobs.load(J)
        if job["status"] in statuses:
            return job
        time.sleep(0.1)
    raise AssertionError(f"job is {job['status']}, not {statuses}: {job['error']} {job['stages']}")


def ready_job(cfg, prompt="box", **fields):
    """A job whose frames stage is done (by its marker), for runner tests with stub commands."""
    job_id, _ = jobs.create_job(cfg.runs, "v.mov")
    J = cfg.runs / job_id
    (J / "frames").mkdir()
    (J / "frames" / "meta.json").write_text("{}")
    jobs.update_stage(J, "frames", status="done")
    jobs.update(J, status="ready", prompt=prompt, **fields)
    return job_id, J


def all_files(root):
    return [p for p in Path(root).rglob("*") if p.is_file() and not p.is_symlink()]


def test_commands_have_the_ice_environment():
    scripts = {s: steps.command(s, JOB, ICE, key_file="/dev/shm/gator-webapp-1/k.env") for s in steps.NAMES
               if s not in ("mjcf", "usd", "urdf")}
    for s, (argv, cwd, env) in scripts.items():
        assert argv[:2] == ["bash", "-c"] and cwd == "/storage/ice1/5/0/ali497/gator/gator_clone"
        assert argv[2].startswith("set -eo pipefail\nG=/storage/ice1/5/0/ali497/gator\n"
                                  "J=/storage/ice1/5/0/ali497/gator/webapp_runs/20261015-120000-abcdef\n")
        assert ("export " + STAGE_A in argv[2]) == (s in ("prep", "da3", "sam3d", "place", "propose", "parts", "agent",
                                                          "align", "turntable")), s
        assert env == {"PYTHONUNBUFFERED": "1"} and KEY not in argv[2]
    sc = {s: v[0][2] for s, v in scripts.items()}
    assert sc["masks"].endswith("\nexport HF_HOME=$G/.cache/huggingface\nrm -f \"$J/scene/sam3_coffee_cup/sam3.json\"\n"
                                "cd $G/gator_clone && $G/envs/sam3seg/bin/python -m gator.stage_a.sam3_masks --scene "
                                "\"$J/scene\" --prompt='coffee cup' --overwrite --pick f00012=1 --pick f00020=none")
    assert sc["prep"].endswith('cd $G/gator_clone && $G/envs/sam3d/bin/python -m gator.stage_a.mvsam3d prep '
                               '--scene "$J/scene" --mask-prompt sam3_coffee_cup --out "$J/pool"')
    assert sc["da3"].endswith('$G/envs/da3/bin/python -m gator.stage_a.mvsam3d da3 --da3-root $G/Depth-Anything-3 --out "$J/pool"')
    assert sc["sam3d"].endswith("$G/envs/sam3d/bin/python -m gator.stage_a.mvsam3d sam3d --mv-root $G/MV-SAM3D "
                                '--sam3d-root $G/sam-3d-objects --out "$J/run"')
    assert sc["agent"].endswith('-m gator.webapp.agent_step --job "$J" --sif $G/sif/gator-agent.sif '
                                "--key-file /dev/shm/gator-webapp-1/k.env --agent astra")
    opus = steps.command("agent", {**JOB, "agent": "opus"}, ICE, key_file="/dev/shm/gator-webapp-1/k.env")[0][2]
    assert opus == sc["agent"][:-len("astra")] + "opus"
    assert "\n".join(VOMP) in sc["vomp"] and sc["vomp"].endswith(
        'cd $G/VoMP && python $G/gator_clone/gator/physics/vomp_run.py --glb "$J/physics/aligned.glb" --out "$J/physics"')
    assert sc["topics"].endswith('$G/envs/web/bin/python -m gator.webapp.video topics --job "$J"')
    assert sc["frames"].endswith('$G/envs/web/bin/python -m gator.webapp.video analyze --job "$J"')
    assert sc["package"].endswith('$G/envs/web/bin/python -m gator.webapp.jobs package --job "$J"')
    # The 360 video: Blender renders the frames in the SIF (envs/sam3d), and envs/web encodes them.
    assert sc["turntable"].endswith(
        f'\nrm -rf "$J/turntable"\nexport {STAGE_A}\n'
        'cd $G/gator_clone && $G/envs/sam3d/bin/python -m gator.webapp.turntable render --job "$J" '
        '--sif $G/sif/gator-agent.sif\ncd $G/gator_clone && $G/envs/web/bin/python -m gator.webapp.turntable encode --job "$J"')
    assert sc["lidar"] == ('set -eo pipefail\nG=/storage/ice1/5/0/ali497/gator\n'
                           'J=/storage/ice1/5/0/ali497/gator/webapp_runs/20261015-120000-abcdef\n'
                           'cd $G/gator_clone && $G/envs/web/bin/python -m gator.webapp.lidar scale --job "$J"')
    with pytest.raises(ValueError):
        steps.command("agent", JOB, ICE)
    fake_cfg = steps.Config(G=ICE.G, runs=ICE.runs, fake=True)
    fake = steps.command("agent", JOB, fake_cfg, key_file="/k/x.env")
    assert fake[0][2] == (f"exec {sys.executable} -m gator.webapp.fake_steps agent --job "
                          "/storage/ice1/5/0/ali497/gator/webapp_runs/20261015-120000-abcdef --key-file /k/x.env --agent astra")
    assert fake[1] == str(ROOT)

    # The parts stages. propose takes the key file only when the VLM lists the parts.
    assert sc["propose"].endswith('\ncd $G/gator_clone && $G/envs/sam3d/bin/python -m gator.parts.stage propose --job "$J"')
    vlm = steps.command("propose", {**ART, "agent": "opus"}, ICE, key_file="/dev/shm/gator-webapp-1/k.env")[0][2]
    assert vlm == sc["propose"] + " --key-file /dev/shm/gator-webapp-1/k.env --agent opus"
    with pytest.raises(ValueError, match="the propose stage needs the key file"):
        steps.command("propose", ART, ICE)
    for typed in ({**ART, "parts": {**ART["parts"], "mode": "user"}}, {**ART, "agent": "none"}):
        assert steps.command("propose", typed, ICE)[0][2] == sc["propose"]
        assert steps.command("propose", typed, fake_cfg)[0][2].endswith("fake_steps propose --job "
                                                                         "/storage/ice1/5/0/ali497/gator/webapp_runs/"
                                                                         "20261015-120000-abcdef")
    assert steps.command("propose", ART, fake_cfg, key_file="/k/x.env")[0][2].endswith(
        "fake_steps propose --job /storage/ice1/5/0/ali497/gator/webapp_runs/20261015-120000-abcdef "
        "--key-file /k/x.env --agent astra")
    stage = 'cd $G/gator_clone && $G/envs/sam3d/bin/python -m gator.parts.stage '
    assert sc["parts"].endswith(
        '\nrm -f "$J/run/parts.json" "$J/run/parts.glb"\nexport HF_HOME=$G/.cache/huggingface\n'
        'cd $G/gator_clone && $G/envs/sam3seg/bin/python -m gator.stage_a.part_masks --run "$J/run" '
        '--proposal "$J/run/proposal.json" --out "$J/run/parts/sam3"\n'
        f'export {STAGE_A}\n{stage}vote --job "$J"\n{stage}split --job "$J" --sif $G/sif/gator-agent.sif\n'
        f'{stage}finish --job "$J"')
    # An articulated job's VoMP runs per link, into physics/vomp; a rigid job's command is as before.
    assert steps.command("vomp", ART, ICE)[0][2] == sc["vomp"].replace(
        '"$J/physics/summary.json"\n', '"$J/physics/summary.json" "$J/physics/vomp"\n') + ' --links "$J/physics/links"'
    assert '"$J/physics/links"' not in sc["vomp"] and '"$J/physics/vomp"' not in sc["vomp"]


def test_masks_command_takes_a_noun_that_starts_with_a_dash(monkeypatch):
    import shlex

    from gator.stage_a import sam3_masks

    ran = []
    monkeypatch.setattr(sam3_masks, "run", lambda scene, prompts, picks, threshold, overwrite: ran.append(prompts))
    for noun in ("-cup", "--mug", "coffee cup"):
        script = steps.command("masks", {**JOB, "prompt": noun}, ICE)[0][2]
        sam3_masks.main(shlex.split(script.split("-m gator.stage_a.sam3_masks ", 1)[1]))
        assert ran[-1] == [noun]


def test_physics_commands_name_the_object():
    pytest.importorskip("gator.webapp.physics_auto")
    mjcf, usd = (steps.command(s, JOB, ICE)[0][2] for s in ("mjcf", "usd"))
    assert mjcf.endswith('rm -rf "$J/physics/mjcf"\ncd $G/gator_clone && $G/envs/mujoco/bin/python -m gator.physics.mjcf '
                         '--glb "$J/physics/aligned.glb" --summary "$J/physics/summary.json" --name coffee_cup '
                         '--out "$J/physics/mjcf"')
    assert usd.endswith('$G/envs/mujoco/bin/python -m gator.physics.usd --glb "$J/physics/aligned.glb" --summary '
                        '"$J/physics/summary.json" --name coffee_cup --mjcf "$J/physics/mjcf" --out "$J/physics/usd"')
    for job in (JOB, ART):  # one link for a rigid object
        assert steps.command("urdf", job, ICE)[0][2].endswith(
            'cd $G/gator_clone && $G/envs/mujoco/bin/python -m gator.physics.urdf --glb "$J/physics/aligned.glb" '
            '--summary "$J/physics/summary.json" --name coffee_cup --mjcf "$J/physics/mjcf" --out "$J/physics/urdf"')


def test_stage_scripts_set_their_environment(tmp_path):
    """The scripts run as written: stand-in interpreters and micromamba report what they were given."""
    G = tmp_path / "G"
    report = textwrap.dedent("""\
        #!/bin/bash
        echo "ran $0 $*"
        echo "pwd $(pwd -P)"
        for v in HF_HOME CONDA_PREFIX TORCH_CUDA_ARCH_LIST TORCH_EXTENSIONS_DIR CPATH LIBRARY_PATH CUDA_HOME; do echo "$v=${!v}"; done
        echo "PATH0=${PATH%%:*}"
        """)
    for env in ("sam3d", "sam3seg", "vomp11"):
        p = G / "envs" / env / "bin" / "python"
        p.parent.mkdir(parents=True)
        p.write_text(report)
        p.chmod(0o755)
    (G / "tools").mkdir()
    (G / "tools" / "micromamba").write_text(
        "#!/bin/bash\necho 'micromamba() { export CONDA_PREFIX=\"$2\"; export PATH=\"$2/bin:$PATH\"; }'\n")
    (G / "tools" / "micromamba").chmod(0o755)
    for d in ("gator_clone", "VoMP"):
        (G / d).mkdir()
    cfg = steps.Config(G=G, runs=tmp_path / "runs")
    J = cfg.runs / JOB["id"]

    def run(stage):
        argv, cwd, env = steps.command(stage, JOB, cfg)
        p = subprocess.run(argv, cwd=cwd, env={**os.environ, **env}, capture_output=True, text=True)
        assert p.returncode == 0, p.stderr
        return dict(line.split("=", 1) for line in p.stdout.splitlines()[2:]), p.stdout.splitlines()[:2]

    env, (ran, pwd) = run("prep")
    G = G.resolve()
    assert ran == f"ran {G}/envs/sam3d/bin/python -m gator.stage_a.mvsam3d prep --scene {J}/scene " \
                  f"--mask-prompt sam3_coffee_cup --out {J}/pool" and pwd == f"pwd {G}/gator_clone"
    assert env["CONDA_PREFIX"] == f"{G}/envs/sam3d" and env["TORCH_CUDA_ARCH_LIST"] == "8.9;9.0"
    assert env["PATH0"] == f"{G}/envs/sam3d/bin" and env["TORCH_EXTENSIONS_DIR"] == f"{G}/.cache/torch_extensions"
    assert env["LIBRARY_PATH"] == f"{G}/envs/sam3d/targets/x86_64-linux/lib:{G}/envs/sam3d/lib"
    env, (ran, pwd) = run("vomp")
    assert ran == f"ran {G}/envs/vomp11/bin/python {G}/gator_clone/gator/physics/vomp_run.py " \
                  f"--glb {J}/physics/aligned.glb --out {J}/physics" and pwd == f"pwd {G}/VoMP"
    assert env["CUDA_HOME"] == env["CONDA_PREFIX"] == f"{G}/envs/vomp11"
    assert env["TORCH_EXTENSIONS_DIR"] == f"{G}/.cache/torch_extensions_vomp11"
    assert env["CPATH"] == f"{G}/envs/vomp11/targets/x86_64-linux/include" and env["HF_HOME"] == f"{G}/.cache/huggingface"

    # The parts stage: part masks in envs/sam3seg with HF_HOME only, then three steps in envs/sam3d.
    argv, cwd, env = steps.command("parts", ART, cfg)
    p = subprocess.run(argv, cwd=cwd, env={**os.environ, **env}, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    out = p.stdout.splitlines()
    ran = [line for line in out if line.startswith("ran ")]
    sam3d = f"ran {G}/envs/sam3d/bin/python -m gator.parts.stage"
    assert ran == [f"ran {G}/envs/sam3seg/bin/python -m gator.stage_a.part_masks --run {J}/run --proposal "
                   f"{J}/run/proposal.json --out {J}/run/parts/sam3", f"{sam3d} vote --job {J}",
                   f"{sam3d} split --job {J} --sif {G}/sif/gator-agent.sif", f"{sam3d} finish --job {J}"]
    blocks = [dict(line.split("=", 1) for line in out[k + 2:k + 10]) for k, line in enumerate(out) if line.startswith("ran ")]
    assert all(b["HF_HOME"] == f"{G}/.cache/huggingface" for b in blocks)
    assert blocks[0]["PATH0"] != f"{G}/envs/sam3d/bin" and blocks[0]["TORCH_EXTENSIONS_DIR"] != f"{G}/.cache/torch_extensions"
    assert all(b["CONDA_PREFIX"] == f"{G}/envs/sam3d" and b["PATH0"] == f"{G}/envs/sam3d/bin" for b in blocks[1:])


def test_config_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("GATOR_ROOT", str(tmp_path))
    monkeypatch.delenv("WEBAPP_RUNS", raising=False)
    monkeypatch.setenv("WEBAPP_FAKE", "1")
    cfg = steps.config_from_env()
    assert cfg.G == tmp_path and cfg.runs == tmp_path / "webapp_runs" and cfg.fake
    monkeypatch.setenv("WEBAPP_RUNS", str(tmp_path / "r"))
    monkeypatch.delenv("WEBAPP_FAKE")
    assert steps.config_from_env().runs == tmp_path / "r" and not steps.config_from_env().fake
    assert [s["name"] for s in steps.STAGES if s["gpu"]] == ["masks", "da3", "sam3d", "parts", "agent", "vomp", "turntable"]
    assert [s["name"] for s in steps.STAGES if s["review_after"]] == ["select"]


def test_done_markers_guard_and_skip(tmp_path):
    J = tmp_path / "job"
    J.mkdir()
    (J / "job.json").write_text(json.dumps({"prompt": "coffee cup", "slug": "coffee_cup"}))
    assert not any(steps.done(s, J) for s in steps.NAMES)
    (J / "frames").mkdir()
    (J / "frames" / "pool.json").write_text(json.dumps({"pool": [3, 9], "names": {"3": "f00003", "9": "f00009"}}))
    (J / "scene" / "images").mkdir(parents=True)
    (J / "scene" / "images" / "f00003.png").write_bytes(b"")
    assert not steps.done("extract", J)
    (J / "scene" / "images" / "f00009.png").write_bytes(b"")
    assert steps.done("extract", J)
    sam = J / "scene" / "sam3_coffee_cup"
    sam.mkdir()
    (sam / "sam3.json").write_text(json.dumps({"views": [{"name": "f00003", "mask": "f00003.png"},
                                                         {"name": "f00009", "mask": None}]}))
    assert steps.done("masks", J)
    assert steps.guard("masks", J) == "SAM 3 found coffee cup in only 1 frames; try another noun"
    (J / "pool" / "input" / "images").mkdir(parents=True)
    for n in ("f00003", "f00009"):
        (J / "pool" / "input" / "images" / f"{n}.png").write_bytes(b"")
    np.savez(J / "pool" / "da3_output.npz", image_files=np.array(["/x/f00003.png"]))
    assert not steps.done("da3", J)  # a view DA3 has not seen
    np.savez(J / "pool" / "da3_output.npz", image_files=np.array(["/x/f00003.png", "/x/f00009.png"]))
    assert steps.done("da3", J)
    (J / "selection.json").write_text(json.dumps({"ok": False}))
    (J / "run").mkdir()
    (J / "run" / "da3_output.npz").write_bytes(b"")
    assert not steps.done("select", J)
    (J / "selection.json").write_text(json.dumps({"ok": True}))
    assert steps.done("select", J)
    # The 360 video: skipped unless the job asks for it (older jobs do not); done when turntable.json is there.
    assert steps.skip("turntable", J) == "turned off" and not steps.turntable({"turntable": "yes"})
    (J / "job.json").write_text(json.dumps({"prompt": "coffee cup", "slug": "coffee_cup", "turntable": False}))
    assert steps.skip("turntable", J) == "turned off"
    (J / "job.json").write_text(json.dumps({"prompt": "coffee cup", "slug": "coffee_cup", "turntable": True}))
    assert steps.skip("turntable", J) is None and steps.turntable(jobs.load(J))
    (J / "turntable").mkdir()
    (J / "turntable" / "turntable.mp4").write_bytes(b"")
    assert not steps.done("turntable", J)
    (J / "turntable" / "turntable.json").write_text("{}")
    assert steps.done("turntable", J) and steps.guard("turntable", J) is None
    assert steps.OPTIONAL == {"turntable": "360 video"} and jobs.DERIVED["turntable"] == ("turntable",)
    (J / "physics" / "mjcf").mkdir(parents=True)
    (J / "physics" / "mjcf" / "mjcf.json").write_text(json.dumps({"name": "x"}))
    assert not steps.done("mjcf", J)
    (J / "physics" / "mjcf" / "mjcf.json").write_text(json.dumps({"name": "x", "compiled": {}}))
    assert steps.done("mjcf", J)
    assert steps.skip("agent", J) is None
    (J / "run" / "placement.json").write_text(json.dumps({"check": {"mean_mask_iou": 0.31}}))
    assert "0.31 is below 0.5" in steps.skip("agent", J) and steps.skip("align", J) is None


def test_parts_stages_done_skip_and_pause(tmp_path):
    J = tmp_path
    job = {"prompt": "toy box", "slug": "toy_box", "agent": "astra"}
    (J / "job.json").write_text(json.dumps(job))
    (J / "run").mkdir()
    assert steps.skip("propose", J) == steps.skip("parts", J) == "turned off: rigid object"  # also an older job
    assert not steps.articulated(job) and steps.needs_key("agent", job) and not steps.needs_key("propose", job)
    assert steps.parts_mode(job) == "vlm" and not steps.needs_key("masks", job)
    for mode, key in (("vlm", True), ("vlm_edit", True), ("user", False)):
        art = {**job, "articulated": True, "parts": {"mode": mode, "list": [], "confirmed": False}}
        none = {**art, "agent": "none"}  # no VLM: the user types the parts, and no stage takes a key
        assert steps.parts_mode(art) == mode and steps.needs_key("propose", art) is key and steps.needs_key("agent", art)
        assert steps.parts_mode(none) == "user" and not steps.needs_key("propose", none) and not steps.needs_key("agent", none)
    with pytest.raises(ValueError, match="the parts mode is one of vlm, vlm_edit, user"):
        steps.parts_mode({"parts": {"mode": "auto"}})

    # An articulated job pauses after propose while its proposal is the VLM's list from "vlm_edit" and is
    # not confirmed; the proposal decides, so a later change of agent does not skip the check.
    art = {**job, "articulated": True, "parts": {"mode": "vlm_edit", "list": [], "confirmed": False}}
    (J / "job.json").write_text(json.dumps(art))
    assert steps.skip("propose", J) is None and steps.skip("parts", J) is None
    assert not steps.done("propose", J) and not steps.pause_after("propose", J)
    proposal = {"source": "vlm", "mode": "vlm_edit", "base": "body", "parts": []}
    (J / "run" / "proposal.json").write_text(json.dumps(proposal))
    assert steps.done("propose", J) and steps.pause_after("propose", J)
    assert not any(steps.pause_after(s, J) for s in steps.NAMES if s != "propose")
    paused = {**art, "status": "review", "stages": {"propose": {"status": "done"}}}
    assert steps.waits_for_parts(J, paused) and not steps.waits_for_parts(J)  # job.json has no status
    assert not steps.waits_for_parts(J, {**paused, "status": "queued"})
    assert not steps.waits_for_parts(J, {**paused, "stages": {"propose": {"status": "running"}}})
    (J / "job.json").write_text(json.dumps({**art, "agent": "none"}))
    assert steps.pause_after("propose", J)
    (J / "job.json").write_text(json.dumps({**art, "articulated": False}))
    assert not steps.pause_after("propose", J)
    (J / "job.json").write_text(json.dumps(art))
    for p in ({**proposal, "mode": "vlm"}, {**proposal, "source": "user", "mode": "user"},
              {**proposal, "source": "user"},  # a typed list, also the list of a job without an agent
              {**proposal, "confirmed": True}):
        (J / "run" / "proposal.json").write_text(json.dumps(p))
        assert not steps.pause_after("propose", J) and not steps.waits_for_parts(J, paused)

    assert not steps.done("parts", J)
    (J / "run" / "parts.json").write_text("{}")
    assert steps.done("parts", J)
    (J / "physics" / "urdf").mkdir(parents=True)
    (J / "physics" / "urdf" / "urdf.json").write_text(json.dumps({"name": "toy_box"}))
    assert not steps.done("urdf", J)
    (J / "physics" / "urdf" / "urdf.json").write_text(json.dumps({"name": "toy_box", "checked": {"links": 2}}))
    assert steps.done("urdf", J)

    # Without the agent the parts get no joints, and the UI says so.
    (J / "job.json").write_text(json.dumps({**art, "agent": "none"}))
    assert steps.skip("agent", J) == "turned off: without the agent the parts get no joints, so physics welds them " \
                                     "into one body"
    (J / "job.json").write_text(json.dumps(art))
    (J / "run" / "placement.json").write_text(json.dumps({"check": {"mean_mask_iou": 0.31}}))
    assert steps.skip("agent", J).endswith("physics uses run/parts.glb, whose parts get no joints")


def test_lidar_stage_done_skip_guard_and_derived(tmp_path):
    J = tmp_path
    (J / "job.json").write_text(json.dumps({"prompt": "box", "slug": "box"}))  # an older job: no "lidar"
    assert steps.lidar({}) is None and steps.lidar({"lidar": None}) is None and steps.lidar({"lidar": False}) is None
    assert steps.skip("lidar", J) == "turned off: no LiDAR" and not steps.done("lidar", J) and steps.guard("lidar", J) is None
    on = {"kind": "rosbag", "topic": "/livox/lidar"}
    assert steps.lidar({"lidar": {**on, "extra": 1}}) == on
    assert steps.lidar({"lidar": {"kind": "stray"}}) == steps.lidar({"lidar": {"kind": "stray", "x": 1}}) == {"kind": "stray"}
    for bad in ({"kind": "video"}, {"kind": "rosbag"}, {"kind": "rosbag", "topic": ""}, "yes", True, ["/x"], "stray"):
        with pytest.raises(ValueError, match=re.escape('lidar is null, {"kind": "rosbag", "topic": T} or {"kind": "stray"}')):
            steps.lidar({"lidar": bad})
    (J / "job.json").write_text(json.dumps({"prompt": "box", "lidar": on}))
    assert steps.skip("lidar", J) is None
    (J / "job.json").write_text(json.dumps({"prompt": "box", "lidar": {"kind": "rosbag"}}))
    assert steps.skip("lidar", J) is None  # malformed: the stage runs and says why
    (J / "pool").mkdir()
    why = "the bag's TF does not link cam to lidar; upload the calibration"
    (J / "pool" / "lidar.json").write_text(json.dumps({"ok": False, "s": None, "reason": why}))
    assert not steps.done("lidar", J) and steps.guard("lidar", J) == f"LiDAR scale: {why}"
    (J / "pool" / "lidar.json").write_text(json.dumps({"ok": "true", "s": 1.25}))
    assert not steps.done("lidar", J) and steps.guard("lidar", J) == "LiDAR scale: no reason given"
    (J / "pool" / "lidar.json").write_text(json.dumps({"ok": True, "s": 1.25, "reason": None}))
    assert steps.done("lidar", J) and steps.guard("lidar", J) is None
    (J / "pool" / "lidar.json").write_text("[]")
    assert not steps.done("lidar", J) and steps.guard("lidar", J) is None
    assert steps.NAMES.index("lidar") == steps.NAMES.index("da3") + 1 == steps.NAMES.index("select") - 1
    assert "lidar" not in steps.GPU and "lidar" in jobs.BUILD

    # A rerun of lidar starts without its record, check images and points; the samples cache and the
    # calibration stay. A new analysis drops the frames' stamps.
    for f in ("pool/lidar.json", "lidar/check/f00001.jpg", "lidar/cloud/f00001.glb", "lidar/samples.npz",
              "lidar/samples.json", "lidar/clouds.npz", "lidar/calibration/c.yaml", "frames/stamps.json",
              "frames/meta.json"):
        (J / f).parent.mkdir(parents=True, exist_ok=True)
        (J / f).write_text("x")
    jobs._clear(J, ["lidar"])
    assert not (J / "pool" / "lidar.json").exists() and not (J / "lidar" / "check").exists()
    assert not (J / "lidar" / "cloud").exists()
    assert all((J / "lidar" / f).is_file() for f in ("samples.npz", "samples.json", "clouds.npz", "calibration/c.yaml"))
    jobs._clear(J, ["frames"])
    assert not (J / "frames" / "stamps.json").exists() and not (J / "frames" / "meta.json").exists()
    assert jobs.DERIVED["lidar"] == ("pool/lidar.json", "lidar/check", "lidar/cloud")


def test_lidar_settings_and_the_build_that_stores_them(make_runner, tmp_path, monkeypatch):
    """lidar_settings checks the LiDAR topic, the TF link or a calibration (made for that topic), and the
    intrinsics, with plain reasons; submit_build stores what it gives, and a bad setting changes nothing."""
    pytest.importorskip("rosbags")
    from test_webapp_lidar import CAMERA, LIVOX, POINTS, T_CAM_LIDAR, upload, write_scene_bag

    from gator.webapp import fake_steps, lidar, rosbag

    runner, cfg = make_runner()

    def job_for(bag, topic=CAMERA):
        """A rosbag job whose analysis is done, as far as lidar_settings looks."""
        job_id, path = jobs.create_job(cfg.runs, "scene.bag")
        J = cfg.runs / job_id
        path.write_bytes(bag.read_bytes())
        (J / "frames").mkdir()
        (J / "frames" / "topics.json").write_text(json.dumps(rosbag.image_topics(path)))
        (J / "frames" / "meta.json").write_text("{}")
        jobs.update_stage(J, "frames", status="done")
        return job_id, J, jobs.update(J, status="ready", prompt="box", source={"kind": "rosbag", "topic": topic})

    def reason(J, job, value):
        with pytest.raises(ValueError) as e:
            jobs.lidar_settings(J, job, value)
        return str(e.value)

    bags = tmp_path / "bags"
    bags.mkdir()
    job_id, J, job = job_for(write_scene_bag(bags / "tf.bag", n=3))
    on = {"kind": "rosbag", "topic": POINTS}
    assert jobs.lidar_settings(J, job, None) is None and jobs.lidar_settings(J, job, False) is None
    assert jobs.lidar_settings(J, job, {"topic": POINTS}) == on and jobs.lidar_settings(J, job, on) == on
    assert reason(J, job, {"topic": "/nope"}) == "/nope is not a LiDAR topic of the bag; its LiDAR topics are /lidar/points"
    for bad in ({"topic": ""}, {"kind": "stray", "topic": POINTS}, POINTS, True):
        assert reason(J, job, bad) == 'lidar is null or {"topic": T}, a LiDAR topic of the bag', bad
    assert reason(J, {**job, "source": {"kind": "video", "topic": None}}, on) == \
        "LiDAR needs a rosbag or a Stray Scanner input; this job's input is a video, so turn LiDAR off"
    assert reason(J, {**job, "source": {"kind": "rosbag", "topic": None}}, on) == "choose the bag's camera topic first"
    topics = json.loads((J / "frames" / "topics.json").read_text())
    (J / "frames" / "topics.json").write_text(json.dumps({k: v for k, v in topics.items() if k != "lidar"}))
    assert reason(J, job, on) == lidar.OLD_TOPICS
    (J / "frames" / "topics.json").write_text(json.dumps(
        {**topics, "lidar": [], "lidar_skipped": [{"topic": POINTS, "type": "PointCloud2", "why": "no messages"}]}))
    assert reason(J, job, on) == "the LiDAR topic /lidar/points cannot be used: no messages"
    (J / "frames" / "topics.json").write_text(json.dumps(topics))

    # Without a TF link only with a calibration; without a camera_info only with the calibration's K.
    _, J2, bare = job_for(write_scene_bag(bags / "bare.bag", n=3, tf=None, info=False))
    assert reason(J2, bare, on) == "the bag's TF does not link camera_optical_frame to lidar; upload the calibration"
    upload(J2, {"T_cam_lidar": T_CAM_LIDAR.tolist()})
    assert reason(J2, jobs.load(J2), on) == ("/cam/image_rect has no usable camera_info (/cam/camera_info); give K (and "
                                             "D) in the calibration")
    assert jobs.lidar_settings(J2, jobs.load(J2), on, read_bag=False) == on  # submit_build's check, under the lock
    upload(J2, {"T_cam_lidar": T_CAM_LIDAR.tolist(), "K": fake_steps.intrinsics(96, 64).tolist()})
    assert jobs.lidar_settings(J2, jobs.load(J2), on) == on
    # K for another aspect ratio than the camera topic's fails at the start, as the stage would; another
    # size of the same aspect ratio is scaled.
    upload(J2, {"T_cam_lidar": T_CAM_LIDAR.tolist(), "K": fake_steps.intrinsics(100, 100).tolist(),
                "image_size": [100, 100], "D": [0.1, 0, 0, 0, 0], "distortion_model": "plumb_bob"})
    assert reason(J2, jobs.load(J2), on) == ("the intrinsics (calibration calib.json: K and D (plumb_bob)) are for "
                                             "100x100 images but the frames are 96x64; give K for the frames' size in "
                                             "the calibration")
    upload(J2, {"T_cam_lidar": T_CAM_LIDAR.tolist(), "K": fake_steps.intrinsics(192, 128).tolist(),
                "image_size": [192, 128]})
    assert jobs.lidar_settings(J2, jobs.load(J2), on) == on
    # A calibration uploaded for another LiDAR topic (its lidar_topic) is refused, also in submit_build's check
    # under the lock; one for this topic passes, and one that names none (older jobs) is not checked.
    cal = jobs.load(J2)["calibration"]
    jobs.update(J2, calibration={**cal, "lidar_topic": LIVOX})
    why = ("the calibration calib.json is for /livox/lidar, not /lidar/points; choose /livox/lidar, or upload a "
           "calibration for /lidar/points")
    assert reason(J2, jobs.load(J2), on) == why
    with pytest.raises(ValueError) as e:
        jobs.lidar_settings(J2, jobs.load(J2), on, read_bag=False)
    assert str(e.value) == why
    for topic in (POINTS, None):
        jobs.update(J2, calibration={**cal, "lidar_topic": topic})
        assert jobs.lidar_settings(J2, jobs.load(J2), on) == on, topic
    jobs.update(J2, calibration=cal)
    assert jobs.lidar_settings(J2, jobs.load(J2), on) == on
    _, J3, raw = job_for(write_scene_bag(bags / "raw.bag", n=3, topic="/cam/image_raw", model="weird_model"),
                         topic="/cam/image_raw")
    assert "unknown distortion model 'weird_model'" in reason(J3, raw, on)

    # The build: a bad setting is refused before anything changes; a good one is stored as checked.
    monkeypatch.setattr(steps, "command", lambda stage, job, cfg, key_file=None: (
        ["bash", "-c", f"touch done_{stage}"], str(cfg.runs / job["id"]), {}))
    monkeypatch.setattr(steps, "done", lambda stage, J: (Path(J) / f"done_{stage}").exists() or stage == "frames")
    with pytest.raises(ValueError, match="/nope is not a LiDAR topic"):
        runner.submit_build(job_id, None, agent="none", lidar={"topic": "/nope"})
    assert jobs.load(J)["status"] == "ready" and jobs.load(J)["lidar"] is None
    runner.submit_build(job_id, None, agent="none", lidar={"topic": POINTS})
    job = wait(J, "review")
    assert job["lidar"] == on and job["stages"]["lidar"]["status"] == "done" and (J / "done_lidar").exists()
    runner.submit_build(job_id, None, agent="none", lidar=None)
    job = wait(J, "review")
    assert job["lidar"] is None and job["stages"]["lidar"] == {**jobs.STAGE, "status": "skipped",
                                                               "note": "turned off: no LiDAR"}


def test_lidar_settings_of_a_stray_capture(tmp_path):
    """A Stray Scanner capture takes {"kind": "stray"} or no LiDAR; a bag's setting is refused for it, and
    Stray's setting for a bag or a video. Its topics stage is skipped."""
    job_id, _ = jobs.create_job(tmp_path, "capture.zip")
    J = tmp_path / job_id
    job = jobs.load(J)
    on = {"kind": "stray"}
    assert jobs.lidar_settings(J, job, on) == jobs.lidar_settings(J, job, {"kind": "stray", "x": 1}) == on
    assert jobs.lidar_settings(J, job, None) is None and jobs.lidar_settings(J, job, False) is None
    assert jobs.lidar_settings(J, job, on, read_bag=False) == on
    for bad in ({"topic": "/lidar/points"}, {"kind": "rosbag", "topic": "/lidar/points"}, "stray", True, {}):
        with pytest.raises(ValueError) as e:
            jobs.lidar_settings(J, job, bad)
        assert str(e.value) == 'lidar is null or {"kind": "stray"} for a Stray Scanner capture', bad
    for source, why in (({"kind": "rosbag", "topic": "/cam"}, 'lidar is null or {"topic": T}, a LiDAR topic of the bag'),
                        ({"kind": "video", "topic": None},
                         "LiDAR needs a rosbag or a Stray Scanner input; this job's input is a video, so turn LiDAR off")):
        with pytest.raises(ValueError) as e:
            jobs.lidar_settings(J, {**job, "source": source}, on)
        assert str(e.value) == why
    assert steps.skip("topics", J) == "Stray Scanner capture" and steps.skip("lidar", J) is None
    jobs.update(J, lidar=None)
    assert steps.skip("lidar", J) == "turned off: no LiDAR"
    for kind, why in (("video", "video input"), ("rosbag", None)):
        jobs.update(J, source={"kind": kind, "topic": None})
        assert steps.skip("topics", J) == why


def test_next_glb(tmp_path):
    J = tmp_path
    (J / "agent" / "s" / "work" / "exports").mkdir(parents=True)
    glb = J / "agent" / "s" / "work" / "exports" / "002.glb"
    glb.write_bytes(b"glTF")
    posed = J / "run" / "posed.glb"

    def result(**r):
        (J / "agent" / "result.json").write_text(json.dumps({"source": "finish", "verdict": "complete",
                                                             "glb": "work/exports/002.glb", **r}))
        return steps.next_glb(J)

    assert steps.next_glb(J) == posed
    assert result() == glb
    assert result(verdict="fallback_original", glb="original.glb") == posed
    assert result(source=None, verdict="needs_review", glb=None) == posed
    assert result(key_found_in=["work/exports/002.glb"]) == posed
    assert result(glb="work/exports/../../../../escape.glb") == posed
    glb.unlink()
    glb.symlink_to(posed)
    assert result() == posed and steps.agent_glb(J) is None

    # An articulated job falls back to its split parts (no joints) once the parts stage made them.
    glb.unlink()
    glb.write_bytes(b"glTF")
    (J / "job.json").write_text(json.dumps({"articulated": True}))
    assert result(verdict="fallback_original", glb="original.glb") == posed
    (J / "run").mkdir()
    (J / "run" / "parts.glb").write_bytes(b"glTF")
    assert result(verdict="fallback_original", glb="original.glb") == J / "run" / "parts.glb"
    assert result(source=None, verdict="needs_review", glb=None) == J / "run" / "parts.glb"
    (J / "agent" / "result.json").unlink()  # the agent stage was skipped
    assert steps.next_glb(J) == J / "run" / "parts.glb" and steps.agent_glb(J) is None
    assert result(verdict="articulated") == glb == steps.agent_glb(J)
    (J / "job.json").write_text(json.dumps({"articulated": False}))
    assert result(verdict="fallback_original", glb="original.glb") == posed


def test_create_job_and_atomic_state(tmp_path):
    job_id, path = jobs.create_job(tmp_path / "runs", "../../My vidéo (1).MOV")
    J = tmp_path / "runs" / job_id
    assert jobs.ID.fullmatch(job_id) and path == J / "video" / "My_vid_o_1_.MOV"
    assert stat.S_IMODE(J.stat().st_mode) == 0o700 and stat.S_IMODE((J / "job.json").stat().st_mode) == 0o600
    job = jobs.load(J)
    assert job["status"] == "uploaded" and set(job["stages"]) == set(steps.NAMES) and job["key_ok"] is False
    assert job["video"]["path"] == str(path.resolve())
    assert job["source"] == {"kind": "video", "topic": None} and job["agent"] == "astra"
    assert job["lidar"] is None and job["calibration"] is None
    assert jobs.load(tmp_path / "runs" / jobs.create_job(tmp_path / "runs", "Run 1.BAG")[0])["source"]["kind"] == "rosbag"
    assert [jobs.kind(n) for n in ("a.webm", "b.MCAP", "c.db3", "d.txt", "bag", "e.ZIP", "f.mp4.zip", "zip")] == \
        ["video", "rosbag", "rosbag", None, None, "stray", "stray", None]
    # A Stray Scanner export starts with LiDAR on; the others without.
    sid, spath = jobs.create_job(tmp_path / "runs", "my capture.zip")
    stray = jobs.load(tmp_path / "runs" / sid)
    assert spath.name == "my_capture.zip" and stray["video"]["path"] == str(spath.resolve())
    assert stray["source"] == {"kind": "stray", "topic": None} and stray["lidar"] == {"kind": "stray"}
    assert stray["calibration"] is None
    jobs.update(J, prompt="mug")
    jobs.update_stage(J, "masks", status="running")
    job = jobs.load(J)
    assert job["prompt"] == "mug" and job["stages"]["masks"]["status"] == "running"
    assert sorted(p.name for p in J.iterdir()) == ["job.json", "logs", "video"]  # no temporary files left
    assert jobs.create_job(tmp_path / "runs", "...")[1].name == "video"
    # Stripping the leading dots keeps the suffix, so the job's kind is that of the saved file.
    assert [jobs.safe_name(n) for n in ("..bag", ".x.MCAP", "a..db3", "..")] == ["video.bag", "x.MCAP", "a..db3", "video"]
    job_id, path = jobs.create_job(tmp_path / "runs", "..bag")
    assert path.name == "video.bag" and jobs.load(path.parents[1])["source"]["kind"] == "rosbag"


def test_progress_reads_the_newest_run_of_the_log(tmp_path):
    job_id, _ = jobs.create_job(tmp_path, "v.mov")
    J = tmp_path / job_id
    (J / "logs" / "da3.log").write_text("== da3 earlier\nPROGRESS 0.9\nfailed\n== da3 now\nloading\nPROGRESS 0.25\n"
                                        "PROGRESS 0.5\nworking\n")
    jobs.update_stage(J, "da3", status="running", started=time.time() - 3)
    st = {s["name"]: s for s in jobs.progress(J)}
    assert [s["name"] for s in jobs.progress(J)] == steps.NAMES
    assert st["da3"]["progress"] == 0.5 and st["da3"]["seconds"] >= 3 and st["da3"]["gpu"]
    assert st["frames"]["progress"] is None and st["frames"]["seconds"] is None
    (J / "logs" / "da3.log").write_text("== da3 earlier\nPROGRESS 0.9\n== da3 now\nloading\n")
    assert jobs.last_progress(J / "logs" / "da3.log") is None

    # A rigid job made before the part stages lacks their records: they show as skipped, urdf as pending.
    # A job made before LiDAR lacks the lidar stage's record and its job.json fields: skipped too.
    # A job made before the 360 video lacks the turntable stage's record and job.json "turntable": skipped too.
    old = jobs.load(J)
    for s in ("propose", "parts", "urdf", "lidar", "turntable"):
        del old["stages"][s]
    del old["articulated"], old["parts"], old["lidar"], old["calibration"], old["turntable"]
    jobs.save(J, old)
    st = {s["name"]: s for s in jobs.progress(J)}
    assert st["propose"]["status"] == st["parts"]["status"] == "skipped"
    assert st["parts"]["note"] == "turned off: rigid object" and st["urdf"]["status"] == "pending"
    assert (st["lidar"]["status"], st["lidar"]["note"], st["lidar"]["gpu"]) == ("skipped", "turned off: no LiDAR", False)
    assert (st["turntable"]["status"], st["turntable"]["note"], st["turntable"]["gpu"]) == ("skipped", "turned off", True)
    jobs.save(J, {**old, "turntable": True})
    assert {s["name"]: s["status"] for s in jobs.progress(J)}["turntable"] == "pending"
    jobs.save(J, {**old, "articulated": True})
    assert {s["name"]: s["status"] for s in jobs.progress(J)}["propose"] == "pending"
    jobs.save(J, {**old, "lidar": {"kind": "rosbag", "topic": "/points"}})  # with LiDAR on it is not skipped
    assert {s["name"]: s["status"] for s in jobs.progress(J)}["lidar"] == "pending"


def test_check_key(monkeypatch):
    import urllib.error
    import urllib.request

    assert jobs.check_key("sk-abc", fake=True)[0] and not jobs.check_key("abc", fake=True)[0]
    assert not jobs.check_key("sk-abc\nOTHER=1", fake=True)[0] and not jobs.check_key("sk abc", fake=True)[0]
    seen = []

    def refuse(req, timeout):
        seen.append((req.full_url, req.get_header("Authorization"), timeout))
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    ok, msg = jobs.check_key(KEY, fake=False)
    assert not ok and msg == "OpenAI rejected the key (HTTP 401)" and KEY not in msg
    assert seen == [("https://api.openai.com/v1/models", f"Bearer {KEY}", 10)]
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout: open(os.devnull, "rb"))
    assert jobs.check_key(KEY, fake=False) == (True, "OpenAI accepted the key")


OPUS_KEY = "sk-ant-fake-" + secrets.token_hex(16)


def test_agent_choice_reads_older_jobs():
    assert [steps.agent(j) for j in ({}, {"agent": True}, {"agent": False}, {"agent": "opus"}, {"agent": "none"})] == \
        ["astra", "astra", "none", "opus", "none"]
    for bad in ("gpt", 1, None, ["opus"]):
        with pytest.raises(ValueError, match="agent is one of astra, opus, none"):
            steps.agent({"agent": bad})


def test_check_anthropic_key(monkeypatch):
    """Opus keys are checked with the anthropic SDK (a stand-in client here); only the kind of error is
    reported, never its text, which could quote the key."""
    pytest.importorskip("anthropic")  # the Claude client
    import anthropic
    import httpx2

    assert jobs.check_key(OPUS_KEY, fake=True, agent="opus") == (True, "fake mode: key accepted")
    assert jobs.check_key(KEY, fake=True, agent="opus") == (False, "fake mode: Anthropic keys start with sk-ant-")
    assert not jobs.check_key(OPUS_KEY + "\nX=1", fake=True, agent="opus")[0]
    req = httpx2.Request("GET", "https://api.anthropic.com/v1/models/claude-opus-5-5")
    seen, raises = [], []

    def status(cls, code):
        return cls(f"refused {OPUS_KEY}", response=httpx2.Response(code, request=req), body=None)

    class Client:
        def __init__(self, **kw):
            seen.append(kw)
            self.models = self

        def retrieve(self, model_id):
            seen.append(model_id)
            if raises[-1]:
                raise raises[-1]

    monkeypatch.setattr(anthropic, "Anthropic", Client)
    for error, want in [(None, (True, "Anthropic accepted the key for claude-opus-5-5")),
                        (status(anthropic.AuthenticationError, 401), (False, "the Anthropic key was rejected")),
                        (status(anthropic.PermissionDeniedError, 403), (False, "this key cannot use claude-opus-5-5")),
                        (status(anthropic.NotFoundError, 404), (False, "this key cannot use claude-opus-5-5")),
                        (status(anthropic.InternalServerError, 500), (False, "the Anthropic API answered HTTP 500")),
                        (anthropic.APIConnectionError(request=req), (False, "could not reach the Anthropic API")),
                        (anthropic.APITimeoutError(req), (False, "could not reach the Anthropic API"))]:
        raises.append(error)
        assert jobs.check_key(OPUS_KEY, fake=False, agent="opus") == want
    assert seen[:2] == [{"api_key": OPUS_KEY, "max_retries": 1, "timeout": 20}, "claude-opus-5-5"]
    monkeypatch.setitem(sys.modules, "anthropic", None)  # envs/web without the SDK
    assert jobs.check_key(OPUS_KEY, fake=False, agent="opus") == (
        False, "the server cannot check Anthropic keys: envs/web needs anthropic==1.15.0")


def test_a_key_of_the_other_provider_is_never_sent(monkeypatch):
    """A key pasted under the wrong agent is refused by its prefix before any request, so it never goes
    to the other company's API; fake mode applies the same rule."""
    pytest.importorskip("anthropic")  # the Claude client
    import urllib.request

    import anthropic

    def no_request(*a, **kw):
        raise AssertionError("a request was made")

    monkeypatch.setattr(urllib.request, "urlopen", no_request)
    monkeypatch.setattr(anthropic, "Anthropic", no_request)
    for fake in (False, True):
        pre = "fake mode: " if fake else ""
        for key, agent, why in [(KEY, "opus", "Anthropic keys start with sk-ant-"),
                                ("sk-proj-" + KEY[8:], "opus", "Anthropic keys start with sk-ant-"),
                                (OPUS_KEY, "astra", "this is an Anthropic key (sk-ant-); GPT-6-Astra needs an OpenAI key"),
                                ("ant-" + KEY, "astra", "OpenAI keys start with sk-")]:
            ok, msg = jobs.check_key(key, fake=fake, agent=agent)
            assert (ok, msg) == (False, pre + why) and key not in msg
    assert jobs.check_key(OPUS_KEY, fake=True, agent="opus") == jobs.check_key(KEY, fake=True) == (True, "fake mode: key accepted")


def test_runner_opus_key_and_a_change_of_agent(make_runner, tmp_path, monkeypatch):
    runner, cfg = make_runner()
    probe = "import os, stat, sys; f = sys.argv[1]; print('key file', oct(stat.S_IMODE(os.stat(f).st_mode)), " \
            "open(f).read().split('=')[0])"

    def command(stage, job, cfg, key_file=None):
        J = cfg.runs / job["id"]
        script = f"touch {J}/done_{stage}"
        if stage == "agent":
            script += f"; {sys.executable} -c \"{probe}\" {key_file}"
        return ["bash", "-c", script], str(J), {}

    monkeypatch.setattr(steps, "command", command)
    monkeypatch.setattr(steps, "done", lambda stage, J: (Path(J) / f"done_{stage}").exists() or stage == "frames")
    job_id, J = ready_job(cfg, auto_continue=True)
    with pytest.raises(ValueError, match="agent is one of"):
        runner.submit_build(job_id, OPUS_KEY, agent="gpt")
    runner.submit_build(job_id, OPUS_KEY, agent="opus")
    job = wait(J, "done")
    assert job["agent"] == "opus" and job["key_ok"] is False and not runner.has_key(job_id)
    assert "key file 0o600 ANTHROPIC_API_KEY" in (J / "logs" / "agent.log").read_text()

    # Selected with GPT-6-Astra, then Build with Opus and no key: the OpenAI key is dropped, and the agent
    # stage waits for the Anthropic one.
    job_id, J = ready_job(cfg)
    runner.submit_build(job_id, KEY)
    wait(J, "review")
    assert runner.has_key(job_id, "astra") and not runner.has_key(job_id, "opus")
    runner.resume(job_id, "sam3d", agent="opus")
    job = wait(J, "review")
    assert job["agent"] == "opus" and job["warnings"] == [jobs.NO_KEY["opus"]] and not runner.has_key(job_id)
    assert job["stages"]["agent"]["note"] == "waiting for the Anthropic API key" and not (J / "done_agent").exists()
    runner.resume(job_id, "sam3d", api_key=OPUS_KEY)
    job = wait(J, "done")
    agent_started, place_started = job["stages"]["agent"]["started"], job["stages"]["place"]["started"]

    # Done; back to GPT-6-Astra from vomp: the agent and every stage after it run again, with an OpenAI key file.
    runner.resume(job_id, "vomp", api_key=KEY, agent="astra")
    job = wait(J, "done")
    assert job["agent"] == "astra" and job["stages"]["agent"]["started"] > agent_started
    assert job["stages"]["place"]["started"] == place_started and job["stages"]["package"]["status"] == "done"
    log = (J / "logs" / "agent.log").read_text()
    assert log.count("== agent ") == 2 and log.index("ANTHROPIC_API_KEY") < log.index("key file 0o600 OPENAI_API_KEY")
    # And no agent (an older client's false): no key, and the agent stage is skipped.
    runner.resume(job_id, "align", agent=False)
    job = wait(J, "done")
    assert job["agent"] == "none" and job["stages"]["agent"]["status"] == "skipped"
    assert not list(runner.key_dir.iterdir())
    assert not any(k.encode() in p.read_bytes() for p in all_files(tmp_path) for k in (KEY, OPUS_KEY))


def test_runner_warns_when_the_360_video_fails(make_runner, monkeypatch):
    """The turntable stage is optional (steps.OPTIONAL): when it fails, its record says why, job.json "warnings" gets
    "360 video: <why>", and the build goes on to package, so the job ends done. A Cancel while it runs still cancels
    the job, before package. A retry from turntable runs package again (Runner.resume), so results.zip gets the
    video."""
    runner, cfg = make_runner()

    def command(stage, job, cfg, key_file=None):
        J = cfg.runs / job["id"]
        script = f"echo PROGRESS 1; touch {J}/done_{stage}"
        if stage == "turntable" and not (J / "works").exists():
            script = "sleep 30" if (J / "slow").exists() else "echo PROGRESS 0.5; echo the GPU is gone; exit 3"
        return ["bash", "-c", script], str(J), {}

    monkeypatch.setattr(steps, "command", command)
    monkeypatch.setattr(steps, "done", lambda stage, J: (Path(J) / f"done_{stage}").exists() or stage == "frames")
    job_id, J = ready_job(cfg, auto_continue=True)
    runner.submit_build(job_id, None, agent="none", turntable=True)
    job = wait(J, "done", "failed")
    tt = job["stages"]["turntable"]
    assert (job["status"], job["error"], job["turntable"]) == ("done", None, True)
    assert (tt["status"], tt["returncode"], tt["note"]) == ("failed", 3, "exit 3: the GPU is gone")
    assert job["warnings"] == ["360 video: exit 3: the GPU is gone"] and job["stages"]["package"]["status"] == "done"

    package_started = job["stages"]["package"]["started"]

    # Retry from the video, cancelled while it runs: the job is cancelled, and package waits for the next run.
    (J / "slow").touch()
    assert runner.resume(job_id, "turntable") == "turntable"
    t0 = time.time()
    while jobs.load(J)["current"] != "turntable" and time.time() - t0 < 30:
        time.sleep(0.05)
    runner.cancel(job_id)
    job = wait(J, "cancelled", timeout=30)
    assert job["stages"]["turntable"]["note"] == "cancelled" and job["warnings"] == []
    assert job["stages"]["package"]["status"] == "pending"

    # Retry again, and the video works: package runs again.
    (J / "works").touch()
    runner.resume(job_id, "turntable")
    job = wait(J, "done", "failed")
    assert job["status"] == "done" and job["warnings"] == [] and job["stages"]["turntable"]["status"] == "done"
    assert job["stages"]["package"]["status"] == "done" and job["stages"]["package"]["started"] > package_started


def test_runner_key_file_lives_only_during_the_agent_stage(make_runner, tmp_path, monkeypatch):
    runner, cfg = make_runner()
    (runner.key_dir / "stale.env").write_text("OPENAI_API_KEY=old\n")
    runner, cfg = make_runner()  # a restart deletes key files a dead server left
    assert not list(runner.key_dir.iterdir()) and stat.S_IMODE(runner.key_dir.stat().st_mode) == 0o700
    probe = "import os, stat, sys; f = sys.argv[1]; print('key file', oct(stat.S_IMODE(os.stat(f).st_mode)), " \
            "open(f).read().startswith('OPENAI_API_KEY='))"

    def command(stage, job, cfg, key_file=None):
        J = cfg.runs / job["id"]
        script = f"echo PROGRESS 1; touch {J}/done_{stage}"
        if stage == "agent":
            script += f"; {sys.executable} -c \"{probe}\" {key_file}"
        return ["bash", "-c", script], str(J), {}

    monkeypatch.setattr(steps, "command", command)
    monkeypatch.setattr(steps, "done", lambda stage, J: (Path(J) / f"done_{stage}").exists() or stage == "frames")
    job_id, J = ready_job(cfg, auto_continue=True)
    runner.submit_build(job_id, KEY)
    job = wait(J, "done")
    skipped = ("lidar", "propose", "parts", "turntable")  # a video has no LiDAR, a rigid object no parts, no 360 video
    assert all(job["stages"][s]["status"] == ("skipped" if s in skipped else "done") for s in ["frames", *jobs.BUILD])
    assert job["key_ok"] is False
    assert job["slug"] == "box" and job_id not in runner._keys
    assert "key file 0o600 True" in (J / "logs" / "agent.log").read_text()
    assert not list(runner.key_dir.iterdir())
    assert not any(KEY.encode() in p.read_bytes() for p in all_files(tmp_path))

    # After a restart the key is gone: the job waits in review at the agent until it is given again.
    job_id, J = ready_job(cfg)
    runner.submit_build(job_id, KEY)
    assert wait(J, "review")["stages"]["sam3d"]["status"] == "pending"
    runner, _ = make_runner()
    assert jobs.load(J)["key_ok"] is False
    runner.resume(job_id, "sam3d")
    job = wait(J, "review")
    assert job["warnings"] == [jobs.NO_KEY["astra"]] and job["stages"]["place"]["status"] == "done"
    assert job["stages"]["agent"]["status"] == "pending" and not (J / "done_agent").exists()
    started = job["stages"]["place"]["started"]
    runner.resume(job_id, "sam3d", api_key=KEY)
    job = wait(J, "done")
    assert job["stages"]["place"]["started"] == started  # done stages are skipped
    assert not any(KEY.encode() in p.read_bytes() for p in all_files(tmp_path))
    with pytest.raises(ValueError):
        runner.resume(job_id, "frames")
    with pytest.raises(ValueError):
        runner.job_dir("../" + job_id)


PROPOSAL = {"source": "vlm", "agent": "astra", "model": "gpt-6-astra", "noun": "box", "base": "body", "warnings": [],
            "parts": [{"name": "body", "parent": None, "joint": "fixed", "count": 1, "axis_hint": None},
                      {"name": "lid", "parent": "body", "joint": "revolute", "count": 1, "axis_hint": "back edge"}]}


def test_runner_pauses_for_the_part_list_and_confirms_it(make_runner, tmp_path, monkeypatch):
    """A "vlm_edit" job stops after propose (whatever auto_continue says) until resume confirms the
    VLM's list or the user's edit of it, also after a cancel or a change of agent; propose gets the key
    file only when the VLM lists the parts."""
    runner, cfg = make_runner()
    for mode in steps.PARTS_MODES:  # what propose writes in each mode; the stub keeps the VLM's list
        (tmp_path / f"proposal_{mode}.json").write_text(json.dumps({**PROPOSAL, "mode": mode,
                                                                   **({"source": "user"} if mode == "user" else {})}))
    probe = "import os, stat, sys; f = sys.argv[1]; print('key file', oct(stat.S_IMODE(os.stat(f).st_mode)), " \
            "open(f).read().split('=')[0])"

    def command(stage, job, cfg, key_file=None):
        J = cfg.runs / job["id"]
        script = f"touch {J}/done_{stage}"
        if stage == "propose":
            mode = (job.get("parts") or {}).get("mode", "vlm")
            script += f"; mkdir -p {J}/run && cp {tmp_path}/proposal_{mode}.json {J}/run/proposal.json"
        if key_file is not None:
            script += f"; {sys.executable} -c \"{probe}\" {key_file}"
        return ["bash", "-c", script], str(J), {}

    monkeypatch.setattr(steps, "command", command)
    monkeypatch.setattr(steps, "done", lambda stage, J: (Path(J) / f"done_{stage}").exists() or stage == "frames")
    job_id, J = ready_job(cfg, auto_continue=True)
    runner.submit_build(job_id, KEY, articulated=True, parts={"mode": "vlm_edit", "list": []})
    job = wait(J, "review")
    assert job["articulated"] is True and job["parts"] == {"mode": "vlm_edit", "list": [], "confirmed": False}
    assert job["stages"]["propose"]["status"] == "done" and job["stages"]["parts"]["status"] == "pending"
    assert steps.waits_for_parts(J, job) and job["warnings"] == [] and runner.has_key(job_id, "astra")
    assert "key file 0o600 OPENAI_API_KEY" in (J / "logs" / "propose.log").read_text()
    assert not list(runner.key_dir.iterdir())

    # Build without a confirmation stops there again; a bad list changes nothing.
    runner.resume(job_id, "sam3d")
    job = wait(J, "review")
    assert job["stages"]["parts"]["status"] == "pending" and steps.waits_for_parts(J, job)
    with pytest.raises(ValueError, match="has the name of the whole object"):
        runner.resume(job_id, "sam3d", confirm_parts=True, parts_list=[{"name": "Box"}, {"name": "lid", "parent": "Box"}])
    with pytest.raises(ValueError, match="only with confirm_parts"):
        runner.resume(job_id, "sam3d", parts_list=PROPOSAL["parts"])
    with pytest.raises(ValueError, match="the job waits for its part list; confirm or edit it first"):
        runner.resume(job_id, "agent")
    assert jobs.load(J)["status"] == "review" and "confirmed" not in json.loads((J / "run" / "proposal.json").read_text())

    edited = [{"name": "body"}, {"name": "top lid", "parent": "body", "joint": "revolute"},
              {"name": "latch", "parent": "top lid", "joint": "fixed"}]
    assert runner.resume(job_id, "sam3d", confirm_parts=True, parts_list=edited) == "parts"
    job = wait(J, "done")
    proposal = json.loads((J / "run" / "proposal.json").read_text())
    assert proposal["confirmed"] and proposal["edited"] and [p["name"] for p in proposal["parts"]] == ["body", "top lid", "latch"]
    assert job["parts"] == {"mode": "vlm_edit", "list": proposal["parts"], "confirmed": True}
    assert all(job["stages"][s]["status"] == ("skipped" if s == "turntable" else "done")
               for s in steps.NAMES[steps.NAMES.index("parts"):])
    assert (J / "logs" / "propose.log").read_text().count("== propose ") == 1
    with pytest.raises(ValueError, match="does not wait for its part list"):
        runner.resume(job_id, "sam3d", confirm_parts=True)

    # Back to the views (a done job holds no key): propose runs anew, its list is not confirmed, and the
    # job stops again; Build with the VLM's own list then confirms it as it is.
    runner.resume(job_id, "select", api_key=KEY)
    job = wait(J, "review")
    assert steps.waits_for_parts(J, job) and job["parts"]["confirmed"] is False
    assert (J / "logs" / "propose.log").read_text().count("key file 0o600 OPENAI_API_KEY") == 2
    runner.resume(job_id, "sam3d", confirm_parts=True)
    job = wait(J, "done")
    proposal = json.loads((J / "run" / "proposal.json").read_text())
    assert proposal["confirmed"] and not proposal["edited"] and job["parts"]["list"] == PROPOSAL["parts"]

    # Cancelled while the list waits: a retry from a later stage stops at the list again, still unconfirmed.
    runner.resume(job_id, "select", api_key=KEY)
    wait(J, "review")
    runner.cancel(job_id)
    assert jobs.load(J)["status"] == "cancelled" and not steps.waits_for_parts(J) and not runner.has_key(job_id)
    assert runner.resume(job_id, "parts", api_key=KEY) == "propose"
    job = wait(J, "review")
    assert steps.waits_for_parts(J, job) and job["stages"]["parts"]["status"] == "pending"
    assert (J / "logs" / "propose.log").read_text().count("== propose ") == 3  # the done propose stage is kept
    assert "confirmed" not in json.loads((J / "run" / "proposal.json").read_text())
    # Nor does a change of agent skip the list: the job stops at it again, and confirming it builds on.
    assert runner.resume(job_id, "sam3d", agent="none") == "sam3d"
    job = wait(J, "review")
    assert job["agent"] == "none" and steps.waits_for_parts(J, job) and job["stages"]["parts"]["status"] == "pending"
    runner.resume(job_id, "sam3d", confirm_parts=True)
    job = wait(J, "done")
    assert job["stages"]["agent"]["status"] == "skipped" and job["parts"]["confirmed"] is True

    # "vlm" runs through; "user" (also any job without an agent) takes no key and needs the list.
    job_id, J = ready_job(cfg, auto_continue=True)
    runner.submit_build(job_id, KEY, articulated=True, parts={"mode": "vlm"})
    job = wait(J, "done")
    assert job["parts"] == {"mode": "vlm", "list": [], "confirmed": False} and job["stages"]["parts"]["status"] == "done"
    job_id, J = ready_job(cfg, auto_continue=True)  # the user's list with an agent: only the agent takes the key
    runner.submit_build(job_id, KEY, articulated=True, parts={"mode": "user", "list": ["base", {"name": "door", "parent": "base"}]})
    job = wait(J, "done")
    assert job["parts"]["mode"] == "user" and "key file" not in (J / "logs" / "propose.log").read_text()
    assert "key file 0o600 OPENAI_API_KEY" in (J / "logs" / "agent.log").read_text() and not list(runner.key_dir.iterdir())
    job_id, J = ready_job(cfg, auto_continue=True)
    for agent, mode in (("astra", "user"), ("none", "vlm_edit")):
        with pytest.raises(ValueError, match="type the parts of the object"):
            runner.submit_build(job_id, KEY if agent != "none" else None, agent=agent, articulated=True,
                                parts={"mode": mode, "list": []})
    with pytest.raises(ValueError, match="parts 'lid' and 'Lid' give the same name"):
        runner.submit_build(job_id, None, agent="none", articulated=True, parts={"list": ["base", "lid", "Lid"]})
    with pytest.raises(ValueError, match="articulated is true or false"):
        runner.submit_build(job_id, None, agent="none", articulated="yes")
    runner.submit_build(job_id, None, agent="none", articulated=True,
                        parts={"mode": "vlm_edit", "list": [{"name": "base"}, {"name": "door", "parent": "base"}]})
    job = wait(J, "done")
    assert job["parts"]["mode"] == "user" and [p["name"] for p in job["parts"]["list"]] == ["base", "door"]
    assert job["stages"]["agent"]["status"] == "skipped" and "key file" not in (J / "logs" / "propose.log").read_text()

    # A rigid job skips both stages; an older job without the parts records resumes too.
    job_id, J = ready_job(cfg, auto_continue=True)
    runner.submit_build(job_id, KEY)
    job = wait(J, "done")
    assert job["articulated"] is False and job["parts"] == {"mode": "vlm", "list": [], "confirmed": False}
    for s in ("propose", "parts"):
        assert job["stages"][s] == {**jobs.STAGE, "status": "skipped", "note": "turned off: rigid object"}
    assert job["stages"]["urdf"]["status"] == "done" and not list(runner.key_dir.iterdir())
    old = jobs.load(J)
    for s in ("propose", "parts", "urdf"):
        del old["stages"][s]
    del old["articulated"], old["parts"]
    jobs.save(J, {**old, "status": "failed"})
    runner.resume(job_id, "sam3d", api_key=KEY)
    job = wait(J, "done")
    assert job["stages"]["propose"]["status"] == "skipped" and job["stages"]["urdf"]["status"] == "done"
    assert not any(KEY.encode() in p.read_bytes() for p in all_files(tmp_path))


def test_a_typed_part_list_is_needed_when_there_is_no_vlm(make_runner, monkeypatch):
    runner, cfg = make_runner()
    monkeypatch.setattr(steps, "command", lambda stage, job, cfg, key_file=None: (
        ["bash", "-c", f"touch done_{stage}"], str(cfg.runs / job["id"]), {}))
    monkeypatch.setattr(steps, "done", lambda stage, J: (Path(J) / f"done_{stage}").exists() or stage == "frames")
    job_id, J = ready_job(cfg)
    runner.submit_build(job_id, KEY, articulated=True, parts={"mode": "vlm"})
    wait(J, "review")  # after select
    with pytest.raises(ValueError, match="this job has no part list: without an agent there is no VLM"):
        runner.resume(job_id, "sam3d", agent="none")
    assert jobs.load(J)["agent"] == "astra" and runner.has_key(job_id, "astra")


def test_runner_failures_select_review_and_resume(make_runner, monkeypatch):
    runner, cfg = make_runner()
    fail = {"masks": "echo 'SAM 3 ran'; echo 'no mask for box' >&2; exit 1",
            "select": """echo '{"ok": false, "warnings": ["5 of 9 masked views disagree"]}' > selection.json; exit 2"""}

    def command(stage, job, cfg, key_file=None):
        J = cfg.runs / job["id"]
        return ["bash", "-c", fail.get(stage, f"touch {J}/done_{stage}")], str(J), {}

    monkeypatch.setattr(steps, "command", command)
    monkeypatch.setattr(steps, "done", lambda stage, J: (Path(J) / f"done_{stage}").exists() or stage == "frames")
    job_id, J = ready_job(cfg)
    runner.submit_build(job_id, KEY)
    job = wait(J, "failed")
    assert job["error"] == "masks: exit 1: no mask for box" and job["stages"]["masks"]["returncode"] == 1
    assert job["stages"]["prep"]["status"] == "pending" and "== masks " in (J / "logs" / "masks.log").read_text()
    del fail["masks"]
    runner.resume(job_id, "masks")
    job = wait(J, "review")
    assert job["stages"]["select"]["status"] == "failed" and job["warnings"][0] == "5 of 9 masked views disagree"
    with pytest.raises(ValueError, match="not selected"):
        runner.resume(job_id, "sam3d")
    del fail["select"]
    runner.resume(job_id, "select")
    job = wait(J, "review")
    assert job["stages"]["select"]["status"] == "done" and job["stages"]["masks"]["status"] == "done"
    runner.resume(job_id, "sam3d")
    assert wait(J, "done")["stages"]["package"]["status"] == "done"


def test_resume_starts_from_a_lidar_stage_that_is_not_current(make_runner, monkeypatch):
    """A LiDAR job whose lidar stage failed, or whose calibration was uploaded or deleted since the stage
    ran, starts from lidar when a retry or Build names a later stage, and stops at the review again."""
    runner, cfg = make_runner()
    real_done, fail = steps.done, {"lidar"}

    def command(stage, job, cfg, key_file=None):
        J = cfg.runs / job["id"]
        if stage == "lidar":  # the record of the real stage, with the calibration it ran with
            rec = {"ok": stage not in fail, "s": 1.25, "reason": "no points",
                   "extrinsic": {"calibration_sha256": (job.get("calibration") or {}).get("sha256")}}
            script = f"mkdir -p pool && echo '{json.dumps(rec)}' > pool/lidar.json && echo lidar >> ran"
            return ["bash", "-c", script], str(J), {}
        return ["bash", "-c", f"touch done_{stage} && echo {stage} >> ran"], str(J), {}

    def ran(J):
        return (J / "ran").read_text().split() if (J / "ran").exists() else []

    monkeypatch.setattr(steps, "command", command)
    monkeypatch.setattr(steps, "done", lambda stage, J: real_done(stage, J) if stage == "lidar"
                        else (Path(J) / f"done_{stage}").exists() or stage == "frames")
    job_id, J = ready_job(cfg, source={"kind": "rosbag", "topic": "/cam"}, lidar={"kind": "rosbag", "topic": "/points"})
    runner.submit_build(job_id, None, agent="none")
    job = wait(J, "failed")
    assert job["error"] == "lidar: LiDAR scale: no points" and ran(J)[-1] == "lidar"

    # The page's retry from select (or from any later stage) runs lidar first.
    fail.clear()
    (J / "ran").unlink()
    assert runner.resume(job_id, "select") == "lidar"
    job = wait(J, "review")
    assert ran(J) == ["lidar", "select"] and job["stages"]["lidar"]["status"] == "done"

    # A calibration uploaded in review: Build finds the scale again and stops in review.
    jobs.update(J, calibration={"name": "c.yaml", "sha256": "1" * 64})
    assert not steps.done("lidar", J)
    (J / "ran").unlink()
    assert runner.resume(job_id, "sam3d") == "lidar"
    job = wait(J, "review")
    assert ran(J) == ["lidar", "select"] and steps.done("lidar", J)
    assert json.loads((J / "pool" / "lidar.json").read_text())["extrinsic"]["calibration_sha256"] == "1" * 64
    assert runner.resume(job_id, "sam3d") == "sam3d"  # current: Build goes on
    job = wait(J, "done")
    assert ran(J)[2] == "sam3d" and "lidar" not in ran(J)[2:]

    # The calibration deleted after the build: a retry of a later stage starts from lidar and stops in review.
    jobs.update(J, calibration=None)
    (J / "ran").unlink()
    assert runner.resume(job_id, "agent") == "lidar"
    assert wait(J, "review") and ran(J) == ["lidar", "select"]

    # A part list waiting for its confirmation cannot be confirmed past a stale scale.
    jobs.update(J, calibration={"name": "c.yaml", "sha256": "2" * 64})
    monkeypatch.setattr(steps, "waits_for_parts", lambda J, job=None: True)
    with pytest.raises(ValueError, match="the LiDAR scale is not current"):
        runner.resume(job_id, "parts", confirm_parts=True)
    assert jobs.load(J)["status"] == "review"


def test_cancel_stops_the_stage_group_and_the_queue(make_runner, tmp_path, monkeypatch):
    runner, cfg = make_runner()
    runner.kill_after_s = 2.0
    pids = tmp_path / "pids"
    sleeper = ("import os, signal, sys, time\n"
               "if sys.argv[2] == 'stubborn': signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
               "open(sys.argv[1], 'a').write(f'{os.getpid()}\\n')\n"
               "time.sleep(60)\n")

    def command(stage, job, cfg, key_file=None):  # the sleeper is a child of bash, as the real stages are
        return ["bash", "-c", f"{sys.executable} -c \"$1\" {pids} {job['prompt']} & wait", "-", sleeper], str(tmp_path), {}

    monkeypatch.setattr(steps, "command", command)
    a, Ja = ready_job(cfg, prompt="polite")
    b, Jb = ready_job(cfg, prompt="queued")
    runner.submit_build(a, KEY)
    runner.submit_build(b, KEY)
    wait(Ja, "running")
    t0 = time.time()
    while not pids.exists() and time.time() - t0 < 20:
        time.sleep(0.05)
    pid = int(pids.read_text().split()[0])
    assert jobs.load(Jb)["status"] == "queued"
    runner.cancel(b)
    assert jobs.load(Jb)["status"] == "cancelled" and b not in runner._keys
    t0 = time.time()
    runner.cancel(a)
    job = wait(Ja, "cancelled", timeout=10)
    assert time.time() - t0 < 1.5 and job["stages"]["extract"]["note"] == "cancelled"
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    time.sleep(0.5)
    assert len(pids.read_text().split()) == 1  # the queued job never started

    c, Jc = ready_job(cfg, prompt="stubborn")
    runner.submit_build(c, KEY)
    while len(pids.read_text().split()) < 2:
        time.sleep(0.05)
    pid = int(pids.read_text().split()[1])
    t0 = time.time()
    runner.cancel(c)
    wait(Jc, "cancelled", timeout=10)
    assert 1.9 < time.time() - t0 < 6  # SIGTERM was ignored, so SIGKILL after kill_after_s
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert not list(runner.key_dir.iterdir())


def test_an_uploading_job_is_busy(make_runner, monkeypatch):
    runner, cfg = make_runner()
    monkeypatch.setattr(steps, "command", lambda stage, job, cfg, key_file=None: (
        ["bash", "-c", "mkdir -p frames && echo {} > frames/meta.json"], str(cfg.runs / job["id"]), {}))
    job_id, path = runner.new_upload("clip.mov")
    J = cfg.runs / job_id
    assert runner.uploading(job_id) and jobs.load(J)["status"] == "uploaded"
    for refused in (runner.delete, runner.submit_analyze):  # the upload still writes into the folder
        with pytest.raises(ValueError, match="still uploading"):
            refused(job_id)
    path.write_bytes(b"video")
    runner.upload_done(job_id, 5)
    job = wait(J, "ready")
    assert job["video"]["bytes"] == 5 and job["stages"]["frames"]["status"] == "done" and not runner.uploading(job_id)
    runner.delete(job_id)
    job_id, _ = runner.new_upload("cut off.mov")
    runner.upload_done(job_id, None)
    assert not (cfg.runs / job_id).exists() and not runner.uploading(job_id)


def test_runner_rosbag_topics_and_the_agent_off(make_runner, monkeypatch):
    runner, cfg = make_runner()
    real_done = steps.done

    def command(stage, job, cfg, key_file=None):
        assert stage != "agent"  # turned off, so it never runs and needs no key
        script = {"topics": "mkdir -p frames && cp topics_in.json frames/topics.json",
                  "frames": f"mkdir -p frames && echo {steps.source(job)['topic']} >> frames_ran && echo {{}} > frames/meta.json"}
        return ["bash", "-c", script.get(stage, f"mkdir -p scene && touch scene/done_{stage}")], str(cfg.runs / job["id"]), {}

    monkeypatch.setattr(steps, "command", command)
    monkeypatch.setattr(steps, "done", lambda stage, J: real_done(stage, J) if stage in ("topics", "frames")
                        else (Path(J) / "scene" / f"done_{stage}").exists())

    def listing(J, topics, skipped=(), lidar=True):
        """topics.json as the topics stage writes it; lidar False: as before LiDAR topics were listed."""
        skipped = [{"topic": t, "type": "Image", "why": "16UC1"} for t in skipped]
        (J / "topics_in.json").write_text(json.dumps({"topics": [{"topic": t} for t in topics], "skipped": skipped,
                                                      **({"lidar": [], "lidar_skipped": []} if lidar else {})}))

    def upload(name, topics, skipped=(), lidar=True):
        job_id, path = runner.new_upload(name)
        J = cfg.runs / job_id
        listing(J, topics, skipped, lidar)
        path.write_bytes(b"bag")
        runner.upload_done(job_id, 3)
        return job_id, J

    _, J = upload("none.bag", [], skipped=["/depth"])
    job = wait(J, "failed")
    assert job["error"] == "topics: no camera image topics in the bag; not usable: /depth (16UC1)"
    assert job["stages"]["topics"]["status"] == "failed" and job["stages"]["frames"]["status"] == "pending"

    _, J = upload("one.MCAP", ["/cam/image_rect"])  # one topic: frames runs on it at once
    job = wait(J, "ready")
    assert job["source"] == {"kind": "rosbag", "topic": "/cam/image_rect"} and (J / "frames_ran").read_text() == "/cam/image_rect\n"

    job_id, J = upload("two.bag", ["/a/image_rect", "/b/image_raw"])
    job = wait(J, "choose_topic")
    assert job["source"]["topic"] is None and job["stages"]["frames"]["status"] == "pending" and not (J / "frames_ran").exists()
    topics_started = job["stages"]["topics"]["started"]
    (J / "lidar" / "calibration").mkdir(parents=True)  # a calibration uploaded before the topic was chosen
    (J / "lidar" / "calibration" / "c.yaml").write_text("T_cam_lidar: ...")
    jobs.update(J, calibration={"name": "c.yaml", "sha256": "0" * 64})
    runner.submit_analyze(job_id, topic="/b/image_raw")
    job = wait(J, "ready")
    assert job["source"]["topic"] == "/b/image_raw" and job["stages"]["topics"]["started"] == topics_started
    assert job["calibration"]["name"] == "c.yaml" and (J / "lidar" / "calibration" / "c.yaml").is_file()

    # A build without the agent: no key, and the agent stage is skipped.
    jobs.update(J, prompt="box", auto_continue=True)
    runner.submit_build(job_id, None, agent=False)
    job = wait(J, "done")
    assert job["stages"]["agent"] == {**jobs.STAGE, "status": "skipped", "note": "turned off: physics uses the Stage A mesh"}
    assert job["agent"] == "none" and job["key_ok"] is False and job_id not in runner._keys
    assert not list(runner.key_dir.iterdir())
    with pytest.raises(ValueError, match="is done"):
        runner.submit_analyze(job_id, topic="/a/image_rect")
    with pytest.raises(ValueError, match="API key"):  # None would queue it without a key (the key comes later)
        runner.submit_build(job_id, "", agent=True)

    # Another topic after a failed build: frames runs on it, and the build's outputs, records and picks are gone;
    # so is the calibration, whose T and K were the other camera's.
    jobs.update(J, status="failed", picks=[3, 9], view0=3)
    runner.submit_analyze(job_id, topic="/a/image_rect")
    job = wait(J, "ready")
    assert (J / "frames_ran").read_text() == "/b/image_raw\n/a/image_rect\n" and not (J / "scene").exists()
    assert all(job["stages"][s] == jobs.STAGE for s in jobs.BUILD) and job["picks"] == [] and job["view0"] is None
    assert job["calibration"] is None and not (J / "lidar" / "calibration").exists()

    # A bag listed before LiDAR topics were: a topic choice lists it again (topics.json then has "lidar").
    job_id, J = upload("old.bag", ["/a/image_rect", "/b/image_raw"], lidar=False)
    job = wait(J, "choose_topic")
    topics_started = job["stages"]["topics"]["started"]
    listing(J, ["/a/image_rect", "/b/image_raw"])  # what the topics stage of today writes
    runner.submit_analyze(job_id, topic="/a/image_rect")
    job = wait(J, "ready")
    assert job["stages"]["topics"]["started"] > topics_started and job["stages"]["topics"]["status"] == "done"
    assert "lidar" in json.loads((J / "frames" / "topics.json").read_text())
    topics_started = job["stages"]["topics"]["started"]
    runner.submit_analyze(job_id, topic="/b/image_raw")  # listed already: topics does not run again
    assert wait(J, "ready")["stages"]["topics"]["started"] == topics_started

    # A video skips topics, also a job made before rosbag input (no "source", no topics stage).
    job_id, J = upload("clip.MOV", [])
    wait(J, "ready")
    old = jobs.load(J)
    del old["source"], old["agent"], old["stages"]["topics"]
    jobs.save(J, {**old, "status": "failed"})
    jobs.update_stage(J, "frames", status="failed")
    runner.submit_analyze(job_id)
    job = wait(J, "ready")
    assert job["stages"]["topics"] == {**jobs.STAGE, "status": "skipped", "note": "video input"}
    with pytest.raises(ValueError, match="video"):
        runner.submit_analyze(job_id, topic="/a/image_rect")


def test_restart_fails_interrupted_jobs(make_runner):
    runner, cfg = make_runner()
    job_id, J = ready_job(cfg)
    t = time.time() - 60
    jobs.update_stage(J, "da3", status="running", started=t)
    jobs.update(J, status="running", current="da3", key_ok=True)
    (J / "logs" / "da3.log").write_text("== da3 earlier\nPROGRESS 0.4\n")
    os.utime(J / "logs" / "da3.log", (t + 30, t + 30))  # its last output, when the server died
    make_runner()
    job = jobs.load(J)
    assert job["status"] == "failed" and job["current"] is None and job["key_ok"] is False
    assert job["stages"]["da3"]["status"] == "failed" and "server stopped" in job["error"]
    assert job["stages"]["da3"]["ended"] == pytest.approx(t + 30)
    assert {s["name"]: s["seconds"] for s in jobs.progress(J)}["da3"] == 30.0  # not still counting


def stub_stages(monkeypatch, hold=()):
    """Stage commands that touch J/done_<stage> (frames counts as done); a stage in hold first waits until
    J/go_<stage> exists."""
    def command(stage, job, cfg, key_file=None):
        J = cfg.runs / job["id"]
        gate = f"while [ ! -e {J}/go_{stage} ]; do sleep 0.05; done; " if stage in hold else ""
        return ["bash", "-c", f"{gate}touch {J}/done_{stage}"], str(J), {}

    monkeypatch.setattr(steps, "command", command)
    monkeypatch.setattr(steps, "done", lambda stage, J: (Path(J) / f"done_{stage}").exists() or stage == "frames")


def until(check, timeout=30):
    t0 = time.time()
    while not check():
        assert time.time() - t0 < timeout, "timed out"
        time.sleep(0.05)


def test_runner_queued_lists_the_builds_that_wait(make_runner, monkeypatch):
    runner, cfg = make_runner()
    stub_stages(monkeypatch, hold=("extract",))
    (a, Ja), (b, Jb), (c, Jc) = (ready_job(cfg, auto_continue=True) for _ in range(3))
    assert runner.queued() == []
    for job_id in (a, b, c):
        runner.submit_build(job_id, None, agent="none")
    wait(Ja, "running")
    q = runner.queued()
    assert q == [(b, "extract"), (c, "extract")]  # the running build is not in it
    q.clear()  # a copy
    assert runner.queued() == [(b, "extract"), (c, "extract")]
    runner.cancel(c)
    assert runner.queued() == [(b, "extract")] and jobs.load(Jc)["status"] == "cancelled"
    for J in (Ja, Jb):
        (J / "go_extract").touch()
    assert wait(Jb, "done")["status"] == "done" and wait(Ja, "done") and runner.queued() == []
    runner.resume(b, "package")
    assert runner.queued() in ([(b, "package")], [])  # [] once the worker took it
    wait(Jb, "done")


def test_runner_json_keeps_a_second_server_off_the_runs_folder(make_runner, monkeypatch):
    """RUNS/.runner.json names the server whose Runner runs the folder's jobs. A second Runner refuses to
    start while that server lives, before it fails any of its jobs; one of the same process (a restart in
    the tests), or of a server that is gone, takes over."""
    monkeypatch.setenv("SLURM_JOB_ID", "1234567")
    runner, cfg = make_runner()
    f = cfg.runs / jobs.RUNNER_FILE
    info = json.loads(f.read_text())
    assert (info["pid"], info["host"], info["slurm_job"]) == (os.getpid(), socket.gethostname(), "1234567")
    assert stat.S_IMODE(f.stat().st_mode) == 0o600 and not list(cfg.runs.glob("..runner.json.*"))
    later, _ = make_runner()
    assert json.loads(f.read_text())["id"] != info["id"]
    runner.shutdown(timeout=5)
    assert f.is_file()  # the file is the later Runner's now
    later.shutdown(timeout=5)
    assert not f.exists()

    job_id, J = ready_job(cfg)
    jobs.update(J, status="running", current="extract")
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "-m", "gator.webapp"])
    try:
        f.write_text(json.dumps({"pid": other.pid, "host": socket.gethostname(), "slurm_job": "1234568"}))
        with pytest.raises(SystemExit) as e:
            make_runner()
        assert str(e.value) == (f"another HARPOR server runs the jobs in {cfg.runs} (pid {other.pid} on "
                                f"{socket.gethostname()}, Slurm job 1234568); stop it first, or delete {f} if it is gone")
        assert jobs.load(J)["status"] == "running" and json.loads(f.read_text())["pid"] == other.pid
    finally:
        other.kill()
        other.wait()
    make_runner()  # that server is gone: this one takes over and fails the job it left
    assert jobs.load(J)["status"] == "failed" and json.loads(f.read_text())["pid"] == os.getpid()
    f.write_text("[not json")
    make_runner()
    assert json.loads(f.read_text())["pid"] == os.getpid()


def test_holder_alive_asks_slurm_about_another_host(monkeypatch, capsys):
    """A server counts as running only when that is known: a HARPOR server process of this user on this host, or a
    running Slurm job on another host. A pid that another program got since, and anything Slurm does not answer,
    count as gone, so a stale .runner.json never keeps the tunnel from starting."""
    gone = subprocess.Popen(["true"])
    gone.wait()
    here = {"host": socket.gethostname()}
    server_ = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "-m", "gator.webapp"])
    other_program = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        assert jobs.holder_alive({**here, "pid": server_.pid}) is True
        assert jobs.holder_alive({**here, "pid": other_program.pid}) is False  # the pid was reused
    finally:
        for p in (server_, other_program):
            p.kill()
            p.wait()
    assert not any(jobs.holder_alive({**here, "pid": p}) for p in (gone.pid, None, "1", 0, -1, True))
    if os.getuid() != 0:
        assert jobs.holder_alive({**here, "pid": 1}) is False  # another user's process

    calls = []

    def squeue(rc, out="", err=""):
        def run(argv, **kw):
            calls.append(argv)
            return subprocess.CompletedProcess(argv, rc, out, err)
        return run

    other = {"pid": 4242, "host": "gpu-node-2.example", "slurm_job": "1234567"}
    for rc, out, err, alive in [(0, "RUNNING\n", "", True), (0, "COMPLETING\n", "", True), (0, "TIMEOUT\n", "", False),
                                (0, "", "", False), (1, "", "slurm_load_jobs error: Invalid job id specified\n", False),
                                (1, "", "slurm_load_jobs error: Unable to contact slurm controller\n", False)]:
        monkeypatch.setattr(jobs.subprocess, "run", squeue(rc, out, err))
        assert jobs.holder_alive(other) is alive, (out, err)
    assert calls[-1] == ["squeue", "-h", "-j", "1234567", "-o", "%T"]
    assert capsys.readouterr().out == ("Slurm did not say whether job 1234567 (the server in .runner.json on "
                                       "gpu-node-2.example) still runs; this server takes the runs folder\n")

    def missing(argv, **kw):
        raise FileNotFoundError("squeue")

    monkeypatch.setattr(jobs.subprocess, "run", missing)
    n = len(calls)
    assert jobs.holder_alive(other) is False and "Slurm did not say" in capsys.readouterr().out
    assert jobs.holder_alive({**other, "slurm_job": None}) is False
    assert jobs.holder_alive({**other, "slurm_job": "1234567 -o %all"}) is False and len(calls) == n


def test_runner_forget_key(make_runner, monkeypatch):
    runner, cfg = make_runner()
    stub_stages(monkeypatch)
    job_id, J = ready_job(cfg)
    runner.submit_build(job_id, KEY)  # it stops in review after select
    job = wait(J, "review")
    assert job["key_ok"] is True and runner.has_key(job_id, "astra")
    assert runner.forget_key(job_id) is True
    assert not runner.has_key(job_id) and jobs.load(J)["key_ok"] is False
    assert not any(runner.forget_key(x) for x in (job_id, "../" + job_id, None))
    runner.resume(job_id, "sam3d")
    job = wait(J, "review")
    assert job["warnings"] == [jobs.NO_KEY["astra"]] and job["stages"]["agent"]["status"] == "pending"


def test_runner_builds_without_a_key_and_asks_for_it_later(make_runner, monkeypatch, tmp_path):
    """submit_build with api_key None queues the build without a key; the job waits in review at the first
    stage that needs it, with NO_KEY_LATER. resume with key_later says the same; resume without it says the
    server lost the key (NO_KEY). A key given wins."""
    runner, cfg = make_runner()
    stub_stages(monkeypatch)
    assert jobs.NO_KEY_LATER == {"astra": "The job needs the OpenAI API key now; paste it and press Build.",
                                 "opus": "The job needs the Anthropic API key now; paste it and press Build."}
    job_id, J = ready_job(cfg, auto_continue=True)
    runner.submit_build(job_id, None, agent="opus")
    assert jobs.load(J)["key_ok"] is False and not runner.has_key(job_id)
    job = wait(J, "review")
    assert job["warnings"] == [jobs.NO_KEY_LATER["opus"]] and job["key_ok"] is False and "key_later" not in job
    assert job["stages"]["agent"]["note"] == "waiting for the Anthropic API key"
    assert job["stages"]["place"]["status"] == "done" and not (J / "done_agent").exists()
    runner.resume(job_id, "sam3d", key_later=True)
    assert wait(J, "review")["warnings"] == [jobs.NO_KEY_LATER["opus"]]
    runner.resume(job_id, "sam3d")
    assert wait(J, "review")["warnings"] == [jobs.NO_KEY["opus"]]
    runner.resume(job_id, "sam3d", api_key=OPUS_KEY, key_later=True)
    job = wait(J, "done")
    assert job["stages"]["agent"]["status"] == "done" and "key_later" not in job

    # A Start without a key drops the key held for the job; an empty key is still refused.
    job_id, J = ready_job(cfg)
    runner.submit_build(job_id, KEY)
    assert wait(J, "review")["key_ok"] is True
    runner.submit_build(job_id, None, agent="astra")
    assert not runner.has_key(job_id) and jobs.load(J)["key_ok"] is False
    wait(J, "review")
    with pytest.raises(ValueError, match="API key"):
        runner.submit_build(job_id, "", agent="astra")
    assert not list(runner.key_dir.iterdir())
    assert not any(k.encode() in p.read_bytes() for p in all_files(tmp_path) for k in (KEY, OPUS_KEY))


def glb_file(path, size=0.2):
    from gator.agent.fake_agent import write_glb

    write_glb(path, size=size)
    return path


def test_package_leaves_out_key_files(tmp_path, capsys):
    J = tmp_path
    for d in ("run", "agent/s/work/exports", "physics/mjcf/assets", "physics/usd"):
        (J / d).mkdir(parents=True)
    glb_file(J / "run" / "posed.glb")
    glb_file(J / "agent" / "s" / "work" / "exports" / "001.glb", 0.3)
    glb_file(J / "physics" / "aligned.glb")
    (J / "pool").mkdir()
    for f in ("run/placement.json", "selection.json", "physics/summary.json", "physics/mjcf/mjcf.json",
              "physics/usd/usd.json", "pool/lidar.json"):
        (J / f).write_text("{}")
    (J / "physics" / "mjcf" / "assets" / "visual_0.obj").write_text("v 0 0 0\n")
    (J / "turntable").mkdir()
    (J / "turntable" / "turntable.mp4").write_bytes(b"\0\0\0\x18ftypmp42")
    (J / "physics" / "up.json").write_text(json.dumps({"glb": str(J / "agent/s/work/exports/001.glb")}))
    r = {"source": "finish", "verdict": "complete", "glb": "work/exports/001.glb", "key_found_in": []}
    (J / "agent" / "result.json").write_text(json.dumps(r))
    jobs.package(J)
    names = set(zipfile.ZipFile(J / "results.zip").namelist())
    assert names == {"posed.glb", "refined.glb", "aligned.glb", "turntable.mp4", "summary.json", "placement.json",
                     "selection.json", "lidar.json", "agent_result.json", "mjcf/mjcf.json", "mjcf/assets/visual_0.obj",
                     "usd/usd.json"}
    with zipfile.ZipFile(J / "results.zip") as z:
        assert z.read("refined.glb") == (J / "agent/s/work/exports/001.glb").read_bytes()
        assert z.read("turntable.mp4") == (J / "turntable" / "turntable.mp4").read_bytes()
    # The agent wrote the key into its GLB: neither it nor what physics made from it (the 360 video too) is packaged.
    (J / "agent" / "result.json").write_text(json.dumps({**r, "key_found_in": ["work/exports/001.glb", "session/x.log"]}))
    jobs.main(["package", "--job", str(J)])
    assert set(zipfile.ZipFile(J / "results.zip").namelist()) == {"posed.glb", "summary.json", "placement.json",
                                                                  "selection.json", "lidar.json", "agent_result.json"}
    assert "hold the API key" in capsys.readouterr().out
    assert not list(J.glob(".results*"))


def run_folder(J, n=2):
    """A finished place output in J/run with n views, as agent_step reads it."""
    run = J / "run"
    (run / "input" / "images").mkdir(parents=True)
    (run / "masks").mkdir()
    cams = []
    for i in range(n):
        img, m = run / "input" / "images" / f"{i}.png", run / "masks" / f"{i}.png"
        Image.fromarray(np.full((48, 64, 3), 120, np.uint8)).save(img)
        a = np.zeros((48, 64), np.uint8)
        a[10:40, 20:44] = 255
        Image.fromarray(a).save(m)
        w2c = np.eye(4)
        w2c[0, 3] = 0.1 * i
        cams.append({"name": str(i), "K": [[60, 0, 31.5], [0, 60, 23.5], [0, 0, 1]], "w2c": w2c.tolist(), "width": 64,
                     "height": 48, "image": str(img), "mask": str(m)})
    (run / "cameras.json").write_text(json.dumps(cams))
    glb_file(run / "posed.glb")
    (J / "job.json").write_text(json.dumps({"id": "x", "prompt": "toy box"}))
    (J / "key.env").write_text(f"OPENAI_API_KEY={KEY}\n")
    return J


FAKE_BLENDER = [sys.executable, "-m", "gator.agent.fake_agent", "--blender"]


def test_agent_step_runs_a_fresh_refinement_session(tmp_path, monkeypatch):
    import tempfile

    from gator.agent import controller as C

    J = run_folder(tmp_path)
    (J / "agent" / "s" / "work").mkdir(parents=True)
    (J / "agent" / "s" / "work" / "stale.txt").write_text("from a failed try")
    calls = []

    def container(root, sif, key_file, budget_s):  # the controller's own session, with an agent that exits at once
        calls.append((Path(root), sif, key_file, budget_s))
        r = C.run_session(root, [sys.executable, "-c", "pass"], blender=FAKE_BLENDER,
                          sock_dir=tempfile.mkdtemp(prefix="gator-"))
        r["key_found_in"] = C.find_key(root, key_file)
        write_json(Path(root) / "result.json", r)
        return r

    monkeypatch.setattr(C, "run_container_session", container)
    r = agent_step.run(J, "/sif/agent.sif", J / "key.env")
    assert calls == [(J / "agent" / "s", "/sif/agent.sif", J / "key.env", 600.0)]
    assert not (J / "agent" / "s" / "work" / "stale.txt").exists()
    m = json.loads((J / "agent" / "s" / "control" / "manifest.json").read_text())
    assert m["mode"] == "refinement" and len(m["views"]) == 2 and m["units"] == "meters"
    assert m["source"] == agent_step.source("toy box", 2)
    assert "2 frames from a phone video of the toy box" in m["source"] and "Depth Anything 3 cameras" in m["source"]
    assert (J / "agent" / "s" / "original.glb").read_bytes() == (J / "run" / "posed.glb").read_bytes()
    assert r["verdict"] == "fallback_original" and r["key_found_in"] == []
    assert json.loads((J / "agent" / "result.json").read_text()) == json.loads((J / "agent" / "s" / "result.json").read_text())
    assert steps.next_glb(J) == J / "run" / "posed.glb"


def test_agent_step_runs_an_articulation_session(tmp_path, monkeypatch, capsys):
    """An articulated job whose parts stage made run/parts.glb gets the articulation mode: parts.glb as the
    original, run/parts.json as input/parts.json, and 900 s."""
    import tempfile

    from gator.agent import controller as C
    from gator.agent.fake_agent import write_glb

    J = run_folder(tmp_path)
    job = {**json.loads((J / "job.json").read_text()), "articulated": True}
    (J / "job.json").write_text(json.dumps(job))
    write_glb(J / "run" / "parts.glb", "parts", 0.2)
    with pytest.raises(SystemExit, match="the parts stage did not finish"):
        agent_step.run(J, "/sif/agent.sif", J / "key.env")
    parts = {"version": 1, "frame": "W", "glb": "parts.glb", "base": "body", "parts": [], "seams": []}
    (J / "run" / "parts.json").write_text(json.dumps(parts))
    budgets = []

    def container(root, sif, key_file, budget_s):
        budgets.append(budget_s)
        r = C.run_session(root, [sys.executable, "-c", "pass"], blender=FAKE_BLENDER,
                          sock_dir=tempfile.mkdtemp(prefix="gator-"), budget_s=budget_s)
        r["key_found_in"] = C.find_key(root, key_file)
        write_json(Path(root) / "result.json", r)
        return r

    monkeypatch.setattr(C, "run_container_session", container)
    r = agent_step.run(J, "/sif/agent.sif", J / "key.env")
    root = J / "agent" / "s"
    m = json.loads((root / "control" / "manifest.json").read_text())
    assert budgets == [900.0] and (m["mode"], m["parts"]) == ("articulation", "input/parts.json")
    assert m["source"] == agent_step.source("toy box", 2, parts=True) and "split into parts" in m["source"]
    assert json.loads((root / "work" / "input" / "parts.json").read_text()) == parts
    assert (root / "original.glb").read_bytes() == (J / "run" / "parts.glb").read_bytes()
    assert (r["mode"], r["verdict"], r["timings"]["budget_s"], r["candidate_built"]) == \
        ("articulation", "fallback_original", 900.0, True)
    assert r["joints_delivered"] == 0
    assert "articulation: 0 joints delivered, 0 set_joint changes, 0 articulation reviews" in capsys.readouterr().out
    assert agent_step.mode(J, {**job, "articulated": False}) == "refinement"
    (J / "run" / "parts.glb").unlink()
    assert agent_step.mode(J, job) == "refinement"


def test_agent_step_runs_a_claude_session_for_opus(tmp_path, monkeypatch, capsys):
    import tempfile

    from gator.agent import controller as C

    J = run_folder(tmp_path)
    (J / "key.env").write_text(f"ANTHROPIC_API_KEY={OPUS_KEY}\n")
    end = {"type": "end", "reason": "end_turn", "turns": 3, "cost_usd": 0.42,
           "totals": {"input_tokens": 9, "output_tokens": 8, "cache_read_input_tokens": 7, "cache_creation_input_tokens": 6}}

    def container(root, sif, key_file, budget_s):  # the controller's own session; the agent writes its end line
        script = f"import json, sys; open(sys.argv[1], 'a').write(json.dumps({end!r}) + chr(10))"
        r = C.run_session(root, [sys.executable, "-c", script, str(Path(root) / "session" / "claude.jsonl")],
                          blender=FAKE_BLENDER, sock_dir=tempfile.mkdtemp(prefix="gator-"))
        r["key_found_in"] = C.find_key(root, key_file)
        write_json(Path(root) / "result.json", r)
        return r

    monkeypatch.setattr(C, "run_container_session", container)
    r = agent_step.run(J, "/sif/agent.sif", J / "key.env", "opus")
    root = J / "agent" / "s"
    assert C.agent_info(root) == {"agent": "claude", "model": "claude-opus-5-5", "effort": "xhigh"}
    assert (root / "session" / "system.md").is_file() and not (root / "work" / "AGENTS.md").exists()
    assert (r["agent"], r["model"], r["cli_exit"], r["turns"], r["cost_usd"]) == ("claude", "claude-opus-5-5", "ended", 3, 0.42)
    assert r["tokens"] == end["totals"] and r["key_found_in"] == [] and C.read_keys(J / "key.env") == [OPUS_KEY]
    out = capsys.readouterr().out
    assert "model: claude-opus-5-5, effort xhigh" in out and "3 turns" in out and "about $0.42" in out
    with pytest.raises(SystemExit):
        agent_step.main(["--job", str(J), "--sif", "x.sif", "--key-file", str(J / "key.env"), "--agent", "gpt"])


def test_fake_agent_stage_stands_in_for_both_agents(tmp_path):
    from gator.webapp import fake_steps

    J = run_folder(tmp_path)
    key = J / "fake.env"
    for which, line, model, cli in (("astra", f"OPENAI_API_KEY={KEY}", "gpt-6-astra", "cancelled"),
                                    ("opus", f"ANTHROPIC_API_KEY={OPUS_KEY}", "claude-opus-5-5", "interrupted")):
        key.unlink(missing_ok=True)
        key.write_text(line + "\n")
        key.chmod(0o600)
        fake_steps.agent(J, key, which)
        r = json.loads((J / "agent" / "result.json").read_text())
        assert (r["model"], r["cli_exit"], r["verdict"]) == (model, cli, "complete") and ("tokens" in r) == (which == "opus")
    assert r["cost_usd"] > 0 and r["turns"] == 12
    with pytest.raises(SystemExit, match="no OPENAI_API_KEY line"):
        fake_steps.agent(J, key, "astra")


def test_agent_step_scrubs_the_key_from_the_result(tmp_path, monkeypatch, capsys):
    """The agent can read the key, so it may write it into its finish text or run a command with it.
    Neither the stage log (stdout), agent/result.json, nor results.zip may hold it."""
    from gator.agent import controller as C

    J = run_folder(tmp_path)

    def container(root, sif, key_file, budget_s):
        r = {"mode": "refinement", "source": "finish", "verdict": "complete", "glb": None, "exit": "finished",
             "cli_exit": "success", "timings": {"wall_s": 3.0}, "reason": f"done; the key is {KEY}",
             "limitations": [f"saw {KEY} in /proc"], "rejected": [{"from": "export", "glb": "a.glb", "why": f"bad {KEY}"}],
             "network_commands": [f"curl -H 'Authorization: Bearer {KEY}' https://api.openai.com/v1/models"]}
        write_json(Path(root) / "result.json", r)
        r["key_found_in"] = C.find_key(root, key_file)
        write_json(Path(root) / "result.json", r)
        return r

    monkeypatch.setattr(C, "run_container_session", container)
    r = agent_step.run(J, "/sif/agent.sif", J / "key.env")
    out = capsys.readouterr().out
    assert KEY not in out and f"reason: done; the key is {agent_step.REDACTED}" in out
    assert "the API key is in ['result.json']" in out
    saved = (J / "agent" / "result.json").read_text()
    assert KEY not in saved and json.loads(saved) == r and r["key_found_in"] == ["result.json"]
    assert r["network_commands"] == [f"curl -H 'Authorization: Bearer {agent_step.REDACTED}' https://api.openai.com/v1/models"]
    assert agent_step.scrub({KEY: [1, None, KEY + "x"]}, KEY) == {agent_step.REDACTED: [1, None, agent_step.REDACTED + "x"]}
    assert agent_step.scrub("text", "") == "text"
    assert agent_step.scrub([f"{KEY} {OPUS_KEY}"], [OPUS_KEY, "", KEY]) == [f"{agent_step.REDACTED} {agent_step.REDACTED}"]
    # The session's result.json held the key, so its copy is not packaged either.
    jobs.package(J)
    assert set(zipfile.ZipFile(J / "results.zip").namelist()) == {"posed.glb"}
    assert "left out (they hold the API key): ['agent_result.json']" in capsys.readouterr().out


def test_agent_step_scrubs_every_key_of_a_shared_env_file(tmp_path, monkeypatch, capsys):
    """With both keys in the env file (the Anthropic one first), neither may reach the stage log or
    agent/result.json, and find_key reports the files that hold either."""
    from gator.agent import controller as C

    J = run_folder(tmp_path)
    (J / "key.env").write_text(f"ANTHROPIC_API_KEY={OPUS_KEY}\nOPENAI_API_KEY={KEY}\n")

    def container(root, sif, key_file, budget_s):
        r = {"mode": "refinement", "source": "finish", "verdict": "complete", "glb": None, "exit": "finished",
             "cli_exit": "success", "timings": {"wall_s": 3.0}, "reason": f"keys {KEY} and {OPUS_KEY}",
             "limitations": [], "rejected": [], "network_commands": []}
        write_json(Path(root) / "result.json", r)
        (Path(root) / "work" / "leak.txt").write_text(KEY)
        r["key_found_in"] = C.find_key(root, key_file)
        write_json(Path(root) / "result.json", r)
        return r

    monkeypatch.setattr(C, "run_container_session", container)
    r = agent_step.run(J, "/sif/agent.sif", J / "key.env")
    out, saved = capsys.readouterr().out, (J / "agent" / "result.json").read_text()
    assert r["key_found_in"] == ["result.json", "work/leak.txt"]
    assert r["reason"] == f"keys {agent_step.REDACTED} and {agent_step.REDACTED}"
    for key in (KEY, OPUS_KEY):
        assert key not in out and key not in saved


def test_agent_step_sigterm_stops_the_agent(tmp_path):
    """The web app's cancel signals the step's group; the agent runs in its own session, so the step
    itself must stop it."""
    J = run_folder(tmp_path)
    wrapper = textwrap.dedent(f"""\
        import sys, tempfile
        from gator.agent import controller as C
        from gator.webapp import agent_step
        AGENT = [sys.executable, "-c", "import os, time; f = open('agent.pid', 'w'); f.write(str(os.getpid())); "
                 "f.close(); time.sleep(120)"]
        C.run_container_session = lambda root, sif, key_file, budget_s: C.run_session(
            root, AGENT, blender={FAKE_BLENDER!r}, sock_dir=tempfile.mkdtemp(prefix="gator-"))
        agent_step.main(sys.argv[1:])
        """)
    p = subprocess.Popen([sys.executable, "-c", wrapper, "--job", str(J), "--sif", "x.sif", "--key-file", str(J / "key.env")],
                         cwd=ROOT, start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    pid_file = J / "agent" / "s" / "work" / "agent.pid"
    t0 = time.time()
    while not (pid_file.exists() and pid_file.read_text()) and time.time() - t0 < 60:
        assert p.poll() is None, p.stdout.read().decode()
        time.sleep(0.1)
    agent = int(pid_file.read_text())
    os.kill(agent, 0)
    os.killpg(p.pid, signal.SIGTERM)
    assert p.wait(15) == 143
    with pytest.raises(ProcessLookupError):
        os.kill(agent, 0)
    assert not (J / "agent" / "result.json").exists()


def test_fake_build_end_to_end(make_runner, tmp_path):
    for m in ("av", "trimesh", "gator.webapp.video", "gator.webapp.views", "gator.webapp.physics_auto"):
        pytest.importorskip(m)
    from test_webapp_server import write_video

    runner, cfg = make_runner()
    job_id, upload = jobs.create_job(cfg.runs, "phone clip.mov")
    J = cfg.runs / job_id
    write_video(upload)
    jobs.update(J, video={**jobs.load(J)["video"], "bytes": upload.stat().st_size})
    runner.submit_analyze(job_id)
    job = wait(J, "ready")
    assert (J / "frames" / "meta.json").is_file() and job["stages"]["frames"]["status"] == "done"
    assert job["stages"]["topics"]["status"] == "skipped" and job["stages"]["topics"]["note"] == "video input"

    jobs.update(J, mode="auto", prompt="Wooden box", n_views=4, auto_continue=False)
    runner.submit_build(job_id, KEY)
    job = wait(J, "review")
    assert {s: job["stages"][s]["status"] for s in ("frames", "extract", "masks", "prep", "da3", "lidar", "select")} == {
        "frames": "done", "extract": "done", "masks": "done", "prep": "done", "da3": "done", "lidar": "skipped",
        "select": "done"}
    assert job["stages"]["lidar"]["note"] == "turned off: no LiDAR" and not (J / "pool" / "lidar.json").exists()
    assert job["stages"]["sam3d"]["status"] == "pending" and job["slug"] == "wooden_box"
    assert not list(runner.key_dir.iterdir())
    sel = json.loads((J / "selection.json").read_text())
    assert sel["ok"] and len(sel["picks"]) >= 3, sel["warnings"]

    # Leave one picked frame out with its instance override: masks, prep and select rerun, da3 does not.
    left_out = sel["picks"][1]["name"]
    da3_started = job["stages"]["da3"]["started"]
    jobs.update(J, overrides={**job["overrides"], "instance": {left_out: "none"}})
    runner.resume(job_id, "masks")
    job = wait(J, "review")
    assert job["stages"]["da3"]["started"] == da3_started and job["stages"]["masks"]["started"] > da3_started
    sam = json.loads((J / "scene" / "sam3_wooden_box" / "sam3.json").read_text())
    assert next(v for v in sam["views"] if v["name"] == left_out)["mask"] is None
    sel = json.loads((J / "selection.json").read_text())
    assert left_out not in [p["name"] for p in sel["picks"]]
    assert next(c for c in sel["candidates"] if c["name"] == left_out)["status"] == "no_mask"

    runner.resume(job_id, "sam3d")
    job = wait(J, "done")
    rigid = ("propose", "parts")  # a rigid object skips the parts stages, and a video the LiDAR
    assert [s for s in steps.NAMES if job["stages"][s]["status"] != "done"] == ["topics", "lidar", "propose", "parts",
                                                                                "turntable"]
    assert job["stages"]["turntable"]["note"] == "turned off" and not (J / "turntable").exists()
    assert all(job["stages"][s]["note"] == "turned off: rigid object" for s in rigid)
    placement = json.loads((J / "run" / "placement.json").read_text())
    assert placement["metric_scale"] is None and placement["units"] == "DA3 units (meters)"
    assert job["key_ok"] is False and job["error"] is None
    assert steps.next_glb(J) == J / "agent" / "s" / "work" / "exports" / "001.glb"
    up = json.loads((J / "physics" / "up.json").read_text())
    assert up["glb"].endswith("work/exports/001.glb")
    assert not (J / "physics" / "links").exists() and "links" not in json.loads((J / "physics" / "summary.json").read_text())
    names = set(zipfile.ZipFile(J / "results.zip").namelist())
    assert {"posed.glb", "refined.glb", "aligned.glb", "summary.json", "placement.json", "selection.json",
            "agent_result.json", "mjcf/mjcf.json", "mjcf/wooden_box.xml", "mjcf/scene.xml", "usd/usd.json",
            "urdf/model.urdf", "urdf/urdf.json"} <= names and not {"parts.glb", "parts.json"} & names
    urdf = json.loads((J / "physics" / "urdf" / "urdf.json").read_text())
    assert urdf["checked"]["links"] == 1 and urdf["checked"]["joints"] == 0 and urdf["checked"]["missing"] == []
    skipped = ("topics", "lidar", *rigid, "turntable")  # topics and lidar: video input; no 360 video asked for
    assert all(s["seconds"] is not None for s in jobs.progress(J) if s["name"] not in skipped)
    assert not list(runner.key_dir.iterdir())
    assert not any(KEY.encode() in p.read_bytes() for p in all_files(tmp_path))


@pytest.mark.parametrize("mode, agent", [("vlm", "astra"), ("vlm_edit", "opus"), ("user", "none")])
def test_fake_articulated_build_end_to_end(make_runner, tmp_path, mode, agent):
    """The fake box with a lid, per parts mode: "vlm" runs through, "vlm_edit" stops for the list after
    propose, and "user" (here without an agent) takes the typed list, whose parts get no joints."""
    for m in ("av", "trimesh", "mujoco", "gator.webapp.video", "gator.webapp.views", "gator.webapp.physics_auto"):
        pytest.importorskip(m)
    from gator.parts import joints as joint_spec
    from gator.parts.stage import mesh_nodes
    from gator.physics.align import read_glb
    from test_webapp_server import write_video

    runner, cfg = make_runner()
    key = {"astra": KEY, "opus": OPUS_KEY, "none": None}[agent]
    job_id, upload = jobs.create_job(cfg.runs, "lid clip.mov")
    J = cfg.runs / job_id
    write_video(upload)
    runner.submit_analyze(job_id)
    wait(J, "ready")
    jobs.update(J, mode="auto", prompt="Wooden box", n_views=4, auto_continue=True)
    typed = [{"name": "box body"}, {"name": "lid", "parent": "box body", "joint": "revolute"},
             {"name": "drawer", "parent": "box body", "joint": "prismatic"}]
    runner.submit_build(job_id, key, agent=agent, articulated=True, parts={"mode": mode, "list": typed if mode == "user" else []},
                        turntable=mode == "vlm")  # one case asks for the 360 video
    base, lid = ("box_body", "lid") if mode == "user" else ("body", "lid")
    if mode == "vlm_edit":  # auto_continue does not skip this pause
        job = wait(J, "review")
        assert steps.waits_for_parts(J, job) and job["stages"]["parts"]["status"] == "pending"
        proposal = json.loads((J / "run" / "proposal.json").read_text())
        assert (proposal["source"], proposal["model"]) == ("vlm", "claude-opus-5-5")
        assert [p["name"] for p in proposal["parts"]] == ["body", "lid"]
        lid = "top_lid"
        assert runner.resume(job_id, "sam3d", confirm_parts=True,
                             parts_list=[{"name": "body"}, {"name": "top lid", "parent": "body", "joint": "revolute"}]) == "parts"
    job = wait(J, "done", "failed")
    assert job["status"] == "done", (job["error"], (J / "logs" / f"{job['error'].split(':')[0]}.log").read_text()[-3000:]
                                     if job["error"] else "")
    skipped = {"topics", "lidar"} | ({"agent"} if agent == "none" else set()) | ({"turntable"} if mode != "vlm" else set())
    assert all(job["stages"][s]["status"] == ("skipped" if s in skipped else "done") for s in steps.NAMES), job["stages"]
    assert job["parts"]["mode"] == mode and job["parts"]["confirmed"] is (mode == "vlm_edit")

    # The split: the base and the lid, the seam's four lines; a third typed part is not found.
    parts = json.loads((J / "run" / "parts.json").read_text())
    found = [p["name"] for p in parts["parts"] if p["status"] == "found"]
    assert found == [base, lid] and [p["parent"] for p in parts["parts"][:2]] == [None, base]
    assert [p["prompt"] for p in parts["parts"] if p["status"] == "not_found"] == (["drawer"] if mode == "user" else [])
    assert len(parts["seams"]) == 1 and len(parts["seams"][0]["lines"]) == 4
    assert sorted(n for n, _, _ in mesh_nodes(J / "run" / "parts.glb")) == sorted(found)
    cams = json.loads((J / "run" / "cameras.json").read_text())
    assert all((J / "run" / "parts" / "sam3" / "check" / f"{c['name']}.jpg").is_file() for c in cams)
    with np.load(J / "run" / "parts" / "faces.npz") as f:
        assert list(f["names"]) == found and set(f["part"].tolist()) == {0, 1}

    aligned = json.loads((J / "physics" / "aligned.json").read_text())
    assert sorted(p.stem for p in (J / "physics" / "links").glob("*.glb")) == sorted(found)
    summary = json.loads((J / "physics" / "summary.json").read_text())
    assert sorted(summary["links"]) == sorted(found) and summary["rigid_body_covers"] == sorted(found)
    assert np.isclose(summary["rigid_body"]["mass"], sum(r["rigid_body"]["mass"] for r in summary["links"].values()))
    mj = json.loads((J / "physics" / "mjcf" / "mjcf.json").read_text())
    urdf = json.loads((J / "physics" / "urdf" / "urdf.json").read_text())["checked"]
    zipped = set(zipfile.ZipFile(J / "results.zip").namelist())
    assert {"parts.glb", "parts.json", "urdf/model.urdf", "mjcf/scene.xml", "usd/usd.json"} <= zipped
    assert ("turntable.mp4" in zipped) is (mode == "vlm")
    if mode == "vlm":  # the 360 video of the agent's parts with joints, all parts in it
        rec = json.loads((J / "turntable" / "turntable.json").read_text())
        assert (rec["glb"], rec["codec"], rec["frames"]) == ("agent/s/work/exports/001.glb", "h264", 240)
    if agent == "none":  # no joints: physics uses the split parts as one body
        assert job["stages"]["agent"]["note"].startswith("turned off: without the agent the parts get no joints")
        assert steps.next_glb(J) == J / "run" / "parts.glb" and steps.agent_glb(J) is None
        assert aligned["joints"] == {} and "links" not in mj and (urdf["links"], urdf["joints"]) == (1, 0)
        assert "refined.glb" not in zipped
    else:
        r = json.loads((J / "agent" / "result.json").read_text())
        assert (r["mode"], r["verdict"], r["joints_delivered"]) == ("articulation", "articulated", 1)
        assert set(r["counts"]) == {"exports", "reviews", "checkpoints", "joints", "articulation_reviews"}
        assert steps.next_glb(J) == steps.agent_glb(J) == J / "agent" / "s" / "work" / "exports" / "001.glb"
        assert joint_spec.read_joints(read_glb(steps.next_glb(J))[0])[lid]["frame"] == "W"
        assert set(aligned["joints"]) == {lid} and aligned["joints"][lid]["frame"] == "S"
        assert (mj["root"], list(mj["links"]), mj["compiled"]["njnt"]) == (base, found, 2)
        assert [(j["child"], j["parent"], j["type"]) for j in mj["joints"]] == [(lid, base, "revolute")]
        probe = mj["compiled"]["probe"][0]  # the lid's front edge goes up as it opens
        assert probe["max_move_m"] > 0.1 and probe["upper"][1][2] > probe["rest"][1][2] + 0.1
        assert (urdf["links"], urdf["joints"], urdf["missing"]) == (2, 1, [])
        usd = json.loads((J / "physics" / "usd" / "usd.json").read_text())["checked"]
        assert [j["type"] for j in usd["joints"]] == ["PhysicsRevoluteJoint"] and "refined.glb" in zipped
    assert not list(runner.key_dir.iterdir())
    assert not any(k.encode() in p.read_bytes() for p in all_files(tmp_path) for k in (KEY, OPUS_KEY))
