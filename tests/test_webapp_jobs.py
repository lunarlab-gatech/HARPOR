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

from harpor.fileio import write_json
from harpor.stage_a import lidar_scale
from harpor.webapp import agent_step, jobs, steps

ROOT = Path(__file__).resolve().parents[1]
KEY = "sk-fake-" + secrets.token_hex(16)
ICE = steps.Config(G=Path("/storage/ice1/5/0/ali497/gator"), runs=Path("/storage/ice1/5/0/ali497/gator/webapp_runs"),
                   code=Path("/storage/ice1/5/0/ali497/gator/HARPOR"))
JOB = {"id": "20261015-120000-abcdef", "prompt": "coffee cup", "slug": "coffee_cup", "agent": "astra",
       "overrides": {"locked": [], "excluded": [], "instance": {"f00012": "1", "f00020": "none"}}}
# The spec's environment lines, copied from our ICE scripts, with the thread count open to the server's environment.
STAGE_A = ('HF_HOME=$G/.cache/huggingface TORCH_HOME=$G/.cache/torch TORCH_EXTENSIONS_DIR=$G/.cache/torch_extensions '
           'OMP_NUM_THREADS=${OMP_NUM_THREADS:-16} CONDA_PREFIX=$G/envs/sam3d '
           'CPATH=$G/envs/sam3d/targets/x86_64-linux/include '
           'LIBRARY_PATH=$G/envs/sam3d/targets/x86_64-linux/lib:$G/envs/sam3d/lib PATH=$G/envs/sam3d/bin:$PATH')
VOMP = ('eval "$($G/tools/micromamba shell hook -s bash)"; micromamba activate $G/envs/vomp11',
        'export CUDA_HOME=$CONDA_PREFIX HF_HOME=$G/.cache/huggingface TORCH_HOME=$G/.cache/torch '
        'TORCH_EXTENSIONS_DIR=$G/.cache/torch_extensions_vomp11 OMP_NUM_THREADS=${OMP_NUM_THREADS:-16} '
        'CPATH=$CONDA_PREFIX/targets/x86_64-linux/include LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/lib:$CONDA_PREFIX/lib')


@pytest.fixture
def make_runner(tmp_path, fake_stages):
    """Runners whose stages are the stand-ins of tests/fake_steps.py, unless a test puts its own steps.command."""
    made = []

    def make(**kw):
        cfg = steps.Config(G=tmp_path / "G", runs=tmp_path / "runs")
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
    scripts = {s: steps.command(s, JOB, ICE, key_file="/dev/shm/harpor-webapp-1/k.env") for s in steps.NAMES
               if s not in ("mjcf", "usd", "urdf")}
    for s, (argv, cwd, env) in scripts.items():
        assert argv[:2] == ["bash", "-c"] and cwd == "/storage/ice1/5/0/ali497/gator/HARPOR"
        assert argv[2].startswith("set -eo pipefail\nG=/storage/ice1/5/0/ali497/gator\n"
                                  "J=/storage/ice1/5/0/ali497/gator/webapp_runs/20261015-120000-abcdef\n"
                                  "C=/storage/ice1/5/0/ali497/gator/HARPOR\n")
        assert ("export " + STAGE_A in argv[2]) == (s in ("prep", "da3", "sam3d", "place", "agent", "align",
                                                          "estimate", "turntable")), s
        assert env == {"PYTHONUNBUFFERED": "1"} and KEY not in argv[2]
    sc = {s: v[0][2] for s, v in scripts.items()}
    assert sc["masks"].endswith("\nexport HF_HOME=$G/.cache/huggingface\nrm -f \"$J/scene/sam3_coffee_cup/sam3.json\"\n"
                                "cd $C && $G/envs/sam3seg/bin/python -m harpor.stage_a.sam3_masks --scene "
                                "\"$J/scene\" --prompt='coffee cup' --pick f00012=1 --pick f00020=none")
    assert sc["prep"].endswith('cd $C && $G/envs/sam3d/bin/python -m harpor.stage_a.mvsam3d prep '
                               '--scene "$J/scene" --mask-prompt sam3_coffee_cup --out "$J/pool"')
    assert sc["da3"].endswith('$G/envs/da3/bin/python -m harpor.stage_a.mvsam3d da3 --da3-root $G/Depth-Anything-3 --out "$J/pool"')
    assert sc["sam3d"].endswith("$G/envs/sam3d/bin/python -m harpor.stage_a.mvsam3d sam3d --mv-root $G/MV-SAM3D "
                                '--sam3d-root $G/sam-3d-objects --out "$J/run"')
    assert sc["agent"].endswith('-m harpor.webapp.agent_step --job "$J" --sif $G/sif/harpor-agent.sif '
                                "--key-file /dev/shm/harpor-webapp-1/k.env --agent astra --lidar-geometry inspect")
    opus = steps.command("agent", {**JOB, "agent": "opus"}, ICE, key_file="/dev/shm/harpor-webapp-1/k.env")[0][2]
    assert opus == sc["agent"].replace("--agent astra", "--agent opus")
    sculpt = steps.command("agent", {**JOB, "lidar_geometry": "sculpt"}, ICE, key_file="/dev/shm/harpor-webapp-1/k.env")
    assert sculpt[0][2] == sc["agent"].replace("--lidar-geometry inspect", "--lidar-geometry sculpt")
    # The estimate stage: the job's model in envs/sam3d, with the same key file and agent.
    assert sc["align"].endswith(f'\nexport {STAGE_A}\ncd $C && $G/envs/sam3d/bin/python -m harpor.webapp.physics_auto '
                                'align --job "$J"')
    assert sc["estimate"].endswith(f'\nexport {STAGE_A}\ncd $C && $G/envs/sam3d/bin/python -m '
                                   'harpor.webapp.estimate_step --job "$J" --key-file /dev/shm/harpor-webapp-1/k.env '
                                   '--agent astra')
    for a in ("opus", "fable"):
        est = steps.command("estimate", {**JOB, "agent": a}, ICE, key_file="/dev/shm/harpor-webapp-1/k.env")[0][2]
        assert est == sc["estimate"][:-len("astra")] + a
    assert "\n".join(VOMP) in sc["vomp"] and sc["vomp"].endswith(
        'cd $G/VoMP && python $C/harpor/physics/vomp_run.py --glb "$J/run/canonical.glb" --out "$J/run/vomp" --opaque-input')
    assert sc["topics"].endswith('$G/envs/web/bin/python -m harpor.webapp.video topics --job "$J"')
    assert sc["frames"].endswith('$G/envs/web/bin/python -m harpor.webapp.video analyze --job "$J"')
    assert sc["package"].endswith('$G/envs/web/bin/python -m harpor.webapp.jobs package --job "$J"')
    # The 360 video: Blender renders the frames in the SIF (envs/sam3d), and envs/web encodes them.
    assert sc["turntable"].endswith(
        f'\nrm -rf "$J/turntable"\nexport {STAGE_A}\n'
        'cd $C && $G/envs/sam3d/bin/python -m harpor.webapp.turntable render --job "$J" '
        '--sif $G/sif/harpor-agent.sif\ncd $C && $G/envs/web/bin/python -m harpor.webapp.turntable encode --job "$J"')
    assert sc["lidar"] == ('set -eo pipefail\nG=/storage/ice1/5/0/ali497/gator\n'
                           'J=/storage/ice1/5/0/ali497/gator/webapp_runs/20261015-120000-abcdef\n'
                           'C=/storage/ice1/5/0/ali497/gator/HARPOR\n'
                           'cd $C && $G/envs/web/bin/python -m harpor.webapp.lidar scale --job "$J"')
    import fake_steps  # the tests' stand-in for steps.command refuses the same calls

    for s in ("agent", "estimate"):
        with pytest.raises(ValueError, match=f"the {s} stage needs the key file"):
            steps.command(s, JOB, ICE)
        with pytest.raises(ValueError, match=f"the {s} stage needs the key file"):
            fake_steps.command(s, JOB, ICE)
        fake = fake_steps.command(s, {**JOB, "agent": "fable"}, ICE, key_file="/k/x.env")
        assert fake[0][2] == (f"exec {sys.executable} {ROOT}/tests/fake_steps.py {s} --job "
                              "/storage/ice1/5/0/ali497/gator/webapp_runs/20261015-120000-abcdef --key-file /k/x.env "
                              "--agent fable")
        assert fake[1] == str(ROOT)
    # Only the agent and estimate stages take the key file: every other stage runs without one, also with an agent.
    for s in set(steps.NAMES) - {"agent", "estimate", "mjcf", "usd", "urdf"}:
        assert steps.command(s, JOB, ICE)[0][2] == sc[s], s
        assert "--key-file" not in fake_steps.command(s, JOB, ICE)[0][2], s
    for s in ("agent", "estimate"):  # a job without an agent skips both (steps.skip), and has no key to give them
        assert steps.command(s, {**JOB, "agent": "none"}, ICE)[0][2] == sc[s].replace(
            " --key-file /dev/shm/harpor-webapp-1/k.env --agent astra", ""), s



def test_masks_command_takes_a_noun_that_starts_with_a_dash(monkeypatch):
    import shlex

    from harpor.stage_a import sam3_masks

    ran = []
    monkeypatch.setattr(sam3_masks, "run", lambda scene, prompt, picks: ran.append(prompt))
    for noun in ("-cup", "--mug", "coffee cup"):
        script = steps.command("masks", {**JOB, "prompt": noun}, ICE)[0][2]
        sam3_masks.main(shlex.split(script.split("-m harpor.stage_a.sam3_masks ", 1)[1]))
        assert ran[-1] == noun


def test_stages_run_from_the_config_code(tmp_path):
    assert steps.Config(G=tmp_path, runs=tmp_path).code == ROOT  # the repo of the running server
    cfg = steps.Config(G=tmp_path / "G", runs=tmp_path / "runs", code=Path("/my repo"))
    for s in set(steps.NAMES) - {"mjcf", "usd", "urdf"}:
        argv, cwd, _ = steps.command(s, JOB, cfg, key_file="/k/x.env")
        assert cwd == "/my repo" and "\nC='/my repo'\n" in argv[2] and str(ICE.code) not in argv[2], s


def test_physics_commands_name_the_object():
    pytest.importorskip("harpor.webapp.physics_auto")
    mjcf, usd = (steps.command(s, JOB, ICE)[0][2] for s in ("mjcf", "usd"))
    assert mjcf.endswith('rm -rf "$J/physics/mjcf"\ncd $C && $G/envs/mujoco/bin/python -m harpor.physics.mjcf '
                         '--glb "$J/physics/aligned.glb" --summary "$J/physics/summary.json" --name coffee_cup '
                         '--out "$J/physics/mjcf"')
    assert usd.endswith('$G/envs/mujoco/bin/python -m harpor.physics.usd --glb "$J/physics/aligned.glb" --summary '
                        '"$J/physics/summary.json" --name coffee_cup --mjcf "$J/physics/mjcf" --out "$J/physics/usd"')
    assert steps.command("urdf", JOB, ICE)[0][2].endswith(  # one link for the rigid object
        'cd $C && $G/envs/mujoco/bin/python -m harpor.physics.urdf --glb "$J/physics/aligned.glb" '
        '--summary "$J/physics/summary.json" --name coffee_cup --mjcf "$J/physics/mjcf" --out "$J/physics/urdf"')


def test_stage_scripts_set_their_environment(tmp_path, monkeypatch):
    """The scripts run as written: stand-in interpreters and micromamba report what they were given."""
    monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    G = tmp_path / "G"
    report = textwrap.dedent("""\
        #!/bin/bash
        echo "ran $0 $*"
        echo "pwd $(pwd -P)"
        for v in HF_HOME CONDA_PREFIX TORCH_CUDA_ARCH_LIST TORCH_EXTENSIONS_DIR CPATH LIBRARY_PATH CUDA_HOME OMP_NUM_THREADS; do
          echo "$v=${!v}"
        done
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
    for d in ("HARPOR", "VoMP"):
        (G / d).mkdir()
    cfg = steps.Config(G=G, runs=tmp_path / "runs", code=G / "HARPOR")
    J = cfg.runs / JOB["id"]

    def run(stage):
        argv, cwd, env = steps.command(stage, JOB, cfg)
        p = subprocess.run(argv, cwd=cwd, env={**os.environ, **env}, capture_output=True, text=True)
        assert p.returncode == 0, p.stderr
        return dict(line.split("=", 1) for line in p.stdout.splitlines()[2:]), p.stdout.splitlines()[:2]

    env, (ran, pwd) = run("prep")
    G = G.resolve()
    assert ran == f"ran {G}/envs/sam3d/bin/python -m harpor.stage_a.mvsam3d prep --scene {J}/scene " \
                  f"--mask-prompt sam3_coffee_cup --out {J}/pool" and pwd == f"pwd {G}/HARPOR"
    assert env["CONDA_PREFIX"] == f"{G}/envs/sam3d" and env["TORCH_CUDA_ARCH_LIST"] == ""  # no stage picks the GPU arch
    assert env["PATH0"] == f"{G}/envs/sam3d/bin" and env["TORCH_EXTENSIONS_DIR"] == f"{G}/.cache/torch_extensions"
    assert env["LIBRARY_PATH"] == f"{G}/envs/sam3d/targets/x86_64-linux/lib:{G}/envs/sam3d/lib"
    env, (ran, pwd) = run("vomp")
    assert ran == f"ran {G}/envs/vomp11/bin/python {G}/HARPOR/harpor/physics/vomp_run.py " \
                  f"--glb {J}/run/canonical.glb --out {J}/run/vomp --opaque-input" and pwd == f"pwd {G}/VoMP"
    assert env["CUDA_HOME"] == env["CONDA_PREFIX"] == f"{G}/envs/vomp11"
    assert env["TORCH_EXTENSIONS_DIR"] == f"{G}/.cache/torch_extensions_vomp11" and env["TORCH_CUDA_ARCH_LIST"] == ""
    assert env["CPATH"] == f"{G}/envs/vomp11/targets/x86_64-linux/include" and env["HF_HOME"] == f"{G}/.cache/huggingface"

    # The thread count is 16 unless the server's environment sets it, in the Stage A line and in VoMP's after activation.
    for stage in ("prep", "vomp"):
        assert run(stage)[0]["OMP_NUM_THREADS"] == "16"
        monkeypatch.setenv("OMP_NUM_THREADS", "")
        assert run(stage)[0]["OMP_NUM_THREADS"] == "16"
        monkeypatch.setenv("OMP_NUM_THREADS", "4")
        assert run(stage)[0]["OMP_NUM_THREADS"] == "4"
        monkeypatch.delenv("OMP_NUM_THREADS")

    # The masks stage: SAM 3 in envs/sam3seg with HF_HOME only, not the Stage A environment of envs/sam3d.
    env, (ran, pwd) = run("masks")
    assert ran == f"ran {G}/envs/sam3seg/bin/python -m harpor.stage_a.sam3_masks --scene {J}/scene " \
                  "--prompt=coffee cup --pick f00012=1 --pick f00020=none" and pwd == f"pwd {G}/HARPOR"
    assert env["HF_HOME"] == f"{G}/.cache/huggingface" and env["CONDA_PREFIX"] != f"{G}/envs/sam3d"
    assert env["PATH0"] != f"{G}/envs/sam3d/bin" and env["TORCH_EXTENSIONS_DIR"] != f"{G}/.cache/torch_extensions"


def test_config_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("HARPOR_ROOT", str(tmp_path))
    monkeypatch.delenv("WEBAPP_RUNS", raising=False)
    cfg = steps.config_from_env()
    assert cfg.G == tmp_path and cfg.runs == tmp_path / "webapp_runs"
    monkeypatch.setenv("WEBAPP_RUNS", str(tmp_path / "r"))
    assert steps.config_from_env().runs == tmp_path / "r"
    assert [s["name"] for s in steps.STAGES if s["gpu"]] == ["masks", "da3", "sam3d", "vomp", "agent", "turntable"]


def test_estimate_stage_order():
    """VoMP runs right after sam3d, on the first mesh, before place and the agent. The estimate stage runs after
    align, which writes summary.json from VoMP's materials, and before mjcf, usd and urdf, which read the friction
    that it adds there. It needs no GPU. A rerun deletes its photo sheet and nothing of align: summary.json is
    align's file."""
    n = steps.NAMES
    assert n[n.index("sam3d"):n.index("agent")] == ["sam3d", "vomp", "place"]
    assert n[n.index("align"):n.index("turntable")] == ["align", "estimate", "mjcf", "usd", "urdf"]
    assert "estimate" in jobs.BUILD and "estimate" not in steps.GPU and "vomp" in jobs.BUILD and "vomp" in steps.GPU
    assert jobs.DERIVED["estimate"] == ("physics/estimate_views.jpg",) and jobs.DERIVED["align"] == ("physics",)
    assert jobs.DERIVED["vomp"] == ("run/vomp",)
    assert steps.KEYED == ("agent", "estimate")


def test_config_from_env_takes_only_harpor_root(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("HARPOR_ROOT", raising=False)
    monkeypatch.delenv("HARPOR_SANDBOX", raising=False)
    cfg = steps.config_from_env()
    assert cfg.G == tmp_path / "harpor" and cfg.sandbox is None


def test_the_sandbox_setting_reaches_the_stages_that_use_blender(monkeypatch, tmp_path):
    """HARPOR_SANDBOX names the docker image; without it the stages take the SIF under the root."""
    job = {"id": "j1", "prompt": "cup", "slug": "cup", "agent": "none", "turntable": True, "overrides": {}}
    cfg = steps.Config(G=tmp_path, runs=tmp_path / "runs")
    assert "--sif $G/sif/harpor-agent.sif" in steps.command("turntable", job, cfg)[0][2]
    monkeypatch.setenv("HARPOR_SANDBOX", "docker:harpor:0.1")
    monkeypatch.setenv("HARPOR_ROOT", str(tmp_path))
    cfg = steps.config_from_env()
    assert cfg.sandbox == "docker:harpor:0.1"
    assert "--sif docker:harpor:0.1" in steps.command("turntable", job, cfg)[0][2]


def test_done_markers_guard_and_skip(tmp_path):
    J = tmp_path / "job"
    J.mkdir()
    (J / "job.json").write_text(json.dumps({"prompt": "coffee cup", "slug": "coffee_cup", "agent": "astra", "turntable": False}))
    assert not any(steps.done(s, J) for s in steps.NAMES)
    (J / "frames").mkdir()
    (J / "frames" / "pool.json").write_text('{"pool": [3')  # a marker that does not parse is not done
    assert not steps.done("extract", J)
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
    # The 360 video: skipped unless the job asks for it; done when turntable.json is there.
    assert steps.skip("turntable", J) == "turned off" and not steps.turntable({"turntable": "yes"})
    (J / "job.json").write_text(json.dumps({"prompt": "coffee cup", "slug": "coffee_cup", "agent": "astra", "turntable": True}))
    assert steps.skip("turntable", J) is None and steps.turntable(jobs.load(J))
    (J / "turntable").mkdir()
    (J / "turntable" / "turntable.mp4").write_bytes(b"")
    assert not steps.done("turntable", J)
    (J / "turntable" / "turntable.json").write_text("{}")
    assert steps.done("turntable", J) and steps.guard("turntable", J) is None
    assert steps.OPTIONAL == {"turntable": "360 video"} and jobs.DERIVED["turntable"] == ("turntable",)
    # align is done with its record and the summary of VoMP's materials moved onto the final mesh. estimate is done
    # when that summary holds the model's "estimate" and the photo sheet is there.
    (J / "physics").mkdir()
    (J / "physics" / "aligned.glb").write_bytes(b"glTF")
    (J / "physics" / "aligned.json").write_text("{}")
    assert not steps.done("align", J) and not steps.done("estimate", J)
    vomp_summary = {"glb": "aligned.glb", "config": "vomp", "median": {"density_kg_m3": 600.0},
                    "rigid_body": {"mass": 2.0, "com": [0, 0, 0], "fullinertia": [1, 1, 1, 0, 0, 0]}}
    for text in (json.dumps(vomp_summary), "{", "[]", json.dumps({**vomp_summary, "friction": 0.4})):
        (J / "physics" / "summary.json").write_text(text)
        assert not steps.done("estimate", J) and steps.done("align", J), text
    (J / "physics" / "summary.json").write_text(json.dumps({**vomp_summary, "friction": 0.4, "estimate": {}}))
    assert not steps.done("estimate", J)  # no photo sheet
    (J / "physics" / "estimate_views.jpg").write_bytes(b"jpg")
    assert steps.done("estimate", J) and steps.guard("estimate", J) is None and steps.skip("estimate", J) is None
    jobs._clear(J, ["estimate"])  # a rerun of estimate keeps align's files, the summary too
    assert not steps.done("estimate", J) and steps.done("align", J)
    assert sorted(p.name for p in (J / "physics").iterdir()) == ["aligned.glb", "aligned.json", "summary.json"]
    (J / "physics" / "mjcf").mkdir(parents=True)
    (J / "physics" / "mjcf" / "mjcf.json").write_text(json.dumps({"name": "x"}))
    assert not steps.done("mjcf", J)
    (J / "physics" / "mjcf" / "mjcf.json").write_text(json.dumps({"name": "x", "compiled": {}}))
    assert steps.done("mjcf", J)
    # USD and URDF are done once their check is in the record.
    for s in ("usd", "urdf"):
        (J / "physics" / s).mkdir()
        (J / "physics" / s / f"{s}.json").write_text(json.dumps({"name": "x"}))
        assert not steps.done(s, J)
        (J / "physics" / s / f"{s}.json").write_text(json.dumps({"name": "x", "checked": {"links": 1}}))
        assert steps.done(s, J) and steps.guard(s, J) is None
    assert steps.skip("agent", J) is None
    (J / "run" / "placement.json").write_text(json.dumps({"check": {"mean_mask_iou": 0.31}}))
    assert "0.31 is below 0.5" in steps.skip("agent", J) and steps.skip("align", J) is None
    # A poor placement skips only the agent: the job's model still gives the friction.
    assert not any(steps.skip(s, J) for s in ("estimate", "mjcf", "usd", "urdf"))


def test_the_agent_and_estimate_stages_take_the_key(tmp_path):
    """needs_key is true for the agent and estimate stages of a job with an agent, and for no other stage. A job
    without one has no stage that takes a key: its agent stage is skipped, so physics uses the Stage A mesh, and
    estimate, mjcf, usd and urdf are skipped, since only the job's model gives the friction. vomp and align still
    run: VoMP gives the mass."""
    J = tmp_path
    job = {"prompt": "toy box", "slug": "toy_box", "agent": "astra"}
    for agent in ("astra", "opus", "fable"):
        assert [s for s in steps.NAMES if steps.needs_key(s, {**job, "agent": agent})] == ["agent", "estimate"]
    assert [s for s in steps.NAMES if steps.needs_key(s, {**job, "agent": "none"})] == []
    with pytest.raises(ValueError, match="agent is one of"):
        steps.needs_key("estimate", {**job, "agent": "gpt"})
    assert steps.NO_KEY_SKIP == "turned off: the friction needs the job's model and API key"
    (J / "job.json").write_text(json.dumps({**job, "agent": "none", "turntable": True}))
    assert steps.skip("agent", J) == "turned off: physics uses the Stage A mesh" and steps.skip("vomp", J) is None
    assert {s: steps.skip(s, J) for s in steps.NAMES[steps.NAMES.index("align"):]} == {
        "align": None, "estimate": steps.NO_KEY_SKIP, "mjcf": steps.NO_KEY_SKIP, "usd": steps.NO_KEY_SKIP,
        "urdf": steps.NO_KEY_SKIP, "turntable": None, "package": None}
    for agent in ("astra", "opus", "fable"):
        (J / "job.json").write_text(json.dumps({**job, "agent": agent}))
        assert not any(steps.skip(s, J) for s in ("agent", "estimate", "mjcf", "usd", "urdf")), agent
    (J / "job.json").write_text(json.dumps(job))
    (J / "run").mkdir()
    (J / "run" / "placement.json").write_text(json.dumps({"check": {"mean_mask_iou": 0.31}}))
    assert steps.skip("agent", J).endswith("physics uses run/posed.glb") and steps.skip("estimate", J) is None


def test_lidar_stage_done_skip_guard_and_derived(tmp_path):
    J = tmp_path
    (J / "job.json").write_text(json.dumps({"prompt": "box", "slug": "box", "lidar": None}))
    assert steps.lidar({"lidar": None}) is None
    assert steps.skip("lidar", J) == "turned off: no LiDAR" and not steps.done("lidar", J) and steps.guard("lidar", J) is None
    on = {"kind": "rosbag", "topic": "/livox/lidar"}
    assert steps.lidar({"lidar": {**on, "extra": 1}}) == on
    assert steps.lidar({"lidar": {"kind": "stray"}}) == steps.lidar({"lidar": {"kind": "stray", "x": 1}}) == {"kind": "stray"}
    for bad in ({"kind": "video"}, {"kind": "rosbag"}, {"kind": "rosbag", "topic": ""}, "yes", True, False, ["/x"], "stray"):
        with pytest.raises(ValueError, match=re.escape('lidar is null, {"kind": "rosbag", "topic": T} or {"kind": "stray"}')):
            steps.lidar({"lidar": bad})
    (J / "job.json").write_text(json.dumps({"prompt": "box", "lidar": {"kind": "rosbag"}}))
    with pytest.raises(ValueError, match="lidar is null"):
        steps.skip("lidar", J)
    (J / "job.json").write_text(json.dumps({"prompt": "box", "lidar": on}))
    assert steps.skip("lidar", J) is None
    (J / "pool").mkdir()
    why = "the bag's TF does not link cam to lidar; upload the calibration"
    (J / "pool" / "lidar.json").write_text(json.dumps({"ok": False, "s": None, "reason": why}))
    assert not steps.done("lidar", J) and steps.guard("lidar", J) == f"LiDAR scale: {why}"
    (J / "pool" / "lidar.json").write_text(json.dumps({"ok": "true", "s": 1.25}))
    assert not steps.done("lidar", J) and steps.guard("lidar", J) == "LiDAR scale: no reason given"
    ok = {"version": lidar_scale.VERSIONS["rosbag"], "ok": True, "s": 1.25, "reason": None}
    (J / "pool" / "lidar.json").write_text(json.dumps(ok))
    assert steps.done("lidar", J) and steps.guard("lidar", J) is None
    # A record of an older estimator, or one without a version, is not done: the lidar stage runs again.
    for old in ({**ok, "version": lidar_scale.VERSIONS["rosbag"] - 1}, {k: v for k, v in ok.items() if k != "version"}):
        (J / "pool" / "lidar.json").write_text(json.dumps(old))
        assert not steps.done("lidar", J) and steps.guard("lidar", J) is None
    # Each source has its own version: a Stray capture has no rays, so its scale of version 2 stays current, and a
    # bag's does not.
    (J / "pool" / "lidar.json").write_text(json.dumps({**ok, "version": 2}))
    assert lidar_scale.VERSIONS == {"rosbag": 3, "stray": 2} and not steps.done("lidar", J)
    (J / "job.json").write_text(json.dumps({"prompt": "box", "lidar": {"kind": "stray"}}))
    assert steps.done("lidar", J)
    (J / "pool" / "lidar.json").write_text(json.dumps(ok))
    assert not steps.done("lidar", J)
    # Without a known source there is no current version: a record without one is not done either, and a kind that
    # is not a string is no error (it was a TypeError, and the job's page answered 500).
    for lid in ({"kind": ["stray"]}, {"kind": "foo"}, {"topic": "/livox/lidar"}, "stray", True, None):
        (J / "job.json").write_text(json.dumps({"prompt": "box", "lidar": lid}))
        for rec in (ok, {**ok, "version": None}, {k: v for k, v in ok.items() if k != "version"}):
            (J / "pool" / "lidar.json").write_text(json.dumps(rec))
            assert not steps.done("lidar", J), (lid, rec)
    (J / "job.json").unlink()
    assert not steps.done("lidar", J)
    (J / "job.json").write_text(json.dumps({"prompt": "box", "lidar": {"kind": "rosbag"}}))
    (J / "pool" / "lidar.json").write_text("[]")
    assert not steps.done("lidar", J) and steps.guard("lidar", J) is None
    # A hand edit, a cut file or one nested too deep for json: not done, and no error.
    for bad in ("{", "", '{"ok": true, "version": 3', "[" * 100_000 + "]" * 100_000):
        (J / "pool" / "lidar.json").write_text(bad)
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

    import fake_steps
    from harpor.webapp import rosbag

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
    assert jobs.lidar_settings(J, job, None) is None
    assert jobs.lidar_settings(J, job, {"topic": POINTS}) == on and jobs.lidar_settings(J, job, on) == on
    assert reason(J, job, {"topic": "/nope"}) == "/nope is not a LiDAR topic of the bag; its LiDAR topics are /lidar/points"
    for bad in ({"topic": ""}, {"kind": "stray", "topic": POINTS}, POINTS, True, False):
        assert reason(J, job, bad) == 'lidar is null or {"topic": T}, a LiDAR topic of the bag', bad
    assert reason(J, {**job, "source": {"kind": "video", "topic": None}}, on) == \
        "LiDAR needs a rosbag or a Stray Scanner input; this job's input is a video, so turn LiDAR off"
    assert reason(J, {**job, "source": {"kind": "rosbag", "topic": None}}, on) == "choose the bag's camera topic first"
    topics = json.loads((J / "frames" / "topics.json").read_text())
    (J / "frames" / "topics.json").write_text(json.dumps({k: v for k, v in topics.items() if k != "lidar"}))
    assert reason(J, job, on) == "/lidar/points is not a LiDAR topic of the bag; the bag has none"
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
    # under the lock; one for this topic passes, and one that names none is not checked.
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
    # Another tab changed the calibration after the server's Start checked the bag with the old one: refused.
    with pytest.raises(ValueError, match="^the calibration changed while the job was starting; press Start again$"):
        runner.submit_build(job_id, None, checked_calibration={"sha256": "0" * 64}, agent="none", lidar={"topic": POINTS})
    assert jobs.load(J)["status"] == "ready" and jobs.load(J)["lidar"] is None
    runner.submit_build(job_id, None, checked_calibration={"sha256": None}, agent="none", lidar={"topic": POINTS})
    job = wait(J, "review")
    assert job["lidar"] == on and job["stages"]["lidar"]["status"] == "done" and (J / "done_lidar").exists()
    runner.submit_build(job_id, None, agent="none", lidar=None)
    job = wait(J, "review")
    assert job["lidar"] is None and job["stages"]["lidar"] == {**jobs.STAGE, "status": "skipped",
                                                               "note": "turned off: no LiDAR"}


def test_lidar_settings_takes_the_tf_link_from_the_listing(tmp_path):
    """The listing reads all of /tf, so lidar_settings believes its tf_links and does not read the bag's TF. This bag's
    link starts late on /tf, and the listing here says no link and has no camera_axes_from."""
    pytest.importorskip("rosbags")
    from test_webapp_lidar import POINTS, write_scene_bag

    from harpor.webapp import rosbag

    job_id, path = jobs.create_job(tmp_path, "scene.bag")
    J = tmp_path / job_id
    path.write_bytes(write_scene_bag(tmp_path / "late.bag", n=12, dt=1.0, tf="dynamic", tf_late_s=5.5).read_bytes())
    found = rosbag.image_topics(path)
    for t in found["topics"]:
        t["tf_links"] = {POINTS: None}
        del t["camera_axes_from"]
    (J / "frames").mkdir()
    (J / "frames" / "topics.json").write_text(json.dumps(found))
    job = jobs.update(J, source={"kind": "rosbag", "topic": found["topics"][0]["topic"]})
    with pytest.raises(ValueError, match="the bag's TF does not link camera_optical_frame to lidar; upload the calibration"):
        jobs.lidar_settings(J, job, {"topic": POINTS})


def test_lidar_settings_of_a_stray_capture(tmp_path):
    """A Stray Scanner capture takes {"kind": "stray"} or no LiDAR; a bag's setting is refused for it, and
    Stray's setting for a bag or a video. Its topics stage is skipped. An export that is gone from the server
    is refused with the lidar stage's reason, so a run again from lidar keeps the results."""
    job_id, path = jobs.create_job(tmp_path, "capture.zip")
    path.write_bytes(b"PK")
    J = tmp_path / job_id
    job = jobs.load(J)
    on = {"kind": "stray"}
    assert jobs.lidar_settings(J, job, on) == jobs.lidar_settings(J, job, {"kind": "stray", "x": 1}) == on
    assert jobs.lidar_settings(J, job, None) is None
    assert jobs.lidar_settings(J, job, on, read_bag=False) == on
    path.unlink()
    with pytest.raises(ValueError, match="^the uploaded export capture.zip is not on the server; upload it again$"):
        jobs.lidar_settings(J, job, on, read_bag=False)
    assert jobs.lidar_settings(J, job, None) is None
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

    # A needs_review GLB is the agent's deliverable too; without a result (the agent stage was skipped) it is posed.glb.
    glb.unlink()
    glb.write_bytes(b"glTF")
    assert result(verdict="needs_review") == glb == steps.agent_glb(J)
    (J / "agent" / "result.json").unlink()
    assert steps.next_glb(J) == posed and steps.agent_glb(J) is None


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


def test_every_stage_has_its_record(tmp_path):
    """create_job writes a record of every stage, and no later write drops one, so a job without one is not read."""
    J = tmp_path / jobs.create_job(tmp_path, "clip.mp4")[0]
    job = jobs.load(J)
    del job["stages"]["turntable"]
    jobs.save(J, job)
    with pytest.raises(KeyError, match="turntable"):
        jobs.progress(J)
    with pytest.raises(KeyError, match="turntable"):
        jobs.update_stage(J, "turntable", status="done")


def test_check_key(monkeypatch, fake_stages):
    import urllib.error
    import urllib.request

    assert jobs.check_key("sk-abc")[0] and not jobs.check_key("abc")[0]
    assert not jobs.check_key("sk-abc\nOTHER=1")[0] and not jobs.check_key("sk abc")[0]
    monkeypatch.undo()  # the real ask_provider below
    seen = []

    def refuse(req, timeout):
        seen.append((req.full_url, req.get_header("Authorization"), timeout))
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    ok, msg = jobs.check_key(KEY)
    assert not ok and msg == "OpenAI rejected the key (HTTP 401)" and KEY not in msg
    assert seen == [("https://api.openai.com/v1/models", f"Bearer {KEY}", 10)]
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout: open(os.devnull, "rb"))
    assert jobs.check_key(KEY) == (True, "OpenAI accepted the key")


OPUS_KEY = "sk-ant-fake-" + secrets.token_hex(16)


def test_agent_choice():
    assert [steps.agent({"agent": a}) for a in steps.AGENTS] == ["astra", "opus", "fable", "none"]
    for job in ({}, *({"agent": bad} for bad in ("gpt", 1, None, ["opus"], True, False))):
        with pytest.raises(ValueError, match="agent is one of astra, opus, fable, none"):
            steps.agent(job)


def test_source_is_job_jsons_own():
    src = {"kind": "rosbag", "topic": "/cam"}
    assert steps.source({"source": src}) == src
    with pytest.raises(KeyError, match="source"):  # not a video
        steps.source({})


def test_check_anthropic_key(monkeypatch):
    """Opus keys are checked with the anthropic SDK (a stand-in client here); only the kind of error is
    reported, never its text, which could quote the key."""
    pytest.importorskip("anthropic")  # the Claude client
    import anthropic
    import httpx2

    assert jobs.check_key(KEY, agent="opus") == (False, "Anthropic keys start with sk-ant-")
    assert not jobs.check_key(OPUS_KEY + "\nX=1", agent="opus")[0]
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
        assert jobs.check_key(OPUS_KEY, agent="opus") == want
    assert seen[:2] == [{"api_key": OPUS_KEY, "max_retries": 1, "timeout": 20}, "claude-opus-5-5"]
    # A fable job's key is checked against Claude Fable 5.2, with the same prefix rule.
    raises.append(None)
    assert jobs.check_key(OPUS_KEY, agent="fable") == (True, "Anthropic accepted the key for claude-fable-5-2")
    assert seen[-1] == "claude-fable-5-2"
    raises.append(status(anthropic.PermissionDeniedError, 403))
    assert jobs.check_key(OPUS_KEY, agent="fable") == (False, "this key cannot use claude-fable-5-2")
    assert jobs.check_key(KEY, agent="fable") == (False, "Anthropic keys start with sk-ant-")
    monkeypatch.setitem(sys.modules, "anthropic", None)  # envs/web without the SDK
    assert jobs.check_key(OPUS_KEY, agent="opus") == (
        False, "the server cannot check Anthropic keys: envs/web needs anthropic==1.15.0")


def test_a_key_of_the_other_provider_is_never_sent(monkeypatch):
    """A key pasted under the wrong agent is refused by its prefix before any request, so it never goes
    to the other company's API."""
    pytest.importorskip("anthropic")  # the Claude client
    import urllib.request

    import anthropic

    def no_request(*a, **kw):
        raise AssertionError("a request was made")

    monkeypatch.setattr(urllib.request, "urlopen", no_request)
    monkeypatch.setattr(anthropic, "Anthropic", no_request)
    for key, agent, why in [(KEY, "opus", "Anthropic keys start with sk-ant-"),
                            ("sk-proj-" + KEY[8:], "opus", "Anthropic keys start with sk-ant-"),
                            (OPUS_KEY, "astra", "this is an Anthropic key (sk-ant-); this agent needs an OpenAI key"),
                            ("ant-" + KEY, "astra", "OpenAI keys start with sk-")]:
        ok, msg = jobs.check_key(key, agent=agent)
        assert (ok, msg) == (False, why) and key not in msg


def test_runner_opus_key_and_a_change_of_agent(make_runner, tmp_path, monkeypatch):
    runner, cfg = make_runner()
    probe = "import os, stat, sys; f = sys.argv[1]; print('key file', oct(stat.S_IMODE(os.stat(f).st_mode)), " \
            "open(f).read().split('=')[0])"

    def command(stage, job, cfg, key_file=None):
        J = cfg.runs / job["id"]
        script = f"touch {J}/done_{stage}"
        if stage in steps.KEYED:
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
    for s in ("agent", "estimate"):  # the job's model refines the mesh and gives the friction
        assert "key file 0o600 ANTHROPIC_API_KEY" in (J / "logs" / f"{s}.log").read_text()

    # Selected with GPT-6.2-Astra, then Build with Opus and no key: the OpenAI key is dropped, and the agent
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

    # Done; back to GPT-6.2-Astra from vomp: the agent and every stage after it run again, with an OpenAI key file.
    runner.resume(job_id, "vomp", api_key=KEY, agent="astra")
    job = wait(J, "done")
    assert job["agent"] == "astra" and job["stages"]["agent"]["started"] > agent_started
    assert job["stages"]["place"]["started"] == place_started and job["stages"]["package"]["status"] == "done"
    for s in ("agent", "estimate"):
        log = (J / "logs" / f"{s}.log").read_text()
        assert log.count(f"== {s} ") == 2
        assert log.index("ANTHROPIC_API_KEY") < log.index("key file 0o600 OPENAI_API_KEY")
    # And no agent: no key, and the agent stage is skipped, and so are the stages that need the friction.
    runner.resume(job_id, "align", agent="none")
    job = wait(J, "done")
    assert job["agent"] == "none" and job["stages"]["agent"]["status"] == "skipped"
    for s in ("estimate", "mjcf", "usd", "urdf"):
        assert job["stages"][s] == {**jobs.STAGE, "status": "skipped", "note": steps.NO_KEY_SKIP}, s
    assert (J / "logs" / "estimate.log").read_text().count("== estimate ") == 2
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


def test_a_job_without_vomps_output_runs_vomp_only_when_align_runs(make_runner, monkeypatch):
    """A job folder whose VoMP output is missing (steps.done("vomp") is false), as a failed VoMP run leaves it, with
    align, the estimate and the exports done. align is the only stage that reads VoMP's
    output. A retry from mjcf does not run align, so it runs no VoMP and needs no key, and the job keeps its physics.
    A done job's Retry from turntable runs only turntable and package, so results.zip is made again, with no key. A
    run that runs align (here align failed, so nothing after it ran) starts from vomp (Runner.resume), and the
    estimate runs after align, so that run needs the key (jobs.key_needed)."""
    runner, cfg = make_runner()
    stub_stages(monkeypatch)
    job_id, J = ready_job(cfg, auto_continue=True)
    runner.submit_build(job_id, KEY, agent="astra", turntable=True)
    built = wait(J, "done")
    started = lambda job: {s: job["stages"][s]["started"] for s in steps.NAMES}  # noqa: E731
    before = started(built)
    (J / "done_vomp").unlink()

    # Failed at mjcf, so the stages after it never ran: they run; VoMP, align and the estimate do not, and no key is
    # needed.
    later = steps.NAMES[steps.NAMES.index("mjcf"):]
    for s in later:
        (J / f"done_{s}").unlink(missing_ok=True)
    jobs.update(J, status="failed", error="mjcf: exit 1",
                stages={**built["stages"], **{s: dict(jobs.STAGE) for s in later},
                        "mjcf": {**jobs.STAGE, "status": "failed", "returncode": 1}})
    assert not jobs.key_needed(J, jobs.load(J), "mjcf")
    assert runner.resume(job_id, "mjcf") == "mjcf"
    job = wait(J, "done", "failed", "review")
    after = started(job)
    assert job["status"] == "done", job
    assert all(after[s] > before[s] for s in later), (before, after)
    assert all(after[s] == before[s] for s in ("vomp", "place", "agent", "align", "estimate"))
    assert not (J / "done_vomp").exists()

    # Done, and its 360 video failed: the Retry from turntable reruns turntable and package only, with no key.
    before = after
    jobs.update(J, warnings=["360 video: exit 3"],
                stages={**job["stages"], "turntable": {**jobs.STAGE, "status": "failed", "returncode": 3}})
    assert not jobs.key_needed(J, jobs.load(J), "turntable")
    assert runner.resume(job_id, "turntable") == "turntable"
    job = wait(J, "done", "failed", "review")
    after = started(job)
    assert job["status"] == "done" and job["warnings"] == [] and job["stages"]["turntable"]["status"] == "done"
    assert after["turntable"] > before["turntable"] and after["package"] > before["package"]
    assert all(after[s] == before[s] for s in steps.NAMES[:steps.NAMES.index("turntable")])

    # align failed, so the stages after it never ran: the retry starts from vomp, since align reads VoMP's output, and
    # the estimate runs after align, so it needs the key. place and the agent stay done.
    before = after
    later = steps.NAMES[steps.NAMES.index("align"):]
    for s in later:
        (J / f"done_{s}").unlink(missing_ok=True)
    stages = {**job["stages"], **{s: dict(jobs.STAGE) for s in later}, "align": {**jobs.STAGE, "status": "failed"}}
    jobs.update(J, status="failed", error="align: exit 1", stages=stages)
    job = jobs.load(J)
    assert jobs.key_needed(J, job, "align") and not jobs.key_needed(J, {**job, "agent": "none"}, "align")
    assert runner.resume(job_id, "align", api_key=KEY) == "vomp"
    job = wait(J, "done", "failed", "review")
    after = started(job)
    assert job["status"] == "done", job
    assert all(after[s] > before[s] for s in ("vomp", *later)) and (J / "done_vomp").is_file()
    assert after["place"] == before["place"] and after["agent"] == before["agent"]


def test_a_run_after_align_whose_outputs_are_gone_starts_from_align(make_runner, monkeypatch):
    """align's record says done but its outputs are gone (here summary.json was deleted). The page's Retry starts from estimate, the first stage whose record is not done.
    Runner.resume starts the run from align instead, since every later stage reads align's outputs, and from vomp
    when VoMP's output is missing too. The estimate runs after align, so the retry needs the key."""
    runner, cfg = make_runner()
    stub_stages(monkeypatch)
    job_id, J = ready_job(cfg, auto_continue=True)
    runner.submit_build(job_id, KEY, agent="astra")
    wait(J, "done")
    later = steps.NAMES[steps.NAMES.index("estimate"):]
    for gone, first in (([], "align"), (["vomp"], "vomp")):  # VoMP's output there, then missing
        before = {s: st["started"] for s, st in jobs.load(J)["stages"].items()}
        for s in ["align", *gone, *later]:
            (J / f"done_{s}").unlink(missing_ok=True)
        stages = {**jobs.load(J)["stages"], **{s: dict(jobs.STAGE) for s in later},
                  "estimate": {**jobs.STAGE, "status": "failed", "returncode": 1}}
        jobs.update(J, status="failed", error="estimate: exit 1", stages=stages)
        job = jobs.load(J)
        assert jobs.key_needed(J, job, "estimate") and not jobs.key_needed(J, {**job, "agent": "none"}, "estimate")
        assert runner.resume(job_id, "estimate", api_key=KEY) == first
        job = wait(J, "done", "failed", "review")
        assert job["status"] == "done", job
        after = {s: st["started"] for s, st in job["stages"].items()}
        assert after["align"] > before["align"] and (after["vomp"] > before["vomp"]) == bool(gone), first
        ran = ("vomp", "align", "estimate", "mjcf", "usd", "urdf", "package")  # no 360 video: turntable is skipped
        assert after["agent"] == before["agent"] and all((J / f"done_{s}").exists() for s in ran)


def test_runner_key_file_lives_only_during_the_keyed_stages(make_runner, tmp_path, monkeypatch):
    runner, cfg = make_runner()
    (runner.key_dir / "stale.env").write_text("OPENAI_API_KEY=old\n")
    runner, cfg = make_runner()  # a restart deletes key files a dead server left
    assert not list(runner.key_dir.iterdir()) and stat.S_IMODE(runner.key_dir.stat().st_mode) == 0o700
    probe = "import os, stat, sys; f = sys.argv[1]; print('key file', oct(stat.S_IMODE(os.stat(f).st_mode)), " \
            "open(f).read().startswith('OPENAI_API_KEY='))"

    def command(stage, job, cfg, key_file=None):
        J = cfg.runs / job["id"]
        script = f"echo PROGRESS 1; touch {J}/done_{stage}"
        if stage in ("agent", "estimate"):
            script += f"; {sys.executable} -c \"{probe}\" {key_file}"
        elif stage == "mjcf":  # the key file of estimate is gone once it ended
            script += f"; ls {runner.key_dir}"
        return ["bash", "-c", script], str(J), {}

    monkeypatch.setattr(steps, "command", command)
    monkeypatch.setattr(steps, "done", lambda stage, J: (Path(J) / f"done_{stage}").exists() or stage == "frames")
    job_id, J = ready_job(cfg, auto_continue=True)
    runner.submit_build(job_id, KEY)
    job = wait(J, "done")
    skipped = ("lidar", "turntable")  # no LiDAR and no 360 video
    assert all(job["stages"][s]["status"] == ("skipped" if s in skipped else "done") for s in ["frames", *jobs.BUILD])
    assert job["key_ok"] is False
    assert job["slug"] == "box" and job_id not in runner._keys
    assert "key file 0o600 True" in (J / "logs" / "agent.log").read_text()
    assert "key file 0o600 True" in (J / "logs" / "estimate.log").read_text()
    assert ".env" not in (J / "logs" / "mjcf.log").read_text()
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


def test_runner_gives_the_key_to_estimate_after_a_skipped_agent(make_runner, tmp_path, monkeypatch):
    """A poor Stage A placement skips the agent stage, which leaves the key unused: the runner holds it for the whole
    build, so estimate gets it. A job that resumes from estimate after a restart, so that the runner holds no key,
    waits in review at estimate with NO_KEY, and runs once the key is given."""
    runner, cfg = make_runner()
    probe = "import os, stat, sys; f = sys.argv[1]; print('key file', oct(stat.S_IMODE(os.stat(f).st_mode)), " \
            "open(f).read().split('=')[0])"

    def command(stage, job, cfg, key_file=None):
        J = cfg.runs / job["id"]
        script = f"touch {J}/done_{stage}"
        if stage == "place":  # mean mask IoU 0.31, below steps.MIN_IOU
            script += " && mkdir -p run && cp placement_in.json run/placement.json"
        if stage == "estimate":
            script = f"{sys.executable} -c \"{probe}\" {key_file}; test ! -e fail_estimate && {script}"
        return ["bash", "-c", script], str(J), {}

    monkeypatch.setattr(steps, "command", command)
    monkeypatch.setattr(steps, "done", lambda stage, J: (Path(J) / f"done_{stage}").exists() or stage == "frames")
    job_id, J = ready_job(cfg, auto_continue=True)
    (J / "placement_in.json").write_text(json.dumps({"check": {"mean_mask_iou": 0.31}}))
    (J / "fail_estimate").touch()
    runner.submit_build(job_id, KEY)
    job = wait(J, "failed")
    assert job["stages"]["agent"]["status"] == "skipped" and "0.31 is below 0.5" in job["stages"]["agent"]["note"]
    assert not (J / "done_agent").exists() and job["stages"]["align"]["status"] == "done"
    assert job["stages"]["estimate"]["status"] == "failed" and job["error"].startswith("estimate: exit 1")
    assert "key file 0o600 OPENAI_API_KEY" in (J / "logs" / "estimate.log").read_text()
    assert runner.has_key(job_id, "astra")  # a failed build keeps the key for its retry

    (J / "fail_estimate").unlink()
    runner, _ = make_runner()  # a restart: no key
    assert runner.resume(job_id, "estimate") == "estimate"
    job = wait(J, "review")
    assert job["warnings"] == [jobs.NO_KEY["astra"]] and job["key_ok"] is False
    assert job["stages"]["estimate"]["note"] == "waiting for the OpenAI API key" and not (J / "done_estimate").exists()
    runner.resume(job_id, "estimate", api_key=KEY)
    job = wait(J, "done")
    assert [job["stages"][s]["status"] for s in ("agent", "align", "estimate", "mjcf", "usd", "urdf", "package")] == [
        "skipped", "done", "done", "done", "done", "done", "done"]
    assert (J / "logs" / "estimate.log").read_text().count("key file 0o600 OPENAI_API_KEY") == 2
    assert not list(runner.key_dir.iterdir())
    assert not any(KEY.encode() in p.read_bytes() for p in all_files(tmp_path))


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
    """A LiDAR job whose lidar stage failed, whose calibration was uploaded or deleted since the stage ran, or
    whose scale an older estimator found, starts from lidar when a retry or Build names a later stage, and stops
    at the review again."""
    runner, cfg = make_runner()
    real_done, fail = steps.done, {"lidar"}

    def command(stage, job, cfg, key_file=None):
        J = cfg.runs / job["id"]
        if stage == "lidar":  # the record of the real stage, with the calibration it ran with
            rec = {"version": lidar_scale.VERSIONS["rosbag"], "ok": stage not in fail, "s": 1.25, "reason": "no points",
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

    # A calibration uploaded in review: Build finds the scale again and stops in review. When it came from
    # another tab after the server checked the LiDAR settings (checked_calibration), the run is refused.
    jobs.update(J, calibration={"name": "c.yaml", "sha256": "1" * 64})
    assert not steps.done("lidar", J)
    with pytest.raises(ValueError, match="^the calibration changed while the run was starting; try again$"):
        runner.resume(job_id, "sam3d", checked_calibration={"sha256": None})
    assert jobs.load(J)["status"] == "review"
    (J / "ran").unlink()
    assert runner.resume(job_id, "sam3d", checked_calibration={"sha256": "1" * 64}) == "lidar"
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

    # A scale that an older estimator found, with the job's calibration: Build finds it again first.
    rec = json.loads((J / "pool" / "lidar.json").read_text())
    (J / "pool" / "lidar.json").write_text(json.dumps({**rec, "version": lidar_scale.VERSIONS["rosbag"] - 1}))
    assert not steps.done("lidar", J)
    (J / "ran").unlink()
    assert runner.resume(job_id, "sam3d") == "lidar"
    assert wait(J, "review") and ran(J) == ["lidar", "select"] and steps.done("lidar", J)


def test_a_stale_scale_says_why(tmp_path, fake_stages):
    """GET /api/jobs/{id} gives why a done lidar stage's scale is stale as lidar_stale, and the selection's LiDAR
    record gives it as calibration_changed and estimator_changed. "calibration": the calibration was uploaded or
    deleted since the scale was found. "estimator": an older lidar_scale.VERSIONS found it with the job's
    calibration, so the page must not say that the calibration changed."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from harpor.webapp import server

    token = "tok-" + secrets.token_hex(16)
    cfg = steps.Config(G=tmp_path / "G", runs=tmp_path / "runs")
    job_id, _ = jobs.create_job(cfg.runs, "scan.bag")
    J = cfg.runs / job_id
    cal = {"name": "c.yaml", "bytes": 300, "sha256": "1" * 64, "T_cam_lidar": np.eye(4).tolist(), "lidar_topic": "/points"}
    jobs.update(J, status="review", prompt="box", slug="box", lidar={"kind": "rosbag", "topic": "/points"}, calibration=cal)
    jobs.update_stage(J, "lidar", status="done")
    (J / "pool").mkdir()
    (J / "selection.json").write_text(json.dumps({"ok": True}))
    rec = {"version": lidar_scale.VERSIONS["rosbag"], "ok": True, "s": 1.25, "reason": None,
           "extrinsic": {"calibration_sha256": "1" * 64}}

    with TestClient(server.make_app(cfg, token, key_dir=tmp_path / "keys")) as c:
        assert c.get(f"/?token={token}", follow_redirects=False).status_code == 303

        def seen(**over):
            (J / "pool" / "lidar.json").write_text(json.dumps({**rec, **over}))
            job, lid = c.get(f"/api/jobs/{job_id}").json(), c.get(f"/api/jobs/{job_id}/selection").json()["lidar"]
            return job["lidar_current"], job["lidar_stale"], lid["calibration_changed"], lid["estimator_changed"]

        assert seen() == (True, None, False, False)
        assert seen(version=lidar_scale.VERSIONS["rosbag"] - 1) == (False, "estimator", False, True)
        assert seen(extrinsic={"calibration_sha256": None}) == (False, "calibration", True, False)
        # Both changed: the calibration is named, since the user changed it.
        assert seen(version=None, extrinsic={"calibration_sha256": "2" * 64}) == (False, "calibration", True, False)
        assert seen(ok=False, reason="no points") == (False, None, False, False)  # not stale: failed
        (J / "pool" / "lidar.json").write_text("{")  # a file that does not parse: not current, and no error (was 500)
        r = c.get(f"/api/jobs/{job_id}")
        assert r.status_code == 200 and (r.json()["lidar_current"], r.json()["lidar_stale"]) == (False, None)


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
        assert stage not in ("agent", "estimate", "mjcf", "usd", "urdf")  # turned off, so they never run
        script = {"topics": "mkdir -p frames && cp topics_in.json frames/topics.json",
                  "frames": f"mkdir -p frames && echo {steps.source(job)['topic']} >> frames_ran && echo {{}} > frames/meta.json"}
        return ["bash", "-c", script.get(stage, f"mkdir -p scene && touch scene/done_{stage}")], str(cfg.runs / job["id"]), {}

    monkeypatch.setattr(steps, "command", command)
    monkeypatch.setattr(steps, "done", lambda stage, J: real_done(stage, J) if stage in ("topics", "frames")
                        else (Path(J) / "scene" / f"done_{stage}").exists())

    def listing(J, topics, skipped=()):
        """topics.json as the topics stage writes it."""
        skipped = [{"topic": t, "type": "Image", "why": "16UC1"} for t in skipped]
        (J / "topics_in.json").write_text(json.dumps({"topics": [{"topic": t} for t in topics], "skipped": skipped,
                                                      "lidar": [], "lidar_skipped": []}))

    def upload(name, topics, skipped=()):
        job_id, path = runner.new_upload(name)
        J = cfg.runs / job_id
        listing(J, topics, skipped)
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

    # A build without the agent: no key, and the agent stage is skipped, and so are estimate, mjcf, usd and urdf,
    # since the friction comes from the job's model. VoMP and align still run.
    jobs.update(J, prompt="box", auto_continue=True)
    runner.submit_build(job_id, None, agent="none")
    job = wait(J, "done")
    assert job["stages"]["agent"] == {**jobs.STAGE, "status": "skipped", "note": "turned off: physics uses the Stage A mesh"}
    for s in ("estimate", "mjcf", "usd", "urdf"):
        assert job["stages"][s] == {**jobs.STAGE, "status": "skipped", "note": steps.NO_KEY_SKIP}, s
    assert job["stages"]["vomp"]["status"] == job["stages"]["align"]["status"] == job["stages"]["package"]["status"] == "done"
    assert job["agent"] == "none" and job["key_ok"] is False and job_id not in runner._keys
    assert not list(runner.key_dir.iterdir())
    with pytest.raises(ValueError, match="is done"):
        runner.submit_analyze(job_id, topic="/a/image_rect")
    with pytest.raises(ValueError, match="API key"):  # None would queue it without a key (the key comes later)
        runner.submit_build(job_id, "", agent="astra")

    # Another topic after a failed build: frames runs on it, and the build's outputs, records and picks are gone;
    # so is the calibration, whose T and K were the other camera's.
    jobs.update(J, status="failed", picks=[3, 9], view0=3)
    runner.submit_analyze(job_id, topic="/a/image_rect")
    job = wait(J, "ready")
    assert (J / "frames_ran").read_text() == "/b/image_raw\n/a/image_rect\n" and not (J / "scene").exists()
    assert all(job["stages"][s] == jobs.STAGE for s in jobs.BUILD) and job["picks"] == [] and job["view0"] is None
    assert job["calibration"] is None and not (J / "lidar" / "calibration").exists()

    # A video skips topics.
    job_id, J = upload("clip.MOV", [])
    wait(J, "ready")
    jobs.update(J, status="failed")
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


def test_key_needed_counts_the_keyed_stages_that_the_run_runs(tmp_path):
    """jobs.key_needed: a keyed stage needs the key when a run from first runs it. Runner._build runs it unless its
    record is done and its outputs are there, and a skipped one runs again unless steps.skip still gives a reason. So
    a skipped estimate record needs the key, from a start at or before estimate. A stage before the start does not
    run. A run that runs align runs the estimate after it, since
    align writes the summary again without it. Such a run starts from vomp when it starts later and VoMP's output is
    missing, so a failed agent stage runs too. A run that does not run align, such as a retry of the 360 video,
    needs no VoMP output and no key. A job without an agent never needs the key."""
    cfg = steps.Config(G=tmp_path / "G", runs=tmp_path / "runs")
    _, J = ready_job(cfg)  # agent astra
    job = jobs.load(J)
    for rel in ("agent/result.json", "run/vomp/summary.json", "physics/aligned.json", "physics/estimate_views.jpg"):
        (J / rel).parent.mkdir(parents=True, exist_ok=True)
        (J / rel).write_text("{}")
    summary = J / "physics" / "summary.json"

    def needs(first, agent="done", estimate="done", model="astra", align="done"):
        """key_needed with these statuses of the agent, align and estimate stages, and model as job.json's agent."""
        st = {**job["stages"], "agent": {**jobs.STAGE, "status": agent}, "align": {**jobs.STAGE, "status": align},
              "estimate": {**jobs.STAGE, "status": estimate}}
        return jobs.key_needed(J, {**job, "agent": model, "stages": st}, first)

    summary.write_text(json.dumps({"rigid_body": {"mass": 1.0}, "estimate": {"friction": 0.4}}))
    assert not any(needs(first) for first in ("extract", "sam3d", "align", "estimate", "turntable"))
    assert needs("align", estimate="failed") and needs("sam3d", estimate="pending") and needs("agent", agent="failed")
    assert not needs("estimate", agent="failed")  # the run starts after the agent stage
    # align runs (its record failed, or its outputs are gone), so the estimate runs after it, done or not.
    assert needs("align", align="failed") and needs("sam3d", align="pending") and not needs("mjcf", align="failed")
    # align's outputs are gone: a run from after align starts from align (Runner.resume), and the estimate follows.
    (J / "physics" / "aligned.json").unlink()
    assert needs("align") and needs("sam3d") and needs("estimate") and needs("turntable")
    assert not needs("estimate", model="none")
    (J / "physics" / "aligned.json").write_text("{}")
    # A job that failed at align: its summary is VoMP's alone, with no estimate, and its estimate record is skipped.
    summary.write_text(json.dumps({"rigid_body": {"mass": 1.0}}))
    assert needs("align", estimate="skipped", align="failed") and needs("sam3d", estimate="skipped")
    assert not needs("mjcf", estimate="skipped") and not needs("turntable", estimate="skipped")
    assert needs("estimate", estimate="done")  # a record that says done without its output runs again
    assert not needs("align", estimate="skipped", model="none")
    # A skipped agent stage that a poor placement skips again needs no key; without that reason it runs.
    summary.write_text(json.dumps({"rigid_body": {"mass": 1.0}, "estimate": {"friction": 0.4}}))
    (J / "run" / "placement.json").write_text(json.dumps({"check": {"mean_mask_iou": 0.31}}))
    assert not needs("sam3d", agent="skipped")
    (J / "run" / "placement.json").unlink()
    assert needs("sam3d", agent="skipped")
    # Without VoMP's output, a run that runs align starts from vomp, so a failed agent stage runs too. A run that does
    # not run align (align done, or a start after it, such as a retry of the 360 video) needs neither.
    (J / "run" / "vomp" / "summary.json").unlink()
    assert needs("align", agent="failed", align="failed") and needs("align", align="failed")
    assert not needs("align", agent="failed") and not needs("estimate", agent="failed") and not needs("turntable")
    assert not needs("align", agent="failed", align="failed", model="none")


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
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "-m", "harpor.webapp"])
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
    server_ = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "-m", "harpor.webapp"])
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


def test_runner_builds_without_a_key_and_asks_for_it(make_runner, monkeypatch, tmp_path):
    """submit_build with api_key None queues the build without a key; the job waits in review at the first
    stage that needs it, with NO_KEY. A key given in resume lets it go on."""
    runner, cfg = make_runner()
    stub_stages(monkeypatch)
    job_id, J = ready_job(cfg, auto_continue=True)
    runner.submit_build(job_id, None, agent="opus")
    assert jobs.load(J)["key_ok"] is False and not runner.has_key(job_id)
    job = wait(J, "review")
    assert job["warnings"] == [jobs.NO_KEY["opus"]] and job["key_ok"] is False
    assert job["stages"]["agent"]["note"] == "waiting for the Anthropic API key"
    assert job["stages"]["place"]["status"] == "done" and not (J / "done_agent").exists()
    runner.resume(job_id, "sam3d", api_key=OPUS_KEY)
    job = wait(J, "done")
    assert job["stages"]["agent"]["status"] == "done"

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
    from harpor.agent.fake_agent import write_glb

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
    # The agent wrote the key into its GLB: neither it nor what physics made from it (the videos too) is packaged.
    (J / "agent" / "result.json").write_text(json.dumps({**r, "key_found_in": ["work/exports/001.glb", "session/x.log"]}))
    jobs.main(["package", "--job", str(J)])
    assert set(zipfile.ZipFile(J / "results.zip").namelist()) == {"posed.glb", "summary.json", "placement.json",
                                                                  "selection.json", "lidar.json", "agent_result.json"}
    assert "hold the API key" in capsys.readouterr().out
    assert not list(J.glob(".results*"))
    # The agent's evidence goes in, unless its source, the session's control/evidence.json, held the key.
    (J / "agent" / "evidence.json").write_text(json.dumps({"version": 1}))
    jobs.package(J)
    assert "evidence.json" in zipfile.ZipFile(J / "results.zip").namelist()
    (J / "agent" / "result.json").write_text(json.dumps({**r, "key_found_in": ["control/evidence.json"]}))
    jobs.package(J)
    assert set(zipfile.ZipFile(J / "results.zip").namelist()) == names
    assert "left out (they hold the API key): ['evidence.json']" in capsys.readouterr().out


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
    (J / "job.json").write_text(json.dumps({"id": "x", "prompt": "toy box", "source": {"kind": "video", "topic": None}}))
    (J / "key.env").write_text(f"OPENAI_API_KEY={KEY}\n")
    return J


FAKE_BLENDER = [sys.executable, "-m", "harpor.agent.fake_agent", "--blender"]


def test_agent_step_runs_a_fresh_refinement_session(tmp_path, monkeypatch):
    import tempfile

    from harpor.agent import controller as C

    J = run_folder(tmp_path)
    (J / "agent" / "s" / "work").mkdir(parents=True)
    (J / "agent" / "s" / "work" / "stale.txt").write_text("from a failed try")
    calls = []

    def container(root, sif, key_file, budget_s):  # the controller's own session, with an agent that exits at once
        calls.append((Path(root), sif, key_file, budget_s))
        r = C.run_session(root, [sys.executable, "-c", "pass"], blender=FAKE_BLENDER,
                          sock_dir=tempfile.mkdtemp(prefix="harpor-"))
        r["key_found_in"] = C.find_key(root, key_file)
        write_json(Path(root) / "result.json", r)
        return r

    monkeypatch.setattr(C, "has_rounds", lambda root: False)  # one session, without the rounds
    monkeypatch.setattr(C, "run_container_session", container)
    r = agent_step.run(J, "/sif/agent.sif", J / "key.env")
    assert calls == [(J / "agent" / "s", "/sif/agent.sif", J / "key.env", agent_step.BUDGET_S)] and agent_step.BUDGET_S == 900.0
    assert not (J / "agent" / "s" / "work" / "stale.txt").exists()
    m = json.loads((J / "agent" / "s" / "control" / "manifest.json").read_text())
    assert "mode" not in m and len(m["views"]) == 2 and m["units"] == "meters"
    assert m["source"] == agent_step.source("toy box", 2, "video")
    assert "2 frames from a video of the toy box" in m["source"] and "Depth Anything 3 cameras" in m["source"]
    assert (J / "agent" / "s" / "original.glb").read_bytes() == (J / "run" / "posed.glb").read_bytes()
    assert r["verdict"] == "fallback_original" and r["key_found_in"] == []
    assert json.loads((J / "agent" / "result.json").read_text()) == json.loads((J / "agent" / "s" / "result.json").read_text())
    assert steps.next_glb(J) == J / "run" / "posed.glb"


def test_agent_step_prints_the_agents_text_on_clean_lines(tmp_path, monkeypatch, capsys):
    """A result.json from before the helpers cleaned the agent's text still prints its reason, each limitation and
    each rejected GLB's reason as one line without control characters, plain text and accents kept."""
    import tempfile

    from harpor.agent import controller as C
    from test_agent_controller import CLEAN, DIRTY

    J = run_folder(tmp_path)

    def container(root, sif, key_file, budget_s):
        r = C.run_session(root, [sys.executable, "-c", "pass"], blender=FAKE_BLENDER,
                          sock_dir=tempfile.mkdtemp(prefix="harpor-"))
        r.update(reason=DIRTY, limitations=[DIRTY, "plain text, é and ü"], key_found_in=[],
                 rejected=[{"from": "finish", "glb": "work/exports/001.glb", "why": f"unparsable: {DIRTY}"}])
        write_json(Path(root) / "result.json", r)
        return r

    monkeypatch.setattr(C, "has_rounds", lambda root: False)  # one session, without the rounds
    monkeypatch.setattr(C, "run_container_session", container)
    agent_step.run(J, "/sif/agent.sif", J / "key.env")
    out = capsys.readouterr().out
    assert (f"  reason: {CLEAN}\n  limitation: {CLEAN}\n  limitation: plain text, é and ü\n"
            f"  rejected finish work/exports/001.glb: unparsable: {CLEAN}\n") in out
    assert [c for c in out if not c.isprintable() and c != "\n"] == []


def test_agent_step_redacts_a_key_that_cleaning_joins(tmp_path, monkeypatch, capsys):
    """A key split by a character that tools.printable drops (ESC, a zero-width space, BEL) is not the key while the
    result is scrubbed, and whole once its line is cleaned: a link name of the agent's GLB in a rejected GLB's reason,
    or an older result's reason and limitations. Each line is scrubbed again after the cleaning."""
    import tempfile

    from harpor.agent import controller as C

    J = run_folder(tmp_path)

    def container(root, sif, key_file, budget_s):
        r = C.run_session(root, [sys.executable, "-c", "pass"], blender=FAKE_BLENDER,
                          sock_dir=tempfile.mkdtemp(prefix="harpor-"))
        r.update(reason=f"done {KEY[:9]}\x1b{KEY[9:]}", limitations=[f"{KEY[:12]}​{KEY[12:]}"], key_found_in=[],
                 rejected=[{"from": "export", "glb": "work/exports/001.glb",
                            "why": f"unparsable: the GLB names (c, {KEY[:20]}\x07{KEY[20:]})"}])
        write_json(Path(root) / "result.json", r)
        return r

    monkeypatch.setattr(C, "has_rounds", lambda root: False)  # one session, without the rounds
    monkeypatch.setattr(C, "run_container_session", container)
    agent_step.run(J, "/sif/agent.sif", J / "key.env")
    out = capsys.readouterr().out
    assert KEY not in out and KEY[9:] not in out
    assert ("  reason: done [redacted API key]\n  limitation: [redacted API key]\n  rejected export work/exports/001.glb: "
            "unparsable: the GLB names (c, [redacted API key])\n") in out


def test_agent_step_names_the_jobs_real_input(tmp_path, monkeypatch):
    """The manifest's source line says what the job started from, as job.json has it."""
    from harpor.agent import controller as C

    J = run_folder(tmp_path)
    job = {**json.loads((J / "job.json").read_text()), "source": {"kind": "rosbag", "topic": "/cam"}}
    (J / "job.json").write_text(json.dumps(job))

    def prepare(root, prompt, views, *, source, **kw):
        raise SystemExit(source)

    monkeypatch.setattr(C, "prepare_session", prepare)
    with pytest.raises(SystemExit, match="2 frames from a rosbag of the toy box"):
        agent_step.run(J, "/sif/agent.sif", J / "key.env")
    assert "2 frames from a Stray Scanner capture of the toy box" in agent_step.source("toy box", 2, "stray")


def test_agent_step_runs_a_claude_session_for_opus(tmp_path, monkeypatch, capsys):
    import tempfile

    from harpor.agent import controller as C

    J = run_folder(tmp_path)
    (J / "key.env").write_text(f"ANTHROPIC_API_KEY={OPUS_KEY}\n")
    end = {"type": "end", "reason": "end_turn", "turns": 3, "cost_usd": 0.42,
           "totals": {"input_tokens": 9, "output_tokens": 8, "cache_read_input_tokens": 7, "cache_creation_input_tokens": 6}}

    def container(root, sif, key_file, budget_s):  # the controller's own session; the agent writes its end line
        script = f"import json, sys; open(sys.argv[1], 'a').write(json.dumps({end!r}) + chr(10))"
        r = C.run_session(root, [sys.executable, "-c", script, str(Path(root) / "session" / "claude.jsonl")],
                          blender=FAKE_BLENDER, sock_dir=tempfile.mkdtemp(prefix="harpor-"))
        r["key_found_in"] = C.find_key(root, key_file)
        write_json(Path(root) / "result.json", r)
        return r

    monkeypatch.setattr(C, "has_rounds", lambda root: False)  # one session, without the rounds
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
    import fake_steps

    J = run_folder(tmp_path)
    key = J / "fake.env"
    for which, line, model, cli in (("astra", f"OPENAI_API_KEY={KEY}", "gpt-6.2-astra", "cancelled"),
                                    ("opus", f"ANTHROPIC_API_KEY={OPUS_KEY}", "claude-opus-5-5", "interrupted")):
        key.unlink(missing_ok=True)
        key.write_text(line + "\n")
        key.chmod(0o600)
        fake_steps.agent(J, key, which)
        r = json.loads((J / "agent" / "result.json").read_text())
        assert (r["model"], r["cli_exit"], r["verdict"]) == (model, cli, "complete") and ("tokens" in r) == (which == "opus")
    assert r["cost_usd"] > 0 and r["turns"] == 12
    # A refinement reports its surface (harpor.agent.surface) and has no rounds, so no evidence.
    assert r["surface"]["shares"]["present"] == 1.0 and "rounds" not in r and "simulations" not in r
    assert not (J / "agent" / "evidence.json").exists()
    with pytest.raises(SystemExit, match="no OPENAI_API_KEY line"):
        fake_steps.agent(J, key, "astra")


def test_fake_agent_stage_alone_skips_the_round_code(tmp_path, monkeypatch, capsys):
    """The fake agent stage writes its refinement result without the round code: neither the controller's session and
    rounds nor the host's check run. It deletes the evidence of an earlier try, writes none
    of its own, and the GLB stays where the fake agent left it."""
    from harpor.agent import controller as C
    from harpor.agent import validate
    import fake_steps

    J = run_folder(tmp_path)

    def not_here(*a, **kw):
        raise AssertionError("the fake agent stage runs no round code")

    for name in ("run_session", "run_rounds", "run_round", "check_round", "store_candidate", "select_candidate"):
        monkeypatch.setattr(C, name, not_here)
    monkeypatch.setattr(validate, "check", not_here)
    key = J / "fake.env"
    key.write_text(f"OPENAI_API_KEY={KEY}\n")
    key.chmod(0o600)
    (J / "agent").mkdir()
    (J / "agent" / "evidence.json").write_text("of an earlier try")
    fake_steps.agent(J, key, "astra")
    r = json.loads((J / "agent" / "result.json").read_text())
    glb = "work/exports/001.glb"
    assert (r["verdict"], r["glb"], r["source"]) == ("complete", glb, "finish")
    assert r["counts"] == {"exports": 1, "reviews": 1, "checkpoints": 0} and "rounds" not in r and "candidate" not in r
    assert r["surface"]["shares"]["present"] == 1.0 and steps.agent_glb(J) == J / "agent" / "s" / glb
    assert r == json.loads((J / "agent" / "s" / "result.json").read_text())
    assert not (J / "agent" / "evidence.json").exists() and not (J / "agent" / "s" / "control" / "evidence.json").exists()
    assert "agent (fake gpt-6.2-astra): finish complete, GLB work/exports/001.glb" in capsys.readouterr().out


def test_agent_step_publishes_the_evidence_of_its_rounds(tmp_path, monkeypatch, capsys, fake_stages):
    """The real agent_step.run on a refinement: the session runs in rounds (control/rounds.json), and the fake agent
    only exports in round 1 and finishes in round 2 with the API key in its reason. agent/evidence.json is control/evidence.json without
    the key and as strict JSON: a number that is not finite (here the diagonal) is null. The stage log prints the
    rounds, and the server's view of the job takes the result's round keys. Since control/evidence.json held the key,
    the file route and results.zip leave the evidence out; a result without that finding serves it."""
    for m in ("fastapi", "av"):
        pytest.importorskip(m)
    import math
    import tempfile

    from fastapi.testclient import TestClient

    from harpor.agent import controller as C
    from harpor.agent import evidence
    from harpor.webapp import server

    cfg = steps.Config(G=tmp_path / "G", runs=tmp_path / "runs")
    job_id, _ = jobs.create_job(cfg.runs, "box.mp4")
    J = cfg.runs / job_id
    job = jobs.load(J)
    run_folder(J)
    jobs.save(J, {**job, "prompt": "toy box", "status": "done"})

    def tool(*args):
        return {"do": "tool", "args": list(args)}

    def edit(size):  # a box of edge size, which the original's 0.2 m keeps within controller.SIZE_RATIO
        return {"do": "edit", "kind": "good", "size": size, "seconds": 0.0}

    finish = tool("finish", "--receipt", "{receipt}", "--verdict", "complete", "--reason", f"refined; key {KEY}")
    script = tmp_path / "script.json"
    script.write_text(json.dumps({"rounds": [[edit(0.24), tool("export_glb"), {"do": "crash"}],
                                             [edit(0.25), tool("review"), finish]]}))
    fast = {"poll_s": 0.02, "kill_after_s": 0.5, "finish_wait_s": 0.5}

    def container(root, sif, key_file, budget_s):  # the controller's own session with the fake agent's rounds
        with tempfile.TemporaryDirectory(prefix="harpor-") as sock_dir:
            env = {**os.environ, "PYTHONPATH": str(ROOT), "HARPOR_TOOL_SOCKET": os.path.join(sock_dir, "tools.sock")}
            r = C.run_session(root, [sys.executable, "-m", "harpor.agent.fake_agent", str(script)], env,
                              blender=FAKE_BLENDER, sock_dir=sock_dir, budget_s=20.0, **fast)
        r["key_found_in"] = C.find_key(root, key_file)
        write_json(Path(root) / "result.json", r)
        return r

    monkeypatch.setattr(C, "run_container_session", container)
    monkeypatch.setattr(evidence, "diagonal", lambda original: math.nan)  # a number that is not finite
    (J / "agent").mkdir()
    (J / "agent" / "evidence.json").write_text("of an earlier try")
    r = agent_step.run(J, "/sif/agent.sif", J / "key.env")
    root = J / "agent" / "s"
    assert (r["verdict"], r["glb"], r["reason"]) == ("complete", "control/candidates/v02/candidate.glb",
                                                     f"refined; key {agent_step.REDACTED}")
    assert [(x["kind"], x["outcome"], x["scope"]) for x in r["rounds"]] == [("build", "fail", "local"),
                                                                           ("repair", "pass", "none")]
    assert (root / "control" / "rounds.json").is_file()
    assert {"result.json", "control/evidence.json"} <= set(r["key_found_in"])
    raw, published = (root / "control" / "evidence.json").read_text(), (J / "agent" / "evidence.json").read_text()
    assert KEY in raw and "NaN" in raw and KEY not in published and agent_step.REDACTED in published

    def strict(text):  # as a browser's JSON.parse reads it
        return json.loads(text, parse_constant=lambda c: pytest.fail(f"{c} in the published evidence"))

    ev = strict(published)
    assert ev == agent_step.finite(agent_step.scrub(json.loads(raw), [KEY])) and ev["diagonal_m"] is None
    assert [c["reason"] for c in ev["candidates"]] == ["uninspected export recovered; the session ended without finish",
                                                       f"refined; key {agent_step.REDACTED}"]
    assert all("physics_review" not in c for c in ev["candidates"])
    assert ev["selected"]["version"] == 2 and ev["rounds"] == r["rounds"]
    out = capsys.readouterr().out
    assert KEY not in out and "round 1 build: fail, scope local, candidate v01 (needs_review); findings: " in out
    assert "round 2 repair: pass, scope none, candidate v02 (complete)" in out
    assert "delivered candidate v02 of round 2: " in out

    token = "tok-" + secrets.token_hex(16)
    with TestClient(server.make_app(cfg, token, key_dir=tmp_path / "keys")) as c:
        assert c.get(f"/?token={token}", follow_redirects=False).status_code == 303
        a = c.get(f"/api/jobs/{job_id}").json()["agent_result"]
        assert {k: a[k] for k in ("rounds", "candidate", "surface", "counts")} == {
            k: r[k] for k in ("rounds", "candidate", "surface", "counts")}
        assert a["candidate"]["version"] == 2 and a["candidate"]["outcome"] == "pass"
        assert "agent/evidence.json" not in c.get(f"/api/jobs/{job_id}").json()["files"]
        assert c.get(f"/api/jobs/{job_id}/file/agent/evidence.json").status_code == 404
        for p in ("agent/s/control/evidence.json", "agent/result.json", "agent/evidence.json/../result.json"):
            assert c.get(f"/api/jobs/{job_id}/file/{p}").status_code == 404, p
        jobs.package(J)
        assert "evidence.json" not in zipfile.ZipFile(J / "results.zip").namelist()
        write_json(J / "agent" / "result.json", {**r, "key_found_in": []})  # as a session whose evidence held no key
        assert "agent/evidence.json" in c.get(f"/api/jobs/{job_id}").json()["files"]
        got = c.get(f"/api/jobs/{job_id}/file/agent/evidence.json")
        assert got.headers["content-type"] == "application/json" and strict(got.text) == ev


def test_agent_step_scrubs_the_key_from_the_result(tmp_path, monkeypatch, capsys):
    """The agent can read the key, so it may write it into its finish text or run a command with it.
    Neither the stage log (stdout), agent/result.json, nor results.zip may hold it."""
    from harpor.agent import controller as C

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

    monkeypatch.setattr(C, "has_rounds", lambda root: False)  # one session, without the rounds
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
    from harpor.agent import controller as C

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

    monkeypatch.setattr(C, "has_rounds", lambda root: False)  # one session, without the rounds
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
        from harpor.agent import controller as C
        from harpor.webapp import agent_step
        C.has_rounds = lambda root: False
        AGENT = [sys.executable, "-c", "import os, time; f = open('agent.pid', 'w'); f.write(str(os.getpid())); "
                 "f.close(); time.sleep(120)"]
        C.run_container_session = lambda root, sif, key_file, budget_s: C.run_session(
            root, AGENT, blender={FAKE_BLENDER!r}, sock_dir=tempfile.mkdtemp(prefix="harpor-"))
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
    for m in ("av", "trimesh", "harpor.webapp.video", "harpor.webapp.views", "harpor.webapp.physics_auto"):
        pytest.importorskip(m)
    import xml.etree.ElementTree as ET

    import fake_steps
    from harpor.agent.tools import MODELS
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
    skipped = ("topics", "lidar", "turntable")  # video input, no LiDAR and no 360 video
    assert [s for s in steps.NAMES if job["stages"][s]["status"] != "done"] == list(skipped)
    assert all(job["stages"][s]["status"] == "skipped" for s in skipped)
    assert job["stages"]["turntable"]["note"] == "turned off" and not (J / "turntable").exists()
    placement = json.loads((J / "run" / "placement.json").read_text())
    assert placement["metric_scale"] is None and placement["units"] == "DA3 units (meters)"
    assert job["key_ok"] is False and job["error"] is None
    assert steps.next_glb(J) == J / "agent" / "s" / "work" / "exports" / "001.glb"
    up = json.loads((J / "physics" / "up.json").read_text())
    assert up["glb"].endswith("work/exports/001.glb")
    assert not (J / "physics" / "links").exists() and "links" not in json.loads((J / "physics" / "summary.json").read_text())
    # VoMP ran after sam3d, and align moved its materials onto the aligned mesh: the fake softwood (600 kg/m^3) in
    # every voxel of the final mesh gives the mass. estimate ran between align and mjcf, the real stage with the fake
    # model's answer, and added the friction to align's summary. The MJCF's mass and friction are the summary's.
    st = job["stages"]
    assert st["sam3d"]["ended"] <= st["vomp"]["started"] <= st["vomp"]["ended"] <= st["place"]["started"]
    assert st["align"]["ended"] <= st["estimate"]["started"] <= st["estimate"]["ended"] <= st["mjcf"]["started"]
    summary = json.loads((J / "physics" / "summary.json").read_text())
    assert summary["glb"] == str((J / "physics" / "aligned.glb").resolve()) and summary["config"] == "fake"
    assert summary["material_transfer"]["method"] == "nearest pre-refinement voxel; final geometry revoxelized"
    assert summary["median"] == {"youngs_modulus_pa": 2.0e9, "poisson_ratio": 0.3, "density_kg_m3": 600.0}
    h, voxels = summary["voxel_edge_m"], summary["voxels_occupied"]
    assert summary["rigid_body"]["mass"] == pytest.approx(600.0 * voxels * h ** 3)
    assert (J / "physics" / "voxels.npz").is_file() and "mass_kg" not in json.dumps(summary)
    est = summary["estimate"]
    assert {k: est[k] for k in fake_steps.ESTIMATE} == fake_steps.ESTIMATE and steps.done("estimate", J)
    n = len(json.loads((J / "run" / "cameras.json").read_text()))  # 3 or 4: up to 4 photos, so all of them
    assert (est["agent"], est["model"], est["attempts"], est["views"]) == ("astra", MODELS["astra"], 1, list(range(n)))
    assert summary["friction"] == 0.4 and (J / "physics" / "estimate_views.jpg").is_file()
    mj = json.loads((J / "physics" / "mjcf" / "mjcf.json").read_text())
    assert mj["friction"] == summary["friction"] and mj["inertial"]["mass"] == summary["rigid_body"]["mass"]
    body = ET.parse(J / "physics" / "mjcf" / "wooden_box.xml").getroot()
    assert float(body.find(".//inertial").get("mass")) == pytest.approx(summary["rigid_body"]["mass"])
    assert float(body.find(".//geom").get("friction").split()[0]) == summary["friction"]
    assert "Pre-refinement materials transferred: mass" in (J / "logs" / "align.log").read_text()
    assert "estimate: friction 0.4 (0.3 to 0.5) for cardboard" in (J / "logs" / "estimate.log").read_text()
    names = set(zipfile.ZipFile(J / "results.zip").namelist())
    assert {"posed.glb", "refined.glb", "aligned.glb", "summary.json", "placement.json", "selection.json",
            "agent_result.json", "mjcf/mjcf.json", "mjcf/wooden_box.xml", "mjcf/scene.xml", "usd/usd.json",
            "urdf/model.urdf", "urdf/urdf.json"} <= names
    urdf = json.loads((J / "physics" / "urdf" / "urdf.json").read_text())
    assert urdf["checked"]["links"] == 1 and urdf["checked"]["missing"] == []
    assert all(s["seconds"] is not None for s in jobs.progress(J) if s["name"] not in skipped)
    assert not list(runner.key_dir.iterdir())
    assert not any(KEY.encode() in p.read_bytes() for p in all_files(tmp_path))


def test_the_lidar_geometry_setting(make_runner, monkeypatch):
    """job.json lidar_geometry: inspect for a new job and for an older record without it; another mode alone reruns
    the agent stage and the stages after it (the keyed stages ran, so the key reached them), not the earlier ones."""
    assert steps.lidar_geometry({}) == steps.lidar_geometry({"lidar_geometry": None}) == "inspect"
    with pytest.raises(ValueError, match="lidar_geometry is one of off, inspect, sculpt"):
        steps.lidar_geometry({"lidar_geometry": "carve"})
    runner, cfg = make_runner()

    def command(stage, job, cfg, key_file=None):
        J = cfg.runs / job["id"]
        return ["bash", "-c", f"touch done_{stage} && echo {stage}:{job.get('lidar_geometry')} >> ran"], str(J), {}

    monkeypatch.setattr(steps, "command", command)
    monkeypatch.setattr(steps, "done", lambda stage, J: (Path(J) / f"done_{stage}").exists() or stage == "frames")
    monkeypatch.setattr(steps, "skip", lambda stage, J: None)
    job_id, J = ready_job(cfg)
    assert jobs.load(J)["lidar_geometry"] == "inspect"
    runner.submit_build(job_id, KEY, agent="astra", auto_continue=True)
    wait(J, "done")
    assert "agent:inspect" in (J / "ran").read_text().split()
    (J / "ran").unlink()
    assert runner.resume(job_id, "package", KEY, lidar_geometry="sculpt") == "agent"
    wait(J, "done")
    ran = (J / "ran").read_text().split()
    assert ran[0] == "agent:sculpt" and "select:sculpt" not in ran and jobs.load(J)["lidar_geometry"] == "sculpt"
