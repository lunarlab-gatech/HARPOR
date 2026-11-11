"""The scripts in setup/ (build_plan.md notes 3 to 10) and the docker image's files in containers/. The scripts run for
real but offline: each tool that needs the network or a GPU is a stub that logs its call."""

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "setup"
NOTES = {"install": ["7"], "web": ["7", "10"], "sam3d": ["4", "7"], "mvsam3d": ["4b", "7"], "sam3seg": ["3", "7"],
         "vomp11": ["7", "9"], "mujoco": ["7", "9"], "agent": ["5", "7"], "weights": ["7"]}
CONTAINERS = ROOT / "containers"
# One stub under every tool's name. Besides logging, it makes what the script's next step needs.
STUB = r"""#!/bin/bash
echo "$0 $* | arch=$TORCH_CUDA_ARCH_LIST jobs=$MAX_JOBS conf=$PIP_CONFIG_FILE hf=$HF_HOME" >> "$STUB_LOG"
case "$(basename "$0") $1" in
  "git clone") url=${@: -1}; url=${url##*/}; mkdir "${url%.git}" ;;
  "tar "*) cp "$0" tools/micromamba ;;
  "micromamba create") mkdir -p "$4/bin"; cp "$0" "$4/bin/python"; cp "$0" "$4/bin/pip" ;;
  "micromamba shell") echo 'micromamba() { export CONDA_PREFIX=$2 PATH=$2/bin:$PATH; }' ;;
  "python -m") mkdir -p "$3/bin"; cp "$0" "$3/bin/python"; cp "$0" "$3/bin/pip" ;;
  "python -c") mkdir -p weights/vomp_1_1 ;;
  "nvidia-smi "*) [ -n "$GPUS" ] || { echo "No devices were found"; exit 6; }; printf '%s\n' $GPUS ;;
  "nproc "*) echo 7 ;;
esac
"""


def stub(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(STUB)
    path.chmod(0o755)


def note(number):
    """One note of build_plan.md, from its heading up to the next note's."""
    text = (ROOT / "docs" / "build_plan.md").read_text()
    return re.search(rf"^### {number}\. .*?(?=^### |\Z)", text, re.M | re.S).group()


@pytest.fixture
def run(tmp_path):
    """run(script, VARIABLE=value, ...) runs setup/<script>.sh with the root tmp_path/root, envs/sam3d from note 4, a
    GPU of capability 8.6 (GPUS) and no module command. None removes a variable. It returns the process and the
    stubs' calls."""
    for tool in ("git", "curl", "tar", "pip", "python", "nvidia-smi", "nproc"):
        stub(tmp_path / "bin" / tool)
    stub(tmp_path / "root" / "envs" / "sam3d" / "bin" / "python")
    (tmp_path / "home").mkdir()
    log = tmp_path / "calls.log"

    def go(script, **env):
        log.write_text("")
        base = {"PATH": f"{tmp_path / 'bin'}:/usr/bin:/bin", "HOME": str(tmp_path / "home"), "STUB_LOG": str(log),
                "HARPOR_ROOT": str(tmp_path / "root"), "GPUS": "8.6"}
        env = {k: v for k, v in {**base, **env}.items() if v is not None}
        p = subprocess.run(["bash", str(SETUP / f"{script}.sh")], cwd=tmp_path, env=env, capture_output=True, text=True)
        return p, log.read_text().splitlines()

    return go


@pytest.mark.parametrize("script", ["common", *NOTES])
def test_script_passes_bash_n(script):
    assert subprocess.run(["bash", "-n", str(SETUP / f"{script}.sh")]).returncode == 0


@pytest.mark.parametrize("script", NOTES)
def test_script_names_the_root_and_its_notes_and_the_readme_name_it(script):
    assert "$HARPOR_ROOT" in (SETUP / f"{script}.sh").read_text()
    assert f"setup/{script}.sh" in (ROOT / "README.md").read_text()
    assert all(f"setup/{script}.sh" in note(n) for n in NOTES[script])


def test_no_doc_names_the_old_run_folders():
    docs = [ROOT / "README.md", *(ROOT / "docs").glob("*.md")]
    assert [p.name for p in docs if re.search(r"runs/(mvsam3d|vomp11)_setup", p.read_text())] == []


@pytest.mark.parametrize("script, env, made, pins", [
    ("web", "envs/web", [], ["python=3.12", "fastapi==0.143.2", "uvicorn==0.54.0", "av==19.0.1", "rosbags==0.11.8",
                             "anthropic==1.15.0", "httpx==0.28.1", "trimesh==5.1.1"]),
    ("sam3seg", "envs/sam3seg", [], ["python=3.12", "torch==2.14.1", "whl/cu130", "transformers==5.20.0"]),
    ("mvsam3d", "envs/da3", ["MV-SAM3D", "Depth-Anything-3"],
     ["abb04b5e8af5bc33b0265bdf19937e76bbb6bcdd", "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4", "torch==2.14.1"]),
    ("vomp11", "envs/vomp11", ["VoMP/weights/vomp_1_1"],
     ["641a22d958054b8f4644aaddd4b1cba2a0d45491", "python=3.11", "cuda-nvcc=12.1.105", "torch==2.4.0",
      "spconv-cu121==2.3.8", "kaolin==0.18.0"]),
    ("mujoco", "envs/mujoco", [],
     ["python=3.11", "mujoco==4.0.0", "coacd==1.0.14", "trimesh==5.1.1", "usd-core==26.8", "numpy==2.4.6",
      "scipy==1.17.1", "pillow==12.3.0"]),
])
def test_script_builds_under_harpor_root_without_a_module_command(run, tmp_path, script, env, made, pins):
    p, calls = run(script)
    root = tmp_path / "root"
    assert p.returncode == 0, p.stderr
    assert all((root / d).is_dir() for d in [env, *made])
    assert all(any(pin in c for c in calls) for pin in pins)
    pips = [c for c in calls if "/pip " in c]
    assert pips and all(c.startswith(f"{root / env}/bin/pip ") for c in pips)
    assert all("conf=/dev/null" in c and f"hf={root}/.cache/huggingface" in c for c in pips)


def test_mujoco_removes_a_3x_mujoco_before_it_installs_4(run):
    # The stub python succeeds at every -c, so the version check takes the env's mujoco for a 3.x one.
    p, calls = run("mujoco")
    assert p.returncode == 0, p.stderr
    pips = [c.split(" | ")[0] for c in calls if "/envs/mujoco/bin/pip " in c]
    assert pips[0].endswith("/envs/mujoco/bin/pip uninstall -y mujoco") and "mujoco==4.0.0" in pips[1]


def test_the_default_root_is_the_one_the_app_reads(run, tmp_path, monkeypatch):
    from harpor.webapp import steps

    p, _ = run("mujoco", HARPOR_ROOT=None)
    assert p.returncode == 0 and (tmp_path / "home" / "harpor" / "envs" / "mujoco").is_dir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HARPOR_ROOT", raising=False)
    assert steps.config_from_env().G == tmp_path / "home" / "harpor"


def test_module_purge_runs_first_where_the_command_exists(run, tmp_path):
    stub(tmp_path / "bin" / "module")
    p, calls = run("mujoco")
    assert p.returncode == 0 and calls[0].startswith(f"{tmp_path / 'bin' / 'module'} purge ")


def test_micromamba_is_fetched_once_and_a_built_env_stays(run):
    p, first = run("mujoco")
    assert p.returncode == 0 and any("/curl " in c for c in first) and any(" create " in c for c in first)
    p, again = run("mujoco")
    assert p.returncode == 0 and not any("/curl " in c or " create " in c for c in again)


def builds(calls):
    """The calls that compile something for the GPUs."""
    return [c for c in calls if "/envs/vomp11/bin/pip install" in c]


@pytest.mark.parametrize("env, arch", [
    ({"GPUS": "8.6"}, "8.6"),
    ({"GPUS": "9.0 8.9 9.0"}, "8.9;9.0"),  # two kinds of GPU, one of them twice
    ({"GPUS": "8.6", "TORCH_CUDA_ARCH_LIST": "8.9;9.0"}, "8.9;9.0"),  # a cluster's mix, which this node cannot see
    ({"GPUS": "", "TORCH_CUDA_ARCH_LIST": "8.6"}, "8.6"),  # no need for a GPU on this machine
])
def test_vomp11_takes_the_gpu_list_from_the_environment_or_else_the_gpus(run, env, arch):
    p, calls = run("vomp11", **env)
    assert p.returncode == 0, p.stderr
    assert builds(calls) and all(f"arch={arch} " in c for c in builds(calls))


def test_vomp11_stops_before_building_when_it_finds_no_gpu(run):
    p, calls = run("vomp11", GPUS="")
    assert p.returncode != 0 and "set TORCH_CUDA_ARCH_LIST" in p.stderr
    assert not any(" create " in c or " install " in c for c in calls)


def test_vomp11_builds_with_the_cores_it_has_unless_told(run):
    _, calls = run("vomp11")
    assert builds(calls) and all(" jobs=7 " in c for c in builds(calls))
    _, calls = run("vomp11", MAX_JOBS="3")
    assert builds(calls) and all(" jobs=3 " in c for c in builds(calls))


def test_mvsam3d_stops_without_envs_sam3d(run, tmp_path):
    (tmp_path / "root" / "envs" / "sam3d" / "bin" / "python").unlink()
    p, calls = run("mvsam3d")
    assert p.returncode != 0 and "note 4" in p.stderr
    assert not any("/git " in c or " install " in c for c in calls)


def test_mujoco_script_says_where_it_came_from():
    text = (SETUP / "mujoco.sh").read_text()
    assert "2026-11-01" in text and "First run as a recipe on 2026-11-08" in text and "file://" not in text


def test_sam3d_builds_the_env_of_note_4_with_anthropic(run, tmp_path):
    """The fixture's envs/sam3d stands for a built env elsewhere; here the script builds its own."""
    import shutil

    root = tmp_path / "root"
    shutil.rmtree(root / "envs" / "sam3d")
    p, calls = run("sam3d", GPUS="8.9 9.0")
    assert p.returncode == 0, p.stderr
    assert any("f91db411c50efee93d8db7aeb323885650f6f722" in c for c in calls)
    assert any(" create -y -p " in c and "environments/default.yml" in c for c in calls)
    pips = [c for c in calls if "/pip " in c]
    assert pips and all(c.startswith(f"{root}/envs/sam3d/bin/pip ") and "arch=8.9;9.0 " in c for c in pips)
    assert any("nvdiffrast.git@v0.3.3" in c for c in pips) and any("anthropic==1.15.0 trimesh==5.1.1" in c for c in pips)
    assert any(c.startswith(f"{root}/envs/sam3d/bin/python patching/hydra ") for c in calls)


def test_a_gpu_newer_than_hopper_stops_the_build(run):
    p, calls = run("vomp11", TORCH_CUDA_ARCH_LIST="9.0;12.0")
    assert p.returncode != 0 and "9.0 (Hopper) at most" in p.stderr
    assert not any(" create " in c or " install " in c for c in calls)
    p, calls = run("sam3d", GPUS="10.0")
    assert p.returncode != 0 and "has 10.0" in p.stderr and not any(" install " in c for c in calls)


def test_weights_needs_a_token_and_then_fetches_the_four_models(run, tmp_path):
    root = tmp_path / "root"
    for e in ("sam3seg", "da3", "vomp11"):
        stub(root / "envs" / e / "bin" / "python")
    p, calls = run("weights")
    assert p.returncode != 0 and "set HF_TOKEN" in p.stderr and not any("snapshot_download" in c for c in calls)
    # What the downloads leave, which the stubs do not write.
    (root / "sam-3d-objects" / "checkpoints" / "hf").mkdir(parents=True)
    (root / "sam-3d-objects" / "checkpoints" / "hf" / "pipeline.yaml").write_text("x")
    (root / "VoMP" / "weights" / "vomp_1_1").mkdir(parents=True)
    (root / "VoMP" / "weights" / "vomp_1_1" / "inference.json").write_text("{}")
    p, calls = run("weights", HF_TOKEN="hf_test")
    assert p.returncode == 0 and "WEIGHTS_OK" in p.stdout, p.stderr
    fetched = "\n".join(calls)  # a python -c of several lines logs several lines
    assert "facebook/sam-3d-objects" not in fetched  # its checkpoints are there
    assert all(m in fetched for m in ("facebook/sam3", "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
                                      "nvidia/PhysicalAI-Simulation-VoMP-Model"))
    assert "HF_TOKEN=hf_test" not in " ".join(calls)
    # The stages load the gated models with the token, so it stays in the cache, for this user only.
    token = root / ".cache" / "huggingface" / "token"
    assert token.read_text() == "hf_test" and token.stat().st_mode & 0o777 == 0o600


def test_install_runs_every_step_in_order_and_skips_on_request(run):
    p, calls = run("install", HARPOR_SKIP_AGENT="1", HARPOR_SKIP_WEIGHTS="1")
    assert p.returncode == 0 and "INSTALL_OK" in p.stdout, p.stderr
    steps = [line.split()[1] for line in p.stdout.splitlines() if line.startswith("==== ") and "INSTALL_OK" not in line]
    assert steps == ["web", "sam3d", "mvsam3d", "sam3seg", "vomp11", "mujoco", "agent:", "weights:"]
    assert "skipped (HARPOR_SKIP_AGENT=1)" in p.stdout and "skipped (HARPOR_SKIP_WEIGHTS=1)" in p.stdout


def test_dockerfile_installs_the_agent_tools_of_the_apptainer_image():
    """The docker image and containers/harpor-agent.def must give the agent the same tools."""
    dockerfile, sif = (CONTAINERS / "Dockerfile").read_text(), (CONTAINERS / "harpor-agent.def").read_text()
    for pin in re.findall(r"https://\S+(?:node-v[\d.]+|blender-[\d.]+)-linux-x64\.tar\.xz|@openai/codex@[\d.]+", sif):
        assert pin in dockerfile
    assert "HARPOR_SKIP_AGENT=1 HARPOR_SKIP_WEIGHTS=1" in dockerfile and "setup/install.sh" in dockerfile


def test_dockerfile_links_point_into_the_data_folder_that_the_entrypoint_makes():
    dockerfile, entry = (CONTAINERS / "Dockerfile").read_text(), (CONTAINERS / "entrypoint.sh").read_text()
    targets = re.findall(r"ln -s /data/(\S+) ", dockerfile)
    assert sorted(targets) == ["cache", "runs", "weights/sam-3d-objects", "weights/vomp"]
    made = re.search(r'mkdir -p "\$HARPOR_DATA"/\{([^}]*)\}', entry).group(1).split(",")
    assert set(targets) <= set(made)
    # The agent's containers find the helper client where Apptainer binds it (controller._mounts).
    assert "ln -s /opt/harpor/code/harpor/agent/bin /opt/gator/bin" in dockerfile
    assert "HARPOR_DATA=/data" in dockerfile and "TMPDIR=/data/tmp" in dockerfile


@pytest.mark.parametrize("script", ["entrypoint.sh", "harpor-docker"])
def test_container_scripts_pass_bash_n(script):
    assert subprocess.run(["bash", "-n", str(CONTAINERS / script)]).returncode == 0
