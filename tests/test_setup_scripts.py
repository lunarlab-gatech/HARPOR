"""The scripts in setup/ (build_plan.md notes 4b, 7 and 9). They run for real but offline: each tool that needs the
network or a GPU is a stub that logs its call."""

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "setup"
NOTES = {"mvsam3d": ["4b", "7"], "vomp11": ["7", "9"], "mujoco": ["7", "9"]}
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
    ("mvsam3d", "envs/da3", ["MV-SAM3D", "Depth-Anything-3"],
     ["abb04b5e8af5bc33b0265bdf19937e76bbb6bcdd", "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"]),
    ("vomp11", "envs/vomp11", ["VoMP/weights/vomp_1_1"],
     ["641a22d958054b8f4644aaddd4b1cba2a0d45491", "python=3.11", "cuda-nvcc=12.1.105", "torch==2.4.0",
      "spconv-cu121==2.3.8", "kaolin==0.18.0"]),
    ("mujoco", "envs/mujoco", [],
     ["python=3.11", "mujoco==3.16.0", "coacd==1.0.14", "trimesh==5.1.1", "usd-core==26.8", "numpy==2.4.6",
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


def test_the_default_root_is_the_one_the_app_reads(run, tmp_path):
    """~/scratch is a link to the real storage, as on ICE."""
    (tmp_path / "storage").mkdir()
    (tmp_path / "home" / "scratch").symlink_to(tmp_path / "storage")
    p, _ = run("mujoco", HARPOR_ROOT=None)
    assert p.returncode == 0 and (tmp_path / "storage" / "gator" / "envs" / "mujoco").is_dir()


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
    assert "2026-11-01" in text and "has not been run as a recipe" in text and "file://" not in text
