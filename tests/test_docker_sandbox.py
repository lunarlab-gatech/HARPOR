"""The docker sandbox (controller.container with "docker:IMAGE") and docker_guard.py, which removes a container
whose `docker run` was killed. A fake docker command stands for the real one."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from harpor.agent import controller as C
from harpor.agent import tools

IMAGE = "docker:harpor:test"
FAKE_DOCKER = r"""#!/bin/bash
echo "$*" >> "$DOCKER_LOG"
if [ "$1" = run ]; then
  [ -n "$RUN_SLEEP" ] && sleep "$RUN_SLEEP"
  echo ran
fi
"""


def flags(cmd, flag):
    return [cmd[i + 1] for i, a in enumerate(cmd) if a == flag]


@pytest.fixture
def inside(monkeypatch, tmp_path):
    """HARPOR's own container: the data folder /data is the host's tmp_path/host."""
    monkeypatch.setenv("HARPOR_DATA", "/data")
    monkeypatch.setenv("HARPOR_HOST_DATA", str(tmp_path / "host"))
    return tmp_path / "host"


def test_docker_image_names_the_image_and_a_sif_none():
    assert C.docker_image("docker:harpor:0.1") == "harpor:0.1" and C.docker_image("/G/sif/harpor-agent.sif") is None


def test_codex_container_in_docker(inside):
    root = Path("/data/runs/j1/agent/s")
    cmd = C.apptainer_command(root, IMAGE, "/dev/shm/k/key.env", "/data/tmp/gator-x")
    assert cmd[:2 + len(C.DOCKER_RUN)] == [sys.executable, str(C.DOCKER_GUARD), *C.DOCKER_RUN]
    assert cmd[cmd.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}" and "-i" in cmd
    # The data folder's paths map to the host's; the helper client is in the image, and the DNS files are not bound.
    assert flags(cmd, "-v") == [f"{inside}/runs/j1/agent/s/work:/work", f"{inside}/runs/j1/agent/s/session:/session",
                                f"{inside}/tmp/gator-x:/opt/gator/sock:ro"]
    assert "--network" not in cmd and "--bind" not in cmd  # Codex needs the network
    assert flags(cmd, "-e") == ["HOME=/tmp", "TMPDIR=/tmp", "CODEX_HOME=/session/codex_home",
                                "GATOR_SOCKET=/opt/gator/sock/tools.sock", "PREPEND_PATH=/opt/gator/bin"]
    assert cmd[cmd.index("--env-file") + 1] == "/dev/shm/k/key.env"
    assert cmd[-len(C.codex_command()) - len(C.CODEX_KEY_WRAPPER) - 3:] == \
        ["-w", "/work", "harpor:test", *C.CODEX_KEY_WRAPPER, *C.codex_command()]


def test_claude_shell_and_blender_in_docker_have_no_network_and_no_key(inside):
    root = Path("/data/runs/j1/agent/s")
    shell = C.shell_command(root, IMAGE, "/data/tmp/gator-x")
    assert shell[shell.index("--network") + 1] == "none" and "-i" in shell and "--env-file" not in shell
    assert not [v for v in flags(shell, "-v") if v.endswith(":/session")]
    assert shell[-4:] == ["harpor:test", "bash", "--noprofile", "--norc"]
    blender = C.blender_command(root, IMAGE)
    assert blender[blender.index("--network") + 1] == "none" and "-i" not in blender and blender[-2:] == \
        ["harpor:test", "blender"]
    work = f"{inside}/runs/j1/agent/s/work"
    assert flags(blender, "-v") == [f"{work}:/data/runs/j1/agent/s/work",
                                    f"{inside}/runs/j1/agent/s/control/staging:/data/runs/j1/agent/s/control/staging",
                                    f"{work}:/work",
                                    f"{inside}/runs/j1/agent/s/control/exports:/data/runs/j1/agent/s/control/exports:ro"]
    # The Blender scripts are in the image at the same path, so they are not bound.
    assert not [v for v in flags(blender, "-v") if str(tools.BLENDER_SCRIPTS) in v]


def test_a_path_outside_the_data_folder_cannot_reach_a_sibling(inside):
    with pytest.raises(ValueError, match="only HARPOR_DATA"):
        C.sif_blender(IMAGE, rw=["/tmp/elsewhere"])


def test_outside_a_container_every_path_is_bound_as_it_is(monkeypatch):
    """HARPOR run directly on the host, with only the sandbox in docker: the daemon sees the same paths."""
    monkeypatch.delenv("HARPOR_HOST_DATA", raising=False)
    cmd = C.sif_blender(IMAGE, rw=["/home/u/runs/j1/turntable"], ro=["/home/u/runs/j1/a.glb"])
    assert flags(cmd, "-v") == ["/home/u/runs/j1/turntable:/home/u/runs/j1/turntable",
                                "/home/u/runs/j1/a.glb:/home/u/runs/j1/a.glb:ro",
                                f"{tools.BLENDER_SCRIPTS}:{tools.BLENDER_SCRIPTS}:ro"]


def test_claude_command_keeps_the_docker_sandbox_name(tmp_path):
    cmd = C.claude_command(tmp_path, IMAGE, "/tmp/gator-x", tmp_path / "k.env")
    assert cmd[cmd.index("--sif") + 1] == IMAGE


@pytest.fixture
def docker(tmp_path):
    """A fake docker command and the log of its calls."""
    exe = tmp_path / "docker"
    exe.write_text(FAKE_DOCKER)
    exe.chmod(0o755)
    log = tmp_path / "docker.log"
    log.write_text("")
    return exe, log


def wait_for(predicate, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_guard_names_the_container_and_removes_it_after_a_normal_exit(docker):
    exe, log = docker
    p = subprocess.run([sys.executable, str(C.DOCKER_GUARD), str(exe), "run", "--rm", "img", "echo"],
                       env={**os.environ, "DOCKER_LOG": str(log)}, capture_output=True, text=True, timeout=30)
    assert p.returncode == 0 and p.stdout == "ran\n"
    assert wait_for(lambda: len(log.read_text().splitlines()) == 2)
    run, rm = log.read_text().splitlines()
    name = run.split()[2]
    assert run == f"run --name {name} --rm img echo" and name.startswith("harpor-") and rm == f"rm -f {name}"


def test_guard_removes_the_container_when_its_docker_run_is_killed(docker):
    """The controller and the web app's cancel SIGKILL a process group: the guardian is in a session of its own."""
    exe, log = docker
    p = subprocess.Popen([sys.executable, str(C.DOCKER_GUARD), str(exe), "run", "img"], start_new_session=True,
                         env={**os.environ, "DOCKER_LOG": str(log), "RUN_SLEEP": "30"}, stdout=subprocess.DEVNULL)
    assert wait_for(lambda: log.read_text().startswith("run --name "))
    name = log.read_text().split()[2]
    os.killpg(p.pid, signal.SIGKILL)
    p.wait()
    assert wait_for(lambda: f"rm -f {name}" in log.read_text().splitlines())


def test_guard_keeps_no_copy_of_the_callers_pipe(docker):
    """The Claude shell's reader waits for end of file on the shell's stdout, so the guardian must not hold it."""
    exe, log = docker
    p = subprocess.Popen([sys.executable, str(C.DOCKER_GUARD), str(exe), "run", "img"], stdout=subprocess.PIPE,
                         env={**os.environ, "DOCKER_LOG": str(log)})
    t0 = time.time()
    assert p.stdout.read() == b"ran\n" and time.time() - t0 < 10
    p.wait()


def test_guard_refuses_anything_but_run(docker):
    exe, _ = docker
    p = subprocess.run([sys.executable, str(C.DOCKER_GUARD), str(exe), "ps"], capture_output=True, text=True)
    assert p.returncode != 0 and "usage" in p.stderr
