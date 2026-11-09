"""Run one `docker run` so that its container cannot outlive it (the docker sandbox, controller.container):

    python docker_guard.py DOCKER run [options] IMAGE [command]

A killed `docker run` leaves its container running, because the container is a child of the docker daemon and not of
the process that started it. The controller, the tool helpers, the Claude shell and the web app's cancel all stop a
sandbox by signalling or killing processes, which works for Apptainer, whose processes are ours. This script makes a
docker container follow its `docker run` the same way: it gives the container a new name, forks a guardian in a
session of its own (so a signal to the caller's process group does not reach it), and replaces itself with
`DOCKER run --name NAME ...`, which keeps the write end of a pipe. However docker run ends, the guardian then reads end
of file and removes the container by name (`DOCKER rm -f NAME`), which does nothing when --rm removed it already.

Standalone, with no harpor imports: it runs from whatever folder the sandbox command runs in.
"""

import os
import secrets
import subprocess
import sys


def main(argv):
    if len(argv) < 2 or argv[1] != "run":
        sys.exit("usage: docker_guard.py DOCKER run [options] IMAGE [command]")
    docker, name = argv[0], f"harpor-{secrets.token_hex(6)}"
    r, w = os.pipe()  # both non-inheritable, as Python makes them
    pid = os.fork()
    if pid == 0:  # the guardian
        try:
            os.setsid()
            if os.fork():  # a grandchild, so that the guardian is nobody's zombie
                os._exit(0)
            # No copy of the caller's files: a pipe that the caller reads to end of file must close when docker run ends.
            null = os.open(os.devnull, os.O_RDWR)
            for fd in (0, 1, 2):
                os.dup2(null, fd)
            os.closerange(3, r)
            os.closerange(r + 1, max(w + 1, min(os.sysconf("SC_OPEN_MAX"), 1 << 16)))  # w too
            while os.read(r, 4096):
                pass
            subprocess.run([docker, "rm", "-f", name], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=120)
        finally:
            os._exit(0)
    os.close(r)
    os.waitpid(pid, 0)
    os.set_inheritable(w, True)
    os.execvp(docker, [docker, "run", "--name", name, *argv[2:]])


if __name__ == "__main__":
    main(sys.argv[1:])
