"""Tool server: runs the agent's helper commands on the host (build_plan.md note 5).

The agent's container sees the socket folder read-only and the client in bin/, nothing else of the
controller. Each request runs `python -m harpor.agent.tools` on the host with the control folder,
which holds the events and the clock record, so the agent cannot forge the export events that move
its deadline. Requests run one at a time. Protocol: one JSON line {"argv": [...]} in, one JSON
object {"output": text, "code": exit code} back.

The helper runs from the repository, not from the work folder: `python -m` puts its starting folder
first on sys.path, so a json.py (or a harpor/ package) that the agent writes into /work would
otherwise run on the host, outside the agent's container. PYTHONSAFEPATH=1 (Python 3.11 and later)
keeps that folder off sys.path as well. The helpers take every path relative to GATOR_WORK.
"""

import contextlib
import json
import os
import shlex
import signal
import socketserver
import subprocess
import sys
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
COMMANDS = ("export_glb", "review", "checkpoint", "finish", "set_joint", "add_part", "review_articulation", "simulate")
MAX_ARGS, MAX_ARG_CHARS, MAX_OUTPUT_CHARS = 64, 4096, 16384  # add_part with 6 planes, a seed and a source has 51 arguments
TERM_GRACE_S = 3.0


def parse_request(line):
    """argv from one request line, or ValueError with the reason."""
    req = json.loads(line)
    argv = req.get("argv") if isinstance(req, dict) else None
    if not (isinstance(argv, list) and 0 < len(argv) <= MAX_ARGS
            and all(isinstance(a, str) and len(a) <= MAX_ARG_CHARS for a in argv)):
        raise ValueError(f"argv must be a list of 1 to {MAX_ARGS} strings")
    if argv[0] not in COMMANDS:
        raise ValueError(f"unknown command {argv[0]!r}; use one of {', '.join(COMMANDS)}")
    return argv


class ToolServer:
    def __init__(self, socket_path, work, control, blender):
        self.socket_path, self.work = str(socket_path), str(work)
        self.env = {**os.environ, "PYTHONPATH": str(REPO), "PYTHONSAFEPATH": "1", "GATOR_WORK": str(work),
                    "GATOR_CONTROL": str(control), "GATOR_BLENDER": shlex.join(blender)}
        self._lock = threading.Lock()
        self._proc, self._stopping, self._thread = None, False, None
        server = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                try:
                    output, code = server.run(parse_request(self.rfile.readline(2 * MAX_ARGS * MAX_ARG_CHARS)))
                except ValueError as e:
                    output, code = f"{e}\n", 2
                with contextlib.suppress(OSError):
                    self.wfile.write(json.dumps({"output": output[:MAX_OUTPUT_CHARS], "code": code}).encode() + b"\n")

        with contextlib.suppress(FileNotFoundError):
            os.unlink(self.socket_path)
        self._server = socketserver.UnixStreamServer(self.socket_path, Handler)

    def run(self, argv):
        """Run one helper command on the host. Returns (output, exit code)."""
        with self._lock:
            if self._stopping:
                return "the session has ended\n", 1
            p = self._proc = subprocess.Popen(
                [sys.executable, "-m", "harpor.agent.tools", *argv], cwd=REPO, env=self.env,
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            out, _ = p.communicate()
        finally:
            with self._lock:
                self._proc = None
        return out.decode(errors="replace"), p.returncode if p.returncode >= 0 else 1

    def start(self):
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        """Refuse new requests, end the running one (its Blender too), then close the socket."""
        with self._lock:
            self._stopping = True
            p = self._proc
        if p is not None:
            with contextlib.suppress(ProcessLookupError):
                p.send_signal(signal.SIGTERM)  # tools kills its Blender group, then exits
            try:
                p.wait(TERM_GRACE_S)
            except subprocess.TimeoutExpired:
                pass
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(p.pid, signal.SIGKILL)
        if self._thread is not None:
            self._server.shutdown()
            self._thread.join()
        self._server.server_close()
        with contextlib.suppress(FileNotFoundError):
            os.unlink(self.socket_path)
