"""Client for the helper commands inside the agent's container (build_plan.md note 5).

The helpers run on the host in the controller's tool server. This script sends its command line over
the Unix socket in GATOR_SOCKET, prints the reply and exits with the helper's exit code. It uses only
the standard library, because it runs on the container's system python3.

    gator_tool.py export_glb|review|checkpoint|finish [args...]
"""

import json
import os
import socket
import sys

SOCKET = "/opt/gator/sock/tools.sock"


def call(argv, path=None):
    """Returns (output, exit code) of one helper command."""
    with socket.socket(socket.AF_UNIX) as s:
        s.connect(path or os.environ.get("GATOR_SOCKET", SOCKET))
        s.sendall(json.dumps({"argv": list(argv)}).encode() + b"\n")
        data = b"".join(iter(lambda: s.recv(65536), b""))
    reply = json.loads(data)
    return reply["output"], reply["code"]


def main():
    try:
        output, code = call(sys.argv[1:])
    except (OSError, ValueError, KeyError) as e:
        sys.exit(f"the helper commands are not available ({e}); the session may have ended")
    sys.stdout.write(output)
    sys.exit(code)


if __name__ == "__main__":
    main()
