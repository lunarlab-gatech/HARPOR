#!/bin/bash
# The entrypoint of HARPOR's docker image (containers/Dockerfile). containers/harpor-docker starts it; its commands:
#
#   serve [ARGS]   the web app on port 8765 (harpor.webapp); it prints the login URL
#   run ARGS       one job from the terminal (harpor.webapp.run)
#   weights        the model weights into the data folder (setup/weights.sh; needs HF_TOKEN)
#   check          the GPU, the docker socket and a sibling container as the agent's sandbox sees it
#   anything else  runs as a command, such as bash
set -eo pipefail
G=$HARPOR_ROOT
# The targets of the image's links into the data folder (the Dockerfile), and HOME.
mkdir -p "$HARPOR_DATA"/{cache,weights/sam-3d-objects,weights/vomp,runs,tmp,home}
cd /opt/harpor/code

check() {
  echo "== GPU"; nvidia-smi -L
  echo "== docker"; docker version --format 'client {{.Client.Version}}, daemon {{.Server.Version}}'
  [ -n "$HARPOR_HOST_DATA" ] || { echo "HARPOR_HOST_DATA is not set: start the image with containers/harpor-docker" >&2; exit 1; }
  echo "== the agent's sandbox: a sibling container with the GPU, no network and no data folder"
  # The same command builder as the agent's Blender (controller.container), with a probe for its command.
  local probe='ls /data 2>&1 | head -1; nvidia-smi -L; (getent hosts example.com || echo "no network"); cd /tmp && blender -b --factory-startup -E CYCLES -o /tmp/smoke_#### -f 1 -- --cycles-device OPTIX 2>&1 | grep -iE "optix|saved|error" | head -5'
  mapfile -d '' cmd < <("$G/envs/sam3d/bin/python" -c '
import os, sys
from harpor.agent import controller
sys.stdout.write("\0".join(controller.container(os.environ["HARPOR_SANDBOX"], [], ["bash", "-c", sys.argv[1]], network=False)))' "$probe")
  "${cmd[@]}"
}

case "${1:-serve}" in
  serve) shift || true; exec "$G/envs/web/bin/python" -m harpor.webapp --host 0.0.0.0 --port 8765 "$@" ;;
  run) shift; exec "$G/envs/web/bin/python" -m harpor.webapp.run "$@" ;;
  weights) exec bash setup/weights.sh ;;
  check) check ;;
  *) exec "$@" ;;
esac
