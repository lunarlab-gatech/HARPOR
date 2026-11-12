#!/bin/bash
# The agent's Apptainer image, sif/harpor-agent.sif (build_plan.md note 5), under $HARPOR_ROOT (by default ~/harpor):
# Node.js, the Codex CLI and Blender, from containers/harpor-agent.def. A machine that runs HARPOR without the docker
# image needs it for the agent and the 360 video. The docker image has these
# tools inside and runs them in sibling containers instead, so it skips this script.
set -eo pipefail
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$CODE/setup/common.sh"
command -v apptainer > /dev/null || { echo "no apptainer on this machine: install it, or use the docker image" >&2; exit 1; }
mkdir -p sif
export APPTAINER_TMPDIR=${APPTAINER_TMPDIR:-${TMPDIR:-/tmp}}
export APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-$APPTAINER_TMPDIR/apptainer_cache}
if [ -s sif/harpor-agent.sif ]; then
  echo "sif/harpor-agent.sif is there; delete it to build it again"
else
  # --fakeroot needs subuid and subgid entries for the user; run the build with sudo where they are missing.
  apptainer build --fakeroot sif/harpor-agent.sif "$CODE/containers/harpor-agent.def"
fi
apptainer exec --containall sif/harpor-agent.sif bash -c 'node --version; codex --version; blender --version | head -1'
echo SETUP_OK
