#!/bin/bash
# Everything HARPOR runs on, under $HARPOR_ROOT (by default ~/harpor): the seven envs, the agent's Apptainer image and
# the weights. It takes one to two hours and about 90 GB. Needs: Linux x86_64, curl, git, bzip2, the X and GL
# libraries (see the Dockerfile), and an NVIDIA GPU from Ampere to Hopper (compute capability 8.0 to 9.0) with a driver
# for CUDA 13.0 or newer. The docker image runs this script itself, with the two skips below.
#
#   HARPOR_SKIP_AGENT=1    skip setup/agent.sh, the Apptainer image (the docker image has its tools inside)
#   HARPOR_SKIP_WEIGHTS=1  skip setup/weights.sh (it needs HF_TOKEN; run it on its own later)
#   TORCH_CUDA_ARCH_LIST   the GPUs to build for, when this machine has no GPU or not all of them (common.sh)
#   MAX_JOBS               parallel compile jobs (default: the number of cores)
#
# Each step can also run on its own, and a step skips what is already built. Run it again after a failure.
set -eo pipefail
S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
for step in web sam3d mvsam3d sam3seg vomp11 mujoco; do
  echo "==== $step $(date +%T)"
  bash "$S/$step.sh"
done
if [ "$HARPOR_SKIP_AGENT" = 1 ]; then
  echo "==== agent: skipped (HARPOR_SKIP_AGENT=1)"
elif ! command -v apptainer > /dev/null; then
  echo "==== agent: skipped, no apptainer. The agent, part split, 360 video and soft-part video need setup/agent.sh." >&2
else
  echo "==== agent $(date +%T)"
  bash "$S/agent.sh"
fi
if [ "$HARPOR_SKIP_WEIGHTS" = 1 ]; then
  echo "==== weights: skipped (HARPOR_SKIP_WEIGHTS=1); run setup/weights.sh with HF_TOKEN set"
else
  echo "==== weights $(date +%T)"
  bash "$S/weights.sh"
fi
echo "==== INSTALL_OK $(date +%T)"
