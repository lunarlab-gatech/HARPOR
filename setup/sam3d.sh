#!/bin/bash
# envs/sam3d: SAM 3D Objects (f91db41) with nvdiffrast and anthropic (build_plan.md notes 4 and 5), about 1 h, under
# $HARPOR_ROOT (by default ~/harpor). MV-SAM3D, the agent controller and most stages run in this env. The weights come
# from setup/weights.sh. The env brings its own CUDA 12.1 toolkit and gcc 12, so the machine needs only an NVIDIA
# driver at run time and no GPU at build time (set TORCH_CUDA_ARCH_LIST then, see common.sh).
set -exo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cuda_arch  # pytorch3d, flash_attn and gsplat are built for these GPUs (common.sh)
[ -d sam-3d-objects ] || git clone https://github.com/facebookresearch/sam-3d-objects.git
git -C sam-3d-objects checkout -q f91db411c50efee93d8db7aeb323885650f6f722
[ -d envs/sam3d ] || tools/micromamba create -y -p "$G/envs/sam3d" -f sam-3d-objects/environments/default.yml
eval "$(tools/micromamba shell hook -s bash)"; micromamba activate "$G/envs/sam3d"
export CUDA_HOME=$CONDA_PREFIX FORCE_CUDA=1 MAX_JOBS=${MAX_JOBS:-$(nproc)}
export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu121"
cd sam-3d-objects
pip install -e '.[dev]'
pip install -e '.[p3d]'
PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html" pip install -e '.[inference]'
# SAM 3D's own patch: it replaces hydra 1.3.2's core/utils.py with a fixed copy, so it can run again.
python patching/hydra
pip install 'huggingface-hub[cli]<1.0'
pip install --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast.git@v0.3.3"
# The agent controller runs Claude Opus 5.5 (note 5) and the part list (note 11) from this env. SAM 3D leaves trimesh
# open, and HARPOR's part code was tested with the 5.1.1 of envs/sam3d on ICE.
pip install anthropic==1.15.0 trimesh==5.1.1
check_imports torch pytorch3d kaolin gsplat nvdiffrast flash_attn anthropic rtree trimesh  # common.sh
python -c "import torch
print('sam3d env', torch.__version__, torch.version.cuda, 'GPU' if torch.cuda.is_available() else 'no GPU')
assert torch.__version__.startswith('2.5.1')"
echo SETUP_OK
