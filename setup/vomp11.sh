#!/bin/bash
# VoMP v1.1 (641a22d) in envs/vomp11 (build_plan.md note 9): Python 3.11, torch 2.4.0 cu121, bpy 4.2.0 pip module, under
# $HARPOR_ROOT (by default ~/harpor). The weights come from setup/weights.sh.
# This follows VoMP's install_env.sh with CUDA_VERSION=12.1, minus the sudo apt steps (they install the X and GL libs,
# which the machine must already have; the Dockerfile installs them), diff-gaussian-rasterization (splat path only)
# and torch-tensorrt (use_trt only).
# On Slurm, start it from the login node, in the repo:
#   nohup setsid srun --jobid=<job> --overlap --ntasks=1 bash setup/vomp11.sh > output.log 2>&1 &
set -exo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cuda_arch  # the GPUs that the CUDA extensions are built for (common.sh)
[ -d VoMP ] || git clone https://github.com/nv-tlabs/VoMP.git
git -C VoMP checkout -q 641a22d958054b8f4644aaddd4b1cba2a0d45491
[ -d envs/vomp11 ] || tools/micromamba create -y -p "$G/envs/vomp11" -c conda-forge python=3.11 \
  cuda-compiler=12.1.1 cuda-libraries-dev=12.1.1 cuda-cudart-dev=12.1.105 cuda-nvcc=12.1.105
eval "$(tools/micromamba shell hook -s bash)"; micromamba activate "$G/envs/vomp11"
export CUDA_HOME=$CONDA_PREFIX FORCE_CUDA=1 MAX_JOBS=${MAX_JOBS:-$(nproc)}
cd VoMP
pip install torch==2.4.0 torchvision==0.19.0 xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121
pip install --no-build-isolation "https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.9.post1/flash_attn-2.5.9.post1+cu122torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
# install_env.sh asks for spconv-cu120==2.3.8 with CUDA 12.1, but spconv-cu120 stops at 2.3.6 on PyPI.
pip install spconv-cu121==2.3.8
pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html
pip install --no-build-isolation -r requirements_inference.txt
pip install --no-build-isolation extensions/vox2seq/
pip install -e .
check_imports torch torchvision xformers flash_attn spconv.pytorch kaolin gsplat igl bpy vox2seq vomp.inference  # common.sh
# No GPU is needed here (a docker build has none), so the device name is printed only when there is one.
python -c "import torch, bpy, kaolin
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU'
print(torch.__version__, torch.version.cuda, gpu, 'bpy', bpy.app.version_string, 'kaolin', kaolin.__version__)
assert torch.__version__.startswith('2.4.0')"
echo SETUP_OK
