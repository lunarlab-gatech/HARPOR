#!/bin/bash
# VoMP v1.1 (641a22d) in envs/vomp11 (build_plan.md note 9): Python 3.11, torch 2.4.0 cu121, bpy 4.2.0 pip module, under
# $HARPOR_ROOT (by default ~/scratch/gator).
# This follows VoMP's install_env.sh with CUDA_VERSION=12.1, minus the sudo apt steps (they install the X and GL libs,
# which a machine must already have, as ICE nodes do), diff-gaussian-rasterization (splat path only) and torch-tensorrt
# (use_trt only).
# Run it on a GPU node. On Slurm, start it from the login node, in the repo:
#   nohup setsid srun --jobid=<job> --overlap --ntasks=1 bash setup/vomp11.sh > output.log 2>&1 &
set -exo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
# The CUDA extensions below are built for the GPUs in TORCH_CUDA_ARCH_LIST, and a build for the wrong GPU does not run.
# The list comes from the environment, or else from the GPUs that this machine sees. On a cluster with mixed GPUs, set
# the whole list by hand: see build_plan.md note 9.
if [ -z "$TORCH_CUDA_ARCH_LIST" ]; then
  TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -u | paste -sd ';' -) \
    || { echo "nvidia-smi found no GPU: run this on a GPU node, or set TORCH_CUDA_ARCH_LIST" >&2; exit 1; }
fi
export TORCH_CUDA_ARCH_LIST
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
python -c "from huggingface_hub import snapshot_download as s; print(s('nvidia/PhysicalAI-Simulation-VoMP-Model', allow_patterns=['vomp_1_1/*'], local_dir='weights'))"
ls -la weights/vomp_1_1
python -c "import torch, torchvision, xformers, flash_attn, spconv.pytorch, kaolin, gsplat, igl, bpy, vox2seq, vomp.inference
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), 'bpy', bpy.app.version_string, 'kaolin', kaolin.__version__)
assert torch.__version__.startswith('2.4.0')"
echo SETUP_OK
