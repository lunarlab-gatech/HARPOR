#!/bin/bash
# envs/sam3seg: SAM 3 masks through transformers 5.20.0 (build_plan.md note 3), a few minutes, under $HARPOR_ROOT (by
# default ~/harpor). transformers 5.20 needs torch 2.6 or newer, so it cannot share envs/sam3d. The SAM 3 weights come
# from setup/weights.sh. torch 2.14.1+cu130 needs an NVIDIA driver for CUDA 13.0 (R580 or newer) at run time.
set -exo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[ -d envs/sam3seg ] || tools/micromamba create -y -p "$G/envs/sam3seg" -c conda-forge python=3.12 pip
"$G/envs/sam3seg/bin/pip" install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
"$G/envs/sam3seg/bin/pip" install transformers==5.20.0 scipy==1.18.1 pillow==12.3.0
"$G/envs/sam3seg/bin/python" -c "import torch, transformers, scipy, PIL
from transformers import Sam3Model, Sam3Processor
print('sam3seg env', torch.__version__, 'transformers', transformers.__version__)"
echo SETUP_OK
