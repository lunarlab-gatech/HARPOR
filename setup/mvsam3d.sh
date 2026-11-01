#!/bin/bash
# MV-SAM3D and Depth Anything 3 (build_plan.md note 4b), about 10 min, under $HARPOR_ROOT (by default
# ~/scratch/gator). Build envs/sam3d and the SAM 3D weights first (note 4). MV-SAM3D reuses both: its requirement
# files are byte-identical, and its code goes first on sys.path. DA3 gets its own venv, because its unpinned
# dependencies could change torch in envs/sam3d, and it runs as its own step.
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[ -x envs/sam3d/bin/python ] || { echo "no $G/envs/sam3d: build it first (build_plan.md note 4)" >&2; exit 1; }
echo "== clones $(date +%T)"
[ -d MV-SAM3D ] || git clone -q https://github.com/devinli123/MV-SAM3D.git
git -C MV-SAM3D checkout -q abb04b5e8af5bc33b0265bdf19937e76bbb6bcdd
[ -d Depth-Anything-3 ] || git clone -q https://github.com/ByteDance-Seed/Depth-Anything-3.git
git -C Depth-Anything-3 checkout -q 3d835ec1a5802d64a8b8b15f817a1ab54809bfe4
git -C MV-SAM3D log -1 --format="MV-SAM3D %h %ad" --date=short; git -C Depth-Anything-3 log -1 --format="DA3 %h %ad" --date=short
echo "== DA3 venv $(date +%T)"
[ -d envs/da3 ] || "$G/envs/sam3d/bin/python" -m venv "$G/envs/da3"
"$G/envs/da3/bin/pip" install -q --upgrade pip
"$G/envs/da3/bin/pip" install -q -e Depth-Anything-3
"$G/envs/da3/bin/python" -c "import torch, numpy, depth_anything_3; print('da3 env', torch.__version__, torch.version.cuda, numpy.__version__, torch.cuda.is_available())"
echo "== DA3 weights $(date +%T)"
"$G/envs/da3/bin/python" -c "
from huggingface_hub import snapshot_download
p = snapshot_download('depth-anything/DA3NESTED-GIANT-LARGE-1.1'); print(p)"
echo "== done $(date +%T)"
