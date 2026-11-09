#!/bin/bash
# envs/web: the web app's server, the terminal command and the CPU stages (build_plan.md note 10), a minute, under
# $HARPOR_ROOT (by default ~/harpor). The versions are those of envs/web on ICE (2026-11-08).
set -exo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[ -d envs/web ] || tools/micromamba create -y -p "$G/envs/web" -c conda-forge python=3.12 pip
"$G/envs/web/bin/pip" install fastapi==0.143.2 uvicorn==0.54.0 av==19.0.1 rosbags==0.11.8 anthropic==1.15.0 \
  httpx==0.28.1 trimesh==5.1.1 numpy==2.5.4 scipy==1.18.1 pillow==12.3.0
"$G/envs/web/bin/python" -c "import fastapi, uvicorn, av, rosbags, anthropic, httpx, trimesh, numpy, scipy, PIL
print('web env', fastapi.__version__, 'av', av.__version__)"
echo SETUP_OK
