#!/bin/bash
# envs/mujoco, the Python of the MJCF, USD and URDF stages (build_plan.md notes 7 and 9), under $HARPOR_ROOT (by default
# ~/scratch/gator).
# Rebuilt on 2026-11-01 from the installed versions in envs/mujoco on ICE (pip freeze). It has not been run as a recipe.
# It pins the packages that HARPOR imports, and pillow, which trimesh needs to read a GLB's textures. pip picks their
# dependencies, such as glfw and PyOpenGL for mujoco.viewer.
set -exo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[ -d envs/mujoco ] || tools/micromamba create -y -p "$G/envs/mujoco" -c conda-forge python=3.11 pip
"$G/envs/mujoco/bin/pip" install mujoco==3.16.0 coacd==1.0.14 trimesh==5.1.1 usd-core==26.8 numpy==2.4.6 scipy==1.17.1 pillow==12.3.0
"$G/envs/mujoco/bin/python" -c "import coacd, mujoco, numpy, PIL, pxr, scipy, trimesh
print('mujoco', mujoco.__version__, 'trimesh', trimesh.__version__, 'numpy', numpy.__version__, 'scipy', scipy.__version__)"
echo SETUP_OK
