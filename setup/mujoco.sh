#!/bin/bash
# envs/mujoco, the Python of the MJCF, USD and URDF stages (build_plan.md notes 7 and 9), under $HARPOR_ROOT (by default
# ~/harpor).
# Rebuilt on 2026-11-01 from the installed versions in envs/mujoco on ICE (pip freeze). First run as a recipe on 2026-11-08.
# MuJoCo moved from 3.16.0 to 4.0.0 on 2026-11-11.
# It pins the packages that HARPOR imports, and pillow, which trimesh needs to read a GLB's textures. pip picks their
# dependencies, such as glfw and PyOpenGL for mujoco.viewer. mujoco 4.0.0 pins mujoco-core and mujoco-studio 4.0.0.
set -exo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[ -d envs/mujoco ] || tools/micromamba create -y -p "$G/envs/mujoco" -c conda-forge python=3.11 pip
# MuJoCo 4.0 split the package into mujoco, mujoco-core and mujoco-studio. The 4.0.0 release notes say to uninstall a
# 3.x mujoco first, because an upgrade in place can remove files that moved into mujoco-core.
if "$G/envs/mujoco/bin/python" -c "import mujoco, sys; sys.exit(int(mujoco.__version__.split('.')[0]) >= 4)" 2>/dev/null; then
  "$G/envs/mujoco/bin/pip" uninstall -y mujoco
fi
"$G/envs/mujoco/bin/pip" install mujoco==4.0.0 coacd==1.0.14 trimesh==5.1.1 usd-core==26.8 numpy==2.4.6 scipy==1.17.1 pillow==12.3.0
"$G/envs/mujoco/bin/python" -c "import coacd, mujoco, numpy, PIL, pxr, scipy, trimesh
print('mujoco', mujoco.__version__, 'trimesh', trimesh.__version__, 'numpy', numpy.__version__, 'scipy', scipy.__version__)"
# A free box on a floor, as the physics check settles a candidate (harpor.physics.supervise): it compiles and comes to
# rest on the floor without a warning.
"$G/envs/mujoco/bin/python" -c "import mujoco
m = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><geom type=\"plane\" size=\"1 1 0.05\"/>
<body pos=\"0 0 0.2\"><freejoint/><geom type=\"box\" size=\"0.05 0.05 0.05\" mass=\"0.5\"/></body>
</worldbody></mujoco>''')
d = mujoco.MjData(m)
mujoco.mj_step(m, d, 1500)
assert not any(w.number for w in d.warning), 'the settling box warns'
assert abs(d.qpos[2] - 0.05) < 2e-3, f'the box rests at z = {d.qpos[2]:.4f}, not on the floor'
print('free body: settles on the floor')"
echo SETUP_OK
