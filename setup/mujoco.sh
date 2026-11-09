#!/bin/bash
# envs/mujoco, the Python of the MJCF, USD and URDF stages (build_plan.md notes 7 and 9), under $HARPOR_ROOT (by default
# ~/harpor).
# Rebuilt on 2026-11-01 from the installed versions in envs/mujoco on ICE (pip freeze). First run as a recipe on 2026-11-08.
# It pins the packages that HARPOR imports, and pillow, which trimesh needs to read a GLB's textures. pip picks their
# dependencies, such as glfw and PyOpenGL for mujoco.viewer.
set -exo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[ -d envs/mujoco ] || tools/micromamba create -y -p "$G/envs/mujoco" -c conda-forge python=3.11 pip
"$G/envs/mujoco/bin/pip" install mujoco==3.16.0 coacd==1.0.14 trimesh==5.1.1 usd-core==26.8 numpy==2.4.6 scipy==1.17.1 pillow==12.3.0
"$G/envs/mujoco/bin/python" -c "import coacd, mujoco, numpy, PIL, pxr, scipy, trimesh
print('mujoco', mujoco.__version__, 'trimesh', trimesh.__version__, 'numpy', numpy.__version__, 'scipy', scipy.__version__)"
# A small soft part with the options of a model with one (harpor.physics.mjcf), which another MuJoCo version can
# change: it compiles, keeps the discrete integrator and the CG solver, and steps without a warning.
"$G/envs/mujoco/bin/python" -c "import mujoco
m = mujoco.MjModel.from_xml_string('''<mujoco><option integrator=\"discrete\" solver=\"CG\"/><worldbody><body name=\"base\">
<flexcomp name=\"pad\" type=\"grid\" count=\"3 3 3\" spacing=\"0.05 0.05 0.05\" dim=\"3\" dof=\"full\" radius=\"0.002\"
mass=\"0.1\"><edge damping=\"0.01\"/><elasticity young=\"3e4\" poisson=\"0.3\"/><pin id=\"0 1 2\"/></flexcomp>
</body></worldbody></mujoco>''')
d = mujoco.MjData(m)
mujoco.mj_step(m, d, 100)
assert (m.opt.integrator, m.opt.solver) == (mujoco.mjtIntegrator.mjINT_DISCRETE, mujoco.mjtSolver.mjSOL_CG)
assert not any(w.number for w in d.warning), 'the soft part warns'
print('flexcomp: discrete integrator, CG solver')"
echo SETUP_OK
