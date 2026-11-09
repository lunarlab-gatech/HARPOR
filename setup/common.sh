# The shared start of the setup scripts, which source it. It clears a cluster's modules, sets G to the root that
# the app reads (HARPOR_ROOT, by default ~/harpor), moves there and prepares the installers.

# A cluster's default modules can break the conda C++ compiler (build_plan.md note 4), so they go first. A machine
# without environment modules has no module command and nothing to clear.
if command -v module > /dev/null; then module purge; fi
G=${HARPOR_ROOT:-$HOME/harpor}
mkdir -p "$G/tools" "$G/envs" "$G/.cache"
cd "$G"
# PIP_CONFIG_FILE=/dev/null skips the user's pip.conf. That file can add an extra index that does not resolve, and pip
# then retries the index on every package (build_plan.md note 7).
export MAMBA_ROOT_PREFIX=$G/micromamba PIP_CACHE_DIR=$G/.cache/pip HF_HOME=$G/.cache/huggingface PIP_CONFIG_FILE=/dev/null
# The micromamba of build_plan.md note 4, for a machine that has not run that note.
[ -x tools/micromamba ] || curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C tools --strip-components=1 bin/micromamba

# cuda_arch sets TORCH_CUDA_ARCH_LIST, the GPUs that the CUDA extensions are built for. A build for the wrong GPU does
# not run. The list comes from the environment, or else from the GPUs that this machine sees. Set it by hand for a
# machine without a GPU (a docker build) or for a cluster with mixed GPUs. torch 2.5.1 (envs/sam3d) and 2.4.0
# (envs/vomp11) build for nothing newer than 9.0, so a Blackwell GPU (10.0 or 12.0) stops the build here.
cuda_arch() {
  if [ -z "$TORCH_CUDA_ARCH_LIST" ]; then
    TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -u | paste -sd ';' -) \
      || { echo "nvidia-smi found no GPU: run this on a GPU machine, or set TORCH_CUDA_ARCH_LIST" >&2; exit 1; }
  fi
  local a
  for a in ${TORCH_CUDA_ARCH_LIST//;/ }; do
    if awk -v a="${a%+PTX}" 'BEGIN { exit !(a + 0 > 9.0) }'; then
      echo "TORCH_CUDA_ARCH_LIST has $a: HARPOR's torch versions build for 9.0 (Hopper) at most" >&2
      exit 1
    fi
  done
  export TORCH_CUDA_ARCH_LIST
}

# check_imports MODULE... imports each module with the python on PATH (the active env). A docker build has no NVIDIA
# driver, so a module that loads libcuda.so.1 on import is reported there instead of failing the build; every other
# import error stops the script.
check_imports() {
  python - "$@" <<'PY'
import importlib, sys
for m in sys.argv[1:]:
    try:
        importlib.import_module(m)
    except ImportError as e:
        if "libcuda.so" not in str(e):
            raise
        print(f"{m}: not imported, as this machine has no NVIDIA driver ({e})")
print("imports checked:", " ".join(sys.argv[1:]))
PY
}
