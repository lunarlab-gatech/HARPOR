# The shared start of the setup scripts, which source it. It clears a cluster's modules, sets G to the root that
# the app reads (HARPOR_ROOT, as in harpor/webapp/start_ice.sh), moves there and prepares the installers.

# A cluster's default modules can break the conda C++ compiler (build_plan.md note 4), so they go first. A machine
# without environment modules has no module command and nothing to clear.
if command -v module > /dev/null; then module purge; fi
G=${HARPOR_ROOT:-$(readlink -f ~/scratch)/gator}
mkdir -p "$G/tools" "$G/envs" "$G/.cache"
cd "$G"
# PIP_CONFIG_FILE=/dev/null skips the user's pip.conf. That file can add an extra index that does not resolve, and pip
# then retries the index on every package (build_plan.md note 7).
export MAMBA_ROOT_PREFIX=$G/micromamba PIP_CACHE_DIR=$G/.cache/pip HF_HOME=$G/.cache/huggingface PIP_CONFIG_FILE=/dev/null
# The micromamba of build_plan.md note 4, for a machine that has not run that note.
[ -x tools/micromamba ] || curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C tools --strip-components=1 bin/micromamba
