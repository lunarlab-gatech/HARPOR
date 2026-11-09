#!/bin/bash
# The model weights, about 30 GB, under $HARPOR_ROOT (by default ~/harpor), after the envs are built. Each download is
# skipped when its files are there. SAM 3 and SAM 3D Objects are gated: request access to facebook/sam3 and
# facebook/sam-3d-objects on Hugging Face, then set HF_TOKEN to a Read token. The script keeps the token in
# $HARPOR_ROOT/.cache/huggingface/token, where `hf auth login` would keep it, because the stages need it too.
#
#   SAM 3D Objects  facebook/sam-3d-objects               sam-3d-objects/checkpoints/hf (note 4)
#   SAM 3           facebook/sam3                         the Hugging Face cache (note 3)
#   DA3             depth-anything/DA3NESTED-GIANT-LARGE-1.1  the Hugging Face cache (note 4b)
#   VoMP v1.1       nvidia/PhysicalAI-Simulation-VoMP-Model   VoMP/weights/vomp_1_1 (note 9)
#
# MoGe and DINOv2 are not here: SAM 3D downloads them into the caches on its first run.
# In the docker image, sam-3d-objects/checkpoints, VoMP/weights and .cache are links into the data folder, so the
# weights land there and not in the image.
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
for e in sam3d sam3seg da3 vomp11; do
  [ -x "envs/$e/bin/python" ] || { echo "no $G/envs/$e: build the envs first (setup/install.sh)" >&2; exit 1; }
done
if [ -z "$HF_TOKEN" ] && [ ! -s "${HF_TOKEN_PATH:-$HF_HOME/token}" ]; then
  echo "no Hugging Face token: set HF_TOKEN to a Read token with access to facebook/sam3 and facebook/sam-3d-objects" >&2
  exit 1
fi
# The stages load the gated models from the cache, and each load still asks the hub, with the token. So the token is
# kept where the stages find it, in $HF_HOME/token (readable by you only), as `hf auth login` keeps it.
if [ ! -s "$HF_HOME/token" ]; then
  mkdir -p "$HF_HOME"
  (umask 077; if [ -n "$HF_TOKEN" ]; then printf %s "$HF_TOKEN" > "$HF_HOME/token"; else cat "$HF_TOKEN_PATH" > "$HF_HOME/token"; fi)
fi

echo "== SAM 3D Objects $(date +%T)"
if [ ! -s sam-3d-objects/checkpoints/hf/pipeline.yaml ]; then
  mkdir -p sam-3d-objects/checkpoints
  (cd sam-3d-objects && "$G/envs/sam3d/bin/python" -c "
from huggingface_hub import snapshot_download
snapshot_download('facebook/sam-3d-objects', local_dir='checkpoints/hf-download', max_workers=4)")
  rm -rf sam-3d-objects/checkpoints/hf
  mv sam-3d-objects/checkpoints/hf-download/checkpoints sam-3d-objects/checkpoints/hf
  rm -rf sam-3d-objects/checkpoints/hf-download
fi
ls sam-3d-objects/checkpoints/hf/pipeline.yaml

echo "== SAM 3 $(date +%T)"
"$G/envs/sam3seg/bin/python" -c "from huggingface_hub import snapshot_download as s; print(s('facebook/sam3'))"

echo "== DA3 $(date +%T)"
"$G/envs/da3/bin/python" -c "from huggingface_hub import snapshot_download as s; print(s('depth-anything/DA3NESTED-GIANT-LARGE-1.1'))"

echo "== VoMP $(date +%T)"
mkdir -p VoMP/weights
(cd VoMP && "$G/envs/vomp11/bin/python" -c "
from huggingface_hub import snapshot_download as s
print(s('nvidia/PhysicalAI-Simulation-VoMP-Model', allow_patterns=['vomp_1_1/*'], local_dir='weights'))")
ls VoMP/weights/vomp_1_1/inference.json
echo WEIGHTS_OK
