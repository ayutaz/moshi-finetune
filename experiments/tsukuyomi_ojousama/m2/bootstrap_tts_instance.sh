#!/usr/bin/env bash

# Bring a fresh CUDA instance to the state the M2 TTS runs expect: Irodori-TTS at a
# pinned commit, its CUDA environment, and the released base checkpoint. The digest of
# those weights is recorded in reports/m2-run-manifest.json.
#
# The tsukuyomi corpus itself is never uploaded to a hub. The precomputed DACVAE latents
# are copied to the instance separately; they are a derived representation, not the audio.
#
# usage: bash bootstrap_tts_instance.sh

set -euo pipefail

irodori_commit="eaf74d6a1913"
irodori_dir="/workspace/Irodori-TTS"
base_repo="Aratako/Irodori-TTS-500M-v3"

mkdir -p /workspace/logs
exec > >(tee /workspace/logs/bootstrap-tts.log) 2>&1

if ! command -v uv >/dev/null 2>&1; then
  if [[ ! -x "${HOME}/.local/bin/uv" ]]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  fi
  export PATH="${HOME}/.local/bin:${PATH}"
fi
uv --version

if [[ ! -d "${irodori_dir}/.git" ]]; then
  git clone https://github.com/Aratako/Irodori-TTS.git "${irodori_dir}"
fi
git -C "${irodori_dir}" fetch --all
git -C "${irodori_dir}" checkout "${irodori_commit}"
echo "Irodori-TTS at $(git -C "${irodori_dir}" rev-parse --short HEAD)"

cd "${irodori_dir}"
uv sync --extra cu128

# The base weights and the codec are public, so they come straight from the hub.
uv run --no-sync python - "${base_repo}" <<'PY'
import sys

from huggingface_hub import hf_hub_download, snapshot_download

repo = sys.argv[1]
weights = hf_hub_download(repo, "model.safetensors")
print(f"base weights: {weights}")
snapshot_download("Aratako/Semantic-DACVAE-Japanese-32dim")
print("codec cached")
PY

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
df -h /workspace 2>/dev/null | tail -1 || df -h / | tail -1
echo "bootstrap complete"
echo "next: copy the precomputed latents and train_manifest.jsonl, then run train.py"
