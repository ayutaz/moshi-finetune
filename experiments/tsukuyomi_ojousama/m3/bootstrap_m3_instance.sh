#!/usr/bin/env bash

# Bring a fresh 2x A100 80GB instance to the state the M3 voice-control runs expect.
#
# Only j-moshi-ext is downloaded here. The datasets were built, tokenised and turned into
# parquet on the laptop, so no audio ever reaches this machine - which keeps the tsukuyomi
# corpus off hardware the experiment does not own, and takes the cheapest part of the
# pipeline off the most expensive place to run it.
#
# usage: bash bootstrap_m3_instance.sh

set -euo pipefail

branch="experiment-j-moshi-character-voice-overfit"
repo_url="https://github.com/ayutaz/moshi-finetune.git"
repo_dir="/workspace/moshi-finetune"
base_repo="nu-dialogue/j-moshi-ext"

mkdir -p /workspace/logs
exec > >(tee /workspace/logs/bootstrap-m3.log) 2>&1

echo "=== host ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
awk '/MemAvailable/ {printf "MemAvailable %.1f GiB\n", $2/1048576}' /proc/meminfo
df -h /workspace 2>/dev/null | tail -1 || df -h / | tail -1

# Refuse rather than discover this at the training launch: ZeRO-3 + fp16 + AdamW needs
# 16 bytes/param resident, 133.94 GB for the 8.37B dep_q=16 model, and one 80 GB card is
# 48 GB short before any activation memory.
gpus=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | wc -l)
smallest=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -1)
[ "$gpus" -ge 2 ] || { echo "FATAL: need 2 GPUs, found $gpus"; exit 1; }
[ "$smallest" -ge 80000 ] || { echo "FATAL: need 80 GB cards, smallest is ${smallest} MiB"; exit 1; }
avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
# finetune.py loads the model on CPU in float32 PER RANK before accelerate partitions
# anything: 2 x 33.49 GB. Too little host RAM OOM-kills at load and no GPU can help.
[ "$avail" -ge 80 ] || { echo "FATAL: need 80 GiB available RAM, have ${avail}"; exit 1; }
echo "host checks passed: ${gpus} GPUs, ${smallest} MiB each, ${avail} GiB RAM available"

if ! command -v uv >/dev/null 2>&1; then
  [ -x "${HOME}/.local/bin/uv" ] || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi
uv --version

[ -d "${repo_dir}/.git" ] || git clone --branch "${branch}" "${repo_url}" "${repo_dir}"
git -C "${repo_dir}" fetch --all
git -C "${repo_dir}" checkout "${branch}"
git -C "${repo_dir}" pull --ff-only origin "${branch}"
echo "repo at $(git -C "${repo_dir}" rev-parse HEAD)"

cd "${repo_dir}"
uv sync --python 3.12

uv run --no-sync python - "${base_repo}" <<'PY'
import sys
from huggingface_hub import snapshot_download
path = snapshot_download(sys.argv[1])
print(f"j-moshi-ext at {path}")
PY

df -h /workspace 2>/dev/null | tail -1 || df -h / | tail -1
echo "bootstrap complete"
