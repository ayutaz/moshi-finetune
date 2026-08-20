#!/usr/bin/env bash

# Bring a fresh Vast.ai instance to the state run_baseline.sh expects:
# repository at the experiment branch, uv environment synced, and both published
# baseline checkpoints recovered at their pinned revisions with verified checksums.
#
# usage: bash bootstrap_instance.sh

set -euo pipefail

branch="experiment-j-moshi-character-voice-overfit"
repository_dir="/workspace/moshi-finetune"
artifact_root="/workspace/experiment-artifacts/baselines"

stage2_repo="ayousanz/moshi-persona-stage2-ojousama-2026-07-06"
stage2_revision="828b0d2b5a7e5262b137cc110d66000a2202cc39"
stage2_sha256="69a4a0112663695371a61d56372f605f549e0613e1e4f767294e2ab3811bc381"

stage3_repo="ayousanz/moshi-persona-stage3-ojousama-2026-07-06"
stage3_revision="224b3ce8408d013cad65c16da213d2f464cc3f90"
stage3_sha256="f34b52b7c2865cc6809e2a1c0ec527de025bdf66d3163e2c9b43ccd1d7c2c072"

mkdir -p /workspace/experiment-artifacts/logs
exec > >(tee /workspace/experiment-artifacts/logs/bootstrap.log) 2>&1

if ! command -v uv >/dev/null 2>&1; then
  if [[ ! -x "${HOME}/.local/bin/uv" ]]; then
    echo "installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
  fi
  export PATH="${HOME}/.local/bin:${PATH}"
fi
uv --version

if [[ ! -d "${repository_dir}/.git" ]]; then
  git clone --branch "${branch}" https://github.com/ayutaz/moshi-finetune.git "${repository_dir}"
fi
git -C "${repository_dir}" fetch origin "${branch}"
git -C "${repository_dir}" checkout "${branch}"
git -C "${repository_dir}" reset --hard "origin/${branch}"
echo "repository at $(git -C "${repository_dir}" rev-parse --short HEAD)"

cd "${repository_dir}"
uv sync --python 3.12

download_stage() {
  local stage="$1" repo="$2" revision="$3" expected="$4"
  local target="${artifact_root}/${stage}"
  mkdir -p "${target}"
  echo "downloading ${stage} from ${repo}@${revision}"
  uv run --no-sync python - "$repo" "$revision" "$target" <<'PY'
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

repo, revision, target = sys.argv[1], sys.argv[2], Path(sys.argv[3])
for name in ("model.safetensors", "moshi_lm_kwargs.json"):
    cached = hf_hub_download(repo, name, revision=revision)
    destination = target / name
    if not destination.exists():
        shutil.copyfile(cached, destination)
    print(f"{name}: {destination.stat().st_size} bytes")
PY
  local actual
  actual="$(sha256sum "${target}/model.safetensors" | cut -d' ' -f1)"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "${stage}: checksum mismatch" >&2
    echo "  expected ${expected}" >&2
    echo "  actual   ${actual}" >&2
    exit 1
  fi
  echo "${stage}: checksum verified"
}

download_stage stage2 "${stage2_repo}" "${stage2_revision}" "${stage2_sha256}"
download_stage stage3 "${stage3_repo}" "${stage3_revision}" "${stage3_sha256}"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
df -h /workspace | tail -1
echo "bootstrap complete"
