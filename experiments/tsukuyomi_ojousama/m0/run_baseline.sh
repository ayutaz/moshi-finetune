#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /workspace/experiment-artifacts/baseline-input/tsukuyomi-heldout" >&2
  exit 2
fi

prompt_source_dir="$1"
repository_dir="/workspace/moshi-finetune"
artifact_root="/workspace/experiment-artifacts/baselines"
shared_root="${artifact_root}/shared"
tokenizer_revision="f464b76739c884d8b0479a0a7705b7fa71c3fd5a"

mkdir -p "${shared_root}/logs"
exec > >(tee "${shared_root}/logs/run-baseline.log") 2>&1

git -C "${repository_dir}" pull --ff-only origin experiment-j-moshi-character-voice-overfit
cd "${repository_dir}"
uv sync --python 3.12

uv run --no-sync python -m tools.prepare_baseline_prompts audio \
  --input-dir "${prompt_source_dir}" \
  --output-dir "${shared_root}/prompt-audio" \
  --target-rate 24000 \
  --report "${shared_root}/prompt-audio-report.json"

uv run --no-sync python -m tools.tokenize_audio \
  --audio_dir "${shared_root}/prompt-audio" \
  --output_dir "${shared_root}/tokenized-audio" \
  --num_workers 2

uv run --no-sync python -m tools.prepare_baseline_prompts padding-text \
  --audio-token-dir "${shared_root}/tokenized-audio" \
  --output-dir "${shared_root}/tokenized-text" \
  --report "${shared_root}/padding-text-report.json"

uv run --no-sync python -m tools.prepare_dataset \
  --tokenized_text_dir "${shared_root}/tokenized-text" \
  --tokenized_audio_dir "${shared_root}/tokenized-audio" \
  --output_prefix "${shared_root}/prompt-dataset/heldout"

for stage in stage2 stage3; do
  CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -m tools.persona_perplexity \
    --model-dir "${artifact_root}/${stage}" \
    --pairs "experiments/tsukuyomi_ojousama/eval/persona-baseline-10.jsonl" \
    --output "${artifact_root}/${stage}/evaluation/persona-perplexity.json" \
    --tokenizer-revision "${tokenizer_revision}" \
    --device cuda:0 \
    --dtype bfloat16

  CUDA_VISIBLE_DEVICES=0 uv run --no-sync accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    generate.py \
    --output_dir "${artifact_root}/${stage}/evaluation/continuation" \
    --model_dir "${artifact_root}/${stage}" \
    --model_dtype bfloat16 \
    --eval_data_files "${shared_root}/prompt-dataset/heldout-*.parquet" \
    --moshi_speakers A \
    --dataset_processing_workers 1 \
    --num_examples 10 \
    --per_device_eval_batch_size 1 \
    --prompt_length 50 \
    --generation_length 125 \
    --temperature 0.8 \
    --top_k 0 \
    --top_p 0.0 \
    --seed 20260818

  CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -m tools.decode_tokens \
    --tokens_dir "${artifact_root}/${stage}/evaluation/continuation/generated_tokens" \
    --output_dir "${artifact_root}/${stage}/evaluation/continuation/generated_wavs" \
    --num_workers 1
done

find "${artifact_root}/stage2" "${artifact_root}/stage3" "${shared_root}" \
  -type f -print0 | sort -z | xargs -0 sha256sum > "${shared_root}/sha256sum.txt"

echo "M0 baseline generation complete"
