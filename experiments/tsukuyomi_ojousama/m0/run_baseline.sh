#!/usr/bin/env bash

set -euo pipefail

# The uv installer puts the binary in ~/.local/bin, which a non-interactive shell
# (nohup, ssh command form) does not have on PATH.
if ! command -v uv >/dev/null 2>&1 && [[ -x "${HOME}/.local/bin/uv" ]]; then
  export PATH="${HOME}/.local/bin:${PATH}"
fi

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /workspace/experiment-artifacts/baseline-input/tsukuyomi-heldout" >&2
  exit 2
fi

prompt_source_dir="$1"
repository_dir="/workspace/moshi-finetune"
artifact_root="/workspace/experiment-artifacts/baselines"
shared_root="${artifact_root}/shared"
tokenizer_revision="f464b76739c884d8b0479a0a7705b7fa71c3fd5a"
prompt_count=10
# Held-out speech is as short as 48 Mimi frames (VOICEACTRESS100_026, 3.802 s). At
# prompt_length=50 that reaches min_length with zero margin, and generate.py drops shorter
# examples without logging, so the prompt is taken from the first 40 frames instead.
prompt_length=40
generation_length=125
# The published checkpoints generate only the Moshi stream, so the user stream is
# teacher-forced from the example; each prompt is padded with silence to cover every
# generated frame, plus 2 frames for the delay pattern.
required_frames=$((prompt_length + generation_length))
audio_context_stem=VOICEACTRESS100_032
# The training text stream is mostly text_padding_id with tokens at their word-timestamp
# frames, so the scored pair is laid out the same way instead of densely from frame 0.
persona_start_frame=12
persona_end_of_text_padding_id=0

mkdir -p "${shared_root}/logs"
exec > >(tee "${shared_root}/logs/run-baseline.log") 2>&1

git -C "${repository_dir}" pull --ff-only origin experiment-j-moshi-character-voice-overfit
cd "${repository_dir}"
uv sync --python 3.12

uv run --no-sync python -m tools.prepare_baseline_prompts audio \
  --input-dir "${prompt_source_dir}" \
  --output-dir "${shared_root}/prompt-audio" \
  --target-rate 24000 \
  --min-frames "$((required_frames + 2))" \
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

# Gate before spending GPU time: every prompt must survive generate.py's silent length filter,
# and example_id -> dialogue_id must be recorded because generate.py drops that column.
uv run --no-sync python -m tools.prepare_baseline_prompts verify-dataset \
  --parquet-glob "${shared_root}/prompt-dataset/heldout-*.parquet" \
  --expected-count "${prompt_count}" \
  --min-frames "${required_frames}" \
  --report "${shared_root}/prompt-dataset-report.json"

# The persona metric and the generation baseline are independent halves of M0, so a
# failure in one must not stop the other. Failures are collected and reported at the end.
failures=()

for stage in stage2 stage3; do
  if ! CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -m tools.persona_perplexity \
    --model-dir "${artifact_root}/${stage}" \
    --pairs "experiments/tsukuyomi_ojousama/eval/persona-baseline-10.jsonl" \
    --output "${artifact_root}/${stage}/evaluation/persona-perplexity.json" \
    --tokenizer-revision "${tokenizer_revision}" \
    --audio-context "${shared_root}/tokenized-audio/${audio_context_stem}.npz" \
    --start-frame "${persona_start_frame}" \
    --end-of-text-padding-id "${persona_end_of_text_padding_id}" \
    --device cuda:0 \
    --dtype bfloat16
  then
    echo "${stage}: persona perplexity failed" >&2
    failures+=("${stage}:persona-perplexity")
  fi

  if ! CUDA_VISIBLE_DEVICES=0 uv run --no-sync accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    generate.py \
    --output_dir "${artifact_root}/${stage}/evaluation/continuation" \
    --model_dir "${artifact_root}/${stage}" \
    --model_dtype bfloat16 \
    --eval_data_files "${shared_root}/prompt-dataset/heldout-*.parquet" \
    --moshi_speakers A \
    --dataset_processing_workers 1 \
    --num_examples "${prompt_count}" \
    --per_device_eval_batch_size 1 \
    --prompt_length "${prompt_length}" \
    --generation_length "${generation_length}" \
    --temperature 0.8 \
    --top_k 0 \
    --top_p 0.0 \
    --seed 20260818
  then
    echo "${stage}: generation failed" >&2
    failures+=("${stage}:generation")
    continue
  fi

  if ! CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -m tools.decode_tokens \
    --tokens_dir "${artifact_root}/${stage}/evaluation/continuation/generated_tokens" \
    --output_dir "${artifact_root}/${stage}/evaluation/continuation/generated_wavs" \
    --num_workers 1
  then
    echo "${stage}: decoding failed" >&2
    failures+=("${stage}:decode")
    continue
  fi

  for kind in "generated_tokens:npy" "generated_wavs:wav"; do
    subdir="${kind%%:*}"
    extension="${kind##*:}"
    produced=$(find "${artifact_root}/${stage}/evaluation/continuation/${subdir}" \
      -type f -name "*.${extension}" | wc -l | tr -d ' ')
    if [[ "${produced}" -ne "${prompt_count}" ]]; then
      echo "${stage}: expected ${prompt_count} ${subdir}, got ${produced}" >&2
      failures+=("${stage}:${subdir}-count")
    fi
  done
done

find "${artifact_root}/stage2" "${artifact_root}/stage3" "${shared_root}" \
  -type f -print0 | sort -z | xargs -0 sha256sum > "${shared_root}/sha256sum.txt"

if [[ ${#failures[@]} -gt 0 ]]; then
  echo "M0 baseline incomplete: ${failures[*]}" >&2
  exit 1
fi

echo "M0 baseline generation complete"
