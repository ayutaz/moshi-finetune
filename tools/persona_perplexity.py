from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


class PairValidationError(ValueError):
    """Raised when the fixed ten-pair persona evaluation is invalid."""


def validate_pairs(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    if len(rows) != 10:
        raise PairValidationError("persona baseline requires exactly 10 pairs")
    validated = []
    identifiers: set[str] = set()
    for row_number, row in enumerate(rows, start=1):
        missing = [
            field
            for field in ("id", "prompt", "preferred", "dispreferred")
            if not isinstance(row.get(field), str) or not row[field].strip()
        ]
        if missing:
            raise PairValidationError(
                f"pair {row_number} is missing non-empty fields: {', '.join(missing)}"
            )
        if row["id"] in identifiers:
            raise PairValidationError(f"duplicate pair id: {row['id']}")
        identifiers.add(row["id"])
        if row["preferred"].strip() == row["dispreferred"].strip():
            raise PairValidationError(f"{row['id']}: preferred and dispreferred must differ")
        validated.append({field: row[field].strip() for field in ("id", "prompt", "preferred", "dispreferred")})
    return validated


def summarise_scores(rows: list[dict[str, float]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("at least one score is required")
    preferred_nll = [float(row["preferred_nll"]) for row in rows]
    dispreferred_nll = [float(row["dispreferred_nll"]) for row in rows]
    if all("preferred_logprob" in row and "dispreferred_logprob" in row for row in rows):
        preferred_logprob = [float(row["preferred_logprob"]) for row in rows]
        dispreferred_logprob = [float(row["dispreferred_logprob"]) for row in rows]
        preferred_wins = sum(
            left > right for left, right in zip(preferred_logprob, dispreferred_logprob)
        )
        ties = sum(left == right for left, right in zip(preferred_logprob, dispreferred_logprob))
    else:
        preferred_logprob = []
        dispreferred_logprob = []
        preferred_wins = sum(left < right for left, right in zip(preferred_nll, dispreferred_nll))
        ties = sum(left == right for left, right in zip(preferred_nll, dispreferred_nll))
    mean_preferred = sum(preferred_nll) / len(rows)
    mean_dispreferred = sum(dispreferred_nll) / len(rows)
    summary = {
        "pair_count": len(rows),
        "preferred_wins": preferred_wins,
        "ties": ties,
        "preferred_win_rate": preferred_wins / len(rows),
        "mean_nll_margin": mean_dispreferred - mean_preferred,
        "preferred_mean_nll": mean_preferred,
        "dispreferred_mean_nll": mean_dispreferred,
        "preferred_perplexity": math.exp(mean_preferred),
        "dispreferred_perplexity": math.exp(mean_dispreferred),
    }
    if preferred_logprob:
        summary.update(
            {
                "preferred_logprob_total": sum(preferred_logprob),
                "dispreferred_logprob_total": sum(dispreferred_logprob),
                "logprob_total_difference": sum(preferred_logprob) - sum(dispreferred_logprob),
            }
        )
    return summary


def _score_completion(
    model: Any,
    *,
    context_ids: list[int],
    completion_ids: list[int],
    device: Any,
) -> dict[str, float]:
    import torch
    import torch.nn.functional as functional

    if not context_ids or not completion_ids:
        raise PairValidationError("tokenizer produced an empty context or completion")
    text_tokens = [model.text_initial_token_id, *context_ids, *completion_ids]
    text_tensor = torch.tensor(text_tokens, dtype=torch.long, device=device).unsqueeze(0)
    audio_tensor = torch.full(
        (1, model.num_audio_codebooks, len(text_tokens)),
        model.zero_token_id,
        dtype=torch.long,
        device=device,
    )
    input_ids = torch.cat((text_tensor.unsqueeze(1), audio_tensor), dim=1)

    with torch.inference_mode():
        hidden = model.text_emb(input_ids[:, 0])
        for codebook_index in range(model.num_audio_codebooks):
            hidden = hidden + model.emb[codebook_index](
                input_ids[:, model.audio_offset + codebook_index]
            )
        hidden = model.transformer(hidden)
        if model.out_norm:
            hidden = model.out_norm(hidden)
        logits = model.text_linear(hidden).float()[..., :-1, :]
        targets = text_tensor[..., 1:]

        completion_start = len(context_ids)
        completion_stop = completion_start + len(completion_ids)
        completion_logits = logits[:, completion_start:completion_stop]
        completion_targets = targets[:, completion_start:completion_stop]
        token_losses = functional.cross_entropy(
            completion_logits.reshape(-1, model.text_card),
            completion_targets.reshape(-1),
            reduction="none",
        )
    return {
        "mean_nll": float(token_losses.mean().item()),
        "total_logprob": float(-token_losses.sum().item()),
    }


def score_pairs(
    *,
    model_dir: Path,
    pairs: list[dict[str, str]],
    tokenizer_path: Path,
    device_name: str,
    dtype_name: str,
) -> list[dict[str, Any]]:
    import torch
    from sentencepiece import SentencePieceProcessor

    from models import MoshiForFinetuning

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[
        dtype_name
    ]
    device = torch.device(device_name)
    tokenizer = SentencePieceProcessor(str(tokenizer_path))
    model = MoshiForFinetuning.from_pretrained(
        str(model_dir), device=device, dtype=dtype
    ).eval()

    scores = []
    for pair in pairs:
        context_ids = tokenizer.encode(pair["prompt"], out_type=int)
        preferred_ids = tokenizer.encode(pair["preferred"], out_type=int)
        dispreferred_ids = tokenizer.encode(pair["dispreferred"], out_type=int)
        preferred_score = _score_completion(
            model,
            context_ids=context_ids,
            completion_ids=preferred_ids,
            device=device,
        )
        dispreferred_score = _score_completion(
            model,
            context_ids=context_ids,
            completion_ids=dispreferred_ids,
            device=device,
        )
        scores.append(
            {
                **pair,
                "preferred_token_count": len(preferred_ids),
                "dispreferred_token_count": len(dispreferred_ids),
                "preferred_nll": preferred_score["mean_nll"],
                "dispreferred_nll": dispreferred_score["mean_nll"],
                "preferred_logprob": preferred_score["total_logprob"],
                "dispreferred_logprob": dispreferred_score["total_logprob"],
                "logprob_difference": (
                    preferred_score["total_logprob"] - dispreferred_score["total_logprob"]
                ),
                "preferred_wins": (
                    preferred_score["total_logprob"] > dispreferred_score["total_logprob"]
                ),
            }
        )
    return scores


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run fixed 10-pair persona perplexity evaluation")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer-repo", default="rinna/japanese-gpt2-medium")
    parser.add_argument("--tokenizer-name", default="spiece.model")
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16"
    )
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download

    pairs = validate_pairs(_read_jsonl(args.pairs))
    tokenizer_path = Path(
        hf_hub_download(
            args.tokenizer_repo,
            args.tokenizer_name,
            revision=args.tokenizer_revision,
        )
    )
    scores = score_pairs(
        model_dir=args.model_dir,
        pairs=pairs,
        tokenizer_path=tokenizer_path,
        device_name=args.device,
        dtype_name=args.dtype,
    )
    report = {
        "schema_version": 1,
        "model_dir": str(args.model_dir),
        "tokenizer_repo": args.tokenizer_repo,
        "tokenizer_name": args.tokenizer_name,
        "tokenizer_revision": args.tokenizer_revision,
        "dtype": args.dtype,
        "scoring": (
            "completion total log probability conditioned on text prompt with audio streams fixed "
            "to zero_token_id; mean token NLL is also reported"
        ),
        "summary": summarise_scores(scores),
        "pairs": scores,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
