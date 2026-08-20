"""Objective intelligibility for the M2 TTS gate.

The gate asks for 27 of 30 unseen sentences to be clearly intelligible. Judging that by
ear is the final word, but a listening pass is expensive to repeat for every candidate
checkpoint, so this gives a cheap screen first: transcribe each rendered sentence and
measure how far the transcript drifts from the script.

Character error rate suits Japanese, where word segmentation is itself a modelling
choice. Reference and hypothesis are both reduced to reading-bearing characters, because
a recogniser does not emit the punctuation a reading script carries and the TTS should
not be charged for that.

The scoring helpers import nothing heavy, so they stay unit-testable without an ASR model.
"""

from __future__ import annotations

import argparse
import json
import statistics
import unicodedata
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_KEEP_CATEGORIES = ("L", "N")


def normalise_for_cer(text: str) -> str:
    """Reduce text to the characters that carry the reading."""
    normalised = unicodedata.normalize("NFKC", text)
    return "".join(char for char in normalised if unicodedata.category(char)[0] in _KEEP_CATEGORIES)


def _edit_distance(reference: str, hypothesis: str) -> int:
    if not hypothesis:
        return len(reference)
    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_char in enumerate(reference, start=1):
        current = [ref_index]
        for hyp_index, hyp_char in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[hyp_index] + 1,
                    current[hyp_index - 1] + 1,
                    previous[hyp_index - 1] + (ref_char != hyp_char),
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Edit distance between the readings, divided by the reference length."""
    ref = normalise_for_cer(reference)
    hyp = normalise_for_cer(hypothesis)
    if not ref:
        raise ValueError("reference must contain reading-bearing characters")
    return _edit_distance(ref, hyp) / len(ref)


def summarise_intelligibility(
    rows: Iterable[dict[str, Any]], *, threshold: float
) -> dict[str, Any]:
    """Count how many sentences fall within `threshold`."""
    rows = list(rows)
    if not rows:
        raise ValueError("at least one scored sentence is required")
    rates = [float(row["cer"]) for row in rows]
    failed = [row["id"] for row in rows if float(row["cer"]) > threshold]
    return {
        "total": len(rows),
        "intelligible": len(rows) - len(failed),
        "threshold": threshold,
        "failed_ids": failed,
        "mean_cer": statistics.fmean(rates),
        "median_cer": statistics.median(rates),
        "max_cer": max(rates),
    }


def _transcribe(audio_dir: Path, model_id: str, device: str) -> dict[str, str]:
    import torch
    from transformers import pipeline

    recogniser = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device=device,
        torch_dtype=torch.float32,
    )
    transcripts = {}
    for path in sorted(audio_dir.glob("*.wav")):
        result = recogniser(
            str(path), generate_kwargs={"language": "japanese", "task": "transcribe"}
        )
        transcripts[path.stem] = result["text"]
    return transcripts


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure TTS intelligibility by ASR + CER")
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--sentences", type=Path, required=True, help="JSONL with id and text")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model", default="openai/whisper-large-v3-turbo")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--required", type=int, default=27)
    parser.add_argument(
        "--transcripts",
        type=Path,
        help="reuse transcripts from a previous run instead of loading the ASR model",
    )
    args = parser.parse_args()

    sentences = {
        row["id"]: row["text"]
        for row in (
            json.loads(line)
            for line in args.sentences.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }

    if args.transcripts and args.transcripts.is_file():
        transcripts = json.loads(args.transcripts.read_text(encoding="utf-8"))
    else:
        transcripts = _transcribe(args.audio_dir, args.model, args.device)
        if args.transcripts:
            args.transcripts.parent.mkdir(parents=True, exist_ok=True)
            args.transcripts.write_text(
                json.dumps(transcripts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )

    rows = []
    for name, hypothesis in sorted(transcripts.items()):
        reference = sentences.get(name)
        if reference is None:
            continue
        rows.append(
            {
                "id": name,
                "reference": reference,
                "hypothesis": hypothesis,
                "cer": character_error_rate(reference, hypothesis),
            }
        )

    summary = summarise_intelligibility(rows, threshold=args.threshold)
    summary["required"] = args.required
    report = {
        "schema_version": 1,
        "audio_dir": str(args.audio_dir),
        "asr_model": args.model,
        "summary": summary,
        "sentences": rows,
        "status": "pass" if summary["intelligible"] >= args.required else "fail",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], **summary}, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
