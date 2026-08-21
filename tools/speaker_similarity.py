"""Speaker embedding similarity between the target speaker and a rendered voice.

The plan asks for this measurement explicitly: extract the voiced regions, RMS-normalise,
then compare speaker embeddings. It also forbids deciding on the number alone, because a
flat, over-smoothed rendering can score higher than one a listener finds more like the
speaker. So this produces supporting evidence for the M2 listening pass, never a verdict.

Silence carries no speaker identity, and two systems that pad differently would otherwise
be scored on their padding, which is why the voiced-region step is not optional.

The signal helpers take plain sequences and import nothing heavy, so they stay
unit-testable without an embedding model.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import wave
from array import array
from collections.abc import Sequence
from pathlib import Path
from typing import Any

DEFAULT_FRAME = 480  # 10 ms at 48 kHz
DEFAULT_THRESHOLD = 500
DEFAULT_TARGET_RMS = 0.05


def voiced_segments(samples: Sequence[int], *, frame: int, threshold: int) -> list[int]:
    """Concatenate the frames whose RMS clears `threshold`."""
    kept: list[int] = []
    for start in range(0, len(samples), frame):
        window = samples[start : start + frame]
        if not window:
            continue
        energy = math.sqrt(sum(value * value for value in window) / len(window))
        if energy >= threshold:
            kept.extend(window)
    return kept


def rms_normalise(samples: Sequence[int], *, target_rms: float) -> list[float]:
    """Scale to a fixed RMS so loudness differences do not colour the comparison."""
    if not samples:
        raise ValueError("cannot normalise a silent signal")
    scale = math.sqrt(sum(value * value for value in samples) / len(samples))
    if scale == 0:
        raise ValueError("cannot normalise a silent signal")
    factor = target_rms * 32768.0 / scale
    return [value * factor / 32768.0 for value in samples]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("cannot compare a zero vector")
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


def paired_comparison(
    base: dict[str, float], candidate: dict[str, float], *, names: tuple[str, str]
) -> dict[str, Any]:
    """Compare two systems item by item on the same set of clips.

    Paired, not pooled: the two systems render the same sentences, so the per-sentence
    difference removes the sentence's own difficulty and is a far tighter measurement than
    two independent means.

    A key present on one side only is an error rather than something to intersect away.
    Quietly dropping it would compare the two systems on different sentences and still
    report the result as paired.
    """
    if not base or not candidate:
        raise ValueError("both systems need at least one scored clip")
    if set(base) != set(candidate):
        missing = sorted(set(base) ^ set(candidate))
        raise ValueError(f"{names[0]} and {names[1]} were scored on different clips: {missing[:5]}")

    from tools.build_listening_page import sign_test_p

    keys = sorted(base)
    deltas = [candidate[key] - base[key] for key in keys]
    wins = sum(1 for d in deltas if d > 0)
    losses = sum(1 for d in deltas if d < 0)
    return {
        "systems": list(names),
        "pairs": len(deltas),
        "higher_on": wins,
        "lower_on": losses,
        "ties": len(deltas) - wins - losses,
        "mean_delta": statistics.fmean(deltas),
        "median_delta": statistics.median(deltas),
        "min_delta": min(deltas),
        "max_delta": max(deltas),
        "sign_test_p_two_sided": sign_test_p(wins, losses),
        "per_file_delta": dict(zip(keys, deltas, strict=True)),
    }


def summarise_similarity(scores: dict[str, float]) -> dict[str, Any]:
    if not scores:
        raise ValueError("at least one score is required")
    values = list(scores.values())
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def to_int16_scale(samples: Sequence[float]) -> list[int]:
    """Bring unit-float samples into the int16 range the voiced threshold is written in.

    Integers pass through: a PCM_16 file is already in that range, while the corpus is
    96 kHz IEEE float and arrives as values in [-1, 1].
    """
    scaled = []
    for value in samples:
        if isinstance(value, int):
            scaled.append(value)
            continue
        scaled.append(max(-32768, min(32767, int(round(value * 32767)))))
    return scaled


def _read_wav(path: Path) -> tuple[list[int], int]:
    """Read a WAV as int16-scaled mono.

    soundfile handles the IEEE-float corpus files that the stdlib wave module rejects
    with "unknown format: 3"; wave remains the fallback for plain PCM.
    """
    try:
        import soundfile

        data, rate = soundfile.read(str(path), always_2d=True)
        return to_int16_scale([float(frame[0]) for frame in data]), int(rate)
    except ImportError:
        with wave.open(str(path)) as handle:
            frames = handle.readframes(handle.getnframes())
            channels = handle.getnchannels()
            samples = list(array("h", frames))
            if channels > 1:
                samples = samples[::channels]
            return samples, handle.getframerate()


def _prepare(path: Path, frame: int, threshold: int, target_rms: float):
    import numpy as np

    samples, rate = _read_wav(path)
    voiced = voiced_segments(samples, frame=frame, threshold=threshold)
    if not voiced:
        raise ValueError(f"{path}: no voiced frames above threshold {threshold}")
    return np.asarray(rms_normalise(voiced, target_rms=target_rms), dtype="float32"), rate


def _embed_all(
    paths: list[Path], model_id: str, device: str, frame: int, threshold: int, target_rms: float
) -> dict[str, list[float]]:
    import torch
    import torchaudio
    from speechbrain.inference.speaker import EncoderClassifier

    classifier = EncoderClassifier.from_hparams(source=model_id, run_opts={"device": device})
    embeddings = {}
    for path in paths:
        signal, rate = _prepare(path, frame, threshold, target_rms)
        wav = torch.from_numpy(signal).unsqueeze(0)
        if rate != 16000:
            wav = torchaudio.functional.resample(wav, rate, 16000)
        with torch.no_grad():
            vector = classifier.encode_batch(wav.to(device)).squeeze().cpu().tolist()
        embeddings[path.stem] = vector
    return embeddings


def main() -> int:
    parser = argparse.ArgumentParser(description="Speaker similarity against the target speaker")
    parser.add_argument("--target-dir", type=Path, required=True, help="natural recordings")
    parser.add_argument(
        "--system", action="append", nargs=2, metavar=("NAME", "DIR"), required=True
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model", default="speechbrain/spkrec-ecapa-voxceleb")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frame", type=int, default=DEFAULT_FRAME)
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD)
    parser.add_argument("--target-rms", type=float, default=DEFAULT_TARGET_RMS)
    args = parser.parse_args()

    import numpy as np

    target_paths = sorted(args.target_dir.glob("*.wav"))
    target_embeddings = _embed_all(
        target_paths, args.model, args.device, args.frame, args.threshold, args.target_rms
    )
    centroid = np.mean(np.asarray(list(target_embeddings.values())), axis=0).tolist()

    systems = {}
    for name, directory in args.system:
        paths = sorted(Path(directory).glob("*.wav"))
        embeddings = _embed_all(
            paths, args.model, args.device, args.frame, args.threshold, args.target_rms
        )
        scores = {key: cosine_similarity(vector, centroid) for key, vector in embeddings.items()}
        systems[name] = {"per_file": scores, "summary": summarise_similarity(scores)}

    report = {
        "schema_version": 1,
        "model": args.model,
        "target_dir": str(args.target_dir),
        "target_clips": len(target_paths),
        "preparation": {
            "voiced_frame_samples": args.frame,
            "voiced_rms_threshold": args.threshold,
            "target_rms": args.target_rms,
        },
        "systems": systems,
        "interpretation": (
            "Supporting evidence only. The plan forbids accepting a system on speaker "
            "similarity alone, because a flat rendering can outscore one a listener judges "
            "closer to the speaker."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for name, payload in systems.items():
        print(f"{name}: {json.dumps(payload['summary'], sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
