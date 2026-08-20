"""Objective measurements for the M2 TTS gate.

The gate asks whether the 30 unseen sentences come out without dropouts or clipping.
Irodori-TTS writes PCM_16 through soundfile, which saturates any float sample past
+/-1.0, so a handful of isolated saturated samples says the write overflowed rather than
that the model distorted. Run length is what separates the two, and the report keeps
both numbers so the distinction stays visible.

The measurement helpers take plain sequences of ints and import nothing heavy, so they
stay unit-testable without torch or soundfile.
"""

from __future__ import annotations

import argparse
import json
import math
import wave
from array import array
from pathlib import Path
from typing import Any, Sequence

DEFAULT_FULL_SCALE = 32767
SILENCE_FLOOR = 200
SILENCE_RATIO = 0.02


def clipped_run_lengths(samples: Sequence[int], *, full_scale: int) -> list[int]:
    """Lengths of each consecutive stretch of samples pinned to either rail."""
    runs: list[int] = []
    run = 0
    for value in samples:
        if value >= full_scale or value <= -full_scale - 1:
            run += 1
        elif run:
            runs.append(run)
            run = 0
    if run:
        runs.append(run)
    return runs


def headroom_gain(*, peak: int, full_scale: int, headroom_db: float) -> float:
    """Gain that brings `peak` down to `headroom_db` below full scale, or 1.0 if already under."""
    if peak <= 0:
        raise ValueError("peak must be positive to compute a gain")
    target = full_scale * (10 ** (-headroom_db / 20))
    if peak <= target:
        return 1.0
    return target / peak


def loud_bounds(samples: Sequence[int], *, threshold: int) -> tuple[int, int] | None:
    """Index of the first and last sample above `threshold`, or None when all are below."""
    first = last = None
    for index, value in enumerate(samples):
        if abs(value) > threshold:
            if first is None:
                first = index
            last = index
    if first is None:
        return None
    return first, last


def summarise_clip(
    samples: Sequence[int],
    *,
    sample_rate: int,
    full_scale: int = DEFAULT_FULL_SCALE,
) -> dict[str, Any]:
    """Measure one rendered utterance."""
    count = len(samples)
    peak = max((abs(value) for value in samples), default=0)
    rms = math.sqrt(sum(value * value for value in samples) / count) if count else 0.0
    runs = clipped_run_lengths(samples, full_scale=full_scale)
    bounds = loud_bounds(samples, threshold=max(SILENCE_FLOOR, int(peak * SILENCE_RATIO)))

    summary: dict[str, Any] = {
        "seconds": count / sample_rate if sample_rate else 0.0,
        "sample_rate": sample_rate,
        "peak": peak,
        "rms": rms,
        "clipped_samples": sum(runs),
        "clipped_runs": len(runs),
        "longest_clipped_run": max(runs, default=0),
        "silent": bounds is None,
    }
    if bounds is None:
        summary["leading_silence_seconds"] = summary["seconds"]
        summary["trailing_silence_seconds"] = summary["seconds"]
    else:
        first, last = bounds
        summary["leading_silence_seconds"] = first / sample_rate
        summary["trailing_silence_seconds"] = (count - 1 - last) / sample_rate
    return summary


def _read_wav(path: Path) -> tuple[list[int], int]:
    with wave.open(str(path)) as handle:
        if handle.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16-bit PCM, got {handle.getsampwidth() * 8}-bit")
        frames = handle.readframes(handle.getnframes())
        return list(array("h", frames)), handle.getframerate()


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure rendered TTS audio for the M2 gate")
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--max-clipped-run",
        type=int,
        default=2,
        help="longest run of full-scale samples treated as write-time saturation",
    )
    args = parser.parse_args()

    paths = sorted(args.audio_dir.glob("*.wav"))
    files: dict[str, Any] = {}
    for path in paths:
        samples, rate = _read_wav(path)
        files[path.stem] = summarise_clip(samples, sample_rate=rate)

    silent = sorted(name for name, row in files.items() if row["silent"])
    distorted = sorted(
        name for name, row in files.items() if row["longest_clipped_run"] > args.max_clipped_run
    )
    saturated = sorted(name for name, row in files.items() if row["clipped_samples"])

    report = {
        "schema_version": 1,
        "audio_dir": str(args.audio_dir),
        "file_count": len(files),
        "expected_count": args.expected_count,
        "max_clipped_run_allowed": args.max_clipped_run,
        "silent_files": silent,
        "files_with_sustained_clipping": distorted,
        "files_with_saturated_samples": saturated,
        "total_saturated_samples": sum(row["clipped_samples"] for row in files.values()),
        "longest_clipped_run": max(
            (row["longest_clipped_run"] for row in files.values()), default=0
        ),
        "files": files,
    }
    report["status"] = (
        "pass" if len(files) == args.expected_count and not silent and not distorted else "fail"
    )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "status",
                    "file_count",
                    "silent_files",
                    "files_with_sustained_clipping",
                    "total_saturated_samples",
                    "longest_clipped_run",
                )
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
