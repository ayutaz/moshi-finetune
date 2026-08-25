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
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

DEFAULT_FRAME = 480  # 10 ms at 48 kHz
DEFAULT_THRESHOLD = 500
DEFAULT_TARGET_RMS = 0.05

#: What a similarity summary has to carry before it can be called a calibration band.
CALIBRATION_BAND_FIELDS = ("count", "mean", "median", "min", "max", "floor")

#: The three things a condition-4 report must publish. M3 published none of them: it
#: reported paired deltas with no absolute cosine, no per-clip spread, and no idea what
#: number a real human scores against the same centroid.
REQUIRED_COMPARISON_SECTIONS = ("calibration_band", "absolute_cosine", "comparisons")


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


def centroid(vectors: Sequence[Sequence[float]]) -> list[float]:
    """Element-wise mean of a set of embeddings.

    Written out rather than deferred to numpy so the calibration band stays testable in a
    suite that must run without torch.
    """
    rows = [list(vector) for vector in vectors]
    if not rows:
        raise ValueError("a centroid needs at least one vector")
    width = len(rows[0])
    if width == 0:
        raise ValueError("a centroid needs vectors with at least one dimension")
    if any(len(row) != width for row in rows):
        raise ValueError("embeddings of different widths cannot be averaged")
    return [sum(row[i] for row in rows) / len(rows) for i in range(width)]


def leave_one_out_similarity(embeddings: Mapping[str, Sequence[float]]) -> dict[str, float]:
    """Score each recording of one speaker against the centroid of that speaker's others.

    This is the calibration band, and without it every other number on this page is
    unreadable. CLAUDE.md states the rule the hard way: a within-group similarity of 0.74
    means nothing until you know what one real human scores. Condition 4 was decided in M3
    with no band at all, so "+0.032 better than control" had no scale to be better on.

    Leave-one-out and not each-against-the-full-centroid, because a recording that is part
    of its own reference is compared against itself and scores high for that reason alone.
    """
    keys = sorted(embeddings)
    if len(keys) < 3:
        raise ValueError(
            f"a leave-one-out band needs at least 3 recordings, got {len(keys)}; "
            "with two, each is scored against a single other recording and there is no band"
        )
    return {
        key: cosine_similarity(embeddings[key], centroid([embeddings[k] for k in keys if k != key]))
        for key in keys
    }


def calibration_band(scores: Mapping[str, float]) -> dict[str, Any]:
    """Turn leave-one-out scores into the band condition 4 is read against.

    `floor` is the band's worst real recording. A candidate below it is not merely worse
    than the target speaker's average - it is outside the range the speaker's own voice
    occupies, which is a different and much stronger statement than a negative delta.
    """
    summary = summarise_similarity(dict(scores))
    return {
        **summary,
        "floor": summary["min"],
        "method": "leave-one-out: each recording against the centroid of the other n-1",
        "per_clip": dict(sorted(scores.items())),
    }


def full_set_delta_vector(
    per_clip_delta: Mapping[str, float], all_keys: Sequence[str]
) -> list[float]:
    """Expand per-clip deltas onto the fixed clip set, charging 0 where nothing was produced.

    The same convention `mean_delta_full_set` uses, in vector form, so an interval estimate
    cannot quietly be computed over a shorter list than the mean it belongs to. A key that
    is not in the fixed set is an error rather than something to ignore: it would mean the
    two statistics were computed over different clips.
    """
    keys = list(all_keys)
    if not keys:
        raise ValueError("a fixed clip set is required")
    stray = sorted(set(per_clip_delta) - set(keys))
    if stray:
        raise ValueError(f"deltas for clips outside the fixed set: {stray[:5]}")
    return [float(per_clip_delta.get(key, 0.0)) for key in keys]


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


def paired_comparison_over_fixed_set(
    base: dict[str, float],
    candidate: dict[str, float],
    *,
    all_keys: Sequence[str],
    names: tuple[str, str],
) -> dict[str, Any]:
    """Compare two systems over a FIXED clip set, where either side may fail to produce one.

    `paired_comparison` requires both sides to cover the same clips. That is right when both
    systems always speak, and wrong here: a degraded checkpoint emits nothing at all on some
    prompts, and those clips have no embedding to compare.

    The trap is counting them inconsistently. Treating an absent clip as not-higher in the
    win count while dropping it from the mean lets an arm pass its effect-size bar on the
    average of exactly the clips where it still behaved - the more it degrades, the better
    its mean looks. So both means are computed and both are named:

    - `mean_delta_survivors` averages the clips both sides produced. Optimistic.
    - `mean_delta_full_set` charges a delta of 0 for every clip the candidate could not
      produce, over the full denominator. This is the one condition 4 is judged on, because
      it uses the same denominator as the win count.

    There is no key called `mean_delta`, so a report cannot quote one without saying which.

    The absolute cosines travel with the deltas for the same reason. A delta of +0.03 is a
    different result at 0.78 than at 0.55, and M3 reported only the delta - which is how an
    arm could be discussed as "improving" with nobody able to say whether it was anywhere
    near the target speaker's own range.
    """
    all_keys = list(all_keys)
    if not all_keys:
        raise ValueError("a fixed clip set is required")

    scorable = [k for k in all_keys if k in base and k in candidate]
    deltas = {k: candidate[k] - base[k] for k in scorable}
    survivors = list(deltas.values())
    wins = sum(1 for d in survivors if d > 0)
    full_set = full_set_delta_vector(deltas, all_keys)

    return {
        "systems": list(names),
        "denominator": len(all_keys),
        "scorable": len(scorable),
        "unscorable": sum(1 for k in all_keys if k not in candidate),
        "base_unscorable": sum(1 for k in all_keys if k not in base),
        "higher_on": wins,
        "lower_on": sum(1 for d in survivors if d < 0),
        "ties": sum(1 for d in survivors if d == 0),
        "mean_delta_survivors": statistics.fmean(survivors) if survivors else None,
        "mean_delta_full_set": sum(survivors) / len(all_keys),
        "mean_delta_is_full_set": len(scorable) == len(all_keys),
        "delta_stdev_full_set": statistics.stdev(full_set) if len(full_set) > 1 else 0.0,
        "delta_stdev_survivors": statistics.stdev(survivors) if len(survivors) > 1 else 0.0,
        "per_clip_delta": deltas,
        "per_clip_absolute": {
            key: {
                "base": base.get(key),
                "candidate": candidate.get(key),
                "delta": deltas.get(key),
            }
            for key in all_keys
        },
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


def require_likeness_report(report: Mapping[str, Any]) -> None:
    """Refuse to publish a likeness report that is missing what condition 4 is read on.

    Three things, and M3 shipped with none of them:

    (a) the calibration band - the target speaker's own within-speaker similarity,
    (b) the absolute cosine of every arm, not only its delta,
    (c) the per-clip paired deltas and their spread.

    Raising rather than warning, in the spirit of `paired_comparison_over_fixed_set`: it
    refuses to expose an ambiguous `mean_delta` at all rather than documenting which one is
    meant. A report that cannot be judged should not exist to be quoted.

    `report_kind` decides what is demanded, and a report cannot dodge the demand by
    mislabelling itself: a report carrying comparisons must declare itself a comparison.
    """
    kind = report.get("report_kind")
    if kind not in ("calibration", "comparison"):
        raise ValueError(f"report_kind must be 'calibration' or 'comparison', got {kind!r}")

    missing: list[str] = []

    band = report.get("calibration_band")
    if not isinstance(band, Mapping):
        missing.append("calibration_band")
    else:
        absent = [field for field in CALIBRATION_BAND_FIELDS if field not in band]
        if absent:
            missing.append(f"calibration_band.{{{','.join(absent)}}}")

    if kind == "calibration":
        if report.get("comparisons"):
            raise ValueError(
                "this report carries comparisons but declares report_kind 'calibration', "
                "which would skip the absolute-cosine and per-clip checks"
            )
        if missing:
            raise ValueError(f"calibration report is missing: {'; '.join(missing)}")
        return

    absolute = report.get("absolute_cosine")
    if not isinstance(absolute, Mapping) or not absolute:
        missing.append("absolute_cosine")
        absolute = {}

    comparisons = report.get("comparisons")
    if not isinstance(comparisons, Mapping) or not comparisons:
        missing.append("comparisons")
        comparisons = {}

    for name, comparison in comparisons.items():
        if not comparison.get("per_clip_delta"):
            missing.append(f"comparisons.{name}.per_clip_delta")
        if not comparison.get("per_clip_absolute"):
            missing.append(f"comparisons.{name}.per_clip_absolute")
        if comparison.get("delta_stdev_full_set") is None:
            missing.append(f"comparisons.{name}.delta_stdev_full_set")
        for system in comparison.get("systems", ()):
            if system not in absolute:
                missing.append(f"absolute_cosine.{system}")

    if missing:
        raise ValueError(f"comparison report is missing: {'; '.join(sorted(set(missing)))}")


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
    parser.add_argument("--system", action="append", nargs=2, metavar=("NAME", "DIR"), default=[])
    parser.add_argument(
        "--baseline",
        help="which --system name is the control; required whenever systems are given, "
        "because condition 4 is a paired comparison and has no meaning without one",
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model", default="speechbrain/spkrec-ecapa-voxceleb")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frame", type=int, default=DEFAULT_FRAME)
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD)
    parser.add_argument("--target-rms", type=float, default=DEFAULT_TARGET_RMS)
    parser.add_argument("--note", default=None, help="free text recorded with the report")
    args = parser.parse_args()

    names = [name for name, _ in args.system]
    if args.system:
        if args.baseline is None:
            parser.error("--baseline is required when --system is given")
        if args.baseline not in names:
            parser.error(f"--baseline {args.baseline!r} is not one of {names}")
        if len(names) < 2:
            parser.error(
                "a comparison report needs the baseline and at least one other system; "
                "with only a baseline there are no paired per-clip deltas to report"
            )

    target_paths = sorted(args.target_dir.glob("*.wav"))
    target_embeddings = _embed_all(
        target_paths, args.model, args.device, args.frame, args.threshold, args.target_rms
    )
    target_centroid = centroid(list(target_embeddings.values()))
    band = calibration_band(leave_one_out_similarity(target_embeddings))

    absolute: dict[str, Any] = {}
    for name, directory in args.system:
        paths = sorted(Path(directory).glob("*.wav"))
        embeddings = _embed_all(
            paths, args.model, args.device, args.frame, args.threshold, args.target_rms
        )
        scores = {
            key: cosine_similarity(vector, target_centroid) for key, vector in embeddings.items()
        }
        absolute[name] = {"per_file": scores, "summary": summarise_similarity(scores)}

    comparisons: dict[str, Any] = {}
    if args.system:
        base_scores = absolute[args.baseline]["per_file"]
        all_keys = sorted(base_scores)
        for name in names:
            if name == args.baseline:
                continue
            comparisons[name] = paired_comparison_over_fixed_set(
                base_scores,
                absolute[name]["per_file"],
                all_keys=all_keys,
                names=(args.baseline, name),
            )

    report = {
        "schema_version": 2,
        "report_kind": "comparison" if args.system else "calibration",
        "model": args.model,
        "target_dir": str(args.target_dir),
        "target_clips": len(target_paths),
        "baseline": args.baseline,
        "preparation": {
            "voiced_frame_samples": args.frame,
            "voiced_rms_threshold": args.threshold,
            "target_rms": args.target_rms,
        },
        "calibration_band": band,
        "absolute_cosine": absolute,
        "comparisons": comparisons,
        "note": args.note,
        "interpretation": (
            "Supporting evidence only. The plan forbids accepting a system on speaker "
            "similarity alone, because a flat rendering can outscore one a listener judges "
            "closer to the speaker. Read every delta against the calibration band: it is "
            "what the target speaker scores against her own centroid, and nothing here can "
            "be called close to her without it."
        ),
    }
    require_likeness_report(report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"calibration band: {json.dumps({k: band[k] for k in CALIBRATION_BAND_FIELDS})}")
    for name, payload in absolute.items():
        print(f"{name}: {json.dumps(payload['summary'], sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
