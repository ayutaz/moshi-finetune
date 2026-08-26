"""Decode a shipped training parquet back to audio and measure it against the source.

The gate this exists for
------------------------
Training reads the parquet and nothing else. If `tools/tokenize_audio.py` had written
speaker B's codebooks into column A, or a delay had been applied twice, or a channel had
come back as noise, **the loss would still fall** - the model would simply learn the wrong
thing, and no curve would say so. M3's costliest correction was exactly this shape: a
number that looked healthy while the thing it stood for was broken. So the tokens are
decoded back to sound and compared with the recording they came from.

Mimi is lossy, so nothing here is a bit comparison. The three questions that survive a
lossy codec are:

1. **Is channel A still channel A?** Answered by scoring every decoded channel against
   every source channel and requiring each one's own source to win. A swap is the one
   defect that leaves both channels individually perfect.
2. **Is the sound intact?** Silence share, clipping, spectral centroid and high-band
   energy, source against decode.
3. **Is speaker A still the target speaker?** Answered outside this module, by
   `tools/speaker_similarity.py`, against the same held-out centroid and calibration band
   the arms are scored on.

Everything above `_HEAVY` is pure and takes plain sequences, so the gates stay testable in
a suite that runs without numpy, torch or a dataset. numpy, pandas, soundfile, torch and
moshi are imported inside the functions that need them.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

#: The dataset is written at 24 kHz and Mimi runs at 12.5 Hz, so one Mimi frame is 1920
#: samples. Every frame-wise measurement here uses that hop, which makes a frame index in
#: the envelope the same frame index in the codebook rows.
SAMPLE_RATE = 24000
FRAME_SAMPLES = 1920

#: `data/experiments/tsukuyomi_ojousama/m3r/v-real` was assembled with this threshold
#: (`m3r-timeline.json` -> `timeline.speech_threshold_rms`). Reusing it rather than picking
#: a new one keeps "A is silent here" the same statement the builder made.
SPEECH_RMS_THRESHOLD = 0.01

CODEBOOKS_PER_SPEAKER = 8
STREAMS_PER_SPEAKER_COLUMN = 1 + CODEBOOKS_PER_SPEAKER

#: Above this the sample is at or past full scale. PCM_16 saturates at 1.0; a decode that
#: rides the ceiling is distorted even though every statistic below is still finite.
CLIPPING_THRESHOLD = 0.99


class RoundtripShapeError(ValueError):
    """A parquet cell is not shaped like something the decoder can be handed."""


def split_speaker_column(
    column: Sequence[Sequence[int]], *, codebooks: int = CODEBOOKS_PER_SPEAKER
) -> tuple[list[int], list[list[int]]]:
    """Split one `[1 + K, T]` speaker column into its text row and its codebook rows.

    `tools/prepare_dataset.merge_text_audio` writes the text row first and the eight Mimi
    codebooks after it. A column of any other height is an error rather than something to
    slice optimistically: the slice would still produce eight rows from a nine-row cell
    that had lost its text stream, and the decode would sound fine.
    """
    rows = [list(row) for row in column]
    if len(rows) != codebooks + 1:
        raise RoundtripShapeError(
            f"speaker column has {len(rows)} streams, expected {codebooks + 1} "
            "(1 text row then 8 Mimi codebooks)"
        )
    widths = {len(row) for row in rows}
    if len(widths) != 1:
        raise RoundtripShapeError(
            f"speaker column has rows of {sorted(widths)} frames; a column is a rectangle "
            "and a ragged one means a row was padded that should not have been"
        )
    if widths == {0}:
        raise RoundtripShapeError("speaker column has no frames")
    return rows[0], rows[1:]


def decoder_token_rows(
    a_column: Sequence[Sequence[int]],
    b_column: Sequence[Sequence[int]],
    *,
    codebooks: int = CODEBOOKS_PER_SPEAKER,
) -> list[list[int]]:
    """Pack the two speaker columns into the `[2K, T]` block `decode_tokens` expects.

    `tools/decode_tokens.decode_audio` splits its input down the middle with `np.split`
    and hands the two halves to Mimi as a batch, so the FIRST K rows come back as decoded
    channel 0 and the last K as channel 1. `tools/tokenize_audio.py` put source channel 0
    into column A. Speaker A therefore has to go first, or the decode is swapped relative
    to the recording and the swap detector would be reading its own bug.

    That ordering is the whole reason this adapter is a named function with a test rather
    than a `np.concatenate` at the call site.
    """
    _, a_audio = split_speaker_column(a_column, codebooks=codebooks)
    _, b_audio = split_speaker_column(b_column, codebooks=codebooks)
    if len(a_audio[0]) != len(b_audio[0]):
        raise RoundtripShapeError(
            f"speaker A has {len(a_audio[0])} frames and speaker B has {len(b_audio[0])}; "
            "the two channels of one dialogue are tokenised from one stereo file and "
            "cannot differ in length"
        )
    return a_audio + b_audio


def frame_rms(samples: Sequence[float], *, hop: int = FRAME_SAMPLES) -> list[float]:
    """RMS of each whole `hop`-sample frame. A trailing partial frame is dropped.

    Dropped rather than zero-padded: a padded tail reads quieter than the sound in it and
    would show up as a silence difference between two files of slightly different length.
    """
    if hop <= 0:
        raise ValueError("hop must be positive")
    count = len(samples) // hop
    envelope = []
    for index in range(count):
        window = samples[index * hop : (index + 1) * hop]
        envelope.append(math.sqrt(sum(float(v) * float(v) for v in window) / hop))
    return envelope


def pearson(left: Sequence[float], right: Sequence[float]) -> float:
    """Pearson correlation. Raises on a constant input rather than returning 0.

    A constant envelope means a silent or DC channel, and calling that "uncorrelated" would
    let a dead channel pass the swap check by being equally unlike both sources.
    """
    if len(left) != len(right):
        raise ValueError(f"lengths differ: {len(left)} vs {len(right)}")
    if len(left) < 2:
        raise ValueError("correlation needs at least two points")
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    left_dev = [float(v) - left_mean for v in left]
    right_dev = [float(v) - right_mean for v in right]
    left_norm = math.sqrt(sum(v * v for v in left_dev))
    right_norm = math.sqrt(sum(v * v for v in right_dev))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("a constant signal has no correlation with anything")
    return sum(a * b for a, b in zip(left_dev, right_dev, strict=True)) / (left_norm * right_norm)


def frame_peak(samples: Sequence[float], *, hop: int = FRAME_SAMPLES) -> list[float]:
    """Largest absolute sample in each whole `hop`-sample frame."""
    if hop <= 0:
        raise ValueError("hop must be positive")
    count = len(samples) // hop
    return [max(abs(float(v)) for v in samples[i * hop : (i + 1) * hop]) for i in range(count)]


def digitally_silent_mask(peaks: Sequence[float]) -> list[bool]:
    """Which frames are EXACTLY zero, as opposed to merely quiet.

    The distinction is the whole point of the room tone. `tools/room_tone.fill_silence`
    replaces digitally silent stretches; a frame that is still exactly zero afterwards is a
    gap the fill did not reach, and it is the frames that are exactly zero which Mimi maps
    onto the two silence codes. Measuring "quiet" alone cannot tell the two apart, and a
    residual silence code in the quiet pool would be read as a failure when it may simply
    be the code for a very low room tone.
    """
    return [float(value) == 0.0 for value in peaks]


def best_lag_correlation(
    source: Sequence[float], decoded: Sequence[float], *, max_lag: int = 10
) -> dict[str, Any]:
    """Where the two envelopes line up best, in frames, and how well.

    This is the check for a delay baked into the parquet. `utils/data.delay_and_pad_streams`
    applies the seventeen delays at TRAINING time; the parquet must hold undelayed streams.
    If a delay had already been applied when the dataset was written it would be applied
    twice, and every statistic computed at lag zero would still look plausible - the shapes
    would simply be shifted. A best lag of anything but 0 says the decode does not sit
    where the recording sits.

    `lag_frames = l` aligns `source[t + l]` with `decoded[t]`, so a decode arriving later
    than the recording gives a negative l.
    """
    if max_lag < 0:
        raise ValueError("max_lag cannot be negative")
    best: tuple[int, float] | None = None
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            left, right = source[lag:], decoded[: len(decoded) - lag if lag else None]
        else:
            left, right = source[: len(source) + lag], decoded[-lag:]
        width = min(len(left), len(right))
        if width < 2:
            continue
        try:
            score = pearson(left[:width], right[:width])
        except ValueError:
            continue
        if best is None or score > best[1]:
            best = (lag, score)
    if best is None:
        raise ValueError("no lag gave a comparable overlap")
    return {"lag_frames": best[0], "correlation": best[1], "max_lag": max_lag}


def voiced_mask(
    envelope: Sequence[float], *, threshold: float = SPEECH_RMS_THRESHOLD
) -> list[bool]:
    """Which frames carry speech, by the builder's own RMS threshold."""
    return [float(value) >= threshold for value in envelope]


def quiet_frame_indices(
    envelope: Sequence[float], *, threshold: float = SPEECH_RMS_THRESHOLD
) -> list[int]:
    """The complement of `voiced_mask`, as indices into the codebook rows."""
    return [index for index, value in enumerate(envelope) if float(value) < threshold]


def mask_agreement(left: Sequence[bool], right: Sequence[bool]) -> dict[str, Any]:
    """How far two voicing masks agree, as counts plus IoU and frame accuracy.

    IoU and accuracy answer different questions and both are reported. On a channel that is
    silent 70% of the time, accuracy is high for free; IoU is not.
    """
    if len(left) != len(right):
        raise ValueError(f"masks differ in length: {len(left)} vs {len(right)}")
    if not left:
        raise ValueError("cannot compare empty masks")
    both = sum(1 for a, b in zip(left, right, strict=True) if a and b)
    left_only = sum(1 for a, b in zip(left, right, strict=True) if a and not b)
    right_only = sum(1 for a, b in zip(left, right, strict=True) if b and not a)
    neither = len(left) - both - left_only - right_only
    union = both + left_only + right_only
    return {
        "frames": len(left),
        "both": both,
        "left_only": left_only,
        "right_only": right_only,
        "neither": neither,
        "iou": (both / union) if union else 1.0,
        "accuracy": (both + neither) / len(left),
    }


def channel_assignment(scores: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    """Read a source x decode score matrix as an assignment, with its margin.

    `scores[source][decoded]`. The verdict is `identity` when each source channel's best
    match is the decode of that same channel, `swapped` when each one's best match is the
    other channel, and `ambiguous` for anything else - including the case where one channel
    is confident and the other is not, which is not a swap but is not a pass either.

    The margin is the smallest own-minus-other gap over the source channels, so one weak
    channel cannot be hidden by a strong one.
    """
    names = sorted(scores)
    if len(names) != 2:
        raise ValueError(f"an assignment is over exactly two channels, got {names}")
    first, second = names
    for source in names:
        missing = [name for name in names if name not in scores[source]]
        if missing:
            raise ValueError(f"scores[{source!r}] has no entry for {missing}")
    margins = {
        source: scores[source][source] - scores[source][_other(source, names)] for source in names
    }
    best = {source: max(names, key=lambda decoded: scores[source][decoded]) for source in names}
    if best[first] == first and best[second] == second:
        verdict = "identity"
    elif best[first] == second and best[second] == first:
        verdict = "swapped"
    else:
        verdict = "ambiguous"
    return {
        "verdict": verdict,
        "best_match": best,
        "margins": margins,
        "min_margin": min(margins.values()),
        "scores": {source: dict(scores[source]) for source in names},
    }


def _other(name: str, names: Sequence[str]) -> str:
    return names[0] if name == names[1] else names[1]


def spectral_centroid_hz(magnitudes: Sequence[float], freqs: Sequence[float]) -> float:
    """Energy-weighted mean frequency of a magnitude spectrum.

    Weighted by magnitude, not power, to match the convention the value is usually read in.
    A spectrum of all zeros has no centroid and raises: a silent window's "centroid" would
    otherwise be 0 Hz and would drag an average towards the low end for a reason that has
    nothing to do with bandwidth.
    """
    if len(magnitudes) != len(freqs):
        raise ValueError(f"spectrum and frequency axis differ: {len(magnitudes)} vs {len(freqs)}")
    total = sum(float(value) for value in magnitudes)
    if total <= 0:
        raise ValueError("a spectrum with no energy has no centroid")
    return sum(float(m) * float(f) for m, f in zip(magnitudes, freqs, strict=True)) / total


def band_energy_ratio(
    magnitudes: Sequence[float], freqs: Sequence[float], *, cutoff_hz: float
) -> float:
    """Share of the spectrum's energy at or above `cutoff_hz`.

    Energy, so the ratio is over squared magnitudes: this is the number that says whether
    the codec threw the top of the band away, and a magnitude-weighted version understates
    how much of the signal's power sits low.
    """
    if len(magnitudes) != len(freqs):
        raise ValueError(f"spectrum and frequency axis differ: {len(magnitudes)} vs {len(freqs)}")
    total = sum(float(value) ** 2 for value in magnitudes)
    if total <= 0:
        raise ValueError("a spectrum with no energy has no band ratio")
    high = sum(
        float(m) ** 2 for m, f in zip(magnitudes, freqs, strict=True) if float(f) >= cutoff_hz
    )
    return high / total


def clipping_stats(
    samples: Sequence[float], *, threshold: float = CLIPPING_THRESHOLD
) -> dict[str, Any]:
    """Peak level and how much of the waveform sits at or past `threshold`."""
    if not samples:
        raise ValueError("cannot measure an empty waveform")
    peak = max(abs(float(value)) for value in samples)
    clipped = sum(1 for value in samples if abs(float(value)) >= threshold)
    return {
        "samples": len(samples),
        "peak": peak,
        "clipped": clipped,
        "clipped_share": clipped / len(samples),
    }


def tokens_at(row: Sequence[int], indices: Sequence[int]) -> list[int]:
    """The codebook tokens at a set of frame indices, bounds checked.

    Silently dropping an out-of-range index is what would happen with a numpy fancy-index
    guard, and it would quietly shrink the sample the silence gate is computed over.
    """
    values = list(row)
    stray = [index for index in indices if index < 0 or index >= len(values)]
    if stray:
        raise IndexError(f"frame indices outside the {len(values)}-frame row: {stray[:5]}")
    return [values[index] for index in indices]


def evenly_spaced(items: Sequence[Any], count: int) -> list[Any]:
    """A deterministic spread of `count` items across a sequence, both ends included.

    Deterministic and not random, so the same dialogues are inspected on a re-run and a
    disagreement between two runs is a real disagreement.
    """
    total = len(items)
    if count <= 0:
        raise ValueError("count must be positive")
    if total == 0:
        raise ValueError("cannot sample an empty sequence")
    if count >= total:
        return list(items)
    if count == 1:
        return [items[0]]
    picked = sorted({round(index * (total - 1) / (count - 1)) for index in range(count)})
    return [items[index] for index in picked]


def leave_one_out_against(
    reference: Mapping[str, Sequence[float]], probes: Mapping[str, Sequence[float]]
) -> dict[str, float]:
    """Score each probe against the centroid of the reference set MINUS its own key.

    This is what makes a codec-adjusted calibration band comparable with an arm's score.
    The published band is each natural recording against the centroid of the other nine
    naturals. An arm is scored against the centroid of all ten. Round-tripping the ten
    naturals through Mimi and scoring each one against the centroid of the other nine
    naturals therefore measures one thing only: what the codec costs, on the same axis the
    arms are read on.

    Scoring a round-tripped clip against a centroid that still contains its own natural
    twin would compare a recording with itself and inflate the band, which is the same trap
    `leave_one_out_similarity` exists to avoid.
    """
    from tools.speaker_similarity import centroid, cosine_similarity

    missing = sorted(set(probes) - set(reference))
    if missing:
        raise ValueError(f"probes with no reference recording of the same name: {missing[:5]}")
    if len(reference) < 3:
        raise ValueError(
            f"a leave-one-out band needs at least 3 reference recordings, got {len(reference)}"
        )
    return {
        key: cosine_similarity(
            probes[key], centroid([reference[other] for other in sorted(reference) if other != key])
        )
        for key in sorted(probes)
    }


def spread(values: Sequence[float]) -> dict[str, float]:
    """min / median / mean / max of a list of measurements."""
    numbers = [float(value) for value in values]
    if not numbers:
        raise ValueError("cannot summarise an empty list")
    return {
        "n": len(numbers),
        "min": min(numbers),
        "median": statistics.median(numbers),
        "mean": statistics.fmean(numbers),
        "max": max(numbers),
    }


# --------------------------------------------------------------------------------------
# _HEAVY: everything below needs numpy, pandas, soundfile, torch or moshi, and imports it
# inside the function so the suite above stays runnable without any of them.
# --------------------------------------------------------------------------------------


def read_parquet_columns(parquet_path: Path) -> dict[str, dict[str, list[list[int]]]]:
    """Load one parquet as `{dialogue_id: {"A": [[...]], "B": [[...]]}}`."""
    import pandas as pd

    frame = pd.read_parquet(parquet_path)
    missing = [column for column in ("dialogue_id", "A", "B") if column not in frame.columns]
    if missing:
        raise RoundtripShapeError(f"{parquet_path}: no {', '.join(missing)} column")
    rows: dict[str, dict[str, list[list[int]]]] = {}
    for record in frame.to_dict("records"):
        rows[str(record["dialogue_id"])] = {
            speaker: [[int(v) for v in row] for row in record[speaker]] for speaker in ("A", "B")
        }
    return rows


def _load_mimi(device: str):
    import torch
    from huggingface_hub import hf_hub_download
    from moshi.models import loaders

    return loaders.get_mimi(
        filename=hf_hub_download(
            "kyutai/moshiko-pytorch-bf16", "tokenizer-e351c8d8-checkpoint125.safetensors"
        ),
        device=torch.device(device),
    )


def _cmd_decode(args: argparse.Namespace) -> int:
    import numpy as np
    import soundfile as sf

    from tools.decode_tokens import decode_audio

    rows = read_parquet_columns(Path(args.parquet))
    ids = evenly_spaced(sorted(rows), args.count)
    out = Path(args.out_dir)
    for name in ("stereo", "source-A", "source-B", "decoded-A", "decoded-B"):
        (out / name).mkdir(parents=True, exist_ok=True)

    mimi = _load_mimi(args.device)
    manifest: list[dict[str, Any]] = []
    for dialogue_id in ids:
        stem = dialogue_id.split("/", 1)[-1]
        block = decoder_token_rows(rows[dialogue_id]["A"], rows[dialogue_id]["B"])
        wav = decode_audio(np.asarray(block, dtype=np.int64), mimi)  # (2, samples)
        sf.write(out / "stereo" / f"{stem}.wav", wav.astype(np.float32).T, SAMPLE_RATE)
        for index, speaker in enumerate(("A", "B")):
            sf.write(
                out / f"decoded-{speaker}" / f"{stem}.wav",
                wav[index].astype(np.float32),
                SAMPLE_RATE,
            )
        source, rate = sf.read(Path(args.source_dir) / f"{stem}.wav", dtype="float32")
        if rate != SAMPLE_RATE or source.ndim != 2 or source.shape[1] != 2:
            raise RoundtripShapeError(
                f"{stem}: source is {rate} Hz with shape {source.shape}; expected stereo "
                f"{SAMPLE_RATE} Hz"
            )
        for index, speaker in enumerate(("A", "B")):
            sf.write(out / f"source-{speaker}" / f"{stem}.wav", source[:, index], SAMPLE_RATE)
        manifest.append(
            {
                "dialogue_id": dialogue_id,
                "stem": stem,
                "frames": len(block[0]),
                "decoded_samples": int(wav.shape[1]),
                "source_samples": int(source.shape[0]),
            }
        )
        print(f"decoded {dialogue_id}: {len(block[0])} frames -> {wav.shape[1]} samples")

    payload = {
        "parquet": str(args.parquet),
        "source_dir": str(args.source_dir),
        "out_dir": str(out),
        "device": args.device,
        "sample_rate": SAMPLE_RATE,
        "selection": f"evenly_spaced over the sorted dialogue_id list, count={args.count}",
        "dialogues": manifest,
    }
    Path(args.report).write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.report}")
    return 0


def _average_spectrum(samples, mask: Sequence[bool], *, hop: int = FRAME_SAMPLES):
    """Mean magnitude spectrum over the frames `mask` selects, with a Hann window."""
    import numpy as np

    picked = [index for index, keep in enumerate(mask) if keep]
    if not picked:
        return None, None
    window = np.hanning(hop)
    frames = np.stack([np.asarray(samples[i * hop : (i + 1) * hop]) * window for i in picked])
    magnitudes = np.abs(np.fft.rfft(frames, axis=1)).mean(axis=0)
    freqs = np.fft.rfftfreq(hop, d=1.0 / SAMPLE_RATE)
    return magnitudes.tolist(), freqs.tolist()


def compare_pair(source_stereo, decoded_stereo) -> dict[str, Any]:
    """All per-dialogue statistics for one source/decode pair of stereo waveforms."""
    import numpy as np

    length = min(source_stereo.shape[0], decoded_stereo.shape[0])
    frames = length // FRAME_SAMPLES
    if frames < 2:
        raise RoundtripShapeError("need at least two whole frames to compare")
    trimmed = length - (length % FRAME_SAMPLES)
    source = np.asarray(source_stereo[:trimmed], dtype=np.float64)
    decoded = np.asarray(decoded_stereo[:trimmed], dtype=np.float64)

    envelopes: dict[str, dict[str, list[float]]] = {"source": {}, "decoded": {}}
    masks: dict[str, dict[str, list[bool]]] = {"source": {}, "decoded": {}}
    for index, speaker in enumerate(("A", "B")):
        envelopes["source"][speaker] = frame_rms(source[:, index].tolist())
        envelopes["decoded"][speaker] = frame_rms(decoded[:, index].tolist())
        masks["source"][speaker] = voiced_mask(envelopes["source"][speaker])
        masks["decoded"][speaker] = voiced_mask(envelopes["decoded"][speaker])

    correlation = {
        src: {dec: pearson(envelopes["source"][src], envelopes["decoded"][dec]) for dec in "AB"}
        for src in "AB"
    }
    voicing_iou = {
        src: {
            dec: mask_agreement(masks["source"][src], masks["decoded"][dec])["iou"] for dec in "AB"
        }
        for src in "AB"
    }

    per_channel: dict[str, Any] = {}
    for index, speaker in enumerate(("A", "B")):
        entry: dict[str, Any] = {
            "envelope_correlation": correlation[speaker][speaker],
            "best_lag": best_lag_correlation(
                envelopes["source"][speaker], envelopes["decoded"][speaker]
            ),
            "voicing": mask_agreement(masks["source"][speaker], masks["decoded"][speaker]),
            "silence_share": {
                "source": 1.0 - sum(masks["source"][speaker]) / frames,
                "decoded": 1.0 - sum(masks["decoded"][speaker]) / frames,
            },
            "rms": {
                "source": float(np.sqrt((source[:, index] ** 2).mean())),
                "decoded": float(np.sqrt((decoded[:, index] ** 2).mean())),
            },
            "clipping": {
                "source": clipping_stats(source[:, index].tolist()),
                "decoded": clipping_stats(decoded[:, index].tolist()),
            },
        }
        entry["silence_share"]["delta"] = (
            entry["silence_share"]["decoded"] - entry["silence_share"]["source"]
        )
        for side, wave in (("source", source), ("decoded", decoded)):
            magnitudes, freqs = _average_spectrum(wave[:, index], masks["source"][speaker])
            if magnitudes is None:
                entry.setdefault("spectrum", {})[side] = None
                continue
            entry.setdefault("spectrum", {})[side] = {
                "centroid_hz": spectral_centroid_hz(magnitudes, freqs),
                "energy_above_4k": band_energy_ratio(magnitudes, freqs, cutoff_hz=4000.0),
                "energy_above_6k": band_energy_ratio(magnitudes, freqs, cutoff_hz=6000.0),
            }
        per_channel[speaker] = entry

    return {
        "frames": frames,
        "seconds": frames * FRAME_SAMPLES / SAMPLE_RATE,
        "envelope_correlation_matrix": correlation,
        "voicing_iou_matrix": voicing_iou,
        "assignment_by_envelope": channel_assignment(correlation),
        "assignment_by_voicing": channel_assignment(voicing_iou),
        "channels": per_channel,
    }


def _cmd_compare(args: argparse.Namespace) -> int:
    import numpy as np
    import soundfile as sf

    decoded_dir = Path(args.decoded_dir)
    source_dir = Path(args.source_dir)
    stems = sorted(path.stem for path in (decoded_dir / "stereo").glob("*.wav"))
    if not stems:
        raise RoundtripShapeError(f"no decoded stereo wavs under {decoded_dir / 'stereo'}")

    per_dialogue: dict[str, Any] = {}
    for stem in stems:
        decoded, decoded_rate = sf.read(decoded_dir / "stereo" / f"{stem}.wav", dtype="float32")
        source, source_rate = sf.read(source_dir / f"{stem}.wav", dtype="float32")
        if decoded_rate != SAMPLE_RATE or source_rate != SAMPLE_RATE:
            raise RoundtripShapeError(f"{stem}: {source_rate} Hz source, {decoded_rate} Hz decode")
        per_dialogue[stem] = compare_pair(np.asarray(source), np.asarray(decoded))
        print(f"compared {stem}")

    swapped = [
        stem
        for stem, entry in per_dialogue.items()
        if entry["assignment_by_envelope"]["verdict"] != "identity"
        or entry["assignment_by_voicing"]["verdict"] != "identity"
    ]
    shifted = [
        f"{stem}/{speaker}"
        for stem, entry in per_dialogue.items()
        for speaker in ("A", "B")
        if entry["channels"][speaker]["best_lag"]["lag_frames"] != 0
    ]
    aggregate = {
        "dialogues": len(per_dialogue),
        "assignment_identity": len(per_dialogue) - len(swapped),
        "not_identity": swapped,
        "channels_whose_best_lag_is_not_zero": shifted,
        "min_envelope_margin": min(
            entry["assignment_by_envelope"]["min_margin"] for entry in per_dialogue.values()
        ),
        "min_voicing_margin": min(
            entry["assignment_by_voicing"]["min_margin"] for entry in per_dialogue.values()
        ),
    }
    for speaker in ("A", "B"):
        rows = [entry["channels"][speaker] for entry in per_dialogue.values()]
        aggregate[speaker] = {
            "envelope_correlation": spread([row["envelope_correlation"] for row in rows]),
            "voicing_iou": spread([row["voicing"]["iou"] for row in rows]),
            "voicing_accuracy": spread([row["voicing"]["accuracy"] for row in rows]),
            "silence_share_source": spread([row["silence_share"]["source"] for row in rows]),
            "silence_share_decoded": spread([row["silence_share"]["decoded"] for row in rows]),
            "silence_share_delta": spread([row["silence_share"]["delta"] for row in rows]),
            "rms_source": spread([row["rms"]["source"] for row in rows]),
            "rms_decoded": spread([row["rms"]["decoded"] for row in rows]),
            "peak_source": spread([row["clipping"]["source"]["peak"] for row in rows]),
            "peak_decoded": spread([row["clipping"]["decoded"]["peak"] for row in rows]),
            "clipped_share_source": spread(
                [row["clipping"]["source"]["clipped_share"] for row in rows]
            ),
            "clipped_share_decoded": spread(
                [row["clipping"]["decoded"]["clipped_share"] for row in rows]
            ),
            "centroid_hz_source": spread(
                [row["spectrum"]["source"]["centroid_hz"] for row in rows]
            ),
            "centroid_hz_decoded": spread(
                [row["spectrum"]["decoded"]["centroid_hz"] for row in rows]
            ),
            "energy_above_4k_source": spread(
                [row["spectrum"]["source"]["energy_above_4k"] for row in rows]
            ),
            "energy_above_4k_decoded": spread(
                [row["spectrum"]["decoded"]["energy_above_4k"] for row in rows]
            ),
            "energy_above_6k_source": spread(
                [row["spectrum"]["source"]["energy_above_6k"] for row in rows]
            ),
            "energy_above_6k_decoded": spread(
                [row["spectrum"]["decoded"]["energy_above_6k"] for row in rows]
            ),
        }

    payload = {
        "decoded_dir": str(decoded_dir),
        "source_dir": str(source_dir),
        "frame_samples": FRAME_SAMPLES,
        "speech_rms_threshold": SPEECH_RMS_THRESHOLD,
        "aggregate": aggregate,
        "per_dialogue": per_dialogue,
    }
    Path(args.report).write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.report}")
    return 0 if not swapped and not shifted else 1


def in_situ_silence_tokens(
    rows: Mapping[str, Mapping[str, Sequence[Sequence[int]]]],
    source_dir: Path,
    *,
    speaker: str = "A",
    codebook: int = 0,
    threshold: float = SPEECH_RMS_THRESHOLD,
) -> dict[str, Any]:
    """Collect one speaker's codebook-`codebook` tokens over the frames where they are quiet.

    The frames are chosen from the SOURCE recording, not from the tokens, because the
    question is whether the tokeniser turned a quiet stretch of real room tone into the
    silence token. Choosing the frames from the tokens would make the answer circular.
    """
    import soundfile as sf

    channel_index = {"A": 0, "B": 1}[speaker]
    collected: list[int] = []
    zero_tokens: list[int] = []
    room_tone_tokens: list[int] = []
    per_dialogue: dict[str, Any] = {}
    for dialogue_id in sorted(rows):
        stem = dialogue_id.split("/", 1)[-1]
        audio, rate = sf.read(source_dir / f"{stem}.wav", dtype="float32")
        if rate != SAMPLE_RATE:
            raise RoundtripShapeError(f"{stem}: {rate} Hz source, expected {SAMPLE_RATE}")
        channel = audio[:, channel_index].tolist()
        envelope = frame_rms(channel)
        zeros = digitally_silent_mask(frame_peak(channel))
        _, codebooks = split_speaker_column(rows[dialogue_id][speaker])
        row = codebooks[codebook]
        usable = min(len(envelope), len(row))
        indices = quiet_frame_indices(envelope[:usable], threshold=threshold)
        tokens = tokens_at(row[:usable], indices)
        zero_indices = [index for index in indices if zeros[index]]
        collected.extend(tokens)
        zero_tokens.extend(tokens_at(row[:usable], zero_indices))
        room_tone_tokens.extend(
            tokens_at(row[:usable], [index for index in indices if not zeros[index]])
        )
        per_dialogue[dialogue_id] = {
            "frames": usable,
            "quiet_frames": len(indices),
            "quiet_share": len(indices) / usable,
            "digitally_silent_frames": len(zero_indices),
            "tokens": tokens,
        }
    if not collected:
        raise RoundtripShapeError("no quiet frames found in any dialogue")
    return {
        "tokens": collected,
        "digitally_silent_tokens": zero_tokens,
        "room_tone_tokens": room_tone_tokens,
        "per_dialogue": per_dialogue,
    }


def _cmd_silence(args: argparse.Namespace) -> int:
    from tools.room_tone import gate_verdict, token_stats

    results = []
    for label, parquet, source_dir in args.dataset:
        rows = read_parquet_columns(Path(parquet))
        gathered = in_situ_silence_tokens(rows, Path(source_dir), speaker=args.speaker)
        stats = token_stats(gathered["tokens"])
        stats["gate"] = gate_verdict(
            stats, min_distinct=args.min_distinct, max_top_share=args.max_top_share
        )
        stats["label"] = label
        stats["parquet"] = parquet
        stats["source_dir"] = source_dir
        stats["dialogues"] = len(gathered["per_dialogue"])
        # The gate's real question. Mimi maps an exactly-zero frame onto its silence codes;
        # a frame carrying room tone gets a code for the tone. Splitting the quiet pool the
        # two ways says whether a residual silence code is a gap the fill missed or simply
        # the code for a very quiet bed.
        zeros = gathered["digitally_silent_tokens"]
        tone = gathered["room_tone_tokens"]
        stats["digitally_silent"] = {
            "frames": len(zeros),
            "share_of_quiet_frames": len(zeros) / len(gathered["tokens"]),
            "stats": token_stats(zeros) if zeros else None,
        }
        stats["room_tone_frames"] = {
            "frames": len(tone),
            "share_of_quiet_frames": len(tone) / len(gathered["tokens"]),
            "stats": token_stats(tone) if tone else None,
        }
        stats["quiet_share"] = spread(
            [entry["quiet_share"] for entry in gathered["per_dialogue"].values()]
        )
        # Distinct-token counts grow with the size of the pool they are counted over, so a
        # pooled figure from 70 dialogues cannot be read against a calibration measured on
        # ~230 frames. The per-dialogue pass puts the gate on a pool of the right size.
        per_dialogue = {}
        for dialogue_id, entry in gathered["per_dialogue"].items():
            one = token_stats(entry["tokens"])
            one["gate"] = gate_verdict(
                one, min_distinct=args.min_distinct, max_top_share=args.max_top_share
            )
            per_dialogue[dialogue_id] = one
        stats["per_dialogue"] = per_dialogue
        stats["per_dialogue_summary"] = {
            "dialogues": len(per_dialogue),
            "gate_passes": sum(1 for one in per_dialogue.values() if one["gate"]["passed"]),
            "quiet_frames": spread([one["frames"] for one in per_dialogue.values()]),
            "distinct": spread([one["distinct"] for one in per_dialogue.values()]),
            "top_share": spread([one["top_share"] for one in per_dialogue.values()]),
            "silence_token_share": spread(
                [one["silence_token_share"] for one in per_dialogue.values()]
            ),
            "entropy_bits": spread([one["entropy_bits"] for one in per_dialogue.values()]),
            "modal_top_token": statistics.mode([one["top_token"] for one in per_dialogue.values()]),
        }
        results.append(stats)
        print(label, {k: v for k, v in stats.items() if k not in ("per_dialogue",)})

    payload = {
        "speaker": args.speaker,
        "codebook": 0,
        "frame_selection": (
            f"frames of the SOURCE recording's channel with RMS < {SPEECH_RMS_THRESHOLD} over "
            f"{FRAME_SAMPLES} samples, the threshold m3r-timeline.json built the dialogues with"
        ),
        "gate": {"min_distinct": args.min_distinct, "max_top_share": args.max_top_share},
        "datasets": results,
    }
    Path(args.report).write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.report}")
    return 0 if all(entry["gate"]["passed"] for entry in results) else 1


def _cmd_codec_band(args: argparse.Namespace) -> int:
    """Round-trip the target speaker's own recordings through Mimi and re-measure the band.

    Without this, a decoded arm that scores 0.78 against a 0.8166 band cannot be read: the
    gap could be the dataset or it could be the codec. Running the band's own recordings
    through the same eight codebooks separates the two.
    """
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    # The private preprocessing path on purpose: the band and the arms have to be embedded
    # by the same voiced-region + RMS-normalise steps, or the two numbers are not on one
    # axis. m3-likeness-calibration.json fixed that requirement.
    from tools.speaker_similarity import (
        DEFAULT_FRAME,
        DEFAULT_TARGET_RMS,
        DEFAULT_THRESHOLD,
        calibration_band,
        leave_one_out_similarity,
    )
    from tools.speaker_similarity import _embed_all as embed_all

    clips = sorted(Path(args.clips_dir).glob("*.wav"))
    if not clips:
        raise RoundtripShapeError(f"no wavs under {args.clips_dir}")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    mimi = _load_mimi(args.device)
    written = []
    for path in clips:
        data, rate = sf.read(path, dtype="float32", always_2d=True)
        wav = torch.from_numpy(np.asarray(data[:, 0], dtype=np.float32)).reshape(1, 1, -1)
        if rate != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, rate, SAMPLE_RATE)
        with torch.no_grad():
            codes = mimi.encode(wav.to(args.device))
            back = mimi.decode(codes).cpu().numpy()[0, 0]
        target = out / path.name
        sf.write(target, back.astype(np.float32), SAMPLE_RATE)
        written.append(
            {"clip": path.stem, "source_rate": int(rate), "codebooks": int(codes.shape[1])}
        )
        print(
            f"round-tripped {path.stem}: {rate} Hz -> {SAMPLE_RATE} Hz, {codes.shape[1]} codebooks"
        )

    natural = embed_all(
        clips, args.model, args.device, DEFAULT_FRAME, DEFAULT_THRESHOLD, DEFAULT_TARGET_RMS
    )
    decoded = embed_all(
        sorted(out.glob("*.wav")),
        args.model,
        args.device,
        DEFAULT_FRAME,
        DEFAULT_THRESHOLD,
        DEFAULT_TARGET_RMS,
    )
    natural_band = calibration_band(leave_one_out_similarity(natural))
    codec_scores = leave_one_out_against(natural, decoded)
    codec_band = calibration_band(codec_scores)
    payload = {
        "clips_dir": str(args.clips_dir),
        "out_dir": str(out),
        "model": args.model,
        "device": args.device,
        "clips": written,
        "natural_band": natural_band,
        "codec_band": codec_band,
        "codec_cost": {
            "mean": codec_band["mean"] - natural_band["mean"],
            "floor": codec_band["floor"] - natural_band["floor"],
        },
        "method": (
            "each held-out recording is Mimi-encoded at 8 codebooks and decoded, then scored "
            "against the centroid of the other nine NATURAL recordings - the same axis the "
            "published band and the arms are read on"
        ),
    }
    Path(args.report).write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.report}")
    return 0


def _cmd_text(args: argparse.Namespace) -> int:
    """Decode both text rows of the selected dialogues, as a second channel-identity check."""
    from huggingface_hub import hf_hub_download
    from sentencepiece import SentencePieceProcessor

    tokenizer = SentencePieceProcessor(
        hf_hub_download(args.text_tokenizer_repo, args.text_tokenizer_name)
    )
    rows = read_parquet_columns(Path(args.parquet))
    ids = evenly_spaced(sorted(rows), args.count)
    out = []
    for dialogue_id in ids:
        entry = {"dialogue_id": dialogue_id}
        for speaker in ("A", "B"):
            text_row, _ = split_speaker_column(rows[dialogue_id][speaker])
            emitted = [t for t in text_row if t not in (args.padding_id, args.end_padding_id)]
            entry[speaker] = tokenizer.decode(emitted)
        out.append(entry)
        print(dialogue_id, entry["A"][:40], "|", entry["B"][:40])
    Path(args.report).write_text(
        json.dumps({"parquet": str(args.parquet), "dialogues": out}, ensure_ascii=False, indent=1)
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.report}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    decode = sub.add_parser("decode", help="decode parquet rows back to WAV")
    decode.add_argument("--parquet", required=True)
    decode.add_argument("--source-dir", required=True, help="the stereo the tokens came from")
    decode.add_argument("--out-dir", required=True)
    decode.add_argument("--report", required=True)
    decode.add_argument("--count", type=int, default=8)
    decode.add_argument("--device", default="cpu")
    decode.set_defaults(func=_cmd_decode)

    compare = sub.add_parser("compare", help="measure decode against source")
    compare.add_argument("--decoded-dir", required=True)
    compare.add_argument("--source-dir", required=True)
    compare.add_argument("--report", required=True)
    compare.set_defaults(func=_cmd_compare)

    silence = sub.add_parser("silence", help="in-situ silence-token stats for one speaker")
    silence.add_argument(
        "--dataset",
        action="append",
        nargs=3,
        metavar=("LABEL", "PARQUET", "SOURCE_DIR"),
        required=True,
    )
    silence.add_argument("--speaker", default="A", choices=["A", "B"])
    silence.add_argument("--min-distinct", type=int, default=35)
    silence.add_argument("--max-top-share", type=float, default=0.35)
    silence.add_argument("--report", required=True)
    silence.set_defaults(func=_cmd_silence)

    codec = sub.add_parser(
        "codec-band", help="re-measure the calibration band after a Mimi round trip"
    )
    codec.add_argument("--clips-dir", required=True, help="the target speaker's held-out clips")
    codec.add_argument("--out-dir", required=True)
    codec.add_argument("--report", required=True)
    codec.add_argument("--model", default="speechbrain/spkrec-ecapa-voxceleb")
    codec.add_argument("--device", default="cpu")
    codec.set_defaults(func=_cmd_codec_band)

    text = sub.add_parser("text", help="decode the two text rows of the selected dialogues")
    text.add_argument("--parquet", required=True)
    text.add_argument("--report", required=True)
    text.add_argument("--count", type=int, default=8)
    text.add_argument("--padding-id", type=int, default=3)
    text.add_argument("--end-padding-id", type=int, default=0)
    text.add_argument("--text-tokenizer-repo", default="nu-dialogue/j-moshi-ext")
    text.add_argument("--text-tokenizer-name", default="tokenizer_spm_32k_3.model")
    text.set_defaults(func=_cmd_text)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
