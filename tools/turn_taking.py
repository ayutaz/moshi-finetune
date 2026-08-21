"""Whether the two channels of a generated conversation take turns.

M3 completion condition 5 asks that turn-taking stay within an acceptable range, and the
repository had no way to measure it. The failure this exists to rule out is the one the
past depformer-only run produced: the model talks continuously and the partner never gets
the floor. A loss curve cannot show that, and neither can speaker similarity.

Frames are 1920 samples at 24 kHz, so one frame is 80 ms and maps 1:1 onto a Mimi frame.
Activity uses the same RMS-over-threshold rule as `tools/speaker_similarity.py`, and the
threshold is recorded in the report rather than left in a constant, because Mimi's decoded
silence is not digital zero and the right cut depends on it.

Everything here takes plain sequences and imports nothing, so the suite runs without numpy
or torch. Only `channel_masks` touches a file.

One warning about reading these numbers. Under `--model_user_stream` the user stream is
teacher-forced from the prompt for every arm, so `user_never_active` describes the PROMPT,
never the model's behaviour. It is useful for verifying that a prompt was built correctly
and is worthless as evidence that a checkpoint listens. `reports/m3-collapse-calibration.json`
records the same caution.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from typing import Any

FRAME_SAMPLES = 1920  # 80 ms at 24 kHz, one Mimi frame
DEFAULT_THRESHOLD = 500


def activity_mask(samples: Sequence[float], *, frame: int, threshold: float) -> list[bool]:
    """One flag per frame: did this frame carry speech?"""
    if frame < 1:
        raise ValueError(f"frame must be at least 1 sample, got {frame}")
    mask: list[bool] = []
    for start in range(0, len(samples), frame):
        window = samples[start : start + frame]
        if not window:
            continue
        energy = math.sqrt(sum(float(v) * float(v) for v in window) / len(window))
        mask.append(energy >= threshold)
    return mask


def runs_of_activity(mask: Sequence[bool]) -> list[tuple[int, int]]:
    """Contiguous active spans as half-open (start, end) frame indices."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, active in enumerate(mask):
        if active and start is None:
            start = index
        elif not active and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def response_latencies(
    *, partner_runs: Sequence[tuple[int, int]], own_runs: Sequence[tuple[int, int]]
) -> list[int]:
    """Frames between each partner turn ending and the next time this side starts.

    Speech already underway when the partner stops is not a response, so only onsets at or
    after the partner's end count. A partner turn nobody answers contributes nothing rather
    than a zero, which would drag the mean towards "instant".
    """
    latencies: list[int] = []
    for _, partner_end in partner_runs:
        onsets = [start for start, _ in own_runs if start >= partner_end]
        if onsets:
            latencies.append(min(onsets) - partner_end)
    return latencies


def summarise_turn_taking(
    *, moshi_mask: Sequence[bool], user_mask: Sequence[bool]
) -> dict[str, Any]:
    """Score one conversation's turn-taking behaviour."""
    if len(moshi_mask) != len(user_mask):
        raise ValueError(
            f"channels must be the same length, got {len(moshi_mask)} and {len(user_mask)}"
        )
    if not moshi_mask:
        raise ValueError("cannot score an empty conversation")

    frames = len(moshi_mask)
    moshi_runs = runs_of_activity(moshi_mask)
    user_runs = runs_of_activity(user_mask)

    # A switch is a frame where the side holding the floor alone changes. Overlapping and
    # silent frames hold the previous holder, so a pause inside one turn is not a switch.
    switches = 0
    holder: str | None = None
    for m, u in zip(moshi_mask, user_mask, strict=True):
        current = "moshi" if m and not u else "user" if u and not m else None
        if current is not None:
            if holder is not None and current != holder:
                switches += 1
            holder = current

    latencies = response_latencies(partner_runs=user_runs, own_runs=moshi_runs)
    return {
        "frames": frames,
        "moshi_speech_ratio": sum(moshi_mask) / frames,
        "user_speech_ratio": sum(user_mask) / frames,
        "overlap_frames": sum(1 for m, u in zip(moshi_mask, user_mask, strict=True) if m and u),
        "speaker_switches": switches,
        "moshi_turns": len(moshi_runs),
        "user_turns": len(user_runs),
        "user_never_active": not any(user_mask),
        "response_latency_frames": {
            "count": len(latencies),
            "mean": statistics.fmean(latencies) if latencies else None,
            "median": statistics.median(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "longest_moshi_run_frames": max((end - start for start, end in moshi_runs), default=0),
    }


def channel_masks(
    path: str, *, frame: int = FRAME_SAMPLES, threshold: float = DEFAULT_THRESHOLD
) -> tuple[list[bool], list[bool]]:
    """Activity masks for the two channels of a decoded stereo conversation.

    Channel 0 is the Moshi stream and channel 1 the user stream, matching the convention
    `tools/tokenize_audio.py` writes and `tools/decode_tokens.py` reads back.
    """
    import soundfile

    data, _ = soundfile.read(path, always_2d=True)
    if data.shape[1] != 2:
        raise ValueError(f"{path}: expected stereo, got {data.shape[1]} channel(s)")
    scale = 32767.0 if data.dtype.kind == "f" else 1.0
    moshi = [float(row[0]) * scale for row in data]
    user = [float(row[1]) * scale for row in data]
    return (
        activity_mask(moshi, frame=frame, threshold=threshold),
        activity_mask(user, frame=frame, threshold=threshold),
    )
