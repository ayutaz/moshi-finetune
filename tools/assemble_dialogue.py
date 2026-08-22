"""Lay rendered turns onto a stereo timeline, with the word timings the text stream needs.

`tools/tokenize_audio.py` reads channel 0 as speaker A and channel 1 as speaker B, and
`tools/tokenize_text.py` wants a flat list of `{speaker, word, start, end}`. This produces
both from the same placement, so the two streams cannot disagree about when someone spoke.

Word timings are mora-weighted, not character-weighted. tokenize_text splits each segment's
duration evenly across its characters, so handing it whole utterances would spread 東寺 -
two characters, three mora - over the same span as a two-mora word and shift everything
after it. pyopenjtalk's frontend already segments and counts mora, so each word gets a
share of the turn proportional to how long it takes to say.

This is proportional allocation, not forced alignment. It is right on average and wrong in
detail; `m3/DATASET_SPEC.md` records that, and the in-voiced-frame check bounds the
consequence rather than measuring the error.

The timeline is strictly sequential - fixed lead-in, fixed gap, no overlap - so the model
never sees barge-in. That is a limitation of this dataset, written down rather than
discovered later.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

SAMPLE_RATE = 24000
FRAME_RATE_HZ = 12.5


@dataclass(frozen=True)
class TimelineSpec:
    lead_in_seconds: float = 0.5
    gap_seconds: float = 0.4
    sample_rate: int = SAMPLE_RATE
    frame_rate_hz: float = FRAME_RATE_HZ


def frames_for(seconds: float, *, frame_rate_hz: float) -> int:
    """Mimi frames covering `seconds`, rounded UP.

    Up, because this counts how many frames the audio occupies: a clip that ends a
    fraction into a frame still needs that frame. The script-length projection floors
    instead, since there the question is how much length is guaranteed.
    """
    return math.ceil(seconds * frame_rate_hz)


def allocate_word_times(
    words: Sequence[tuple[str, int]], *, start: float, end: float
) -> list[dict[str, Any]]:
    """Give each word a slice of [start, end] proportional to its mora count.

    Words with no mora - punctuation - still get a slice, because tokenize_text consumes
    the characters of every segment it is handed and a zero-length span would collide with
    its neighbour. They are weighted as a fraction of one mora so they take almost nothing.
    """
    words = list(words)
    if not words:
        raise ValueError("a turn needs at least one word")
    if end <= start:
        raise ValueError(f"turn span must be positive, got start={start} end={end}")

    weights = [max(float(mora), 0.25) for _, mora in words]
    total = sum(weights)
    span = end - start

    placed: list[dict[str, Any]] = []
    cursor = start
    for index, ((word, _), weight) in enumerate(zip(words, weights, strict=True)):
        # The last word ends exactly at `end`; accumulating floats would drift otherwise.
        finish = end if index == len(words) - 1 else cursor + span * weight / total
        placed.append({"word": word, "start": cursor, "end": finish})
        cursor = finish
    return placed


def dialogue_timeline(
    turns: Sequence[tuple[str, float]], *, spec: TimelineSpec
) -> list[dict[str, Any]]:
    """Place each (speaker, duration) turn sequentially with a fixed gap between them."""
    placed: list[dict[str, Any]] = []
    cursor = spec.lead_in_seconds
    for index, (speaker, duration) in enumerate(turns):
        if index:
            cursor += spec.gap_seconds
        placed.append({"speaker": speaker, "start": cursor, "end": cursor + duration})
        cursor += duration
    return placed


def channel_gate(*, a_rms: float, b_rms: float, ratio: float) -> dict[str, Any]:
    """Is speaker A really on channel 0?

    A left/right swap survives every other check in the pipeline: the file is still stereo,
    still the right length, still matches its transcript. It only shows up as A's voiced
    energy no longer dominating channel 0, so the gate carries a number rather than an
    assertion that someone looked.

    Silence on either channel fails too. A silent B satisfies any ratio while meaning the
    user never speaks at all.
    """
    if a_rms <= 0 or b_rms <= 0:
        return {
            "ok": False,
            "reason": "silent channel" if a_rms <= 0 else "silent user channel",
            "a_rms": a_rms,
            "b_rms": b_rms,
        }
    observed = a_rms / b_rms
    return {
        "ok": observed >= ratio,
        "reason": None if observed >= ratio else f"a/b rms {observed:.2f} below {ratio}",
        "a_rms": a_rms,
        "b_rms": b_rms,
        "ratio": observed,
    }


def word_units(text: str) -> list[tuple[str, int]]:
    """Segment `text` into words with their mora counts."""
    import pyopenjtalk

    units = []
    for entry in pyopenjtalk.run_frontend(text):
        surface = entry["string"]
        if not surface:
            continue
        units.append((surface, int(entry.get("mora_size") or 0)))
    if not units:
        raise ValueError(f"no words segmented from {text!r}")
    return units
