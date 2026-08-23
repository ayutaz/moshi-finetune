"""Detect the two failures that a falling loss hides.

The past depformer-only run drove audio loss from 6.88 to 1.02 while the model stopped
listening: it talked over the user, repeated "ありがとうございました" and collapsed into a
generic voice. Every number being collected at the time looked like success. M3 completion
condition 3 asks for the absence of exactly that, and nothing in the repository could
measure it - a grep for repeat, repetition, monologue, loop and collapse across `tools/`,
`models/`, `utils/` and the two scripts finds only a tensor `.repeat()` and a CSS rule.

This reads the generated token array directly, so it needs no model, no decoder and no GPU.
Row 0 is the text stream; padding frames are the ones where the model chose to say nothing,
which makes the text stream the cheapest place to see both failures:

- a loop shows up as one n-gram repeated far past anything natural;
- a monologue shows up as text on nearly every frame, never yielding the floor.

The scoring functions take plain sequences and import nothing, so the suite runs without
numpy or torch. Only `load_generation` touches the filesystem.

Thresholds are NOT chosen here. They are calibrated once, recorded in
`reports/m3-collapse-calibration.json`, and passed in - because a threshold picked after
seeing the candidate is indistinguishable from a threshold picked to admit it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

TEXT_ROW = 0


class RepeatedNgram(NamedTuple):
    ngram: tuple[int, ...]
    count: int


@dataclass(frozen=True)
class CollapseThresholds:
    """The numbers that decide a verdict, fixed before any candidate is scored."""

    min_n: int = 3
    max_ngram_repeats: int = 4
    min_distinct_ratio: float = 0.3
    max_emitted_ratio: float = 0.95


def emitted_text_tokens(
    text_row: Sequence[int], *, padding_id: int, end_padding_id: int
) -> list[int]:
    """The tokens the model actually emitted, in order, with padding removed."""
    return [int(token) for token in text_row if token not in (padding_id, end_padding_id)]


def emitted_text_ratio(text_row: Sequence[int], *, padding_id: int, end_padding_id: int) -> float:
    """Fraction of frames that carried text.

    Near 1.0 means the model never stopped talking. In a full-duplex dialogue that is the
    signature of a monologue, not of fluency.
    """
    if len(text_row) == 0:
        raise ValueError("cannot score an empty text stream")
    emitted = emitted_text_tokens(text_row, padding_id=padding_id, end_padding_id=end_padding_id)
    return len(emitted) / len(text_row)


def longest_repeated_ngram(tokens: Sequence[int], *, min_n: int) -> RepeatedNgram | None:
    """The most-repeated n-gram, preferring the longest one at that repeat count.

    The count is what the threshold is applied to, so it is maximised first: taking the
    maximum over every n is the most sensitive form of the detector.

    Length only breaks ties, and it has to break them towards the longer gram. In
    `1,2,3,4` repeated four times, `1,2` also occurs four times; reporting the 2-gram would
    describe a four-token loop as a two-token one. Reaching for the longest gram outright
    is equally wrong - there the 8-gram `1,2,3,4,1,2,3,4` occurs three times, so "longest"
    would report a period the model never had.
    """
    if min_n < 1:
        raise ValueError(f"min_n must be at least 1, got {min_n}")
    tokens = [int(token) for token in tokens]
    best: RepeatedNgram | None = None
    for n in range(min_n, len(tokens) // 2 + 1):
        counts: dict[tuple[int, ...], int] = {}
        for start in range(len(tokens) - n + 1):
            gram = tuple(tokens[start : start + n])
            counts[gram] = counts.get(gram, 0) + 1
        repeated = {gram: count for gram, count in counts.items() if count > 1}
        if not repeated:
            # No n-gram of this length repeats, so no longer one can either.
            break
        gram = max(repeated, key=lambda g: repeated[g])
        candidate = RepeatedNgram(gram, repeated[gram])
        if best is None or candidate.count >= best.count:
            best = candidate
    return best


def distinct_ratio(tokens: Sequence[int]) -> float:
    """Unique tokens over total. A model stuck on one phrase scores near zero."""
    if len(tokens) == 0:
        raise ValueError("cannot score an empty token sequence")
    return len({int(token) for token in tokens}) / len(tokens)


def summarise_generation(
    text_row: Sequence[int],
    *,
    padding_id: int,
    end_padding_id: int,
    thresholds: CollapseThresholds,
) -> dict[str, Any]:
    """Score one generation and decide whether it collapsed."""
    ratio = emitted_text_ratio(text_row, padding_id=padding_id, end_padding_id=end_padding_id)
    tokens = emitted_text_tokens(text_row, padding_id=padding_id, end_padding_id=end_padding_id)

    if not tokens:
        # Saying nothing is not health. Without this branch a silent generation scores zero
        # repeats and a zero emission ratio, and passes every other check.
        return {
            "frames": len(text_row),
            "emitted_tokens": 0,
            "emitted_ratio": 0.0,
            "distinct_ratio": None,
            "longest_repeat_n": None,
            "longest_repeat_count": None,
            "silent": True,
            "monologue_loop": False,
            "exact_repeat_collapse": False,
        }

    repeat = longest_repeated_ngram(tokens, min_n=thresholds.min_n)
    variety = distinct_ratio(tokens)
    return {
        "frames": len(text_row),
        "emitted_tokens": len(tokens),
        "emitted_ratio": ratio,
        "distinct_ratio": variety,
        "longest_repeat_n": len(repeat.ngram) if repeat else None,
        "longest_repeat_count": repeat.count if repeat else None,
        "silent": False,
        "monologue_loop": ratio > thresholds.max_emitted_ratio,
        "exact_repeat_collapse": (
            (repeat is not None and repeat.count > thresholds.max_ngram_repeats)
            or variety < thresholds.min_distinct_ratio
        ),
    }


def verdict_for(summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one run's generations into the counts condition 3 is judged on."""
    summaries = list(summaries)
    if not summaries:
        raise ValueError("at least one generation is required")
    monologue = sum(1 for s in summaries if s["monologue_loop"])
    repeats = sum(1 for s in summaries if s["exact_repeat_collapse"])
    silent = sum(1 for s in summaries if s["silent"])
    return {
        "total": len(summaries),
        "monologue_loop_count": monologue,
        "exact_repeat_collapse_count": repeats,
        "silent_count": silent,
        # Generations with at least one failure, counted ONCE each. A silent generation
        # trips neither collapse flag - summarise_generation returns early with both False -
        # so quoting the two collapse counts alone makes a mute checkpoint look clean, and
        # the absolute bar gets EASIER the more an arm degrades. Summing the three counts
        # instead would double-count a generation that both repeats and monologues and can
        # exceed the total, which is worse than the problem it fixes.
        "degenerate_count": sum(
            1
            for s in summaries
            if s["monologue_loop"] or s["exact_repeat_collapse"] or s["silent"]
        ),
        "passes": monologue == 0 and repeats == 0 and silent == 0,
    }


def load_generation(path: str) -> list[int]:
    """Read the text stream out of a generated token file.

    Kept separate from the scoring so the rest of this module needs no numpy.
    """
    import numpy as np

    tokens = np.load(path)
    if tokens.ndim != 2:
        raise ValueError(f"{path}: expected a 2-D (streams, frames) array, got {tokens.shape}")
    return [int(value) for value in tokens[TEXT_ROW]]
