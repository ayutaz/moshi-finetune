"""Detect the failures that a falling loss hides - in the text stream AND in the audio.

The past depformer-only run drove audio loss from 6.88 to 1.02 while the model stopped
listening: it talked over the user, repeated "ありがとうございました" and collapsed into a
generic voice. Every number being collected at the time looked like success. M3 completion
condition 3 asks for the absence of exactly that, and nothing in the repository could
measure it - a grep for repeat, repetition, monologue, loop and collapse across `tools/`,
`models/`, `utils/` and the two scripts finds only a tensor `.repeat()` and a CSS rule.

This reads the generated token array directly, so it needs no model, no decoder and no GPU.
The array is (17, frames): row 0 is the text stream, rows 1-8 are speaker A's Mimi
codebooks 0-7, rows 9-16 are speaker B's.

**The text stream alone is not enough, and M3 was judged on it alone.** Two failures live
there:

- a loop shows up as one n-gram repeated far past anything natural;
- a monologue shows up as text on nearly every frame, never yielding the floor.

A third lives only in the audio. In 17 of the control's 30 general30 generations, speaker
A's codebook 0 is the Mimi silence token 1316 on 113 of 124 frames - the model emits two to
eight text tokens over ten seconds with a dead voice channel underneath. The text-side
`silent` test is a knife edge at zero emitted tokens, so those two stray tokens were enough
for the whole run to be recorded as `silent_count: 0`. The baseline the entire M3 verdict
was measured against was itself collapsed, and the detector could not see it, because the
detector never looked at row 1. `summarise_acoustics` looks.

The scoring functions take plain sequences and import nothing outside the standard library,
so the suite runs without numpy or torch. Only `load_generation` and `load_streams` touch
the filesystem.

Thresholds are NOT chosen here. They are calibrated once, recorded in
`reports/m3-collapse-calibration.json` (text) and
`reports/m3-collapse-acoustic-calibration.json` (audio), and passed in - because a threshold
picked after seeing the candidate is indistinguishable from a threshold picked to admit it.
`AcousticThresholds` therefore has no defaults: `CollapseThresholds()` hands back 0.3 and
0.95 while the frozen text calibration says 0.4 and 0.85, so a caller who forgets to pass
thresholds silently gets a laxer detector than the one the report was written against. The
audio half cannot be constructed without saying which numbers it is using.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

TEXT_ROW = 0
CODEBOOKS_PER_SPEAKER = 8
SPEAKER_A_CODEBOOK0_ROW = 1
SPEAKER_B_CODEBOOK0_ROW = 1 + CODEBOOKS_PER_SPEAKER


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


@dataclass(frozen=True)
class AcousticThresholds:
    """The audio-side numbers, fixed in `reports/m3-collapse-acoustic-calibration.json`.

    No defaults, deliberately: see the module docstring. Both are ceilings, and either one
    on its own condemns a generation, because the two statistics fail on different shapes.
    A window dominated by one token with a long tail of singletons keeps its distinct count
    up while its entropy falls; a window of five tokens spread more evenly keeps its entropy
    up while its distinct count falls. The calibration measured real examples of each.
    """

    max_entropy_bits: float
    max_distinct_tokens: int


def token_frequencies(tokens: Sequence[int]) -> dict[int, int]:
    """How often each token occurs. The one place the audio statistics get their counts."""
    if len(tokens) == 0:
        raise ValueError("cannot score an empty token sequence")
    counts: dict[int, int] = {}
    for token in tokens:
        key = int(token)
        counts[key] = counts.get(key, 0) + 1
    return counts


def distinct_token_count(tokens: Sequence[int]) -> int:
    """How many different tokens the window visits.

    Unlike `distinct_ratio` this is NOT divided by the length. The audio row has one token
    per frame with no padding to strip, so every window of a generation is the same length
    and the raw count is the quantity the calibration band is expressed in.
    """
    return len(token_frequencies(tokens))


def top_token(tokens: Sequence[int]) -> tuple[int, float]:
    """The most frequent token and the share of frames it occupies.

    Which token it is matters as much as the share. Mimi's codebook-0 silence token is
    1316: it fills 91.13% of the teacher-forced silent partner stream, so a speaker-A row
    whose top token is 1316 at a 0.9 share is not speaking, whatever the text row says.
    Ties go to the smaller token id so the answer does not depend on dict ordering.
    """
    counts = token_frequencies(tokens)
    best = max(counts, key=lambda token: (counts[token], -token))
    return best, counts[best] / len(tokens)


def top_token_share(tokens: Sequence[int]) -> float:
    """Share of frames held by the single most frequent token."""
    return top_token(tokens)[1]


def token_entropy_bits(tokens: Sequence[int]) -> float:
    """Shannon entropy of the token distribution, in bits.

    This is the statistic the collapse threshold is set on, because it is the only one of
    the three with a scale that means something on its own: 2**entropy is the effective
    number of tokens the window uses. A healthy generation from a working checkpoint runs
    at 2**4.95 ~ 31 effective tokens over 124 frames. A collapsed one runs at 2**0.53 ~ 1.4,
    which is to say it is one token with a rounding error attached.
    """
    counts = token_frequencies(tokens)
    total = len(tokens)
    return -sum((n / total) * math.log2(n / total) for n in counts.values())


def effective_vocabulary(tokens: Sequence[int]) -> float:
    """2**entropy: the token count a uniform window would need to be this varied.

    Reported alongside the entropy because a reader can judge "1.4 tokens" without having
    to exponentiate a log scale in their head, and because the threshold's headroom is a
    ratio in this unit rather than a difference.
    """
    return 2.0 ** token_entropy_bits(tokens)


def summarise_acoustics(
    codebook0_row: Sequence[int], *, thresholds: AcousticThresholds
) -> dict[str, Any]:
    """Score one speaker's coarse audio token row and decide whether it went dead.

    Codebook 0 alone. It carries the coarsest acoustic content, so a channel that has
    collapsed there has collapsed; the finer codebooks can still look busy while the sound
    is a constant, which is exactly the way a collapse would hide from a naive average over
    all eight.
    """
    token, share = top_token(codebook0_row)
    entropy = token_entropy_bits(codebook0_row)
    distinct = distinct_token_count(codebook0_row)
    return {
        "frames": len(codebook0_row),
        "distinct_tokens": distinct,
        "top_token": token,
        "top_token_share": share,
        "entropy_bits": entropy,
        "effective_vocabulary": 2.0**entropy,
        "acoustic_collapse": (
            entropy <= thresholds.max_entropy_bits or distinct <= thresholds.max_distinct_tokens
        ),
    }


def acoustic_thresholds_from_calibration(calibration: Mapping[str, Any]) -> AcousticThresholds:
    """Read the frozen numbers out of a parsed calibration document.

    Takes the already-parsed mapping rather than a path, so the module still touches no
    files above `load_generation`. Every key is required: a calibration missing a threshold
    is a calibration that never fixed it, and defaulting the gap would be the failure this
    whole module exists to prevent.
    """
    try:
        thresholds = calibration["thresholds"]
        return AcousticThresholds(
            max_entropy_bits=float(thresholds["max_entropy_bits"]),
            max_distinct_tokens=int(thresholds["max_distinct_tokens"]),
        )
    except KeyError as missing:
        raise ValueError(f"calibration is missing {missing}") from missing


def speaker_codebook0(streams: Sequence[Sequence[int]], *, row: int) -> Sequence[int]:
    """One codebook-0 row out of a (17, frames) generation, with the shape checked.

    The shape check is the point. M3's detector read `tokens[0]` and would have read it
    just as happily out of a 9-row array or a transposed one; naming the row it wants and
    refusing an array that cannot have it is what stops the next silent misread.
    """
    if row < 0:
        raise ValueError(f"row must be non-negative, got {row}")
    if len(streams) <= row:
        raise ValueError(
            f"expected at least {row + 1} streams to read row {row}, got {len(streams)}"
        )
    if len(streams[row]) == 0:
        raise ValueError(f"stream row {row} is empty")
    return streams[row]


def summarise_streams(
    streams: Sequence[Sequence[int]],
    *,
    padding_id: int,
    end_padding_id: int,
    thresholds: CollapseThresholds,
    acoustic_thresholds: AcousticThresholds,
) -> dict[str, Any]:
    """Score one generation on both streams and merge the verdicts.

    The merge is a disjunction, and it has to be. The text failures and the audio failure
    are not different views of one thing: over the 550 M3 generations, 21 are acoustically
    collapsed while every text test passes, and 105 fail a text test with a healthy audio
    row. Requiring both would have certified all 126.

    `acoustic_only` names the 21. It is not a failure class of its own - it is the size of
    the blind spot the text-only detector had, and it belongs in the report next to the
    counts so nobody has to rediscover it.
    """
    text_row = speaker_codebook0(streams, row=TEXT_ROW)
    audio_row = speaker_codebook0(streams, row=SPEAKER_A_CODEBOOK0_ROW)
    if len(text_row) != len(audio_row):
        raise ValueError(
            f"text stream has {len(text_row)} frames but speaker A's codebook 0 has "
            f"{len(audio_row)}; these are the same timeline and must agree"
        )
    summary = summarise_generation(
        text_row,
        padding_id=padding_id,
        end_padding_id=end_padding_id,
        thresholds=thresholds,
    )
    acoustics = summarise_acoustics(audio_row, thresholds=acoustic_thresholds)
    summary.update({f"audio_{key}": value for key, value in acoustics.items() if key != "frames"})
    summary["acoustic_collapse"] = acoustics["acoustic_collapse"]
    summary["text_failure"] = bool(
        summary["monologue_loop"] or summary["exact_repeat_collapse"] or summary["silent"]
    )
    summary["acoustic_only"] = bool(summary["acoustic_collapse"] and not summary["text_failure"])
    return summary


def verdict_for(summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one run's generations into the counts condition 3 is judged on.

    A run scored without the audio cannot pass. `acoustic_scored` is false when the
    summaries carry no `acoustic_collapse` key, and `passes` is false with it - because the
    only thing this detector is asked for is the certified ABSENCE of collapse, and a
    detector that did not look at half the signal cannot certify absence of anything. That
    is not hypothetical: the M3 verdict was computed from text-only summaries and recorded
    the control as `silent_count: 0` while 17 of its 30 general30 generations had a dead
    voice channel.
    """
    summaries = list(summaries)
    if not summaries:
        raise ValueError("at least one generation is required")
    scored = [s for s in summaries if "acoustic_collapse" in s]
    if scored and len(scored) != len(summaries):
        raise ValueError(
            f"{len(scored)} of {len(summaries)} summaries carry an acoustic verdict; "
            "mixing scored and unscored generations would report a count against the wrong "
            "denominator"
        )
    acoustic_scored = bool(scored)
    monologue = sum(1 for s in summaries if s["monologue_loop"])
    repeats = sum(1 for s in summaries if s["exact_repeat_collapse"])
    silent = sum(1 for s in summaries if s["silent"])
    acoustic = sum(1 for s in summaries if s.get("acoustic_collapse"))

    def failed(summary: dict[str, Any]) -> bool:
        return bool(
            summary["monologue_loop"]
            or summary["exact_repeat_collapse"]
            or summary["silent"]
            or summary.get("acoustic_collapse")
        )

    return {
        "total": len(summaries),
        "monologue_loop_count": monologue,
        "exact_repeat_collapse_count": repeats,
        "silent_count": silent,
        "acoustic_scored": acoustic_scored,
        "acoustic_collapse_count": acoustic if acoustic_scored else None,
        # The size of the blind spot, not a fourth failure class: generations whose audio
        # died while every text test passed. 21 of M3's 550, 16 of them the control's.
        "acoustic_only_count": (
            sum(
                1
                for s in summaries
                if s.get("acoustic_collapse")
                and not (s["monologue_loop"] or s["exact_repeat_collapse"] or s["silent"])
            )
            if acoustic_scored
            else None
        ),
        # Generations with at least one failure, counted ONCE each. A silent generation
        # trips neither collapse flag - summarise_generation returns early with both False -
        # so quoting the two collapse counts alone makes a mute checkpoint look clean, and
        # the absolute bar gets EASIER the more an arm degrades. Summing the counts instead
        # would double-count a generation that both repeats and monologues and can exceed
        # the total, which is worse than the problem it fixes.
        "degenerate_count": sum(1 for s in summaries if failed(s)),
        "passes": (
            acoustic_scored and monologue == 0 and repeats == 0 and silent == 0 and acoustic == 0
        ),
    }


def load_generation(path: str) -> list[int]:
    """Read the text stream out of a generated token file.

    Kept separate from the scoring so the rest of this module needs no numpy.
    """
    return load_streams(path)[TEXT_ROW]


def load_streams(path: str) -> list[list[int]]:
    """Read every stream out of a generated token file as plain lists.

    Kept separate from the scoring so the rest of this module needs no numpy. The row count
    is checked here rather than at the call site: a (17, frames) array is what dep_q=16
    produces, and anything else means the caller is about to read speaker A's voice out of
    a row that is not it.
    """
    import numpy as np

    tokens = np.load(path)
    if tokens.ndim != 2:
        raise ValueError(f"{path}: expected a 2-D (streams, frames) array, got {tokens.shape}")
    expected = 1 + 2 * CODEBOOKS_PER_SPEAKER
    if tokens.shape[0] != expected:
        raise ValueError(
            f"{path}: expected {expected} streams (text + two speakers x "
            f"{CODEBOOKS_PER_SPEAKER} codebooks), got {tokens.shape[0]}"
        )
    return [[int(value) for value in row] for row in tokens]


def score_directory(
    directory: str,
    *,
    padding_id: int,
    end_padding_id: int,
    thresholds: CollapseThresholds,
    acoustic_thresholds: AcousticThresholds,
) -> list[dict[str, Any]]:
    """Score every `*.npy` generation in one directory, ordered by numeric filename.

    Touches the filesystem, so it lives down here with the other loaders.
    """
    import pathlib

    paths = sorted(
        pathlib.Path(directory).glob("*.npy"),
        key=lambda path: (0, int(path.stem)) if path.stem.isdigit() else (1, path.stem),
    )
    if not paths:
        raise ValueError(f"{directory}: no generations to score")
    summaries = []
    for path in paths:
        summary = summarise_streams(
            load_streams(str(path)),
            padding_id=padding_id,
            end_padding_id=end_padding_id,
            thresholds=thresholds,
            acoustic_thresholds=acoustic_thresholds,
        )
        summary["generation"] = path.name
        summaries.append(summary)
    return summaries


def _spread(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {"min": ordered[0], "median": statistics.median(ordered), "max": ordered[-1]}


def group_report(summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """One group's verdict with the distributions a reader needs to check it."""
    report = verdict_for(summaries)
    report["audio_distinct_tokens"] = _spread([s["audio_distinct_tokens"] for s in summaries])
    report["audio_top_token_share"] = _spread([s["audio_top_token_share"] for s in summaries])
    report["audio_entropy_bits"] = _spread([s["audio_entropy_bits"] for s in summaries])
    counts: dict[int, int] = {}
    for summary in summaries:
        counts[summary["audio_top_token"]] = counts.get(summary["audio_top_token"], 0) + 1
    report["modal_top_token"] = max(counts, key=lambda token: (counts[token], -token))
    report["collapsed_generations"] = [s["generation"] for s in summaries if s["acoustic_collapse"]]
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Re-score a tree of generations and print the numbers as JSON.

    Only the computed part. The narrative belongs in the report the numbers are quoted
    into, not in the tool that produces them.
    """
    import argparse
    import json
    import pathlib

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        required=True,
        help="directory containing generated_tokens/ folders at any depth; each one is a"
        " group, named by its path relative to the root",
    )
    parser.add_argument("--text-calibration", required=True)
    parser.add_argument("--acoustic-calibration", required=True)
    parser.add_argument("--output", help="write here instead of stdout")
    args = parser.parse_args(argv)

    text_calibration = json.loads(pathlib.Path(args.text_calibration).read_text(encoding="utf-8"))
    acoustic_calibration = json.loads(
        pathlib.Path(args.acoustic_calibration).read_text(encoding="utf-8")
    )
    thresholds = CollapseThresholds(**text_calibration["thresholds"])
    acoustic_thresholds = acoustic_thresholds_from_calibration(acoustic_calibration)
    source = text_calibration["calibration_source"]

    root = pathlib.Path(args.root)
    directories = sorted(root.rglob("generated_tokens"))
    if not directories:
        raise SystemExit(f"{root}: no generated_tokens directories found")

    groups: dict[str, Any] = {}
    for directory in directories:
        summaries = score_directory(
            str(directory),
            padding_id=source["text_padding_id"],
            end_padding_id=source["end_of_text_padding_id"],
            thresholds=thresholds,
            acoustic_thresholds=acoustic_thresholds,
        )
        groups[str(directory.parent.relative_to(root))] = group_report(summaries)

    document = {
        "root": str(root),
        "text_calibration": args.text_calibration,
        "acoustic_calibration": args.acoustic_calibration,
        "thresholds": {
            "text": text_calibration["thresholds"],
            "audio": {
                "max_entropy_bits": acoustic_thresholds.max_entropy_bits,
                "max_distinct_tokens": acoustic_thresholds.max_distinct_tokens,
            },
        },
        "groups": groups,
        "totals": {
            "groups": len(groups),
            "generations": sum(g["total"] for g in groups.values()),
            "degenerate": sum(g["degenerate_count"] for g in groups.values()),
            "acoustic_collapse": sum(g["acoustic_collapse_count"] for g in groups.values()),
            "acoustic_only": sum(g["acoustic_only_count"] for g in groups.values()),
        },
    }
    rendered = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        pathlib.Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    raise SystemExit(main())
