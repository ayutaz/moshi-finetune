"""Build and check the 80 dialogue scripts V-real and V-tts share.

Each dialogue is three turns - B opens, A speaks one corpus sentence, B closes - and both
datasets use the same scripts, so that V0 and V1 differ in exactly one thing: the bytes of
channel A. A speaks the corpus sentence VERBATIM, because in V-real the audio for that turn
IS the corpus recording; any edit to the text desynchronises it from the recording.

Two things this has to get right before any audio is rendered.

The A texts must equal the 80 train sentences as a SET. Counting them is not enough: eighty
rows where one sentence appears twice and another never appears counts eighty, trains on a
sentence twice, and silently drops the other. Set equality catches it, a count does not.

Dialogue length has to clear the floor. A's duration is fixed by the corpus (2.24 s to
13.45 s), so only B's turns can be sized, and they have to be written long enough before
they are synthesised. Projection uses mora count over a measured speaking rate, and the
gate assumes the FAST end of the rate: a voice that speaks faster than projected lands
shorter than planned, which is the direction that breaks the floor.

Measured rates, for whoever sets `mora_per_second`:

- tsukuyomi corpus, 80 sentences: 6.475 mora/s pooled, mean 6.589, stdev 0.717
- speaker B under `--ref-wav` (the frozen reference M3 uses): about 6.13 mora/s
- speaker B under `--no-ref`: about 4.91 mora/s, notably slower - the reference conditions
  the duration predictor, so a rate measured without one does not apply

The pure functions import nothing; `mora_count` loads pyopenjtalk lazily.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

# Small kana do not carry their own mora; they combine with the preceding one.
_SMALL_KANA = set("ャュョァィゥェォ")
_KANA_ONLY = re.compile(r"[^ァ-ヴー]")


@dataclass(frozen=True)
class TimelineSpec:
    """How turns are laid out on the dialogue timeline."""

    lead_in_seconds: float = 0.5
    gap_seconds: float = 0.4
    frame_rate_hz: float = 12.5


def mora_count(text: str) -> int:
    """Mora in `text`, via pyopenjtalk's kana reading."""
    import pyopenjtalk

    kana = _KANA_ONLY.sub("", pyopenjtalk.g2p(text, kana=True))
    return sum(1 for char in kana if char not in _SMALL_KANA)


def project_seconds(mora: int, *, mora_per_second: float) -> float:
    """How long `mora` takes at a given speaking rate."""
    if mora_per_second <= 0:
        raise ValueError(f"mora_per_second must be positive, got {mora_per_second}")
    return mora / mora_per_second


def project_frames(seconds: float, *, frame_rate_hz: float) -> int:
    """Mimi frames in `seconds`, rounded DOWN so the floor gate never overstates length."""
    if frame_rate_hz <= 0:
        raise ValueError(f"frame_rate_hz must be positive, got {frame_rate_hz}")
    return int(seconds * frame_rate_hz)


def project_dialogue_seconds(turn_seconds: Sequence[float], *, spec: TimelineSpec) -> float:
    """Total wall-clock length of a dialogue laid out sequentially with fixed gaps."""
    turns = list(turn_seconds)
    if not turns:
        raise ValueError("a dialogue needs at least one turn")
    return spec.lead_in_seconds + sum(turns) + spec.gap_seconds * (len(turns) - 1)


def _normalise(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip()


def validate_scripts(
    rows: Sequence[Mapping[str, Any]],
    *,
    corpus_texts: Mapping[str, str],
    eval_texts: Iterable[str],
) -> dict[str, Any]:
    """Check the script set against everything that has to hold before any audio exists."""
    eval_set = {_normalise(text) for text in eval_texts}

    seen_ids: set[str] = set()
    duplicate_ids: list[str] = []
    bad_structure: list[str] = []
    altered_a: list[dict[str, str]] = []
    empty_turns: list[str] = []
    eval_overlaps: list[dict[str, str]] = []
    used_artifacts: list[str] = []

    for row in rows:
        dialogue_id = row.get("dialogue_id", "<missing>")
        if dialogue_id in seen_ids:
            duplicate_ids.append(dialogue_id)
        seen_ids.add(dialogue_id)

        turns = row.get("turns") or []
        if [turn.get("speaker") for turn in turns] != ["B", "A", "B"]:
            bad_structure.append(dialogue_id)
            continue

        artifact_id = row.get("source_artifact_id", "")
        used_artifacts.append(artifact_id)

        expected = corpus_texts.get(artifact_id)
        actual = turns[1].get("text", "")
        if expected is None or _normalise(actual) != _normalise(expected):
            altered_a.append({"dialogue_id": dialogue_id, "artifact_id": artifact_id})

        for index, turn in enumerate(turns):
            text = turn.get("text", "")
            if not _normalise(text):
                empty_turns.append(f"{dialogue_id}#{index}")
            elif turn["speaker"] == "B" and _normalise(text) in eval_set:
                eval_overlaps.append({"dialogue_id": dialogue_id, "text": text})

    a_texts_match = set(used_artifacts) == set(corpus_texts) and len(used_artifacts) == len(
        corpus_texts
    )

    failures = {
        "duplicate_dialogue_ids": sorted(set(duplicate_ids)),
        "bad_turn_structure": bad_structure,
        "altered_a_turns": altered_a,
        "empty_turns": empty_turns,
        "eval_overlaps": eval_overlaps,
    }
    passed = a_texts_match and not any(failures.values())
    return {
        "status": "pass" if passed else "fail",
        "rows": len(rows),
        "a_texts_match_corpus": a_texts_match,
        "corpus_sentences": len(corpus_texts),
        "eval_overlap_count": len(eval_overlaps),
        **failures,
    }
