"""Build and check the dialogue scripts the V-real and V-tts datasets share.

M3 shipped three turns per dialogue - B opens, A speaks one corpus sentence, B closes - and
both datasets used the same scripts, so that V0 and V1 differed in exactly one thing: the
bytes of channel A. A speaks the corpus sentence VERBATIM, because in V-real the audio for
that turn IS the corpus recording; any edit to the text desynchronises it from the recording.

That shape is what M3-R has to change. With A speaking once, in the middle, on a timeline
with no overlap, channel A is silent 68.8% of the time and `text is pad` coincides with
`A is silent` for 98% of frames - a deterministic shortcut, and the mechanism the M3
verification settled on for the collapse (`docs/experiments/j-moshi-tsukuyomi-ojousama-m3-verification.md`
§3.2). `split_at_central_comma` and `assign_backchannels` below cut each A sentence in two
around a short B backchannel, which doubles A's turns WITHOUT dropping a dialogue: 72 train
dialogues stay 72, so the step count stays 45 and only the shape changes.

Splitting is a partition, never an edit. The two fragments concatenate back to the corpus
sentence character for character. If they did not, V-real's audio would no longer match its
text and the M3-R/M3 comparison would no longer be about shape alone.

Two further things this has to get right before any audio is rendered.

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

The pure functions import only the standard library; `mora_count` loads pyopenjtalk lazily.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Small kana do not carry their own mora; they combine with the preceding one.
_SMALL_KANA = set("ャュョァィゥェォ")
_KANA_ONLY = re.compile(r"[^ァ-ヴー]")

# The Japanese comma. A sentence is cut here and nowhere else: a comma is already a pause
# the reader takes, so a turn boundary on one is a boundary the recording also has.
CLAUSE_COMMA = "、"

# A fragment shorter than this is not a turn, it is a hiccup. Six characters is the floor
# the M3-R plan measured against: 70 of the 72 train sentences split with both halves at or
# above it, and the two that fail have no comma at all.
MIN_FRAGMENT_CHARS = 6

# The turn shapes this experiment has agreed to render. B-A-B is M3; B-A-B-A-B is M3-R with
# the backchannel between A's two fragments. Anything else is a script nobody has costed.
ALLOWED_SPEAKER_SHAPES: tuple[tuple[str, ...], ...] = (
    ("B", "A", "B"),
    ("B", "A", "B", "A", "B"),
)

# Short B backchannels for the seam between A's two fragments. Twelve, not one, because a
# single phrase in all 78 seams is itself a fixed pattern - one more thing that is always
# true of the training data, which is the class of defect M3-R exists to remove.
BACKCHANNEL_POOL: tuple[str, ...] = (
    "ええ。",
    "はい。",
    "なるほど。",
    "そうですか。",
    "ええ、ええ。",
    "はい、はい。",
    "ええ、なるほど。",
    "はい、なるほど。",
    "なるほど、ええ。",
    "そうなのですね。",
    "ええ、そうなのですね。",
    "なるほど、そうなのですね。",
)

# Particles that leave the listener mid-list. A cut in front of one of these is the least
# natural place to put a backchannel, so `summarise_split_points` counts them separately.
COORDINATION_PARTICLES: tuple[str, ...] = ("と", "や")

_CLAUSE_END = re.compile(r"[、。！？]")


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


def split_at_central_comma(
    text: str,
    *,
    comma: str = CLAUSE_COMMA,
    min_fragment_chars: int = MIN_FRAGMENT_CHARS,
) -> tuple[str, str] | None:
    """Cut `text` in two at the comma nearest its middle, or return None if it will not cut.

    A partition, not an edit: the comma stays on the left fragment, nothing is inserted or
    dropped, and `left + right == text` exactly. In V-real the audio for A's turn IS the
    corpus recording, so a fragment that is not a substring of the sentence has no audio to
    go with it, and the shape-only comparison against M3 stops being shape-only.

    The comma NEAREST THE MIDDLE, not the first or the last. A cut near one end leaves a
    fragment of two or three characters, and a turn that short teaches the same lesson as
    no turn at all - A appears for a moment and is silent again. `min_fragment_chars`
    refuses the ones still too lopsided rather than emitting them quietly.

    Ties go to the earlier comma so the result never depends on iteration order.

    Returns None when the sentence has no comma, or when the central comma would leave a
    fragment under the floor. Both cases mean "leave this dialogue as B-A-B"; neither is an
    error, because 2 of the 72 train sentences have no comma and must still be trained on.
    """
    if min_fragment_chars < 1:
        raise ValueError(f"min_fragment_chars must be positive, got {min_fragment_chars}")
    if len(comma) != 1:
        raise ValueError(f"comma must be a single character, got {comma!r}")

    positions = [index for index, character in enumerate(text) if character == comma]
    if not positions:
        return None

    centre = len(text) / 2
    # index + 1 is how many characters land on the left, so that is the quantity to centre.
    chosen = min(positions, key=lambda index: (abs((index + 1) - centre), index))
    left, right = text[: chosen + 1], text[chosen + 1 :]
    if len(left) < min_fragment_chars or len(right) < min_fragment_chars:
        return None
    return left, right


def leading_clause(text: str) -> str:
    """Everything in `text` before its first comma or full stop."""
    return _CLAUSE_END.split(_normalise(text), maxsplit=1)[0]


def assign_backchannels(
    followers: Sequence[str],
    *,
    pool: Sequence[str] = BACKCHANNEL_POOL,
    seed: int,
) -> list[str]:
    """One backchannel per seam, spread evenly over `pool`, ordered by `seed`.

    `followers[k]` is the turn that comes after seam k - B's closing turn in the dialogue.

    Even spread matters more than variety here. The failure this rebuild is undoing is a
    model that found something always true of the data and predicted it; one backchannel
    repeated 78 times would hand it another such thing. Each phrase is therefore used
    `n // len(pool)` or one more times, and the order is a seeded shuffle so the file can
    be regenerated byte for byte.

    A backchannel is also kept away from the turn it precedes when the two open with the
    same clause - `なるほど。` in front of `なるほど、中央に配されるのですね。` is a repetition
    inside a single dialogue, audible in a way the counts would not show. The fix is a swap
    between two seams, which changes the order and not the counts.

    Raises ValueError on an empty pool or a pool that repeats a phrase; a repeated phrase
    would quietly double that phrase's share of the seams.
    """
    if not pool:
        raise ValueError("backchannel pool must not be empty")
    if len(set(pool)) != len(pool):
        raise ValueError("backchannel pool must not repeat a phrase")

    count = len(followers)
    chosen = [pool[index % len(pool)] for index in range(count)]
    random.Random(seed).shuffle(chosen)

    heads = [leading_clause(text) for text in followers]
    for seam in range(count):
        if leading_clause(chosen[seam]) != heads[seam]:
            continue
        for other in range(seam + 1, count):
            # The swap has to fix this seam without breaking the one it borrows from.
            if (
                leading_clause(chosen[other]) != heads[seam]
                and leading_clause(chosen[seam]) != heads[other]
            ):
                chosen[seam], chosen[other] = chosen[other], chosen[seam]
                break
    return chosen


def backchannel_clashes(chosen: Sequence[str], followers: Sequence[str]) -> list[int]:
    """Seams where the backchannel and the turn after it open with the same clause.

    `assign_backchannels` tries to leave none, but with a small pool and many followers
    sharing an opening it can run out of swaps. It returns the assignment either way; this
    reports what is left so the count lands in the validation record instead of nowhere.
    """
    if len(chosen) != len(followers):
        raise ValueError(f"length mismatch: {len(chosen)} backchannels, {len(followers)} followers")
    return [
        seam
        for seam, (backchannel, follower) in enumerate(zip(chosen, followers, strict=True))
        if leading_clause(backchannel) == leading_clause(follower)
    ]


def split_dialogue_row(
    row: Mapping[str, Any],
    *,
    backchannel: str | None,
    min_fragment_chars: int = MIN_FRAGMENT_CHARS,
) -> dict[str, Any]:
    """Rebuild one B-A-B row as B-A₁-b-A₂-B, or return it unchanged when A will not split.

    `backchannel` is the B turn for the seam; pass None to leave the row three turns even
    when A could be split. Turns gain a `role` field (`open` / `body` / `backchannel` /
    `close`) that v1 did not have - `speaker` and `text` keep their name and meaning, so a
    reader that only knows v1 still reads this file correctly.

    `projected_frames_fast` is carried over untouched: it is a projection over mora, and
    only `project_script_frames` (which loads pyopenjtalk) can recompute it.
    """
    turns = list(row.get("turns") or [])
    if [turn.get("speaker") for turn in turns] != ["B", "A", "B"]:
        raise ValueError(f"{row.get('dialogue_id', '<missing>')}: expected a B-A-B row")

    rebuilt = dict(row)
    opening, middle, closing = turns
    fragments = (
        None
        if backchannel is None
        else split_at_central_comma(middle["text"], min_fragment_chars=min_fragment_chars)
    )
    if fragments is None:
        rebuilt["turns"] = [
            {"speaker": "B", "text": opening["text"], "role": "open"},
            {"speaker": "A", "text": middle["text"], "role": "body"},
            {"speaker": "B", "text": closing["text"], "role": "close"},
        ]
        return rebuilt

    left, right = fragments
    rebuilt["turns"] = [
        {"speaker": "B", "text": opening["text"], "role": "open"},
        {"speaker": "A", "text": left, "role": "body"},
        {"speaker": "B", "text": backchannel, "role": "backchannel"},
        {"speaker": "A", "text": right, "role": "body"},
        {"speaker": "B", "text": closing["text"], "role": "close"},
    ]
    return rebuilt


def project_script_frames(
    texts: Sequence[str],
    *,
    mora_per_second: float,
    spec: TimelineSpec,
    mora_of: Callable[[str], int] = mora_count,
) -> int:
    """Projected frames for one dialogue's turns, laid out sequentially with fixed gaps.

    `mora_of` is injectable so the projection can be tested without pyopenjtalk installed.
    """
    seconds = [project_seconds(mora_of(text), mora_per_second=mora_per_second) for text in texts]
    return project_frames(
        project_dialogue_seconds(seconds, spec=spec), frame_rate_hz=spec.frame_rate_hz
    )


def summarise_structure(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The structural counts the M3-R data gates are stated in.

    `a_characters` is the invariant that makes the rebuild a rebuild: splitting A moves
    characters between turns and must not add or lose one, so the total has to equal v1's
    3149 exactly. A mean turn count that moved while this total also moved would mean the
    script was rewritten, not reshaped, and M3-R would no longer be comparable with M3.
    """
    if not rows:
        raise ValueError("cannot summarise an empty script set")

    turn_counts: list[int] = []
    a_turn_counts: list[int] = []
    shapes: dict[str, int] = {}
    a_characters = 0
    b_characters = 0
    for row in rows:
        turns = list(row.get("turns") or [])
        turn_counts.append(len(turns))
        speakers = [str(turn.get("speaker")) for turn in turns]
        shapes["-".join(speakers)] = shapes.get("-".join(speakers), 0) + 1
        a_turn_counts.append(sum(1 for speaker in speakers if speaker == "A"))
        for turn in turns:
            if turn.get("speaker") == "A":
                a_characters += len(turn.get("text", ""))
            else:
                b_characters += len(turn.get("text", ""))

    histogram: dict[str, int] = {}
    for count in a_turn_counts:
        histogram[str(count)] = histogram.get(str(count), 0) + 1
    return {
        "dialogues": len(rows),
        "turns_total": sum(turn_counts),
        "turns_per_dialogue": statistics.fmean(turn_counts),
        "a_turns_total": sum(a_turn_counts),
        "a_turns_per_dialogue": statistics.fmean(a_turn_counts),
        "a_turns_histogram": dict(sorted(histogram.items())),
        "speaker_shapes": dict(sorted(shapes.items())),
        "a_characters": a_characters,
        "b_characters": b_characters,
    }


def summarise_split_points(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Where the cuts landed, counted by the character standing in front of the comma.

    A backchannel after a topic marker (`〜は、`) is about the most natural place Japanese
    has for one. A backchannel in the middle of a coordination (`〜と、`, `〜や、`) is about
    the least, because the listener answers having heard half a list. Counting both turns
    "the split points read well" from an assertion into a number somebody can re-check, and
    names the dialogues to look at first if the rendered audio sounds wrong.
    """
    tails: dict[str, int] = {}
    mid_coordination: list[str] = []
    splits = 0
    for row in rows:
        turns = list(row.get("turns") or [])
        if len(turns) < 4 or turns[1].get("speaker") != "A":
            continue
        left = str(turns[1].get("text", ""))
        if not left.endswith(CLAUSE_COMMA) or len(left) < 2:
            continue
        splits += 1
        tail = left[-2]
        tails[tail] = tails.get(tail, 0) + 1
        if tail in COORDINATION_PARTICLES:
            mid_coordination.append(str(row.get("dialogue_id", "<missing>")))
    return {
        "splits": splits,
        "preceding_character": dict(sorted(tails.items(), key=lambda item: (-item[1], item[0]))),
        "mid_coordination": {
            "particles": list(COORDINATION_PARTICLES),
            "count": len(mid_coordination),
            "dialogue_ids": mid_coordination,
        },
    }


def validate_scripts(
    rows: Sequence[Mapping[str, Any]],
    *,
    corpus_texts: Mapping[str, str],
    eval_texts: Iterable[str],
    held_out_texts: Iterable[str] = (),
    min_a_turn_chars: int = 0,
) -> dict[str, Any]:
    """Check the script set against everything that has to hold before any audio exists.

    A's turns are compared to the corpus sentence AFTER being joined back together, so a
    B-A-B row and a B-A₁-b-A₂-B row are held to the same rule: A said the corpus sentence,
    verbatim, whole. A split that dropped the comma or a fragment would show up here.

    `held_out_texts` are checked against every turn AND against the joined A text - the
    evaluation set is only worth what its separation from training is worth, and a held-out
    sentence cut in two would slip past a per-turn check. `eval_texts` stays a B-turn check:
    ten of them are the `seen` voice references, which ARE corpus train sentences and are
    supposed to appear as A.

    `min_a_turn_chars` above 0 also rejects A turns shorter than that; the splitter already
    refuses them, so this catches a script edited by hand afterwards.
    """
    eval_set = {_normalise(text) for text in eval_texts}
    held_out_set = {_normalise(text) for text in held_out_texts}

    seen_ids: set[str] = set()
    duplicate_ids: list[str] = []
    bad_structure: list[str] = []
    altered_a: list[dict[str, str]] = []
    empty_turns: list[str] = []
    eval_overlaps: list[dict[str, str]] = []
    held_out_overlaps: list[dict[str, str]] = []
    short_a_turns: list[dict[str, Any]] = []
    used_artifacts: list[str] = []

    for row in rows:
        dialogue_id = row.get("dialogue_id", "<missing>")
        if dialogue_id in seen_ids:
            duplicate_ids.append(dialogue_id)
        seen_ids.add(dialogue_id)

        turns = row.get("turns") or []
        if tuple(turn.get("speaker") for turn in turns) not in ALLOWED_SPEAKER_SHAPES:
            bad_structure.append(dialogue_id)
            continue

        artifact_id = row.get("source_artifact_id", "")
        used_artifacts.append(artifact_id)

        a_texts = [turn.get("text", "") for turn in turns if turn.get("speaker") == "A"]
        joined_a = "".join(a_texts)
        expected = corpus_texts.get(artifact_id)
        if expected is None or _normalise(joined_a) != _normalise(expected):
            altered_a.append({"dialogue_id": dialogue_id, "artifact_id": artifact_id})
        # Only when A was split: a single A turn is already covered by the per-turn check
        # below, and counting it twice would overstate the leak.
        if len(a_texts) > 1 and _normalise(joined_a) in held_out_set:
            held_out_overlaps.append({"dialogue_id": dialogue_id, "text": joined_a})

        for index, turn in enumerate(turns):
            text = turn.get("text", "")
            if not _normalise(text):
                empty_turns.append(f"{dialogue_id}#{index}")
                continue
            if _normalise(text) in held_out_set:
                held_out_overlaps.append({"dialogue_id": dialogue_id, "text": text})
            if turn["speaker"] == "B" and _normalise(text) in eval_set:
                eval_overlaps.append({"dialogue_id": dialogue_id, "text": text})
            if turn["speaker"] == "A" and 0 < len(text) < min_a_turn_chars:
                short_a_turns.append(
                    {"dialogue_id": dialogue_id, "turn": index, "characters": len(text)}
                )

    a_texts_match = set(used_artifacts) == set(corpus_texts) and len(used_artifacts) == len(
        corpus_texts
    )

    failures = {
        "duplicate_dialogue_ids": sorted(set(duplicate_ids)),
        "bad_turn_structure": bad_structure,
        "altered_a_turns": altered_a,
        "empty_turns": empty_turns,
        "eval_overlaps": eval_overlaps,
        "held_out_overlaps": held_out_overlaps,
        "short_a_turns": short_a_turns,
    }
    passed = a_texts_match and not any(failures.values())
    return {
        "status": "pass" if passed else "fail",
        "rows": len(rows),
        "a_texts_match_corpus": a_texts_match,
        "corpus_sentences": len(corpus_texts),
        "eval_overlap_count": len(eval_overlaps),
        "held_out_overlap_count": len(held_out_overlaps),
        "held_out_texts_compared": len(held_out_set),
        **failures,
    }


def build_v2_scripts(
    rows: Sequence[Mapping[str, Any]],
    *,
    pool: Sequence[str] = BACKCHANNEL_POOL,
    seed: int,
    min_fragment_chars: int = MIN_FRAGMENT_CHARS,
) -> dict[str, Any]:
    """Rebuild a set of B-A-B rows as B-A₁-b-A₂-B, and record how each one was rebuilt.

    Pure: no file is read, no clock is called, nothing but `seed` decides the backchannel
    order. `projected_frames_fast` is carried over from the input rows and has to be
    recomputed by the caller, which is the only part that needs pyopenjtalk.

    Rows whose A sentence will not split stay three turns. That is not a fallback to
    apologise for - dropping them would drop two of the 72 train dialogues, and the whole
    point of splitting rather than pairing sentences was to keep the step count at 45.
    """
    seams: list[dict[str, Any]] = []
    unsplit: list[dict[str, Any]] = []
    fragments_by_id: dict[str, tuple[str, str]] = {}

    for row in rows:
        dialogue_id = str(row.get("dialogue_id", "<missing>"))
        turns = list(row.get("turns") or [])
        if [turn.get("speaker") for turn in turns] != ["B", "A", "B"]:
            raise ValueError(f"{dialogue_id}: expected a B-A-B row, got {len(turns)} turns")
        sentence = turns[1].get("text", "")
        fragments = split_at_central_comma(sentence, min_fragment_chars=min_fragment_chars)
        if fragments is None:
            unsplit.append(
                {
                    "dialogue_id": dialogue_id,
                    "reason": (
                        "no-comma"
                        if CLAUSE_COMMA not in sentence
                        else "central-comma-leaves-a-fragment-under-the-floor"
                    ),
                    "text": sentence,
                    "characters": len(sentence),
                }
            )
            continue
        fragments_by_id[dialogue_id] = fragments
        seams.append(
            {
                "dialogue_id": dialogue_id,
                "left_characters": len(fragments[0]),
                "right_characters": len(fragments[1]),
                "follower": turns[2].get("text", ""),
            }
        )

    followers = [seam["follower"] for seam in seams]
    chosen = assign_backchannels(followers, pool=pool, seed=seed)
    clashes = backchannel_clashes(chosen, followers)
    backchannel_by_id = {
        seam["dialogue_id"]: backchannel for seam, backchannel in zip(seams, chosen, strict=True)
    }
    for seam, backchannel in zip(seams, chosen, strict=True):
        seam["backchannel"] = backchannel
        del seam["follower"]

    rebuilt = [
        split_dialogue_row(
            row,
            backchannel=backchannel_by_id.get(str(row.get("dialogue_id", ""))),
            min_fragment_chars=min_fragment_chars,
        )
        for row in rows
    ]

    counts: dict[str, int] = dict.fromkeys(pool, 0)
    for backchannel in chosen:
        counts[backchannel] += 1
    fragment_lengths = [
        length
        for fragments in fragments_by_id.values()
        for length in (len(fragments[0]), len(fragments[1]))
    ]
    return {
        "rows": rebuilt,
        "seams": seams,
        "unsplit": unsplit,
        "backchannel_counts": counts,
        "clash_seams": [seams[index]["dialogue_id"] for index in clashes],
        "fragment_characters": {
            "count": len(fragment_lengths),
            "mean": statistics.fmean(fragment_lengths) if fragment_lengths else None,
            "min": min(fragment_lengths, default=None),
            "max": max(fragment_lengths, default=None),
        },
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


_EVAL_TEXT_FIELDS = ("text", "prompt", "user_prompt", "preferred", "dispreferred")


def collect_eval_texts(rows: Iterable[Mapping[str, Any]]) -> set[str]:
    """Every human-readable string in a fixed evaluation file.

    Wide on purpose. A script that reuses an evaluation PROMPT is nearly as bad as one that
    reuses its answer: the model would have been trained on the very sentence the judge is
    about to hand it. The 256 strings this returns over the five eval files are the same 256
    v1 was checked against (`reports/m3-script-validation.json`).
    """
    texts: set[str] = set()
    for row in rows:
        for field in _EVAL_TEXT_FIELDS:
            value = row.get(field)
            if isinstance(value, str) and value.strip():
                texts.add(_normalise(value))
    return texts


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_v2_command(args: argparse.Namespace) -> int:
    spec = TimelineSpec(
        lead_in_seconds=args.lead_in_seconds,
        gap_seconds=args.gap_seconds,
        frame_rate_hz=args.frame_rate_hz,
    )
    source_rows = _read_jsonl(args.source)
    corpus_rows = _read_jsonl(args.corpus_manifest)
    corpus_texts = {
        str(row["artifact_id"]): str(row["text"])
        for row in corpus_rows
        if row.get("split") == "train"
    }
    held_out_texts = {
        _normalise(str(row["text"])) for row in corpus_rows if row.get("split") in {"dev", "test"}
    }
    eval_texts: set[str] = set()
    for path in args.eval or []:
        eval_texts |= collect_eval_texts(_read_jsonl(path))

    built = build_v2_scripts(
        source_rows, seed=args.seed, min_fragment_chars=args.min_fragment_chars
    )
    rows = built["rows"]
    for row in rows:
        row["projected_frames_fast"] = project_script_frames(
            [turn["text"] for turn in row["turns"]],
            mora_per_second=args.mora_per_second,
            spec=spec,
        )

    # The committed v1 field is not reproducible with today's pyopenjtalk dictionary, so the
    # v1/v2 comparison uses THIS toolchain on both sides. Recording the gap keeps anyone from
    # reading the two committed fields as if they were measured the same way.
    v1_recomputed = [
        project_script_frames(
            [turn["text"] for turn in row["turns"]],
            mora_per_second=args.mora_per_second,
            spec=spec,
        )
        for row in source_rows
    ]
    v1_committed = [int(row["projected_frames_fast"]) for row in source_rows]
    v1_deltas = [
        committed - recomputed
        for committed, recomputed in zip(v1_committed, v1_recomputed, strict=True)
    ]

    # What v1 actually rendered, for whoever has to judge whether a projection this close to
    # the floor is a risk. Passed in rather than assumed, and recorded with its source.
    realised: dict[str, Any] | None = None
    if args.v1_realised_source:
        realised = {
            "min_frames": args.v1_realised_min_frames,
            "mean_frames": args.v1_realised_mean_frames,
            "source": args.v1_realised_source,
            "realised_over_projected_mean": (
                args.v1_realised_mean_frames / statistics.fmean(v1_recomputed)
                if args.v1_realised_mean_frames
                else None
            ),
            "note": (
                "v1 rendered LONGER than it projected, so the projection is the conservative "
                "side of the floor. It is still a projection."
            ),
        }

    checks = validate_scripts(
        rows,
        corpus_texts=corpus_texts,
        eval_texts=eval_texts,
        held_out_texts=held_out_texts,
        min_a_turn_chars=args.min_fragment_chars,
    )
    before = summarise_structure(source_rows)
    after = summarise_structure(rows)
    frames = [int(row["projected_frames_fast"]) for row in rows]
    # The plan's other length gate is per turn, not per dialogue: every turn has to be at
    # least two mora, or it is a noise the renderer will smear over a whole frame.
    turn_mora = [
        (str(row["dialogue_id"]), index, mora_count(turn["text"]))
        for row in rows
        for index, turn in enumerate(row["turns"])
    ]
    shortest_turn = min(turn_mora, key=lambda item: item[2])

    _write_jsonl(args.out_scripts, rows)

    assignment = {str(row["dialogue_id"]): str(row["split"]) for row in rows}
    counts: dict[str, int] = {"train": 0, "dev": 0, "test": 0}
    for split in assignment.values():
        counts[split] = counts.get(split, 0) + 1
    inherited = all(
        str(source["split"]) == assignment[str(source["dialogue_id"])] for source in source_rows
    )
    _write_json(
        args.out_split_map,
        {
            "schema_version": 1,
            "created_at": args.captured_at,
            "seed": args.split_seed,
            "method": (
                "Inherited unchanged from split-map-v1.json. M3-R rebuilds the SHAPE of the "
                "dialogues and nothing else, so the 72/8 assignment has to be the same one "
                "M3 used - a dialogue that moved between train and dev would make the two "
                "runs incomparable for a reason that has nothing to do with the rebuild."
            ),
            "inherited_from": args.split_map_source,
            "inherited_unchanged": inherited,
            "shared_by": list(args.shared_by),
            "counts": counts,
            "test_note": (
                "No dialogue-level test split. Held-out evaluation uses the corpus test split "
                "audio, which never enters the dataset."
            ),
            "assignment": assignment,
        },
    )

    by_split: dict[str, Any] = {}
    for split in sorted(counts):
        if not counts[split]:
            continue
        subset = [row for row in rows if row["split"] == split]
        by_split[split] = summarise_structure(subset)

    report = {
        "schema_version": 1,
        "milestone": "M3-R",
        "step": "2-2 comma-split rebuild of the dialogue scripts",
        "captured_at": args.captured_at,
        "artifact": {
            "path": str(args.out_scripts),
            "rows": len(rows),
            "byte_size": args.out_scripts.stat().st_size,
            "sha256": _sha256(args.out_scripts),
        },
        "derived_from": {
            "path": str(args.source),
            "rows": len(source_rows),
            "sha256": _sha256(args.source),
            "structure": before,
        },
        "split_rule": {
            "function": "tools.dialogue_scripts.split_at_central_comma",
            "punctuation": CLAUSE_COMMA,
            "min_fragment_chars": args.min_fragment_chars,
            "tie_break": "earlier comma",
            "split_dialogues": len(built["seams"]),
            "unsplit_dialogues": len(built["unsplit"]),
            "unsplit": built["unsplit"],
            "fragment_characters": built["fragment_characters"],
            "why": (
                "A speaks twice per dialogue without the dialogue count falling, so the 72 "
                "train dialogues stay 72 and the run stays 45 steps. Pairing sentences "
                "instead would have halved both."
            ),
            "split_points": summarise_split_points(rows),
        },
        "backchannels": {
            "pool": list(BACKCHANNEL_POOL),
            "pool_size": len(BACKCHANNEL_POOL),
            "seed": args.seed,
            "seams": len(built["seams"]),
            "counts": built["backchannel_counts"],
            "clash_seams": built["clash_seams"],
            "clash_rule": (
                "A backchannel never opens with the same clause as the turn it precedes."
            ),
        },
        "structure": {
            "m3_v1": {
                "turns_per_dialogue": before["turns_per_dialogue"],
                "a_turns_per_dialogue": before["a_turns_per_dialogue"],
            },
            "m3r_v2": {
                "turns_per_dialogue": after["turns_per_dialogue"],
                "a_turns_per_dialogue": after["a_turns_per_dialogue"],
            },
            "target": {"turns_per_dialogue": 5.0, "a_turns_per_dialogue": 2.0},
            "speaker_shapes": after["speaker_shapes"],
            "a_turns_histogram": after["a_turns_histogram"],
            "by_split": by_split,
            "train_dialogues": counts["train"],
            "projected_steps": {
                "value": counts["train"] // 8 * 5,
                "how": "train dialogues // batch 8, times 5 epochs",
                "m3_value": 45,
            },
        },
        "invariants": {
            "a_characters_v1": before["a_characters"],
            "a_characters_v2": after["a_characters"],
            "a_characters_unchanged": before["a_characters"] == after["a_characters"],
            "why": (
                "Splitting moves characters between turns; it must not add or lose one. If "
                "this total moved, the scripts were rewritten rather than reshaped and the "
                "M3-R/M3 comparison would no longer isolate structure."
            ),
            "b_characters_v1": before["b_characters"],
            "b_characters_v2": after["b_characters"],
            "b_characters_added_by_backchannels": after["b_characters"] - before["b_characters"],
        },
        "checks": {
            **checks,
            "eval_texts_compared": len(eval_texts),
            "why_held_out_is_checked_joined": (
                "A held-out sentence cut in two would pass a per-turn check. The joined A "
                "text is compared as well."
            ),
        },
        "length": {
            "floor_frames": args.floor_frames,
            "floor_seconds": args.floor_frames / spec.frame_rate_hz,
            "projection": "mora / mora_per_second at the FAST end of the measured rate, frames floored",
            "mora_per_second_used": args.mora_per_second,
            "timeline": {
                "lead_in_seconds": spec.lead_in_seconds,
                "gap_seconds": spec.gap_seconds,
                "frame_rate_hz": spec.frame_rate_hz,
            },
            "min_frames": min(frames),
            "max_frames": max(frames),
            "mean_frames": statistics.fmean(frames),
            "median_seconds": statistics.median(frames) / spec.frame_rate_hz,
            "below_floor": sum(1 for value in frames if value < args.floor_frames),
            "turn_mora": {
                "floor": 2,
                "min": shortest_turn[2],
                "shortest_turn": {"dialogue_id": shortest_turn[0], "turn": shortest_turn[1]},
                "turns_under_floor": sum(1 for _, _, mora in turn_mora if mora < 2),
            },
            "v1_same_toolchain": {
                "min_frames": min(v1_recomputed),
                "max_frames": max(v1_recomputed),
                "mean_frames": statistics.fmean(v1_recomputed),
                "committed_field_delta_mean": statistics.fmean(v1_deltas),
                "committed_field_delta_max": max(v1_deltas, key=abs),
                "rows_reproduced": sum(1 for delta in v1_deltas if delta == 0),
                "note": (
                    "The projected_frames_fast committed with v1 cannot be reproduced by "
                    "today's pyopenjtalk dictionary. Both columns above are recomputed here, "
                    "so the v1/v2 comparison is like for like; the committed v1 field is not."
                ),
            },
            "v1_realised": realised,
            "floor_margin_frames": min(frames) - args.floor_frames,
        },
        "limits": [
            "Projection is mora over a measured rate, not synthesis. Realised durations are "
            "gated again after rendering.",
            f"The smallest projection clears the {args.floor_frames}-frame floor by "
            f"{min(frames) - args.floor_frames} frame(s). Step 2-5 (overlap) removes length "
            "and step 2-6 may concatenate dialogues; whichever happens has to re-measure "
            "against the floor rather than inherit this number.",
            f"Each backchannel phrase is used {min(built['backchannel_counts'].values())}-"
            f"{max(built['backchannel_counts'].values())} times across the "
            f"{len(built['seams'])} seams. Rendering one WAV per phrase and reusing it would "
            "put an identical waveform in that many dialogues, which is the repetition this "
            "pool exists to avoid - step 2-3 renders per dialogue.",
            "A five-turn dialogue projects to about 20 seconds, so the M3-R plan's 60-second "
            "sequence-length target is not met by one dialogue. That target belongs to step "
            "2-6 (concatenation), not to this file.",
            "Split points were read one by one. They all land on a comma, which is a pause "
            "the reader already takes, but a comma is not always a syntactic boundary; see "
            "split_rule.split_points.mid_coordination for the ones a listener would find "
            "least natural.",
            "The 2 dialogues that keep the B-A-B shape have no comma at all. They stay in "
            "training because dropping them would cost 2 of the 72 train dialogues, and "
            "holding the step count at 45 is the reason this rebuild splits rather than pairs.",
        ],
    }
    _write_json(args.report, report)
    print(json.dumps({"status": checks["status"], "report": str(args.report)}, ensure_ascii=False))
    return 0 if checks["status"] == "pass" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and validate the dialogue scripts")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser(
        "build-v2", help="rebuild B-A-B scripts as B-A₁-b-A₂-B by splitting A at a comma"
    )
    build.add_argument("--source", type=Path, required=True)
    build.add_argument("--corpus-manifest", type=Path, required=True)
    build.add_argument("--eval", type=Path, action="append")
    build.add_argument("--out-scripts", type=Path, required=True)
    build.add_argument("--out-split-map", type=Path, required=True)
    build.add_argument("--report", type=Path, required=True)
    build.add_argument("--seed", type=int, required=True)
    build.add_argument("--split-seed", type=int, default=20260822)
    build.add_argument("--split-map-source", default="m3/scripts/split-map-v1.json")
    build.add_argument("--shared-by", nargs="*", default=["v-real-v2"])
    build.add_argument("--captured-at", required=True)
    build.add_argument("--min-fragment-chars", type=int, default=MIN_FRAGMENT_CHARS)
    build.add_argument("--v1-realised-min-frames", type=int)
    build.add_argument("--v1-realised-mean-frames", type=float)
    build.add_argument("--v1-realised-source")
    build.add_argument("--mora-per-second", type=float, default=7.5)
    build.add_argument("--floor-frames", type=int, default=200)
    build.add_argument("--lead-in-seconds", type=float, default=0.5)
    build.add_argument("--gap-seconds", type=float, default=0.4)
    build.add_argument("--frame-rate-hz", type=float, default=12.5)

    args = parser.parse_args()
    if args.command == "build-v2":
        return _build_v2_command(args)
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
