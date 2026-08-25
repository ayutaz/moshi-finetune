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

M3 and M3-R
-----------
`dialogue_timeline` is the M3 placement: fixed lead-in, fixed gap, strictly sequential, no
overlap. It stays because the M3 dataset has to remain reproducible, and because it is the
control the new placement is measured against.

`place_turns` is the M3-R placement and everything new is opt-in through it. Three things
differ, each answering a measured defect of the M3 data.

- **Turns are joined speech-end to speech-start, not clip-end to clip-start.** A rendered
  clip carries its own padding - a corpus recording opens 25 ms and closes 110 ms in
  silence, an Irodori turn 28 ms and 65 ms - so a "0.4 s gap" between clips was never 0.4 s
  of conversational silence. Joining on the speech makes the number in the spec the number
  on the timeline, and it is the only way to state an overlap at all: an overlap defined
  between clip edges can be entirely padding and never put two voices in the same frame.
- **A negative join is an overlap.** M3 taught the model that nobody ever speaks while
  someone else is speaking. The backchannel join is always negative, because an aizuchi
  that waits for a gap is not an aizuchi.
- **`Join(mode="clip")` butt-joins two clips.** Speaker A's two turns are the two halves of
  one recording cut at its own pause; butt-joining them puts every sample of that recording
  back on the timeline in its original order, so A's placed audio is the same bytes M3
  placed and the arms stay comparable. The pause between A's turns is then the pause the
  speaker actually took, and the backchannel is placed inside it.

The room tone under the non-speaking channel comes from `tools/room_tone.py` and is applied
here rather than there, because only the timeline knows which stretches are silent by
construction rather than by accident.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Sequence
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


@dataclass(frozen=True)
class Clip:
    """One rendered turn: how long the file is, and where the speech sits inside it.

    `speech_start` and `speech_end` are offsets into the clip, so a placement can align two
    turns on what is audible rather than on where their files happen to begin.
    """

    speaker: str
    duration: float
    speech_start: float
    speech_end: float
    role: str = ""

    def __post_init__(self) -> None:
        if self.duration <= 0:
            raise ValueError(f"a clip needs a positive duration, got {self.duration}")
        if not 0 <= self.speech_start < self.speech_end <= self.duration:
            raise ValueError(
                f"speech {self.speech_start}..{self.speech_end} does not sit inside a clip "
                f"of {self.duration}"
            )


@dataclass(frozen=True)
class Join:
    """Where one turn sits relative to an earlier one.

    `anchor` is an index into the turn list and must point backwards; the backchannel and
    speaker A's second turn are both anchored to A's *first* turn rather than to whatever
    happens to precede them, which is what lets the backchannel land inside A's own pause.

    `seconds` is positive for a gap and negative for an overlap. In `mode="speech"` it is
    measured from the anchor's last audible sample to this turn's first audible one; in
    `mode="clip"` from file edge to file edge, so `Join(anchor, "clip", 0.0)` concatenates
    two clips without losing or repeating a sample.
    """

    anchor: int
    mode: str = "speech"
    seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in ("speech", "clip"):
            raise ValueError(f"unknown join mode {self.mode!r}")


@dataclass(frozen=True)
class OverlapSpec:
    """The offsets a dialogue's joins are drawn from, in seconds.

    Drawn rather than fixed, because a constant offset is a pattern the model can learn
    instead of the conversation - the same reason the room tone is drawn from a reshuffled
    deck. Every value is recorded per dialogue so a timeline can be rebuilt from the seed.

    The two outer sets straddle zero: most transitions overlap slightly, a few leave a gap.
    The backchannel set does not, because an aizuchi placed after the speaker stops is a
    reply, not a backchannel, and would teach the opposite of what it is here to teach.
    """

    open_to_body: tuple[float, ...] = (-0.45, -0.35, -0.25, -0.15, -0.05, 0.05)
    body_to_backchannel: tuple[float, ...] = (-0.55, -0.45, -0.35, -0.25)
    body_to_close: tuple[float, ...] = (-0.45, -0.35, -0.25, -0.15, -0.05, 0.05)


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
    """Place each (speaker, duration) turn sequentially with a fixed gap between them.

    The M3 placement. Kept so the M3 dataset can be rebuilt; `place_turns` is what M3-R
    uses.
    """
    placed: list[dict[str, Any]] = []
    cursor = spec.lead_in_seconds
    for index, (speaker, duration) in enumerate(turns):
        if index:
            cursor += spec.gap_seconds
        placed.append({"speaker": speaker, "start": cursor, "end": cursor + duration})
        cursor += duration
    return placed


def place_turns(
    clips: Sequence[Clip],
    joins: Sequence[Join | None],
    *,
    spec: TimelineSpec,
    min_same_speaker_gap: float = 0.0,
) -> list[dict[str, Any]]:
    """Place clips against their anchors, allowing overlap.

    Returns one row per turn with absolute `clip_start`/`clip_end` and
    `speech_start`/`speech_end`. The first join is ignored - turn 0 starts at the lead-in -
    and every other must anchor to a turn already placed, so a cycle cannot be expressed.

    A placement that would start before time zero is refused rather than shifted: shifting
    would silently move every other turn and change the overlaps the caller asked for.

    A placement that would start *before* the same speaker's previous clip ends is shifted,
    to `min_same_speaker_gap` after it, and the row records how far by. One channel carrying
    two clips at once is not an overlap, it is a sum of two voices, and it happens for real:
    a backchannel long enough to still be sounding when a two-word second half of speaker
    A's sentence has already finished. Refusing would throw the dialogue away over a hundred
    milliseconds; shifting keeps it and says so. A clip that starts exactly where the
    previous one ended is left alone - that is the butt joint, not a collision.
    """
    if len(clips) != len(joins):
        raise ValueError(f"{len(clips)} clips but {len(joins)} joins")
    if not clips:
        raise ValueError("a dialogue needs at least one turn")

    placed: list[dict[str, Any]] = []
    for index, (clip, join) in enumerate(zip(clips, joins, strict=True)):
        if index == 0:
            clip_start = spec.lead_in_seconds
        else:
            if join is None:
                raise ValueError(f"turn {index} has no join")
            if not 0 <= join.anchor < index:
                raise ValueError(f"turn {index} anchors to {join.anchor}, which is not earlier")
            anchor = placed[join.anchor]
            if join.mode == "speech":
                clip_start = anchor["speech_end"] + join.seconds - clip.speech_start
            else:
                clip_start = anchor["clip_end"] + join.seconds
        if clip_start < 0:
            raise ValueError(f"turn {index} would start at {clip_start:.3f} s, before the file")
        earlier = [row["clip_end"] for row in placed if row["speaker"] == clip.speaker]
        latest = max(earlier) if earlier else None
        # Only a real overlap is repaired. A clip that starts exactly where the previous one
        # ended is the butt joint speaker A's two halves rely on, and pushing it by the gap
        # would put room tone inside the recording.
        deferred = (
            latest + min_same_speaker_gap - clip_start
            if latest is not None and clip_start < latest
            else 0.0
        )
        clip_start += deferred
        placed.append(
            {
                "speaker": clip.speaker,
                "role": clip.role,
                "clip_start": clip_start,
                "clip_end": clip_start + clip.duration,
                "speech_start": clip_start + clip.speech_start,
                "speech_end": clip_start + clip.speech_end,
                "deferred_seconds": deferred,
            }
        )
    return placed


def timeline_seconds(placed: Sequence[dict[str, Any]]) -> float:
    """How long the assembled file is: the last clip edge, not the last audible sample."""
    return max(row["clip_end"] for row in placed)


def merge_intervals(spans: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    """Union of possibly overlapping spans, in order.

    Summing spans without merging double-counts every overlap, which is exactly the
    quantity a dataset built to contain overlaps has to get right.
    """
    ordered = sorted((s, e) for s, e in spans if e > s)
    merged: list[tuple[float, float]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def speaking_seconds(
    placed: Sequence[dict[str, Any]], speaker: str, *, extent: str = "clip"
) -> float:
    """Seconds one speaker holds the floor, counting each moment once.

    `extent="clip"` measures the placed file, which is the M3 convention and what the
    published 68.8% is computed from: a turn's silent head and tail count as that speaker's
    time. `extent="speech"` measures first audible sample to last. Both are reported,
    because the clip convention rewards leaving silence inside a turn and the speech
    convention rewards cutting it out, and neither is the honest number on its own.
    """
    key = ("clip_start", "clip_end") if extent == "clip" else ("speech_start", "speech_end")
    spans = [(row[key[0]], row[key[1]]) for row in placed if row["speaker"] == speaker]
    return sum(end - start for start, end in merge_intervals(spans))


def silence_share(placed: Sequence[dict[str, Any]], speaker: str, *, extent: str = "clip") -> float:
    """Fraction of the dialogue in which `speaker` is not holding the floor."""
    total = timeline_seconds(placed)
    if total <= 0:
        raise ValueError("a dialogue with no length has no silence share")
    return 1.0 - speaking_seconds(placed, speaker, extent=extent) / total


def frame_mask(
    spans: Sequence[tuple[float, float]], *, frames: int, frame_rate_hz: float
) -> list[bool]:
    """Which frames any of `spans` touches.

    A frame counts as spoken when any part of it is inside a span, matching `frames_for`:
    a syllable that starts three quarters of the way into a frame is still in that frame.
    """
    mask = [False] * frames
    for start, end in spans:
        first = max(0, int(math.floor(start * frame_rate_hz)))
        last = min(frames, int(math.ceil(end * frame_rate_hz)))
        for index in range(first, last):
            mask[index] = True
    return mask


def overlap_frames(
    placed: Sequence[dict[str, Any]], *, frame_rate_hz: float = FRAME_RATE_HZ
) -> dict[str, Any]:
    """Frames in which both speakers are audible.

    This is the gate M3 could not pass by construction. It is counted on the speech
    extents, not the clip extents, because two clips can overlap entirely in their padding
    and put nothing in the same frame.
    """
    frames = frames_for(timeline_seconds(placed), frame_rate_hz=frame_rate_hz)
    masks = {
        speaker: frame_mask(
            [
                (row["speech_start"], row["speech_end"])
                for row in placed
                if row["speaker"] == speaker
            ],
            frames=frames,
            frame_rate_hz=frame_rate_hz,
        )
        for speaker in ("A", "B")
    }
    both = [a and b for a, b in zip(masks["A"], masks["B"], strict=True)]
    return {
        "frames": frames,
        "simultaneous_frames": sum(both),
        "a_frames": sum(masks["A"]),
        "b_frames": sum(masks["B"]),
        "simultaneous_share": sum(both) / frames if frames else 0.0,
    }


def same_channel_collisions(placed: Sequence[dict[str, Any]]) -> list[tuple[int, int]]:
    """Pairs of turns by the same speaker whose clips overlap.

    Two clips on one channel would be summed, and a sum of two voices is not a turn. This
    is the check that stops an overlap spec from quietly producing one.
    """
    clashes = []
    for i, left in enumerate(placed):
        for j, right in enumerate(placed[i + 1 :], start=i + 1):
            if left["speaker"] != right["speaker"]:
                continue
            if left["clip_start"] < right["clip_end"] and right["clip_start"] < left["clip_end"]:
                clashes.append((i, j))
    return clashes


def stable_seed(base: int, *parts: object) -> int:
    """A seed derived from `base` and an identity, stable across processes.

    `hash()` is salted per process, so a run resumed tomorrow would draw a different
    timeline for the dialogues it had not reached.
    """
    key = ":".join([str(base), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=4).digest(), "big")


def draw_offsets(rng, spec: OverlapSpec, *, has_backchannel: bool) -> dict[str, float]:
    """One offset per drawn join of a dialogue.

    The A1 -> A2 join is not here: it is fixed by the recording, which is cut at its own
    pause and put back together unchanged.
    """
    drawn = {
        "open_to_body": rng.choice(spec.open_to_body),
        "body_to_close": rng.choice(spec.body_to_close),
    }
    if has_backchannel:
        drawn["body_to_backchannel"] = rng.choice(spec.body_to_backchannel)
    return drawn


def dialogue_joins(roles: Sequence[str], offsets: dict[str, float]) -> list[Join | None]:
    """Joins for a B-A-b-A-B or a B-A-B dialogue, from its roles and drawn offsets.

    The five-turn shape anchors the backchannel and A's second half to A's *first* half, so
    the aizuchi sits inside the speaker's own pause and the two halves of the recording stay
    contiguous no matter how far the backchannel reaches.
    """
    if roles == ["open", "body", "backchannel", "body", "close"]:
        return [
            None,
            Join(0, "speech", offsets["open_to_body"]),
            Join(1, "speech", offsets["body_to_backchannel"]),
            Join(1, "clip", 0.0),
            Join(3, "speech", offsets["body_to_close"]),
        ]
    if roles == ["open", "body", "close"]:
        return [
            None,
            Join(0, "speech", offsets["open_to_body"]),
            Join(1, "speech", offsets["body_to_close"]),
        ]
    raise ValueError(f"no join plan for roles {roles!r}")


def group_dialogues(dialogue_ids: Sequence[str], *, group_size: int) -> list[list[str]]:
    """Split dialogues into the sequences that become one training row.

    A dialogue is about 19 s, and the sequence length M3-R is aiming at is 60 s or more, so
    a row has to hold several. Grouping is non-overlapping: a dialogue appearing in two rows
    would be seen twice per epoch, which is a change to how much training each recording
    gets and has nothing to do with sequence length.

    A short final group is kept rather than dropped - dropping it would lose dialogues, and
    padding it by repeating one would put a recording in twice.
    """
    if group_size < 1:
        raise ValueError(f"group_size must be at least 1, got {group_size}")
    ids = list(dialogue_ids)
    return [ids[i : i + group_size] for i in range(0, len(ids), group_size)]


def steps_for_grouping(
    *, dialogues: int, group_size: int, global_batch: int, epochs: int
) -> dict[str, Any]:
    """What grouping does to the training schedule.

    Grouping trades rows for length at a fixed amount of audio, so the step count falls
    unless the batch falls with it. The frames a step averages over is the quantity that
    actually sets the gradient noise, and it is reported here so a change of batch can be
    judged against it rather than against the row count.
    """
    from tools.training_shape import steps_per_epoch, total_steps

    rows = len(group_dialogues([str(i) for i in range(dialogues)], group_size=group_size))
    return {
        "rows": rows,
        "group_size": group_size,
        "global_batch": global_batch,
        "steps_per_epoch": steps_per_epoch(examples=rows, batch=global_batch),
        "total_steps": total_steps(examples=rows, batch=global_batch, epochs=epochs),
        "dialogues_per_step": group_size * global_batch,
    }


def grouping_options(
    *,
    dialogues: int,
    epochs: int,
    target_steps: int,
    seconds_per_dialogue: float,
    gap_seconds: float = 0.0,
    group_sizes: Sequence[int] = (1, 2, 3, 4, 6, 8),
    batches: Sequence[int] = (1, 2, 3, 4, 6, 8),
) -> list[dict[str, Any]]:
    """Every (group size, batch) pair that lands on `target_steps`, with what it costs.

    M3-R has to hold two things at once: sequences of 60 s or more, and the 45 steps M3 ran.
    They pull against each other, because grouping trades rows for length and the step count
    is set by rows. Enumerating rather than asserting means the choice can be seen to be a
    choice, and the fallback is visible if the chosen one does not fit in memory.
    """
    from tools.training_shape import total_steps

    options = []
    for group_size in group_sizes:
        rows = len(group_dialogues([str(i) for i in range(dialogues)], group_size=group_size))
        seconds = group_size * seconds_per_dialogue + (group_size - 1) * gap_seconds
        for batch in batches:
            if total_steps(examples=rows, batch=batch, epochs=epochs) != target_steps:
                continue
            options.append(
                {
                    "group_size": group_size,
                    "global_batch": batch,
                    "rows": rows,
                    "sequence_seconds": seconds,
                    "sequence_frames": frames_for(seconds, frame_rate_hz=FRAME_RATE_HZ),
                    "dialogues_per_step": group_size * batch,
                    "meets_60s": seconds >= 60.0,
                }
            )
    return options


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


# --------------------------------------------------------------------------------------
# signal helpers - numpy only inside the functions, so the suite runs without it
# --------------------------------------------------------------------------------------


def quiet_runs(levels: Sequence[float], *, threshold: float) -> list[tuple[int, int]]:
    """Half-open [start, end) runs of frames at or below `threshold`."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, level in enumerate(levels):
        if level <= threshold:
            if start is None:
                start = index
        elif start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(levels)))
    return runs


def speech_extent(
    levels: Sequence[float], *, threshold: float, hop_seconds: float
) -> tuple[float, float]:
    """First and last audible instant of a clip, in seconds.

    Returns the whole clip when nothing clears the threshold, rather than raising: a caller
    that has already decided this file is a turn should get a usable extent, and the
    silent-channel gate is what catches a turn that is not really there.
    """
    loud = [index for index, level in enumerate(levels) if level > threshold]
    if not loud:
        return 0.0, len(levels) * hop_seconds
    return loud[0] * hop_seconds, (loud[-1] + 1) * hop_seconds


def choose_pause(
    runs: Sequence[tuple[int, int]],
    *,
    target_seconds: float,
    hop_seconds: float,
    min_frames: int,
    edge_frames: int,
    frames: int,
) -> tuple[int, int] | None:
    """The pause nearest `target_seconds`, ignoring the ones at the clip's edges.

    `target_seconds` comes from the mora split of the sentence, so "nearest" is nearest to
    where the speaker should have drawn breath. Runs shorter than `min_frames` are consonant
    closures rather than pauses, and runs touching either edge are the recording's own head
    and tail, which are not places to cut a sentence in half.
    """
    best: tuple[float, tuple[int, int]] | None = None
    for start, end in runs:
        if end - start < min_frames:
            continue
        if start < edge_frames or end > frames - edge_frames:
            continue
        centre = (start + end) / 2 * hop_seconds
        distance = abs(centre - target_seconds)
        if best is None or distance < best[0]:
            best = (distance, (start, end))
    return None if best is None else best[1]


def frame_levels(samples, *, hop: int):
    """RMS of each `hop`-sample frame."""
    import numpy as np

    data = np.asarray(samples, dtype=np.float64)
    count = len(data) // hop
    if count == 0:
        return np.zeros(0)
    frames = data[: count * hop].reshape(count, hop)
    return np.sqrt((frames**2).mean(axis=1))


#: Frame hop the edge and pause detectors work at. 10 ms is short enough to place a cut
#: inside a 60 ms pause and long enough that one glottal period does not read as silence.
HOP_SECONDS = 0.01
#: A frame counts as audible above this share of the clip's 95th-percentile frame RMS, or
#: above `LEVEL_FLOOR`, whichever is larger. The floor is what keeps the corpus recordings'
#: noise-gated tails - 8.9e-5 RMS, under three PCM_16 LSB - from reading as speech.
LEVEL_RATIO = 0.03
LEVEL_FLOOR = 3e-4


def clip_from_samples(samples, *, speaker: str, role: str, sample_rate: int) -> Clip:
    """Measure a rendered turn: its length and where the audible part of it sits."""
    hop = int(round(HOP_SECONDS * sample_rate))
    levels = frame_levels(samples, hop=hop)
    threshold = _level_threshold(levels)
    start, end = speech_extent(list(levels), threshold=threshold, hop_seconds=HOP_SECONDS)
    duration = len(samples) / sample_rate
    return Clip(
        speaker=speaker,
        role=role,
        duration=duration,
        speech_start=min(start, duration - 1e-6),
        speech_end=min(max(end, start + 1e-3), duration),
    )


def _level_threshold(levels) -> float:
    import numpy as np

    if len(levels) == 0:
        return LEVEL_FLOOR
    return max(float(np.percentile(levels, 95)) * LEVEL_RATIO, LEVEL_FLOOR)


def split_recording(
    samples,
    *,
    sample_rate: int,
    mora_before: int,
    mora_after: int,
    min_pause_seconds: float = 0.08,
    edge_seconds: float = 0.08,
) -> dict[str, Any]:
    """Cut one recording in two at the pause nearest its mora split.

    The script splits speaker A's sentence at a reading comma, and the recording of that
    sentence has a pause where the comma is. Finding that pause is what turns one turn into
    two without a forced aligner: the mora ratio says roughly where to look, and the
    recording says exactly where the speaker stopped.

    The cut is the middle of the pause and nothing is discarded, so the two halves put back
    together are the recording, sample for sample.
    """
    hop = int(round(HOP_SECONDS * sample_rate))
    levels = frame_levels(samples, hop=hop)
    threshold = _level_threshold(levels)
    speech_start, speech_end = speech_extent(
        list(levels), threshold=threshold, hop_seconds=HOP_SECONDS
    )
    total_mora = mora_before + mora_after
    if total_mora <= 0:
        raise ValueError("a split needs mora on both sides")
    target = speech_start + (speech_end - speech_start) * mora_before / total_mora
    pause = choose_pause(
        quiet_runs(list(levels), threshold=threshold),
        target_seconds=target,
        hop_seconds=HOP_SECONDS,
        min_frames=max(1, int(round(min_pause_seconds / HOP_SECONDS))),
        edge_frames=max(1, int(round(edge_seconds / HOP_SECONDS))),
        frames=len(levels),
    )
    if pause is None:
        return {"found": False, "target_seconds": target}
    centre_frame = (pause[0] + pause[1]) // 2
    cut = centre_frame * hop
    return {
        "found": True,
        "target_seconds": target,
        "cut_seconds": cut / sample_rate,
        "cut_sample": int(cut),
        "pause_seconds": (pause[1] - pause[0]) * HOP_SECONDS,
        "pause_start_seconds": pause[0] * HOP_SECONDS,
        "pause_end_seconds": pause[1] * HOP_SECONDS,
        "distance_seconds": abs(cut / sample_rate - target),
    }


def _read_stereo(path: str):
    import numpy as np
    import soundfile

    data, rate = soundfile.read(str(path), dtype="float64", always_2d=True)
    return np.asarray(data), int(rate)


def _write_stereo(path: str, data, rate: int) -> None:
    import numpy as np
    import soundfile

    soundfile.write(str(path), np.clip(data, -1.0, 1.0), rate, subtype="PCM_16")


def speaker_spans(
    word_transcript: Sequence[dict[str, Any]], speaker: str
) -> list[tuple[float, float]]:
    """Contiguous runs of one speaker's words: one span per turn M3 placed.

    The M3 stereo files carry a turn's clip exactly where its words are, so reading the
    spans back out of the transcript recovers the clip boundaries without having to
    re-derive them from the lead-in, the gap and the source durations.
    """
    words = sorted(
        (w for w in word_transcript if w["speaker"] == speaker), key=lambda w: w["start"]
    )
    spans: list[tuple[float, float]] = []
    for word in words:
        if spans and word["start"] <= spans[-1][1] + 1e-9:
            spans[-1] = (spans[-1][0], max(spans[-1][1], word["end"]))
        else:
            spans.append((word["start"], word["end"]))
    return spans


def extract_m3_clips(stereo, word_transcript, *, sample_rate: int) -> dict[str, Any]:
    """Cut A's turn and B's two turns back out of an M3 stereo dialogue.

    Re-using M3's own bytes rather than re-rendering from the 96 kHz corpus and the 48 kHz
    Irodori turns removes the resampler from the comparison entirely: speaker A's audio in
    M3-R is the same audio M3 trained on, so a difference between the arms cannot be a
    difference in how the recordings were downsampled.

    Everything outside those spans must be digital silence, which is the assertion that the
    spans really are the clips and not part of them.
    """
    import numpy as np

    out: dict[str, Any] = {}
    for speaker, channel in (("A", 0), ("B", 1)):
        clips = []
        covered = np.zeros(len(stereo), dtype=bool)
        for start, end in speaker_spans(word_transcript, speaker):
            first, last = round(start * sample_rate), round(end * sample_rate)
            clips.append(stereo[first:last, channel].copy())
            covered[first:last] = True
        leftover = stereo[~covered, channel]
        if leftover.size and np.abs(leftover).max() > 0:
            raise ValueError(f"speaker {speaker} has audio outside its word spans")
        out[speaker] = clips
    return out


def resample(samples, *, source_rate: int, target_rate: int):
    """Rate conversion for the one clip type M3 never rendered.

    Only the backchannels need this: A and both of B's long turns are lifted from the M3
    stereo at 24 kHz. torchaudio rather than a stride, because a stride folds everything
    above 12 kHz back into the band.
    """
    import numpy as np
    import torch
    import torchaudio

    if source_rate == target_rate:
        return np.asarray(samples, dtype=np.float64)
    tensor = torch.from_numpy(np.asarray(samples, dtype=np.float32)).unsqueeze(0)
    converted = torchaudio.functional.resample(tensor, source_rate, target_rate)
    return converted.squeeze(0).numpy().astype(np.float64)


def render_channels(placed: Sequence[dict[str, Any]], clips: Sequence[Any], *, sample_rate: int):
    """Write every clip onto its speaker's channel at the placed offset."""
    import numpy as np

    starts = [round(row["clip_start"] * sample_rate) for row in placed]
    # The length is the later of the timeline's own end and the last sample any clip
    # actually occupies: rounding a clip start up can push its final sample one past
    # ceil(end * rate), and a silently truncated turn is worse than a one-sample file.
    total = max(
        int(math.ceil(timeline_seconds(placed) * sample_rate)),
        max(start + len(samples) for start, samples in zip(starts, clips, strict=True)),
    )
    stereo = np.zeros((total, 2), dtype=np.float64)
    for row, start, samples in zip(placed, starts, clips, strict=True):
        channel = 0 if row["speaker"] == "A" else 1
        stereo[start : start + len(samples), channel] += np.asarray(samples, dtype=np.float64)
    return stereo


def lay_room_tone(stereo, pool, *, seed: int, sample_rate: int):
    """Put recorded background under every stretch that is silent by construction.

    Digital silence is one Mimi code, so in M3 `text is pad` and `speaker A is quiet` were
    the same event to 98% and the cheapest way down the loss was to stop speaking. The two
    channels get different seeds: one bed under both would make the two channels' silence
    identical, which is a different way of saying the same thing.
    """
    import numpy as np

    from tools import room_tone

    out = np.array(stereo, dtype=np.float64)
    for channel in (0, 1):
        out[:, channel] = room_tone.fill_silence(
            out[:, channel], pool, seed=seed + channel * 5000, spec=room_tone.FLOOR_WITH_EVENTS
        )
    return out


def word_transcript_for(
    placed: Sequence[dict[str, Any]], texts: Sequence[str]
) -> list[dict[str, Any]]:
    """Word timings for the whole dialogue, allocated over each turn's audible span.

    Over the audible span, not the clip: a corpus recording ends 110 ms after the last
    syllable, and stretching the final word across that silence puts a text token in a frame
    where nothing is being said.
    """
    rows: list[dict[str, Any]] = []
    for row, text in zip(placed, texts, strict=True):
        for word in allocate_word_times(
            word_units(text), start=row["speech_start"], end=row["speech_end"]
        ):
            rows.append({"speaker": row["speaker"], **word})
    rows.sort(key=lambda w: (w["start"], w["speaker"]))
    return rows


def acoustic_silence(
    stereo, *, sample_rate: int, frame_rate_hz: float, threshold: float
) -> dict[str, Any]:
    """How much of each channel carries no speech, measured from the samples.

    Measured before the room tone goes on. Afterwards the channel is never digitally silent
    and any threshold would be measuring the bed rather than the speech, which would make
    the number look better for a reason that has nothing to do with the dialogue.
    """
    import numpy as np

    hop = int(round(sample_rate / frame_rate_hz))
    count = len(stereo) // hop
    frames = np.asarray(stereo[: count * hop], dtype=np.float64).reshape(count, hop, 2)
    rms = np.sqrt((frames**2).mean(axis=1))
    quiet = rms < threshold
    return {
        "frames": int(count),
        "a_quiet_frames": int(quiet[:, 0].sum()),
        "b_quiet_frames": int(quiet[:, 1].sum()),
        "both_loud_frames": int((~quiet[:, 0] & ~quiet[:, 1]).sum()),
    }


def _summary(values: Sequence[float]) -> dict[str, float]:
    import statistics

    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "max": ordered[-1],
    }


def build_dialogue(
    row: dict[str, Any],
    *,
    stereo,
    word_transcript: Sequence[dict[str, Any]],
    backchannel,
    backchannel_rate: int,
    spec: TimelineSpec,
    overlap_spec: OverlapSpec,
    seed: int,
) -> dict[str, Any]:
    """Turn one v2 script row and its M3 audio into a placed, rendered dialogue."""
    import random

    sample_rate = spec.sample_rate
    turns = row["turns"]
    roles = [turn.get("role", "") for turn in turns]
    m3 = extract_m3_clips(stereo, word_transcript, sample_rate=sample_rate)
    if len(m3["A"]) != 1:
        raise ValueError(f"{row['dialogue_id']}: expected one A clip, got {len(m3['A'])}")
    if len(m3["B"]) != 2:
        raise ValueError(f"{row['dialogue_id']}: expected two B clips, got {len(m3['B'])}")

    a_full = m3["A"][0]
    split: dict[str, Any] = {"split": False}
    if roles == ["open", "body", "backchannel", "body", "close"]:
        before = sum(mora for _, mora in word_units(turns[1]["text"]))
        after = sum(mora for _, mora in word_units(turns[3]["text"]))
        split = split_recording(
            a_full, sample_rate=sample_rate, mora_before=before, mora_after=after
        )
        if not split["found"]:
            raise ValueError(f"{row['dialogue_id']}: no pause to split the recording at")
        split["split"] = True
        cut = split["cut_sample"]
        samples = [
            m3["B"][0],
            a_full[:cut],
            resample(backchannel, source_rate=backchannel_rate, target_rate=sample_rate),
            a_full[cut:],
            m3["B"][1],
        ]
    elif roles == ["open", "body", "close"]:
        samples = [m3["B"][0], a_full, m3["B"][1]]
    else:
        raise ValueError(f"{row['dialogue_id']}: unexpected roles {roles!r}")

    clips = [
        clip_from_samples(
            data, speaker=turn["speaker"], role=turn.get("role", ""), sample_rate=sample_rate
        )
        for data, turn in zip(samples, turns, strict=True)
    ]
    offsets = draw_offsets(
        random.Random(stable_seed(seed, row["dialogue_id"])),
        overlap_spec,
        has_backchannel="backchannel" in roles,
    )
    placed = place_turns(
        clips, dialogue_joins(roles, offsets), spec=spec, min_same_speaker_gap=0.05
    )
    collisions = same_channel_collisions(placed)
    if collisions:
        raise ValueError(f"{row['dialogue_id']}: same-channel clips overlap at {collisions}")
    return {
        "placed": placed,
        "clips": clips,
        "samples": samples,
        "offsets": offsets,
        "split": split,
        "words": word_transcript_for(placed, [turn["text"] for turn in turns]),
    }


def _dialogue_stats(
    dialogue_id: str,
    split: str,
    built: dict[str, Any],
    stereo,
    *,
    spec: TimelineSpec,
    threshold: float,
) -> dict[str, Any]:
    placed = built["placed"]
    overlap = overlap_frames(placed, frame_rate_hz=spec.frame_rate_hz)
    seconds = timeline_seconds(placed)
    acoustic = acoustic_silence(
        stereo,
        sample_rate=spec.sample_rate,
        frame_rate_hz=spec.frame_rate_hz,
        threshold=threshold,
    )
    return {
        "dialogue_id": dialogue_id,
        "split": split,
        "turns": len(placed),
        "a_turns": sum(1 for row in placed if row["speaker"] == "A"),
        "seconds": seconds,
        "frames": frames_for(seconds, frame_rate_hz=spec.frame_rate_hz),
        "a_silence_share_clip": silence_share(placed, "A", extent="clip"),
        "a_silence_share_speech": silence_share(placed, "A", extent="speech"),
        "b_silence_share_clip": silence_share(placed, "B", extent="clip"),
        "simultaneous_frames": overlap["simultaneous_frames"],
        "simultaneous_share": overlap["simultaneous_share"],
        "offsets": built["offsets"],
        "deferred_seconds": sum(row.get("deferred_seconds", 0.0) for row in placed),
        "pause_seconds": built["split"].get("pause_seconds"),
        "pause_distance_seconds": built["split"].get("distance_seconds"),
        "acoustic": acoustic,
        "acoustic_a_silence_share": acoustic["a_quiet_frames"] / acoustic["frames"],
        "roles": [row["role"] for row in placed],
    }


def _tone_gap(samples: int, pool, *, seed: int, sample_rate: int):
    """A stereo stretch of room tone `samples` long, ramped in and out.

    Built by handing `fill_silence` a silent block, so the gap between two dialogues is made
    of the same beds, at the same levels, with the same edges as every gap inside one.
    """
    import numpy as np

    if pool is None:
        return np.zeros((samples, 2))
    from tools import room_tone

    return np.stack(
        [
            room_tone.fill_silence(
                np.zeros(samples),
                pool,
                seed=seed + channel * 5000,
                spec=room_tone.FLOOR_WITH_EVENTS,
            )
            for channel in (0, 1)
        ],
        axis=1,
    ).astype(np.float64)


def _build_sequences(
    rows,
    assignment,
    rendered,
    words_by_id,
    *,
    out_dir,
    spec: TimelineSpec,
    pool,
    seed: int,
    group_size: int,
    gap: float,
) -> dict[str, Any]:
    """Concatenate the finished dialogues into the rows a training step actually sees.

    One dialogue is about 19 s and Kyutai's guidance for this model is 100-300 s, so the
    length has to come from putting dialogues together. Concatenating the *rendered* files,
    rather than re-placing the turns on a longer timeline, keeps every sequence an exact
    concatenation of files that can be listened to on their own.

    The gap between two dialogues is filled with room tone like any other gap, so the join
    is a pause in a conversation rather than a splice of digital silence - but the tone is
    rendered into the gap alone and the dialogues are pasted in untouched. Running the fill
    over the joined array instead changes them: it re-frames the channel from a different
    offset, so a stretch of digital silence that fell across two frames in the dialogue
    falls inside one frame in the sequence and gets filled there and not there. Measured, 36
    of 80 segments came back different. Nothing was made worse by it, but "the sequence is
    these files in a row" stops being true, and that is the property that lets someone
    listen to one dialogue and know what the model was trained on.
    """
    import json

    import numpy as np

    by_split: dict[str, list[str]] = {}
    for row in rows:
        by_split.setdefault(assignment[row["dialogue_id"]], []).append(row["dialogue_id"])

    out: dict[str, Any] = {"group_size": group_size, "gap_seconds": gap, "splits": {}}
    for split, ids in by_split.items():
        groups = group_dialogues(ids, group_size=group_size)
        directory = out_dir / "sequences" / split
        (directory / "audio").mkdir(parents=True, exist_ok=True)
        (directory / "text").mkdir(parents=True, exist_ok=True)
        entries = []
        gap_samples = int(round(gap * spec.sample_rate))
        for index, group in enumerate(groups, start=1):
            pieces, words, cursor = [], [], 0.0
            for position, did in enumerate(group):
                if position:
                    pieces.append(
                        _tone_gap(
                            gap_samples,
                            pool,
                            seed=stable_seed(seed, split, index, position, "gap") % 10**6,
                            sample_rate=spec.sample_rate,
                        )
                    )
                    cursor += gap_samples / spec.sample_rate
                pieces.append(rendered[did])
                words.extend(
                    {**word, "start": word["start"] + cursor, "end": word["end"] + cursor}
                    for word in words_by_id[did]
                )
                cursor += len(rendered[did]) / spec.sample_rate
            joined = np.concatenate(pieces, axis=0)
            name = f"{split}-seq-{index:03d}"
            _write_stereo(str(directory / "audio" / f"{name}.wav"), joined, spec.sample_rate)
            (directory / "text" / f"{name}.json").write_text(
                json.dumps(words, ensure_ascii=False), encoding="utf-8"
            )
            seconds = len(joined) / spec.sample_rate
            entries.append(
                {
                    "name": name,
                    "dialogues": list(group),
                    "seconds": seconds,
                    "frames": frames_for(seconds, frame_rate_hz=spec.frame_rate_hz),
                }
            )
        out["splits"][split] = {
            "rows": len(entries),
            "dialogues": len(ids),
            "seconds": _summary([entry["seconds"] for entry in entries]),
            "frames": _summary([float(entry["frames"]) for entry in entries]),
            "entries": entries,
        }
    return out


def m3_reference(
    audio_dir, text_dir, *, spec: TimelineSpec, threshold: float
) -> list[dict[str, Any]]:
    """The same statistics, computed on the M3 dialogues by the same code.

    The published "speaker A is silent 68.8% of the time" is a turn-span number, and a
    turn-span number moves when clip padding moves. Recomputing M3 here means the M3-R
    figure is compared against a measurement rather than against a quotation, and the two
    other conventions - speech extent and frame energy - exist for M3 as well.
    """
    import json
    from pathlib import Path

    rows: list[dict[str, Any]] = []
    for audio_path in sorted(Path(audio_dir).glob("*.wav")):
        stereo, rate = _read_stereo(str(audio_path))
        if rate != spec.sample_rate:
            raise ValueError(f"{audio_path}: {rate} Hz")
        transcript = json.loads(
            (Path(text_dir) / f"{audio_path.stem}.json").read_text(encoding="utf-8")
        )
        placed = []
        for speaker, channel in (("A", 0), ("B", 1)):
            for start, end in speaker_spans(transcript, speaker):
                first, last = round(start * spec.sample_rate), round(end * spec.sample_rate)
                clip = clip_from_samples(
                    stereo[first:last, channel],
                    speaker=speaker,
                    role="",
                    sample_rate=spec.sample_rate,
                )
                placed.append(
                    {
                        "speaker": speaker,
                        "role": "",
                        "clip_start": start,
                        "clip_end": end,
                        "speech_start": start + clip.speech_start,
                        "speech_end": start + clip.speech_end,
                    }
                )
        placed.sort(key=lambda row: row["clip_start"])
        seconds = len(stereo) / spec.sample_rate
        overlap = overlap_frames(placed, frame_rate_hz=spec.frame_rate_hz)
        acoustic = acoustic_silence(
            stereo,
            sample_rate=spec.sample_rate,
            frame_rate_hz=spec.frame_rate_hz,
            threshold=threshold,
        )
        rows.append(
            {
                "dialogue_id": audio_path.stem,
                "split": "m3",
                "turns": len(placed),
                "a_turns": sum(1 for row in placed if row["speaker"] == "A"),
                "seconds": seconds,
                "frames": frames_for(seconds, frame_rate_hz=spec.frame_rate_hz),
                "a_silence_share_clip": 1 - speaking_seconds(placed, "A") / seconds,
                "a_silence_share_speech": 1
                - speaking_seconds(placed, "A", extent="speech") / seconds,
                "b_silence_share_clip": 1 - speaking_seconds(placed, "B") / seconds,
                "simultaneous_frames": overlap["simultaneous_frames"],
                "simultaneous_share": overlap["simultaneous_share"],
                "offsets": {},
                "deferred_seconds": 0.0,
                "pause_seconds": None,
                "pause_distance_seconds": None,
                "acoustic": acoustic,
                "acoustic_a_silence_share": acoustic["a_quiet_frames"] / acoustic["frames"],
                "roles": [row["role"] for row in placed],
            }
        )
    return rows


def _report(
    per_dialogue, sequences, m3, *, spec: TimelineSpec, overlap_spec: OverlapSpec, args
) -> dict[str, Any]:
    """The timeline report: every number the 2-5 and 2-6 gates are judged on."""
    train = [row for row in per_dialogue if row["split"] == "train"]

    def block(rows):
        total = sum(row["seconds"] for row in rows)
        a_clip = sum(row["seconds"] * (1 - row["a_silence_share_clip"]) for row in rows)
        a_speech = sum(row["seconds"] * (1 - row["a_silence_share_speech"]) for row in rows)
        frames = sum(row["acoustic"]["frames"] for row in rows)
        return {
            "dialogues": len(rows),
            "turns_per_dialogue": sum(row["turns"] for row in rows) / len(rows),
            "a_turns_per_dialogue": sum(row["a_turns"] for row in rows) / len(rows),
            "seconds": _summary([row["seconds"] for row in rows]),
            "frames": _summary([float(row["frames"]) for row in rows]),
            "total_seconds": total,
            "a_silence_share_clip": 1 - a_clip / total,
            "a_silence_share_speech": 1 - a_speech / total,
            "a_silence_share_acoustic": sum(row["acoustic"]["a_quiet_frames"] for row in rows)
            / frames,
            "simultaneous_frames": sum(row["simultaneous_frames"] for row in rows),
            "simultaneous_share": sum(row["simultaneous_frames"] for row in rows)
            / sum(row["frames"] for row in rows),
            "dialogues_without_overlap": sum(1 for row in rows if row["simultaneous_frames"] == 0),
            "acoustic_simultaneous_frames": sum(
                row["acoustic"]["both_loud_frames"] for row in rows
            ),
        }

    return {
        "schema_version": 1,
        "milestone": "M3-R",
        "step": "2-3, 2-5, 2-6",
        "captured_at": args.captured_at,
        "builder": "tools/assemble_dialogue.py",
        "commands": args.command,
        "timeline": {
            "lead_in_seconds": spec.lead_in_seconds,
            "sample_rate": spec.sample_rate,
            "frame_rate_hz": spec.frame_rate_hz,
            "channel_0": "speaker A",
            "channel_1": "speaker B",
            "seed": args.seed,
            "offset_sets": {
                "open_to_body": list(overlap_spec.open_to_body),
                "body_to_backchannel": list(overlap_spec.body_to_backchannel),
                "body_to_close": list(overlap_spec.body_to_close),
            },
            "a1_to_a2": "butt joint - the recording's own pause, nothing added or removed",
            "room_tone": str(args.roomtone_dir) if args.roomtone_dir else None,
            "speech_threshold_rms": args.speech_threshold,
        },
        "m3_recomputed": block(m3) if m3 else None,
        "all": block(per_dialogue),
        "train": block(train),
        "dev": block([row for row in per_dialogue if row["split"] == "dev"]),
        "sequences": sequences,
        "schedule": steps_for_grouping(
            dialogues=len(train),
            group_size=args.group_size,
            global_batch=args.global_batch,
            epochs=args.epochs,
        ),
        "schedule_options": grouping_options(
            dialogues=len(train),
            epochs=args.epochs,
            target_steps=args.target_steps,
            seconds_per_dialogue=sum(row["seconds"] for row in train) / len(train),
            gap_seconds=args.sequence_gap,
        ),
        "min_frames_floor": {
            "floor": args.min_frames_floor,
            "why": "m3/DATASET_SPEC.md sets 200 frames as the shortest dialogue that may "
            "enter the dataset, and says not to lower it.",
            "dialogues_below": sorted(
                (row["dialogue_id"], row["frames"])
                for row in per_dialogue
                if row["frames"] < args.min_frames_floor
            ),
            "sequences_below": sorted(
                (entry["name"], entry["frames"])
                for block in sequences["splits"].values()
                for entry in block["entries"]
                if entry["frames"] < args.min_frames_floor
            ),
            "note": "The floor was written when one dialogue was one training row. It is "
            "now the sequence, which is four dialogues; the per-dialogue files are the "
            "auditable unit, not the trained one. Where a dialogue falls under 200 it is "
            "because dead time was removed, not because content was.",
        },
        "splits": {
            "found": sum(1 for row in per_dialogue if row["pause_seconds"] is not None),
            "pause_seconds": _summary(
                [row["pause_seconds"] for row in per_dialogue if row["pause_seconds"] is not None]
            ),
            "distance_to_mora_target_seconds": _summary(
                [
                    row["pause_distance_seconds"]
                    for row in per_dialogue
                    if row["pause_distance_seconds"] is not None
                ]
            ),
        },
        "per_dialogue": per_dialogue,
    }


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Assemble the M3-R stereo dialogues")
    parser.add_argument("--scripts", type=Path, required=True, help="dialogues-v2.jsonl")
    parser.add_argument("--split-map", type=Path, required=True)
    parser.add_argument("--m3-audio-dir", type=Path, required=True, help="m3/v-real/audio")
    parser.add_argument("--m3-text-dir", type=Path, required=True, help="m3/v-real/text")
    parser.add_argument("--backchannel-dir", type=Path, required=True)
    parser.add_argument("--roomtone-dir", type=Path, default=None, help="omit to leave silence")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--lead-in", type=float, default=0.3)
    parser.add_argument("--group-size", type=int, default=4, help="dialogues per training row")
    parser.add_argument("--sequence-gap", type=float, default=0.4)
    parser.add_argument("--global-batch", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--target-steps", type=int, default=45, help="the M3 step count")
    parser.add_argument("--min-frames-floor", type=int, default=200)
    parser.add_argument("--speech-threshold", type=float, default=0.01)
    parser.add_argument(
        "--compare-m3",
        action="store_true",
        help="recompute the same statistics on the M3 dialogues, so the comparison is a "
        "measurement rather than a quotation",
    )
    parser.add_argument("--captured-at", default=None)
    parser.add_argument(
        "--command",
        action="append",
        default=[],
        help="a command that produced this report, recorded verbatim; repeatable",
    )
    args = parser.parse_args(argv)

    import numpy as np
    import soundfile

    spec = TimelineSpec(lead_in_seconds=args.lead_in)
    overlap_spec = OverlapSpec()
    rows = [
        json.loads(line)
        for line in args.scripts.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assignment = json.loads(args.split_map.read_text(encoding="utf-8"))["assignment"]
    pool = None
    if args.roomtone_dir is not None:
        from tools import room_tone

        pool = room_tone.load_pool(str(args.roomtone_dir))

    per_dialogue: list[dict[str, Any]] = []
    rendered: dict[str, Any] = {}
    words_by_id: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        did = row["dialogue_id"]
        stereo, rate = _read_stereo(str(args.m3_audio_dir / f"{did}.wav"))
        if rate != spec.sample_rate:
            raise ValueError(f"{did}: {rate} Hz, expected {spec.sample_rate}")
        transcript = json.loads((args.m3_text_dir / f"{did}.json").read_text(encoding="utf-8"))
        backchannel, bc_rate = None, spec.sample_rate
        bc_path = args.backchannel_dir / f"{did}-t2-B.wav"
        if bc_path.is_file():
            data, bc_rate = soundfile.read(str(bc_path), dtype="float64", always_2d=True)
            backchannel = np.asarray(data)[:, 0]
        built = build_dialogue(
            row,
            stereo=stereo,
            word_transcript=transcript,
            backchannel=backchannel,
            backchannel_rate=int(bc_rate),
            spec=spec,
            overlap_spec=overlap_spec,
            seed=args.seed,
        )
        dry = render_channels(built["placed"], built["samples"], sample_rate=spec.sample_rate)
        wet = (
            lay_room_tone(
                dry,
                pool,
                seed=stable_seed(args.seed, did, "tone") % 10**6,
                sample_rate=spec.sample_rate,
            )
            if pool is not None
            else dry
        )
        split = assignment[did]
        for directory in ("audio", "text"):
            (args.out_dir / directory).mkdir(parents=True, exist_ok=True)
        _write_stereo(str(args.out_dir / "audio" / f"{did}.wav"), wet, spec.sample_rate)
        (args.out_dir / "text" / f"{did}.json").write_text(
            json.dumps(built["words"], ensure_ascii=False), encoding="utf-8"
        )
        per_dialogue.append(
            _dialogue_stats(did, split, built, dry, spec=spec, threshold=args.speech_threshold)
        )
        rendered[did] = wet
        words_by_id[did] = built["words"]
        print(f"{did} {split} {timeline_seconds(built['placed']):.2f}s", flush=True)

    sequences = _build_sequences(
        rows,
        assignment,
        rendered,
        words_by_id,
        out_dir=args.out_dir,
        spec=spec,
        pool=pool,
        seed=args.seed,
        group_size=args.group_size,
        gap=args.sequence_gap,
    )

    m3 = (
        m3_reference(
            args.m3_audio_dir, args.m3_text_dir, spec=spec, threshold=args.speech_threshold
        )
        if args.compare_m3
        else []
    )
    report = _report(
        per_dialogue,
        sequences,
        m3,
        spec=spec,
        overlap_spec=overlap_spec,
        args=args,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
