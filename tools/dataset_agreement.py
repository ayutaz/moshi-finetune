"""Re-run M3's nine dataset agreement checks on a rebuilt dialogue dataset.

Why this exists
---------------
`reports/m3-dataset-agreement.json` recorded nine counts that had to be exactly zero before
M3 could train, and the script that produced them was never committed. M3-R rebuilt the
dataset - room tone under the non-speaking channel, a backchannel turn that overlaps, two
dialogues dropped at the 200-frame floor - so the same nine counts have to be taken again
on the new material, and two of the nine no longer mean what they meant in M3.

What changed, and why the check had to change with it
-----------------------------------------------------
`channel_mismatches` in M3 read "during each turn the speaking channel carries energy and
the other carries none". In M3-R the other channel carries room tone by design, and during
speaker A's body turn it also carries speaker B's backchannel by design. Neither is a
defect; both would fail M3's wording. The check is therefore restated over the frames of a
turn that no other speaker's turn covers, as two clauses:

  a. the speaking channel's median frame RMS exceeds the other channel's, and
  b. the other channel stays under the speech threshold on those same frames.

Clause (a) is a comparison, not a level. An absolute floor was tried first, at the
assembler's own `timeline.speech_threshold_rms`, and it flagged 22 of the 154 speaker-A
turns in the shipped dataset - because speaker A is about 19 dB quieter than speaker B
throughout, not because anything was silent. A left/right swap moves energy between the
channels and changes no absolute level, so the comparison is the clause with power and it
has no number to tune. Clause (b) keeps the absolute threshold, for the different question
of whether speech leaked onto a channel whose transcript says nobody is talking.

`--negative_control` swaps the two channels and requires the same check to fail. On the
shipped dataset the narrowest correct turn sits +9.56 dB and the widest swapped one -9.56,
so the two populations are 19.1 dB apart with nothing in between: that is what separates
"the channels are right" from "the check cannot tell".

`text_frames_exceeding_audio` in M3 read as a length comparison. Taken literally it fires on
every dialogue in any dataset, M3's included, because `tools/tokenize_text.py` pads the text
stream to `(last_token_end + 1s) * frame_rate` - a second of trailing padding past the last
word. What `tools/prepare_dataset.merge_text_audio` then truncates is that padding. The
defect the count was named for is a *token* falling off the end, so that is what is counted
here, with the raw length difference reported beside it rather than gated.

The pure functions take plain sequences and import nothing heavy, so the segmentation, the
window arithmetic and the verdicts are testable without numpy, soundfile or a dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

#: Mimi's frame rate and the samples one frame covers at 24 kHz.
SAMPLE_RATE = 24000
FRAME_RATE_HZ = 12.5
FRAME_SAMPLES = 1920

#: m3/DATASET_SPEC.md: the shortest dialogue that may enter the dataset. Not to be lowered.
MIN_FRAMES = 200

#: reports/m3r-timeline.json timeline.speech_threshold_rms - the assembler's own value.
SPEECH_RMS_THRESHOLD = 0.01

#: tools/tts_audio_report.py --max-clipped-run: isolated full-scale samples are a write-time
#: artefact and inaudible (verified in M2); a longer run is real distortion.
MAX_SATURATED_RUN = 2

#: A digital-silence stretch this long on a non-speaking channel is what room tone exists to
#: prevent: tools/room_tone.py measured Mimi mapping a long run of exact zeros onto one code.
MAX_DIGITAL_SILENCE_SECONDS = 1.0

#: tools/tokenize_text.py pads to (last_token_end + 1s) * frame_rate.
TEXT_STREAM_TAIL_SECONDS = 1.0

_PUNCTUATION = re.compile(r"[\s、。，．,.!?！？「」『』・…ー]")

#: Frame edges are compared in float seconds; this absorbs the division, nothing more.
_EDGE_EPSILON = 1e-9


class AgreementError(ValueError):
    """The dataset cannot be measured as described - a missing file, a broken shape."""


# --------------------------------------------------------------------------------------
# text
# --------------------------------------------------------------------------------------


def normalise_text(text: str) -> str:
    """NFKC, then drop whitespace and punctuation, so spacing cannot hide a difference."""
    return _PUNCTUATION.sub("", unicodedata.normalize("NFKC", text))


def reading(text: str) -> str:
    """Katakana reading, via pyopenjtalk's grapheme-to-phoneme.

    Surface comparison is not enough: the frontend that produced the word transcripts
    expands numerals, so `1931年` in the script is `千九百三十一年` in the transcript. M3 hit
    exactly this on v-063 and switched to readings; M2's intelligibility gate had the same
    fault and failed a working checkpoint 26 of 30 times.
    """
    import pyopenjtalk

    return _PUNCTUATION.sub("", pyopenjtalk.g2p(text, kana=True))


# --------------------------------------------------------------------------------------
# turns
# --------------------------------------------------------------------------------------


def turn_intervals(
    segments: Sequence[Mapping[str, Any]], *, gap_tolerance: float = 1e-6
) -> list[dict[str, Any]]:
    """Recover the turns from a word transcript.

    The transcript is one flat list sorted by start time, so consecutive entries alternate
    between speakers wherever two turns overlap - grouping by runs of the same speaker
    recovers eleven fragments from a five-turn dialogue, not five turns. What does separate
    the turns is time: `tools/assemble_dialogue.py` allocates word times inside a turn back
    to back, so `end` of one word is bit-identical to `start` of the next, and any
    discontinuity inside one speaker's words is a turn boundary.
    """
    turns: list[dict[str, Any]] = []
    for speaker in sorted({str(segment["speaker"]) for segment in segments}):
        words = [segment for segment in segments if str(segment["speaker"]) == speaker]
        current: list[Mapping[str, Any]] = []
        for word in words:
            if current and abs(float(word["start"]) - float(current[-1]["end"])) > gap_tolerance:
                turns.append(_turn(speaker, current))
                current = []
            current.append(word)
        if current:
            turns.append(_turn(speaker, current))
    turns.sort(key=lambda turn: (turn["start"], turn["speaker"]))
    for index, turn in enumerate(turns):
        turn["index"] = index
    return turns


def _turn(speaker: str, words: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "speaker": speaker,
        "start": float(words[0]["start"]),
        "end": float(words[-1]["end"]),
        "words": len(words),
        "text": "".join(str(word["word"]) for word in words),
    }


def timestamp_problems(
    segments: Sequence[Mapping[str, Any]],
    *,
    duration_seconds: float,
    tolerance: float = 1e-3,
) -> list[dict[str, Any]]:
    """Word times that leave the audio, invert, or run backwards inside one speaker."""
    problems: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        start = float(segment["start"])
        end = float(segment["end"])
        if start < -tolerance:
            problems.append({"kind": "start_before_zero", "index": index, "start": start})
        if end > duration_seconds + tolerance:
            problems.append(
                {"kind": "end_past_audio", "index": index, "end": end, "audio": duration_seconds}
            )
        if end < start - tolerance:
            problems.append(
                {"kind": "end_before_start", "index": index, "start": start, "end": end}
            )
    for speaker in sorted({str(segment["speaker"]) for segment in segments}):
        words = [segment for segment in segments if str(segment["speaker"]) == speaker]
        for earlier, later in zip(words, words[1:], strict=False):
            if float(later["start"]) < float(earlier["start"]) - tolerance:
                problems.append(
                    {
                        "kind": "not_monotonic",
                        "speaker": speaker,
                        "start": float(later["start"]),
                        "previous_start": float(earlier["start"]),
                    }
                )
    return problems


def exclusive_windows(
    turn: Mapping[str, float], others: Iterable[Mapping[str, Any]]
) -> list[tuple[float, float]]:
    """The parts of `turn` that no other speaker's turn covers.

    A backchannel is placed to overlap the body turn it answers, so during that overlap the
    other channel is *supposed* to carry sound. Subtracting the overlaps leaves the stretch
    where only one person is speaking, which is where a left/right swap would show.
    """
    windows = [(float(turn["start"]), float(turn["end"]))]
    for other in others:
        if str(other.get("speaker")) == str(turn.get("speaker")):
            continue
        low, high = float(other["start"]), float(other["end"])
        cut: list[tuple[float, float]] = []
        for start, end in windows:
            if high <= start or low >= end:
                cut.append((start, end))
                continue
            if low > start:
                cut.append((start, low))
            if high < end:
                cut.append((high, end))
        windows = cut
    return [(start, end) for start, end in windows if end > start]


def frames_inside(
    windows: Sequence[tuple[float, float]], *, frame_count: int, frame_seconds: float
) -> list[int]:
    """Frame indices lying wholly inside one of `windows`.

    Wholly inside, not merely touching: a frame straddling a turn boundary carries both
    sides and cannot testify about either.
    """
    import math

    chosen: set[int] = set()
    for start, end in windows:
        # The epsilon is not slack in the criterion, it is float division: 0.16 / 0.08 is
        # 1.9999999999999998, which would drop a frame that lies exactly inside the window.
        first = max(0, math.ceil(start / frame_seconds - _EDGE_EPSILON))
        last = min(frame_count, math.floor(end / frame_seconds + _EDGE_EPSILON))
        chosen.update(range(first, last))
    return sorted(chosen)


def longest_true_run(flags: Sequence[bool]) -> int:
    """Longest run of consecutive True values."""
    best = run = 0
    for flag in flags:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


# --------------------------------------------------------------------------------------
# verdicts
# --------------------------------------------------------------------------------------


def channel_problems(
    measurement: Mapping[str, Any], *, threshold: float = SPEECH_RMS_THRESHOLD
) -> list[dict[str, Any]]:
    """Decide one turn from its measured numbers.

    Clause (a) is a comparison between the two channels over the same frames, not a level
    against a constant. The event this check exists to catch is a left/right swap, which
    moves energy from one channel to the other and changes no absolute level at all - so
    the comparison is what has discriminating power, and it has no threshold to tune.

    An earlier form of this function used an absolute floor, `own_median_rms >=
    speech_threshold_rms`, and it flagged 22 of the 154 speaker-A turns in the shipped
    v-real-v2. Measured against the same dialogues' room tone those turns sit 30-40 times
    above the floor of the channel; what the flag actually found is that speaker A is
    about 19 dB quieter than speaker B throughout the dataset. That is worth recording -
    `reports/m3r-dataset-agreement.json` records it - but it is not a channel error, and
    the absolute number is kept in the row rather than gated on.

    Clause (b) does use the absolute threshold, for a different question: whether speech
    has leaked onto a channel whose transcript says nobody is talking. A leak is loud in
    absolute terms by definition, and the room tone underneath it is 50 times lower.

    A turn with no exclusive part is not a channel error either - eight of the nine are
    backchannels the assembler deliberately buried inside speaker A's turn. It is returned
    as `no_exclusive_frames` so the caller can count it separately, because clause (b)
    cannot be evaluated for such a turn and silence about that would be a false pass.
    """
    problems: list[dict[str, Any]] = []
    if int(measurement["exclusive_frames"]) == 0:
        problems.append({"kind": "no_exclusive_frames"})
        for problem in problems:
            problem.update(
                {
                    "dialogue": measurement.get("dialogue"),
                    "turn": measurement.get("turn"),
                    "speaker": measurement.get("speaker"),
                }
            )
        return problems
    own = float(measurement["own_exclusive_median_rms"])
    other = float(measurement["other_exclusive_median_rms"])
    if own <= other:
        problems.append(
            {
                "kind": "speaking_channel_not_dominant",
                "own_exclusive_median_rms": own,
                "other_exclusive_median_rms": other,
            }
        )
    if float(measurement["other_max_rms"]) >= threshold:
        problems.append(
            {
                "kind": "other_channel_loud",
                "other_max_rms": float(measurement["other_max_rms"]),
                "threshold": threshold,
            }
        )
    for problem in problems:
        problem.update(
            {
                "dialogue": measurement.get("dialogue"),
                "turn": measurement.get("turn"),
                "speaker": measurement.get("speaker"),
            }
        )
    return problems


def text_tokens_lost(
    text_ids: Sequence[int], *, audio_frames: int, padding_id: int
) -> dict[str, Any]:
    """How much of the text stream `merge_text_audio` will cut off.

    It truncates to the audio length. Trailing padding being cut is the design; a
    non-padding token being cut is a word the model will never be asked to predict.
    """
    last_token = -1
    for index, value in enumerate(text_ids):
        if int(value) != padding_id:
            last_token = index
    lost = [int(value) for value in text_ids[audio_frames:] if int(value) != padding_id]
    return {
        "text_frames": len(text_ids),
        "audio_frames": audio_frames,
        "length_difference": len(text_ids) - audio_frames,
        "last_non_padding_index": last_token,
        "non_padding_tokens_truncated": len(lost),
    }


def excluded_id_sightings(
    excluded: Sequence[str], places: Mapping[str, Iterable[str]]
) -> list[dict[str, Any]]:
    """Where a dropped dialogue still appears.

    `places` maps a place name to the identifiers it holds. An identifier counts as a
    sighting when the excluded name appears in it as a path component or whole token, so
    `train/v-047` is found and `v-0470` is not.
    """
    sightings: list[dict[str, Any]] = []
    for name in excluded:
        pattern = re.compile(rf"(?<![0-9A-Za-z-]){re.escape(name)}(?![0-9A-Za-z])")
        for place, identifiers in places.items():
            hits = sorted({value for value in identifiers if pattern.search(str(value))})
            if hits:
                sightings.append({"dialogue": name, "place": place, "found": hits})
    return sightings


#: Counted but not gated. `turns_never_alone` is a shape the assembler produces on purpose;
#: it belongs in the record, not in a gate that M3 never wrote it into.
OBSERVED_COUNTS = ("turns_never_alone",)


def counts_from_problems(problems: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """The nine M3 counts, all of which must be exactly zero."""
    counts = {
        "channel_mismatches": 0,
        "timestamp_violations": 0,
        "text_mismatches": 0,
        "non_stereo": 0,
        "wrong_sample_rate": 0,
        "zero_length": 0,
        "below_min_frames": 0,
        "text_frames_exceeding_audio": 0,
        "saturated_files": 0,
    }
    for problem in problems:
        key = str(problem["count"])
        if key in OBSERVED_COUNTS:
            continue
        if key not in counts:
            raise AgreementError(f"unknown count {key!r}")
        counts[key] += 1
    return counts


def observed_counts(problems: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """The tallies that are recorded rather than gated."""
    counts = dict.fromkeys(OBSERVED_COUNTS, 0)
    for problem in problems:
        key = str(problem["count"])
        if key in counts:
            counts[key] += 1
    return counts


# --------------------------------------------------------------------------------------
# audio - heavy imports stay inside
# --------------------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_wav(path: Path):
    """Stereo float samples plus the header fields the counts are taken from."""
    import soundfile

    data, rate = soundfile.read(str(path), dtype="float64", always_2d=True)
    return data, int(rate)


def frame_rms(channel, *, frame_samples: int = FRAME_SAMPLES):
    import numpy as np

    count = len(channel) // frame_samples
    if count == 0:
        return np.zeros(0)
    frames = np.asarray(channel[: count * frame_samples]).reshape(count, frame_samples)
    return np.sqrt((frames**2).mean(axis=1))


def frame_zero_share(channel, *, frame_samples: int = FRAME_SAMPLES):
    import numpy as np

    count = len(channel) // frame_samples
    if count == 0:
        return np.zeros(0)
    frames = np.asarray(channel[: count * frame_samples]).reshape(count, frame_samples)
    return (frames == 0.0).mean(axis=1)


def best_lag_ncc(needle, haystack) -> dict[str, float]:
    """Normalised cross-correlation of `needle` against `haystack`, maximised over lag.

    FFT-based, because the direct form is 24k x 530k multiplications per dialogue.
    """
    import numpy as np

    needle = np.asarray(needle, dtype=np.float64)
    haystack = np.asarray(haystack, dtype=np.float64)
    n = len(needle)
    if n == 0 or len(haystack) < n:
        raise AgreementError("needle is empty or longer than the haystack")
    needle = needle - needle.mean()
    size = 1 << (len(haystack) + n).bit_length()
    correlation = np.fft.irfft(
        np.fft.rfft(haystack, size) * np.conj(np.fft.rfft(needle, size)), size
    )[: len(haystack) - n + 1]
    # Sliding sums by prefix sum, not np.convolve: the direct convolution against an
    # n-long box is 24k x 530k multiplications per dialogue and takes minutes each.
    cumulative = np.concatenate(([0.0], np.cumsum(haystack)))
    cumulative_sq = np.concatenate(([0.0], np.cumsum(haystack**2)))
    window_sum = cumulative[n:] - cumulative[:-n]
    window_sq = cumulative_sq[n:] - cumulative_sq[:-n]
    variance = np.maximum(window_sq - window_sum**2 / n, 0.0)
    denominator = np.sqrt(variance) * np.sqrt((needle**2).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = np.where(denominator > 0, correlation / denominator, 0.0)
    index = int(np.argmax(scores))
    return {"ncc": float(scores[index]), "lag_seconds": index / SAMPLE_RATE}


def resample_to(samples, *, source_rate: int, target_rate: int):
    """The assembler's resampler, so the comparison is not measuring two of them."""
    import torch
    import torchaudio

    if source_rate == target_rate:
        return samples
    torch.set_num_threads(2)
    tensor = torch.as_tensor(samples, dtype=torch.float64)
    return torchaudio.functional.resample(tensor, source_rate, target_rate).numpy()


# --------------------------------------------------------------------------------------
# the check
# --------------------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _summarise(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "min": float(min(values)),
        "median": float(statistics.median(values)),
        "mean": float(statistics.fmean(values)),
        "max": float(max(values)),
    }


def _measure_dialogue(
    *,
    dialogue_id: str,
    split: str,
    audio_path: Path,
    transcript_path: Path,
    script: Mapping[str, Any],
    text_ids: Mapping[str, Sequence[int]],
    audio_frames: int,
    padding_id: int,
    threshold: float,
    swap_channels: bool,
) -> dict[str, Any]:
    import numpy as np

    data, rate = read_wav(audio_path)
    channels = data.shape[1]
    if swap_channels and channels == 2:
        data = data[:, ::-1]
    duration = data.shape[0] / rate if rate else 0.0
    segments = json.loads(transcript_path.read_text(encoding="utf-8"))
    turns = turn_intervals(segments)

    problems: list[dict[str, Any]] = []
    if channels != 2:
        problems.append({"count": "non_stereo", "dialogue": dialogue_id, "channels": channels})
    if rate != SAMPLE_RATE:
        problems.append({"count": "wrong_sample_rate", "dialogue": dialogue_id, "rate": rate})
    if data.shape[0] == 0:
        problems.append({"count": "zero_length", "dialogue": dialogue_id})
    if audio_frames < MIN_FRAMES:
        problems.append(
            {"count": "below_min_frames", "dialogue": dialogue_id, "frames": audio_frames}
        )

    for problem in timestamp_problems(segments, duration_seconds=duration):
        problems.append({"count": "timestamp_violations", "dialogue": dialogue_id, **problem})

    # --- text: the script's turns against the transcript's turns, by reading
    script_turns = list(script["turns"])
    text_rows: list[dict[str, Any]] = []
    if len(script_turns) != len(turns):
        problems.append(
            {
                "count": "text_mismatches",
                "dialogue": dialogue_id,
                "kind": "turn_count",
                "script": len(script_turns),
                "transcript": len(turns),
            }
        )
    else:
        for position, (want, got) in enumerate(zip(script_turns, turns, strict=False)):
            if str(want["speaker"]) != got["speaker"]:
                problems.append(
                    {
                        "count": "text_mismatches",
                        "dialogue": dialogue_id,
                        "kind": "speaker",
                        "turn": position,
                        "script": want["speaker"],
                        "transcript": got["speaker"],
                    }
                )
                continue
            surface_equal = normalise_text(str(want["text"])) == normalise_text(got["text"])
            want_reading = reading(str(want["text"]))
            got_reading = reading(got["text"])
            if want_reading != got_reading:
                problems.append(
                    {
                        "count": "text_mismatches",
                        "dialogue": dialogue_id,
                        "kind": "reading",
                        "turn": position,
                        "script": str(want["text"])[:60],
                        "transcript": got["text"][:60],
                    }
                )
            got["role"] = want.get("role")
            text_rows.append(
                {
                    "turn": position,
                    "role": want.get("role"),
                    "speaker": got["speaker"],
                    "surface_equal": surface_equal,
                    "reading_equal": want_reading == got_reading,
                }
            )

    # --- channels
    channel_of = {"A": 0, "B": 1}
    rms = {name: frame_rms(data[:, index]) for name, index in channel_of.items()}
    zero_share = {name: frame_zero_share(data[:, index]) for name, index in channel_of.items()}
    frame_count = min(len(rms["A"]), len(rms["B"]))
    frame_seconds = 1.0 / FRAME_RATE_HZ
    turn_rows: list[dict[str, Any]] = []
    for turn in turns:
        speaker = turn["speaker"]
        other = "B" if speaker == "A" else "A"
        own_frames = frames_inside(
            [(turn["start"], turn["end"])], frame_count=frame_count, frame_seconds=frame_seconds
        )
        exclusive = frames_inside(
            exclusive_windows(turn, turns), frame_count=frame_count, frame_seconds=frame_seconds
        )
        measurement = {
            "dialogue": dialogue_id,
            "turn": turn["index"],
            "speaker": speaker,
            "role": turn.get("role"),
            "start": round(turn["start"], 4),
            "end": round(turn["end"], 4),
            "seconds": round(turn["end"] - turn["start"], 4),
            "own_frames": len(own_frames),
            "exclusive_frames": len(exclusive),
            "own_median_rms": float(np.median(rms[speaker][own_frames])) if own_frames else 0.0,
            "other_median_rms": float(np.median(rms[other][own_frames])) if own_frames else 0.0,
            "own_exclusive_median_rms": (
                float(np.median(rms[speaker][exclusive])) if exclusive else 0.0
            ),
            "other_exclusive_median_rms": (
                float(np.median(rms[other][exclusive])) if exclusive else 0.0
            ),
            "other_max_rms": float(np.max(rms[other][exclusive])) if exclusive else 0.0,
            "own_below_speech_threshold": bool(
                own_frames and float(np.median(rms[speaker][own_frames])) < threshold
            ),
        }
        turn_rows.append(measurement)
        for problem in channel_problems(measurement, threshold=threshold):
            # `no_exclusive_frames` is a shape of the timeline, not a channel error: the
            # assembler buries a backchannel inside speaker A's turn on purpose. It is
            # counted separately so that it neither inflates the gate nor disappears.
            key = (
                "turns_never_alone"
                if problem["kind"] == "no_exclusive_frames"
                else "channel_mismatches"
            )
            problems.append({"count": key, **problem})

    # --- room tone: the frames where this channel's speaker is not in any turn
    room_rows: list[dict[str, Any]] = []
    for name in ("A", "B"):
        speaking = set()
        for turn in turns:
            if turn["speaker"] != name:
                continue
            speaking.update(
                frames_inside(
                    [(turn["start"], turn["end"])],
                    frame_count=frame_count,
                    frame_seconds=frame_seconds,
                )
            )
        quiet = [index for index in range(frame_count) if index not in speaking]
        flags = [bool(zero_share[name][index] == 1.0) for index in quiet]
        room_rows.append(
            {
                "dialogue": dialogue_id,
                "channel": name,
                "quiet_frames": len(quiet),
                "median_rms": float(np.median(rms[name][quiet])) if quiet else 0.0,
                "min_rms": float(np.min(rms[name][quiet])) if quiet else 0.0,
                "all_zero_frames": sum(flags),
                "longest_all_zero_seconds": longest_true_run(flags) * frame_seconds,
            }
        )

    # --- saturation
    peak = np.abs(data).max() if data.size else 0.0
    saturated = np.abs(data) >= (32767.0 / 32768.0)
    longest = max(longest_true_run(saturated[:, index].tolist()) for index in range(data.shape[1]))
    if longest > MAX_SATURATED_RUN:
        problems.append(
            {"count": "saturated_files", "dialogue": dialogue_id, "longest_run": int(longest)}
        )

    # --- text stream truncation
    stream_rows = []
    for name in ("A", "B"):
        row = text_tokens_lost(
            list(text_ids[name]), audio_frames=audio_frames, padding_id=padding_id
        )
        row.update({"dialogue": dialogue_id, "speaker": name})
        stream_rows.append(row)
        if row["non_padding_tokens_truncated"]:
            problems.append({"count": "text_frames_exceeding_audio", **row})

    return {
        "dialogue_id": dialogue_id,
        "split": split,
        "duration_seconds": duration,
        "frames": audio_frames,
        "channels": channels,
        "sample_rate": rate,
        "peak": float(peak),
        "longest_saturated_run": int(longest),
        "turns": turn_rows,
        "text": text_rows,
        "room_tone": room_rows,
        "text_streams": stream_rows,
        "problems": problems,
    }


def _cmd_check(args: argparse.Namespace) -> int:
    import numpy as np

    root = Path(args.repo_root).resolve()
    data_root = root / args.data_root
    dataset_root = root / args.dataset_root
    manifest_rows = read_jsonl(root / args.manifest)
    scripts = {row["dialogue_id"]: row for row in read_jsonl(root / args.dialogues)}
    split_map = json.loads((root / args.split_map).read_text(encoding="utf-8"))

    per_dialogue: list[dict[str, Any]] = []
    problems: list[dict[str, Any]] = []
    backchannel_rows: list[dict[str, Any]] = []

    for row in manifest_rows:
        dialogue_id = str(row["group_id"])
        split = str(row["split"])
        tokenized = row["tokenized"]
        audio_path = data_root / tokenized["audio_wav"]
        transcript_path = data_root / tokenized["word_transcript"]
        tok_audio = data_root / tokenized["audio_wav"].replace("/audio/", "/tok-audio/").replace(
            ".wav", ".npz"
        )
        tok_text = data_root / tokenized["word_transcript"].replace("/text/", "/tok-text/").replace(
            ".json", ".npz"
        )
        with np.load(tok_audio) as bundle:
            audio_frames = int(bundle["A"].shape[-1])
            if int(bundle["B"].shape[-1]) != audio_frames:
                raise AgreementError(f"{tok_audio}: speaker columns disagree on frame count")
        with np.load(tok_text) as bundle:
            text_ids = {name: bundle[name].tolist() for name in ("A", "B")}

        measured = _measure_dialogue(
            dialogue_id=dialogue_id,
            split=split,
            audio_path=audio_path,
            transcript_path=transcript_path,
            script=scripts[dialogue_id],
            text_ids=text_ids,
            audio_frames=audio_frames,
            padding_id=args.text_padding_id,
            threshold=args.speech_threshold,
            swap_channels=args.negative_control,
        )
        problems.extend(measured.pop("problems"))
        per_dialogue.append(measured)

        if not args.skip_backchannel:
            backchannel = row.get("backchannel")
            script_roles = [str(turn.get("role")) for turn in scripts[dialogue_id]["turns"]]
            if not backchannel:
                # Two of the 80 source sentences carried no comma to split on, so their
                # scripts are the three-turn M3 shape and have no backchannel to place.
                # Absent-and-absent is agreement; absent on one side only is not.
                backchannel_rows.append(
                    {
                        "dialogue": dialogue_id,
                        "kind": "absent",
                        "script_has_backchannel": "backchannel" in script_roles,
                        "agrees": "backchannel" not in script_roles,
                    }
                )
            else:
                backchannel_rows.append(
                    _check_backchannel(
                        dialogue_id=dialogue_id,
                        data_root=data_root,
                        audio_path=audio_path,
                        backchannel=backchannel,
                        script=scripts[dialogue_id],
                        turns=measured["turns"],
                        swap_channels=args.negative_control,
                        lag_tolerance=args.backchannel_lag_tolerance,
                    )
                )

    counts = counts_from_problems(problems)
    observed = observed_counts(problems)

    shipped_ids = {str(row["group_id"]) for row in manifest_rows}
    parquet_ids: list[str] = []
    for path in sorted((dataset_root / "parquet").glob("*.parquet")):
        import pandas as pd

        parquet_ids.extend(str(value) for value in pd.read_parquet(path)["dialogue_id"])
    places = {
        "manifest": sorted(shipped_ids),
        "parquet": sorted(parquet_ids),
        "tok-audio": sorted(path.stem for path in dataset_root.glob("*/tok-audio/*.npz")),
        "tok-text": sorted(path.stem for path in dataset_root.glob("*/tok-text/*.npz")),
        "split-audio": sorted(path.stem for path in dataset_root.glob("*/audio/*.wav")),
        "split-text": sorted(path.stem for path in dataset_root.glob("*/text/*.json")),
        "split-map": sorted(split_map.get("assignment", {})),
        "dialogue-scripts": sorted(scripts),
        "staging-audio": sorted(path.stem for path in (dataset_root / "audio").glob("*.wav")),
        "staging-text": sorted(path.stem for path in (dataset_root / "text").glob("*.json")),
    }
    sightings = excluded_id_sightings(args.excluded, places)
    shipping_places = {"manifest", "parquet", "tok-audio", "tok-text", "split-audio", "split-text"}
    blocking = [
        row for row in sightings if row["place"] in shipping_places or row["place"] == "split-map"
    ]

    room = [row for dialogue in per_dialogue for row in dialogue["room_tone"]]
    room_failures = [
        row
        for row in room
        if row["longest_all_zero_seconds"] >= MAX_DIGITAL_SILENCE_SECONDS
        or row["median_rms"] <= 0.0
    ]
    backchannel_failures = [
        row
        for row in backchannel_rows
        if not row.get("agrees", False)
        or (
            "ncc" in row
            and (
                float(row["ncc"]) < args.backchannel_floor
                or not row.get("text_matches_script", False)
                or not row.get("sha256_matches_manifest", False)
                or not row.get("placed_where_the_transcript_says", False)
            )
        )
    ]
    level = _level_report(per_dialogue)

    payload = {
        "schema_version": 1,
        "milestone": "M3-R",
        "dataset_id": args.dataset_id,
        "negative_control": bool(args.negative_control),
        "dialogues": len(per_dialogue),
        "counts": counts,
        "observed_counts": observed,
        "speaker_levels": level,
        "problems": problems,
        "excluded": {
            "dialogues": list(args.excluded),
            "sightings": sightings,
            "blocking": blocking,
            "places_scanned": sorted(places),
        },
        "room_tone": {
            "channels_checked": len(room),
            "failures": room_failures,
            "median_rms": _summarise([row["median_rms"] for row in room]),
            "longest_all_zero_seconds": _summarise(
                [row["longest_all_zero_seconds"] for row in room]
            ),
        },
        "backchannel": {
            "checked": len(backchannel_rows),
            "with_audio": sum(1 for row in backchannel_rows if "ncc" in row),
            "absent_and_absent_from_the_script": sum(
                1 for row in backchannel_rows if row.get("kind") == "absent" and row.get("agrees")
            ),
            "failures": backchannel_failures,
            "ncc": _summarise([float(row["ncc"]) for row in backchannel_rows if "ncc" in row]),
            "lag_error_seconds": _summarise(
                [float(row["lag_error_seconds"]) for row in backchannel_rows if "ncc" in row]
            ),
            "turn_median_rms": _summarise(
                [float(row["turn_median_rms"]) for row in backchannel_rows if "ncc" in row]
            ),
        },
        "per_dialogue": per_dialogue,
        "backchannel_rows": backchannel_rows,
    }
    payload["passed"] = (
        all(value == 0 for value in counts.values())
        and not blocking
        and not room_failures
        and not backchannel_failures
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0 if payload["passed"] else 1


def _check_backchannel(
    *,
    dialogue_id: str,
    data_root: Path,
    audio_path: Path,
    backchannel: Mapping[str, Any],
    script: Mapping[str, Any],
    turns: Sequence[Mapping[str, Any]],
    swap_channels: bool,
    lag_tolerance: float,
) -> dict[str, Any]:
    """Does the backchannel the script asks for agree with the audio that was written?

    Three separate questions, because a single one of them can pass while the turn is
    still wrong. The clip has to be the one the manifest names (sha256); its text has to
    be the script's text; and it has to sit where the word transcript says the turn is -
    a perfect cross-correlation only says the clip is *somewhere* in the channel, and the
    text stream is aligned to the transcript's times, not to wherever the audio landed.
    """
    import soundfile

    wav_path = data_root / str(backchannel["path"])
    clip, clip_rate = soundfile.read(str(wav_path), dtype="float64", always_2d=True)
    clip = clip.mean(axis=1)
    clip = resample_to(clip, source_rate=int(clip_rate), target_rate=SAMPLE_RATE)
    data, _ = read_wav(audio_path)
    if swap_channels and data.shape[1] == 2:
        data = data[:, ::-1]
    scored = best_lag_ncc(clip, data[:, 1])
    turn_index = int(backchannel["turn_index"])
    script_text = str(script["turns"][turn_index]["text"])
    row = [turn for turn in turns if int(turn["turn"]) == turn_index]
    transcript_start = float(row[0]["start"]) if row else float("nan")
    lag_error = abs(float(scored["lag_seconds"]) - transcript_start)
    return {
        "dialogue": dialogue_id,
        "turn_index": turn_index,
        "role": script["turns"][turn_index].get("role"),
        "seconds": len(clip) / SAMPLE_RATE,
        "clip_rms": float((clip**2).mean() ** 0.5),
        "turn_median_rms": float(row[0]["own_median_rms"]) if row else 0.0,
        "transcript_start": transcript_start,
        "lag_error_seconds": lag_error,
        "placed_where_the_transcript_says": bool(lag_error <= lag_tolerance),
        "text_matches_script": normalise_text(str(backchannel["text"]))
        == normalise_text(script_text),
        "sha256_matches_manifest": _sha256(wav_path) == str(backchannel["sha256"]),
        "agrees": True,
        **scored,
    }


def _level_report(per_dialogue: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """How loud each speaker is, and how far the other channel sits below them.

    Recorded rather than gated: nothing in M3 pre-registered a level criterion, and a
    number invented now to judge material that already exists is a number chosen to judge
    this material. It goes in the record because it is a property of what the model will
    be trained on that no report so far states.
    """
    import math

    rows = [turn for dialogue in per_dialogue for turn in dialogue["turns"]]
    per_speaker = {}
    for speaker in ("A", "B"):
        values = [float(row["own_median_rms"]) for row in rows if row["speaker"] == speaker]
        per_speaker[speaker] = _summarise(values)
    gap = None
    if per_speaker["A"].get("median") and per_speaker["B"].get("median"):
        gap = 20.0 * math.log10(per_speaker["A"]["median"] / per_speaker["B"]["median"])
    return {
        "unit": "median frame RMS over the turn, linear full scale",
        "turns": len(rows),
        "per_speaker": per_speaker,
        "a_minus_b_db": gap,
        "turns_below_speech_threshold": sum(
            1 for row in rows if row.get("own_below_speech_threshold")
        ),
        "note": (
            "The turns under the speech threshold are all speaker A. They sit 30-40 times "
            "above the room tone on the same channel, so they are quiet speech, not silence."
        ),
    }


def dominance_margins_db(measurement: Mapping[str, Any]) -> list[dict[str, Any]]:
    """How far each judgeable turn's own channel sits above the other, in dB.

    The calibration the verdict needs: "the speaking channel dominates" means nothing as a
    boolean until the margin is known, and until the same margin is known for material that
    is definitely wrong.
    """
    import math

    rows = []
    for dialogue in measurement["per_dialogue"]:
        for turn in dialogue["turns"]:
            if not turn["exclusive_frames"]:
                continue
            own = float(turn["own_exclusive_median_rms"])
            other = float(turn["other_exclusive_median_rms"])
            if own <= 0 or other <= 0:
                continue
            rows.append(
                {
                    "dialogue": turn["dialogue"],
                    "turn": turn["turn"],
                    "speaker": turn["speaker"],
                    "role": turn.get("role"),
                    "margin_db": 20.0 * math.log10(own / other),
                }
            )
    return rows


def _merge(base: dict[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in extra.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def assemble_report(
    *,
    measured: Mapping[str, Any],
    swapped: Mapping[str, Any],
    parquet: Mapping[str, Any],
    annotations: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the milestone report out of the measurements, transcribing no number by hand.

    M3-R's second stage found that the six scripts which produced `v-real-v2` lived under
    `data/`, which is gitignored, so the procedure that made the dataset was not in the
    repository. Assembling the report from the measurement files rather than from a person
    reading them keeps the same failure from happening to the report.
    """
    per_dialogue = list(measured["per_dialogue"])
    splits: dict[str, Any] = {}
    for split in sorted({str(row["split"]) for row in per_dialogue}):
        rows = [row for row in per_dialogue if str(row["split"]) == split]
        splits[split] = {
            "dialogues": len(rows),
            "frames": _summarise([int(row["frames"]) for row in rows]),
            "seconds": _summarise([float(row["duration_seconds"]) for row in rows]),
        }

    shipped = dominance_margins_db(measured)
    inverted = dominance_margins_db(swapped)
    turns = [turn for row in per_dialogue for turn in row["turns"]]
    streams = [row for dialogue in per_dialogue for row in dialogue["text_streams"]]

    report: dict[str, Any] = {
        "schema_version": 1,
        "milestone": "M3-R",
        "dataset_id": measured["dataset_id"],
        "checks": dict(measured["counts"]),
        "scope": {
            "dialogues": len(per_dialogue),
            "turns": len(turns),
            "turns_judgeable_for_channel": len(shipped),
            "turns_compared_against_the_script": sum(len(row["text"]) for row in per_dialogue),
            "text_streams": len(streams),
            "splits": splits,
        },
        "parquet": dict(parquet),
        "negative_control": {
            "what": "the two channels are swapped and the same check is run again",
            "why": (
                "A verdict that never fails is not a verdict. Swapping is the exact defect "
                "the channel check exists for and it changes no absolute level, so it "
                "separates 'the channels are right' from 'the check cannot tell'."
            ),
            "counts": dict(swapped["counts"]),
            "dialogues_with_a_channel_mismatch": len(
                {
                    problem["dialogue"]
                    for problem in swapped["problems"]
                    if problem["count"] == "channel_mismatches"
                }
            ),
            "backchannel_ncc": dict(swapped["backchannel"]["ncc"]),
        },
        "channel_margin_db": {
            "definition": (
                "20*log10(median frame RMS of the speaking channel / of the other channel), "
                "over the frames of the turn that no other speaker's turn covers"
            ),
            "shipped": _summarise([row["margin_db"] for row in shipped]),
            "channels_swapped": _summarise([row["margin_db"] for row in inverted]),
            "shipped_turns_at_or_below_zero": sum(1 for row in shipped if row["margin_db"] <= 0),
            "swapped_turns_above_zero": sum(1 for row in inverted if row["margin_db"] > 0),
            "narrowest_shipped": sorted(shipped, key=lambda row: row["margin_db"])[:5],
            "separation_db": (
                min(row["margin_db"] for row in shipped) - max(row["margin_db"] for row in inverted)
                if shipped and inverted
                else None
            ),
        },
        "m3r_additions": {
            "excluded_dialogues": dict(measured["excluded"]),
            "backchannel": {
                key: value for key, value in measured["backchannel"].items() if key != "failures"
            }
            | {"failures": measured["backchannel"]["failures"]},
            "room_tone": dict(measured["room_tone"]),
        },
        "observed_not_gated": {
            "counts": dict(measured["observed_counts"]),
            "turns_never_alone": [
                problem
                for problem in measured["problems"]
                if problem["count"] == "turns_never_alone"
            ],
            "speaker_levels": dict(measured["speaker_levels"]),
            "text_stream_length_difference": _summarise(
                [int(row["length_difference"]) for row in streams]
            ),
            "non_padding_tokens_truncated": sum(
                int(row["non_padding_tokens_truncated"]) for row in streams
            ),
        },
        "problems": list(measured["problems"]),
    }
    report["nine_counts_status"] = (
        "pass" if all(value == 0 for value in report["checks"].values()) else "fail"
    )
    report["status"] = "pass" if measured.get("passed") else "fail"
    return _merge(report, annotations or {})


def _cmd_report(args: argparse.Namespace) -> int:
    report = assemble_report(
        measured=json.loads(Path(args.measured).read_text(encoding="utf-8")),
        swapped=json.loads(Path(args.negative_control).read_text(encoding="utf-8")),
        parquet=json.loads(Path(args.parquet).read_text(encoding="utf-8")),
        annotations=(
            json.loads(Path(args.annotations).read_text(encoding="utf-8"))
            if args.annotations
            else None
        ),
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0 if report["status"] == "pass" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    report = sub.add_parser("report", help="assemble the milestone report from measurements")
    report.add_argument("--measured", required=True)
    report.add_argument("--negative_control", required=True)
    report.add_argument("--parquet", required=True)
    report.add_argument("--annotations", help="authored prose merged over the measured numbers")
    report.add_argument("--out")
    report.set_defaults(func=_cmd_report)
    check = sub.add_parser("check", help="the nine counts plus the M3-R additions")
    check.add_argument("--repo_root", default=".")
    check.add_argument("--data_root", default="data/experiments/tsukuyomi_ojousama")
    check.add_argument("--dataset_root", required=True)
    check.add_argument("--manifest", required=True)
    check.add_argument("--dialogues", required=True)
    check.add_argument("--split_map", required=True)
    check.add_argument("--dataset_id", required=True)
    check.add_argument("--excluded", nargs="*", default=[])
    check.add_argument("--text_padding_id", type=int, default=3)
    check.add_argument("--speech_threshold", type=float, default=SPEECH_RMS_THRESHOLD)
    check.add_argument("--backchannel_floor", type=float, default=0.9)
    check.add_argument(
        "--backchannel_lag_tolerance",
        type=float,
        default=1.0 / FRAME_RATE_HZ,
        help="how far the placed clip may sit from the transcript's turn start, in seconds",
    )
    check.add_argument("--skip_backchannel", action="store_true")
    check.add_argument(
        "--negative_control",
        action="store_true",
        help="swap the two channels; every dialogue must then fail",
    )
    check.add_argument("--out", help="write the JSON here instead of stdout")
    check.set_defaults(func=_cmd_check)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
