"""Reference-free intelligibility: ASR the generated voice, then perplex the transcript.

M3 shipped condition 5 blank, with "no way to measure it" written where the number
belonged. The reason given was that a generated dialogue has no ground-truth text, and
comparing ASR output against the model's own decoded text stream is self-referential - the
more the model repeats itself, the better it scores. That argument is sound, and it rules
out exactly one family of methods: the ones that need a reference.

J-Moshi's own paper does not need one. It runs the generated audio through Whisper and
reports the perplexity a Japanese language model assigns to the transcript
(Interspeech 2025 §4.2). No reference text, no reference audio, no GPU.

**A repeating model scores BETTER on that number, not worse.** Measured here with
llm-jp-3-150m, on strings chosen before any generation was scored:

    今日はとても良い天気ですね。散歩に行きましょうか。      ppl    91.3   (natural)
    ライトンとニューゼルゼンの服と呪縁のトリドンドです      ppl  2637.9   (gibberish)
    ありがとうございました ×4                              ppl   171.0   (a loop)
    あああああああああああああああああああ                  ppl    13.9   (a stuck token)

A model stuck on one mora scores seven times more "fluent" than a real Japanese sentence.
Quoting a perplexity from this module without the repetition beside it would therefore
rank the worst possible failure first. So the two are computed together and published
together: `fluency_row` returns both halves or neither, there is no function here that
hands back a perplexity on its own, and `require_joint_intelligibility` refuses a document
that carries one without the other. This is the same discipline
`tools/speaker_similarity.py` applies to `mean_delta` - a number that cannot be read
correctly should not exist in a form that can be quoted.

The second trap is the denominator. Eight of the ten v-real/epoch5 held-out generations
produce no sound at all, so they produce no transcript, so they contribute no perplexity.
Averaging over "the clips that had a transcript" makes an arm look better the more of it
has gone silent. Every aggregate here is therefore named for its denominator -
`..._transcribed` never stands alone, `denominator` and `empty_transcript` travel with it,
and `clean_transcribed_count` is counted over the full set.

The third trap is Whisper itself: given silence it sometimes invents a fluent Japanese
sentence. A hallucinated transcript over a dead channel would score as the most intelligible
generation in the run. `flag_transcripts_over_collapsed_audio` marks those by joining the
transcript to the acoustic verdict `tools/dialogue_collapse.py` already computes from the
frozen calibration, so no new threshold is invented to catch them.

Everything above `load_channel` imports nothing outside the standard library, so the suite
runs without torch, numpy, soundfile or an ASR model.

Thresholds are NOT chosen here. `RepetitionThresholds` has no defaults, for the reason
`AcousticThresholds` has none: a caller who forgets to pass them would otherwise silently
get a laxer detector than the one the report was written against.
"""

from __future__ import annotations

import math
import statistics
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

#: What a fluency summary must carry before it can be read. Named for the denominators and
#: the two halves, because M3's condition 5 was blank and its replacement must not be a
#: number that looks answerable on its own.
REQUIRED_SUMMARY_FIELDS = (
    "denominator",
    "transcribed",
    "empty_transcript",
    "repetitive_transcripts",
    "median_perplexity_transcribed",
    "median_perplexity_nonrepetitive",
    "perplexity_deflation_from_repetition",
    "repeat_coverage",
    "distinct_char_ratio",
    "clean_transcribed_count",
)

#: What every scored clip must carry. The perplexity fields and the repetition fields are
#: in one tuple deliberately: there is no partial row.
REQUIRED_ROW_FIELDS = (
    "transcribed",
    "perplexity",
    "bits_per_character",
    "scored_tokens",
    "characters",
    "repeat_coverage",
    "distinct_char_ratio",
)


@dataclass(frozen=True)
class RepetitionThresholds:
    """The lines that decide whether a transcript is a loop. No defaults, deliberately.

    Two statistics rather than one, because they fail on different shapes. A transcript
    that cycles a long phrase keeps a high distinct-character ratio while its coverage goes
    to 1.0; a transcript stuck on a single mora keeps its coverage moderate at short n while
    its distinct ratio collapses towards 1/len. Either one alone condemns the transcript.

    `min_scored_tokens` does not condemn anything. It marks transcripts too short for a
    perplexity to mean much: 「ありがとうございました」 is two tokens to this tokenizer and
    scores ppl 973 as a fragment, against 171 when it is repeated four times.
    """

    min_ngram: int
    max_repeat_coverage: float
    min_distinct_char_ratio: float
    min_scored_tokens: int


class TranscriptRepeat(NamedTuple):
    ngram: str
    count: int
    coverage: float


def normalise_transcript(text: str) -> str:
    """NFKC, with the recogniser's whitespace removed.

    Whisper punctuates and spaces Japanese to its own taste, and neither is spoken. Leaving
    the spaces in would let a transcript's repetition hide behind them - 「はい はい はい」
    and 「はいはいはい」 are the same failure - and would charge the language model for
    tokens the model never uttered.
    """
    normalised = unicodedata.normalize("NFKC", text)
    return "".join(char for char in normalised if not char.isspace())


def perplexity_from_nll(total_nll_nats: float, scored_tokens: int) -> float:
    """exp of the mean per-token negative log likelihood.

    Nats in, because that is what a torch log-softmax hands back, and converting once at
    the boundary keeps every caller from choosing a base.
    """
    if scored_tokens <= 0:
        raise ValueError(f"a perplexity needs at least one scored token, got {scored_tokens}")
    if total_nll_nats < 0:
        raise ValueError(f"negative log likelihood cannot be negative, got {total_nll_nats}")
    return math.exp(total_nll_nats / scored_tokens)


def bits_per_character(total_nll_nats: float, characters: int) -> float:
    """The same likelihood normalised by characters instead of tokens.

    Reported alongside the perplexity because perplexity is a per-token quantity and the
    token is the language model's unit, not the speaker's. Two transcripts of the same
    length in characters can differ threefold in token count. This is the version that
    stays comparable when the tokenizer changes, so a later run on a different LM can still
    be lined up against this one.
    """
    if characters <= 0:
        raise ValueError(f"bits per character needs at least one character, got {characters}")
    if total_nll_nats < 0:
        raise ValueError(f"negative log likelihood cannot be negative, got {total_nll_nats}")
    return total_nll_nats / math.log(2.0) / characters


def distinct_character_ratio(text: str) -> float:
    """Unique characters over total. A transcript stuck on one mora scores near zero."""
    if not text:
        raise ValueError("cannot score an empty transcript")
    return len(set(text)) / len(text)


def most_covering_repeat(text: str, *, min_ngram: int) -> TranscriptRepeat | None:
    """The repeated substring that accounts for the largest share of the transcript.

    Occurrences are counted WITHOUT overlap, greedily left to right. Overlapping counts are
    the wrong unit here: in 「ああああ」 the bigram 「ああ」 occurs three times overlapping,
    which is more copies than the string has room for, and a coverage built from that
    exceeds 1.0. Non-overlapping copies are the ones a listener would actually hear.

    Coverage is `count * n / len(text)`: the fraction of the transcript that is copies of
    one phrase. It is the quantity the threshold is applied to, so it is maximised first.
    Ties break towards the SHORTER n-gram, because a longer gram at equal coverage is only
    a multiple of the true period - 「ありがとうございました」 repeated four times is a
    four-fold loop of eleven characters, not a two-fold loop of twenty-two.

    Returns None when nothing repeats, rather than a zero-coverage sentinel, so a caller
    cannot mistake "no repeat found" for "a repeat covering nothing".
    """
    if min_ngram < 1:
        raise ValueError(f"min_ngram must be at least 1, got {min_ngram}")
    length = len(text)
    best: TranscriptRepeat | None = None
    for n in range(min_ngram, length // 2 + 1):
        # One left-to-right pass counts non-overlapping occurrences of every n-gram at
        # once: a gram is counted only when it starts at or after the end of its own
        # previous accepted copy.
        counts: dict[str, int] = {}
        next_free: dict[str, int] = {}
        for start in range(length - n + 1):
            gram = text[start : start + n]
            if start >= next_free.get(gram, 0):
                counts[gram] = counts.get(gram, 0) + 1
                next_free[gram] = start + n
        repeated = {gram: count for gram, count in counts.items() if count > 1}
        if not repeated:
            # Nothing of this length repeats without overlap, so nothing longer can either.
            break
        gram = max(repeated, key=lambda g: repeated[g])
        candidate = TranscriptRepeat(gram, repeated[gram], repeated[gram] * n / length)
        if best is None or candidate.coverage > best.coverage:
            best = candidate
    return best


def describe_repetition(text: str, *, min_ngram: int) -> dict[str, Any]:
    """Both repetition statistics for one transcript, always together."""
    if not text:
        raise ValueError("cannot score an empty transcript")
    repeat = most_covering_repeat(text, min_ngram=min_ngram)
    return {
        "repeat_ngram": repeat.ngram if repeat else None,
        "repeat_count": repeat.count if repeat else 0,
        "repeat_coverage": repeat.coverage if repeat else 0.0,
        "distinct_char_ratio": distinct_character_ratio(text),
    }


def fluency_row(
    *,
    clip_id: str,
    transcript: str,
    total_nll_nats: float | None,
    scored_tokens: int | None,
    thresholds: RepetitionThresholds,
) -> dict[str, Any]:
    """Score one clip on fluency AND repetition, or record that it said nothing.

    The two halves are produced by one function on purpose. Nothing in this module returns
    a perplexity without the repetition beside it, so a report cannot be assembled that
    quotes the fluency number alone - which, on the evidence in the module docstring, would
    rank a model stuck on 「あ」 above a model speaking Japanese.

    A clip with no transcript is not a clip with a good score. It gets `transcribed: False`
    and null metrics, never a zero, because a zero would average in as excellent and the
    arms that failed hardest are exactly the ones that fell silent.
    """
    cleaned = normalise_transcript(transcript)
    if not cleaned:
        return {
            "id": clip_id,
            "transcript": "",
            "transcribed": False,
            "characters": 0,
            "scored_tokens": 0,
            "perplexity": None,
            "bits_per_character": None,
            "repeat_ngram": None,
            "repeat_count": 0,
            "repeat_coverage": None,
            "distinct_char_ratio": None,
            "repetitive": False,
            "short": True,
            "clean": False,
        }
    if total_nll_nats is None or scored_tokens is None:
        raise ValueError(f"{clip_id}: a non-empty transcript must carry a likelihood")

    repetition = describe_repetition(cleaned, min_ngram=thresholds.min_ngram)
    repetitive = (
        repetition["repeat_coverage"] >= thresholds.max_repeat_coverage
        or repetition["distinct_char_ratio"] < thresholds.min_distinct_char_ratio
    )
    short = scored_tokens < thresholds.min_scored_tokens
    return {
        "id": clip_id,
        "transcript": cleaned,
        "transcribed": True,
        "characters": len(cleaned),
        "scored_tokens": scored_tokens,
        "perplexity": perplexity_from_nll(total_nll_nats, scored_tokens),
        "bits_per_character": bits_per_character(total_nll_nats, len(cleaned)),
        **repetition,
        "repetitive": repetitive,
        "short": short,
        "clean": not repetitive and not short,
    }


def _spread(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "max": ordered[-1],
        "stdev": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
    }


def summarise_fluency(rows: Iterable[Mapping[str, Any]], *, denominator: int) -> dict[str, Any]:
    """Aggregate one group, with every denominator named.

    Three perplexity aggregates and no key called `mean_perplexity`:

    - `median_perplexity_transcribed` averages the clips that produced sound. It includes
      the loops, so it is the optimistic number in two directions at once.
    - `median_perplexity_nonrepetitive` drops the loops. This is the one that answers
      "how fluent is this model when it is actually saying something".
    - `perplexity_deflation_from_repetition` is the first minus the second. It is negative
      exactly when repetition is pulling the fluency number down, and its size says by how
      much. It is the reason both are computed.

    `clean_transcribed_count` is over the full denominator, not over the transcribed ones,
    so an arm cannot improve its score by going silent.
    """
    rows = list(rows)
    if denominator <= 0:
        raise ValueError(f"a group needs a positive denominator, got {denominator}")
    if len(rows) != denominator:
        raise ValueError(
            f"got {len(rows)} scored clips against a denominator of {denominator}; a "
            "denominator that does not match the rows would report a count against the "
            "wrong total"
        )

    transcribed = [row for row in rows if row["transcribed"]]
    nonrepetitive = [row for row in transcribed if not row["repetitive"]]
    all_ppl = [float(row["perplexity"]) for row in transcribed]
    clean_ppl = [float(row["perplexity"]) for row in nonrepetitive]
    median_all = statistics.median(all_ppl) if all_ppl else None
    median_clean = statistics.median(clean_ppl) if clean_ppl else None

    return {
        "denominator": denominator,
        "transcribed": len(transcribed),
        "empty_transcript": denominator - len(transcribed),
        "transcribed_ratio": len(transcribed) / denominator,
        "repetitive_transcripts": sum(1 for row in transcribed if row["repetitive"]),
        "short_transcripts": sum(1 for row in transcribed if row["short"]),
        "clean_transcribed_count": sum(1 for row in rows if row["clean"]),
        "clean_transcribed_ratio": sum(1 for row in rows if row["clean"]) / denominator,
        "median_perplexity_transcribed": median_all,
        "median_perplexity_nonrepetitive": median_clean,
        "perplexity_deflation_from_repetition": (
            median_all - median_clean
            if median_all is not None and median_clean is not None
            else None
        ),
        "perplexity_transcribed": _spread(all_ppl),
        "perplexity_nonrepetitive": _spread(clean_ppl),
        "bits_per_character_transcribed": _spread(
            [float(row["bits_per_character"]) for row in transcribed]
        ),
        "repeat_coverage": _spread([float(row["repeat_coverage"]) for row in transcribed]),
        "distinct_char_ratio": _spread([float(row["distinct_char_ratio"]) for row in transcribed]),
        "characters": _spread([float(row["characters"]) for row in transcribed]),
    }


def merge_by_id(
    fluency_rows: Sequence[Mapping[str, Any]],
    token_rows: Sequence[Mapping[str, Any]],
    *,
    token_key: str = "generation",
) -> list[dict[str, Any]]:
    """Put the transcript metrics and the token metrics on one row, or refuse.

    The plan asks for perplexity and repetition in the same table. Building that table by
    zipping two orderings is how the M3 record ended up with an arm scored 9 clips against
    another's 10 and nobody able to say which clip went missing, so the ids are matched and
    a mismatch raises.
    """
    token_by_id = {}
    for row in token_rows:
        key = str(row[token_key])
        stem = key[:-4] if key.endswith(".npy") else key
        if stem in token_by_id:
            raise ValueError(f"duplicate token row for {stem!r}")
        token_by_id[stem] = row

    fluency_ids = {str(row["id"]) for row in fluency_rows}
    if fluency_ids != set(token_by_id):
        only_audio = sorted(fluency_ids - set(token_by_id))
        only_tokens = sorted(set(token_by_id) - fluency_ids)
        raise ValueError(
            f"transcripts and token files disagree: {len(only_audio)} only in audio "
            f"{only_audio[:5]}, {len(only_tokens)} only in tokens {only_tokens[:5]}"
        )

    merged = []
    for row in fluency_rows:
        token_row = token_by_id[str(row["id"])]
        merged.append(
            {
                **row,
                "audio_entropy_bits": token_row.get("audio_entropy_bits"),
                "audio_distinct_tokens": token_row.get("audio_distinct_tokens"),
                "audio_top_token_share": token_row.get("audio_top_token_share"),
                "acoustic_collapse": token_row.get("acoustic_collapse"),
                "text_stream_distinct_ratio": token_row.get("distinct_ratio"),
                "text_stream_emitted_tokens": token_row.get("emitted_tokens"),
                "text_stream_longest_repeat_count": token_row.get("longest_repeat_count"),
                "text_stream_silent": token_row.get("silent"),
            }
        )
    return merged


def flag_transcripts_over_collapsed_audio(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Count the clips whose transcript cannot be trusted because the channel was dead.

    Whisper given silence sometimes emits a fluent Japanese sentence. Such a transcript
    would score as the most intelligible clip in the run while the model made no sound, so
    it has to be visible. The verdict used is the acoustic one `tools/dialogue_collapse.py`
    already computes against its frozen calibration - no new threshold is introduced here,
    because a threshold invented to catch an inconvenient clip is indistinguishable from a
    threshold invented to admit one.
    """
    rows = list(rows)
    scored = [row for row in rows if row.get("acoustic_collapse") is not None]
    if len(scored) != len(rows):
        raise ValueError(
            f"{len(scored)} of {len(rows)} rows carry an acoustic verdict; a hallucination "
            "count over a partial join would understate itself"
        )
    suspect = [str(row["id"]) for row in rows if row["transcribed"] and row["acoustic_collapse"]]
    return {
        "collapsed_audio": sum(1 for row in rows if row["acoustic_collapse"]),
        "collapsed_audio_with_transcript": len(suspect),
        "suspect_ids": suspect,
    }


def require_joint_intelligibility(document: Mapping[str, Any]) -> None:
    """Refuse to publish an intelligibility document that can be misquoted.

    Four demands, and the first is the one this module exists for:

    (a) every scored clip carries its repetition statistics next to its perplexity,
    (b) every group names its denominator and how much of it fell silent,
    (c) the deflation between the two perplexity aggregates is published,
    (d) the ASR model and the language model are named, because a perplexity is a property
        of the scoring model as much as of the speech.

    Raising rather than warning, in the spirit of `require_likeness_report`: a report that
    cannot be read correctly should not exist to be quoted.
    """
    missing: list[str] = []

    for field in ("asr_model", "language_model"):
        if not document.get(field):
            missing.append(field)

    groups = document.get("groups")
    if not isinstance(groups, Mapping) or not groups:
        missing.append("groups")
        groups = {}

    for name, group in groups.items():
        summary = group.get("summary") if isinstance(group, Mapping) else None
        if not isinstance(summary, Mapping):
            missing.append(f"groups.{name}.summary")
            continue
        for field in REQUIRED_SUMMARY_FIELDS:
            if field not in summary:
                missing.append(f"groups.{name}.summary.{field}")
        rows = group.get("clips")
        if not isinstance(rows, Sequence) or not rows:
            missing.append(f"groups.{name}.clips")
            continue
        for row in rows:
            for field in REQUIRED_ROW_FIELDS:
                if field not in row:
                    missing.append(f"groups.{name}.clips[{row.get('id')}].{field}")

    if missing:
        raise ValueError(
            "intelligibility document is missing: " + "; ".join(sorted(set(missing))[:20])
        )


def calibration_band(rows: Iterable[Mapping[str, Any]], *, label: str) -> dict[str, Any]:
    """The band a measurement is read against, in the same fields as a group summary.

    `CLAUDE.md`: a similarity of 0.74 means nothing until you know a real human scores 0.70.
    A perplexity of 2600 means nothing until you know what the same Whisper and the same
    language model score on a clean recording of a real person.
    """
    rows = list(rows)
    transcribed = [row for row in rows if row["transcribed"]]
    if not transcribed:
        raise ValueError(f"{label}: a calibration band needs at least one transcribed clip")
    return {
        "label": label,
        "count": len(rows),
        "transcribed": len(transcribed),
        "perplexity": _spread([float(row["perplexity"]) for row in transcribed]),
        "bits_per_character": _spread([float(row["bits_per_character"]) for row in transcribed]),
        "repeat_coverage": _spread([float(row["repeat_coverage"]) for row in transcribed]),
        "distinct_char_ratio": _spread([float(row["distinct_char_ratio"]) for row in transcribed]),
        "characters": _spread([float(row["characters"]) for row in transcribed]),
    }


# --------------------------------------------------------------------------------------
# Everything below here touches the filesystem or a model. Heavy imports stay inside the
# functions so the suite runs without torch, numpy, soundfile or an ASR model.
# --------------------------------------------------------------------------------------


def load_channel(path: str, *, channel: int, target_rate: int = 16000):
    """One channel of a wav, resampled for the recogniser, as float32 in [-1, 1].

    Channel 0 of an M3 decode is the model's own voice; channel 1 is the partner stream,
    which the generation leaves at the Mimi silence floor. Mixing them would measure the
    silence as well as the speech.
    """
    import numpy as np
    import soundfile as sf
    import soxr

    samples, rate = sf.read(path, dtype="float32", always_2d=True)
    if channel >= samples.shape[1]:
        raise ValueError(
            f"{path}: asked for channel {channel} of a {samples.shape[1]}-channel file"
        )
    mono = np.ascontiguousarray(samples[:, channel])
    rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
    if rate != target_rate:
        mono = soxr.resample(mono, rate, target_rate).astype("float32")
    return mono, rms


def transcribe_tree(
    audio_root: str,
    *,
    channel: int,
    model_size: str,
    compute_type: str,
    cpu_threads: int,
    beam_size: int,
) -> dict[str, Any]:
    """Transcribe every wav under `audio_root`, grouped by its directory.

    Decoding is the expensive half, so it is a separate step with its own artifact: the
    transcripts are written once and scored many times, and a change to the language model
    never costs another ASR pass.
    """
    import pathlib

    from faster_whisper import WhisperModel

    root = pathlib.Path(audio_root)
    paths = sorted(root.rglob("*.wav"))
    if not paths:
        raise ValueError(f"{audio_root}: no wav files to transcribe")

    model = WhisperModel(
        model_size, device="cpu", compute_type=compute_type, cpu_threads=cpu_threads, num_workers=1
    )
    groups: dict[str, list[dict[str, Any]]] = {}
    for path in paths:
        audio, rms = load_channel(str(path), channel=channel)
        segments, _info = model.transcribe(
            audio,
            language="ja",
            beam_size=beam_size,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        segments = list(segments)
        text = "".join(segment.text for segment in segments)
        group = str(path.parent.relative_to(root))
        groups.setdefault(group, []).append(
            {
                "id": path.stem,
                "transcript": text,
                "channel_rms": rms,
                "channel_dbfs": 20.0 * math.log10(rms) if rms > 0 else None,
                "no_speech_prob": (
                    min(segment.no_speech_prob for segment in segments) if segments else None
                ),
            }
        )
    for rows in groups.values():
        rows.sort(key=lambda row: (0, int(row["id"])) if row["id"].isdigit() else (1, row["id"]))
    return {
        "audio_root": audio_root,
        "channel": channel,
        "asr_model": f"faster-whisper/{model_size}",
        "asr_settings": {
            "compute_type": compute_type,
            "cpu_threads": cpu_threads,
            "beam_size": beam_size,
            "temperature": 0.0,
            "language": "ja",
            "vad_filter": False,
            "condition_on_previous_text": False,
            "resampled_to_hz": 16000,
        },
        "groups": groups,
    }


def _language_model(name: str, threads: int):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(threads)
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).eval()
    return tokenizer, model


def score_texts(texts: Sequence[str], *, model_name: str, threads: int) -> list[dict[str, Any]]:
    """Total negative log likelihood, in nats, of each text under a Japanese LM.

    A BOS token is prepended and every real token after it is scored; no EOS is appended,
    because these are fragments of a dialogue and charging the model for failing to end a
    sentence would measure turn length rather than fluency.
    """
    import torch

    tokenizer, model = _language_model(model_name, threads)
    bos = tokenizer.bos_token_id
    if bos is None:
        raise ValueError(f"{model_name}: no BOS token, so the first real token cannot be scored")

    scored: list[dict[str, Any]] = []
    for text in texts:
        if not text:
            scored.append({"total_nll_nats": None, "scored_tokens": 0})
            continue
        ids = [bos, *tokenizer.encode(text, add_special_tokens=False)]
        if len(ids) < 2:
            scored.append({"total_nll_nats": None, "scored_tokens": 0})
            continue
        tensor = torch.tensor([ids])
        with torch.no_grad():
            logits = model(tensor).logits.float()
        log_probs = torch.log_softmax(logits[0, :-1], dim=-1)
        targets = tensor[0, 1:]
        nll = -log_probs.gather(1, targets[:, None]).squeeze(1)
        scored.append(
            {"total_nll_nats": float(nll.sum().item()), "scored_tokens": int(targets.numel())}
        )
    return scored


def _thresholds_from_calibration(calibration: Mapping[str, Any]) -> RepetitionThresholds:
    try:
        block = calibration["repetition_thresholds"]
        return RepetitionThresholds(
            min_ngram=int(block["min_ngram"]),
            max_repeat_coverage=float(block["max_repeat_coverage"]),
            min_distinct_char_ratio=float(block["min_distinct_char_ratio"]),
            min_scored_tokens=int(block["min_scored_tokens"]),
        )
    except KeyError as missing:
        raise ValueError(f"calibration is missing {missing}") from missing


def _load_token_groups(token_root: str, text_calibration: str, acoustic_calibration: str):
    import json
    import pathlib

    from tools.dialogue_collapse import (
        CollapseThresholds,
        acoustic_thresholds_from_calibration,
        score_directory,
    )

    text_doc = json.loads(pathlib.Path(text_calibration).read_text(encoding="utf-8"))
    acoustic_doc = json.loads(pathlib.Path(acoustic_calibration).read_text(encoding="utf-8"))
    thresholds = CollapseThresholds(**text_doc["thresholds"])
    acoustic = acoustic_thresholds_from_calibration(acoustic_doc)
    source = text_doc["calibration_source"]

    root = pathlib.Path(token_root)
    groups: dict[str, list[dict[str, Any]]] = {}
    for directory in sorted(root.rglob("generated_tokens")):
        groups[str(directory.parent.relative_to(root))] = score_directory(
            str(directory),
            padding_id=source["text_padding_id"],
            end_padding_id=source["end_of_text_padding_id"],
            thresholds=thresholds,
            acoustic_thresholds=acoustic,
        )
    return groups


def main(argv: Sequence[str] | None = None) -> int:
    """Transcribe a tree of decoded audio, or score a set of transcripts.

    Two subcommands rather than one, because the ASR pass costs minutes and the scoring
    pass costs seconds, and a language model swap must not force the ASR pass again.
    """
    import argparse
    import json
    import pathlib

    parser = argparse.ArgumentParser(description="Reference-free intelligibility, with repetition")
    sub = parser.add_subparsers(dest="command", required=True)

    asr = sub.add_parser("transcribe", help="decoded audio -> transcripts")
    asr.add_argument("--audio-root", required=True)
    asr.add_argument("--output", required=True)
    asr.add_argument("--channel", type=int, default=0, help="0 is the model's own voice")
    asr.add_argument("--model-size", default="small", help="small or smaller; this runs on a Mac")
    asr.add_argument("--compute-type", default="int8")
    asr.add_argument("--cpu-threads", type=int, default=2)
    asr.add_argument("--beam-size", type=int, default=1)

    score = sub.add_parser("score", help="transcripts -> joint fluency and repetition report")
    score.add_argument("--transcripts", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--calibration", required=True, help="repetition thresholds, frozen")
    score.add_argument("--language-model", default="llm-jp/llm-jp-3-150m")
    score.add_argument("--threads", type=int, default=2)
    score.add_argument("--token-root", help="generated_tokens tree, to join the token metrics")
    score.add_argument("--text-calibration")
    score.add_argument("--acoustic-calibration")

    args = parser.parse_args(argv)

    if args.command == "transcribe":
        document = transcribe_tree(
            args.audio_root,
            channel=args.channel,
            model_size=args.model_size,
            compute_type=args.compute_type,
            cpu_threads=args.cpu_threads,
            beam_size=args.beam_size,
        )
        pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.output).write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        total = sum(len(rows) for rows in document["groups"].values())
        print(json.dumps({"groups": len(document["groups"]), "clips": total}))
        return 0

    transcripts = json.loads(pathlib.Path(args.transcripts).read_text(encoding="utf-8"))
    calibration = json.loads(pathlib.Path(args.calibration).read_text(encoding="utf-8"))
    thresholds = _thresholds_from_calibration(calibration)

    flat: list[tuple[str, dict[str, Any]]] = []
    for group, rows in transcripts["groups"].items():
        for row in rows:
            flat.append((group, row))
    likelihoods = score_texts(
        [normalise_transcript(row["transcript"]) for _group, row in flat],
        model_name=args.language_model,
        threads=args.threads,
    )

    scored: dict[str, list[dict[str, Any]]] = {}
    for (group, row), likelihood in zip(flat, likelihoods, strict=True):
        entry = fluency_row(
            clip_id=row["id"],
            transcript=row["transcript"],
            total_nll_nats=likelihood["total_nll_nats"],
            scored_tokens=likelihood["scored_tokens"],
            thresholds=thresholds,
        )
        entry["channel_dbfs"] = row.get("channel_dbfs")
        scored.setdefault(group, []).append(entry)

    token_groups = {}
    if args.token_root:
        if not (args.text_calibration and args.acoustic_calibration):
            raise SystemExit("--token-root needs --text-calibration and --acoustic-calibration")
        token_groups = _load_token_groups(
            args.token_root, args.text_calibration, args.acoustic_calibration
        )

    groups: dict[str, Any] = {}
    for group, rows in sorted(scored.items()):
        if group in token_groups:
            rows = merge_by_id(rows, token_groups[group])
            hallucination = flag_transcripts_over_collapsed_audio(rows)
        else:
            hallucination = None
        groups[group] = {
            "summary": summarise_fluency(rows, denominator=len(rows)),
            "transcripts_over_collapsed_audio": hallucination,
            "clips": rows,
        }

    document = {
        "schema_version": 1,
        "asr_model": transcripts["asr_model"],
        "asr_settings": transcripts["asr_settings"],
        "language_model": args.language_model,
        "repetition_thresholds": {
            "min_ngram": thresholds.min_ngram,
            "max_repeat_coverage": thresholds.max_repeat_coverage,
            "min_distinct_char_ratio": thresholds.min_distinct_char_ratio,
            "min_scored_tokens": thresholds.min_scored_tokens,
        },
        "repetition_calibration": args.calibration,
        "groups": groups,
    }
    require_joint_intelligibility(document)
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.output).write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                group: {
                    key: body["summary"][key]
                    for key in (
                        "denominator",
                        "transcribed",
                        "repetitive_transcripts",
                        "clean_transcribed_count",
                        "median_perplexity_transcribed",
                        "median_perplexity_nonrepetitive",
                    )
                }
                for group, body in groups.items()
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    raise SystemExit(main())
