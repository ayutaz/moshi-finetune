"""Collect room tone from the corpus and lay it under a channel that is not speaking.

Why this exists
---------------
M3 wrote digital silence into the non-speaking channel. Mimi maps a long stretch of exact
zeros onto one code - token 1316 takes 96.8% of a 60 s bed - so `pad` and `speaker A is
silent` became the same event to 98% accuracy and the cheapest way to lower the loss was to
stop speaking. Replacing that silence with real recorded background breaks the equivalence.

What the corpus actually contains
---------------------------------
The premise "collect the silent stretches of the 100 corpus recordings and you have room
tone" does not survive measurement. The recordings are noise-gated: 15.8% of all samples
are exactly 0.0, 39% of the quiet frames in a typical recording are digital zero, and the
quiet frames that are not zero sit at an RMS of 8.9e-5 - about -98 dBFS, under three LSB of
the PCM_16 the dataset is written in, so what survives the write is a three-level signal.
141 s of that material spliced back at its own level still tokenises as digital silence for
41.6% of frames. There is no usable noise floor to harvest.

What is left is the material just above that floor: the tails, breaths and mouth noise in
the 2e-3..1e-2 RMS band. That is real acoustics from the same recording chain, and it is
what this module collects. Voiced segments are rejected by an autocorrelation test so no
speech ends up on a channel whose transcript says nobody is speaking - real speech scores
0.94 on that test, the material kept here scores under 0.5.

How the bed is built
--------------------
A short loop teaches its own period, so units are drawn from a reshuffled deck: no segment
repeats until every segment has been used. Each unit is randomly reversed, resampled by
+/-15%, given a random 8-band EQ and normalised to a level drawn from a small fixed set,
then equal-power crossfaded onto its neighbour. The level set is the parameter that matters
most - Mimi's code for a quiet stationary bed is chosen almost entirely by its level, so
drawing from several levels is what spreads the histogram.

How it is judged
----------------
The calibration figure - natural silence at distinct 37 / top share 0.31 - was measured on
silent frames *inside recordings that also contain speech*, so that is the protocol
`verify --channels-dir` reproduces: fill the gaps of a real dialogue channel, tokenise the
whole channel, and count over the gap frames only. An isolated 30-60 s bed is a different
and much harsher question, and is reported separately rather than conflated with it.
"""

from __future__ import annotations

import json
import math
import os
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

SAMPLE_RATE = 24000
FRAME_RATE_HZ = 12.5
#: Median RMS of a speech frame on channel A of the shipped v-real dialogues. Levels are
#: quoted against this so "how loud is the room tone" has an answer that is not dBFS.
SPEECH_REFERENCE_RMS = 0.0258
#: Mimi's code for a long stretch of digital silence, confirmed in the M3 verification.
SILENCE_TOKEN = 1316


@dataclass(frozen=True)
class CollectSpec:
    """Which parts of a recording become room tone."""

    band_low: float = 2e-3
    band_high: float = 1e-2
    hop_seconds: float = 0.01
    min_seconds: float = 0.10
    guard_seconds: float = 0.0
    max_voicing: float = 0.5
    sample_rate: int = SAMPLE_RATE

    @property
    def hop_samples(self) -> int:
        return int(round(self.hop_seconds * self.sample_rate))

    @property
    def min_frames(self) -> int:
        return max(1, int(round(self.min_seconds / self.hop_seconds)))

    @property
    def guard_frames(self) -> int:
        return int(round(self.guard_seconds / self.hop_seconds))


@dataclass(frozen=True)
class RenderSpec:
    """How the collected segments become an arbitrary length of room tone.

    `levels` is the RMS each unit is normalised to and `level_weights` how often each is
    drawn. Three quiet levels a factor of two apart land in three different Mimi codes -
    one level lands in one - and the two rare loud levels are what keeps a long gap from
    settling: a bed that never changes regime converges however many quiet levels it has.
    The loud draws are 7% of units, so the floor a listener hears stays where the quiet
    levels put it while the RMS rises.
    """

    levels: tuple[float, ...] = (1e-4, 2e-4, 4e-4, 3.2e-3, 6.4e-3)
    level_weights: tuple[float, ...] = (9.0, 9.0, 9.0, 1.0, 1.0)
    eq_db: float = 28.0
    speed_jitter: float = 0.15
    reverse_probability: float = 0.5
    crossfade_seconds: float = 0.02
    sample_rate: int = SAMPLE_RATE
    eq_bands_hz: tuple[float, ...] = (60.0, 150.0, 350.0, 800.0, 1800.0, 4000.0, 8000.0, 12000.0)

    @property
    def crossfade_samples(self) -> int:
        return int(round(self.crossfade_seconds * self.sample_rate))

    @property
    def weights(self) -> tuple[float, ...]:
        return self.level_weights or tuple(1.0 for _ in self.levels)


#: The whole pool at one quiet level band. Passes on real dialogue channels whose gaps are
#: short, degrades past a 10 s gap. Keep for a timeline that has no long gaps.
QUIET_FLOOR = RenderSpec(levels=(1e-4, 2e-4, 4e-4), level_weights=(1.0, 1.0, 1.0))
#: The default: the same quiet floor with rare louder events. Holds the gate from a 2 s gap
#: to a 20 s one, which is the range a 60 s M3-R sequence produces.
FLOOR_WITH_EVENTS = RenderSpec()


@dataclass(frozen=True)
class UnitDraw:
    """One decision the renderer made, recoverable from the seed alone."""

    segment: int
    level: float
    reverse: bool
    speed: float
    eq_gains: tuple[float, ...] = field(default=())


# --------------------------------------------------------------------------------------
# pure logic - no numpy, no torch, no soundfile
# --------------------------------------------------------------------------------------


def runs_in_band(values: Sequence[float], *, low: float, high: float) -> list[tuple[int, int]]:
    """Half-open [start, end) runs where `low <= value < high`."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(values):
        inside = low <= value < high
        if inside and start is None:
            start = index
        elif not inside and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(values)))
    return runs


def trim_runs(
    runs: Sequence[tuple[int, int]], *, guard_frames: int, min_frames: int
) -> list[tuple[int, int]]:
    """Shrink each run by `guard_frames` at both ends and drop what is left too short.

    The guard exists because the frame next to speech still carries its decay. It is a
    separate knob from `min_frames` so a guard of zero - which is what the collected band
    wants, since the tail *is* the material - stays expressible.
    """
    if guard_frames < 0:
        raise ValueError(f"guard_frames must not be negative, got {guard_frames}")
    kept = []
    for start, end in runs:
        a, b = start + guard_frames, end - guard_frames
        if b - a >= min_frames:
            kept.append((a, b))
    return kept


def equal_power_ramp(length: int) -> list[float]:
    """Rising ramp whose square, plus the reversed ramp's square, is 1 at every point.

    Two uncorrelated recordings crossfaded with a linear ramp dip in the middle - their
    powers add, not their amplitudes. Room tone is uncorrelated with itself, so the fade
    has to be equal-power or every splice is an audible hole.
    """
    if length <= 0:
        return []
    if length == 1:
        return [1.0]
    return [math.sqrt(0.5 * (1.0 - math.cos(math.pi * i / (length - 1)))) for i in range(length)]


def deck_order(count: int, needed: int, rng: random.Random) -> list[int]:
    """Indices in reshuffled-deck order: nothing repeats until everything has been used.

    Sampling with replacement would let one segment land twice in a row often enough to
    hear; a loop would teach its period. A deck does neither.
    """
    if count <= 0:
        raise ValueError("cannot draw from an empty pool")
    order: list[int] = []
    deck: list[int] = []
    while len(order) < needed:
        if not deck:
            deck = list(range(count))
            rng.shuffle(deck)
        order.append(deck.pop())
    return order


def plan_units(
    durations: Sequence[float], seconds: float, *, spec: RenderSpec, seed: int
) -> list[UnitDraw]:
    """Decide every unit of a bed before any audio is touched.

    Separating the plan from the rendering is what makes a seed mean something: the same
    seed gives the same sequence of segments, levels, reversals and EQ curves whatever the
    audio backend does.
    """
    if not durations:
        raise ValueError("cannot plan a bed from an empty pool")
    if seconds <= 0:
        raise ValueError(f"seconds must be positive, got {seconds}")
    if not spec.levels:
        raise ValueError("RenderSpec.levels must not be empty")
    if len(spec.weights) != len(spec.levels):
        raise ValueError(f"{len(spec.levels)} levels but {len(spec.weights)} weights")

    rng = random.Random(seed)
    overlap = spec.crossfade_seconds
    # Shortest advance a single unit can make: the shortest segment, sped up as far as the
    # jitter allows, minus the crossfade it gives back. Counting from that bound over-draws,
    # and the loop below stops the moment the bed is covered. A plan that runs out early is
    # a hole in the middle of the bed, so the error goes the other way on purpose.
    slowest = max(min(durations) / (1.0 + spec.speed_jitter) - overlap, overlap)
    order = deck_order(len(durations), int(seconds / slowest) + 8, rng)

    plan: list[UnitDraw] = []
    covered = 0.0
    for index in order:
        level = rng.choices(spec.levels, weights=spec.weights, k=1)[0]
        reverse = rng.random() < spec.reverse_probability
        speed = rng.uniform(1.0 - spec.speed_jitter, 1.0 + spec.speed_jitter)
        gains = tuple(rng.uniform(-spec.eq_db, spec.eq_db) for _ in spec.eq_bands_hz)
        plan.append(
            UnitDraw(segment=index, level=level, reverse=reverse, speed=speed, eq_gains=gains)
        )
        covered += max(durations[index] / speed - overlap, overlap)
        if covered >= seconds:
            return plan
    raise RuntimeError(f"plan covered {covered:.2f} s of the {seconds:.2f} s asked for")


def sources_excluding(paths: Sequence[str], held_out: Sequence[str]) -> list[str]:
    """Drop any source whose stem appears in `held_out`.

    The evaluation set is ten of the same 100 recordings. Room tone taken from a held-out
    file would put that file's acoustics into training, which is the one contamination this
    experiment cannot survive, so the exclusion is a function with a test rather than a
    flag someone remembers to pass.
    """
    blocked = {os.path.splitext(os.path.basename(name))[0] for name in held_out}
    return [p for p in paths if os.path.splitext(os.path.basename(p))[0] not in blocked]


def token_stats(tokens: Sequence[int]) -> dict[str, Any]:
    """distinct / top share / entropy / silence-token share for one codebook row."""
    if len(tokens) == 0:
        raise ValueError("no tokens to summarise")
    counts: dict[int, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    total = len(tokens)
    top_token, top_count = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
    return {
        "frames": total,
        "distinct": len(counts),
        "top_token": top_token,
        "top_share": top_count / total,
        "silence_token_share": counts.get(SILENCE_TOKEN, 0) / total,
        "entropy_bits": entropy,
    }


def gate_verdict(
    stats: dict[str, Any], *, min_distinct: int = 35, max_top_share: float = 0.35
) -> dict[str, Any]:
    """The 2-4 gate, applied to one measurement."""
    passed = stats["distinct"] >= min_distinct and stats["top_share"] <= max_top_share
    return {
        "passed": bool(passed),
        "min_distinct": min_distinct,
        "max_top_share": max_top_share,
        "distinct": stats["distinct"],
        "top_share": stats["top_share"],
    }


def level_in_db(rms: float, *, reference: float = SPEECH_REFERENCE_RMS) -> float:
    """Room tone level relative to a speech frame. Positive would mean it is louder."""
    if rms <= 0:
        raise ValueError(f"rms must be positive, got {rms}")
    return 20.0 * math.log10(rms / reference)


# --------------------------------------------------------------------------------------
# audio - every heavy import is local so the test suite runs without torch
# --------------------------------------------------------------------------------------


def _load_mono_24k(path: str, sample_rate: int):
    import soundfile as sf
    import torch
    import torchaudio

    samples, rate = sf.read(path, dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    tensor = torch.from_numpy(samples)
    if rate != sample_rate:
        tensor = torchaudio.functional.resample(tensor, rate, sample_rate)
    return tensor.numpy()


def frame_rms(samples, hop: int):
    import numpy as np

    count = len(samples) // hop
    if count == 0:
        return np.zeros(0)
    frames = np.asarray(samples[: count * hop], dtype=np.float64).reshape(count, hop)
    return np.sqrt((frames**2).mean(axis=1))


def max_voicing(samples, *, sample_rate: int = SAMPLE_RATE, window_seconds: float = 0.04) -> float:
    """Highest normalised autocorrelation peak in the pitch range over any window.

    Speech is periodic and room tone is not, so this is the test that keeps words off a
    channel whose transcript says nobody is talking. `max` rather than `median`: one voiced
    window in a segment is one voiced window too many.
    """
    import numpy as np

    window = int(window_seconds * sample_rate)
    count = len(samples) // window
    if count == 0:
        return 1.0
    best = 0.0
    low, high = sample_rate // 400, sample_rate // 70
    for i in range(count):
        chunk = np.asarray(samples[i * window : (i + 1) * window], dtype=np.float64)
        chunk = chunk - chunk.mean()
        spectrum = np.fft.rfft(chunk * np.hanning(len(chunk)), 2 * len(chunk))
        auto = np.fft.irfft(spectrum * np.conj(spectrum))[: len(chunk)]
        if auto[0] <= 0:
            continue
        best = max(best, float(np.max(auto[low:high] / auto[0])))
    return best


def collect_pool(paths: Sequence[str], *, spec: CollectSpec) -> dict[str, Any]:
    """Pull every qualifying segment out of `paths`. Returns segments plus their provenance."""
    import numpy as np

    segments = []
    index = []
    rejected_voiced = 0
    for path in sorted(paths):
        stem = os.path.splitext(os.path.basename(path))[0]
        audio = _load_mono_24k(path, spec.sample_rate)
        rms = frame_rms(audio, spec.hop_samples)
        runs = trim_runs(
            runs_in_band(rms.tolist(), low=spec.band_low, high=spec.band_high),
            guard_frames=spec.guard_frames,
            min_frames=spec.min_frames,
        )
        for start, end in runs:
            segment = audio[start * spec.hop_samples : end * spec.hop_samples]
            voicing = max_voicing(segment, sample_rate=spec.sample_rate)
            if voicing >= spec.max_voicing:
                rejected_voiced += 1
                continue
            segments.append(np.asarray(segment, dtype=np.float32))
            index.append(
                {
                    "source": stem,
                    "start_seconds": round(start * spec.hop_seconds, 3),
                    "duration_seconds": round((end - start) * spec.hop_seconds, 3),
                    "rms": float(np.sqrt((segment.astype(np.float64) ** 2).mean())),
                    "voicing": round(voicing, 4),
                }
            )
    return {
        "segments": segments,
        "index": index,
        "rejected_voiced": rejected_voiced,
        "sources_used": sorted({entry["source"] for entry in index}),
        "total_seconds": round(sum(e["duration_seconds"] for e in index), 3),
        "spec": {
            "band_low": spec.band_low,
            "band_high": spec.band_high,
            "hop_seconds": spec.hop_seconds,
            "min_seconds": spec.min_seconds,
            "guard_seconds": spec.guard_seconds,
            "max_voicing": spec.max_voicing,
            "sample_rate": spec.sample_rate,
        },
    }


def save_pool(pool: dict[str, Any], directory: str) -> None:
    import numpy as np

    os.makedirs(directory, exist_ok=True)
    np.savez_compressed(
        os.path.join(directory, "segments.npz"),
        **{f"s{i}": s for i, s in enumerate(pool["segments"])},
    )
    meta = {k: v for k, v in pool.items() if k != "segments"}
    with open(os.path.join(directory, "index.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=1)


def load_pool(directory: str) -> dict[str, Any]:
    import numpy as np

    with open(os.path.join(directory, "index.json"), encoding="utf-8") as handle:
        meta = json.load(handle)
    archive = np.load(os.path.join(directory, "segments.npz"))
    meta["segments"] = [archive[f"s{i}"] for i in range(len(meta["index"]))]
    return meta


def _apply_eq(unit, gains: Sequence[float], bands: Sequence[float], sample_rate: int):
    import numpy as np

    spectrum = np.fft.rfft(unit)
    freqs = np.fft.rfftfreq(len(unit), 1.0 / sample_rate)
    curve = np.interp(
        np.log2(np.maximum(freqs, 20.0)),
        np.log2(np.asarray(bands, dtype=np.float64)),
        np.asarray(gains, dtype=np.float64),
    )
    return np.fft.irfft(spectrum * 10 ** (curve / 20.0), len(unit))


def render_room_tone(
    pool: dict[str, Any], seconds: float, *, seed: int, spec: RenderSpec | None = None
):
    """Build `seconds` of room tone. Same pool + same seed gives the same samples."""
    import numpy as np

    spec = spec or RenderSpec()
    segments = pool["segments"]
    durations = [len(s) / spec.sample_rate for s in segments]
    plan = plan_units(durations, seconds, spec=spec, seed=seed)

    total = int(round(seconds * spec.sample_rate))
    fade = spec.crossfade_samples
    ramp = np.asarray(equal_power_ramp(fade), dtype=np.float64)
    out = np.zeros(total + 4 * spec.sample_rate, dtype=np.float64)
    position = 0
    for draw in plan:
        if position >= total:
            break
        unit = np.asarray(segments[draw.segment], dtype=np.float64)
        if draw.reverse:
            unit = unit[::-1].copy()
        if spec.speed_jitter:
            length = max(2 * fade, int(len(unit) / draw.speed))
            unit = np.interp(np.linspace(0, len(unit) - 1, length), np.arange(len(unit)), unit)
        if len(unit) < 2 * fade:
            continue
        if spec.eq_db and draw.eq_gains:
            unit = _apply_eq(unit, draw.eq_gains, spec.eq_bands_hz, spec.sample_rate)
        rms = math.sqrt(float((unit**2).mean()))
        if rms <= 0:
            continue
        unit = unit * (draw.level / rms)
        if position == 0:
            out[: len(unit)] += unit
            position = len(unit)
        else:
            unit[:fade] *= ramp
            out[position - fade : position] *= ramp[::-1]
            out[position - fade : position - fade + len(unit)] += unit
            position += len(unit) - fade
    if position < total:
        raise RuntimeError(f"plan covered {position / spec.sample_rate:.2f} s of {seconds:.2f} s")
    return out[:total].astype(np.float32)


def fill_silence(
    channel,
    pool: dict[str, Any],
    *,
    seed: int,
    spec: RenderSpec | None = None,
    frame_samples: int = 1920,
    edge_seconds: float = 0.02,
):
    """Replace every digitally silent stretch of `channel` with room tone.

    The fill is ramped in and out at the gap edges: a hard cut from speech to a bed that is
    40 dB down is still a step, and a step is a click.
    """
    import numpy as np

    spec = spec or RenderSpec()
    out = np.asarray(channel, dtype=np.float64).copy()
    count = len(out) // frame_samples
    frames = out[: count * frame_samples].reshape(count, frame_samples)
    silent = np.abs(frames).max(axis=1) == 0.0
    edge = int(edge_seconds * spec.sample_rate)
    for start, end in runs_in_band(silent.astype(float).tolist(), low=0.5, high=2.0):
        first, last = start * frame_samples, end * frame_samples
        span = last - first
        if span < 2 * edge:
            continue
        bed = np.asarray(
            render_room_tone(pool, span / spec.sample_rate, seed=seed, spec=spec), dtype=np.float64
        )
        bed = bed[:span]
        ramp = np.asarray(equal_power_ramp(edge), dtype=np.float64)
        bed[:edge] *= ramp
        bed[-edge:] *= ramp[::-1]
        out[first:last] = bed
        seed += 1
    return out.astype(np.float32)


_MIMI_CACHE: dict[tuple[str, str, str], Any] = {}


def codebook0(
    wav,
    *,
    device: str = "mps",
    repo: str = "kyutai/moshiko-pytorch-bf16",
    name: str = "tokenizer-e351c8d8-checkpoint125.safetensors",
):
    """Mimi's first audio codebook for one mono waveform."""
    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download
    from moshi.models import loaders

    key = (repo, name, device)
    if key not in _MIMI_CACHE:
        _MIMI_CACHE[key] = loaders.get_mimi(filename=hf_hub_download(repo, name), device=device)
    mimi = _MIMI_CACHE[key]
    tensor = torch.from_numpy(np.asarray(wav, dtype=np.float32)).reshape(1, 1, -1).to(device)
    with torch.no_grad():
        return mimi.encode(tensor).cpu()[0].numpy()[0]


def as_pcm16(wav):
    """Round-trip through the format the dataset is written in.

    Room tone at -98 dBFS does not survive this; the level has to be chosen on the far side
    of the quantiser, not before it.
    """
    import numpy as np

    clipped = np.clip(np.asarray(wav, dtype=np.float64), -1.0, 1.0)
    return (np.round(clipped * 32767).astype(np.int16).astype(np.float64) / 32767.0).astype(
        np.float32
    )


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _cmd_collect(args) -> None:
    import glob

    paths = sorted(glob.glob(os.path.join(args.source_dir, "*.wav")))
    held_out = (
        sorted(glob.glob(os.path.join(args.held_out_dir, "*.wav"))) if args.held_out_dir else []
    )
    kept = sources_excluding(paths, held_out)
    spec = CollectSpec(
        band_low=args.band_low,
        band_high=args.band_high,
        min_seconds=args.min_seconds,
        max_voicing=args.max_voicing,
    )
    pool = collect_pool(kept, spec=spec)
    pool["excluded_held_out"] = sorted(os.path.splitext(os.path.basename(p))[0] for p in held_out)
    pool["source_dir"] = args.source_dir
    save_pool(pool, args.output_dir)
    print(
        f"{len(pool['segments'])} segments, {pool['total_seconds']} s, "
        f"{len(pool['sources_used'])} sources, {pool['rejected_voiced']} rejected as voiced -> {args.output_dir}"
    )


def _cmd_render(args) -> None:
    import soundfile as sf

    pool = load_pool(args.pool_dir)
    wav = render_room_tone(pool, args.seconds, seed=args.seed)
    sf.write(args.output, wav, SAMPLE_RATE, subtype="PCM_16")
    print(f"{args.seconds} s, seed {args.seed} -> {args.output}")


def _cmd_verify(args) -> None:
    import numpy as np

    pool = load_pool(args.pool_dir)
    report: dict[str, Any] = {"pool_dir": args.pool_dir, "isolated": [], "in_situ": []}
    for seconds in args.seconds:
        for seed in args.seeds:
            wav = as_pcm16(render_room_tone(pool, seconds, seed=seed))
            stats = token_stats(codebook0(wav, device=args.device).tolist())
            stats.update(
                {
                    "seconds": seconds,
                    "seed": seed,
                    "level_db": round(
                        level_in_db(float(np.sqrt((wav.astype(np.float64) ** 2).mean()))), 1
                    ),
                }
            )
            stats["gate"] = gate_verdict(stats)
            report["isolated"].append(stats)
            print("isolated", stats)
    if args.channels_dir:
        import glob

        import soundfile as sf

        files = sorted(glob.glob(os.path.join(args.channels_dir, "*.wav")))[: args.channel_count]
        for seed in args.seeds:
            tokens: list[int] = []
            for path in files:
                data, _ = sf.read(path, dtype="float32")
                channel = data[:, 0] if data.ndim > 1 else data
                filled = as_pcm16(fill_silence(channel, pool, seed=seed * 1000))
                count = len(channel) // 1920
                frames = np.asarray(channel[: count * 1920]).reshape(count, 1920)
                silent = np.abs(frames).max(axis=1) == 0.0
                ids = codebook0(filled, device=args.device)
                usable = min(len(ids), count)
                tokens.extend(ids[:usable][silent[:usable]].tolist())
            stats = token_stats(tokens)
            stats.update({"seed": seed, "channels": len(files)})
            stats["gate"] = gate_verdict(stats)
            report["in_situ"].append(stats)
            print("in-situ", stats)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=1)
        print(f"wrote {args.report}")


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect", help="Build a room-tone pool from a directory of wavs.")
    collect.add_argument("--source-dir", required=True)
    collect.add_argument("--held-out-dir", default=None, help="Sources to exclude, by stem.")
    collect.add_argument("--output-dir", required=True)
    collect.add_argument("--band-low", type=float, default=CollectSpec.band_low)
    collect.add_argument("--band-high", type=float, default=CollectSpec.band_high)
    collect.add_argument("--min-seconds", type=float, default=CollectSpec.min_seconds)
    collect.add_argument("--max-voicing", type=float, default=CollectSpec.max_voicing)
    collect.set_defaults(func=_cmd_collect)

    render = sub.add_parser("render", help="Write one WAV of room tone.")
    render.add_argument("--pool-dir", required=True)
    render.add_argument("--seconds", type=float, required=True)
    render.add_argument("--seed", type=int, required=True)
    render.add_argument("--output", required=True)
    render.set_defaults(func=_cmd_render)

    verify = sub.add_parser("verify", help="Tokenise with Mimi and apply the gate.")
    verify.add_argument("--pool-dir", required=True)
    verify.add_argument("--seconds", type=float, nargs="+", default=[30.0, 60.0])
    verify.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    verify.add_argument(
        "--channels-dir", default=None, help="Stereo dialogues; channel 0 is speaker A."
    )
    verify.add_argument("--channel-count", type=int, default=12)
    verify.add_argument("--device", default="mps")
    verify.add_argument("--report", default=None)
    verify.set_defaults(func=_cmd_verify)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
