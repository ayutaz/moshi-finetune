from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

STREAMS_PER_SPEAKER = 9  # 1 text + 8 audio codebooks
MIMI_FRAME_RATE_HZ = 12.5


def minimum_sample_count(*, min_frames: int, sample_rate: int) -> int:
    """Samples needed for Mimi to emit at least `min_frames` frames."""
    if min_frames <= 0:
        raise ValueError("min_frames must be positive")
    return math.ceil(min_frames / MIMI_FRAME_RATE_HZ * sample_rate)


class PromptDatasetError(ValueError):
    """Raised when the fixed baseline prompt dataset would not be evaluated as recorded."""


def build_stereo_prompt(
    mono_waveform: Any, *, channel: str = "A", lead_in_samples: int = 0
) -> tuple[Any, Any]:
    """Put the mono audio on one channel and silence on the other.

    `channel="B"` is what the fixed conversations need: the USER speaks and the model
    answers. With the audio on A, "does it ignore the user" and every turn-taking figure
    would be measured against a silent user - the degenerate regime the past
    depformer-only run collapsed into, where nothing distinguishes health from failure.

    `lead_in_samples` delays the audio so it starts inside the window generate.py actually
    returns. Speech that finishes during the prompt is invisible to the measurement.
    """
    if channel not in ("A", "B"):
        raise ValueError(f"channel must be A or B, got {channel!r}")
    if lead_in_samples < 0:
        raise ValueError(f"lead_in_samples must not be negative, got {lead_in_samples}")

    shape = getattr(mono_waveform, "shape", None)
    channel_count = int(shape[0]) if shape is not None else len(mono_waveform)
    if channel_count != 1:
        raise ValueError(f"baseline reference must be mono, got {channel_count} channels")

    speech = mono_waveform[0]
    if lead_in_samples:
        if hasattr(speech, "new_zeros"):
            import torch

            speech = torch.cat((speech.new_zeros(lead_in_samples), speech))
        else:
            speech = [0.0] * lead_in_samples + list(speech)

    if hasattr(speech, "new_zeros"):
        silence = speech.new_zeros(speech.shape)
    else:
        silence = [0.0 for _ in speech]
    return (speech, silence) if channel == "A" else (silence, speech)


def user_voiced_frames_in_window(
    user_mask: Any, *, prompt_frames: int, generation_frames: int
) -> int:
    """How many frames the user speaks inside the span generate.py returns.

    generate.py emits only the generated frames, so user speech during the prompt is
    invisible to every downstream measurement. A prompt scoring zero here would have the
    model judged on whether it ignores a user who never speaks.
    """
    window = list(user_mask)[prompt_frames : prompt_frames + generation_frames]
    return sum(1 for active in window if active)


def select_audio_token_stems(names: Iterable[str]) -> list[str]:
    return sorted(Path(name).stem for name in names if Path(name).suffix == ".npz")


def prepare_stereo_audio(
    input_dir: Path,
    output_dir: Path,
    *,
    target_rate: int,
    min_frames: int | None = None,
    channel: str = "A",
    lead_in_frames: int = 0,
    frame_rate_hz: float = 12.5,
) -> dict[str, Any]:
    import torch
    import torchaudio

    paths = sorted(input_dir.glob("*.wav"))
    if not paths:
        raise ValueError(f"no WAV prompts found in {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    minimum_samples = (
        minimum_sample_count(min_frames=min_frames, sample_rate=target_rate) if min_frames else 0
    )
    durations = {}
    speech_durations = {}
    for input_path in paths:
        waveform, sample_rate = torchaudio.load(str(input_path))
        if sample_rate != target_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, target_rate)
        speech_durations[input_path.stem] = round(waveform.shape[-1] / target_rate, 6)
        if waveform.shape[-1] < minimum_samples:
            # Trailing silence, so the user stream can be teacher-forced past the prompt.
            padding = waveform.new_zeros((waveform.shape[0], minimum_samples - waveform.shape[-1]))
            waveform = torch.cat((waveform, padding), dim=-1)
        lead_in_samples = int(round(lead_in_frames / frame_rate_hz * target_rate))
        speaker_a, speaker_b = build_stereo_prompt(
            waveform, channel=channel, lead_in_samples=lead_in_samples
        )
        stereo = torch.stack((speaker_a, speaker_b))
        output_path = output_dir / input_path.name
        torchaudio.save(str(output_path), stereo, target_rate)
        durations[input_path.stem] = round(stereo.shape[-1] / target_rate, 6)
    return {
        "status": "pass",
        "prompt_count": len(paths),
        "sample_rate_hz": target_rate,
        "channel": channel,
        "lead_in_frames": lead_in_frames,
        "speaker_a": ("Tsukuyomi held-out reference then silence" if channel == "A" else "silence"),
        "speaker_b": ("silence" if channel == "A" else "user prompt after the lead-in"),
        "min_frames_requested": min_frames,
        "minimum_samples": minimum_samples,
        "speech_durations_seconds": speech_durations,
        "durations_seconds": durations,
    }


def create_padding_text_tokens(audio_token_dir: Path, output_dir: Path) -> dict[str, Any]:
    import numpy as np

    stems = select_audio_token_stems(path.name for path in audio_token_dir.iterdir())
    if not stems:
        raise ValueError(f"no audio token archives found in {audio_token_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        with np.load(audio_token_dir / f"{stem}.npz") as audio_tokens:
            for speaker in ("A", "B"):
                if speaker not in audio_tokens or audio_tokens[speaker].ndim != 2:
                    raise ValueError(f"{stem}: invalid {speaker} audio token stream")
        empty = np.array([], dtype=np.int64)
        np.savez_compressed(output_dir / f"{stem}.npz", A=empty, B=empty)
    return {"status": "pass", "prompt_count": len(stems)}


def verify_prompt_dataset(
    rows: list[dict[str, Any]],
    *,
    expected_count: int,
    min_frames: int,
) -> dict[str, Any]:
    """Fail before generation when the prompt dataset would not be evaluated as recorded.

    `utils.data.filter_out_short_streams` silently discards examples shorter than
    `generate.py --prompt_length`, so a too-short prompt would shrink the baseline without
    any log line. `example_id` is also assigned by row order after `dialogue_id` has been
    dropped, so the mapping back to each prompt is recorded here.
    """
    if len(rows) != expected_count:
        raise PromptDatasetError(f"expected {expected_count} prompt rows, got {len(rows)}")

    examples = []
    too_short = []
    for example_id, row in enumerate(rows):
        dialogue_id = row["dialogue_id"]
        speaker_frames = {}
        for speaker in ("A", "B"):
            streams = row[speaker]
            if len(streams) != STREAMS_PER_SPEAKER:
                raise PromptDatasetError(
                    f"{dialogue_id}: speaker {speaker} must have "
                    f"{STREAMS_PER_SPEAKER} streams, got {len(streams)}"
                )
            frame_counts = {len(stream) for stream in streams}
            if len(frame_counts) != 1:
                raise PromptDatasetError(
                    f"{dialogue_id}: speaker {speaker} has ragged streams {sorted(frame_counts)}"
                )
            speaker_frames[speaker] = frame_counts.pop()
        if speaker_frames["A"] != speaker_frames["B"]:
            raise PromptDatasetError(
                f"{dialogue_id}: A and B frame counts differ "
                f"({speaker_frames['A']} != {speaker_frames['B']})"
            )
        frames = speaker_frames["A"]
        if frames < min_frames:
            too_short.append((dialogue_id, frames))
        examples.append({"example_id": example_id, "dialogue_id": dialogue_id, "frames": frames})

    if too_short:
        detail = ", ".join(f"{dialogue_id}={frames}" for dialogue_id, frames in too_short)
        raise PromptDatasetError(
            f"prompts below min_frames={min_frames} would be dropped silently: {detail}"
        )

    return {
        "status": "pass",
        "prompt_count": len(rows),
        "min_frames_required": min_frames,
        "min_frames_observed": min(example["frames"] for example in examples),
        "examples": examples,
    }


def _read_prompt_dataset(parquet_glob: str) -> list[dict[str, Any]]:
    from glob import glob

    import pandas as pd

    paths = sorted(glob(parquet_glob))
    if not paths:
        raise PromptDatasetError(f"no parquet files matched {parquet_glob}")
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    return frame.to_dict(orient="records")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare fixed voice-only baseline prompts")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audio_parser = subparsers.add_parser("audio")
    audio_parser.add_argument("--input-dir", type=Path, required=True)
    audio_parser.add_argument("--output-dir", type=Path, required=True)
    audio_parser.add_argument("--target-rate", type=int, default=24_000)
    audio_parser.add_argument("--min-frames", type=int, default=None)
    audio_parser.add_argument("--report", type=Path, required=True)
    audio_parser.add_argument(
        "--channel",
        choices=("A", "B"),
        default="A",
        help="which stream carries the audio; B puts the USER on the prompt",
    )
    audio_parser.add_argument(
        "--lead-in-frames",
        type=int,
        default=0,
        help="silence before the audio, so it starts inside the generated window",
    )

    text_parser = subparsers.add_parser("padding-text")
    text_parser.add_argument("--audio-token-dir", type=Path, required=True)
    text_parser.add_argument("--output-dir", type=Path, required=True)
    text_parser.add_argument("--report", type=Path, required=True)

    verify_parser = subparsers.add_parser("verify-dataset")
    verify_parser.add_argument("--parquet-glob", required=True)
    verify_parser.add_argument("--expected-count", type=int, required=True)
    verify_parser.add_argument("--min-frames", type=int, required=True)
    verify_parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "audio":
        report = prepare_stereo_audio(
            args.input_dir,
            args.output_dir,
            target_rate=args.target_rate,
            min_frames=args.min_frames,
            channel=args.channel,
            lead_in_frames=args.lead_in_frames,
        )
    elif args.command == "padding-text":
        report = create_padding_text_tokens(args.audio_token_dir, args.output_dir)
    else:
        report = verify_prompt_dataset(
            _read_prompt_dataset(args.parquet_glob),
            expected_count=args.expected_count,
            min_frames=args.min_frames,
        )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
