from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def build_stereo_prompt(mono_waveform: Any) -> tuple[Any, Any]:
    """Return speaker A's mono channel and a same-length silent B channel."""
    shape = getattr(mono_waveform, "shape", None)
    channel_count = int(shape[0]) if shape is not None else len(mono_waveform)
    if channel_count != 1:
        raise ValueError(f"baseline reference must be mono, got {channel_count} channels")
    speaker_a = mono_waveform[0]
    if hasattr(speaker_a, "new_zeros"):
        speaker_b = speaker_a.new_zeros(speaker_a.shape)
    else:
        speaker_b = [0.0 for _ in speaker_a]
    return speaker_a, speaker_b


def select_audio_token_stems(names: Iterable[str]) -> list[str]:
    return sorted(Path(name).stem for name in names if Path(name).suffix == ".npz")


def prepare_stereo_audio(input_dir: Path, output_dir: Path, *, target_rate: int) -> dict[str, Any]:
    import torch
    import torchaudio

    paths = sorted(input_dir.glob("*.wav"))
    if not paths:
        raise ValueError(f"no WAV prompts found in {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    durations = {}
    for input_path in paths:
        waveform, sample_rate = torchaudio.load(str(input_path))
        if sample_rate != target_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, target_rate)
        speaker_a, speaker_b = build_stereo_prompt(waveform)
        stereo = torch.stack((speaker_a, speaker_b))
        output_path = output_dir / input_path.name
        torchaudio.save(str(output_path), stereo, target_rate)
        durations[input_path.stem] = round(stereo.shape[-1] / target_rate, 6)
    return {
        "status": "pass",
        "prompt_count": len(paths),
        "sample_rate_hz": target_rate,
        "speaker_a": "Tsukuyomi held-out reference",
        "speaker_b": "silence",
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare fixed voice-only baseline prompts")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audio_parser = subparsers.add_parser("audio")
    audio_parser.add_argument("--input-dir", type=Path, required=True)
    audio_parser.add_argument("--output-dir", type=Path, required=True)
    audio_parser.add_argument("--target-rate", type=int, default=24_000)
    audio_parser.add_argument("--report", type=Path, required=True)

    text_parser = subparsers.add_parser("padding-text")
    text_parser.add_argument("--audio-token-dir", type=Path, required=True)
    text_parser.add_argument("--output-dir", type=Path, required=True)
    text_parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "audio":
        report = prepare_stereo_audio(args.input_dir, args.output_dir, target_rate=args.target_rate)
    else:
        report = create_padding_text_tokens(args.audio_token_dir, args.output_dir)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
