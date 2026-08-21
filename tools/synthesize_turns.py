"""Render every dialogue turn with one long-lived Irodori-TTS runtime.

`infer.py` builds the runtime per process: on this Mac that is about 18 seconds of the 40
it takes to produce one clip, so the 320 turns M3 needs would spend well over an hour
loading the same weights again and again. Loading once and looping cuts that.

Two speakers, two checkpoints, one rule each:

- **B** is the frozen reference. Every B turn passes the same `--ref-wav`, because a voice
  sampled per utterance from a caption is not one speaker - measured at 0.45-0.52 pairwise
  ECAPA against a one-real-human floor of 0.565 (`reports/m3-speaker-b-probe.json`).
- **A** in V-tts is the M2 speaker-inversion embedding. In V-real, A is not synthesised at
  all: the audio is the corpus recording.

Resumable, because a three-hour render should not restart from zero. A turn whose wav
already exists is skipped, and every turn appends a sidecar row recording exactly what
produced it - so the manifest can be built from what was rendered rather than from what
was intended.

Watermarking must be off before this runs. Put `m3/nowatermark/` first on PYTHONPATH; the
sidecar records whether the runtime agreed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

VOICEDESIGN = "Aratako/Irodori-TTS-600M-v3-VoiceDesign"
BASE_500M = "Aratako/Irodori-TTS-500M-v3"


def turn_filename(dialogue_id: str, index: int, speaker: str) -> str:
    """Stable name for one rendered turn.

    The turn index is in the name because a dialogue has two B turns, and the speaker is
    in it because a later pass reads these back without the script in hand.
    """
    return f"{dialogue_id}-t{index}-{speaker}.wav"


def turns_to_render(
    dialogues: list[dict[str, Any]], *, speaker: str, out_dir: Path
) -> list[dict[str, Any]]:
    """Every turn for `speaker` that has no wav yet, in dialogue order."""
    pending = []
    for dialogue in dialogues:
        for index, turn in enumerate(dialogue["turns"]):
            if turn["speaker"] != speaker:
                continue
            name = turn_filename(dialogue["dialogue_id"], index, speaker)
            if (out_dir / name).exists():
                continue
            pending.append(
                {
                    "dialogue_id": dialogue["dialogue_id"],
                    "turn_index": index,
                    "speaker": speaker,
                    "text": turn["text"],
                    "filename": name,
                }
            )
    return pending


def resolve_checkpoint(checkpoint: str) -> str:
    """A local safetensors path, downloading from the hub if `checkpoint` is a repo id.

    RuntimeKey takes a path and torch.load fails with a bare FileNotFoundError on a repo
    id, which reads as a missing file rather than as the wrong kind of argument. infer.py
    resolves this before building the runtime; doing it here keeps both entry points
    accepting the same values.
    """
    if Path(checkpoint).expanduser().is_file():
        return str(Path(checkpoint).expanduser())

    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=checkpoint, filename="model.safetensors")


def _build_runtime(checkpoint: str, device: str):
    from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey

    key = RuntimeKey(
        checkpoint=resolve_checkpoint(checkpoint), model_device=device, codec_device=device
    )
    return InferenceRuntime.from_key(key)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render dialogue turns with one runtime")
    parser.add_argument("--scripts", type=Path, required=True, help="dialogues JSONL")
    parser.add_argument("--speaker", choices=("A", "B"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True, help="JSONL render log")
    parser.add_argument("--checkpoint", default=None, help="defaults by speaker")
    parser.add_argument("--ref-wav", default=None, help="frozen reference for speaker B")
    parser.add_argument("--ref-embed", default=None, help="speaker-inversion embedding for A")
    parser.add_argument("--caption", default=None)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--limit", type=int, default=None, help="render at most N turns")
    args = parser.parse_args()

    checkpoint = args.checkpoint or (VOICEDESIGN if args.speaker == "B" else BASE_500M)
    dialogues = [
        json.loads(line)
        for line in args.scripts.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.sidecar.parent.mkdir(parents=True, exist_ok=True)

    pending = turns_to_render(dialogues, speaker=args.speaker, out_dir=args.out_dir)
    if args.limit is not None:
        pending = pending[: args.limit]
    print(f"{len(pending)} turn(s) to render for speaker {args.speaker}", flush=True)
    if not pending:
        return 0

    from irodori_tts.inference_runtime import SamplingRequest, save_wav

    runtime = _build_runtime(checkpoint, args.device)
    watermarked = bool(runtime.watermarker.ready)
    if watermarked:
        # Refuse rather than warn: a watermark in the training audio is invisible in every
        # downstream number, so nothing later would catch it.
        raise SystemExit(
            "the SilentCipher watermarker loaded; put m3/nowatermark/ first on PYTHONPATH"
        )

    started = time.perf_counter()
    with args.sidecar.open("a", encoding="utf-8") as log:
        for position, turn in enumerate(pending, start=1):
            request = SamplingRequest(
                text=turn["text"],
                caption=args.caption,
                ref_wav=args.ref_wav,
                ref_embed=args.ref_embed,
                no_ref=not (args.ref_wav or args.ref_embed),
                seed=args.seed,
            )
            result = runtime.synthesize(request)
            # save_wav, not soundfile.write: result.audio is a torch tensor and the
            # project's own writer handles the layout and dtype it comes in.
            path = save_wav(args.out_dir / turn["filename"], result.audio, result.sample_rate)
            log.write(
                json.dumps(
                    {
                        **turn,
                        "path": str(path),
                        "sample_rate": result.sample_rate,
                        "seconds": result.audio.numel() / result.sample_rate,
                        "checkpoint": checkpoint,
                        "ref_wav": args.ref_wav,
                        "ref_embed": args.ref_embed,
                        "caption": args.caption,
                        "seed": args.seed,
                        "watermarked": watermarked,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            log.flush()
            elapsed = time.perf_counter() - started
            rate = elapsed / position
            print(
                f"[{position}/{len(pending)}] {turn['filename']} "
                f"{rate:.1f}s/turn, {(len(pending) - position) * rate / 60:.0f} min left",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
