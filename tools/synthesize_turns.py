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

M3-R adds two options, both off by default so the M3 render stays reproducible.

- `--roles` renders only the turns carrying one of the named `role` values. M3-R re-uses
  M3's open and close turns byte for byte and needs the twelve backchannel texts alone.
- `--seed-per-turn` derives each turn's seed from the base seed and the turn's identity.
  One seed for the whole run is right when every text is different; M3-R's backchannel pool
  is twelve texts spread over 78 dialogues, so a single seed would put the *same waveform*
  into six or seven dialogues - which is the repeating pattern this rebuild exists to
  remove. The derivation is a hash of the id, not `hash()`, because Python salts that per
  process and the seed has to survive a resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
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


def recorded_filenames(sidecar: Path) -> set[str]:
    """Turn filenames the sidecar already has a row for."""
    if not sidecar.is_file():
        return set()
    names = set()
    for line in sidecar.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            names.add(json.loads(line)["filename"])
        except (json.JSONDecodeError, KeyError):
            # A row torn in half by a kill is not a record of anything.
            continue
    return names


def turn_seed(base: int, dialogue_id: str, index: int) -> int:
    """A per-turn seed that depends only on the base seed and the turn's identity.

    Deterministic across processes and across a resume, which `hash()` is not: PYTHONHASHSEED
    is random by default, so a run killed halfway would re-render the remaining turns from a
    different voice draw than the ones already on disk.
    """
    digest = hashlib.blake2b(f"{base}:{dialogue_id}:{index}".encode(), digest_size=4).digest()
    return int.from_bytes(digest, "big")


@dataclass(frozen=True)
class DurationModel:
    """How long a turn should be, when the model's own duration predictor cannot say.

    Irodori predicts the length of a turn from its text and then generates exactly that
    much audio. On the 44-character turns M3 rendered it is right - 0.151 s per mora, which
    is 6.8 mora per second. On a three-character aizuchi it is not: it asked for 5.12 s of
    「はい、はい。」 and the model filled the excess with invented speech. Whisper reads that
    file back as ハイゾシティイリアシマスHUR大吐き…; nothing in a duration log or a loss curve
    would have shown it, and it would have gone into 78 dialogues.

    So a short turn gets a length instead of a prediction, and the length has to be close to
    how long the words actually take. Asking for *more* than that fails the same way the
    predictor did. Measured on 「ええ。」 and 「はい。」, two mora each, two seeds apiece and
    Whisper reading the result back:

    | asked for | 0.55 s | 0.70 s | 0.90 s | 1.1 s | 1.4 s | 1.7 s |
    | correct   |  4/4   |  3/4   |  1/4   |  0/6  |  0/6  |  0/6  |

    A first pass used a 1.1 s floor and all seven 「ええ。」 backchannels came back as
    invented speech - 説自は, インチステイ, ギズティー, Aスイーツ道場 - while four of the seven
    「はい。」 had junk after the はい. Four mora and up were unaffected, which is why the
    defect was invisible until every clip was transcribed. The floor is now the runtime's own
    `min_seconds`, and the slope is the 6.8 mora per second the 160 shipped speaker B turns
    were rendered at.

    A clip can still come out wrong at a plausible length - 「はい。」 at 0.70 s did, on one
    seed of two - so this sets the odds, not the outcome. What settles it is transcribing
    every rendered clip and re-drawing the seed for the ones that do not read back.
    """

    floor: float = 0.5
    base: float = 0.30
    per_mora: float = 0.135
    per_comma: float = 0.15

    def seconds(self, *, mora: int, commas: int) -> float:
        return max(self.floor, self.base + self.per_mora * mora + self.per_comma * commas)


def text_shape(text: str) -> tuple[int, int]:
    """Mora count and reading-comma count, the two things length depends on."""
    import pyopenjtalk

    mora = sum(int(entry.get("mora_size") or 0) for entry in pyopenjtalk.run_frontend(text))
    return mora, text.count("、")


def turns_to_render(
    dialogues: list[dict[str, Any]],
    *,
    speaker: str,
    out_dir: Path,
    recorded: set[str],
    roles: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Every turn for `speaker` still to render, in dialogue order.

    A turn counts as done only when the wav exists AND the sidecar has a row for it. The
    wav is written first, so a kill in between leaves audio nobody can account for - and
    the sidecar is what the manifest is built from, so an unrecorded wav would simply
    vanish from the dataset while sitting on disk. Re-rendering it is cheap; a dialogue
    silently missing a turn is not.

    `roles` narrows the selection to turns whose script row carries one of those `role`
    values. A turn with no `role` never matches, so asking for a role on a script that has
    none renders nothing rather than everything.
    """
    wanted = None if roles is None else set(roles)
    pending = []
    for dialogue in dialogues:
        for index, turn in enumerate(dialogue["turns"]):
            if turn["speaker"] != speaker:
                continue
            if wanted is not None and turn.get("role") not in wanted:
                continue
            name = turn_filename(dialogue["dialogue_id"], index, speaker)
            if (out_dir / name).exists() and name in recorded:
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
    parser.add_argument(
        "--roles",
        nargs="+",
        default=None,
        help="render only turns whose script role is one of these (default: every turn)",
    )
    parser.add_argument(
        "--seed-per-turn",
        action="store_true",
        help="derive each turn's seed from --seed and the turn id, so a text that repeats "
        "across dialogues does not become the same waveform in every one of them",
    )
    parser.add_argument(
        "--manual-duration",
        action="store_true",
        help="set each turn's length from its mora count instead of trusting the model's "
        "duration predictor, which over-predicts short texts and fills the excess with "
        "invented speech",
    )
    parser.add_argument("--duration-floor", type=float, default=DurationModel.floor)
    parser.add_argument("--duration-base", type=float, default=DurationModel.base)
    parser.add_argument("--duration-per-mora", type=float, default=DurationModel.per_mora)
    parser.add_argument("--duration-per-comma", type=float, default=DurationModel.per_comma)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--limit", type=int, default=None, help="render at most N turns")
    args = parser.parse_args()

    checkpoint = args.checkpoint or (VOICEDESIGN if args.speaker == "B" else BASE_500M)
    duration_model = DurationModel(
        floor=args.duration_floor,
        base=args.duration_base,
        per_mora=args.duration_per_mora,
        per_comma=args.duration_per_comma,
    )
    dialogues = [
        json.loads(line)
        for line in args.scripts.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.sidecar.parent.mkdir(parents=True, exist_ok=True)

    recorded = recorded_filenames(args.sidecar)
    pending = turns_to_render(
        dialogues,
        speaker=args.speaker,
        out_dir=args.out_dir,
        recorded=recorded,
        roles=args.roles,
    )
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
            seed = (
                turn_seed(args.seed, turn["dialogue_id"], turn["turn_index"])
                if args.seed_per_turn
                else args.seed
            )
            seconds = None
            if args.manual_duration:
                mora, commas = text_shape(turn["text"])
                seconds = duration_model.seconds(mora=mora, commas=commas)
            request = SamplingRequest(
                seconds=seconds,
                text=turn["text"],
                caption=args.caption,
                ref_wav=args.ref_wav,
                ref_embed=args.ref_embed,
                no_ref=not (args.ref_wav or args.ref_embed),
                seed=seed,
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
                        "seed": seed,
                        "requested_seconds": seconds,
                        "base_seed": args.seed,
                        "seed_per_turn": bool(args.seed_per_turn),
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
