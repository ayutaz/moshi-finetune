"""Materialise a dialogue dataset's split directories and write its manifest.

Why this is in `tools/` and not in a script
-------------------------------------------
`v-real-v2` was built by six scripts that lived under
`data/experiments/tsukuyomi_ojousama/m3r/v-real/build-scripts/`. `data/` is gitignored, so
the procedure for rebuilding the dataset was not in the repository: the manifest could be
read but not remade, and the logic that assembled each row had no test. That is the same
defect as M3's dropped `--no_whitespace_before_word` - a step nobody wrote down - one layer
up, and it is the reason the two halves below are here.

The two halves
--------------
**Materialising the split directories.** The builder writes `sequences/<split>/` rows. With
`group_size=1` each row is exactly one dialogue, and it is copied to `<split>/{audio,text}`
under *the dialogue's own name*, because the basename becomes the `dialogue_id`
(`train/v-001`, not `train/train-seq-001`). That namespace is what joins this dataset's
parquet to the manifest and to M3's rows, so the naming is load-bearing rather than
cosmetic.

Nothing is trusted on the way: every copy is checked against both the sequence file the
builder wrote and the dialogue file it claims to equal, and against the sha256 the build
report recorded. Three sources have to agree before a byte is written.

**The manifest row.** `tools/experiment_data.build_manifest` inventories a directory of raw
recordings and assigns splits from a seed. A dialogue dataset is the other shape: the rows
already exist, the split is already fixed in a split map, and each row has to carry the
things that let a trained checkpoint be traced back - which sequence row it became, which
backchannel wav was mixed into it, which room-tone pool, and the checksums of the tokenized
artifacts. `manifest_row` builds that, and it is pure: every checksum arrives as an argument
that some caller measured off the disk. A row assembled from a remembered number is a row
that can agree with nothing.

Splitting the two means the assembly can be tested without a dataset, which is the half that
was untestable while it lived under `data/`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# The dataset-level fields a manifest row repeats. `tools/experiment_data.REQUIRED_FIELDS`
# is the validator's list; this is the subset a caller has to supply, and the difference is
# exactly the fields derived per row.
DATASET_METADATA_FIELDS = (
    "retrieved_at",
    "credit",
    "license_id",
    "license_url",
    "redistribution",
    "source_url",
    "source_version",
    "generation_method",
)

MEDIA_TYPE = "audio/x-wav"


class SequencePlanError(ValueError):
    """The build report does not describe rows that can be copied one-to-one."""


class ManifestRowError(ValueError):
    """A row cannot be assembled from the inputs given."""


@dataclass(frozen=True)
class SequenceRow:
    """One row of the built dataset, and the dialogue it is a copy of."""

    split: str
    name: str
    dialogue: str
    sha256: str
    group_size: int


@dataclass(frozen=True)
class DialogueArtifacts:
    """The facts about one dialogue that had to be read off the disk.

    Every field here is measured. Keeping them in one frozen object is what lets
    `manifest_row` stay pure while still refusing to write a checksum nobody took.
    """

    path: str
    sha256: str
    byte_size: int
    audio: Mapping[str, Any]
    split_audio_path: str
    sequence_audio_path: str
    word_transcript_path: str
    word_transcript_sha256: str
    tok_audio_sha256: str
    tok_text_sha256: str
    parquet_path: str
    parquet_sha256: str


def sequence_plan(build_report: Mapping[str, Any]) -> list[SequenceRow]:
    """Read the build report's `sequences` block into a copy plan, or refuse.

    Every problem is collected before raising. A plan that stops at the first bad entry
    tells you about one dialogue when the dataset may be wrong in eighty places.
    """
    sequences = build_report.get("sequences")
    if not isinstance(sequences, Mapping):
        raise SequencePlanError("the build report has no `sequences` block")
    group_size = sequences.get("group_size")
    if group_size != 1:
        raise SequencePlanError(
            f"group_size is {group_size!r}; this copies one dialogue per row and a row "
            "holding several has no single dialogue name to take"
        )
    splits = sequences.get("splits")
    if not isinstance(splits, Mapping) or not splits:
        raise SequencePlanError("the build report lists no splits")

    problems: list[str] = []
    rows: list[SequenceRow] = []
    for split, block in splits.items():
        entries = (block or {}).get("entries")
        if not isinstance(entries, Sequence) or not entries:
            problems.append(f"{split}: no entries")
            continue
        for entry in entries:
            name = entry.get("name", "?")
            dialogues = entry.get("dialogues") or []
            if len(dialogues) != 1:
                problems.append(f"{split}/{name}: {len(dialogues)} dialogues in one row")
                continue
            if not entry.get("identical_to_dialogue"):
                problems.append(f"{split}/{name}: the builder did not mark it identical")
                continue
            if not entry.get("sha256"):
                problems.append(f"{split}/{name}: no sha256 recorded")
                continue
            rows.append(
                SequenceRow(
                    split=str(split),
                    name=str(name),
                    dialogue=str(dialogues[0]),
                    sha256=str(entry["sha256"]),
                    group_size=group_size,
                )
            )

    seen: dict[str, str] = {}
    for row in rows:
        if row.dialogue in seen:
            problems.append(
                f"{row.split}/{row.name}: dialogue {row.dialogue} is also row {seen[row.dialogue]}"
            )
        seen[row.dialogue] = f"{row.split}/{row.name}"

    if problems:
        raise SequencePlanError("; ".join(problems))
    return rows


def room_tone_block(
    index: Mapping[str, Any], *, pool: str, index_sha256: str, segments_sha256: str
) -> dict[str, Any]:
    """The room-tone facts a manifest row carries.

    The pool is shared by every dialogue, so this block is identical on all rows. It is
    repeated anyway: a row that names its own bed can be checked on its own, and
    `registry/*.json` already claims which recordings the bed excludes.
    """
    return {
        "pool": pool,
        "index_sha256": index_sha256,
        "segments_sha256": segments_sha256,
        "segments": len(index.get("index", [])),
        "seconds": index.get("total_seconds"),
        "sources": len(index.get("sources_used", [])),
        "held_out_excluded": list(index.get("excluded_held_out", [])),
    }


def backchannel_block(
    log_row: Mapping[str, Any] | None, *, path: str | None = None, sha256: str | None = None
) -> dict[str, Any] | None:
    """The backchannel wav mixed into one dialogue, or None if it has none.

    None is a real answer: a dialogue with no backchannel turn exists, and writing an empty
    object instead would make "no backchannel" and "backchannel not recorded" the same row.
    """
    if log_row is None:
        return None
    if not path or not sha256:
        raise ManifestRowError(
            f"backchannel for {log_row.get('dialogue_id')!r} has no measured path and sha256"
        )
    return {
        "path": path,
        "sha256": sha256,
        "text": log_row["text"],
        "seconds": log_row["seconds"],
        "seed": log_row["seed"],
        "turn_index": log_row["turn_index"],
    }


def manifest_row(
    *,
    dialogue: Mapping[str, Any],
    split: str,
    sequence: SequenceRow,
    artifacts: DialogueArtifacts,
    dataset_id: str,
    metadata: Mapping[str, Any],
    room_tone: Mapping[str, Any],
    backchannel: Mapping[str, Any] | None,
    extra_derivation: Sequence[str] = (),
) -> dict[str, Any]:
    """Assemble one manifest row. Pure: every measurement arrives as an argument."""
    group_id = dialogue.get("dialogue_id")
    if not group_id:
        raise ManifestRowError("a dialogue with no dialogue_id cannot be a manifest row")
    missing = [field for field in DATASET_METADATA_FIELDS if not metadata.get(field)]
    if missing:
        raise ManifestRowError(f"missing dataset metadata: {', '.join(missing)}")

    # Three sources name the split, and disagreeing on it puts a dialogue in two places at
    # once. The split map is the authority; the other two have to match it.
    if dialogue.get("split") != split:
        raise ManifestRowError(
            f"{group_id}: the split map says {split!r} and the script says "
            f"{dialogue.get('split')!r}"
        )
    if sequence.split != split:
        raise ManifestRowError(
            f"{group_id}: the split map says {split!r} and the build report puts its "
            f"sequence row in {sequence.split!r}"
        )
    if sequence.dialogue != group_id:
        raise ManifestRowError(
            f"{group_id}: sequence row {sequence.name} is a copy of {sequence.dialogue!r}"
        )
    if artifacts.sha256 != sequence.sha256:
        raise ManifestRowError(
            f"{group_id}: the wav on disk is {artifacts.sha256[:12]}... and the build "
            f"report recorded {sequence.sha256[:12]}..."
        )

    turns = dialogue.get("turns") or []
    if not turns:
        raise ManifestRowError(f"{group_id}: no turns")
    for index, turn in enumerate(turns):
        if "role" not in turn:
            raise ManifestRowError(f"{group_id}: turn {index} has no role")

    source_artifact_id = dialogue.get("source_artifact_id")
    if not source_artifact_id:
        raise ManifestRowError(f"{group_id}: no source_artifact_id")

    return {
        "artifact_id": f"{dataset_id}:{group_id}",
        "audio": dict(artifacts.audio),
        "backchannel": dict(backchannel) if backchannel is not None else None,
        "byte_size": artifacts.byte_size,
        "credit": metadata["credit"],
        "dataset_id": dataset_id,
        "derivation": [source_artifact_id, *extra_derivation],
        "generation_method": metadata["generation_method"],
        "group_id": group_id,
        "license_id": metadata["license_id"],
        "license_url": metadata["license_url"],
        "media_type": MEDIA_TYPE,
        "path": artifacts.path,
        "redistribution": metadata["redistribution"],
        "retrieved_at": metadata["retrieved_at"],
        "room_tone": dict(room_tone),
        "schema_version": 1,
        "sequence_row": {
            "name": sequence.name,
            "path": artifacts.sequence_audio_path,
            "group_size": sequence.group_size,
            "identical_to_dialogue": True,
        },
        "sha256": artifacts.sha256,
        "source_artifact_id": source_artifact_id,
        "source_url": metadata["source_url"],
        "source_version": metadata["source_version"],
        "split": split,
        "text": " ".join(turn["text"] for turn in turns),
        "tokenized": {
            "dialogue_id": f"{split}/{group_id}",
            "audio_wav": artifacts.split_audio_path,
            "word_transcript": artifacts.word_transcript_path,
            "word_transcript_sha256": artifacts.word_transcript_sha256,
            "tok_audio_npz_sha256": artifacts.tok_audio_sha256,
            "tok_text_npz_sha256": artifacts.tok_text_sha256,
            "parquet": artifacts.parquet_path,
            "parquet_sha256": artifacts.parquet_sha256,
        },
        "turn_count": len(turns),
        "turn_roles": [turn["role"] for turn in turns],
    }


# --------------------------------------------------------------------------------------
# I/O. Everything above is pure; everything below reads or writes files.
# --------------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audio_metadata(path: Path) -> dict[str, Any]:
    """The audio block, rounded the way `tools/experiment_data._audio_metadata` rounds it.

    `validate_manifest` compares a row against that function, so a sixth decimal place of
    disagreement here would be a validation failure with no cause anybody could see.
    """
    import wave

    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
        return {
            "channels": handle.getnchannels(),
            "duration_seconds": round(frames / rate, 6),
            "frame_count": frames,
            "sample_rate_hz": rate,
            "sample_width_bytes": handle.getsampwidth(),
        }


def materialise_splits(
    dataset_root: Path, plan: Sequence[SequenceRow], *, dry_run: bool = False
) -> list[dict[str, Any]]:
    """Copy each sequence row into `<split>/{audio,text}` under its dialogue's name.

    Returns one check per file copied. Raises before writing anything if any source
    disagrees with the build report, so a half-materialised dataset is not a state this can
    leave behind.
    """
    problems: list[str] = []
    work: list[tuple[SequenceRow, str, Path, Path, Path, str]] = []
    for row in plan:
        for kind, suffix in (("audio", ".wav"), ("text", ".json")):
            source = dataset_root / "sequences" / row.split / kind / f"{row.name}{suffix}"
            twin = dataset_root / kind / f"{row.dialogue}{suffix}"
            destination = dataset_root / row.split / kind / f"{row.dialogue}{suffix}"
            if not source.is_file():
                problems.append(f"{row.split}/{kind}/{row.name}: {source} is missing")
                continue
            if not twin.is_file():
                problems.append(f"{row.split}/{kind}/{row.dialogue}: {twin} is missing")
                continue
            source_sha = sha256_file(source)
            if source_sha != sha256_file(twin):
                problems.append(
                    f"{row.split}/{kind}/{row.name}: the sequence row differs from dialogue "
                    f"{row.dialogue}"
                )
                continue
            if kind == "audio" and source_sha != row.sha256:
                problems.append(
                    f"{row.split}/{kind}/{row.name}: the sequence row differs from the "
                    f"sha256 in the build report"
                )
                continue
            work.append((row, kind, source, twin, destination, source_sha))

    if problems:
        raise SequencePlanError("; ".join(problems))

    checks: list[dict[str, Any]] = []
    for row, kind, source, _twin, destination, source_sha in work:
        if not dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                destination.unlink()
            shutil.copyfile(source, destination)
            written = sha256_file(destination)
            if written != source_sha:
                raise SequencePlanError(f"{destination}: the copy differs from its source")
        checks.append(
            {
                "split": row.split,
                "kind": kind,
                "sequence": row.name,
                "dialogue": row.dialogue,
                "sha256": source_sha,
            }
        )
    return checks


def collect_artifacts(
    *,
    data_root: Path,
    dataset_dir: str,
    dialogue_id: str,
    split: str,
    sequence_name: str,
) -> DialogueArtifacts:
    """Measure everything a manifest row repeats, off the files themselves."""
    root = data_root / dataset_dir
    wav = root / "audio" / f"{dialogue_id}.wav"
    split_wav = root / split / "audio" / f"{dialogue_id}.wav"
    sequence_wav = root / "sequences" / split / "audio" / f"{sequence_name}.wav"
    transcript = root / "text" / f"{dialogue_id}.json"
    split_transcript = root / split / "text" / f"{dialogue_id}.json"
    parquet = root / "parquet" / f"{split}-001-of-001.parquet"

    wav_sha = sha256_file(wav)
    for twin in (split_wav, sequence_wav):
        if sha256_file(twin) != wav_sha:
            raise ManifestRowError(f"{dialogue_id}: {twin} differs from the dialogue wav")
    transcript_sha = sha256_file(transcript)
    if sha256_file(split_transcript) != transcript_sha:
        raise ManifestRowError(
            f"{dialogue_id}: {split_transcript} differs from the dialogue transcript"
        )

    def relative(path: Path) -> str:
        return path.relative_to(data_root).as_posix()

    return DialogueArtifacts(
        path=relative(wav),
        sha256=wav_sha,
        byte_size=wav.stat().st_size,
        audio=audio_metadata(wav),
        split_audio_path=relative(split_wav),
        sequence_audio_path=relative(sequence_wav),
        word_transcript_path=relative(split_transcript),
        word_transcript_sha256=transcript_sha,
        tok_audio_sha256=sha256_file(root / split / "tok-audio" / f"{dialogue_id}.npz"),
        tok_text_sha256=sha256_file(root / split / "tok-text" / f"{dialogue_id}.npz"),
        parquet_path=relative(parquet),
        parquet_sha256=sha256_file(parquet),
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def build_rows(spec: Mapping[str, Any], *, repository_root: Path) -> list[dict[str, Any]]:
    """Build every manifest row of the dataset the spec describes."""
    data_root = repository_root / spec["data_root"]
    dataset_dir = spec["dataset_dir"]
    dataset_id = spec["dataset_id"]

    dialogues = read_jsonl(repository_root / spec["scripts"]["dialogues"])
    split_map = json.loads(
        (repository_root / spec["scripts"]["split_map"]).read_text(encoding="utf-8")
    )["assignment"]
    build_report = json.loads((repository_root / spec["build_report"]).read_text(encoding="utf-8"))
    plan = {row.dialogue: row for row in sequence_plan(build_report)}

    backchannel_log = {
        row["dialogue_id"]: row for row in read_jsonl(data_root / spec["backchannel_log"])
    }
    pool = spec["room_tone_pool"]
    index_path = data_root / pool / "index.json"
    segments_path = data_root / pool / "segments.npz"
    room_tone = room_tone_block(
        json.loads(index_path.read_text(encoding="utf-8")),
        pool=pool,
        index_sha256=sha256_file(index_path),
        segments_sha256=sha256_file(segments_path),
    )

    rows = []
    for dialogue in dialogues:
        group_id = dialogue["dialogue_id"]
        if group_id not in split_map:
            raise ManifestRowError(f"{group_id}: the split map does not assign it")
        if group_id not in plan:
            raise ManifestRowError(f"{group_id}: the build report has no sequence row")
        split = split_map[group_id]
        sequence = plan[group_id]
        log_row = backchannel_log.get(group_id)
        backchannel = None
        if log_row is not None:
            wav = data_root / Path(spec["backchannel_log"]).parent / log_row["filename"]
            backchannel = backchannel_block(
                log_row,
                path=wav.relative_to(data_root).as_posix(),
                sha256=sha256_file(wav),
            )
        rows.append(
            manifest_row(
                dialogue=dialogue,
                split=split,
                sequence=sequence,
                artifacts=collect_artifacts(
                    data_root=data_root,
                    dataset_dir=dataset_dir,
                    dialogue_id=group_id,
                    split=split,
                    sequence_name=sequence.name,
                ),
                dataset_id=dataset_id,
                metadata=spec["metadata"],
                room_tone=room_tone,
                backchannel=backchannel,
                extra_derivation=spec["metadata"].get("extra_derivation", ()),
            )
        )
    rows.sort(key=lambda row: row["group_id"])
    return rows


def write_manifest(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    from collections import Counter

    return {
        "rows": len(rows),
        "splits": dict(Counter(row["split"] for row in rows)),
        "with_backchannel": sum(1 for row in rows if row["backchannel"]),
        "turn_counts": dict(Counter(row["turn_count"] for row in rows)),
    }


def _cmd_materialise(args: argparse.Namespace) -> int:
    build_report = json.loads(Path(args.build_report).read_text(encoding="utf-8"))
    checks = materialise_splits(
        Path(args.dataset_root), sequence_plan(build_report), dry_run=args.dry_run
    )
    per_directory: dict[str, int] = {}
    for check in checks:
        key = f"{check['split']}/{check['kind']}"
        per_directory[key] = per_directory.get(key, 0) + 1
    payload = {"verified_triples": len(checks), "per_directory": per_directory}
    print(json.dumps(payload, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    repository_root = Path(args.repository_root).resolve()
    spec = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    rows = build_rows(spec, repository_root=repository_root)
    out = Path(args.out) if args.out else repository_root / spec["manifest"]
    write_manifest(out, rows)
    print(
        json.dumps(
            {**_summary(rows), "out": out.relative_to(repository_root).as_posix()},
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    materialise = sub.add_parser(
        "materialise-splits",
        help="copy each sequence row into <split>/{audio,text} under its dialogue's name",
    )
    materialise.add_argument("--dataset_root", required=True)
    materialise.add_argument("--build_report", required=True)
    materialise.add_argument("--out", help="write the per-file checks here")
    materialise.add_argument(
        "--dry_run",
        action="store_true",
        help="verify the three-way agreement and copy nothing",
    )
    materialise.set_defaults(func=_cmd_materialise)

    build = sub.add_parser("build", help="write the manifest the dataset spec describes")
    build.add_argument("--dataset", required=True, help="path to the dataset spec JSON")
    build.add_argument("--out", help="manifest path; defaults to the spec's `manifest`")
    build.add_argument(
        "--repository_root",
        default=str(Path(__file__).resolve().parents[1]),
        help="paths in the spec are resolved against this",
    )
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
