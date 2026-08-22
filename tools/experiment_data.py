from __future__ import annotations

import argparse
import hashlib
import json
import math
import mimetypes
import re
import struct
import unicodedata
import wave
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = (
    "schema_version",
    "artifact_id",
    "dataset_id",
    "path",
    "media_type",
    "byte_size",
    "sha256",
    "source_url",
    "source_version",
    "retrieved_at",
    "license_id",
    "license_url",
    "credit",
    "redistribution",
    "generation_method",
    "derivation",
    "group_id",
    "split",
)
SPLITS = ("train", "dev", "test")


class ManifestValidationError(ValueError):
    """Raised when an experiment manifest violates a data gate."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _riff_audio_metadata(path: Path) -> dict[str, Any]:
    with path.open("rb") as file_handle:
        if file_handle.read(4) != b"RIFF":
            raise ManifestValidationError(f"corrupt audio file: {path}")
        file_handle.seek(8)
        if file_handle.read(4) != b"WAVE":
            raise ManifestValidationError(f"corrupt audio file: {path}")

        format_data = None
        data_size = None
        while True:
            chunk_header = file_handle.read(8)
            if not chunk_header:
                break
            if len(chunk_header) != 8:
                raise ManifestValidationError(f"corrupt audio chunk header: {path}")
            chunk_id, chunk_size = struct.unpack("<4sI", chunk_header)
            if chunk_id == b"fmt ":
                format_data = file_handle.read(chunk_size)
                if len(format_data) != chunk_size:
                    raise ManifestValidationError(f"truncated audio format chunk: {path}")
            elif chunk_id == b"data":
                data_size = chunk_size
                file_handle.seek(chunk_size, 1)
            else:
                file_handle.seek(chunk_size, 1)
            if chunk_size % 2:
                file_handle.seek(1, 1)

    if format_data is None or data_size is None or len(format_data) < 16:
        raise ManifestValidationError(f"missing audio format/data chunk: {path}")
    format_tag, channels, sample_rate, _, block_align, bits_per_sample = struct.unpack(
        "<HHIIHH", format_data[:16]
    )
    if format_tag == 0xFFFE and len(format_data) >= 40:
        format_tag = struct.unpack("<H", format_data[24:26])[0]
    if format_tag not in (1, 3):
        raise ManifestValidationError(f"unsupported audio format {format_tag}: {path}")
    if sample_rate <= 0 or channels <= 0 or block_align <= 0 or bits_per_sample <= 0:
        raise ManifestValidationError(f"invalid audio metadata: {path}")
    if data_size % block_align:
        raise ManifestValidationError(f"misaligned audio data chunk: {path}")
    frame_count = data_size // block_align
    if frame_count <= 0:
        raise ManifestValidationError(f"empty audio data chunk: {path}")
    return {
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "sample_width_bytes": math.ceil(bits_per_sample / 8),
        "frame_count": frame_count,
        "duration_seconds": round(frame_count / sample_rate, 6),
    }


def _audio_metadata(path: Path) -> dict[str, Any]:
    try:
        with wave.open(str(path), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_count = wav_file.getnframes()
            if sample_rate <= 0 or channels <= 0 or sample_width <= 0 or frame_count <= 0:
                raise ManifestValidationError(f"invalid audio metadata: {path}")
    except (EOFError, wave.Error):
        return _riff_audio_metadata(path)

    return {
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "sample_width_bytes": sample_width,
        "frame_count": frame_count,
        "duration_seconds": round(frame_count / sample_rate, 6),
    }


def _allocate_split_counts(count: int) -> dict[str, int]:
    if count <= 0:
        return dict.fromkeys(SPLITS, 0)

    ratios = {"train": 0.8, "dev": 0.1, "test": 0.1}
    raw_counts = {split: count * ratio for split, ratio in ratios.items()}
    counts = {split: math.floor(value) for split, value in raw_counts.items()}
    remainder = count - sum(counts.values())
    order = sorted(
        SPLITS,
        key=lambda split: (raw_counts[split] - counts[split], -SPLITS.index(split)),
        reverse=True,
    )
    for split in order[:remainder]:
        counts[split] += 1
    return counts


def _assign_splits(group_ids: Iterable[str], seed: str) -> dict[str, str]:
    unique_groups = sorted(set(group_ids))
    ranked_groups = sorted(
        unique_groups,
        key=lambda group_id: hashlib.sha256(f"{seed}\0{group_id}".encode()).hexdigest(),
    )
    counts = _allocate_split_counts(len(ranked_groups))
    assignments: dict[str, str] = {}
    cursor = 0
    for split in SPLITS:
        for group_id in ranked_groups[cursor : cursor + counts[split]]:
            assignments[group_id] = split
        cursor += counts[split]
    return assignments


def resolve_splits(
    group_ids: Iterable[str], *, seed: str, override: dict[str, str] | None
) -> dict[str, str]:
    """Split assignment for each group, from an explicit map when one is given.

    M3 needs the override. Its split is committed in `m3/scripts/split-map-v1.json` and
    shared by V-real and V-tts, which are built from the same scripts; re-deriving it from
    a hash per dataset would give the two different splits and the paired comparison would
    stop being paired.

    A group the override does not mention is an error rather than a fallback to the hash.
    Falling back is how half a dataset ends up on a different split from the other half,
    with nothing to show for it.
    """
    if override is None:
        return _assign_splits(group_ids, seed)

    groups = set(group_ids)
    missing = sorted(groups - set(override))
    unknown = sorted(set(override) - groups)
    if missing:
        raise ValueError(f"split override does not cover: {missing[:10]}")
    if unknown:
        raise ValueError(f"split override names groups that do not exist: {unknown[:10]}")
    bad = sorted({value for value in override.values()} - set(SPLITS))
    if bad:
        raise ValueError(f"unknown split name(s): {bad}")
    return dict(override)


def apply_row_overrides(
    rows: list[dict[str, Any]], overrides: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Merge per-row fields into a built manifest, keyed by group_id.

    `derivation` is the reason this exists. build_manifest copies one dataset-wide value
    onto every row, but each V dialogue quotes a different corpus sentence, and the leakage
    assertion in tests/test_experiment_assets.py reads derivation PER ROW. One shared value
    would make that check vacuous.

    A row without an override, or an override without a row, is an error: both mean the
    caller's idea of the dataset and the manifest's have diverged.
    """
    by_group = {row["group_id"]: row for row in rows}
    missing = sorted(set(by_group) - set(overrides))
    unknown = sorted(set(overrides) - set(by_group))
    if missing:
        raise ValueError(f"no row override for: {missing[:10]}")
    if unknown:
        raise ValueError(f"row overrides for absent rows: {unknown[:10]}")
    return [{**row, **overrides[row["group_id"]]} for row in rows]


def build_manifest(
    *,
    data_root: Path,
    source_dir: Path,
    metadata: dict[str, Any],
    seed: str,
    transcripts: dict[str, str] | None = None,
    split_override: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Inventory WAV files and deterministically assign 80/10/10 splits."""
    data_root = data_root.resolve()
    source_dir = source_dir.resolve()
    try:
        source_dir.relative_to(data_root)
    except ValueError as exc:
        raise ValueError("source_dir must be inside data_root") from exc

    missing_metadata = [field for field in REQUIRED_FIELDS[6:13] if not metadata.get(field)]
    # REQUIRED_FIELDS includes row-only fields; check the dataset-level contract explicitly.
    dataset_fields = (
        "dataset_id",
        "source_url",
        "source_version",
        "retrieved_at",
        "license_id",
        "license_url",
        "credit",
        "redistribution",
        "generation_method",
    )
    missing_metadata = [field for field in dataset_fields if not metadata.get(field)]
    if missing_metadata:
        raise ValueError(f"missing dataset metadata: {', '.join(missing_metadata)}")

    audio_paths = sorted(path for path in source_dir.rglob("*.wav") if path.is_file())
    if not audio_paths:
        raise ValueError(f"no WAV files found below {source_dir}")

    group_ids = [path.relative_to(source_dir).with_suffix("").as_posix() for path in audio_paths]
    assignments = resolve_splits(group_ids, seed=seed, override=split_override)
    transcript_map = transcripts or {}
    rows = []
    for path, group_id in zip(audio_paths, group_ids, strict=True):
        relative_path = path.relative_to(data_root).as_posix()
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        row = {
            "schema_version": 1,
            "artifact_id": f"{metadata['dataset_id']}:{group_id}",
            "dataset_id": metadata["dataset_id"],
            "path": relative_path,
            "media_type": media_type,
            "byte_size": path.stat().st_size,
            "sha256": _sha256(path),
            "source_url": metadata["source_url"],
            "source_version": metadata["source_version"],
            "retrieved_at": metadata["retrieved_at"],
            "license_id": metadata["license_id"],
            "license_url": metadata["license_url"],
            "credit": metadata["credit"],
            "redistribution": metadata["redistribution"],
            "generation_method": metadata["generation_method"],
            "derivation": list(metadata.get("derivation", [])),
            "group_id": group_id,
            "split": assignments[group_id],
            "text": transcript_map.get(group_id, ""),
            "audio": _audio_metadata(path),
        }
        rows.append(row)
    return rows


def load_voiceactress_transcripts(path: Path) -> dict[str, str]:
    transcripts: dict[str, str] = {}
    pattern = re.compile(r"^(VOICEACTRESS100_\d{3}):(.*)$")
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        match = pattern.fullmatch(line)
        if not match or not match.group(2).strip():
            raise ValueError(f"invalid transcript at line {line_number}")
        utterance_id, text = match.groups()
        if utterance_id in transcripts:
            raise ValueError(f"duplicate transcript id: {utterance_id}")
        transcripts[utterance_id] = text.strip()

    expected_ids = {f"VOICEACTRESS100_{index:03d}" for index in range(1, 101)}
    if transcripts.keys() != expected_ids:
        missing = sorted(expected_ids - transcripts.keys())
        extra = sorted(transcripts.keys() - expected_ids)
        raise ValueError(
            f"transcript must contain exactly 100 utterances; missing={missing}, extra={extra}"
        )
    return transcripts


def _normalise_text(text: str) -> str:
    normalised = unicodedata.normalize("NFKC", text).casefold()
    return "".join(character for character in normalised if character.isalnum())


def _character_shingles(text: str, width: int = 3) -> set[str]:
    if len(text) <= width:
        return {text} if text else set()
    return {text[index : index + width] for index in range(len(text) - width + 1)}


def _near_duplicate(left: str, right: str) -> bool:
    left_normalised = _normalise_text(left)
    right_normalised = _normalise_text(right)
    if not left_normalised or not right_normalised:
        return False
    if left_normalised == right_normalised:
        return True
    left_shingles = _character_shingles(left_normalised)
    right_shingles = _character_shingles(right_normalised)
    union = left_shingles | right_shingles
    return bool(union) and len(left_shingles & right_shingles) / len(union) >= 0.9


def _contains_coeiroink(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_coeiroink(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_coeiroink(item) for item in value)
    return "coeiroink" in str(value).casefold()


def validate_manifest(rows: list[dict[str, Any]], *, data_root: Path) -> dict[str, Any]:
    data_root = data_root.resolve()
    errors: list[str] = []
    coeiroink_count = 0
    corrupt_audio_count = 0
    cross_split_duplicate_count = 0
    seen_artifact_ids: set[str] = set()
    checksum_splits: dict[str, set[str]] = defaultdict(set)
    checksum_artifacts: dict[str, list[str]] = defaultdict(list)
    group_splits: dict[str, set[str]] = defaultdict(set)

    for index, row in enumerate(rows, start=1):
        missing = [
            field for field in REQUIRED_FIELDS if field not in row or row[field] in (None, "")
        ]
        if missing:
            errors.append(f"row {index}: missing required fields: {', '.join(missing)}")
            continue
        if row["schema_version"] != 1:
            errors.append(f"row {index}: unsupported schema_version")
        if row["artifact_id"] in seen_artifact_ids:
            errors.append(f"row {index}: duplicate artifact_id {row['artifact_id']}")
        seen_artifact_ids.add(row["artifact_id"])
        if row["split"] not in SPLITS:
            errors.append(f"row {index}: invalid split {row['split']}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(row["sha256"])):
            errors.append(f"row {index}: invalid checksum format")

        if _contains_coeiroink(row):
            coeiroink_count += 1
            errors.append(f"row {index}: COEIROINK provenance is prohibited")

        path = (data_root / row["path"]).resolve()
        try:
            path.relative_to(data_root)
        except ValueError:
            errors.append(f"row {index}: path escapes data_root")
            continue
        if not path.is_file():
            errors.append(f"row {index}: file does not exist: {row['path']}")
            continue
        if path.stat().st_size != row["byte_size"]:
            errors.append(f"row {index}: byte_size mismatch")
        if _sha256(path) != row["sha256"]:
            errors.append(f"row {index}: checksum mismatch")

        if row["media_type"] == "audio/wav":
            try:
                actual_audio = _audio_metadata(path)
                recorded_audio = row.get("audio")
                if recorded_audio != actual_audio:
                    errors.append(f"row {index}: audio metadata mismatch")
            except ManifestValidationError as exc:
                corrupt_audio_count += 1
                errors.append(f"row {index}: {exc}")

        checksum_splits[str(row["sha256"])].add(str(row["split"]))
        checksum_artifacts[str(row["sha256"])].append(str(row["artifact_id"]))
        group_key = f"{row['dataset_id']}:{row['group_id']}"
        group_splits[group_key].add(str(row["split"]))

    duplicate_artifact_count = 0
    for checksum, artifact_ids in checksum_artifacts.items():
        if len(artifact_ids) > 1:
            duplicate_artifact_count += len(artifact_ids) - 1
            errors.append(f"checksum {checksum}: exact duplicate artifacts {sorted(artifact_ids)}")
    for checksum, splits in checksum_splits.items():
        if len(splits) > 1:
            cross_split_duplicate_count += 1
            errors.append(
                f"checksum {checksum}: exact duplicate occurs across splits {sorted(splits)}"
            )
    for group_id, splits in group_splits.items():
        if len(splits) > 1:
            cross_split_duplicate_count += 1
            errors.append(f"group_id {group_id} occurs across splits {sorted(splits)}")

    text_rows = [row for row in rows if row.get("text") and row.get("split") in SPLITS]
    for left_index, left in enumerate(text_rows):
        for right in text_rows[left_index + 1 :]:
            if left["split"] == right["split"]:
                continue
            if _near_duplicate(str(left["text"]), str(right["text"])):
                cross_split_duplicate_count += 1
                errors.append(
                    "near-duplicate text across splits: "
                    f"{left.get('artifact_id')} ({left['split']}) / "
                    f"{right.get('artifact_id')} ({right['split']})"
                )

    summary = {
        "status": "pass" if not errors else "fail",
        "artifact_count": len(rows),
        "coeiroink_artifact_count": coeiroink_count,
        "corrupt_audio_count": corrupt_audio_count,
        "cross_split_duplicate_count": cross_split_duplicate_count,
        "duplicate_artifact_count": duplicate_artifact_count,
        "split_counts": {split: sum(row.get("split") == split for row in rows) for split in SPLITS},
        "errors": errors,
    }
    if errors:
        raise ManifestValidationError("; ".join(errors))
    return summary


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file_handle:
        return [json.loads(line) for line in file_handle if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    path.write_text(content, encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and validate experiment data manifests")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="inventory WAV files")
    build_parser.add_argument("--data-root", type=Path, required=True)
    build_parser.add_argument("--source-dir", type=Path, required=True)
    build_parser.add_argument("--metadata", type=Path, required=True)
    build_parser.add_argument("--transcripts", type=Path)
    build_parser.add_argument("--seed", required=True)
    build_parser.add_argument("--output", type=Path, required=True)

    validate_parser = subparsers.add_parser("validate", help="validate a JSONL manifest")
    validate_parser.add_argument("--data-root", type=Path, required=True)
    validate_parser.add_argument("--manifest", type=Path, required=True)
    validate_parser.add_argument("--report", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "build":
        metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
        transcripts = load_voiceactress_transcripts(args.transcripts) if args.transcripts else None
        rows = build_manifest(
            data_root=args.data_root,
            source_dir=args.source_dir,
            metadata=metadata,
            seed=args.seed,
            transcripts=transcripts,
        )
        _write_jsonl(args.output, rows)
        print(json.dumps({"status": "built", "artifact_count": len(rows)}, sort_keys=True))
        return 0

    rows = _read_jsonl(args.manifest)
    try:
        report = validate_manifest(rows, data_root=args.data_root)
    except ManifestValidationError as exc:
        report = {"status": "fail", "error": str(exc)}
        if args.report:
            _write_json(args.report, report)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 1
    if args.report:
        _write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
