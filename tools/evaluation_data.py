from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path
from typing import Any, Iterable


STYLE_MARKERS = (
    "ですわ",
    "ますわ",
    "ですの",
    "ますの",
    "ましたわ",
    "ましたの",
    "でしたわ",
    "でしたの",
    "かしら",
    "わたくし",
    "よろしくて",
    "ございま",
    "ませんこと",
    "くださいませ",
    "くださいな",
)


class EvaluationValidationError(ValueError):
    """Raised when a fixed evaluation set violates the M1 contract."""


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


def build_voice_evaluation_index(manifest: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select ten deterministic seen and ten held-out voice references."""
    rows = list(manifest)
    train_rows = sorted(
        (row for row in rows if row.get("split") == "train"),
        key=lambda row: str(row.get("artifact_id", "")),
    )
    test_rows = sorted(
        (row for row in rows if row.get("split") == "test"),
        key=lambda row: str(row.get("artifact_id", "")),
    )
    if len(train_rows) < 10 or len(test_rows) < 10:
        raise EvaluationValidationError(
            "voice evaluation requires at least 10 train and 10 test artifacts"
        )

    selected: list[dict[str, Any]] = []
    for partition, source_rows in (("seen", train_rows[:10]), ("held-out", test_rows[:10])):
        for source in source_rows:
            missing = [field for field in ("artifact_id", "text", "sha256") if not source.get(field)]
            if missing:
                raise EvaluationValidationError(
                    f"voice evaluation source is missing: {', '.join(missing)}"
                )
            selected.append(
                {
                    "schema_version": 1,
                    "id": f"voice-{partition}-{len(selected) + 1:02d}",
                    "partition": partition,
                    "source_split": source["split"],
                    "artifact_id": source["artifact_id"],
                    "text": source["text"],
                    "sha256": source["sha256"],
                }
            )
    return selected


def _validate_rows(
    rows: list[dict[str, Any]],
    *,
    label: str,
    count: int,
    required_fields: tuple[str, ...],
) -> None:
    if len(rows) != count:
        raise EvaluationValidationError(f"fixed evaluation requires exactly {count} {label} rows")
    identifiers: set[str] = set()
    for row_number, row in enumerate(rows, start=1):
        missing = [field for field in required_fields if not row.get(field)]
        if missing:
            raise EvaluationValidationError(
                f"{label} row {row_number} is missing: {', '.join(missing)}"
            )
        identifier = str(row["id"])
        if identifier in identifiers:
            raise EvaluationValidationError(f"{label} contains duplicate id: {identifier}")
        identifiers.add(identifier)


def _evaluation_texts(
    tts_rows: list[dict[str, Any]],
    style_rows: list[dict[str, Any]],
    general_rows: list[dict[str, Any]],
) -> Iterable[tuple[str, str]]:
    for row in tts_rows:
        yield str(row["id"]), str(row["text"])
    for row in style_rows:
        yield f"{row['id']}:prompt", str(row["prompt"])
        yield f"{row['id']}:preferred", str(row["preferred"])
        yield f"{row['id']}:dispreferred", str(row["dispreferred"])
    for row in general_rows:
        yield str(row["id"]), str(row["user_prompt"])


def validate_fixed_evaluation(
    *,
    tts_rows: list[dict[str, Any]],
    style_rows: list[dict[str, Any]],
    general_rows: list[dict[str, Any]],
    training_manifest: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate cardinality, schema, style markers, and training isolation."""
    _validate_rows(
        tts_rows,
        label="TTS",
        count=30,
        required_fields=("id", "text", "tags"),
    )
    _validate_rows(
        style_rows,
        label="style pair",
        count=50,
        required_fields=("id", "topic", "prompt", "preferred", "dispreferred"),
    )
    _validate_rows(
        general_rows,
        label="general dialogue",
        count=30,
        required_fields=("id", "category", "user_prompt", "success_criteria"),
    )

    for row in style_rows:
        preferred = str(row["preferred"])
        dispreferred = str(row["dispreferred"])
        if preferred == dispreferred:
            raise EvaluationValidationError(f"{row['id']}: preferred and dispreferred must differ")
        if not any(marker in preferred for marker in STYLE_MARKERS):
            raise EvaluationValidationError(
                f"{row['id']}: preferred response has no recognised style marker"
            )

    train_texts = [
        str(row["text"])
        for row in training_manifest
        if row.get("split") == "train" and row.get("text")
    ]
    leakage = []
    for evaluation_id, evaluation_text in _evaluation_texts(tts_rows, style_rows, general_rows):
        if any(_near_duplicate(evaluation_text, train_text) for train_text in train_texts):
            leakage.append(evaluation_id)
    if leakage:
        raise EvaluationValidationError(
            f"training leakage detected in fixed evaluation: {', '.join(leakage)}"
        )

    return {
        "status": "pass",
        "tts_count": len(tts_rows),
        "style_pair_count": len(style_rows),
        "general_dialogue_count": len(general_rows),
        "training_leakage_count": 0,
        "style_markers": list(STYLE_MARKERS),
        "near_duplicate_threshold": 0.9,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvaluationValidationError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(value, dict):
            raise EvaluationValidationError(f"{path}:{line_number}: expected JSON object")
        rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and validate fixed M1 evaluation data")
    subparsers = parser.add_subparsers(dest="command", required=True)

    voice_parser = subparsers.add_parser("build-voice-index")
    voice_parser.add_argument("--manifest", type=Path, required=True)
    voice_parser.add_argument("--output", type=Path, required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--tts", type=Path, required=True)
    validate_parser.add_argument("--style", type=Path, required=True)
    validate_parser.add_argument("--general", type=Path, required=True)
    validate_parser.add_argument("--training-manifest", type=Path, required=True)
    validate_parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "build-voice-index":
        _write_jsonl(args.output, build_voice_evaluation_index(_read_jsonl(args.manifest)))
        return 0

    report = validate_fixed_evaluation(
        tts_rows=_read_jsonl(args.tts),
        style_rows=_read_jsonl(args.style),
        general_rows=_read_jsonl(args.general),
        training_manifest=_read_jsonl(args.training_manifest),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
