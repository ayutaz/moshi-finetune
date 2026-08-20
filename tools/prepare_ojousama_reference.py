from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def prepare_reference(
    rows: list[dict[str, Any]], *, source_commit: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    accepted = []
    seen_prompts: set[str] = set()
    rejected_source_rows = []
    duplicate_prompts = []
    for source_row, row in enumerate(rows, start=1):
        if set(row) != {"prompt", "completion"}:
            raise ValueError(f"source row {source_row}: expected prompt/completion keys")
        prompt = row["prompt"].strip()
        completion = row["completion"].strip()
        if not prompt or not completion:
            raise ValueError(f"source row {source_row}: prompt/completion must be non-empty")
        if prompt in seen_prompts:
            rejected_source_rows.append(source_row)
            duplicate_prompts.append(prompt)
            continue
        seen_prompts.add(prompt)
        accepted.append(
            {
                "schema_version": 1,
                "id": f"ojousama-reference-{len(accepted) + 1:03d}",
                "source_commit": source_commit,
                "source_row": source_row,
                "prompt": prompt,
                "completion": completion,
                "usage": "reference-only",
                "split": "held-out",
            }
        )

    report = {
        "status": "pass-with-deduplication" if rejected_source_rows else "pass",
        "raw_row_count": len(rows),
        "accepted_row_count": len(accepted),
        "rejected_row_count": len(rejected_source_rows),
        "duplicate_prompt_count": len(set(duplicate_prompts)),
        "duplicate_prompts": sorted(set(duplicate_prompts)),
        "rejected_source_rows": rejected_source_rows,
        "source_commit": source_commit,
        "usage": "reference-only",
    }
    return accepted, report


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and deduplicate Ojousama reference data")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    accepted, report = prepare_reference(_read_jsonl(args.input), source_commit=args.source_commit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in accepted),
        encoding="utf-8",
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
