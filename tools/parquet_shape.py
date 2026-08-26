"""Measure a shipped dialogue parquet against the shape the trainer will feed the model.

The gate this exists for
------------------------
`utils/data.main_speaker_streams` is what `finetune.py` calls on every batch: it takes the
two speaker columns and lays them out as one example - the main speaker's text row and eight
Mimi codebooks, plus the other speaker's eight - seventeen streams, matching the seventeen
delays in `moshi_lm_kwargs.json`. A parquet whose rows do not come out at seventeen streams
trains anyway and learns the wrong thing.

Counting it with the trainer's own function rather than asserting the `[9, T]` column shape
is the point. The column shape is what `tools/prepare_dataset.py` wrote; the stream count is
what the model receives, and the two are only equal while the function in between behaves
the way the person reading the column shape assumed.

`summarise_streams` is pure and takes plain sequences, so the gate can be tested without
numpy, pandas or a dataset. Everything that needs those is below it and imports them inside
the function.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

# Seventeen rows: text plus eight codebooks for the main speaker, eight for the other. The
# delay pattern in moshi_lm_kwargs.json has the same seventeen entries, and
# docs/experiments/j-moshi-tsukuyomi-ojousama-m3r-dataset-audit.md section 1 matched them
# index by index.
EXPECTED_STREAMS_PER_EXAMPLE = 17

# `tools/prepare_dataset.merge_text_audio` writes one text row ahead of the eight codebooks,
# so a speaker column is [9, T].
EXPECTED_STREAMS_PER_SPEAKER_COLUMN = 9


class ParquetShapeError(ValueError):
    """The parquet is not shaped like something `finetune.py` can train on."""


def summarise_streams(examples: Iterable[Sequence[Sequence[Any]]]) -> dict[str, Any]:
    """Reduce the trainer's view of a split to counts.

    `examples` is what `main_speaker_streams` returns: one entry per training row, each a
    sequence of streams, each stream a sequence of tokens. Ragged streams inside one example
    are an error rather than a shrug - the example is a rectangle by construction, and a
    ragged one means something upstream padded a row it should not have.
    """
    stream_counts: set[int] = set()
    frame_counts: list[int] = []
    count = 0
    for index, example in enumerate(examples):
        count += 1
        streams = list(example)
        if not streams:
            raise ParquetShapeError(f"example {index} has no streams")
        lengths = {len(stream) for stream in streams}
        if len(lengths) != 1:
            raise ParquetShapeError(
                f"example {index} has streams of {sorted(lengths)} frames; an example is a "
                "rectangle and a ragged one means a row was padded that should not have been"
            )
        stream_counts.add(len(streams))
        frame_counts.append(lengths.pop())
    if not count:
        raise ParquetShapeError("no examples")
    return {
        "examples": count,
        "streams_per_example": sorted(stream_counts),
        "frames": {
            "min": min(frame_counts),
            "max": max(frame_counts),
            "median": float(statistics.median(frame_counts)),
        },
    }


def stream_shape_problems(
    summary: dict[str, Any], *, expected: int = EXPECTED_STREAMS_PER_EXAMPLE
) -> list[str]:
    """Problems with a shape summary, as sentences. Empty means it holds up."""
    problems = []
    counts = summary.get("streams_per_example") or []
    if counts != [expected]:
        problems.append(
            f"streams per example is {counts}, not [{expected}]; the delay pattern has "
            f"{expected} entries and a row that does not match them trains on the wrong "
            "alignment"
        )
    return problems


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarise_parquet(parquet_path: Path, *, speakers: Sequence[str] = ("A", "B")) -> dict:
    """Read one parquet and measure it through the trainer's own collation."""
    import numpy as np
    import pandas as pd

    from utils.data import main_speaker_streams

    frame = pd.read_parquet(parquet_path)
    missing = [column for column in ("dialogue_id", *speakers) if column not in frame.columns]
    if missing:
        raise ParquetShapeError(
            f"{parquet_path}: no {', '.join(missing)} column; tools/prepare_dataset.py "
            "writes dialogue_id, A and B"
        )

    columns: dict[str, list[Any]] = {}
    per_speaker_rows: dict[str, list[int]] = {}
    for speaker in speakers:
        arrays = []
        for cell in frame[speaker]:
            merged = np.asarray(cell.tolist() if hasattr(cell, "tolist") else cell)
            if merged.ndim != 2:
                raise ParquetShapeError(
                    f"{parquet_path}: speaker {speaker} is {merged.ndim}D, expected "
                    "[streams, frames]"
                )
            arrays.append(merged)
        columns[speaker] = arrays
        per_speaker_rows[speaker] = sorted({int(a.shape[0]) for a in arrays})

    streams = main_speaker_streams(columns, list(speakers))
    summary = summarise_streams([list(example) for example in streams])
    ids = [str(value) for value in frame["dialogue_id"]]
    return {
        "path": parquet_path.as_posix(),
        "sha256": _sha256(parquet_path),
        "byte_size": parquet_path.stat().st_size,
        "rows": int(len(frame)),
        "streams_per_speaker_column": per_speaker_rows,
        "dialogue_id_namespace": sorted({value.split("/", 1)[0] for value in ids}),
        "dialogue_ids_unique": len(set(ids)) == len(ids),
        **summary,
    }


def _cmd_measure(args: argparse.Namespace) -> int:
    measurements = [summarise_parquet(Path(path)) for path in args.parquet]
    problems = [
        f"{measurement['path']}: {problem}"
        for measurement in measurements
        for problem in stream_shape_problems(measurement, expected=args.expected_streams)
    ]
    payload = {"parquets": measurements, "problems": problems, "passed": not problems}
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if not problems else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    measure = sub.add_parser("measure", help="count the streams the trainer will receive")
    measure.add_argument("parquet", nargs="+")
    measure.add_argument("--expected_streams", type=int, default=EXPECTED_STREAMS_PER_EXAMPLE)
    measure.add_argument("--out", help="write the JSON here as well as to stdout")
    measure.set_defaults(func=_cmd_measure)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
