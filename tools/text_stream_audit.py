"""Measure the text stream of a tokenized dialogue dataset, and record how it was made.

M3 shipped a dataset tokenized without ``--no_whitespace_before_word``. That flag is not
optional for Japanese - ``README-ja.md:109`` and ``README.md:112`` both require it - and
without it ``tools/tokenize_text.py`` prefixes every pyopenjtalk word with a space, which
SentencePiece then emits as a bare ``U+2581``. In the shipped v-real train parquet 44.6%
of the tokens the tokenizer wrote are that marker and nothing else.

Nothing caught it. The eighteen-step local review before M3 billed a GPU found thirteen
defects, and this was not one of them, because the flag leaves no trace in any artifact
anybody looked at: the parquet is well formed, the row counts are right, the training loop
starts. It shows only in the token histogram, which this module computes.

Two things are needed to close that, and this module provides both.

* A measurement. :func:`summarise_text_stream` reduces a text row to counts, and
  :func:`audit_parquet` applies it to a shipped parquet.
* A declaration. :func:`resolve_tokenize_invocation` turns the argv of a
  ``tools/tokenize_text.py`` run into a record in which every flag appears with its
  resolved value, so a *dropped* flag is written down as ``false`` rather than being
  invisible by absence. :func:`write_tokenize_record` files that beside the manifest.

The declaration is what runs in CI: it is JSON, and ``tests/test_experiment_assets.py``
checks it with no dependency beyond the standard library. The measurement needs pandas,
pyarrow and sentencepiece and runs wherever the data lives - the machine that built it.

Heavy imports are inside the functions that need them, so the counting logic stays
importable in a bare environment.
"""

import argparse
import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Pieces rather than ids. `tokenizer_spm_32k_3.model` happens to put `[PAD]` at 3 and
# `U+2581` at 9, but writing those integers here would make this module a second,
# independent guess at what the tokenizer holds - and a wrong guess would not fail, it
# would quietly count the wrong tokens.
PAD_PIECE = "[PAD]"
WHITESPACE_PIECE = "▁"

# `--end_of_text_padding_id` is a setting of tokenize_text.py, not a property of the
# tokenizer: the tokenizer's id 0 is `<unk>`, and the pipeline reuses it as the
# end-of-text marker. So it is carried as an id, and the piece it maps to is recorded in
# the audit output so a reader can see which token was borrowed.
DEFAULT_END_OF_TEXT_PADDING_ID = 0

# The row `tools/prepare_dataset.merge_text_audio` writes the text stream to, ahead of the
# eight Mimi codebooks.
TEXT_ROW = 0

# Every flag of `tools/tokenize_text.py`, with the default its argparse block declares.
# The parser there is built inside `if __name__ == "__main__"` and so cannot be imported;
# this is a copy, and `tests/test_experiment_assets.py` asserts the copy still names the
# same flags as the source - a flag added there and forgotten here fails the suite.
TOKENIZE_TEXT_FLAGS: dict[str, Any] = {
    "word_transcript_dir": None,
    "output_dir": None,
    "text_tokenizer_repo": "kyutai/moshiko-pytorch-bf16",
    "text_tokenizer_name": "tokenizer_spm_32k_3.model",
    "no_whitespace_before_word": False,
    "text_padding_id": 3,
    "end_of_text_padding_id": 0,
    "audio_tokenizer_frame_rate": 12.5,
    "num_workers": 1,
    "resume": False,
}

# The two flags with no default: their entry in the table above is None, so the type of
# the default cannot stand in for the parser's type the way it does for the rest.
REQUIRED_FLAGS = frozenset({"word_transcript_dir", "output_dir"})
STORE_TRUE_FLAGS = frozenset({"no_whitespace_before_word", "resume"})


class TokenizerVocabularyError(RuntimeError):
    """A piece the audit counts is not the id the tokenizer says it is."""


class NotADialogueParquetError(ValueError):
    """The file is a parquet, but not one `tools/prepare_dataset.py` wrote.

    Raised rather than returning empty counts, so a caller sweeping a directory can tell
    "this file has no text stream to measure" from "this file's text stream is clean".
    """


@dataclass(frozen=True)
class TextStreamVocabulary:
    """The three ids the text stream is made of, resolved from a real tokenizer."""

    pad_id: int
    whitespace_id: int
    end_of_text_padding_id: int
    pad_piece: str
    whitespace_piece: str
    end_of_text_padding_piece: str

    def as_record(self) -> dict[str, Any]:
        return {
            "pad": {"id": self.pad_id, "piece": self.pad_piece},
            "bare_whitespace": {"id": self.whitespace_id, "piece": self.whitespace_piece},
            "end_of_text_padding": {
                "id": self.end_of_text_padding_id,
                "piece": self.end_of_text_padding_piece,
            },
        }


@dataclass(frozen=True)
class TextStreamCounts:
    """What one speaker's text stream is made of, across a whole split.

    ``frames`` counts every position in the stream, most of which are padding by design:
    the stream is one token per Mimi frame and speech is sparse. The ratio that matters is
    not padding but ``bare_whitespace_share_of_text_tokens`` - of the tokens the tokenizer
    actually emitted, how many carry no content.
    """

    streams: int
    frames: int
    pad: int
    bare_whitespace: int
    end_of_text_padding: int
    words: int

    @property
    def non_pad(self) -> int:
        return self.frames - self.pad

    @property
    def text_tokens(self) -> int:
        """Tokens the text tokenizer produced: words plus bare whitespace markers.

        End-of-text padding is excluded. It is written by tokenize_text.py's frame layout,
        not by the tokenizer, so counting it would dilute the ratio with something the
        flag cannot change - which is exactly why the parquet reads 32.1% against non-pad
        while the tokenizer path reads 44.6%.
        """
        return self.words + self.bare_whitespace

    @property
    def bare_whitespace_share_of_text_tokens(self) -> float:
        if self.text_tokens == 0:
            return 0.0
        return self.bare_whitespace / self.text_tokens

    @property
    def bare_whitespace_share_of_non_pad(self) -> float:
        if self.non_pad == 0:
            return 0.0
        return self.bare_whitespace / self.non_pad

    @property
    def pad_share_of_frames(self) -> float:
        if self.frames == 0:
            return 0.0
        return self.pad / self.frames

    def as_record(self) -> dict[str, Any]:
        return {
            "streams": self.streams,
            "frames": self.frames,
            "pad": self.pad,
            "pad_share_of_frames": self.pad_share_of_frames,
            "non_pad": self.non_pad,
            "bare_whitespace": self.bare_whitespace,
            "bare_whitespace_share_of_non_pad": self.bare_whitespace_share_of_non_pad,
            "end_of_text_padding": self.end_of_text_padding,
            "words": self.words,
            "text_tokens": self.text_tokens,
            "bare_whitespace_share_of_text_tokens": self.bare_whitespace_share_of_text_tokens,
        }


def resolve_vocabulary(
    tokenizer: Any,
    *,
    pad_piece: str = PAD_PIECE,
    whitespace_piece: str = WHITESPACE_PIECE,
    end_of_text_padding_id: int = DEFAULT_END_OF_TEXT_PADDING_ID,
) -> TextStreamVocabulary:
    """Look the ids up in the tokenizer, and confirm each one round-trips.

    SentencePiece returns the unknown id for a piece it does not hold, so ``piece_to_id``
    on its own cannot tell "``[PAD]`` is 3" from "``[PAD]`` is absent and 0 means unknown".
    Both would return an integer and the audit would count whatever sat there. Asking the
    tokenizer to map the id back closes that: a piece that is not in the vocabulary cannot
    round-trip to itself.
    """
    resolved = {}
    for name, piece in (("pad", pad_piece), ("whitespace", whitespace_piece)):
        token_id = tokenizer.piece_to_id(piece)
        back = tokenizer.id_to_piece(token_id)
        if back != piece:
            raise TokenizerVocabularyError(
                f"{piece!r} resolved to id {token_id}, which is {back!r}; this tokenizer "
                f"does not hold {piece!r}"
            )
        resolved[name] = token_id

    if resolved["pad"] == resolved["whitespace"]:
        raise TokenizerVocabularyError(
            f"{pad_piece!r} and {whitespace_piece!r} are both id {resolved['pad']}"
        )
    if end_of_text_padding_id in (resolved["pad"], resolved["whitespace"]):
        raise TokenizerVocabularyError(
            f"end_of_text_padding_id {end_of_text_padding_id} collides with "
            f"{pad_piece!r} or {whitespace_piece!r}; the three counts would overlap"
        )

    return TextStreamVocabulary(
        pad_id=resolved["pad"],
        whitespace_id=resolved["whitespace"],
        end_of_text_padding_id=end_of_text_padding_id,
        pad_piece=pad_piece,
        whitespace_piece=whitespace_piece,
        end_of_text_padding_piece=tokenizer.id_to_piece(end_of_text_padding_id),
    )


def summarise_text_stream(
    streams: Iterable[Sequence[int]], vocabulary: TextStreamVocabulary
) -> TextStreamCounts:
    """Reduce one speaker's text rows, across a whole split, to counts.

    ``words`` is what is left after the three known markers, so it never disagrees with the
    total by construction.
    """
    stream_count = frames = pad = whitespace = end_of_text_padding = 0
    for stream in streams:
        stream_count += 1
        for token_id in stream:
            frames += 1
            if token_id == vocabulary.pad_id:
                pad += 1
            elif token_id == vocabulary.whitespace_id:
                whitespace += 1
            elif token_id == vocabulary.end_of_text_padding_id:
                end_of_text_padding += 1
    return TextStreamCounts(
        streams=stream_count,
        frames=frames,
        pad=pad,
        bare_whitespace=whitespace,
        end_of_text_padding=end_of_text_padding,
        words=frames - pad - whitespace - end_of_text_padding,
    )


def resolve_tokenize_invocation(argv: Sequence[str]) -> dict[str, Any]:
    """Expand a ``tools/tokenize_text.py`` command line into a complete flag record.

    The point is the expansion. ``--no_whitespace_before_word`` is ``store_true``: a run
    that omits it and a run that was never recorded look identical in a shell history, and
    that is how M3's defect survived. Here every flag is present with its resolved value,
    so omitting it writes ``"no_whitespace_before_word": false`` - a claim a reviewer can
    see and a test can fail on.
    """
    parser = argparse.ArgumentParser(prog="tools/tokenize_text.py", add_help=False)
    for name, default in TOKENIZE_TEXT_FLAGS.items():
        option = f"--{name}"
        if name in STORE_TRUE_FLAGS:
            parser.add_argument(option, action="store_true")
        elif name in REQUIRED_FLAGS:
            parser.add_argument(option, type=str, required=True)
        else:
            parser.add_argument(option, type=type(default), default=default)

    parsed = vars(parser.parse_args(list(argv)))
    given = {token.split("=", 1)[0] for token in argv if token.startswith("--")}
    return {
        "argv": list(argv),
        "flags": {name: parsed[name] for name in TOKENIZE_TEXT_FLAGS},
        "defaults_used": sorted(name for name in TOKENIZE_TEXT_FLAGS if f"--{name}" not in given),
    }


def audit_parquet(
    parquet_path: Path, tokenizer_path: Path, *, end_of_text_padding_id: int
) -> dict[str, Any]:
    """Count the text stream of both speakers in one shipped parquet."""
    import numpy as np
    import pandas as pd
    from sentencepiece import SentencePieceProcessor

    vocabulary = resolve_vocabulary(
        SentencePieceProcessor(str(tokenizer_path)),
        end_of_text_padding_id=end_of_text_padding_id,
    )
    frame = pd.read_parquet(parquet_path)
    missing = [column for column in ("dialogue_id", "A", "B") if column not in frame.columns]
    if missing:
        raise NotADialogueParquetError(
            f"{parquet_path}: no {', '.join(missing)} column; "
            "tools/prepare_dataset.py writes dialogue_id, A and B"
        )

    speakers = {}
    for speaker in ("A", "B"):
        rows = []
        for example in frame[speaker]:
            merged = np.asarray(example.tolist() if hasattr(example, "tolist") else example)
            if merged.ndim != 2:
                raise ValueError(
                    f"{parquet_path}: speaker {speaker} is {merged.ndim}D, expected "
                    "[streams, frames] as written by tools/prepare_dataset.merge_text_audio"
                )
            rows.append(merged[TEXT_ROW])
        speakers[speaker] = summarise_text_stream(rows, vocabulary).as_record()

    return {
        "path": parquet_path.as_posix(),
        "sha256": hashlib.sha256(parquet_path.read_bytes()).hexdigest(),
        "byte_size": parquet_path.stat().st_size,
        "dialogues": int(len(frame)),
        "vocabulary": vocabulary.as_record(),
        "speakers": speakers,
    }


def counterfactual_shares(
    word_transcript_dir: Path,
    dialogue_names: Sequence[str],
    tokenizer_path: Path,
    *,
    text_padding_id: int,
    end_of_text_padding_id: int,
    audio_tokenizer_frame_rate: float,
) -> dict[str, Any]:
    """Re-tokenize the same transcripts both ways, and report both token histograms.

    The calibration a threshold needs cannot be a number somebody typed. This runs
    ``tokenize_and_pad_text`` over the dataset's own word transcripts twice - once with
    ``no_whitespace_before_word`` and once without - so the compliant and non-compliant
    poles are measured on the same sentences, by the same counter, on demand.

    It doubles as the proof of what a shipped dataset was tokenized with: the run whose
    non-padding counts match the parquet is the flag setting that produced it.
    """
    from sentencepiece import SentencePieceProcessor

    from tools.tokenize_text import tokenize_and_pad_text

    tokenizer = SentencePieceProcessor(str(tokenizer_path))
    vocabulary = resolve_vocabulary(tokenizer, end_of_text_padding_id=end_of_text_padding_id)
    if vocabulary.pad_id != text_padding_id:
        raise TokenizerVocabularyError(
            f"--text_padding_id {text_padding_id} is not the id of {vocabulary.pad_piece!r} "
            f"({vocabulary.pad_id}); the audit would count a different token as padding"
        )

    transcripts = []
    for name in dialogue_names:
        transcripts.append(
            json.loads((word_transcript_dir / f"{name}.json").read_text(encoding="utf-8"))
        )

    paths = {}
    for no_whitespace_before_word in (True, False):
        speakers = {}
        for speaker in ("A", "B"):
            streams = []
            for transcript in transcripts:
                # tokenize_and_pad_text rewrites the `word` of every segment in place, so
                # each pass gets its own copy or the second one reads the first one's edits.
                segments = [
                    dict(segment) for segment in transcript if segment["speaker"] == speaker
                ]
                if not segments:
                    continue
                streams.append(
                    tokenize_and_pad_text(
                        word_transcript=segments,
                        no_whitespace_before_word=no_whitespace_before_word,
                        text_tokenizer=tokenizer,
                        text_padding_id=text_padding_id,
                        end_of_text_padding_id=end_of_text_padding_id,
                        audio_tokenizer_frame_rate=audio_tokenizer_frame_rate,
                    )
                )
            speakers[speaker] = summarise_text_stream(streams, vocabulary).as_record()
        key = "with_flag" if no_whitespace_before_word else "without_flag"
        paths[key] = speakers

    return {
        "word_transcript_dir": word_transcript_dir.as_posix(),
        "dialogues": len(transcripts),
        "audio_tokenizer_frame_rate": audio_tokenizer_frame_rate,
        "vocabulary": vocabulary.as_record(),
        "paths": paths,
    }


def write_tokenize_record(
    path: Path,
    *,
    dataset_id: str,
    manifest: str,
    invocations: list[dict[str, Any]],
    provenance: str,
    recorded_at: str,
    append: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """File the flag record beside the manifest it describes.

    A sidecar rather than a column on every manifest row. ``v-real-v1.jsonl`` has eighty
    rows that share one tokenize run, so a per-row copy would be eighty chances to
    disagree, and rewriting those rows would move fields the existing manifest tests pin.
    The precedent is ``tsukuyomi-corpus-v1-extraction.json``, which already records how the
    manifest next to it was produced.
    """
    record = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "manifest": manifest,
        "tool": "tools/tokenize_text.py",
        "provenance": provenance,
        "recorded_at": recorded_at,
        "invocations": invocations,
    }
    if append and path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        for field in ("dataset_id", "manifest", "provenance"):
            if existing.get(field) != record[field]:
                raise ValueError(
                    f"{path}: cannot append, {field} is {existing.get(field)!r} there and "
                    f"{record[field]!r} here"
                )
        record["invocations"] = list(existing.get("invocations", [])) + invocations
        record = {**existing, **record}
    record.update(extra or {})
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record


def _cmd_audit(args: argparse.Namespace) -> None:
    measurements = [
        audit_parquet(
            Path(parquet),
            Path(args.text_tokenizer_path),
            end_of_text_padding_id=args.end_of_text_padding_id,
        )
        for parquet in args.parquet
    ]
    payload = json.dumps(measurements, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


def _cmd_counterfactual(args: argparse.Namespace) -> None:
    names = args.dialogue_names or sorted(
        path.stem for path in Path(args.word_transcript_dir).glob("*.json")
    )
    payload = json.dumps(
        counterfactual_shares(
            Path(args.word_transcript_dir),
            names,
            Path(args.text_tokenizer_path),
            text_padding_id=args.text_padding_id,
            end_of_text_padding_id=args.end_of_text_padding_id,
            audio_tokenizer_frame_rate=args.audio_tokenizer_frame_rate,
        ),
        ensure_ascii=False,
        indent=2,
    )
    if args.out:
        Path(args.out).write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


def _cmd_record_tokenize(args: argparse.Namespace) -> None:
    invocation = resolve_tokenize_invocation(args.tokenize_argv)
    if args.split:
        invocation = {"split": args.split, **invocation}
    record = write_tokenize_record(
        Path(args.out),
        dataset_id=args.dataset_id,
        manifest=args.manifest,
        invocations=[invocation],
        provenance=args.provenance,
        recorded_at=args.recorded_at,
        append=args.append,
    )
    print(json.dumps(record, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="count the text stream of shipped parquet files")
    audit.add_argument("parquet", nargs="+", help="parquet files to audit")
    audit.add_argument(
        "--text_tokenizer_path",
        required=True,
        help="path to the SentencePiece model the dataset was tokenized with",
    )
    audit.add_argument(
        "--end_of_text_padding_id",
        type=int,
        default=DEFAULT_END_OF_TEXT_PADDING_ID,
        help="the --end_of_text_padding_id the dataset was tokenized with",
    )
    audit.add_argument("--out", help="write the JSON here instead of stdout")
    audit.set_defaults(func=_cmd_audit)

    counterfactual = sub.add_parser(
        "counterfactual",
        help="re-tokenize word transcripts with and without --no_whitespace_before_word",
    )
    counterfactual.add_argument("--word_transcript_dir", required=True)
    counterfactual.add_argument("--text_tokenizer_path", required=True)
    counterfactual.add_argument(
        "--dialogue_names",
        nargs="*",
        help="restrict to these stems; defaults to every .json in the directory",
    )
    counterfactual.add_argument("--text_padding_id", type=int, default=3)
    counterfactual.add_argument(
        "--end_of_text_padding_id", type=int, default=DEFAULT_END_OF_TEXT_PADDING_ID
    )
    counterfactual.add_argument("--audio_tokenizer_frame_rate", type=float, default=12.5)
    counterfactual.add_argument("--out", help="write the JSON here instead of stdout")
    counterfactual.set_defaults(func=_cmd_counterfactual)

    record = sub.add_parser(
        "record-tokenize",
        help="expand a tokenize_text.py command line into a manifest sidecar",
    )
    record.add_argument("--dataset_id", required=True)
    record.add_argument("--manifest", required=True, help="repository-relative manifest path")
    record.add_argument("--out", required=True, help="sidecar path to write")
    record.add_argument("--split", default="", help="which split this invocation produced")
    record.add_argument("--provenance", default="recorded", choices=["recorded", "reconstructed"])
    record.add_argument("--recorded_at", required=True)
    record.add_argument(
        "--append",
        action="store_true",
        help="add to an existing sidecar rather than replacing it, for one split at a time",
    )
    record.add_argument(
        "tokenize_argv",
        nargs="*",
        help="the tokenize_text.py argv, after a bare --",
    )
    record.set_defaults(func=_cmd_record_tokenize)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
