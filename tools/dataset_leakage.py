"""Prove that the held-out recordings are not in the training data, three ways.

Why three ways
--------------
The held-out set is the ten corpus `test` recordings. Condition 4 of
`m3/DATASET_SPEC.md` judges the whole experiment on how a checkpoint scores against them,
so a single one of them leaking into training makes the number meaningless while leaving
every other gate green.

One check is not enough because each has a blind spot the next covers:

1. **By artifact id.** Every dialogue names the corpus sentence its speaker A quotes. None
   may be a corpus `dev` or `test` row. Blind to a file that was renamed or re-derived from
   a held-out sentence - the id would be innocent and the audio would not.
2. **By text, on the transcripts that were tokenized.** A held-out sentence's own words may
   not appear in the word transcripts. Blind to a transcript that disagrees with what was
   tokenized.
3. **By text, on the decoded parquet.** The text stream decoded back through SentencePiece.
   This is the artifact the model actually reads, and it is downstream of everything.

Checks 2 and 3 report the *longest run* a dialogue shares with any held-out sentence rather
than a yes/no. A boolean answers "did a whole sentence leak"; the run length also answers
"how close did anything get", and a number that has been 5 for two builds is evidence in a
way that `false` is not.

Normalisation
-------------
Comparison runs on `tools.experiment_data._normalise_text` - NFKC, casefolded, everything
that is not alphanumeric removed - deliberately the same normaliser
`tools/memorisation.py` borrows for the same reason it gives: two normalisers that disagree
about what "the same text" means would let a sentence be a leak to one tool and original to
the other. Dropping punctuation and whitespace can only bring two strings closer together,
so a leak cannot hide behind different punctuation.

Everything here is pure except the loaders at the bottom; the parquet path imports pandas
and sentencepiece inside the function that needs them.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Shared on purpose. See the module docstring.
from tools.experiment_data import _normalise_text as normalise

DEFAULT_HELD_OUT_SPLIT = "test"


class LeakageError(ValueError):
    """The inputs cannot support the check that was asked for."""


@dataclass(frozen=True)
class RunMatch:
    """The closest any document came to any held-out sentence."""

    chars: int
    document: str | None
    held_out: str | None

    def as_record(self) -> dict[str, Any]:
        return {
            "longest_shared_normalised_run_chars": self.chars,
            "where": self.document,
            "against": self.held_out,
        }


def longest_shared_run(needle: str, haystack: str) -> int:
    """Longest run of `needle` that occurs contiguously in `haystack`, in characters.

    Both are expected to be normalised already; this counts characters, it does not decide
    what a character is.

    The search walks lengths upward and stops at the first miss. That is exact, not an
    approximation: every window of length n+1 contains a window of length n, so if no
    window of length n occurs then none of length n+1 can.
    """
    if not needle or not haystack:
        return 0
    best = 0
    for length in range(1, min(len(needle), len(haystack)) + 1):
        window_found = any(
            needle[start : start + length] in haystack for start in range(len(needle) - length + 1)
        )
        if not window_found:
            return best
        best = length
    return best


def held_out_texts(
    corpus_rows: Iterable[Mapping[str, Any]],
    *,
    split: str = DEFAULT_HELD_OUT_SPLIT,
    stems: Sequence[str] | None = None,
) -> dict[str, str]:
    """The held-out sentences, keyed by corpus artifact id.

    `stems` narrows the set to recordings that exist on disk - the ten wav files the
    baseline prompts were built from. Passing it and getting fewer rows back than stems is
    an error rather than a smaller set: a stem with no corpus row is a recording nobody can
    say the split of.
    """
    by_stem: dict[str, tuple[str, Mapping[str, Any]]] = {}
    selected: dict[str, str] = {}
    for row in corpus_rows:
        artifact_id = row.get("artifact_id")
        if not artifact_id:
            continue
        stem = str(artifact_id).split(":", 1)[-1]
        by_stem[stem] = (str(artifact_id), row)
        if stems is None and row.get("split") == split:
            selected[str(artifact_id)] = str(row.get("text", ""))

    if stems is None:
        return selected

    missing = [stem for stem in stems if stem not in by_stem]
    if missing:
        raise LeakageError(f"no corpus row for {', '.join(missing)}")
    wrong_split = [stem for stem in stems if by_stem[stem][1].get("split") != split]
    if wrong_split:
        raise LeakageError(
            f"{', '.join(wrong_split)} are held out on disk but the corpus does not put "
            f"them in the {split!r} split"
        )
    return {by_stem[stem][0]: str(by_stem[stem][1].get("text", "")) for stem in stems}


def source_id_overlap(
    dialogues: Iterable[Mapping[str, Any]],
    corpus_splits: Mapping[str, str],
    *,
    training_split: str = "train",
    held_out_ids: Iterable[str] = (),
) -> dict[str, Any]:
    """Which corpus sentences the dialogues quote, and whether any is off limits.

    Two answers, not one. `sources_that_are_held_out` is the leak; `sources_outside_train`
    is the wider rule `m3/DATASET_SPEC.md` states - corpus dev sentences may not enter
    either dataset - and a dev sentence is not held out but is still a rule broken.
    """
    held_out = set(held_out_ids)
    sources = sorted(
        {
            str(dialogue["source_artifact_id"])
            for dialogue in dialogues
            if dialogue.get("source_artifact_id")
        }
    )
    unknown = [source for source in sources if source not in corpus_splits]
    return {
        "dialogue_sources": sources,
        "sources_not_in_the_corpus": unknown,
        "sources_outside_train": sorted(
            source
            for source in sources
            if source not in unknown and corpus_splits[source] != training_split
        ),
        "sources_that_are_held_out": sorted(held_out.intersection(sources)),
    }


def text_overlap(documents: Mapping[str, str], held_out: Mapping[str, str]) -> dict[str, Any]:
    """Compare every document against every held-out sentence, on normalised text.

    Returns the worst run seen and every document that contains a whole held-out sentence.
    `whole_sentence_hits` is what fails the gate; the run length is what makes a passing
    result readable.
    """
    if not held_out:
        raise LeakageError("no held-out sentences to compare against")
    needles = {artifact_id: normalise(text) for artifact_id, text in held_out.items() if text}
    if not needles:
        raise LeakageError("every held-out sentence normalised to nothing")

    worst = RunMatch(0, None, None)
    hits: list[dict[str, Any]] = []
    for name, text in documents.items():
        haystack = normalise(text)
        for artifact_id, needle in needles.items():
            if needle in haystack:
                hits.append({"document": name, "held_out": artifact_id, "match": "whole"})
            run = longest_shared_run(needle, haystack)
            if run > worst.chars:
                worst = RunMatch(run, name, artifact_id)
    return {
        "documents": len(documents),
        "held_out_sentences": len(needles),
        "whole_sentence_hits": hits,
        "shortest_held_out_sentence_chars": min(len(needle) for needle in needles.values()),
        **worst.as_record(),
    }


def transcript_agreement(
    transcripts: Mapping[str, Sequence[Mapping[str, Any]]],
    scripts: Mapping[str, Mapping[str, Any]],
    *,
    speakers: Sequence[str] = ("A", "B"),
) -> dict[str, Any]:
    """Check each word transcript against the script it was built from, per speaker.

    Per speaker rather than per dialogue: a transcript that puts speaker A's words on
    channel B agrees with the script as a whole and is a dataset where the target voice is
    the user stream.
    """
    mismatches: list[dict[str, Any]] = []
    for name, segments in sorted(transcripts.items()):
        script = scripts.get(name)
        if script is None:
            mismatches.append({"dialogue": name, "problem": "no script"})
            continue
        for speaker in speakers:
            got = normalise(
                "".join(
                    str(segment["word"])
                    for segment in segments
                    if segment.get("speaker") == speaker
                )
            )
            want = normalise(
                "".join(
                    str(turn["text"])
                    for turn in script.get("turns", [])
                    if turn.get("speaker") == speaker
                )
            )
            if got != want:
                mismatches.append(
                    {
                        "dialogue": name,
                        "speaker": speaker,
                        "transcript": got[:80],
                        "script": want[:80],
                    }
                )
    return {"dialogues": len(transcripts), "mismatches": mismatches}


# --------------------------------------------------------------------------------------
# I/O.
# --------------------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def dialogue_documents(dialogues: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    """One document per dialogue script: every turn's text, joined."""
    return {
        str(dialogue["dialogue_id"]): "".join(
            str(turn["text"]) for turn in dialogue.get("turns", [])
        )
        for dialogue in dialogues
    }


def word_transcript_documents(directories: Iterable[Path]) -> dict[str, str]:
    """One document per word transcript, named `<split>/<dialogue>`."""
    documents: dict[str, str] = {}
    for directory in directories:
        for path in sorted(directory.glob("*.json")):
            segments = json.loads(path.read_text(encoding="utf-8"))
            documents[f"{directory.parent.name}/{path.stem}"] = "".join(
                str(segment["word"]) for segment in segments
            )
    return documents


def parquet_documents(
    parquet_paths: Iterable[Path],
    tokenizer_path: Path,
    *,
    text_padding_id: int = 3,
    end_of_text_padding_id: int = 0,
    speakers: Sequence[str] = ("A", "B"),
) -> dict[str, str]:
    """Decode each parquet row's text stream back to text, one document per row.

    This is the only check downstream of tokenization, so it is the only one that sees a
    transcript that was built from something other than the script.
    """
    import numpy as np
    import pandas as pd
    from sentencepiece import SentencePieceProcessor

    from tools.text_stream_audit import TEXT_ROW

    tokenizer = SentencePieceProcessor(str(tokenizer_path))
    skip = {text_padding_id, end_of_text_padding_id}
    documents: dict[str, str] = {}
    for path in parquet_paths:
        frame = pd.read_parquet(path)
        for _, row in frame.iterrows():
            decoded = []
            for speaker in speakers:
                cell = row[speaker]
                merged = np.asarray(cell.tolist() if hasattr(cell, "tolist") else cell)
                ids = [int(token) for token in merged[TEXT_ROW] if int(token) not in skip]
                decoded.append(tokenizer.decode(ids))
            documents[str(row["dialogue_id"])] = "".join(decoded)
    return documents


def _cmd_check(args: argparse.Namespace) -> int:
    corpus = read_jsonl(Path(args.corpus_manifest))
    corpus_splits = {str(row["artifact_id"]): str(row.get("split", "")) for row in corpus}
    stems = None
    if args.held_out_dir:
        stems = sorted(path.stem for path in Path(args.held_out_dir).glob("*.wav"))
        if not stems:
            raise LeakageError(f"{args.held_out_dir} holds no wav files")
    held_out = held_out_texts(corpus, split=args.held_out_split, stems=stems)

    dialogues = read_jsonl(Path(args.dialogues))
    result: dict[str, Any] = {
        "held_out_split": args.held_out_split,
        "held_out_ids": sorted(held_out),
        "by_artifact_id": source_id_overlap(dialogues, corpus_splits, held_out_ids=held_out),
        "by_text": {"dialogue_scripts": text_overlap(dialogue_documents(dialogues), held_out)},
    }
    if args.word_transcript_dir:
        directories = [Path(directory) for directory in args.word_transcript_dir]
        result["by_text"]["word_transcripts"] = text_overlap(
            word_transcript_documents(directories), held_out
        )
        if args.scripts_agree:
            segments_by_dialogue = {
                path.stem: json.loads(path.read_text(encoding="utf-8"))
                for directory in directories
                for path in sorted(directory.glob("*.json"))
            }
            result["transcript_matches_script"] = transcript_agreement(
                segments_by_dialogue,
                {str(dialogue["dialogue_id"]): dialogue for dialogue in dialogues},
            )
    if args.parquet:
        documents = parquet_documents(
            [Path(path) for path in args.parquet],
            Path(args.text_tokenizer_path),
            text_padding_id=args.text_padding_id,
            end_of_text_padding_id=args.end_of_text_padding_id,
        )
        result["by_text"]["decoded_parquet_text_stream"] = text_overlap(documents, held_out)

    # `sources_outside_train` counts too: DATASET_SPEC.md forbids corpus dev sentences as
    # well as test ones, and a dev sentence that slipped in is a rule broken even though it
    # is not the held-out set.
    failures = {
        "sources_that_are_held_out": result["by_artifact_id"]["sources_that_are_held_out"],
        "sources_not_in_the_corpus": result["by_artifact_id"]["sources_not_in_the_corpus"],
        "sources_outside_train": result["by_artifact_id"]["sources_outside_train"],
        "whole_sentence_hits": [
            hit for block in result["by_text"].values() for hit in block["whole_sentence_hits"]
        ],
        "transcript_mismatches": result.get("transcript_matches_script", {}).get("mismatches", []),
    }
    passed = not any(failures.values())
    result["failures"] = failures
    result["passed"] = passed
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if passed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="held-out leakage, by id and by text")
    check.add_argument("--corpus_manifest", required=True)
    check.add_argument("--dialogues", required=True, help="the dialogue script jsonl")
    check.add_argument("--held_out_split", default=DEFAULT_HELD_OUT_SPLIT)
    check.add_argument(
        "--held_out_dir",
        help="narrow the held-out set to the wav stems in this directory",
    )
    check.add_argument(
        "--word_transcript_dir",
        nargs="*",
        default=[],
        help="the <split>/text directories that were tokenized",
    )
    check.add_argument(
        "--scripts_agree",
        action="store_true",
        help="also check each word transcript against its script, per speaker",
    )
    check.add_argument("--parquet", nargs="*", default=[], help="shipped parquet files")
    check.add_argument("--text_tokenizer_path", help="required with --parquet")
    check.add_argument("--text_padding_id", type=int, default=3)
    check.add_argument("--end_of_text_padding_id", type=int, default=0)
    check.add_argument("--out", help="write the JSON here as well as to stdout")
    check.set_defaults(func=_cmd_check)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "parquet", None) and not args.text_tokenizer_path:
        raise SystemExit("--parquet needs --text_tokenizer_path")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
