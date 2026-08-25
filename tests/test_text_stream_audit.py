import ast
import json
import tempfile
import unittest
from pathlib import Path

from tools.text_stream_audit import (
    TOKENIZE_TEXT_FLAGS,
    TokenizerVocabularyError,
    resolve_tokenize_invocation,
    resolve_vocabulary,
    summarise_text_stream,
    write_tokenize_record,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class FakeTokenizer:
    """A SentencePiece stand-in, including the part that makes the audit tricky.

    `piece_to_id` returns the unknown id for a piece the model does not hold, rather than
    raising. Every test of `resolve_vocabulary` depends on reproducing that.
    """

    def __init__(self, pieces: list[str]) -> None:
        self._pieces = list(pieces)

    def piece_to_id(self, piece: str) -> int:
        return self._pieces.index(piece) if piece in self._pieces else 0

    def id_to_piece(self, token_id: int) -> str:
        return self._pieces[token_id]


def j_moshi_like() -> FakeTokenizer:
    """`<unk>` at 0, `[PAD]` at 3, `U+2581` at 9, as tokenizer_spm_32k_3.model has them."""
    return FakeTokenizer(["<unk>", "<s>", "</s>", "[PAD]", "[CLS]", "あ", "い", "う", "え", "▁"])


class VocabularyResolutionTests(unittest.TestCase):
    """The audit counts three ids. Guessing any of them would miscount in silence."""

    def test_ids_come_from_the_tokenizer(self) -> None:
        vocabulary = resolve_vocabulary(j_moshi_like())

        self.assertEqual(vocabulary.pad_id, 3)
        self.assertEqual(vocabulary.whitespace_id, 9)
        self.assertEqual(vocabulary.end_of_text_padding_id, 0)
        self.assertEqual(vocabulary.end_of_text_padding_piece, "<unk>")

    def test_a_tokenizer_without_the_whitespace_piece_raises(self) -> None:
        """`piece_to_id` would return 0 here, and 0 is the end-of-text marker.

        Without the round-trip check the audit would count every end-of-text marker as a
        bare whitespace token and report a broken dataset as clean, or the reverse.
        """
        without_whitespace = FakeTokenizer(["<unk>", "<s>", "</s>", "[PAD]"])

        with self.assertRaises(TokenizerVocabularyError):
            resolve_vocabulary(without_whitespace)

    def test_a_tokenizer_without_the_padding_piece_raises(self) -> None:
        with self.assertRaises(TokenizerVocabularyError):
            resolve_vocabulary(FakeTokenizer(["<unk>", "<s>", "</s>", "x", "▁"]))

    def test_an_end_of_text_id_that_collides_with_padding_raises(self) -> None:
        """The three counts partition the stream, so they may not overlap."""
        with self.assertRaises(TokenizerVocabularyError):
            resolve_vocabulary(j_moshi_like(), end_of_text_padding_id=3)


class TextStreamCountTests(unittest.TestCase):
    def _counts(self, *streams: list[int]):
        return summarise_text_stream(streams, resolve_vocabulary(j_moshi_like()))

    def test_every_frame_lands_in_exactly_one_bucket(self) -> None:
        counts = self._counts([3, 3, 0, 9, 5, 3], [3, 9, 6])

        self.assertEqual(counts.frames, 9)
        self.assertEqual(counts.pad, 4)
        self.assertEqual(counts.end_of_text_padding, 1)
        self.assertEqual(counts.bare_whitespace, 2)
        self.assertEqual(counts.words, 2)
        self.assertEqual(
            counts.pad + counts.end_of_text_padding + counts.bare_whitespace + counts.words,
            counts.frames,
        )

    def test_the_gated_share_excludes_padding_of_both_kinds(self) -> None:
        """Two whitespace markers and two words is half, however much padding surrounds it.

        Measured against non-pad instead, the same stream reads 2/5. That is the difference
        between the 32.1% the shipped parquet shows against non-pad and the 44.6% it shows
        against emitted tokens, and only the second compares to the with-flag pole.
        """
        counts = self._counts([3, 3, 3, 0, 9, 5, 9, 6])

        self.assertEqual(counts.text_tokens, 4)
        self.assertEqual(counts.bare_whitespace_share_of_text_tokens, 0.5)
        self.assertEqual(counts.bare_whitespace_share_of_non_pad, 0.4)

    def test_an_empty_stream_does_not_divide_by_zero(self) -> None:
        counts = self._counts([3, 3, 3])

        self.assertEqual(counts.text_tokens, 0)
        self.assertEqual(counts.bare_whitespace_share_of_text_tokens, 0.0)
        self.assertEqual(counts.bare_whitespace_share_of_non_pad, 0.0)

    def test_streams_are_counted(self) -> None:
        self.assertEqual(self._counts([3], [3], [3]).streams, 3)


class TokenizeInvocationRecordTests(unittest.TestCase):
    """A dropped `store_true` flag leaves no trace. That is how M3's defect survived."""

    def _argv(self, *extra: str) -> list[str]:
        return ["--word_transcript_dir", "in", "--output_dir", "out", *extra]

    def test_omitting_the_whitespace_flag_records_it_as_false(self) -> None:
        record = resolve_tokenize_invocation(self._argv())

        self.assertIs(record["flags"]["no_whitespace_before_word"], False)
        self.assertIn("no_whitespace_before_word", record["defaults_used"])

    def test_passing_the_whitespace_flag_records_it_as_true(self) -> None:
        record = resolve_tokenize_invocation(self._argv("--no_whitespace_before_word"))

        self.assertIs(record["flags"]["no_whitespace_before_word"], True)
        self.assertNotIn("no_whitespace_before_word", record["defaults_used"])

    def test_every_flag_appears_with_its_resolved_value(self) -> None:
        record = resolve_tokenize_invocation(self._argv())

        self.assertEqual(set(record["flags"]), set(TOKENIZE_TEXT_FLAGS))
        self.assertEqual(record["flags"]["text_padding_id"], 3)
        self.assertEqual(record["flags"]["audio_tokenizer_frame_rate"], 12.5)

    def test_a_fractional_frame_rate_survives_the_record(self) -> None:
        """The record must not repeat the int-coercion bug it exists to make visible."""
        record = resolve_tokenize_invocation(self._argv("--audio_tokenizer_frame_rate", "12.5"))

        self.assertEqual(record["flags"]["audio_tokenizer_frame_rate"], 12.5)
        self.assertNotIn("audio_tokenizer_frame_rate", record["defaults_used"])

    def test_the_equals_form_counts_as_given(self) -> None:
        record = resolve_tokenize_invocation(self._argv("--num_workers=4"))

        self.assertEqual(record["flags"]["num_workers"], 4)
        self.assertNotIn("num_workers", record["defaults_used"])

    def test_the_argv_is_kept_verbatim(self) -> None:
        argv = self._argv("--text_tokenizer_repo", "nu-dialogue/j-moshi-ext")

        self.assertEqual(resolve_tokenize_invocation(argv)["argv"], argv)


class TokenizeRecordFileTests(unittest.TestCase):
    def _write(self, path: Path, *, dataset_id: str = "d-v1", append: bool = False, split: str):
        return write_tokenize_record(
            path,
            dataset_id=dataset_id,
            manifest="manifests/d-v1.jsonl",
            invocations=[
                {
                    "split": split,
                    **resolve_tokenize_invocation(
                        ["--word_transcript_dir", "in", "--output_dir", "out"]
                    ),
                }
            ],
            provenance="recorded",
            recorded_at="2026-08-25",
            append=append,
        )

    def test_appending_keeps_the_earlier_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "d-v1-tokenize.json"
            self._write(path, split="train")
            record = self._write(path, split="dev", append=True)

            self.assertEqual([i["split"] for i in record["invocations"]], ["train", "dev"])
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), record)

    def test_writing_without_append_replaces(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "d-v1-tokenize.json"
            self._write(path, split="train")
            record = self._write(path, split="dev")

            self.assertEqual([i["split"] for i in record["invocations"]], ["dev"])

    def test_appending_to_another_dataset_raises(self) -> None:
        """Two datasets sharing a sidecar would attribute one's flags to the other."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "d-v1-tokenize.json"
            self._write(path, split="train")

            with self.assertRaises(ValueError):
                self._write(path, dataset_id="other-v1", split="dev", append=True)


class TokenizeTextFrameRateTests(unittest.TestCase):
    """`--audio_tokenizer_frame_rate` was declared `type=int` with a default of `12.5`.

    Mimi runs at 12.5 Hz. Under `type=int` the documented value cannot even be passed -
    `int("12.5")` raises - and `12` is accepted without complaint, which slides the text
    stream one frame further from the audio every two seconds. No artifact shows it: the
    parquet is well formed and the training loop starts. M3 escaped it only because the
    flag was never passed and argparse leaves a default alone.
    """

    def _frame_rate_argument(self) -> ast.Call:
        source = (REPOSITORY_ROOT / "tools" / "tokenize_text.py").read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call):
                continue
            if getattr(node.func, "attr", None) != "add_argument":
                continue
            if node.args and getattr(node.args[0], "value", None) == "--audio_tokenizer_frame_rate":
                return node
        raise AssertionError("tokenize_text.py no longer declares --audio_tokenizer_frame_rate")

    def _add_argument_calls(self) -> list[ast.Call]:
        source = (REPOSITORY_ROOT / "tools" / "tokenize_text.py").read_text(encoding="utf-8")
        return [
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call)
            and getattr(node.func, "attr", None) == "add_argument"
            and node.args
            and isinstance(getattr(node.args[0], "value", None), str)
        ]

    def test_the_recorded_flag_table_still_matches_tokenize_text(self) -> None:
        """`TOKENIZE_TEXT_FLAGS` is a copy, and a copy can go stale.

        tokenize_text.py builds its parser inside `if __name__ == "__main__"`, so it cannot
        be imported and the flag table has to be duplicated. A flag added there and not
        here would be omitted from every future record while the record still claimed to be
        complete - which is the exact failure the record exists to prevent.
        """
        declared = {call.args[0].value.removeprefix("--") for call in self._add_argument_calls()}

        self.assertEqual(declared, set(TOKENIZE_TEXT_FLAGS))

    def test_the_recorded_defaults_still_match_tokenize_text(self) -> None:
        for call in self._add_argument_calls():
            name = call.args[0].value.removeprefix("--")
            keywords = {kw.arg: kw.value for kw in call.keywords}
            if "default" not in keywords:
                continue
            self.assertEqual(
                keywords["default"].value,
                TOKENIZE_TEXT_FLAGS[name],
                f"--{name}: the recorded default no longer matches tokenize_text.py",
            )

    def test_the_flag_is_parsed_as_a_float(self) -> None:
        keywords = {kw.arg: kw.value for kw in self._frame_rate_argument().keywords}

        self.assertEqual(getattr(keywords["type"], "id", None), "float")
        self.assertEqual(keywords["default"].value, 12.5)

    def test_a_fractional_frame_rate_changes_the_stream_length(self) -> None:
        """The behavioural half: 12.5 Hz and 12 Hz do not produce the same stream.

        Without this, `type=float` could be reverted and the AST check patched to match
        without anything measuring what the difference costs.
        """
        try:
            from tools.tokenize_text import tokenize_and_pad_text
        except ImportError as error:  # numpy / sentencepiece / tqdm are not in the test env
            self.skipTest(f"tokenize_text dependencies are absent: {error}")

        class OneWordTokenizer:
            def encode_as_pieces(self, text: str) -> list[str]:
                return [text]

            def piece_to_id(self, piece: str) -> int:
                return 5

        def transcript() -> list[dict]:
            return [{"speaker": "A", "word": "あい", "start": 0.0, "end": 0.8}]

        at_twelve_and_a_half = tokenize_and_pad_text(
            word_transcript=transcript(),
            no_whitespace_before_word=True,
            text_tokenizer=OneWordTokenizer(),
            text_padding_id=3,
            end_of_text_padding_id=0,
            audio_tokenizer_frame_rate=12.5,
        )
        at_twelve = tokenize_and_pad_text(
            word_transcript=transcript(),
            no_whitespace_before_word=True,
            text_tokenizer=OneWordTokenizer(),
            text_padding_id=3,
            end_of_text_padding_id=0,
            audio_tokenizer_frame_rate=12,
        )

        self.assertEqual(len(at_twelve_and_a_half), 22)
        self.assertEqual(len(at_twelve), 21)


if __name__ == "__main__":
    unittest.main()
