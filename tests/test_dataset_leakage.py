import unittest

from tools.dataset_leakage import (
    LeakageError,
    held_out_texts,
    longest_shared_run,
    normalise,
    source_id_overlap,
    text_overlap,
    transcript_agreement,
)


class LongestSharedRunTests(unittest.TestCase):
    """The measurement behind "the longest run any dialogue shares is 5 characters".

    A boolean answers "did a whole sentence leak". The run length also answers "how close
    did anything get", which is what makes a passing result evidence rather than an absence.
    """

    def test_no_overlap_is_zero(self) -> None:
        self.assertEqual(longest_shared_run("あいうえお", "かきくけこ"), 0)

    def test_a_whole_needle_scores_its_length(self) -> None:
        self.assertEqual(longest_shared_run("あいう", "ぜんぶあいうはいってる"), 3)

    def test_the_longest_contiguous_run_wins_not_the_total(self) -> None:
        """Two separated matches of 2 are not a run of 4."""
        self.assertEqual(longest_shared_run("あいXXうえ", "あいYYうえ"), 2)

    def test_an_empty_side_is_zero(self) -> None:
        self.assertEqual(longest_shared_run("", "あいうえお"), 0)
        self.assertEqual(longest_shared_run("あいうえお", ""), 0)

    def test_the_upward_walk_does_not_stop_early(self) -> None:
        """Every window of length n+1 contains one of length n, so stopping at the first
        miss is exact. A needle whose long run appears late in the string pins it."""
        self.assertEqual(longest_shared_run("XXXあいうえおか", "まったく別のあいうえおか"), 6)


class NormalisationTests(unittest.TestCase):
    """Dropping punctuation can only bring two strings closer, never further apart."""

    def test_punctuation_and_spacing_do_not_hide_a_leak(self) -> None:
        sentence = "また、東寺のように、五大明王と呼ばれる。"
        repunctuated = "また 東寺のように 五大明王と呼ばれる"

        self.assertEqual(normalise(sentence), normalise(repunctuated))

    def test_width_is_normalised(self) -> None:
        self.assertEqual(normalise("ABC１２３"), normalise("ＡＢＣ123"))


CORPUS = [
    {"artifact_id": "c:001", "split": "train", "text": "訓練の文です。"},
    {"artifact_id": "c:002", "split": "dev", "text": "開発の文です。"},
    {"artifact_id": "c:003", "split": "test", "text": "評価のためだけの長い文章です。"},
    {"artifact_id": "c:004", "split": "test", "text": "もうひとつの評価専用の文章です。"},
]


class HeldOutTextTests(unittest.TestCase):
    def test_the_split_selects_the_held_out_set(self) -> None:
        self.assertEqual(sorted(held_out_texts(CORPUS)), ["c:003", "c:004"])

    def test_stems_narrow_the_set_to_what_is_on_disk(self) -> None:
        self.assertEqual(sorted(held_out_texts(CORPUS, stems=["003"])), ["c:003"])

    def test_a_stem_with_no_corpus_row_is_refused(self) -> None:
        """A recording nobody can say the split of is not a held-out set."""
        with self.assertRaises(LeakageError):
            held_out_texts(CORPUS, stems=["999"])

    def test_a_stem_the_corpus_puts_in_train_is_refused(self) -> None:
        with self.assertRaises(LeakageError) as raised:
            held_out_texts(CORPUS, stems=["001"])

        self.assertIn("does not put them in", str(raised.exception))


class SourceIdOverlapTests(unittest.TestCase):
    def _splits(self):
        return {row["artifact_id"]: row["split"] for row in CORPUS}

    def test_a_held_out_source_is_named(self) -> None:
        result = source_id_overlap(
            [{"source_artifact_id": "c:003"}], self._splits(), held_out_ids=["c:003", "c:004"]
        )

        self.assertEqual(result["sources_that_are_held_out"], ["c:003"])

    def test_a_dev_source_is_a_rule_broken_even_though_it_is_not_held_out(self) -> None:
        """DATASET_SPEC.md forbids corpus dev sentences in either dataset."""
        result = source_id_overlap(
            [{"source_artifact_id": "c:002"}], self._splits(), held_out_ids=["c:003"]
        )

        self.assertEqual(result["sources_that_are_held_out"], [])
        self.assertEqual(result["sources_outside_train"], ["c:002"])

    def test_a_source_the_corpus_does_not_hold_is_named_separately(self) -> None:
        """Otherwise it would be silently counted as inside the train split."""
        result = source_id_overlap([{"source_artifact_id": "c:999"}], self._splits())

        self.assertEqual(result["sources_not_in_the_corpus"], ["c:999"])
        self.assertEqual(result["sources_outside_train"], [])

    def test_a_clean_set_reports_nothing(self) -> None:
        result = source_id_overlap(
            [{"source_artifact_id": "c:001"}], self._splits(), held_out_ids=["c:003"]
        )

        self.assertEqual(result["sources_that_are_held_out"], [])
        self.assertEqual(result["sources_outside_train"], [])


class TextOverlapTests(unittest.TestCase):
    def _held_out(self):
        return {row["artifact_id"]: row["text"] for row in CORPUS if row["split"] == "test"}

    def test_a_clean_corpus_reports_a_short_run(self) -> None:
        result = text_overlap({"v-001": "まったく無関係な内容。"}, self._held_out())

        self.assertEqual(result["whole_sentence_hits"], [])
        self.assertLess(result["longest_shared_normalised_run_chars"], 5)

    def test_a_whole_sentence_that_leaked_is_caught(self) -> None:
        result = text_overlap(
            {"v-001": "前置き。評価のためだけの長い文章です。後置き。"}, self._held_out()
        )

        self.assertEqual([hit["held_out"] for hit in result["whole_sentence_hits"]], ["c:003"])

    def test_a_leak_hiding_behind_different_punctuation_is_caught(self) -> None:
        result = text_overlap({"v-001": "評価のためだけの、長い文章です！"}, self._held_out())

        self.assertEqual(len(result["whole_sentence_hits"]), 1)

    def test_the_worst_match_names_both_sides(self) -> None:
        result = text_overlap(
            {"v-001": "無関係。", "v-002": "評価のためだけの長い文章です。"}, self._held_out()
        )

        self.assertEqual(result["where"], "v-002")
        self.assertEqual(result["against"], "c:003")

    def test_no_held_out_sentences_is_refused(self) -> None:
        """An empty needle set would make every dataset pass."""
        with self.assertRaises(LeakageError):
            text_overlap({"v-001": "なにか。"}, {})


class TranscriptAgreementTests(unittest.TestCase):
    """Per speaker, not per dialogue.

    A transcript that puts speaker A's words on channel B agrees with the script as a whole
    and is a dataset in which the target voice is the user stream - the failure
    `m3/DATASET_SPEC.md` says is visible nowhere else.
    """

    def _script(self):
        return {
            "v-001": {
                "turns": [
                    {"speaker": "B", "text": "こんにちは。"},
                    {"speaker": "A", "text": "はい、そうです。"},
                ]
            }
        }

    def test_an_agreeing_transcript_has_no_mismatches(self) -> None:
        transcripts = {
            "v-001": [
                {"speaker": "B", "word": "こんにちは"},
                {"speaker": "A", "word": "はい"},
                {"speaker": "A", "word": "そうです"},
            ]
        }

        self.assertEqual(transcript_agreement(transcripts, self._script())["mismatches"], [])

    def test_swapped_speakers_are_caught(self) -> None:
        transcripts = {
            "v-001": [
                {"speaker": "A", "word": "こんにちは"},
                {"speaker": "B", "word": "はいそうです"},
            ]
        }

        mismatches = transcript_agreement(transcripts, self._script())["mismatches"]

        self.assertEqual(sorted(m["speaker"] for m in mismatches), ["A", "B"])

    def test_a_dropped_word_is_caught(self) -> None:
        transcripts = {
            "v-001": [
                {"speaker": "B", "word": "こんにちは"},
                {"speaker": "A", "word": "はい"},
            ]
        }

        mismatches = transcript_agreement(transcripts, self._script())["mismatches"]

        self.assertEqual([m["speaker"] for m in mismatches], ["A"])

    def test_a_transcript_with_no_script_is_reported(self) -> None:
        mismatches = transcript_agreement({"v-999": []}, self._script())["mismatches"]

        self.assertEqual(mismatches, [{"dialogue": "v-999", "problem": "no script"}])


if __name__ == "__main__":
    unittest.main()
