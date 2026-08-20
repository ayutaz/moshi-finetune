import unittest

from tools.tts_intelligibility import (
    character_error_rate,
    normalise_for_cer,
    summarise_intelligibility,
)


class NormalisationTests(unittest.TestCase):
    """ASR output will not reproduce the punctuation a reading script carries.

    Comparing raw strings would charge the TTS for commas the recogniser never emits, so
    both sides are reduced to the characters that carry the reading.
    """

    def test_drops_japanese_and_ascii_punctuation(self) -> None:
        self.assertEqual(normalise_for_cer("朝露に、きらめく。"), "朝露にきらめく")

    def test_drops_whitespace(self) -> None:
        self.assertEqual(normalise_for_cer("こんにちは 世界\n"), "こんにちは世界")

    def test_keeps_kana_kanji_and_digits(self) -> None:
        self.assertEqual(normalise_for_cer("九時45分ティー"), "九時45分ティー")


class CharacterErrorRateTests(unittest.TestCase):
    def test_identical_strings_score_zero(self) -> None:
        self.assertEqual(character_error_rate("こんにちは", "こんにちは"), 0.0)

    def test_counts_a_substitution(self) -> None:
        self.assertAlmostEqual(character_error_rate("こんにちは", "こんにちわ"), 0.2)

    def test_counts_a_deletion(self) -> None:
        self.assertAlmostEqual(character_error_rate("こんにちは", "こんにち"), 0.2)

    def test_counts_an_insertion(self) -> None:
        self.assertAlmostEqual(character_error_rate("こんにち", "こんにちは"), 0.25)

    def test_an_empty_hypothesis_scores_one(self) -> None:
        self.assertEqual(character_error_rate("こんにちは", ""), 1.0)

    def test_rejects_an_empty_reference(self) -> None:
        with self.assertRaisesRegex(ValueError, "reference"):
            character_error_rate("", "こんにちは")

    def test_normalises_before_comparing(self) -> None:
        self.assertEqual(character_error_rate("朝露に、きらめく。", "朝露にきらめく"), 0.0)


class SummaryTests(unittest.TestCase):
    """The gate asks for 27 of 30 sentences to be clearly intelligible."""

    def test_counts_files_under_the_threshold(self) -> None:
        rows = [{"id": "a", "cer": 0.0}, {"id": "b", "cer": 0.5}, {"id": "c", "cer": 0.1}]

        summary = summarise_intelligibility(rows, threshold=0.15)

        self.assertEqual(summary["intelligible"], 2)
        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["failed_ids"], ["b"])

    def test_reports_mean_and_median(self) -> None:
        rows = [{"id": "a", "cer": 0.0}, {"id": "b", "cer": 0.2}, {"id": "c", "cer": 0.4}]

        summary = summarise_intelligibility(rows, threshold=0.5)

        self.assertAlmostEqual(summary["mean_cer"], 0.2)
        self.assertAlmostEqual(summary["median_cer"], 0.2)

    def test_a_threshold_on_the_boundary_counts_as_intelligible(self) -> None:
        summary = summarise_intelligibility([{"id": "a", "cer": 0.15}], threshold=0.15)

        self.assertEqual(summary["intelligible"], 1)

    def test_rejects_an_empty_set(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            summarise_intelligibility([], threshold=0.15)


if __name__ == "__main__":
    unittest.main()
