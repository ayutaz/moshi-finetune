import unittest

from tools.dialogue_collapse import (
    CollapseThresholds,
    distinct_ratio,
    emitted_text_ratio,
    emitted_text_tokens,
    longest_repeated_ngram,
    summarise_generation,
    verdict_for,
)

PADDING = 3
END_PADDING = 0


class EmittedTextTests(unittest.TestCase):
    def test_padding_is_stripped_but_order_is_kept(self) -> None:
        row = [PADDING, PADDING, 539, END_PADDING, 33, PADDING, 7]
        self.assertEqual(
            emitted_text_tokens(row, padding_id=PADDING, end_padding_id=END_PADDING), [539, 33, 7]
        )

    def test_ratio_counts_frames_that_carried_text(self) -> None:
        row = [PADDING, 539, PADDING, 33]
        self.assertEqual(
            emitted_text_ratio(row, padding_id=PADDING, end_padding_id=END_PADDING), 0.5
        )

    def test_a_silent_generation_has_ratio_zero(self) -> None:
        self.assertEqual(
            emitted_text_ratio([PADDING] * 8, padding_id=PADDING, end_padding_id=END_PADDING), 0.0
        )

    def test_an_empty_row_is_rejected_rather_than_scored(self) -> None:
        with self.assertRaises(ValueError):
            emitted_text_ratio([], padding_id=PADDING, end_padding_id=END_PADDING)


class LongestRepeatedNgramTests(unittest.TestCase):
    def test_it_finds_the_loop_and_counts_it(self) -> None:
        tokens = [9, 9, 9] + [1, 2, 3, 4] * 5 + [7]
        found = longest_repeated_ngram(tokens, min_n=3)
        self.assertIsNotNone(found)
        self.assertEqual(found.count, 5)
        self.assertEqual(list(found.ngram), [1, 2, 3, 4])

    def test_it_prefers_the_longest_repeat_over_a_shorter_one(self) -> None:
        # 1,2 appears inside every 1,2,3,4; reporting the 2-gram would understate the loop.
        tokens = [1, 2, 3, 4] * 4
        found = longest_repeated_ngram(tokens, min_n=2)
        self.assertEqual(len(found.ngram), 4)

    def test_non_repeating_text_returns_nothing(self) -> None:
        self.assertIsNone(longest_repeated_ngram(list(range(30)), min_n=3))

    def test_a_sequence_shorter_than_two_ngrams_returns_nothing(self) -> None:
        self.assertIsNone(longest_repeated_ngram([1, 2, 3], min_n=3))


class DistinctRatioTests(unittest.TestCase):
    def test_degenerate_vocabulary_scores_low(self) -> None:
        self.assertAlmostEqual(distinct_ratio([5, 5, 5, 5]), 0.25)

    def test_varied_text_scores_high(self) -> None:
        self.assertEqual(distinct_ratio([1, 2, 3, 4]), 1.0)

    def test_no_tokens_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            distinct_ratio([])


class VerdictTests(unittest.TestCase):
    """The detector has to fire on the failure it exists to detect, and stay quiet otherwise.

    A detector that never fires cannot certify absence, which is the only thing M3
    completion condition 3 asks of it.
    """

    thresholds = CollapseThresholds(
        min_n=3, max_ngram_repeats=4, min_distinct_ratio=0.3, max_emitted_ratio=0.95
    )

    def _row(self, tokens: list[int]) -> list[int]:
        return tokens

    def test_it_fires_on_a_four_gram_repeated_five_times(self) -> None:
        row = [PADDING] + [11, 12, 13, 14] * 5 + [PADDING]
        summary = summarise_generation(
            row, padding_id=PADDING, end_padding_id=END_PADDING, thresholds=self.thresholds
        )
        self.assertTrue(summary["exact_repeat_collapse"])

    def test_it_stays_quiet_on_a_non_repeating_generation(self) -> None:
        row = [PADDING, PADDING] + list(range(100, 140)) + [PADDING]
        summary = summarise_generation(
            row, padding_id=PADDING, end_padding_id=END_PADDING, thresholds=self.thresholds
        )
        self.assertFalse(summary["exact_repeat_collapse"])
        self.assertFalse(summary["monologue_loop"])

    def test_never_yielding_the_floor_is_a_monologue(self) -> None:
        row = list(range(100, 200))  # text on every single frame, never a pause
        summary = summarise_generation(
            row, padding_id=PADDING, end_padding_id=END_PADDING, thresholds=self.thresholds
        )
        self.assertTrue(summary["monologue_loop"])

    def test_a_degenerate_vocabulary_is_a_repeat_collapse(self) -> None:
        row = [PADDING] + [42] * 40 + [PADDING]
        summary = summarise_generation(
            row, padding_id=PADDING, end_padding_id=END_PADDING, thresholds=self.thresholds
        )
        self.assertTrue(summary["exact_repeat_collapse"])

    def test_a_silent_generation_is_flagged_rather_than_passed(self) -> None:
        # Emitting nothing is not health; without this it would score zero repeats and pass.
        row = [PADDING] * 60
        summary = summarise_generation(
            row, padding_id=PADDING, end_padding_id=END_PADDING, thresholds=self.thresholds
        )
        self.assertTrue(summary["silent"])

    def test_verdict_for_aggregates_over_a_run(self) -> None:
        summaries = [
            {"exact_repeat_collapse": False, "monologue_loop": False, "silent": False},
            {"exact_repeat_collapse": True, "monologue_loop": False, "silent": False},
        ]
        verdict = verdict_for(summaries)
        self.assertEqual(verdict["total"], 2)
        self.assertEqual(verdict["exact_repeat_collapse_count"], 1)
        self.assertEqual(verdict["monologue_loop_count"], 0)
        self.assertFalse(verdict["passes"])

    def test_a_mute_run_is_degenerate_even_though_no_collapse_flag_fires(self) -> None:
        # This is how the first condition-3 verdict went wrong: the two collapse counts were
        # quoted without the silence beside them, and a checkpoint that had stopped speaking
        # read as clean. degenerate_count cannot be quoted without the silence in it.
        summaries = [{"exact_repeat_collapse": False, "monologue_loop": False, "silent": True}] * 8
        verdict = verdict_for(summaries)
        self.assertEqual(verdict["monologue_loop_count"], 0)
        self.assertEqual(verdict["exact_repeat_collapse_count"], 0)
        self.assertEqual(verdict["degenerate_count"], 8)
        self.assertFalse(verdict["passes"])

    def test_a_generation_failing_two_ways_is_counted_once(self) -> None:
        # Summing the three counts gave 70 of 50 for v-real/epoch1, because a generation can
        # both repeat and monologue. A count that exceeds its own total is not a count.
        summaries = [{"exact_repeat_collapse": True, "monologue_loop": True, "silent": False}] * 5
        verdict = verdict_for(summaries)
        self.assertEqual(verdict["degenerate_count"], 5)
        self.assertLessEqual(verdict["degenerate_count"], verdict["total"])

    def test_degenerate_count_never_exceeds_the_total(self) -> None:
        summaries = [
            {"exact_repeat_collapse": True, "monologue_loop": True, "silent": False},
            {"exact_repeat_collapse": True, "monologue_loop": False, "silent": False},
            {"exact_repeat_collapse": False, "monologue_loop": False, "silent": True},
        ]
        verdict = verdict_for(summaries)
        self.assertEqual(verdict["degenerate_count"], 3)

    def test_degenerate_count_counts_each_distinct_failing_generation(self) -> None:
        summaries = [
            {"exact_repeat_collapse": True, "monologue_loop": False, "silent": False},
            {"exact_repeat_collapse": False, "monologue_loop": True, "silent": False},
            {"exact_repeat_collapse": False, "monologue_loop": False, "silent": True},
            {"exact_repeat_collapse": False, "monologue_loop": False, "silent": False},
        ]
        self.assertEqual(verdict_for(summaries)["degenerate_count"], 3)

    def test_a_clean_run_passes(self) -> None:
        summaries = [
            {"exact_repeat_collapse": False, "monologue_loop": False, "silent": False}
        ] * 30
        self.assertTrue(verdict_for(summaries)["passes"])


if __name__ == "__main__":
    unittest.main()
