import unittest

from tools.dialogue_collapse import (
    SPEAKER_A_CODEBOOK0_ROW,
    SPEAKER_B_CODEBOOK0_ROW,
    AcousticThresholds,
    CollapseThresholds,
    acoustic_thresholds_from_calibration,
    distinct_ratio,
    distinct_token_count,
    effective_vocabulary,
    emitted_text_ratio,
    emitted_text_tokens,
    group_report,
    longest_repeated_ngram,
    speaker_codebook0,
    summarise_acoustics,
    summarise_generation,
    summarise_streams,
    token_entropy_bits,
    top_token,
    top_token_share,
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

    def test_a_clean_run_passes_only_once_the_audio_has_been_scored(self) -> None:
        # A run whose summaries carry no audio verdict cannot pass: see
        # AcousticAggregationTests.test_a_run_scored_without_the_audio_cannot_pass.
        text_only = [
            {"exact_repeat_collapse": False, "monologue_loop": False, "silent": False}
        ] * 30
        self.assertFalse(verdict_for(text_only)["passes"])
        scored = [dict(summary, acoustic_collapse=False) for summary in text_only]
        self.assertTrue(verdict_for(scored)["passes"])


PADDING_ROW = [PADDING] * 124
SILENCE = 1316  # Mimi's codebook-0 silence token, 91.13% of the teacher-forced partner


def a_collapsed_audio_row(frames: int = 124) -> list[int]:
    """The control's general30 signature: silence texture, four distinct tokens."""
    row = [SILENCE] * frames
    for index in range(0, 8):
        row[index * 3] = 768
    row[5] = 318
    row[11] = 318
    row[60] = 1926
    return row  # 1316 x 113, 768 x 8, 318 x 2, 1926 x 1 - the control's exact profile


def a_healthy_audio_row(frames: int = 124) -> list[int]:
    """Roughly M0's shape: dozens of tokens, no single one owning the window."""
    return [100 + (index * 7) % 55 for index in range(frames)]


class AcousticMetricTests(unittest.TestCase):
    def test_distinct_counts_tokens_not_a_ratio(self) -> None:
        # distinct_ratio divides by length; the audio row has one token per frame and no
        # padding to strip, so the calibration band is expressed as a raw count.
        self.assertEqual(distinct_token_count([5, 5, 5, 7]), 2)

    def test_top_token_reports_which_token_holds_the_window(self) -> None:
        # Which token it is decides the reading: 1316 means the channel is silent.
        token, share = top_token([SILENCE] * 9 + [768])
        self.assertEqual(token, SILENCE)
        self.assertAlmostEqual(share, 0.9)

    def test_top_token_breaks_ties_on_the_smaller_id(self) -> None:
        # Without a rule the answer would follow dict insertion order.
        self.assertEqual(top_token([9, 9, 4, 4])[0], 4)
        self.assertEqual(top_token([4, 4, 9, 9])[0], 4)

    def test_top_token_share_agrees_with_top_token(self) -> None:
        row = [SILENCE] * 113 + [768] * 8 + [318] * 3
        self.assertEqual(top_token_share(row), top_token(row)[1])

    def test_entropy_of_a_constant_row_is_zero(self) -> None:
        self.assertEqual(token_entropy_bits([SILENCE] * 124), 0.0)

    def test_entropy_of_a_uniform_row_is_log2_of_its_vocabulary(self) -> None:
        self.assertAlmostEqual(token_entropy_bits([1, 2, 3, 4] * 10), 2.0)

    def test_effective_vocabulary_is_two_to_the_entropy(self) -> None:
        # The unit the threshold's headroom is a ratio in.
        self.assertAlmostEqual(effective_vocabulary([1, 2, 3, 4] * 10), 4.0)

    def test_an_empty_row_is_rejected_by_every_metric(self) -> None:
        for metric in (
            distinct_token_count,
            top_token,
            top_token_share,
            token_entropy_bits,
            effective_vocabulary,
        ):
            with self.assertRaises(ValueError):
                metric([])

    def test_the_metrics_take_plain_sequences(self) -> None:
        # No numpy anywhere in the scoring path: the suite has to run without it.
        self.assertEqual(distinct_token_count((1, 1, 2)), 2)
        self.assertEqual(distinct_token_count(range(4)), 4)


class AcousticThresholdTests(unittest.TestCase):
    calibrated = AcousticThresholds(max_entropy_bits=1.43, max_distinct_tokens=9)

    def test_thresholds_have_no_defaults(self) -> None:
        # CollapseThresholds() returns 0.3 and 0.95 while the frozen text calibration says
        # 0.4 and 0.85. A caller who forgets gets a laxer detector than the report used.
        # The audio half cannot be built without naming its numbers.
        with self.assertRaises(TypeError):
            AcousticThresholds()  # type: ignore[call-arg]

    def test_they_are_read_out_of_a_calibration_document(self) -> None:
        thresholds = acoustic_thresholds_from_calibration(
            {"thresholds": {"max_entropy_bits": 1.43, "max_distinct_tokens": 9}}
        )
        self.assertEqual(thresholds, self.calibrated)

    def test_a_calibration_missing_a_threshold_is_rejected(self) -> None:
        # Defaulting the gap is the failure this module exists to prevent.
        with self.assertRaises(ValueError):
            acoustic_thresholds_from_calibration({"thresholds": {"max_entropy_bits": 1.43}})
        with self.assertRaises(ValueError):
            acoustic_thresholds_from_calibration({})


class AcousticVerdictTests(unittest.TestCase):
    thresholds = AcousticThresholds(max_entropy_bits=1.43, max_distinct_tokens=9)

    def test_it_fires_on_the_control_general30_signature(self) -> None:
        # The 17 of 30 that the text-only detector could not see at all.
        summary = summarise_acoustics(a_collapsed_audio_row(), thresholds=self.thresholds)
        self.assertTrue(summary["acoustic_collapse"])
        self.assertEqual(summary["top_token"], SILENCE)
        self.assertLess(summary["entropy_bits"], 1.0)

    def test_it_stays_quiet_on_a_varied_row(self) -> None:
        summary = summarise_acoustics(a_healthy_audio_row(), thresholds=self.thresholds)
        self.assertFalse(summary["acoustic_collapse"])
        self.assertGreater(summary["distinct_tokens"], 9)

    def test_a_long_tail_of_singletons_cannot_hide_a_dead_channel(self) -> None:
        # control/general30/7.npy: 10 distinct tokens, so the distinct ceiling misses it,
        # but 0.946 bits, so the entropy ceiling catches it.
        row = [SILENCE] * 107 + [768] * 7 + list(range(200, 210))
        summary = summarise_acoustics(row, thresholds=self.thresholds)
        self.assertGreater(summary["distinct_tokens"], self.thresholds.max_distinct_tokens)
        self.assertTrue(summary["acoustic_collapse"])

    def test_a_few_tokens_spread_evenly_cannot_hide_either(self) -> None:
        # v-real/epoch4/seen/9.npy shape: 7 distinct tokens at 1.55 bits, above the entropy
        # ceiling, below the distinct one. Neither ceiling contains the other.
        row = [2029] * 59 + [SILENCE] * 40 + [768] * 15 + [84] * 5 + [318] * 3 + [49, 716]
        summary = summarise_acoustics(row, thresholds=self.thresholds)
        self.assertGreater(summary["entropy_bits"], self.thresholds.max_entropy_bits)
        self.assertTrue(summary["acoustic_collapse"])

    def test_the_reported_numbers_agree_with_the_metric_functions(self) -> None:
        row = a_collapsed_audio_row()
        summary = summarise_acoustics(row, thresholds=self.thresholds)
        self.assertEqual(summary["frames"], len(row))
        self.assertEqual(summary["distinct_tokens"], distinct_token_count(row))
        self.assertEqual(summary["top_token_share"], top_token_share(row))
        self.assertEqual(summary["entropy_bits"], token_entropy_bits(row))
        self.assertEqual(summary["effective_vocabulary"], effective_vocabulary(row))


class StreamExtractionTests(unittest.TestCase):
    def test_speaker_a_codebook0_is_row_one(self) -> None:
        streams = [[0] * 4, [11] * 4] + [[0] * 4 for _ in range(15)]
        self.assertEqual(list(speaker_codebook0(streams, row=SPEAKER_A_CODEBOOK0_ROW)), [11] * 4)

    def test_speaker_b_codebook0_is_row_nine(self) -> None:
        self.assertEqual(SPEAKER_B_CODEBOOK0_ROW, 9)

    def test_an_array_too_short_to_hold_the_row_is_rejected(self) -> None:
        # M3's detector read tokens[0] and would have read it out of a 9-row array just as
        # happily. Naming the row and refusing the array is what stops the next misread.
        with self.assertRaises(ValueError):
            speaker_codebook0([[1, 2, 3]], row=SPEAKER_A_CODEBOOK0_ROW)

    def test_an_empty_row_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            speaker_codebook0([[1], []], row=SPEAKER_A_CODEBOOK0_ROW)


class CombinedVerdictTests(unittest.TestCase):
    text = CollapseThresholds(
        min_n=3, max_ngram_repeats=4, min_distinct_ratio=0.4, max_emitted_ratio=0.85
    )
    audio = AcousticThresholds(max_entropy_bits=1.43, max_distinct_tokens=9)

    def _streams(self, text_row: list[int], audio_row: list[int]) -> list[list[int]]:
        streams = [text_row, audio_row]
        streams.extend([list(audio_row) for _ in range(15)])
        return streams

    def _summarise(self, text_row: list[int], audio_row: list[int]) -> dict:
        return summarise_streams(
            self._streams(text_row, audio_row),
            padding_id=PADDING,
            end_padding_id=END_PADDING,
            thresholds=self.text,
            acoustic_thresholds=self.audio,
        )

    def test_a_text_clean_generation_with_a_dead_voice_is_a_failure(self) -> None:
        # This is the control's general30: two stray text tokens clear the knife-edge
        # silent test, so the text-only detector recorded silent_count 0 for the arm.
        text_row = list(PADDING_ROW)
        text_row[2] = 11
        text_row[40] = 12
        summary = self._summarise(text_row, a_collapsed_audio_row())
        self.assertFalse(summary["silent"])
        self.assertFalse(summary["text_failure"])
        self.assertTrue(summary["acoustic_collapse"])
        self.assertTrue(summary["acoustic_only"])

    def test_a_text_failure_with_healthy_audio_is_still_a_failure(self) -> None:
        # 105 of M3's 550. Requiring both halves would have certified them clean.
        text_row = [PADDING] + [11, 12, 13, 14] * 5 + [PADDING] * 103
        summary = self._summarise(text_row, a_healthy_audio_row())
        self.assertTrue(summary["exact_repeat_collapse"])
        self.assertFalse(summary["acoustic_collapse"])
        self.assertFalse(summary["acoustic_only"])
        self.assertTrue(summary["text_failure"])

    def test_a_clean_generation_trips_nothing(self) -> None:
        text_row = list(PADDING_ROW)
        text_row[10:40] = list(range(100, 130))
        summary = self._summarise(text_row, a_healthy_audio_row())
        self.assertFalse(summary["text_failure"])
        self.assertFalse(summary["acoustic_collapse"])

    def test_the_audio_numbers_are_carried_into_the_summary(self) -> None:
        summary = self._summarise(list(PADDING_ROW), a_collapsed_audio_row())
        self.assertEqual(summary["audio_top_token"], SILENCE)
        self.assertEqual(summary["audio_distinct_tokens"], 4)
        self.assertIn("audio_entropy_bits", summary)

    def test_streams_of_different_lengths_are_rejected(self) -> None:
        # Text and audio are the same timeline; a mismatch means the rows are misaligned.
        with self.assertRaises(ValueError):
            summarise_streams(
                self._streams(list(PADDING_ROW), a_collapsed_audio_row(120)),
                padding_id=PADDING,
                end_padding_id=END_PADDING,
                thresholds=self.text,
                acoustic_thresholds=self.audio,
            )


class AcousticAggregationTests(unittest.TestCase):
    def _summary(self, **flags: bool) -> dict:
        summary = {"monologue_loop": False, "exact_repeat_collapse": False, "silent": False}
        summary.update(flags)
        return summary

    def test_an_acoustic_collapse_lands_in_degenerate_count(self) -> None:
        summaries = [self._summary(acoustic_collapse=True) for _ in range(17)]
        summaries += [self._summary(acoustic_collapse=False) for _ in range(13)]
        verdict = verdict_for(summaries)
        self.assertEqual(verdict["acoustic_collapse_count"], 17)
        self.assertEqual(verdict["degenerate_count"], 17)
        self.assertFalse(verdict["passes"])

    def test_a_generation_failing_both_ways_is_counted_once(self) -> None:
        summaries = [self._summary(silent=True, acoustic_collapse=True)] * 6
        verdict = verdict_for(summaries)
        self.assertEqual(verdict["degenerate_count"], 6)
        self.assertLessEqual(verdict["degenerate_count"], verdict["total"])

    def test_acoustic_only_count_names_the_blind_spot(self) -> None:
        summaries = [
            self._summary(acoustic_collapse=True),
            self._summary(silent=True, acoustic_collapse=True),
            self._summary(exact_repeat_collapse=True, acoustic_collapse=False),
            self._summary(acoustic_collapse=False),
        ]
        verdict = verdict_for(summaries)
        self.assertEqual(verdict["acoustic_collapse_count"], 2)
        self.assertEqual(verdict["acoustic_only_count"], 1)
        self.assertEqual(verdict["degenerate_count"], 3)

    def test_a_run_scored_without_the_audio_cannot_pass(self) -> None:
        # The M3 verdict was computed from text-only summaries and certified the control
        # as silent_count 0 while 17 of its 30 general30 generations had a dead channel.
        # A detector that did not look cannot certify absence.
        verdict = verdict_for([self._summary() for _ in range(30)])
        self.assertFalse(verdict["acoustic_scored"])
        self.assertIsNone(verdict["acoustic_collapse_count"])
        self.assertIsNone(verdict["acoustic_only_count"])
        self.assertFalse(verdict["passes"])

    def test_a_fully_scored_clean_run_passes(self) -> None:
        verdict = verdict_for([self._summary(acoustic_collapse=False) for _ in range(30)])
        self.assertTrue(verdict["acoustic_scored"])
        self.assertTrue(verdict["passes"])

    def test_mixing_scored_and_unscored_generations_is_rejected(self) -> None:
        # A count against the wrong denominator is worse than no count.
        with self.assertRaises(ValueError):
            verdict_for([self._summary(acoustic_collapse=False), self._summary()])


class GroupReportTests(unittest.TestCase):
    """The per-group block the report quotes, so a reader can check the verdict."""

    def _scored(
        self, name: str, distinct: int, share: float, entropy: float, collapse: bool
    ) -> dict:
        return {
            "generation": name,
            "monologue_loop": False,
            "exact_repeat_collapse": False,
            "silent": False,
            "acoustic_collapse": collapse,
            "audio_distinct_tokens": distinct,
            "audio_top_token_share": share,
            "audio_entropy_bits": entropy,
            "audio_top_token": SILENCE if collapse else 2029,
        }

    def test_it_carries_the_distributions_and_names_the_failures(self) -> None:
        summaries = [
            self._scored("0.npy", 4, 0.91, 0.53, True),
            self._scored("1.npy", 60, 0.20, 5.10, False),
            self._scored("2.npy", 4, 0.90, 0.57, True),
        ]
        report = group_report(summaries)
        self.assertEqual(report["acoustic_collapse_count"], 2)
        self.assertEqual(report["collapsed_generations"], ["0.npy", "2.npy"])
        self.assertEqual(report["audio_distinct_tokens"], {"min": 4, "median": 4, "max": 60})
        self.assertEqual(report["modal_top_token"], SILENCE)

    def test_the_modal_token_breaks_ties_on_the_smaller_id(self) -> None:
        # Otherwise the reported token follows the order the directory happened to be read.
        summaries = [
            self._scored("0.npy", 4, 0.91, 0.53, True),
            self._scored("1.npy", 60, 0.20, 5.10, False),
        ]
        self.assertEqual(group_report(summaries)["modal_top_token"], SILENCE)
        self.assertEqual(group_report(summaries[::-1])["modal_top_token"], SILENCE)


if __name__ == "__main__":
    unittest.main()
