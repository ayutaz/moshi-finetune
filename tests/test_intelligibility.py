import unittest

from tools.intelligibility import (
    REQUIRED_ROW_FIELDS,
    REQUIRED_SUMMARY_FIELDS,
    RepetitionThresholds,
    bits_per_character,
    calibration_band,
    describe_repetition,
    distinct_character_ratio,
    flag_transcripts_over_collapsed_audio,
    fluency_row,
    merge_by_id,
    most_covering_repeat,
    normalise_transcript,
    perplexity_from_nll,
    require_joint_intelligibility,
    summarise_fluency,
)

# The frozen numbers, derived in reports/m3r-intelligibility.json from 401 pieces of real
# human Japanese and copied here. Named rather than defaulted, for the same reason the
# dataclass has no defaults: a test that quietly used a different line than the report
# would certify nothing.
THRESHOLDS = RepetitionThresholds(
    min_ngram=3,
    max_repeat_coverage=0.5,
    min_distinct_char_ratio=0.4,
    min_scored_tokens=5,
)


def _row(clip_id, transcript, nll, tokens, thresholds=THRESHOLDS):
    return fluency_row(
        clip_id=clip_id,
        transcript=transcript,
        total_nll_nats=nll,
        scored_tokens=tokens,
        thresholds=thresholds,
    )


class NormalisationTests(unittest.TestCase):
    def test_whitespace_is_removed_because_it_is_not_spoken(self) -> None:
        self.assertEqual(normalise_transcript(" はい はい\nはい "), "はいはいはい")

    def test_width_is_normalised_so_one_mora_is_one_character(self) -> None:
        self.assertEqual(normalise_transcript("ﾊｲ"), "ハイ")

    def test_a_transcript_of_only_spaces_is_empty(self) -> None:
        self.assertEqual(normalise_transcript("   \n "), "")


class PerplexityMathTests(unittest.TestCase):
    def test_uniform_likelihood_gives_the_expected_perplexity(self) -> None:
        # Four tokens each costing ln(2) nats is one bit per token, so perplexity is 2.
        import math

        self.assertAlmostEqual(perplexity_from_nll(4 * math.log(2), 4), 2.0)

    def test_bits_per_character_divides_by_characters_not_tokens(self) -> None:
        import math

        self.assertAlmostEqual(bits_per_character(10 * math.log(2), 5), 2.0)

    def test_zero_tokens_is_refused_rather_than_returning_infinity(self) -> None:
        with self.assertRaises(ValueError):
            perplexity_from_nll(0.0, 0)

    def test_zero_characters_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            bits_per_character(1.0, 0)

    def test_a_negative_likelihood_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            perplexity_from_nll(-1.0, 3)


class RepeatDetectionTests(unittest.TestCase):
    def test_a_four_fold_loop_covers_the_whole_transcript(self) -> None:
        repeat = most_covering_repeat("ありがとうございました" * 4, min_ngram=3)
        assert repeat is not None
        self.assertEqual(repeat.ngram, "ありがとうございました")
        self.assertEqual(repeat.count, 4)
        self.assertAlmostEqual(repeat.coverage, 1.0)

    def test_ties_break_towards_the_period_not_towards_the_longest_gram(self) -> None:
        # 「ありがとうございました」x4 is covered completely by the 11-gram at four copies
        # and by the 22-gram at two. Reporting the 22-gram would describe a four-fold loop
        # as a two-fold one.
        repeat = most_covering_repeat("ありがとうございました" * 4, min_ngram=3)
        assert repeat is not None
        self.assertEqual(len(repeat.ngram), 11)

    def test_occurrences_are_counted_without_overlap(self) -> None:
        # Overlapping, 「ああ」 occurs three times in 「ああああ」 - more copies than the
        # string has room for, and a coverage of 6/4.
        repeat = most_covering_repeat("ああああ", min_ngram=2)
        assert repeat is not None
        self.assertEqual(repeat.count, 2)
        self.assertLessEqual(repeat.coverage, 1.0)

    def test_a_natural_sentence_has_no_qualifying_repeat(self) -> None:
        self.assertIsNone(
            most_covering_repeat("今日はとても良い天気ですね散歩に行きましょうか", min_ngram=3)
        )

    def test_nothing_repeating_returns_none_rather_than_zero_coverage(self) -> None:
        # A caller must be able to tell "no repeat was found" from "a repeat covering
        # nothing", which is why this is None and not a sentinel.
        self.assertIsNone(most_covering_repeat("あいうえお", min_ngram=3))

    def test_min_ngram_below_one_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            most_covering_repeat("ああ", min_ngram=0)

    def test_a_stuck_mora_is_caught_by_the_distinct_ratio(self) -> None:
        self.assertAlmostEqual(distinct_character_ratio("あ" * 20), 0.05)

    def test_distinct_ratio_of_an_empty_transcript_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            distinct_character_ratio("")

    def test_describe_repetition_publishes_both_statistics(self) -> None:
        described = describe_repetition("はいはいはいはい", min_ngram=3)
        self.assertIn("repeat_coverage", described)
        self.assertIn("distinct_char_ratio", described)


class FluencyRowTests(unittest.TestCase):
    def test_a_row_carries_the_repetition_beside_the_perplexity(self) -> None:
        row = _row("0", "今日はとても良い天気ですね", 20.0, 10)
        for field in REQUIRED_ROW_FIELDS:
            self.assertIn(field, row)

    def test_a_loop_is_flagged_even_though_its_perplexity_is_low(self) -> None:
        # The whole point of the module. Measured with llm-jp-3-150m, four copies of
        # 「ありがとうございました」 score ppl 171 against 2638 for one gibberish sentence,
        # so the fluency number alone ranks the loop as the better generation.
        loop = _row("loop", "ありがとうございました" * 4, 10.0, 20)
        gibberish = _row("gib", "ライトンとニューゼルゼンの服と呪縁のトリドンドです", 120.0, 16)
        self.assertLess(loop["perplexity"], gibberish["perplexity"])
        self.assertTrue(loop["repetitive"])
        self.assertFalse(gibberish["repetitive"])
        self.assertFalse(loop["clean"])

    def test_a_stuck_mora_is_flagged_by_the_distinct_ratio_not_the_coverage(self) -> None:
        row = _row("stuck", "あ" * 20, 5.0, 7)
        self.assertLess(row["distinct_char_ratio"], THRESHOLDS.min_distinct_char_ratio)
        self.assertTrue(row["repetitive"])

    def test_an_empty_transcript_is_not_scored_as_zero(self) -> None:
        row = _row("silent", "", None, None)
        self.assertFalse(row["transcribed"])
        self.assertIsNone(row["perplexity"])
        self.assertIsNone(row["repeat_coverage"])
        self.assertFalse(row["clean"])

    def test_a_non_empty_transcript_without_a_likelihood_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            _row("0", "はい", None, None)

    def test_a_short_transcript_is_marked_rather_than_trusted(self) -> None:
        row = _row("short", "ありがとうございました", 13.8, 2)
        self.assertTrue(row["short"])
        self.assertFalse(row["clean"])


class SummaryTests(unittest.TestCase):
    def _mixed_group(self):
        return [
            _row("0", "今日はとても良い天気ですね散歩に行きましょうか", 45.0, 10),
            _row("1", "そうですねお散歩は気持ちがよいものですわ", 40.0, 9),
            _row("2", "ありがとうございました" * 4, 10.0, 20),
            _row("3", "", None, None),
        ]

    def test_the_denominator_travels_with_the_mean(self) -> None:
        summary = summarise_fluency(self._mixed_group(), denominator=4)
        self.assertEqual(summary["denominator"], 4)
        self.assertEqual(summary["transcribed"], 3)
        self.assertEqual(summary["empty_transcript"], 1)

    def test_there_is_no_key_called_mean_perplexity(self) -> None:
        # An unqualified perplexity cannot be read: it hides both which denominator it used
        # and whether the loops are in it. `speaker_similarity` refuses `mean_delta` for
        # the same reason.
        summary = summarise_fluency(self._mixed_group(), denominator=4)
        for forbidden in ("perplexity", "mean_perplexity", "median_perplexity"):
            self.assertNotIn(forbidden, summary)

    def test_repetition_deflates_the_headline_perplexity(self) -> None:
        # The loop is the cheapest clip in the group, so including it pulls the median
        # down. A negative deflation is the signal that the fluency number is being helped
        # by a failure.
        summary = summarise_fluency(self._mixed_group(), denominator=4)
        self.assertLess(
            summary["median_perplexity_transcribed"], summary["median_perplexity_nonrepetitive"]
        )
        self.assertLess(summary["perplexity_deflation_from_repetition"], 0.0)

    def test_clean_count_is_over_the_full_denominator(self) -> None:
        # Two of four clips are usable. Counting them against the three that spoke would
        # let an arm improve its score by going silent.
        summary = summarise_fluency(self._mixed_group(), denominator=4)
        self.assertEqual(summary["clean_transcribed_count"], 2)
        self.assertAlmostEqual(summary["clean_transcribed_ratio"], 0.5)

    def test_a_silent_group_reports_nulls_rather_than_a_flattering_zero(self) -> None:
        summary = summarise_fluency([_row(str(i), "", None, None) for i in range(3)], denominator=3)
        self.assertEqual(summary["transcribed"], 0)
        self.assertIsNone(summary["median_perplexity_transcribed"])
        self.assertIsNone(summary["perplexity_deflation_from_repetition"])
        self.assertEqual(summary["clean_transcribed_count"], 0)

    def test_a_denominator_that_does_not_match_the_rows_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            summarise_fluency(self._mixed_group(), denominator=10)

    def test_a_non_positive_denominator_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            summarise_fluency([], denominator=0)


class MergeTests(unittest.TestCase):
    def test_the_token_metrics_land_on_the_same_row(self) -> None:
        rows = [_row("0", "はいそうですね本当にそう思いますわ", 30.0, 8)]
        tokens = [
            {
                "generation": "0.npy",
                "audio_entropy_bits": 4.9,
                "audio_distinct_tokens": 55,
                "audio_top_token_share": 0.29,
                "acoustic_collapse": False,
                "distinct_ratio": 0.8,
                "emitted_tokens": 20,
                "longest_repeat_count": 2,
                "silent": False,
            }
        ]
        merged = merge_by_id(rows, tokens)
        self.assertEqual(merged[0]["audio_entropy_bits"], 4.9)
        self.assertEqual(merged[0]["perplexity"], rows[0]["perplexity"])

    def test_a_missing_token_file_raises_rather_than_dropping_the_clip(self) -> None:
        rows = [_row("0", "はい", 5.0, 2), _row("1", "はい", 5.0, 2)]
        tokens = [{"generation": "0.npy", "acoustic_collapse": False}]
        with self.assertRaises(ValueError):
            merge_by_id(rows, tokens)

    def test_duplicate_token_rows_are_refused(self) -> None:
        rows = [_row("0", "はい", 5.0, 2)]
        tokens = [{"generation": "0.npy"}, {"generation": "0"}]
        with self.assertRaises(ValueError):
            merge_by_id(rows, tokens)


class HallucinationTests(unittest.TestCase):
    def test_a_transcript_over_a_dead_channel_is_counted(self) -> None:
        rows = [
            {**_row("0", "ご視聴ありがとうございました", 20.0, 6), "acoustic_collapse": True},
            {**_row("1", "", None, None), "acoustic_collapse": True},
            {**_row("2", "今日はよい天気ですね", 25.0, 7), "acoustic_collapse": False},
        ]
        flagged = flag_transcripts_over_collapsed_audio(rows)
        self.assertEqual(flagged["collapsed_audio"], 2)
        self.assertEqual(flagged["collapsed_audio_with_transcript"], 1)
        self.assertEqual(flagged["suspect_ids"], ["0"])

    def test_an_unjoined_row_is_refused_rather_than_counted_as_healthy(self) -> None:
        rows = [{**_row("0", "はい", 5.0, 2), "acoustic_collapse": True}, _row("1", "はい", 5.0, 2)]
        with self.assertRaises(ValueError):
            flag_transcripts_over_collapsed_audio(rows)


class DocumentGuardTests(unittest.TestCase):
    def _document(self):
        rows = [_row("0", "今日はとても良い天気ですね散歩に行きましょうか", 45.0, 10)]
        return {
            "asr_model": "faster-whisper/small",
            "language_model": "llm-jp/llm-jp-3-150m",
            "groups": {
                "control/held-out": {
                    "summary": summarise_fluency(rows, denominator=1),
                    "clips": rows,
                }
            },
        }

    def test_a_complete_document_passes(self) -> None:
        require_joint_intelligibility(self._document())

    def test_a_document_without_the_repetition_half_is_refused(self) -> None:
        document = self._document()
        for row in document["groups"]["control/held-out"]["clips"]:
            del row["repeat_coverage"]
        with self.assertRaises(ValueError):
            require_joint_intelligibility(document)

    def test_a_document_without_the_deflation_is_refused(self) -> None:
        document = self._document()
        del document["groups"]["control/held-out"]["summary"][
            "perplexity_deflation_from_repetition"
        ]
        with self.assertRaises(ValueError):
            require_joint_intelligibility(document)

    def test_a_document_that_does_not_name_its_language_model_is_refused(self) -> None:
        # A perplexity is a property of the scoring model as much as of the speech, so a
        # document that does not say which model produced it cannot be compared to anything.
        document = self._document()
        del document["language_model"]
        with self.assertRaises(ValueError):
            require_joint_intelligibility(document)

    def test_a_document_with_no_groups_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            require_joint_intelligibility({"asr_model": "x", "language_model": "y", "groups": {}})

    def test_every_required_summary_field_is_actually_produced(self) -> None:
        rows = [_row("0", "今日はとても良い天気ですね", 30.0, 8)]
        summary = summarise_fluency(rows, denominator=1)
        for field in REQUIRED_SUMMARY_FIELDS:
            self.assertIn(field, summary)


class CalibrationBandTests(unittest.TestCase):
    def test_the_band_reports_the_spread_not_only_a_centre(self) -> None:
        rows = [
            _row("0", "今日はとても良い天気ですね散歩に行きましょうか", 45.0, 10),
            _row("1", "そうですねお散歩は気持ちがよいものですわ", 40.0, 9),
        ]
        band = calibration_band(rows, label="tsukuyomi-corpus-v1")
        self.assertEqual(band["count"], 2)
        for field in ("perplexity", "repeat_coverage", "distinct_char_ratio"):
            self.assertIn("median", band[field])
            self.assertIn("min", band[field])
            self.assertIn("max", band[field])

    def test_a_band_with_nothing_transcribed_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            calibration_band([_row("0", "", None, None)], label="empty")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
