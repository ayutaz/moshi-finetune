import unittest

from tools.speaker_similarity import paired_comparison_over_fixed_set


class PairedComparisonOverFixedSetTests(unittest.TestCase):
    """Condition 4 counts wins out of ten. The mean must use the same ten."""

    ALL = ("a", "b", "c", "d")

    def test_a_complete_pair_behaves_like_the_plain_comparison(self) -> None:
        result = paired_comparison_over_fixed_set(
            base={"a": 0.5, "b": 0.5, "c": 0.5, "d": 0.5},
            candidate={"a": 0.6, "b": 0.6, "c": 0.6, "d": 0.6},
            all_keys=self.ALL,
            names=("control", "candidate"),
        )
        self.assertEqual(result["higher_on"], 4)
        self.assertEqual(result["scorable"], 4)
        self.assertAlmostEqual(result["mean_delta_full_set"], 0.1)
        self.assertTrue(result["mean_delta_is_full_set"])

    def test_an_unscorable_candidate_clip_counts_as_a_loss_in_both_statistics(self) -> None:
        # The defect this exists for: counting a silent clip as not-higher while excluding
        # it from the mean lets an arm pass on the average of exactly the clips where it
        # still behaved.
        result = paired_comparison_over_fixed_set(
            base={"a": 0.5, "b": 0.5, "c": 0.5, "d": 0.5},
            candidate={"a": 0.9, "b": 0.9},  # c and d produced nothing scorable
            all_keys=self.ALL,
            names=("control", "candidate"),
        )
        self.assertEqual(result["higher_on"], 2)
        self.assertEqual(result["unscorable"], 2)
        self.assertEqual(result["denominator"], 4)
        # survivors-only mean would be +0.4; over the fixed set the two absences drag it down
        self.assertAlmostEqual(result["mean_delta_survivors"], 0.4)
        self.assertAlmostEqual(result["mean_delta_full_set"], 0.2)
        self.assertFalse(result["mean_delta_is_full_set"])

    def test_the_two_means_are_reported_separately_and_labelled(self) -> None:
        result = paired_comparison_over_fixed_set(
            base={"a": 0.5, "b": 0.5},
            candidate={"a": 0.6},
            all_keys=("a", "b"),
            names=("control", "candidate"),
        )
        self.assertIn("mean_delta_survivors", result)
        self.assertIn("mean_delta_full_set", result)
        self.assertNotIn("mean_delta", result)  # ambiguous name is not offered at all

    def test_an_unscorable_base_clip_removes_the_pair_and_is_counted(self) -> None:
        # A degraded control clip is not a win for the candidate; there is nothing to beat.
        result = paired_comparison_over_fixed_set(
            base={"a": 0.5},
            candidate={"a": 0.9, "b": 0.9},
            all_keys=("a", "b"),
            names=("control", "candidate"),
        )
        self.assertEqual(result["base_unscorable"], 1)
        self.assertEqual(result["higher_on"], 1)
        self.assertEqual(result["denominator"], 2)

    def test_an_arm_that_produces_nothing_scores_zero_rather_than_raising(self) -> None:
        result = paired_comparison_over_fixed_set(
            base={"a": 0.5, "b": 0.5},
            candidate={},
            all_keys=("a", "b"),
            names=("control", "candidate"),
        )
        self.assertEqual(result["higher_on"], 0)
        self.assertEqual(result["unscorable"], 2)
        self.assertEqual(result["mean_delta_full_set"], 0.0)
        self.assertIsNone(result["mean_delta_survivors"])

    def test_an_empty_key_set_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            paired_comparison_over_fixed_set(
                base={}, candidate={}, all_keys=(), names=("control", "candidate")
            )

    def test_the_absolute_cosines_travel_with_the_deltas(self) -> None:
        # A delta of +0.03 is a different result at 0.78 than at 0.37, and M3 reported only
        # the delta - which is how an arm could be discussed as "improving" with nobody able
        # to say whether it was anywhere near the target speaker's own range.
        result = paired_comparison_over_fixed_set(
            base={"a": 0.5, "b": 0.4},
            candidate={"a": 0.6},
            all_keys=("a", "b"),
            names=("control", "candidate"),
        )
        entry = result["per_clip_absolute"]["a"]
        self.assertEqual(entry["base"], 0.5)
        self.assertEqual(entry["candidate"], 0.6)
        self.assertAlmostEqual(entry["delta"], 0.1)

    def test_a_clip_the_candidate_could_not_produce_is_named_rather_than_dropped(self) -> None:
        result = paired_comparison_over_fixed_set(
            base={"a": 0.5, "b": 0.4},
            candidate={"a": 0.6},
            all_keys=("a", "b"),
            names=("control", "candidate"),
        )
        self.assertEqual(set(result["per_clip_absolute"]), {"a", "b"})
        self.assertIsNone(result["per_clip_absolute"]["b"]["candidate"])
        self.assertEqual(result["per_clip_absolute"]["b"]["base"], 0.4)

    def test_the_spread_of_the_deltas_is_reported_over_the_full_set(self) -> None:
        # Condition 4 now rests on an interval, and an interval needs a spread. Reporting
        # only a mean is what let "+0.032" be quoted for a week with no idea how noisy it was.
        result = paired_comparison_over_fixed_set(
            base={"a": 0.5, "b": 0.5, "c": 0.5, "d": 0.5},
            candidate={"a": 0.6, "b": 0.6, "c": 0.6, "d": 0.6},
            all_keys=self.ALL,
            names=("control", "candidate"),
        )
        self.assertAlmostEqual(result["delta_stdev_full_set"], 0.0)

    def test_the_full_set_spread_counts_the_clips_charged_zero(self) -> None:
        # Survivors-only, these two clips agree exactly; over the fixed set the two absences
        # are part of the variability and the interval has to see them.
        result = paired_comparison_over_fixed_set(
            base={"a": 0.5, "b": 0.5, "c": 0.5, "d": 0.5},
            candidate={"a": 0.9, "b": 0.9},
            all_keys=self.ALL,
            names=("control", "candidate"),
        )
        self.assertAlmostEqual(result["delta_stdev_survivors"], 0.0)
        self.assertGreater(result["delta_stdev_full_set"], 0.2)


if __name__ == "__main__":
    unittest.main()
