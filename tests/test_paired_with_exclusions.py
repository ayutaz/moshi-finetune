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


if __name__ == "__main__":
    unittest.main()
