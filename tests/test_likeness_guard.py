import statistics
import unittest

from tools.likeness_guard import (
    DEFAULT_MIN_BAND_CLOSURE,
    apply_degeneracy_guard,
    band_closure,
    condition4_verdict,
    paired_mean_interval,
    student_t_critical,
    voiced_frames_floor,
)


class VoicedFramesFloorTests(unittest.TestCase):
    """A clip too short to carry speaker identity must not be scored as if it did."""

    def test_a_clip_above_the_floor_is_scorable(self) -> None:
        self.assertTrue(voiced_frames_floor(voiced_samples=48000, frame=480, minimum_frames=25))

    def test_a_clip_below_the_floor_is_not(self) -> None:
        # 10 frames of 480 samples is 0.2 s at 24 kHz - a breath, not a voice.
        self.assertFalse(voiced_frames_floor(voiced_samples=4800, frame=480, minimum_frames=25))

    def test_exactly_at_the_floor_counts(self) -> None:
        self.assertTrue(voiced_frames_floor(voiced_samples=25 * 480, frame=480, minimum_frames=25))

    def test_no_voiced_audio_is_not_scorable(self) -> None:
        self.assertFalse(voiced_frames_floor(voiced_samples=0, frame=480, minimum_frames=25))


class DegeneracyGuardTests(unittest.TestCase):
    """ECAPA rewards the collapse this milestone exists to reject."""

    def test_a_clip_from_a_degenerate_generation_is_withheld(self) -> None:
        # speaker_similarity.py's own docstring warns that a flat, over-smoothed rendering
        # can outscore one a listener finds closer. A repeating or mute generation is
        # exactly that, so its similarity is not evidence of speaker likeness.
        scores = {"a": 0.8, "b": 0.9}
        flags = {
            "a": {"silent": False, "exact_repeat_collapse": True, "monologue_loop": False},
            "b": {"silent": False, "exact_repeat_collapse": False, "monologue_loop": False},
        }
        kept, withheld = apply_degeneracy_guard(scores, flags)
        self.assertEqual(set(kept), {"b"})
        self.assertEqual(withheld, {"a": "exact_repeat_collapse"})

    def test_a_silent_clip_is_withheld_and_named(self) -> None:
        scores = {"a": 0.7}
        flags = {"a": {"silent": True, "exact_repeat_collapse": False, "monologue_loop": False}}
        kept, withheld = apply_degeneracy_guard(scores, flags)
        self.assertEqual(kept, {})
        self.assertEqual(withheld, {"a": "silent"})

    def test_a_clean_set_passes_through_untouched(self) -> None:
        scores = {"a": 0.7, "b": 0.8}
        flags = {
            k: {"silent": False, "exact_repeat_collapse": False, "monologue_loop": False}
            for k in scores
        }
        kept, withheld = apply_degeneracy_guard(scores, flags)
        self.assertEqual(kept, scores)
        self.assertEqual(withheld, {})

    def test_a_score_with_no_flags_is_an_error(self) -> None:
        # Silently keeping an unflagged clip is how the guard would be bypassed.
        with self.assertRaises(ValueError):
            apply_degeneracy_guard({"a": 0.7}, {})


class StudentTCriticalTests(unittest.TestCase):
    """scipy is not a dependency here, and an interval that needs it would not be computed."""

    def test_matches_the_published_value_for_nine_degrees_of_freedom(self) -> None:
        # n = 10 clips is the fixed held-out set, so df = 9 is the case that matters.
        self.assertAlmostEqual(student_t_critical(df=9), 2.262, places=3)

    def test_a_ninety_percent_interval_is_narrower(self) -> None:
        self.assertLess(student_t_critical(df=9, confidence=0.90), student_t_critical(df=9))

    def test_between_tabulated_rows_it_rounds_towards_the_wider_interval(self) -> None:
        # df 45 is not tabulated; using df 50's value would make the interval too narrow.
        self.assertEqual(student_t_critical(df=45), student_t_critical(df=40))

    def test_very_large_samples_fall_back_to_the_normal_value(self) -> None:
        self.assertAlmostEqual(student_t_critical(df=5000), 1.960, places=3)

    def test_an_untabulated_confidence_level_raises(self) -> None:
        # Returning the 95% number for a 99% request would be silently wrong.
        with self.assertRaisesRegex(ValueError, "confidence"):
            student_t_critical(df=9, confidence=0.99)

    def test_zero_degrees_of_freedom_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            student_t_critical(df=0)


class PairedMeanIntervalTests(unittest.TestCase):
    """The replacement for the sign test, which threw away every magnitude."""

    CONSISTENT = [0.05, 0.06, 0.04, 0.05, 0.06, 0.05, 0.04, 0.06, 0.05, 0.05]
    SCATTERED = [0.5, -0.4, 0.6, -0.5, 0.4, -0.3, 0.5, -0.4, 0.3, -0.2]

    def test_a_consistent_improvement_has_an_interval_that_clears_zero(self) -> None:
        self.assertGreater(paired_mean_interval(self.CONSISTENT)["lower_bound"], 0)

    def test_the_same_mean_with_more_scatter_does_not_clear_zero(self) -> None:
        # The distinction the sign test could not draw. These two have the same mean and
        # each has five clips above zero; only one of them is a measured effect.
        self.assertAlmostEqual(
            sum(self.SCATTERED) / len(self.SCATTERED),
            sum(self.CONSISTENT) / len(self.CONSISTENT),
            places=2,
        )
        self.assertLess(paired_mean_interval(self.SCATTERED)["lower_bound"], 0)

    def test_the_gate_uses_whichever_lower_bound_is_lower(self) -> None:
        result = paired_mean_interval(self.CONSISTENT, bootstrap_iterations=2000)
        self.assertEqual(result["lower_bound"], min(result["t"]["low"], result["bootstrap"]["low"]))

    def test_the_same_numbers_and_seed_give_the_same_interval(self) -> None:
        # An interval that moves between runs of the same data is not an interval, and
        # every count in M3 was a single unrecorded draw.
        first = paired_mean_interval(self.SCATTERED, bootstrap_iterations=500, seed=7)
        second = paired_mean_interval(self.SCATTERED, bootstrap_iterations=500, seed=7)
        self.assertEqual(first["bootstrap"], second["bootstrap"])

    def test_the_seed_is_reported(self) -> None:
        self.assertEqual(paired_mean_interval(self.CONSISTENT, seed=99)["bootstrap"]["seed"], 99)

    def test_a_single_clip_has_no_interval(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two"):
            paired_mean_interval([0.05])


class BandClosureTests(unittest.TestCase):
    """+0.02 means nothing until you know how far away the target speaker is."""

    def test_half_the_gap_reads_as_half(self) -> None:
        result = band_closure(control_mean=0.4, candidate_mean=0.6, band_mean=0.8, band_floor=0.74)
        self.assertAlmostEqual(result["closed"], 0.5)
        self.assertAlmostEqual(result["closed_percent"], 50.0)

    def test_the_same_delta_is_a_different_result_at_a_different_distance(self) -> None:
        # The defect in the old +0.02 bar: it is most of the way there in one case and a
        # rounding error in the other, and M3 recorded no absolute cosine to tell them apart.
        near = band_closure(control_mean=0.78, candidate_mean=0.80, band_mean=0.81, band_floor=0.74)
        far = band_closure(control_mean=0.37, candidate_mean=0.39, band_mean=0.81, band_floor=0.74)
        self.assertGreater(near["closed"], 0.5)
        self.assertLess(far["closed"], 0.05)

    def test_a_candidate_below_the_floor_is_outside_the_speakers_own_range(self) -> None:
        result = band_closure(
            control_mean=0.37, candidate_mean=0.45, band_mean=0.8166, band_floor=0.7405
        )
        self.assertFalse(result["candidate_within_band"])
        self.assertFalse(result["candidate_exceeds_band_mean"])

    def test_moving_away_from_the_speaker_reads_negative_rather_than_zero(self) -> None:
        # Clamping would flatten "went backwards" and "went nowhere" into one number.
        result = band_closure(control_mean=0.5, candidate_mean=0.4, band_mean=0.8, band_floor=0.74)
        self.assertLess(result["closed"], 0)

    def test_passing_the_band_mean_reads_above_one_rather_than_one(self) -> None:
        result = band_closure(control_mean=0.5, candidate_mean=0.9, band_mean=0.8, band_floor=0.74)
        self.assertGreater(result["closed"], 1.0)
        self.assertTrue(result["candidate_exceeds_band_mean"])

    def test_a_control_already_at_the_band_leaves_no_gap_to_close(self) -> None:
        result = band_closure(
            control_mean=0.85, candidate_mean=0.87, band_mean=0.81, band_floor=0.74
        )
        self.assertIsNone(result["closed"])
        self.assertIsNone(result["closed_percent"])
        self.assertIn("no gap to close", result["undefined_reason"])


class PreRegisteredConstantTests(unittest.TestCase):
    """The numbers the verdict turns on are pinned here, not only in prose.

    m3/DATASET_SPEC.md fixed 0.25 before any candidate was measured, and derived it: over
    the gap widths this experiment expects (0.08-0.15) it reproduces the old +0.02 bar,
    and over the measured gap of 0.4438 it is stricter. A constant that carries that much
    of the verdict has to fail a test when it moves, or the spec is the only thing holding
    it and specs do not run.
    """

    def test_the_band_closure_bar_is_the_pre_registered_quarter(self) -> None:
        self.assertEqual(DEFAULT_MIN_BAND_CLOSURE, 0.25)

    def test_moving_the_bar_changes_a_verdict_that_sits_between_the_two_values(self) -> None:
        # The live case: an arm whose interval clears zero but which closes only part of
        # the gap. It fails at the pre-registered 0.25 and passes below its closure, so the
        # constant is not decorative - it decides the arm on its own.
        deltas = [0.08, 0.07, 0.09, 0.08, 0.06, 0.09, 0.07, 0.08, 0.07, 0.05]
        control_mean = 0.3728
        candidate_mean = control_mean + statistics.fmean(deltas)
        common = {
            "paired_deltas": deltas,
            "denominator": len(deltas),
            "control_mean": control_mean,
            "candidate_mean": candidate_mean,
            "band": {"mean": 0.8166, "min": 0.7405},
            "collapse": {"degenerate_count": 0, "total": 10},
            "memorisation": "generalisation",
        }
        strict = condition4_verdict(**common, min_band_closure=0.25)
        lax = condition4_verdict(**common, min_band_closure=0.05)
        closed = strict["band_closure"]["closed"]
        self.assertGreater(closed, 0.05)
        self.assertLess(closed, 0.25)
        self.assertFalse(strict["passes"])
        self.assertTrue(lax["passes"])


class Condition4VerdictTests(unittest.TestCase):
    """Conditions 3 and 4 still interlock, and the sign count no longer decides."""

    CLEAN = {"degenerate_count": 0, "total": 10}
    BAND = {"mean": 0.8166, "min": 0.7405}
    # A clear, consistent improvement that closes most of the gap from a control at 0.60.
    STRONG = [0.13, 0.14, 0.12, 0.15, 0.13, 0.14, 0.13, 0.12, 0.15, 0.14]

    def verdict(self, deltas, *, control_mean=0.60, **overrides):
        values = list(deltas.values()) if isinstance(deltas, dict) else list(deltas)
        kwargs = {
            "paired_deltas": deltas,
            "denominator": len(values),
            "control_mean": control_mean,
            "candidate_mean": control_mean + sum(values) / len(values),
            "band": self.BAND,
            "collapse": self.CLEAN,
            "memorisation": "generalisation",
            "bootstrap_iterations": 2000,
        }
        kwargs.update(overrides)
        return condition4_verdict(**kwargs)

    def test_a_real_improvement_that_closes_the_gap_passes(self) -> None:
        result = self.verdict(self.STRONG)
        self.assertTrue(result["passes"], result["reason"])
        self.assertGreater(result["band_closure"]["closed"], DEFAULT_MIN_BAND_CLOSURE)

    def test_a_degenerate_arm_cannot_pass_however_high_the_similarity(self) -> None:
        # The mechanism the guard exists for: an arm that has collapsed scores well on
        # ECAPA precisely because it collapsed.
        result = self.verdict([0.20] * 10, collapse={"degenerate_count": 6, "total": 10})
        self.assertFalse(result["passes"])
        self.assertIn("degenerate", result["reason"])

    def test_memorisation_blocks_a_pass(self) -> None:
        result = self.verdict(self.STRONG, memorisation="memorisation")
        self.assertFalse(result["passes"])
        self.assertIn("memoris", result["reason"])

    def test_a_scattered_improvement_fails_on_the_interval_not_the_win_count(self) -> None:
        scattered = [0.5, -0.4, 0.6, -0.5, 0.4, -0.3, 0.5, -0.4, 0.4, -0.2]
        result = self.verdict(scattered)
        self.assertFalse(result["passes"])
        self.assertIn("does not exclude zero", result["reason"])

    def test_a_real_but_tiny_effect_fails_on_the_distance_to_the_band(self) -> None:
        # Statistically solid and practically nothing: 10 clips all improved by 0.005 out
        # of a 0.22 gap. The old +0.02 bar would also have caught this one; the point is
        # that the new bar states what the 0.02 was a proxy for.
        result = self.verdict([0.005] * 10)
        self.assertGreater(result["interval"]["lower_bound"], 0)
        self.assertFalse(result["passes"])
        self.assertIn("gap to the calibration band", result["reason"])

    def test_the_case_the_old_eight_of_ten_rule_got_wrong(self) -> None:
        # v-tts/epoch3's measured shape, re-measured 2026-08-25 without the degeneracy
        # guard: control 0.3728, arm 0.4466, six clips higher out of ten. The old rule
        # rejected it at 5-of-10 while it cleared the +0.02 effect bar - "no effect" and
        # "not enough clips to tell" were the same output. The new criterion separates
        # them: the interval straddles zero, so it fails as UNDERPOWERED, and it is
        # recorded as having closed only 17% of the distance to the speaker.
        measured = [
            -0.1868,
            -0.0214,
            0.0702,
            0.1786,
            0.4163,
            0.2202,
            0.1186,
            -0.0434,
            -0.1150,
            0.1003,
        ]
        result = self.verdict(measured, control_mean=0.3728)
        self.assertEqual(result["descriptive_sign_count"]["higher_on"], 6)
        self.assertLess(result["interval"]["lower_bound"], 0)
        self.assertFalse(result["passes"])
        self.assertLess(result["band_closure"]["closed"], 0.25)
        self.assertFalse(result["band_closure"]["candidate_within_band"])

    def test_the_sign_count_is_recorded_but_is_not_a_criterion(self) -> None:
        result = self.verdict(self.STRONG)
        self.assertEqual(result["descriptive_sign_count"]["higher_on"], 10)
        self.assertIn("0.383", result["descriptive_sign_count"]["not_a_criterion"])

    def test_deltas_short_of_the_denominator_are_refused(self) -> None:
        # Judging an arm on the clips where it still behaved is how a collapsing arm's
        # number improves as it collapses.
        with self.assertRaisesRegex(ValueError, "full fixed set"):
            condition4_verdict(
                paired_deltas=[0.1] * 7,
                denominator=10,
                control_mean=0.6,
                candidate_mean=0.7,
                band=self.BAND,
                collapse=self.CLEAN,
                memorisation="generalisation",
            )

    def test_absolute_means_that_disagree_with_the_deltas_are_refused(self) -> None:
        # Pairing a survivors-only candidate mean with a full-set delta vector.
        with self.assertRaisesRegex(ValueError, "different clip sets"):
            condition4_verdict(
                paired_deltas=[0.01] * 10,
                denominator=10,
                control_mean=0.6,
                candidate_mean=0.9,
                band=self.BAND,
                collapse=self.CLEAN,
                memorisation="generalisation",
            )

    def test_a_missing_calibration_band_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "calibration band"):
            condition4_verdict(
                paired_deltas=[0.1] * 10,
                denominator=10,
                control_mean=0.6,
                candidate_mean=0.7,
                band={"min": 0.74},
                collapse=self.CLEAN,
                memorisation="generalisation",
            )

    def test_a_mapping_of_deltas_is_accepted_in_a_fixed_order(self) -> None:
        as_mapping = {str(i): value for i, value in enumerate(self.STRONG)}
        self.assertEqual(
            self.verdict(as_mapping)["paired_mean_delta_full_set"],
            self.verdict(self.STRONG)["paired_mean_delta_full_set"],
        )

    def test_no_headroom_falls_back_to_a_no_regression_test(self) -> None:
        # A control already at or above the band: there is no gap, so the ratio is not a
        # number and the magnitude bar cannot be applied as one.
        result = self.verdict([0.01] * 10, control_mean=0.90)
        self.assertIsNone(result["band_closure"]["closed"])
        self.assertTrue(result["passes"], result["reason"])
        self.assertIn("vacuous", result["magnitude_note"])

    def test_no_headroom_still_rejects_a_regression(self) -> None:
        result = self.verdict([-0.01] * 10, control_mean=0.90)
        self.assertFalse(result["passes"])
        self.assertIn("below the control mean", result["reason"])


if __name__ == "__main__":
    unittest.main()
