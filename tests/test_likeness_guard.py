import unittest

from tools.likeness_guard import (
    apply_degeneracy_guard,
    condition4_verdict,
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


class Condition4VerdictTests(unittest.TestCase):
    """Conditions 3 and 4 must interlock: a collapsed arm cannot pass on likeness."""

    CLEAN = {"degenerate_count": 0, "total": 10}

    def test_all_three_criteria_and_a_clean_arm_passes(self) -> None:
        v = condition4_verdict(
            higher_on=8,
            denominator=10,
            mean_delta_full_set=0.03,
            collapse=self.CLEAN,
            memorisation="generalisation",
            min_delta=0.02,
        )
        self.assertTrue(v["passes"])

    def test_a_degenerate_arm_cannot_pass_however_high_the_similarity(self) -> None:
        # The mechanism the guard exists for: an arm that has collapsed scores well on
        # ECAPA precisely because it collapsed.
        v = condition4_verdict(
            higher_on=10,
            denominator=10,
            mean_delta_full_set=0.20,
            collapse={"degenerate_count": 6, "total": 10},
            memorisation="generalisation",
            min_delta=0.02,
        )
        self.assertFalse(v["passes"])
        self.assertIn("degenerate", v["reason"])

    def test_memorisation_blocks_a_pass(self) -> None:
        v = condition4_verdict(
            higher_on=9,
            denominator=10,
            mean_delta_full_set=0.05,
            collapse=self.CLEAN,
            memorisation="memorisation",
            min_delta=0.02,
        )
        self.assertFalse(v["passes"])
        self.assertIn("memoris", v["reason"])

    def test_too_few_wins_fails_even_with_a_large_mean(self) -> None:
        v = condition4_verdict(
            higher_on=6,
            denominator=10,
            mean_delta_full_set=0.09,
            collapse=self.CLEAN,
            memorisation="generalisation",
            min_delta=0.02,
        )
        self.assertFalse(v["passes"])

    def test_too_small_an_effect_fails_even_with_every_win(self) -> None:
        v = condition4_verdict(
            higher_on=10,
            denominator=10,
            mean_delta_full_set=0.005,
            collapse=self.CLEAN,
            memorisation="generalisation",
            min_delta=0.02,
        )
        self.assertFalse(v["passes"])


if __name__ == "__main__":
    unittest.main()
