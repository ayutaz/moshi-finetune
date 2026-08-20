import math
import unittest

from tools.persona_perplexity import (
    ScoringError,
    assert_better_than_chance,
    build_delayed_audio_context,
)


class DelayedAudioContextTests(unittest.TestCase):
    """The 2026-08-18 run conditioned every codebook on `zero_token_id` for the whole
    sequence and scored worse than chance. `zero_token_id` is a loss ignore marker and
    post-sequence padding, never audio conditioning, so the context now comes from real
    Mimi tokens with the model's own delay pattern applied.
    """

    def test_applies_the_per_codebook_delay(self) -> None:
        context = build_delayed_audio_context(
            [[10, 11, 12, 13], [20, 21, 22, 23]],
            delays=[0, 1],
            initial_token_id=99,
            length=4,
        )

        self.assertEqual(context, [[10, 11, 12, 13], [99, 20, 21, 22]])

    def test_fills_the_initial_token_before_the_delay(self) -> None:
        context = build_delayed_audio_context(
            [[5, 6, 7]], delays=[2], initial_token_id=99, length=3
        )

        self.assertEqual(context, [[99, 99, 5]])

    def test_truncates_a_longer_context_to_the_requested_length(self) -> None:
        context = build_delayed_audio_context(
            [[1, 2, 3, 4, 5]], delays=[0], initial_token_id=99, length=2
        )

        self.assertEqual(context, [[1, 2]])

    def test_rejects_a_context_shorter_than_the_scored_sequence(self) -> None:
        with self.assertRaisesRegex(ScoringError, "at least 5 frames"):
            build_delayed_audio_context([[1, 2, 3]], delays=[0], initial_token_id=99, length=5)

    def test_rejects_a_delay_count_that_does_not_match_the_codebooks(self) -> None:
        with self.assertRaisesRegex(ScoringError, "delay"):
            build_delayed_audio_context(
                [[1, 2], [3, 4]], delays=[0], initial_token_id=99, length=2
            )


class ChanceBoundTests(unittest.TestCase):
    """A baseline that scores worse than a uniform distribution is not a metric.

    Recording one as the Stage 2 / Stage 3 reference is the failure this gate prevents.
    """

    def test_accepts_a_summary_better_than_uniform(self) -> None:
        assert_better_than_chance({"preferred_mean_nll": 4.2}, text_card=32000)

    def test_rejects_a_summary_at_or_above_the_uniform_bound(self) -> None:
        with self.assertRaises(ScoringError) as caught:
            assert_better_than_chance({"preferred_mean_nll": 12.878}, text_card=32000)

        message = str(caught.exception)
        self.assertIn("12.878", message)
        self.assertIn(f"{math.log(32000):.3f}", message)

    def test_rejects_exactly_at_the_bound(self) -> None:
        with self.assertRaises(ScoringError):
            assert_better_than_chance(
                {"preferred_mean_nll": math.log(32000)}, text_card=32000
            )


if __name__ == "__main__":
    unittest.main()
