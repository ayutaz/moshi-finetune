import unittest

from tools.persona_perplexity import ScoringError, build_delayed_audio_context


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
            build_delayed_audio_context([[1, 2], [3, 4]], delays=[0], initial_token_id=99, length=2)


if __name__ == "__main__":
    unittest.main()
