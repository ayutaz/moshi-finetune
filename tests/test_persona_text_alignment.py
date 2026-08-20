import unittest

from tools.persona_perplexity import ScoringError, build_aligned_text_stream


class AlignedTextStreamTests(unittest.TestCase):
    """Mirror the text stream `tools/tokenize_text.py` writes during training.

    `tokenize_and_pad_text` fills every frame with `text_padding_id`, writes each token at
    its word-timestamp frame, and sets `end_of_text_padding_id` in the frame before a token
    whose predecessor is still padding. Scoring dense tokens from frame 0 instead is what
    made the 2026-08-18 and 2026-08-20 runs score worse than chance.
    """

    def test_pads_the_lead_in_and_marks_the_start_of_the_run(self) -> None:
        stream, positions = build_aligned_text_stream(
            [11, 12],
            [21],
            num_frames=8,
            start_frame=4,
            text_padding_id=3,
            end_of_text_padding_id=0,
        )

        self.assertEqual(stream, [3, 3, 3, 0, 11, 12, 21, 3])
        self.assertEqual(positions, [6])

    def test_marks_the_run_only_once(self) -> None:
        stream, _ = build_aligned_text_stream(
            [11],
            [21, 22, 23],
            num_frames=7,
            start_frame=2,
            text_padding_id=3,
            end_of_text_padding_id=0,
        )

        self.assertEqual(stream, [3, 0, 11, 21, 22, 23, 3])
        self.assertEqual(stream.count(0), 1)

    def test_reports_a_frame_per_completion_token(self) -> None:
        _, positions = build_aligned_text_stream(
            [11, 12, 13],
            [21, 22],
            num_frames=12,
            start_frame=5,
            text_padding_id=3,
            end_of_text_padding_id=0,
        )

        self.assertEqual(positions, [8, 9])

    def test_rejects_a_window_too_short_for_the_tokens(self) -> None:
        with self.assertRaisesRegex(ScoringError, "num_frames"):
            build_aligned_text_stream(
                [11],
                [21],
                num_frames=3,
                start_frame=2,
                text_padding_id=3,
                end_of_text_padding_id=0,
            )

    def test_requires_room_for_the_end_of_padding_marker(self) -> None:
        with self.assertRaisesRegex(ScoringError, "start_frame"):
            build_aligned_text_stream(
                [11],
                [21],
                num_frames=8,
                start_frame=0,
                text_padding_id=3,
                end_of_text_padding_id=0,
            )

    def test_rejects_an_empty_completion(self) -> None:
        with self.assertRaisesRegex(ScoringError, "completion"):
            build_aligned_text_stream(
                [11],
                [],
                num_frames=8,
                start_frame=2,
                text_padding_id=3,
                end_of_text_padding_id=0,
            )


if __name__ == "__main__":
    unittest.main()
