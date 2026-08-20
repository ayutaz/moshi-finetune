import unittest

from tools.prepare_baseline_prompts import (
    PromptDatasetError,
    build_stereo_prompt,
    minimum_sample_count,
    select_audio_token_stems,
    verify_prompt_dataset,
)


def _row(dialogue_id: str, frames: int, *, b_frames: int | None = None) -> dict:
    return {
        "dialogue_id": dialogue_id,
        "A": [[0] * frames for _ in range(9)],
        "B": [[0] * (frames if b_frames is None else b_frames) for _ in range(9)],
    }


class StereoPromptTests(unittest.TestCase):
    def test_places_mono_reference_on_a_and_silence_on_b(self) -> None:
        mono = [[0.25, -0.5, 0.75]]

        stereo = build_stereo_prompt(mono)

        self.assertEqual(stereo, ([0.25, -0.5, 0.75], [0.0, 0.0, 0.0]))

    def test_rejects_non_mono_reference(self) -> None:
        with self.assertRaisesRegex(ValueError, "mono"):
            build_stereo_prompt([[0.0] * 10, [0.0] * 10])


class PaddingTextTokenTests(unittest.TestCase):
    def test_selects_sorted_npz_stems_only(self) -> None:
        stems = select_audio_token_stems(["prompt-02.npz", "README.md", "prompt-01.npz"])

        self.assertEqual(stems, ["prompt-01", "prompt-02"])


class PromptDatasetVerificationTests(unittest.TestCase):
    """`generate.py` drops examples shorter than `--prompt_length` without warning.

    The baseline must fail loudly instead, otherwise Stage 2 / Stage 3 are compared
    on fewer prompts than the protocol records.
    """

    def test_accepts_a_dataset_that_matches_the_fixed_protocol(self) -> None:
        rows = [_row(f"heldout/prompt-{index:02d}", 47 + index) for index in range(10)]

        report = verify_prompt_dataset(rows, expected_count=10, min_frames=40)

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["prompt_count"], 10)
        self.assertEqual(report["min_frames_observed"], 47)
        self.assertEqual(report["min_frames_required"], 40)

    def test_reports_the_example_id_to_prompt_mapping(self) -> None:
        rows = [_row("heldout/second", 60), _row("heldout/first", 50)]

        report = verify_prompt_dataset(rows, expected_count=2, min_frames=40)

        self.assertEqual(
            report["examples"],
            [
                {"example_id": 0, "dialogue_id": "heldout/second", "frames": 60},
                {"example_id": 1, "dialogue_id": "heldout/first", "frames": 50},
            ],
        )

    def test_rejects_a_prompt_that_generate_py_would_drop(self) -> None:
        rows = [_row("heldout/ok", 60), _row("heldout/too-short", 47)]

        with self.assertRaises(PromptDatasetError) as caught:
            verify_prompt_dataset(rows, expected_count=2, min_frames=50)

        self.assertIn("heldout/too-short=47", str(caught.exception))
        self.assertIn("min_frames=50", str(caught.exception))

    def test_rejects_a_missing_prompt(self) -> None:
        rows = [_row(f"heldout/prompt-{index:02d}", 60) for index in range(9)]

        with self.assertRaisesRegex(PromptDatasetError, "expected 10 prompt rows, got 9"):
            verify_prompt_dataset(rows, expected_count=10, min_frames=40)

    def test_rejects_mismatched_speaker_lengths(self) -> None:
        rows = [_row("heldout/ragged", 60, b_frames=59)]

        with self.assertRaisesRegex(PromptDatasetError, "A and B frame counts differ"):
            verify_prompt_dataset(rows, expected_count=1, min_frames=40)

    def test_rejects_an_unexpected_stream_count(self) -> None:
        rows = [_row("heldout/broken", 60)]
        rows[0]["A"] = rows[0]["A"][:8]

        with self.assertRaisesRegex(PromptDatasetError, "9 streams"):
            verify_prompt_dataset(rows, expected_count=1, min_frames=40)


if __name__ == "__main__":
    unittest.main()


class MinimumSampleCountTests(unittest.TestCase):
    """Teacher-forcing the user stream needs frames beyond the prompt.

    `generate.py` reads the user-stream codebooks out of the example itself, so a prompt
    audio file must cover `prompt_length + generation_length` Mimi frames, not just the
    prompt.
    """

    def test_converts_mimi_frames_to_samples_at_24khz(self) -> None:
        self.assertEqual(minimum_sample_count(min_frames=165, sample_rate=24000), 316800)

    def test_rounds_up_so_the_last_frame_is_complete(self) -> None:
        self.assertEqual(minimum_sample_count(min_frames=1, sample_rate=24000), 1920)
        self.assertEqual(minimum_sample_count(min_frames=48, sample_rate=24000), 92160)

    def test_rejects_a_non_positive_frame_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            minimum_sample_count(min_frames=0, sample_rate=24000)
