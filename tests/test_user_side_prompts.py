import unittest

from tools.prepare_baseline_prompts import (
    build_stereo_prompt,
    user_voiced_frames_in_window,
)


class BuildStereoPromptTests(unittest.TestCase):
    def test_speaker_a_carries_the_audio_by_default(self) -> None:
        a, b = build_stereo_prompt([[1.0, 2.0, 3.0]])
        self.assertEqual(list(a), [1.0, 2.0, 3.0])
        self.assertEqual(list(b), [0.0, 0.0, 0.0])

    def test_channel_b_puts_the_audio_on_the_user_stream(self) -> None:
        # The fixed conversations need the USER speaking, not the assistant. With the audio
        # on A, "does it ignore the user" would be measured against a silent user.
        a, b = build_stereo_prompt([[1.0, 2.0, 3.0]], channel="B")
        self.assertEqual(list(a), [0.0, 0.0, 0.0])
        self.assertEqual(list(b), [1.0, 2.0, 3.0])

    def test_a_lead_in_delays_the_audio_and_lengthens_both_channels(self) -> None:
        a, b = build_stereo_prompt([[1.0, 2.0]], channel="B", lead_in_samples=3)
        self.assertEqual(list(b), [0.0, 0.0, 0.0, 1.0, 2.0])
        self.assertEqual(list(a), [0.0] * 5)

    def test_a_lead_in_of_zero_changes_nothing(self) -> None:
        a, b = build_stereo_prompt([[1.0, 2.0]], channel="B", lead_in_samples=0)
        self.assertEqual(list(b), [1.0, 2.0])

    def test_a_negative_lead_in_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_stereo_prompt([[1.0]], channel="B", lead_in_samples=-1)

    def test_a_stereo_input_is_still_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_stereo_prompt([[1.0], [2.0]])

    def test_an_unknown_channel_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_stereo_prompt([[1.0]], channel="C")


class UserVoicedFramesInWindowTests(unittest.TestCase):
    """The window is the part generate.py actually returns; speech outside it is invisible."""

    def test_it_counts_only_frames_inside_the_window(self) -> None:
        mask = [False] * 40 + [True] * 20 + [False] * 105
        self.assertEqual(
            user_voiced_frames_in_window(mask, prompt_frames=40, generation_frames=125), 20
        )

    def test_speech_entirely_before_the_window_counts_zero(self) -> None:
        # This is the failure the check exists for: a prompt whose user speaks only during
        # the prompt is silent for every frame the model is judged on.
        mask = [True] * 30 + [False] * 135
        self.assertEqual(
            user_voiced_frames_in_window(mask, prompt_frames=40, generation_frames=125), 0
        )

    def test_speech_running_past_the_window_is_truncated_not_counted_whole(self) -> None:
        mask = [False] * 40 + [True] * 200
        self.assertEqual(
            user_voiced_frames_in_window(mask, prompt_frames=40, generation_frames=125), 125
        )

    def test_a_mask_shorter_than_the_window_counts_what_exists(self) -> None:
        mask = [False] * 40 + [True] * 10
        self.assertEqual(
            user_voiced_frames_in_window(mask, prompt_frames=40, generation_frames=125), 10
        )


if __name__ == "__main__":
    unittest.main()
