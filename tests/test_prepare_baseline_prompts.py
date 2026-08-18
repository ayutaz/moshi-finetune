import unittest

from tools.prepare_baseline_prompts import build_stereo_prompt, select_audio_token_stems


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


if __name__ == "__main__":
    unittest.main()
