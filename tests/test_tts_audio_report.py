import math
import unittest

from tools.tts_audio_report import (
    clipped_run_lengths,
    headroom_gain,
    loud_bounds,
    summarise_clip,
)


class ClippedRunTests(unittest.TestCase):
    """Irodori-TTS writes PCM_16 through soundfile, which saturates float samples past
    +/-1.0. Isolated saturated samples are a write-time artefact; a sustained run is real
    distortion, so the run length is what distinguishes them.
    """

    def test_finds_no_runs_in_clean_audio(self) -> None:
        self.assertEqual(clipped_run_lengths([0, 1000, -1000, 32766], full_scale=32767), [])

    def test_reports_each_run_separately(self) -> None:
        samples = [32767, 0, -32768, -32768, 0, 32767, 32767, 32767]

        self.assertEqual(clipped_run_lengths(samples, full_scale=32767), [1, 2, 3])

    def test_treats_both_rails_as_clipping(self) -> None:
        self.assertEqual(clipped_run_lengths([32767, -32768], full_scale=32767), [2])


class HeadroomTests(unittest.TestCase):
    def test_scales_a_full_scale_peak_down_to_the_target(self) -> None:
        gain = headroom_gain(peak=32768, full_scale=32767, headroom_db=1.0)

        self.assertAlmostEqual(32768 * gain, 32767 * 10 ** (-1.0 / 20), places=3)

    def test_leaves_quiet_audio_untouched(self) -> None:
        self.assertEqual(headroom_gain(peak=1000, full_scale=32767, headroom_db=1.0), 1.0)

    def test_rejects_a_silent_signal(self) -> None:
        with self.assertRaisesRegex(ValueError, "peak"):
            headroom_gain(peak=0, full_scale=32767, headroom_db=1.0)


class LoudBoundsTests(unittest.TestCase):
    def test_locates_the_first_and_last_loud_sample(self) -> None:
        samples = [0, 0, 500, 0, 800, 0, 0]

        self.assertEqual(loud_bounds(samples, threshold=100), (2, 4))

    def test_returns_none_for_silence(self) -> None:
        self.assertIsNone(loud_bounds([0, 1, -1], threshold=100))


class SummaryTests(unittest.TestCase):
    def test_summarises_a_clean_file(self) -> None:
        summary = summarise_clip([0, 5000, -5000, 0], sample_rate=48000, full_scale=32767)

        self.assertEqual(summary["clipped_samples"], 0)
        self.assertEqual(summary["longest_clipped_run"], 0)
        self.assertAlmostEqual(summary["seconds"], 4 / 48000)
        self.assertEqual(summary["peak"], 5000)

    def test_reports_leading_and_trailing_silence_as_durations(self) -> None:
        samples = [0] * 4800 + [8000] * 48000 + [0] * 9600

        summary = summarise_clip(samples, sample_rate=48000, full_scale=32767)

        self.assertAlmostEqual(summary["leading_silence_seconds"], 0.1, places=3)
        self.assertAlmostEqual(summary["trailing_silence_seconds"], 0.2, places=3)

    def test_flags_a_file_that_is_entirely_silent(self) -> None:
        summary = summarise_clip([0] * 100, sample_rate=48000, full_scale=32767)

        self.assertTrue(summary["silent"])

    def test_reports_rms(self) -> None:
        summary = summarise_clip([3, -3, 3, -3], sample_rate=48000, full_scale=32767)

        self.assertAlmostEqual(summary["rms"], 3.0)
        self.assertFalse(math.isnan(summary["rms"]))


if __name__ == "__main__":
    unittest.main()
