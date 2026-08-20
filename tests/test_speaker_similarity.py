import math
import unittest

from tools.speaker_similarity import (
    cosine_similarity,
    rms_normalise,
    summarise_similarity,
    to_int16_scale,
    voiced_segments,
)


class VoicedSegmentTests(unittest.TestCase):
    """The plan measures similarity on voiced regions only.

    Leading and trailing silence carries no speaker identity, and a rendering that simply
    pads differently would otherwise shift the score.
    """

    def test_keeps_only_frames_above_the_threshold(self) -> None:
        samples = [0, 0, 900, 1000, 0, 0, 800, 0]

        self.assertEqual(voiced_segments(samples, frame=1, threshold=500), [900, 1000, 800])

    def test_uses_frame_energy_not_single_samples(self) -> None:
        # a lone spike inside an otherwise silent frame must not qualify the frame
        samples = [0, 0, 0, 3000, 0, 0, 0, 0]

        self.assertEqual(voiced_segments(samples, frame=4, threshold=2000), [])

    def test_returns_empty_for_silence(self) -> None:
        self.assertEqual(voiced_segments([0, 0, 0, 0], frame=2, threshold=100), [])


class RmsNormaliseTests(unittest.TestCase):
    def test_scales_to_the_target_rms(self) -> None:
        result = rms_normalise([100, -100, 100, -100], target_rms=0.1)

        self.assertAlmostEqual(math.sqrt(sum(v * v for v in result) / len(result)), 0.1, places=6)

    def test_rejects_a_silent_signal(self) -> None:
        with self.assertRaisesRegex(ValueError, "silent"):
            rms_normalise([0, 0, 0], target_rms=0.1)


class CosineTests(unittest.TestCase):
    def test_identical_vectors_score_one(self) -> None:
        self.assertAlmostEqual(cosine_similarity([1.0, 2.0], [1.0, 2.0]), 1.0)

    def test_orthogonal_vectors_score_zero(self) -> None:
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_opposite_vectors_score_minus_one(self) -> None:
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [-1.0, 0.0]), -1.0)

    def test_rejects_a_zero_vector(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero"):
            cosine_similarity([0.0, 0.0], [1.0, 0.0])


class SummaryTests(unittest.TestCase):
    def test_reports_mean_and_spread_per_system(self) -> None:
        summary = summarise_similarity({"a": 0.6, "b": 0.8})

        self.assertAlmostEqual(summary["mean"], 0.7)
        self.assertAlmostEqual(summary["min"], 0.6)
        self.assertAlmostEqual(summary["max"], 0.8)
        self.assertEqual(summary["count"], 2)

    def test_rejects_an_empty_set(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            summarise_similarity({})


if __name__ == "__main__":
    unittest.main()


class Int16ScaleTests(unittest.TestCase):
    """The tsukuyomi corpus is 96 kHz IEEE float, which the stdlib wave module cannot read.

    Reading through soundfile yields floats in [-1, 1], and the voiced-frame threshold is
    expressed in int16 units, so the two have to meet somewhere explicit.
    """

    def test_scales_unit_floats_to_int16_range(self) -> None:
        self.assertEqual(to_int16_scale([0.0, 1.0, -1.0]), [0, 32767, -32767])

    def test_rounds_to_the_nearest_integer(self) -> None:
        self.assertEqual(to_int16_scale([0.5]), [16384])

    def test_clamps_values_beyond_full_scale(self) -> None:
        self.assertEqual(to_int16_scale([1.5, -1.5]), [32767, -32768])

    def test_passes_integers_through_unchanged(self) -> None:
        self.assertEqual(to_int16_scale([0, 1000, -1000]), [0, 1000, -1000])
