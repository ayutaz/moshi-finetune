import json
import math
import unittest
from pathlib import Path

from tools.speaker_similarity import (
    calibration_band,
    centroid,
    cosine_similarity,
    full_set_delta_vector,
    leave_one_out_similarity,
    require_likeness_report,
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


class CentroidTests(unittest.TestCase):
    def test_averages_element_wise(self) -> None:
        self.assertEqual(centroid([[0.0, 2.0], [2.0, 4.0]]), [1.0, 3.0])

    def test_a_single_vector_is_its_own_centroid(self) -> None:
        self.assertEqual(centroid([[1.0, -1.0]]), [1.0, -1.0])

    def test_rejects_an_empty_set(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            centroid([])

    def test_rejects_ragged_embeddings(self) -> None:
        # Averaging different widths would silently compare different embedding models.
        with self.assertRaisesRegex(ValueError, "widths"):
            centroid([[1.0, 2.0], [1.0]])


class LeaveOneOutTests(unittest.TestCase):
    """The calibration band: what the target speaker scores against her own centroid.

    CLAUDE.md states the rule this exists for - a within-group similarity of 0.74 means
    nothing until you know a real human scores 0.70. M3 decided condition 4 with no band,
    so "+0.032 better than control" had no scale to be better on.
    """

    def test_scores_each_recording_against_the_others(self) -> None:
        # 'c' is the odd one out, so it must score lowest against the rest.
        scores = leave_one_out_similarity({"a": [1.0, 0.0], "b": [1.0, 0.05], "c": [0.0, 1.0]})
        self.assertEqual(set(scores), {"a", "b", "c"})
        self.assertLess(scores["c"], scores["a"])

    def test_a_recording_is_never_part_of_its_own_reference(self) -> None:
        # Against the full centroid every clip is compared partly with itself and scores
        # high for that reason alone, which would inflate the band the arms are read against.
        embeddings = {"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [1.0, 1.0]}
        self.assertAlmostEqual(
            leave_one_out_similarity(embeddings)["a"],
            cosine_similarity([1.0, 0.0], centroid([embeddings["b"], embeddings["c"]])),
        )

    def test_two_recordings_are_not_a_band(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 3"):
            leave_one_out_similarity({"a": [1.0, 0.0], "b": [0.0, 1.0]})


class CalibrationBandTests(unittest.TestCase):
    def test_names_the_worst_real_recording_as_the_floor(self) -> None:
        band = calibration_band({"a": 0.9, "b": 0.8, "c": 0.74})

        self.assertAlmostEqual(band["floor"], 0.74)
        self.assertAlmostEqual(band["mean"], 0.8133333, places=6)
        self.assertEqual(band["count"], 3)
        self.assertEqual(set(band["per_clip"]), {"a", "b", "c"})


class FullSetDeltaVectorTests(unittest.TestCase):
    """An interval must be computed over the same clips as the mean it belongs to."""

    def test_charges_zero_for_a_clip_the_candidate_could_not_produce(self) -> None:
        self.assertEqual(
            full_set_delta_vector({"a": 0.1, "c": -0.2}, ("a", "b", "c")), [0.1, 0.0, -0.2]
        )

    def test_follows_the_order_of_the_fixed_set(self) -> None:
        self.assertEqual(full_set_delta_vector({"b": 0.5}, ("b", "a")), [0.5, 0.0])

    def test_a_delta_outside_the_fixed_set_is_an_error(self) -> None:
        # It would mean the mean and the interval were computed over different clips.
        with self.assertRaisesRegex(ValueError, "outside the fixed set"):
            full_set_delta_vector({"z": 0.1}, ("a", "b"))

    def test_an_empty_fixed_set_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            full_set_delta_vector({}, ())


class RequireLikenessReportTests(unittest.TestCase):
    """A report missing the band, the absolute cosine or the per-clip spread cannot ship.

    M3's condition-4 report had none of the three, which is why "+0.032" could be argued
    about for a week with nobody able to say how far 0.032 was from the target speaker.
    """

    BAND = {"count": 10, "mean": 0.81, "median": 0.81, "min": 0.74, "max": 0.87, "floor": 0.74}

    def comparison(self, **overrides):
        report = {
            "report_kind": "comparison",
            "calibration_band": dict(self.BAND),
            "absolute_cosine": {"control": {"summary": {}}, "arm": {"summary": {}}},
            "comparisons": {
                "arm": {
                    "systems": ["control", "arm"],
                    "per_clip_delta": {"0": 0.1},
                    "per_clip_absolute": {"0": {"base": 0.5, "candidate": 0.6, "delta": 0.1}},
                    "delta_stdev_full_set": 0.0,
                }
            },
        }
        report.update(overrides)
        return report

    def test_a_complete_comparison_report_passes(self) -> None:
        require_likeness_report(self.comparison())

    def test_a_missing_calibration_band_is_refused(self) -> None:
        report = self.comparison()
        del report["calibration_band"]
        with self.assertRaisesRegex(ValueError, "calibration_band"):
            require_likeness_report(report)

    def test_a_band_without_its_floor_is_refused(self) -> None:
        report = self.comparison()
        report["calibration_band"] = {k: v for k, v in self.BAND.items() if k != "floor"}
        with self.assertRaisesRegex(ValueError, "floor"):
            require_likeness_report(report)

    def test_a_missing_absolute_cosine_is_refused(self) -> None:
        report = self.comparison(absolute_cosine={})
        with self.assertRaisesRegex(ValueError, "absolute_cosine"):
            require_likeness_report(report)

    def test_an_arm_scored_but_not_reported_in_absolute_terms_is_refused(self) -> None:
        report = self.comparison(absolute_cosine={"control": {"summary": {}}})
        with self.assertRaisesRegex(ValueError, "absolute_cosine.arm"):
            require_likeness_report(report)

    def test_a_comparison_without_per_clip_spread_is_refused(self) -> None:
        report = self.comparison()
        report["comparisons"]["arm"]["delta_stdev_full_set"] = None
        with self.assertRaisesRegex(ValueError, "delta_stdev_full_set"):
            require_likeness_report(report)

    def test_a_comparison_without_per_clip_deltas_is_refused(self) -> None:
        report = self.comparison()
        report["comparisons"]["arm"]["per_clip_delta"] = {}
        with self.assertRaisesRegex(ValueError, "per_clip_delta"):
            require_likeness_report(report)

    def test_a_calibration_report_needs_only_the_band(self) -> None:
        require_likeness_report({"report_kind": "calibration", "calibration_band": dict(self.BAND)})

    def test_a_comparison_cannot_relabel_itself_to_escape_the_checks(self) -> None:
        # The one way the validator could be bypassed: carry comparisons, claim to be a
        # calibration file, and skip the absolute-cosine and per-clip requirements.
        report = self.comparison(report_kind="calibration")
        with self.assertRaisesRegex(ValueError, "declares report_kind"):
            require_likeness_report(report)

    def test_an_unknown_report_kind_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "report_kind"):
            require_likeness_report({"calibration_band": dict(self.BAND)})


class ShippedCalibrationFileTests(unittest.TestCase):
    """The band condition 4 is read against has to stay on disk and stay complete."""

    PATH = (
        Path(__file__).resolve().parents[1]
        / "experiments/tsukuyomi_ojousama/reports/m3-likeness-calibration.json"
    )

    def test_the_calibration_report_is_present_and_valid(self) -> None:
        require_likeness_report(json.loads(self.PATH.read_text(encoding="utf-8")))

    def test_the_band_is_the_measured_one(self) -> None:
        # Measured 2026-08-25 on the 10 test-split recordings through the same ECAPA path
        # the arms are scored on. If this moves, the preparation changed and every arm's
        # number has to be recomputed before it can be compared with the band.
        band = json.loads(self.PATH.read_text(encoding="utf-8"))["calibration_band"]
        self.assertEqual(band["count"], 10)
        self.assertAlmostEqual(band["mean"], 0.8166, places=4)
        self.assertAlmostEqual(band["median"], 0.8175, places=4)
        self.assertAlmostEqual(band["floor"], 0.7405, places=4)
        self.assertAlmostEqual(band["max"], 0.8780, places=4)


if __name__ == "__main__":
    unittest.main()
