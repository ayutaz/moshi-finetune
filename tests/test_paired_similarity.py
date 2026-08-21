import json
import unittest
from pathlib import Path

from tools.speaker_similarity import paired_comparison

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
M2_GATE = REPOSITORY_ROOT / "experiments" / "tsukuyomi_ojousama" / "reports" / "m2-tts-gate.json"
# The per-file scores live under data/, which is gitignored: they derive from audio that may
# not be redistributed. The committed gate carries only the summary they roll up to.
M2_SCORES = (
    REPOSITORY_ROOT
    / "data"
    / "experiments"
    / "tsukuyomi_ojousama"
    / "m2"
    / "speaker-similarity.json"
)


class PairedComparisonTests(unittest.TestCase):
    def test_it_pairs_by_name_and_counts_wins(self) -> None:
        result = paired_comparison(
            {"a": 0.5, "b": 0.7, "c": 0.4},
            {"a": 0.6, "b": 0.6, "c": 0.9},
            names=("base", "candidate"),
        )
        self.assertEqual(result["pairs"], 3)
        self.assertEqual(result["higher_on"], 2)
        self.assertAlmostEqual(result["mean_delta"], (0.1 - 0.1 + 0.5) / 3)

    def test_a_tie_counts_for_neither_side(self) -> None:
        result = paired_comparison({"a": 0.5}, {"a": 0.5}, names=("base", "candidate"))
        self.assertEqual(result["higher_on"], 0)
        self.assertEqual(result["ties"], 1)

    def test_mismatched_keys_are_rejected_rather_than_intersected(self) -> None:
        # Silently comparing whichever names happen to overlap would compare two systems on
        # different sentences and report it as paired.
        with self.assertRaises(ValueError):
            paired_comparison({"a": 0.5, "b": 0.5}, {"a": 0.5}, names=("base", "candidate"))

    def test_an_empty_comparison_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            paired_comparison({}, {}, names=("base", "candidate"))


class M2ReplayTests(unittest.TestCase):
    """Reproduce M2's committed numbers exactly, or the statistic is not the one M2 was judged on."""

    def test_the_committed_m2_paired_block_is_reproduced_to_the_digit(self) -> None:
        expected = json.loads(M2_GATE.read_text(encoding="utf-8"))["speaker_similarity"][
            "paired_comparison"
        ]
        if not M2_SCORES.is_file():
            self.skipTest(f"{M2_SCORES} is gitignored and absent on this host")
        systems = json.loads(M2_SCORES.read_text(encoding="utf-8"))["systems"]

        result = paired_comparison(
            systems["T0_zero_shot"]["per_file"],
            systems["T1_speaker_inversion"]["per_file"],
            names=("T0_zero_shot", "T1_speaker_inversion"),
        )
        self.assertEqual(result["pairs"], expected["pairs"])
        self.assertEqual(result["higher_on"], expected["t1_higher_on"])
        self.assertEqual(result["mean_delta"], expected["mean_delta"])
        self.assertEqual(result["median_delta"], expected["median_delta"])
        self.assertEqual(result["min_delta"], expected["min_delta"])
        self.assertEqual(result["max_delta"], expected["max_delta"])
        self.assertEqual(result["sign_test_p_two_sided"], expected["sign_test_p_two_sided"])


if __name__ == "__main__":
    unittest.main()
