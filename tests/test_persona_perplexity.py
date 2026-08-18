import math
import unittest

from tools.persona_perplexity import PairValidationError, summarise_scores, validate_pairs


def _pairs(count: int = 10):
    return [
        {
            "id": f"pair-{index:02d}",
            "prompt": f"質問{index}",
            "preferred": f"お答えしますわ{index}",
            "dispreferred": f"答えます{index}",
        }
        for index in range(1, count + 1)
    ]


class PersonaPairValidationTests(unittest.TestCase):
    def test_requires_exactly_ten_unique_pairs(self) -> None:
        validated = validate_pairs(_pairs())

        self.assertEqual(len(validated), 10)

    def test_wrong_pair_count_fails(self) -> None:
        with self.assertRaisesRegex(PairValidationError, "10 pairs"):
            validate_pairs(_pairs(9))

    def test_identical_candidates_fail(self) -> None:
        rows = _pairs()
        rows[0]["dispreferred"] = rows[0]["preferred"]

        with self.assertRaisesRegex(PairValidationError, "must differ"):
            validate_pairs(rows)


class PersonaScoreSummaryTests(unittest.TestCase):
    def test_reports_win_rate_margin_and_perplexity(self) -> None:
        rows = [
            {"preferred_nll": 1.0, "dispreferred_nll": 2.0},
            {"preferred_nll": 3.0, "dispreferred_nll": 2.5},
        ]

        summary = summarise_scores(rows)

        self.assertEqual(summary["pair_count"], 2)
        self.assertEqual(summary["preferred_wins"], 1)
        self.assertEqual(summary["preferred_win_rate"], 0.5)
        self.assertAlmostEqual(summary["mean_nll_margin"], 0.25)
        self.assertAlmostEqual(summary["preferred_perplexity"], math.exp(2.0))
        self.assertAlmostEqual(summary["dispreferred_perplexity"], math.exp(2.25))

    def test_reproduces_blog_total_log_probability_metric(self) -> None:
        rows = [
            {
                "preferred_nll": 1.0,
                "dispreferred_nll": 2.0,
                "preferred_logprob": -3.0,
                "dispreferred_logprob": -5.0,
            },
            {
                "preferred_nll": 3.0,
                "dispreferred_nll": 2.5,
                "preferred_logprob": -6.0,
                "dispreferred_logprob": -4.0,
            },
        ]

        summary = summarise_scores(rows)

        self.assertEqual(summary["preferred_wins"], 1)
        self.assertEqual(summary["preferred_logprob_total"], -9.0)
        self.assertEqual(summary["dispreferred_logprob_total"], -9.0)
        self.assertEqual(summary["logprob_total_difference"], 0.0)


if __name__ == "__main__":
    unittest.main()
