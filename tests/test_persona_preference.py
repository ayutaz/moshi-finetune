import math
import unittest

from tools.persona_perplexity import ScoringError, assert_scores_discriminate, summarise_scores


def _row(pref_nll, disp_nll, pref_lp, disp_lp):
    return {
        "preferred_nll": pref_nll,
        "dispreferred_nll": disp_nll,
        "preferred_logprob": pref_lp,
        "dispreferred_logprob": disp_lp,
    }


class LengthNormalisedPreferenceTests(unittest.TestCase):
    """Total log-probability penalises the longer candidate.

    The persona pairs put the longer candidate on the preferred side in 6 of 10 cases
    (「ですわ」 against 「ですね」), so summing log-probabilities scored the persona as
    losing while per-token NLL scored it as winning. The comparison is per-token.
    """

    def test_counts_a_win_when_the_preferred_candidate_has_lower_mean_nll(self) -> None:
        summary = summarise_scores([_row(12.0, 13.0, -36.0, -26.0)])

        self.assertEqual(summary["preferred_wins"], 1)
        self.assertEqual(summary["preferred_win_rate"], 1.0)

    def test_a_longer_preferred_candidate_still_wins_on_mean_nll(self) -> None:
        # three tokens at 12.0 against two at 13.0: worse total, better per token
        summary = summarise_scores([_row(12.0, 13.0, -36.0, -26.0)])

        self.assertEqual(summary["preferred_wins"], 1)
        self.assertEqual(summary["preferred_wins_by_total_logprob"], 0)

    def test_reports_both_criteria_side_by_side(self) -> None:
        rows = [_row(12.0, 13.0, -36.0, -26.0), _row(15.0, 14.0, -30.0, -42.0)]

        summary = summarise_scores(rows)

        self.assertEqual(summary["preferred_wins"], 1)
        self.assertEqual(summary["preferred_wins_by_total_logprob"], 1)
        self.assertEqual(summary["win_criterion"], "mean token NLL")

    def test_still_reports_the_margin_and_perplexities(self) -> None:
        summary = summarise_scores([_row(12.0, 13.0, -36.0, -26.0)])

        self.assertAlmostEqual(summary["mean_nll_margin"], 1.0)
        self.assertAlmostEqual(summary["preferred_perplexity"], math.exp(12.0))


class ScoreValidityTests(unittest.TestCase):
    """A paired comparison is broken when the candidates cannot move the score.

    Reading the wrong logit positions would give both candidates the same number, which
    is the failure worth refusing. Absolute NLL is not the test: the conditioning audio
    is a different utterance from the scored text, so it is high by construction.
    """

    def test_accepts_scores_that_separate_the_candidates(self) -> None:
        assert_scores_discriminate([_row(12.0, 13.0, -36.0, -26.0)])

    def test_rejects_identical_scores_for_every_pair(self) -> None:
        rows = [_row(12.0, 12.0, -36.0, -36.0), _row(9.0, 9.0, -27.0, -27.0)]

        with self.assertRaisesRegex(ScoringError, "identical"):
            assert_scores_discriminate(rows)

    def test_rejects_a_non_finite_score(self) -> None:
        with self.assertRaisesRegex(ScoringError, "finite"):
            assert_scores_discriminate([_row(float("nan"), 13.0, -36.0, -26.0)])


if __name__ == "__main__":
    unittest.main()
