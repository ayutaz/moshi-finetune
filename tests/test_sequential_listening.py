import unittest

from tools.build_listening_page import pairs_needed_for_significance, sequential_verdict


class PairsNeededTests(unittest.TestCase):
    """A listener should know how short the pass can be before starting it.

    The M2 pass judged 6 pairs out of 30 and stopped, which turned out to be the right
    call: an unbroken run reaches two-sided p < 0.05 at six non-tie judgements, so the
    other 24 could never have been decisive on their own.
    """

    def test_six_straight_wins_settle_it_at_five_percent(self) -> None:
        self.assertEqual(pairs_needed_for_significance(0.05), 6)

    def test_a_stricter_threshold_needs_more(self) -> None:
        self.assertEqual(pairs_needed_for_significance(0.01), 8)

    def test_rejects_an_impossible_threshold(self) -> None:
        with self.assertRaisesRegex(ValueError, "alpha"):
            pairs_needed_for_significance(0)


class SequentialVerdictTests(unittest.TestCase):
    def test_keeps_going_while_undecided(self) -> None:
        verdict = sequential_verdict(wins=2, losses=0, ties=1, remaining=27, alpha=0.05)

        self.assertEqual(verdict["state"], "continue")
        self.assertEqual(verdict["non_tie"], 2)

    def test_settles_once_the_sign_test_clears_alpha(self) -> None:
        verdict = sequential_verdict(wins=6, losses=0, ties=0, remaining=24, alpha=0.05)

        self.assertEqual(verdict["state"], "settled-better")
        self.assertLess(verdict["p_two_sided"], 0.05)

    def test_settles_against_the_adapted_system_too(self) -> None:
        verdict = sequential_verdict(wins=0, losses=6, ties=0, remaining=24, alpha=0.05)

        self.assertEqual(verdict["state"], "settled-worse")

    def test_reports_how_many_more_wins_would_settle_it(self) -> None:
        verdict = sequential_verdict(wins=4, losses=0, ties=2, remaining=24, alpha=0.05)

        self.assertEqual(verdict["state"], "continue")
        self.assertEqual(verdict["wins_still_needed"], 2)

    def test_calls_it_exhausted_when_nothing_is_left_to_judge(self) -> None:
        verdict = sequential_verdict(wins=3, losses=2, ties=0, remaining=0, alpha=0.05)

        self.assertEqual(verdict["state"], "exhausted")

    def test_a_run_broken_by_a_loss_needs_more_than_the_minimum(self) -> None:
        clean = sequential_verdict(wins=4, losses=0, ties=0, remaining=26, alpha=0.05)
        broken = sequential_verdict(wins=4, losses=1, ties=0, remaining=25, alpha=0.05)

        self.assertGreater(broken["wins_still_needed"], clean["wins_still_needed"])

    def test_ties_do_not_count_toward_the_sign_test(self) -> None:
        with_ties = sequential_verdict(wins=6, losses=0, ties=5, remaining=19, alpha=0.05)

        self.assertEqual(with_ties["non_tie"], 6)
        self.assertEqual(with_ties["state"], "settled-better")


if __name__ == "__main__":
    unittest.main()
