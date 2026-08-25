import unittest

from tools.experiment_budget import (
    ACTIVE_STOP_FRACTION,
    DEFAULT_HARD_CAP,
    NEW_RUN_FRACTION,
    WARNING_FRACTION,
    BudgetDecision,
    evaluate_budget,
)


class ExperimentBudgetTests(unittest.TestCase):
    """The thresholds are fractions of the cap, so the policy moves when the cap does.

    They were absolutes once. When the cap went from US$100 to US$125 the absolutes stayed
    behind, and the tool returned non-zero for every possible plan - a preflight that always
    says no is one nobody reads. Each test below therefore names the cap it judges against.
    """

    def test_allows_a_run_below_the_new_run_limit(self) -> None:
        decision = evaluate_budget(spent=10.0, hourly_rate=2.5, planned_hours=4.0, cap=100.0)

        self.assertEqual(decision, BudgetDecision("allow", 20.0, False, "within budget"))

    def test_warns_at_three_quarters_of_the_cap(self) -> None:
        decision = evaluate_budget(spent=74.0, hourly_rate=1.0, planned_hours=2.0, cap=100.0)

        self.assertEqual(decision.status, "allow-with-warning")
        self.assertTrue(decision.warning)

    def test_rejects_a_new_run_predicted_past_nine_tenths_of_the_cap(self) -> None:
        decision = evaluate_budget(spent=80.0, hourly_rate=3.0, planned_hours=4.0, cap=100.0)

        self.assertEqual(decision.status, "reject-new-run")
        self.assertEqual(decision.predicted_spend, 92.0)

    def test_stops_active_work_at_nineteen_twentieths_of_the_cap(self) -> None:
        decision = evaluate_budget(spent=95.0, hourly_rate=0.0, planned_hours=0.0, cap=100.0)

        self.assertEqual(decision.status, "stop-active-work")

    def test_never_allows_values_above_the_cap(self) -> None:
        decision = evaluate_budget(spent=101.0, hourly_rate=0.0, planned_hours=0.0, cap=100.0)

        self.assertEqual(decision.status, "hard-cap-exceeded")


class CapIsAParameterTests(unittest.TestCase):
    """Raising the cap has to move every threshold, not only the last one."""

    def test_the_same_spend_reads_differently_under_the_two_caps(self) -> None:
        # US$96 is past the old US$100 cap's safety stop and inside the raised one.
        self.assertEqual(
            evaluate_budget(spent=96.0, hourly_rate=0.0, planned_hours=0.0, cap=100.0).status,
            "stop-active-work",
        )
        self.assertEqual(
            evaluate_budget(spent=96.0, hourly_rate=0.0, planned_hours=0.0, cap=125.0).status,
            "allow-with-warning",
        )

    def test_the_default_cap_is_the_approved_one(self) -> None:
        # US$125, approved 2026-08-24. m0/spend-ledger.json is the record.
        self.assertEqual(DEFAULT_HARD_CAP, 125.0)

    def test_the_fractions_are_the_original_policy(self) -> None:
        # 75 / 90 / 95 against the original US$100 cap.
        self.assertEqual(
            (WARNING_FRACTION, NEW_RUN_FRACTION, ACTIVE_STOP_FRACTION), (0.75, 0.90, 0.95)
        )

    def test_a_non_positive_cap_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_budget(spent=0.0, hourly_rate=0.0, planned_hours=0.0, cap=0.0)

    def test_the_m3r_phase4_plan_is_rejected_under_the_approved_cap(self) -> None:
        # Recorded because it is a live decision, not a hypothetical: US$102.697 accrued,
        # the phase-4 estimate is 6.0 h at US$3.0567/h, and that lands at US$121.04 -
        # inside the US$125 cap but past the US$112.50 new-run limit. The preflight says no
        # and the plan has to be split or re-authorised. Weakening this test to make the
        # plan pass is the failure it exists to prevent.
        decision = evaluate_budget(spent=102.697, hourly_rate=3.0566666, planned_hours=6.0)

        self.assertEqual(decision.status, "reject-new-run")

    def test_the_forward_probe_alone_is_allowed(self) -> None:
        decision = evaluate_budget(spent=102.697, hourly_rate=0.35, planned_hours=0.5)

        self.assertEqual(decision.status, "allow-with-warning")


if __name__ == "__main__":
    unittest.main()
