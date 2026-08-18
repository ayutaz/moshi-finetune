import unittest

from tools.experiment_budget import BudgetDecision, evaluate_budget


class ExperimentBudgetTests(unittest.TestCase):
    def test_allows_a_run_below_the_new_run_limit(self) -> None:
        decision = evaluate_budget(spent=10.0, hourly_rate=2.5, planned_hours=4.0)

        self.assertEqual(decision, BudgetDecision("allow", 20.0, False, "within budget"))

    def test_warns_after_75_dollars(self) -> None:
        decision = evaluate_budget(spent=74.0, hourly_rate=1.0, planned_hours=2.0)

        self.assertEqual(decision.status, "allow-with-warning")
        self.assertTrue(decision.warning)

    def test_rejects_a_new_run_predicted_to_exceed_90_dollars(self) -> None:
        decision = evaluate_budget(spent=80.0, hourly_rate=3.0, planned_hours=4.0)

        self.assertEqual(decision.status, "reject-new-run")
        self.assertEqual(decision.predicted_spend, 92.0)

    def test_stops_active_work_at_95_dollars(self) -> None:
        decision = evaluate_budget(spent=95.0, hourly_rate=0.0, planned_hours=0.0)

        self.assertEqual(decision.status, "stop-active-work")

    def test_never_allows_values_above_hard_cap(self) -> None:
        decision = evaluate_budget(spent=101.0, hourly_rate=0.0, planned_hours=0.0)

        self.assertEqual(decision.status, "hard-cap-exceeded")


if __name__ == "__main__":
    unittest.main()
