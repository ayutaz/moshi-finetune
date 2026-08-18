from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass


WARNING_SPEND = 75.0
NEW_RUN_LIMIT = 90.0
ACTIVE_STOP = 95.0
HARD_CAP = 100.0


@dataclass(frozen=True)
class BudgetDecision:
    status: str
    predicted_spend: float
    warning: bool
    reason: str


def evaluate_budget(*, spent: float, hourly_rate: float, planned_hours: float) -> BudgetDecision:
    if min(spent, hourly_rate, planned_hours) < 0:
        raise ValueError("budget inputs must be non-negative")
    predicted_spend = round(spent + hourly_rate * planned_hours, 6)
    if spent > HARD_CAP:
        return BudgetDecision(
            "hard-cap-exceeded", predicted_spend, True, "actual spend exceeds US$100"
        )
    if spent >= ACTIVE_STOP:
        return BudgetDecision(
            "stop-active-work", predicted_spend, True, "actual spend reached US$95 safety stop"
        )
    if predicted_spend > NEW_RUN_LIMIT:
        return BudgetDecision(
            "reject-new-run",
            predicted_spend,
            True,
            "predicted cumulative spend exceeds US$90 new-run limit",
        )
    if max(spent, predicted_spend) >= WARNING_SPEND:
        return BudgetDecision(
            "allow-with-warning",
            predicted_spend,
            True,
            "US$75 warning threshold reached; re-estimate remaining work",
        )
    return BudgetDecision("allow", predicted_spend, False, "within budget")


def main() -> int:
    parser = argparse.ArgumentParser(description="Enforce the Vast.ai experiment budget policy")
    parser.add_argument("--spent", type=float, required=True)
    parser.add_argument("--hourly-rate", type=float, required=True)
    parser.add_argument("--planned-hours", type=float, required=True)
    args = parser.parse_args()
    decision = evaluate_budget(
        spent=args.spent,
        hourly_rate=args.hourly_rate,
        planned_hours=args.planned_hours,
    )
    print(json.dumps(asdict(decision), sort_keys=True))
    return 0 if decision.status in {"allow", "allow-with-warning"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
