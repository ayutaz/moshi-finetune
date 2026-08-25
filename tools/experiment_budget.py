from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

# The cap is the user's decision and it has moved once: US$100 approved 2026-08-18,
# raised to US$125 on 2026-08-24 after the M3 verification. The thresholds below are
# fractions of it rather than absolutes, so that raising the cap does not silently leave
# this tool judging against the old one. m0/spend-ledger.json is the record; `--cap`
# overrides for a caller that has read it.
DEFAULT_HARD_CAP = 125.0
# Fractions of the cap, not absolutes. The original thresholds were 75/90/95 against a
# US$100 cap; keeping them as fractions means raising the cap moves the policy with it
# instead of leaving the old absolute numbers behind, which is the failure this had.
WARNING_FRACTION = 0.75
NEW_RUN_FRACTION = 0.90
ACTIVE_STOP_FRACTION = 0.95


@dataclass(frozen=True)
class BudgetDecision:
    status: str
    predicted_spend: float
    warning: bool
    reason: str


def evaluate_budget(
    *, spent: float, hourly_rate: float, planned_hours: float, cap: float = DEFAULT_HARD_CAP
) -> BudgetDecision:
    """Decide whether a run may start, against a cap the caller may override.

    The thresholds are fractions of the cap, not absolutes. M3 raised the cap from US$100
    to US$125 and this tool kept judging against US$100, so it returned non-zero for every
    possible plan - and a preflight that always says no is one nobody reads.
    """
    if min(spent, hourly_rate, planned_hours) < 0:
        raise ValueError("budget inputs must be non-negative")
    if cap <= 0:
        raise ValueError(f"cap must be positive, got {cap}")

    warning_spend = cap * WARNING_FRACTION
    new_run_limit = cap * NEW_RUN_FRACTION
    active_stop = cap * ACTIVE_STOP_FRACTION

    predicted_spend = round(spent + hourly_rate * planned_hours, 6)
    if spent > cap:
        return BudgetDecision(
            "hard-cap-exceeded", predicted_spend, True, f"actual spend exceeds US${cap:g}"
        )
    if spent >= active_stop:
        return BudgetDecision(
            "stop-active-work",
            predicted_spend,
            True,
            f"actual spend reached the US${active_stop:g} safety stop",
        )
    if predicted_spend > new_run_limit:
        return BudgetDecision(
            "reject-new-run",
            predicted_spend,
            True,
            f"predicted cumulative spend exceeds the US${new_run_limit:g} new-run limit",
        )
    if max(spent, predicted_spend) >= warning_spend:
        return BudgetDecision(
            "allow-with-warning",
            predicted_spend,
            True,
            f"US${warning_spend:g} warning threshold reached; re-estimate remaining work",
        )
    return BudgetDecision("allow", predicted_spend, False, "within budget")


def main() -> int:
    parser = argparse.ArgumentParser(description="Enforce the Vast.ai experiment budget policy")
    parser.add_argument("--spent", type=float, required=True)
    parser.add_argument("--hourly-rate", type=float, required=True)
    parser.add_argument("--planned-hours", type=float, required=True)
    parser.add_argument(
        "--cap",
        type=float,
        default=DEFAULT_HARD_CAP,
        help="Experiment cap in USD; read it from m0/spend-ledger.json experiment_cap",
    )
    args = parser.parse_args()
    decision = evaluate_budget(
        spent=args.spent,
        hourly_rate=args.hourly_rate,
        planned_hours=args.planned_hours,
        cap=args.cap,
    )
    print(json.dumps(asdict(decision), sort_keys=True))
    return 0 if decision.status in {"allow", "allow-with-warning"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
