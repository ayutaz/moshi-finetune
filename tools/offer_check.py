"""Judge a Vast.ai offer before renting it, on the two things that cost money this session.

Both failures were in the offer, not the code, and both were visible in the search result
before anything was rented:

- **The advertised rate is not the rate.** `dph_total` covers compute; the disk bills
  separately at `storage_cost` US$/GB/month. An offer at US$2.0896/h with 900 GB at
  US$1.00/GB/month bills US$3.3327/h. That inverted which line binds - the budget allowed
  2.91 h against a 3.376 h plan - and the instance was destroyed before training started,
  for US$0.20.
- **"A100" is two different machines.** M3 trained on A100-SXM4-80GB. M3-R rented A100 80GB
  PCIe, NCCL initialised, and both ranks then spun at 100% CPU with no I/O until the run was
  abandoned at US$4.29. A search result says `A100` for both.

So this refuses an offer whose real rate breaks the budget, and warns when a multi-GPU
training offer is not on the interconnect the working run used. It is arithmetic and string
matching, nothing else - no network, no imports beyond the standard library - so it runs
before `vastai create` rather than after.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

HOURS_PER_MONTH = 730.0

# Interconnects a 2-GPU ZeRO-3 run is known to work on here. SXM4 is what M3 used; PCIe is
# what hung. The list is what has been observed, not what is theoretically fine - an offer
# outside it gets a warning, not a refusal, because nobody has tested the rest.
KNOWN_GOOD_MULTI_GPU = ("SXM4", "SXM5", "NVLINK")


@dataclass(frozen=True)
class OfferVerdict:
    usable: bool
    hourly_rate: float
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]


def true_hourly_rate(*, dph_total: float, storage_cost: float, disk_gb: float) -> float:
    """What the instance actually bills per hour.

    `storage_cost` is US$/GB/month in a search result, so the disk contributes
    `storage_cost * disk_gb / 730`. Reading `dph_total` alone understates a 900 GB box at
    US$1.00/GB/month by US$1.23/h.
    """
    if min(dph_total, storage_cost, disk_gb) < 0:
        raise ValueError("offer figures must be non-negative")
    return dph_total + storage_cost * disk_gb / HOURS_PER_MONTH


def affordable_hours(*, spent: float, limit: float, hourly_rate: float) -> float:
    """Hours the preflight limit still allows at this rate. Zero when already past it."""
    if hourly_rate <= 0:
        raise ValueError(f"hourly_rate must be positive, got {hourly_rate}")
    return max(0.0, (limit - spent) / hourly_rate)


def interconnect_is_known_good(gpu_name: str) -> bool:
    """Has a multi-GPU run of this experiment worked on this interconnect?

    Matches on the name because that is all a search result carries. Absence of a match is
    'untested', not 'broken' - the caller warns rather than refusing.
    """
    upper = gpu_name.upper().replace("-", " ").replace("_", " ")
    return any(token in upper for token in KNOWN_GOOD_MULTI_GPU)


def check_offer(
    offer: dict[str, Any],
    *,
    spent: float,
    limit: float,
    planned_hours: float,
    num_gpus_needed: int = 1,
) -> OfferVerdict:
    """Decide whether to rent, on the real rate and the interconnect.

    Refusals are budget facts: the run does not fit in what the preflight allows. The
    interconnect is a warning because one bad experience is not a rule - but it is the one
    that cost the most, so it is never silent.
    """
    rate = true_hourly_rate(
        dph_total=float(offer.get("dph_total", 0.0)),
        storage_cost=float(offer.get("storage_cost", 0.0)),
        disk_gb=float(offer.get("disk_space", 0.0)),
    )
    reasons: list[str] = []
    warnings: list[str] = []

    hours = affordable_hours(spent=spent, limit=limit, hourly_rate=rate)
    if hours < planned_hours:
        reasons.append(
            f"US${rate:.4f}/h buys {hours:.2f} h against a {planned_hours:.2f} h plan "
            f"(spent US${spent:.3f}, limit US${limit:.2f})"
        )

    advertised = float(offer.get("dph_total", 0.0))
    # 10% is the line where the disk starts changing decisions rather than just the total.
    # The offer that had to be destroyed was 59% over its advertised rate; its replacement
    # was 6% over and caused no trouble.
    if advertised > 0 and rate > advertised * 1.10:
        warnings.append(
            f"advertised US${advertised:.4f}/h but bills US${rate:.4f}/h - "
            f"disk {offer.get('disk_space', 0):.0f} GB at US${offer.get('storage_cost', 0):.2f}/GB/mo"
        )

    gpus = int(offer.get("num_gpus", 1))
    if gpus < num_gpus_needed:
        reasons.append(f"{gpus} GPU against {num_gpus_needed} needed")
    if num_gpus_needed > 1 and not interconnect_is_known_good(str(offer.get("gpu_name", ""))):
        warnings.append(
            f"{offer.get('gpu_name', '?')} is not an interconnect a multi-GPU run has worked "
            f"on here (known good: {', '.join(KNOWN_GOOD_MULTI_GPU)}). M3-R lost US$4.29 to a "
            "collective that hung on PCIe. Smoke-test two steps before committing."
        )

    return OfferVerdict(
        usable=not reasons,
        hourly_rate=round(rate, 6),
        reasons=tuple(reasons),
        warnings=tuple(warnings),
    )
