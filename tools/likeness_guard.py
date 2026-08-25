"""Stop speaker similarity from rewarding the collapse M3 exists to reject.

`tools/speaker_similarity.py` says the hazard in its own docstring: a flat, over-smoothed
rendering can score higher than one a listener finds closer to the speaker. Its preparation
makes that worse by design - `voiced_segments` keeps only sustained voiced energy and
throws away everything else, so a checkpoint that has degraded into a short, steady hum is
embedded from its most speaker-like fragment while a healthy one is embedded from varied
speech.

M3 measured that degradation directly. Across all three prompt sets, v-real goes from 0 to
24 silent generations out of 50 between epochs 1 and 5. Scoring those arms on similarity
alone would read their collapse as an improvement.

Three defences, each of which the M3 critique showed was needed:

1. A voiced-duration floor, so a clip too short to carry identity is not scored as if it did.
2. A degeneracy guard, so a clip from a generation the collapse detector already flagged is
   withheld from the likeness statistic rather than counted as a win.
3. An interlock, so no arm passes condition 4 while failing condition 3. Conditions 3 and 4
   were computed into separate files that never referenced each other, which is how an arm
   could be reported as more speaker-like *because* it had stopped speaking.
4. A criterion that can tell a null result from an unmeasured one (added 2026-08-25). The
   original rule - higher on 8 of 10 clips - has power 0.383 against a true win rate of
   0.70, so it rejected v-tts/epoch3 at 5 of 10 while that same arm cleared the effect-size
   bar it was supposed to be measured on. Magnitudes, an interval, and a calibration band
   replaced the sign count. See `condition4_verdict`.

Everything here is pure and imports nothing outside the standard library.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

# 25 frames of 480 samples is 0.5 s at 24 kHz. Below that there is not enough voiced audio
# for a speaker embedding to mean anything, whatever number comes back.
DEFAULT_MIN_VOICED_FRAMES = 25

# Condition 4's magnitude bar, in units of the gap between the control and the target
# speaker's own calibration band. See `band_closure` and m3/DATASET_SPEC.md for why a
# fraction of a measured gap replaced the bare +0.02 the old spec used.
DEFAULT_MIN_BAND_CLOSURE = 0.25

DEFAULT_CONFIDENCE = 0.95
DEFAULT_BOOTSTRAP_ITERATIONS = 10000
# Recorded, not incidental: M3's counts were each a single unrecorded draw, and an interval
# that moves between runs of the same data is not an interval.
DEFAULT_BOOTSTRAP_SEED = 20260824

# Two-sided Student-t critical values. scipy is not a dependency of this repository and a
# gate that only runs where scipy is installed is a gate that will one day be skipped.
_T_CRITICAL: dict[float, dict[int, float]] = {
    0.95: {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        21: 2.080,
        22: 2.074,
        23: 2.069,
        24: 2.064,
        25: 2.060,
        26: 2.056,
        27: 2.052,
        28: 2.048,
        29: 2.045,
        30: 2.042,
        40: 2.021,
        50: 2.009,
        60: 2.000,
        80: 1.990,
        100: 1.984,
        120: 1.980,
    },
    0.90: {
        1: 6.314,
        2: 2.920,
        3: 2.353,
        4: 2.132,
        5: 2.015,
        6: 1.943,
        7: 1.895,
        8: 1.860,
        9: 1.833,
        10: 1.812,
        11: 1.796,
        12: 1.782,
        13: 1.771,
        14: 1.761,
        15: 1.753,
        16: 1.746,
        17: 1.740,
        18: 1.734,
        19: 1.729,
        20: 1.725,
        21: 1.721,
        22: 1.717,
        23: 1.714,
        24: 1.711,
        25: 1.708,
        26: 1.706,
        27: 1.703,
        28: 1.701,
        29: 1.699,
        30: 1.697,
        40: 1.684,
        50: 1.676,
        60: 1.671,
        80: 1.664,
        100: 1.660,
        120: 1.658,
    },
}
_T_ASYMPTOTIC = {0.95: 1.960, 0.90: 1.645}


def voiced_frames_floor(*, voiced_samples: int, frame: int, minimum_frames: int) -> bool:
    """Does this clip carry enough voiced audio to be worth embedding?

    Expressed in frames rather than samples because `voiced_segments` extends its output by
    whole frame-sized windows, so a sample-count test can only ever land on multiples of the
    frame and reads as a threshold that is never actually crossed.
    """
    if frame < 1:
        raise ValueError(f"frame must be at least 1, got {frame}")
    return voiced_samples // frame >= minimum_frames


def apply_degeneracy_guard(
    scores: dict[str, float], flags: dict[str, dict[str, bool]]
) -> tuple[dict[str, float], dict[str, str]]:
    """Withhold similarity scores whose generation the collapse detector already flagged.

    Returns the kept scores and, separately, what was withheld and why - so a report cannot
    quietly shrink its denominator. A score with no flags raises rather than passing, since
    an unflagged clip is the one way the guard could be bypassed.
    """
    kept: dict[str, float] = {}
    withheld: dict[str, str] = {}
    for key, score in scores.items():
        flag = flags.get(key)
        if flag is None:
            raise ValueError(f"no collapse flags for {key}; the guard cannot be applied")
        if flag.get("silent"):
            withheld[key] = "silent"
        elif flag.get("exact_repeat_collapse"):
            withheld[key] = "exact_repeat_collapse"
        elif flag.get("monologue_loop"):
            withheld[key] = "monologue_loop"
        else:
            kept[key] = score
    return kept, withheld


def student_t_critical(*, df: int, confidence: float = DEFAULT_CONFIDENCE) -> float:
    """Two-sided Student-t critical value, from a table rather than from scipy.

    Between tabulated degrees of freedom the next *lower* df is used, so the interval is
    never narrower than the true one. Only the confidence levels this project actually
    quotes are supported: an unsupported level raises instead of silently returning a
    number that is wrong for it.
    """
    if df < 1:
        raise ValueError(f"df must be at least 1, got {df}")
    table = _T_CRITICAL.get(confidence)
    if table is None:
        raise ValueError(
            f"no critical values tabulated for confidence {confidence}; "
            f"supported: {sorted(_T_CRITICAL)}"
        )
    candidates = [key for key in table if key <= df]
    if not candidates:
        raise ValueError(f"df {df} is below every tabulated value")
    if df > max(table):
        return _T_ASYMPTOTIC[confidence]
    return table[max(candidates)]


def paired_mean_interval(
    deltas: Sequence[float],
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Interval estimate for a paired mean delta, computed two ways and reported as both.

    This replaces the sign test, which was the actual defect in condition 4. A sign test on
    ten clips discards every magnitude and keeps only the direction; at a true win rate of
    0.70 its power is 0.383, so "did not reach 8 of 10" and "there is no effect" are the
    same output. An interval on the magnitudes separates them: a mean whose interval clears
    zero is an effect, and a mean whose interval straddles zero with ten clips is an
    underpowered measurement that says so in its width.

    Both a t interval and a percentile bootstrap are computed, and the gate uses whichever
    lower bound is lower. Reporting one of two after seeing them is choosing a result; using
    the more conservative of the two by rule is not, and the disagreement between them is
    itself informative at n = 10.

    The bootstrap seed is part of the output. An interval that moves between runs of the
    same numbers is not an interval, and M3's counts were each a single unrecorded draw.
    """
    values = [float(value) for value in deltas]
    if len(values) < 2:
        raise ValueError(f"an interval needs at least two clips, got {len(values)}")

    mean = statistics.fmean(values)
    stdev = statistics.stdev(values)
    stderr = stdev / math.sqrt(len(values))
    critical = student_t_critical(df=len(values) - 1, confidence=confidence)

    rng = random.Random(seed)
    n = len(values)
    resampled = sorted(
        statistics.fmean([values[rng.randrange(n)] for _ in range(n)])
        for _ in range(bootstrap_iterations)
    )
    tail = (1.0 - confidence) / 2.0
    low_index = max(0, min(len(resampled) - 1, int(math.floor(tail * len(resampled)))))
    high_index = max(0, min(len(resampled) - 1, int(math.ceil((1.0 - tail) * len(resampled))) - 1))

    t_low = mean - critical * stderr
    boot_low = resampled[low_index]
    return {
        "n": n,
        "mean": mean,
        "stdev": stdev,
        "stderr": stderr,
        "confidence": confidence,
        "t": {"critical": critical, "low": t_low, "high": mean + critical * stderr},
        "bootstrap": {
            "iterations": bootstrap_iterations,
            "seed": seed,
            "low": boot_low,
            "high": resampled[high_index],
        },
        "lower_bound": min(t_low, boot_low),
        "lower_bound_method": "t" if t_low <= boot_low else "bootstrap",
        "methods_agree_on_sign": (t_low > 0) == (boot_low > 0),
    }


def band_closure(
    *, control_mean: float, candidate_mean: float, band_mean: float, band_floor: float
) -> dict[str, Any]:
    """How much of the distance from the control to the target speaker's own band was closed.

    The old magnitude bar was `mean_delta >= +0.02`, an absolute number with no denominator.
    +0.02 is most of the way there if the control sits 0.03 below the band and a rounding
    error if it sits 0.4 below, and nothing in the M3 report distinguished those cases
    because it never recorded an absolute cosine.

    `closed` = (candidate - control) / (band_mean - control). 1.0 means the arm reached the
    similarity the target speaker's own recordings average against her centroid.

    Two edges have to be handled rather than divided through:

    - The gap can be zero or negative, when the control already sits at or above the band.
      There is then no distance to close and the ratio is not a number; it is reported as
      None with the reason, and the caller must fall back to a no-regression test.
    - `closed` can exceed 1 or go negative. Both are real and are reported as they are, not
      clamped: an arm past the band mean and an arm that moved away from the speaker are
      different results and should not be flattened to 1.0 and 0.0.
    """
    headroom = band_mean - control_mean
    closed: float | None
    reason: str | None
    if headroom > 0:
        closed = (candidate_mean - control_mean) / headroom
        reason = None
    else:
        closed = None
        reason = (
            f"the control mean {control_mean:.4f} is already at or above the band mean "
            f"{band_mean:.4f}, so there is no gap to close"
        )
    return {
        "control_mean": control_mean,
        "candidate_mean": candidate_mean,
        "band_mean": band_mean,
        "band_floor": band_floor,
        "headroom": headroom,
        "closed": closed,
        "closed_percent": None if closed is None else 100.0 * closed,
        "undefined_reason": reason,
        "control_within_band": control_mean >= band_floor,
        "candidate_within_band": candidate_mean >= band_floor,
        "candidate_exceeds_band_mean": candidate_mean >= band_mean,
    }


def condition4_verdict(
    *,
    paired_deltas: Sequence[float] | Mapping[str, float],
    denominator: int,
    control_mean: float,
    candidate_mean: float,
    band: Mapping[str, float],
    collapse: Mapping[str, Any],
    memorisation: str,
    min_band_closure: float = DEFAULT_MIN_BAND_CLOSURE,
    confidence: float = DEFAULT_CONFIDENCE,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Decide condition 4 on magnitudes and a calibration band, with condition 3 wired in.

    Four criteria, all of which must hold:

    1. **Interlock.** No generation of this arm is degenerate. ECAPA scores a flat, mute or
       repeating rendering favourably - `speaker_similarity.py` says so in its own docstring
       - so without this the more an arm collapsed the better its likeness number would look.
    2. **Not memorisation.** The channel-A audio of the V-real arms is the training audio.
    3. **The improvement is real.** The lower bound of the interval on the paired mean delta
       clears zero. This is what replaced "higher on 8 of 10", whose power at a true win
       rate of 0.70 is 0.383 - it could not tell an absent effect from an unmeasured one.
    4. **The improvement is worth having.** The arm closes at least `min_band_closure` of
       the distance from the control to the target speaker's own calibration band.

    The inputs are checked against each other rather than trusted. `paired_deltas` must
    cover the whole fixed clip set, so an interval cannot be computed over the survivors
    while the mean it belongs to uses the full denominator; and the two absolute means must
    differ by exactly the mean of the deltas, so an optimistic survivors-only candidate mean
    cannot be paired with a full-set delta vector. Both mistakes flatter a collapsing arm.
    """
    if isinstance(paired_deltas, Mapping):
        deltas = [float(paired_deltas[key]) for key in sorted(paired_deltas)]
    else:
        deltas = [float(value) for value in paired_deltas]
    if len(deltas) != denominator:
        raise ValueError(
            f"condition 4 is judged over the full fixed set: got {len(deltas)} deltas for a "
            f"denominator of {denominator}. Charge 0 for every clip the candidate could not "
            "produce (tools.speaker_similarity.full_set_delta_vector) rather than shortening "
            "the list, which would judge the arm on the clips where it still behaved."
        )
    for field in ("mean", "min"):
        if field not in band:
            raise ValueError(
                f"the calibration band is missing {field!r}; condition 4 cannot be decided "
                "without knowing what the target speaker scores against her own centroid"
            )

    mean_delta_full_set = statistics.fmean(deltas)
    drift = (candidate_mean - control_mean) - mean_delta_full_set
    if abs(drift) > 1e-6:
        raise ValueError(
            f"candidate_mean - control_mean is {candidate_mean - control_mean:+.6f} but the "
            f"deltas average {mean_delta_full_set:+.6f} (difference {drift:+.2e}); the "
            "absolute means and the paired deltas were computed over different clip sets"
        )

    interval = paired_mean_interval(
        deltas,
        confidence=confidence,
        bootstrap_iterations=bootstrap_iterations,
        seed=bootstrap_seed,
    )
    closure = band_closure(
        control_mean=control_mean,
        candidate_mean=candidate_mean,
        band_mean=float(band["mean"]),
        band_floor=float(band["min"]),
    )

    reasons: list[str] = []
    degenerate = collapse.get("degenerate_count", 0)
    if degenerate:
        reasons.append(
            f"{degenerate} of {collapse.get('total')} generations are degenerate "
            "(silent, repeating or monologuing), so the similarity is not evidence of likeness"
        )
    if memorisation == "memorisation":
        reasons.append("the memorisation verdict is memorisation")
    if interval["lower_bound"] <= 0:
        reasons.append(
            f"the {confidence:.0%} interval on the paired mean delta "
            f"{mean_delta_full_set:+.4f} runs from {interval['lower_bound']:+.4f}, so it does "
            "not exclude zero"
        )

    if closure["closed"] is None:
        magnitude_note = (
            f"{closure['undefined_reason']}; the magnitude bar is vacuous here and the "
            "decision rests on the interval and on the listening pass"
        )
        if candidate_mean < control_mean:
            reasons.append(
                f"no band headroom, and the candidate mean {candidate_mean:.4f} is below the "
                f"control mean {control_mean:.4f}"
            )
    else:
        magnitude_note = (
            f"closed {closure['closed_percent']:.1f}% of the {closure['headroom']:+.4f} gap "
            f"between the control and the calibration band"
        )
        if closure["closed"] < min_band_closure:
            reasons.append(
                f"closed {closure['closed_percent']:.1f}% of the gap to the calibration band, "
                f"below the {100.0 * min_band_closure:.0f}% required"
            )

    wins = sum(1 for value in deltas if value > 0)
    losses = sum(1 for value in deltas if value < 0)
    return {
        "passes": not reasons,
        "reason": "; ".join(reasons) if reasons else "all criteria met",
        "denominator": denominator,
        "paired_mean_delta_full_set": mean_delta_full_set,
        "interval": interval,
        "band_closure": closure,
        "min_band_closure": min_band_closure,
        "magnitude_note": magnitude_note,
        "degenerate_count": degenerate,
        "memorisation": memorisation,
        "descriptive_sign_count": {
            "higher_on": wins,
            "lower_on": losses,
            "ties": denominator - wins - losses,
            "not_a_criterion": (
                "recorded because it is comparable with the M3 record, and not used to "
                "decide: at a true win rate of 0.70 the 8-of-10 rule it belonged to had "
                "power 0.383"
            ),
        },
    }
