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

Everything here is pure and imports nothing.
"""

from __future__ import annotations

from typing import Any

# 25 frames of 480 samples is 0.5 s at 24 kHz. Below that there is not enough voiced audio
# for a speaker embedding to mean anything, whatever number comes back.
DEFAULT_MIN_VOICED_FRAMES = 25


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


def condition4_verdict(
    *,
    higher_on: int,
    denominator: int,
    mean_delta_full_set: float,
    collapse: dict[str, Any],
    memorisation: str,
    min_delta: float,
) -> dict[str, Any]:
    """Decide condition 4, with condition 3 wired in as a precondition.

    The criteria are the ones m3/DATASET_SPEC.md fixed before any candidate existed:
    direction, at least 8 of 10 higher, and an effect of at least `min_delta` - plus the
    memorisation verdict not being "memorisation".

    The interlock is the addition. An arm whose generations are degenerate cannot pass on
    likeness, because ECAPA scores that degeneracy favourably; without this, the more an arm
    collapsed the better its condition-4 number would look.
    """
    reasons: list[str] = []

    degenerate = collapse.get("degenerate_count", 0)
    if degenerate:
        reasons.append(
            f"{degenerate} of {collapse.get('total')} generations are degenerate "
            "(silent, repeating or monologuing), so the similarity is not evidence of likeness"
        )
    if memorisation == "memorisation":
        reasons.append("the memorisation verdict is memorisation")
    if higher_on < 8:
        reasons.append(f"higher on {higher_on} of {denominator}, below the 8 required")
    if mean_delta_full_set < min_delta:
        reasons.append(
            f"full-set mean delta {mean_delta_full_set:+.4f} is below the {min_delta:+.4f} required"
        )

    return {
        "passes": not reasons,
        "reason": "; ".join(reasons) if reasons else "all criteria met",
        "higher_on": higher_on,
        "denominator": denominator,
        "mean_delta_full_set": mean_delta_full_set,
        "min_delta": min_delta,
        "degenerate_count": degenerate,
        "memorisation": memorisation,
    }
