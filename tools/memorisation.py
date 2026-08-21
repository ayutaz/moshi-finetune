"""Tell a checkpoint that learned a voice from one that learned 72 sentences.

V-real's channel A is the training audio itself: the model hears the same 72 corpus
sentences, spoken by the target speaker, five times over. A model that simply stores them
passes every other measurement M3 takes. Speaker similarity on held-out prompts goes up,
because the voice it reproduces is the right voice. Intelligibility goes up, because a
memorised sentence is pronounced perfectly. `eval/RUBRIC.md` already names the verdict -
seenだけが改善してheld-outが改善しないcheckpointは暗記と判定する - and nothing computed it.

Two independent signals, either of which is sufficient:

1. **Seen against held-out.** The comparison RUBRIC.md describes, needing the seen-10
   prompt set that M3 builds alongside the held-out one.
2. **Verbatim reproduction.** Whether a training sentence comes back whole in the decoded
   text.

The second needs a different instrument than the one already here. `experiment_data`'s
`_near_duplicate` is symmetric Jaccard at 0.9, which answers "are these two texts the same
text" - the right question for deduplicating a corpus, the wrong one here. A 50-character
training sentence embedded in a 150-character generation shares every one of its shingles
with it and still scores far below 0.9, because the union counts everything the model
added. Containment asks instead what fraction of the TRAINING sentence survived, and is
unaffected by length. `tests/test_memorisation.py` pins both halves of that.

Everything here is pure and imports nothing heavy.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

# Shared with experiment_data on purpose. Two normalisers that disagree about what "the
# same text" means would let a sentence be a duplicate to one tool and original to the other.
from tools.experiment_data import _character_shingles, _normalise_text


def containment(needle: str, haystack: str) -> float:
    """Fraction of `needle`'s character shingles that appear in `haystack`.

    Asymmetric on purpose: 1.0 means all of `needle` is present, however much else
    `haystack` contains.
    """
    needle_normalised = _normalise_text(needle)
    haystack_normalised = _normalise_text(haystack)
    if not needle_normalised:
        raise ValueError("needle must contain text")
    needle_shingles = _character_shingles(needle_normalised)
    if not needle_shingles:
        raise ValueError(f"needle is too short to shingle: {needle!r}")
    haystack_shingles = _character_shingles(haystack_normalised)
    return len(needle_shingles & haystack_shingles) / len(needle_shingles)


def reproduced_training_texts(
    generation: str, training_texts: Iterable[str], *, threshold: float
) -> list[dict[str, Any]]:
    """Training sentences this generation gives back, with how completely."""
    if not 0 < threshold <= 1:
        raise ValueError(f"threshold must lie in (0, 1], got {threshold}")
    normalised_generation = _normalise_text(generation)
    hits: list[dict[str, Any]] = []
    for text in training_texts:
        if not _normalise_text(text):
            continue
        score = containment(text, generation)
        if score >= threshold:
            hits.append(
                {
                    "training_text": text,
                    "containment": score,
                    "exact_substring": _normalise_text(text) in normalised_generation,
                }
            )
    return sorted(hits, key=lambda hit: hit["containment"], reverse=True)


def memorisation_verdict(
    *, seen_delta: float, heldout_delta: float, verbatim_hits: int, min_delta: float
) -> dict[str, Any]:
    """Decide whether the improvement generalised, following RUBRIC.md.

    `seen_delta` and `heldout_delta` are speaker-likeness improvements over the control on
    the seen-10 and held-out-10 prompt sets. `min_delta` is the effect size M3 pre-registered
    as meaningful, so "improved" means the same thing here as in condition 4.
    """
    if verbatim_hits > 0:
        return {
            "verdict": "memorisation",
            "reason": (
                f"{verbatim_hits} generation(s) reproduce a training sentence verbatim or "
                "near-verbatim"
            ),
            "seen_delta": seen_delta,
            "heldout_delta": heldout_delta,
            "verbatim_hits": verbatim_hits,
        }

    seen_improved = seen_delta >= min_delta
    heldout_improved = heldout_delta >= min_delta
    if seen_improved and not heldout_improved:
        verdict, reason = (
            "memorisation",
            (
                f"seen improved by {seen_delta:.4f} while held-out moved {heldout_delta:.4f}, "
                f"below the {min_delta} that counts as improvement"
            ),
        )
    elif seen_improved and heldout_improved:
        verdict, reason = "generalisation", "both seen and held-out improved"
    elif heldout_improved:
        # Held-out beating seen is not memorisation, but it is not a clean result either:
        # something other than the target voice is probably moving the number.
        verdict, reason = (
            "inconsistent",
            (
                "held-out improved while seen did not, which the memorisation model does not "
                "predict - investigate before adopting"
            ),
        )
    else:
        verdict, reason = "no-improvement", "neither seen nor held-out improved"

    return {
        "verdict": verdict,
        "reason": reason,
        "seen_delta": seen_delta,
        "heldout_delta": heldout_delta,
        "verbatim_hits": verbatim_hits,
    }


def count_verbatim_hits(
    generations: Sequence[str], training_texts: Sequence[str], *, threshold: float
) -> dict[str, Any]:
    """Scan a run's decoded text streams for training sentences coming back."""
    per_generation = []
    for index, generation in enumerate(generations):
        if not _normalise_text(generation):
            continue
        hits = reproduced_training_texts(generation, training_texts, threshold=threshold)
        if hits:
            per_generation.append({"index": index, "hits": hits})
    return {
        "generations_scanned": len(generations),
        "generations_with_hits": len(per_generation),
        "threshold": threshold,
        "details": per_generation,
    }
