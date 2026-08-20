---
name: experiment-log
description: Use after any experiment step that produces a result - a GPU run, a data audit, a gate that passed or failed, a spend change. Records the outcome in the milestone document, the reports directory and the spend ledger so a milestone can be judged complete from evidence rather than memory.
---

# Recording an experiment result

`docs/experiments/j-moshi-tsukuyomi-ojousama-milestones.md` is the authority on progress and
completion. It states its own rule: a milestone is never complete because a run happened,
only when each completion condition points at a file, log, checkpoint or report that proves
it. This skill is how that rule gets followed.

## What to write where

| Artifact | Holds |
| --- | --- |
| `docs/.../milestones.md` | State, completion checkboxes, evidence links, why a gate failed and where it sends you back |
| `docs/.../plan.md` | Technical conditions and the run matrix. Change this *first* when a run condition changes, then sync the milestone |
| `experiments/tsukuyomi_ojousama/reports/*.json` | The measured numbers, per run, with the commit that produced them |
| `experiments/tsukuyomi_ojousama/m0/spend-ledger.json` | Invoice lines, accrued estimate, preflight decisions, instance state |
| `experiments/tsukuyomi_ojousama/registry/*.json` | Any dataset touched: source, version, terms, and whether it is used |

## Rules that are easy to get wrong

**Report the outcome, not the intent.** If a gate failed, say so with the number and the
threshold. A run that produced no artifact is `blocked` or `partial`, never `pass`.

**Write the report before enforcing a gate.** A rejected run's numbers are the evidence you
need to diagnose it. `tools/persona_perplexity.py` writes its report and then fails.

**A superseded report stays.** Add `supersedes` to the newer one rather than editing history.
`m0-baseline-final.json` supersedes two earlier runs that recorded real failures worth keeping.

**Record the commit.** Artifacts from different commits in one report need
`artifact_provenance` saying which produced what.

**Never record a secret.** No API keys, no instance tokens, no account balance. The ledger
tracks charges per instance; `starting_account_balance` is the only balance figure and it is
a fixed historical datum.

**Cite the file, not the memory.** "checksums verified" is worth nothing without the path and
the digest.

## When a gate fails

Record which gate, the observed value, the threshold, and the milestone's stated fallback.
Then ask whether the gate itself is right before iterating on the code: an absolute-NLL gate
on `persona_perplexity` rejected a working paired metric three times because its premise did
not fit a paired comparison. A gate that keeps rejecting plausible results is a hypothesis
about correctness, and hypotheses can be wrong.

## Before claiming a milestone complete

Walk its completion conditions one at a time and name the evidence for each. If a condition
covers two things - fixed audio *and* evaluation values - and only one exists, split the
checkbox rather than ticking it. If a condition depends on data that was never obtained,
register that source with `used_in_experiment: false`, a rationale and a reopen condition,
so the ledger is complete even though the data is not.

Then run the suite: `uv run --python 3.12 --with pytest --no-sync python -m pytest tests -q`.
The data gates in `tests/test_experiment_assets.py` are what keep the M1 guarantees true.
