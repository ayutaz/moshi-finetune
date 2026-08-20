# moshi-finetune

Fine-tuning Moshi / J-Moshi on spoken dialogue data. `main` tracks the upstream project;
the `experiment-j-moshi-character-voice-overfit` branch carries the tsukuyomi voice and
persona experiment.

## Running Python

Everything goes through `uv`. The project targets Python 3.12 and the system `python3` may
be older.

```bash
uv run --no-sync python -m tools.<module>
uv run --python 3.12 --with pytest --no-sync python -m pytest tests -q
```

`python -m pytest`, never bare `pytest`: the tests import `tools` and `models` from the
repository root, and only the module form puts the working directory on `sys.path`.

`uv.lock` is gitignored, so dependency resolution differs per host. Pin it before any run
whose numbers have to be reproducible.

## The experiment

Read [`experiments/tsukuyomi_ojousama/README.md`](experiments/tsukuyomi_ojousama/README.md)
first. Progress and completion are judged in
[the milestone document](docs/experiments/j-moshi-tsukuyomi-ojousama-milestones.md); the run
matrix lives in [the plan](docs/experiments/j-moshi-tsukuyomi-ojousama-plan.md). A milestone
is complete only when each condition points at a file that proves it.

Two skills cover the recurring work: `vast-run` for GPU jobs on Vast.ai, `experiment-log`
for writing a result into the record.

## Rules that cost money or rights if broken

**GPU spend is capped at US$100**, approved 2026-08-18. Run
`tools/experiment_budget.py` before starting an instance and record the decision in
`m0/spend-ledger.json`. **Stop the instance when the run ends** - a stopped instance still
bills for its disk, and Vast.ai's invoice lags real runtime badly, so budget against
`accrued_estimate`, not `invoiced_to_date`.

**Never commit** raw tsukuyomi audio, generated audio, checkpoints that have not passed a
publication review, API keys, or instance tokens. They belong in `data/`, which is
gitignored. See `experiments/tsukuyomi_ojousama/DATA_CREDITS.md`.

**Third-party data carries its licence with it.** `reference/ojousama-talk-script-201.jsonl`
is MIT-derived and published here, so `reference/LICENSE.OjousamaTalkScriptDataset` ships
alongside it. A copyright line alone does not satisfy MIT for a redistributed derivative.

**Register a dataset before using it.** Every `dataset_id` in a manifest must have a registry
entry that is not marked `used_in_experiment: false`. A source that was never obtained still
gets an entry, with its terms, a rationale and a reopen condition.

## Tests are the data gates

`tests/test_experiment_assets.py` holds the guarantees the experiment rests on: raw-audio
checksums, evaluation-registry coverage in both directions, agreement between the voice
evaluation set and the corpus manifest, and the bundled licence. They run in CI and in a
pre-commit hook. When one fails, the data ledger is wrong - fix the data, not the test.

New pure logic gets a test. Heavy dependencies stay behind lazy imports inside functions so
the suite runs without torch.

## Evaluating

Loss alone never decides anything. Voice quality, intelligibility, persona and full-duplex
behaviour are judged separately, and a checkpoint that collapses in live dialogue is
rejected whatever its loss.

When a gate keeps rejecting plausible results, question the gate. An absolute-NLL gate on
the persona metric rejected a working paired comparison three times before the real problem
turned out to be a length-biased win criterion.
