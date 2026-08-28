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

M0, M1 and M2 are complete. M3 (Voice control) is **complete and failed**: the verdict was
right, the diagnosis was not, and the diagnosis has been retracted - see
[the M3 verification record](docs/experiments/j-moshi-tsukuyomi-ojousama-m3-verification.md).
The work now runs under **M3-R**, which repairs the record, the instruments and the data
before re-taking the control
([the plan](docs/experiments/j-moshi-tsukuyomi-ojousama-m3r-plan.md)). Where it stands -
what is done, what stopped it, what the user has to decide - is
[the M3-R status](docs/experiments/j-moshi-tsukuyomi-ojousama-m3r-status.md): phases 0 to 3
and 4-1 are done, 4-2's run1 hung before training started and bought nothing, and the
preflight rejects a retry under the current cap. M4 stays blocked until M3-R hands it a
control checkpoint.

Two skills cover the recurring work: `vast-run` for GPU jobs on Vast.ai, `experiment-log`
for writing a result into the record.

## Rules that cost money or rights if broken

**GPU spend is capped at US$125**, raised from US$100 on 2026-08-24 after the M3 session
breached the old cap at US$102.697. Run `tools/experiment_budget.py` before starting an
instance **and again at every progress check while it runs** - M3 was authorised 14.0 hours,
ran 25.21, and called the preflight exactly once - then record the decision in
`experiments/tsukuyomi_ojousama/m0/spend-ledger.json`. **Stop the instance when the run ends**
- a stopped instance still bills for its disk, and Vast.ai's invoice lags real runtime badly,
so budget against `accrued_estimate`, not `invoiced_to_date`.

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

## Measure the premise before building on it

Every costly correction in this experiment came from an assumption a few minutes of
measurement would have settled, so measure first and record the number. Reports live in
`experiments/tsukuyomi_ojousama/reports/`.

- The neutral speaker B was to be generated per utterance from a caption. Measured, that
  produces a different voice nearly every line - below the band ten recordings of one real
  human occupy - which would have made one channel of every training dialogue a crowd.
  Nothing in a loss curve shows this. `m3-speaker-b-probe.json`.
- Mimi was assumed to need a GPU because the tools hardcoded CUDA. It tokenises 160
  dialogues in 1.2 minutes on this Mac. `m3-local-compute-probe.json`. M3-R 4-2 repeated the
  mistake on a rented box: an hour of paid suspicion fell on `preprocess_function`, which
  runs the shipped 70 rows in 0.01 s here. The CPU-only stage runs before renting, not after,
  and a two-step smoke test comes before the real launch. `m3r-run1-failure.json`.
- An intelligibility gate scored Whisper's choice of orthography rather than the model's
  pronunciation, and failed a working checkpoint 26/30 until readings were compared instead.
- A 60-second sequence length was a premise I set without checking it. Both citations were
  mine and both were wrong; the one run that has ever worked here is 19.02 s at one dialogue
  per example. Cite this project's own record before an upstream README.
  `docs/experiments/j-moshi-tsukuyomi-ojousama-m3r-dataset-audit.md`.
- A high base audio loss was not evidence about the target speaker. 80.1% of it sits on the
  untrained head `models/utils.py` deepcopies for the user stream. A total says nothing until
  it is split. M3 computed that split ten times and kept none. `m3r-forward-breakdown.json`.
- Room tone was not "collect the silences and lay them down" - the corpus holds none, and
  every cheap substitute moved the shortcut rather than removing it. What makes natural
  silence varied is the decay tails and breath, not the quiet. `m3r-roomtone.json`.

A calibration band is part of the measurement, not an extra: a within-group similarity of
0.74 means nothing until you know one real human scores 0.70.

## Evaluating

Loss alone never decides anything. Voice quality, intelligibility, persona and full-duplex
behaviour are judged separately, and a checkpoint that collapses in live dialogue is
rejected whatever its loss.

Some measurements invert when the model fails, so they are never published alone. Transcript
perplexity is the sharp case: a voice stuck on one mora scores 13.9 where a real sentence
scores 91.3, so `tools/intelligibility.py` has no function that returns a perplexity without
the repetition beside it, and condition 5 is judged on clean transcripts over a fixed
denominator (control 0.80). Conditions 3 and 4 interlock for the same reason
(`tools/likeness_guard.py`): an arm that fell silent must not read as more speaker-like. And
the collapse detector reads the audio row, not only the text row - the control everything was
measured against was itself collapsed in 17 of its 30 general30 generations, and nothing on
the text side could see it.

When a gate keeps rejecting plausible results, question the gate. An absolute-NLL gate on
the persona metric rejected a working paired comparison three times before the real problem
turned out to be a length-biased win criterion.
