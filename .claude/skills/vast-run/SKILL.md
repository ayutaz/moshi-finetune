---
name: vast-run
description: Use when running any GPU job for the tsukuyomi experiment on Vast.ai - renting or restarting an instance, bootstrapping it, running run_baseline.sh or a training run, exporting artifacts, and stopping the instance afterwards. Covers the budget ledger and the failure modes that cost real money.
---

# Running a GPU job on Vast.ai

The instance bills by the hour whether or not it is doing anything. Every step below
exists because skipping it cost time or money at least once.

## Before starting anything

Run the budget preflight and record the decision. **The cap is US$125, approved
2026-08-24.** It replaced the US$100 cap of 2026-08-18, which the M3 session breached at
US$102.697; 4-1 has since taken it to US$102.812. Headroom was US$22.30 when the cap moved,
and `m0/spend-ledger.json` records under `cap_raise` exactly what that headroom is for - one
V-real re-run and one forward-only measurement. Spending it elsewhere needs the user, not you.

**The preflight works and its answer is binding.** `tools/experiment_budget.py` takes the
cap as a parameter (default US$125) and derives its thresholds as fractions of it - warning
at 0.75, new-run at 0.90, stop at 0.95 - so raising the cap moves the whole policy instead
of leaving old absolutes behind. That was the earlier failure: the tool tested against a
hardcoded US$100 after the cap moved, refused every possible plan, and a preflight that
always says no is one nobody reads. The ledger's thresholds (US$93.75 / 112.50 / 118.75)
are the same fractions.

So a non-zero exit now means something. Against accrued US$102.812 the new-run limit of
US$112.50 leaves US$9.688 - under four hours at A100x2 rates, less than one full V-real run.
Read `accrued_estimate` from the ledger rather than this line; it moves with every run.
**Do not widen a fraction, raise the default, or skip the check to admit a
plan.** Split the plan, or ask the user for a cap. `tests/test_experiment_budget.py` pins
the phase-4 verdict as a test for exactly this reason.

```bash
uv run --no-sync python -m tools.experiment_budget \
  --spent <accrued_estimate from spend-ledger.json> \
  --hourly-rate <instance $/h> \
  --planned-hours <estimate>
```

Non-zero exit means do not start. Use `accrued_estimate`, not `invoiced_to_date`:
Vast.ai's invoice lags actual runtime badly (0.793 invoiced GPU hours against ~6.5 real
ones was observed) and `vastai show invoices` returns only the current bucket.

## Starting the instance

```bash
vastai start instance <id>
```

**"Required resources are currently unavailable, state change queued" means the host is
full.** The request is not held: `intended_status` reverts to `stopped`. Retry on a loop,
but do not assume it will free up - 20 retries over 41 minutes all failed once, and the
fix was renting a different host. To migrate:

```bash
vastai search offers 'gpu_name=A100_SXM4 num_gpus=2 gpu_ram>=79 disk_space>=200 reliability>0.98' --raw
vastai create instance <offer_id> --image pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel \
  --disk 120 --label <label> --ssh --direct
```

Prefer a high `inet_down`, but size that requirement to the bootstrap you are actually
running: `m0/bootstrap_instance.sh` re-downloads both published checkpoints (about 31 GB),
while `m3/bootstrap_m3_instance.sh` pulls j-moshi-ext alone - the ledger's preflight for
that run recorded 15.76 GB. Disk costs money per hour, so size it for the job (120 GB is
enough for generation and baselines; the retired 300 GB instance billed US$0.139/h while
stopped against US$0.033/h for 120 GB).

**`--disk` cannot be changed after create.** Undersizing it is not a cost saving, it is a
run that dies partway through and has to be paid for twice.

**`dph_total` in a search result does not include the disk.** It is the compute rate; the
disk bills separately at `storage_cost` US$/GB/month. Multiply it out before you believe a
price:

```
rate = dph_total + storage_cost * disk_gb / 730
```

M3-R 4-2 rented at an advertised US$2.0896/h and was billed US$3.3327/h - the offer charged
US$1.00/GB/month and 900 GB added US$1.23/h. At that rate the budget line fell to 2.91 h
against a 3.376 h plan, leaving 0.24 h for a conversion that M3 measured at 0.5 h. **The
instance was destroyed before the training started**, at a cost of US$0.20, rather than
discovering it with a half-converted checkpoint on the clock. The replacement charged
US$0.20/GB/month and billed US$2.4936/h.

Size the disk for what is resident at once, not for the total written. Five ZeRO states are
502 GB, but a run that converts and reclaims each checkpoint holds at most two - about
328 GB with the fp32 intermediate, the base model and the exports.

The API returns an `instance_api_key`. **Never write it to a file or a commit.**

### Sizing a full-parameter Moshi training run

Generation and baselines are small. Training the 8.37B `dep_q=16` model is not, and none
of the numbers that matter are visible in a GPU spec:

```bash
vastai search offers 'gpu_name=A100_SXM4 num_gpus=2 gpu_ram>=79 disk_space>=1000 \
  cpu_ram>=100000 reliability>0.98' --raw
vastai create instance <offer_id> --image pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel \
  --disk 900 --label <label> --ssh --direct
```

| Constraint | Value | Why it bites |
| --- | --- | --- |
| GPUs | **2× A100 80GB SXM4** | ZeRO-3 + fp16 + AdamW is 16 bytes/param resident = 133.94 GB against 85.90 GB on one card. One GPU is 48 GB short before any activation memory. **Ask for SXM4 by name.** M3-R 4-2 rented A100 **PCIe** and hung in a collective right after the dataset load - NCCL initialised, then both ranks spun at 100% CPU with no I/O for 75 minutes across two attempts. M3 ran the same code on SXM4 without trouble. NCCL P2P over PCIe is the hypothesis and it is untested, but the run cost US$4.29 and produced nothing. |
| Host RAM | **80 GB available, 85 GiB+ allocated** | `finetune.py` loads the model on CPU in float32 *per rank* before accelerate partitions anything: 2 × 33.49 GB. `zero_to_fp32` peaks near 67 GB separately. Too little RAM OOM-kills at load, and no GPU is large enough to fix it. |
| Disk | **900 GB** | A ZeRO-3 `save_state` is 12 bytes/param = 100.46 GB, and `finetune.py` has **no rotation at all** - no `--save_total_limit`, and accelerate's own rotation is never enabled. Five epochs accumulate 502 GB. |

Check `MemAvailable` on the instance before launching, not the advertised figure:

```bash
ssh -p <port> root@<ip> "awk '/MemAvailable/ {print \$2/1048576 \" GiB\"}' /proc/meminfo; nvidia-smi --query-gpu=memory.total --format=csv,noheader; df -h /workspace | tail -1"
```

If checkpoints have to accumulate faster than they can be converted, run a reclaim watcher
alongside the training that converts each closed checkpoint and deletes the ZeRO state -
but start it **after** the startup assertion passes, never at launch, when 67 GB of fp32
CPU copies are still live.

### Sizing a forward-only measurement

`m3/bootstrap_m3_instance.sh` stops on anything smaller than 2 GPUs of 80 GB with 80 GiB of
host RAM available. That gate is right for training and far too large for a forward pass.
**Do not loosen a gate to admit a job it was not written for - give the different job its
own gate.** The 4-1 gate refuses under a 20 GiB card or 40 GiB of available RAM - a floor,
set just above the 31.19 GiB the fp32 CPU load needs. Ask an offer for 64 GB so the floor is
not the thing you are betting on; the box 4-1 actually ran on reported 720 GiB.

| Constraint | Forward-only | Why |
| --- | --- | --- |
| GPU | one card, **20 GiB** | fp16 weights are 15.59 GiB. No optimiser state, no gradients, no ZeRO partitioning to pay for. |
| Host RAM | **31.19 GiB**, so ask for 64 GB | `finetune.py` still loads the model on CPU in float32 before anything is partitioned. This, not the GPU, is what a cheap offer fails. |
| Launcher | DeepSpeed anyway | the launcher refuses without it even on one GPU - see below. |

4-1 measured the base-loss breakdown on a single V100-SXM2-32GB at US$0.3017/h: 0.382 h,
US$0.115, a quarter of its 1.50 h stop line. The 2× A100 box the training gate demands
bills US$3.0567/h and would have returned the same numbers.

## Connecting

The `sshN.vast.ai` proxy sometimes refuses the account key even when it is attached.
Use the direct address instead:

```bash
vastai ssh-url <id>          # ssh://root@<ip>:<port>
ssh -o StrictHostKeyChecking=accept-new -p <port> root@<ip>
```

## Bootstrapping

```bash
scp -P <port> experiments/tsukuyomi_ojousama/m0/bootstrap_instance.sh root@<ip>:/workspace/
ssh -p <port> root@<ip> 'bash /workspace/bootstrap_instance.sh'
```

It installs uv, clones the branch, syncs the environment, and re-downloads both published
checkpoints, **failing loudly if either SHA-256 does not match** the values in
`m0/artifact-recovery.md`. Do not skip that check; it is the only thing standing between a
corrupted download and a day of confusing results.

## Uploading prompt audio

Upload from `data/experiments/tsukuyomi_ojousama/baseline-input/` and verify the remote
checksums against `manifests/tsukuyomi-corpus-v1.jsonl` before running anything. The raw
audio is not redistributable, so it never goes through git.

## Running

Run detached. An SSH drop must not kill a job you are paying for.

```bash
ssh -p <port> root@<ip> '
export PATH="$HOME/.local/bin:$PATH"
cd /workspace/moshi-finetune && git pull --ff-only origin <branch>
nohup setsid bash -c "bash <script> <args>; echo EXIT_CODE=\$? > /workspace/run.done" \
  > /workspace/run.nohup 2>&1 < /dev/null &'
```

`export PATH` matters: uv installs to `~/.local/bin`, which a non-interactive shell does
not have. `run_baseline.sh` handles this itself, but ad-hoc commands do not.

**Smoke-test before the real launch.** M3's plan had a smoke-test stage; M3-R's 4-2 launched
straight into the run and spent US$4.29 discovering that the collective hangs. Two steps with
`--max_train_steps 2` costs minutes and answers whether this box can train at all. When
distributed init is the suspect, add `NCCL_DEBUG=INFO`, and `NCCL_P2P_DISABLE=1` to test
whether P2P is what is stuck.

And run the CPU-only stages at home first. `preprocess_function` on the shipped 70-row
parquet takes 0.01 s on a laptop - it was suspected for an hour of paid time before anyone
ran it for free.

Watch the log with a filter that covers **failure as well as progress**. A filter matching
only success markers stays silent through a crash, and silence looks like "still running".
Include `Traceback`, `Error`, `assert`, `OOM`, `Killed`.

### `finetune.py` runs only under DeepSpeed

`finetune.py:425-426` raises `NotImplementedError: Only DeepSpeed is supported for now.`
when `--use_deepspeed` is absent, and errors again if `--deepspeed_config_file` does not
come with it. That holds on a single GPU and for a forward-only pass, not just for
distributed training: the first 4-1 launch died on exactly this.

```bash
accelerate launch --use_deepspeed \
  --deepspeed_config_file ds_configs/zero3-fp16-offload-act_ckpt.json \
  finetune.py <args>
```

`ds_configs/` holds the configs the runs have used and `examples/finetune_accelerate.sh`
shows the full argument list. When the point of a run is to compare against an earlier one,
reuse that run's config: 4-1 kept M3's fp16 offload config so the losses were comparable,
even though one GPU has no need to offload an optimiser it never allocates.

## The stop line

M3 was authorised to 14.0 hours and ran 25.21. That overrun alone cost US$34.27, and it is
what carried cumulative spend past the cap - before it, accrued was US$25.638 of US$100.
Three habits caused it, and each has a replacement.

**Fix the stop line as a UTC timestamp before the run starts, not as an elapsed-hours
budget.** Every progress check during M3 recomputed a forecast from the plan's per-stage
estimates. The forecast stayed reassuring while the clock did not. Derive the deadline once
from the instance's own `start_date` and compare wall clock against it at every check:

```bash
vastai show instance <id> --raw | python3 -c "
import datetime as dt, json, sys
start = dt.datetime.fromtimestamp(json.load(sys.stdin)['start_date'], dt.timezone.utc)
now = dt.datetime.now(dt.timezone.utc)
print('start  ', start.isoformat())
print('now    ', now.isoformat())
print('elapsed', round((now - start).total_seconds() / 3600, 2), 'h')"
```

`start_date` is the authority. An elapsed figure carried forward in your head, or recomputed
from stage estimates, is not - and the two diverge in the direction that costs money.

**Put the export in the estimate.** M3 measured roughly 30 minutes per 15.4 GB checkpoint
file, and no elapsed-time check ran between the last generation and the export finishing.
Five checkpoints is a multi-hour tail on a meter that is still running. Budget those hours
before renting, and treat "training finished" as the middle of the run, not the end.

**Call the preflight during the run, not only before it.** M3 invoked
`tools/experiment_budget.py` once, before renting. The one instrument that would have said
"stop" was never asked again. Re-run it at each progress check, with the hours actually
elapsed plus what is left:

```bash
uv run --no-sync python -m tools.experiment_budget \
  --spent <accrued_estimate> --hourly-rate <instance $/h> --planned-hours <elapsed + remaining>
```

A non-zero exit mid-run means export what exists and stop. It does not mean "finish this
stage first" - the stage is what the money is going into.

## Before stopping

Export everything you need first. The disk survives stop/start but dies with the instance.

```bash
rsync -az -e "ssh -p <port>" --include='*/' --include='*.json' --include='*.log' \
  --include='*.wav' --include='*.npy' --exclude='*' \
  root@<ip>:/workspace/experiment-artifacts/baselines/ <local export dir>/
```

Export into `data/`, which is gitignored. Generated audio and any checkpoint that has not
passed a publication review are non-public - see `DATA_CREDITS.md`.

**The export is limited by your own link, not by the instance - so do not wait it out on the
expensive machine.** The phase-4 estimate has 84 GB coming down at 8.56 MB/s: 2.718 hours,
39% of the whole run, spent watching a US$3.0567/h box do nothing. Push the converted
checkpoints to a cheap instance, destroy the expensive one, then pull at your leisure. Live
`vastai search offers` found an RTX 3060 12GB at US$0.0525/h and a GTX 1070 at US$0.0481/h -
five cents an hour to hold the files, about an eighth of the US$0.45/h that
`m3r/STOP_LINE.md` §6 had assumed when it costed the split. Search for the real rate rather
than carrying an assumed one into a plan.

## Stopping - do not skip this

```bash
vastai stop instance <id>
```

Then update `m0/spend-ledger.json`: append the invoice lines, refresh `accrued_estimate`,
and record the run's outcome. A stopped instance still bills for its disk, so destroy one
you have finished with, **after** confirming its contents are reproducible or exported:

```bash
vastai destroy instance <id> -y
```

The disk rate scales with size, so a large training instance is expensive to leave stopped:
900 GB is about **US$10.00/day** doing nothing, against US$3.33/day at 300 GB. Destroy a
finished training instance the same day rather than stopping it.

**Only destroy instances this experiment created.** The account runs unrelated work in
parallel - `spend-ledger.json` lists them under `other_instance_ids_excluded` - and
`vastai show instances` gives no hint which is which. An instance that merely looks idle is
not yours to end; confirm the id against the ledger, and ask if it is not there.

## Then

Invoke the `experiment-log` skill to write the result into the milestone document and the
reports directory.
