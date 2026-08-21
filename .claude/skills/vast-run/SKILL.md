---
name: vast-run
description: Use when running any GPU job for the tsukuyomi experiment on Vast.ai - renting or restarting an instance, bootstrapping it, running run_baseline.sh or a training run, exporting artifacts, and stopping the instance afterwards. Covers the budget ledger and the failure modes that cost real money.
---

# Running a GPU job on Vast.ai

The instance bills by the hour whether or not it is doing anything. Every step below
exists because skipping it cost time or money at least once.

## Before starting anything

Run the budget preflight and record the decision. The cap is US$100, approved
2026-08-18; `tools/experiment_budget.py` encodes the thresholds.

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

Prefer a high `inet_down`: bootstrap re-downloads 31 GB of checkpoints. Disk costs money
per hour, so size it for the job (120 GB is enough for generation and baselines; the
retired 300 GB instance billed US$0.139/h while stopped against US$0.033/h for 120 GB).

**`--disk` cannot be changed after create.** Undersizing it is not a cost saving, it is a
run that dies partway through and has to be paid for twice.

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
| GPUs | **2× A100 80GB minimum** | ZeRO-3 + fp16 + AdamW is 16 bytes/param resident = 133.94 GB against 85.90 GB on one card. One GPU is 48 GB short before any activation memory. |
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

Watch the log with a filter that covers **failure as well as progress**. A filter matching
only success markers stays silent through a crash, and silence looks like "still running".
Include `Traceback`, `Error`, `assert`, `OOM`, `Killed`.

## Before stopping

Export everything you need first. The disk survives stop/start but dies with the instance.

```bash
rsync -az -e "ssh -p <port>" --include='*/' --include='*.json' --include='*.log' \
  --include='*.wav' --include='*.npy' --exclude='*' \
  root@<ip>:/workspace/experiment-artifacts/baselines/ <local export dir>/
```

Export into `data/`, which is gitignored. Generated audio and any checkpoint that has not
passed a publication review are non-public - see `DATA_CREDITS.md`.

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
