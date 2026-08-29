#!/usr/bin/env bash
# PreToolUse(Bash, vastai create instance): judge the offer before it becomes a charge.
#
# This session lost US$4.49 to two offers, and both were readable in the search result
# before anything was rented:
#
#   US$0.20  Advertised US$2.0896/h, billed US$3.3327/h. `dph_total` is compute; the disk
#            bills separately at `storage_cost` US$/GB/month, and 900 GB at US$1.00 added
#            US$1.23/h. At the real rate the budget allowed 2.91 h against a 3.376 h plan,
#            so the instance was destroyed before training started.
#   US$4.29  A100 *PCIe*. M3 trained on A100-SXM4-80GB. NCCL initialised and then both
#            ranks spun at 100% CPU with no I/O until the run was abandoned. A search
#            result calls both machines "A100".
#
# Price and interconnect are independent - the first offer was SXM4 and the second was
# affordable - so `tools/offer_check.py` tests both and this hook runs it on the offer id
# in the command.
#
# It warns rather than denies. The lookup can fail for reasons that have nothing to do with
# the offer (the id has already been taken, the search is slow, vastai is not installed),
# and a hook that blocks work when its own evidence is missing is a hook people route
# around. What it must never do is stay silent about an offer it did judge.

set -u

command_text=$(jq -r '.tool_input.command // empty' 2>/dev/null)
case "${command_text}" in
  *"vastai create instance"*) ;;
  *) exit 0 ;;
esac

command -v vastai >/dev/null 2>&1 || exit 0
command -v uv >/dev/null 2>&1 || exit 0

cd "${CLAUDE_PROJECT_DIR:-.}" 2>/dev/null || exit 0
[ -f tools/offer_check.py ] || exit 0
[ -f experiments/tsukuyomi_ojousama/m0/spend-ledger.json ] || exit 0

offer_id=$(printf '%s' "${command_text}" | sed -n 's/.*vastai[[:space:]]\{1,\}create[[:space:]]\{1,\}instance[[:space:]]\{1,\}\([0-9]\{1,\}\).*/\1/p')
[ -n "${offer_id}" ] || exit 0
disk_gb=$(printf '%s' "${command_text}" | sed -n 's/.*--disk[[:space:]]\{1,\}\([0-9]\{1,\}\).*/\1/p')
[ -n "${disk_gb}" ] || disk_gb=0

# Offers are not addressable by id, so sweep and filter. Two sweeps because a training box
# and an export box do not appear in the same query.
offers=$( { vastai search offers 'num_gpus=1' --raw 2>/dev/null; vastai search offers 'num_gpus=2' --raw 2>/dev/null; } )

printf '%s' "${offers}" | uv run --no-sync python -c '
import json, sys

sys.path.insert(0, ".")
raw = sys.stdin.read()
offer_id, disk_gb = int(sys.argv[1]), float(sys.argv[2])

# vastai pretty-prints, so two --raw calls concatenate into two multi-line JSON
# documents rather than two lines. raw_decode walks them one after the other.
rows, decoder, pos = [], json.JSONDecoder(), 0
while pos < len(raw):
    while pos < len(raw) and raw[pos] in " \t\r\n":
        pos += 1
    if pos >= len(raw):
        break
    try:
        value, pos = decoder.raw_decode(raw, pos)
    except ValueError:
        break
    if isinstance(value, list):
        rows.extend(value)

match = next((r for r in rows if int(r.get("id", -1)) == offer_id), None)
if match is None:
    sys.exit(0)  # nothing to say, rather than a guess

try:
    from tools.offer_check import check_offer
    ledger = json.load(open("experiments/tsukuyomi_ojousama/m0/spend-ledger.json"))
except Exception:
    sys.exit(0)

spent = float(ledger["accrued_estimate"]["total"])
limit = float(ledger["new_run_prediction_limit"])
offer = dict(match)
if disk_gb:
    offer["disk_space"] = disk_gb

# The plan length is unknown here, so judge the offer on what the limit still buys at its
# real rate. That answers "is this offer priced so the run cannot finish", which is the
# question the destroyed instance failed.
verdict = check_offer(
    offer, spent=spent, limit=limit, planned_hours=0.0,
    num_gpus_needed=int(offer.get("num_gpus", 1)),
)
hours = (limit - spent) / verdict.hourly_rate if verdict.hourly_rate > 0 else 0.0

lines = [
    "offer {} - {}, {} GPU, disk {:.0f} GB".format(
        offer_id, offer.get("gpu_name", "?"), offer.get("num_gpus", "?"), float(offer.get("disk_space", 0))
    ),
    "billed US${:.4f}/h (advertised US${:.4f}/h), and US${:.3f} of headroom buys {:.2f} h".format(
        verdict.hourly_rate, float(offer.get("dph_total", 0)), limit - spent, hours
    ),
]
lines += ["WARNING: " + w for w in verdict.warnings]
if hours <= 0:
    lines.append("WARNING: accrued spend is already past the new-run limit")

print(json.dumps({"systemMessage": "\n".join(lines)}))
' "${offer_id}" "${disk_gb}" 2>/dev/null

exit 0
