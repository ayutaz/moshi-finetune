#!/usr/bin/env bash
# Stop: warn while a Vast.ai instance is still burning money, and say how far past the
# stop line it is.
#
# Two costs this project has actually paid:
#
#   US$34.27  M3 was authorised 14.0 h and ran 25.21. Every progress check recomputed a
#             forecast from the plan's per-stage estimates instead of reading the
#             instance's own start_date. The forecast stayed reassuring while the clock
#             did not, and the overrun is what carried spend past the cap.
#   US$4.29   4-2 hung before training. Watching elapsed against a fixed line is what let
#             it be abandoned at 1.72 h instead of running to the deadline.
#
# So this reads `start_date` from the instance rather than any estimate, and compares it
# against a stop line recorded in `m0/spend-ledger.json` under `active_stop_lines`
# ({"<instance id>": "<UTC ISO-8601>"}). Without a recorded line it still reports elapsed
# time and cost - a number on the screen beats a number in your head.
#
# Silent on machines without the vastai CLI, so it costs collaborators nothing.

set -u

command -v vastai >/dev/null 2>&1 || exit 0

ledger="${CLAUDE_PROJECT_DIR:-.}/experiments/tsukuyomi_ojousama/m0/spend-ledger.json"

vastai show instances --raw 2>/dev/null | python3 -c '
import datetime as dt
import json
import sys

try:
    rows = json.load(sys.stdin)
except Exception:
    sys.exit(0)

running = [r for r in rows if r.get("actual_status") == "running"]
if not running:
    sys.exit(0)

stop_lines = {}
try:
    with open(sys.argv[1], encoding="utf-8") as handle:
        stop_lines = json.load(handle).get("active_stop_lines") or {}
except Exception:
    pass

now = dt.datetime.now(dt.timezone.utc)
hourly = sum(float(r.get("dph_total") or 0) for r in running)
lines = []
overdue = False

for r in running:
    rid = str(r.get("id"))
    label = r.get("label") or "no label"
    rate = float(r.get("dph_total") or 0)
    piece = "{} ({}) at ${:.4f}/h".format(rid, label, rate)

    start = r.get("start_date")
    if start:
        started = dt.datetime.fromtimestamp(float(start), dt.timezone.utc)
        elapsed = (now - started).total_seconds() / 3600
        piece += " - {:.2f} h elapsed, about ${:.2f} so far".format(elapsed, elapsed * rate)

    line = stop_lines.get(rid)
    if line:
        try:
            deadline = dt.datetime.fromisoformat(line)
            remaining = (deadline - now).total_seconds() / 3600
            if remaining <= 0:
                overdue = True
                piece += " - PAST ITS STOP LINE by {:.2f} h. Export and stop.".format(-remaining)
            else:
                piece += " - {:.2f} h left of its stop line".format(remaining)
        except Exception:
            pass
    else:
        piece += " - no stop line recorded (add one under active_stop_lines)"
    lines.append(piece)

header = "Vast.ai still running, about ${:.2f}/h total:".format(hourly)
if overdue:
    header = "Vast.ai PAST A STOP LINE, about ${:.2f}/h total:".format(hourly)
body = "\n".join("  " + p for p in lines)
tail = "Stop with: vastai stop instance <id>   Destroy with: vastai destroy instance <id> -y"
print(json.dumps({"systemMessage": "{}\n{}\n{}".format(header, body, tail)}))
' "${ledger}"
exit 0
