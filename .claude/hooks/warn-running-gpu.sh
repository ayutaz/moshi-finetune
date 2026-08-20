#!/usr/bin/env bash
# Stop: warn while a Vast.ai instance is still burning money.
#
# The costliest mistake available in this project is leaving a rented A100 running.
# Silent on machines without the vastai CLI, so it costs collaborators nothing.

set -u

command -v vastai >/dev/null 2>&1 || exit 0

vastai show instances --raw 2>/dev/null | python3 -c '
import json
import sys

try:
    rows = json.load(sys.stdin)
except Exception:
    sys.exit(0)

running = [r for r in rows if r.get("actual_status") == "running"]
if not running:
    sys.exit(0)

hourly = sum(float(r.get("dph_total") or 0) for r in running)
names = ", ".join(
    "{} ({})".format(r.get("id"), r.get("label") or "no label") for r in running
)
message = (
    "Vast.ai still running: {} - about ${:.2f}/h. "
    "Stop it with: vastai stop instance <id>".format(names, hourly)
)
print(json.dumps({"systemMessage": message}))
'
exit 0
