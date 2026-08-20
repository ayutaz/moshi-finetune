#!/usr/bin/env bash
# PreToolUse(Bash, git commit *): refuse to commit while the suite is red.
#
# tests/ holds the M1 data gates: raw-audio checksums, evaluation-registry coverage,
# voice-set agreement with the corpus manifest, and the bundled MIT notice. Those are
# the guarantees the experiment rests on, so a broken one must not reach a commit.
#
# `python -m pytest`, not `pytest`: the tests import `tools` and `models` from the
# repository root, and only the module form puts the working directory on sys.path.

set -u

# Check the command ourselves rather than relying on the settings `if` filter, which was
# observed firing this hook on unrelated Bash calls. That mattered: a red suite is the
# normal mid-TDD state, so gating every command on a green suite blocks the work.
command_text=$(jq -r '.tool_input.command // empty' 2>/dev/null)
case "${command_text}" in
  *"git commit"*) ;;
  *) exit 0 ;;
esac

[ -d tests ] || exit 0

if command -v uv >/dev/null 2>&1; then
  output=$(uv run --python 3.12 --with pytest --no-sync python -m pytest tests -q 2>&1)
elif python3 -c 'import pytest' >/dev/null 2>&1; then
  output=$(python3 -m pytest tests -q 2>&1)
else
  exit 0
fi

status=$?
[ "$status" -eq 0 ] && exit 0

{
  echo "Commit blocked: the experiment data gates or unit tests are failing."
  echo "$output" | tail -12
} >&2
exit 2
