#!/usr/bin/env bash
# PostToolUse(Write|Edit): keep edited Python formatted and lint-clean.
#
# CI runs `ruff check` and `ruff format --check` over the whole repository. Without this,
# drift accumulates silently until CI goes red, which is what happened on the
# experiment branch (14 lint errors and 7 unformatted files).
#
# Never fails the tool call: a formatter problem must not block work.

set -u

file=$(jq -r '.tool_response.filePath // .tool_input.file_path // empty' 2>/dev/null)
case "$file" in
  *.py) ;;
  *) exit 0 ;;
esac
[ -f "$file" ] || exit 0

if command -v ruff >/dev/null 2>&1; then
  ruff format "$file" >/dev/null 2>&1
  ruff check --fix "$file" >/dev/null 2>&1
elif command -v uv >/dev/null 2>&1; then
  uv run --with ruff --no-sync ruff format "$file" >/dev/null 2>&1
  uv run --with ruff --no-sync ruff check --fix "$file" >/dev/null 2>&1
fi
exit 0
