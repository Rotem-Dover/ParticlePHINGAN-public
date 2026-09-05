#!/usr/bin/env bash
# PostToolUse: when a file under src/physics/<process>/ is edited, run the
# matching test directory if one exists. Best-effort, non-blocking on failure.
set -u

payload="$(cat)"
file_path="$(printf '%s' "$payload" | python3 -c 'import json,sys; d=json.load(sys.stdin); print((d.get("tool_input") or {}).get("file_path",""))' 2>/dev/null)"

[ -z "$file_path" ] && exit 0

# Only act on edits under src/physics/<process>/
case "$file_path" in
  */src/physics/*) ;;
  *) exit 0 ;;
esac

# Extract the process name (e.g. g4h_ionisation)
process="$(printf '%s' "$file_path" | sed -E 's|.*/src/physics/([^/]+)/.*|\1|')"
[ -z "$process" ] && exit 0

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
test_dir="$repo_root/tests/physics/$process"

[ -d "$test_dir" ] || exit 0

cd "$repo_root" || exit 0
echo "[hook] running pytest for $process"
PYTHONPATH=src .venv/bin/pytest "$test_dir" -q --no-header -x 2>&1 | tail -40
# Always exit 0 so failing tests don't block subsequent edits — they're just informational.
exit 0
