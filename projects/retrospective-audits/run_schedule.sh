#!/bin/bash
# Run every entry of a schedule file in order.
#   ./run_schedule.sh tasks/schedule-main.json [validation-task-dir]
# Refuses to start unless every frozen hash in the schedule matches the working tree.
# Harness exit codes: 0 eligible, 2 ineligible (dropped, not replaced), 1 harness failure (abort).
# Logs go to ~/.codex-runs/logs, outside the model-readable filesystem.
set -u -o pipefail
cd "$(dirname "$0")"
SCHED=${1:?schedule file}; VALIDATE=${2:-}
L=~/.codex-runs/logs; mkdir -p "$L"; STAMP=$(date +%Y%m%d-%H%M%S)

python3 - "$SCHED" <<'PY' || { echo "=== ABORT: frozen hashes do not match the working tree"; exit 1; }
import hashlib, json, sys
from pathlib import Path
def tree_hash(root):
    h = hashlib.sha256()
    for f in sorted(p for p in Path(root).rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        h.update(str(f.relative_to(root)).encode()); h.update(f.read_bytes())
    return h.hexdigest()
s = json.load(open(sys.argv[1])); bad = []
for task, h in s["task_source_hashes"].items():
    if tree_hash(Path("tasks") / task) != h: bad.append(task)
for name, path in (("harness_sha256", "fork_smoke.py"), ("later_verify_sha256", "tasks/later_verify.py"),
                   ("analyze_sha256", "analyze.py"), ("label_sha256", "label.py"), ("runner_sha256", "run_schedule.sh")):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != s[name]: bad.append(path)
if tree_hash("tasks/_devtools") != s["devtools_tree_sha256"]: bad.append("tasks/_devtools")
if bad: print("hash mismatch:", bad); sys.exit(1)
print("frozen hashes match")
PY

if [ -n "$VALIDATE" ]; then
  echo "=== VALIDATE $VALIDATE $(date +%T)"
  python3 fork_smoke.py "runs/$STAMP-validate-$(basename "$VALIDATE")" --task-dir "$VALIDATE" > "$L/$STAMP-validate.log" 2>&1 < /dev/null
  rc=$?; echo "=== VALIDATE EXIT $rc $(date +%T)"; [ $rc -ne 0 ] && { echo "=== ABORT: validation failed"; exit 1; }
fi

eligible=0; ineligible=0; failed=0; skipped=0
while read -r idx task rdir shift; do
  if [ -d "$rdir" ]; then
    if [ -f "$rdir/result.json" ]; then echo "=== SKIP $idx $task (complete)"; skipped=$((skipped+1)); continue
    else echo "=== ABORT: incomplete run directory exists: $rdir"; exit 1; fi
  fi
  echo "=== START $idx $task shift=$shift $(date +%T)"
  python3 fork_smoke.py "$rdir" --task-dir "tasks/$task" --order-shift "$shift" > "$L/$STAMP-$idx.log" 2>&1 < /dev/null
  rc=$?; echo "=== EXIT $rc $idx $task $(date +%T)"
  case $rc in
    0) eligible=$((eligible+1)) ;;
    2) ineligible=$((ineligible+1)) ;;
    *) failed=$((failed+1)); echo "=== ABORT: harness failure on $idx (see $L/$STAMP-$idx.log)"; break ;;
  esac
done < <(python3 - "$SCHED" <<'PY'
import json, sys
for e in json.load(open(sys.argv[1]))["sequence"]:
    print(e["index"], e["task"], e["run_dir"], e.get("order_shift", 0))
PY
)
echo "=== SCHEDULE DONE eligible=$eligible ineligible=$ineligible failed=$failed skipped=$skipped $(date +%T)"
[ $failed -eq 0 ] || exit 1
