#!/bin/bash
# Experiment 2 over the 16 main parents (prereg-2.md). Order shift = (index-1) mod 4.
#   ./run_belief.sh tasks/schedule-main.json
# Refuses to start unless the hashes frozen in tasks/schedule-belief.json match the tree.
set -u -o pipefail
cd "$(dirname "$0")"
SCHED=${1:?schedule}; L=~/.codex-runs/logs; mkdir -p "$L"; STAMP=$(date +%Y%m%d-%H%M%S)

python3 - <<'PY' || { echo "=== ABORT: frozen hashes do not match the working tree"; exit 1; }
import hashlib, json, sys
from pathlib import Path
s = json.load(open("tasks/schedule-belief.json")); bad = []
for name, path in s["hashes"].items():
    if hashlib.sha256(Path(name).read_bytes()).hexdigest() != path: bad.append(name)
if bad: print("hash mismatch:", bad); sys.exit(1)
print("frozen hashes match")
PY

ok=0; failed=0; skipped=0
while read -r idx rdir; do
  if [ -d "$rdir/belief" ]; then
    if [ -f "$rdir/belief/result.json" ]; then echo "=== SKIP $idx (complete)"; skipped=$((skipped+1)); continue
    else echo "=== ABORT: incomplete belief dir exists: $rdir/belief"; exit 1; fi
  fi
  shift=$(( (idx - 1) % 4 ))
  echo "=== START $idx $rdir shift=$shift $(date +%T)"
  python3 fork_belief.py "$rdir" --order-shift "$shift" > "$L/$STAMP-belief-$idx.log" 2>&1 < /dev/null
  rc=$?; echo "=== EXIT $rc $idx $(date +%T)"
  if [ $rc -eq 0 ]; then ok=$((ok+1)); else failed=$((failed+1)); echo "=== ABORT: failure on $idx (see $L/$STAMP-belief-$idx.log)"; break; fi
done < <(python3 -c "
import json, sys
for e in json.load(open(sys.argv[1]))['sequence']: print(e['index'], e['run_dir'])" "$SCHED")
echo "=== BELIEF DONE ok=$ok failed=$failed skipped=$skipped $(date +%T)"
[ $failed -eq 0 ] || exit 1
