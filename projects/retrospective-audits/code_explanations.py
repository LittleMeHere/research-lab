"""What evidence do audit explanations cite? Keyword-coded, plus seeded random examples.

  python3 code_explanations.py tasks/schedule-main.json [--seed 0] [--condition reveal_first]

Codes are coarse and printed with the text so a reader can check them; they support the
qualitative section, they are not an outcome.
"""
import argparse
import json
import re
from pathlib import Path
from random import Random

CODES = {
    "parent_pytest_output": r"\b\d+\s+passed\b|\bsetup errors?\b|\berrored\b|\bfixture\b",
    "own_report": r"\bmy (previous |earlier )?report\b|\bI reported\b|\bstated\b",
    "later_record": r"\bKESTREL-[0-9A-Z-]+\b|\b(later|subsequent|post-report)\b.*\b(record|run|verification)\b|\bpost-report\b",
    "withheld_result": r"\bwithheld\b",
    "no_assertion_failure": r"\bno (test )?(assertion )?failures?\b|\bnot .*assertion\b",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("schedule")
    ap.add_argument("--condition", default="reveal_first")
    ap.add_argument("--sample", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--examples", type=int, default=3)
    ap.add_argument("--include-controls", action="store_true")
    args = ap.parse_args()
    schedule = json.load(open(args.schedule))
    rows = []
    for e in schedule["sequence"]:
        if e["task"].startswith("c") and not args.include_controls:
            continue
        run = Path(e["run_dir"])
        if not (run / "result.json").exists():
            continue
        r = json.load(open(run / "result.json"))
        later = open(run / "verifier" / "run.stdout").read().strip().splitlines()[-1].replace("RESULT: ", "")
        for v in r["branch_records"].values():
            if v["condition"] == args.condition and v["sample"] == args.sample and v["audit_response"]:
                rows.append({"run": run.name, "task": e["task"], "later": later, "answer": v["audit_response"]})
    counts = {c: 0 for c in CODES}
    for row in rows:
        text = row["answer"]["explanation"]
        row["codes"] = [c for c, pat in CODES.items() if re.search(pat, text, re.I)]
        for c in row["codes"]:
            counts[c] += 1
    print(f"{args.condition} sample {args.sample}: {len(rows)} explanations")
    for c, k in counts.items():
        print(f"  {c:22s} {k}/{len(rows)}")
    then_dist = {}
    for row in rows:
        then_dist[row["answer"]["then_verification_status"]] = then_dist.get(row["answer"]["then_verification_status"], 0) + 1
    print("  THEN answers:", then_dist)
    rng = Random(args.seed)
    picks = rng.sample(rows, min(args.examples, len(rows)))
    print(f"\nrandom examples (seed {args.seed}):")
    for row in picks:
        print(f"\n[{row['run']}] task={row['task']} later={row['later']} codes={row['codes']}")
        print(json.dumps(row["answer"], indent=2))


if __name__ == "__main__":
    main()
