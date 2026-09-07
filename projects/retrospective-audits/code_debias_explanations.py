"""Post-hoc Experiment 3 explanation coding, fixed after data.

  python3 code_debias_explanations.py tasks/schedule-main.json

The quantitative Experiment 3 conclusions do not depend on these lexical patterns. This
script records the exact patterns behind the qualitative counts in the write-up.
"""
import json
import re
import sys
from pathlib import Path

CONDS = ("withheld", "reveal_fail", "withheld_debias", "reveal_fail_debias")
PATTERNS = {
    "risk-emphasis": r"\bbut\b.*(unexecuted|never (executed|ran|reached)|risk|uncertain)",
    "discounting": r"\b(only|merely)\b.*(fixture|setup)",
    "synthetic-record": r"KESTREL|record|verification run|post-report|subsequent",
    "instruction": r"\binstruct(?:ion|ed|s)?\b|do not let|\bignore\b|influence your estimate",
}
POSSIBLE_VERIFICATION = r"\bverification\b"


def main():
    schedule = Path(sys.argv[1] if len(sys.argv) > 1 else "tasks/schedule-main.json")
    rows = {condition: [] for condition in CONDS}
    for entry in json.loads(schedule.read_text())["sequence"]:
        run = Path(entry["run_dir"])
        for branch in sorted((run / "debias" / "branches").iterdir()):
            if not branch.is_dir():
                continue
            condition = branch.name.split("__")[0]
            answer = json.loads((branch / "answer.json").read_text())
            rows[condition].append((entry["index"], branch.name, answer["explanation"]))

    print("Experiment 3 explanation coding (post-hoc; all three samples)")
    for condition in CONDS:
        explanations = rows[condition]
        counts = "  ".join(
            f"{label} {sum(bool(re.search(pattern, text, re.I)) for _, _, text in explanations)}/{len(explanations)}"
            for label, pattern in PATTERNS.items()
        )
        print(f"{condition:20s} {counts}")

    print("\nBroad 'verification' scan for manual disambiguation:")
    for condition in CONDS:
        for index, branch, text in rows[condition]:
            if re.search(POSSIBLE_VERIFICATION, text, re.I):
                print(f"run {index:02d}  {branch:24s}  {text}")


if __name__ == "__main__":
    main()
