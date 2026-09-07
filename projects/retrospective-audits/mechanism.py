"""Post-hoc mechanism reading for the no-definition arms, from raw branch rollouts.

  python3 mechanism.py tasks/schedule-main.json

Reasoning content is encrypted and summaries are empty, so deliberation is measured, not
read: tool calls during the audit, reasoning tokens summed over an audit's model calls,
and what the explanations key on. All of this was framed after seeing the data.
"""
import json
import re
import statistics
import sys
from pathlib import Path

CONDS = ("audit_first", "withheld", "reveal_first", "audit_first_nodef", "reveal_first_nodef")
FEATURES = {
    "exit / nonzero / unsuccessfully": r"\bexit(ed)?\b|\bnonzero\b|\bunsuccessful",
    "targeted / subset passed": r"\btargeted\b|\bsubset\b|\b3/3\b",
    "incomplete / could not run": r"\bincomplete\b|\bcould not (run|complete)\b|\bnot complete\b|\bdid not (complete|finish)\b|\bremained\b",
    "cache": r"\bcache\b",
    "'also'": r"\balso\b",
}


def branches(schedule_path):
    rows = []
    for e in json.load(open(schedule_path))["sequence"]:
        run = Path(e["run_dir"])
        if not (run / "result.json").exists():
            continue
        r = json.load(open(run / "result.json"))
        later = open(run / "verifier" / "run.stdout").read().strip().splitlines()[-1].replace("RESULT: ", "")
        for name, v in r["branch_records"].items():
            calls = reason = 0
            cmds = []
            saw_cache = False
            for line in open(run / "branches" / name / "rollout.jsonl"):
                d = json.loads(line)
                p = d.get("payload", {})
                t = p.get("type") or d.get("type")
                if t == "token_count":
                    calls += 1
                    reason += p["info"]["last_token_usage"]["reasoning_output_tokens"]
                elif t in ("function_call", "custom_tool_call"):
                    a = str(p.get("arguments") or p.get("input"))
                    m = re.search(r'"cmd":"(.*?)","workdir', a)
                    cmds.append((m.group(1) if m else a)[:80])
                elif t in ("function_call_output", "custom_tool_call_output"):
                    out = p.get("output")
                    out = out if isinstance(out, str) else json.dumps(out)
                    if '"tests/test_reports.py::' in out:
                        saw_cache = True
            a = v["audit_response"]
            rows.append({"run": run.name, "cond": v["condition"], "sample": v["sample"], "later": later,
                         "then": a["then_verification_status"] if a else None, "expl": a["explanation"] if a else "",
                         "calls": calls, "reason": reason, "cmds": cmds, "saw_cache": saw_cache})
    return rows


def main():
    rows = branches(sys.argv[1] if len(sys.argv) > 1 else "tasks/schedule-main.json")
    print(f"branches: {len(rows)}\n")
    print("== tool use during the audit, and reasoning tokens summed over model calls ==")
    for c in CONDS:
        xs = [r for r in rows if r["cond"] == c]
        used = [r for r in xs if r["cmds"]]
        print(f"{c:19s} ran tools {len(used):2d}/{len(xs)}   median reasoning {statistics.median(r['reason'] for r in xs):5.0f}   "
              f"FAIL among tool-users {sum(r['then'] == 'FAIL' for r in used)}/{len(used)}   among non-users "
              f"{sum(r['then'] == 'FAIL' for r in xs if not r['cmds'])}/{len(xs) - len(used)}")
    nodef = [r for r in rows if r["cond"].endswith("nodef")]
    print("\n== reasoning by THEN answer, no-definition arms ==")
    for c in ("audit_first_nodef", "reveal_first_nodef"):
        for then in ("UNKNOWN", "FAIL", "PASS"):
            xs = [r["reason"] for r in nodef if r["cond"] == c and r["then"] == then]
            if xs:
                print(f"  {c:19s} THEN={then:7s} n={len(xs):2d} median {statistics.median(xs):5.0f}")
    print("\n== what AUDIT-NODEF audits ran (top commands) ==")
    from collections import Counter
    for cmd, k in Counter(cmd for r in nodef if r["cond"] == "audit_first_nodef" for cmd in r["cmds"]).most_common(6):
        print(f"  {k:2d}x  {cmd}")
    an = [r for r in nodef if r["cond"] == "audit_first_nodef"]
    saw = [r for r in an if r["saw_cache"]]
    print(f"\n== pytest lastfailed cache (AUDIT-NODEF) ==\n  saw cache contents {len(saw)}/{len(an)}: FAIL {sum(r['then'] == 'FAIL' for r in saw)}/{len(saw)}; "
          f"did not: FAIL {sum(r['then'] == 'FAIL' for r in an if not r['saw_cache'])}/{len(an) - len(saw)}")
    print("\n== explanation features, no-definition arms (all samples) ==")
    fail = [r for r in nodef if r["then"] == "FAIL"]
    unk = [r for r in nodef if r["then"] == "UNKNOWN"]
    for k, pat in FEATURES.items():
        f = sum(bool(re.search(pat, r["expl"], re.I)) for r in fail)
        u = sum(bool(re.search(pat, r["expl"], re.I)) for r in unk)
        print(f"  {k:32s} FAIL {f:2d}/{len(fail)} ({100 * f / len(fail):3.0f}%)   UNKNOWN {u:2d}/{len(unk)} ({100 * u / len(unk):3.0f}%)")
    also = re.compile(r"\balso\b")
    print("\n== 'also' among REVEAL-NODEF FAIL readings, by revealed result ==")
    for later in ("FAIL", "PASS"):
        xs = [r for r in nodef if r["cond"] == "reveal_first_nodef" and r["later"] == later and r["then"] == "FAIL"]
        print(f"  revealed {later}: {sum(bool(also.search(r['expl'])) for r in xs)}/{len(xs)}")
    d = [r for r in rows if r["cond"] == "reveal_first"]
    print(f"  (definitions arm REVEAL-FIRST, any answer: {sum(bool(also.search(r['expl'])) for r in d)}/{len(d)})")


if __name__ == "__main__":
    main()
