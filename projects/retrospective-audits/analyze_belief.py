"""Experiment 2 analysis (prereg-2.md). Reads runs/<parent>/belief/result.json per scheduled parent.

  python3 analyze_belief.py tasks/schedule-main.json

Primary unit: per-trajectory mean of the three samples per condition. Sample 1 alone is a
secondary. Each contrast is complete-case for its own conditions.
"""
import json
import sys
from pathlib import Path
from random import Random

CONDS = ("baseline", "withheld", "reveal_pass", "reveal_fail")


def load(schedule_path):
    rows = []
    for e in json.load(open(schedule_path))["sequence"]:
        run = Path(e["run_dir"])
        f = run / "belief" / "result.json"
        if not f.exists():
            continue
        r = json.load(open(f))
        if not r["checks"]["pipeline_passed"]:
            print(f"  skipped {run.name}: pipeline failed {[k for k, v in r['checks'].items() if v is False]}")
            continue
        rec = {"run": run.name, "task": e["task"], "later": r["later_result"], "index": e["index"], "vals": {}, "expl": {}}
        for v in r["branch_records"].values():
            a = v["answer"]
            rec["vals"].setdefault(v["condition"], {})[v["sample"]] = None if a is None else a["p_pass_then"]
            rec["expl"].setdefault(v["condition"], {})[v["sample"]] = None if a is None else a["explanation"]
        rows.append(rec)
    return rows


def cond_value(t, c, sample=None):
    vals = [v for v in t["vals"].get(c, {}).values() if v is not None] if sample is None else [t["vals"].get(c, {}).get(sample)]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def sign_flip(diffs, n=20000, seed=0):
    """Randomization test: flipping a paired difference = swapping the two labels within a trajectory."""
    if not diffs:
        return float("nan"), float("nan")
    rng = Random(seed)
    obs = mean(diffs)
    ge = ge_abs = 0
    for _ in range(n):
        m = mean([d if rng.random() < 0.5 else -d for d in diffs])
        ge += m >= obs - 1e-12
        ge_abs += abs(m) >= abs(obs) - 1e-12
    return (ge + 1) / (n + 1), (ge_abs + 1) / (n + 1)


def boot_ci(diffs, n=4000, seed=0):
    if len(diffs) < 2:
        return float("nan"), float("nan")
    rng = Random(seed)
    ms = sorted(mean([diffs[rng.randrange(len(diffs))] for _ in diffs]) for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n) - 1]


def contrast(rows, a, b, sample, label, randomization, predicted=+1):
    pairs = [(cond_value(t, a, sample), cond_value(t, b, sample)) for t in rows]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    diffs = [x - y for x, y in pairs]
    p1, p2 = sign_flip([predicted * d for d in diffs])  # one-sided p in the predicted direction
    lo, hi = boot_ci(diffs)
    kind = "sign-flip randomization" if randomization else "one-sample symmetry (sign-flip)"
    print(f"  {label:34s} mean {mean(diffs):+6.1f}  n={len(diffs):2d}  bootstrap 95% [{lo:+.1f}, {hi:+.1f}]  {kind}: one-sided (predicted direction) p={p1:.4f} two-sided p={p2:.4f}")
    return diffs, pairs


def report(rows, sample, label):
    print(f"\n==== {label} ====")
    for c in CONDS:
        vals = [cond_value(t, c, sample) for t in rows]
        vals = [v for v in vals if v is not None]
        by_task = "  ".join(f"{task.split('-')[0]}={mean([v for t, v in zip(rows, [cond_value(t, c, sample) for t in rows]) if t['task'] == task and v is not None]):.1f}"
                            for task in sorted({t['task'] for t in rows}))
        print(f"  {c:12s} mean {mean(vals):5.1f}  n={len(vals):2d}  by task: {by_task}")
    spread, _ = contrast(rows, "reveal_pass", "reveal_fail", sample, "PRIMARY spread rP - rF", True)
    dP, _ = contrast(rows, "reveal_pass", "withheld", sample, "direction rP - w", False)
    dF, _ = contrast(rows, "reveal_fail", "withheld", sample, "direction rF - w", False, predicted=-1)
    contrast(rows, "withheld", "baseline", sample, "record presence w - b", False)
    # headroom-normalized secondary
    hp = [(cond_value(t, "reveal_pass", sample) - cond_value(t, "withheld", sample)) / (100 - cond_value(t, "withheld", sample))
          for t in rows if cond_value(t, "reveal_pass", sample) is not None and cond_value(t, "withheld", sample) not in (None, 100)]
    hf = [(cond_value(t, "withheld", sample) - cond_value(t, "reveal_fail", sample)) / cond_value(t, "withheld", sample)
          for t in rows if cond_value(t, "reveal_fail", sample) is not None and cond_value(t, "withheld", sample) not in (None, 0)]
    print(f"  headroom-normalized: PASS-side rise {mean(hp):+.2f} of available headroom (n={len(hp)}); FAIL-side drop {mean(hf):+.2f} of available (n={len(hf)})")
    h1 = mean(spread) >= 10 and mean(dP) > 0 and mean(dF) < 0
    print(f"  H1 rule (spread >= 10 and rP - w > 0 and rF - w < 0): {'SUPPORTED' if h1 else 'not supported'}"
          f"   [spread {mean(spread):+.1f}; rP - w {mean(dP):+.1f}; rF - w {mean(dF):+.1f}]")
    return spread


def main():
    rows = load(sys.argv[1] if len(sys.argv) > 1 else "tasks/schedule-main.json")
    print(f"parents with belief data: {len(rows)}")
    if not rows:
        return
    spread = report(rows, None, "PRIMARY UNIT: mean of 3 samples per condition")
    report(rows, 1, "SECONDARY: sample 1 only")
    print("\n== per-trajectory (mean of 3): index task later | b w rP rF | spread ==")
    for t in rows:
        b, w, rp, rf = (cond_value(t, c) for c in CONDS)
        fmt = lambda v: "  --" if v is None else f"{v:4.0f}"
        sp = "  --" if rp is None or rf is None else f"{rp - rf:+4.0f}"
        print(f"  {t['index']:2d} {t['task'].split('-')[0]} {t['later']:4s} | {fmt(b)} {fmt(w)} {fmt(rp)} {fmt(rf)} | {sp}")
    print("\n== H2 (exploratory): spread by WITHHELD level ==")
    unc = [cond_value(t, "reveal_pass") - cond_value(t, "reveal_fail") for t in rows if cond_value(t, "withheld") is not None and 25 <= cond_value(t, "withheld") <= 75 and cond_value(t, "reveal_pass") is not None and cond_value(t, "reveal_fail") is not None]
    ext = [cond_value(t, "reveal_pass") - cond_value(t, "reveal_fail") for t in rows if cond_value(t, "withheld") is not None and not 25 <= cond_value(t, "withheld") <= 75 and cond_value(t, "reveal_pass") is not None and cond_value(t, "reveal_fail") is not None]
    print(f"  w in [25,75]: mean spread {mean(unc):+.1f} (n={len(unc)});  extremes: {mean(ext):+.1f} (n={len(ext)})")
    print("\n== spread by task (= by true later result; confounded by design) ==")
    for task in sorted({t["task"] for t in rows}):
        xs = [cond_value(t, "reveal_pass") - cond_value(t, "reveal_fail") for t in rows if t["task"] == task and cond_value(t, "reveal_pass") is not None and cond_value(t, "reveal_fail") is not None]
        print(f"  {task}: mean spread {mean(xs):+.1f} (n={len(xs)})")
    print("\n== explanation framing (all samples; post-hoc coding, patterns fixed 2026-08-30 13:25) ==")
    import re
    pats = {
        "risk-emphasis": r"\bbut\b.*(unexecuted|never (executed|ran|reached)|risk|uncertain)",
        "discounting": r"\b(only|merely)\b.*(fixture|setup)",
        "mentions record": r"KESTREL|record|verification run|post-report|subsequent",
        "dispute-like": r"implausible|surprising|unexpected|doubt|unlikely that|contradict|cannot explain",
    }
    for c in CONDS:
        xs = [x for t in rows for x in t["expl"].get(c, {}).values() if x]
        counts = "   ".join(f"{k} {sum(bool(re.search(pat, x, re.I)) for x in xs)}/{len(xs)}" for k, pat in pats.items())
        print(f"  {c:12s} {counts}")
    print("\n== missingness by condition (all samples) ==")
    for c in CONDS:
        tot = sum(len(t["vals"].get(c, {})) for t in rows)
        miss = sum(v is None for t in rows for v in t["vals"].get(c, {}).values())
        print(f"  {c:12s} {miss}/{tot}")


if __name__ == "__main__":
    main()
