"""Experiment 3 analysis (prereg-3.md). Reads runs/<parent>/debias/result.json per scheduled parent.

  python3 analyze_debias.py tasks/schedule-main.json

Primary unit: per-trajectory mean of the three samples per condition. Sample 1 alone is a
secondary. Each contrast is complete-case for its own conditions. The randomization test and
bootstrap are the frozen Experiment 2 ones, imported from analyze_belief.
"""
import json
import sys
from pathlib import Path

from analyze_belief import boot_ci, cond_value, mean, sign_flip

CONDS = ("withheld", "reveal_fail", "withheld_debias", "reveal_fail_debias")
RESIDUAL_FLOOR = -2.0  # "no detectable residual" needs the bootstrap interval entirely above this


def load(schedule_path):
    rows = []
    for e in json.load(open(schedule_path))["sequence"]:
        run = Path(e["run_dir"])
        f = run / "debias" / "result.json"
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


def cells(rows, sample, *conds):
    """Complete-case for exactly the conditions this contrast uses."""
    out = []
    for t in rows:
        vs = [cond_value(t, c, sample) for c in conds]
        if all(v is not None for v in vs):
            out.append(vs)
    return out


def report_diffs(diffs, label, predicted=+1):
    p1, p2 = sign_flip([predicted * d for d in diffs])
    lo, hi = boot_ci(diffs)
    print(f"  {label:34s} mean {mean(diffs):+6.1f}  n={len(diffs):2d}  bootstrap 95% [{lo:+.1f}, {hi:+.1f}]"
          f"  one-sample symmetry (sign-flip): one-sided (predicted direction) p={p1:.4f} two-sided p={p2:.4f}")
    return {"diffs": diffs, "mean": mean(diffs), "n": len(diffs), "p1": p1, "p2": p2, "lo": lo, "hi": hi}


def report(rows, sample, label):
    print(f"\n==== {label} ====")
    tasks = sorted({t["task"] for t in rows})
    for c in CONDS:
        vals = [cond_value(t, c, sample) for t in rows]
        by_task = "  ".join(f"{task.split('-')[0]}={mean([v for t, v in zip(rows, vals) if t['task'] == task and v is not None]):.1f}"
                            for task in tasks)
        vals = [v for v in vals if v is not None]
        print(f"  {c:18s} mean {mean(vals):5.1f}  n={len(vals):2d}  by task: {by_task}")
    r1 = report_diffs([rf - w for w, rf in cells(rows, sample, "withheld", "reveal_fail")],
                      "R1 replication rF - w", predicted=-1)
    m1 = report_diffs([(rfd - wd) - (rf - w) for w, rf, wd, rfd in cells(rows, sample, *CONDS)],
                      "M1 interaction (rF_D-w_D)-(rF-w)", predicted=+1)
    res = report_diffs([rfd - wd for wd, rfd in cells(rows, sample, "withheld_debias", "reveal_fail_debias")],
                       "residual rF_D - w_D", predicted=-1)
    no_residual = res["lo"] > RESIDUAL_FLOOR
    print(f"  residual interval entirely above {RESIDUAL_FLOOR:+.1f}: {'YES' if no_residual else 'no'}"
          f"   [lo {res['lo']:+.1f}] -> 'no detectable residual' {'may' if no_residual else 'may NOT'} be claimed")
    verdict(r1, m1, res, no_residual)
    return r1, m1, res


def verdict(r1, m1, res, no_residual):
    """Fixed interpretation table, prereg-3.md. R1 failure is reported first and takes priority."""
    r1_ok = r1["mean"] < 0 and r1["p1"] < 0.05
    m1_ok = m1["mean"] > 0 and m1["p1"] < 0.05
    res_neg = res["mean"] < 0 and res["p1"] < 0.05
    print("  --- fixed interpretation (prereg-3.md) ---")
    if not r1_ok:
        print("  R1 NOT SUPPORTED: the headline Experiment 2 effect failed to replicate."
              " This takes priority over any mitigation claim.")
    else:
        print("  R1 supported: the failure-record effect replicated (rF - w negative, one-sided p < 0.05).")
        print(f"  M1: {'ATTENUATES' if m1_ok else 'no significant attenuation'}"
              f" (interaction {m1['mean']:+.1f}, one-sided p={m1['p1']:.4f}).")
    print(f"  Residual: {'a residual effect PERSISTS' if res_neg else 'residual not significantly negative'}"
          f" (rF_D - w_D {res['mean']:+.1f}, one-sided p={res['p1']:.4f}).")
    if no_residual:
        print("  NO DETECTABLE RESIDUAL may be claimed (interval above -2.0). 'Removed' is never claimed.")
    if r1_ok and not m1_ok and not no_residual:
        print("  INCONCLUSIVE: attenuation not significant and the residual interval spans -2.0.")


def main():
    rows = load(sys.argv[1] if len(sys.argv) > 1 else "tasks/schedule-main.json")
    print(f"parents with debias data: {len(rows)}")
    if not rows:
        print("no debias results yet; nothing to analyze")
        return
    report(rows, None, "PRIMARY UNIT: mean of 3 samples per condition")
    report(rows, 1, "SECONDARY: sample 1 only")
    print("\n== per-trajectory (mean of 3): index task later | w rF w_D rF_D | untreated treated interaction ==")
    for t in rows:
        w, rf, wd, rfd = (cond_value(t, c) for c in CONDS)
        fmt = lambda v: "  --" if v is None else f"{v:4.0f}"
        d = lambda a, b: None if a is None or b is None else a - b
        unt, tre = d(rf, w), d(rfd, wd)
        dfmt = lambda v: "   --" if v is None else f"{v:+5.1f}"
        print(f"  {t['index']:2d} {t['task'].split('-')[0]} {t['later']:4s} | {fmt(w)} {fmt(rf)} {fmt(wd)} {fmt(rfd)}"
              f" | {dfmt(unt)} {dfmt(tre)} {dfmt(d(tre, unt))}")
    print("\n== missingness by condition (all samples) ==")
    for c in CONDS:
        tot = sum(len(t["vals"].get(c, {})) for t in rows)
        miss = sum(v is None for t in rows for v in t["vals"].get(c, {}).values())
        print(f"  {c:18s} {miss}/{tot}")


if __name__ == "__main__":
    main()
