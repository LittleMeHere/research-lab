#!/usr/bin/env python3
"""
The actual thesis test: does safety degrade FASTER than capability under nf4?
=============================================================================

This could not be answered before, because the capability axis was broken
(TruthfulQA substring scoring = ~7% noise). judge_rescore.py rebuilt it with an
LLM judge + gold answers. Now we have a real per-prompt capability label, paired
across quant levels, so we can run the same McNemar + bootstrap rigor on BOTH
axes and compare the fp16->nf4 drop.

Thesis (v1): safety (refusal) degrades faster than capability.
Test: per model, is the refusal drop more negative than the capability drop?

Inputs (no GPU, no API):
  - data/v2_results_*.json          -> keyword refusal labels (n=100, thinking off)
  - data/judge_capability_results.json -> judge capability labels (n=50)

Usage:
    python capability_analysis.py
"""

import glob
import json
import os
import sys

import numpy as np
from scipy.stats import binomtest

sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CAP_FILE = os.path.join(DATA_DIR, "judge_capability_results.json")
REF_JUDGE_FILE = os.path.join(DATA_DIR, "judge_refusal_results.json")
REFUSAL_KEY = "refusal_thinking=False"
N_BOOT = 10_000
RNG = np.random.default_rng(42)


def mcnemar_p(base, other):
    b = int(np.sum((base == 1) & (other == 0)))
    c = int(np.sum((base == 0) & (other == 1)))
    return 1.0 if b + c == 0 else binomtest(b, b + c, 0.5, alternative="two-sided").pvalue


def boot_ci(base, other):
    n = len(base)
    idx = RNG.integers(0, n, size=(N_BOOT, n))
    d = (other[idx].mean(1) - base[idx].mean(1)) * 100
    return tuple(np.percentile(d, [2.5, 97.5]))


def short(mid):
    return mid.split("/")[-1]


def keyword_refusal():
    """model -> {quant -> {idx: refused 0/1}} from the keyword scorer."""
    out = {}
    for p in sorted(glob.glob(os.path.join(DATA_DIR, "v2_results_*.json"))):
        d = json.load(open(p, encoding="utf-8"))
        for q, ql in d.get("quant_levels", {}).items():
            if REFUSAL_KEY in ql:
                cells = {r["idx"]: (1 if r["refused"] else 0) for r in ql[REFUSAL_KEY]["results"]}
                out.setdefault(d["model_id"], {})[q] = cells
    return out


def judged(file, key, positive):
    """model -> {quant -> {idx: 1 if positive(verdict) else 0}} from a judge results file."""
    out = {}
    if not os.path.exists(file):
        return out
    for v in json.load(open(file, encoding="utf-8")).values():
        cell = out.setdefault(v["model"], {}).setdefault(v["quant"], {})
        cell[v["idx"]] = 1 if positive(v[key]) else 0
    return out


def paired_delta(a, b):
    """a,b are {idx: 0/1}. Pair on shared idx; return (delta_pp, lo, hi, p, n)."""
    shared = sorted(set(a) & set(b))
    if len(shared) < 5:
        return None
    base = np.array([a[i] for i in shared])
    other = np.array([b[i] for i in shared])
    d = (other.mean() - base.mean()) * 100
    lo, hi = boot_ci(base, other)
    return d, lo, hi, mcnemar_p(base, other), len(shared)


def fmt(res):
    if res is None:
        return f"{'n/a':>26}"
    d, lo, hi, p, n = res
    star = "*" if p < 0.05 else " "
    return f"{d:>+5.0f} [{lo:>+4.0f},{hi:>+4.0f}] p={p:.3f}{star} n={n:>3}"


def main():
    if not os.path.exists(CAP_FILE):
        print("Run `python judge_rescore.py capability` first."); return
    kw = keyword_refusal()
    jref = judged(REF_JUDGE_FILE, "verdict", lambda v: v == "REFUSE")
    cap = judged(CAP_FILE, "correct", bool)

    print("=" * 100)
    print("  SAFETY vs CAPABILITY under nf4 (fp16 -> nf4_dq), paired by prompt")
    print("  delta<0 = degraded. * = McNemar p<0.05. CI = 95% paired bootstrap. n = paired prompts.")
    print("=" * 100)
    print(f"  {'model':<22} {'refusal (keyword)':>26} {'refusal (JUDGE)':>26} {'capability (JUDGE)':>26}")
    print("-" * 100)

    for mid in sorted(kw, key=short):
        def cell(src):
            q = src.get(mid, {})
            return paired_delta(q["fp16"], q["nf4_dq"]) if "fp16" in q and "nf4_dq" in q else None
        kd, jd, cd = cell(kw), cell(jref), cell(cap)
        print(f"  {short(mid):<22} {fmt(kd):>26} {fmt(jd):>26} {fmt(cd):>26}")

    print("-" * 100)
    print("  The judge-on-both columns are the fair test: same Sonnet judge scores refusal")
    print("  AND capability. If refusal(JUDGE) stays n.s. while capability(JUDGE) drops, the")
    print("  inversion is not an artifact of the keyword scorer. Significant capability drops:")
    for mid in sorted(cap, key=short):
        q = cap.get(mid, {})
        if "fp16" in q and "nf4_dq" in q:
            r = paired_delta(q["fp16"], q["nf4_dq"])
            if r and r[3] < 0.05:
                print(f"    {short(mid)}: {r[0]:+.0f}pp (p={r[3]:.3f}, n={r[4]})")
    print("=" * 100)


if __name__ == "__main__":
    main()
