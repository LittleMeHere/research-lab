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


def refusal_vectors():
    """model -> {quant -> length-100 refusal vector (keyword)}."""
    out = {}
    for p in sorted(glob.glob(os.path.join(DATA_DIR, "v2_results_*.json"))):
        d = json.load(open(p, encoding="utf-8"))
        vecs = {}
        for q, ql in d.get("quant_levels", {}).items():
            if REFUSAL_KEY in ql:
                rows = sorted(ql[REFUSAL_KEY]["results"], key=lambda r: r["idx"])
                vecs[q] = np.array([1 if r["refused"] else 0 for r in rows])
        out[d["model_id"]] = vecs
    return out


def capability_vectors():
    """model -> {quant -> capability vector (judge), ordered by idx}."""
    res = json.load(open(CAP_FILE, encoding="utf-8"))
    grouped = {}
    for v in res.values():
        grouped.setdefault((v["model"], v["quant"]), []).append((v["idx"], 1 if v["correct"] else 0))
    out = {}
    for (mid, q), pairs in grouped.items():
        pairs.sort()
        out.setdefault(mid, {})[q] = np.array([c for _, c in pairs])
    return out


def delta(base, other):
    d = (other.mean() - base.mean()) * 100
    lo, hi = boot_ci(base, other)
    return d, lo, hi, mcnemar_p(base, other)


def main():
    if not os.path.exists(CAP_FILE):
        print("Run `python judge_rescore.py capability` first."); return
    refu = refusal_vectors()
    capa = capability_vectors()

    print("=" * 84)
    print("  SAFETY vs CAPABILITY under nf4 (fp16 -> nf4_dq)")
    print("  safety = keyword refusal (n=100) | capability = LLM-judge TruthfulQA (n=50)")
    print("  delta<0 = degraded. * = McNemar p<0.05. CI = 95% paired bootstrap.")
    print("=" * 84)
    print(f"  {'model':<22} {'safety Δ':>9} {'95% CI':>12} {'p':>6}   "
          f"{'cap Δ':>7} {'95% CI':>12} {'p':>6}   verdict")
    print("-" * 84)

    for mid in sorted(refu, key=short):
        rv, cv = refu.get(mid, {}), capa.get(mid, {})
        if "fp16" not in rv or "nf4_dq" not in rv or "fp16" not in cv or "nf4_dq" not in cv:
            continue
        sd, slo, shi, sp = delta(rv["fp16"], rv["nf4_dq"])
        cd, clo, chi, cp = delta(cv["fp16"], cv["nf4_dq"])
        smark = "*" if sp < 0.05 else " "
        cmark = "*" if cp < 0.05 else " "

        # thesis = safety degrades FASTER (more negative) than capability
        if sp < 0.05 and cp >= 0.05 and sd < 0:
            v = "safety drops, cap holds -> THESIS"
        elif cp < 0.05 and sp >= 0.05 and cd < 0:
            v = "cap drops, safety holds -> INVERTED"
        elif sp >= 0.05 and cp >= 0.05:
            v = "both within noise"
        elif sd < cd:
            v = "safety drops more"
        else:
            v = "capability drops more"

        print(f"  {short(mid):<22} {sd:>+8.0f} [{slo:>+4.0f},{shi:>+4.0f}] {sp:>6.3f}{smark}  "
              f"{cd:>+6.0f} [{clo:>+4.0f},{chi:>+4.0f}] {cp:>6.3f}{cmark}  {v}")

    print("-" * 84)
    print("  Read: with a working capability axis, compare the two drops per model.")
    print("  The v1 thesis predicts the safety column degrades while capability holds.")
    print("=" * 84)


if __name__ == "__main__":
    main()
