#!/usr/bin/env python3
"""Paired analysis of thinking-on and thinking-off refusal labels.

Every quantization configuration received the same 100 HarmBench rows in both
thinking modes. The FP16 comparison uses an exact McNemar test and paired bootstrap
interval. Results across the four quantization configurations are correlated
because they reuse prompts and related model states.

The script reads saved results and does not use a GPU or API.

Usage:
    python code/thinking_mode_analysis.py
"""

import glob
import json
import os
import sys

import numpy as np
from scipy.stats import binomtest

sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
QUANT_ORDER = ["fp16", "int8", "int4_fp4", "nf4_dq"]
OFF_KEY = "refusal_thinking=False"
ON_KEY = "refusal_thinking=True"
N_BOOTSTRAP = 10_000
RNG = np.random.default_rng(42)


def vec(quant_level: dict, key: str) -> np.ndarray:
    results = sorted(quant_level[key]["results"], key=lambda r: r["idx"])
    return np.array([1 if r["refused"] else 0 for r in results], dtype=int)


def mcnemar_exact(off: np.ndarray, on: np.ndarray) -> dict:
    """off = thinking-OFF refused, on = thinking-ON refused (paired)."""
    gained = int(np.sum((off == 0) & (on == 1)))   # thinking flipped comply->refuse
    lost = int(np.sum((off == 1) & (on == 0)))      # thinking flipped refuse->comply
    disc = gained + lost
    p = 1.0 if disc == 0 else binomtest(gained, disc, 0.5, alternative="two-sided").pvalue
    return {"gained": gained, "lost": lost, "p": p}


def bootstrap_delta_ci(off: np.ndarray, on: np.ndarray) -> tuple:
    n = len(off)
    idx = RNG.integers(0, n, size=(N_BOOTSTRAP, n))
    deltas = (on[idx].mean(axis=1) - off[idx].mean(axis=1)) * 100
    return tuple(np.percentile(deltas, [2.5, 97.5]))


def short(mid):
    return mid.split("/")[-1]


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "v2_results_*.json")))
    models = {json.load(open(f, encoding="utf-8"))["model_id"]: f for f in files}

    print("=" * 80)
    print("  THINKING-MODE REFUSAL AUDIT (thinking ON minus OFF, paired)")
    print("  delta>0 = higher refusal with thinking. 10k bootstrap, exact McNemar.")
    print("=" * 80)
    print(f"  {'model':<24} {'fp16 Δ':>7} {'95% CI':>13} {'p':>7}   {'sign @ 4 quant levels':>22}")
    print("-" * 80)

    summary = []
    for mid in sorted(models, key=short):
        d = json.load(open(models[mid], encoding="utf-8"))
        ql = d.get("quant_levels", {})
        if "fp16" not in ql or ON_KEY not in ql["fp16"]:
            continue

        # Paired comparison at FP16.
        off0, on0 = vec(ql["fp16"], OFF_KEY), vec(ql["fp16"], ON_KEY)
        d0 = (on0.mean() - off0.mean()) * 100
        lo, hi = bootstrap_delta_ci(off0, on0)
        mc = mcnemar_exact(off0, on0)

        # Direction of the observed difference at each quantization configuration.
        signs = []
        for q in QUANT_ORDER:
            if q in ql and ON_KEY in ql[q]:
                dq = (vec(ql[q], ON_KEY).mean() - vec(ql[q], OFF_KEY).mean()) * 100
                signs.append("+" if dq > 0.5 else "-" if dq < -0.5 else "0")
            else:
                signs.append("?")
        crosses0 = lo <= 0 <= hi

        print(f"  {short(mid):<24} {d0:>+6.0f} [{lo:>+5.0f},{hi:>+5.0f}] {mc['p']:>7.3f}   {' '.join(signs):>22}")
        summary.append({"model": short(mid), "d0": d0, "lo": lo, "hi": hi,
                        "p": mc["p"], "signs": signs, "crosses0": crosses0})

    print("-" * 80)
    excludes_zero = [s for s in summary if not s["crosses0"]]
    print(f"  FP16 intervals excluding zero: {len(excludes_zero)}/{len(summary)}")
    for s in excludes_zero:
        print(f"    {s['model']}: {s['d0']:+.0f}pp  CI[{s['lo']:+.0f},{s['hi']:+.0f}]  p={s['p']:.3f}")

    # Dependence across quantization configurations.
    print()
    print("  The four direction columns are not independent replications: they reuse")
    print("  the same prompts and related model states. Do not pool them as separate")
    print("  observations. These labels use the keyword heuristic; semantic rescoring")
    print("  is required before selecting a thinking-mode confirmation target.")
    print("=" * 80)


if __name__ == "__main__":
    main()
