#!/usr/bin/env python3
"""Paired statistical analysis of the keyword-scored refusal results.

Each quantization configuration received the same 100 HarmBench rows. This script
uses that prompt pairing in two analyses:

1. McNemar's exact test compares refusal-to-compliance and
   compliance-to-refusal changes.
2. A paired bootstrap reports a 95% interval for each percentage-point
   difference.

The script reads saved JSON results and does not load a model.

Usage:
    python code/stats_analysis.py
"""

import glob
import json
import os
import sys

import numpy as np
from scipy.stats import binomtest

sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
QUANT_ORDER = ["fp16", "int8", "int4_fp4", "nf4_dq"]
REFUSAL_KEY = "refusal_thinking=False"   # non-thinking mode, present in all 6 files
N_BOOTSTRAP = 10_000
RNG = np.random.default_rng(42)          # fixed seed => reproducible intervals


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def refusal_vector(quant_level: dict) -> np.ndarray:
    """Return a length-100 array of 0/1 refusal flags, ordered by prompt idx.

    Ordering by idx is what makes the comparison *paired*: position i is the
    same HarmBench prompt at every quant level.
    """
    results = quant_level[REFUSAL_KEY]["results"]
    by_idx = sorted(results, key=lambda r: r["idx"])
    return np.array([1 if r["refused"] else 0 for r in by_idx], dtype=int)


def load_models() -> dict:
    """model_id -> {quant_name -> refusal vector}."""
    models = {}
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "v2_results_*.json"))):
        d = json.load(open(path, encoding="utf-8"))
        vecs = {}
        for q in QUANT_ORDER:
            ql = d.get("quant_levels", {}).get(q)
            if ql and REFUSAL_KEY in ql:
                vecs[q] = refusal_vector(ql)
        models[d["model_id"]] = vecs
    return models


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def mcnemar_exact(base: np.ndarray, other: np.ndarray) -> dict:
    """Exact McNemar test on two paired binary vectors (base vs other).

    b = lost   : refused at base, complied at other
    c = gained : complied at base, refused at other

    Under the null hypothesis (quantization doesn't change refusal), each
    discordant prompt is equally likely to flip either way, so b ~ Binomial(b+c, 0.5).
    The exact two-sided p-value is how surprising the observed split is.
    """
    lost = int(np.sum((base == 1) & (other == 0)))     # b
    gained = int(np.sum((base == 0) & (other == 1)))   # c
    discordant = lost + gained
    if discordant == 0:
        p = 1.0
    else:
        p = binomtest(lost, discordant, 0.5, alternative="two-sided").pvalue
    return {
        "lost": lost,
        "gained": gained,
        "discordant": discordant,
        "p_value": p,
    }


def bootstrap_delta_ci(base: np.ndarray, other: np.ndarray, alpha: float = 0.05) -> tuple:
    """Paired bootstrap 95% CI for the refusal-rate delta (other - base), in pp.

    We resample PROMPT INDICES (not the two vectors independently) so each
    resample keeps base[i] and other[i] together -- preserving the pairing.
    """
    n = len(base)
    idx = RNG.integers(0, n, size=(N_BOOTSTRAP, n))   # 10k resamples of prompt positions
    deltas = (other[idx].mean(axis=1) - base[idx].mean(axis=1)) * 100
    lo, hi = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi


def bootstrap_rate_ci(vec: np.ndarray, alpha: float = 0.05) -> tuple:
    """Bootstrap 95% CI for a single refusal rate (in %)."""
    n = len(vec)
    idx = RNG.integers(0, n, size=(N_BOOTSTRAP, n))
    rates = vec[idx].mean(axis=1) * 100
    lo, hi = np.percentile(rates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def short(model_id: str) -> str:
    return model_id.split("/")[-1]


def main():
    models = load_models()

    print("=" * 78)
    print("  KEYWORD-SCORED REFUSAL AUDIT (non-thinking mode)")
    print(f"  {N_BOOTSTRAP:,} bootstrap resamples, seed=42, exact McNemar test")
    print("=" * 78)
    print("""
  How to read this:
    delta      = nf4 refusal rate minus fp16 refusal rate (percentage points)
    95% CI     = paired bootstrap interval for that difference
    lost/gained= prompts that flipped refused->complied / complied->refused
    p          = exact McNemar two-sided p-value on those flips
""")

    # ---- Per-model detail: every quant level vs fp16 baseline ----
    summary = []
    all_tests = []   # every (model, quant) McNemar p-value, for the multiple-comparisons check
    for mid in sorted(models, key=short):
        vecs = models[mid]
        if "fp16" not in vecs:
            continue
        base = vecs["fp16"]
        base_rate = base.mean() * 100
        b_lo, b_hi = bootstrap_rate_ci(base)

        print("-" * 78)
        print(f"  {short(mid)}")
        print(f"    fp16 baseline refusal: {base_rate:.0f}%  (95% CI {b_lo:.0f}–{b_hi:.0f}%)")
        print(f"    {'compare':<10} {'rate':>5} {'delta':>7} {'95% CI':>15} "
              f"{'lost':>5} {'gain':>5} {'p':>8}")

        for q in ["int8", "int4_fp4", "nf4_dq"]:
            if q not in vecs:
                continue
            other = vecs[q]
            rate = other.mean() * 100
            delta = rate - base_rate
            lo, hi = bootstrap_delta_ci(base, other)
            mc = mcnemar_exact(base, other)
            all_tests.append({"model": short(mid), "quant": q, "p": mc["p_value"], "delta": delta})
            ci = f"[{lo:+.0f}, {hi:+.0f}]"
            print(f"    {q:<10} {rate:>4.0f}% {delta:>+6.0f}  {ci:>15} "
                  f"{mc['lost']:>5} {mc['gained']:>5} {mc['p_value']:>8.3f}")

            if q == "nf4_dq":
                crosses_zero = lo <= 0 <= hi
                summary.append({
                    "model": short(mid),
                    "fp16": base_rate, "nf4": rate, "delta": delta,
                    "ci_lo": lo, "ci_hi": hi,
                    "lost": mc["lost"], "gained": mc["gained"],
                    "p": mc["p_value"], "crosses_zero": crosses_zero,
                })
        print()

    # ---- FP16-to-NF4 summary across all models ----
    print("=" * 78)
    print("  FP16 -> NF4_DQ SUMMARY")
    print("=" * 78)
    print(f"  {'model':<26} {'fp16':>5} {'nf4':>5} {'delta':>7} {'95% CI':>14} {'p':>8}  interval")
    print("-" * 78)
    for s in sorted(summary, key=lambda x: x["delta"]):
        interval_status = "includes 0" if s["crosses_zero"] else "excludes 0"
        ci = f"[{s['ci_lo']:+.0f},{s['ci_hi']:+.0f}]"
        print(f"  {s['model']:<26} {s['fp16']:>4.0f}% {s['nf4']:>4.0f}% {s['delta']:>+6.0f}  "
              f"{ci:>14} {s['p']:>8.3f}  {interval_status}")

    print()
    excludes_zero = [s for s in summary if not s["crosses_zero"]]
    print(f"  Intervals excluding zero: {len(excludes_zero)}/{len(summary)}")
    if excludes_zero:
        labels = ", ".join(
            "{} ({:+.0f}pp)".format(s["model"], s["delta"])
            for s in excludes_zero
        )
        print(f"  Comparisons: {labels}")

    # ---- Multiple-comparison summary ----
    print()
    print("=" * 78)
    print("  MULTIPLE COMPARISONS CHECK (all quant levels, all models)")
    print("=" * 78)
    n_tests = len(all_tests)
    raw_below_threshold = [t for t in all_tests if t["p"] < 0.05]
    expected_false = 0.05 * n_tests
    bonferroni = 0.05 / n_tests
    survives = [t for t in all_tests if t["p"] < bonferroni]
    print(f"  Total tests run            : {n_tests}  (3 quant levels x 6 models)")
    print(f"  Raw p<0.05                  : {len(raw_below_threshold)}")
    print(f"  Expected if all differences are zero: {expected_false:.1f}")
    print(f"  Bonferroni threshold        : p < {bonferroni:.4f}")
    print(f"  Pass Bonferroni threshold   : {len(survives)}")
    if raw_below_threshold:
        print("  Comparisons with raw p<0.05:")
        for t in sorted(raw_below_threshold, key=lambda x: x["p"]):
            keep = "passes correction" if t["p"] < bonferroni else "does not pass correction"
            print(f"    {t['model']} {t['quant']}: {t['delta']:+.0f}pp, p={t['p']:.3f}  ({keep})")
    print()
    print(f"  Result: {len(raw_below_threshold)} raw p-value(s) below 0.05; ")
    print(f"          {len(survives)} pass the Bonferroni threshold.")
    print("=" * 78)


if __name__ == "__main__":
    main()
