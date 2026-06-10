#!/usr/bin/env python3
"""
Statistical rigor pass over the quantization-alignment results.
================================================================

The v2 writeup reports point estimates like "SmolLM2 drops -8pp under nf4."
But each number comes from a SINGLE run of 100 prompts. This script asks the
question the writeup couldn't: are those deltas real, or within noise?

It uses two tools, both of which exploit a fact about the experiment that the
naive rate-vs-rate comparison throws away: every quant level was evaluated on
the *exact same 100 prompts*. That pairing is statistical gold.

  1. McNemar's exact test  -- looks only at prompts that FLIPPED between two
     quant levels. If "refused -> complied" flips (safety lost) massively
     outnumber "complied -> refused" flips (safety gained), the change is real.
     If they're roughly balanced, it's noise.

  2. Paired bootstrap CI    -- resamples the 100 prompts with replacement many
     times to put an error bar on each delta. If the 95% interval crosses 0,
     we cannot distinguish the effect from noise.

No GPU, no model loading -- this reads the JSON that already exists in data/.

Usage:
    python stats_analysis.py
"""

import glob
import json
import os
import sys

import numpy as np
from scipy.stats import binomtest

sys.stdout.reconfigure(encoding="utf-8")   # render em-dashes on Windows consoles

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

    b = lost   : refused at base, complied at other  (safety erosion)
    c = gained : complied at base, refused at other   (safety improvement)

    Under the null hypothesis (quantization doesn't change refusal), each
    discordant prompt is equally likely to flip either way, so b ~ Binomial(b+c, 0.5).
    The exact two-sided p-value is how surprising the observed split is.
    """
    lost = int(np.sum((base == 1) & (other == 0)))     # b
    gained = int(np.sum((base == 0) & (other == 1)))   # c
    discordant = lost + gained
    if discordant == 0:
        p = 1.0   # nothing flipped at all -> no evidence of change
    else:
        p = binomtest(lost, discordant, 0.5, alternative="two-sided").pvalue
    return {
        "lost": lost,
        "gained": gained,
        "net": gained - lost,          # +ve = net safety gain, -ve = net loss
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


def stars(p: float) -> str:
    """Significance markers — the usual social-science thresholds."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"   # not significant


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def short(model_id: str) -> str:
    return model_id.split("/")[-1]


def main():
    models = load_models()

    print("=" * 78)
    print("  STATISTICAL RIGOR PASS — refusal under quantization (non-thinking mode)")
    print(f"  {N_BOOTSTRAP:,} bootstrap resamples, seed=42, exact McNemar test")
    print("=" * 78)
    print("""
  How to read this:
    delta      = nf4 refusal rate minus fp16 refusal rate (percentage points)
    95% CI     = bootstrap interval for that delta; if it CROSSES 0, the
                 effect is indistinguishable from noise
    lost/gained= prompts that flipped refused->complied / complied->refused
    p          = exact McNemar two-sided p-value on those flips
    sig        = *** p<.001  ** p<.01  * p<.05  ns = not significant
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
              f"{'lost':>5} {'gain':>5} {'p':>8}  sig")

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
                  f"{mc['lost']:>5} {mc['gained']:>5} {mc['p_value']:>8.3f}  {stars(mc['p_value'])}")

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

    # ---- Headline summary: fp16 -> nf4 across all models ----
    print("=" * 78)
    print("  HEADLINE: fp16 -> nf4_dq (the most aggressive standard quantization)")
    print("=" * 78)
    print(f"  {'model':<26} {'fp16':>5} {'nf4':>5} {'delta':>7} {'95% CI':>14} {'p':>8}  verdict")
    print("-" * 78)
    for s in sorted(summary, key=lambda x: x["delta"]):
        verdict = "NOISE (CI crosses 0)" if s["crosses_zero"] else "REAL effect"
        ci = f"[{s['ci_lo']:+.0f},{s['ci_hi']:+.0f}]"
        print(f"  {s['model']:<26} {s['fp16']:>4.0f}% {s['nf4']:>4.0f}% {s['delta']:>+6.0f}  "
              f"{ci:>14} {s['p']:>8.3f}  {verdict}")

    print()
    real = [s for s in summary if not s["crosses_zero"]]
    print(f"  => {len(real)} of {len(summary)} models show a statistically distinguishable")
    print(f"     fp16->nf4 refusal change. The rest are within single-run noise.")
    if real:
        labels = ", ".join("{} ({:+.0f}pp)".format(s["model"], s["delta"]) for s in real)
        print(f"     Real effects: {labels}")

    # ---- Multiple comparisons: the crucial sanity check ----
    # We ran many tests. Even with NO real effect anywhere, ~5% of tests come up
    # "significant" at p<0.05 by chance. So the right question isn't "did any test
    # pass?" but "did MORE pass than chance alone would produce?"
    print()
    print("=" * 78)
    print("  MULTIPLE COMPARISONS CHECK (all quant levels, all models)")
    print("=" * 78)
    n_tests = len(all_tests)
    sig = [t for t in all_tests if t["p"] < 0.05]
    expected_false = 0.05 * n_tests
    bonferroni = 0.05 / n_tests
    survives = [t for t in all_tests if t["p"] < bonferroni]
    print(f"  Total tests run            : {n_tests}  (3 quant levels x 6 models)")
    print(f"  Significant at p<0.05       : {len(sig)}")
    print(f"  Expected by chance (noise)  : {expected_false:.1f}")
    print(f"  Bonferroni threshold        : p < {bonferroni:.4f}")
    print(f"  Survive Bonferroni          : {len(survives)}")
    if sig:
        print("  The p<0.05 hits:")
        for t in sorted(sig, key=lambda x: x["p"]):
            keep = "survives correction" if t["p"] < bonferroni else "does NOT survive correction"
            print(f"    {t['model']} {t['quant']}: {t['delta']:+.0f}pp, p={t['p']:.3f}  ({keep})")
    print()
    print(f"  Read: {len(sig)} significant hit(s) vs {expected_false:.1f} expected from pure noise.")
    print("  When observed significant results match the false-positive rate, the")
    print("  honest conclusion is: no reliable quantization effect on refusal here.")
    print("=" * 78)


if __name__ == "__main__":
    main()
