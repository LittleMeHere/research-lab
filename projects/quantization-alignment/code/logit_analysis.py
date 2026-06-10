#!/usr/bin/env python3
"""
First-token logit analysis across quantization levels.
=======================================================

The experiment captured the top-50 first-token probabilities for every refusal
prompt at every quant level. This is the "decision moment": before generating a
single token, what does the model want to say first, and how sure is it?

Under greedy decoding the top token is almost always ~1.0, so "probability of a
refusal token" is nearly binary and uninformative. A better, label-free signal
is UNCERTAINTY at that first position:

    top1 confidence  = probability mass on the single most likely token
    entropy          = spread of the (top-50) distribution, in nats
                       higher entropy = model is less committed

Hypothesis worth visualizing: does aggressive quantization (fp16 -> nf4) raise
first-token entropy -- i.e. does compression blur the model's initial decision?

Reads existing data/, no GPU. Writes a per-(model, quant) metrics table and a
machine-readable JSON next to the data.

Usage:
    python logit_analysis.py
"""

import glob
import json
import os
import sys
from math import log

sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
QUANT_ORDER = ["fp16", "int8", "int4_fp4", "nf4_dq"]
REFUSAL_KEY = "refusal_thinking=False"


def entropy(probs: list[float]) -> float:
    """Shannon entropy in nats over a (truncated top-k) probability list."""
    return -sum(p * log(p) for p in probs if p > 0)


def metrics_for_level(quant_level: dict) -> dict:
    """Mean top-1 confidence and mean first-token entropy over all prompts."""
    results = quant_level[REFUSAL_KEY]["results"]
    top1s, ents = [], []
    for r in results:
        probs = r.get("logits", {}).get("probs", [])
        if not probs:
            continue
        top1s.append(probs[0])              # already sorted descending
        ents.append(entropy(probs))
    n = len(top1s)
    return {
        "n": n,
        "mean_top1": sum(top1s) / n,
        "mean_entropy": sum(ents) / n,
    }


def short(model_id: str) -> str:
    return model_id.split("/")[-1]


def main():
    out = {}
    rows = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "v2_results_*.json"))):
        d = json.load(open(path, encoding="utf-8"))
        mid = d["model_id"]
        out[mid] = {}
        for q in QUANT_ORDER:
            ql = d.get("quant_levels", {}).get(q)
            if not ql or REFUSAL_KEY not in ql:
                continue
            m = metrics_for_level(ql)
            out[mid][q] = m
            rows.append((short(mid), q, m))

    # ---- Text report ----
    print("=" * 74)
    print("  FIRST-TOKEN UNCERTAINTY ACROSS QUANTIZATION (refusal prompts)")
    print("=" * 74)
    print("  top1 = mean prob on the argmax token (higher = more certain)")
    print("  H    = mean first-token entropy in nats (higher = less certain)")
    print()

    models = sorted({short(m) for m in out})
    for mshort in models:
        print("-" * 74)
        print(f"  {mshort}")
        print(f"    {'quant':<10} {'top1':>8} {'entropy(H)':>12}   {'dH vs fp16':>11}")
        base_h = None
        for short_name, q, m in rows:
            if short_name != mshort:
                continue
            if q == "fp16":
                base_h = m["mean_entropy"]
            dh = "" if base_h is None or q == "fp16" else f"{m['mean_entropy'] - base_h:+.3f}"
            print(f"    {q:<10} {m['mean_top1']:>8.3f} {m['mean_entropy']:>12.3f}   {dh:>11}")
        print()

    # ---- Save machine-readable for plotting ----
    out_path = os.path.join(DATA_DIR, "logit_uncertainty_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("=" * 74)
    print(f"  Saved metrics -> {os.path.relpath(out_path, os.path.dirname(DATA_DIR))}")
    print("  Plot with matplotlib (optional dep):")
    print("    pip install matplotlib && python logit_plot.py")
    print("=" * 74)


if __name__ == "__main__":
    main()
