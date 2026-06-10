#!/usr/bin/env python3
"""
Plot first-token entropy vs quantization level (one line per model).

Reads data/logit_uncertainty_summary.json (produced by logit_analysis.py) and
saves a PNG into notes/. Kept separate from logit_analysis.py so the analysis
has no plotting dependency — install matplotlib only if you want the figure.

Usage:
    pip install matplotlib
    python logit_plot.py
"""

import json
import os

import matplotlib.pyplot as plt

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(CODE_DIR)
SUMMARY = os.path.join(PROJECT, "data", "logit_uncertainty_summary.json")
OUT = os.path.join(PROJECT, "notes", "logit_entropy.png")

QUANT_ORDER = ["fp16", "int8", "int4_fp4", "nf4_dq"]

# family color + generation linestyle so the encoding carries meaning
STYLE = {
    "microsoft/Phi-4-mini-instruct":        ("#185FA5", "-",  "Phi-4-mini"),
    "Qwen/Qwen3.5-4B":                      ("#0F6E56", "-",  "Qwen3.5-4B"),
    "Qwen/Qwen3-1.7B":                      ("#0F6E56", "--", "Qwen3-1.7B"),
    "google/gemma-4-e2b-it":                ("#534AB7", "-",  "gemma-4-e2b"),
    "HuggingFaceTB/SmolLM3-3B":             ("#D85A30", "-",  "SmolLM3-3B"),
    "HuggingFaceTB/SmolLM2-1.7B-Instruct":  ("#D85A30", "--", "SmolLM2-1.7B"),
}


def main():
    data = json.load(open(SUMMARY, encoding="utf-8"))

    fig, ax = plt.subplots(figsize=(8, 5))
    for mid, (color, ls, label) in STYLE.items():
        levels = data.get(mid, {})
        ys = [levels.get(q, {}).get("mean_entropy") for q in QUANT_ORDER]
        ax.plot(QUANT_ORDER, ys, color=color, linestyle=ls, marker="o",
                linewidth=2, label=label)

    ax.set_xlabel("quantization level (more aggressive →)")
    ax.set_ylabel("mean first-token entropy (nats)")
    ax.set_title("first-token uncertainty across quantization")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"saved -> {os.path.relpath(OUT, PROJECT)}")


if __name__ == "__main__":
    main()
