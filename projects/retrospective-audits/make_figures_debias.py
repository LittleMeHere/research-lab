"""Experiment 3 figure: per-trajectory FAIL-record effects with and without instruction.

  python3 make_figures_debias.py tasks/schedule-main.json [--out figures]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analyze_belief as ab
import analyze_debias as ad

SURFACE = "#fcfcfb"; TEXT = "#0b0b0b"; TEXT2 = "#52514e"; GRID = "#e6e5e1"
UNTREATED = "#eb6834"; INSTRUCTED = "#2a78d6"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("schedule", nargs="?", default="tasks/schedule-main.json")
    parser.add_argument("--out", default="figures")
    args = parser.parse_args()
    rows = ad.load(args.schedule)
    rows = [row for row in rows if all(ab.cond_value(row, condition) is not None for condition in ad.CONDS)]
    if not rows:
        print("no complete rows")
        return

    untreated = [ab.cond_value(row, "reveal_fail") - ab.cond_value(row, "withheld") for row in rows]
    instructed = [ab.cond_value(row, "reveal_fail_debias") - ab.cond_value(row, "withheld_debias") for row in rows]
    ys = list(range(len(rows)))[::-1]
    fig, ax = plt.subplots(figsize=(7.6, 0.32 * len(rows) + 2.0))
    for y, before, after in zip(ys, untreated, instructed):
        ax.plot([before, after], [y, y], color="#b9b8b3", linewidth=2, zorder=1)
        ax.scatter(before, y, s=62, facecolor=UNTREATED, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.scatter(after, y, s=62, facecolor=INSTRUCTED, edgecolor=SURFACE, linewidth=2, zorder=3)

    mean_untreated = sum(untreated) / len(untreated)
    mean_instructed = sum(instructed) / len(instructed)
    ax.axvline(0, color=TEXT2, linewidth=1)
    ax.axvline(mean_untreated, color=UNTREATED, linewidth=1.5, alpha=0.65)
    ax.axvline(mean_instructed, color=INSTRUCTED, linewidth=1.5, alpha=0.65)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"run {row['index']:02d} {row['task'].split('-')[0]}" for row in rows],
                       color=TEXT2, fontsize=8.5)
    ax.set_xlim(-18.5, 2.5)
    ax.set_xticks([-15, -10, -5, 0])
    ax.set_xlabel("FAIL record - WITHHELD (probability points; mean of 3 samples)", color=TEXT2)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    handles = [
        plt.Line2D([], [], marker="o", markerfacecolor=UNTREATED, markeredgecolor=SURFACE,
                   linestyle="", markersize=8),
        plt.Line2D([], [], marker="o", markerfacecolor=INSTRUCTED, markeredgecolor=SURFACE,
                   linestyle="", markersize=8),
    ]
    fig.legend(handles, [f"no instruction (mean {mean_untreated:+.1f})",
                         f"explicit instruction (mean {mean_instructed:+.1f})"],
               loc="lower center", ncol=2, frameon=False, fontsize=8.5)
    ax.set_title("Instruction attenuates, but does not remove, the FAIL-record effect",
                 loc="left", color=TEXT, fontsize=10.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    fig.savefig(out / "fig7_debias.png", dpi=200)
    plt.close(fig)
    print("written", out / "fig7_debias.png", "n =", len(rows))


if __name__ == "__main__":
    main()
