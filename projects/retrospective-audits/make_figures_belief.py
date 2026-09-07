"""Experiment 2 figure: per-trajectory retrospective probability under WITHHELD vs after each reveal.

  python3 make_figures_belief.py tasks/schedule-main.json [--out figures]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analyze_belief as ab

SURFACE = "#fcfcfb"; TEXT = "#0b0b0b"; TEXT2 = "#52514e"; GRID = "#e6e5e1"
PASS_C = "#2a78d6"; FAIL_C = "#eb6834"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("schedule", nargs="?", default="tasks/schedule-main.json")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    rows = ab.load(args.schedule)
    rows = [t for t in rows if all(ab.cond_value(t, c) is not None for c in ("withheld", "reveal_pass", "reveal_fail"))]
    if not rows:
        print("no complete rows"); return
    fig, ax = plt.subplots(figsize=(7.6, 0.32 * len(rows) + 1.8))
    ys = list(range(len(rows)))[::-1]
    for y, t in zip(ys, rows):
        w, rp, rf = (ab.cond_value(t, c) for c in ("withheld", "reveal_pass", "reveal_fail"))
        ax.plot([w, rp], [y, y], color=PASS_C, linewidth=2, zorder=1)
        ax.plot([rf, w], [y, y], color=FAIL_C, linewidth=2, zorder=1)
        ax.scatter([w], [y], s=60, facecolor=SURFACE, edgecolor=TEXT2, linewidth=2, zorder=3)
        ax.scatter([rp], [y], s=60, facecolor=PASS_C, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.scatter([rf], [y], s=60, facecolor=FAIL_C, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.set_yticks(ys); ax.set_yticklabels([f"run {t['index']:02d} {t['task'].split('-')[0]}" for t in rows], color=TEXT2, fontsize=8.5)
    ax.set_xlim(-3, 103); ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("retrospective P(suite passes), mean of 3 samples", color=TEXT2)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0); ax.grid(axis="x", color=GRID, linewidth=1); ax.set_axisbelow(True)
    handles = [plt.Line2D([], [], marker="o", markerfacecolor=SURFACE, markeredgecolor=TEXT2, markeredgewidth=2, linestyle="", markersize=8),
               plt.Line2D([], [], marker="o", markerfacecolor=PASS_C, markeredgecolor=SURFACE, linestyle="", markersize=8),
               plt.Line2D([], [], marker="o", markerfacecolor=FAIL_C, markeredgecolor=SURFACE, linestyle="", markersize=8)]
    fig.legend(handles, ["record withheld", "after PASS record", "after FAIL record"], loc="lower center", ncol=3, frameon=False, fontsize=8.5)
    spread = [ab.cond_value(t, "reveal_pass") - ab.cond_value(t, "reveal_fail") for t in rows]
    ax.set_title(f"What probability would you have assigned? (mean spread PASS - FAIL = {sum(spread)/len(spread):+.1f} points)", loc="left", color=TEXT, fontsize=10.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out = Path(args.out); out.mkdir(exist_ok=True)
    fig.savefig(out / "fig6_belief.png", dpi=200); plt.close(fig)
    print("written", out / "fig6_belief.png", "n =", len(rows))


if __name__ == "__main__":
    main()
