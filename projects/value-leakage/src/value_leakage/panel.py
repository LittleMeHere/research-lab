"""Mega panel: one row per model, three columns.

  all rollouts | first estimate ABOVE threshold | first estimate BELOW threshold

Conditioning on where the trace STARTS separates two things the pooled plot
confounds: regression toward the threshold (a trace that opens high tends to
come down whatever the incentive) and genuine incentive-driven drift. Within a
stratum both conditions share a starting side, so above-minus-below is the
drift that survives that control.

  uv run python -m value_leakage.panel
"""

import glob
import json
import os

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from value_leakage.plot import (
    COLORS, INK, INK_MUTED, LABELS, N_GRID, ORDER, drift, resample, valid)

MODES = (("all", "all rollouts"),
         ("above", "first estimate ABOVE threshold"),
         ("below", "first estimate BELOW threshold"))


def subset(trajectories: list, threshold: float, mode: str) -> list:
    kept = [t for t in trajectories if isinstance(t, list) and t]
    if mode == "above":
        return [t for t in kept if t[0] > threshold]
    if mode == "below":
        return [t for t in kept if t[0] < threshold]
    return kept


def band(trajectories: list, threshold: float):
    kept = valid(trajectories, threshold)
    if not kept:
        return None
    stacked = (np.vstack([resample(t) for t in kept]) - threshold) / threshold
    lo, hi = np.percentile(stacked, [25, 75], axis=0)
    return np.median(stacked, axis=0), lo, hi, len(kept)


def main(out: str = "mega_panel.png", dpi: int = 110):
    runs = []
    for f in sorted(glob.glob("runs/*/factor.json")):
        d = os.path.dirname(f) + "/"
        cfg = json.loads(open(d + "config.json").read())
        if cfg.get("count", 0) < 100:
            continue
        runs.append((cfg["model"], d, json.loads(open(f).read())))
    runs.sort(key=lambda r: -r[2]["motivated_reasoning_factor"])

    grid = np.linspace(0, 1, N_GRID)
    fig, axes = plt.subplots(len(runs), 3, figsize=(16.5, 3.0 * len(runs)),
                             squeeze=False)

    for row, (model, d, stats) in enumerate(runs):
        thr = stats["threshold"]
        tr = json.loads(open(d + "trajectories.json").read())
        for col, (mode, label) in enumerate(MODES):
            ax = axes[row][col]
            sub = {c: subset(tr.get(c, []), thr, mode) for c in ORDER}
            for condition in ORDER:
                packed = band(sub[condition], thr)
                if packed is None:
                    continue
                centre, lo, hi, n = packed
                ax.fill_between(grid, lo, hi, color=COLORS[condition],
                                alpha=0.13, linewidth=0)
                ax.plot(grid, centre, color=COLORS[condition], linewidth=1.8,
                        label=f"{LABELS[condition]} (n={n})")

            da = drift(sub["above_good"], thr)
            db = drift(sub["below_good"], thr)
            mrf = None if da is None or db is None else da - db
            ax.axhline(0, color=INK_MUTED, linewidth=0.8, linestyle="--", zorder=0)
            ax.set_xlim(0, 1)
            ax.grid(True, alpha=0.22, linewidth=0.5)
            ax.set_axisbelow(True)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            ax.tick_params(colors=INK_MUTED, labelsize=8)
            ax.set_title(f"{model} — {label}" if col == 0 else label,
                         fontsize=9.5, color=INK if col == 0 else INK_MUTED)
            ax.text(0.02, 0.96,
                    "MRF = n/a" if mrf is None else f"MRF = {mrf:+.3f}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
                    color=INK, bbox=dict(boxstyle="round,pad=0.35",
                                         facecolor="white", edgecolor="#dcdcdc",
                                         linewidth=0.7))
            if col == 0:
                ax.set_ylabel("(est − thr) / thr", fontsize=8.5, color=INK)
                ax.legend(fontsize=7.2, loc="lower right", frameon=False)
            if row == len(runs) - 1:
                ax.set_xlabel("Normalised position in reasoning", fontsize=9,
                              color=INK)

    fig.suptitle("Motivated-reasoning drift, split by where the trace starts",
                 fontsize=13, color=INK, y=0.999)
    fig.tight_layout(rect=(0, 0, 1, 0.995))
    fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"saved {out}  ({len(runs)} models x 3)")


if __name__ == "__main__":
    fire.Fire(main)
