"""One-page summary figure from the three experiments.

  python3 make_figure_summary.py tasks/schedule-main.json [--out figures]
"""
import argparse
import json
from pathlib import Path
from random import Random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analyze_belief as belief
import analyze_debias as debias

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e6e5e1"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#1b9e77"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
})


def status_counts(schedule_path):
    counts = {condition: 0 for condition in ("audit_first", "withheld", "reveal_first")}
    for entry in json.loads(Path(schedule_path).read_text())["sequence"]:
        result = json.loads((Path(entry["run_dir"]) / "result.json").read_text())
        for branch in result["branch_records"].values():
            condition = branch["condition"]
            if condition in counts and branch["sample"] == 1:
                answer = branch["audit_response"]
                counts[condition] += answer["then_verification_status"] == "UNKNOWN"
    return counts


def differences(rows, treatment, control):
    return [belief.cond_value(row, treatment) - belief.cond_value(row, control) for row in rows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("schedule", nargs="?", default="tasks/schedule-main.json")
    parser.add_argument("--out", default="figures")
    args = parser.parse_args()

    status = status_counts(args.schedule)
    belief_rows = belief.load(args.schedule)
    debias_rows = debias.load(args.schedule)
    series = [
        ("Exp. 2: PASS record", differences(belief_rows, "reveal_pass", "withheld"), BLUE),
        ("Exp. 2: FAIL record", differences(belief_rows, "reveal_fail", "withheld"), ORANGE),
        ("Exp. 3: FAIL record", differences(debias_rows, "reveal_fail", "withheld"), ORANGE),
        ("Exp. 3: FAIL record + instruction",
         differences(debias_rows, "reveal_fail_debias", "withheld_debias"), GREEN),
    ]

    fig, (ax_status, ax_probability) = plt.subplots(
        1, 2, figsize=(8.2, 3.7), gridspec_kw={"width_ratios": [0.9, 1.75]}
    )

    conditions = ["AUDIT-FIRST", "WITHHELD", "REVEAL-FIRST"]
    values = [status[key] for key in status]
    ys = [2, 1, 0]
    ax_status.barh(ys, values, height=0.46, color=BLUE)
    for y, value in zip(ys, values):
        ax_status.text(value / 2, y, f"{value}/16 UNKNOWN", ha="center", va="center",
                       color="white", fontsize=8.5)
    ax_status.set_yticks(ys)
    ax_status.set_yticklabels(conditions, color=TEXT)
    ax_status.set_xlim(0, 16)
    ax_status.set_xticks([0, 8, 16])
    ax_status.set_xlabel("trajectories (sample 1)", color=TEXT2)
    ax_status.set_title("A  Defined status did not move", loc="left", color=TEXT, fontsize=10.5)

    rng = Random(0)
    ys = [3, 2, 1, 0]
    for y, (label, values, color) in zip(ys, series):
        jitter = [rng.uniform(-0.13, 0.13) for _ in values]
        ax_probability.scatter(values, [y + offset for offset in jitter], s=25, color=color,
                               alpha=0.5, edgecolor="none", zorder=2)
        mean = belief.mean(values)
        lo, hi = belief.boot_ci(values)
        ax_probability.plot([lo, hi], [y, y], color=TEXT, linewidth=2.2, zorder=3)
        ax_probability.scatter([mean], [y], marker="D", s=48, color=color,
                               edgecolor=SURFACE, linewidth=1.2, zorder=4)
        ax_probability.text(4.5, y, f"{mean:+.1f}  [{lo:+.1f}, {hi:+.1f}]",
                            va="center", color=TEXT2, fontsize=8)
    ax_probability.axvline(0, color=TEXT2, linewidth=1)
    ax_probability.set_yticks(ys)
    ax_probability.set_yticklabels([label for label, _, _ in series], color=TEXT)
    ax_probability.set_xlim(-18.5, 17.5)
    ax_probability.set_xticks([-15, -10, -5, 0, 5, 10])
    ax_probability.set_xlabel("change in retrospective P(PASS), percentage points", color=TEXT2)
    ax_probability.set_title("B  FAIL records lowered probability", loc="left",
                             color=TEXT, fontsize=10.5)

    for ax in (ax_status, ax_probability):
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)
        ax.grid(axis="x", color=GRID, linewidth=1)
        ax.set_axisbelow(True)

    fig.suptitle("Defined status stayed stable; FAIL records shifted retrospective probability reports",
                 x=0.01, ha="left", color=TEXT, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=2.4)
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    fig.savefig(out / "fig0_summary.png", dpi=200)
    plt.close(fig)
    print("written", out / "fig0_summary.png")


if __name__ == "__main__":
    main()
