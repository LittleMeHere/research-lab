"""Figures for the write-up, from the same loader as analyze.py.

  python3 make_figures.py tasks/schedule-main.json [--no-manual] [--allow-drift] [--out figures]

Figure 1: THEN-answer classes per condition (sample 1), stacked horizontal bars, NOW accuracy
          in the label. Figure 2: paired AUDIT-FIRST -> REVEAL-FIRST transitions, 3x3 grid.
Figure 3: controls, THEN recovered per condition. Palette validated with the dataviz
validator (light surface): blue / orange / aqua; aqua carries direct labels for contrast.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analyze

SURFACE = "#fcfcfb"; TEXT = "#0b0b0b"; TEXT2 = "#52514e"; GRID = "#e6e5e1"
COLORS = {"unknown": "#2a78d6", "assimilated": "#eb6834", "other": "#1baf7a", "missing": "#c3c2b7"}
LABELS = {"audit_first": "AUDIT-FIRST", "withheld": "WITHHELD", "reveal_first": "REVEAL-FIRST",
          "audit_first_nodef": "AUDIT-FIRST (no definitions)", "reveal_first_nodef": "REVEAL-FIRST (no definitions)"}
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TEXT,
                     "xtick.color": TEXT2, "ytick.color": TEXT2, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def clean_axes(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(length=0)
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def fig_then_classes(main, conds, path, sample=1):
    rows = []
    for c in conds:
        if not any(c in t["answers"] for t in main):
            continue
        classes = [analyze.classify(t["answers"].get(c, {}).get(sample), t["later"]) for t in main]
        n_obs = sum(cl != "missing" for cl in classes)
        now_ok = sum(1 for t in main if (a := t["answers"].get(c, {}).get(sample)) is not None
                     and a["now_verified_status"] == (t["later"] if c.startswith("reveal_first") else "UNKNOWN"))
        rows.append((c, {k: classes.count(k) for k in COLORS}, n_obs, now_ok))
    fig, ax = plt.subplots(figsize=(7.2, 0.62 * len(rows) + 1.4))
    ys = list(range(len(rows)))[::-1]
    for y, (c, counts, n_obs, now_ok) in zip(ys, rows):
        left = 0
        for k in ("unknown", "assimilated", "other", "missing"):
            v = counts[k]
            if v == 0:
                continue
            ax.barh(y, v, left=left, height=0.42, color=COLORS[k], edgecolor=SURFACE, linewidth=2)
            if v >= 2:
                ax.text(left + v / 2, y, str(v), ha="center", va="center", color="white", fontsize=9)
            left += v
        ax.text(left + 0.25, y, f"NOW as expected {now_ok}/{n_obs}", va="center", color=TEXT2, fontsize=8.5)
    ax.set_yticks(ys)
    ax.set_yticklabels([LABELS[c] for c, *_ in rows], color=TEXT)
    ax.set_xlim(0, max(n for _, _, n, _ in rows) * 1.55)
    ax.set_xlabel("trajectories (sample 1 of each condition)", color=TEXT2)
    clean_axes(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[k]) for k in ("unknown", "assimilated", "other")]
    fig.legend(handles, ["THEN = UNKNOWN (correct)", "THEN = later result (assimilated)", "THEN = the other status"],
               loc="lower center", frameon=False, fontsize=8.5, ncol=3)
    ax.set_title("Retrospective THEN answers by condition", loc="left", color=TEXT, fontsize=11)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_transitions(main, path, base="audit_first", treat="reveal_first", sample=1):
    order = ["unknown", "assimilated", "other"]
    grid = [[0] * 3 for _ in order]
    for t in main:
        a = analyze.classify(t["answers"].get(base, {}).get(sample), t["later"])
        b = analyze.classify(t["answers"].get(treat, {}).get(sample), t["later"])
        if a in order and b in order:
            grid[order.index(a)][order.index(b)] += 1
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    vmax = max(max(r) for r in grid) or 1
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            shade = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab"][min(4, int(4 * v / vmax))] if v else SURFACE
            ax.add_patch(plt.Rectangle((j, 2 - i), 1, 1, facecolor=shade, edgecolor=SURFACE, linewidth=2))
            ax.text(j + 0.5, 2 - i + 0.5, str(v), ha="center", va="center", color="white" if v and v / vmax > 0.5 else TEXT, fontsize=12)
    ax.set_xlim(0, 3); ax.set_ylim(0, 3)
    ax.set_xticks([0.5, 1.5, 2.5]); ax.set_xticklabels(order, color=TEXT)
    ax.set_yticks([2.5, 1.5, 0.5]); ax.set_yticklabels(order, color=TEXT)
    ax.set_xlabel(f"{LABELS[treat]} THEN answer", color=TEXT2); ax.set_ylabel(f"{LABELS[base]} THEN answer", color=TEXT2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Paired transitions per trajectory (sample 1)", loc="left", color=TEXT, fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_controls(controls, conds, path):
    rows = []
    for c in conds:
        ok = tot = 0
        for t in controls:
            for a in t["answers"].get(c, {}).values():
                if a is None:
                    continue
                tot += 1; ok += a["then_verification_status"] == t["then"]
        if tot:
            rows.append((c, ok, tot))
    fig, ax = plt.subplots(figsize=(6.4, 0.55 * len(rows) + 1.2))
    ys = list(range(len(rows)))[::-1]
    for y, (c, ok, tot) in zip(ys, rows):
        ax.barh(y, tot, height=0.42, color="#cde2fb")
        ax.barh(y, ok, height=0.42, color="#2a78d6")
        ax.text(tot + 0.4, y, f"{ok}/{tot}", va="center", color=TEXT2, fontsize=9)
    ax.set_yticks(ys); ax.set_yticklabels([LABELS[c] for c, _, _ in rows], color=TEXT)
    ax.set_xlim(0, max(t for _, _, t in rows) * 1.15)
    ax.set_xlabel("control audits (all samples): THEN recovered", color=TEXT2)
    clean_axes(ax)
    ax.set_title("Audit calibration on PASS/FAIL controls", loc="left", color=TEXT, fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_fail_reading(main, path):
    """Dumbbell per trajectory: FAIL-reading rate in AUDIT-NODEF -> REVEAL-NODEF, split by revealed result."""
    def rate(t, c):
        answers = [a for a in t["answers"].get(c, {}).values() if a is not None]
        return sum(a["then_verification_status"] == "FAIL" for a in answers) / len(answers) if answers else None
    groups = [("FAIL", "#eb6834"), ("PASS", "#2a78d6")]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    for ax, (later, color) in zip(axes, groups):
        sub = [t for t in main if t["later"] == later and rate(t, "audit_first_nodef") is not None]
        ys = list(range(len(sub)))[::-1]
        for y, t in zip(ys, sub):
            b, r = rate(t, "audit_first_nodef"), rate(t, "reveal_first_nodef")
            ax.plot([b, r], [y, y], color=color, linewidth=2, solid_capstyle="round", zorder=1)
            ax.scatter([b], [y], s=64, facecolor=SURFACE, edgecolor=color, linewidth=2, zorder=2)
            ax.scatter([r], [y], s=64, facecolor=color, edgecolor=SURFACE, linewidth=2, zorder=3)
        mb = sum(rate(t, "audit_first_nodef") for t in sub) / len(sub); mr = sum(rate(t, "reveal_first_nodef") for t in sub) / len(sub)
        ax.axvline(mb, color=GRID, linewidth=1); ax.axvline(mr, color=color, linewidth=1, alpha=0.5)
        ax.set_yticks(ys); ax.set_yticklabels([f"run {t['index']:02d}" for t in sub], color=TEXT2, fontsize=8.5)
        ax.set_xlim(-0.08, 1.08); ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_title(f"later result revealed: {later}   mean {mb:.2f} \u2192 {mr:.2f}", loc="left", color=TEXT, fontsize=10)
        clean_axes(ax)
    axes[0].set_xlabel("share of 3 samples answering THEN = FAIL", color=TEXT2); axes[1].set_xlabel("share of 3 samples answering THEN = FAIL", color=TEXT2)
    handles = [plt.Line2D([], [], marker="o", markerfacecolor=SURFACE, markeredgecolor=TEXT2, markeredgewidth=2, linestyle="", markersize=8),
               plt.Line2D([], [], marker="o", markerfacecolor=TEXT2, markeredgecolor=SURFACE, linestyle="", markersize=8)]
    fig.legend(handles, ["no reveal (AUDIT-FIRST, no definitions)", "after reveal (REVEAL-FIRST, no definitions)"], loc="lower center", ncol=2, frameon=False, fontsize=8.5)
    fig.suptitle("Without status definitions, a revealed result shifts the 'setup errors = FAIL' reading", x=0.01, ha="left", color=TEXT, fontsize=11)
    fig.tight_layout(rect=(0, 0.1, 1, 0.94)); fig.savefig(path, dpi=200); plt.close(fig)


def fig_reasoning(main, path):
    """Reasoning tokens per audit by condition (strip + median): definitions turn the audit into a lookup."""
    import json as _json
    conds = list(LABELS)
    data = {c: [] for c in conds}
    for t in main:
        run = Path("runs") / t["run"]
        r = _json.load(open(run / "result.json"))
        for name, v in r["branch_records"].items():
            total = 0  # one token_count per model call; an audit that runs tools has several
            for line in open(run / "branches" / name / "rollout.jsonl"):
                d = _json.loads(line); pl = d.get("payload", {})
                if (pl.get("type") or d.get("type")) == "token_count":
                    total += pl["info"]["last_token_usage"]["reasoning_output_tokens"]
            data[v["condition"]].append(total)
    import statistics
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ys = list(range(len(conds)))[::-1]
    from random import Random
    rng = Random(0)
    for y, c in zip(ys, conds):
        xs = data[c]
        ax.scatter(xs, [y + rng.uniform(-0.16, 0.16) for _ in xs], s=22, color="#2a78d6", alpha=0.45, edgecolor="none")
        med = statistics.median(xs)
        ax.plot([med, med], [y - 0.28, y + 0.28], color="#eb6834", linewidth=2)
        ax.text(med, y + 0.34, f"median {med:.0f}", ha="center", va="bottom", color=TEXT2, fontsize=8)
    ax.set_yticks(ys); ax.set_yticklabels([LABELS[c] for c in conds], color=TEXT)
    ax.set_xscale("symlog", linthresh=50); ax.set_xlim(-2, 2000); ax.set_xticks([0, 25, 50, 100, 250, 500, 1000]); ax.set_xticklabels(["0", "25", "50", "100", "250", "500", "1000"])
    ax.set_xlabel("reasoning tokens per audit, summed over its model calls (symlog scale)", color=TEXT2)
    clean_axes(ax)
    ax.set_title("How much the model deliberates before answering THEN", loc="left", color=TEXT, fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def main_():
    ap = argparse.ArgumentParser()
    ap.add_argument("schedule", nargs="?", default="tasks/schedule-main.json")
    ap.add_argument("--controls", default="tasks/schedule.json")
    ap.add_argument("--no-manual", action="store_true")
    ap.add_argument("--allow-drift", action="store_true")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(exist_ok=True)
    schedule = json.load(open(args.schedule))
    analyze.check_frozen(schedule, strict=not args.allow_drift)
    skipped = []
    main = [r for e in schedule["sequence"] if not e["task"].startswith("c")
            if (r := analyze.load_run(e, schedule, True, not args.no_manual, skipped))]
    cs = json.load(open(args.controls))
    controls = [r for e in cs["sequence"] if e["task"].startswith("c") if (r := analyze.load_run(e, cs, False, False, skipped))]
    if not main:
        print("no main trajectories loaded:", skipped); return
    fig_then_classes(main, analyze.PRIMARY + analyze.EXPLORATORY, out / "fig1_then_classes.png")
    fig_transitions(main, out / "fig2_transitions.png")
    if controls:
        fig_controls(controls, analyze.PRIMARY + analyze.EXPLORATORY, out / "fig3_controls.png")
    if any(analyze.EXPLORATORY[0] in t["answers"] for t in main):
        fig_fail_reading(main, out / "fig4_fail_reading.png")
    fig_reasoning(main, out / "fig5_reasoning_tokens.png")
    print("figures written to", out, "| main n =", len(main), "| controls n =", len(controls))


if __name__ == "__main__":
    main_()
