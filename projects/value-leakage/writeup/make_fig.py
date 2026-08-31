"""Main figure for the write-up. Panel A: forced-answer source-condition gap
at E1 and cuts c1-c4 with per-parent points (E2, resolved values) and the
parents' own final-answer gap as reference. Panel B: E3 within-parent
high(300)-minus-low(150) effects by condition vs the full-retention line.
Reads only committed run artifacts. Run from repo root:
  python3 writeup/make_fig.py
"""
import json, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VL = "runs"  # run from the project root: python writeup/make_fig.py
E12 = f"{VL}/fw_e1e2_20260830_152305"
E3 = f"{VL}/fw_e3_20260830_161108"
BELOW, ABOVE = "#1f77b4", "#c85a00"
INK = "#1a1a1a"

s = json.load(open(f"{E12}/summary_v2.json"))
pr = json.load(open(f"{E12}/parent_rows_resolved.json"))
pp = json.load(open(f"{E3}/per_parent_v2.json"))
pr_all = pr

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6), width_ratios=[1.55, 1])
fig.subplots_adjust(wspace=0.28, left=0.07, right=0.98, top=0.86, bottom=0.14)

# ---- Panel A --------------------------------------------------------------
cuts = ["c1", "c2", "c3", "c4"]
xs = {"E1": 0, "c1": 1, "c2": 2, "c3": 3, "c4": 4}
# per-parent gap can't be drawn (gap is cross-condition); draw per-parent medians
# as paired condition strips per cut, and the gap line on a twin axis? Simpler:
# plot the GAP curve with CIs, plus per-parent median_log points per condition
# offset around each cut (secondary, light) on a second y-axis is confusing.
# Choose: main axis = gap (log units); light points = per-parent deviations from
# their condition median (spread), shown as vertical jitter strips.
gaps = [s["e1"]["above_minus_below_median_log_gap"]] + [s["e2_curve"][c]["gap"] for c in cuts]
los = [None] + [s["e2_curve"][c]["ci95"][0] for c in cuts]
his = [None] + [s["e2_curve"][c]["ci95"][1] for c in cuts]
ref = s["parents_own_final_gap_judged"] if "parents_own_final_gap_judged" in s else 0.344
ax.axhline(0, color="#999999", lw=0.8)
ax.axhline(ref, color=INK, lw=1.1, ls="--")
ax.annotate("parents' own final-answer gap\n(0.344 → ×1.41)", xy=(3.6, ref),
            xytext=(2.72, ref + 0.62), fontsize=8.5, color=INK,
            arrowprops=dict(arrowstyle="-", color=INK, lw=0.7))
X = list(xs.values())
ax.errorbar(X[1:], gaps[1:],
            yerr=[[g - l for g, l in zip(gaps[1:], los[1:])],
                  [h - g for g, h in zip(gaps[1:], his[1:])]],
            fmt="o-", color=INK, lw=1.6, capsize=3, ms=5, zorder=5,
            label="E2 forced-answer gap (median over 20+20 parents, 95% CI)")
ax.plot(X[0], gaps[0], "s", color="#777777", ms=7, zorder=5)
ax.annotate("E1: empty reasoning\n(pathological regime,\nmedians in billions)",
            xy=(X[0], gaps[0]), xytext=(X[0] - 0.28, gaps[0] + 0.35), fontsize=8, color="#555555")
# red-flag shading for c1, c2
ax.axvspan(0.6, 2.45, color="#f2b705", alpha=0.10, lw=0)
ax.text(1.52, 4.28, "red-flagged: gap exceeds endpoint gap\n(thin-context dispersion, not transfer)",
        ha="center", va="bottom", fontsize=7.6, color="#8a6d00")
ax.set_xticks(X)
ax.set_xticklabels(["E1\nempty", "c1\nfirst\nsentence", "c2\npopulation\nselected",
                    "c3\nspots value\nfloated", "c4\nfull\nreasoning"], fontsize=8.5)
ax.set_ylabel("above − below forced-answer gap (log units)", fontsize=9)
ax.set_title("A · When does the same-prompt prefix recover the gap?\n"
             "GLM-5.2, giraffe Donation Bet, forced answers under the original prompt", fontsize=9.5, loc="left")
ax.set_ylim(-1.9, 5.05)
ax.set_xlim(-0.55, 4.45)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=7.5, loc="lower right", bbox_to_anchor=(1.0, 0.02), frameon=False)

# per-parent medians (protocol: show parent-level data) on a secondary axis
import math as _m
axr = ax.twinx()
_rng2 = __import__("random").Random(7)
for cut_i, cut in enumerate(cuts, start=1):
    for cond, col, off in (("below_good", BELOW, -0.16), ("above_good", ABOVE, 0.16)):
        ys = [r["median_log"] for r in pr_all if r["cut"] == cut and r["condition"] == cond and r["median_log"]]
        axr.scatter([cut_i + off + _rng2.uniform(-0.05, 0.05) for _ in ys], ys,
                    s=9, color=col, alpha=0.38, lw=0, zorder=2)
axr.set_ylabel("per-parent median forced estimate (log)", fontsize=8, color="#888888")
axr.set_ylim(_m.log(1e6), _m.log(1e11))
axr.set_yticks([_m.log(v) for v in (1e6, 1e7, 1e8, 1e9, 1e10)])
axr.set_yticklabels(["1M", "10M", "100M", "1B", "10B"], fontsize=7, color="#888888")
axr.spines[["top"]].set_visible(False)
axr.tick_params(length=2, color="#bbbbbb")

# ---- Panel B --------------------------------------------------------------
import random, statistics
rng = random.Random(3)
for i, cond in enumerate(["below_good", "above_good"]):
    ds = [r["high_minus_low_log"] for r in pp if r["condition"] == cond and r["high_minus_low_log"] is not None]
    xj = [i + rng.uniform(-0.13, 0.13) for _ in ds]
    ax2.scatter(xj, ds, s=26, color=BELOW if cond == "below_good" else ABOVE, alpha=0.75, zorder=4)
    med = statistics.median(ds)
    ax2.hlines(med, i - 0.24, i + 0.24, color=INK, lw=2.4, zorder=5)
ax2.axhline(0, color="#999999", lw=0.8)
ax2.axhline(math.log(2), color=INK, lw=1.1, ls="--")
ax2.text(0.52, math.log(2) + 0.045, "full retention of the ×2 edit (log 2)", fontsize=8, ha="center", color=INK)
ax2.set_xticks([0, 1])
ax2.set_xticklabels(["below-favoured\n(14 parents)", "above-favoured\n(10 parents)"], fontsize=8.5)
ax2.set_ylabel("E3: final-answer shift, 300 vs 150 edit (log units)", fontsize=9)
ax2.set_title("B · A ×2 factor edit shifts answers ~11–17%, not 100%\n"
              "within-parent high−low; medians +0.111 / +0.155", fontsize=9.5, loc="left")
ax2.set_ylim(-1.0, 1.35)
ax2.spines[["top", "right"]].set_visible(False)

fig.savefig("writeup/fig_main.png", dpi=200)
print("wrote writeup/fig_main.png")
