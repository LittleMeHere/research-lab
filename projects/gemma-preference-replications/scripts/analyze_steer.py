"""Summaries for runs/steer (swing by layer) and runs/patch (flip rates). Bootstrap CIs over pairs.
Out: runs/steer/summary.csv, runs/patch/summary.csv, printed tables."""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, read_jsonl


def boot_ci(pair_ids, values, fn, B=1000, seed=0):
    df = pd.DataFrame({"p": pair_ids, "v": values})
    per = df.groupby("p").v.mean()  # one number per pair (both orders averaged)
    rng = np.random.default_rng(seed)
    stats = [fn(per.values[rng.integers(0, len(per), len(per))]) for _ in range(B)]
    return fn(per.values), np.percentile(stats, 2.5), np.percentile(stats, 97.5)


def steer():
    rows = pd.DataFrame(read_jsonl(os.path.join(ROOT, "runs/steer/results.jsonl")))
    if rows.empty:
        return
    rows["pa"] = np.exp(rows.logp_a) / (np.exp(rows.logp_a) + np.exp(rows.logp_b))
    if "mode" not in rows: rows["mode"] = "contrastive"
    rows["mode"] = rows["mode"].fillna("contrastive")
    out = []
    for (layer, d, mode), g in rows.groupby(["layer", "dir", "mode"]):
        for c in sorted(g.c.unique()):
            if c <= 0:
                continue
            gp, gm = g[g.c == c], g[g.c == -c]
            if gp.empty or gm.empty:
                continue
            m = gp.merge(gm, on=["pair_id", "order"], suffixes=("_p", "_m"))
            swing, lo, hi = boot_ci(m.pair_id, m.pa_p - m.pa_m, np.mean)
            base = g[g.c == 0]
            out.append({"layer": layer, "dir": d, "mode": mode, "c": c, "n_pairs": m.pair_id.nunique(), "P_A_plus": m.pa_p.mean(),
                        "P_A_minus": m.pa_m.mean(), "P_A_zero": base.pa.mean() if len(base) else np.nan,
                        "swing": swing, "ci_lo": lo, "ci_hi": hi, "mass_plus": gp.mass.mean(), "mass_minus": gm.mass.mean()})
    s = pd.DataFrame(out).sort_values(["mode", "dir", "c", "layer"])
    s.to_csv(os.path.join(ROOT, "runs/steer/summary.csv"), index=False)
    print("=== steering swing E_L = P(A|+c) - P(A|-c), both orders, bootstrap 95% CI over pairs")
    print(s.round(3).to_string(index=False))


def patch():
    p = os.path.join(ROOT, "runs/patch/results.jsonl")
    rows = pd.DataFrame(read_jsonl(p)) if os.path.exists(p) else pd.DataFrame()
    if rows.empty:
        return
    rows["pa"] = np.exp(rows.logp_a) / (np.exp(rows.logp_a) + np.exp(rows.logp_b))
    base = rows[rows.cond == "none"].set_index(["pair_id", "order"]).pa
    assert base.index.is_unique, "duplicate baseline rows"
    out = []
    for (cond, layer), g in rows[rows.cond != "none"].groupby(["cond", "layer"]):
        g = g.set_index(["pair_id", "order"])
        b = base.reindex(g.index)
        dec = (b - 0.5).abs() > 0.2  # baseline-decisive trials only
        flip = ((g.pa > 0.5) != (b > 0.5))[dec]
        fr, lo, hi = boot_ci(g.index.get_level_values(0)[dec], flip.values.astype(float), np.mean)
        out.append({"cond": cond, "layer": layer, "n_decisive": int(dec.sum()), "flip_rate": fr, "ci_lo": lo, "ci_hi": hi,
                    "mean_dPA_toward_flip": float(((g.pa - b) * np.sign(0.5 - b))[dec].mean()), "mass": g.mass.mean()})
    s = pd.DataFrame(out).sort_values(["cond", "layer"])
    s.to_csv(os.path.join(ROOT, "runs/patch/summary.csv"), index=False)
    print("=== patching flip rate on baseline-decisive trials (|P(A)-0.5| > 0.2), bootstrap 95% CI over pairs")
    print(s.round(3).to_string(index=False))


steer(); patch()
