"""Per-layer ridge probes: eot activations -> BT utility. Train on canonical-train pool tasks, alpha by 5-fold CV
on train, report held-out Pearson r on canonical-eval tasks. Shuffled-label probes as the null.
One unit-norm direction per layer in raw activation space, sign: + = higher utility.
Out: runs/<run>/directions.npy [62, 5376] float32, runs/<run>/probe_r.csv
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from scipy.stats import pearsonr
sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, N_LAYERS, D_MODEL

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="runs/probe")
ap.add_argument("--acts", default="runs/extract/acts_eot.f32.npy")
ap.add_argument("--utilities", default="runs/pairwise/utilities.csv")
ap.add_argument("--layers", default="all")
args = ap.parse_args()
run = os.path.join(ROOT, args.run)
os.makedirs(run, exist_ok=True)
acts = np.load(os.path.join(ROOT, args.acts), mmap_mode="r")
ut = pd.read_csv(os.path.join(ROOT, args.utilities))
assert len(ut) == acts.shape[0]
tr = (ut.split == "train").values; ev = (ut.split == "eval").values
y = ut.u.values.astype(np.float64)
rng = np.random.default_rng(0)
y_shuf = y.copy(); y_shuf[tr] = rng.permutation(y[tr])
layers = range(N_LAYERS) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
alphas = np.logspace(0, 5, 11)
dirs = np.zeros((N_LAYERS, D_MODEL), dtype=np.float32)
recs = []
for l in layers:
    X = np.asarray(acts[:, l, :], dtype=np.float64)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Z = (X - mu) / sd
    m = RidgeCV(alphas=alphas, cv=5).fit(Z[tr], y[tr])
    r_ev = pearsonr(m.predict(Z[ev]), y[ev])[0]
    r_tr = pearsonr(m.predict(Z[tr]), y[tr])[0]
    ms = RidgeCV(alphas=alphas, cv=5).fit(Z[tr], y_shuf[tr])
    r_shuf = pearsonr(ms.predict(Z[ev]), y[ev])[0]
    w = m.coef_ / sd  # back to raw activation space
    dirs[l] = (w / np.linalg.norm(w)).astype(np.float32)
    # sign check: projection of raw eval activations on the direction must correlate positively with utility
    r_proj = pearsonr(X[ev] @ dirs[l], y[ev])[0]
    recs.append({"layer": l, "r_eval": r_ev, "r_train": r_tr, "r_eval_shuffled": r_shuf, "alpha": m.alpha_, "r_proj_eval": r_proj,
                 "n_train": int(tr.sum()), "n_eval": int(ev.sum())})
    print(f"L{l:2d} r_eval {r_ev:.3f} (train {r_tr:.3f}, shuffled {r_shuf:+.3f}) alpha {m.alpha_:g} r_proj {r_proj:.3f}", flush=True)
np.save(os.path.join(run, "directions.npy"), dirs)
pd.DataFrame(recs).to_csv(os.path.join(run, "probe_r.csv"), index=False)
