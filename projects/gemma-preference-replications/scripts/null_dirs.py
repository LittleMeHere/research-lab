"""Control directions for the random-direction question (pre-registered, Sep 3).

Observed: isotropic unit-Gaussian directions, norm-matched to the probe, move explicit choice by mean |E| 0.30 at
c = 0.05 (0.09 at 0.02) vs probe 0.97 (0.79). Hypotheses and predictions, written before running:
  H-shape : isotropic vectors are off-manifold (energy in low-variance dims) and therefore loud.
            -> covariance-matched random directions much weaker than isotropic; top-PC random weaker too.
  H-broad : choice is sensitive to a broad (~60-80 dim) subspace where the data lives.
            -> top-PC random directions STRONGER than isotropic; covariance-matched similar or stronger;
               low-PC random near zero.
  H-probe-shape : the real probe's 0.97 is partly "any probe-shaped vector".
            -> shuffled-label probe directions (3 seeds) give |E| well above isotropic random.
Decision rule: whichever null is largest becomes the reported null; the study's c is chosen where the probe beats
that null by >= 5x. Nothing about the probe direction or c is tuned toward the paper.

Builds, at layers L in --layers, from runs/extract/acts_eot.f32.npy (2,993 tasks) and runs/pairwise/utilities.csv:
  shufK  : ridge probe on utilities permuted within train (seed K), same pipeline as probe.py
  covK   : Gaussian in PC coordinates scaled by per-PC std (covariance-matched), unit norm
  topK   : Gaussian restricted to the top-50 PCs, unit norm
  lowK   : Gaussian restricted to PCs 500-1499, unit norm
All unit-norm in raw space; steer.py scales by c * norm_L exactly as for the probe.
Also prints the offline diagnostics for the 8 isotropic random directions already run (cosine to probe, PC-energy).
Out: runs/null/dirs.npz  (keys "<name>_L<layer>")
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from scipy.stats import pearsonr
sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, D_MODEL

ap = argparse.ArgumentParser()
ap.add_argument("--layers", default="20,23")
ap.add_argument("--n_pcs", type=int, default=1500)
args = ap.parse_args()
os.makedirs(os.path.join(ROOT, "runs/null"), exist_ok=True)
acts = np.load(os.path.join(ROOT, "runs/extract/acts_eot.f32.npy"), mmap_mode="r")
ut = pd.read_csv(os.path.join(ROOT, "runs/pairwise/utilities.csv"))
tr, ev = (ut.split == "train").values, (ut.split == "eval").values
y = ut.u.values.astype(np.float64)
dirs_probe = np.load(os.path.join(ROOT, "runs/probe/directions.npy"))
summ = pd.read_csv(os.path.join(ROOT, "runs/steer/summary.csv"))
out = {}
alphas = np.logspace(0, 5, 11)
for L in [int(x) for x in args.layers.split(",")]:
    X = np.asarray(acts[:, L, :], dtype=np.float64)
    mu = X[tr].mean(0)
    Xc = X - mu
    # PCA on train activations
    U, S, Vt = np.linalg.svd(Xc[tr], full_matrices=False)
    sd_pc = S / np.sqrt(tr.sum() - 1)
    pcs = Vt  # [n_train, d] rows = PC directions
    d_probe = dirs_probe[L]
    print(f"\n=== L{L}: top-PC variance share: top10 {np.sum(sd_pc[:10]**2) / np.sum(sd_pc**2):.3f}, top50 {np.sum(sd_pc[:50]**2) / np.sum(sd_pc**2):.3f}, "
          f"top500 {np.sum(sd_pc[:500]**2) / np.sum(sd_pc**2):.3f}; per-dim std: max {Xc[tr].std(0).max():.0f}, median {np.median(Xc[tr].std(0)):.0f}")

    def energy(v):
        c = pcs @ v
        return {"top10": float(np.sum(c[:10]**2)), "top50": float(np.sum(c[:50]**2)), "top500": float(np.sum(c[:500]**2)),
                "maha": float(np.sqrt(np.sum((c / np.maximum(sd_pc, 1e-6))**2)))}
    print(f"probe L{L}: energy {energy(d_probe)}")
    # diagnostics for the 8 isotropic random directions already run
    rows = []
    for k in range(8):
        g = np.random.default_rng(k * 1000 + L).standard_normal(D_MODEL); g /= np.linalg.norm(g)
        e = energy(g)
        sw = summ[(summ.layer == L) & (summ.dir == f"rand{k}") & (summ.c == 0.05)].swing
        rows.append({"dir": f"rand{k}", "cos_probe": float(g @ d_probe), **e, "swing_c05": float(sw.iloc[0]) if len(sw) else np.nan})
    df = pd.DataFrame(rows); print(df.round(4).to_string(index=False))
    if df.swing_c05.notna().sum() >= 6:
        for col in ["cos_probe", "top50", "maha"]:
            print(f"  corr(swing, {col}) = {pearsonr(df.swing_c05, df[col])[0]:+.3f};  corr(|swing|, {col}) = {pearsonr(df.swing_c05.abs(), df[col])[0]:+.3f}")
    sdd = X[tr].std(0) + 1e-6
    Z = (X - mu) / sdd
    for k in range(3):
        rng = np.random.default_rng(500 + k)
        ys = y.copy(); ys[tr] = rng.permutation(y[tr])
        m = RidgeCV(alphas=alphas, cv=5).fit(Z[tr], ys[tr])
        w = m.coef_ / sdd; w /= np.linalg.norm(w)
        r_real = pearsonr(X[ev] @ w, y[ev])[0]
        if r_real < 0:  # orient like the real probe so the sign of E is comparable; the magnitude is what matters
            w = -w; r_real = -r_real
        out[f"shuf{k}_L{L}"] = w.astype(np.float32)
        print(f"shuf{k} L{L}: r_eval on true utilities {r_real:+.3f}, cos(probe) {float(w @ d_probe):+.3f}, energy {energy(w)}")
        # covariance-matched: coefficients ~ N(0, sd_pc^2) in PC space (rank-limited to the train PCs)
        g = rng.standard_normal(len(sd_pc)) * sd_pc
        v = g @ pcs; v /= np.linalg.norm(v)
        out[f"cov{k}_L{L}"] = v.astype(np.float32)
        print(f"cov{k}  L{L}: cos(probe) {float(v @ d_probe):+.3f}, energy {energy(v)}")
        g = rng.standard_normal(50); v = g @ pcs[:50]; v /= np.linalg.norm(v)
        out[f"top{k}_L{L}"] = v.astype(np.float32)
        print(f"top{k}  L{L}: cos(probe) {float(v @ d_probe):+.3f}, energy {energy(v)}")
        g = rng.standard_normal(1000); v = g @ pcs[500:1500]; v /= np.linalg.norm(v)
        out[f"low{k}_L{L}"] = v.astype(np.float32)
        print(f"low{k}  L{L}: cos(probe) {float(v @ d_probe):+.3f}, energy {energy(v)}")
np.savez(os.path.join(ROOT, "runs/null/dirs.npz"), **out)
print("saved", len(out), "directions")
