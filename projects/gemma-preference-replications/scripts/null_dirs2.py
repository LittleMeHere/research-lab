"""Second batch of control directions and the natural-SD matching (Sep 4 review, P13 and the cosine hint).

  orthK : isotropic random (seeded on name only, as steer.py now does) with the layer's probe direction projected out,
          renormalised. If random swings are partly carried by the probe-aligned component, orth swings shrink.
  natsd : per-direction multiplier so that c * norm_L * d moves the projection X·d by the same number of natural SDs
          as the probe does: scale(d) = std(X·d_probe) / std(X·d), std over the 2,000 train eot activations.
Out: runs/null/dirs.npz (adds orthK_L*), runs/null/natsd_scale.json, printed std table."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, D_MODEL
acts = np.load(os.path.join(ROOT, "runs/extract/acts_eot.f32.npy"), mmap_mode="r")
import pandas as pd
ut = pd.read_csv(os.path.join(ROOT, "runs/pairwise/utilities.csv")); tr = (ut.split == "train").values
probe = np.load(os.path.join(ROOT, "runs/probe/directions.npy"))
dirs = dict(np.load(os.path.join(ROOT, "runs/null/dirs.npz")))
scale = {}
for L in (20, 23):
    X = np.asarray(acts[tr, L, :], dtype=np.float64); Xc = X - X.mean(0)
    fam = {"probe": probe[L]}
    for k in range(4):
        g = np.random.default_rng(k).standard_normal(D_MODEL); g /= np.linalg.norm(g); fam[f"rand{k}"] = g
        o = g - (g @ probe[L]) * probe[L]; o /= np.linalg.norm(o); fam[f"orth{k}"] = o; dirs[f"orth{k}_L{L}"] = o.astype(np.float32)
    for name in ["top0", "top1", "top2", "shuf0", "shuf1", "shuf2", "cov0", "low0"]:
        fam[name] = dirs[f"{name}_L{L}"]
    sd = {n: float((Xc @ d).std()) for n, d in fam.items()}
    print(f"\nL{L}: std of the projection X·d over train eot activations (natural SD along each direction); probe = {sd['probe']:.1f}")
    for n, v in sd.items():
        raw = sd["probe"] / v
        scale[f"{n}_L{L}"] = float(np.clip(raw, 0.05, 4.0))  # clipped: a >4x coefficient would leave the tested regime
        print(f"  {n:6s} std {v:8.1f}   c-multiplier to match the probe's natural-SD excursion: {raw:6.3f} (used {scale[f'{n}_L{L}']:.3f})   cos(probe) {float(fam[n] @ probe[L]):+.3f}")
np.savez(os.path.join(ROOT, "runs/null/dirs.npz"), **dirs)
json.dump(scale, open(os.path.join(ROOT, "runs/null/natsd_scale.json"), "w"), indent=1)
print("saved", len(dirs), "directions;", len(scale), "scales")
