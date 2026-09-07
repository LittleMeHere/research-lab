"""Content baseline for the probe (Gilg App. A.1): predict utility from a generic sentence embedding of the task text.
If a text encoder predicts held-out utility about as well as the residual-stream probe, the probe is mostly reading
task content, not the model's valuation. Same train/eval split and ridge pipeline as probe.py.
Encoder: all-MiniLM-L6-v2 (Gilg's default content encoder; they also report Qwen3-Embedding-8B)."""
import json, os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from scipy.stats import pearsonr
from sentence_transformers import SentenceTransformer
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
pool = json.load(open(os.path.join(ROOT, "runs/pairwise/pool.json")))
ut = pd.read_csv(os.path.join(ROOT, "runs/pairwise/utilities.csv")).set_index("id").loc[[p["id"] for p in pool]]
y = ut.u.values; tr = (ut.split == "train").values; ev = ~tr
enc = SentenceTransformer("all-MiniLM-L6-v2")
X = enc.encode([p["text"] for p in pool], batch_size=64, show_progress_bar=False, normalize_embeddings=True)
alphas = np.logspace(-3, 3, 13)
def fit(Xm, label):
    Z = (Xm - Xm[tr].mean(0)) / (Xm[tr].std(0) + 1e-6)
    m = RidgeCV(alphas=alphas, cv=5).fit(Z[tr], y[tr])
    print(f"{label:38s} held-out r = {pearsonr(m.predict(Z[ev]), y[ev])[0]:.3f}  (alpha {m.alpha_:g})")
fit(X, "MiniLM text embedding (384-d)")
S = pd.get_dummies(ut.origin).values.astype(float)
fit(S, "source one-hot only (5-d)")
fit(np.hstack([X, S]), "embedding + source")
L = np.array([[len(p["text"]), np.log1p(len(p["text"]))] for p in pool]); fit(np.hstack([S, L]), "source + length")
pr = pd.read_csv(os.path.join(ROOT, "runs/probe/probe_r.csv")).set_index("layer")
print(f"{'residual-stream probe (for reference)':38s} held-out r = {pr.loc[23,'r_eval']:.3f} (L23), {pr.r_eval.max():.3f} (best, L{pr.r_eval.idxmax()})")
# within-source: does the probe target still vary meaningfully after removing source means, and can the encoder track it?
resid = y - pd.Series(y).groupby(ut.origin.values).transform("mean").values
print(f"utility SD {y.std():.2f}; within-source residual SD {resid.std():.2f}")
Z = (X - X[tr].mean(0)) / (X[tr].std(0) + 1e-6); m = RidgeCV(alphas=alphas, cv=5).fit(Z[tr], resid[tr])
print(f"{'MiniLM -> within-source residual':38s} held-out r = {pearsonr(m.predict(Z[ev]), resid[ev])[0]:.3f}")
