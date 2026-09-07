"""Bradley-Terry utilities from runs/pairwise/results.jsonl.

Outcome per (pair, order) row: y = P(A)/(P(A)+P(B)) from the logit readout (soft label). Model:
P(first beats second) = sigmoid(u_first - u_second + beta), beta = position (Task A slot) bias.
L2 on u (lambda small) fixes the gauge. Rows with A/B mass < --min_mass are dropped and counted.
Out: runs/pairwise/utilities.csv (id, split, origin, u, n_comp), runs/pairwise/bt_fit.json
"""
import argparse, json, os, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, read_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="runs/pairwise")
ap.add_argument("--min_mass", type=float, default=0.5)
ap.add_argument("--l2", type=float, default=0.01)
args = ap.parse_args()
run = os.path.join(ROOT, args.run)
pool = json.load(open(os.path.join(run, "pool.json")))
rows = pd.DataFrame(read_jsonl(os.path.join(run, "results.jsonl")))
n_all = len(rows)
rows = rows[rows.mass >= args.min_mass]
print(f"{n_all} rows, {n_all - len(rows)} dropped for mass < {args.min_mass}; mean mass {rows.mass.mean():.4f}")
pa, pb = np.exp(rows.logp_a), np.exp(rows.logp_b)
rows["y"] = pa / (pa + pb)

# order consistency: the same pair in both orders should give y(order0) ~ 1 - y(order1)
piv = rows.pivot_table(index="pair_id", columns="order", values="y")
both = piv.dropna()
agree = ((both[0] > 0.5) == (both[1] < 0.5)).mean()
print(f"pairs with both orders: {len(both)}; sign agreement across orders {agree:.3f}; "
      f"mean y(order0) {both[0].mean():.3f}, mean 1-y(order1) {(1 - both[1]).mean():.3f}")

idx = {p["id"]: k for k, p in enumerate(pool)}
a = torch.tensor([idx[t] for t in rows.task_a]); b = torch.tensor([idx[t] for t in rows.task_b])
y = torch.tensor(rows.y.values, dtype=torch.float64)
u = torch.zeros(len(pool), dtype=torch.float64, requires_grad=True)
beta = torch.zeros(1, dtype=torch.float64, requires_grad=True)
opt = torch.optim.LBFGS([u, beta], max_iter=500, line_search_fn="strong_wolfe", tolerance_grad=1e-9)


def closure():
    opt.zero_grad()
    logit = u[a] - u[b] + beta
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y) + args.l2 * (u ** 2).mean()
    loss.backward()
    return loss


opt.step(closure)
loss = closure().item()  # LBFGS.step returns the *initial* loss; report the converged one
u = u.detach().numpy(); beta = beta.item()
p_hat = 1 / (1 + np.exp(-(u[a.numpy()] - u[b.numpy()] + beta)))
acc = ((p_hat > 0.5) == (rows.y.values > 0.5)).mean()
n_comp = pd.Series(np.concatenate([a.numpy(), b.numpy()])).value_counts().reindex(range(len(pool)), fill_value=0)
out = pd.DataFrame({"id": [p["id"] for p in pool], "split": [p["split"] for p in pool], "origin": [p["origin"] for p in pool],
                    "u": u, "n_comp": n_comp.values})
out.to_csv(os.path.join(run, "utilities.csv"), index=False)
summary = {"n_rows_used": int(len(rows)), "n_dropped_mass": int(n_all - len(rows)), "loss": loss, "beta_position_bias": beta,
           "train_acc_hard": float(acc), "order_sign_agreement": float(agree), "l2": args.l2,
           "u_by_origin": out.groupby("origin").u.agg(["mean", "std", "count"]).round(3).to_dict("index"),
           "u_std": float(out.u.std())}
json.dump(summary, open(os.path.join(run, "bt_fit.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
print(out.sort_values("u").head(5)[["id", "origin", "u"]].to_string()); print(out.sort_values("u").tail(5)[["id", "origin", "u"]].to_string())
