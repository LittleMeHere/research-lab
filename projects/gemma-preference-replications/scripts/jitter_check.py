"""How much does batch composition (padded length, batch-mates) move the analysed quantity, with no intervention?
Runs the 240 held-out pairwise prompts three ways — alone (batch 1), length-sorted batches (as steer.py does), and
randomly shuffled batches — and reports the distribution of |Δ margin| where margin = logP(A) − logP(B).
This bounds the numerical jitter that would enter any between-condition difference whose prompts land in different
batches (review P9). Out: runs/steer/jitter.json"""
import json, os, random, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, load_model, pair_prompt, last_logits, ab_readout, token_batches
pool = {p["id"]: p["text"] for p in json.load(open(os.path.join(ROOT, "runs/pairwise/pool.json")))}
pairs = json.load(open(os.path.join(ROOT, "runs/steer/pairs.json")))
tok, model = load_model()
items = []
for p in pairs:
    for o in (0, 1):
        a, b = (p["i"], p["j"]) if o == 0 else (p["j"], p["i"])
        pp = pair_prompt(tok, pool[a], pool[b]); items.append({"key": (p["pair_id"], o), "ids": pp["ids"], "n": len(pp["ids"])})
def run(batches):
    out = {}
    for batch in batches:
        la, lb, mass = ab_readout(last_logits(model, tok, [q["ids"] for q in batch]))
        for k, q in enumerate(batch):
            out[q["key"]] = (la[k].item() - lb[k].item(), mass[k].item())
    return out
alone = run([[q] for q in items])
sorted_b = run(list(token_batches(items, key=lambda q: q["n"], budget_tokens=6000)))
sh = items[:]; random.Random(0).shuffle(sh)
shuffled = run([sh[i:i + 12] for i in range(0, len(sh), 12)])
res = {}
for name, other in (("sorted_vs_alone", sorted_b), ("shuffled_vs_alone", shuffled), ("shuffled_vs_sorted", None)):
    ref = alone if other is not None else sorted_b; oth = other if other is not None else shuffled
    d = np.array([abs(oth[k][0] - ref[k][0]) for k in alone]); m = np.array([alone[k][1] for k in alone])
    res[name] = {"median": float(np.median(d)), "p90": float(np.percentile(d, 90)), "max": float(d.max()),
                 "median_mass>0.9": float(np.median(d[m > 0.9])), "max_mass>0.9": float(d[m > 0.9].max()), "n_mass>0.9": int((m > 0.9).sum())}
    print(name, {k: round(v, 4) for k, v in res[name].items()})
json.dump(res, open(os.path.join(ROOT, "runs/steer/jitter.json"), "w"), indent=1)
