"""Patching known-positives (Gilg et al. swap hooks and App. K eot-token transplant) on the same held-out pairs
as runs/steer/pairs.json. Single prefill pass per trial, choice read as P(A) vs P(B). Sign test only.

Conditions, each at decoder layer L (hook on the layer output, prefill only):
  swap    : swap residuals across the two task spans (right-aligned, min length) -> should flip choice
  eot     : replace the recipient's <end_of_turn> residual with the donor's, donor = same pair in the other
            order -> should flip choice (App. K: window L25-L35)
  eot_all : the eot transplant at every layer at once (App. K headline: 56.9% flips)
  none    : baseline
Rows -> runs/<run>/results.jsonl (resumable).
"""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import (ROOT, load_model, decoder_layers, pair_prompt, last_logits, ab_readout, token_batches,
                    append_jsonl, read_jsonl, git_hash, N_LAYERS)

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="runs/patch")
ap.add_argument("--pairs", default="runs/steer/pairs.json")
ap.add_argument("--layers", required=True)
ap.add_argument("--conds", default="none,swap,eot,eot_all")
ap.add_argument("--budget", type=int, default=6000)
args = ap.parse_args()
run = os.path.join(ROOT, args.run)
os.makedirs(run, exist_ok=True)
res_path = os.path.join(run, "results.jsonl")
pool = json.load(open(os.path.join(ROOT, "runs/pairwise/pool.json")))
text = {p["id"]: p["text"] for p in pool}
pairs = json.load(open(os.path.join(ROOT, args.pairs)))

tok, model = load_model()
layers = decoder_layers(model)
eot_id = tok.convert_tokens_to_ids("<end_of_turn>")
prompts = {}
for p in pairs:
    for o in (0, 1):
        a, b = (p["i"], p["j"]) if o == 0 else (p["j"], p["i"])
        pp = pair_prompt(tok, text[a], text[b])
        eot = (pp["ids"] == eot_id).nonzero().flatten().tolist()
        assert len(eot) == 1
        prompts[(p["pair_id"], o)] = {"pair_id": p["pair_id"], "order": o, "task_a": a, "task_b": b, "ids": pp["ids"],
                                      "span_a": pp["span_a"], "span_b": pp["span_b"], "eot": eot[0], "n_tokens": len(pp["ids"])}

done = {(r["pair_id"], r["order"], r["cond"], r["layer"]) for r in read_jsonl(res_path)}
state = {"mode": None, "batch": None, "donor": {}}


def make_hook(l):
    def h(mod, inp, out):
        x = out[0] if isinstance(out, tuple) else out
        if state["mode"] == "swap":
            for i, q in enumerate(state["batch"]):
                (a0, a1), (b0, b1) = q["span_a"], q["span_b"]
                n = min(a1 - a0, b1 - b0)
                ta, tb = x[i, a1 - n:a1].clone(), x[i, b1 - n:b1].clone()
                x[i, a1 - n:a1], x[i, b1 - n:b1] = tb, ta
        elif state["mode"] == "capture":
            state["donor"][l] = torch.stack([x[i, q["eot"]].clone() for i, q in enumerate(state["batch"])])
        elif state["mode"] == "paste":
            for i, q in enumerate(state["batch"]):
                x[i, q["eot"]] = state["donor"][l][i]
        return out
    return h


def run_cond(cond, layer, todo):
    """Yield (q, logits row) for prompts in todo under cond at layer (layer=-1 for none/eot_all)."""
    hook_layers = list(range(N_LAYERS)) if cond == "eot_all" else ([layer] if cond != "none" else [])
    handles = [layers[l].register_forward_hook(make_hook(l)) for l in hook_layers]
    try:
        for batch in token_batches(todo, key=lambda q: q["n_tokens"], budget_tokens=args.budget):
            state["batch"] = batch
            if cond in ("eot", "eot_all"):
                donors = [prompts[(q["pair_id"], 1 - q["order"])] for q in batch]
                state["batch"], state["mode"] = donors, "capture"
                last_logits(model, tok, [d["ids"] for d in donors])
                state["batch"], state["mode"] = batch, "paste"
            else:
                state["mode"] = cond if cond == "swap" else None
            logits = last_logits(model, tok, [q["ids"] for q in batch])
            la, lb, mass = ab_readout(logits)
            yield [{"pair_id": q["pair_id"], "order": q["order"], "cond": cond, "layer": layer, "task_a": q["task_a"],
                    "task_b": q["task_b"], "logp_a": round(la[k].item(), 5), "logp_b": round(lb[k].item(), 5),
                    "mass": round(mass[k].item(), 5)} for k, q in enumerate(batch)]
    finally:
        for h in handles:
            h.remove()


t0, n = time.time(), 0
for cond in args.conds.split(","):
    for layer in ([-1] if cond in ("none", "eot_all") else [int(x) for x in args.layers.split(",")]):
        todo = [q for q in prompts.values() if (q["pair_id"], q["order"], cond, layer) not in done]
        for rows in run_cond(cond, layer, todo):
            append_jsonl(res_path, rows)
            n += len(rows)
        if todo:
            pa = np.exp([r["logp_a"] for r in read_jsonl(res_path) if r["cond"] == cond and r["layer"] == layer])
            print(f"{cond} L{layer}: mean P(A) {pa.mean():.3f} n={len(pa)} [{n} rows, {n / (time.time() - t0):.1f}/s]", flush=True)
print("done")
