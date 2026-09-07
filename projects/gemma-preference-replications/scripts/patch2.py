"""eot-transplant follow-up (review deviation e / REPLICATION item 2): two hypotheses for the 9%-vs-majority gap.
  --format completion : Gilg's completion template; readout = forced prefix "Task" then P(" A") vs P(" B") — the
                        model's stated choice in their format, in one pass (their judge read the same thing).
  --format letter     : our original one-letter prompt (P("A") vs P("B")).
Conditions: none; eot (patch the <end_of_turn> residual from the other-order donor); eot_nl (eot + the following
newline, as in their footnote); eot_all / eot_nl_all (every layer at once).
Rows -> runs/patch_<format>/results.jsonl (resumable)."""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import (ROOT, load_model, decoder_layers, pair_prompt, last_logits, token_batches, append_jsonl, read_jsonl, N_LAYERS, A_ID, B_ID)
ap = argparse.ArgumentParser()
ap.add_argument("--format", default="completion", choices=["completion", "letter"])
ap.add_argument("--layers", default="20,23,26,29,32,35")
ap.add_argument("--conds", default="none,eot,eot_nl,eot_all,eot_nl_all")  # also: nl (newline token only), nl_all
ap.add_argument("--budget", type=int, default=6000)
args = ap.parse_args()
run = os.path.join(ROOT, f"runs/patch_{args.format}"); os.makedirs(run, exist_ok=True)
res_path = os.path.join(run, "results.jsonl")
pool = {p["id"]: p["text"] for p in json.load(open(os.path.join(ROOT, "runs/pairwise/pool.json")))}
pairs = json.load(open(os.path.join(ROOT, "runs/steer/pairs.json")))
tok, model = load_model(); layers = decoder_layers(model)
eot_id = tok.convert_tokens_to_ids("<end_of_turn>")
if args.format == "completion":
    HEAD = "You will be given two tasks. Choose one and complete it.\n\nBegin with 'Task A:' or 'Task B:' to indicate your choice, then complete that task.\n\n\nTask A:\n\n"; MID = "\n\n\nTask B:\n\n"
    PREFIX = torch.tensor(tok.encode("Task", add_special_tokens=False)); IDA, IDB = 562, 603  # " A", " B"
else:
    HEAD = MID = None; PREFIX = torch.tensor([], dtype=torch.long); IDA, IDB = A_ID, B_ID
prompts = {}
for p in pairs:
    for o in (0, 1):
        a, b = (p["i"], p["j"]) if o == 0 else (p["j"], p["i"])
        pp = pair_prompt(tok, pool[a], pool[b]) if HEAD is None else pair_prompt(tok, pool[a], pool[b], head=HEAD, mid=MID)
        eot = (pp["ids"] == eot_id).nonzero().flatten().tolist(); assert len(eot) == 1
        assert tok.convert_ids_to_tokens(int(pp["ids"][eot[0] + 1])) == "\n"
        prompts[(p["pair_id"], o)] = {"pair_id": p["pair_id"], "order": o, "task_a": a, "task_b": b, "ids": torch.cat([pp["ids"], PREFIX]), "eot": eot[0], "n_tokens": len(pp["ids"]) + len(PREFIX)}
done = {(r["pair_id"], r["order"], r["cond"], r["layer"]) for r in read_jsonl(res_path)}
state = {"mode": None, "batch": None, "donor": {}, "width": 1, "offset": 0}
def make_hook(l):
    def h(mod, inp, out):
        x = out[0] if isinstance(out, tuple) else out
        w, off = state["width"], state["offset"]
        if state["mode"] == "capture":
            state["donor"][l] = torch.stack([x[i, q["eot"] + off:q["eot"] + off + w].clone() for i, q in enumerate(state["batch"])])
        elif state["mode"] == "paste":
            for i, q in enumerate(state["batch"]):
                x[i, q["eot"] + off:q["eot"] + off + w] = state["donor"][l][i]
        return out
    return h
def readout(logits):
    lp = torch.log_softmax(logits, -1); return lp[:, IDA], lp[:, IDB], lp[:, IDA].exp() + lp[:, IDB].exp()
def run_cond(cond, layer, todo):
    # eot: [eot]; eot_nl: [eot, \n]; nl: [\n] only (the newline-only control)
    width = 2 if cond.startswith("eot_nl") else 1
    offset = 1 if cond.startswith("nl") else 0
    hook_layers = list(range(N_LAYERS)) if cond.endswith("all") else ([layer] if cond != "none" else [])
    handles = [layers[l].register_forward_hook(make_hook(l)) for l in hook_layers]
    try:
        for batch in token_batches(todo, key=lambda q: q["n_tokens"], budget_tokens=args.budget):
            state["width"], state["offset"] = width, offset
            if cond != "none":
                donors = [prompts[(q["pair_id"], 1 - q["order"])] for q in batch]
                state["batch"], state["mode"] = donors, "capture"; last_logits(model, tok, [d["ids"] for d in donors])
                state["batch"], state["mode"] = batch, "paste"
            else:
                state["mode"] = None
            la, lb, mass = readout(last_logits(model, tok, [q["ids"] for q in batch]))
            yield [{"pair_id": q["pair_id"], "order": q["order"], "cond": cond, "layer": layer, "format": args.format, "task_a": q["task_a"], "task_b": q["task_b"],
                    "logp_a": round(la[k].item(), 5), "logp_b": round(lb[k].item(), 5), "mass": round(mass[k].item(), 5)} for k, q in enumerate(batch)]
    finally:
        for h in handles: h.remove()
t0, n = time.time(), 0
for cond in args.conds.split(","):
    for layer in ([-1] if cond in ("none",) or cond.endswith("all") else [int(x) for x in args.layers.split(",")]):
        todo = [q for q in prompts.values() if (q["pair_id"], q["order"], cond, layer) not in done]
        for rows in run_cond(cond, layer, todo):
            append_jsonl(res_path, rows); n += len(rows)
        if todo:
            rr = [r for r in read_jsonl(res_path) if r["cond"] == cond and r["layer"] == layer]
            pa = np.array([np.exp(r["logp_a"]) / (np.exp(r["logp_a"]) + np.exp(r["logp_b"])) for r in rr]); ms = np.mean([r["mass"] for r in rr])
            print(f"{cond} L{layer}: mean P(A|A or B) {pa.mean():.3f}, mass {ms:.3f}, n={len(pa)} [{n} rows, {n / (time.time() - t0):.1f}/s]", flush=True)
print("done")
