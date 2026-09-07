"""Pairwise task-preference elicitation on Gemma-3-27B-it NF4 (Stage B instrument, charter §5).

Pool: build_pool() (2000 canonical-train + ~990 canonical-eval Gilg tasks). Design: R rounds; each round is a random
perfect matching of the pool, so every task gets exactly one new comparison per round and the first r rounds of a
partial run are a valid degree-r design. Every pair is asked in both orders (order 0: i shown as Task A).
One forward pass per (pair, order); choice is read as P("A") vs P("B") at the decision position.

Raw rows -> runs/<run>/results.jsonl (append-only, resumable). Inputs -> runs/<run>/pool.json, pairs.json, run.json.
"""
import argparse, json, os, random, sys, time
import torch
sys.path.insert(0, os.path.dirname(__file__))
from common import (ROOT, build_pool, load_model, pair_prompt, last_logits, ab_readout, token_batches,
                    append_jsonl, read_jsonl, git_hash, A_ID, B_ID)

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="runs/pairwise")
ap.add_argument("--rounds", type=int, default=8)
ap.add_argument("--limit", type=int, default=0, help="stop after this many new rows (smoke test)")
ap.add_argument("--budget", type=int, default=6000, help="tokens per batch")
ap.add_argument("--resume", action="store_true")
args = ap.parse_args()

run = os.path.join(ROOT, args.run)
os.makedirs(run, exist_ok=True)
pool_path, pairs_path, res_path = (os.path.join(run, f) for f in ("pool.json", "pairs.json", "results.jsonl"))

if os.path.exists(pool_path):
    pool = json.load(open(pool_path))
else:
    pool = build_pool()
    json.dump(pool, open(pool_path, "w"), indent=0)
text = {p["id"]: p["text"] for p in pool}
ids = [p["id"] for p in pool]

if os.path.exists(pairs_path):
    pairs = json.load(open(pairs_path))
else:
    pairs, seen = [], set()
    for r in range(args.rounds):
        order = ids[:]
        random.Random(1000 + r).shuffle(order)
        for k in range(0, len(order) - 1, 2):
            i, j = order[k], order[k + 1]
            if (i, j) in seen or (j, i) in seen:
                continue  # rare duplicate across rounds; the task simply gets one fewer comparison this round
            seen.add((i, j))
            pairs.append({"pair_id": len(pairs), "round": r, "i": i, "j": j})
    json.dump(pairs, open(pairs_path, "w"), indent=0)
print(f"pool {len(pool)} tasks, {len(pairs)} pairs, {2 * len(pairs)} passes")

done = {(r["pair_id"], r["order"]) for r in read_jsonl(res_path)} if args.resume else set()
if not args.resume and os.path.exists(res_path):
    sys.exit(f"{res_path} exists; pass --resume")
todo = [(p, o) for p in pairs for o in (0, 1) if (p["pair_id"], o) not in done]
print(f"{len(done)} done, {len(todo)} to do")

tok, model = load_model()
json.dump({"model": "google/gemma-3-27b-it", "precision": "nf4-bnb-doublequant-bf16compute", "git": git_hash(),
           "rounds": args.rounds, "pool": len(pool), "pairs": len(pairs), "readout": "logP(A) vs logP(B) at last prompt token",
           "torch": torch.__version__}, open(os.path.join(run, "run.json"), "w"), indent=1)

# Gate-0 self-test: gathered-lm_head path equals the model's own logits; right-padding does not change them.
p0 = pair_prompt(tok, text[pairs[0]["i"]], text[pairs[0]["j"]])
p1 = pair_prompt(tok, text[pairs[1]["i"]][:200], text[pairs[1]["j"]][:200])
with torch.no_grad():
    ref = model(input_ids=p0["ids"][None].cuda()).logits[0, -1].float()
mine = last_logits(model, tok, [p0["ids"]])[0]
padded = last_logits(model, tok, [p0["ids"], p1["ids"]])[0]
d_ref = (torch.log_softmax(ref, -1)[[A_ID, B_ID]] - torch.log_softmax(mine, -1)[[A_ID, B_ID]]).abs().max().item()
d_pad = (torch.log_softmax(padded, -1)[[A_ID, B_ID]] - torch.log_softmax(mine, -1)[[A_ID, B_ID]]).abs().max().item()
print(f"selftest: |dlogp| lm_head-path vs model.logits = {d_ref:.4f}; padded vs unpadded = {d_pad:.4f}")
assert d_ref < 1e-2 and d_pad < 5e-2, (d_ref, d_pad)
json.dump({"d_ref": d_ref, "d_pad": d_pad}, open(os.path.join(run, "selftest.json"), "w"))

def build(p, o):
    a, b = (p["i"], p["j"]) if o == 0 else (p["j"], p["i"])
    pp = pair_prompt(tok, text[a], text[b])
    return {"pair_id": p["pair_id"], "round": p["round"], "order": o, "task_a": a, "task_b": b,
            "n_tokens": len(pp["ids"]), "_ids": pp["ids"]}

items = [build(p, o) for p, o in todo]
t0, n_new, first = time.time(), 0, True
for batch in token_batches(items, key=lambda it: it["n_tokens"], budget_tokens=args.budget):
    logits = last_logits(model, tok, [it["_ids"] for it in batch])
    la, lb, mass = ab_readout(logits)
    top = torch.topk(logits, 5, dim=-1)
    rows = []
    for k, it in enumerate(batch):
        r = {kk: v for kk, v in it.items() if kk != "_ids"}
        r.update(logp_a=round(la[k].item(), 5), logp_b=round(lb[k].item(), 5), mass=round(mass[k].item(), 5),
                 top=[[tok.convert_ids_to_tokens(int(i)), round(float(v), 2)] for v, i in zip(top.values[k], top.indices[k])])
        rows.append(r)
    append_jsonl(res_path, rows)
    n_new += len(rows)
    if first:  # label-token verification against sampled outputs (charter §12.5 item 4)
        for r in rows[:4]:
            print("  top5", r["top"], "mass", r["mass"])
        first = False
    if n_new % 200 < len(rows):
        el = time.time() - t0
        print(f"{n_new}/{len(items)} rows, {n_new / el:.2f} rows/s, mean mass so far {mass.mean():.3f}, "
              f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GB peak", flush=True)
    if args.limit and n_new >= args.limit:
        break
print(f"done: {n_new} new rows in {time.time() - t0:.0f}s")
