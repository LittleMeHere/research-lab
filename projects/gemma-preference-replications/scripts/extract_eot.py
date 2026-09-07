"""Residual-stream activations at the <end_of_turn> token of every pool task, all 62 decoder-layer outputs
(Gilg et al. Fig. 2 / App. J probe position), captured with forward hooks. Also records per-layer mean residual
norms at the eot token and over the task-content tokens, used to scale steering coefficients.

Out: runs/<run>/acts_eot.f32.npy  [N, 62, 5376] float32 memmap (float16 overflows: Gemma residual norms pass 65504 after ~L45) (row order = pool.json order)
     runs/<run>/norms.json, runs/<run>/done.txt (rows written; resumable)
"""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, load_model, decoder_layers, single_task_ids, pad_batch, token_batches, git_hash, N_LAYERS, D_MODEL

ap = argparse.ArgumentParser()
ap.add_argument("--pool", default="runs/pairwise/pool.json")
ap.add_argument("--run", default="runs/extract")
ap.add_argument("--budget", type=int, default=8000)
args = ap.parse_args()
run = os.path.join(ROOT, args.run)
os.makedirs(run, exist_ok=True)
pool = json.load(open(os.path.join(ROOT, args.pool)))
N = len(pool)
acts_path, done_path = os.path.join(run, "acts_eot.f32.npy"), os.path.join(run, "done.txt")
acts = (np.lib.format.open_memmap(acts_path, mode="r+") if os.path.exists(acts_path)
        else np.lib.format.open_memmap(acts_path, mode="w+", dtype=np.float32, shape=(N, N_LAYERS, D_MODEL)))
done = int(open(done_path).read()) if os.path.exists(done_path) else 0
print(f"{N} tasks, {done} done")

tok, model = load_model()
layers = decoder_layers(model)
state = {}
cap = {}
sums = {"eot": np.zeros(N_LAYERS), "task": np.zeros(N_LAYERS), "n_eot": 0, "n_task": 0}


def mk(l):
    def h(mod, inp, out):
        x = out[0] if isinstance(out, tuple) else out
        B = x.shape[0]
        cap[l] = x[torch.arange(B), state["eot"]].detach().float()
        # norms: eot token and mean over task-content tokens [4, eot) (after "<bos><start_of_turn>user\n")
        sums["eot"][l] += cap[l].norm(dim=-1).sum().item()
        tn = x.norm(dim=-1).float()  # [B, T]
        m = state["task_mask"]
        sums["task"][l] += (tn * m).sum().item()
    return h


handles = [layers[l].register_forward_hook(mk(l)) for l in range(N_LAYERS)]
items = [{"row": r, "ids": single_task_ids(tok, p["text"])[0], "eot": single_task_ids(tok, p["text"])[1]}
         for r, p in enumerate(pool) if r >= done]
t0 = time.time()
n = 0
with torch.no_grad():
    for batch in token_batches(items, key=lambda it: len(it["ids"]), budget_tokens=args.budget, max_batch=32):
        input_ids, attn, _ = pad_batch(tok, [it["ids"] for it in batch])
        state["eot"] = torch.tensor([it["eot"] for it in batch]).cuda()
        mask = torch.zeros_like(attn, dtype=torch.float)
        for i, it in enumerate(batch):
            mask[i, 4:it["eot"]] = 1
        state["task_mask"] = mask
        model.model(input_ids=input_ids, attention_mask=attn)
        sums["n_eot"] += len(batch)
        sums["n_task"] += mask.sum().item()
        for i, it in enumerate(batch):
            acts[it["row"]] = torch.stack([cap[l][i] for l in range(N_LAYERS)]).cpu().numpy().astype(np.float32)
        n += len(batch)
        # rows are written out of pool order (length-sorted batches), so `done` is only advanced at the end
        if n % 512 < len(batch):
            print(f"{n}/{len(items)} {n / (time.time() - t0):.1f} tasks/s", flush=True)
acts.flush()
open(done_path, "w").write(str(N))
assert sums["n_eot"] == N, f"norms would be biased: this invocation processed {sums['n_eot']} of {N} tasks; delete done.txt and re-run"
json.dump({"precision": os.environ.get("PRECISION", "nf4"), "mean_norm_eot": (sums["eot"] / max(sums["n_eot"], 1)).tolist(),
           "mean_norm_task_tokens": (sums["task"] / max(sums["n_task"], 1)).tolist(),
           "n_tasks": N, "git": git_hash(), "note": "norms over the tasks processed in this invocation only"},
          open(os.path.join(run, "norms.json"), "w"), indent=1)
print(f"done {n} in {time.time() - t0:.0f}s; peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
