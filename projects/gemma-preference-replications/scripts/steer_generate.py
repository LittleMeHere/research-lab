"""Read what the model writes under contrastive steering (coherence check, charter §12.5 item 6 / Gilg App. F.1).
Same hook as steer.py (prefill only; the added vector never touches generated tokens), greedy decoding, batch 1.
Out: data/prompts/steered_completions.md (for the owner to read) and runs/steer_gen/results.jsonl."""
import argparse, json, os, random, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, load_model, decoder_layers, pair_prompt, append_jsonl, git_hash, A_ID, B_ID

ap = argparse.ArgumentParser()
ap.add_argument("--layer", type=int, default=23)
ap.add_argument("--n_pairs", type=int, default=20)
ap.add_argument("--max_new_tokens", type=int, default=60)
ap.add_argument("--format", default="letter", choices=["letter", "completion"])
args = ap.parse_args()
pool = {p["id"]: p["text"] for p in json.load(open(os.path.join(ROOT, "runs/pairwise/pool.json")))}
pairs = json.load(open(os.path.join(ROOT, "runs/steer/pairs.json")))
norm = json.load(open(os.path.join(ROOT, "runs/extract/norms.json")))["mean_norm_eot"][args.layer]
probe = np.load(os.path.join(ROOT, "runs/probe/directions.npy"))[args.layer]
null = dict(np.load(os.path.join(ROOT, "runs/null/dirs.npz")))
conds = [("probe", 0.0), ("probe", 0.02), ("probe", 0.05), ("probe", -0.05), ("top0", 0.05), ("shuf0", 0.05)]
tok, model = load_model()
layers = decoder_layers(model)
state = {}


def hook(mod, inp, out):
    x = out[0] if isinstance(out, tuple) else out
    if x.shape[1] > 1:  # prefill only
        (a0, a1), (b0, b1) = state["spans"]
        x[0, a0:a1] += state["vec"]; x[0, b0:b1] -= state["vec"]
    return out


h = layers[args.layer].register_forward_hook(hook)
os.makedirs(os.path.join(ROOT, "runs/steer_gen"), exist_ok=True)
os.makedirs(os.path.join(ROOT, "data/prompts"), exist_ok=True)
res_path = os.path.join(ROOT, f"runs/steer_gen/results_{args.format}.jsonl")
md = [f"# Steered completions ({args.format} format), L{args.layer}, greedy, {args.n_pairs} random held-out pairs (git {git_hash()})\n",
      "Vector added to Task A span and subtracted from Task B span during prefill. `[ok]` = first token is A or B.\n"]
rng = random.Random(11)
for p in rng.sample(pairs, args.n_pairs):
    for order in (0, 1):  # order 1 puts the higher-utility task in slot B; without it every trial sits at the slot-bias ceiling
        a, b = (p["i"], p["j"]) if order == 0 else (p["j"], p["i"])
        if args.format == "completion":  # Gilg et al. completion_preference template: the model performs the chosen task
            pp = pair_prompt(tok, pool[a], pool[b], head="You will be given two tasks. Choose one and complete it.\n\nBegin with 'Task A:' or 'Task B:' to indicate your choice, then complete that task.\n\n\nTask A:\n\n", mid="\n\n\nTask B:\n\n")
        else:
            pp = pair_prompt(tok, pool[a], pool[b])
        md.append(f"\n## pair {p['pair_id']} order {order} (u_A={(p['u_i'] if order == 0 else p['u_j']):.2f}, u_B={(p['u_j'] if order == 0 else p['u_i']):.2f})\nTask A: {pool[a][:160]!r}\nTask B: {pool[b][:160]!r}\n")
        for dname, c in conds:
            d = probe if dname == "probe" else null[f"{dname}_L{args.layer}"]
            state["vec"] = (torch.tensor(c * norm * d, dtype=torch.float32)).to(torch.bfloat16).cuda()
            state["spans"] = (pp["span_a"], pp["span_b"])
            with torch.no_grad():
                out = model.generate(pp["ids"][None].cuda(), max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
            text = tok.decode(out[0, len(pp["ids"]):], skip_special_tokens=False)
            first = int(out[0, len(pp["ids"])])
            ok = first in (A_ID, B_ID)
            append_jsonl(res_path, [{"pair_id": p["pair_id"], "order": order, "dir": dname, "c": c, "first_token": tok.convert_ids_to_tokens(first), "ok": ok, "text": text}])
            md.append(f"- {dname} c={c:+.2f} {'[ok]' if ok else '[??]'}: {text.replace(chr(10), ' ')[:200]!r}")
open(os.path.join(ROOT, f"data/prompts/steered_completions_{args.format}.md"), "w").write("\n".join(md))
rows = pd.DataFrame([json.loads(l) for l in open(res_path)])
print(rows.groupby(["dir", "c"]).agg(first_is_AB=("ok", "mean"), n=("ok", "size"), first_A=("first_token", lambda s: (s == "A").mean())).round(2).to_string())
