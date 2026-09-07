"""Contrastive steering on held-out pairwise prompts (Gilg et al. §2.2, Fig. 33): at decoder layer L, add
+c*norm_L*d to the residual over the Task-A span and -c*norm_L*d over the Task-B span during the single prefill
pass, read P(A) vs P(B). Every pair is run in both orders, so each task sits in slot A once. Swing at a layer:
E_L(c) = mean P(A | +c) - mean P(A | -c)  (= P(chose steered task | +c) - P(chose steered task | -c)).

Directions: 'probe' (runs/probe/directions.npy, per layer, + = higher utility) or 'randK' (seed-K Gaussian unit
vector, same norm scaling). c=0 with the hook attached must reproduce the no-hook baseline bit-for-bit (Gate 0).
Before any steered pass, 20 random prompts with the steered spans marked are written to
data/prompts/examples_highlighted.md.

Rows -> runs/<run>/results.jsonl (resumable; key = pair_id, order, layer, c, dir).
"""
import argparse, json, os, random, sys, time
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import (ROOT, load_model, decoder_layers, pair_prompt, last_logits, ab_readout, token_batches,
                    append_jsonl, read_jsonl, git_hash, D_MODEL, A_ID, B_ID)

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="runs/steer")
ap.add_argument("--layers", required=True, help="comma list")
ap.add_argument("--coefs", required=True, help="comma list, fractions of mean eot residual norm at the layer")
ap.add_argument("--dirs", default="probe", help="comma list: probe, rand0, rand1, ...")
ap.add_argument("--n_pairs", type=int, default=120)
ap.add_argument("--norm", default="eot", choices=["eot", "task"])
ap.add_argument("--budget", type=int, default=6000)
ap.add_argument("--mode", default="contrastive", choices=["contrastive", "a_only"])
ap.add_argument("--probe", default="runs/probe/directions.npy")
ap.add_argument("--norms", default="runs/extract/norms.json")
ap.add_argument("--null", default="runs/null/dirs.npz")
ap.add_argument("--scale_json", default="", help="per-direction c multipliers (natural-SD matching); rows tagged mode+natsd")
ap.add_argument("--tag", default="", help="extra tag appended to the mode key so runs never collide")
ap.add_argument("--random-seed-mode", default="layer", choices=["layer", "name"],
                help="layer reproduces Sep 2–3 rows; name reproduces Sep 4 seedname/natsd controls")
args = ap.parse_args()
run = os.path.join(ROOT, args.run)
os.makedirs(run, exist_ok=True)
res_path = os.path.join(run, "results.jsonl")

pool = json.load(open(os.path.join(ROOT, "runs/pairwise/pool.json")))
text = {p["id"]: p["text"] for p in pool}
ut = pd.read_csv(os.path.join(ROOT, "runs/pairwise/utilities.csv")).set_index("id")
norms = json.load(open(os.path.join(ROOT, args.norms)))
norm_l = norms["mean_norm_eot" if args.norm == "eot" else "mean_norm_task_tokens"]
dirs_all = np.load(os.path.join(ROOT, args.probe))

# held-out pair set: random pairs of canonical-eval tasks (never seen by the probe), fixed once per run dir
pairs_path = os.path.join(run, "pairs.json")
if os.path.exists(pairs_path):
    pairs = json.load(open(pairs_path))
else:
    ev = [p["id"] for p in pool if p["split"] == "eval"]
    rng = random.Random(7)
    pairs = []
    for k in range(args.n_pairs):
        i, j = rng.sample(ev, 2)
        if ut.u[i] < ut.u[j]:
            i, j = j, i  # Gilg convention: task i (= higher utility) listed first in the pair record
        pairs.append({"pair_id": k, "i": i, "j": j, "u_i": float(ut.u[i]), "u_j": float(ut.u[j])})
    json.dump(pairs, open(pairs_path, "w"), indent=0)

tok, model = load_model()
layers = decoder_layers(model)


null_path = os.path.join(ROOT, args.null)
null_dirs = dict(np.load(null_path)) if os.path.exists(null_path) else {}


def direction(name, layer):
    if name == "probe":
        return dirs_all[layer]
    if name.startswith("rand"):
        # Historical contrastive rows use independent draws per layer; later controls reuse one vector.
        seed = int(name[4:]) * 1000 + layer if args.random_seed_mode == "layer" else int(name[4:])
        g = np.random.default_rng(seed).standard_normal(D_MODEL)
        return (g / np.linalg.norm(g)).astype(np.float32)
    return null_dirs[f"{name}_L{layer}"]  # control directions from scripts/null_dirs.py


# prompts for every (pair, order); spans located once
prompts = {}
for p in pairs:
    for o in (0, 1):
        a, b = (p["i"], p["j"]) if o == 0 else (p["j"], p["i"])
        pp = pair_prompt(tok, text[a], text[b])
        prompts[(p["pair_id"], o)] = {"pair_id": p["pair_id"], "order": o, "task_a": a, "task_b": b, "u_a": float(ut.u[a]),
                                      "u_b": float(ut.u[b]), "ids": pp["ids"], "span_a": pp["span_a"], "span_b": pp["span_b"],
                                      "n_tokens": len(pp["ids"])}

# highlighted examples (charter §12.5 item 1) — the exact token spans the hooks will touch
os.makedirs(os.path.join(ROOT, "data/prompts"), exist_ok=True)
with open(os.path.join(ROOT, "data/prompts/examples_highlighted.md"), "w") as f:
    f.write(f"# Steered spans, 20 random held-out prompts (run {args.run}, git {git_hash()})\n\n"
            "`⟦+ ... +⟧` = Task A span (+c·d), `⟦- ... -⟧` = Task B span (−c·d); everything else untouched. "
            "Decision position = final token.\n\n")
    for key in random.Random(0).sample(sorted(prompts), 20):
        q = prompts[key]; ids = q["ids"].tolist(); (a0, a1), (b0, b1) = q["span_a"], q["span_b"]
        s = (tok.decode(ids[:a0]) + "⟦+" + tok.decode(ids[a0:a1]) + "+⟧" + tok.decode(ids[a1:b0]) + "⟦-"
             + tok.decode(ids[b0:b1]) + "-⟧" + tok.decode(ids[b1:]))
        f.write(f"## pair {q['pair_id']} order {q['order']} (u_a={q['u_a']:.2f}, u_b={q['u_b']:.2f})\n```\n{s}\n```\n\n")

tag = args.tag or ("seedname" if args.random_seed_mode == "name" else "")
MODE = args.mode + ("+" + tag if tag else "")
if args.scale_json:
    assert args.random_seed_mode == "name", "Saved natural-SD scales use name-seeded randoms"
scales = json.load(open(os.path.join(ROOT, args.scale_json))) if args.scale_json else {}
done = {(r["pair_id"], r["order"], r["layer"], r["c"], r.get("mode", "contrastive") + ":" + r["dir"]) for r in read_jsonl(res_path)}
coefs = [float(c) for c in args.coefs.split(",")]
state = {"vec": None, "spans": None}


def make_hook(layer):
    def h(mod, inp, out):
        x = out[0] if isinstance(out, tuple) else out
        for i, (sa, sb) in enumerate(state["spans"]):
            x[i, sa[0]:sa[1]] += state["vec"]
            if args.mode == "contrastive":
                x[i, sb[0]:sb[1]] -= state["vec"]
        return out
    return h


# Gate 0: a hook that adds c=0 (a zero vector) must reproduce the no-hook logits bit-for-bit.
_L = int(args.layers.split(",")[0])
_qs = list(prompts.values())[:16]
state["vec"] = torch.zeros(D_MODEL, dtype=torch.bfloat16).cuda()
state["spans"] = [(q["span_a"], q["span_b"]) for q in _qs]
_h = layers[_L].register_forward_hook(make_hook(_L))
_hooked = last_logits(model, tok, [q["ids"] for q in _qs])
_h.remove()
_ref = last_logits(model, tok, [q["ids"] for q in _qs])
_dmax = (_hooked - _ref).abs().max().item()
print(f"gate0: hooked c=0 vs no hook at L{_L}, 16 prompts, max |dlogit| = {_dmax}", flush=True)
assert _dmax == 0.0, _dmax
# positive control: the same hook with a large vector on (i) the decision position must move the logits; (ii) a padding
# position beyond each prompt's length must not. Proves the hook fires where the spans say, not just that zero is zero.
_big = (0.5 * norm_l[_L] * torch.tensor(direction("probe", _L))).to(torch.bfloat16).cuda()
_ids = [q["ids"] for q in _qs] + [torch.cat([_qs[0]["ids"], _qs[0]["ids"][:4]])]  # one longer prompt forces padding on the others
_ref2 = torch.log_softmax(last_logits(model, tok, _ids)[:-1], -1)[:, [A_ID, B_ID]]  # same batch shape, no hook
for _where, _spans in (("decision", [((len(q["ids"]) - 1, len(q["ids"])), (0, 0)) for q in _qs]),
                       ("padding", [((len(q["ids"]) + 1, len(q["ids"]) + 2), (0, 0)) for q in _qs])):
    state["vec"], state["spans"] = _big, _spans
    _h = layers[_L].register_forward_hook(make_hook(_L))
    _out = torch.log_softmax(last_logits(model, tok, _ids)[:-1], -1)[:, [A_ID, B_ID]]
    _h.remove()
    _d = (_out - _ref2).abs().max().item()
    print(f"gate0 positive control: 0.5*norm probe vector on the {_where} position -> max |d logP(A/B)| = {_d:.4f} (same batch, hook on vs off)", flush=True)
    assert (_d > 0.5) if _where == "decision" else (_d == 0.0), (_where, _d)
_dshape = (torch.log_softmax(_ref, -1)[:, [A_ID, B_ID]] - _ref2).abs().max().item()
print(f"gate0 note: batch-shape (padding-length) jitter on logP(A/B), no hook: {_dshape:.4f}", flush=True)

t0, n_new = time.time(), 0
for layer in [int(x) for x in args.layers.split(",")]:
    handle = layers[layer].register_forward_hook(make_hook(layer))
    for dname in args.dirs.split(","):
        d = torch.tensor(direction(dname, layer), dtype=torch.float32)
        for c in coefs:
            todo = [q for q in prompts.values() if (q["pair_id"], q["order"], layer, c, MODE + ":" + dname) not in done]
            if not todo:
                continue
            mult = scales.get(f"{dname}_L{layer}", 1.0)
            state["vec"] = (c * mult * norm_l[layer] * d).to(torch.bfloat16).cuda()
            for batch in token_batches(todo, key=lambda q: q["n_tokens"], budget_tokens=args.budget):
                state["spans"] = [(q["span_a"], q["span_b"]) for q in batch]
                logits = last_logits(model, tok, [q["ids"] for q in batch])
                la, lb, mass = ab_readout(logits)
                rows = [{"pair_id": q["pair_id"], "order": q["order"], "layer": layer, "c": c, "dir": dname, "mode": MODE, "task_a": q["task_a"],
                         "task_b": q["task_b"], "u_a": q["u_a"], "u_b": q["u_b"], "norm": norm_l[layer], "c_mult": mult,
                         "logp_a": round(la[k].item(), 5), "logp_b": round(lb[k].item(), 5), "mass": round(mass[k].item(), 5)}
                        for k, q in enumerate(batch)]
                append_jsonl(res_path, rows)
                n_new += len(rows)
            rr = [r for r in read_jsonl(res_path) if r["layer"] == layer and r["c"] == c and r["dir"] == dname and r.get("mode", "contrastive") == MODE]
            pa = np.array([np.exp(r["logp_a"]) / (np.exp(r["logp_a"]) + np.exp(r["logp_b"])) for r in rr])
            print(f"L{layer} {dname} c={c:+.3f}: mean P(A|A or B)={pa.mean():.3f} (n={len(pa)}) [{n_new} rows, {n_new / (time.time() - t0):.1f}/s]", flush=True)
    handle.remove()
print("done")
