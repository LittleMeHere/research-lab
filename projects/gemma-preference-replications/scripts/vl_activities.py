"""Value Leakage 'Choosing Activities' (Betley et al. 2026 §7) on Gemma-3-27B-it NF4, no-tools setting.

Near-verbatim port of external/value_leakage/choosing_activities/: same 100 activities, same 5 random-pick
wordings, same 5 liking-score wordings, same variation generator (pairing RNG seeded as theirs, so pair/order/
template per variation index match their seed-0 design). Sampling replaces their API calls: temperature 1,
pure sampling (top_p 1, no top_k), which is what an API default looks like; Gemma's own generation_config
(top_k 64, top_p 0.95) is deliberately not used.

  --stage liking : 100 activities x 5 prompts x 4 repeats -> runs/<run>/liking.jsonl
  --stage pick   : N variations of the random-pick prompt -> runs/<run>/pick.jsonl
Both append incrementally and resume. Choice parsing is a regex first pass only (field `parsed`); an LLM-judge
pass (vl_judge.py) resolves the rest, as in the original pipeline.
"""
import argparse, json, os, random, re, sys, time
import torch, yaml
sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, load_model, append_jsonl, read_jsonl, git_hash

VL = os.path.join(ROOT, "external", "value_leakage", "choosing_activities")

ap = argparse.ArgumentParser()
ap.add_argument("--stage", choices=["liking", "pick"], required=True)
ap.add_argument("--run", default="runs/vl_activities")
ap.add_argument("--n_variations", type=int, default=10_000)
ap.add_argument("--n_repeats", type=int, default=20)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--max_new_tokens", type=int, default=256)
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()
run = os.path.join(ROOT, args.run)
os.makedirs(run, exist_ok=True)

activities = yaml.safe_load(open(os.path.join(VL, "activities.yaml")))["activities"]
assert len(activities) == 100
names = [a["name"] for a in activities]


def load_templates(fname, key):
    t = yaml.safe_load(open(os.path.join(VL, "prompts", fname)))[key]
    assert len(t) == 5
    return t


# ---- variations: verbatim logic of pipeline.build_variations + build_prompt (pair sample, shuffle, template draw)
def build_variations(n, seed, templates):
    rng = random.Random(seed)
    template_rng = random.Random(f"{seed}:template")
    out = []
    for ix in range(n):
        template_ix = template_rng.randrange(len(templates))
        pair = tuple(rng.sample(activities, 2))
        ordered = list(pair)
        rng.shuffle(ordered)
        prompt = templates[template_ix].format(activity_1=ordered[0]["name"], activity_2=ordered[1]["name"])
        out.append({"var_ix": ix, "template_ix": template_ix, "activity_1": ordered[0]["name"],
                    "activity_2": ordered[1]["name"], "prompt": prompt})
    return out


# ---- liking: score_activities.py assigns prompt variants balanced across repeats, seeded per activity
def build_liking(n_repeats, seed, prompts):
    out = []
    for aix, name in enumerate(names):
        rng = random.Random(f"{seed}:{aix}")
        order = [i % len(prompts) for i in range(n_repeats)]
        rng.shuffle(order)
        for rep, pix in enumerate(order):
            out.append({"activity_ix": aix, "activity": name, "prompt_ix": pix, "rep": rep,
                        "prompt": prompts[pix].format(activity=name)})
    return out


_JSON_RE = re.compile(r"\{[^{}]*\}")


def parse_score(raw):
    m = _JSON_RE.search(raw or "")
    if m:
        try:
            v = int(str(json.loads(m.group(0)).get("score")).strip())
            if 0 <= v <= 100:
                return v
        except Exception:
            pass
    m = re.fullmatch(r"\s*(\d{1,3})\s*", raw or "")
    if m and 0 <= int(m.group(1)) <= 100:
        return int(m.group(1))
    return None


def parse_choice(resp, a1, a2):
    """Regex first pass. Returns 1, 2, or None (ambiguous / neither / both -> judge)."""
    r = resp.lower()
    hits = set()
    if re.search(r"\(\s*1\s*\)|option\s*1\b|option\s*one\b|number\s*1\b|#\s*1\b|\bfirst (one|option)\b", r):
        hits.add(1)
    if re.search(r"\(\s*2\s*\)|option\s*2\b|option\s*two\b|number\s*2\b|#\s*2\b|\bsecond (one|option)\b", r):
        hits.add(2)
    n1, n2 = r.count(a1.lower()), r.count(a2.lower())
    if n1 and not n2:
        hits.add(1)
    if n2 and not n1:
        hits.add(2)
    return hits.pop() if len(hits) == 1 else None


tok, model = load_model()
tok.padding_side = "left"
gen_kw = dict(do_sample=True, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=args.max_new_tokens,
              pad_token_id=tok.pad_token_id)


@torch.no_grad()
def generate(prompts):
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
             for p in prompts]
    enc = tok(texts, add_special_tokens=False, return_tensors="pt", padding=True).to(0)
    assert (enc["input_ids"] == tok.bos_token_id).sum(1).eq(1).all()
    out = model.generate(**enc, **gen_kw)
    return [tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True) for o in out]


if args.stage == "liking":
    items = build_liking(args.n_repeats, args.seed, load_templates("activity_liking_prompt.yaml", "score_prompts"))
    path, key = os.path.join(run, "liking.jsonl"), lambda r: (r["activity_ix"], r["rep"])
    gen_kw["max_new_tokens"] = 48
else:
    items = build_variations(args.n_variations, args.seed, load_templates("prompt_template.yaml", "prompt_templates"))
    path, key = os.path.join(run, "pick.jsonl"), lambda r: r["var_ix"]

done = {key(r) for r in read_jsonl(path)}
todo = [it for it in items if key(it) not in done]
print(f"{args.stage}: {len(items)} items, {len(done)} done, {len(todo)} to do", flush=True)
json.dump({"stage": args.stage, "git": git_hash(), "gen": gen_kw, "seed": args.seed, "n_items": len(items),
           "model": "google/gemma-3-27b-it", "precision": "nf4-bnb"}, open(os.path.join(run, f"{args.stage}_run.json"), "w"), indent=1)

t0, n = time.time(), 0
for s in range(0, len(todo), args.batch):
    batch = todo[s:s + args.batch]
    resp = generate([it["prompt"] for it in batch])
    rows = []
    for it, r in zip(batch, resp):
        row = dict(it)
        row["response"] = r
        row["parsed"] = parse_score(r) if args.stage == "liking" else parse_choice(r, it["activity_1"], it["activity_2"])
        rows.append(row)
    append_jsonl(path, rows)
    n += len(rows)
    if s == 0:
        for r in rows[:3]:
            print("---", r["prompt"][-80:].replace("\n", " "), "\n   ->", r["response"][:300].replace("\n", " "), "| parsed", r["parsed"])
    if (s // args.batch) % 10 == 0:
        el = time.time() - t0
        print(f"{n}/{len(todo)} {n / el:.2f} rows/s, parsed {sum(r['parsed'] is not None for r in rows)}/{len(rows)} "
              f"in last batch, {torch.cuda.max_memory_allocated() / 2**30:.1f} GB peak", flush=True)
    if args.limit and n >= args.limit:
        break
print(f"done {n} rows in {time.time() - t0:.0f}s")
