"""Choice judge for runs/vl_activities/pick.jsonl using the original choice_judge_prompt.yaml, run on
Gemma-3-27B-it itself (greedy) instead of their gpt-5.4-mini. Judges every row the regex could not parse, plus
an audit sample of regex-parsed rows to measure agreement. Out: runs/<run>/judged.jsonl (var_ix, judge_raw, judge)."""
import argparse, json, os, random, re, sys, time
import torch, yaml
sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, load_model, append_jsonl, read_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="runs/vl_activities")
ap.add_argument("--audit", type=int, default=300)
ap.add_argument("--batch", type=int, default=16)
args = ap.parse_args()
run = os.path.join(ROOT, args.run)
tmpl = yaml.safe_load(open(os.path.join(ROOT, "external/value_leakage/choosing_activities/prompts/choice_judge_prompt.yaml")))["choice_judge_prompt"]
picks = read_jsonl(os.path.join(run, "pick.jsonl"))
out_path = os.path.join(run, "judged.jsonl")
done = {r["var_ix"] for r in read_jsonl(out_path)}
unparsed = [r for r in picks if r["parsed"] is None]
parsed = [r for r in picks if r["parsed"] is not None]
audit = random.Random(0).sample(parsed, min(args.audit, len(parsed)))
todo = [r for r in unparsed + audit if r["var_ix"] not in done]
print(f"{len(picks)} picks, {len(unparsed)} unparsed, audit {len(audit)}, {len(todo)} to judge", flush=True)
_JSON_RE = re.compile(r"\{[^{}]*\}")


def parse_judge(raw):
    m = _JSON_RE.search(raw or "")
    if not m:
        return "refusal"
    try:
        c = json.loads(m.group(0)).get("choice")
    except Exception:
        return "refusal"
    return 1 if c in (1, "1") else 2 if c in (2, "2") else "refusal"


tok, model = load_model()
tok.padding_side = "left"
t0 = time.time()
for s in range(0, len(todo), args.batch):
    batch = todo[s:s + args.batch]
    texts = [tok.apply_chat_template([{"role": "user", "content": tmpl.format(llm_text=json.dumps(
        {"activity_1": r["activity_1"], "activity_2": r["activity_2"], "response": r["response"]}, ensure_ascii=False))}],
        tokenize=False, add_generation_prompt=True) for r in batch]
    enc = tok(texts, add_special_tokens=False, return_tensors="pt", padding=True).to(0)
    with torch.no_grad():
        out = model.generate(**enc, do_sample=False, max_new_tokens=24, pad_token_id=tok.pad_token_id)
    raws = [tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True) for o in out]
    append_jsonl(out_path, [{"var_ix": r["var_ix"], "regex": r["parsed"], "judge_raw": raw, "judge": parse_judge(raw)}
                            for r, raw in zip(batch, raws)])
    if (s // args.batch) % 20 == 0:
        print(f"{s + len(batch)}/{len(todo)} {(s + len(batch)) / (time.time() - t0):.1f}/s", flush=True)
print("done")
