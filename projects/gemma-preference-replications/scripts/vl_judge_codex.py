"""Independent choice judge for the Value Leakage picks using the Codex CLI (GPT), in place of Gemma-as-judge.
Judges the same rows Gemma judged (all regex-unparsed picks + the 300-row audit sample), in batches, with the original
choice_judge_prompt.yaml instructions. Gemma's verdicts are never shown to the judge. Resumable.
Logs: CLI version, model, reasoning effort. Out: runs/vl_activities/codex/judged_codex.jsonl"""
import json, os, random, re, subprocess, sys, time
import yaml
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import argparse
ap = argparse.ArgumentParser(); ap.add_argument("--batch", type=int, default=25); ap.add_argument("--limit", type=int, default=0); args = ap.parse_args()
MODEL, EFFORT, BATCH = "gpt-5.6-sol", "low", args.batch
tmpl = yaml.safe_load(open(os.path.join(ROOT, "external/value_leakage/choosing_activities/prompts/choice_judge_prompt.yaml")))["choice_judge_prompt"]
picks = {r["var_ix"]: r for r in map(json.loads, open(os.path.join(ROOT, "runs/vl_activities/pick.jsonl")))}
gem = [json.loads(l) for l in open(os.path.join(ROOT, "runs/vl_activities/judged.jsonl"))]
targets = [j["var_ix"] for j in gem]  # identical row set to Gemma's judge pass (unparsed + audit)
out_path = os.path.join(ROOT, "runs/vl_activities/codex/judged_codex.jsonl")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
done = {json.loads(l)["var_ix"] for l in open(out_path)} if os.path.exists(out_path) else set()
todo = [v for v in targets if v not in done]
if args.limit: todo = todo[:args.limit]
ver = subprocess.run(["codex", "--version"], capture_output=True, text=True).stdout.strip()
json.dump({"cli": ver, "model": MODEL, "reasoning_effort": EFFORT, "batch": BATCH, "n_targets": len(targets)}, open(os.path.join(ROOT, "runs/vl_activities/codex/judge_run.json"), "w"), indent=1)
print(f"{ver} model={MODEL} effort={EFFORT}; {len(targets)} targets, {len(done)} done, {len(todo)} to do", flush=True)
# the single-case instructions, kept verbatim, then a batch wrapper
instructions = tmpl.split("<case>")[0].replace("Below is a JSON object containing the two displayed options and the assistant's\nresponse:", "Below are several JSON cases, each with an id, the two displayed options and the assistant's response:")
schema = {"type": "object", "properties": {"verdicts": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "integer"}, "choice": {"type": ["integer", "string"]}}, "required": ["id", "choice"], "additionalProperties": False}}}, "required": ["verdicts"], "additionalProperties": False}
schema_path = os.path.join(ROOT, "runs/vl_activities/codex/schema.json"); json.dump(schema, open(schema_path, "w"))
rules = tmpl.split("</case>")[1]
for s in range(0, len(todo), BATCH):
    ids = todo[s:s + BATCH]
    cases = "\n".join(json.dumps({"id": v, "activity_1": picks[v]["activity_1"], "activity_2": picks[v]["activity_2"], "response": picks[v]["response"]}, ensure_ascii=False) for v in ids)
    prompt = (instructions + "\n<cases>\n" + cases + "\n</cases>\n" + rules.replace('one JSON object, nothing else', 'one JSON object with a "verdicts" array, one entry per case id, nothing else')
              + "\nYou are a classifier only: do not use tools, do not read or write files, do not run commands. Return exactly the JSON.")
    outf = os.path.join(ROOT, f"runs/vl_activities/codex/last_{s}.json")
    for attempt in range(3):
        r = subprocess.run(["codex", "exec", "--ephemeral", "--skip-git-repo-check", "-s", "read-only", "-m", MODEL, "-c", f'model_reasoning_effort="{EFFORT}"',
                            "--output-schema", schema_path, "-o", outf, "-C", os.path.join(ROOT, "runs/vl_activities/codex"), "-"], input=prompt, capture_output=True, text=True, timeout=600)
        try:
            v = json.loads(open(outf).read())["verdicts"]; got = {int(x["id"]): x["choice"] for x in v}
            assert set(got) == set(ids), (len(got), len(ids))
            break
        except Exception as e:
            print("retry", s, attempt, type(e).__name__, str(e)[:120], r.stderr[-300:].replace("\n", " "), flush=True); got = None; time.sleep(5)
    if got is None: print("FAILED batch", s, flush=True); continue
    with open(out_path, "a") as f:
        for v in ids:
            c = got[v]; c = 1 if c in (1, "1") else 2 if c in (2, "2") else "refusal"
            f.write(json.dumps({"var_ix": v, "judge_codex": c}) + "\n")
    print(f"{s + len(ids)}/{len(todo)} done", flush=True)
print("JUDGE_DONE")
