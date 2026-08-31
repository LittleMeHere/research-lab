"""E3: paired coherent factor edit (protocol v3, manifest v2, post Gate D).

For each E3-eligible parent (14 below / 10 above), the adopted
spots-per-giraffe sentence is rewritten with the factor set to 150 (low arm)
and 300 (high arm); the prefix is cut immediately after the edited sentence,
`<think>` is left open, and the model continues reasoning to its natural end.
Three continuations per arm, temperature 1.0, top_p 1.0, max_tokens 4000.

Prompts are strings (the edit forces retokenization): rendered template +
condition prompt + reasoning[:sentence_start] + edited sentence. Every
response's `prompt_token_ids` are checked against the parent's frozen
prompt_ids + completion_ids; the first divergence must lie at or after the
token boundary of the sentence start (small boundary-merge fuzz allowed) and
is recorded per request as `drift`.

Final answers: text after the LAST `</think>` in the completion, parsed with
the O0 deterministic rules; unfinished (`finish_reason=length` or no
`</think>`) and unparsed rows are flagged for the judge/manual pass. The
retained/revised/restarted/confused classification of every continuation is a
separate manual step (protocol: read all of them).

  uv run python -m value_leakage.fw_e3 --cap_confirmed=True
"""

import asyncio
import json
import math
import random
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path

from value_leakage.fw_e1e2 import PARENTS_RUN, THRESHOLD, one, med, medlog
from value_leakage.o0_decompose import parse_final_deterministic
from value_leakage.sample import build_prompt

ARMS = {"low": "150", "high": "300"}
REPS = 3
SEED = 260832
RENDERED = "[gMASK]<sop><|system|>Reasoning Effort: High<|user|>{prompt}<|assistant|><think>"
NUM = re.compile(r"\d[\d,]*")
FUZZ = 2  # tokens of allowed boundary-merge fuzz before the sentence start


def edited_prefix(reasoning: str, start: int, end: int, new_value: str) -> tuple[str, str, str]:
    """reasoning[:start] + sentence with its single 20..5000 value replaced."""
    sentence = reasoning[start:end]
    cands = [m for m in NUM.finditer(sentence)
             if 20 <= int(m.group(0).replace(",", "")) <= 5000]
    assert len(cands) == 1, f"ambiguous factor in {sentence!r}"
    m = cands[0]
    edited = sentence[:m.start()] + new_value + sentence[m.end():]
    return reasoning[:start] + edited, sentence, edited


def final_from_completion(text: str, finish: str) -> tuple[float | None, str]:
    if "</think>" not in (text or ""):
        return None, "no_think_close"
    tail = text.rsplit("</think>", 1)[1]
    if not tail.strip():
        return None, "empty_after_think"
    v, rule = parse_final_deterministic(tail)
    if v is not None:
        return v, f"parser_{rule}"
    return None, "unparsed"


async def run(cap_confirmed: bool, max_concurrent: int = 8) -> Path:
    if not cap_confirmed:
        raise SystemExit("pass --cap_confirmed=True only with the <=$20 dashboard cap in place")
    from dotenv import load_dotenv
    import os
    load_dotenv()
    key = os.getenv("FIREWORKS_API_KEY")
    from value_leakage.fw_e0 import MODEL
    manifest = json.loads((PARENTS_RUN / "parent_annotations_v2.json").read_text())
    assert manifest["version"] == 2
    eligible = [p for p in manifest["parents"] if p.get("e3_eligible")]
    assert len(eligible) == 24

    parents = {}
    for p in eligible:
        raw = json.loads((PARENTS_RUN / f"parent_{p['parent_id']}.json").read_text())["response"]
        ch = raw["choices"][0]
        parents[p["parent_id"]] = {
            **p,
            "reasoning": ch["message"]["reasoning_content"],
            "prompt_ids": ch["raw_output"]["prompt_token_ids"],
            "completion_ids": ch["raw_output"]["completion_token_ids"],
        }
    # token boundary at/before sentence start, from the frozen echo maps
    from value_leakage.fw_annotate_prep import load_parent
    for pid, par in parents.items():
        lp = load_parent(PARENTS_RUN, pid)
        offs = lp["comp_token_offsets"]
        start = par["e3_sentence"]["start"]
        par["start_boundary_tok"] = max(j for j, o in enumerate(offs) if o <= start)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path("runs") / f"fw_e3_{stamp}"
    out.mkdir(parents=True)
    sample_params = {"temperature": 1.0, "top_p": 1.0, "max_tokens": 4000}
    config = {
        "purpose": "E3 paired coherent factor edit (protocol v3, post Gate D)",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL, "threshold": THRESHOLD,
        "manifest": str(PARENTS_RUN / "parent_annotations_v2.json"),
        "arms": ARMS, "reps": REPS, "shuffle_seed": SEED,
        "sample_params": sample_params,
        "eligible_parents": sorted(parents),
        "prompt_form": "string; rendered template + condition prompt + reasoning[:start] + edited sentence; "
                       "no </think>; per-response token-prefix drift check vs frozen ids",
        "edit_rule": "single 20..5000 value in the adopted sentence -> 150 (low) / 300 (high)",
        "return_token_ids": True,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))

    jobs = []
    edits = {}
    for pid, par in parents.items():
        user_prompt = build_prompt(par["condition"], THRESHOLD)
        s = par["e3_sentence"]
        for arm, val in ARMS.items():
            prefix, orig_sent, edited_sent = edited_prefix(par["reasoning"], s["start"], s["end"], val)
            edits[(pid, arm)] = {"original_sentence": orig_sent, "edited_sentence": edited_sent}
            prompt = RENDERED.format(prompt=user_prompt) + prefix
            for rep in range(REPS):
                jobs.append((f"e3_{pid}_{arm}_r{rep}",
                             {"model": MODEL, "prompt": prompt, "return_token_ids": True,
                              **sample_params}))
    (out / "edits.json").write_text(json.dumps(
        {f"{pid}|{arm}": v for (pid, arm), v in edits.items()}, indent=1, ensure_ascii=False))
    random.Random(SEED).shuffle(jobs)
    print(f"{len(jobs)} E3 requests")

    sem = asyncio.Semaphore(max_concurrent)
    responses = await asyncio.gather(*[one(sem, name, body, out, key) for name, body in jobs])

    rows = []
    usage_in = usage_out = 0
    for (name, body), resp in zip(jobs, responses):
        parts = name.split("_")
        pid, arm, rep = "_".join(parts[1:-2]), parts[-2], parts[-1]
        meta = {"request_name": name, "parent_id": pid, "arm": arm, "replicate": rep,
                "condition": parents[pid]["condition"]}
        if resp is None:
            rows.append({**meta, "error": "api_failure_after_retries"})
            continue
        u = resp.get("usage") or {}
        usage_in += u.get("prompt_tokens", 0); usage_out += u.get("completion_tokens", 0)
        ch = resp["choices"][0]
        text, finish = ch.get("text") or "", ch.get("finish_reason")
        # drift check
        got = ch.get("prompt_token_ids") or resp.get("prompt_token_ids") or []
        exp = parents[pid]["prompt_ids"] + parents[pid]["completion_ids"]
        n_agree = 0
        for a, b in zip(got, exp):
            if a != b:
                break
            n_agree += 1
        boundary = len(parents[pid]["prompt_ids"]) + parents[pid]["start_boundary_tok"]
        val, source = final_from_completion(text, finish)
        rows.append({**meta, "finish_reason": finish, "text": text,
                     "final": val, "final_source": source,
                     "prompt_tokens_agree": n_agree, "expected_boundary": boundary,
                     "drift_before_sentence": n_agree < boundary - FUZZ})
    (out / "results.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))

    # --- summary (numeric part; manual classification comes separately) ------
    per_parent = []
    for pid, par in parents.items():
        rec = {"parent_id": pid, "condition": par["condition"]}
        for arm in ARMS:
            vals = [r["final"] for r in rows
                    if r.get("parent_id") == pid and r.get("arm") == arm and r.get("final")]
            rec[f"{arm}_n"] = len(vals)
            rec[f"{arm}_median"] = med(vals)
            rec[f"{arm}_median_log"] = medlog(vals)
        if rec["low_median_log"] is not None and rec["high_median_log"] is not None:
            rec["high_minus_low_log"] = rec["high_median_log"] - rec["low_median_log"]
        else:
            rec["high_minus_low_log"] = None
        per_parent.append(rec)
    (out / "per_parent.json").write_text(json.dumps(per_parent, indent=1))

    rng = random.Random(SEED + 1)
    summary = {"per_condition": {}, "pooled": {}}
    for scope, sel in (("below_good", lambda r: r["condition"] == "below_good"),
                       ("above_good", lambda r: r["condition"] == "above_good"),
                       ("pooled", lambda r: True)):
        diffs = [r["high_minus_low_log"] for r in per_parent
                 if sel(r) and r["high_minus_low_log"] is not None]
        entry = {"n_parents": len(diffs),
                 "median_high_minus_low_log": statistics.median(diffs) if diffs else None,
                 "ratio": math.exp(statistics.median(diffs)) if diffs else None}
        if len(diffs) >= 3:
            boots = sorted(statistics.median(rng.choices(diffs, k=len(diffs))) for _ in range(2000))
            entry["ci95"] = [boots[50], boots[1949]]
        (summary["per_condition"] if scope != "pooled" else summary)[
            "pooled" if scope == "pooled" else scope] = entry
    ln2 = math.log(2)
    summary["reference"] = {"log_ratio_if_fully_retained_150_to_300": round(ln2, 4),
                            "note": "a parent that keeps the edited factor and its population "
                                    "unchanged should shift by ~log(300/150)=0.693"}
    drift = sum(1 for r in rows if r.get("drift_before_sentence"))
    unfinished = sum(1 for r in rows if r.get("final_source") == "no_think_close")
    unparsed = sum(1 for r in rows if r.get("final_source") in ("unparsed", "empty_after_think"))
    summary["quality"] = {"rows": len(rows), "drift_before_sentence": drift,
                          "no_think_close": unfinished, "unparsed_final": unparsed}
    summary["usage"] = {"prompt_tokens": usage_in, "completion_tokens": usage_out,
                        "usd_at_list_price": round(usage_in * 1.4e-6 + usage_out * 4.4e-6, 4)}
    summary["out"] = str(out)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1, default=str)[:2500])
    return out


def main(cap_confirmed: bool = False, max_concurrent: int = 8) -> None:
    asyncio.run(run(cap_confirmed, max_concurrent))


if __name__ == "__main__":
    import fire

    fire.Fire(main)
