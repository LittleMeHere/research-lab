"""E1 (empty-reasoning forced-answer control) and E2 (chronological
truncate-and-force curve) on Fireworks GLM-5.2, protocol v3, manifest v2.

E1: prompt = rendered(<condition prompt>) + "</think>\\nFinal estimate:".
    below/above are built from the sampled parents' exact prompt token ids
    (identical across a condition, asserted) + the forcing suffix; baseline is
    a string prompt whose returned token ids must end with the suffix.
    100 samples per condition (4 requests x n=25), temperature 1.0, top_p 1.0,
    max_tokens 32, stop at newline.

E2: for every parent x cut in parent_annotations_v2.json (40 x c1..c4),
    prompt = prompt_ids + completion_ids[:cut_token_index] + SUFFIX, n=5,
    same sampling params. Request order seeded-shuffled.

The forcing suffix token ids [154842, 198, 19321, 16031, 25] decode to
"</think>\\nFinal estimate:" (captured in E0 run 04_D); the run begins with an
echo verification call and aborts if the decoded tail differs.

Outputs under runs/fw_e1e2_<stamp>/: config.json (written before any call),
00_suffix_verification.json, e1_*.json and e2_*.json raw request+response,
results.json (parsed rows), summary.json (gap curve, sensitivities, c4
validity, attrition, bootstrap CIs).

  uv run python -m value_leakage.fw_e1e2 --cap_confirmed=True
"""

import asyncio
import json
import math
import os
import random
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from value_leakage.fw_e0 import BASE, MODEL, PRICE_IN, PRICE_OUT, post
from value_leakage.sample import build_prompt

PARENTS_RUN = Path("runs/fw_parents_20260830_134529")
THRESHOLD = 20_874_000
SUFFIX = [154842, 198, 19321, 16031, 25]  # </think>\nFinal estimate:
SUFFIX_TEXT = "</think>\nFinal estimate:"
CUTS = ("c1", "c2", "c3", "c4")
SEED = 260830
NUM_RE = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s*(million|billion)?", re.I)


def parse_forced(text: str) -> float | None:
    """Exactly one number on the forced line (markdown/units tolerated)."""
    m = NUM_RE.findall(text or "")
    if len(m) != 1:
        return None
    v = float(m[0][0].replace(",", ""))
    return v * {"": 1, "million": 1e6, "billion": 1e9}[m[0][1].lower()]


async def one(sem, name, body, out, key, attempts=4):
    async with sem:
        for attempt in range(1, attempts + 1):
            try:
                status, resp = await asyncio.to_thread(post, "/completions", body, key, 600)
            except Exception as e:
                status, resp = None, {"transport_error": f"{type(e).__name__}: {e}"}
            rec = {"sent_at": datetime.now(timezone.utc).isoformat(), "attempt": attempt,
                   "request": body, "http_status": status, "response": resp}
            if status == 200:
                (out / f"{name}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
                return resp
            (out / f"{name}.attempt{attempt}_http{status}.json").write_text(
                json.dumps(rec, indent=1, ensure_ascii=False))
            await asyncio.sleep(2 ** attempt)
    return None


def med(xs):
    xs = [x for x in xs if x is not None and x > 0]
    return statistics.median(xs) if xs else None


def medlog(xs):
    xs = [x for x in xs if x is not None and x > 0]
    return statistics.median(map(math.log, xs)) if xs else None


async def run(cap_confirmed: bool, max_concurrent: int = 8) -> Path:
    if not cap_confirmed:
        raise SystemExit("pass --cap_confirmed=True only with the <=$20 dashboard cap in place")
    load_dotenv()
    key = os.getenv("FIREWORKS_API_KEY")
    manifest = json.loads((PARENTS_RUN / "parent_annotations_v2.json").read_text())
    assert manifest["version"] == 2 and manifest["annotation_status"] == "frozen_before_any_forced_outcome"
    parents = {}
    for p in manifest["parents"]:
        raw = json.loads((PARENTS_RUN / f"parent_{p['parent_id']}.json").read_text())["response"]
        ro = raw["choices"][0]["raw_output"]
        parents[p["parent_id"]] = {**p, "prompt_ids": ro["prompt_token_ids"],
                                   "completion_ids": ro["completion_token_ids"],
                                   "content": raw["choices"][0]["message"].get("content") or ""}
    # per-condition prompt ids must be identical across parents
    cond_pids = {}
    for cond in ("below_good", "above_good"):
        ids = {tuple(v["prompt_ids"]) for v in parents.values() if v["condition"] == cond}
        assert len(ids) == 1, f"{cond}: prompt ids differ across parents"
        cond_pids[cond] = list(ids.pop())

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path("runs") / f"fw_e1e2_{stamp}"
    out.mkdir(parents=True)
    sample_params = {"temperature": 1.0, "top_p": 1.0, "max_tokens": 32, "stop": ["\n"]}
    config = {
        "purpose": "E1 empty-reasoning control + E2 chronological forced curve (protocol v3)",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL, "threshold": THRESHOLD,
        "manifest": str(PARENTS_RUN / "parent_annotations_v2.json"),
        "manifest_version": 2,
        "suffix_token_ids": SUFFIX, "suffix_text": SUFFIX_TEXT,
        "sample_params": sample_params,
        "e1": {"per_condition": 100, "requests": "100 independent n=1 requests per condition"},
        "e2": {"continuations_per_cell": 5, "cells": "40 parents x c1..c4",
               "requests": "5 independent n=1 requests per cell"},
        "n_bug_note": "n>1 on /completions conditions only choice 0 on the prompt; "
                      "see runs/fw_n_bug_probe_20260830 and the invalidated run "
                      "fw_e1e2_20260830_151944",
        "shuffle_seed": SEED,
        "parser": parse_forced.__doc__,
        "c4_validity_rule": "parent median forced estimate within 10% of the parent's own visible "
                            "final answer AND on the same side of the threshold; >20% failures "
                            "blocks mechanistic interpretation (protocol v3)",
        "preregistered_sensitivities": manifest["preregistered_sensitivities"],
    }
    (out / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))

    # --- suffix verification (echo, 1 token) ---------------------------------
    p0 = parents["below_good_00"]
    ver_body = {"model": MODEL, "prompt": p0["prompt_ids"] + p0["completion_ids"][:p0["cuts"]["c1"]["token_index"]] + SUFFIX,
                "temperature": 0, "max_tokens": 1, "echo": True, "logprobs": 1}
    status, ver = await asyncio.to_thread(post, "/completions", ver_body, key, 300)
    (out / "00_suffix_verification.json").write_text(json.dumps(
        {"request": ver_body, "http_status": status, "response": ver}, indent=1, ensure_ascii=False))
    n_echo = len(ver_body["prompt"])
    offs = ver["choices"][0]["logprobs"]["text_offset"]
    echoed_prompt_text = ver["choices"][0]["text"][:offs[n_echo]] if len(offs) > n_echo else ver["choices"][0]["text"]
    assert echoed_prompt_text.endswith(SUFFIX_TEXT), f"suffix decode mismatch: {echoed_prompt_text[-60:]!r}"
    print("suffix verification ok:", repr(echoed_prompt_text[-40:]))

    # --- build jobs ----------------------------------------------------------
    jobs = []
    # E1 baseline: string prompt (no sampled parent); template from below prompt ids is not
    # reusable textually, so rebuild the rendered string via the known template shape.
    rendered = "[gMASK]<sop><|system|>Reasoning Effort: High<|user|>{prompt}<|assistant|><think>"
    for cond in ("baseline", "below_good", "above_good"):
        if cond == "baseline":
            prompt = rendered.format(prompt=build_prompt("baseline", None)) + SUFFIX_TEXT
        else:
            prompt = cond_pids[cond] + SUFFIX
        for rep in range(100):
            jobs.append((f"e1_{cond}_r{rep}", {"model": MODEL, "prompt": prompt,
                                               **sample_params}))
    for pid, par in parents.items():
        for cut in CUTS:
            k = par["cuts"][cut]["token_index"]
            for rep in range(5):
                jobs.append((f"e2_{pid}_{cut}_r{rep}", {"model": MODEL,
                                                        "prompt": par["prompt_ids"] + par["completion_ids"][:k] + SUFFIX,
                                                        **sample_params}))
    random.Random(SEED).shuffle(jobs)
    print(f"{len(jobs)} requests (300 E1 + {len(parents)*4*5} E2)")

    sem = asyncio.Semaphore(max_concurrent)
    responses = await asyncio.gather(*[one(sem, name, body, out, key) for name, body in jobs])

    # --- parse ---------------------------------------------------------------
    rows = []
    usage_in = usage_out = 0
    for (name, body), resp in zip(jobs, responses):
        kind = "e1" if name.startswith("e1_") else "e2"
        meta = {"request_name": name}
        parts = name.split("_")
        if kind == "e1":
            meta.update({"condition": "_".join(parts[1:-1]), "replicate": parts[-1]})
        else:
            meta.update({"parent_id": "_".join(parts[1:-2]), "cut": parts[-2],
                         "replicate": parts[-1], "condition": "_".join(parts[1:3])})
        if resp is None:
            rows.append({**meta, "error": "api_failure_after_retries"})
            continue
        u = resp.get("usage") or {}
        usage_in += u.get("prompt_tokens", 0); usage_out += u.get("completion_tokens", 0)
        for ch in resp["choices"]:
            rows.append({**meta, "choice_index": ch["index"], "text": ch.get("text"),
                         "finish_reason": ch.get("finish_reason"),
                         "parsed": parse_forced(ch.get("text"))})
    (out / "results.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))

    # --- E1 summary ----------------------------------------------------------
    e1 = {}
    for cond in ("baseline", "below_good", "above_good"):
        vals = [r["parsed"] for r in rows if r.get("request_name", "").startswith(f"e1_{cond}")
                and "choice_index" in r]
        ok = [v for v in vals if v is not None and v > 0]
        e1[cond] = {"n": len(vals), "parsed_n": len(ok), "median": med(ok),
                    "median_log": medlog(ok),
                    "frac_above_threshold": (sum(v > THRESHOLD for v in ok) / len(ok)) if ok else None}
    e1["above_minus_below_median_log_gap"] = (e1["above_good"]["median_log"] - e1["below_good"]["median_log"]
                                              if e1["above_good"]["median_log"] and e1["below_good"]["median_log"] else None)

    # --- E2 summary ----------------------------------------------------------
    by_cell = {}
    for r in rows:
        if "parent_id" in r and "choice_index" in r:
            by_cell.setdefault((r["parent_id"], r["cut"]), []).append(r["parsed"])
    parent_rows = []
    for (pid, cut), vals in sorted(by_cell.items()):
        ok = [v for v in vals if v is not None and v > 0]
        parent_rows.append({"parent_id": pid, "cut": cut,
                            "condition": parents[pid]["condition"],
                            "n": len(vals), "parsed_n": len(ok),
                            "median": med(ok), "median_log": medlog(ok),
                            "c2_hedged": parents[pid]["c2_hedged"],
                            "c3_target_calc": parents[pid]["c3_contains_target_calc"]})

    def gap(cut, exclude=lambda row: False):
        cell = [r for r in parent_rows if r["cut"] == cut and r["median_log"] is not None
                and not exclude(r)]
        b = [r["median_log"] for r in cell if r["condition"] == "below_good"]
        a = [r["median_log"] for r in cell if r["condition"] == "above_good"]
        if not (a and b):
            return None
        return {"n_below": len(b), "n_above": len(a),
                "gap": statistics.median(a) - statistics.median(b)}

    rng = random.Random(SEED + 1)

    def boot(cut, exclude=lambda row: False, n_boot=2000):
        cell = [r for r in parent_rows if r["cut"] == cut and r["median_log"] is not None
                and not exclude(r)]
        b = [r["median_log"] for r in cell if r["condition"] == "below_good"]
        a = [r["median_log"] for r in cell if r["condition"] == "above_good"]
        if len(a) < 3 or len(b) < 3:
            return None
        gaps = sorted(statistics.median(rng.choices(a, k=len(a))) - statistics.median(rng.choices(b, k=len(b)))
                      for _ in range(n_boot))
        return {"ci95": [gaps[int(0.025 * n_boot)], gaps[int(0.975 * n_boot)]]}

    curve = {cut: {**(gap(cut) or {}), **(boot(cut) or {})} for cut in CUTS}
    sensitivities = {
        "c2_excluding_hedged": {**(gap("c2", lambda r: r["c2_hedged"]) or {}),
                                **(boot("c2", lambda r: r["c2_hedged"]) or {})},
        "c3_excluding_target_calc": {**(gap("c3", lambda r: r["c3_target_calc"]) or {}),
                                     **(boot("c3", lambda r: r["c3_target_calc"]) or {})},
    }

    # --- c4 validity ---------------------------------------------------------
    from value_leakage.o0_decompose import parse_final_deterministic
    c4 = []
    for pid, par in parents.items():
        forced = next((r for r in parent_rows if r["parent_id"] == pid and r["cut"] == "c4"), None)
        own, rule = parse_final_deterministic(par["content"])
        rec = {"parent_id": pid, "own_final": own, "own_parse_rule": rule,
               "forced_median": forced["median"] if forced else None}
        if own and forced and forced["median"]:
            rec["within_10pct"] = abs(forced["median"] - own) / own <= 0.10
            rec["same_side"] = (forced["median"] > THRESHOLD) == (own > THRESHOLD)
            rec["pass"] = rec["within_10pct"] and rec["same_side"]
        else:
            rec["pass"] = None
        c4.append(rec)
    c4_evaluable = [r for r in c4 if r["pass"] is not None]
    c4_fail = [r["parent_id"] for r in c4_evaluable if not r["pass"]]

    # --- attrition -----------------------------------------------------------
    attrition = {}
    for cut in CUTS:
        for cond in ("below_good", "above_good"):
            cell = [r for r in parent_rows if r["cut"] == cut and r["condition"] == cond]
            tot = sum(r["n"] for r in cell); okn = sum(r["parsed_n"] for r in cell)
            attrition[f"{cut}_{cond}"] = {"continuations": tot, "parsed": okn,
                                          "rate": round(1 - okn / tot, 3) if tot else None}

    summary = {
        "e1": e1, "e2_curve": curve, "sensitivities": sensitivities,
        "c4_validity": {"evaluable": len(c4_evaluable), "failures": c4_fail,
                        "failure_rate": round(len(c4_fail) / len(c4_evaluable), 3) if c4_evaluable else None,
                        "unevaluable": [r["parent_id"] for r in c4 if r["pass"] is None],
                        "detail": c4},
        "attrition": attrition,
        "usage": {"prompt_tokens": usage_in, "completion_tokens": usage_out,
                  "usd_at_list_price": round(usage_in * PRICE_IN + usage_out * PRICE_OUT, 4)},
        "out": str(out),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("e1", "e2_curve", "sensitivities", "c4_validity", "usage")},
                     indent=1, default=str)[:4000])
    return out


def main(cap_confirmed: bool = False, max_concurrent: int = 8) -> None:
    asyncio.run(run(cap_confirmed, max_concurrent))


if __name__ == "__main__":
    import fire

    fire.Fire(main)
