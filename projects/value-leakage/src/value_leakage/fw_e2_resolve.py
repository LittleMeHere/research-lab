"""Resolve pass for an fw_e1e2 run (protocol v3): the deterministic parser
rejects any forced line without exactly one number, but the audit shows most
rejects carry a clear final value (full multiplications "117,000 x 150 =
17,550,000", "</think>58,500,000" forms, "**N** (X billion)" restatements).
Per protocol these go to the GLM judge (GCP, thinking off, paper-verbatim
NUMBER_JUDGE_PROMPT). The same judge also reads all 40 parents' visible
answers, replacing the deterministic own-final parse in the c4 validity check
(the deterministic rule mis-parsed above_good_03/05, whose true finals match
their forced medians exactly).

Writes into the run dir: resolve_judge_raw.json, results_resolved.json,
parent_finals_judged.json, summary_v2.json (curve, sensitivities, c4
validity, attrition and unresolved classes, red-flag comparison of each cut
gap against the parents' own final-answer gap).

  uv run python -m value_leakage.fw_e2_resolve --run_dir=runs/fw_e1e2_20260830_152305
"""

import asyncio
import json
import math
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path

from value_leakage.fw_e1e2 import CUTS, PARENTS_RUN, THRESHOLD, med, medlog
from value_leakage.gcp_go_no_go import MODEL as GCP_MODEL, client, one_call
from value_leakage.judge import NUMBER_JUDGE_PROMPT, parse_tagged_estimate

SEED = 260831


def classify_unresolved(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return "empty"
    if "</think>" in t:
        return "reopened_think"
    if any(w in t.lower() for w in ("wait", "let's", "refine", "reconsider")):
        return "continued_reasoning"
    return "other"


async def run(run_dir: str, max_concurrent: int = 8) -> None:
    run = Path(run_dir)
    rows = json.loads((run / "results.json").read_text())
    manifest = json.loads((PARENTS_RUN / "parent_annotations_v2.json").read_text())
    parents = {p["parent_id"]: p for p in manifest["parents"]}
    contents = {}
    for pid in parents:
        raw = json.loads((PARENTS_RUN / f"parent_{pid}.json").read_text())["response"]
        contents[pid] = raw["choices"][0]["message"].get("content") or ""

    api = client()
    sem = asyncio.Semaphore(max_concurrent)

    def judge_body(text):
        return {"model": GCP_MODEL,
                "messages": [{"role": "user", "content": NUMBER_JUDGE_PROMPT.format(llm_text=text)}],
                "temperature": 0, "max_tokens": 64,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}

    todo = [i for i, r in enumerate(rows)
            if "text" in r and r.get("parsed") is None and (r.get("text") or "").strip()]
    parent_ids = sorted(parents)
    print(f"judging {len(todo)} unresolved forced lines + {len(parent_ids)} parent finals on GCP")
    responses = await asyncio.gather(
        *[one_call(api, sem, judge_body(rows[i]["text"])) for i in todo],
        *[one_call(api, sem, judge_body(contents[pid])) for pid in parent_ids],
        return_exceptions=True)
    await api.close()

    judge_raw = {"forced": {}, "parents": {}}
    for i, resp in zip(todo, responses[:len(todo)]):
        rec = ({"error": str(resp)} if isinstance(resp, Exception) else resp.model_dump(mode="json"))
        judge_raw["forced"][str(i)] = rec
        val = None if isinstance(resp, Exception) else parse_tagged_estimate(
            resp.choices[0].message.content)
        rows[i]["judge"] = val
    for pid, resp in zip(parent_ids, responses[len(todo):]):
        rec = ({"error": str(resp)} if isinstance(resp, Exception) else resp.model_dump(mode="json"))
        judge_raw["parents"][pid] = rec
    (run / "resolve_judge_raw.json").write_text(json.dumps(judge_raw, indent=1, ensure_ascii=False))

    parent_finals = {}
    for pid in parent_ids:
        rec = judge_raw["parents"][pid]
        val = None if "error" in rec else parse_tagged_estimate(rec["choices"][0]["message"]["content"])
        parent_finals[pid] = val
    (run / "parent_finals_judged.json").write_text(json.dumps(parent_finals, indent=2))

    for r in rows:
        if "text" not in r:
            continue
        if r.get("parsed") is not None:
            r["resolved"], r["source"] = r["parsed"], "parser"
        elif r.get("judge") is not None:
            r["resolved"], r["source"] = r["judge"], "judge"
        else:
            r["resolved"], r["source"] = None, classify_unresolved(r.get("text"))
    (run / "results_resolved.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))

    # ---- summaries over resolved values -------------------------------------
    e1 = {}
    for cond in ("baseline", "below_good", "above_good"):
        vals = [r["resolved"] for r in rows if r.get("request_name", "").startswith(f"e1_{cond}")
                and "text" in r]
        ok = [v for v in vals if v is not None and v > 0]
        e1[cond] = {"n": len(vals), "resolved_n": len(ok), "median": med(ok), "median_log": medlog(ok),
                    "frac_above_threshold": (sum(v > THRESHOLD for v in ok) / len(ok)) if ok else None}
    e1["above_minus_below_median_log_gap"] = (
        e1["above_good"]["median_log"] - e1["below_good"]["median_log"]
        if e1["above_good"]["median_log"] and e1["below_good"]["median_log"] else None)

    by_cell = {}
    for r in rows:
        if r.get("parent_id") and "text" in r:
            by_cell.setdefault((r["parent_id"], r["cut"]), []).append(r["resolved"])
    parent_rows = []
    for (pid, cut), vals in sorted(by_cell.items()):
        ok = [v for v in vals if v is not None and v > 0]
        parent_rows.append({"parent_id": pid, "cut": cut, "condition": parents[pid]["condition"],
                            "n": len(vals), "resolved_n": len(ok), "median": med(ok),
                            "median_log": medlog(ok),
                            "c2_hedged": parents[pid]["c2_hedged"],
                            "c3_target_calc": parents[pid]["c3_contains_target_calc"]})
    (run / "parent_rows_resolved.json").write_text(json.dumps(parent_rows, indent=1))

    def gap(cut, exclude=lambda r: False):
        cell = [r for r in parent_rows if r["cut"] == cut and r["median_log"] is not None and not exclude(r)]
        b = [r["median_log"] for r in cell if r["condition"] == "below_good"]
        a = [r["median_log"] for r in cell if r["condition"] == "above_good"]
        return ({"n_below": len(b), "n_above": len(a),
                 "gap": statistics.median(a) - statistics.median(b)} if a and b else None)

    rng = random.Random(SEED)

    def boot(cut, exclude=lambda r: False, n_boot=2000):
        cell = [r for r in parent_rows if r["cut"] == cut and r["median_log"] is not None and not exclude(r)]
        b = [r["median_log"] for r in cell if r["condition"] == "below_good"]
        a = [r["median_log"] for r in cell if r["condition"] == "above_good"]
        if len(a) < 3 or len(b) < 3:
            return {}
        gaps = sorted(statistics.median(rng.choices(a, k=len(a))) - statistics.median(rng.choices(b, k=len(b)))
                      for _ in range(n_boot))
        return {"ci95": [gaps[int(0.025 * n_boot)], gaps[int(0.975 * n_boot)]]}

    curve = {cut: {**(gap(cut) or {}), **boot(cut)} for cut in CUTS}
    sens = {"c2_excluding_hedged": {**(gap("c2", lambda r: r["c2_hedged"]) or {}),
                                    **boot("c2", lambda r: r["c2_hedged"])},
            "c3_excluding_target_calc": {**(gap("c3", lambda r: r["c3_target_calc"]) or {}),
                                         **boot("c3", lambda r: r["c3_target_calc"])}}

    own_b = [math.log(v) for pid, v in parent_finals.items()
             if parents[pid]["condition"] == "below_good" and v]
    own_a = [math.log(v) for pid, v in parent_finals.items()
             if parents[pid]["condition"] == "above_good" and v]
    own_gap = statistics.median(own_a) - statistics.median(own_b) if own_a and own_b else None

    c4 = []
    for pid in parent_ids:
        forced = next((r for r in parent_rows if r["parent_id"] == pid and r["cut"] == "c4"), None)
        own = parent_finals[pid]
        rec = {"parent_id": pid, "own_final_judged": own,
               "forced_median": forced["median"] if forced else None}
        if own and forced and forced["median"]:
            rec["within_10pct"] = abs(forced["median"] - own) / own <= 0.10
            rec["same_side"] = (forced["median"] > THRESHOLD) == (own > THRESHOLD)
            rec["pass"] = rec["within_10pct"] and rec["same_side"]
        else:
            rec["pass"] = None
        c4.append(rec)
    ev = [r for r in c4 if r["pass"] is not None]
    fails = [r["parent_id"] for r in ev if not r["pass"]]

    unresolved = {}
    for r in rows:
        if "text" in r and r["resolved"] is None:
            unresolved[r["source"]] = unresolved.get(r["source"], 0) + 1
    attrition = {}
    for cut in CUTS:
        for cond in ("below_good", "above_good"):
            cell = [r for r in parent_rows if r["cut"] == cut and r["condition"] == cond]
            tot = sum(r["n"] for r in cell); okn = sum(r["resolved_n"] for r in cell)
            attrition[f"{cut}_{cond}"] = round(1 - okn / tot, 3) if tot else None

    summary = {
        "resolution": {"judged_lines": len(todo),
                       "judge_resolved": sum(1 for i in todo if rows[i].get("judge") is not None),
                       "still_unresolved_by_class": unresolved},
        "e1": e1,
        "e2_curve": curve,
        "sensitivities": sens,
        "parents_own_final_gap_judged": own_gap,
        "red_flag_cut_gap_exceeds_parents_final_gap": {
            cut: (curve[cut].get("gap") is not None and own_gap is not None
                  and curve[cut]["gap"] > own_gap) for cut in CUTS},
        "c4_validity": {"evaluable": len(ev), "failures": fails,
                        "failure_rate": round(len(fails) / len(ev), 3) if ev else None,
                        "detail": c4},
        "attrition_post_resolution": attrition,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (run / "summary_v2.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1, default=str)[:3500])


def main(run_dir: str = "runs/fw_e1e2_20260830_152305", max_concurrent: int = 8) -> None:
    asyncio.run(run(run_dir, max_concurrent))


if __name__ == "__main__":
    import fire

    fire.Fire(main)
