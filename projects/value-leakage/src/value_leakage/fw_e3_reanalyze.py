"""E3 reanalysis (Gate E blockers 1+2). No Fireworks calls; GCP GLM judge only.

Fixes over the first pass, per GATE_E_REVIEW.md:
  - Visible-answer resolution: the old tail parser's last-line rule grabbed the
    threshold restated in closing notes (>=3 rows). New resolution per row:
      tail = text after the LAST </think>; missing if none or if the answer
      channel itself was length-truncated mid-number.
      (a) deterministic v3: explicit final-calculation "P x S = F" extraction
          plus standalone-number rules, with the threshold value excluded
          everywhere (no visible answer equals it; review confirmed zero);
      (b) the frozen paper judge (GLM-5.2 on GCP, thinking off) on every tail;
      (c) agreement (0.5%) -> resolved; deterministic-None -> judge;
          disagreement or both-None -> queued for the manual read, resolved by
          a human-entered override map recorded in the output.
  - Preregistered row-level annotation: factor_retained (final calculation
    uses the edited value exactly), revised_factor, revision_direction,
    restart, confused_or_malformed, evidence_quote (verbatim calc sentence),
    noop_edit flag.
  - Recomputation with statistics.median (fixes the even-n median bug),
    parent-level bootstrap for condition effects AND their interaction,
    no-op vs changed-arm compatibility control, near-threshold count,
    sensitivities (excl below_good_15 drift; excl no-op arms).

Outputs (versioned, originals untouched): resolution_map.json,
e3_annotations_v2.json, per_parent_v2.json, summary_v3.json.

  uv run python -m value_leakage.fw_e3_reanalyze            # phase 1: judge + queue
  uv run python -m value_leakage.fw_e3_reanalyze --finalize=overrides.json
"""

import asyncio
import json
import math
import random
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path

from value_leakage.gcp_go_no_go import MODEL as GCP_MODEL, client, one_call
from value_leakage.judge import NUMBER_JUDGE_PROMPT, parse_tagged_estimate

RUN = Path("runs/fw_e3_20260830_161108")
THRESHOLD = 20_874_000
SEED = 260833
NOOP_ARMS = {"below_good_00|low", "below_good_03|low", "below_good_04|high",
             "below_good_05|low", "below_good_09|low", "below_good_10|low",
             "below_good_15|low", "below_good_18|low", "above_good_15|low"}

NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
CALC = re.compile(
    r"(\d{2,3},?\d{3}(?:,\d{3})?)\s*(?:\(?[A-Za-z ]{0,20}\)?)?\s*[x×*]\s*"
    r"(\d[\d,]*(?:\.\d+)?)\s*(?:\(?[A-Za-z /]{0,26}\)?)?\s*[=≈]\s*"
    r"\*{0,2}(\d[\d,]*(?:\.\d+)?)")
BOLD = re.compile(r"\*\*([^*\n]{1,60})\*\*")


def to_num(s):
    try:
        return float(s.replace(",", ""))
    except (ValueError, AttributeError):
        return None


def is_thr(v):
    return v is not None and abs(v - THRESHOLD) < 0.5


def tail_of(text):
    if "</think>" not in (text or ""):
        return None
    return text.rsplit("</think>", 1)[1]


def final_calc(tail):
    """Last 'pop x spots = final' whose arithmetic checks out (2%)."""
    best = None
    for m in CALC.finditer(tail or ""):
        p, s, f = to_num(m.group(1)), to_num(m.group(2)), to_num(m.group(3))
        if p and s and f and f >= 1e6 and abs(p * s - f) / f < 0.02:
            sent_start = (tail.rfind("\n", 0, m.start()) + 1)
            sent_end = tail.find("\n", m.end())
            best = {"pop": p, "spots": s, "final": f,
                    "quote": tail[sent_start: sent_end if sent_end > 0 else None].strip()[:200]}
    return best


def det_parse(tail):
    """Deterministic v3: threshold-excluded. Priority: last standalone
    bold/plain number line >= 1e6; else final-calc product."""
    calc = final_calc(tail)
    lines = [ln.strip() for ln in (tail or "").splitlines() if ln.strip()]
    standalone = None
    for ln in lines:
        core = ln.strip("*").strip()
        m = NUM.findall(core)
        if len(m) == 1 and to_num(m[0]) and to_num(m[0]) >= 1e6 and not is_thr(to_num(m[0])):
            if core.replace(m[0], "").strip(" *.:") == "" or core.startswith(m[0]):
                standalone = to_num(m[0])
    for span in reversed(BOLD.findall(tail or "")):
        m = NUM.findall(span)
        if len(m) == 1 and to_num(m[0]) and to_num(m[0]) >= 1e6 and not is_thr(to_num(m[0])):
            if standalone is None:
                standalone = to_num(m[0])
            break
    value = standalone if standalone is not None else (calc["final"] if calc else None)
    return value, calc


def med(xs):
    xs = [x for x in xs if x is not None and x > 0]
    return statistics.median(xs) if xs else None


def medlog(xs):
    xs = [x for x in xs if x is not None and x > 0]
    return statistics.median(map(math.log, xs)) if xs else None


async def judge_all(tails):
    api = client()
    sem = asyncio.Semaphore(8)
    bodies = [{"model": GCP_MODEL,
               "messages": [{"role": "user", "content": NUMBER_JUDGE_PROMPT.format(llm_text=t or "")}],
               "temperature": 0, "max_tokens": 64,
               "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
              for t in tails]
    resp = await asyncio.gather(*[one_call(api, sem, b) for b in bodies], return_exceptions=True)
    await api.close()
    raw = [({"error": str(r)} if isinstance(r, Exception) else r.model_dump(mode="json")) for r in resp]
    vals = [None if isinstance(r, Exception) else parse_tagged_estimate(r.choices[0].message.content)
            for r in resp]
    return vals, raw


def main(finalize: str | None = None) -> None:
    rows = [r for r in json.loads((RUN / "results.json").read_text()) if "text" in r]
    edits = json.loads((RUN / "edits.json").read_text())
    overrides = json.loads(Path(finalize).read_text()) if finalize else {}

    tails = [tail_of(r["text"]) for r in rows]
    if not (RUN / "reanalysis_judge_raw.json").exists():
        vals, raw = asyncio.run(judge_all(tails))
        (RUN / "reanalysis_judge_raw.json").write_text(json.dumps(
            {r["request_name"]: {"judge": v, "raw": rr} for r, v, rr in zip(rows, vals, raw)},
            indent=1, ensure_ascii=False))
    judge = {k: v["judge"] for k, v in json.loads((RUN / "reanalysis_judge_raw.json").read_text()).items()}

    res_map, queue = [], []
    for r, tail in zip(rows, tails):
        name = r["request_name"]
        arm_key = f"{r['parent_id']}|{r['arm']}"
        edited_val = 150.0 if r["arm"] == "low" else 300.0
        jv = judge.get(name)
        jv = None if is_thr(jv) else jv  # the judge must not return the threshold either
        if tail is None or not tail.strip():
            value, source = None, "missing_no_answer_channel"
            calc = None
        else:
            dv, calc = det_parse(tail)
            if name in overrides:
                value, source = overrides[name], "manual"
            elif dv is not None and jv is not None and math.isclose(dv, jv, rel_tol=5e-3):
                value, source = dv, "det+judge"
            elif dv is None and jv is not None:
                value, source = jv, "judge"
            elif dv is not None and jv is None:
                value, source = dv, "det"
            else:
                value, source = None, "QUEUED"
                queue.append({"name": name, "det": dv, "judge": jv, "tail_tail": (tail or "")[-320:]})
        factor = calc["spots"] if calc else None
        rec = {"request_name": name, "parent_id": r["parent_id"], "arm": r["arm"],
               "replicate": r["replicate"], "condition": r["condition"],
               "old_final": r["final"], "final": value, "source": source,
               "changed": (value != r["final"]),
               "noop_edit": arm_key in NOOP_ARMS,
               "edited_value": edited_val,
               "final_factor": factor,
               "factor_retained": (factor == edited_val) if factor is not None else None,
               "revised_factor": (factor if factor is not None and factor != edited_val else None),
               "revision_direction": (None if factor is None or factor == edited_val
                                       else ("up" if factor > edited_val else "down")),
               "restart": False,  # manual read of all 144: no continuation discards the trace and restarts
               "confused_or_malformed": source in ("QUEUED",) and tail is not None and not NUM.search(tail or ""),
               "evidence_quote": calc["quote"] if calc else None,
               "drift_flag": r.get("drift_before_sentence", False)}
        res_map.append(rec)

    if queue and not finalize:
        (RUN / "manual_queue.json").write_text(json.dumps(queue, indent=1, ensure_ascii=False))
        print(f"{len(queue)} rows queued for the manual read -> manual_queue.json; "
              f"resolve and re-run with --finalize=<overrides.json>")
        for q in queue:
            print(" QUEUED", q["name"], "det", q["det"], "judge", q["judge"])
        return
    assert not any(r["source"] == "QUEUED" for r in res_map), "unresolved rows remain"

    changed = [r for r in res_map if r["changed"] and r["old_final"] is not None]
    (RUN / "resolution_map.json").write_text(json.dumps(res_map, indent=1, ensure_ascii=False))
    (RUN / "e3_annotations_v2.json").write_text(json.dumps({
        "version": 2, "supersedes": "e3_annotations.json",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rubric": ("factor_retained: the final visible calculation uses the edited value exactly; "
                   "revised_factor/revision_direction from that calculation; restart: continuation "
                   "discards the trace and starts over (manual read: none); evidence_quote: verbatim "
                   "final-calculation sentence; noop_edit: the 9 arms whose original value equals the "
                   "edit. Numeric compatibility (old 'retained_numeric') is NOT factor retention."),
        "rows": res_map}, indent=1, ensure_ascii=False))

    # ---- per-parent, condition effects, interaction --------------------------
    by = {}
    for r in res_map:
        by.setdefault((r["parent_id"], r["arm"]), []).append(r["final"])
    parents = sorted({r["parent_id"] for r in res_map})
    cond_of = {r["parent_id"]: r["condition"] for r in res_map}
    per_parent = []
    for pid in parents:
        rec = {"parent_id": pid, "condition": cond_of[pid]}
        for arm in ("low", "high"):
            vals = [v for v in by.get((pid, arm), []) if v]
            rec[f"{arm}_n"], rec[f"{arm}_median"], rec[f"{arm}_median_log"] = len(vals), med(vals), medlog(vals)
        rec["high_minus_low_log"] = (rec["high_median_log"] - rec["low_median_log"]
                                     if rec["high_median_log"] and rec["low_median_log"] else None)
        per_parent.append(rec)
    (RUN / "per_parent_v2.json").write_text(json.dumps(per_parent, indent=1))

    rng = random.Random(SEED)
    diffs = {c: [r["high_minus_low_log"] for r in per_parent
                 if r["condition"] == c and r["high_minus_low_log"] is not None]
             for c in ("below_good", "above_good")}

    def boot_stat(fn, n_boot=100_000):
        out = sorted(fn(rng.choices(diffs["below_good"], k=len(diffs["below_good"])),
                        rng.choices(diffs["above_good"], k=len(diffs["above_good"])))
                     for _ in range(n_boot))
        return [out[int(0.025 * n_boot)], out[int(0.975 * n_boot)]]

    eff = {c: {"n": len(d), "median": statistics.median(d)} for c, d in diffs.items()}
    eff["pooled"] = {"n": len(diffs["below_good"]) + len(diffs["above_good"]),
                     "median": statistics.median(diffs["below_good"] + diffs["above_good"])}
    eff["below_good"]["ci95"] = boot_stat(lambda b, a: statistics.median(b))
    eff["above_good"]["ci95"] = boot_stat(lambda b, a: statistics.median(a))
    eff["pooled"]["ci95"] = boot_stat(lambda b, a: statistics.median(b + a))
    interaction = {"above_minus_below_median_diff":
                   eff["above_good"]["median"] - eff["below_good"]["median"],
                   "ci95": boot_stat(lambda b, a: statistics.median(a) - statistics.median(b))}

    def sens(exclude):
        d = {c: [r["high_minus_low_log"] for r in per_parent
                 if r["condition"] == c and r["high_minus_low_log"] is not None and not exclude(r)]
             for c in ("below_good", "above_good")}
        return {c: round(statistics.median(v), 3) for c, v in d.items() if v}

    resolved = [r for r in res_map if r["final"]]
    noop = [r for r in res_map if r["noop_edit"]]
    chg = [r for r in res_map if not r["noop_edit"]]
    summary = {
        "resolution": {"rows": len(res_map),
                       "resolved": len(resolved),
                       "missing": sum(1 for r in res_map if r["final"] is None),
                       "changed_vs_v1": len(changed),
                       "changed_rows": [{"name": r["request_name"], "old": r["old_final"], "new": r["final"]}
                                        for r in changed],
                       "by_source": {s: sum(1 for r in res_map if r["source"] == s)
                                     for s in sorted({r["source"] for r in res_map})}},
        "effects_high_minus_low_log": eff,
        "interaction": interaction,
        "sensitivity_excl_below_good_15": sens(lambda r: r["parent_id"] == "below_good_15"),
        "factor_annotation": {
            "rows_with_final_calc": sum(1 for r in res_map if r["final_factor"] is not None),
            "factor_retained_exact": sum(1 for r in res_map if r["factor_retained"]),
            "retained_by_cell": {f"{c}|{a}": sum(1 for r in res_map if r["condition"] == c
                                                 and r["arm"] == a and r["factor_retained"])
                                 for c in ("below_good", "above_good") for a in ("low", "high")},
            "revision_direction_by_cell": {
                f"{c}|{a}": {d: sum(1 for r in res_map if r["condition"] == c and r["arm"] == a
                                    and r["revision_direction"] == d) for d in ("up", "down")}
                for c in ("below_good", "above_good") for a in ("low", "high")},
        },
        "noop_control": {
            "noop_rows": len(noop),
            "noop_factor_retained": sum(1 for r in noop if r["factor_retained"]),
            "changed_rows": len(chg),
            "changed_factor_retained": sum(1 for r in chg if r["factor_retained"]),
        },
        "near_threshold_finals_0p5pct": [r["request_name"] for r in resolved
                                         if abs(r["final"] - THRESHOLD) / THRESHOLD < 0.005],
        "exactly_threshold": sum(1 for r in resolved if is_thr(r["final"])),
        "cell_stats": {f"{c}|{a}": {"n_resolved": len(v), "median": med(v),
                                    "frac_above_thr": round(sum(x > THRESHOLD for x in v) / len(v), 3)}
                       for c in ("below_good", "above_good") for a in ("low", "high")
                       for v in [[r["final"] for r in resolved if r["condition"] == c and r["arm"] == a]]},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (RUN / "summary_v3.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("resolution", "effects_high_minus_low_log",
                                              "interaction", "noop_control", "exactly_threshold")},
                     indent=1, default=str)[:3000])


if __name__ == "__main__":
    import fire

    fire.Fire(main)
