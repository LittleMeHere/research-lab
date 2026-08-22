"""Print descriptive summaries from the saved v2 results and refusal labels."""
import json
import os
import sys
import glob

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_DIR, "data")
NEWER_CHECKPOINTS = {
    "google/gemma-4-e2b-it",
    "microsoft/Phi-4-mini-instruct",
    "HuggingFaceTB/SmolLM3-3B",
    "Qwen/Qwen3.5-4B",
}

QUANT_ORDER = ["fp16", "int8", "int4_fp4", "nf4_dq"]
CATEGORY_LABELS = {
    "chemical_biological": "chemical/bio",
    "copyright": "copyright",
    "cybercrime_intrusion": "cybercrime",
    "harassment_bullying": "harassment",
    "harmful": "harmful",
    "illegal": "illegal",
    "misinformation_disinformation": "misinformation",
}

def load_all():
    files = glob.glob(os.path.join(RESULTS_DIR, "v2_results_*.json"))
    out = []
    for f in sorted(files):
        with open(f, encoding="utf-8") as fh:
            out.append(json.load(fh))
    return out

def sep(c="=", w=90):
    print(c * w)

def pct(v):
    """Format percentage value (stored as 74.0 meaning 74%)"""
    return f"{v:.0f}%" if v is not None else "N/A"


def print_category_refusal(results):
    """Print pooled keyword/judge refusal rates by category and configuration."""
    judge_path = os.path.join(RESULTS_DIR, "judge_refusal_results.json")
    judged = {}
    if os.path.exists(judge_path):
        with open(judge_path, encoding="utf-8") as f:
            for value in json.load(f).values():
                judged[(value["model"], value["quant"], value["idx"])] = (
                    value.get("verdict") == "REFUSE"
                )

    cells = {}
    for data in results:
        model = data["model_id"]
        for quant in QUANT_ORDER:
            rows = data["quant_levels"][quant]["refusal_thinking=False"]["results"]
            for row in rows:
                cell = cells.setdefault(row["category"], {}).setdefault(
                    quant, {"keyword": [], "judge": []}
                )
                cell["keyword"].append(bool(row["refused"]))
                key = (model, quant, row["idx"])
                if key in judged:
                    cell["judge"].append(judged[key])

    print()
    sep()
    print("  CATEGORY-LEVEL REFUSAL — POOLED ACROSS SIX MODELS")
    print("  Each cell is keyword / judge refusal; judge rates use available labels.")
    sep()
    print(f"  {'Category':<18} | {'fp16':>14} | {'int8':>14} | "
          f"{'fp4':>14} | {'nf4':>14}")
    sep("-")
    for category, label in CATEGORY_LABELS.items():
        formatted = []
        for quant in QUANT_ORDER:
            cell = cells[category][quant]
            keyword_rate = sum(cell["keyword"]) / len(cell["keyword"]) * 100
            judge_rate = (sum(cell["judge"]) / len(cell["judge"]) * 100
                          if cell["judge"] else None)
            formatted.append(
                f"{keyword_rate:.1f}% / {judge_rate:.1f}%" if judge_rate is not None
                else f"{keyword_rate:.1f}% / N/A"
            )
        print(f"  {label:<18} | " + " | ".join(f"{value:>14}" for value in formatted))

def analyze():
    results = load_all()
    
    sep()
    print("  LEGACY KEYWORD/SUBSTRING SCORE SUMMARY")
    sep()
    print(f"  Models loaded: {len(results)}")
    if not results:
        print(f"  No v2 result files found in: {RESULTS_DIR}")
        return
    print("  NOTE: Refusal uses a keyword heuristic; Factual uses the legacy substring metric.")
    print("        Use stats_analysis.py and capability_analysis.py for paired analyses.")
    
    # Build rows
    rows = []
    for data in results:
        mid = data["model_id"]
        newer = mid in NEWER_CHECKPOINTS
        has_think = data.get("has_thinking_mode", False)
        
        for q in QUANT_ORDER:
            ql = data.get("quant_levels", {}).get(q)
            if not ql:
                continue
            
            # Keys use "refusal_thinking=False" and "refusal_thinking=True"
            ref_no = ql.get("refusal_thinking=False", {})
            ref_yes = ql.get("refusal_thinking=True", {})
            fac = ql.get("factual", {})
            ins = ql.get("instruction", {})
            
            rows.append({
                "model": mid,
                "newer": newer,
                "has_think": has_think,
                "quant": q,
                "ref": ref_no.get("refusal_rate"),
                "ref_t": ref_yes.get("refusal_rate") if ref_yes else None,
                "fac": fac.get("accuracy") if isinstance(fac.get("accuracy"), (int, float)) else fac.get("correct"),
                "ins": ins.get("pass_rate") if isinstance(ins.get("pass_rate"), (int, float)) else ins.get("passed"),
                "mem": ql.get("memory_mb"),
                "lat": ref_no.get("avg_latency"),
            })
    
    # Per-model tables
    seen = []
    for data in results:
        mid = data["model_id"]
        if mid in seen:
            continue
        seen.append(mid)
        
        tag = "newer checkpoint" if mid in NEWER_CHECKPOINTS else "earlier checkpoint"
        mrows = [r for r in rows if r["model"] == mid]
        has_think = mrows[0]["has_think"] if mrows else False
        
        print()
        sep("-")
        print(f"  {mid}  [{tag}]")
        
        if has_think:
            print(f"  {'Quant':<12} {'Refusal*':>8} {'w/Think*':>8} {'Fact*':>8} {'Instr':>8} {'GPU MB':>8} {'Lat(s)':>8}")
        else:
            print(f"  {'Quant':<12} {'Refusal*':>8} {'Fact*':>8} {'Instr':>8} {'GPU MB':>8} {'Lat(s)':>8}")
        sep("-")
        
        fp16_ref = None
        for r in mrows:
            if r["quant"] == "fp16" and r["ref"] is not None:
                fp16_ref = r["ref"]
            
            mem = f"{r['mem']:.0f}" if r['mem'] else "?"
            lat = f"{r['lat']:.1f}" if r['lat'] else "?"
            
            if has_think:
                print(f"  {r['quant']:<12} {pct(r['ref']):>8} {pct(r['ref_t']):>8} {pct(r['fac']):>8} {pct(r['ins']):>8} {mem:>8} {lat:>8}")
            else:
                print(f"  {r['quant']:<12} {pct(r['ref']):>8} {pct(r['fac']):>8} {pct(r['ins']):>8} {mem:>8} {lat:>8}")
        
        if fp16_ref is not None:
            nf4 = [r for r in mrows if r["quant"] == "nf4_dq"]
            if nf4 and nf4[0]["ref"] is not None:
                d = nf4[0]["ref"] - fp16_ref
                sign = "+" if d > 0 else ""
                print(f"\n  Keyword-refusal point estimate (fp16 -> nf4): {sign}{d:.0f} pp")
    
    # Cross-model comparison for the selected newer checkpoints.
    print()
    sep()
    print("  SELECTED NEWER CHECKPOINTS — KEYWORD REFUSAL RATE")
    sep()
    print(f"  {'Model':<32} {'fp16':>7} {'int8':>7} {'int4':>7} {'nf4':>7} {'Delta':>8}")
    sep("-")
    
    for mid in sorted(NEWER_CHECKPOINTS):
        mrows = [r for r in rows if r["model"] == mid]
        vals = {r["quant"]: r["ref"] for r in mrows}
        short = mid.split("/")[-1]
        fp16 = vals.get("fp16")
        nf4 = vals.get("nf4_dq")
        delta = f"{nf4-fp16:+.0f}pp" if fp16 is not None and nf4 is not None else "N/A"
        
        print(f"  {short:<32} {pct(vals.get('fp16')):>7} {pct(vals.get('int8')):>7} {pct(vals.get('int4_fp4')):>7} {pct(vals.get('nf4_dq')):>7} {delta:>8}")
    
    # Generational comparison
    gen_pairs = [
        ("Qwen/Qwen3-1.7B", "Qwen/Qwen3.5-4B", "Qwen 3 -> 3.5"),
        ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "HuggingFaceTB/SmolLM3-3B", "SmolLM 2 -> 3"),
    ]
    
    print()
    sep()
    print("  CROSS-VERSION KEYWORD REFUSAL COMPARISON")
    sep()
    print(f"  {'Family':<25} {'Earlier':>10} {'Newer':>10} {'Difference':>10}")
    sep("-")
    
    for old_id, new_id, label in gen_pairs:
        old_fp16 = [r for r in rows if r["model"] == old_id and r["quant"] == "fp16"]
        new_fp16 = [r for r in rows if r["model"] == new_id and r["quant"] == "fp16"]
        
        if old_fp16 and new_fp16 and old_fp16[0]["ref"] is not None and new_fp16[0]["ref"] is not None:
            o = old_fp16[0]["ref"]
            n = new_fp16[0]["ref"]
            d = n - o
            print(f"  {label:<25} {pct(o):>10} {pct(n):>10} {d:+.0f}pp")
        else:
            print(f"  {label:<25}       N/A       N/A      N/A")

    print_category_refusal(results)

    # Key findings
    print()
    sep()
    print("  DESCRIPTIVE POINT ESTIMATES (NOT INFERENTIAL FINDINGS)")
    sep()
    
    newer_fp16 = [(r["model"].split("/")[-1], r["ref"])
                  for r in rows if r["newer"] and r["quant"] == "fp16" and r["ref"] is not None]
    
    if newer_fp16:
        best = max(newer_fp16, key=lambda x: x[1])
        worst = min(newer_fp16, key=lambda x: x[1])
        print(f"  1. Highest FP16 keyword refusal: {best[0]} at {best[1]:.0f}%")
        print(f"  2. Lowest FP16 keyword refusal:  {worst[0]} at {worst[1]:.0f}%")
        print(f"  3. Cross-family gap:          {best[1]-worst[1]:.0f} percentage points")
    
    # FP16-to-NF4 differences of at least four percentage points.
    for mid in NEWER_CHECKPOINTS:
        fp16 = [r for r in rows if r["model"] == mid and r["quant"] == "fp16"]
        nf4 = [r for r in rows if r["model"] == mid and r["quant"] == "nf4_dq"]
        if fp16 and nf4 and fp16[0]["ref"] is not None and nf4[0]["ref"] is not None:
            d = nf4[0]["ref"] - fp16[0]["ref"]
            short = mid.split("/")[-1]
            if abs(d) >= 4:
                print(f"  *  {short} NF4−FP16 keyword-refusal difference: {d:+.0f}pp")
    
    # Thinking mode effects
    think_models = [r for r in rows if r["ref_t"] is not None and r["quant"] == "fp16"]
    if think_models:
        print()
        print("  THINKING-MODE KEYWORD POINT ESTIMATES (FP16):")
        for r in think_models:
            if r["ref"] is not None and r["ref_t"] is not None:
                d = r["ref_t"] - r["ref"]
                short = r["model"].split("/")[-1]
                direction = "higher" if d > 0 else "lower" if d < 0 else "same"
                print(f"    {short}: {r['ref']:.0f}% -> {r['ref_t']:.0f}% with thinking ({direction})")
    
    # Totals
    total_configs = len(rows)
    total_models = len(seen)
    newer_count = sum(1 for m in seen if m in NEWER_CHECKPOINTS)
    print()
    print(f"  Total: {total_configs} experiment configs across {total_models} models")
    print(f"         {newer_count} selected newer + {total_models - newer_count} earlier checkpoints")
    sep()

if __name__ == "__main__":
    analyze()
