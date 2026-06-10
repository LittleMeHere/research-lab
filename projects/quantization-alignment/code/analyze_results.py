"""
Quantization Safety Degradation - Results Analyzer
Reads all v2_results_*.json files and produces a summary report.
"""
import json
import os
import sys
import glob

sys.stdout.reconfigure(encoding='utf-8')

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
LATEST_GEN = {
    "google/gemma-4-e2b-it",
    "microsoft/Phi-4-mini-instruct",
    "HuggingFaceTB/SmolLM3-3B",
    "Qwen/Qwen3.5-4B",
}

QUANT_ORDER = ["fp16", "int8", "int4_fp4", "nf4_dq"]

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

def analyze():
    results = load_all()
    
    sep()
    print("  QUANTIZATION SAFETY DEGRADATION — COMPLETE RESULTS")
    sep()
    print(f"  Models loaded: {len(results)}")
    
    # Build rows
    rows = []
    for data in results:
        mid = data["model_id"]
        latest = mid in LATEST_GEN
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
                "latest": latest,
                "has_think": has_think,
                "quant": q,
                "ref": ref_no.get("refusal_rate"),
                "ref_t": ref_yes.get("refusal_rate") if ref_yes else None,
                "fac": fac.get("accuracy") if isinstance(fac.get("accuracy"), (int, float)) else fac.get("correct"),
                "ins": ins.get("pass_rate") if isinstance(ins.get("pass_rate"), (int, float)) else ins.get("passed"),
                "mem": ql.get("memory_mb"),
                "lat": ref_no.get("avg_latency"),
            })
    
    # Check what factual key actually is
    # Let me also check if accuracy is stored differently
    sample_ql = results[0]["quant_levels"]["fp16"]
    fac_data = sample_ql.get("factual", {})
    fac_keys = [k for k in fac_data.keys() if k != "results"]
    
    # Per-model tables
    seen = []
    for data in results:
        mid = data["model_id"]
        if mid in seen:
            continue
        seen.append(mid)
        
        tag = "LATEST GEN" if mid in LATEST_GEN else "prior gen"
        mrows = [r for r in rows if r["model"] == mid]
        has_think = mrows[0]["has_think"] if mrows else False
        
        print()
        sep("-")
        print(f"  {mid}  [{tag}]")
        
        if has_think:
            print(f"  {'Quant':<12} {'Refusal':>8} {'w/Think':>8} {'Factual':>8} {'Instr':>8} {'GPU MB':>8} {'Lat(s)':>8}")
        else:
            print(f"  {'Quant':<12} {'Refusal':>8} {'Factual':>8} {'Instr':>8} {'GPU MB':>8} {'Lat(s)':>8}")
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
                arrow = "+" if d > 0 else "" if d < 0 else "="
                print(f"\n  Safety delta (fp16 -> nf4): {arrow}{d:.0f} pp")
    
    # Cross-model latest gen
    print()
    sep()
    print("  CROSS-MODEL COMPARISON — LATEST GENERATION ONLY (Refusal Rate)")
    sep()
    print(f"  {'Model':<32} {'fp16':>7} {'int8':>7} {'int4':>7} {'nf4':>7} {'Delta':>8}")
    sep("-")
    
    for mid in sorted(LATEST_GEN):
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
    print("  GENERATIONAL SAFETY IMPROVEMENT")
    sep()
    print(f"  {'Family':<25} {'Old fp16':>10} {'New fp16':>10} {'Gain':>8}")
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
    
    # Key findings
    print()
    sep()
    print("  KEY FINDINGS")
    sep()
    
    latest_fp16 = [(r["model"].split("/")[-1], r["ref"]) 
                   for r in rows if r["latest"] and r["quant"] == "fp16" and r["ref"] is not None]
    
    if latest_fp16:
        best = max(latest_fp16, key=lambda x: x[1])
        worst = min(latest_fp16, key=lambda x: x[1])
        print(f"  1. Strongest baseline safety: {best[0]} at {best[1]:.0f}%")
        print(f"  2. Weakest baseline safety:   {worst[0]} at {worst[1]:.0f}%")
        print(f"  3. Cross-family gap:          {best[1]-worst[1]:.0f} percentage points")
    
    # Max degradation
    for mid in LATEST_GEN:
        fp16 = [r for r in rows if r["model"] == mid and r["quant"] == "fp16"]
        nf4 = [r for r in rows if r["model"] == mid and r["quant"] == "nf4_dq"]
        if fp16 and nf4 and fp16[0]["ref"] is not None and nf4[0]["ref"] is not None:
            d = nf4[0]["ref"] - fp16[0]["ref"]
            short = mid.split("/")[-1]
            if abs(d) >= 4:
                print(f"  *  Notable degradation: {short} dropped {abs(d):.0f}pp under nf4")
    
    # Thinking mode effects
    think_models = [r for r in rows if r["ref_t"] is not None and r["quant"] == "fp16"]
    if think_models:
        print()
        print("  THINKING MODE EFFECTS (fp16 baseline):")
        for r in think_models:
            if r["ref"] is not None and r["ref_t"] is not None:
                d = r["ref_t"] - r["ref"]
                short = r["model"].split("/")[-1]
                direction = "higher" if d > 0 else "lower" if d < 0 else "same"
                print(f"    {short}: {r['ref']:.0f}% -> {r['ref_t']:.0f}% with thinking ({direction})")
    
    # Totals
    total_configs = len(rows)
    total_models = len(seen)
    latest_count = sum(1 for m in seen if m in LATEST_GEN)
    print()
    print(f"  Total: {total_configs} experiment configs across {total_models} models")
    print(f"         {latest_count} latest gen + {total_models - latest_count} prior gen")
    sep()

if __name__ == "__main__":
    analyze()
