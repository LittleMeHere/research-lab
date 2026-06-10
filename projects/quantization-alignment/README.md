# quantization as an alignment lens

Does quantization degrade safety faster than capability? When you
compress a safety-tuned model from fp16 → int8 → int4 → nf4, what
breaks first — refusal of harmful prompts, factual recall, or
instruction following?

If refusal dies before recall, that's a flag worth raising before
someone ships a worse-aligned model.

**Status:** Data collection complete. 6 models × 4 quant levels, ~6K
inferences on GCP L4s. Writeup and raw data below.

> [!WARNING]
> The `data/` directory contains raw model responses to harmful prompts
> (from [HarmBench](https://github.com/centerforaisafety/HarmBench)).
> Some responses include models complying with harmful requests. This
> data is published for AI safety research purposes.

---

## key findings

Full analysis: [quantization as an alignment lens](notes/01_quantization_alignment_lens.md)

> [!IMPORTANT]
> **Updated after a statistical-rigor pass** ([02](notes/02_statistical_rigor.md),
> [03](notes/03_capability_axis_and_inverted_thesis.md)). The original refusal
> findings below (#1, #4) do **not** survive McNemar + bootstrap — they are within
> single-run noise. With the broken capability scorer rebuilt via an LLM judge, the
> thesis actually **inverts**: under nf4, *capability* degrades significantly
> (Qwen3.5-4B −14pp, SmolLM2 −18pp) while refusal holds. Refusal is the robust
> behavior; factual knowledge is the fragile one. Findings #2 and #3 (baseline gap,
> copyright) still stand. Read the addenda for the corrected record.

1. **Safety degrades under quantization — but it's model-dependent.** SmolLM2
   drops 8pp, Gemma drops 4pp, Phi-4 and Qwen3.5 are immune.
2. **Baseline safety matters more than quantization.** The 52pp gap between
   the safest and least safe model dwarfs the worst quantization effect (8pp).
3. **Copyright is the universal blind spot.** Every model except Phi-4 refuses
   copyright prompts below 20%.
4. **Thinking mode is a double-edged sword.** Helps large aligned models
   (+5-6pp), hurts small ones (−9pp for Qwen3-1.7B).

---

## files

| Path | Contents |
|------|----------|
| `code/v2_experiment.py` | Cross-family experiment runner (v2) |
| `code/quantization_alignment_experiment.py` | Original v1 experiment (Gemma-only) |
| `code/analyze_results.py` | Results aggregator — `python analyze_results.py` |
| `code/stats_analysis.py` | McNemar + bootstrap rigor pass (no GPU) |
| `code/logit_analysis.py` | First-token uncertainty from saved logits (no GPU) |
| `code/logit_plot.py` | Plots logit entropy → `notes/logit_entropy.png` (needs matplotlib) |
| `code/judge_rescore.py` | LLM-as-judge — rebuilds capability axis + validates refusal scorer (runs via `claude -p`) |
| `code/capability_analysis.py` | Safety-vs-capability thesis test on judged labels (no GPU) |
| `data/v2_results_*.json` (×6) | Per-model results: prompts, responses, logit snapshots |
| `data/judge_capability_results.json` | LLM-judge TruthfulQA labels (rebuilt capability axis) |
| `data/truthfulqa_gold.json` | Cached TruthfulQA gold answers (for the judge) |
| `data/results_e2b.json`, `results_e4b.json` | v1 Gemma results |
| `logs/v2_log_vm*.txt` (×3) | GCP L4 execution logs |
| `notes/00_why_im_here.md` | Personal motivation |
| `notes/01_quantization_alignment_lens.md` | Full analysis writeup |
| `notes/02_statistical_rigor.md` | Rigor addendum — the quant refusal deltas are within noise |
| `notes/03_capability_axis_and_inverted_thesis.md` | Capability axis rebuilt — thesis inverts (cap degrades, not safety) |

---

## reproduce

```bash
# requires: GPU with 24GB+ VRAM, CUDA, Python 3.10+
pip install -r requirements.txt

# run experiments (choose model pairs to fit your GPU)
MODELS="google/gemma-4-e2b-it,microsoft/Phi-4-mini-instruct" python3 code/v2_experiment.py
MODELS="Qwen/Qwen3.5-4B,Qwen/Qwen3-1.7B" python3 code/v2_experiment.py
MODELS="HuggingFaceTB/SmolLM3-3B,HuggingFaceTB/SmolLM2-1.7B-Instruct" python3 code/v2_experiment.py

# analyze
python3 code/analyze_results.py
```

**Hardware used:** NVIDIA L4 (24GB), GCP us-central1
**Seed:** 42 (HarmBench), 43 (TruthfulQA)
**Generation:** Greedy (temperature=0, do_sample=False, max_new_tokens=256)

---

## next

- [x] Statistical rigor pass — McNemar + bootstrap ([notes/02](notes/02_statistical_rigor.md)).
      Result: the fp16→nf4 refusal deltas are within single-run noise.
- [x] First-token uncertainty analysis (`code/logit_analysis.py` + plot)
- [ ] **Multi-seed confidence intervals — now the critical path.** The rigor pass
      shows a single run can't separate a ~5pp quant effect from noise; need 3–5
      seeds (or n≈400 prompts) per config before any quant claim holds.
- [ ] LLM-as-judge rescore at scale (`code/judge_rescore.py` ready; needs API key)
      to quantify the keyword scorer's false-positive rate.
- [ ] Cross-quant activation norm comparison (only captured at fp16 — needs re-run)
- [ ] Probing classifiers — find the refusal direction (nnsight; bitsandbytes
      doesn't play well with TransformerLens)
- [ ] Steering vectors — can we restore safety in quantized models?
