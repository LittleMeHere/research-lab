# quantization as an alignment lens

This project asks whether weight quantization changes refusal behavior faster than
factual accuracy or instruction following in small, instruction-tuned language
models.

## status

The exploratory phase is complete: six models, four quantization configurations,
and 6,240 saved generations. The current data do not support the original general
claim that safety degrades faster than capability. Several model-level differences
have raw p-values below 0.05, but none remains below the significance threshold
after accounting for the 12 comparisons. They remain exploratory targets.

A draft protocol for a follow-up experiment is awaiting independent review. No
confirmation data have been collected.

> [!WARNING]
> Some raw responses contain harmful or code-like text that security software may
> flag. Treat response text as untrusted data: inspect it in an isolated environment
> and do not execute it.

## research record

- [Current exploratory results](notes/03_capability_axis_and_inverted_thesis.md)
- [Statistical audit of the keyword-scored refusal results](notes/02_statistical_rigor.md)
- [Archived original writeup](notes/01_quantization_alignment_lens.md), retained to
  show how the interpretation changed

## data coverage

| Data | Saved responses or labels |
|---|---:|
| HarmBench, thinking off | 2,400 responses |
| HarmBench, thinking on | 2,400 responses |
| TruthfulQA | 1,200 responses |
| Instruction following | 240 responses |
| TruthfulQA judge A | 1,200/1,200 labels |
| TruthfulQA judge B | 1,200/1,200 labels |
| Refusal judge | 2,076/2,400 labels |

`code/data_audit.py` reports completeness and provenance warnings. The current
results note discusses the warnings that affect interpretation.

## files

| Path | Contents |
|---|---|
| `code/v2_experiment.py` | Six-model experiment runner |
| `code/data_audit.py` | Saved-data and label-integrity checks |
| `code/analyze_results.py` | Aggregate descriptive results |
| `code/stats_analysis.py` | Paired refusal analysis |
| `code/thinking_mode_analysis.py` | Paired thinking-mode analysis |
| `code/judge_rescore.py` | Semantic TruthfulQA and refusal labeling |
| `code/capability_analysis.py` | Judge-scored safety/capability comparisons |
| `code/cross_judge.py` | Sonnet/Opus capability-label comparison |
| `data/v2_results_*.json` | Six raw v2 result files |
| `notes/01_quantization_alignment_lens.md` | Archived original interpretation |
| `notes/02_statistical_rigor.md` | Keyword-refusal statistical audit |
| `notes/03_capability_axis_and_inverted_thesis.md` | Current exploratory results |

## reproduce the exploratory analysis

Run from this project directory:

```bash
pip install -r requirements.txt
python3 code/data_audit.py
python3 code/analyze_results.py
python3 code/stats_analysis.py
python3 code/thinking_mode_analysis.py
python3 code/capability_analysis.py
python3 code/cross_judge.py
```

The original generation used NVIDIA L4 GPUs in GCP `us-central1`, greedy decoding
(`do_sample=False`), and `max_new_tokens=256`.

## next stage

The next experiment will use prompts that were not included in the exploratory
data. Its protocol will specify the hypotheses, scoring rules, and statistical
decision rule before data collection.
