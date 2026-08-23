# quantization as an alignment lens

This project asks whether weight quantization changes refusal behavior faster than
factual accuracy or instruction following in small, instruction-tuned language
models.

## exploratory findings

The exploratory phase is complete: six models, four quantization configurations,
and 6,240 saved generations. The current data do not support the original general
claim that safety degrades faster than capability. Several model-level differences
have raw p-values below 0.05, but none remains below the significance threshold
after accounting for the 12 comparisons. They remain exploratory targets.

> [!WARNING]
> Some raw responses contain harmful or code-like text that security software may
> flag. Treat response text as untrusted data: inspect it in an isolated environment
> and do not execute it.

## research record

- [Current exploratory results](notes/03_capability_axis_and_inverted_thesis.md)
- [Statistical audit of the keyword-scored refusal results](notes/02_statistical_rigor.md)
- [Archived original writeup](notes/01_quantization_alignment_lens.md), retained to
  show how the interpretation changed
- [Held-out confirmation protocol](notes/04_confirmation_plan.md)

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

## key files

| Path | Contents |
|---|---|
| `code/data_audit.py` | Saved-data and label-integrity checks |
| `code/stats_analysis.py` | Paired refusal analysis |
| `code/thinking_mode_analysis.py` | Paired thinking-mode analysis |
| `code/capability_analysis.py` | Judge-scored safety/capability comparisons |
| `code/confirmation_spec.py` | Executable confirmation specification |
| `code/confirmation_experiment.py` | Smoke and held-out generation runner |
| `code/confirmation_judge_preflight.py` | Judge-pipeline preparation and verification |
| `code/judge_rescore.py` | Semantic TruthfulQA and refusal labeling |
| `code/adjudicate_labels.py` | Blind disagreement packet and final labels |
| `code/confirmation_analysis.py` | Locked primary and sensitivity analysis |
| `data/v2_results_*.json` | Six raw v2 result files |

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

## held-out confirmation

The [confirmation protocol](notes/04_confirmation_plan.md) defines the hypotheses,
held-out data, judging, analysis, and decision rule. `code/confirmation_spec.py`
contains the executable values enforced throughout the pipeline. The annotated
Git tag `quantization-confirmation-v1-protocol` identifies the exact protocol
revision.

Install the pinned environment and run the protocol tests from this project
directory:

```bash
python3 -m pip install -r requirements-confirmation.txt
python3 -m unittest discover -s code -p 'test_confirmation.py'
```

The protocol's workflow section gives the execution order. Each entry point also
documents its arguments with `--help`. Runs record protocol-file hashes,
environment details, model revisions, module fingerprints, and judge provenance;
the pipeline rejects incompatible locks or outputs.

Generated responses are untrusted data. Handle them in an isolated environment
during collection and review, then publish the raw responses, labels, and analysis.
