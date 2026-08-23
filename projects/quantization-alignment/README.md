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

The held-out confirmation protocol has passed independent review. No held-out
confirmation responses have been generated.

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
| `code/confirmation_spec.py` | Executable confirmation specification |
| `code/confirmation_experiment.py` | Smoke and held-out generation runner |
| `code/confirmation_judge_preflight.py` | Judge-pipeline preparation and verification |
| `code/adjudicate_labels.py` | Blind disagreement packet and final labels |
| `code/confirmation_analysis.py` | Locked primary and sensitivity analysis |
| `data/v2_results_*.json` | Six raw v2 result files |
| `notes/01_quantization_alignment_lens.md` | Archived original interpretation |
| `notes/02_statistical_rigor.md` | Keyword-refusal statistical audit |
| `notes/03_capability_axis_and_inverted_thesis.md` | Current exploratory results |
| `notes/04_confirmation_plan.md` | Reviewed held-out confirmation protocol |

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

Read the [confirmation plan](notes/04_confirmation_plan.md) before operating the
pipeline. `code/confirmation_spec.py` is the source of truth for executable
constants. Do not edit the protocol files listed there after the environment
smoke test; the runner rejects stale locks and outputs.

Use one isolated Python environment throughout smoke and generation. It needs
CUDA, a C/C++ compiler, development headers matching Python, `pip`, and:

```bash
python3 -m pip install -r requirements-confirmation.txt
python3 -m unittest discover -s code -p 'test_confirmation.py'
```

Do not install or update packages in that environment after the smoke test; full
generation requires its recorded `pip freeze` to match exactly.

The judging steps require authenticated Claude and Codex CLIs. Keep both CLI
versions fixed until judging finishes; a changed version invalidates saved labels
from that backend.

Run the complete smoke on the GPU host. A partial model list does not write the
environment lock.

```bash
MODELS="google/gemma-4-e2b-it,Qwen/Qwen3.5-4B,HuggingFaceTB/SmolLM2-1.7B-Instruct,Qwen/Qwen3-1.7B" \
  python3 code/confirmation_experiment.py --smoke
```

Then exercise both judges and blind adjudication on the exploratory responses:

```bash
python3 code/confirmation_judge_preflight.py prepare

python3 code/judge_rescore.py capability --backend claude --results-dir data/confirmation-judge-preflight
python3 code/judge_rescore.py capability --backend codex --results-dir data/confirmation-judge-preflight
python3 code/judge_rescore.py refusal --backend claude --results-dir data/confirmation-judge-preflight
python3 code/judge_rescore.py refusal --backend codex --results-dir data/confirmation-judge-preflight

python3 code/adjudicate_labels.py capability --results-dir data/confirmation-judge-preflight
python3 code/adjudicate_labels.py refusal --results-dir data/confirmation-judge-preflight
```

If either adjudication command reports disagreements, complete its generated
resolution template using the blinded packet, then rerun that command with
`--resolutions PATH`. Once both adjudicated files exist:

```bash
python3 code/confirmation_judge_preflight.py verify
```

Only after `data/confirmation_environment.json` and
`data/confirmation_judge_lock.json` exist, generate the held-out responses from
a clean Git worktree:

```bash
MODELS="google/gemma-4-e2b-it,Qwen/Qwen3.5-4B,HuggingFaceTB/SmolLM2-1.7B-Instruct,Qwen/Qwen3-1.7B" \
  python3 code/confirmation_experiment.py
```

Repeat the four judge commands and two adjudication commands above with
`data/confirmation` as `--results-dir`, resolve any disagreements, and run:

```bash
python3 code/confirmation_analysis.py
```

Generated harmful responses are untrusted data. Keep raw confirmation artifacts
in the isolated environment described in the plan.
