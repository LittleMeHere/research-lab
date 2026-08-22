# exploratory results after scorer and statistical audit

*Current account of the exploratory phase. This supersedes the interpretation in
[`01_quantization_alignment_lens.md`](01_quantization_alignment_lens.md).*

All model-level quantization comparisons in this document are exploratory. Raw
p-values are unadjusted unless stated otherwise. Holm and Bonferroni corrections
account for testing 12 comparisons at once; neither procedure identifies a
statistically significant difference. This qualification applies to every
model-level difference reported below.

## scope

The v2 experiment saved 6,240 generations from six models and four quantization
configurations. The comparisons in the main table use non-thinking responses and
pair each prompt's requested FP16 output with its NF4 double-quantized output.

The study measured three outcomes:

- refusal on HarmBench prompts;
- accuracy on TruthfulQA questions; and
- programmatically checked instruction following.

The original TruthfulQA substring metric and the refusal keyword heuristic were
not sufficiently reliable for the model-level comparisons. The current analysis
uses LLM-judge labels for TruthfulQA and compares keyword and judge labels for
refusal.

## measurement audit

### TruthfulQA

The substring metric reported 7% aggregate accuracy. A Sonnet judge reported 57%
on the same 1,200 responses.

Official TruthfulQA references were matched by question for 1,080 responses. The
remaining 120 were judged from model knowledge because the exploratory cache did
not contain their references. The follow-up experiment will require matched
references for every item.

### refusal

The Sonnet refusal pass produced 2,076 labels for 2,400 responses. The remaining
324 labels are missing, and missingness varies by model and quantization
configuration. Judge-scored refusal percentages therefore describe an incomplete
subset.

Across the 2,076 available labels, the keyword heuristic and judge agree 85.7% of
the time. Keyword scoring and judge scoring produce different model-level
estimates in two comparisons central to the original interpretation:

- SmolLM2 changes from −8pp under keyword scoring to +9pp under judge scoring.
- Gemma changes from −4pp under keyword scoring to −12pp under judge scoring.

The comparison does not treat the judge as ground truth. It shows that differences
of the size studied here depend on the scoring method.

## FP16-to-NF4 comparisons

Differences are quantized minus requested FP16, in percentage points. Capability
uses 50 paired TruthfulQA questions per cell. Keyword refusal uses 100 HarmBench
rows; judge refusal uses 60–100 paired labels depending on missingness.

| Model | Refusal, keyword | Refusal, judge | TruthfulQA, judge |
|---|---:|---:|---:|
| Gemma-4-e2b | −4pp | −12pp (`p=0.007`) | 0pp |
| Qwen3.5-4B | 0pp | 0pp | −14pp (`p=0.039`) |
| SmolLM2-1.7B | −8pp | +9pp | −18pp (`p=0.022`) |
| Qwen3-1.7B | +1pp | +5pp | −10pp (`p=0.125`) |
| SmolLM3-3B | −1pp | +3pp | 0pp |
| Phi-4-mini | +2pp | 0pp | +4pp |

Three of the 12 judge-scored model-by-outcome comparisons have raw p-values below
0.05. The smallest is Gemma at `p=0.007`; the Bonferroni threshold for 12 tests is
`0.05 / 12 = 0.0042`. None passes the Holm-adjusted 0.05 threshold.

## second-judge comparison

Opus re-labeled all 1,200 TruthfulQA responses. Sonnet and Opus are different
models from the same provider, so this is a two-model robustness comparison rather
than a cross-provider replication.

| Model | Sonnet difference | Opus difference | Label agreement | Both-agree subset |
|---|---:|---:|---:|---:|
| Qwen3.5-4B | −14pp (`p=0.039`) | −10pp (`p=0.125`) | 92% | −13pp (`p=0.031`) |
| SmolLM2-1.7B | −18pp (`p=0.022`) | −14pp (`p=0.118`) | 88% | −18pp (`p=0.039`) |
| Qwen3-1.7B | −10pp (`p=0.125`) | −8pp (`p=0.125`) | 94% | −7pp (`p=0.250`) |
| Gemma / SmolLM3 / Phi-4 | approximately 0pp | approximately 0pp | 90–94% | approximately 0pp |

The two judges give the same direction for the Qwen3.5 and SmolLM2 comparisons,
but neither comparison has a raw p-value below 0.05 under Opus alone. The
both-agree subset is selected using judge agreement, so its p-values are
descriptive and are not independent confirmation.

## comparisons selected for confirmation

The audit identified four comparisons for a follow-up confirmation experiment:

| Model and quantization | Outcome | Exploratory estimate |
|---|---|---:|
| Qwen3.5-4B, NF4 | TruthfulQA accuracy | −14pp Sonnet; −10pp Opus |
| SmolLM2-1.7B, NF4 | TruthfulQA accuracy | −18pp Sonnet; −14pp Opus |
| Gemma-4-e2b, NF4 | Responses containing materially harmful assistance | +8pp on 88 paired labels |
| Qwen3-1.7B, FP4 | TruthfulQA accuracy | −24pp Sonnet; −20pp Opus |

The first three came from the NF4 analysis. The Qwen3 FP4 comparison was selected
during the subsequent all-configuration audit. Selection occurred after inspection
of the exploratory data, so new prompts are required to test them.

## additional provenance limits

- The v2 `fp16` label records the requested configuration, not a verified loaded
  parameter dtype. Gemma and Qwen3.5 use composite model configurations that may
  have retained native BF16 parameters under the Transformers version used for v2.
- HarmBench contains 100 sampled rows but 99 unique normalized prompt texts.

## interpretation

### the models do not share one quantization response

The exploratory data do not support a general claim that quantization degrades
safety faster than capability. Under judge scoring, Gemma is the only model with
lower refusal under NF4; the other five are unchanged or have higher refusal.
TruthfulQA accuracy falls for Qwen3.5, SmolLM2, and Qwen3, while it is unchanged or
higher for the other three models. The observed direction therefore depends on the
model and outcome rather than following one safety-versus-capability pattern.

### baseline and category differences

Differences between models are larger than the FP16-to-NF4 differences within a
model. On the available FP16 judge labels, refusal ranges from 18% for SmolLM3 to
88% for Phi-4. Exact rates are affected by the uneven missing-label counts described
above, but the spread is also present under the complete keyword scorer.

Copyright prompts provide the clearest category-level pattern. Pooling the six
models within each quantization configuration gives the following refusal rates:

| Scoring method | FP16 | INT8 | FP4 | NF4 |
|---|---:|---:|---:|---:|
| Keyword scorer | 20.0% | 17.8% | 13.3% | 20.0% |
| Judge, available labels | 29.1% | 31.2% | 32.5% | 37.3% |

Copyright is the lowest-refusal category at every configuration under both scoring
methods. At FP16, five of the six models have copyright refusal rates of 0–20%
under the keyword scorer and 8–33% on available judge labels; Phi-4 is the exception
at 87% and 80%, respectively. This is not a quantization effect—the pooled rate does
not fall consistently as precision decreases—and the category contains only 15
prompts per model. The judge's separate materially-harmful label is near zero for
these responses, so the result concerns refusal behavior on copyright prompts
rather than the severity of their content.

### model-specific capability results

The Qwen3.5 and SmolLM2 TruthfulQA declines have the same direction under both
judges, although only the Sonnet comparisons have raw p-values below 0.05. The
Qwen3 FP4 decline also has the same direction under both judges. These patterns
motivate model-specific confirmation tests; they do not show that quantization
generally reduces factual accuracy.

### thinking mode and instruction following

Thinking mode has no consistent direction across models, and all six FP16
intervals include zero. Instruction-following scores range from 80% to 100%, but
each cell contains only 10 tests, making that measure too coarse to support a
claim that general capability was preserved.

### compression

NF4 reduced recorded peak GPU memory by 21–53% relative to the requested FP16
configuration. This is the practical benefit being compared with behavioral
changes. The exact reductions are subject to the dtype provenance limitation above.
