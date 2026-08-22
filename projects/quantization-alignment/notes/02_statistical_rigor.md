# statistical audit of the keyword-scored refusal results

*Scope: paired statistical analysis of the refusal and thinking-mode claims in the
archived original writeup. Analysis: `code/stats_analysis.py` and
`code/thinking_mode_analysis.py`.*

## data and methods

The original writeup compared refusal percentages from one run of 100 HarmBench
rows per model and quantization configuration. Every configuration received the
same prompts, so the analysis is paired by prompt.

- **Exact McNemar test:** compares the number of prompts that changed from refusal
  to compliance with the number that changed from compliance to refusal.
- **Paired bootstrap interval:** resamples prompts 10,000 times and reports a 95%
  interval for the percentage-point difference.

An interval containing zero does not identify the direction of the difference at
the stated confidence level. It does not establish that the underlying difference
is exactly zero.

## NF4 refusal comparison

The table compares FP16 with NF4 double quantization in non-thinking mode. Refusal
is classified by the original keyword heuristic.

| Model | FP16 | NF4 | Difference | 95% interval | McNemar p |
|---|---:|---:|---:|:---:|---:|
| SmolLM2-1.7B | 33% | 25% | −8pp | [−18, +1] | 0.152 |
| Gemma-4-e2b | 74% | 70% | −4pp | [−11, +3] | 0.388 |
| SmolLM3-3B | 36% | 35% | −1pp | [−9, +6] | 1.000 |
| Qwen3.5-4B | 84% | 84% | 0pp | [−3, +3] | 1.000 |
| Qwen3-1.7B | 52% | 53% | +1pp | [−7, +9] | 1.000 |
| Phi-4-mini | 88% | 90% | +2pp | [−3, +7] | 0.688 |

All six intervals contain zero. The SmolLM2 and Gemma comparisons therefore do not
support the original writeup's claims of −8pp and −4pp safety degradation under
this metric and sample.

## multiple comparisons

The exploratory analysis contains 18 FP16-to-quantized refusal comparisons: six
models and three quantized configurations. One raw p-value is below 0.05:
SmolLM2 FP16-to-INT8 at +12pp (`p=0.008`). Its direction is higher refusal under
INT8, and it does not pass the Bonferroni threshold of `0.05 / 18 = 0.0028`.

Under a global null, 18 tests at an unadjusted 0.05 threshold produce 0.9 such
results in expectation. The observed count is one. The analysis therefore does
not identify a keyword-scored refusal difference that passes a family-wise error
threshold.

## thinking-mode comparison

The original writeup also compared thinking on with thinking off at FP16. The same
100 prompts were used in both modes.

| Model | Difference | 95% interval | McNemar p | Direction across FP16/INT8/FP4/NF4 |
|---|---:|:---:|---:|:---:|
| Qwen3-1.7B | −9pp | [−20, +1] | 0.136 | − − − − |
| SmolLM3-3B | −2pp | [−10, +7] | 0.824 | − − − + |
| Phi-4-mini | 0pp | [0, 0] | 1.000 | 0 0 0 0 |
| SmolLM2-1.7B | 0pp | [0, 0] | 1.000 | 0 0 0 0 |
| Gemma-4-e2b | +5pp | [−3, +13] | 0.302 | + + + + |
| Qwen3.5-4B | +6pp | [−2, +14] | 0.238 | + + + + |

All six FP16 intervals contain zero. The direction patterns across quantization
configurations are not independent replications because they reuse the same prompts
and closely related model states.

## conclusion

The keyword-scored refusal data do not support the original claims about either
quantization or thinking mode. This conclusion is limited to the keyword metric and
the exploratory sample. The semantic rescoring and its remaining limitations are
reported in
[`03_capability_axis_and_inverted_thesis.md`](03_capability_axis_and_inverted_thesis.md).
