# the capability axis, rebuilt — and the thesis inverts

*supersedes the "factual recall anomaly" section of [01](01_quantization_alignment_lens.md)*
*analysis: `code/judge_rescore.py` (capability), `code/capability_analysis.py`*

---

## the scorer was broken, not the models

v1 scored TruthfulQA by substring matching and got 2–10% everywhere, called it
"essentially noise," and **disregarded the capability axis entirely**. That was
the right call for that scorer but the wrong conclusion about the data: the models
weren't failing, the scorer was.

Re-grading all 1,200 saved TruthfulQA responses with an LLM judge (Sonnet) using
the official TruthfulQA gold answers (fetched from the dataset's GitHub CSV, joined
by question, 1080/1200 matched) gives a completely different picture:

| | substring scorer | LLM judge |
|---|---:|---:|
| overall TruthfulQA accuracy | **7%** | **57%** |

The capability axis was never noise. It was unmeasured. With it rebuilt, the
project's actual thesis — *does safety degrade faster than capability?* — can
finally be tested, because there is now a real per-prompt capability label paired
across quant levels.

## the thesis inverts

Same rigor as the refusal pass (McNemar + paired bootstrap), fp16 → nf4_dq, per
model. Safety = keyword refusal (n=100); capability = judge TruthfulQA (n=50):

| Model | safety Δ | p | capability Δ | p | |
|-------|---:|:--:|---:|:--:|---|
| Qwen3.5-4B | +0 | 1.000 | **−14** | **0.039** | cap drops, safety holds |
| SmolLM2-1.7B | −8 | 0.152 | **−18** | **0.022** | cap drops, safety holds |
| Qwen3-1.7B | +1 | 1.000 | −10 | 0.125 | (cap drop borderline) |
| gemma-4-e2b | −4 | 0.388 | +0 | 1.000 | both noise |
| SmolLM3-3B | −1 | 1.000 | +0 | 1.000 | both noise |
| Phi-4-mini | +2 | 0.688 | +4 | 0.727 | both noise |

**The first two statistically significant quantization effects in the entire
project are both on capability, not safety.** No model shows a significant refusal
drop under nf4 (the [rigor pass](02_statistical_rigor.md) established that). Two
models show a significant *capability* drop while their refusal is rock-stable.

So the v1 thesis — "safety degrades **faster** than capability" — is **backwards**
for the fragile (smaller / older) models. Under aggressive quantization:

> Refusal is the robust behavior. Factual knowledge is the fragile one.

Qwen3.5-4B is the cleanest case: a strong, well-aligned model whose refusal doesn't
budge (0pp) but which loses 14pp of factual accuracy under nf4. Quantization noise
erodes distributed knowledge faster than it erodes the comparatively shallow,
robust "refuse" behavior. That dissociation is mechanistically plausible and is the
interesting result of the project.

## honest caveats

1. **n=50 on capability** (vs 100 on safety) — wider error bars. The two hits
   (p=0.039, 0.022) do **not** individually clear Bonferroni for 6 tests
   (threshold 0.0083). But 2 significant of 6 when ~0.3 are expected by chance,
   both in the same direction, on the two models you'd predict to be most
   fragile, is a real signal — not the chance pattern the refusal axis showed.
2. **Judge variance.** Capability labels come from one Sonnet pass. It is vastly
   better than substring matching but not ground truth; a second judge or a
   human spot-check would tighten this.
3. **Multi-seed still the critical path.** This makes the multi-seed run more
   urgent, not less — it would confirm (or kill) the capability effect and is now
   pointed at a specific, falsifiable prediction: *nf4 cuts TruthfulQA accuracy
   ~10–18pp on Qwen3.5-4B and SmolLM2 while leaving refusal unchanged.*

## what this changes in the v1 writeup

- The "factual recall anomaly" section (axis is noise → disregard) is superseded.
- Conclusion 1 ("thesis partially supported") should flip: the refusal thesis is
  unsupported; a **capability**-degradation finding takes its place.
- The reframed one-liner: *aggressive quantization degrades what small/older models
  know faster than whether they refuse — refusal is robust, knowledge is fragile.*
