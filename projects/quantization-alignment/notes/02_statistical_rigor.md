# statistical rigor addendum

*methods addendum to [01_quantization_alignment_lens.md](01_quantization_alignment_lens.md)*
*analysis: `code/stats_analysis.py` — runs on existing data, no GPU*

---

## why this exists

The v1 writeup reports refusal deltas (e.g. "SmolLM2 −8pp under nf4") as
point estimates from a **single run of 100 prompts**. Its own limitations
section flags this: "the ±2-4pp variations could be within noise." This
addendum tests that directly, using two methods that exploit the paired
design (the same 100 prompts were run at every quant level).

- **McNemar's exact test** — looks only at prompts that *flipped* between
  two quant levels. Compares "refused→complied" flips (safety lost) against
  "complied→refused" flips (safety gained). Balanced flips = noise.
- **Paired bootstrap 95% CI** — resamples the 100 prompts 10,000× (seed 42)
  to put an error bar on each delta. A CI that crosses 0 = indistinguishable
  from noise.

---

## headline result: the quantization deltas are noise

fp16 → nf4_dq (the thesis's strongest claim), non-thinking mode:

| Model | fp16 | nf4 | Δ | 95% CI | McNemar p | verdict |
|-------|-----:|----:|---:|:------:|:---------:|---------|
| SmolLM2-1.7B | 33% | 25% | −8 | [−18, +1] | 0.152 | noise (CI crosses 0) |
| gemma-4-e2b | 74% | 70% | −4 | [−11, +3] | 0.388 | noise |
| SmolLM3-3B | 36% | 35% | −1 | [−9, +6] | 1.000 | noise |
| Qwen3.5-4B | 84% | 84% | +0 | [−3, +3] | 1.000 | noise |
| Qwen3-1.7B | 52% | 53% | +1 | [−7, +9] | 1.000 | noise |
| Phi-4-mini | 88% | 90% | +2 | [−3, +7] | 0.688 | noise |

**0 of 6** fp16→nf4 refusal changes are statistically distinguishable from
single-run noise. The two cases the v1 writeup counted as thesis support
(SmolLM2 −8pp "strong support", Gemma −4pp "moderate support") both have
confidence intervals that include zero and non-significant McNemar p-values.

## multiple comparisons

Across all 18 tests (3 quant levels × 6 models), exactly **1** reaches
p<0.05: SmolLM2 fp16→**int8** at **+12pp** (p=0.008) — note the sign:
quantization *raised* refusal here, opposite to the thesis. But:

- Pure noise across 18 tests is expected to produce ~0.9 hits at p<0.05.
  We observed 1. The hit rate matches the false-positive rate.
- It does **not** survive Bonferroni correction (threshold p<0.0028).

So even the one "significant" result is consistent with chance.

---

## what this means for the project

1. **The thesis as originally stated is not supported.** "Quantization
   degrades safety faster than capability" — at n=100, single run, the
   refusal effect of quantization is not measurable above noise for any
   model tested. This is a clean negative result, and the v1 limitations
   section predicted it.

2. **The real signals in the dataset are the ones that aren't about
   quantization at all,** and they are large:
   - the **52pp baseline gap** between safest (Phi-4 88%) and least safe
     (SmolLM3 36%) latest-gen models
   - the **copyright blind spot** (every family but Phi-4 below 20%),
     which replicates across all 6 models — exactly the kind of
     cross-model consistency that noise does not produce.

3. **The reframed contribution** is stronger and more honest:
   *single-run quantization deltas are within noise; the dominant axes of
   safety variation are baseline tuning and harm category, not bit-width.*
   The "quantization as a stress test" framing survives only if multi-seed
   runs (next) reveal an effect the single run couldn't.

## to upgrade from "noise" to a real measurement

The bottleneck is statistical power, not GPU. To detect a true ~5pp effect
you need either more prompts per run (n≈400+) or multiple seeds per config
(3–5 runs) so within-config variance can be separated from the quant effect.
Both are listed in v1 future work; the stats here show they are the
*critical path*, not a nicety.
