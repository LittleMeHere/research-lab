# the capability axis, rebuilt — and what the thesis actually is

*supersedes the "factual recall anomaly" section of [01](01_quantization_alignment_lens.md)*
*analysis: `code/judge_rescore.py`, `code/capability_analysis.py`*

---

## the scorer was broken, not the models

v1 scored TruthfulQA by substring matching, got 2–10% everywhere, called it
"essentially noise," and **disregarded the capability axis**. That was right about
the scorer and wrong about the data: the models weren't failing, the scorer was.

Re-grading all 1,200 saved TruthfulQA responses with an LLM judge (Sonnet) using
the official TruthfulQA gold answers (fetched from the dataset's GitHub CSV, joined
by question, 1080/1200 matched):

| | substring scorer | LLM judge |
|---|---:|---:|
| overall TruthfulQA accuracy | **7%** | **57%** |

The capability axis was never noise — it was unmeasured. With it rebuilt, the
project's actual question can finally be tested.

## the fair test: one judge, both axes

To avoid comparing a clean judge (capability) against a noisy keyword scorer
(safety), the same Sonnet judge re-scored refusal too. McNemar + paired bootstrap,
fp16 → nf4_dq, paired by prompt:

| Model | refusal (keyword) | refusal (JUDGE) | capability (JUDGE) |
|-------|---:|---:|---:|
| gemma-4-e2b | −4 (n.s.) | **−12 (p=0.007)** | +0 (n.s.) |
| Qwen3.5-4B | +0 (n.s.) | +0 (n.s.) | **−14 (p=0.039)** |
| SmolLM2-1.7B | −8 (n.s.) | +9 (n.s.) | **−18 (p=0.022)** |
| Qwen3-1.7B | +1 (n.s.) | +5 (n.s.) | −10 (p=0.125) |
| SmolLM3-3B | −1 (n.s.) | +3 (n.s.) | +0 (n.s.) |
| Phi-4-mini | +2 (n.s.) | +0 (n.s.) | +4 (n.s.) |

*(refusal n=100 keyword; refusal-judge n=60–100 after dropped chunks; capability
n=50. Three significant effects emerge where the keyword-only pass found none.)*

## the finding: degradation is real, but bidirectional and model-specific

The v1 thesis — "safety degrades **faster** than capability" — is **wrong as a
universal claim.** With both axes measured properly, quantization degradation is
real but splits into **two opposite failure modes**:

- **Knowledge-fragile (thesis inverted):** Qwen3.5-4B (capability −14, p=0.039) and
  SmolLM2 (−18, p=0.022) lose factual accuracy under nf4 while refusal holds. For
  these models, quantization erodes *what they know* faster than *whether they refuse*.
- **Safety-fragile (thesis as stated):** gemma-4-e2b loses refusal (−12, p=0.007 by
  the judge) while capability holds. For gemma, the original direction is right.

So both things happen — just to different models. There is no single direction.

## the keyword scorer was actively misleading

The judge-on-both columns expose the methodological punchline: the keyword scorer
didn't just add noise, it gave **wrong answers** on the safety axis.

- **SmolLM2** — v1's "strongest thesis support" (refusal −8pp) — has the **wrong
  sign**: the judge says refusal *rose* +9pp. The headline support was a scorer
  artifact.
- **gemma** — the one real safety degradation (−12pp, p=0.007) — was **invisible**
  to the keyword scorer (it saw −4pp, n.s.).

A keyword-based safety eval got the wrong answer on 2 of the 3 real effects in this
dataset. That is the strongest argument for judge-based scoring here.

In aggregate over ~2,400 responses the keyword scorer agrees with the judge **85.7%**
of the time — but the disagreements matter: **7.7%** are false safety positives
(keyword "refused", judge "complied"), and **6.0% of responses scored as refusals
actually contained harmful content**. The ~14% per-label error rate is *larger than
any quantization delta the study was trying to detect*, which is a second reason the
keyword refusal deltas were unmeasurable.

## cross-judge robustness (Sonnet vs Opus)

To check that the capability drops aren't an artifact of one judge's grading, all
1,200 responses were re-graded by a second, different Anthropic model (Opus 4.8)
via `code/judge_rescore.py --backend opus`, and compared (`code/cross_judge.py`).
(The Gemini and codex CLIs were tried first but are filesystem-exploring *agents*,
not one-shot endpoints — Gemini read the script instead of answering — so a clean
cross-provider judge wasn't practical; Opus is the independent second model.)

| Model | Sonnet Δ | Opus Δ | agreement | consensus Δ (both agree) |
|-------|---:|---:|:--:|---:|
| Qwen3.5-4B | −14 (p=0.039) | −10 (p=0.125) | 92% | **−13 (p=0.031)** |
| SmolLM2-1.7B | −18 (p=0.022) | −14 (p=0.118) | 88% | **−18 (p=0.039)** |
| Qwen3-1.7B | −10 (p=0.125) | −8 (p=0.125) | 94% | −7 (p=0.250) |
| gemma / SmolLM3 / Phi-4 | ~0 | ~0 | 90–94% | ~0 |

**What this establishes:** the two drops are **not a single-judge hallucination** —
Opus independently reproduces the same direction and similar magnitude (judges agree
on 88–94% of individual labels), and on the high-confidence subset where both judges
agree, both effects stay significant (−13 p=0.031, −18 p=0.039).

**What it doesn't:** neither drop clears p<0.05 under Opus *alone* (both p≈0.12) — at
n=50 each judge is underpowered and the CIs just touch zero. So this is corroboration
of direction and magnitude, not independent confirmation of significance. It rules out
"Sonnet invented it" and sharpens the case for more samples.

## rigor caveats

1. **Multiple comparisons:** 12 judge tests (6 models × 2 axes), 3 significant at
   p<0.05 vs ~0.6 expected by chance — clearly above the noise floor (unlike the
   keyword refusal pass, where 1 hit ≈ 0.9 expected). gemma's p=0.007 nearly clears
   Bonferroni (0.05/12 = 0.0042); the two capability hits do not individually.
2. **Small n** (50 capability, 60–100 refusal-judge). Two judges (Sonnet + Opus)
   now agree on direction/magnitude (see cross-judge section), but n=50 leaves each
   underpowered — suggestive-strong, not airtight.
2b. **Refusal-judge selection bias.** ~13% of refusal items never parse, even on
   retry, and the misses concentrate in the weakest-safety models/levels (e.g.
   SmolLM3 nf4 n=60) — most likely because the judge itself balks at classifying
   the most harmful *compliance* responses. So the judged refusal subset is skewed
   toward the benign cases; the true refusal rates (and any drop) could be larger
   than measured. The effect *directions* hold, but treat the refusal-judge
   magnitudes as conservative.
3. **Multi-seed is now the confirmation path**, aimed at three falsifiable
   predictions: nf4 cuts TruthfulQA ~14–18pp on Qwen3.5-4B / SmolLM2, and cuts
   gemma's refusal ~12pp, with the other models flat.

## what this changes in the v1 writeup

- "Factual recall anomaly" (axis is noise → disregard): superseded.
- Conclusion 1 ("thesis partially supported"): the thesis is **not** supported as a
  general claim. Replace with: *quantization degradation is real but model- and
  axis-specific — some models lose knowledge while refusing fine, one loses refusal
  while knowing fine — and keyword-based refusal scoring is unreliable for measuring it.*
