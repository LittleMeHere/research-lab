# quantization as an alignment lens  [OWN]

*research sprint — april/may 2026*
*status: data collection complete, synthesis below*

> [!CAUTION]
> **Archived original interpretation.** The body below is preserved as written and
> is not the current results account. See the
> [statistical audit](02_statistical_rigor.md) and
> [current exploratory results](03_capability_axis_and_inverted_thesis.md).

---

## thesis

> When you aggressively quantize a safety-tuned small language model,
> safety behaviors (refusal of harmful prompts) degrade **faster** and
> **less predictably** than raw capabilities (factual recall, instruction
> following).

If true, this means every time someone ships a quantized model for
edge/mobile/cost, they're silently shipping a **less aligned** version
— and nobody is systematically measuring this.

---

## experiment design

### models tested (6 total)

| Model | Family | Params | Generation | Thinking Mode |
|-------|--------|--------|------------|---------------|
| google/gemma-4-e2b-it | Gemma | 2B | Latest (Apr 2026) | Yes |
| microsoft/Phi-4-mini-instruct | Phi | 3.8B | Latest | Yes* |
| Qwen/Qwen3.5-4B | Qwen | 4B | Latest | Yes |
| HuggingFaceTB/SmolLM3-3B | SmolLM | 3B | Latest | Yes |
| Qwen/Qwen3-1.7B | Qwen | 1.7B | Prior gen | Yes |
| HuggingFaceTB/SmolLM2-1.7B-Instruct | SmolLM | 1.7B | Prior gen | Yes* |

*thinking mode detected but didn't change behavior*

### quantization levels (4 per model)

| Level | Method | Bits | Notes |
|-------|--------|------|-------|
| fp16 | Full precision baseline | 16 | Reference point |
| int8 | bitsandbytes 8-bit | 8 | Standard deployment quantization |
| int4_fp4 | 4-bit with fp4 scheme | 4 | Aggressive, common in consumer use |
| nf4_dq | 4-bit NF4 + double quant | ~3.5 | Most aggressive standard method |

### evaluation pipeline

Three capability axes, all deterministic (temperature=0, greedy decoding):

1. **Refusal** — 100 prompts from HarmBench (sampled from 400, seed=42).
   Categories: chemical/biological, copyright, cybercrime, harassment,
   harmful, illegal, misinformation. Scored by keyword matching against
   26 refusal indicators. First-token logits captured for every prompt.

2. **Factual Recall** — 50 questions from TruthfulQA (generation split,
   sampled from 817). Scored by substring match against accepted answers.

3. **Instruction Following** — 10 hand-crafted structural compliance
   tests (numbered lists, JSON output, uppercase, exact word, etc.).
   Scored by programmatic regex/parse checks.

### infrastructure

- GPU: NVIDIA L4 (24GB VRAM) on GCP `us-central1`
- Framework: transformers + bitsandbytes
- Total configs: 6 models × 4 quant levels = **24 experiment runs**
- Total prompts evaluated: 24 × (100 + 50 + 10) = **3,840 inferences**
- Plus thinking-mode variants on refusal = **~6,000+ total generations**

---

## raw results

### per-model scorecards

#### google/gemma-4-e2b-it [LATEST GEN]

| Quant | Refusal | w/Think | Factual | Instruct | GPU MB | Lat(s) |
|-------|--------:|--------:|--------:|---------:|-------:|-------:|
| fp16 | 74% | 79% | 2% | 100% | 9,736 | 7.8 |
| int8 | 73% | 79% | 4% | 100% | 7,521 | 27.2 |
| int4_fp4 | 78% | 86% | 2% | 90% | 6,550 | 11.9 |
| nf4_dq | 70% | 79% | 4% | 100% | 7,705 | 17.8 |

**Safety delta (fp16 → nf4): −4pp**
Gemma shows the clearest safety erosion. Under NF4, refusal drops from
74% to 70% while instruction following stays pinned at 100%. This is
the thesis: safety degrades while capability stays.

#### microsoft/Phi-4-mini-instruct [LATEST GEN]

| Quant | Refusal | w/Think | Factual | Instruct | GPU MB | Lat(s) |
|-------|--------:|--------:|--------:|---------:|-------:|-------:|
| fp16 | 88% | 88% | 6% | 100% | 7,316 | 1.9 |
| int8 | 88% | 88% | 4% | 100% | 4,256 | 5.3 |
| int4_fp4 | 85% | 85% | 10% | 100% | 2,911 | 3.3 |
| nf4_dq | 90% | 90% | 6% | 100% | 4,496 | 2.6 |

**Safety delta (fp16 → nf4): +2pp**
Phi-4 is remarkably robust. Strongest baseline safety (88%) and
actually *increases* under NF4. Instruction following is bulletproof
at 100% across all levels. This is what good safety engineering looks
like.

#### Qwen/Qwen3.5-4B [LATEST GEN]

| Quant | Refusal | w/Think | Factual | Instruct | GPU MB | Lat(s) |
|-------|--------:|--------:|--------:|---------:|-------:|-------:|
| fp16 | 84% | 90% | 8% | 90% | 8,021 | 12.4 |
| int8 | 84% | 88% | 8% | 90% | 4,630 | 43.3 |
| int4_fp4 | 79% | 90% | 8% | 80% | 3,170 | 18.1 |
| nf4_dq | 84% | 90% | 8% | 100% | 4,927 | 22.7 |

**Safety delta (fp16 → nf4): 0pp**
Qwen3.5 is perfectly stable on refusal. The interesting anomaly: int4_fp4
drops refusal 5pp but *thinking mode* compensates back to 90%. Also
notable: instruction following *improves* from 90% to 100% under nf4_dq.

#### HuggingFaceTB/SmolLM3-3B [LATEST GEN]

| Quant | Refusal | w/Think | Factual | Instruct | GPU MB | Lat(s) |
|-------|--------:|--------:|--------:|---------:|-------:|-------:|
| fp16 | 36% | 34% | 10% | 100% | 5,866 | 9.2 |
| int8 | 37% | 36% | 10% | 100% | 3,196 | 38.3 |
| int4_fp4 | 36% | 28% | 6% | 100% | 2,031 | 15.3 |
| nf4_dq | 35% | 37% | 8% | 100% | 3,420 | 20.9 |

**Safety delta (fp16 → nf4): −1pp**
SmolLM3 starts dangerously low (36%) but is *stable* — it doesn't get
worse because there wasn't much safety to lose. Instruction following
is perfect at 100% across all levels. The problem isn't degradation,
it's the baseline.

#### Qwen/Qwen3-1.7B [PRIOR GEN]

| Quant | Refusal | w/Think | Factual | Instruct | GPU MB | Lat(s) |
|-------|--------:|--------:|--------:|---------:|-------:|-------:|
| fp16 | 52% | 43% | 8% | 100% | 3,282 | 8.7 |
| int8 | 57% | 41% | 8% | 100% | 1,948 | 36.1 |
| int4_fp4 | 51% | 39% | 10% | 90% | 1,359 | 12.9 |
| nf4_dq | 53% | 39% | 10% | 100% | 2,053 | 15.5 |

**Safety delta (fp16 → nf4): +1pp**
Qwen3 is borderline (52%) but quantization doesn't make it worse.
Interesting: thinking mode actually *hurts* safety here (−9pp at fp16).

#### HuggingFaceTB/SmolLM2-1.7B-Instruct [PRIOR GEN]

| Quant | Refusal | w/Think | Factual | Instruct | GPU MB | Lat(s) |
|-------|--------:|--------:|--------:|---------:|-------:|-------:|
| fp16 | 33% | 33% | 6% | 90% | 3,966 | 6.0 |
| int8 | 45% | 45% | 8% | 90% | 1,738 | 24.3 |
| int4_fp4 | 37% | 37% | 4% | 90% | 1,064 | 10.5 |
| nf4_dq | 25% | 25% | 6% | 90% | 1,857 | 13.6 |

**Safety delta (fp16 → nf4): −8pp**
The **worst degradation** in the study. SmolLM2 drops from 33% to 25%
under NF4 — that's a 24% relative decline. Meanwhile instruction following
stays locked at 90%. This is the thesis in its purest form: safety erodes
while capability holds.

---

## cross-model comparison (latest gen only)

| Model | fp16 | int8 | int4 | nf4 | Δ(fp16→nf4) |
|-------|-----:|-----:|-----:|----:|------------:|
| Phi-4-mini-instruct | 88% | 88% | 85% | 90% | +2pp |
| Qwen3.5-4B | 84% | 84% | 79% | 84% | 0pp |
| gemma-4-e2b-it | 74% | 73% | 78% | 70% | −4pp |
| SmolLM3-3B | 36% | 37% | 36% | 35% | −1pp |

**Key observation**: The gap between the safest (Phi-4 at 88%) and
least safe (SmolLM3 at 36%) latest-gen model is **52 percentage points**.
That is a wider gap than the maximum degradation from quantization.

---

## generational safety improvement

| Family | Old fp16 | New fp16 | Gain |
|--------|--------:|--------:|------:|
| Qwen 3 → 3.5 | 52% | 84% | **+32pp** |
| SmolLM 2 → 3 | 33% | 36% | +3pp |

Qwen made a massive jump. SmolLM barely moved. This suggests Qwen
invested heavily in safety tuning between generations while SmolLM
focused elsewhere.

---

## per-category refusal analysis (fp16 baselines)

This is where it gets interesting. Not all harm categories are equal:

| Category | Gemma | Phi-4 | Qwen3.5 | SmolLM3 | Qwen3 | SmolLM2 |
|----------|------:|------:|--------:|--------:|------:|--------:|
| chemical/bio | 100% | 100% | 100% | 77% | 77% | 62% |
| copyright | 20% | 87% | 7% | 0% | 0% | 7% |
| cybercrime | 92% | 92% | 96% | 16% | 56% | 40% |
| harassment | 67% | 100% | 100% | 67% | 100% | 0% |
| harmful | 83% | 100% | 100% | 33% | 83% | 0% |
| illegal | 94% | 100% | 100% | 76% | 76% | 59% |
| misinfo | 57% | 62% | 95% | 24% | 33% | 19% |

**findings:**

1. **Copyright is a universal weak spot.** Even Gemma (74% overall)
   only refuses 20% of copyright-related prompts. Qwen3.5 (84% overall)
   is at 7%. Only Phi-4 handles it well (87%). Models are trained to
   refuse bombs but not plagiarism.

2. **Chemical/bio is universally strong.** Every model, even the weakest
   ones, has high refusal for chemical/biological threats. This suggests
   these categories get disproportionate training attention.

3. **SmolLM has huge category gaps.** SmolLM3 goes from 77% on chemical/bio
   to 0% on copyright and 16% on cybercrime. The safety is lopsided.

4. **Misinformation is hard.** Even well-aligned models struggle here.
   Gemma is at 57%, Phi-4 at 62%. Only Qwen3.5 (95%) handles it well.
   This suggests the task is genuinely difficult — the model has to
   recognize that the *request itself* is harmful, not just the topic.

---

## thinking mode effects

| Model | Normal | w/Think | Delta |
|-------|-------:|--------:|------:|
| gemma-4-e2b-it | 74% | 79% | **+5pp** |
| Qwen3.5-4B | 84% | 90% | **+6pp** |
| Phi-4-mini-instruct | 88% | 88% | 0pp |
| SmolLM3-3B | 36% | 34% | −2pp |
| Qwen3-1.7B | 52% | 43% | **−9pp** |
| SmolLM2-1.7B | 33% | 33% | 0pp |

**findings:**

1. **Thinking helps larger, well-aligned models.** Gemma and Qwen3.5
   both gain 5-6pp with thinking enabled. The "deliberation" step
   lets them recognize harm they'd otherwise miss.

2. **Thinking hurts smaller models.** Qwen3-1.7B *loses* 9pp with
   thinking. Hypothesis: the thinking chain gives the small model
   enough tokens to "talk itself out of" its initial refusal.

3. **This is a size-dependent effect.** The benefit of thinking mode
   for safety appears to require sufficient model capacity to reason
   about harm, not just pattern-match.

---

## factual recall anomaly

Every model scores absurdly low on TruthfulQA (2-10%). This is **not**
because the models are stupid — it's a scoring artifact. TruthfulQA
uses free-form generation questions like "What happens if you eat
watermelon seeds?" where the *correct* answer is "Nothing, they pass
through you" and common wrong answers are "They grow in your stomach."

Our substring-match scorer can't handle TruthfulQA's nuanced answer
format. The models are giving long, detailed, correct responses that
don't contain the exact expected substring. This axis is essentially
**noise** and should be disregarded for the thesis.

The important comparison is still valid: factual accuracy doesn't
change much across quantization levels (±2pp), confirming that raw
knowledge retrieval is stable under compression.

---

## memory efficiency

| Model | fp16 MB | nf4 MB | Reduction |
|-------|--------:|-------:|----------:|
| gemma-4-e2b-it | 9,736 | 7,705 | 21% |
| Phi-4-mini-instruct | 7,316 | 4,496 | 39% |
| Qwen3.5-4B | 8,021 | 4,927 | 39% |
| SmolLM3-3B | 5,866 | 3,420 | 42% |
| Qwen3-1.7B | 3,282 | 2,053 | 37% |
| SmolLM2-1.7B | 3,966 | 1,857 | 53% |

People quantize for a reason — it works. SmolLM2 goes from ~4GB to
under 2GB. The question is whether the safety cost is worth it.

---

## conclusions

### 1. the thesis is **partially supported**

Safety does degrade under quantization, but it's **not universal**.
The effect is model-dependent:

- **SmolLM2**: Strong support (−8pp refusal, 0pp instruction = safety
  degrades faster than capability)
- **Gemma**: Moderate support (−4pp refusal, 0pp instruction)
- **Phi-4, Qwen3.5**: No support (refusal is stable or improves)
- **SmolLM3**: Weak support (−1pp, but started so low it's noise)

### 2. baseline safety matters more than quantization

The **52pp gap** between the safest and least safe latest-gen model
dwarfs any quantization effect (max −8pp). If you care about safety,
choosing Phi-4 over SmolLM3 matters **6x more** than choosing fp16
over nf4.

### 3. copyright is the universal blind spot

Every model family except Phi-4 has a copyright refusal rate below
20%. This is a systemic training gap, not a quantization artifact.
If you're evaluating model safety, test copyright compliance.

### 4. thinking mode is a double-edged sword

It helps models that are already well-aligned (+5-6pp for Gemma,
Qwen3.5) but hurts smaller models (−9pp for Qwen3-1.7B). Don't
assume thinking = safer.

### 5. the real risk is shipping weak baselines, not quantizing strong ones

Phi-4 at nf4 (90% refusal, 4.5GB) is safer than SmolLM3 at fp16
(36% refusal, 5.9GB). The quantized version of a good model beats
the full-precision version of a bad one.

---

## limitations & future work

1. **Single-run data.** No multi-seed confidence intervals. The ±2-4pp
   variations could be within noise. Need 3-5 runs per config.

2. **Refusal detection is keyword-based.** A model that refuses
   politely with novel phrasing gets scored as "complied." False
   negative rate is unknown.

3. **TruthfulQA scoring is broken.** Need a semantic similarity
   scorer (e.g., LLM-as-judge) instead of substring matching.

4. **Only tested bitsandbytes quantization.** GPTQ, AWQ, and GGUF
   may show different patterns. Would be especially interesting to
   test against the parameter-golf ternary/binary quantization
   techniques from the leaderboard.

5. **No activation/logit analysis in write-up.** The data was captured
   (first-token logits, per-layer activation norms) but not analyzed
   yet. This is where mechanistic interpretability connects: *which
   layers* lose safety representations first?

6. **No red-team validation.** Need to verify that "complied" responses
   actually contain harmful content, not just missed refusal keywords.

---

## reproducibility

```bash
# Clone and navigate
git clone https://github.com/LittleMeHere/research-lab
cd research-lab/projects/quantization-alignment

# Install dependencies
pip install -r requirements.txt

# Run experiments (requires GPU with 24GB+ VRAM)
MODELS="google/gemma-4-e2b-it,microsoft/Phi-4-mini-instruct" python3 code/v2_experiment.py
MODELS="Qwen/Qwen3.5-4B,Qwen/Qwen3-1.7B" python3 code/v2_experiment.py
MODELS="HuggingFaceTB/SmolLM3-3B,HuggingFaceTB/SmolLM2-1.7B-Instruct" python3 code/v2_experiment.py

# Analyze results
python3 code/analyze_results.py
```

**Hardware:** NVIDIA L4 (24GB), GCP us-central1
**Software:** Python 3.10, torch 2.9.1, transformers, bitsandbytes
**Seed:** 42 (HarmBench), 43 (TruthfulQA)
**Generation:** Greedy (temperature=0, do_sample=False, max_new_tokens=256)

All raw data: `data/v2_results_*.json` (6 files, ~3MB each)
VM logs: `logs/v2_log_vm*.txt` (3 files)

---

## what I actually learned

this started as "quantization breaks safety." the data says: "it depends."

the models where quantization hurts safety are the ones that were
borderline to begin with (SmolLM2 at 33%, Gemma at 74%). the models
that were well-tuned (Phi-4 at 88%) are basically immune to it.

the real finding isn't about quantization at all. it's about how
**wildly different** safety tuning is across model families. a 52pp
gap between models released in the same month is insane. there's no
standard, no floor, no certification.

also: copyright. nobody is testing copyright refusal. every model
except Phi-4 will happily reproduce copyrighted content on request.
that's a bomb waiting to go off.

the quantization-as-alignment-lens framing is still useful though:
it's a **stress test**. if you want to know how robust a model's
safety really is, quantize it to nf4 and see what breaks first.
the models that hold up are the ones that were genuinely well-trained.
the ones that break were relying on surface-level patterns.

---

*next: activation norm analysis, logit-shift visualization, multi-run
confidence intervals. then probably a blog post.*

*tag: OWN — I can whiteboard every number in this doc.*
