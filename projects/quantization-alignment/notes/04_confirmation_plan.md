# Held-out confirmation plan

## Purpose

The v2 runs were exploratory. They produced many comparisons, used 50
TruthfulQA questions, and did not obtain complete refusal labels. The results
identify four effects worth retesting; they do not establish a general effect of
quantization on safety or capability.

This phase asks whether those four effects recur on unused prompts. It does not
include activation analysis, steering, or a search for additional effects.

## Terms

- **Held-out** means the prompt was not used in v2.
- **Paired** means the fp16 and quantized model answer the same prompt; each test
  compares those two answers.
- **fp16** stores model weights as 16-bit floating-point numbers and serves as the
  control. **NF4** and **FP4** are two 4-bit storage formats.
- An **endpoint** is the outcome being measured: TruthfulQA accuracy or materially
  harmful compliance.
- A **judge** is a separate language model that assigns a binary endpoint label.
  **Adjudication** resolves cases in which the two judges disagree.
- **Holm correction** adjusts the four p-values together so that testing four
  predictions does not inflate the chance of a false positive.

## Primary hypotheses

All deltas are quantized minus fp16, in percentage points.

1. Qwen3.5-4B NF4 lowers TruthfulQA accuracy. The exploratory estimates were
   −14 points under Sonnet and −10 under Opus.
2. SmolLM2-1.7B NF4 lowers TruthfulQA accuracy. The exploratory estimates were
   −18 points under Sonnet and −14 under Opus.
3. Gemma-4-e2b NF4 raises materially harmful compliance. The exploratory estimate
   was +8 points among 88 prompts with labels for both conditions.
4. Qwen3-1.7B FP4 lowers TruthfulQA accuracy. The exploratory estimates were
   −24 points under Sonnet and −20 under Opus. This signal was found in the later
   audit of all v2 quantization levels.

Qwen3-1.7B NF4 versus fp16 is a descriptive format comparison, not a fifth
primary hypothesis.

## Data and generation

The runner uses pinned source and model revisions recorded in
`code/confirmation_spec.py`. It excludes normalized v2 prompt texts, removes
duplicate source prompts, and sorts the remaining rows by normalized prompt and
category. The resulting sets contain:

- 294 HarmBench prompts;
- 767 TruthfulQA validation questions, each stored with the pinned correct and
  incorrect reference answers.

The HarmBench identity hash covers prompt and category. The TruthfulQA identity
hash also covers both reference-answer lists. A changed question, order, category,
or scoring key therefore produces a different hash.

Generation is greedy (`do_sample=False`), limited to 256 new tokens, and uses seed
42. Thinking mode is disabled. Only the endpoint needed for a hypothesis is run:

| Model | Configurations | Benchmark | Generations |
|---|---|---|---:|
| Gemma-4-e2b | fp16, NF4 | HarmBench | 588 |
| Qwen3.5-4B | fp16, NF4 | TruthfulQA | 1,534 |
| SmolLM2-1.7B | fp16, NF4 | TruthfulQA | 1,534 |
| Qwen3-1.7B | fp16, FP4, NF4 | TruthfulQA | 2,301 |
| **Total** | | | **5,957** |

This focused design avoids 3,592 generations that would not contribute to a
declared test or the Qwen3 format comparison.

The HarmBench loader retains the v2 input format: it uses each row's behavior
text without the separate `ContextString` field. The held-out split therefore
includes 100 contextual behaviors that may be underspecified when read alone.
This limitation is reported with the confirmation results.

## Treatment and environment checks

Every model load requests fp16 for unquantized parameters, including 4-bit
conditions. For each named linear module, the runner records its class, weight
storage type, 4-bit format, computation type, and double-quantization state. It
rejects a condition if any floating parameter or 4-bit computation is not fp16,
if the named format is absent, or if module coverage differs between Qwen3's FP4
and NF4 conditions.

Transformers 5.14.0 is required because it includes the relevant composite-model
dtype fix. A one-prompt-per-benchmark smoke test exercises all nine model/config
cells without using held-out prompts. Its environment record includes package
versions, CUDA and GPU details, `pip freeze`, and module fingerprints. Full
generation requires the same record. Results use stable filenames and save after
each response, so an interrupted run resumes only after verifying its existing
metadata and prompt prefix.

The host also needs a C/C++ compiler and development headers matching its Python
installation because the pinned PyTorch/Triton stack compiles a CUDA helper on
first use.

## Labels

The primary judges are Claude Sonnet 5 through the Claude CLI and GPT-5.6 Sol
through the Codex CLI. They run independently, receive responses in different
shuffled orders, and do not receive model or quantization identities. The full
response and, for TruthfulQA, the pinned reference answers are included. Codex
uses a strict JSON schema. Both paths reject invalid types, retry omitted items
individually, and must label every response.

Two judges are used because a single grader can impose one model's systematic
interpretation on every endpoint. Different model families reduce shared grading
tendencies; they do not guarantee independence, so each judge's result is also
reported.

Agreements become the primary label. Disagreements go to Claude Opus 5 or a
named human resolver in a salt-shuffled packet that omits model and quantization
identities.
Opaque item IDs use a random private salt; only the salt hash accompanies the
packet. The packet, resolution file, resolver identity/interface, and hashes are
retained. Partial resolution files do not produce adjudicated output, and existing
packets or final labels are never overwritten.

The adjudicator writes the packet and a resolution template but does not invoke
the resolver. The packet is reviewed separately, the template records the exact
resolver and interface, and the adjudicator is rerun with the completed resolution
file. The private salt remains local and is excluded from version control.

Before held-out generation, this complete judging and adjudication path runs on
the relevant v2 model/configuration cells. The resulting lock requires 100%
coverage from both judges and the adjudicated labels. This step tests the pipeline;
its labels are not confirmation data.

## Analysis and decision rule

Each primary comparison uses a two-sided exact McNemar test, which tests whether
paired binary outcomes switch in the two directions equally often. The effect is
the quantized rate minus the fp16 rate. Its 95% interval is a paired percentile
bootstrap with 10,000 resamples and seed 20260822. Holm correction covers exactly
the four primary tests.

An effect advances to a mechanistic follow-up only when all of the following hold:

1. the effect has the predicted direction;
2. the Holm-adjusted p-value is below 0.05;
3. the paired 95% interval excludes zero;
4. the absolute effect is at least 8 percentage points; and
5. both cells have complete labels.

The 8-point rule is a follow-up threshold, not the null hypothesis. H3 uses every
available held-out HarmBench prompt, but its exploratory estimate lies at that
threshold. Even under an optimistic extrapolation of the v2 transition pattern,
an exactly 8-point true effect has only about a one-half chance of producing an
observed estimate at or above 8 points. The report therefore includes 4-, 6-, and
8-point sensitivity rows and states that a null result does not exclude smaller
safety effects. The threshold is not lowered after results are seen.

All four results, both judges' versions, direction reversals, nulls, discordant
pair counts, and the Qwen3 NF4 format comparison are reported. Missing or stale
inputs make the analysis invalid; a complete result that does not meet the rule is
reported as `STOP`, not as evidence of no effect.

## Workflow integrity

`code/confirmation_spec.py` is the single executable source for models, revisions,
splits, judges, hypotheses, and statistical constants. The explanatory plan does
not duplicate revision hashes. Each run records one manifest containing the Git
commit and hashes of the plan, specification, runner, judging, adjudication,
analysis, and environment-input files. Full generation requires those files to be
committed and unchanged. The file hashes define the protocol identity; the Git
commit is provenance and may differ after result-only commits. Analysis verifies
the protocol identity across all outputs.

The sequence is:

1. complete independent design and code review;
2. commit the reviewed package;
3. run all model/configuration smoke cells and write the environment record;
4. run and verify the judge/adjudication preflight on v2 responses;
5. generate the held-out responses;
6. run both judges, resolve every disagreement, and run the primary analysis.

## Raw safety data

Generated harmful responses may contain code patterns detected by endpoint
protection. Treat them as inert research data: store and inspect them in an
isolated environment, never execute response text, and move only aggregate or
reviewed artifacts to general-purpose systems.
