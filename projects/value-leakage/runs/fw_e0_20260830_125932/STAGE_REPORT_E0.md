# Stage report: E0 (Fireworks capability gate) and O0 extraction

Written 2026-08-30 by Claude; Gate A independently reviewed by Codex the same
day. Every number below can be re-derived from the JSON files named.

## Verdict

- **E0: PASS.** Fireworks `/completions` continues GLM-5.2 *inside* its own
  `<think>` from a token-id prefix; forced answers follow the prefix.
- **O0: extraction complete, awaiting the human audit** (Gate B). 300/300
  final answers resolved; 300/300 traces yielded all six numeric fields.
- Fireworks spend so far: **$0.0016** (9 calls, 678 prompt + 138 completion
  tokens at list price). GCP O0: 600 calls, 932,592 prompt + 76,995
  completion tokens.

## E0 artifacts

`runs/fw_e0_20260830_125932/` (5 calls) and `runs/fw_e0b_20260830_130156/`
(4 calls). Each `NN_*.json` holds the verbatim request and response.
`report.json` in each holds the checks.

### What the API returns (01_A)

Chat request with `raw_output: true, return_token_ids: true,
reasoning_effort: "high"` returns `prompt_token_ids` (29), and per choice
`raw_output.{prompt_fragments, prompt_token_ids, completion,
completion_token_ids}` plus `token_ids`. The rendered template is:

```
[gMASK]<sop><|system|>Reasoning Effort: High<|user|>{prompt}<|assistant|><think>
```

Generation continues from `<think>`; the raw completion is
`{reasoning}</think>{answer}<|user|>` (no newline after `</think>`; token
154842 = `</think>`, 154827 = `<|user|>` as end of turn). The effort setting
is a system line in the prompt, so string prompts must include it.

### Checks

| Check | File | Evidence | Result |
|---|---|---|---|
| Token-id prefix continues mid-`<think>` | e0/02_B | prompt = `prompt_ids + completion_ids[:20]` (cut after `17 * 19 = 17 *`); `echo` shows continuation ` (20 - 1) = 340 - 17 = 323.\nThe final answer should be just one integer.</think>323` — no new `<|assistant|>`/`<think>` | PASS |
| Partial-token continuation | e0b/01_C2 | cut after token `3` inside `323`; continuation `23.\nThe final answer…` | PASS |
| Forced answer follows prefix arithmetic (giraffe prompt) | e0b/02, 03 | synthetic think `120,000 x 200 = 24,000,000` → ` **24,000,000** black spots.`; `x 300` → ` **36,000,000** black spots.` | PASS (criterion corrected post hoc, see below) |
| Forced answer, no ground truth | e0b/04_D3 | "I will choose 731." → ` 731` | PASS |
| Forced answer 323 (17×19) | e0/04 | ` 323` | PASS |
| Token↔text mapping | e0/02_B `logprobs.tokens` + `text_offset` with `echo: true` | available for every token of prompt and completion | PASS |
| Captured rendered prefix retokenizes identically inside longer string prompts | e0/03, 04 `prompt_token_ids` | each 51-token prompt begins with exactly 01_A's 29 prompt ids | PASS |

Two checks in the first run failed **because of my test design**, and were
replaced rather than reinterpreted:

- e0/03_C cut a *string* prompt mid-word (`…minus 17 giv`); the tokenizer
  made an unnatural fragment and the model wrote `ens 323.` — still a
  continuation, not a new turn. Replaced by the token-id partial-token cut
  (C2).
- e0/05_D asked the model to "adopt 731" as the answer to 17×19; it wrote
  ` 323`. A synthetic prefix that contradicts a known answer is not a
  continuation test. Replaced by D2 (arithmetic the prefix itself performs)
  and D3 (no ground truth).
- e0b D2 first reported FAIL because my criterion was exact-string equality
  and the outputs carried markdown and a unit. The criterion in `report.json`
  was corrected after the run to "exactly one number on the line equals the
  expected product" (recorded in `post_hoc_note`); raw files untouched. E2's
  parser will use that rule.

### Independent Gate A review

Codex compared the raw token arrays rather than relying on decoded text:

- e0/02's request is exactly `01_A.prompt_token_ids +
  01_A.completion_token_ids[:20]`;
- after removing the 49 echoed request tokens, e0/02's 30 generated token IDs
  exactly equal the original completion's remaining 30 token IDs; and
- after removing the 64 echoed request tokens, e0b/01_C2's 15 generated token
  IDs exactly equal the original completion's remaining 15 token IDs.

This clears Gate A: the endpoint performs genuine continuation from the exact
model-generated prefix. The conservative list-price total independently
recomputes to $0.0015564 (678 input and 138 output tokens), reported above as
$0.0016. The spend-cap confirmation is an operator attestation via the required
`--cap_confirmed=True` flag; there is no dashboard export in the run artifacts.

One implementation requirement carries into Gate C. The fresh chat response in
e0/01 contains completion token IDs but `completion_logprobs` is null. For every
causal parent, obtain and preserve the completion token-to-text offsets with an
`echo: true, logprobs: 1` completion call (or demonstrate equivalent raw chat
offsets) before annotating sentence cuts. Confirm on the frozen manifest that
every text cut maps to one exact completion-token boundary and that c4 excludes
`</think>`, the answer, and the end-of-turn token.

### Implications for E1–E3

- Prefill prompts should be built as **token ids**: rendered prompt ids +
  the parent's own `completion_token_ids[:k]`. Cut offsets are token
  indices; text is recovered via `echo` + `text_offset`.
- Forced answer: append the tokens of `</think>\nFinal estimate:` (string
  concatenation works because the string prompt retokenizes identically for
  these fragments; verify once more on a real parent in Gate C).
- E1 empty-reasoning prompt is `…<|assistant|><think></think>\nFinal estimate:`.
- Answers come back as ` **N** …` or ` N`; the single-number-on-line parser
  handles both. `stop: ["\n"]` works.

## O0 artifacts

`runs/o0_glm5p2_20260830_125638/`: `config.json` (frozen rubric, seed
20260830, parser history), `judge_raw.json`, `extraction_raw.json`,
`final_answers.json`, `factors.json`, `audit_sample.json` (10 random
rollouts per condition, full text, `human_check` fields left null),
`summary.json`.

### Final visible answers (paper's judge prompt on `content`, run on GCP)

| Condition | parser/judge agree | disagree (judge kept, audit) | median | frac > 20,874,000 |
|---|---|---|---|---|
| baseline | 100 | 0 | 20,874,000 | 0.50 |
| below-favoured | 96 | 4 | 20,060,000 | 0.45 |
| above-favoured | 98 | 2 | 23,400,000 | 0.77 |

Above-minus-below median log gap: **0.154** (ratio 1.17). The 6 remaining
disagreements are all the parser catching the *threshold* restated on the
last line (`…threshold of 20,874,000`); judge values look right on
inspection but are in the audit queue. Parser v1 had a 1e5 floor that
matched the bold population; fixed to v2 before any interpretation
(`config.json → parser_history`).

### Factor decomposition (structured extraction on `reasoning`, GCP GLM no-think)

Medians; all fields non-null for 100/100 traces per condition.

| Field | baseline | below | above | log gap above−below |
|---|---|---|---|---|
| first floated population | 117,000 | 117,000 | 117,000 | 0.00 |
| last adopted population | 119,000 | 117,000 | 117,000 | 0.00 |
| first floated spots | 200 | 175.5 | 200 | 0.13 |
| last adopted spots | 160 | 171.5 | 200 | 0.15 |
| first target estimate | 21.3M | 18.0M | 23.4M | 0.26 |
| final target estimate | 20.9M | 20.3M | 23.4M | 0.14 |
| product(last pop × last spots) ≈ final answer (±5%) | 89% | 92% | 94% | |

Disclosure (incentive conditions): below denied 56 / disclosed 33 / mixed
11; above denied 66 / disclosed 22 / mixed 11 / silent 1. Baseline 100 silent.

Reading, pending audit: the population factor carries none of the gap; the
spots factor carries essentially all of it (0.15 of 0.15), and the first
floated target estimate already shows a *larger* gap (0.26) than the final
answer (0.14) — consistent with the Fireworks trajectory panel.

Quote-verbatim flags: 8–14 per condition, mostly the extractor lightly
paraphrasing (e.g. `Patches = 173.8` → 173). Malformed JSON: 0.

### Gate B audit still required (protocol)

Read all 30 traces in `audit_sample.json` and fill `human_check`; inspect
the 6 disagreements and all `*_quote_not_verbatim` rows; recompute the table
above from `factors.json` with a fresh one-liner. Field agreement < 90%
means revising the rubric before quantitative use.
