# Stage report: E1 + E2 (Gate D input)

Written 2026-08-30 by Claude. Manifest: `parent_annotations_v2.json`. All
1,101 Fireworks requests and raw responses are in this directory (300 E1,
800 E2, one suffix verification); `results_resolved.json`
has every forced line with its parser/judge value and source;
`summary_v2.json` is the authoritative summary (see "n bug" below for why
`summary.json` from the first pass of this run uses parser-only values).

Codex's independent Gate D ruling is in `GATE_D_REVIEW.md`. Gate D passes,
with the interpretation narrowed as recorded there.

## Headline

**By c3 (first floated spots value), the forced-answer gap's point estimate is
already the size of the endpoint gap, and the endpoint reproduces the parents'
own answers almost perfectly. The early cuts and the empty-reasoning control
are a different, wildly dispersed regime that cannot localize transfer.**

| Point | gap (log, above−below) | 95% CI (parent bootstrap) | ratio | red flag |
|---|---|---|---|---|
| E1 empty reasoning (pooled, n=100/cond) | **−1.10** | — | 0.33× | regime incomparable |
| c1 first sentence | 2.98 | [−1.22, 3.88] | 19.7× | **yes** (gap ≫ parents' final gap) |
| c2 population selected | 0.79 | [0.29, 3.81] | 2.20× | **yes** |
| c3 spots floated | 0.32 | [−0.06, 0.60] | 1.38× | no |
| c4 full reasoning | 0.344 | [0.005, 0.52] | 1.41× | no |
| parents' own final answers (judged) | 0.344 | — | 1.41× | reference |

- **c4 validity: 40/40 evaluable, 1 failure (2.5%)** — far under the 20%
  block threshold. 35/40 parents reproduce their own visible final
  **exactly**; 39/40 are within 10%, and 40/40 stay on the same side of the
  threshold. The one failure,
  `above_good_02`, lands 12% above its own final (64.35M vs 57.33M), same
  side of threshold.
- **Sensitivities** (preregistered): c2 excluding hedged parents 0.83
  [0.25, 4.13]; c3 excluding target-calc parents 0.29 [−0.45, 0.54].
- **Attrition after judge resolution: ≤5% per condition × cut, symmetric**
  (worst c3: 4%/5%). 103 parser-rejected lines were sent to the GLM judge
  (GCP, thinking off, paper-verbatim prompt); 97 resolved. Of 15 still
  unresolved overall, 13 are E2 rows. Fifteen of 800 E2 outputs emit a stray
  `</think>`, almost all still carrying a final value. One c3 parent has only
  2/5 resolved replicates; requiring at least three changes the c3 gap from
  0.321 to 0.354.
- The stored 2,000-draw c4 CI is just above zero. A reviewer sensitivity with
  500,000 draws is approximately [−0.01, 0.53], so its lower endpoint should
  not be used as a significance threshold. A paired c3-minus-c4 bootstrap is
  also broad: approximately [−0.305, 0.301].

## Reading (for Gate D discussion, not a final claim)

1. **E1 does not show the clean H-early signature, but cannot reject early
   prompt influence.** With no reasoning text at all (E1), the model's
   forced estimates explode to the billions (medians 4.0B / 1.04B / 0.35B;
   84–95% above threshold in every condition) and the below/above gap is
   *negative*. The empty-prefix probe therefore fails to recover the natural
   scale or direction. Its pathological scale prevents reading that failure
   as evidence that the conditioned prompt had no early effect.
2. **c3 is the first interpretable boundary whose point-estimate gap is
   endpoint-sized.** Once the first spots-per-giraffe value is on the page,
   forcing an answer gives a 1.38× above/below ratio versus 1.41× at c4. The
   c3-minus-c4 uncertainty is broad, and c3 is a cumulative prefix under the
   original conditioned prompt. E2 shows recoverability by c3; it does not
   establish that the spots statement caused the gap.
3. **c1/c2 numbers are not interpretable as transfer** (preregistered red
   flag: gap exceeds the parents' final gap). Parent-median dispersion at c1
   spans 15M–41B (IQR ≈ 3.4 log units vs 0.35–0.58 at c3/c4): with only a
   sentence or a population in context, forced answers are near-E1-regime
   noise, and a 20× "gap" on n=20 medians is dispersion, not signal.
4. E3 (paired coherent factor edit) is the confirmatory causal test of the
   spots factor and is authorized by the Gate D pass. This E2 result is not a
   neutral-transplant or text-alone sufficiency test; its claim must retain
   "under the original conditioned prompt."

## The n bug (why this directory replaces run 151944)

Fireworks `/completions` with `n>1` conditions **only choice 0** on the
prompt; the other samples are prompt-free garbage (93% vs ~1% coherent in the
invalidated first run `fw_e1e2_20260830_151944/`, minimal repro in
`runs/fw_n_bug_probe_20260830/` for both token-id and string prompts). This
run therefore issues every sample as an independent n=1 request (300 E1 +
800 E2 + suffix verification). `summary.json` here was produced by the
parser-only pass; `summary_v2.json` (post judge resolution) supersedes it.

## Costs

This run $1.02; Fireworks total to date **$2.15 of the $20 cap** (e0 $0.002,
parents+maps $0.53, aborted sampling $0.32, invalidated n-bug run $0.25,
probes $0.02). GCP judge calls: 143 (resolution + parent finals), negligible.

## For Codex (Gate D)

- Recompute the curve from `results_resolved.json` + `parent_rows_resolved.json`.
- Spot-check judge resolutions in `resolve_judge_raw.json` (the parser-reject
  classes are: full multiplication shown, `</think>`+value, `**N** (X
  billion)` restatements).
- Check the c4 exact-reproduction claim against `parent_finals_judged.json`.
- Rule on interpretation 1–3 and whether E3 proceeds.
