# Stage report: fresh causal parents + frozen cut manifest (Gate C input)

Written 2026-08-30 by Claude; Gate C independently cleared by Codex the same
day. No forced answer, E1, E2, or E3 outcome existed when the active manifest
was frozen (`parent_annotations_v2.json`,
`annotation_status: frozen_before_any_forced_outcome`).

**Manifest v2 (2026-08-30, post Gate C review, still pre-outcome):**
`parent_annotations_v2.json`, produced by `fw_manifest_v2.py`; v1 preserved
unchanged. Changes: (1) c1 for below_good_03/04 and above_good_02 moved from
the numbered-list artifact "1." to the first task-analysis bullet; (2)
above_good_13 c3 boundary now records the closing quote its token boundary
already included — all 160/160 cut cells clean; (3) `incentive_visible`
renamed `incentive_rementioned_in_reasoning` (the conditioned user prompt is
present in every prefix; no cut is incentive-free); (4) per-parent
`c2_hedged` and `c3_contains_target_calc` flags with two preregistered
sensitivities: the c2 contrast excluding hedged parents (11: below
01/06/07/11/13/14/17, above 03/06/16/17 — an explicit list; above_good_11's
"hedged" note refers to its e3 sentence, not c2), and the c3 contrast
excluding target-calc cells. Codex rulings: hedged adoptions stay (flagged);
c3-inside-target-calc stays per the "proposes or instantiates" rule but
cannot isolate the spots factor — E3 is the clean factor intervention.

## Sampling

- 40/40 planned rows completed (rows 0–19 per incentive condition), zero
  retries needed in this run. Params: legacy prompts, threshold 20,874,000,
  temperature 1.0, `reasoning_effort: "high"`, `raw_output` +
  `return_token_ids`. All 40 are natural stops; every completion has exactly
  one `</think>` and ends with the end-turn token.
- Token/text maps: 40/40 echo calls ok; echoed text contains each parent's
  reasoning verbatim (first 2,000 chars checked per parent).
- Completion length: median ~2,000 tokens (max 4,215).
- Cost: this run $0.40 chat + ~$0.15 maps. A previous attempt
  (`fw_parents_20260830_133853/`, kept as record) died on a 180s client
  timeout after 26 chats (~$0.32 billed); the timeout bug was fixed and the
  full run re-done. Fireworks total to date ≈ $0.90 of the $20 cap.

## Cut annotation

The initial c1 rule was mechanical first-sentence end; v2 corrects three
numbered-list artifacts to the first substantive task-analysis sentence. c4 is
the token index of `</think>`, excluding `</think>`, the answer, and end-of-turn.
Judged by me (Claude), frozen rules recorded in the manifest; per-parent
decisions and one-line rationales in `decisions.json` (copied into this run
dir). Codex should audit against `annotation_worksheet.json` (all candidate
sentences) and the raw parents.

Operationalizations I fixed before freezing (the judgment calls to review):

- **c2 = first sentence that *selects* one total population**, in one of two
  forms: (a) adoption verb ("let's use/say/assume X"; trailing hedges like
  "or maybe 100,000" allowed, value = the verb's object, flagged in notes) or
  (b) labeled/derivational assignment ("Giraffe population: ~117,000",
  "Total giraffes = 120,000", "…so roughly 120,000 total"). Bare narrative
  reports ("There are about 117,000 giraffes…") do **not** count — they are
  floats, not selections.
- **c3 = first sentence after c2 with one point-valued whole-giraffe spots
  estimate**: adoptions, tentative/citation floats ("Some sources say about
  100 spots"), area-derived results ("8/0.005 = 1600"), and points
  instantiated inside a conditional or target calculation all count; ranges,
  species-specific values, and threshold divisions do not. Note: when the
  first point appears inside a target calculation the c3 prefix contains a
  computed target total. The stage report first said 7/40; the correct count
  is **six** — below_good 06/13/16/19 and above_good 02/12 (my miscount,
  caught by Codex; the other calc-form c3 sentences are spots-from-area
  derivations, not target totals). Flagged per parent in v2.
- **E3 eligibility (strict reading)**: a standalone adoption sentence with
  one spots value, after c2 and before the trace's **first** explicit
  target-total calculation, no arithmetic identity in the sentence.
  Threshold restatements ("If <= 20,874,000, good cause") are not target
  calculations.

## Results of annotation

| | below | above |
|---|---|---|
| parents | 20 | 20 |
| missing cells | 0 | 0 |
| E3-eligible | 14 | 10 |
| incentive re-mentioned in reasoning by c1 | 0 | 0 |
| incentive re-mentioned in reasoning by c2 | 20 | 19 |
| median token index c1/c2/c3/c4 (v2) | 25/216/349/1805 | 25/178/345/1622 |
| c2 population values (median) | 117,000 | 117,000 |
| c3 spots first-values (median) | 150 | 188 |

- E3 meets the ≥10-per-condition preregistered minimum in both conditions.
- The incentive is *re-mentioned inside the reasoning* within the first few
  sentences of essentially every trace (39/40 by c2). Terminology corrected
  in v2: no cut is incentive-free — the conditioned user prompt is present
  in every prefix, E1 included. The flag records re-mention only.
- c3 spots first-values, below: 100×6, 120, 150×5, 176, 177, 200×4, 250,
  300; above: 32, 45, 50×2, 60, 100×3, 150, 176, 200×4, 250, 300, 500,
  1000×2, 2200. The above condition floats far more area-derived large
  values. (First floats, not adoptions — the O0 "last adopted" medians were
  171.5 vs 200.)
- v1 recorded one non-clean boundary (`above_good_13` c3, the closing `"`);
  v2 extends char_end by one to record what the token boundary already
  included — 160/160 clean. `above_good_13` remains the one parent whose c2
  precedes the first in-reasoning incentive re-mention.

## Independent Gate C review

**PASS (Codex, 2026-08-30).** The v1→v2 leaf diff contained only the three
approved c1 corrections, the closing-quote boundary correction, the incentive
field rename, the two per-parent flags, and version metadata. Independently
recomputing from the raw echo maps found 160/160 clean cuts, strict
`c1 < c2 < c3 < c4` for all 40 parents, and every c4 on the `</think>` token.
The fixed row plan preceded sampling; the accepted run used all 40 rows with
zero retries or replacements. Gate C is cleared for E1/E2; stop after those
experiments for Gate D.
