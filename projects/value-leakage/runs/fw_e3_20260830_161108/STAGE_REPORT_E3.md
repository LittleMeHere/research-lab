# Stage report: E3 paired coherent factor edit — v3.1 (Gate E input, fourth pass)

Written 2026-08-30/31 by Claude, rewritten clean after three Gate E reviews
and cleared by Codex on the fourth pass;
the superseded v1/v2 report texts live in git (6875cb1, 1ade0ca-era files).
Authoritative artifacts in this directory: `resolution_map.json` (visible
answers), `e3_annotations_v3.json` (v3.1 destination audit),
`summary_v3.json` (primary effects), `e3_overrides.json` (the two manual
missing decisions), `GATE_E_REVIEW.md` (all four reviews and the final ruling).

## Design and validity (accepted at earlier passes)

144 requests: 24 frozen eligible parents (14 below / 10 above) × {150, 300}
× 3 independent continuations; string prompts with per-request token-prefix
verification (only `below_good_15` drifts, flagged; exclusion sensitivity
0.080 / 0.155). All 144 request bodies reconstruct exactly from frozen
inputs. 0 restarts, 0 confusion, 6 in-think length-caps, 2 answer-channel
truncations. Cost $1.42; Fireworks total ≈ $3.6 of $20.

## Answer resolution (accepted at pass two)

136 finals via deterministic-v3/frozen-judge agreement, 2 answer-channel
truncations manually confirmed missing (`e3_overrides.json`), 6 completions
with no answer channel. Three v1 parser errors corrected (threshold captured
in place of the answer). Near-threshold finals: 4; exactly at threshold: 0.

## Primary outcome (accepted at pass two; unchanged)

Within-parent median high(300)−low(150) log difference: below **+0.111**
[−0.022, 0.362], above **+0.155** [−0.008, 0.358], pooled **+0.130**
[−0.008, 0.279]; mechanical retention would give +0.693. Doubling the
adopted factor changes the median answer by ~14%, far less than doubling:
the stated factor is a **weak, revisable commitment**. The above−below
interaction is 0.043 [−0.274, 0.318]: incentive-specific repair is
unresolved. Substitution does not test necessity, so the sentence is not
shown to be non-causal.

## Destination audit v3.1 (`e3_annotations_v3.json`)

Scope, stated honestly: an **answer-justification audit** — the destination
factor is read from each answer's own justification, with the reasoning
channel consulted only for the two rows whose justification is truncated
(`below_good_04_low_r1` → effective 1,368 at its stated 117,000;
`below_good_03_high_r1` → 279). It is not a sentence-by-sentence audit of
the reasoning channel; restart/confused are per-row attestations from the
manual read (none observed), not automated detections. Mixture rows divide
by the population **stated in the answer** (7 corrected at this pass;
stated-total-vs-subtotal inconsistencies flagged in notes). Every resolved
row now carries a verbatim evidence span containing the labeled value (or,
for derived/mixture values, the final-total span).

Classified 136/144: 106 revised, 15 retained (stated-aggregate convention),
13 mixture-effective, 2 reasoning-effective; 8 missing.

**Retention, stratified by changed vs no-op (the confound-corrected view):**

| | retained / classified |
|---|---|
| genuinely changed arms | 5/109 (6/109 under a numerical convention — `above_good_13_high_r2`'s mixture is effectively 300) |
| no-op arms (edit equals original value) | 10/27 |
| changed low arms, below vs above | 2/20 vs 2/24 — **no condition asymmetry** |

The earlier 27.5%-vs-11.1% per-cell comparison was a no-op confound (7 of
the below/low arms are no-ops vs 1 above/low) and is withdrawn. Reopening
occurs even with no changed token; because changed-arm retention is much
lower, an additional edit-detection response remains possible.

**Revision directions, changed arms only:** 150 moves up (below 15:3, above
16:6), 300 moves down (below 33:4, above 23:3) — condition-symmetric,
toward the baseline belief range (median 160, IQR 140–250; 150 inside it,
300 above it but adopted by 22/100 baseline traces). Ordinary-belief
correction is the dominant visible pattern. Threshold anchoring remains
pervasive: 112/144 continuations compute 178.4 or 173.95; the closest final
is 20,872,800 = 178.4 × 117,000.

## Interpretation within the accepted claim boundary

Under the original conditioned prompt, the cumulative prefix recovers an
endpoint-sized point gap by c3 (E2), and doubling a later adopted factor
changes the median answer by ~14% (E3): the factor is a weak, revisable
commitment. Incentive-specific repair is unresolved (null interaction;
no changed-arm retention asymmetry; direction pattern condition-symmetric).
Not established: text-alone sufficiency, sentence necessity, non-operative
rationales, or hidden-answer rationalization (E1 cannot arbitrate that).

## Gate E ruling

**Passed on the fourth review.** The final review independently reproduced
the classification and changed-only tables and checked the nontrivial rows
against the raw continuations. Two direction-preserving arithmetic provenance
fixes were made before clearance (`below_good_04_high_r0`: 1,009.2 at the
answer's stated 119,000; `below_good_03_low_r1`: 139.2 at the stated 117,000).
Neither affects a retention label, direction count, primary effect, or
conclusion. See the top of `GATE_E_REVIEW.md`.
