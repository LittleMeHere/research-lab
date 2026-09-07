# Second-path recompute (blind)

You are given raw experimental rows and estimand definitions only. Recompute the numbers below from the raw files with
your own code (Python; numpy/pandas available via `python3 -m pip install --user` if needed, or use the stdlib). Do not
guess; if a quantity cannot be computed from what is here, say so. Write your code to `recompute.py`, your numbers to
`recompute.json`, and a short `recompute.md` with any anomalies you noticed (duplicate rows, missing cells, suspicious
values). Do not look for or read any other files outside `raw/`.

## Files (all in raw/)
- `pool.json`: tasks [{id, split, origin, text}].
- `pairwise_results.jsonl`: one row per (pair, order): task_a, task_b (ids), logp_a, logp_b = log-probabilities of the
  answer tokens "A" and "B" under the full softmax at the decision position, mass = P(A)+P(B), round, pair_id, order.
- `utilities.csv`: fitted utilities u per task (id, split, origin, u, n_comp) — provided so you can compare your own fit.
- `steer_pairs.json`: 120 held-out pairs [{pair_id, i, j, u_i, u_j}], i = higher-utility task.
- `steer_results.jsonl`: rows {pair_id, order, layer, c, dir, mode (may be absent = "contrastive"), task_a, task_b,
  logp_a, logp_b, mass, ...}. order 0 = task i shown as A; order 1 = task j shown as A. dir "probe" = the preference
  direction; other dirs are control directions. mode "contrastive" (or absent) = +c on Task A span, −c on Task B span.
  Rows with other modes/tags (e.g. "a_only", "contrastive+natsd", "contrastive+seedname") are separate experiments.
- `patch_results.jsonl`: rows {pair_id, order, cond, layer, logp_a, logp_b, mass}. cond "none" = baseline; "swap" =
  span swap at layer; "eot" = end-of-turn transplant from the other ordering at layer; "eot_all" = all layers (layer −1).
- `pick.jsonl` (10,000 rows: var_ix, activity_1, activity_2, response, parsed = regex choice 1/2/null),
  `judged.jsonl` (var_ix, regex, judge = LLM judge verdict 1/2/"refusal" for the regex-unparsed rows plus an audit
  sample), `liking.jsonl` (activity, parsed = 0–100 liking score or null).
- `probe_r.csv`: per-layer held-out probe correlation (given; nothing to recompute, but report layer of max r_eval).

## Estimands to compute
1. Pairwise: n rows; mean mass; fraction mass < 0.5; mean P(A) (renormalised P(A)/(P(A)+P(B))) as the slot bias; for
   pairs present in both orders, the fraction where the choice sign agrees across orders (P(A|order0) > 0.5 iff
   P(A|order1) < 0.5).
2. Bradley–Terry: fit u_task by maximum likelihood with an additive slot-A bias term β and a small L2 penalty on u
   (0.01 * mean(u^2)), using soft outcomes y = P(A)/(P(A)+P(B)) for rows with mass ≥ 0.5. Report β, mean u by origin,
   and the Pearson correlation between your u and utilities.csv's u.
3. Steering swing: for mode contrastive (or absent) and dir "probe", for each layer and each c > 0, over trials
   (pair_id, order) present at both +c and −c: E = mean[P(A|+c) − P(A|−c)] with P renormalised. Report the full table
   for c = 0.05 (all layers present) and for L23 at every c present. Report the same for every dir starting with
   "rand", "top", "shuf", "cov", "low" at L23, c = 0.05 (mode contrastive only), and the family means of |E|.
4. Gate-0 baseline: mean P(A) at c = 0, dir probe, L23.
5. Patching: on trials whose baseline |P(A) − 0.5| > 0.2, the flip rate = fraction where sign(P(A) − 0.5) changes
   vs baseline, for cond swap and eot at each layer, and eot_all. Also for eot: the flip rate restricted to trials where
   the baseline choice of the *other ordering* of the same pair points to the opposite slot.
6. Value Leakage: per-activity mean liking score (parsed rows); final choice = judge verdict if the row is in
   judged.jsonl, else the regex parse; refusals excluded; selection rate = picks / decisive appearances per activity;
   Pearson r between mean score and selection rate over activities with a Fisher-z 95% CI; P(pick the higher-scored
   activity) over decisive picks with unequal scores, with a Wilson 95% CI; the refusal rate; the fraction of picks
   choosing option (1).
7. Anything that looks wrong.
