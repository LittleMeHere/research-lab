# Task preferences in Gemma-3-27B-it: adapted replications of probing, steering, and value leakage

Experiments: September 2–5, 2026. Release recomputation: September 6, 2026.
These runs reproduce central behaviors from [Gilg et al.](https://arxiv.org/abs/2605.13339)
and [Betley et al., Value Leakage §7](https://arxiv.org/abs/2607.14345), with
measurement changes described below. They are not exact replications of every condition.

## 1. Task preferences, linear probes and steering

**Original result.** Gilg et al. report consistent task preferences, a linear utility
probe with held-out r around 0.87, and a direction that steers choice most strongly at
layer 23 within a layer 17–26 window. Their reported random-direction control is inert.

**Setup.** We use 2,993 task texts from their files: 2,000 training-split and 993
evaluation-split tasks, filtered to at most 1,000 characters. Eight rounds of random
matchings produce 11,954 pairs in both orders, or 23,908 forward passes. Utilities
come from a Bradley–Terry model with an additive slot-A bias, using soft A/B outcomes.
The two answer tokens carry mean probability mass 0.991; 203 rows with mass below
0.5 are excluded from the utility fit. The fitted slot-A bias is 1.010 logits.

We fit ridge probes on single-task end-of-turn activations from the training split,
with five-fold regularization selection, and evaluate on the evaluation split.
Saved held-out correlations peak at **r = 0.8887, layer 30**. This release reads that
fit table; it does not refit the probes from the omitted activation array.

**Steering.** For each of 120 held-out pairs in both slot orders, add
`c × mean_activation_norm[layer] × unit_direction` to Task A's tokens and subtract it
from Task B's. Choice is P(A), renormalized over the A/B answer tokens at the first
answer position. Swing is the mean of P(A | +c) − P(A | −c) over the 240 trials.
Layer numbers are zero-based decoder-block output indices.

| Layer | 17 | 20 | 23 | 26 | 29 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| Probe swing, c=0.05 | 0.236 | 0.827 | **0.971** | 0.739 | 0.246 | 0.150 |

At layer 23, swing increases from 0.046 at c=0.001 to 0.214 at 0.005, 0.788 at 0.02,
0.952 at 0.04, 0.971 at 0.05 and 0.996 at 0.06. The peak and sign reproduce; the
window extends later than the original. Small reversed effects occur at layers 5–8.

**Controls.** Random directions also change choice. The table gives the mean absolute
swing at layer 23; it is not the largest effect in each family.

| Family | Raw norm, c=0.02 | Raw norm, c=0.05 | Natural-SD matched, c=0.05 |
|---|---:|---:|---:|
| Preference probe | **0.788** | **0.971** | **0.971** |
| Shuffled-label probe | 0.136 (3) | 0.366 (3) | 0.311 (3) |
| Random in top 50 PCs | 0.216 (3) | 0.435 (3) | 0.153 (3) |
| Isotropic random | 0.092 (6) | 0.302 (8) | 0.174 (4) |
| Random in PCs 500–1500 | 0.081 (3) | 0.174 (3) | — |
| Covariance-matched random | 0.022 (3) | 0.074 (3) | 0.001 (1) |

Parentheses count directions. Natural-SD matching scales displacement by the train
activation SD along each direction; factors are clipped to [0.05, 4]. Different columns
use different random draws, so this is not a fully paired comparison of matching rules.
The probe is the reference displacement in both schemes. Shuffled probes were
sign-oriented using true evaluation utilities, so their signed effects are not an
unbiased sign null. Use their magnitudes. The strongest individual top-PC draw swings
0.696 at c=0.05; the 2.2× probe advantage is over the **family mean**, not that draw.

At c=0.05 the eight isotropic draws have absolute swings from 0.100 to 0.518. Probe
steering is strong, but the inert-random-control finding does not hold under this
adapted setup. This supports effective steering more strongly than preference-specific
causal interpretation. Additional orthogonalized, single-sided and tiny-coefficient
controls are retained in the raw rows and release tables. Saved generated completions
provide examples of behavioral effects, not a quantitative capability-preservation test.

**Precision.** On an A100 with bf16 weights and the same fitted directions/pairs, the
L23 probe swing is 0.977 versus NF4's 0.971 at c=0.05. Across matched direction-family
means and c=0.02/0.05, the largest difference is 0.027; the largest individual-direction
difference is **0.114**. Main effects persist in bf16. This check does not establish
numerical equivalence for every direction, task or experiment.

## 2. Turn-boundary activation patching

**Original result.** Gilg et al., Appendix K, report that transplanting turn-boundary
activations from the opposite task ordering can flip the chosen slot, with a later
layer window than task-span steering. The intervention includes the end-of-turn token
and following newline. Their all-layer flip rate is 56.9%.

**Setup.** Use the same 120 pairs in both orders. The donor is the other ordering of
the same tasks. Each format has its own unpatched baseline. Retain trials with
|P(A)−0.5| > 0.2, then count changes in the sign of P(A)−0.5 after patching.
Letter format reads A/B logits. Completion format uses the paper's completion prompt
but forces the prefix “Task” and reads the next A/B token; it is still a logit readout.

| Patched tokens / layers | Letter format | Completion format |
|---|---:|---:|
| Both tokens, L26 | 0.247 | 0.241 |
| Both tokens, L29 | 0.430 [0.362, 0.496] | 0.338 [0.278, 0.397] |
| Both tokens, L32 | 0.357 | 0.354 |
| Both tokens, L35 | 0.021 | 0.000 |
| Both tokens, all layers | **0.570 [0.500, 0.641]** | **0.502 [0.430, 0.570]** |
| Newline only, all layers | 0.174 | 0.122 |
| End-of-turn only, all layers | 0.166¹ | 0.160 |

Denominators are 235 confident letter trials and 237 completion trials. Intervals
are percentile 95% CIs from 2,000 resamples of whole task pairs, seed 0; point estimates
weight retained trials equally. ¹The end-of-turn-only letter result comes from the
original letter experiment, whose own baseline is used. Full tables include that
experiment's task-span swaps as well as both follow-up formats.

The all-layer letter result matches the published magnitude. Single-layer rates are
below the published majority-flip plateau, while the broad late-layer pattern agrees.
Both boundary tokens matter jointly; neither alone reproduces the all-layer effect.
The coarse layer grid does not locate an exact onset or cutoff.

## 3. “Random” activity choices

**Original result.** Value Leakage §7 finds that models asked to choose leisure
activities at random favor activities they rate more highly.

**Setup.** Use the upstream 100 activities, five prompt wordings and seed-0 pairing
protocol. Collect 2,000 liking ratings and 10,000 choices with Gemma NF4, temperature
1, top-p 1 and top-k 0. Resolve ambiguous regex parses plus a 300-row audit using
Gemma at greedy decoding and, separately, GPT (`gpt-5.6-sol`, low reasoning effort,
codex-cli 0.153.2). Each judge overrides regex decisions on its judged rows.

**Results.** Gemma and GPT agree on **97.7% of 2,805 judged rows**. Primary results
use Gemma judgments consistently: 62 refusals, 9,938 decisive choices, and
**r = 0.702 [0.587, 0.790]** between activity mean liking and selection rate. GPT
judgments give 17 refusals and **r = 0.704 [0.589, 0.791]**. Correlation intervals
use Fisher z across 100 activities; they do not model uncertainty in the liking ratings.

The higher-rated activity is chosen in **65.9%** of 9,907 decisive unequal-score
trials. Option 1 is selected only 28.2% of the time. A position-adjusted logistic model,
P(option 1) = sigmoid(a + b × score_gap/100), yields a = −1.186 and b = 3.025
(model-based SE 0.079). This SE treats choices as independent conditional on fixed
activity scores; it does not account for all shared-activity uncertainty.

As a post-hoc sensitivity analysis, retaining only activities rated at least 40 leaves
60 activities and 3,544 decisive choices: r = **0.426 [0.193, 0.614]**, with the
higher-rated activity chosen 57.1% of the time. This rating threshold is not an
independent safety annotation. The study establishes preference-associated choice bias;
it does not distinguish liking from safety or perceived user welfare.

## Differences, verification and limits

These are one-model adapted replications. The main differences from the originals are
NF4 weights; logit rather than generated-choice readouts for items 1–2; random task
pairing and Bradley–Terry rather than active sampling and Thurstonian utility fitting;
and locally sampled Gemma activity choices. Train/eval task identities are disjoint
for the probe, but utilities are estimated jointly from one comparison dataset.

The original blind analysis recomputed elicitation, original steering/patching and
primary activity arithmetic. It read the probe peak from a supplied summary table.
The release script reruns that analysis and separately computes the later controls;
those additions are not an independent blind validation. Artifact hashes, unique keys,
paired-coefficient coverage and per-cell trial counts are checked before tables are
written. Original raw rows and derived fit artifacts remain separate from new outputs.

The historical intervention scripts contain zero-hook identity, positive-hook and
padding controls. Saved jitter measurements show a roughly 0.25-nat single-trial
readout resolution at relevant bf16 logit magnitudes. Matching batches reduces
batch-composition confounding; averaging does not guarantee rounding error is unbiased.
GPU execution and those controls were not rerun for release preparation.

Recorded experimental effort was approximately 16.5 hours and 26 L4-hours plus
1.3 A100-hours, about $27 of GPU/VM compute at the time; this is a historical estimate,
not a rerun price quote or an accounting of every model-judge cost. See
[PROVENANCE.md](PROVENANCE.md), [ARTIFACTS.md](ARTIFACTS.md), and
[REPRODUCING.md](REPRODUCING.md) for the code/data chain and reproduction limits.
