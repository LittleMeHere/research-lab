# Artifact guide

`runs/` preserves the historical layout. A JSONL line is one saved trial or response;
the directories also contain clearly named input and fitted artifacts. Release analysis
reads these files without rewriting them and writes only to `derived/`.

| Directory | Raw observations | Inputs / fitted or derived artifacts |
|---|---|---|
| `runs/pairwise/` | `results.jsonl`: task IDs, pair/order, answer log probabilities and mass | `pool.json`: IDs, texts, source, train/eval split; `pairs.json`: elicitation pairs; `run.json`: metadata; `utilities.csv`, `bt_fit.json`: Bradley–Terry fit |
| `runs/extract/` | Activation array omitted | `norms.json`: measured per-layer norms |
| `runs/probe/` | Activation array omitted | `directions.npy`: unit directions, shape [62, 5376]; `probe_r.csv`: saved per-layer fitted correlations |
| `runs/steer/` | `results.jsonl`: pair/order, layer, coefficient, direction, mode, answer log probabilities | `pairs.json`: 120 fixed held-out pairs; `summary.csv`: original analysis; `jitter.json`: saved readout-control measurements |
| `runs/null/` | — | `dirs.npz`: fitted/sampled null directions; `natsd_scale.json`: natural-SD scale factors |
| `runs/patch/` | `results.jsonl`: original letter-format span/boundary patches | `summary.csv`: original pair-weighted analysis |
| `runs/patch_letter/`, `runs/patch_completion/` | `results.jsonl`: two-token and newline-only follow-ups, with separate baselines | — |
| `runs/bf16/` | `steer_bf16_results.jsonl`: NF4-fit directions with bf16 weights; `steer_bf16_fit_results.jsonl`: bf16-fit probe | Norms, pairs, directions and probe-fit correlations |
| `runs/steer_gen/` | `results_completion.jsonl`: 240 saved completion texts | Simple first-token parsing is a diagnostic, not a quality score |
| `runs/vl_activities/` | `liking.jsonl`, `pick.jsonl`: prompts, responses and regex parses; `judged.jsonl`: Gemma judgments; `codex/judged_codex.jsonl`: independent GPT judgments | Run metadata; original activity and aggregate summaries |
| `runs/second_path/` | — | Original blind-analysis brief, unmodified analysis script and saved results; the release wrapper supplies its historical `raw/` layout in a temporary directory |
| `artifacts/` | `kickoff_check.json`: initial model/token checks | `pod_env.txt`: historical L4 package/hardware record |

## Intervention keys

Steering rows are unique on `(pair_id, order, layer, dir, mode, c)`. Missing `mode`
means `contrastive`. Do not pool modes:

- `contrastive`: +v on A, −v on B; historical random seed = `1000*k + layer`.
- `a_only`: +v on A only; the same historical seed convention.
- `contrastive+seedname`: seed = `k`, shared across layers; includes orthogonalized controls.
- `contrastive+natsd`: name-seeded randoms, coefficients rescaled using saved natural SDs.

The raw-norm isotropic mean at c=0.02 uses six directions (`rand2`–`rand7`); at
c=0.05 it uses eight. Natural-SD matching uses four isotropic, three shuffled,
three top-PC, two orthogonalized, and one covariance-matched directions. It is not
the same sample of directions across every column. Scale factors are clipped to
[0.05, 4]; covariance matching hits the lower bound.

Patching rows are unique on `(pair_id, order, cond, layer)` within each directory.
`none` is that run's own baseline; layer −1 denotes an all-layer intervention.
Always join to the baseline from the same run and format. The new release tables
use trial-weighted flip rates with pair-cluster bootstrap intervals; original
`runs/patch/summary.csv` averaged each pair's retained trials first. Small numerical
differences between these summaries are expected and are not new model runs.

Activity picks are keyed by `var_ix`; ratings by `(activity_ix, rep)`. Judge rows
override regex parses, including disagreements in the audit sample. Invalid or
refusal verdicts are excluded from decisive-choice estimands. Gemma is the primary
judge throughout the public report; GPT results are shown separately.
