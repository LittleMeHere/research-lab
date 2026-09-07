# Recomputed results

All probabilities below are renormalized over the A/B answer tokens where applicable.

## Pairwise and Bradley–Terry

Pairwise: n=23908; mean mass=0.991461; mass<0.5=0.008491; mean P(A)=0.607444; order agreement=0.735570 (8793/11954).

Bradley–Terry: beta=1.009973; correlation with supplied utilities=1.000000 (n=2993). Mean u by origin: alpaca=0.037139, bailbench=-3.510482, math=1.542356, stress_test=-1.816854, wildchat=0.878560.

## Steering

Probe direction, c=0.05:

| Layer | E | n |
|---:|---:|---:|
| 2 | -0.010762 | 240 |
| 5 | -0.040261 | 240 |
| 8 | -0.090996 | 240 |
| 11 | -0.001535 | 240 |
| 14 | 0.020415 | 240 |
| 17 | 0.236434 | 240 |
| 20 | 0.827132 | 240 |
| 23 | 0.971159 | 240 |
| 26 | 0.738874 | 240 |
| 29 | 0.245713 | 240 |
| 32 | 0.149617 | 240 |
| 35 | 0.022426 | 240 |
| 38 | 0.004634 | 240 |
| 41 | -0.003228 | 240 |
| 44 | 0.002200 | 240 |
| 47 | -0.005729 | 240 |
| 50 | -0.015795 | 240 |
| 53 | -0.009926 | 240 |
| 56 | -0.002125 | 240 |
| 59 | 0.000740 | 240 |

Probe direction, layer 23:

| c | E | n |
|---:|---:|---:|
| 0.001 | 0.046396 | 240 |
| 0.005 | 0.214275 | 240 |
| 0.02 | 0.787908 | 240 |
| 0.04 | 0.952080 | 240 |
| 0.05 | 0.971159 | 240 |
| 0.06 | 0.995638 | 240 |

Controls at layer 23, c=0.05:

| Direction | E | n |
|---|---:|---:|
| cov0 | -0.023509 | 240 |
| cov1 | 0.108883 | 240 |
| cov2 | -0.090514 | 240 |
| low0 | 0.097022 | 240 |
| low1 | -0.173654 | 240 |
| low2 | -0.252562 | 240 |
| rand0 | 0.286296 | 240 |
| rand1 | -0.203022 | 240 |
| rand2 | 0.517968 | 240 |
| rand3 | 0.369155 | 240 |
| rand4 | -0.450775 | 240 |
| rand5 | 0.157000 | 240 |
| rand6 | -0.333425 | 240 |
| rand7 | -0.100392 | 240 |
| shuf0 | -0.435347 | 240 |
| shuf1 | -0.334710 | 240 |
| shuf2 | 0.328120 | 240 |
| top0 | 0.521947 | 240 |
| top1 | 0.087346 | 240 |
| top2 | 0.696184 | 240 |

Control family means of |E|: rand=0.302254, top=0.435159, shuf=0.366059, cov=0.074302, low=0.174412.

Gate-0 baseline (probe, L23): mean P(A)=0.606419, n=240.

## Patching

The confidence filter retains 235 of 240 baseline trials.

| Layer | Swap flip rate | EOT flip rate | EOT restricted flip rate | n | restricted n |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.553191 | 0.000000 | 0.000000 | 235 | 162 |
| 5 | 0.565957 | 0.008511 | 0.000000 | 235 | 162 |
| 8 | 0.544681 | 0.008511 | 0.006173 | 235 | 162 |
| 11 | 0.548936 | 0.025532 | 0.012346 | 235 | 162 |
| 14 | 0.565957 | 0.012766 | 0.006173 | 235 | 162 |
| 17 | 0.582979 | 0.025532 | 0.006173 | 235 | 162 |
| 20 | 0.582979 | 0.042553 | 0.018519 | 235 | 162 |
| 23 | 0.455319 | 0.055319 | 0.030864 | 235 | 162 |
| 26 | 0.208511 | 0.085106 | 0.074074 | 235 | 162 |
| 29 | 0.055319 | 0.102128 | 0.111111 | 235 | 162 |
| 32 | 0.021277 | 0.085106 | 0.080247 | 235 | 162 |
| 35 | 0.000000 | 0.004255 | 0.000000 | 235 | 162 |
| 38 | 0.000000 | 0.000000 | 0.000000 | 235 | 162 |
| 41 | 0.000000 | 0.000000 | 0.000000 | 235 | 162 |
| 44 | 0.000000 | 0.000000 | 0.000000 | 235 | 162 |
| 47 | 0.000000 | 0.000000 | 0.000000 | 235 | 162 |
| 50 | 0.000000 | 0.000000 | 0.000000 | 235 | 162 |
| 53 | 0.000000 | 0.000000 | 0.000000 | 235 | 162 |
| 56 | 0.000000 | 0.000000 | 0.000000 | 235 | 162 |
| 59 | 0.000000 | 0.000000 | 0.000000 | 235 | 162 |

EOT-all flip rate: 0.165957 (39/235).

## Value leakage

Across 100 activities, score/selection Pearson r=0.702154, Fisher-z 95% CI [0.586643, 0.789665].

Higher-scored activity picked with probability 0.658928 (6528/9907), Wilson 95% CI [0.649533, 0.668200]. Refusal rate=0.006200 (62/10000); option-(1) fraction among decisive picks=0.282351 (2806/9938).

Per-activity mean scores and selection rates are in `recompute.json`.

## Probe

Maximum held-out r_eval is at layer 30 (r_eval=0.888663).

## Anomalies

- judged audit rows override regex: 3 decisive disagreements and 5 refusals among 300 audited parsed rows
- pairwise rows with A/B mass below 0.5 (excluded from BT only): 203
- exact A/B log-probability ties (likely affected by logged precision): pairwise=102, steering=506, patching=48
- large control steering swings at L23/c=0.05; largest is top2, E=0.696184
