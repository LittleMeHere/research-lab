# value-leakage

When does GLM-5.2's visible Donation Bet reasoning start to matter? A causal
study of chain-of-thought recoverability and factor-edit sensitivity, done as
a SPAR take-home (mentor: Aditya Singh, Model Forensics).

**Write-up:** [writeup/WRITEUP.md](writeup/WRITEUP.md) (figure:
`writeup/fig_main.png`, regenerate with `python writeup/make_fig.py` from
this directory).

## Attribution

- The starting observational data — 100 baseline / 100 below-favoured /
  100 above-favoured GLM-5.2 rollouts — are **Aditya Singh's replication**,
  not reproduced here. Fetch them from his repository at the exact commit
  and place them at `runs/glm-5p2_20260815_030703/`:
  `git clone https://github.com/adsingh-64/value-leakage && git -C value-leakage checkout 16d1298`
  Verify against [INPUT_CHECKSUMS.sha256](INPUT_CHECKSUMS.sha256)
  (`sha256sum -c` inside that directory).
- `src/` is a fork of the pipeline in that repository; the analysis and
  experiment scripts added for this study are the `fw_*.py`, `o0_*.py`,
  `gcp_*.py`, and `judge.py` modules.
- Paper: [Value Leakage](https://arxiv.org/abs/2607.14345)
  ([paper code](https://github.com/TruthfulAI-research/value_leakage)).

## What is here (and why)

Everything required to reproduce every quantitative claim in the write-up:

| Path | Claim it supports |
|---|---|
| `runs/o0_glm5p2_20260830_125638/` | observational decomposition (77%/45% crossing, spots factor, disclosure audit) |
| `runs/fw_parents_20260830_134529/` | 40 frozen parents, preregistered cuts, token maps |
| `runs/fw_e1e2_20260830_152305/` | forced-answer curve (c1–c4), empty-reasoning control, c4 validity (35/40 exact) |
| `runs/fw_e3_20260830_161108/` | 150-vs-300 factor edit, v3.1 destination audit, review trail |
| `runs/fw_e0*`, `runs/fw_n_bug_probe_20260830/` | token-level continuation proofs ("323"→"23", temp-0 regeneration) and the Fireworks n>1 bug |
| `runs/gcp_glm52_{smoke,native_prefill,content_prefill,capability_probe}*` | evidence that Vertex/GCP cannot continue a prefilled reasoning turn |
| `src/`, `pyproject.toml`, `uv.lock` | all generation and analysis code (`uv sync` to set up) |

Every analysis and the figure re-run from the saved artifacts with no API
access. To sample fresh generations you need a Fireworks API key
(`FIREWORKS_API_KEY` in a local `.env`); to re-run the GCP probes you also
need your own Google Cloud project with Vertex AI access (`GCP_PROJECT`
environment variable).

This directory keeps its own self-contained layout (rather than the
repo-wide `code/data/notes` convention) so the scripts' relative paths run
unmodified from the project root.
