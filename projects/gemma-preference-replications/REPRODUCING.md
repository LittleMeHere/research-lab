# Reproduction

## CPU analysis of the frozen release

Run from this project directory with Python 3.12:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-analysis.txt
.venv/bin/python scripts/reproduce.py
```

The analysis uses only NumPy, pandas, SciPy and Matplotlib, runs offline, and writes
to `derived/recomputed/`. Choose another output directory with `--out PATH`.
It never modifies frozen `runs/` files. `ARTIFACTS.sha256` is checked first.

The original blind script expected a temporary `raw/` directory that was not retained
in the workspace. The wrapper reconstructs that layout, executes the unmodified
script, and compares its main sections with the saved JSON. It then recomputes the
later patch conditions, null-family summaries, matched NF4/bf16 comparisons, both
activity-judge analyses and the overview figure. CSV files contain the complete
tables behind the rounded report values.

**Limits:** probe correlations are read from `runs/probe/probe_r.csv`, not refitted.
Raw activation arrays were not exported. The CPU command is not a rerun of the GPU
experiment, the GPT judge, or the optional content-embedding baseline. Regenerated
plot PDFs may differ in creation metadata even when plotted values agree.

## GPU rerun using the saved directions

Use a separate directory so published evidence cannot be overwritten. From the
release directory:

```bash
python3 scripts/prepare_rerun.py /tmp/gemma-replication-rerun
cd /tmp/gemma-replication-rerun
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-gpu.txt
```

The helper refuses existing destinations. It copies experiment code, frozen task/pair
definitions, utilities, norms and direction arrays, **without trial outputs**. Its
`STARTING_INPUTS.json` hashes both starting inputs and code. Keep that file with new
results. The CPU release-recompute command applies to the original release directory,
not this incomplete, newly prepared run directory.

The NF4 runs used 24 GB L4 GPUs. `common.py` pins Gemma and its tokenizer to revision
`005ad3404e59d6023443cb575daa05336842228a`. Accept the model's access terms using your
own account and authenticate locally (for example, `hf auth login` from the venv).
Provide `HF_TOKEN` through your environment if needed; do not put credentials in code.
Weights require about 55 GB of download/cache space before quantization at load time.
The historical L4 torch build used CUDA 13.0; install a compatible torch build and
driver for your host. The GPU dependency installation was not freshly validated during
release preparation; recorded versions are in `artifacts/pod_env.txt` in the release.

Probe sweep and historical isotropic nulls:

```bash
.venv/bin/python scripts/steer.py --layers 2,5,8,11,14,17,20,23,26,29,32,35,38,41,44,47,50,53,56,59 --coefs=-0.05,0,0.05 --dirs probe
.venv/bin/python scripts/steer.py --layers 23 --coefs=-0.06,-0.04,-0.02,-0.005,-0.001,0.001,0.005,0.02,0.04,0.06 --dirs probe
.venv/bin/python scripts/steer.py --layers 20,23 --coefs=-0.05,0.05 --dirs rand0,rand1,rand2,rand3,rand4,rand5,rand6,rand7 --random-seed-mode layer
.venv/bin/python scripts/steer.py --layers 20,23 --coefs=-0.02,0.02 --dirs rand2,rand3,rand4,rand5,rand6,rand7 --random-seed-mode layer
.venv/bin/python scripts/steer.py --layers 20,23 --coefs=-0.05,-0.02,0.02,0.05 --dirs top0,top1,top2,shuf0,shuf1,shuf2,cov0,cov1,cov2,low0,low1,low2
```

The current release defaults to historical layer-dependent seeds. The old private
script defaulted to name-only seeding after September 4; using it unmodified would
silently regenerate different random directions. Use name seeding for later controls:

```bash
.venv/bin/python scripts/steer.py --layers 23 --coefs=-0.05,0.05 --dirs rand0,rand1,rand2,rand3,orth0,orth1,orth2,orth3 --random-seed-mode name
.venv/bin/python scripts/steer.py --layers 23 --coefs=-0.05,-0.02,0.02,0.05 --dirs top0,top1,top2,shuf0,shuf1,shuf2,rand0,rand1,rand2,rand3,orth0,orth1,cov0 --random-seed-mode name --scale_json runs/null/natsd_scale.json --tag natsd
.venv/bin/python scripts/steer.py --layers 23 --coefs=-0.05,0.05 --dirs probe,rand0,rand1,cov0 --mode a_only
```

Patching and saved-format completion examples:

```bash
.venv/bin/python scripts/patch.py --layers 2,5,8,11,14,17,20,23,26,29,32,35,38,41,44,47,50,53,56,59
.venv/bin/python scripts/patch2.py --format letter --conds none,eot_nl,eot_nl_all
.venv/bin/python scripts/patch2.py --format completion
.venv/bin/python scripts/patch2.py --format letter --layers 26,29,32 --conds nl,nl_all
.venv/bin/python scripts/patch2.py --format completion --layers 26,29,32 --conds nl,nl_all
.venv/bin/python scripts/steer_generate.py --format completion --max_new_tokens 100
.venv/bin/python scripts/analyze_steer.py
```

The last analysis script writes the historical steering/original-patching summaries;
its original patch summary is pair-weighted. Use the release `patching()` estimator
when comparing to the report's trial-weighted follow-up rates. These commands reproduce
the stated conditions, not the exact original launch sequence or every historical cell.
Do not combine interrupted runs from different hardware or changed code under the same
resume keys. Use another prepared directory for a different configuration.

## Refit utilities and directions

In a **new prepared rerun directory**, fetch the pinned upstream inputs, then rebuild
elicitation, utilities, activations and probes before running the interventions above:

```bash
bash scripts/fetch_inputs.sh
.venv/bin/python scripts/pairwise.py
.venv/bin/python scripts/fit_bt.py
.venv/bin/python scripts/extract_eot.py
.venv/bin/python scripts/probe.py
```

The full activation array is roughly 4 GB and is written as float32 because later
Gemma residuals overflow float16. Run extraction to completion; its norm accumulator
must cover all tasks in one completed invocation. An interrupted extraction restarts
all rows when no completion marker exists. Do not reuse a completion marker with a
different pool or precision.

To regenerate null directions, first run the probe/isotropic steering commands and
`scripts/analyze_steer.py` to supply their diagnostic table, then:

```bash
.venv/bin/python scripts/null_dirs.py --layers 20,23
.venv/bin/python scripts/null_dirs2.py
```

These overwrite the *new workspace's* null arrays. The shuffled-label probes are
oriented using true evaluation utility; their magnitudes are the control estimand.
Run null steering afterward. The optional `scripts/content_baseline.py` downloads
all-MiniLM-L6-v2 and prints text/source baselines; it is outside the frozen release
recompute and its encoder revision was not recorded historically.

## Activity experiment

In a prepared rerun directory with upstream inputs fetched:

```bash
.venv/bin/python scripts/vl_activities.py --stage liking
.venv/bin/python scripts/vl_activities.py --stage pick
.venv/bin/python scripts/vl_judge.py
.venv/bin/python scripts/vl_analyze.py
```

For the historical independent GPT judge, `scripts/vl_judge_codex.py` records its CLI,
model, reasoning effort and batch size. It requires your own authenticated Codex CLI
and incurs model usage; it is not invoked by CPU recomputation. Published GPT verdicts
are already included for offline analysis. Sampled outputs are not guaranteed to be
bit-identical across batching, hardware or library changes despite fixed seeds.

## bf16 check

Prepare another directory, on an 80 GB GPU. Using `PRECISION=bf16`, extract its norms
separately and steer with the saved NF4-fit directions. For example:

```bash
PRECISION=bf16 .venv/bin/python scripts/extract_eot.py --run runs/extract_bf16
PRECISION=bf16 .venv/bin/python scripts/steer.py --run runs/steer_bf16 --norms runs/extract_bf16/norms.json --layers 20,23 --coefs=-0.05,-0.02,0,0.02,0.05 --dirs probe,rand0,rand1,rand2,rand3,shuf0,shuf1,shuf2,top0,top1,top2 --random-seed-mode layer
.venv/bin/python scripts/probe.py --run runs/probe_bf16 --acts runs/extract_bf16/acts_eot.f32.npy --layers 17,20,23,26,29
PRECISION=bf16 .venv/bin/python scripts/steer.py --run runs/steer_bf16_fit --probe runs/probe_bf16/directions.npy --norms runs/extract_bf16/norms.json --layers 17,20,23,26,29 --coefs=-0.05,0.05 --dirs probe
```

The original bf16 run used torch 2.6/cu124 rather than the L4 environment. Preserve
your actual package versions and hardware alongside new results; these commands
specify the comparison but do not guarantee exact reconstruction of that environment.
