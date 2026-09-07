# Retrospective audits: code and reproduction

Three experiments on 16 saved Codex + GPT-5.6 Sol coding runs. A post-report record
labeled FAIL lowered the agent's retrospective probability estimate by about six points,
a PASS label made no measurable difference, and an instruction to ignore the record halved
the effect. Findings, design, and limits are in the [report](writeup.md). This file covers
reproduction only.

## Recompute the recorded results

Starting at the repository root, run the commands below. Numerical analysis and the raw-response audit use
only Python's standard library; Python 3.12.3 was used for collection and for the release checks.
No model account, network access, or private session directory is needed.

```bash
cd projects/retrospective-audits
tar -xzf data/replication-artifacts.tar.gz
python3 audit_replication.py
python3 analyze.py tasks/schedule-main.json --controls tasks/schedule.json
python3 analyze_belief.py tasks/schedule-main.json
python3 analyze_debias.py tasks/schedule-main.json
python3 code_debias_explanations.py tasks/schedule-main.json
```

Extract in a fresh checkout. The archive writes the original `runs/` paths; do not extract
over existing experimental data. `runs/` remains ignored by Git. The archive and its
[SHA-256 manifest](data/artifact-manifest.json) are the distribution copies.

Expected main results:

| Check | Expected result |
|---|---|
| Raw audit | 240 status, 192 probability, 192 instruction, 90 pilot, 108 control answers matched |
| Experiment 1, primary | 0/16 assimilation in each defined condition; 144/144 repeated THEN answers UNKNOWN |
| Experiment 2 | PASS − FAIL +5.770833; FAIL − WITHHELD −5.604167; H1 not supported |
| Experiment 3 | untreated −6.375000; attenuation +3.229167; instructed residual −3.145833 |

`audit_replication.py` fails on missing scheduled cells and checks raw final messages against
both saved answers and aggregate results. It also checks prompts, normalized contexts,
parent offsets and initial workspace manifests, and enumerates all 65,536 parent-level
sign flips for the headline probability contrasts. This supplements the historical
analyzers, whose loaders can skip absent runs. Its assertions must be enabled: do not run
it with Python's `-O` option.

Script output keeps its original terminology: “manual, blind” applies only to labels 13–16,
“true” records are relative to the augmented verifier, and “randomization” tests assume
exchangeability. The report explains each.

## Recheck the tests and regenerate figures

The pinned packages below were verified before release. The collection
rollouts also record pytest 9.1.1; a complete collection-time dependency lock was not saved.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-reproduce.txt
.venv/bin/python audit_replication.py --check-tests
.venv/bin/python make_figure_summary.py tasks/schedule-main.json
.venv/bin/python make_figures.py tasks/schedule-main.json
.venv/bin/python make_figures_belief.py tasks/schedule-main.json
.venv/bin/python make_figures_debias.py tasks/schedule-main.json
```

The test check copies each saved verifier workspace into a temporary directory, supplies
the canonical fixture shim, and runs the visible and augmented suites separately. Expected:
11 visible tests pass for every parent; augmented t001 has 13 passes; augmented t002 has
12 passes and two `day_count` failures. It never reruns the network-blocked parent task or
changes a recorded workspace.

The figure scripts regenerate the existing PNGs. For a separate destination use
`--out /absolute/path/to/an/existing/output-directory`. Additional descriptive analyses:

```bash
python3 code_explanations.py tasks/schedule-main.json
python3 mechanism.py tasks/schedule-main.json
```

## Follow one trajectory

After extraction, start at `runs/2026-08-29-main-13-t001/`:

| Artifact | Purpose |
|---|---|
| `task/task.json`, `prompts/initial.txt` | Original task and user prompt |
| `parent/rollout.jsonl`, `parent/codex.stdout` | Parent actions, tool outputs, and final report |
| `ground_truth/then_label.json`, `ground_truth/manual_label.json` | Automatic label and manual review, including blinding caveat |
| `fork_point/parent_prefix.jsonl`, `fork_point/workspace/` | Saved conversation and workspace |
| `verifier/run.stdout`, `verifier/workspace/` | Augmented test result and executed verifier workspace |
| `prompts/`, `branches/` | Experiment 1 inputs and raw continuations |
| `belief/prompts/`, `belief/branches/` | Experiment 2 inputs and raw continuations |
| `debias/prompts/`, `debias/branches/` | Experiment 3 inputs and raw continuations |
| `result.json`, `belief/result.json`, `debias/result.json` | Derived answers, branch identities and recorded checks |

What the archive includes and omits, and how it was built, is described in
[data/README.md](data/README.md).

## Collect new model runs

Offline reanalysis reproduces the recorded numerical results. New collection tests whether
the effect recurs. It is stochastic and requires model access; exact responses are not
guaranteed.

Collection used Linux/WSL2, Codex CLI 0.150.1, `gpt-5.6-sol`, medium reasoning, a
workspace-write sandbox, approval policy `never`, and the pragmatic personality. The
commands and resolved policies are in each run's `environment/` and raw records. The harness
depends on that CLI's native `codex exec fork` interface and permission configuration.
Compatibility with other CLI versions or currently available model access has not been
established.

Use a separate checkout without extracted `runs/`. A local Codex login must provide
`~/.codex/auth.json`; the harness copies it into isolated `~/.codex-runs/<run-name>/` homes.
Those homes are private and must not be published. Validate the recorded CLI, runtime
read paths, and canary checks on this machine before collecting data. A changed platform
or model configuration is a new implementation and should get its own recorded protocol.

For a compatible installation, a single validation run uses:

```bash
python3 fork_smoke.py runs/fresh-validation-c001 --task-dir tasks/c001-control-pass
```

The scheduled workflow is:

```bash
./run_schedule.sh tasks/schedule.json
./run_schedule.sh tasks/schedule-main.json
python3 label.py tasks/schedule-main.json
./run_belief.sh tasks/schedule-main.json
./run_debias.sh tasks/schedule-main.json
```

This reproduces the recorded prompts, including their test-set ambiguity. Correcting the
question or record would be a new experiment. To preserve blinding, label each completed
main parent before inspecting its later result or branches. The schedules enforce source
hashes and refuse incomplete existing runs. Runs already containing `result.json` are
skipped; invoking these commands on the released artifacts will not collect new data.

Experiments 2 and 3 locate the parent in the local per-run Codex home and restore its
recorded temporary workspace path. The released rollout alone cannot be passed to these
fork scripts as a portable replacement for that session state. Collect fresh parents and
keep their local homes for follow-ups.

## Provenance and scope

This release is a snapshot; development Git history is not included. No development
checkout or commit is needed for offline reanalysis. References to old commits and paths
inside frozen protocols or raw records are retained as provenance, not setup instructions.
The reported timing of design decisions is not independently established by this snapshot.

The initial [preregistration](minimal-prereg), the follow-up plans
([2](prereg-2.md), [3](prereg-3.md)), their errata
([2](prereg-2-errata.md), [3](prereg-3-errata.md)), and the schedules
([1](tasks/schedule-main.json), [2](tasks/schedule-belief.json),
[3](tasks/schedule-debias.json)) preserve the design decisions, deviations, and source hashes.
The historical harness, tasks, and analysis sources have
not been changed since collection. [Experiment 4](prereg-4.md) is an unrun plan and is not
part of these results.

## Licensing

Copyright (c) 2026 LittleMeHere. Unless a separate notice states otherwise:

| Material | License |
|---|---|
| Original source code, scripts, tests, and code snippets, including copies inside recorded artifacts | [MIT](LICENSE) |
| Original report, figures, research documentation, prompts, metadata, and research data, including rights in the dataset's selection and arrangement | [CC BY 4.0](LICENSE-CC-BY-4.0.txt) |
| Third-party material, including the adapted task code | Its existing terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) |

The archive is mixed content, and these grants cover only rights the author holds. Scope
details, third-party exceptions, and attribution wording are in [data/README.md](data/README.md).
