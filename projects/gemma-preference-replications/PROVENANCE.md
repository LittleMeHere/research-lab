# Provenance and release changes

## Sources

| Source | Revision |
|---|---|
| [Gilg repository](https://github.com/oscar-gilg/probing-persona-preferences) | `11869a5ef93a30f8d8856246f57ceeefdc9b3b1f` |
| [Value Leakage repository](https://github.com/TruthfulAI-research/value_leakage) | `f7e5480cfe8abeb64b7007ba24fb0164519c3b68` |
| [Gemma-3-27B-it](https://huggingface.co/google/gemma-3-27b-it) | `005ad3404e59d6023443cb575daa05336842228a` |

The model revision was recovered from the initial setup log's downloaded snapshot
path; `common.py` now pins tokenizer and weights to it. Original run metadata did
not record model revisions on each run, so this is evidence from setup rather than
per-run proof. Upstream repository revisions are the checked-out revisions at export.

The original experiment code was uncommitted. This release cannot identify the exact
source tree for every historical invocation. `provenance/source_snapshot.json` records
the selected files' hashes **before release edits**. `ARTIFACTS.sha256` records the
unchanged frozen artifacts actually shipped. Neither supplies missing historical commits.

`artifacts/pod_env.txt` records the L4 environment: Python setup used 3.12, torch
2.14/cu130, transformers 5.16.1 and bitsandbytes 0.50.2. The A100 check used torch
2.6/cu124 and transformers 5.16.1. `requirements-analysis.txt` pins the packages used
for the release CPU check. GPU requirements combine recorded core versions and
locally available analysis dependencies; they are not a complete historical lockfile.

## Changes made for release

- Added public documentation, licensing scope, artifact hashes, CPU reproduction,
  derived tables/figure, upstream fetch commands, and a separate GPU-rerun workspace helper.
- Pinned Gemma tokenizer/weights to the recovered setup revision.
- Added `--random-seed-mode` to `steer.py`: `layer` reproduces the original
  `1000*k + layer` draws; `name` reproduces the later controls. Default is `layer`.
  Name-seeded runs automatically receive `+seedname` unless explicitly tagged.
- Kept the original blind recomputation script and results unchanged. The wrapper
  reconstructs its missing input layout temporarily and compares its outputs.
- Made the optional GPT-judge script create its output directory on a fresh rerun.
- Replaced the draft report with a release report tied to the shipped artifacts.
  Corrected the “all bf16 numbers within 0.02” claim: the largest matched L23
  individual-direction difference is 0.114; the largest family-mean difference is 0.027.
- Made patch estimates trial-weighted throughout the release tables. Completion-format
  all-layer two-token flips are 119/237 = 0.502, compared with 0.51 in the earlier draft.
- Used Gemma judgments consistently for primary activity analyses: the rating≥40
  subset contains 3,544 decisive picks; the position coefficient is 3.025 per 100
  rating points. GPT judgments produce a slightly different sample.
- Narrowed verification claims. The historical blind analysis checks elicitation,
  original steering/patching and primary activity arithmetic; it reads probe r from
  a supplied fit table. Later controls are checked by the release analysis, not
  retroactively described as independently blind-verified.

## Scope of verification

Frozen rows and direction arrays were copied byte-for-byte. CPU analysis was rerun;
GPU execution, activation extraction, probe refitting and the GPT judge were not.
The historical activation array and full per-run source snapshots are absent.
The draft's text-embedding/within-source baselines and qualitative coherence judgments
are not independently reconstructed by this release command; their scripts or saved
completion texts are provided, but those numerical baselines are omitted from the
release's central claims. Private design reviews and agent transcripts are not shipped.

## Release checks (2026-09-06)

CPU recomputation passed in an isolated copy with no upstream checkout or private
scratch inputs. Its CSV and JSON outputs matched the release outputs byte-for-byte
in the installed pinned analysis environment. The historical comparison allows tiny
floating-point differences across platforms (rtol 1e-8, atol 1e-9); counts and keys
must match exactly. Dependencies were already installed; a fresh network installation
and GPU execution were not tested.

Python/shell syntax, local document links, frozen artifact hashes, historical RNG
seeding, and the fresh-rerun helper's input copying and refusal to overwrite an
existing directory were checked. A limited pattern scan of the curated files found
no credential tokens or private machine/account references. The figure was inspected.
