# Release status

The three adapted replication experiments are complete. This directory contains a
curated public snapshot, not the subsequent research project.

- Frozen artifacts: 23,908 elicitation rows; 120 held-out task pairs in both orders
  for interventions; 2,000 activity ratings and 10,000 activity choices.
- Release analysis: `scripts/reproduce.py`; checked outputs in `derived/release/`.
- Probe fitting is represented by saved correlations and direction arrays. The large
  activation array is omitted; it must be regenerated to refit the probes.
- GPU experiments were not repeated during release preparation. Historical code was
  uncommitted; current release code includes documented portability and seed fixes.
- Known scientific limits and exact estimators: `REPLICATION.md`.
- Reproduction commands and historical provenance: `REPRODUCING.md`, `PROVENANCE.md`.

Last checked: 2026-09-06. Publication is a separate step from preparing this snapshot.
