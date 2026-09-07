# Experiment 2 preregistration erratum

Written after Experiment 2. This file corrects the provenance claim in `prereg-2.md` without
modifying that hash-recorded artifact.

The design, prompts, endpoint, H1 threshold, and parent set were recorded before Experiment 2
data, but Experiment 2 was not committed or tagged before collection. The runner checked
hashes in `tasks/schedule-belief.json`; the final schedule also records later changes: an auth
refresh after a failed validation attempt that produced no data, correction of the one-sided
FAIL test during collection after the included validation parent had been read, and post-hoc
explanation-framing code after data. Parent 1 was used as the successful validation run, its
condition means were inspected, and it remained in the 16-parent analysis.

The repository therefore preserves the stated timeline, final hashes, and raw artifacts, but
does not provide an immutable external timestamp for the exact pre-data preregistration
snapshot. The sentence in `prereg-2.md` saying the implementation and preregistration were
"committed and hashed ... before launch" is incorrect as to **committed**. The original file
is retained unchanged because its SHA-256 is part of the historical schedule.

The phrase "only a symmetric result supports H1" in the preregistration's interpretability
section was informal and inconsistent with the formal H1 rule. The formal rule governs: spread
at least 10 points, `rP - w > 0`, and `rF - w < 0`; it does not require equal-sized PASS and
FAIL movements.
