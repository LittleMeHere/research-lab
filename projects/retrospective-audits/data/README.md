# Recorded research artifacts

`replication-artifacts.tar.gz` contains the selected original artifacts for the three
experiments. `artifact-manifest.json` records their hashes and the archive hash. See the
[reproduction guide](../README.md#recompute-the-recorded-results) before extracting.

## Contents

The archive holds byte-preserving copies of selected artifacts from all 16 main and
22 pilot runs, including workspace bytecode and pytest caches because they were part of
the saved experimental state. Timestamps and permissions are normalized and empty
directories are omitted; the original workspace manifests retain their entries and modes.
Redundant branch workspace copies and unrelated diagnostic runs are omitted. Full Codex
homes, credentials, and session databases are not included. Raw records retain original
local paths, which need not exist for offline reanalysis, and encrypted reasoning fields,
which do not expose readable reasoning. The manifest records content integrity at
packaging time, not an independent timestamp of collection.

[`package_artifacts.py`](../package_artifacts.py) documents the selection and builds the
archive from the original local runs. It refuses to overwrite a package. Rebuilding is a
maintainer operation, not a prerequisite for reanalysis.

## Licensing

Original research data, metadata, prompts, and rights in the dataset's selection and
arrangement are licensed under [CC BY 4.0](../LICENSE-CC-BY-4.0.txt), to the extent
of the author's rights. Attribution: LittleMeHere (2026), *Records labeled FAIL lower
a coding agent's retrospective probability estimates*.

This is not a blanket license for every archive member or transcript passage:

- Original code remains [MIT-licensed](../LICENSE), including recorded code snippets
  and saved workspace copies.
- Adapted task code retains the upstream MIT notice. Other third-party material,
  including provider-supplied instructions and quoted text, retains its existing terms
  and is not relicensed here. See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
- No copyright is asserted over facts or otherwise unprotected material, including
  model-generated outputs where no such rights exist. No third-party rights are granted.

The archive preserves recorded bytes and contains no newly inserted licensing files.
When redistributing it, include these licensing notes, the applicable license texts,
and the third-party notices alongside it.
