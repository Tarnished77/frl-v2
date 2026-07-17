# Locked Compact Results

This directory is the tracked result layer for the revised manuscript.
`results_manifest.json` identifies the exact analysis source commit, software
versions, samples, estimands, input-manifest hashes, and output checksums.

The large immutable CSV outputs remain under `research/frl_v2/outputs/` and are
not committed. Their experiment manifests are retained in `input_manifests/`.
The formal random streams and pilot lock are in `research/frl_v2/design/`.

All CSV files in this directory were reproduced byte for byte in an independent
analysis run before they were committed. Manuscript tables and figures must use
these files or `results_manifest.json`; numbers must not be transcribed from the
legacy experiment directories.
