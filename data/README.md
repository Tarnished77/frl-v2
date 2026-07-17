# Data layout

`sample/smoke-v1/` is a small deterministic example containing all 27 primary
conditions, two replication blocks per condition, ten investors per run, and a
five-day horizon. It is committed so that `python reproduce.py` can verify a
byte-identical model replay without downloading the full data.

The formal data are not stored in ordinary Git history. Download and extract
`frl-v2-formal-data-v1.zip`, then run:

```text
python reproduce.py --formal-data-dir PATH_TO_EXTRACTED_ARCHIVE/formal
```

The separate archive contains the complete primary, homogeneous, N=50, and
N=200 CSV outputs, experiment manifests, seed manifests, design lock, data
dictionary, license, and SHA-256 checksums.
