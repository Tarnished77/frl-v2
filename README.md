# FRL v3.1 reproducibility package

This package reproduces the computational evidence for:

> Funding Withdrawals, Market Losses, and Fire-Sale Amplification: An
> Agent-Based Stress Experiment

It contains the finance-native v3.1 model, locked configuration and random
streams, block-level formal results, deterministic analysis and figure
builders, tests, and expected publication assets.

## What is not included

The package deliberately excludes:

- the manuscript and Supplementary source or PDFs;
- the Cover Letter, Highlights, submission notes, and contact details;
- editable PowerPoint paper artwork;
- development conversations, prompts, logs, secrets, caches, and local paths;
- third-party raw OFR and FRED observations;
- legacy game-backed, v2, and invalid v3.0 evidence; and
- large agent-level and daily formal CSV files.

The omitted simulation files can be regenerated from the included model,
locked configuration, experiment manifests, and seed manifests. Their original
SHA-256 values remain in each `experiment_manifest.full.json` and compact
`experiment_manifest.json`.

## Environment

Python 3.12 or 3.13 is required. Exact direct dependencies are listed in
`requirements-lock.txt`.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe reproduce.py
```

macOS or Linux:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements-lock.txt
./.venv/bin/python reproduce.py
```

The default check:

1. verifies every package checksum;
2. runs the model and analysis tests;
3. replays the small smoke experiment byte for byte; and
4. rebuilds all publication tables and figures, requiring byte-identical
   numerical/text outputs, quantitatively equivalent PNG rendering, and
   matching vector-PDF page count, near-identical page geometry, and embedded
   fonts.

Binary PNG compression and PDF container bytes may vary across Python wheels
and font-rasterization builds. The replay therefore permits at most a one-pixel
border difference and tightly bounded anti-aliasing variation while comparing
PNG RGB content, and checks vector-PDF structure instead of requiring
cross-environment file-hash equality for those two rendered formats.

To rerun the block-level formal analysis before rebuilding the publication
assets:

```bash
python reproduce.py --reanalyze
```

All commands refuse to overwrite an existing output directory.

## Package map

- `frl_v3/`: model, random-stream, experiment, and analysis modules.
- `configs/`: model and resilience-design configuration.
- `resilience_design/`: locked design, seed manifest, calibration audit, and
  finite-population records.
- `anchors/`: locked derived anchors and a source ledger. Third-party raw
  observations are not redistributed.
- `data/sample/`: a complete small experiment used for byte replay.
- `data/formal/`: compact block-level formal inputs and provenance manifests.
- `derived-resilience-*/`: locked analysis outputs used by the publication
  builder.
- `expected/publication/`: expected CSV, LaTeX, PDF, and PNG outputs plus their
  manifest.
- `tests/`: deterministic accounting, random-stream, design, and analysis
  tests.

`DATA_DICTIONARY.md` documents the compact formal and derived data.

## Full simulation

The full formal experiment is substantially more expensive than the default
verification. A new primary run can be generated with:

```bash
python run_resilience.py run primary \
  --locked-design resilience_design/locked_resilience_design.json \
  --output-dir outputs/resilience-primary-replay
```

The same interface provides the four declared sensitivity families. Do not
reuse an existing output path. The primary design contains 54 conditions, 200
paired replication blocks per condition, 100 institutions per block, and a
30-day horizon.

## Statistical unit and evidence boundary

The independent Monte Carlo unit is the replication block. The 1,080,000
institution-condition trajectories are not independent empirical
observations. Primary outcomes are mean 30-day equity loss and mean loss among
the worst-loss 10 percent of institutions in each run. No exit occurs in the
locked primary or sensitivity envelope, so no survival-model estimate is part
of the submitted evidence.

## Integrity and licenses

Run `python reproduce.py` before using the package. `checksums.sha256` covers
every payload file.

Code is licensed under the MIT License. Generated simulation data and figures
are licensed under CC BY 4.0. Third-party raw source observations are not
redistributed and remain subject to their source providers' terms.

## Repository and citation

- Public source repository: <https://github.com/Tarnished77/frl-v2>
- Archived release (version 1.0.1): <https://doi.org/10.5281/zenodo.21491979>
- All-version DOI: <https://doi.org/10.5281/zenodo.21406157>

The included `CITATION.cff` provides machine-readable citation metadata.
