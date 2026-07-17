# FRL v2 public reproducibility repository

This repository contains the finance-native simulation and analysis code for:

`Liquidity Pressure, Market Risk, and Investor Exit: An Agent-Based Calibration Analysis`

Its purpose is to reproduce the computational evidence, not to distribute the
submission manuscript. Manuscript source/PDFs, Supplementary Materials, cover
letter, Highlights, author contact details, submission records, PowerPoint
source, AI conversations, prompts, logs, local paths, secrets, and caches are
intentionally excluded.

## Repository contents

```text
frl-v2-public-repository/
|-- README.md
|-- LICENSE
|-- CITATION.cff
|-- pyproject.toml
|-- requirements-lock.txt
|-- reproduce.py
|-- run_pipeline.py
|-- analyze_results.py
|-- plot_results.py
|-- frl_v2/                 # simulation and analysis library
|-- tests/                  # deterministic and accounting tests
|-- configs/                # model and experimental configuration
|-- design/                 # pilot record, design lock, and formal seeds
|-- data/
|   |-- README.md
|   `-- sample/smoke-v1/   # small, byte-replayable example output
|-- results/                # compact locked statistical outputs
|-- figures/                # figures regenerated from compact results
|-- DATA_DICTIONARY.md
|-- package_manifest.json
`-- checksums.sha256
```

The complete formal CSV outputs are distributed separately as
`frl-v2-formal-data-v1.zip`. This avoids committing a 151.5 MiB CSV to normal
Git history while keeping the full data available for independent reanalysis.

## Environment

- Python 3.12 or 3.13
- exact dependencies in `requirements-lock.txt`

```text
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements-lock.txt
```

On macOS or Linux, replace `.venv\Scripts\python` with `.venv/bin/python`.

## Quick verification

```text
python reproduce.py
```

This command verifies repository checksums, runs all tests, reruns the 27-cell
smoke experiment, and requires every sample output to match byte for byte.

## Full formal-data reproduction

Extract the separate formal-data ZIP and provide its `formal/` directory:

```text
python reproduce.py --formal-data-dir ../frl-v2-formal-data-v1/formal
```

The command verifies the data archive, regenerates all compact statistical
outputs and figures, and checks them byte for byte against the locked versions.
It does not build or distribute the manuscript.

## Run the formal experiments from scratch

The runner refuses to overwrite output directories.

```text
python run_pipeline.py dry-run
python run_pipeline.py run primary --output-dir NEW_PRIMARY_DIRECTORY
python run_pipeline.py run homogeneous --output-dir NEW_HOMOGENEOUS_DIRECTORY
python run_pipeline.py run finite-50 --output-dir NEW_N50_DIRECTORY
python run_pipeline.py run finite-200 --output-dir NEW_N200_DIRECTORY
```

## Formal samples

| Experiment | Runs | Agent rows | Exits |
| --- | ---: | ---: | ---: |
| Primary 27-cell design | 5,400 | 540,000 | 253,230 |
| Homogeneous sensitivity | 1,800 | 180,000 | 78,331 |
| Finite-size, N=50 | 450 | 22,500 | 10,592 |
| Finite-size, N=200 | 450 | 90,000 | 42,518 |

The model uses order-independent SHA-256-derived random streams. All primary
cells contain 200 paired replication blocks. Historical game-backed experiments
are not part of this repository or its evidence chain.

## Citation and licenses

Use `CITATION.cff` for the software citation and cite the formal-data record
after its persistent identifier is assigned. Source code is MIT licensed.
Generated sample data, compact results, and figures are CC BY 4.0.
