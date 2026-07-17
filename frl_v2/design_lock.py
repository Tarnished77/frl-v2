"""Create a self-contained, immutable confirmatory-design record."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

from .config import DesignConfig
from .experiment import replication_ids
from .io_utils import require_new_directory, sha256_file, write_json
from .rng import stream_seed_manifest


def lock_confirmatory_design(
    design: DesignConfig,
    pilot_dir: Path,
    output_dir: Path,
    source_commit: str,
) -> dict[str, object]:
    if not source_commit or len(source_commit) < 7:
        raise ValueError("source_commit must identify the research-model revision")

    pilot_results_path = pilot_dir / "pilot_results.csv"
    pilot_lock_path = pilot_dir / "locked_design.json"
    with pilot_lock_path.open("r", encoding="utf-8") as handle:
        pilot_lock = json.load(handle)
    if not pilot_lock.get("selected_is_acceptable"):
        raise ValueError("The selected pilot budget is outside the pre-specified range")

    with pilot_results_path.open("r", encoding="utf-8", newline="") as handle:
        pilot_rows = list(csv.DictReader(handle))
    if len(pilot_rows) != len(design.pilot_stress_budgets):
        raise ValueError("Pilot output does not contain every pre-specified budget")

    require_new_directory(output_dir)
    shutil.copyfile(pilot_results_path, output_dir / "pilot_results.csv")
    shutil.copyfile(pilot_lock_path, output_dir / "locked_design.json")

    formal_ids = {
        "primary_and_homogeneous": replication_ids(
            "primary", design.primary_replications
        ),
        "finite_size": replication_ids(
            "finite-size", design.finite_size_replications
        ),
    }
    seed_manifest = {
        group: stream_seed_manifest(design.experiment_namespace, identifiers)
        for group, identifiers in formal_ids.items()
    }
    write_json(output_dir / "formal_seed_manifest.json", seed_manifest)

    manifest = {
        "manifest_version": "1.0",
        "status": "locked_before_confirmatory outcomes were generated",
        "source_commit": source_commit,
        "experiment_namespace": design.experiment_namespace,
        "selected_stress_budget": pilot_lock["selected_stress_budget"],
        "replication_groups": {
            group: len(identifiers) for group, identifiers in formal_ids.items()
        },
        "files": {
            filename: sha256_file(output_dir / filename)
            for filename in (
                "pilot_results.csv",
                "locked_design.json",
                "formal_seed_manifest.json",
            )
        },
        "pilot_source_files": {
            "pilot_results.csv": sha256_file(pilot_results_path),
            "locked_design.json": sha256_file(pilot_lock_path),
        },
    }
    write_json(output_dir / "design_lock_manifest.json", manifest)
    return manifest
