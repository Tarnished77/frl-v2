"""Verify the public code package and optionally reproduce formal results."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formal-data-dir",
        type=Path,
        help="Path to the extracted formal/ directory from the separate data archive.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reproduced",
        help="New directory for replayed sample and formal outputs.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def run(command: list[str], cwd: Path = ROOT) -> None:
    print("+", " ".join(command), flush=True)
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(command, cwd=cwd, check=True, env=environment)


def verify_checksum_list(package_root: Path) -> None:
    checksum_path = package_root / "checksums.sha256"
    if not checksum_path.is_file():
        raise FileNotFoundError(f"Checksum list is missing: {checksum_path}")
    checked = 0
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", maxsplit=1)
        path = package_root / Path(relative)
        if not path.is_file():
            raise FileNotFoundError(f"Package payload is missing: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"Checksum mismatch for {relative}: {actual} != {expected}")
        checked += 1
    print(f"Verified {checked} payload files in {package_root.name}.", flush=True)


def compare_manifest_files(manifest_path: Path, candidate_dir: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for filename, expected in manifest["files"].items():
        actual = sha256_file(candidate_dir / filename)
        if actual != expected:
            raise ValueError(f"Reproduced hash mismatch for {filename}: {actual} != {expected}")


def compare_file(expected: Path, actual: Path) -> None:
    expected_hash = sha256_file(expected)
    actual_hash = sha256_file(actual)
    if actual_hash != expected_hash:
        raise ValueError(
            f"Reproduced file differs: {actual} ({actual_hash} != {expected_hash})"
        )


def replay_sample(output_dir: Path) -> None:
    expected = ROOT / "data" / "sample" / "smoke-v1"
    replay = output_dir / "sample-smoke-v1"
    run(
        [
            sys.executable,
            "run_pipeline.py",
            "smoke",
            "--output-dir",
            str(replay),
        ]
    )
    for filename in (
        "agent_survival.csv",
        "run_daily.csv",
        "run_summary.csv",
        "seed_manifest.json",
        "experiment_manifest.json",
    ):
        compare_file(expected / filename, replay / filename)
    print("Sample simulation replay is byte-identical.", flush=True)


def locate_data_package_root(formal_data_dir: Path) -> Path:
    formal_data_dir = formal_data_dir.resolve()
    required = [formal_data_dir / experiment for experiment in (
        "primary-v1",
        "homogeneous-v1",
        "finite-50-v1",
        "finite-200-v1",
    )]
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Formal data directories are missing: {missing}")
    package_root = formal_data_dir.parent
    verify_checksum_list(package_root)
    return package_root


def reproduce_formal(formal_data_dir: Path, output_dir: Path) -> None:
    locate_data_package_root(formal_data_dir)
    locked_results = json.loads(
        (ROOT / "results" / "results_manifest.json").read_text(encoding="utf-8")
    )
    reproduced_results = output_dir / "results"
    run(
        [
            sys.executable,
            "analyze_results.py",
            "--primary-dir",
            str(formal_data_dir / "primary-v1"),
            "--homogeneous-dir",
            str(formal_data_dir / "homogeneous-v1"),
            "--finite-50-dir",
            str(formal_data_dir / "finite-50-v1"),
            "--finite-200-dir",
            str(formal_data_dir / "finite-200-v1"),
            "--output-dir",
            str(reproduced_results),
            "--analysis-source-commit",
            locked_results["analysis_source_commit"],
        ]
    )
    compare_manifest_files(ROOT / "results" / "results_manifest.json", reproduced_results)
    compare_file(
        ROOT / "results" / "results_manifest.json",
        reproduced_results / "results_manifest.json",
    )

    locked_figures = json.loads(
        (ROOT / "figures" / "figure_manifest.json").read_text(encoding="utf-8")
    )
    reproduced_figures = output_dir / "figures"
    run(
        [
            sys.executable,
            "plot_results.py",
            "--results-dir",
            str(reproduced_results),
            "--design-dir",
            "design",
            "--output-dir",
            str(reproduced_figures),
            "--source-commit",
            locked_figures["source_commit"],
        ]
    )
    compare_manifest_files(ROOT / "figures" / "figure_manifest.json", reproduced_figures)
    compare_file(
        ROOT / "figures" / "figure_manifest.json",
        reproduced_figures / "figure_manifest.json",
    )
    print("Formal analysis and figures are byte-identical.", flush=True)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)

    verify_checksum_list(ROOT)
    run([sys.executable, "-m", "unittest", "discover", "tests", "-v"])
    replay_sample(output_dir)

    if args.formal_data_dir is None:
        print(
            "Code, tests, and sample replay passed. To reproduce the formal "
            "analysis, extract the separate data archive and rerun with "
            "--formal-data-dir PATH_TO_FORMAL_DIRECTORY.",
            flush=True,
        )
    else:
        reproduce_formal(args.formal_data_dir.resolve(), output_dir)
    print("Reproduction checks completed successfully.", flush=True)


if __name__ == "__main__":
    main()
