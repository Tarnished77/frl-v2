"""Verify and reproduce the FRL v3.1 public computational package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
PNG_MAX_DIMENSION_DELTA = 1
PNG_MAX_MEAN_ABSOLUTE_ERROR = 2.0
PNG_MAX_HIGH_DIFFERENCE_FRACTION = 0.015
PDF_MAX_MEDIA_BOX_DELTA_POINTS = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reproduced",
        help="New directory for replayed outputs.",
    )
    parser.add_argument(
        "--reanalyze",
        action="store_true",
        help="Rerun formal block-level analysis before rebuilding publication assets.",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip unit tests; checksums and deterministic replays still run.",
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


def verify_checksums() -> None:
    checksum_path = ROOT / "checksums.sha256"
    if not checksum_path.is_file():
        raise FileNotFoundError(checksum_path)
    checked = 0
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", maxsplit=1)
        path = ROOT / Path(relative)
        if not path.is_file():
            raise FileNotFoundError(f"Missing package payload: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"Checksum mismatch for {relative}: {actual} != {expected}"
            )
        checked += 1
    print(f"Verified {checked} package payload files.", flush=True)


def compare_file(expected: Path, actual: Path) -> None:
    expected_hash = sha256_file(expected)
    actual_hash = sha256_file(actual)
    if actual_hash != expected_hash:
        raise ValueError(
            f"Reproduced file differs: {actual} "
            f"({actual_hash} != {expected_hash})"
        )


def compare_png_pixels(expected: Path, actual: Path) -> None:
    with (
        Image.open(expected) as expected_image,
        Image.open(actual) as actual_image,
    ):
        width_delta = abs(expected_image.width - actual_image.width)
        height_delta = abs(expected_image.height - actual_image.height)
        if (
            width_delta > PNG_MAX_DIMENSION_DELTA
            or height_delta > PNG_MAX_DIMENSION_DELTA
        ):
            raise ValueError(
                f"Reproduced PNG geometry differs: {actual} "
                f"{actual_image.size} != {expected_image.size}"
            )
        common_width = min(expected_image.width, actual_image.width)
        common_height = min(expected_image.height, actual_image.height)
        crop = (0, 0, common_width, common_height)
        expected_rgb = np.asarray(
            expected_image.convert("RGB").crop(crop),
            dtype=np.int16,
        )
        actual_rgb = np.asarray(
            actual_image.convert("RGB").crop(crop),
            dtype=np.int16,
        )
        difference = np.abs(expected_rgb - actual_rgb)
        mean_absolute_error = float(difference.mean())
        high_difference_fraction = float(
            np.any(difference > 64, axis=2).mean()
        )
        if (
            mean_absolute_error > PNG_MAX_MEAN_ABSOLUTE_ERROR
            or high_difference_fraction > PNG_MAX_HIGH_DIFFERENCE_FRACTION
        ):
            raise ValueError(
                f"Reproduced PNG is not visually equivalent: {actual}; "
                f"mean absolute RGB error={mean_absolute_error:.6f}, "
                "high-difference pixel fraction="
                f"{high_difference_fraction:.6f}"
            )


def pdf_structure(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    if not payload.startswith(b"%PDF-"):
        raise ValueError(f"Not a PDF file: {path}")
    if len(payload) < 1_000:
        raise ValueError(f"Unexpectedly small PDF file: {path}")
    media_boxes = [
        tuple(
            float(value)
            for value in match.decode("ascii", errors="strict").split()
        )
        for match in re.findall(rb"/MediaBox\s*\[([^\]]+)\]", payload)
    ]
    return {
        "page_objects": payload.count(b"/Type /Page"),
        "media_boxes": media_boxes,
        "embedded_truetype_fonts": payload.count(b"/FontFile2"),
        "embedded_cff_fonts": payload.count(b"/FontFile3"),
    }


def compare_pdf_structure(expected: Path, actual: Path) -> None:
    expected_structure = pdf_structure(expected)
    actual_structure = pdf_structure(actual)
    for key in (
        "page_objects",
        "embedded_truetype_fonts",
        "embedded_cff_fonts",
    ):
        if actual_structure[key] != expected_structure[key]:
            raise ValueError(
                f"Reproduced PDF structure differs: {actual} "
                f"({actual_structure} != {expected_structure})"
            )
    expected_boxes = expected_structure["media_boxes"]
    actual_boxes = actual_structure["media_boxes"]
    if len(actual_boxes) != len(expected_boxes):
        raise ValueError(
            f"Reproduced PDF structure differs: {actual} "
            f"({actual_structure} != {expected_structure})"
        )
    for expected_box, actual_box in zip(expected_boxes, actual_boxes):
        if len(actual_box) != len(expected_box) or any(
            abs(actual_value - expected_value)
            > PDF_MAX_MEDIA_BOX_DELTA_POINTS
            for expected_value, actual_value in zip(expected_box, actual_box)
        ):
            raise ValueError(
                f"Reproduced PDF page geometry differs: {actual} "
                f"({actual_boxes} != {expected_boxes})"
            )
    if not (
        actual_structure["embedded_truetype_fonts"]
        or actual_structure["embedded_cff_fonts"]
    ):
        raise ValueError(f"Reproduced PDF has no embedded font program: {actual}")


def replay_smoke(output_root: Path) -> None:
    expected = ROOT / "data" / "sample" / "resilience-smoke-v1"
    candidate = output_root / "resilience-smoke-replay"
    run(
        [
            sys.executable,
            "run_resilience.py",
            "smoke",
            "--output-dir",
            str(candidate),
        ]
    )
    for filename in (
        "agent_survival.csv",
        "run_daily.csv",
        "run_summary.csv",
        "seed_manifest.json",
        "experiment_manifest.json",
    ):
        compare_file(expected / filename, candidate / filename)
    print("Smoke experiment replay is byte-identical.", flush=True)


def publication_command(
    output_dir: Path,
    *,
    primary: Path,
    high_impact: Path,
    frequency: Path,
    cash: Path,
    budget: Path,
) -> list[str]:
    return [
        sys.executable,
        "build_publication.py",
        "--output-dir",
        str(output_dir),
        "--primary",
        str(primary),
        "--high-impact",
        str(high_impact),
        "--frequency",
        str(frequency),
        "--cash",
        str(cash),
        "--budget",
        str(budget),
        "--finite-50",
        str(ROOT / "data" / "formal" / "finite" / "n50"),
        "--finite-100",
        str(ROOT / "data" / "formal" / "finite" / "n100"),
        "--finite-200",
        str(ROOT / "data" / "formal" / "finite" / "n200"),
        "--calibration-audit",
        str(
            ROOT
            / "resilience_design"
            / "calibration_audit_summary.csv"
        ),
    ]


def compare_publication(candidate: Path, *, exact_manifest: bool) -> None:
    expected_root = ROOT / "expected" / "publication"
    expected_manifest_path = expected_root / "publication_manifest.json"
    candidate_manifest_path = candidate / "publication_manifest.json"
    expected_manifest = json.loads(
        expected_manifest_path.read_text(encoding="utf-8")
    )
    candidate_manifest = json.loads(
        candidate_manifest_path.read_text(encoding="utf-8")
    )
    metadata_exclusions = {"files"}
    if not exact_manifest:
        metadata_exclusions.update(
            {
                "primary_results_manifest_sha256",
                "sensitivity_results_manifest_sha256",
            }
        )
    expected_metadata = {
        key: value
        for key, value in expected_manifest.items()
        if key not in metadata_exclusions
    }
    candidate_metadata = {
        key: value
        for key, value in candidate_manifest.items()
        if key not in metadata_exclusions
    }
    if candidate_metadata != expected_metadata:
        raise ValueError("Publication manifest metadata differs.")
    if set(candidate_manifest["files"]) != set(expected_manifest["files"]):
        raise ValueError("Publication manifest file inventory differs.")

    for relative, expected_hash in expected_manifest["files"].items():
        expected = expected_root / relative
        actual = candidate / relative
        actual_hash = sha256_file(actual).lower()
        if candidate_manifest["files"][relative] != actual_hash:
            raise ValueError(
                f"Publication manifest is inconsistent for {relative}: "
                f"{candidate_manifest['files'][relative]} != {actual_hash}"
            )
        suffix = Path(relative).suffix.lower()
        if suffix == ".png":
            compare_png_pixels(expected, actual)
        elif suffix == ".pdf":
            compare_pdf_structure(expected, actual)
        elif actual_hash != expected_hash:
            raise ValueError(
                f"Publication payload mismatch for {relative}: "
                f"{actual_hash} != {expected_hash}"
            )

    qualifier = "locked" if exact_manifest else "reanalyzed"
    print(
        f"Publication payload ({qualifier}) has byte-identical data/text, "
        "visually equivalent PNGs, and matching vector-PDF structure.",
        flush=True,
    )


def replay_locked_publication(output_root: Path) -> None:
    candidate = output_root / "publication-from-locked-derived"
    run(
        publication_command(
            candidate,
            primary=ROOT / "derived-resilience-primary-v1",
            high_impact=(
                ROOT / "derived-resilience-sensitivity-high-impact-v1"
            ),
            frequency=(
                ROOT / "derived-resilience-sensitivity-frequency-v1"
            ),
            cash=ROOT / "derived-resilience-sensitivity-cash-v1",
            budget=ROOT / "derived-resilience-sensitivity-budget-v2",
        )
    )
    compare_publication(candidate, exact_manifest=True)


def analyze(
    kind: str,
    inputs: list[Path],
    output_dir: Path,
) -> None:
    if kind == "primary":
        command = [
            sys.executable,
            "analyze_resilience.py",
            "primary",
        ]
    else:
        command = [
            sys.executable,
            "analyze_resilience.py",
            "sensitivity",
            kind,
        ]
    for input_dir in inputs:
        command.extend(["--input-dir", str(input_dir)])
    command.extend(
        [
            "--output-dir",
            str(output_dir),
            "--locked-design",
            str(
                ROOT
                / "resilience_design"
                / "locked_resilience_design.json"
            ),
        ]
    )
    run(command)


def compare_analysis_payload(
    expected: Path,
    actual: Path,
    filenames: tuple[str, ...],
) -> None:
    for filename in filenames:
        compare_file(expected / filename, actual / filename)


def rerun_formal_analysis(output_root: Path) -> None:
    analysis_root = output_root / "analysis"
    primary = analysis_root / "derived-resilience-primary-v1"
    high_impact = analysis_root / "derived-resilience-sensitivity-high-impact-v1"
    frequency = analysis_root / "derived-resilience-sensitivity-frequency-v1"
    cash = analysis_root / "derived-resilience-sensitivity-cash-v1"
    budget = analysis_root / "derived-resilience-sensitivity-budget-v2"

    analyze(
        "primary",
        [ROOT / "data" / "formal" / "primary"],
        primary,
    )
    analyze(
        "high_impact",
        [ROOT / "data" / "formal" / "sensitivity" / "high_impact"],
        high_impact,
    )
    analyze(
        "market_frequency",
        [ROOT / "data" / "formal" / "sensitivity" / "market_frequency"],
        frequency,
    )
    analyze(
        "cash_buffer",
        [ROOT / "data" / "formal" / "sensitivity" / "cash_buffer"],
        cash,
    )
    budget_root = ROOT / "data" / "formal" / "sensitivity" / "stress_budget"
    analyze(
        "stress_budget",
        [
            budget_root / "budget-0p10",
            budget_root / "budget-0p15",
            budget_root / "budget-0p30",
        ],
        budget,
    )

    compare_analysis_payload(
        ROOT / "derived-resilience-primary-v1",
        primary,
        (
            "mechanism_validation.json",
            "precision_decision.json",
            "resilience_cells.csv",
            "resilience_contrasts.csv",
        ),
    )
    for expected_name, candidate in (
        ("derived-resilience-sensitivity-high-impact-v1", high_impact),
        ("derived-resilience-sensitivity-frequency-v1", frequency),
        ("derived-resilience-sensitivity-cash-v1", cash),
        ("derived-resilience-sensitivity-budget-v2", budget),
    ):
        compare_analysis_payload(
            ROOT / expected_name,
            candidate,
            ("sensitivity_cells.csv", "sensitivity_contrasts.csv"),
        )
    print("Formal block-level analysis payload is byte-identical.", flush=True)

    publication = output_root / "publication-from-reanalysis"
    run(
        publication_command(
            publication,
            primary=primary,
            high_impact=high_impact,
            frequency=frequency,
            cash=cash,
            budget=budget,
        )
    )
    compare_publication(publication, exact_manifest=False)


def main() -> None:
    args = parse_args()
    output_root = args.output_dir.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {output_root}")

    verify_checksums()
    output_root.mkdir(parents=True)
    if not args.skip_tests:
        run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
    replay_smoke(output_root)
    replay_locked_publication(output_root)
    if args.reanalyze:
        rerun_formal_analysis(output_root)
    print("All requested reproducibility checks passed.", flush=True)


if __name__ == "__main__":
    main()
