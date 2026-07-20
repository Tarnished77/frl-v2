"""Stable output helpers and checksum generation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().lower()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")


def stable_value(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return format(value, ".12g")
    return value


class StableCsvWriter:
    def __init__(self, path: Path, fieldnames: list[str]):
        self.path = path
        self.fieldnames = fieldnames
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(
            self._handle,
            fieldnames=fieldnames,
            extrasaction="raise",
            lineterminator="\n",
        )
        self._writer.writeheader()
        self.rows_written = 0

    def write(self, row: dict[str, Any]) -> None:
        self._writer.writerow(
            {key: stable_value(row[key]) for key in self.fieldnames}
        )
        self.rows_written += 1

    def write_many(self, rows: Iterable[dict[str, Any]]) -> None:
        for row in rows:
            self.write(row)

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "StableCsvWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def require_new_directory(path: Path) -> None:
    if path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output directory: {path}"
        )
    path.mkdir(parents=True)
