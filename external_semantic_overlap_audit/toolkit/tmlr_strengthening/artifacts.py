"""Deterministic output helpers for strengthening analyses."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Iterable, Mapping


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )


def write_csv(path: Path, rows: list[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_macros(path: Path, macros: Mapping[str, str]) -> None:
    lines = []
    for name, value in macros.items():
        if not name.isalpha():
            raise ValueError(f"LaTeX macro names must contain only letters: {name}")
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def run_info(
    *,
    seed: int | list[int] | None,
    script_path: str,
    input_paths: Iterable[Path],
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    working_directory = Path.cwd().resolve()

    def portable_path(path: Path) -> str:
        resolved = path.resolve()
        try:
            return str(resolved.relative_to(working_directory))
        except ValueError:
            return path.name

    packages = {}
    for name in ("matplotlib", "numpy", "scipy", "torch"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    result: dict[str, object] = {
        "input_sha256": {
            portable_path(path): sha256_file(path)
            for path in sorted(input_paths, key=str)
        },
        "packages": packages,
        "platform": platform.platform(),
        "python": sys.version,
        "script": script_path,
        "seed": seed,
        "utc_time": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        result.update(extra)
    return result


def write_sha256sums(directory: Path) -> None:
    entries = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            entries.append(f"{sha256_file(path)}  {path.relative_to(directory)}")
    (directory / "SHA256SUMS").write_text(
        "\n".join(entries) + "\n",
        encoding="ascii",
    )
