"""Run, validate, and atomically release the Mathematics collision census."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .collision_census import build_census
from .collision_common import (
    ANALYSIS_DIR,
    ANALYSIS_ID,
    IMPLEMENTATION_DIR,
    MODULES,
    PROJECT_ROOT,
    PROTOCOL_DIR,
    STAGING_DIR,
    read_json,
    sha256_file,
    verify_inputs,
    write_json,
)
from .collision_validate import validate_census


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_implementation_freeze() -> dict[str, Any]:
    path = IMPLEMENTATION_DIR / "implementation_manifest.json"
    if not path.is_file():
        raise RuntimeError("Collision-census implementation is not frozen.")
    manifest = read_json(path)
    if (
        manifest.get("status") != "frozen"
        or manifest.get("frozen_before_join") is not True
    ):
        raise RuntimeError("Invalid collision-census implementation freeze.")
    for relative, expected in manifest["implementation_sha256"].items():
        source = PROJECT_ROOT / relative
        if not source.is_file() or sha256_file(source) != expected:
            raise RuntimeError(f"Frozen census implementation changed: {relative}")
    if (
        sha256_file(IMPLEMENTATION_DIR / "environment.json")
        != manifest["environment_sha256"]
        or sha256_file(IMPLEMENTATION_DIR / "requirements.lock")
        != manifest["requirements_lock_sha256"]
        or sha256_file(PROTOCOL_DIR / "preregistration.json")
        != manifest["preregistration_sha256"]
    ):
        raise RuntimeError("Frozen census environment or protocol changed.")
    current_lock = sorted(
        line.strip()
        for line in subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            text=True,
        ).splitlines()
        if line.strip()
    )
    frozen_lock = sorted(
        line.strip()
        for line in (IMPLEMENTATION_DIR / "requirements.lock")
        .read_text(encoding="ascii")
        .splitlines()
        if line.strip()
    )
    if current_lock != frozen_lock:
        raise RuntimeError("Current environment differs from census lock.")
    return manifest


def _report_cn(results: dict[str, Any], validation: dict[str, Any]) -> str:
    lines = [
        "# Mathematics Collision Census v1",
        "",
        "## Gate",
        "",
        f"- Materialization and independent validation: "
        f"`{validation['status']}`",
        f"- Unresolved rows: `{results['overall']['unresolved_row_count']}`",
        "- Behavioral claim: not allowed",
        "",
        "## Module Results",
        "",
        "| Module | Eval rows | Verbatim overlap | Semantic overlap | "
        "Unique semantic overlap |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in results["by_module"]:
        lines.append(
            f"| `{row['module']}` | {row['interpolate_row_denominator']} | "
            f"{row['verbatim_question_overlap_row_count']} "
            f"({100 * row['verbatim_question_overlap_row_prevalence']:.4f}%) | "
            f"{row['canonical_semantic_overlap_row_count']} "
            f"({100 * row['canonical_semantic_overlap_row_prevalence']:.4f}%) | "
            f"{row['overlapping_unique_semantic_id_count']}/"
            f"{row['interpolate_unique_semantic_id_denominator']} "
            f"({100 * row['overlapping_unique_semantic_id_prevalence']:.4f}%) |"
        )
    overall = results["overall"]
    lines.extend(
        [
            "",
            "## Overall",
            "",
            f"- Verbatim overlap: "
            f"`{overall['verbatim_question_overlap_row_count']}/"
            f"{overall['interpolate_row_denominator']}` "
            f"({100 * overall['verbatim_question_overlap_row_prevalence']:.4f}%)",
            f"- Canonical semantic overlap: "
            f"`{overall['canonical_semantic_overlap_row_count']}/"
            f"{overall['interpolate_row_denominator']}` "
            f"({100 * overall['canonical_semantic_overlap_row_prevalence']:.4f}%)",
            f"- Unique semantic overlap: "
            f"`{overall['overlapping_unique_semantic_id_count']}/"
            f"{overall['interpolate_unique_semantic_id_denominator']}` "
            f"({100 * overall['overlapping_unique_semantic_id_prevalence']:.4f}%)",
            "",
            "All semantic collision rows have an independently verified exact "
            "symbolic zero-difference witness. Results apply only to the two "
            "regenerated modules under the frozen source revision and seeds.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_checksums(directory: Path) -> Path:
    paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    )
    checksum_path = directory / "SHA256SUMS"
    checksum_path.write_text(
        "".join(
            f"{sha256_file(path)}  {path.name}\n" for path in paths
        ),
        encoding="ascii",
    )
    return checksum_path


def run_and_release() -> dict[str, Any]:
    implementation = validate_implementation_freeze()
    inputs = verify_inputs()
    if ANALYSIS_DIR.exists() or STAGING_DIR.exists():
        raise RuntimeError("Collision-census output or staging already exists.")
    ANALYSIS_DIR.parent.mkdir(parents=True, exist_ok=True)

    build_census(STAGING_DIR)
    validation = validate_census(STAGING_DIR, write_artifacts=True)
    if validation["status"] != "pass":
        raise RuntimeError(
            f"Collision census failed independent validation: "
            f"{validation['failed_checks']}"
        )
    results = read_json(STAGING_DIR / "results.json")
    (STAGING_DIR / "report_cn.md").write_text(
        _report_cn(results, validation),
        encoding="utf-8",
    )
    output_paths = sorted(
        path for path in STAGING_DIR.iterdir() if path.is_file()
    )
    manifest = {
        "aggregate_released_at_utc": _now(),
        "analysis_id": ANALYSIS_ID,
        "implementation_manifest_sha256": sha256_file(
            IMPLEMENTATION_DIR / "implementation_manifest.json"
        ),
        "input_manifest_sha256": inputs["input_manifest_sha256"],
        "output_sha256": {
            path.name: sha256_file(path) for path in output_paths
        },
        "preregistration_sha256": sha256_file(
            PROTOCOL_DIR / "preregistration.json"
        ),
        "schema_version": "mathematics-collision-census-release-v1",
        "status": "released_after_independent_validation",
    }
    write_json(STAGING_DIR / "analysis_manifest.json", manifest)
    _write_checksums(STAGING_DIR)
    STAGING_DIR.replace(ANALYSIS_DIR)
    return {
        "analysis_dir": ANALYSIS_DIR.relative_to(PROJECT_ROOT).as_posix(),
        "analysis_id": ANALYSIS_ID,
        "overall": results["overall"],
        "status": "released",
        "validation_status": validation["status"],
    }


def validate_release_bundle() -> dict[str, Any]:
    failures: list[str] = []
    for line in (
        ANALYSIS_DIR / "SHA256SUMS"
    ).read_text(encoding="ascii").splitlines():
        expected, filename = line.split("  ", 1)
        path = ANALYSIS_DIR / filename
        if not path.is_file():
            failures.append(f"missing:{filename}")
        elif sha256_file(path) != expected:
            failures.append(f"hash:{filename}")
    manifest = read_json(ANALYSIS_DIR / "analysis_manifest.json")
    validation = read_json(ANALYSIS_DIR / "validation_report.json")
    results = read_json(ANALYSIS_DIR / "results.json")
    if manifest["status"] != "released_after_independent_validation":
        failures.append("manifest_status")
    if validation["status"] != "pass":
        failures.append("validation_status")
    if results["aggregate_release_status"] != "validated_for_atomic_release":
        failures.append("results_status")
    return {
        "analysis_id": ANALYSIS_ID,
        "failures": failures,
        "status": "pass" if not failures else "fail",
    }
