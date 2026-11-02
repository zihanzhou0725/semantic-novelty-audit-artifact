#!/usr/bin/env python3
"""Validate the frozen Mathematics collision-census protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROTOCOL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROTOCOL_DIR.parents[2]
V2_MANIFEST_DIR = PROJECT_ROOT / "manifests" / "external-behavior-v2"
ANALYSIS_DIR = (
    PROJECT_ROOT
    / "analysis"
    / "external-behavior-v2"
    / "mathematics-collision-census-v1"
)
IMPLEMENTATION_DIR = (
    V2_MANIFEST_DIR / "mathematics-collision-census-v1"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("freeze", "execution"),
        default="freeze",
    )
    args = parser.parse_args()
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: Any = None) -> None:
        checks.append({"detail": detail, "name": name, "pass": bool(condition)})

    prereg = load(PROTOCOL_DIR / "preregistration.json")
    inputs = load(PROTOCOL_DIR / "input_manifest.json")
    receipt = load(PROTOCOL_DIR / "freeze_receipt.json")
    parent_validation = load(
        V2_MANIFEST_DIR / "materialization_validation_report.json"
    )

    check(
        "parent_materialization_passed",
        parent_validation["status"] == "pass"
        and parent_validation["materialization_gate"] == "pass"
        and parent_validation["failed_count"] == 0,
    )
    check(
        "parent_validation_hash",
        sha256(V2_MANIFEST_DIR / "materialization_validation_report.json")
        == inputs["materialization_validation_report_sha256"],
    )
    check(
        "parent_aggregate_was_sealed",
        parent_validation["aggregate_overlap_released"] is False
        and receipt["state_at_freeze"]["train_interpolate_join_run"] is False
        and receipt["state_at_freeze"][
            "semantic_overlap_count_inspected"
        ]
        is False,
    )
    check(
        "module_isolation",
        prereg["scope"]["cross_module_matching_forbidden"] is True,
    )
    check(
        "denominators_are_separate",
        prereg["identity_definitions"]["evaluation_unit"]
        == "One retained interpolate row."
        and prereg["identity_definitions"]["unique_semantic_unit"]
        == "One distinct interpolate semantic_id within a module.",
    )
    check(
        "input_row_arithmetic",
        sum(item["rows"] for item in inputs["streams"]) == 220000
        and sum(
            item["rows"]
            for item in inputs["streams"]
            if item["regime"] == "interpolate"
        )
        == 20000,
    )
    observed_counts: dict[str, int] = {}
    for stream in inputs["streams"]:
        path = PROJECT_ROOT / stream["path"]
        observed = line_count(path) if path.is_file() else -1
        observed_counts[stream["path"]] = observed
        check(
            f"input:{stream['regime']}:{stream['module']}",
            path.is_file()
            and sha256(path) == stream["sha256"]
            and observed == stream["rows"],
        )
    check(
        "symbolic_verifier_is_required",
        prereg["independent_symbolic_verifier"]["applies_to"]
        == "Every semantically overlapping interpolate row and its selected train witness."
        and "exact integer zero"
        in prereg["independent_symbolic_verifier"]["test"],
    )
    check(
        "claim_scope_is_bounded",
        prereg["scope"]["behavioral_claim_allowed"] is False
        and len(prereg["claim_gate"]["forbidden"]) >= 4,
    )
    if args.stage == "freeze":
        check(
            "no_analysis_or_implementation_outputs",
            not ANALYSIS_DIR.exists() and not IMPLEMENTATION_DIR.exists(),
            {
                "analysis_exists": ANALYSIS_DIR.exists(),
                "implementation_exists": IMPLEMENTATION_DIR.exists(),
            },
        )

    failed = [item for item in checks if not item["pass"]]
    document_names = (
        "README.md",
        "checklist.md",
        "freeze_receipt.json",
        "input_manifest.json",
        "preregistration.json",
        "validate.py",
    )
    report = {
        "analysis_id": "mathematics-collision-census-v1",
        "check_count": len(checks),
        "checks": checks,
        "document_sha256": {
            name: sha256(PROTOCOL_DIR / name) for name in document_names
        },
        "failed_count": len(failed),
        "input_line_counts": observed_counts,
        "schema_version": "mathematics-collision-census-protocol-validation-v1",
        "stage": args.stage,
        "status": "pass" if not failed else "fail",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
