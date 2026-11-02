#!/usr/bin/env python3
"""Validate the frozen external-behavior-v2 amendment."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROTOCOL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROTOCOL_DIR.parents[1]
V1_PROTOCOL_DIR = PROJECT_ROOT / "protocols" / "external-behavior-v1"
V1_MANIFEST_DIR = PROJECT_ROOT / "manifests" / "external-behavior-v1"
UPSTREAM = PROJECT_ROOT / "upstream" / "external_behavior_v1_mathematics"


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


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(UPSTREAM), *args],
        text=True,
    ).strip()


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
    receipt = load(PROTOCOL_DIR / "amendment_receipt.json")
    source = load(PROTOCOL_DIR / "source_manifest.json")
    v1_prereg = load(V1_PROTOCOL_DIR / "preregistration.json")
    v1_failure = load(V1_MANIFEST_DIR / "materialization_failure_report.json")

    check(
        "parent_preregistration_hash",
        sha256(V1_PROTOCOL_DIR / "preregistration.json")
        == prereg["relationship_to_v1"]["parent_preregistration_sha256"],
    )
    check(
        "parent_failure_report_hash",
        sha256(V1_MANIFEST_DIR / "materialization_failure_report.json")
        == prereg["relationship_to_v1"]["parent_failure_report_sha256"],
    )
    check(
        "parent_decisions_preserved",
        prereg["relationship_to_v1"]["clutrr_design_gate_remains"] == "no_go"
        and prereg["relationship_to_v1"][
            "external_behavior_v1_suite_decision_remains"
        ]
        == "no_go"
        and prereg["relationship_to_v1"][
            "retroactive_reclassification_allowed"
        ]
        is False,
    )
    check(
        "no_overlap_result_available_at_amendment",
        receipt["evidence_available_at_freeze"][
            "mathematics_overlap_aggregate_available"
        ]
        is False
        and v1_failure["aggregate_mathematics_overlap_released"] is False,
    )
    check(
        "candidate_source_and_scope_unchanged",
        prereg["immutable_from_v1"]["candidate_id"]
        == v1_prereg["mathematics"]["candidate_id"]
        and prereg["immutable_from_v1"]["source_revision"]
        == v1_prereg["mathematics"]["source"]["revision"]
        and prereg["immutable_from_v1"]["modules"]
        == v1_prereg["mathematics"]["modules"],
    )
    expected_rows = 2 * (100000 + 10000)
    check(
        "sample_budget_is_220000",
        expected_rows == 220000
        and prereg["immutable_from_v1"]["regimes"][
            "train_examples_per_module"
        ]
        == 100000
        and prereg["immutable_from_v1"]["regimes"][
            "interpolate_examples_per_module"
        ]
        == 10000,
    )
    recovery = prereg["generation_recovery_rule"]
    check(
        "recovery_rule_is_narrow",
        recovery["recoverable_exception_class"] == "AssertionError"
        and recovery["required_terminal_frame_relative_path"]
        == "mathematics_dataset/sample/polynomials.py"
        and recovery["required_terminal_frame_line"] == 107
        and recovery["required_exception_message"] == ""
        and recovery["all_other_exceptions"] == "fail_immediately",
    )
    source_line = (
        UPSTREAM / recovery["required_terminal_frame_relative_path"]
    ).read_text(encoding="utf-8").splitlines()[
        recovery["required_terminal_frame_line"] - 1
    ]
    check(
        "registered_assertion_exists_at_pinned_line",
        source_line.strip() == "assert entropy == 0",
        source_line,
    )
    check(
        "v1_prefix_hash_matches_receipt",
        sha256(
            V1_MANIFEST_DIR
            / ".data.build"
            / "mathematics_train_polynomials__expand.jsonl.tmp"
        )
        == prereg["rerun_and_compatibility"][
            "v1_train_expand_prefix_sha256"
        ],
    )
    check(
        "upstream_revision",
        git_output("rev-parse", "HEAD") == source["mathematics"]["revision"],
        git_output("rev-parse", "HEAD"),
    )
    check(
        "upstream_checkout_clean",
        git_output("status", "--porcelain=v1") == "",
        git_output("status", "--porcelain=v1"),
    )
    for relative, expected in source["mathematics"]["files"].items():
        path = UPSTREAM / relative
        check(
            f"source_hash:{relative}",
            path.is_file() and sha256(path) == expected,
        )
    if args.stage == "freeze":
        forbidden = [
            PROJECT_ROOT / "manifests" / "external-behavior-v2",
            PROJECT_ROOT / "runs" / "external-behavior-v2",
            PROJECT_ROOT / "analysis" / "external-behavior-v2",
        ]
        check(
            "no_v2_result_directories",
            not any(path.exists() for path in forbidden),
            [str(path) for path in forbidden if path.exists()],
        )

    failed = [item for item in checks if not item["pass"]]
    document_names = (
        "README.md",
        "amendment_receipt.json",
        "checklist.md",
        "preregistration.json",
        "source_manifest.json",
        "validate.py",
    )
    report = {
        "check_count": len(checks),
        "checks": checks,
        "document_sha256": {
            name: sha256(PROTOCOL_DIR / name) for name in document_names
        },
        "failed_count": len(failed),
        "protocol_id": "external-behavior-v2",
        "schema_version": "external-behavior-v2-protocol-validation-v1",
        "stage": args.stage,
        "status": "pass" if not failed else "fail",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
