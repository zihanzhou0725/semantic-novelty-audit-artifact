#!/usr/bin/env python3
"""Validate the frozen external-behavior-v3 protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


PROTOCOL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROTOCOL_DIR.parents[1]
CLUTRR_ROOT = PROJECT_ROOT / "upstream" / "external_behavior_v1_clutrr"
V1_PROTOCOL_DIR = PROJECT_ROOT / "protocols" / "external-behavior-v1"
V1_MANIFEST_DIR = PROJECT_ROOT / "manifests" / "external-behavior-v1"


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


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(CLUTRR_ROOT), *args],
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
    inputs = load(PROTOCOL_DIR / "input_manifest.json")
    source = load(PROTOCOL_DIR / "source_manifest.json")
    receipt = load(PROTOCOL_DIR / "freeze_receipt.json")
    source_rows = PROJECT_ROOT / inputs["source_rows"]["path"]

    check(
        "v1_decisions_preserved",
        prereg["relationship_to_prior_versions"][
            "external_behavior_v1_clutrr_design_gate_remains"
        ]
        == "no_go"
        and prereg["relationship_to_prior_versions"][
            "retroactive_reclassification_allowed"
        ]
        is False,
    )
    check(
        "no_behavioral_outcome_before_freeze",
        receipt["state_at_freeze"]["v1_behavioral_training_started"] is False
        and receipt["state_at_freeze"][
            "v1_behavioral_outcome_inspected"
        ]
        is False
        and receipt["state_at_freeze"][
            "v3_expanded_candidate_rows_generated"
        ]
        is False,
    )
    check(
        "source_rows_hash_and_count",
        source_rows.is_file()
        and sha256(source_rows) == inputs["source_rows"]["sha256"]
        and line_count(source_rows) == inputs["source_rows"]["rows"],
    )
    check(
        "parent_hashes",
        sha256(V1_PROTOCOL_DIR / "preregistration.json")
        == inputs["parent_artifacts"][
            "external_behavior_v1_preregistration_sha256"
        ]
        and sha256(V1_MANIFEST_DIR / "materialization_failure_report.json")
        == inputs["parent_artifacts"][
            "external_behavior_v1_failure_report_sha256"
        ],
    )
    model_path = (
        PROJECT_ROOT
        / inputs["frozen_implementation_dependencies"][
            "byte_transformer_model_path"
        ]
    )
    check(
        "model_hash",
        sha256(model_path)
        == inputs["frozen_implementation_dependencies"][
            "byte_transformer_model_sha256"
        ],
    )
    check(
        "expansion_arithmetic",
        prereg["sentence_permutation_expansion"][
            "expanded_row_cap_before_deduplication"
        ]
        == inputs["source_rows"]["rows"] * 6,
    )
    check(
        "allocation_arithmetic",
        prereg["evaluation_allocation"]["rows_per_pattern"] == 33 * 6 + 2
        and prereg["exposure_allocation"]["rows_per_pattern"] == 10 * 6 + 4
        and prereg["target_selection"]["eligibility"][
            "distinct_complete_source_groups_at_least"
        ]
        == 34 + 11,
    )
    check(
        "arm_arithmetic",
        prereg["training_arms"]["episodes_per_arm"]
        == prereg["training_arms"]["common_base_episodes"]
        + prereg["training_arms"]["target_patterns_exposed_per_arm"]
        * prereg["training_arms"]["exposure_rows_per_exposed_pattern"]
        and prereg["model_and_optimization"]["run_count"] == 16,
    )
    relation_store = yaml.safe_load(
        (
            CLUTRR_ROOT / "clutrr/store/relations_store.yaml"
        ).read_text(encoding="utf-8")
    )
    templates = [
        template
        for relation in relation_store.values()
        for gender in relation.values()
        for template in gender["p"]
    ]
    check(
        "templates_have_no_internal_period",
        len(templates)
        == inputs["template_sources"]["relation_template_count"]
        and all("." not in template for template in templates),
        len(templates),
    )
    check(
        "upstream_revision_and_clean",
        git_output("rev-parse", "HEAD") == source["clutrr"]["revision"]
        and git_output("status", "--porcelain=v1") == "",
    )
    for relative, expected in source["clutrr"]["files"].items():
        path = CLUTRR_ROOT / relative
        check(
            f"source_hash:{relative}",
            path.is_file() and sha256(path) == expected,
        )
    if args.stage == "freeze":
        forbidden = [
            PROJECT_ROOT / "manifests" / "external-behavior-v3",
            PROJECT_ROOT / "runs" / "external-behavior-v3",
            PROJECT_ROOT / "analysis" / "external-behavior-v3",
        ]
        check(
            "no_v3_result_directories",
            not any(path.exists() for path in forbidden),
            [str(path) for path in forbidden if path.exists()],
        )

    failed = [item for item in checks if not item["pass"]]
    document_names = (
        "README.md",
        "checklist.md",
        "freeze_receipt.json",
        "input_manifest.json",
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
        "protocol_id": "external-behavior-v3",
        "schema_version": "external-behavior-v3-protocol-validation-v1",
        "stage": args.stage,
        "status": "pass" if not failed else "fail",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
