#!/usr/bin/env python3
"""Validate the frozen CLUTRR V3 execution protocol."""

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
V3_MANIFEST_DIR = PROJECT_ROOT / "manifests" / "external-behavior-v3"
RUN_ROOT = PROJECT_ROOT / "runs" / "external-behavior-v3"
EXECUTION_MANIFEST_DIR = V3_MANIFEST_DIR / "execution-v1"


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
    receipt = load(PROTOCOL_DIR / "freeze_receipt.json")
    design = load(V3_MANIFEST_DIR / "design_validation_report.json")
    design_sha = sha256(V3_MANIFEST_DIR / "design_validation_report.json")

    check(
        "design_gate_passed",
        design["status"] == "pass"
        and design["design_gate"] == "pass"
        and design["failed_count"] == 0,
    )
    check(
        "design_hash_bound",
        design_sha
        == prereg["frozen_design"]["design_validation_sha256"],
        design_sha,
    )
    check(
        "run_arithmetic",
        prereg["frozen_design"]["run_count"]
        == len(prereg["frozen_design"]["model_seeds"])
        * len(prereg["frozen_design"]["arms"])
        == 16,
    )
    check(
        "update_arithmetic",
        prereg["optimization"]["optimizer_updates_per_epoch"]
        == (
            prereg["frozen_design"]["train_rows_per_run"]
            + prereg["tensorization"]["train_batch_size"]
            - 1
        )
        // prereg["tensorization"]["train_batch_size"]
        and prereg["optimization"]["optimizer_updates_per_run"]
        == prereg["optimization"]["epochs"]
        * prereg["optimization"]["optimizer_updates_per_epoch"],
    )
    check(
        "paired_randomness_declared",
        prereg["randomness"]["model_initialization_namespace"]
        == "clutrr:model:{model_seed}"
        and prereg["randomness"]["epoch_order_namespace"]
        == "clutrr:epoch:{model_seed}:{epoch_1_based}"
        and prereg["randomness"]["deterministic_algorithms"] is True,
    )
    check(
        "fixed_shape_declared",
        prereg["tensorization"]["fixed_padding_length"] == 185
        and prereg["tensorization"]["truncation"] is False,
    )
    model_path = PROJECT_ROOT / "toolkit/external_behavior/model.py"
    check(
        "model_hash",
        sha256(model_path) == prereg["model"]["implementation_sha256"],
    )
    check(
        "effect_analysis_sealed",
        prereg["outcome_blinding"]["primary_effect_computed_in_this_stage"]
        is False
        and prereg["outcome_blinding"][
            "paired_arm_aggregate_computed_in_this_stage"
        ]
        is False,
    )
    check(
        "no_training_at_protocol_freeze",
        receipt["state_at_freeze"]["behavioral_training_started"] is False
        and receipt["state_at_freeze"]["run_metrics_inspected"] is False,
    )
    if args.stage == "freeze":
        check(
            "no_execution_manifests_or_runs",
            not EXECUTION_MANIFEST_DIR.exists()
            and (not RUN_ROOT.exists() or not any(RUN_ROOT.iterdir())),
            {
                "execution_manifest_exists": EXECUTION_MANIFEST_DIR.exists(),
                "run_root_nonempty": (
                    RUN_ROOT.exists() and any(RUN_ROOT.iterdir())
                ),
            },
        )

    failed = [item for item in checks if not item["pass"]]
    document_names = (
        "README.md",
        "checklist.md",
        "freeze_receipt.json",
        "preregistration.json",
        "validate.py",
    )
    report = {
        "check_count": len(checks),
        "checks": checks,
        "document_sha256": {
            name: sha256(PROTOCOL_DIR / name) for name in document_names
        },
        "execution_id": "external-behavior-v3-execution-v1",
        "failed_count": len(failed),
        "schema_version": "external-behavior-v3-execution-protocol-validation-v1",
        "stage": args.stage,
        "status": "pass" if not failed else "fail",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
