#!/usr/bin/env python3
"""Freeze the P2 prediction-source census before any new confirmatory join."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quality_p1_common import EXTERNAL_ROOT, PROTOCOL_DIR, ROOT, sha256_file, write_json


SCHEMA_VERSION = "quality-enhancement-p2-sources-v1"
P2_PROTOCOL_DIR = PROTOCOL_DIR / "p2"
RAW_OUTPUT = EXTERNAL_ROOT / "upstream" / "llama31_8b_slr_model_outputs.json"
STRATIFICATION = (
    EXTERNAL_ROOT / "artifacts" / "slr_official_output_stratification.json"
)
PER_ITEM = EXTERNAL_ROOT / "artifacts" / "slr_official_output_per_item.csv"


def fail(message: str) -> None:
    raise SystemExit(f"P2 source freeze failed: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def validate_parent() -> dict[str, Any]:
    preregistration_path = PROTOCOL_DIR / "preregistration.json"
    validation = json.loads(
        (PROTOCOL_DIR / "validation_report.json").read_text(encoding="ascii")
    )
    require(validation.get("status") == "passed", "parent validation failed")
    require(
        validation.get("preregistration_sha256")
        == sha256_file(preregistration_path),
        "parent validation refers to another preregistration",
    )
    preregistration = json.loads(
        preregistration_path.read_text(encoding="ascii")
    )
    for pin in preregistration["source_pins"]:
        path = ROOT / pin["path"]
        require(path.is_file(), f"missing parent source pin: {pin['path']}")
        require(
            sha256_file(path) == pin["sha256"],
            f"parent source pin changed: {pin['path']}",
        )
    return preregistration


def main() -> int:
    preregistration = validate_parent()
    manifest_path = P2_PROTOCOL_DIR / "source_manifest.json"
    checksum_path = P2_PROTOCOL_DIR / "SHA256SUMS"
    require(not manifest_path.exists(), "source manifest already exists")
    require(not checksum_path.exists(), "P2 checksum manifest already exists")
    for path in (RAW_OUTPUT, STRATIFICATION, PER_ITEM):
        require(path.is_file(), f"missing source artifact: {path}")

    stratification = json.loads(STRATIFICATION.read_text(encoding="utf-8"))
    source = stratification["source"]
    crosscheck = stratification["paper_crosscheck"]
    raw_hash = sha256_file(RAW_OUTPUT)
    require(
        raw_hash == source["model_outputs_sha256"],
        "raw repository output hash differs from the pinned crosscheck",
    )
    require(source["output_rows"] == 1000, "repository output row count")
    require(
        source["test_alignment"]
        == "problem_id and ground-truth rule match 100%",
        "repository output is not aligned to the complete test split",
    )
    require(
        crosscheck["behavioral_use_status"]
        == "exploratory_only_not_reported_main_table_behavior",
        "pinned output eligibility status changed",
    )
    require(
        preregistration["p2"]["known_status_at_freeze"][
            "published_main_table_reproduced_by_pinned_output"
        ]
        is False,
        "parent protocol no longer records the known source mismatch",
    )

    raw_rows = json.loads(RAW_OUTPUT.read_text(encoding="utf-8"))
    require(len(raw_rows) == 1000, "raw repository output is not 1,000 rows")
    require(
        {row.get("model_name") for row in raw_rows} == {"merged"},
        "raw output model labels changed",
    )
    require(
        all("model_revision" not in row for row in raw_rows),
        "raw output unexpectedly gained checkpoint revisions",
    )
    require(
        all("generation_config" not in row for row in raw_rows),
        "raw output unexpectedly gained complete generation configs",
    )

    manifest = {
        "confirmatory_target": {
            "checkpoint_result": "ACL 2026 final paper Table 2",
            "model": "Llama-3.1-8B-IT-SLR",
            "substitution_allowed": False,
        },
        "freeze_script_sha256": sha256_file(Path(__file__).resolve()),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "new_prediction_overlap_or_correctness_join_performed": False,
        "parent_preregistration_sha256": sha256_file(
            PROTOCOL_DIR / "preregistration.json"
        ),
        "schema_version": SCHEMA_VERSION,
        "selection_decision": {
            "eligible_confirmatory_source_count": 0,
            "status": "no_reproducible_confirmatory_prediction_source",
        },
        "sources": [
            {
                "artifact": relative(RAW_OUTPUT),
                "artifact_sha256": raw_hash,
                "candidate_identity": "repository_hosted_AIML-TUDA_Llama-3.1-8B-SLR",
                "confirmatory_target_eligible": False,
                "failure_reasons": [
                    "checkpoint_revision_not_cryptographically_bound",
                    "complete_decoding_configuration_not_present",
                    "artifact_regeneration_not_demonstrated",
                    "acl_2026_final_table_2_tier_counts_not_reproduced",
                ],
                "known_at_parent_freeze": True,
                "output_rows": 1000,
                "supporting_artifacts": {
                    relative(STRATIFICATION): sha256_file(STRATIFICATION),
                    relative(PER_ITEM): sha256_file(PER_ITEM),
                },
                "test_id_and_rule_alignment": "100_percent",
                "use": "exploratory_only",
            }
        ],
        "status": "frozen_before_new_confirmatory_prediction_overlap_or_correctness_join",
    }
    P2_PROTOCOL_DIR.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, manifest)
    checksum_path.write_text(
        f"{sha256_file(manifest_path)}  source_manifest.json\n",
        encoding="ascii",
    )
    print(
        json.dumps(
            {
                "eligible_confirmatory_source_count": 0,
                "source_manifest": relative(manifest_path),
                "source_manifest_sha256": sha256_file(manifest_path),
                "status": manifest["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
