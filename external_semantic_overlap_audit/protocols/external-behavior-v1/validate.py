#!/usr/bin/env python3
"""Validate the frozen external-behavior-v1 preregistration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROTOCOL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROTOCOL_DIR.parents[1]
UPSTREAM_ROOT = PROJECT_ROOT / "upstream"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(name: str) -> dict[str, Any]:
    with (PROTOCOL_DIR / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("freeze", "execution"),
        default="freeze",
        help="Freeze additionally requires that no result directories exist.",
    )
    args = parser.parse_args()

    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: Any = None) -> None:
        checks.append({"name": name, "pass": bool(condition), "detail": detail})

    candidate = load_json("candidate_manifest.json")
    receipt = load_json("candidate_freeze_receipt.json")
    prereg = load_json("preregistration.json")
    sources = load_json("source_manifest.json")

    check(
        "candidate_manifest_hash_matches_receipt",
        sha256(PROTOCOL_DIR / "candidate_manifest.json")
        == receipt["candidate_manifest_sha256"],
    )
    check(
        "availability_screening_hash_matches_receipt",
        sha256(PROTOCOL_DIR / "availability_screening.csv")
        == receipt["availability_screening_sha256"],
    )
    check("candidate_count_is_two", candidate["candidate_count"] == 2)
    check(
        "candidate_selection_was_result_independent",
        candidate["result_dependent_selection_used"] is False,
    )
    check(
        "candidate_freeze_precedes_result_actions",
        all(value is False for value in receipt["state_at_freeze"].values()),
    )
    check(
        "protocol_is_independent_expansion",
        prereg["relationship_to_prior_protocol"][
            "independent_confirmatory_expansion"
        ]
        is True
        and prereg["relationship_to_prior_protocol"][
            "retroactive_p3_reclassification_allowed"
        ]
        is False,
    )

    candidate_by_id = {
        item["candidate_id"]: item for item in candidate["candidates"]
    }
    check(
        "candidate_ids_match_preregistration",
        set(candidate_by_id)
        == {prereg["clutrr"]["candidate_id"], prereg["mathematics"]["candidate_id"]},
    )
    check(
        "source_revisions_match_candidate_manifest",
        prereg["clutrr"]["source"]["revision"]
        == candidate_by_id[prereg["clutrr"]["candidate_id"]]["code_revision"][
            "revision"
        ]
        and prereg["mathematics"]["source"]["revision"]
        == candidate_by_id[prereg["mathematics"]["candidate_id"]][
            "code_revision"
        ]["revision"],
    )

    clutrr = prereg["clutrr"]
    check(
        "clutrr_target_count_arithmetic",
        clutrr["target_selection"]["required_target_patterns"]
        == 2 * clutrr["target_selection"]["required_pattern_pairs"],
    )
    check(
        "clutrr_evaluation_count_arithmetic",
        clutrr["evaluation"]["rows_total"]
        == clutrr["evaluation"]["target_patterns"]
        * clutrr["evaluation"]["rows_per_pattern"],
    )
    check(
        "clutrr_treatment_count_arithmetic",
        clutrr["training_arms"]["target_exposure_episodes_per_arm"]
        == clutrr["training_arms"]["target_patterns_exposed_per_arm"]
        * clutrr["training_arms"]["exposure_rows_per_exposed_pattern"],
    )
    check(
        "clutrr_training_count_arithmetic",
        clutrr["training_arms"]["episodes_per_arm"]
        == clutrr["training_arms"]["common_base_episodes"]
        + clutrr["training_arms"]["target_exposure_episodes_per_arm"],
    )
    check(
        "clutrr_run_count_arithmetic",
        clutrr["optimization"]["run_count"]
        == len(clutrr["optimization"]["model_seeds"])
        * len(clutrr["training_arms"]["arms"]),
    )
    check(
        "clutrr_effect_direction_not_suite_gate",
        prereg["suite_gate"]["effect_direction_is_not_a_suite_gate"] is True,
    )

    mathematics = prereg["mathematics"]
    expected_math_rows = len(mathematics["modules"]) * sum(
        regime["examples_per_module"]
        for regime in mathematics["regimes"].values()
    )
    check(
        "mathematics_total_count_arithmetic",
        mathematics["generation"]["total_retained_examples"]
        == expected_math_rows,
        expected_math_rows,
    )
    check(
        "mathematics_behavioral_claim_forbidden",
        mathematics["behavioral_claim_allowed"] is False,
    )

    with (PROTOCOL_DIR / "availability_screening.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        screening_rows = list(csv.DictReader(handle))
    included = [row["candidate"] for row in screening_rows if row["decision"] == "include"]
    check(
        "availability_screening_includes_frozen_candidates",
        included == ["CLUTRR", "DeepMind Mathematics Dataset"],
        included,
    )

    source_locations = {
        "clutrr": UPSTREAM_ROOT / "external_behavior_v1_clutrr",
        "mathematics": UPSTREAM_ROOT / "external_behavior_v1_mathematics",
    }
    for source_name, source in sources["sources"].items():
        repo = source_locations[source_name]
        check(f"{source_name}_repository_exists", (repo / ".git").is_dir())
        if (repo / ".git").is_dir():
            revision = git_output(repo, "rev-parse", "HEAD")
            status = git_output(repo, "status", "--porcelain=v1")
            check(
                f"{source_name}_revision_matches",
                revision == source["revision"],
                revision,
            )
            check(f"{source_name}_checkout_clean", status == "", status)
        for relative_path, expected_hash in source["files"].items():
            path = repo / relative_path
            check(
                f"{source_name}_hash_{relative_path}",
                path.is_file() and sha256(path) == expected_hash,
            )

    if args.stage == "freeze":
        result_paths = [
            PROJECT_ROOT / "analysis" / "external-behavior-v1",
            PROJECT_ROOT / "runs" / "external-behavior-v1",
            PROJECT_ROOT / "manifests" / "external-behavior-v1",
        ]
        check(
            "no_result_directories_at_protocol_freeze",
            not any(path.exists() for path in result_paths),
            [str(path) for path in result_paths if path.exists()],
        )

    failed = [item for item in checks if not item["pass"]]
    report = {
        "schema_version": "external-behavior-protocol-validation-v1",
        "protocol_id": "external-behavior-v1",
        "stage": args.stage,
        "status": "pass" if not failed else "fail",
        "check_count": len(checks),
        "failed_count": len(failed),
        "checks": checks,
        "document_sha256": {
            name: sha256(PROTOCOL_DIR / name)
            for name in (
                "availability_screening.csv",
                "candidate_freeze_receipt.json",
                "candidate_manifest.json",
                "checklist.md",
                "preregistration.json",
                "README.md",
                "source_manifest.json",
                "validate.py",
            )
        },
    }
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
