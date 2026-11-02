"""Orchestrate the frozen external-behavior-v1 data materialization."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .clutrr import CANDIDATE_CAP, materialize_clutrr
from .common import (
    DATA_DIR,
    MANIFEST_DIR,
    PROJECT_ROOT,
    PROTOCOL_DIR,
    PROTOCOL_ID,
    read_json,
    require_pythonhashseed,
    sha256_file,
    verify_source_manifest,
    write_checksum_manifest,
    write_json,
)
from .mathematics import materialize_mathematics


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_implementation_freeze() -> dict[str, Any]:
    path = MANIFEST_DIR / "implementation_manifest.json"
    if not path.is_file():
        raise RuntimeError("Implementation manifest has not been frozen.")
    manifest = read_json(path)
    if (
        manifest.get("status") != "frozen"
        or not manifest.get("frozen_before_data_materialization")
    ):
        raise RuntimeError("Implementation manifest is not a valid pre-data freeze.")
    for relative, expected in manifest["implementation_sha256"].items():
        source = PROJECT_ROOT / relative
        if not source.is_file() or sha256_file(source) != expected:
            raise RuntimeError(f"Frozen implementation changed: {relative}")
    environment_path = MANIFEST_DIR / "environment.json"
    requirements_path = MANIFEST_DIR / "requirements.lock"
    if sha256_file(environment_path) != manifest["environment_sha256"]:
        raise RuntimeError("Frozen environment manifest changed.")
    if sha256_file(requirements_path) != manifest["requirements_lock_sha256"]:
        raise RuntimeError("Frozen dependency lock changed.")
    if (
        sha256_file(PROTOCOL_DIR / "preregistration.json")
        != manifest["parent_preregistration_sha256"]
    ):
        raise RuntimeError("Parent preregistration changed after implementation freeze.")
    return manifest


def _clutrr_progress(completed: int, total: int) -> None:
    print(f"CLUTRR materialization {completed}/{total}", flush=True)


def _mathematics_progress(
    regime: str,
    module_name: str,
    completed: int,
    total: int,
) -> None:
    print(
        f"Mathematics {regime}/{module_name} {completed}/{total}",
        flush=True,
    )


def materialize_all() -> dict[str, Any]:
    require_pythonhashseed()
    implementation = validate_implementation_freeze()
    source_validation = verify_source_manifest()
    if DATA_DIR.exists():
        raise RuntimeError(f"Data directory already exists: {DATA_DIR}")
    staging = MANIFEST_DIR / ".data.build"
    if staging.exists():
        raise RuntimeError(
            f"Incomplete staging directory exists; inspect before removal: {staging}"
        )
    staging.mkdir(parents=True)
    started = _now()
    try:
        clutrr = materialize_clutrr(
            staging,
            candidate_cap=CANDIDATE_CAP,
            progress=_clutrr_progress,
        )
        clutrr_gate = {
            "checked_before_mathematics_materialization": True,
            "design_ready_for_full_validation": clutrr[
                "design_ready_for_validation"
            ],
            "eligible_pattern_count": clutrr["eligible_pattern_count"],
            "pair_count": clutrr["pair_count"],
            "status": (
                "provisional_pass"
                if clutrr["design_ready_for_validation"]
                else "provisional_no_go"
            ),
            "target_count": clutrr["target_count"],
        }
        write_json(staging / "clutrr_pre_math_gate.json", clutrr_gate)

        mathematics = materialize_mathematics(
            staging,
            progress=_mathematics_progress,
        )
        files = sorted(path for path in staging.iterdir() if path.is_file())
        checksum_path = write_checksum_manifest(staging, files)
        files.append(checksum_path)
        staging.replace(DATA_DIR)
    except BaseException:
        raise

    materialization = {
        "aggregate_mathematics_overlap_released": False,
        "clutrr": clutrr,
        "completed_at_utc": _now(),
        "data_checksum_manifest_sha256": sha256_file(DATA_DIR / "SHA256SUMS"),
        "data_directory": DATA_DIR.relative_to(PROJECT_ROOT).as_posix(),
        "implementation_manifest_sha256": sha256_file(
            MANIFEST_DIR / "implementation_manifest.json"
        ),
        "mathematics": mathematics,
        "parent_preregistration_sha256": sha256_file(
            PROTOCOL_DIR / "preregistration.json"
        ),
        "protocol_id": PROTOCOL_ID,
        "schema_version": "external-behavior-materialization-v1",
        "source_manifest_sha256": source_validation[
            "source_manifest_sha256"
        ],
        "started_at_utc": started,
        "status": "materialized_pending_independent_validation",
        "training_started": False,
    }
    write_json(MANIFEST_DIR / "materialization_manifest.json", materialization)
    return materialization
