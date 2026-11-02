"""Orchestrate external-behavior-v3 design materialization."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from toolkit.external_behavior.common import require_pythonhashseed

from .common import (
    DATA_DIR,
    MANIFEST_DIR,
    PROJECT_ROOT,
    PROTOCOL_DIR,
    PROTOCOL_ID,
    read_json,
    sha256_file,
    verify_frozen_inputs,
    write_checksum_manifest,
    write_json,
)
from .design import materialize_design


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_implementation_freeze() -> dict[str, Any]:
    path = MANIFEST_DIR / "implementation_manifest.json"
    if not path.is_file():
        raise RuntimeError("V3 implementation is not frozen.")
    manifest = read_json(path)
    if (
        manifest.get("status") != "frozen"
        or manifest.get("frozen_before_design_materialization") is not True
    ):
        raise RuntimeError("Invalid V3 implementation freeze.")
    for relative, expected in manifest["implementation_sha256"].items():
        source = PROJECT_ROOT / relative
        if not source.is_file() or sha256_file(source) != expected:
            raise RuntimeError(f"Frozen V3 implementation changed: {relative}")
    if (
        sha256_file(MANIFEST_DIR / "environment.json")
        != manifest["environment_sha256"]
        or sha256_file(MANIFEST_DIR / "requirements.lock")
        != manifest["requirements_lock_sha256"]
        or sha256_file(PROTOCOL_DIR / "preregistration.json")
        != manifest["preregistration_sha256"]
    ):
        raise RuntimeError("Frozen V3 environment or protocol changed.")
    return manifest


def materialize() -> dict[str, Any]:
    require_pythonhashseed()
    implementation = validate_implementation_freeze()
    inputs = verify_frozen_inputs()
    if DATA_DIR.exists():
        raise RuntimeError(f"V3 data already exists: {DATA_DIR}")
    staging = MANIFEST_DIR / ".data.build"
    if staging.exists():
        raise RuntimeError(f"V3 staging already exists: {staging}")
    staging.parent.mkdir(parents=True, exist_ok=True)
    started = _now()
    report = materialize_design(staging)
    files = sorted(path for path in staging.iterdir() if path.is_file())
    write_checksum_manifest(staging, files)
    staging.replace(DATA_DIR)
    manifest = {
        "completed_at_utc": _now(),
        "data_checksum_manifest_sha256": sha256_file(DATA_DIR / "SHA256SUMS"),
        "data_directory": DATA_DIR.relative_to(PROJECT_ROOT).as_posix(),
        "design_report": report,
        "implementation_manifest_sha256": sha256_file(
            MANIFEST_DIR / "implementation_manifest.json"
        ),
        "input_manifest_sha256": inputs["input_manifest_sha256"],
        "model_training_started": False,
        "protocol_id": PROTOCOL_ID,
        "preregistration_sha256": sha256_file(
            PROTOCOL_DIR / "preregistration.json"
        ),
        "schema_version": "external-behavior-v3-materialization-v1",
        "started_at_utc": started,
        "status": "materialized_pending_independent_validation",
    }
    write_json(MANIFEST_DIR / "materialization_manifest.json", manifest)
    return manifest
