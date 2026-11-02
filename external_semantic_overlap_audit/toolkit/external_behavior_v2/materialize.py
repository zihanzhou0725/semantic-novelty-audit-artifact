"""Orchestrate external-behavior-v2 Mathematics materialization."""

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
    verify_source_manifest,
    write_checksum_manifest,
    write_json,
)
from .mathematics import materialize_all_streams


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_implementation_freeze() -> dict[str, Any]:
    path = MANIFEST_DIR / "implementation_manifest.json"
    if not path.is_file():
        raise RuntimeError("V2 implementation has not been frozen.")
    manifest = read_json(path)
    if (
        manifest.get("status") != "frozen"
        or manifest.get("frozen_before_v2_generation") is not True
    ):
        raise RuntimeError("V2 implementation freeze is invalid.")
    for relative, expected in manifest["implementation_sha256"].items():
        source = PROJECT_ROOT / relative
        if not source.is_file() or sha256_file(source) != expected:
            raise RuntimeError(f"Frozen V2 implementation changed: {relative}")
    if (
        sha256_file(MANIFEST_DIR / "environment.json")
        != manifest["environment_sha256"]
    ):
        raise RuntimeError("V2 environment manifest changed.")
    if (
        sha256_file(MANIFEST_DIR / "requirements.lock")
        != manifest["requirements_lock_sha256"]
    ):
        raise RuntimeError("V2 dependency lock changed.")
    if (
        sha256_file(PROTOCOL_DIR / "preregistration.json")
        != manifest["preregistration_sha256"]
    ):
        raise RuntimeError("V2 preregistration changed.")
    return manifest


def materialize() -> dict[str, Any]:
    require_pythonhashseed()
    implementation = validate_implementation_freeze()
    source = verify_source_manifest()
    if DATA_DIR.exists():
        raise RuntimeError(f"V2 data already exists: {DATA_DIR}")
    staging = MANIFEST_DIR / ".data.build"
    if staging.exists():
        raise RuntimeError(f"V2 staging directory already exists: {staging}")
    staging.mkdir(parents=True)
    started = _now()
    mathematics = materialize_all_streams(staging)
    files = sorted(path for path in staging.iterdir() if path.is_file())
    write_checksum_manifest(staging, files)
    staging.replace(DATA_DIR)
    report = {
        "aggregate_overlap_released": False,
        "completed_at_utc": _now(),
        "data_checksum_manifest_sha256": sha256_file(DATA_DIR / "SHA256SUMS"),
        "data_directory": DATA_DIR.relative_to(PROJECT_ROOT).as_posix(),
        "implementation_manifest_sha256": sha256_file(
            MANIFEST_DIR / "implementation_manifest.json"
        ),
        "join_or_collision_census_executed": False,
        "mathematics": mathematics,
        "protocol_id": PROTOCOL_ID,
        "preregistration_sha256": sha256_file(
            PROTOCOL_DIR / "preregistration.json"
        ),
        "schema_version": "external-behavior-v2-materialization-v1",
        "source_manifest_sha256": source["source_manifest_sha256"],
        "started_at_utc": started,
        "status": "materialized_pending_independent_validation",
    }
    write_json(MANIFEST_DIR / "materialization_manifest.json", report)
    return report
