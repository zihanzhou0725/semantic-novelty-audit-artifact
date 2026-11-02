"""Shared paths and frozen-input checks for external-behavior-v3."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from toolkit.external_behavior.common import (
    CLUTRR_ROOT,
    PROJECT_ROOT,
    git_output,
    read_json,
    sha256_file,
    write_checksum_manifest,
    write_json,
    write_jsonl,
    write_lines,
)


PROTOCOL_ID = "external-behavior-v3"
PROTOCOL_DIR = PROJECT_ROOT / "protocols" / PROTOCOL_ID
MANIFEST_DIR = PROJECT_ROOT / "manifests" / PROTOCOL_ID
DATA_DIR = MANIFEST_DIR / "data"
SOURCE_ROWS = (
    PROJECT_ROOT
    / "manifests"
    / "external-behavior-v1"
    / ".data.build"
    / "clutrr_candidates_deduplicated.jsonl"
)


def verify_frozen_inputs() -> dict[str, Any]:
    inputs = read_json(PROTOCOL_DIR / "input_manifest.json")
    source = read_json(PROTOCOL_DIR / "source_manifest.json")
    checks: list[dict[str, Any]] = []
    expected_rows = inputs["source_rows"]
    observed_lines = -1
    if SOURCE_ROWS.is_file():
        with SOURCE_ROWS.open("rb") as handle:
            observed_lines = sum(1 for _ in handle)
    checks.append(
        {
            "check": "source_rows",
            "observed_rows": observed_lines,
            "observed_sha256": (
                sha256_file(SOURCE_ROWS) if SOURCE_ROWS.is_file() else None
            ),
            "pass": SOURCE_ROWS.is_file()
            and observed_lines == expected_rows["rows"]
            and sha256_file(SOURCE_ROWS) == expected_rows["sha256"],
        }
    )
    revision = git_output(CLUTRR_ROOT, "rev-parse", "HEAD")
    status = git_output(CLUTRR_ROOT, "status", "--porcelain=v1")
    checks.extend(
        [
            {
                "check": "upstream_revision",
                "observed": revision,
                "pass": revision == source["clutrr"]["revision"],
            },
            {
                "check": "upstream_clean",
                "observed": status,
                "pass": status == "",
            },
        ]
    )
    for relative, expected in source["clutrr"]["files"].items():
        path = CLUTRR_ROOT / relative
        checks.append(
            {
                "check": f"source:{relative}",
                "pass": path.is_file() and sha256_file(path) == expected,
            }
        )
    for key in ("byte_transformer_model", "v1_common"):
        path = PROJECT_ROOT / inputs["frozen_implementation_dependencies"][
            f"{key}_path"
        ]
        expected = inputs["frozen_implementation_dependencies"][
            f"{key}_sha256"
        ]
        checks.append(
            {
                "check": f"implementation:{key}",
                "pass": path.is_file() and sha256_file(path) == expected,
            }
        )
    failed = [item for item in checks if not item["pass"]]
    if failed:
        raise RuntimeError(f"Frozen V3 inputs changed: {failed}")
    return {
        "checks": checks,
        "input_manifest_sha256": sha256_file(
            PROTOCOL_DIR / "input_manifest.json"
        ),
        "source_manifest_sha256": sha256_file(
            PROTOCOL_DIR / "source_manifest.json"
        ),
        "status": "pass",
    }


__all__ = [
    "CLUTRR_ROOT",
    "DATA_DIR",
    "MANIFEST_DIR",
    "PROJECT_ROOT",
    "PROTOCOL_DIR",
    "PROTOCOL_ID",
    "SOURCE_ROWS",
    "read_json",
    "sha256_file",
    "verify_frozen_inputs",
    "write_checksum_manifest",
    "write_json",
    "write_jsonl",
    "write_lines",
]
