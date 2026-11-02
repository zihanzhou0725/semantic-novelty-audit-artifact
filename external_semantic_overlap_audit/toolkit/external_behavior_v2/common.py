"""Shared paths and source checks for external-behavior-v2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from toolkit.external_behavior.common import (
    MATHEMATICS_ROOT,
    PROJECT_ROOT,
    git_output,
    read_json,
    sha256_file,
    write_checksum_manifest,
    write_json,
    write_jsonl,
)


PROTOCOL_ID = "external-behavior-v2"
PROTOCOL_DIR = PROJECT_ROOT / "protocols" / PROTOCOL_ID
MANIFEST_DIR = PROJECT_ROOT / "manifests" / PROTOCOL_ID
DATA_DIR = MANIFEST_DIR / "data"
V1_MANIFEST_DIR = PROJECT_ROOT / "manifests" / "external-behavior-v1"


def verify_source_manifest() -> dict[str, Any]:
    manifest = read_json(PROTOCOL_DIR / "source_manifest.json")
    source = manifest["mathematics"]
    revision = git_output(MATHEMATICS_ROOT, "rev-parse", "HEAD")
    status = git_output(MATHEMATICS_ROOT, "status", "--porcelain=v1")
    checks: list[dict[str, Any]] = [
        {
            "check": "revision",
            "expected": source["revision"],
            "observed": revision,
            "pass": revision == source["revision"],
        },
        {
            "check": "checkout_clean",
            "observed": status,
            "pass": status == "",
        },
    ]
    for relative, expected in source["files"].items():
        path = MATHEMATICS_ROOT / relative
        observed = sha256_file(path) if path.is_file() else None
        checks.append(
            {
                "check": f"source:{relative}",
                "expected": expected,
                "observed": observed,
                "pass": observed == expected,
            }
        )
    failed = [item for item in checks if not item["pass"]]
    if failed:
        raise RuntimeError(f"V2 source validation failed: {failed}")
    return {
        "checks": checks,
        "source_manifest_sha256": sha256_file(
            PROTOCOL_DIR / "source_manifest.json"
        ),
        "status": "pass",
    }


__all__ = [
    "DATA_DIR",
    "MANIFEST_DIR",
    "MATHEMATICS_ROOT",
    "PROJECT_ROOT",
    "PROTOCOL_DIR",
    "PROTOCOL_ID",
    "V1_MANIFEST_DIR",
    "read_json",
    "sha256_file",
    "verify_source_manifest",
    "write_checksum_manifest",
    "write_json",
    "write_jsonl",
]
