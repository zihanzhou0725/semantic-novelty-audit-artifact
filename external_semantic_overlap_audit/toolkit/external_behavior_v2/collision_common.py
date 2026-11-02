"""Shared paths and frozen-input checks for the Mathematics census."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from toolkit.external_behavior.common import (
    PROJECT_ROOT,
    read_json,
    sha256_file,
    write_json,
    write_jsonl,
)


ANALYSIS_ID = "mathematics-collision-census-v1"
PROTOCOL_DIR = (
    PROJECT_ROOT
    / "protocols"
    / "external-behavior-v2"
    / ANALYSIS_ID
)
IMPLEMENTATION_DIR = (
    PROJECT_ROOT
    / "manifests"
    / "external-behavior-v2"
    / ANALYSIS_ID
)
ANALYSIS_PARENT = PROJECT_ROOT / "analysis" / "external-behavior-v2"
ANALYSIS_DIR = ANALYSIS_PARENT / ANALYSIS_ID
STAGING_DIR = ANALYSIS_PARENT / f".{ANALYSIS_ID}.build"
MODULES = ("polynomials__expand", "polynomials__collect")
REGIMES = ("train", "interpolate")


def input_manifest() -> dict[str, Any]:
    return read_json(PROTOCOL_DIR / "input_manifest.json")


def stream_path(regime: str, module_name: str) -> Path:
    matches = [
        item
        for item in input_manifest()["streams"]
        if item["regime"] == regime and item["module"] == module_name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Frozen stream is not unique: {regime}/{module_name}")
    return PROJECT_ROOT / matches[0]["path"]


def verify_inputs() -> dict[str, Any]:
    manifest = input_manifest()
    checks: list[dict[str, Any]] = []
    for stream in manifest["streams"]:
        path = PROJECT_ROOT / stream["path"]
        observed = sha256_file(path) if path.is_file() else None
        checks.append(
            {
                "module": stream["module"],
                "observed_sha256": observed,
                "pass": observed == stream["sha256"],
                "regime": stream["regime"],
            }
        )
    failed = [item for item in checks if not item["pass"]]
    if failed:
        raise RuntimeError(f"Frozen census inputs changed: {failed}")
    return {
        "checks": checks,
        "input_manifest_sha256": sha256_file(
            PROTOCOL_DIR / "input_manifest.json"
        ),
        "status": "pass",
    }


__all__ = [
    "ANALYSIS_DIR",
    "ANALYSIS_ID",
    "IMPLEMENTATION_DIR",
    "MODULES",
    "PROJECT_ROOT",
    "PROTOCOL_DIR",
    "STAGING_DIR",
    "input_manifest",
    "read_json",
    "sha256_file",
    "stream_path",
    "verify_inputs",
    "write_json",
    "write_jsonl",
]
