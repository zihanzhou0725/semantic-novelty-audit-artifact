"""Freeze external-behavior-v2 implementation and environment."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

from .common import (
    DATA_DIR,
    MANIFEST_DIR,
    PROJECT_ROOT,
    PROTOCOL_DIR,
    PROTOCOL_ID,
    V1_MANIFEST_DIR,
    read_json,
    sha256_file,
    verify_source_manifest,
    write_json,
)


IMPLEMENTATION_PATHS = (
    "scripts/freeze_external_behavior_v2.py",
    "scripts/materialize_external_behavior_v2.py",
    "scripts/validate_external_behavior_v2.py",
    "tests/test_external_behavior_v2.py",
    "toolkit/external_behavior/common.py",
    "toolkit/external_behavior/mathematics.py",
    "toolkit/external_behavior_v2/__init__.py",
    "toolkit/external_behavior_v2/common.py",
    "toolkit/external_behavior_v2/freeze.py",
    "toolkit/external_behavior_v2/materialize.py",
    "toolkit/external_behavior_v2/mathematics.py",
    "toolkit/external_behavior_v2/validate.py",
)
RELEVANT_DISTRIBUTIONS = (
    "absl-py",
    "numpy",
    "six",
    "sympy",
)


def _environment() -> tuple[dict[str, Any], str]:
    freeze_lines = sorted(
        line.strip()
        for line in subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            text=True,
        ).splitlines()
        if line.strip()
    )
    environment = {
        "architecture": platform.machine(),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in RELEVANT_DISTRIBUTIONS
        },
        "executable": sys.executable,
        "macos_version": platform.mac_ver()[0],
        "platform": platform.platform(),
        "protocol_id": PROTOCOL_ID,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
    }
    return environment, "\n".join(freeze_lines) + "\n"


def freeze_implementation() -> dict[str, Any]:
    protocol_validation = read_json(PROTOCOL_DIR / "validation_report.json")
    if protocol_validation.get("status") != "pass":
        raise RuntimeError("V2 protocol validation has not passed.")
    if DATA_DIR.exists() or (MANIFEST_DIR / ".data.build").exists():
        raise RuntimeError("Refusing to freeze after V2 data generation began.")
    missing = [
        relative
        for relative in IMPLEMENTATION_PATHS
        if not (PROJECT_ROOT / relative).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"V2 implementation files are missing: {missing}")

    source_validation = verify_source_manifest()
    environment, lock_text = _environment()
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    requirements_path = MANIFEST_DIR / "requirements.lock"
    requirements_path.write_text(lock_text, encoding="ascii")
    environment_path = MANIFEST_DIR / "environment.json"
    write_json(environment_path, environment)
    manifest = {
        "environment_sha256": sha256_file(environment_path),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_before_v2_generation": True,
        "implementation_sha256": {
            relative: sha256_file(PROJECT_ROOT / relative)
            for relative in IMPLEMENTATION_PATHS
        },
        "parent_failure_report_sha256": sha256_file(
            V1_MANIFEST_DIR / "materialization_failure_report.json"
        ),
        "parent_preregistration_sha256": read_json(
            PROTOCOL_DIR / "preregistration.json"
        )["relationship_to_v1"]["parent_preregistration_sha256"],
        "protocol_id": PROTOCOL_ID,
        "preregistration_sha256": sha256_file(
            PROTOCOL_DIR / "preregistration.json"
        ),
        "requirements_lock_sha256": sha256_file(requirements_path),
        "schema_version": "external-behavior-v2-implementation-freeze-v1",
        "source_validation": source_validation,
        "status": "frozen",
    }
    manifest_path = MANIFEST_DIR / "implementation_manifest.json"
    write_json(manifest_path, manifest)
    checksum_path = MANIFEST_DIR / "IMPLEMENTATION_SHA256SUMS"
    checksum_path.write_text(
        "".join(
            f"{sha256_file(path)}  {path.name}\n"
            for path in (environment_path, manifest_path, requirements_path)
        ),
        encoding="ascii",
    )
    return manifest
