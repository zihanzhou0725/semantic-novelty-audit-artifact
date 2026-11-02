"""Freeze Mathematics collision-census implementation and environment."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

from .collision_common import (
    ANALYSIS_DIR,
    ANALYSIS_ID,
    IMPLEMENTATION_DIR,
    PROJECT_ROOT,
    PROTOCOL_DIR,
    read_json,
    sha256_file,
    verify_inputs,
    write_json,
)


IMPLEMENTATION_PATHS = (
    "scripts/freeze_math_collision_census_v1.py",
    "scripts/run_math_collision_census_v1.py",
    "scripts/validate_math_collision_census_v1.py",
    "tests/test_math_collision_census_v1.py",
    "toolkit/external_behavior/common.py",
    "toolkit/external_behavior_v2/collision_census.py",
    "toolkit/external_behavior_v2/collision_common.py",
    "toolkit/external_behavior_v2/collision_freeze.py",
    "toolkit/external_behavior_v2/collision_run.py",
    "toolkit/external_behavior_v2/collision_validate.py",
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
            for name in ("numpy", "sympy")
        },
        "executable": sys.executable,
        "macos_version": platform.mac_ver()[0],
        "platform": platform.platform(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
    }
    return environment, "\n".join(freeze_lines) + "\n"


def freeze_implementation() -> dict[str, Any]:
    protocol_validation = read_json(PROTOCOL_DIR / "validation_report.json")
    if protocol_validation.get("status") != "pass":
        raise RuntimeError("Collision-census protocol validation has not passed.")
    if (
        IMPLEMENTATION_DIR.exists()
        or ANALYSIS_DIR.exists()
        or ANALYSIS_DIR.with_name(f".{ANALYSIS_ID}.build").exists()
    ):
        raise RuntimeError("Collision-census execution has already begun.")
    missing = [
        relative
        for relative in IMPLEMENTATION_PATHS
        if not (PROJECT_ROOT / relative).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Census implementation files missing: {missing}")

    inputs = verify_inputs()
    environment, lock_text = _environment()
    IMPLEMENTATION_DIR.mkdir(parents=True)
    requirements_path = IMPLEMENTATION_DIR / "requirements.lock"
    requirements_path.write_text(lock_text, encoding="ascii")
    environment_path = IMPLEMENTATION_DIR / "environment.json"
    write_json(environment_path, environment)
    manifest = {
        "analysis_id": ANALYSIS_ID,
        "environment_sha256": sha256_file(environment_path),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_before_join": True,
        "implementation_sha256": {
            relative: sha256_file(PROJECT_ROOT / relative)
            for relative in IMPLEMENTATION_PATHS
        },
        "input_manifest_sha256": inputs["input_manifest_sha256"],
        "preregistration_sha256": sha256_file(
            PROTOCOL_DIR / "preregistration.json"
        ),
        "requirements_lock_sha256": sha256_file(requirements_path),
        "schema_version": "mathematics-collision-census-implementation-v1",
        "status": "frozen",
    }
    manifest_path = IMPLEMENTATION_DIR / "implementation_manifest.json"
    write_json(manifest_path, manifest)
    (IMPLEMENTATION_DIR / "IMPLEMENTATION_SHA256SUMS").write_text(
        "".join(
            f"{sha256_file(path)}  {path.name}\n"
            for path in (environment_path, manifest_path, requirements_path)
        ),
        encoding="ascii",
    )
    return manifest
