"""Freeze external-behavior-v3 implementation and environment."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

from toolkit.external_behavior.model import inspect_architecture

from .common import (
    DATA_DIR,
    MANIFEST_DIR,
    PROJECT_ROOT,
    PROTOCOL_DIR,
    PROTOCOL_ID,
    read_json,
    sha256_file,
    verify_frozen_inputs,
    write_json,
)


IMPLEMENTATION_PATHS = (
    "scripts/freeze_external_behavior_v3.py",
    "scripts/materialize_external_behavior_v3.py",
    "scripts/validate_external_behavior_v3.py",
    "tests/test_external_behavior_v3.py",
    "toolkit/external_behavior/clutrr.py",
    "toolkit/external_behavior/common.py",
    "toolkit/external_behavior/model.py",
    "toolkit/external_behavior_v3/__init__.py",
    "toolkit/external_behavior_v3/common.py",
    "toolkit/external_behavior_v3/design.py",
    "toolkit/external_behavior_v3/freeze.py",
    "toolkit/external_behavior_v3/materialize.py",
    "toolkit/external_behavior_v3/validate.py",
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
            for name in ("numpy", "torch", "PyYAML")
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
        raise RuntimeError("V3 protocol validation has not passed.")
    if DATA_DIR.exists() or (MANIFEST_DIR / ".data.build").exists():
        raise RuntimeError("Refusing V3 freeze after materialization began.")
    missing = [
        relative
        for relative in IMPLEMENTATION_PATHS
        if not (PROJECT_ROOT / relative).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"V3 implementation files missing: {missing}")

    inputs = verify_frozen_inputs()
    architecture = inspect_architecture(23)
    environment, lock_text = _environment()
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    requirements_path = MANIFEST_DIR / "requirements.lock"
    requirements_path.write_text(lock_text, encoding="ascii")
    environment_path = MANIFEST_DIR / "environment.json"
    write_json(environment_path, environment)
    manifest = {
        "environment_sha256": sha256_file(environment_path),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_before_design_materialization": True,
        "implementation_sha256": {
            relative: sha256_file(PROJECT_ROOT / relative)
            for relative in IMPLEMENTATION_PATHS
        },
        "input_manifest_sha256": inputs["input_manifest_sha256"],
        "model_architecture": architecture,
        "preregistration_sha256": sha256_file(
            PROTOCOL_DIR / "preregistration.json"
        ),
        "protocol_id": PROTOCOL_ID,
        "requirements_lock_sha256": sha256_file(requirements_path),
        "schema_version": "external-behavior-v3-implementation-freeze-v1",
        "status": "frozen",
    }
    manifest_path = MANIFEST_DIR / "implementation_manifest.json"
    write_json(manifest_path, manifest)
    (MANIFEST_DIR / "IMPLEMENTATION_SHA256SUMS").write_text(
        "".join(
            f"{sha256_file(path)}  {path.name}\n"
            for path in (environment_path, manifest_path, requirements_path)
        ),
        encoding="ascii",
    )
    return manifest
