"""Freeze implementation and environment before data materialization."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .clutrr import _args, _load_store, relation_classes
from .common import (
    DATA_DIR,
    MANIFEST_DIR,
    PROJECT_ROOT,
    PROTOCOL_DIR,
    PROTOCOL_ID,
    read_json,
    sha256_file,
    verify_source_manifest,
    write_json,
)
from .model import inspect_architecture


IMPLEMENTATION_PATHS = (
    "scripts/freeze_external_behavior_implementation.py",
    "scripts/materialize_external_behavior.py",
    "scripts/validate_external_behavior_materialization.py",
    "tests/test_external_behavior_materialization.py",
    "toolkit/external_behavior/__init__.py",
    "toolkit/external_behavior/clutrr.py",
    "toolkit/external_behavior/common.py",
    "toolkit/external_behavior/freeze.py",
    "toolkit/external_behavior/materialize.py",
    "toolkit/external_behavior/mathematics.py",
    "toolkit/external_behavior/model.py",
    "toolkit/external_behavior/validate.py",
)
RELEVANT_DISTRIBUTIONS = (
    "absl-py",
    "matplotlib",
    "names",
    "networkx",
    "numpy",
    "PyYAML",
    "six",
    "sympy",
    "torch",
    "tqdm",
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
    distributions = {
        name: importlib.metadata.version(name) for name in RELEVANT_DISTRIBUTIONS
    }
    environment = {
        "architecture": platform.machine(),
        "dependencies": distributions,
        "executable": sys.executable,
        "macos_version": platform.mac_ver()[0],
        "platform": platform.platform(),
        "protocol_id": PROTOCOL_ID,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "torch_default_dtype": str(torch.get_default_dtype()),
        "torch_version": torch.__version__,
    }
    return environment, "\n".join(freeze_lines) + "\n"


def freeze_implementation() -> dict[str, Any]:
    protocol_validation = read_json(PROTOCOL_DIR / "validation_report.json")
    if protocol_validation.get("status") != "pass":
        raise RuntimeError("Frozen parent protocol validation has not passed.")
    if DATA_DIR.exists() and any(DATA_DIR.iterdir()):
        raise RuntimeError(
            "Refusing to refreeze implementation after data materialization."
        )
    missing = [
        relative
        for relative in IMPLEMENTATION_PATHS
        if not (PROJECT_ROOT / relative).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Implementation files are missing: {missing}")

    source_validation = verify_source_manifest()
    store = _load_store(_args())
    classes = relation_classes(store)
    architecture = inspect_architecture(len(classes))
    environment, lock_text = _environment()

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    requirements_path = MANIFEST_DIR / "requirements.lock"
    requirements_path.write_text(lock_text, encoding="ascii")
    environment_path = MANIFEST_DIR / "environment.json"
    write_json(environment_path, environment)
    implementation_hashes = {
        relative: sha256_file(PROJECT_ROOT / relative)
        for relative in IMPLEMENTATION_PATHS
    }
    manifest = {
        "architecture_inspection": architecture,
        "environment_sha256": sha256_file(environment_path),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_before_data_materialization": True,
        "implementation_sha256": implementation_hashes,
        "model_output_classes": classes,
        "parent_preregistration_sha256": sha256_file(
            PROTOCOL_DIR / "preregistration.json"
        ),
        "protocol_id": PROTOCOL_ID,
        "requirements_lock_sha256": sha256_file(requirements_path),
        "schema_version": "external-behavior-implementation-freeze-v1",
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
