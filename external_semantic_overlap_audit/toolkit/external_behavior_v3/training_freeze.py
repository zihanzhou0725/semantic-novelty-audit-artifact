"""Freeze CLUTRR V3 execution implementation and randomization manifests."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

from .training_common import (
    EXECUTION_DIR,
    PROJECT_ROOT,
    PROTOCOL_DIR,
    RUN_ROOT,
    build_run_manifest,
    build_tensor_cache,
    classes,
    epoch_order_manifest,
    initialization_manifest,
    sha256_file,
    write_csv,
    write_json,
)


IMPLEMENTATION_PATHS = (
    "scripts/freeze_external_behavior_v3_execution.py",
    "scripts/run_external_behavior_v3.py",
    "scripts/run_external_behavior_v3_matrix.py",
    "scripts/validate_external_behavior_v3_execution.py",
    "tests/test_external_behavior_v3_training.py",
    "toolkit/external_behavior/common.py",
    "toolkit/external_behavior/model.py",
    "toolkit/external_behavior_v3/train_driver.py",
    "toolkit/external_behavior_v3/train_integrity.py",
    "toolkit/external_behavior_v3/train_run.py",
    "toolkit/external_behavior_v3/training_common.py",
    "toolkit/external_behavior_v3/training_freeze.py",
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
            for name in ("numpy", "torch")
        },
        "executable": sys.executable,
        "macos_version": platform.mac_ver()[0],
        "platform": platform.platform(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "torch_deterministic_algorithms": True,
        "torch_interop_threads_per_run": 1,
        "torch_intraop_threads_per_run": 2,
        "maximum_concurrent_runs": 8,
    }
    return environment, "\n".join(freeze_lines) + "\n"


def freeze_execution() -> dict[str, Any]:
    protocol_validation = json_load(PROTOCOL_DIR / "validation_report.json")
    if protocol_validation.get("status") != "pass":
        raise RuntimeError("Execution protocol validation has not passed.")
    design_validation = json_load(
        PROJECT_ROOT
        / "manifests"
        / "external-behavior-v3"
        / "design_validation_report.json"
    )
    if design_validation.get("design_gate") != "pass":
        raise RuntimeError("V3 design gate has not passed.")
    if EXECUTION_DIR.exists() or (
        RUN_ROOT.exists() and any(RUN_ROOT.iterdir())
    ):
        raise RuntimeError("Execution freeze or training already exists.")
    missing = [
        relative
        for relative in IMPLEMENTATION_PATHS
        if not (PROJECT_ROOT / relative).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Execution implementation missing: {missing}")

    EXECUTION_DIR.mkdir(parents=True)
    environment, lock_text = _environment()
    environment_path = EXECUTION_DIR / "environment.json"
    write_json(environment_path, environment)
    requirements_path = EXECUTION_DIR / "requirements.lock"
    requirements_path.write_text(lock_text, encoding="ascii")
    write_json(EXECUTION_DIR / "classes.json", {"classes": classes()})

    cache = build_tensor_cache()
    initialization = initialization_manifest()
    write_json(
        EXECUTION_DIR / "initialization_manifest.json",
        initialization,
    )
    orders = epoch_order_manifest()
    write_json(EXECUTION_DIR / "epoch_order_manifest.json", orders)
    run_rows = build_run_manifest(initialization, orders)
    write_csv(EXECUTION_DIR / "run_manifest.csv", run_rows)

    implementation_hashes = {
        relative: sha256_file(PROJECT_ROOT / relative)
        for relative in IMPLEMENTATION_PATHS
    }
    manifest = {
        "cache": cache,
        "classes_sha256": sha256_file(EXECUTION_DIR / "classes.json"),
        "design_validation_sha256": sha256_file(
            PROJECT_ROOT
            / "manifests"
            / "external-behavior-v3"
            / "design_validation_report.json"
        ),
        "environment_sha256": sha256_file(environment_path),
        "epoch_order_manifest_sha256": sha256_file(
            EXECUTION_DIR / "epoch_order_manifest.json"
        ),
        "execution_id": "external-behavior-v3-execution-v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_before_training": True,
        "implementation_sha256": implementation_hashes,
        "initialization_manifest_sha256": sha256_file(
            EXECUTION_DIR / "initialization_manifest.json"
        ),
        "protocol_preregistration_sha256": sha256_file(
            PROTOCOL_DIR / "preregistration.json"
        ),
        "requirements_lock_sha256": sha256_file(requirements_path),
        "run_manifest_sha256": sha256_file(
            EXECUTION_DIR / "run_manifest.csv"
        ),
        "schema_version": "external-behavior-v3-execution-manifest-v1",
        "status": "frozen_before_training",
    }
    manifest_path = EXECUTION_DIR / "execution_manifest.json"
    write_json(manifest_path, manifest)
    frozen_paths = [
        EXECUTION_DIR / "classes.json",
        environment_path,
        EXECUTION_DIR / "epoch_order_manifest.json",
        EXECUTION_DIR / "evaluation_manifest.npz",
        manifest_path,
        EXECUTION_DIR / "initialization_manifest.json",
        requirements_path,
        EXECUTION_DIR / "run_manifest.csv",
        EXECUTION_DIR / "tensor_cache.npz",
        EXECUTION_DIR / "train_cache_indices.npz",
    ]
    (EXECUTION_DIR / "EXECUTION_IMPLEMENTATION_SHA256SUMS").write_text(
        "".join(
            f"{sha256_file(path)}  {path.name}\n"
            for path in frozen_paths
        ),
        encoding="ascii",
    )
    return manifest


def json_load(path: Any) -> dict[str, Any]:
    import json

    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value
