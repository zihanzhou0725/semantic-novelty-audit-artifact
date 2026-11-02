"""Freeze and validate the 32-run directional V4 execution bundle."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from toolkit.external_behavior.common import iter_jsonl

from .training_common import (
    ARMS,
    BLOCKS,
    EPOCHS,
    EVAL_ROWS,
    EXECUTION_DIR,
    EXECUTION_ID,
    FIXED_LENGTH,
    MATERIALIZATION_DIR,
    MODEL_SEEDS,
    PROJECT_ROOT,
    PROTOCOL_DIR,
    RUN_COUNT,
    RUN_ROOT,
    TRAIN_ROWS,
    array_sha256,
    build_run_manifest,
    build_tensor_cache,
    classes,
    epoch_order,
    epoch_order_manifest,
    initialization_manifest,
    initialize_model,
    load_npz,
    read_csv,
    read_json,
    sha256_file,
    tensor_cache_summary,
    write_csv,
    write_json,
)


IMPLEMENTATION_PATHS = (
    "scripts/freeze_external_behavior_v4_execution.py",
    "scripts/run_external_behavior_v4.py",
    "scripts/run_external_behavior_v4_matrix.py",
    "scripts/validate_external_behavior_v4_execution.py",
    "tests/test_external_behavior_v4_training.py",
    "toolkit/external_behavior/common.py",
    "toolkit/external_behavior/model.py",
    "toolkit/external_behavior_v3/training_common.py",
    "toolkit/external_behavior_v4/train_driver.py",
    "toolkit/external_behavior_v4/train_integrity.py",
    "toolkit/external_behavior_v4/train_run.py",
    "toolkit/external_behavior_v4/training_common.py",
    "toolkit/external_behavior_v4/training_freeze.py",
    "protocols/external-behavior-v4/execution-v1/validate.py",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _checksum_entries(path: Path) -> list[tuple[str, Path]]:
    entries: list[tuple[str, Path]] = []
    for line in path.read_text(encoding="ascii").splitlines():
        if line:
            digest, name = line.split("  ", 1)
            entries.append((digest, path.parent / name))
    return entries


def _protocol_input_paths() -> list[Path]:
    return sorted(
        [
            PROJECT_ROOT
            / "protocols"
            / "external-behavior-v4"
            / "materialization-v1"
            / "SHA256SUMS",
            MATERIALIZATION_DIR / "MATERIALIZATION_SHA256SUMS",
            MATERIALIZATION_DIR / "materialization_validation_report.json",
            MATERIALIZATION_DIR / "run_manifest.csv",
            MATERIALIZATION_DIR / "data" / "SHA256SUMS",
            PROJECT_ROOT
            / "manifests"
            / "external-behavior-v4"
            / "directional-design-v1"
            / "DESIGN_SHA256SUMS",
            PROJECT_ROOT
            / "manifests"
            / "external-behavior-v4"
            / "directional-design-v1"
            / "design_manifest.json",
        ],
        key=lambda path: path.relative_to(PROJECT_ROOT).as_posix(),
    )


def _implementation_hashes() -> dict[str, str]:
    missing = [
        name
        for name in IMPLEMENTATION_PATHS
        if not (PROJECT_ROOT / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing V4 execution code: {missing}")
    return {
        name: sha256_file(PROJECT_ROOT / name)
        for name in IMPLEMENTATION_PATHS
    }


def validate_protocol(*, require_execution_absent: bool) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: Any = None) -> None:
        checks.append({"detail": detail, "name": name, "pass": bool(condition)})

    prereg = read_json(PROTOCOL_DIR / "preregistration.json")
    materialization = read_json(
        MATERIALIZATION_DIR / "materialization_validation_report.json"
    )
    check(
        "materialization_gate",
        materialization.get("status") == "pass"
        and materialization.get("materialization_gate") == "go"
        and materialization.get("training_started") is False,
    )
    check(
        "run_arithmetic",
        prereg["frozen_design"]["run_count"] == RUN_COUNT
        and prereg["frozen_design"]["block_count"] == 2
        and prereg["frozen_design"]["model_seed_count"] == 8
        and prereg["frozen_design"]["arm_count"] == 2,
    )
    check(
        "update_arithmetic",
        prereg["optimization"]["updates_per_epoch"] == 188
        and prereg["optimization"]["updates_per_run"] == 5_640
        and prereg["optimization"]["epochs"] == EPOCHS,
    )
    check(
        "four_way_pairing",
        prereg["randomness"]["shared_over"] == [
            "block_0_arm_A",
            "block_0_arm_B",
            "block_1_arm_A",
            "block_1_arm_B",
        ]
        and prereg["randomness"]["model_initialization_namespace"]
        == "clutrr:model:{model_seed}",
    )
    manifest_path = PROTOCOL_DIR / "input_manifest.json"
    receipt_path = PROTOCOL_DIR / "freeze_receipt.json"
    check(
        "freeze_artifacts",
        manifest_path.is_file() and receipt_path.is_file(),
    )
    input_failures: list[str] = []
    code_failures: list[str] = []
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        for name, expected in manifest["input_sha256"].items():
            path = PROJECT_ROOT / name
            if not path.is_file() or sha256_file(path) != expected:
                input_failures.append(name)
        for name, expected in manifest["implementation_sha256"].items():
            path = PROJECT_ROOT / name
            if not path.is_file() or sha256_file(path) != expected:
                code_failures.append(name)
    check("input_hashes", not input_failures, input_failures)
    check("implementation_hashes", not code_failures, code_failures)
    if require_execution_absent:
        check(
            "execution_absent",
            not EXECUTION_DIR.exists() and not RUN_ROOT.exists(),
            {
                "execution_dir_exists": EXECUTION_DIR.exists(),
                "run_root_exists": RUN_ROOT.exists(),
            },
        )
    failed = [item for item in checks if not item["pass"]]
    document_names = (
        "README.md",
        "checklist.md",
        "freeze_receipt.json",
        "input_manifest.json",
        "preregistration.json",
        "validate.py",
    )
    return {
        "check_count": len(checks),
        "checks": checks,
        "document_sha256": {
            name: sha256_file(PROTOCOL_DIR / name)
            for name in document_names
        },
        "execution_id": EXECUTION_ID,
        "failed_count": len(failed),
        "schema_version": (
            "external-behavior-v4-execution-protocol-validation-v1"
        ),
        "stage": (
            "pre_execution_freeze"
            if require_execution_absent
            else "post_execution_freeze_recheck"
        ),
        "status": "pass" if not failed else "fail",
        "validated_at_utc": _now(),
    }


def freeze_protocol() -> dict[str, Any]:
    if (PROTOCOL_DIR / "input_manifest.json").exists():
        raise RuntimeError("V4 execution protocol is already frozen.")
    if EXECUTION_DIR.exists() or RUN_ROOT.exists():
        raise RuntimeError("V4 execution artifacts or runs already exist.")
    materialization = read_json(
        MATERIALIZATION_DIR / "materialization_validation_report.json"
    )
    if materialization.get("status") != "pass":
        raise RuntimeError("V4 materialization Gate did not pass.")
    paths = _protocol_input_paths()
    missing = [
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in paths
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing V4 execution inputs: {missing}")
    manifest = {
        "execution_id": EXECUTION_ID,
        "frozen_at_utc": _now(),
        "implementation_sha256": _implementation_hashes(),
        "input_sha256": {
            path.relative_to(PROJECT_ROOT).as_posix(): sha256_file(path)
            for path in paths
        },
        "schema_version": (
            "external-behavior-v4-execution-input-manifest-v1"
        ),
        "status": "frozen_before_execution_bundle",
    }
    write_json(PROTOCOL_DIR / "input_manifest.json", manifest)
    receipt = {
        "execution_id": EXECUTION_ID,
        "frozen_at_utc": _now(),
        "schema_version": (
            "external-behavior-v4-execution-freeze-receipt-v1"
        ),
        "state_at_freeze": {
            "execution_artifacts_absent": True,
            "initialization_hashes_materialized": False,
            "run_root_absent": True,
            "tensor_cache_materialized": False,
            "training_started": False,
        },
        "status": "frozen_before_execution_bundle",
    }
    write_json(PROTOCOL_DIR / "freeze_receipt.json", receipt)
    report = validate_protocol(require_execution_absent=True)
    if report["status"] != "pass":
        raise RuntimeError(f"V4 execution protocol failed: {report}")
    write_json(PROTOCOL_DIR / "validation_report.json", report)
    names = (
        "README.md",
        "checklist.md",
        "freeze_receipt.json",
        "input_manifest.json",
        "preregistration.json",
        "validate.py",
        "validation_report.json",
    )
    (PROTOCOL_DIR / "SHA256SUMS").write_text(
        "".join(
            f"{sha256_file(PROTOCOL_DIR / name)}  {name}\n"
            for name in names
        ),
        encoding="ascii",
    )
    return report


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
        "maximum_concurrent_runs": 8,
        "platform": platform.platform(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "torch_deterministic_algorithms": True,
        "torch_interop_threads_per_run": 1,
        "torch_intraop_threads_per_run": 2,
    }
    return environment, "\n".join(freeze_lines) + "\n"


def freeze_execution() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_DIR / "validation_report.json")
    if protocol.get("status") != "pass":
        raise RuntimeError("V4 execution protocol has not passed.")
    if EXECUTION_DIR.exists() or RUN_ROOT.exists():
        raise RuntimeError("V4 execution bundle or runs already exist.")
    EXECUTION_DIR.mkdir(parents=True)
    environment, lock_text = _environment()
    environment_path = EXECUTION_DIR / "environment.json"
    write_json(environment_path, environment)
    requirements_path = EXECUTION_DIR / "requirements.lock"
    requirements_path.write_text(lock_text, encoding="ascii")
    write_json(EXECUTION_DIR / "classes.json", {"classes": classes()})
    cache = build_tensor_cache()
    if cache["cached_rows"] != 43_840:
        raise RuntimeError(
            f"Unexpected V4 tensor-cache rows: {cache['cached_rows']}"
        )
    initialization = initialization_manifest()
    write_json(
        EXECUTION_DIR / "initialization_manifest.json",
        initialization,
    )
    orders = epoch_order_manifest()
    write_json(EXECUTION_DIR / "epoch_order_manifest.json", orders)
    run_rows = build_run_manifest(initialization, orders)
    write_csv(EXECUTION_DIR / "run_manifest.csv", run_rows)
    implementation_hashes = _implementation_hashes()
    manifest = {
        "cache": cache,
        "classes_sha256": sha256_file(EXECUTION_DIR / "classes.json"),
        "environment_sha256": sha256_file(environment_path),
        "epoch_order_manifest_sha256": sha256_file(
            EXECUTION_DIR / "epoch_order_manifest.json"
        ),
        "execution_id": EXECUTION_ID,
        "frozen_at_utc": _now(),
        "frozen_before_training": True,
        "implementation_sha256": implementation_hashes,
        "initialization_manifest_sha256": sha256_file(
            EXECUTION_DIR / "initialization_manifest.json"
        ),
        "materialization_validation_sha256": sha256_file(
            MATERIALIZATION_DIR
            / "materialization_validation_report.json"
        ),
        "protocol_preregistration_sha256": sha256_file(
            PROTOCOL_DIR / "preregistration.json"
        ),
        "requirements_lock_sha256": sha256_file(requirements_path),
        "run_count": RUN_COUNT,
        "run_manifest_sha256": sha256_file(
            EXECUTION_DIR / "run_manifest.csv"
        ),
        "schema_version": (
            "external-behavior-v4-execution-manifest-v1"
        ),
        "status": "frozen_before_training",
    }
    manifest_path = EXECUTION_DIR / "execution_manifest.json"
    write_json(manifest_path, manifest)
    frozen_paths = [
        EXECUTION_DIR / "classes.json",
        environment_path,
        EXECUTION_DIR / "epoch_order_manifest.json",
        EXECUTION_DIR / "evaluation_block0.npz",
        EXECUTION_DIR / "evaluation_block1.npz",
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


def _freeze_checklist(checks: list[dict[str, Any]]) -> str:
    status = {item["name"]: item["pass"] for item in checks}
    entries = (
        ("artifact_hashes", "Validate every frozen bundle artifact."),
        ("tensor_cache", "Validate the 43,840-row tensor cache."),
        ("train_indices", "Validate all 32 train-index arrays."),
        ("evaluations", "Validate both 8,000-row evaluation caches."),
        ("initialization", "Validate eight initialization records."),
        ("epoch_orders", "Validate all 240 epoch-order hashes."),
        ("run_manifest", "Validate all 32 frozen run records."),
        ("four_way_pairing", "Validate four-way pairing within seed."),
        ("training_absent", "Confirm no V4 training has started."),
    )
    lines = ["# Directional V4 Execution Freeze Integrity", ""]
    for name, label in entries:
        lines.append(f"- [{'x' if status.get(name) else ' '}] {label}")
    lines.append("")
    return "\n".join(lines)


def validate_execution_freeze() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: Any = None) -> None:
        checks.append({"detail": detail, "name": name, "pass": bool(condition)})

    checksum_failures: list[str] = []
    for expected, path in _checksum_entries(
        EXECUTION_DIR / "EXECUTION_IMPLEMENTATION_SHA256SUMS"
    ):
        if not path.is_file() or sha256_file(path) != expected:
            checksum_failures.append(path.name)
    check("artifact_hashes", not checksum_failures, checksum_failures)
    cache = load_npz(EXECUTION_DIR / "tensor_cache.npz")
    cache_ok = (
        set(cache)
        == {"attention_mask", "labels", "row_ids", "token_ids"}
        and cache["token_ids"].shape == (43_840, FIXED_LENGTH)
        and cache["attention_mask"].shape == (43_840, FIXED_LENGTH)
        and cache["labels"].shape == (43_840,)
        and cache["row_ids"].shape == (43_840,)
        and len(set(cache["row_ids"].tolist())) == 43_840
    )
    check("tensor_cache", cache_ok, tensor_cache_summary(
        EXECUTION_DIR / "tensor_cache.npz"
    ))
    cache_row_ids = cache["row_ids"].astype("U64")
    train_archive = load_npz(
        EXECUTION_DIR / "train_cache_indices.npz"
    )
    materialized = read_csv(MATERIALIZATION_DIR / "run_manifest.csv")
    train_failures: list[str] = []
    for row in materialized:
        identifier = row["run_id"]
        if identifier not in train_archive:
            train_failures.append(f"missing:{identifier}")
            continue
        indices = train_archive[identifier]
        expected_ids = (
            MATERIALIZATION_DIR
            / "data"
            / row["train_manifest"]
        ).read_text(encoding="ascii").splitlines()
        if (
            indices.shape != (TRAIN_ROWS,)
            or indices.dtype != np.int32
            or not np.array_equal(
                cache_row_ids[indices],
                np.asarray(expected_ids),
            )
        ):
            train_failures.append(identifier)
    check(
        "train_indices",
        set(train_archive) == {row["run_id"] for row in materialized}
        and not train_failures,
        train_failures,
    )
    evaluation_failures: list[str] = []
    class_to_id = {
        name: index for index, name in enumerate(classes())
    }
    for block in BLOCKS:
        observed = load_npz(
            EXECUTION_DIR / f"evaluation_block{block}.npz"
        )
        source = list(
            iter_jsonl(
                MATERIALIZATION_DIR
                / "data"
                / f"evaluation_block{block}.jsonl"
            )
        )
        if (
            len(source) != EVAL_ROWS
            or observed["cache_indices"].shape != (EVAL_ROWS,)
            or not np.array_equal(
                cache_row_ids[observed["cache_indices"]],
                np.asarray([row["surface_id"] for row in source]),
            )
            or not np.array_equal(
                observed["labels"],
                np.asarray(
                    [
                        class_to_id[row["target_relation"]]
                        for row in source
                    ],
                    dtype=np.int16,
                ),
            )
        ):
            evaluation_failures.append(str(block))
    check("evaluations", not evaluation_failures, evaluation_failures)
    initialization = read_json(
        EXECUTION_DIR / "initialization_manifest.json"
    )
    init_failures: list[str] = []
    for row in initialization["records"]:
        seed = int(row["model_seed"])
        model, model_hash, rng_hash = initialize_model(seed)
        if (
            model_hash != row["initial_model_sha256"]
            or rng_hash != row["training_start_torch_rng_sha256"]
        ):
            init_failures.append(str(seed))
        del model
    check(
        "initialization",
        len(initialization["records"]) == 8 and not init_failures,
        init_failures,
    )
    orders = read_json(EXECUTION_DIR / "epoch_order_manifest.json")
    order_failures: list[str] = []
    for row in orders["records"]:
        seed = int(row["model_seed"])
        epoch = int(row["epoch"])
        if array_sha256(epoch_order(seed, epoch)) != row["order_sha256"]:
            order_failures.append(f"{seed}:{epoch}")
    check(
        "epoch_orders",
        len(orders["records"]) == 240 and not order_failures,
        order_failures,
    )
    run_rows = read_csv(EXECUTION_DIR / "run_manifest.csv")
    run_failures: list[str] = []
    for row in run_rows:
        train = MATERIALIZATION_DIR / "data" / row["train_manifest"]
        evaluation = EXECUTION_DIR / row["evaluation_manifest"]
        if (
            sha256_file(train) != row["train_manifest_sha256"]
            or sha256_file(evaluation)
            != row["evaluation_manifest_sha256"]
        ):
            run_failures.append(row["run_id"])
    check(
        "run_manifest",
        len(run_rows) == RUN_COUNT
        and len({row["run_id"] for row in run_rows}) == RUN_COUNT
        and not run_failures,
        run_failures,
    )
    pairing_failures: list[str] = []
    for seed in MODEL_SEEDS:
        selected = [
            row for row in run_rows if int(row["model_seed"]) == seed
        ]
        if (
            len(selected) != 4
            or len(
                {row["expected_initial_model_sha256"] for row in selected}
            )
            != 1
            or len(
                {
                    row["expected_training_start_torch_rng_sha256"]
                    for row in selected
                }
            )
            != 1
            or len(
                {row["combined_epoch_order_sha256"] for row in selected}
            )
            != 1
        ):
            pairing_failures.append(str(seed))
    check(
        "four_way_pairing",
        not pairing_failures,
        pairing_failures,
    )
    check(
        "training_absent",
        not RUN_ROOT.exists(),
        str(RUN_ROOT),
    )
    failed = [item for item in checks if not item["pass"]]
    report = {
        "check_count": len(checks),
        "checks": checks,
        "execution_freeze_gate": "pass" if not failed else "no_go",
        "execution_id": EXECUTION_ID,
        "failed_checks": [item["name"] for item in failed],
        "failed_count": len(failed),
        "schema_version": (
            "external-behavior-v4-execution-freeze-validation-v1"
        ),
        "status": "pass" if not failed else "no_go",
        "training_started": False,
        "validated_at_utc": _now(),
    }
    checklist = EXECUTION_DIR / "freeze_checklist.md"
    checklist.write_text(_freeze_checklist(checks), encoding="ascii")
    report_path = EXECUTION_DIR / "freeze_validation_report.json"
    write_json(report_path, report)
    frozen_names = (
        "EXECUTION_IMPLEMENTATION_SHA256SUMS",
        "classes.json",
        "environment.json",
        "epoch_order_manifest.json",
        "evaluation_block0.npz",
        "evaluation_block1.npz",
        "execution_manifest.json",
        "freeze_checklist.md",
        "freeze_validation_report.json",
        "initialization_manifest.json",
        "requirements.lock",
        "run_manifest.csv",
        "tensor_cache.npz",
        "train_cache_indices.npz",
    )
    (EXECUTION_DIR / "EXECUTION_FREEZE_SHA256SUMS").write_text(
        "".join(
            f"{sha256_file(EXECUTION_DIR / name)}  {name}\n"
            for name in frozen_names
        ),
        encoding="ascii",
    )
    return report


def freeze_all() -> dict[str, Any]:
    protocol = freeze_protocol()
    execution = freeze_execution()
    validation = validate_execution_freeze()
    if validation["status"] != "pass":
        raise RuntimeError(f"V4 execution freeze failed: {validation}")
    return {
        "execution_id": EXECUTION_ID,
        "execution_manifest_sha256": sha256_file(
            EXECUTION_DIR / "execution_manifest.json"
        ),
        "freeze_validation_status": validation["status"],
        "protocol_validation_status": protocol["status"],
        "run_count": execution["run_count"],
        "status": "frozen_before_training",
        "tensor_cache_rows": execution["cache"]["cached_rows"],
    }
