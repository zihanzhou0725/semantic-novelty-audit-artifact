"""Integrity validation for the 16 CLUTRR V3 runs."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from toolkit.external_behavior.common import iter_jsonl

from .training_common import (
    ARMS,
    DATA_DIR,
    EPOCHS,
    EVAL_ROWS,
    EXECUTION_DIR,
    MODEL_SEEDS,
    PROJECT_ROOT,
    RUN_ROOT,
    UPDATES_PER_RUN,
    array_sha256,
    artifact_hashes,
    epoch_order,
    finite_tree,
    load_npz,
    read_csv,
    read_json,
    sha256_file,
    state_dict_sha256,
    write_json,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ece(probabilities: np.ndarray, labels: np.ndarray) -> float:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == labels
    result = 0.0
    for index in range(15):
        lower = index / 15
        upper = (index + 1) / 15
        selected = (
            (confidence >= lower) & (confidence <= upper)
            if index == 14
            else (confidence >= lower) & (confidence < upper)
        )
        if selected.any():
            result += float(selected.mean()) * abs(
                float(correct[selected].mean())
                - float(confidence[selected].mean())
            )
    return result


def _prediction_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    predictions = logits.argmax(axis=1)
    one_hot = np.eye(23, dtype=np.float32)[labels]
    return {
        "accuracy": float((predictions == labels).mean()),
        "brier_score": float(
            np.square(probabilities - one_hot).sum(axis=1).mean()
        ),
        "cross_entropy": float(
            (
                -shifted[np.arange(len(labels)), labels]
                + np.log(exp_logits.sum(axis=1))
            ).mean()
        ),
        "ece_15": _ece(probabilities, labels),
    }


def _checklist(checks: list[dict[str, Any]]) -> str:
    status = {item["name"]: item["pass"] for item in checks}
    entries = [
        ("Complete all 16 frozen runs.", "run_count"),
        ("Reach epoch 30 and update 5,640 for every run.", "progress"),
        ("Validate every run artifact hash.", "artifact_hashes"),
        ("Validate finite model and optimizer states.", "checkpoint_finite"),
        ("Validate 30 frozen epoch orders per run.", "epoch_orders"),
        ("Validate frozen paired initializations.", "initialization"),
        ("Validate paired initial and final Torch RNG hashes.", "paired_rng"),
        ("Validate 8,000 ordered predictions per run.", "predictions"),
        ("Independently recompute each run's metrics.", "metrics"),
        ("Validate frozen train manifests and complementary exposure.", "training_rows"),
        ("Confirm no paired effect aggregate was computed.", "aggregate_blindness"),
    ]
    lines = ["# External Behavioral Validation v3 Execution Integrity", ""]
    for label, name in entries:
        lines.append(f"- [{'x' if status.get(name, False) else ' '}] {label}")
    lines.append("")
    return "\n".join(lines)


def validate_execution() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: Any = None) -> None:
        checks.append({"detail": detail, "name": name, "pass": bool(condition)})

    execution = read_json(EXECUTION_DIR / "execution_manifest.json")
    rows = read_csv(EXECUTION_DIR / "run_manifest.csv")
    check(
        "run_count",
        len(rows) == 16
        and all((RUN_ROOT / row["run_id"]).is_dir() for row in rows),
        len(rows),
    )
    expected_eval = load_npz(EXECUTION_DIR / "evaluation_manifest.npz")
    expected_epoch_manifest = read_json(
        EXECUTION_DIR / "epoch_order_manifest.json"
    )
    expected_epoch_hash = {
        (int(row["model_seed"]), int(row["epoch"])): row["order_sha256"]
        for row in expected_epoch_manifest["records"]
    }
    initialization = read_json(
        EXECUTION_DIR / "initialization_manifest.json"
    )
    init_by_seed = {
        int(row["model_seed"]): row for row in initialization["records"]
    }

    progress_failures: list[str] = []
    artifact_failures: list[str] = []
    checkpoint_failures: list[str] = []
    epoch_failures: list[str] = []
    initialization_failures: list[str] = []
    prediction_failures: list[str] = []
    metric_failures: list[str] = []
    training_failures: list[str] = []
    rng_by_seed: dict[int, dict[str, tuple[str, str]]] = defaultdict(dict)
    run_artifacts: list[dict[str, Any]] = []

    exposure_rows = list(
        iter_jsonl(DATA_DIR / "clutrr_v3_exposure.jsonl")
    )
    exposure_by_surface = {
        row["surface_id"]: row for row in exposure_rows
    }
    pairs = list(iter_jsonl(DATA_DIR / "clutrr_v3_pairs.jsonl"))

    for row in rows:
        seed = int(row["model_seed"])
        arm = row["arm"]
        run_id = row["run_id"]
        run_dir = RUN_ROOT / run_id
        completed_path = run_dir / "COMPLETED.json"
        if not completed_path.is_file():
            artifact_failures.append(f"missing_completed:{run_id}")
            continue
        completed = read_json(completed_path)
        for filename, expected in completed["artifacts_sha256"].items():
            path = run_dir / filename
            if not path.is_file() or sha256_file(path) != expected:
                artifact_failures.append(f"{run_id}:{filename}")
        progress = read_json(run_dir / "progress.json")
        if (
            progress.get("status") != "completed"
            or progress.get("epoch") != EPOCHS
            or progress.get("optimizer_updates") != UPDATES_PER_RUN
        ):
            progress_failures.append(run_id)

        config = read_json(run_dir / "config.json")
        if (
            config["train_manifest_sha256"]
            != row["train_manifest_sha256"]
            or config["expected_initial_model_sha256"]
            != row["expected_initial_model_sha256"]
            or config["model_seed"] != seed
            or config["arm"] != arm
        ):
            initialization_failures.append(f"config:{run_id}")
        history = read_csv(run_dir / "history.csv")
        if len(history) != EPOCHS:
            epoch_failures.append(f"history:{run_id}")
        else:
            for epoch, history_row in enumerate(history, start=1):
                expected_hash = expected_epoch_hash[(seed, epoch)]
                regenerated_hash = array_sha256(epoch_order(seed, epoch))
                if (
                    history_row["epoch_order_sha256"] != expected_hash
                    or regenerated_hash != expected_hash
                    or int(history_row["optimizer_updates"]) != epoch * 188
                    or not math.isfinite(float(history_row["train_loss"]))
                    or not math.isfinite(
                        float(history_row["train_accuracy"])
                    )
                ):
                    epoch_failures.append(f"{run_id}:{epoch}")

        checkpoint = torch.load(
            run_dir / "final_checkpoint.pt",
            map_location="cpu",
            weights_only=False,
        )
        if (
            checkpoint["epoch"] != EPOCHS
            or checkpoint["optimizer_updates"] != UPDATES_PER_RUN
            or not finite_tree(checkpoint["model_state"])
            or not finite_tree(checkpoint["optimizer_state"])
        ):
            checkpoint_failures.append(run_id)
        final_model_hash = state_dict_sha256(checkpoint["model_state"])
        if (
            checkpoint["initial_model_sha256"]
            != init_by_seed[seed]["initial_model_sha256"]
            or checkpoint["initial_model_sha256"]
            != row["expected_initial_model_sha256"]
            or checkpoint["training_start_torch_rng_sha256"]
            != init_by_seed[seed]["training_start_torch_rng_sha256"]
            or final_model_hash != checkpoint["final_model_sha256"]
        ):
            initialization_failures.append(run_id)
        rng_by_seed[seed][arm] = (
            checkpoint["training_start_torch_rng_sha256"],
            checkpoint["final_torch_rng_sha256"],
        )

        predictions = load_npz(run_dir / "eval_predictions.npz")
        required_prediction_keys = {
            "labels",
            "logits",
            "pair_index",
            "pair_member",
            "predictions",
            "row_ids",
            "semantic_ids",
            "source_group_ids",
        }
        if (
            set(predictions) != required_prediction_keys
            or predictions["logits"].shape != (EVAL_ROWS, 23)
            or not np.isfinite(predictions["logits"]).all()
            or not np.array_equal(
                predictions["row_ids"], expected_eval["row_ids"]
            )
            or not np.array_equal(
                predictions["semantic_ids"],
                expected_eval["semantic_ids"],
            )
            or not np.array_equal(
                predictions["source_group_ids"],
                expected_eval["source_group_ids"],
            )
            or not np.array_equal(
                predictions["labels"], expected_eval["labels"]
            )
            or not np.array_equal(
                predictions["pair_index"], expected_eval["pair_index"]
            )
            or not np.array_equal(
                predictions["pair_member"], expected_eval["pair_member"]
            )
            or not np.array_equal(
                predictions["predictions"],
                predictions["logits"].argmax(axis=1),
            )
        ):
            prediction_failures.append(run_id)
        metrics = read_json(run_dir / "metrics.json")
        recomputed = _prediction_metrics(
            predictions["logits"],
            predictions["labels"].astype(np.int64),
        )
        if any(
            not math.isclose(
                recomputed[name],
                float(metrics[name]),
                rel_tol=1e-7,
                abs_tol=1e-8,
            )
            for name in recomputed
        ):
            metric_failures.append(run_id)

        train_path = DATA_DIR / row["train_manifest"]
        train_ids = train_path.read_text(encoding="ascii").splitlines()
        if (
            len(train_ids) != 24000
            or sha256_file(train_path) != row["train_manifest_sha256"]
        ):
            training_failures.append(f"manifest:{run_id}")
        exposure_counts = Counter(
            exposure_by_surface[row_id]["semantic_id"]
            for row_id in train_ids
            if row_id in exposure_by_surface
        )
        for pair in pairs:
            first = pair["member_0"]
            second = pair["member_1"]
            counts = (exposure_counts[first], exposure_counts[second])
            if sorted(counts) != [0, 64]:
                training_failures.append(
                    f"exposure:{run_id}:{pair['pair_index']}"
                )
        run_artifacts.append(
            {
                "artifacts_sha256": completed["artifacts_sha256"],
                "completed_sha256": sha256_file(completed_path),
                "run_id": run_id,
            }
        )

    rng_failures: list[int] = []
    for seed in MODEL_SEEDS:
        if set(rng_by_seed[seed]) != set(ARMS):
            rng_failures.append(seed)
            continue
        if rng_by_seed[seed]["A"] != rng_by_seed[seed]["B"]:
            rng_failures.append(seed)

    check("progress", not progress_failures, progress_failures)
    check("artifact_hashes", not artifact_failures, artifact_failures)
    check(
        "checkpoint_finite",
        not checkpoint_failures,
        checkpoint_failures,
    )
    check("epoch_orders", not epoch_failures, epoch_failures[:100])
    check(
        "initialization",
        not initialization_failures,
        initialization_failures,
    )
    check("paired_rng", not rng_failures, rng_failures)
    check("predictions", not prediction_failures, prediction_failures)
    check("metrics", not metric_failures, metric_failures)
    check("training_rows", not training_failures, training_failures[:100])
    analysis_root = PROJECT_ROOT / "analysis" / "external-behavior-v3"
    aggregate_files = (
        list(analysis_root.rglob("*")) if analysis_root.exists() else []
    )
    check(
        "aggregate_blindness",
        not aggregate_files
        and read_json(RUN_ROOT / "execution_state.json")[
            "primary_effect_computed"
        ]
        is False,
        [str(path) for path in aggregate_files],
    )

    required = {
        "run_count",
        "progress",
        "artifact_hashes",
        "checkpoint_finite",
        "epoch_orders",
        "initialization",
        "paired_rng",
        "predictions",
        "metrics",
        "training_rows",
        "aggregate_blindness",
    }
    by_name = {item["name"]: item["pass"] for item in checks}
    passed = all(by_name[name] for name in required)
    write_json(
        EXECUTION_DIR / "run_artifact_manifest.json",
        {
            "runs": sorted(run_artifacts, key=lambda item: item["run_id"]),
            "schema_version": "external-behavior-v3-run-artifacts-v1",
        },
    )
    report = {
        "check_count": len(checks),
        "checks": checks,
        "execution_gate": "pass" if passed else "no_go",
        "failed_checks": [
            item["name"] for item in checks if not item["pass"]
        ],
        "failed_count": sum(not item["pass"] for item in checks),
        "primary_effect_computed": False,
        "schema_version": "external-behavior-v3-execution-validation-v1",
        "status": "pass" if passed else "no_go",
        "validated_at_utc": _now(),
    }
    checklist_path = EXECUTION_DIR / "execution_checklist.md"
    checklist_path.write_text(_checklist(checks), encoding="ascii")
    report_path = EXECUTION_DIR / "execution_validation_report.json"
    write_json(report_path, report)
    freeze_paths = [
        EXECUTION_DIR / "EXECUTION_IMPLEMENTATION_SHA256SUMS",
        EXECUTION_DIR / "environment.json",
        EXECUTION_DIR / "execution_manifest.json",
        EXECUTION_DIR / "run_manifest.csv",
        EXECUTION_DIR / "initialization_manifest.json",
        EXECUTION_DIR / "epoch_order_manifest.json",
        EXECUTION_DIR / "tensor_cache.npz",
        EXECUTION_DIR / "train_cache_indices.npz",
        EXECUTION_DIR / "evaluation_manifest.npz",
        EXECUTION_DIR / "run_artifact_manifest.json",
        checklist_path,
        report_path,
        RUN_ROOT / "execution_state.json",
        RUN_ROOT / "matrix_status.json",
    ]
    (EXECUTION_DIR / "EXECUTION_SHA256SUMS").write_text(
        "".join(
            f"{sha256_file(path)}  "
            f"{path.relative_to(PROJECT_ROOT).as_posix()}\n"
            for path in freeze_paths
        ),
        encoding="ascii",
    )
    return report
