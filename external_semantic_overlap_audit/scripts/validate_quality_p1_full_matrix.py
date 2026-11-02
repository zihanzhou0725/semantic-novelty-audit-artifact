#!/usr/bin/env python3
"""Validate the completed 60-run P1 matrix without comparing arm outcomes."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from quality_p1_common import (
    ARMS,
    BATCH_SIZE,
    EPOCHS,
    EVAL_EPISODES_PER_FUNCTION,
    EXTERNAL_ROOT,
    MODEL_SEEDS,
    MODELS,
    P1_DIR,
    P1_SCHEMA_VERSION,
    TRAIN_EPISODES,
    load_npz,
    parse_checksum_file,
    sha256_file,
    validate_checksum_file,
    write_json,
)


RUN_ROOT = EXTERNAL_ROOT / "runs" / "quality-enhancement-v1" / "p1"
EXPECTED_PREDICTION_DTYPES = {
    "correct": np.dtype(np.bool_),
    "episode_within_function": np.dtype(np.int16),
    "exposed": np.dtype(np.bool_),
    "function_id": np.dtype(np.int64),
    "function_multiplier": np.dtype(np.int8),
    "logits": np.dtype(np.float32),
    "pair_intercept": np.dtype(np.int8),
    "per_episode_brier": np.dtype(np.float32),
    "per_episode_ce": np.dtype(np.float32),
    "prediction": np.dtype(np.uint8),
    "probabilities": np.dtype(np.float32),
    "target": np.dtype(np.uint8),
}
EXPECTED_PREDICTION_SHAPES = {
    "correct": (10_000,),
    "episode_within_function": (10_000,),
    "exposed": (10_000,),
    "function_id": (10_000,),
    "function_multiplier": (10_000,),
    "logits": (10_000, 20),
    "pair_intercept": (10_000,),
    "per_episode_brier": (10_000,),
    "per_episode_ce": (10_000,),
    "prediction": (10_000,),
    "probabilities": (10_000, 20),
    "target": (10_000,),
}
REQUIRED_RUN_FILES = (
    "config.json",
    "history.csv",
    "final_model.pt",
    "eval_predictions.npz",
    "metrics.json",
    "progress.json",
)


def fail(message: str) -> None:
    raise SystemExit(f"P1 full-matrix validation failed: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def require_close(
    observed: float,
    expected: float,
    message: str,
    *,
    atol: float = 2e-6,
) -> None:
    require(
        math.isfinite(observed)
        and math.isfinite(expected)
        and math.isclose(observed, expected, rel_tol=0.0, abs_tol=atol),
        f"{message}: {observed} != {expected}",
    )


def recursively_finite(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all())
    if isinstance(value, dict):
        return all(recursively_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(recursively_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def load_manifest_rows() -> list[dict[str, str]]:
    with (P1_DIR / "run_manifest.csv").open(
        newline="", encoding="ascii"
    ) as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 60, "frozen run manifest does not have 60 rows")
    require(
        len({row["run_id"] for row in rows}) == 60,
        "frozen run IDs are not unique",
    )
    return rows


def validate_matrix_state(expected_run_ids: list[str]) -> tuple[dict, dict]:
    validation_path = P1_DIR / "validation_report.json"
    design_validation = json.loads(validation_path.read_text(encoding="ascii"))
    require(
        design_validation.get("status") == "passed"
        and design_validation.get("design_gate") == "passed",
        "P1 Design Gate is not passed",
    )
    validate_checksum_file(P1_DIR, P1_DIR / "SHA256SUMS")

    matrix_path = RUN_ROOT / "matrix_status.json"
    matrix = json.loads(matrix_path.read_text(encoding="ascii"))
    require(matrix.get("protocol") == P1_SCHEMA_VERSION, "matrix protocol")
    require(matrix.get("state") == "completed", "matrix is not completed")
    require(matrix.get("total") == 60, "matrix total is not 60")
    require(matrix.get("completed") == 60, "matrix completed count is not 60")
    require(matrix.get("failures") == [], "matrix records failures")
    require(matrix.get("current_run_id") is None, "matrix still has a current run")
    require(
        matrix.get("completed_run_ids") == expected_run_ids,
        "matrix completion order differs from frozen run manifest",
    )
    require(
        matrix.get("design_validation_sha256") == sha256_file(validation_path),
        "matrix refers to another Design Gate report",
    )

    execution_path = RUN_ROOT / "execution_state.json"
    execution = json.loads(execution_path.read_text(encoding="ascii"))
    require(execution.get("protocol") == P1_SCHEMA_VERSION, "execution protocol")
    require(execution.get("training_started") is True, "training not marked started")
    require(
        execution.get("design_validation_sha256") == sha256_file(validation_path),
        "execution state refers to another Design Gate report",
    )
    require(
        execution.get("design_checksum_manifest_sha256")
        == sha256_file(P1_DIR / "SHA256SUMS"),
        "execution state refers to another design checksum manifest",
    )
    return matrix, execution


def expected_exposed_ids(
    train_arrays: dict[str, np.ndarray],
    arm: str,
) -> list[int]:
    target_mask = train_arrays["slot_kind"] == 1
    return sorted(
        set(
            int(value)
            for value in train_arrays[f"function_id_{arm}"][target_mask]
        )
    )


def validate_config(
    config: dict[str, Any],
    row: dict[str, str],
    exposed_ids: list[int],
) -> None:
    run_id = row["run_id"]
    seed = int(row["model_seed"])
    arm = row["arm"]
    expected_scalars = {
        "arm": arm,
        "batch_size": BATCH_SIZE,
        "checkpoint_selection": "fixed_final_epoch_60",
        "deterministic_algorithms": True,
        "dropout": 0.05,
        "epochs": EPOCHS,
        "ff_dim": 128,
        "gradient_clip_norm": 1.0,
        "heads": 4,
        "layers": 2,
        "learning_rate": 0.001,
        "model": row["model"],
        "model_dim": 64,
        "model_seed": seed,
        "optimizer": "AdamW",
        "optimizer_foreach": False,
        "protocol": P1_SCHEMA_VERSION,
        "public_latent_count": 32,
        "run_id": run_id,
        "support_noise": 0.1,
        "support_size": 8,
        "weight_decay": 0.0001,
    }
    for key, expected in expected_scalars.items():
        require(config.get(key) == expected, f"{run_id} config {key}")
    require(config.get("exposed_function_ids") == exposed_ids, f"{run_id} exposure")
    require(
        isinstance(config.get("parameters"), int) and config["parameters"] > 0,
        f"{run_id} parameter count",
    )
    require(config.get("device") in {"cpu", "mps", "cuda"}, f"{run_id} device")

    expected_inputs = {
        "train_manifest": P1_DIR / row["train_manifest"],
        "epoch_manifest": P1_DIR / row["epoch_manifest"],
        "eval_manifest": P1_DIR / row["eval_manifest"],
    }
    for key, expected_path in expected_inputs.items():
        observed_path = Path(config.get(key, ""))
        require(
            observed_path.resolve() == expected_path.resolve(),
            f"{run_id} {key} path",
        )
        require(
            config.get(f"{key}_sha256") == sha256_file(expected_path),
            f"{run_id} {key} hash",
        )


def validate_history(path: Path, run_id: str) -> None:
    with path.open(newline="", encoding="ascii") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == EPOCHS, f"{run_id} history row count")
    require(
        list(rows[0]) == [
            "epoch",
            "train_loss",
            "train_accuracy",
            "epoch_wall_seconds",
        ],
        f"{run_id} history fields",
    )
    for expected_epoch, row in enumerate(rows, start=1):
        require(int(row["epoch"]) == expected_epoch, f"{run_id} history epoch")
        loss = float(row["train_loss"])
        accuracy = float(row["train_accuracy"])
        wall_seconds = float(row["epoch_wall_seconds"])
        require(
            math.isfinite(loss) and loss >= 0.0,
            f"{run_id} nonfinite train loss",
        )
        require(
            math.isfinite(accuracy) and 0.0 <= accuracy <= 1.0,
            f"{run_id} invalid train accuracy",
        )
        require(
            math.isfinite(wall_seconds) and wall_seconds > 0.0,
            f"{run_id} invalid epoch time",
        )


def validate_checkpoint(path: Path, run_id: str, parameters: int) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    require(
        isinstance(checkpoint, dict) and checkpoint,
        f"{run_id} checkpoint is not a nonempty state dict",
    )
    require(recursively_finite(checkpoint), f"{run_id} checkpoint is nonfinite")
    observed_parameters = sum(
        int(value.numel())
        for value in checkpoint.values()
        if torch.is_tensor(value)
    )
    require(
        observed_parameters == parameters,
        f"{run_id} checkpoint parameter count",
    )


def validate_predictions(
    predictions: dict[str, np.ndarray],
    eval_arrays: dict[str, np.ndarray],
    exposed_ids: set[int],
    metrics: dict[str, Any],
    run_id: str,
) -> None:
    require(
        set(predictions) == set(EXPECTED_PREDICTION_DTYPES),
        f"{run_id} prediction fields differ",
    )
    for name in sorted(EXPECTED_PREDICTION_DTYPES):
        require(
            predictions[name].dtype == EXPECTED_PREDICTION_DTYPES[name],
            f"{run_id} prediction dtype {name}",
        )
        require(
            predictions[name].shape == EXPECTED_PREDICTION_SHAPES[name],
            f"{run_id} prediction shape {name}",
        )

    for name in (
        "episode_within_function",
        "function_id",
        "function_multiplier",
        "pair_intercept",
        "target",
    ):
        require(
            np.array_equal(predictions[name], eval_arrays[name]),
            f"{run_id} prediction metadata {name}",
        )
    expected_exposed = np.asarray(
        [int(value) in exposed_ids for value in eval_arrays["function_id"]],
        dtype=np.bool_,
    )
    require(
        np.array_equal(predictions["exposed"], expected_exposed),
        f"{run_id} episode exposure flags",
    )

    logits = predictions["logits"]
    probabilities = predictions["probabilities"]
    target = predictions["target"].astype(np.int64)
    require(np.isfinite(logits).all(), f"{run_id} nonfinite logits")
    require(np.isfinite(probabilities).all(), f"{run_id} nonfinite probabilities")
    require(
        np.all((probabilities >= 0.0) & (probabilities <= 1.0)),
        f"{run_id} probabilities outside [0,1]",
    )
    require(
        np.allclose(
            probabilities.sum(axis=1),
            np.ones(len(probabilities)),
            rtol=0.0,
            atol=2e-6,
        ),
        f"{run_id} probabilities do not sum to one",
    )
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    reconstructed_probabilities = np.exp(shifted)
    reconstructed_probabilities /= reconstructed_probabilities.sum(
        axis=1, keepdims=True
    )
    require(
        np.allclose(
            probabilities,
            reconstructed_probabilities,
            rtol=0.0,
            atol=2e-6,
        ),
        f"{run_id} logits/probabilities mismatch",
    )
    prediction = logits.argmax(axis=1).astype(np.uint8)
    correct = prediction == predictions["target"]
    require(
        np.array_equal(predictions["prediction"], prediction),
        f"{run_id} predictions are not logit argmax",
    )
    require(
        np.array_equal(predictions["correct"], correct),
        f"{run_id} correctness flags differ",
    )

    rows = np.arange(len(target))
    ce = -np.log(reconstructed_probabilities[rows, target])
    one_hot = np.eye(20, dtype=np.float64)[target]
    brier = np.square(reconstructed_probabilities - one_hot).sum(axis=1)
    require(
        np.allclose(
            predictions["per_episode_ce"], ce, rtol=0.0, atol=5e-5
        ),
        f"{run_id} per-episode CE mismatch",
    )
    require(
        np.allclose(
            predictions["per_episode_brier"], brier, rtol=0.0, atol=5e-6
        ),
        f"{run_id} per-episode Brier mismatch",
    )

    accuracy = float(correct.mean())
    require_close(float(metrics["eval_accuracy"]), accuracy, f"{run_id} accuracy")
    require_close(float(metrics["eval_loss"]), float(ce.mean()), f"{run_id} loss")
    require_close(float(metrics["eval_ce"]), float(ce.mean()), f"{run_id} CE")
    require_close(
        float(metrics["eval_brier"]),
        float(brier.mean()),
        f"{run_id} Brier",
    )
    for key in (
        "eval_context_0_accuracy",
        "eval_out_support_accuracy",
        "eval_unseen_accuracy",
    ):
        require_close(float(metrics[key]), accuracy, f"{run_id} {key}")
    require_close(
        float(metrics["eval_seen_function_accuracy"]),
        float(correct[expected_exposed].mean()),
        f"{run_id} seen-function accuracy",
    )
    require_close(
        float(metrics["eval_unseen_function_accuracy"]),
        float(correct[~expected_exposed].mean()),
        f"{run_id} unseen-function accuracy",
    )

    function_rows = metrics.get("per_function")
    require(
        isinstance(function_rows, list) and len(function_rows) == 40,
        f"{run_id} per-function metrics",
    )
    require(
        len({int(row["function_id"]) for row in function_rows}) == 40,
        f"{run_id} duplicate per-function metrics",
    )
    for row in function_rows:
        function_id = int(row["function_id"])
        mask = predictions["function_id"] == function_id
        require(int(mask.sum()) == EVAL_EPISODES_PER_FUNCTION, f"{run_id} function n")
        require(int(row["n"]) == EVAL_EPISODES_PER_FUNCTION, f"{run_id} row n")
        require(
            bool(row["exposed"]) == (function_id in exposed_ids),
            f"{run_id} function exposure",
        )
        require(
            int(row["multiplier"])
            == int(np.unique(predictions["function_multiplier"][mask])[0]),
            f"{run_id} function multiplier",
        )
        require(
            int(row["intercept"])
            == int(np.unique(predictions["pair_intercept"][mask])[0]),
            f"{run_id} function intercept",
        )
        require_close(
            float(row["accuracy"]),
            float(correct[mask].mean()),
            f"{run_id} function accuracy",
        )
        require_close(
            float(row["ce"]),
            float(ce[mask].mean()),
            f"{run_id} function CE",
        )
        require_close(
            float(row["brier"]),
            float(brier[mask].mean()),
            f"{run_id} function Brier",
        )


def validate_metrics(
    metrics: dict[str, Any],
    progress: dict[str, Any],
    row: dict[str, str],
    parameters: int,
) -> None:
    run_id = row["run_id"]
    expected = {
        "arm": row["arm"],
        "epochs_completed": EPOCHS,
        "exposed_function_count": 20,
        "final_checkpoint": "final_model.pt",
        "final_epoch": EPOCHS,
        "model": row["model"],
        "model_seed": int(row["model_seed"]),
        "optimizer_updates": EPOCHS
        * ((TRAIN_EPISODES + BATCH_SIZE - 1) // BATCH_SIZE),
        "parameters": parameters,
        "run_id": run_id,
    }
    for key, value in expected.items():
        require(metrics.get(key) == value, f"{run_id} metrics {key}")
    require(recursively_finite(metrics), f"{run_id} metrics contain nonfinite values")
    require(
        float(metrics["training_wall_seconds"]) > 0.0
        and float(metrics["evaluation_seconds"]) > 0.0,
        f"{run_id} timing metrics",
    )
    require(progress.get("status") == "completed", f"{run_id} progress status")
    require(progress.get("run_id") == run_id, f"{run_id} progress identity")
    require(progress.get("epoch_completed") == EPOCHS, f"{run_id} progress epoch")
    require_close(
        float(progress["eval_accuracy"]),
        float(metrics["eval_accuracy"]),
        f"{run_id} progress accuracy",
    )


def main() -> int:
    manifest_rows = load_manifest_rows()
    expected_run_ids = [row["run_id"] for row in manifest_rows]
    matrix, execution = validate_matrix_state(expected_run_ids)
    frozen_hashes = parse_checksum_file(P1_DIR / "SHA256SUMS")
    eval_arrays = load_npz(P1_DIR / "eval.npz")
    train_by_seed = {
        seed: load_npz(P1_DIR / f"train_seed{seed}.npz")
        for seed in MODEL_SEEDS
    }

    observed_run_dirs = sorted(
        path.name
        for path in RUN_ROOT.iterdir()
        if path.is_dir() and "__seed" in path.name
    )
    require(
        observed_run_dirs == sorted(expected_run_ids),
        "run directories differ from frozen manifest",
    )

    run_reports: dict[str, Any] = {}
    package_versions: set[tuple[str, str]] = set()
    paired_metadata: dict[tuple[str, int, str], dict[str, np.ndarray]] = {}
    total_training_seconds = 0.0
    for row in manifest_rows:
        run_id = row["run_id"]
        model = row["model"]
        seed = int(row["model_seed"])
        arm = row["arm"]
        require(model in MODELS, f"{run_id} model is not frozen")
        require(seed in MODEL_SEEDS, f"{run_id} seed is not frozen")
        require(arm in ARMS, f"{run_id} arm is not frozen")
        run_dir = RUN_ROOT / run_id
        for filename in REQUIRED_RUN_FILES:
            require((run_dir / filename).is_file(), f"{run_id} missing {filename}")

        config = json.loads((run_dir / "config.json").read_text(encoding="ascii"))
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="ascii"))
        progress = json.loads((run_dir / "progress.json").read_text(encoding="ascii"))
        exposed_ids = expected_exposed_ids(train_by_seed[seed], arm)
        require(len(exposed_ids) == 20, f"{run_id} does not expose 20 functions")
        validate_config(config, row, exposed_ids)
        validate_history(run_dir / "history.csv", run_id)
        validate_checkpoint(
            run_dir / "final_model.pt", run_id, int(config["parameters"])
        )
        validate_metrics(metrics, progress, row, int(config["parameters"]))
        predictions = load_npz(run_dir / "eval_predictions.npz")
        validate_predictions(
            predictions,
            eval_arrays,
            set(exposed_ids),
            metrics,
            run_id,
        )

        paired_metadata[(model, seed, arm)] = {
            key: predictions[key]
            for key in (
                "episode_within_function",
                "function_id",
                "function_multiplier",
                "pair_intercept",
                "target",
                "exposed",
            )
        }
        package_versions.add(
            (str(config.get("torch_version")), str(config.get("platform")))
        )
        total_training_seconds += float(metrics["training_wall_seconds"])
        artifact_hashes = {
            filename: sha256_file(run_dir / filename)
            for filename in REQUIRED_RUN_FILES
        }
        run_reports[run_id] = {
            "artifact_sha256": artifact_hashes,
            "input_manifest_sha256": {
                "train_manifest": config["train_manifest_sha256"],
                "epoch_manifest": config["epoch_manifest_sha256"],
                "eval_manifest": config["eval_manifest_sha256"],
            },
            "checkpoint_and_outputs_valid": True,
        }

    for model in MODELS:
        for seed in MODEL_SEEDS:
            arm_a = paired_metadata[(model, seed, "A")]
            arm_b = paired_metadata[(model, seed, "B")]
            for key in (
                "episode_within_function",
                "function_id",
                "function_multiplier",
                "pair_intercept",
                "target",
            ):
                require(
                    np.array_equal(arm_a[key], arm_b[key]),
                    f"{model} seed {seed} paired metadata {key}",
                )
            require(
                np.array_equal(arm_a["exposed"], ~arm_b["exposed"]),
                f"{model} seed {seed} exposure is not complementary",
            )

    report = {
        "checks": {
            "complete_frozen_60_run_matrix": "passed",
            "design_and_execution_hashes": "passed",
            "run_configs_match_frozen_matrix": "passed",
            "epoch_60_histories_complete_and_finite": "passed",
            "final_checkpoints_complete_and_finite": "passed",
            "prediction_schema_metadata_and_probabilities": "passed",
            "metrics_reconstructed_within_each_run": "passed",
            "common_eval_metadata_between_arms": "passed",
            "complementary_exposure_flags": "passed",
        },
        "effect_metrics_compared": False,
        "eligible_run_count": 60,
        "frozen_design_files": frozen_hashes,
        "full_run_count": 60,
        "matrix_status_sha256": sha256_file(RUN_ROOT / "matrix_status.json"),
        "execution_state_sha256": sha256_file(RUN_ROOT / "execution_state.json"),
        "package_versions": [
            {"torch_version": torch_version, "platform": platform}
            for torch_version, platform in sorted(package_versions)
        ],
        "protocol": P1_SCHEMA_VERSION,
        "runs": run_reports,
        "status": "passed",
        "total_training_seconds": total_training_seconds,
    }
    output_path = RUN_ROOT / "full_validation.json"
    write_json(output_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
