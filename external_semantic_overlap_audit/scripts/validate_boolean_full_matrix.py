#!/usr/bin/env python3
"""Validate the frozen 18-run Boolean behavioral result matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from freeze_boolean_behavioral_protocol import (
    EVAL_FUNCTION_DRAWS,
    N_POINTS,
    SCHEMA_VERSION,
    sha256_file,
)


FULL_STEPS = 20_000


def fail(message: str) -> None:
    raise SystemExit(f"Full-matrix validation failed: {message}")


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def all_tensors_finite(value: object) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(all_tensors_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_tensors_finite(item) for item in value)
    return True


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full-dir",
        type=Path,
        default=root / "runs" / SCHEMA_VERSION / "full",
    )
    args = parser.parse_args()

    protocol_dir = root / "protocols" / SCHEMA_VERSION
    matrix_status_path = (
        root / "runs" / SCHEMA_VERSION / "matrix" / "full_matrix_status.json"
    )
    matrix_status = json.loads(matrix_status_path.read_text())
    if matrix_status["state"] != "complete":
        fail(f"matrix state is {matrix_status['state']}, not complete")
    if matrix_status["failure"] is not None:
        fail("matrix status records a failure")
    if matrix_status["effect_metrics_examined"]:
        fail("matrix driver unexpectedly examined effect metrics")

    frozen_hashes = {}
    for line in (protocol_dir / "SHA256SUMS").read_text().splitlines():
        digest, filename = line.split("  ", maxsplit=1)
        frozen_hashes[filename] = digest
    split_manifest = json.loads((protocol_dir / "split_manifest.json").read_text())
    adapter_hash = split_manifest["training_adapter_sha256"]

    with (protocol_dir / "run_manifest.csv").open(newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    manifest_by_id = {row["run_id"]: row for row in manifest_rows}
    expected_run_ids = [row["run_id"] for row in manifest_rows]
    if len(expected_run_ids) != 18 or len(manifest_by_id) != 18:
        fail("frozen run manifest does not contain 18 unique runs")
    if matrix_status["completed_run_ids"] != expected_run_ids:
        fail("matrix completion order differs from frozen run manifest")

    eval_rows_by_task = defaultdict(list)
    for row in load_jsonl(protocol_dir / "eval_functions.jsonl"):
        eval_rows_by_task[row["task"]].append(row)

    observed_directories = sorted(
        path.name for path in args.full_dir.iterdir() if path.is_dir()
    )
    if observed_directories != sorted(expected_run_ids):
        fail("full output directories differ from the frozen 18-run matrix")

    pair_predictions = {}
    run_reports = {}
    package_versions = None
    total_training_seconds = 0.0
    for run_id in expected_run_ids:
        run_dir = args.full_dir / run_id
        required_paths = {
            "completed": run_dir / "COMPLETED.json",
            "metrics": run_dir / "metrics.json",
            "checkpoint": run_dir / "state.pt",
            "predictions": run_dir / "eval_predictions.jsonl",
            "training_log": run_dir / "training_log.jsonl",
        }
        for name, path in required_paths.items():
            if not path.exists():
                fail(f"{run_id} is missing {name}: {path}")

        completed = json.loads(required_paths["completed"].read_text())
        metrics = json.loads(required_paths["metrics"].read_text())
        manifest_row = manifest_by_id[run_id]
        task_name = manifest_row["task"]
        protocol = manifest_row["protocol"]
        model_seed = int(manifest_row["model_seed"])

        if completed["status"] != "complete":
            fail(f"{run_id} status is not complete")
        if not completed["eligible_for_gate"]:
            fail(f"{run_id} is not marked eligible for Gate analysis")
        if completed["mode"] != "full" or completed["run_id"] != run_id:
            fail(f"{run_id} completion identity or mode mismatch")
        if completed["execution"]["completed_steps"] != FULL_STEPS:
            fail(f"{run_id} did not complete {FULL_STEPS} steps")
        if completed["execution"]["starting_step"] not in (0, FULL_STEPS):
            fail(f"{run_id} has unexpected resume starting step")
        if completed["frozen_hashes"] != frozen_hashes:
            fail(f"{run_id} frozen hashes differ")
        if completed["adapter_sha256"] != adapter_hash:
            fail(f"{run_id} adapter hash differs from frozen split")
        if completed["run_manifest_row"] != manifest_row:
            fail(f"{run_id} embedded run manifest row differs")
        if completed["metrics"] != metrics:
            fail(f"{run_id} metrics file differs from completion metadata")
        if metrics["eval_episode_count"] != EVAL_FUNCTION_DRAWS:
            fail(f"{run_id} eval episode count differs")
        if protocol == "strict_function_disjoint" and metrics[
            "actual_exposed_eval_episode_count"
        ] != 0:
            fail(f"{run_id} strict run has nonzero actual exposure")
        if protocol == "original" and metrics[
            "actual_exposed_eval_episode_count"
        ] != int(manifest_row["eval_draw_exposure_count"]):
            fail(f"{run_id} original exposure count differs from manifest")

        if package_versions is None:
            package_versions = completed["package_versions"]
        elif completed["package_versions"] != package_versions:
            fail(f"{run_id} package versions differ across runs")

        artifact_info = completed["artifacts"]
        if sha256_file(required_paths["checkpoint"]) != artifact_info[
            "checkpoint_sha256"
        ]:
            fail(f"{run_id} checkpoint checksum mismatch")
        if sha256_file(required_paths["predictions"]) != artifact_info[
            "predictions_sha256"
        ]:
            fail(f"{run_id} predictions checksum mismatch")

        checkpoint = torch.load(
            required_paths["checkpoint"], map_location="cpu", weights_only=False
        )
        if checkpoint["run_id"] != run_id or checkpoint["mode"] != "full":
            fail(f"{run_id} checkpoint identity mismatch")
        if checkpoint["completed_steps"] != FULL_STEPS:
            fail(f"{run_id} checkpoint step mismatch")
        if checkpoint["frozen_hashes"] != frozen_hashes:
            fail(f"{run_id} checkpoint frozen hashes differ")
        if not all_tensors_finite(checkpoint["model_state_dict"]):
            fail(f"{run_id} model state contains nonfinite tensors")
        if not all_tensors_finite(checkpoint["optimizer_state_dict"]):
            fail(f"{run_id} optimizer state contains nonfinite tensors")

        predictions = load_jsonl(required_paths["predictions"])
        if len(predictions) != EVAL_FUNCTION_DRAWS:
            fail(f"{run_id} prediction row count differs")
        expected_eval_rows = eval_rows_by_task[task_name]
        for index, (prediction, expected_eval) in enumerate(
            zip(predictions, expected_eval_rows)
        ):
            if prediction["eval_index"] != index:
                fail(f"{run_id} eval index differs at row {index}")
            if prediction["semantic_id"] != expected_eval["semantic_id"]:
                fail(f"{run_id} semantic ID differs at row {index}")
            if prediction["task"] != task_name:
                fail(f"{run_id} task differs at row {index}")
            if prediction["run_protocol"] != protocol:
                fail(f"{run_id} protocol differs at row {index}")
            if len(prediction["labels"]) != N_POINTS:
                fail(f"{run_id} label length differs at row {index}")
            if len(prediction["logits"]) != N_POINTS:
                fail(f"{run_id} logit length differs at row {index}")
            if not all(math.isfinite(value) for value in prediction["logits"]):
                fail(f"{run_id} has nonfinite logits at row {index}")
            for metric_name in (
                "tail_accuracy",
                "final_position_accuracy",
                "all_positions_1_through_39_accuracy",
                "tail_binary_cross_entropy",
            ):
                if not math.isfinite(prediction[metric_name]):
                    fail(f"{run_id} has nonfinite {metric_name} at row {index}")
            if protocol == "strict_function_disjoint" and prediction[
                "seen_in_actual_training_stream"
            ]:
                fail(f"{run_id} strict row {index} is marked actually seen")

        log_rows = load_jsonl(required_paths["training_log"])
        logged_steps = [row["completed_steps"] for row in log_rows]
        if logged_steps != sorted(set(logged_steps)):
            fail(f"{run_id} training log steps are not unique and sorted")
        if logged_steps[0] != 1 or logged_steps[-1] != FULL_STEPS:
            fail(f"{run_id} training log boundaries differ")
        if not all(
            math.isfinite(row["loss"])
            and math.isfinite(row["steps_per_second_since_start"])
            for row in log_rows
        ):
            fail(f"{run_id} training log contains nonfinite values")

        pair_predictions[(task_name, model_seed, protocol)] = predictions
        training_seconds = completed["execution"][
            "training_seconds_this_invocation"
        ]
        total_training_seconds += training_seconds
        run_reports[run_id] = {
            "checkpoint_sha256": artifact_info["checkpoint_sha256"],
            "predictions_sha256": artifact_info["predictions_sha256"],
            "training_seconds": training_seconds,
            "checkpoint_and_outputs_valid": True,
        }

    for task_name in eval_rows_by_task:
        for model_seed in (0, 1, 2):
            original = pair_predictions[(task_name, model_seed, "original")]
            strict = pair_predictions[
                (task_name, model_seed, "strict_function_disjoint")
            ]
            for original_row, strict_row in zip(original, strict):
                for key in (
                    "eval_index",
                    "semantic_id",
                    "labels",
                    "seen_in_original_training_stream",
                ):
                    if original_row[key] != strict_row[key]:
                        fail(
                            f"{task_name} seed {model_seed} paired field {key} differs"
                        )

    report = {
        "status": "passed",
        "schema_version": SCHEMA_VERSION,
        "full_run_count": len(expected_run_ids),
        "eligible_run_count": len(expected_run_ids),
        "effect_metrics_compared": False,
        "matrix_status_sha256": sha256_file(matrix_status_path),
        "package_versions": package_versions,
        "checks": {
            "complete_frozen_matrix": "passed",
            "completion_and_artifact_hashes": "passed",
            "checkpoint_model_and_optimizer_finite": "passed",
            "prediction_structure_and_finite_values": "passed",
            "strict_actual_exposure_zero": "passed",
            "paired_eval_semantic_ids_labels_and_exposure_flags": "passed",
            "training_logs_complete_and_finite": "passed",
        },
        "total_training_seconds": total_training_seconds,
        "runs": run_reports,
    }
    output_path = args.full_dir / "full_validation.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
