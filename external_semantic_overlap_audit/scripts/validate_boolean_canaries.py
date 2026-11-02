#!/usr/bin/env python3
"""Validate all six frozen seed-0 Boolean engineering canaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from freeze_boolean_behavioral_protocol import (
    EVAL_FUNCTION_DRAWS,
    N_POINTS,
    SCHEMA_VERSION,
    TASKS,
    TRAIN_BATCH_SIZE,
    decode_parameter,
    sample_ids,
    semantic_id,
    sha256_file,
)


CANARY_STEPS = 200
CANARY_EVAL_EPISODES = 128


def fail(message: str) -> None:
    raise SystemExit(f"Canary validation failed: {message}")


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def expected_run_ids() -> list[str]:
    result = []
    for task_name in TASKS:
        result.extend(
            (
                f"{task_name}__original__seed0",
                f"{task_name}__strict__seed0",
            )
        )
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--canary-dir",
        type=Path,
        default=root / "runs" / SCHEMA_VERSION / "canary",
    )
    args = parser.parse_args()

    protocol_dir = root / "protocols" / SCHEMA_VERSION
    frozen_hashes = {}
    for line in (protocol_dir / "SHA256SUMS").read_text().splitlines():
        digest, filename = line.split("  ", maxsplit=1)
        frozen_hashes[filename] = digest

    with (protocol_dir / "run_manifest.csv").open(newline="") as handle:
        manifest_rows = {
            row["run_id"]: row for row in csv.DictReader(handle)
        }
    eval_rows_by_task = {task_name: [] for task_name in TASKS}
    for row in load_jsonl(protocol_dir / "eval_functions.jsonl"):
        eval_rows_by_task[row["task"]].append(row)

    expected = expected_run_ids()
    observed_directories = sorted(
        path.name for path in args.canary_dir.iterdir() if path.is_dir()
    )
    if observed_directories != sorted(expected):
        fail(
            f"canary directories differ from expected seed-0 matrix: "
            f"{observed_directories}"
        )

    run_reports = {}
    total_training_seconds = 0.0
    total_training_steps = 0
    pair_predictions = {}
    for run_id in expected:
        run_dir = args.canary_dir / run_id
        completed_path = run_dir / "COMPLETED.json"
        metrics_path = run_dir / "metrics.json"
        checkpoint_path = run_dir / "state.pt"
        predictions_path = run_dir / "eval_predictions.jsonl"
        log_path = run_dir / "training_log.jsonl"
        for path in (
            completed_path,
            metrics_path,
            checkpoint_path,
            predictions_path,
            log_path,
        ):
            if not path.exists():
                fail(f"missing {path}")

        completed = json.loads(completed_path.read_text())
        metrics = json.loads(metrics_path.read_text())
        if completed["status"] != "complete":
            fail(f"{run_id} status is not complete")
        if completed["eligible_for_gate"]:
            fail(f"{run_id} is incorrectly eligible for the behavioral Gate")
        if completed["mode"] != "canary":
            fail(f"{run_id} mode is not canary")
        if completed["run_id"] != run_id:
            fail(f"{run_id} completion identity mismatch")
        if completed["frozen_hashes"] != frozen_hashes:
            fail(f"{run_id} frozen hashes differ")
        if completed["metrics"] != metrics:
            fail(f"{run_id} metrics.json differs from COMPLETED.json")
        if completed["execution"]["completed_steps"] != CANARY_STEPS:
            fail(f"{run_id} did not complete {CANARY_STEPS} steps")
        if metrics["eval_episode_count"] != CANARY_EVAL_EPISODES:
            fail(f"{run_id} eval episode count mismatch")

        artifact_info = completed["artifacts"]
        if sha256_file(checkpoint_path) != artifact_info["checkpoint_sha256"]:
            fail(f"{run_id} checkpoint checksum mismatch")
        if sha256_file(predictions_path) != artifact_info["predictions_sha256"]:
            fail(f"{run_id} predictions checksum mismatch")

        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if checkpoint["run_id"] != run_id or checkpoint["mode"] != "canary":
            fail(f"{run_id} checkpoint identity mismatch")
        if checkpoint["completed_steps"] != CANARY_STEPS:
            fail(f"{run_id} checkpoint step mismatch")
        if checkpoint["frozen_hashes"] != frozen_hashes:
            fail(f"{run_id} checkpoint frozen hashes differ")
        for name, value in checkpoint["model_state_dict"].items():
            if not torch.isfinite(value).all():
                fail(f"{run_id} has nonfinite model tensor {name}")

        predictions = load_jsonl(predictions_path)
        if len(predictions) != CANARY_EVAL_EPISODES:
            fail(f"{run_id} prediction row count mismatch")
        task_name = manifest_rows[run_id]["task"]
        protocol = manifest_rows[run_id]["protocol"]
        expected_eval_rows = eval_rows_by_task[task_name][
            :CANARY_EVAL_EPISODES
        ]
        for index, (prediction, expected_eval) in enumerate(
            zip(predictions, expected_eval_rows)
        ):
            if prediction["eval_index"] != index:
                fail(f"{run_id} eval indices are not contiguous")
            if prediction["semantic_id"] != expected_eval["semantic_id"]:
                fail(f"{run_id} semantic ID differs at eval row {index}")
            if len(prediction["labels"]) != N_POINTS:
                fail(f"{run_id} label length differs at eval row {index}")
            if len(prediction["logits"]) != N_POINTS:
                fail(f"{run_id} logit length differs at eval row {index}")
            if not all(math.isfinite(value) for value in prediction["logits"]):
                fail(f"{run_id} has nonfinite logits at eval row {index}")
            if protocol == "strict_function_disjoint" and prediction[
                "seen_in_actual_training_stream"
            ]:
                fail(f"{run_id} marks a strict eval episode as actually seen")

        log_rows = load_jsonl(log_path)
        logged_steps = [row["completed_steps"] for row in log_rows]
        if logged_steps != sorted(set(logged_steps)):
            fail(f"{run_id} training log steps are not unique and sorted")
        if logged_steps[-1] != CANARY_STEPS:
            fail(f"{run_id} training log does not end at {CANARY_STEPS}")
        if not all(math.isfinite(row["loss"]) for row in log_rows):
            fail(f"{run_id} training log contains a nonfinite loss")

        task_eval_ids = np.asarray(
            [
                row["encoded_semantic_id"]
                for row in eval_rows_by_task[task_name]
            ],
            dtype=np.uint64,
        )
        original_stream = sample_ids(
            task_name,
            int(manifest_rows[run_id]["function_draw_count"]),
            int(manifest_rows[run_id]["train_function_seed"]),
        )
        prefix = original_stream[: CANARY_STEPS * TRAIN_BATCH_SIZE]
        prefix_collision_count = int(
            np.isin(prefix, np.unique(task_eval_ids)).sum()
        )

        pair_predictions[(task_name, protocol)] = predictions
        training_seconds = completed["execution"][
            "training_seconds_this_invocation"
        ]
        total_training_seconds += training_seconds
        total_training_steps += CANARY_STEPS
        run_reports[run_id] = {
            "checkpoint_sha256": artifact_info["checkpoint_sha256"],
            "predictions_sha256": artifact_info["predictions_sha256"],
            "training_seconds": training_seconds,
            "steps_per_second": CANARY_STEPS / training_seconds,
            "prefix_original_eval_collision_count": prefix_collision_count,
            "actual_exposed_eval_episode_count": metrics[
                "actual_exposed_eval_episode_count"
            ],
            "checkpoint_and_outputs_valid": True,
        }

    for task_name in TASKS:
        original = pair_predictions[(task_name, "original")]
        strict = pair_predictions[(task_name, "strict_function_disjoint")]
        for original_row, strict_row in zip(original, strict):
            for key in ("eval_index", "semantic_id", "labels"):
                if original_row[key] != strict_row[key]:
                    fail(f"{task_name} paired eval field {key} differs")

    aggregate_steps_per_second = total_training_steps / total_training_seconds
    estimated_core_hours = (
        len(TASKS)
        * 2
        * 3
        * 20_000
        / aggregate_steps_per_second
        / 3600
    )
    report = {
        "status": "passed",
        "schema_version": SCHEMA_VERSION,
        "canary_run_count": len(expected),
        "all_runs_eligible_for_gate": False,
        "effect_metrics_compared": False,
        "checks": {
            "complete_matrix": "passed",
            "frozen_hashes": "passed",
            "checkpoint_hashes_and_finite_tensors": "passed",
            "prediction_hashes_shapes_and_finite_logits": "passed",
            "strict_actual_exposure_zero": "passed",
            "paired_eval_semantic_ids_and_labels": "passed",
            "training_logs_complete_and_finite": "passed",
        },
        "runtime": {
            "total_training_steps": total_training_steps,
            "total_training_seconds": total_training_seconds,
            "aggregate_steps_per_second": aggregate_steps_per_second,
            "estimated_18_run_core_training_hours_at_canary_throughput": (
                estimated_core_hours
            ),
            "estimate_scope": (
                "Training-loop extrapolation only; excludes process startup, "
                "evaluation, checkpoint I/O, and thermal throttling."
            ),
        },
        "runs": run_reports,
    }
    output_path = args.canary_dir / "canary_validation.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
