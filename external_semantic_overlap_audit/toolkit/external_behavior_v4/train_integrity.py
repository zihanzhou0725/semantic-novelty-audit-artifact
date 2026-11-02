"""Post-training integrity validation for 32 directional V4 runs."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch

from toolkit.external_behavior.common import iter_jsonl

from .training_common import (
    ARMS,
    BLOCKS,
    DATA_DIR,
    EPOCHS,
    EVAL_ROWS,
    EXECUTION_DIR,
    MATERIALIZATION_DIR,
    MODEL_SEEDS,
    PROJECT_ROOT,
    RUN_COUNT,
    RUN_ROOT,
    UPDATES_PER_RUN,
    array_sha256,
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


def _metrics(
    logits: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
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
                + np.log(exponentials.sum(axis=1))
            ).mean()
        ),
        "ece_15": _ece(probabilities, labels),
    }


def _checklist(checks: list[dict[str, Any]]) -> str:
    status = {item["name"]: item["pass"] for item in checks}
    entries = (
        ("run_count", "Complete all 32 frozen runs."),
        ("progress", "Reach epoch 30 and update 5,640 in every run."),
        ("artifact_hashes", "Validate every run artifact hash."),
        ("checkpoint_finite", "Validate finite model and optimizer states."),
        ("epoch_orders", "Validate all frozen epoch orders."),
        ("initialization", "Validate frozen model initialization."),
        ("paired_rng", "Validate four-way paired Torch RNG trajectories."),
        ("predictions", "Validate ordered 8,000-row predictions."),
        ("metrics", "Independently recompute every metric."),
        ("training_rows", "Validate frozen training-row manifests."),
        ("aggregate_blindness", "Confirm no V4 effect aggregate exists."),
    )
    lines = ["# Directional V4 Execution Integrity", ""]
    for name, label in entries:
        lines.append(f"- [{'x' if status.get(name) else ' '}] {label}")
    lines.append("")
    return "\n".join(lines)


def validate_execution() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: Any = None) -> None:
        checks.append({"detail": detail, "name": name, "pass": bool(condition)})

    rows = read_csv(EXECUTION_DIR / "run_manifest.csv")
    check(
        "run_count",
        len(rows) == RUN_COUNT
        and all((RUN_ROOT / row["run_id"]).is_dir() for row in rows),
        len(rows),
    )
    evaluations = {
        block: load_npz(EXECUTION_DIR / f"evaluation_block{block}.npz")
        for block in BLOCKS
    }
    orders = read_json(EXECUTION_DIR / "epoch_order_manifest.json")
    expected_order = {
        (int(row["model_seed"]), int(row["epoch"])): row["order_sha256"]
        for row in orders["records"]
    }
    initialization = read_json(
        EXECUTION_DIR / "initialization_manifest.json"
    )
    init_by_seed = {
        int(row["model_seed"]): row for row in initialization["records"]
    }
    failures: dict[str, list[str]] = defaultdict(list)
    rng_by_seed: dict[int, dict[str, tuple[str, str]]] = defaultdict(dict)
    run_artifacts: list[dict[str, Any]] = []
    materialized_rows = {
        row["run_id"]: row
        for row in read_csv(MATERIALIZATION_DIR / "run_manifest.csv")
    }
    for row in rows:
        block = int(row["block_index"])
        seed = int(row["model_seed"])
        arm = row["arm"]
        identifier = row["run_id"]
        run_dir = RUN_ROOT / identifier
        completed_path = run_dir / "COMPLETED.json"
        if not completed_path.is_file():
            failures["artifact_hashes"].append(
                f"missing_completed:{identifier}"
            )
            continue
        completed = read_json(completed_path)
        for filename, expected in completed["artifacts_sha256"].items():
            path = run_dir / filename
            if not path.is_file() or sha256_file(path) != expected:
                failures["artifact_hashes"].append(
                    f"{identifier}:{filename}"
                )
        progress = read_json(run_dir / "progress.json")
        if (
            progress.get("status") != "completed"
            or progress.get("epoch") != EPOCHS
            or progress.get("optimizer_updates") != UPDATES_PER_RUN
            or progress.get("block_index") != block
        ):
            failures["progress"].append(identifier)
        config = read_json(run_dir / "config.json")
        if (
            config["train_manifest_sha256"]
            != row["train_manifest_sha256"]
            or config["expected_initial_model_sha256"]
            != row["expected_initial_model_sha256"]
            or config["block_index"] != block
            or config["model_seed"] != seed
            or config["arm"] != arm
        ):
            failures["initialization"].append(f"config:{identifier}")
        history = read_csv(run_dir / "history.csv")
        if len(history) != EPOCHS:
            failures["epoch_orders"].append(f"history:{identifier}")
        else:
            for epoch, history_row in enumerate(history, start=1):
                expected = expected_order[(seed, epoch)]
                if (
                    history_row["epoch_order_sha256"] != expected
                    or array_sha256(epoch_order(seed, epoch)) != expected
                    or int(history_row["optimizer_updates"]) != epoch * 188
                    or not math.isfinite(float(history_row["train_loss"]))
                    or not math.isfinite(
                        float(history_row["train_accuracy"])
                    )
                ):
                    failures["epoch_orders"].append(
                        f"{identifier}:{epoch}"
                    )
        checkpoint = torch.load(
            run_dir / "final_checkpoint.pt",
            map_location="cpu",
            weights_only=False,
        )
        if (
            checkpoint["block_index"] != block
            or checkpoint["epoch"] != EPOCHS
            or checkpoint["optimizer_updates"] != UPDATES_PER_RUN
            or not finite_tree(checkpoint["model_state"])
            or not finite_tree(checkpoint["optimizer_state"])
        ):
            failures["checkpoint_finite"].append(identifier)
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
            failures["initialization"].append(identifier)
        rng_by_seed[seed][f"block{block}_arm{arm}"] = (
            checkpoint["training_start_torch_rng_sha256"],
            checkpoint["final_torch_rng_sha256"],
        )
        predictions = load_npz(run_dir / "eval_predictions.npz")
        expected_eval = evaluations[block]
        required = set(expected_eval) - {"cache_indices"}
        required.update({"logits", "predictions"})
        metadata = set(expected_eval) - {"cache_indices"}
        if (
            set(predictions) != required
            or predictions["logits"].shape != (EVAL_ROWS, 23)
            or not np.isfinite(predictions["logits"]).all()
            or any(
                not np.array_equal(
                    predictions[name],
                    expected_eval[name],
                )
                for name in metadata
            )
            or not np.array_equal(
                predictions["predictions"],
                predictions["logits"].argmax(axis=1),
            )
        ):
            failures["predictions"].append(identifier)
        metrics = read_json(run_dir / "metrics.json")
        recomputed = _metrics(
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
            failures["metrics"].append(identifier)
        materialized = materialized_rows[identifier]
        train_path = DATA_DIR / materialized["train_manifest"]
        if (
            len(train_path.read_text(encoding="ascii").splitlines())
            != 24_000
            or sha256_file(train_path) != row["train_manifest_sha256"]
        ):
            failures["training_rows"].append(identifier)
        run_artifacts.append(
            {
                "artifacts_sha256": completed["artifacts_sha256"],
                "completed_sha256": sha256_file(completed_path),
                "run_id": identifier,
            }
        )
    expected_rng_keys = {
        f"block{block}_arm{arm}"
        for block in BLOCKS
        for arm in ARMS
    }
    for seed in MODEL_SEEDS:
        if set(rng_by_seed[seed]) != expected_rng_keys or len(
            set(rng_by_seed[seed].values())
        ) != 1:
            failures["paired_rng"].append(str(seed))
    for name in (
        "progress",
        "artifact_hashes",
        "checkpoint_finite",
        "epoch_orders",
        "initialization",
        "paired_rng",
        "predictions",
        "metrics",
        "training_rows",
    ):
        check(name, not failures[name], failures[name][:100])
    effect_root = (
        PROJECT_ROOT
        / "analysis"
        / "external-behavior-v4"
        / "effect-analysis-v1"
    )
    effect_artifacts = (
        list(effect_root.rglob("*")) if effect_root.exists() else []
    )
    check(
        "aggregate_blindness",
        not effect_artifacts
        and read_json(RUN_ROOT / "execution_state.json")[
            "primary_effect_computed"
        ]
        is False,
        [str(path) for path in effect_artifacts],
    )
    passed = all(item["pass"] for item in checks)
    write_json(
        EXECUTION_DIR / "run_artifact_manifest.json",
        {
            "runs": sorted(run_artifacts, key=lambda item: item["run_id"]),
            "schema_version": "external-behavior-v4-run-artifacts-v1",
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
        "schema_version": (
            "external-behavior-v4-execution-validation-v1"
        ),
        "status": "pass" if passed else "no_go",
        "validated_at_utc": _now(),
    }
    checklist = EXECUTION_DIR / "execution_checklist.md"
    checklist.write_text(_checklist(checks), encoding="ascii")
    report_path = EXECUTION_DIR / "execution_validation_report.json"
    write_json(report_path, report)
    frozen_paths = [
        EXECUTION_DIR / "EXECUTION_FREEZE_SHA256SUMS",
        EXECUTION_DIR / "run_artifact_manifest.json",
        checklist,
        report_path,
        RUN_ROOT / "execution_state.json",
        RUN_ROOT / "matrix_status.json",
    ]
    (EXECUTION_DIR / "EXECUTION_SHA256SUMS").write_text(
        "".join(
            f"{sha256_file(path)}  "
            f"{path.relative_to(PROJECT_ROOT).as_posix()}\n"
            for path in frozen_paths
        ),
        encoding="ascii",
    )
    return report
