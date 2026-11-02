"""Execute one frozen CLUTRR V3 training run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .training_common import (
    ARMS,
    DATA_DIR,
    EPOCHS,
    EVAL_BATCH_SIZE,
    EVAL_ROWS,
    EXECUTION_DIR,
    FIXED_LENGTH,
    MODEL_SEEDS,
    PROJECT_ROOT,
    RUN_ROOT,
    TORCH_INTEROP_THREADS,
    TORCH_INTRAOP_THREADS,
    TRAIN_BATCH_SIZE,
    TRAIN_ROWS,
    UPDATES_PER_EPOCH,
    UPDATES_PER_RUN,
    array_sha256,
    artifact_hashes,
    canonical_json_bytes,
    classes,
    epoch_order,
    finite_tree,
    initialize_model,
    load_npz,
    read_csv,
    read_json,
    run_id,
    sha256_bytes,
    sha256_file,
    state_dict_sha256,
    torch_rng_sha256,
    write_csv,
    write_deterministic_npz,
    write_json,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_json(temporary, value)
    temporary.replace(path)


def _run_row(seed: int, arm: str) -> dict[str, str]:
    matches = [
        row
        for row in read_csv(EXECUTION_DIR / "run_manifest.csv")
        if int(row["model_seed"]) == seed and row["arm"] == arm
    ]
    if len(matches) != 1:
        raise RuntimeError("Frozen run is not unique.")
    return matches[0]


def _epoch_hashes(seed: int) -> dict[int, str]:
    manifest = read_json(EXECUTION_DIR / "epoch_order_manifest.json")
    return {
        int(row["epoch"]): row["order_sha256"]
        for row in manifest["records"]
        if int(row["model_seed"]) == seed
    }


def _batch(
    cache: dict[str, np.ndarray],
    cache_indices: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = torch.from_numpy(
        cache["token_ids"][cache_indices].astype(np.int64, copy=False)
    )
    attention_mask = torch.from_numpy(
        cache["attention_mask"][cache_indices]
    )
    labels = torch.from_numpy(
        cache["labels"][cache_indices].astype(np.int64, copy=False)
    )
    return input_ids, attention_mask, labels


def _ece(probabilities: np.ndarray, labels: np.ndarray) -> float:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == labels
    result = 0.0
    for index in range(15):
        lower = index / 15
        upper = (index + 1) / 15
        if index == 14:
            selected = (confidence >= lower) & (confidence <= upper)
        else:
            selected = (confidence >= lower) & (confidence < upper)
        if not selected.any():
            continue
        result += float(selected.mean()) * abs(
            float(correct[selected].mean())
            - float(confidence[selected].mean())
        )
    return result


def _evaluate(
    model: torch.nn.Module,
    cache: dict[str, np.ndarray],
    evaluation: dict[str, np.ndarray],
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    model.eval()
    logits = np.zeros((EVAL_ROWS, 23), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, EVAL_ROWS, EVAL_BATCH_SIZE):
            stop = min(start + EVAL_BATCH_SIZE, EVAL_ROWS)
            indices = evaluation["cache_indices"][start:stop]
            input_ids, attention_mask, _ = _batch(cache, indices)
            output = model(input_ids, attention_mask)
            if not torch.isfinite(output).all():
                raise FloatingPointError("Non-finite evaluation logits.")
            logits[start:stop] = output.cpu().numpy()
    labels = evaluation["labels"].astype(np.int64)
    predictions = logits.argmax(axis=1).astype(np.int16)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    cross_entropy = float(
        (
            -shifted[np.arange(EVAL_ROWS), labels]
            + np.log(exp_logits.sum(axis=1))
        ).mean()
    )
    one_hot = np.eye(23, dtype=np.float32)[labels]
    metrics = {
        "accuracy": float((predictions == labels).mean()),
        "brier_score": float(
            np.square(probabilities - one_hot).sum(axis=1).mean()
        ),
        "cross_entropy": cross_entropy,
        "ece_15": _ece(probabilities, labels),
    }
    arrays = {
        "labels": labels.astype(np.int16),
        "logits": logits,
        "pair_index": evaluation["pair_index"],
        "pair_member": evaluation["pair_member"],
        "predictions": predictions,
        "row_ids": evaluation["row_ids"],
        "semantic_ids": evaluation["semantic_ids"],
        "source_group_ids": evaluation["source_group_ids"],
    }
    return metrics, arrays


def run(seed: int, arm: str) -> dict[str, Any]:
    if seed not in MODEL_SEEDS or arm not in ARMS:
        raise ValueError("Unknown seed or arm.")
    torch.set_num_threads(TORCH_INTRAOP_THREADS)
    torch.set_num_interop_threads(TORCH_INTEROP_THREADS)
    torch.use_deterministic_algorithms(True)

    frozen = _run_row(seed, arm)
    identifier = run_id(seed, arm)
    run_dir = RUN_ROOT / identifier
    if run_dir.exists():
        raise RuntimeError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    started_at = _now()

    cache = load_npz(EXECUTION_DIR / "tensor_cache.npz")
    train_indices_archive = load_npz(
        EXECUTION_DIR / "train_cache_indices.npz"
    )
    train_indices = train_indices_archive[f"seed{seed}_arm{arm}"]
    evaluation = load_npz(EXECUTION_DIR / "evaluation_manifest.npz")
    if (
        train_indices.shape != (TRAIN_ROWS,)
        or evaluation["cache_indices"].shape != (EVAL_ROWS,)
        or cache["token_ids"].shape[1] != FIXED_LENGTH
    ):
        raise RuntimeError("Frozen tensor shapes changed.")

    model, initial_model_hash, start_rng_hash = initialize_model(seed)
    if (
        initial_model_hash != frozen["expected_initial_model_sha256"]
        or start_rng_hash
        != frozen["expected_training_start_torch_rng_sha256"]
    ):
        raise RuntimeError("Frozen initialization hash mismatch.")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0003,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )
    expected_epoch_hashes = _epoch_hashes(seed)
    combined_hash = sha256_bytes(
        canonical_json_bytes(
            [expected_epoch_hashes[epoch] for epoch in range(1, EPOCHS + 1)]
        )
    )
    if combined_hash != frozen["combined_epoch_order_sha256"]:
        raise RuntimeError("Combined epoch-order hash mismatch.")

    config = {
        "arm": arm,
        "batch_size": TRAIN_BATCH_SIZE,
        "device": "cpu",
        "epochs": EPOCHS,
        "evaluation_batch_size": EVAL_BATCH_SIZE,
        "evaluation_manifest_sha256": sha256_file(
            EXECUTION_DIR / "evaluation_manifest.npz"
        ),
        "execution_manifest_sha256": sha256_file(
            EXECUTION_DIR / "execution_manifest.json"
        ),
        "expected_initial_model_sha256": initial_model_hash,
        "expected_training_start_torch_rng_sha256": start_rng_hash,
        "fixed_padding_length": FIXED_LENGTH,
        "model_seed": seed,
        "optimizer": "AdamW",
        "precision": "float32",
        "run_id": identifier,
        "threading": {
            "interop": TORCH_INTEROP_THREADS,
            "intraop": TORCH_INTRAOP_THREADS,
        },
        "train_manifest": frozen["train_manifest"],
        "train_manifest_sha256": frozen["train_manifest_sha256"],
        "updates_per_run": UPDATES_PER_RUN,
    }
    write_json(run_dir / "config.json", config)
    history: list[dict[str, Any]] = []
    updates = 0
    wall_start = time.perf_counter()
    model.train()
    for epoch in range(1, EPOCHS + 1):
        order = epoch_order(seed, epoch)
        order_hash = array_sha256(order)
        if order_hash != expected_epoch_hashes[epoch]:
            raise RuntimeError(f"Epoch order changed at epoch {epoch}.")
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_rows = 0
        epoch_start = time.perf_counter()
        model.train()
        for start in range(0, TRAIN_ROWS, TRAIN_BATCH_SIZE):
            slots = order[start : start + TRAIN_BATCH_SIZE]
            cache_indices = train_indices[slots]
            input_ids, attention_mask, labels = _batch(
                cache,
                cache_indices,
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids, attention_mask)
            loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss.")
            loss.backward()
            for parameter in model.parameters():
                if (
                    parameter.grad is not None
                    and not torch.isfinite(parameter.grad).all()
                ):
                    raise FloatingPointError("Non-finite gradient.")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_rows = len(slots)
            epoch_loss += float(loss.detach()) * batch_rows
            epoch_correct += int(
                (logits.detach().argmax(dim=1) == labels).sum()
            )
            epoch_rows += batch_rows
            updates += 1
        if updates != epoch * UPDATES_PER_EPOCH:
            raise RuntimeError("Optimizer update count changed.")
        if not finite_tree(model.state_dict()):
            raise FloatingPointError("Non-finite model state.")
        history.append(
            {
                "epoch": epoch,
                "epoch_order_sha256": order_hash,
                "optimizer_updates": updates,
                "seconds": time.perf_counter() - epoch_start,
                "train_accuracy": epoch_correct / epoch_rows,
                "train_loss": epoch_loss / epoch_rows,
            }
        )
        write_csv(run_dir / "history.csv", history)
        _atomic_json(
            run_dir / "progress.json",
            {
                "arm": arm,
                "epoch": epoch,
                "model_seed": seed,
                "optimizer_updates": updates,
                "run_id": identifier,
                "status": "training",
                "updated_at_utc": _now(),
            },
        )

    final_rng_hash = torch_rng_sha256()
    final_model_hash = state_dict_sha256(model.state_dict())
    if updates != UPDATES_PER_RUN:
        raise RuntimeError("Final optimizer update count changed.")
    if not finite_tree(optimizer.state_dict()):
        raise FloatingPointError("Non-finite optimizer state.")
    metrics, predictions = _evaluate(model, cache, evaluation)
    if not all(math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("Non-finite evaluation metric.")

    checkpoint_path = run_dir / "final_checkpoint.pt"
    torch.save(
        {
            "arm": arm,
            "epoch": EPOCHS,
            "final_model_sha256": final_model_hash,
            "final_torch_rng_sha256": final_rng_hash,
            "initial_model_sha256": initial_model_hash,
            "model_seed": seed,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "optimizer_updates": updates,
            "training_start_torch_rng_sha256": start_rng_hash,
        },
        checkpoint_path,
    )
    predictions_path = run_dir / "eval_predictions.npz"
    write_deterministic_npz(predictions_path, predictions)
    metrics_payload = {
        **metrics,
        "arm": arm,
        "epochs_completed": EPOCHS,
        "final_model_sha256": final_model_hash,
        "final_torch_rng_sha256": final_rng_hash,
        "initial_model_sha256": initial_model_hash,
        "model_seed": seed,
        "optimizer_updates": updates,
        "prediction_rows": EVAL_ROWS,
        "run_id": identifier,
        "seconds": time.perf_counter() - wall_start,
        "training_start_torch_rng_sha256": start_rng_hash,
    }
    write_json(run_dir / "metrics.json", metrics_payload)
    _atomic_json(
        run_dir / "progress.json",
        {
            "arm": arm,
            "epoch": EPOCHS,
            "model_seed": seed,
            "optimizer_updates": updates,
            "run_id": identifier,
            "status": "completed",
            "updated_at_utc": _now(),
        },
    )
    artifact_names = (
        "config.json",
        "eval_predictions.npz",
        "final_checkpoint.pt",
        "history.csv",
        "metrics.json",
        "progress.json",
    )
    completed = {
        "artifacts_sha256": artifact_hashes(run_dir, artifact_names),
        "completed_at_utc": _now(),
        "run_id": identifier,
        "status": "completed",
    }
    write_json(run_dir / "COMPLETED.json", completed)
    return metrics_payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    args = parser.parse_args(argv)
    metrics = run(args.seed, args.arm)
    print(
        json.dumps(
            {
                "run_id": metrics["run_id"],
                "status": "completed",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
