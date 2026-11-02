#!/usr/bin/env python3
"""Run one frozen quality-enhancement P1 model/seed/arm training job."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from quality_p1_common import (
    ARMS,
    BATCH_SIZE,
    EPOCHS,
    EXTERNAL_ROOT,
    MODEL_SEEDS,
    MODELS,
    P1_DIR,
    P1_SCHEMA_VERSION,
    ROOT,
    SUPPORT_SIZE,
    TRAIN_EPISODES,
    load_npz,
    model_args,
    set_deterministic_state,
    sha256_file,
    validate_checksum_file,
    write_deterministic_npz,
    write_json,
)


KFWA_ROOT = (
    ROOT
    / "semantic_novelty_audit_supplement"
    / "src"
    / "kfwa_experiments"
)
if str(KFWA_ROOT) not in sys.path:
    sys.path.insert(0, str(KFWA_ROOT))

from kfwa.metrics import classification_metrics  # noqa: E402
from kfwa.models import build_model  # noqa: E402
from kfwa.training import unpack_model_output  # noqa: E402
from kfwa.utils import count_parameters, get_device  # noqa: E402


RUN_ROOT = EXTERNAL_ROOT / "runs" / "quality-enhancement-v1" / "p1"


def atomically_write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_json(temporary, value)
    temporary.replace(path)


def make_batch(
    arrays: dict[str, np.ndarray],
    arm: str,
    indices: np.ndarray,
    device: torch.device,
    *,
    evaluation: bool,
    exposed_ids: set[int] | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    count = len(indices)
    if evaluation:
        family_np = arrays["family_id"][indices].astype(np.int64)
        identity_np = arrays["identity_id"][indices].astype(np.int64)
        function_np = arrays["function_id"][indices].astype(np.int64)
        alias_np = arrays["function_alias_count"][indices].astype(np.int64)
        labels_np = arrays["support_labels"][indices].astype(np.int64)
        target_np = arrays["target"][indices].astype(np.int64)
        seen_function_np = np.asarray(
            [int(value) in (exposed_ids or set()) for value in function_np],
            dtype=np.bool_,
        )
        seen_identity_np = np.zeros(count, dtype=np.bool_)
    else:
        family_np = arrays[f"family_id_{arm}"][indices].astype(np.int64)
        identity_np = arrays[f"identity_id_{arm}"][indices].astype(np.int64)
        function_np = arrays[f"function_id_{arm}"][indices].astype(np.int64)
        alias_np = arrays[f"function_alias_count_{arm}"][indices].astype(
            np.int64
        )
        labels_np = arrays[f"support_labels_{arm}"][indices].astype(np.int64)
        target_np = arrays[f"target_{arm}"][indices].astype(np.int64)
        seen_function_np = np.ones(count, dtype=np.bool_)
        seen_identity_np = np.ones(count, dtype=np.bool_)

    family = torch.from_numpy(family_np)
    seen_identity = torch.from_numpy(seen_identity_np)
    batch = {
        "query_digit": torch.from_numpy(
            arrays["query_digit"][indices].astype(np.int64)
        ),
        "support_digits": torch.from_numpy(
            arrays["support_digits"][indices].astype(np.int64)
        ),
        "support_labels": torch.from_numpy(labels_np),
        "support_mask": torch.ones((count, SUPPORT_SIZE), dtype=torch.bool),
        "support_k": torch.full((count,), SUPPORT_SIZE, dtype=torch.long),
        "rule_id": torch.from_numpy(identity_np),
        "identity_id": torch.from_numpy(identity_np.copy()),
        "function_id": torch.from_numpy(function_np),
        "function_alias_count": torch.from_numpy(alias_np),
        "rule_family": family,
        "context_id": family.clone(),
        "seen_identity": seen_identity,
        "seen_function": torch.from_numpy(seen_function_np),
        "seen_context": seen_identity.clone(),
        "query_in_support": torch.zeros(count, dtype=torch.bool),
    }
    return (
        {key: value.to(device) for key, value in batch.items()},
        torch.from_numpy(target_np).to(device),
    )


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    arrays: dict[str, np.ndarray],
    arm: str,
    order: np.ndarray,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_rows = 0
    for start in range(0, TRAIN_EPISODES, BATCH_SIZE):
        indices = order[start : start + BATCH_SIZE].astype(np.int64)
        batch, target = make_batch(
            arrays,
            arm,
            indices,
            device,
            evaluation=False,
        )
        logits, auxiliary = unpack_model_output(model(batch))
        loss = torch.nn.functional.cross_entropy(logits, target)
        if auxiliary is not None:
            loss = loss + auxiliary
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        count = int(target.shape[0])
        total_loss += float(loss.detach().cpu().item()) * count
        total_correct += int(
            logits.detach().argmax(dim=-1).eq(target).sum().cpu().item()
        )
        total_rows += count
    return total_loss / total_rows, total_correct / total_rows


def evaluate(
    model: torch.nn.Module,
    arrays: dict[str, np.ndarray],
    arm: str,
    exposed_ids: set[int],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    model.eval()
    logits_parts: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    contexts: list[torch.Tensor] = []
    seen_functions: list[torch.Tensor] = []
    query_in_support: list[torch.Tensor] = []
    size = len(arrays["target"])
    with torch.no_grad():
        for start in range(0, size, BATCH_SIZE):
            indices = np.arange(
                start,
                min(start + BATCH_SIZE, size),
                dtype=np.int64,
            )
            batch, target = make_batch(
                arrays,
                arm,
                indices,
                device,
                evaluation=True,
                exposed_ids=exposed_ids,
            )
            logits, _ = unpack_model_output(model(batch))
            logits_parts.append(logits.detach().cpu())
            targets.append(target.detach().cpu())
            contexts.append(batch["context_id"].detach().cpu())
            seen_functions.append(batch["seen_function"].detach().cpu())
            query_in_support.append(batch["query_in_support"].detach().cpu())

    logits = torch.cat(logits_parts)
    target = torch.cat(targets)
    seen_function = torch.cat(seen_functions)
    metrics = classification_metrics(
        logits,
        target,
        context_id=torch.cat(contexts),
        seen_context=torch.zeros(size, dtype=torch.bool),
        seen_function=seen_function,
        query_in_support=torch.cat(query_in_support),
    )
    probabilities = logits.softmax(dim=-1)
    prediction = logits.argmax(dim=-1)
    correct = prediction.eq(target)
    per_episode_ce = torch.nn.functional.cross_entropy(
        logits,
        target,
        reduction="none",
    )
    one_hot = torch.nn.functional.one_hot(target, num_classes=20).float()
    per_episode_brier = (probabilities - one_hot).square().sum(dim=-1)

    function_rows: list[dict[str, Any]] = []
    function_ids = arrays["function_id"].astype(np.int64)
    for function_id in sorted(set(int(value) for value in function_ids)):
        mask_np = function_ids == function_id
        mask = torch.from_numpy(mask_np)
        function_rows.append(
            {
                "function_id": function_id,
                "exposed": function_id in exposed_ids,
                "n": int(mask_np.sum()),
                "accuracy": float(correct[mask].float().mean().item()),
                "ce": float(per_episode_ce[mask].mean().item()),
                "brier": float(per_episode_brier[mask].mean().item()),
                "multiplier": int(
                    np.unique(arrays["function_multiplier"][mask_np])[0]
                ),
                "intercept": int(
                    np.unique(arrays["pair_intercept"][mask_np])[0]
                ),
            }
        )
    outputs = {
        "arm": arm,
        "exposed_function_count": len(exposed_ids),
        "per_function": function_rows,
        **metrics.as_dict(prefix="eval_"),
    }
    prediction_arrays = {
        "logits": logits.numpy().astype(np.float32),
        "probabilities": probabilities.numpy().astype(np.float32),
        "prediction": prediction.numpy().astype(np.uint8),
        "target": target.numpy().astype(np.uint8),
        "correct": correct.numpy().astype(np.bool_),
        "per_episode_ce": per_episode_ce.numpy().astype(np.float32),
        "per_episode_brier": per_episode_brier.numpy().astype(np.float32),
        "function_id": function_ids,
        "function_multiplier": arrays["function_multiplier"].copy(),
        "pair_intercept": arrays["pair_intercept"].copy(),
        "episode_within_function": arrays[
            "episode_within_function"
        ].copy(),
        "exposed": np.asarray(
            [int(value) in exposed_ids for value in function_ids],
            dtype=np.bool_,
        ),
    }
    return outputs, prediction_arrays


def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one frozen P1 job.")
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--seed", choices=MODEL_SEEDS, type=int, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--retry", action="store_true")
    args = parser.parse_args()

    gate = json.loads(
        (P1_DIR / "validation_report.json").read_text(encoding="ascii")
    )
    if gate.get("design_gate") != "passed" or not gate.get(
        "training_may_start"
    ):
        raise SystemExit("P1 Design Gate has not passed.")
    validate_checksum_file(P1_DIR, P1_DIR / "SHA256SUMS")

    run_id = f"{args.model}__seed{args.seed}__arm{args.arm}"
    run_dir = RUN_ROOT / run_id
    if (run_dir / "metrics.json").exists():
        print(f"Completed run already exists: {run_dir}")
        return 0
    if run_dir.exists() and any(run_dir.iterdir()) and not args.retry:
        raise SystemExit(
            f"Incomplete run exists; pass --retry to reuse the frozen run: {run_dir}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    train_path = P1_DIR / f"train_seed{args.seed}.npz"
    epoch_path = P1_DIR / f"epoch_orders_seed{args.seed}.npz"
    eval_path = P1_DIR / "eval.npz"
    train_arrays = load_npz(train_path)
    epoch_orders = load_npz(epoch_path)["epoch_order"]
    eval_arrays = load_npz(eval_path)
    target_mask = train_arrays["slot_kind"] == 1
    exposed_ids = set(
        int(value)
        for value in train_arrays[f"function_id_{args.arm}"][target_mask]
    )
    if len(exposed_ids) != 20:
        raise SystemExit("Run does not expose exactly 20 target functions.")

    set_deterministic_state(args.seed)
    device = get_device(args.device)
    model = build_model(
        model_args(args.model),
        context_dim=0,
        target_dim=20,
    )
    parameters = count_parameters(model)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001,
        foreach=False,
    )
    config = {
        "arm": args.arm,
        "batch_size": BATCH_SIZE,
        "checkpoint_selection": "fixed_final_epoch_60",
        "deterministic_algorithms": True,
        "device": str(device),
        "dropout": 0.05,
        "epochs": EPOCHS,
        "epoch_manifest": str(epoch_path),
        "epoch_manifest_sha256": sha256_file(epoch_path),
        "eval_manifest": str(eval_path),
        "eval_manifest_sha256": sha256_file(eval_path),
        "exposed_function_ids": sorted(exposed_ids),
        "ff_dim": 128,
        "gradient_clip_norm": 1.0,
        "heads": 4,
        "layers": 2,
        "learning_rate": 0.001,
        "model": args.model,
        "model_dim": 64,
        "model_seed": args.seed,
        "optimizer": "AdamW",
        "optimizer_foreach": False,
        "parameters": parameters,
        "platform": platform.platform(),
        "protocol": P1_SCHEMA_VERSION,
        "public_latent_count": 32,
        "run_id": run_id,
        "support_noise": 0.1,
        "support_size": SUPPORT_SIZE,
        "torch_version": torch.__version__,
        "train_manifest": str(train_path),
        "train_manifest_sha256": sha256_file(train_path),
        "weight_decay": 0.0001,
    }
    atomically_write_json(run_dir / "config.json", config)
    atomically_write_json(
        run_dir / "progress.json",
        {
            "epoch_completed": 0,
            "run_id": run_id,
            "status": "running",
        },
    )

    history: list[dict[str, Any]] = []
    start_time = time.time()
    for epoch_index in range(EPOCHS):
        epoch_start = time.time()
        loss, accuracy = train_one_epoch(
            model,
            optimizer,
            train_arrays,
            args.arm,
            epoch_orders[epoch_index],
            device,
        )
        synchronize(device)
        row = {
            "epoch": epoch_index + 1,
            "train_loss": loss,
            "train_accuracy": accuracy,
            "epoch_wall_seconds": time.time() - epoch_start,
        }
        history.append(row)
        atomically_write_json(
            run_dir / "progress.json",
            {
                **row,
                "elapsed_seconds": time.time() - start_time,
                "epoch_completed": epoch_index + 1,
                "run_id": run_id,
                "status": "running",
            },
        )
        print(
            f"{run_id} epoch={epoch_index + 1:02d}/{EPOCHS} "
            f"loss={loss:.6f} acc={accuracy:.4f} "
            f"seconds={row['epoch_wall_seconds']:.1f}",
            flush=True,
        )

    evaluation_start = time.time()
    metrics, prediction_arrays = evaluate(
        model,
        eval_arrays,
        args.arm,
        exposed_ids,
        device,
    )
    synchronize(device)
    evaluation_seconds = time.time() - evaluation_start

    checkpoint = {
        key: value.detach().cpu() for key, value in model.state_dict().items()
    }
    torch.save(checkpoint, run_dir / "final_model.pt")
    write_history(run_dir / "history.csv", history)
    write_deterministic_npz(
        run_dir / "eval_predictions.npz",
        prediction_arrays,
    )
    final_metrics = {
        **metrics,
        "arm": args.arm,
        "device": str(device),
        "epochs_completed": EPOCHS,
        "evaluation_seconds": evaluation_seconds,
        "final_checkpoint": "final_model.pt",
        "final_epoch": EPOCHS,
        "model": args.model,
        "model_seed": args.seed,
        "optimizer_updates": EPOCHS
        * ((TRAIN_EPISODES + BATCH_SIZE - 1) // BATCH_SIZE),
        "parameters": parameters,
        "run_id": run_id,
        "training_wall_seconds": time.time() - start_time,
    }
    atomically_write_json(run_dir / "metrics.json", final_metrics)
    atomically_write_json(
        run_dir / "progress.json",
        {
            "elapsed_seconds": time.time() - start_time,
            "epoch_completed": EPOCHS,
            "eval_accuracy": metrics["eval_accuracy"],
            "run_id": run_id,
            "status": "completed",
        },
    )
    print(
        f"completed {run_id} eval_accuracy={metrics['eval_accuracy']:.4f} "
        f"seconds={final_metrics['training_wall_seconds']:.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
