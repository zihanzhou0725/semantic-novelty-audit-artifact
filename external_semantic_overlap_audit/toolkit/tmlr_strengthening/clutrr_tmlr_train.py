"""Independent CLUTRR training loop for TMLR strengthening experiments."""

from __future__ import annotations

import hashlib
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from toolkit.external_behavior.common import derive_seed
from toolkit.external_behavior.model import (
    ByteTransformerClassifier,
    encode_normalized_input,
)
from toolkit.external_behavior_v3.training_common import (
    load_npz,
    state_dict_sha256,
    write_deterministic_npz,
)
from toolkit.tmlr_strengthening.artifacts import (
    sha256_file,
    write_csv,
    write_json,
)


FIXED_LENGTH = 185
TRAIN_BATCH_SIZE = 128
EVAL_BATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 3e-4
BETAS = (0.9, 0.999)
EPSILON = 1e-8
WEIGHT_DECAY = 0.01
GRADIENT_CLIP = 1.0
TORCH_INTRAOP_THREADS = 2
TORCH_INTEROP_THREADS = 1


def _surface_sha256(rows: list[dict[str, object]]) -> str:
    payload = "\n".join(str(row["surface_id"]) for row in rows) + "\n"
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def epoch_order(
    seed: int,
    epoch: int,
    row_count: int,
    namespace: str,
) -> np.ndarray:
    if seed < 0 or epoch < 1 or row_count < 1 or not namespace:
        raise ValueError("Invalid epoch-order input.")
    rng = np.random.Generator(
        np.random.PCG64(
            derive_seed(f"{namespace}:epoch:{seed}:{epoch}")
        )
    )
    return rng.permutation(row_count).astype(np.int32)


def initialize_model(
    seed: int,
    namespace: str,
) -> tuple[ByteTransformerClassifier, str]:
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(derive_seed(f"{namespace}:model:{seed}"))
    model = ByteTransformerClassifier(23)
    return model, state_dict_sha256(model.state_dict())


def build_execution_inputs(
    design: dict[str, object],
    output_dir: Path,
    classes: list[str],
) -> dict[str, object]:
    """Encode the union of arm and evaluation rows once."""

    if len(classes) != 23:
        raise ValueError("Expected 23 CLUTRR classes.")
    arm_rows = design["arm_rows"]
    evaluation_rows = design["evaluation_rows"]
    if not isinstance(arm_rows, dict) or not isinstance(
        evaluation_rows,
        list,
    ):
        raise TypeError("Malformed T6 design.")
    selected: dict[str, dict[str, object]] = {}
    for rows in list(arm_rows.values()) + [evaluation_rows]:
        for row in rows:
            surface_id = str(row["surface_id"])
            previous = selected.setdefault(surface_id, row)
            if previous != row:
                raise ValueError(f"Inconsistent row payload: {surface_id}")
    row_ids = sorted(selected)
    label_to_id = {name: index for index, name in enumerate(classes)}
    token_ids = np.zeros((len(row_ids), FIXED_LENGTH), dtype=np.uint16)
    attention_mask = np.zeros(
        (len(row_ids), FIXED_LENGTH),
        dtype=np.bool_,
    )
    labels = np.zeros(len(row_ids), dtype=np.int16)
    for index, row_id in enumerate(row_ids):
        row = selected[row_id]
        tokens = encode_normalized_input(str(row["normalized_input"]))
        if len(tokens) > FIXED_LENGTH:
            raise ValueError(f"Input exceeds fixed length: {row_id}")
        token_ids[index, : len(tokens)] = tokens
        attention_mask[index, : len(tokens)] = True
        labels[index] = label_to_id[str(row["target_relation"])]
    cache_index = {row_id: index for index, row_id in enumerate(row_ids)}
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "tensor_cache.npz"
    train_path = output_dir / "train_indices.npz"
    evaluation_path = output_dir / "evaluation.npz"
    write_deterministic_npz(
        cache_path,
        {
            "attention_mask": attention_mask,
            "labels": labels,
            "row_ids": np.asarray(row_ids, dtype="S64"),
            "token_ids": token_ids,
        },
    )
    write_deterministic_npz(
        train_path,
        {
            arm: np.asarray(
                [
                    cache_index[str(row["surface_id"])]
                    for row in arm_rows[arm]
                ],
                dtype=np.int32,
            )
            for arm in sorted(arm_rows)
        },
    )
    pattern_by_semantic = {
        str(record["semantic_id"]): int(record["pattern_index"])
        for record in design["allocation"]
    }
    write_deterministic_npz(
        evaluation_path,
        {
            "cache_indices": np.asarray(
                [
                    cache_index[str(row["surface_id"])]
                    for row in evaluation_rows
                ],
                dtype=np.int32,
            ),
            "labels": np.asarray(
                [
                    label_to_id[str(row["target_relation"])]
                    for row in evaluation_rows
                ],
                dtype=np.int16,
            ),
            "pattern_indices": np.asarray(
                [
                    pattern_by_semantic[str(row["semantic_id"])]
                    for row in evaluation_rows
                ],
                dtype=np.int16,
            ),
            "row_ids": np.asarray(
                [str(row["surface_id"]) for row in evaluation_rows],
                dtype="S64",
            ),
            "semantic_ids": np.asarray(
                [str(row["semantic_id"]) for row in evaluation_rows],
                dtype="S64",
            ),
        },
    )
    manifest = {
        "arm_surface_id_sha256": {
            arm: _surface_sha256(arm_rows[arm])
            for arm in sorted(arm_rows)
        },
        "cache_rows": len(row_ids),
        "evaluation_rows": len(evaluation_rows),
        "evaluation_surface_id_sha256": _surface_sha256(
            evaluation_rows
        ),
        "files": {
            path.name: sha256_file(path)
            for path in (cache_path, train_path, evaluation_path)
        },
        "label_counts": design["label_counts"],
        "train_rows": {
            arm: len(arm_rows[arm]) for arm in sorted(arm_rows)
        },
    }
    write_json(output_dir / "execution_inputs_manifest.json", manifest)
    return manifest


def _batch(
    cache: dict[str, np.ndarray],
    cache_indices: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = torch.from_numpy(
        cache["token_ids"][cache_indices].astype(np.int64, copy=False)
    ).to(device)
    attention_mask = torch.from_numpy(
        cache["attention_mask"][cache_indices]
    ).to(device)
    labels = torch.from_numpy(
        cache["labels"][cache_indices].astype(np.int64, copy=False)
    ).to(device)
    return input_ids, attention_mask, labels


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def _select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    if device_name not in {"cpu", "mps"}:
        raise ValueError("device must be auto, cpu, or mps")
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable.")
    return torch.device(device_name)


def train_and_eval(
    *,
    input_dir: Path,
    arm: str,
    seed: int,
    namespace: str,
    output_dir: Path,
    epochs: int = EPOCHS,
    device_name: str = "auto",
) -> dict[str, object]:
    """Train one arm and evaluate the shared block-0 target set."""

    if arm not in {"C0", "C1", "C2"} or seed not in range(4):
        raise ValueError("Unknown T6 arm or seed.")
    if not 1 <= epochs <= EPOCHS:
        raise ValueError("epochs must lie in [1, 30]")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(TORCH_INTRAOP_THREADS)
    try:
        torch.set_num_interop_threads(TORCH_INTEROP_THREADS)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    device = _select_device(device_name)
    cache = load_npz(input_dir / "tensor_cache.npz")
    train_indices = load_npz(input_dir / "train_indices.npz")[arm]
    evaluation = load_npz(input_dir / "evaluation.npz")
    row_count = len(train_indices)
    if row_count != 24_640 or len(evaluation["labels"]) != 8_000:
        raise ValueError("T6 train or evaluation shape changed.")

    model, initial_model_hash = initialize_model(seed, namespace)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=BETAS,
        eps=EPSILON,
        weight_decay=WEIGHT_DECAY,
    )
    history: list[dict[str, object]] = []
    updates = 0
    _synchronize(device)
    wall_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        order = epoch_order(seed, epoch, row_count, namespace)
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_rows = 0
        _synchronize(device)
        epoch_start = time.perf_counter()
        model.train()
        for start in range(0, row_count, TRAIN_BATCH_SIZE):
            slots = order[start : start + TRAIN_BATCH_SIZE]
            input_ids, attention_mask, labels = _batch(
                cache,
                train_indices[slots],
                device,
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids, attention_mask)
            loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite T6 training loss.")
            loss.backward()
            for parameter in model.parameters():
                if (
                    parameter.grad is not None
                    and not torch.isfinite(parameter.grad).all()
                ):
                    raise FloatingPointError("Non-finite T6 gradient.")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRADIENT_CLIP,
            )
            optimizer.step()
            batch_rows = len(slots)
            epoch_loss += float(loss.detach().cpu()) * batch_rows
            epoch_correct += int(
                (logits.detach().argmax(dim=1) == labels)
                .sum()
                .cpu()
            )
            epoch_rows += batch_rows
            updates += 1
        _synchronize(device)
        history.append(
            {
                "epoch": epoch,
                "seconds": time.perf_counter() - epoch_start,
                "train_accuracy": epoch_correct / epoch_rows,
                "train_loss": epoch_loss / epoch_rows,
                "updates": updates,
            }
        )
        write_csv(output_dir / "history.csv", history)

    model.eval()
    evaluation_count = len(evaluation["labels"])
    logits_array = np.zeros(
        (evaluation_count, 23),
        dtype=np.float32,
    )
    with torch.no_grad():
        for start in range(0, evaluation_count, EVAL_BATCH_SIZE):
            stop = min(start + EVAL_BATCH_SIZE, evaluation_count)
            indices = evaluation["cache_indices"][start:stop]
            input_ids, attention_mask, _ = _batch(cache, indices, device)
            logits = model(input_ids, attention_mask)
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Non-finite T6 evaluation logits.")
            logits_array[start:stop] = logits.detach().cpu().numpy()
    _synchronize(device)
    labels = evaluation["labels"].astype(np.int64)
    predictions = logits_array.argmax(axis=1).astype(np.int16)
    shifted = logits_array - logits_array.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    probability_correct = probabilities[
        np.arange(evaluation_count),
        labels,
    ]
    cross_entropy = float(
        (
            -shifted[np.arange(evaluation_count), labels]
            + np.log(np.exp(shifted).sum(axis=1))
        ).mean()
    )
    seconds = time.perf_counter() - wall_start
    final_state = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
    }
    final_model_hash = state_dict_sha256(final_state)
    torch.save(
        {
            "arm": arm,
            "epochs": epochs,
            "final_model_sha256": final_model_hash,
            "initial_model_sha256": initial_model_hash,
            "model_state": final_state,
            "namespace": namespace,
            "seed": seed,
            "updates": updates,
        },
        output_dir / "final_checkpoint.pt",
    )
    write_deterministic_npz(
        output_dir / "eval_predictions.npz",
        {
            "labels": evaluation["labels"],
            "pattern_indices": evaluation["pattern_indices"],
            "predictions": predictions,
            "prob_correct": probability_correct.astype(np.float32),
            "row_ids": evaluation["row_ids"],
            "semantic_ids": evaluation["semantic_ids"],
        },
    )
    metrics = {
        "accuracy": float((predictions == labels).mean()),
        "arm": arm,
        "cross_entropy": cross_entropy,
        "device": str(device),
        "epochs": epochs,
        "evaluation_rows": evaluation_count,
        "final_model_sha256": final_model_hash,
        "initial_model_sha256": initial_model_hash,
        "n_train": row_count,
        "namespace": namespace,
        "seconds": seconds,
        "seed": seed,
        "updates": updates,
    }
    write_json(output_dir / "metrics.json", metrics)
    label_counts = Counter(
        int(value) for value in cache["labels"][train_indices]
    )
    write_json(
        output_dir / "config.json",
        {
            "arm": arm,
            "batch_size": TRAIN_BATCH_SIZE,
            "betas": list(BETAS),
            "device": str(device),
            "epochs": epochs,
            "epsilon": EPSILON,
            "evaluation_batch_size": EVAL_BATCH_SIZE,
            "fixed_length": FIXED_LENGTH,
            "gradient_clip": GRADIENT_CLIP,
            "input_hashes": {
                path.name: sha256_file(path)
                for path in (
                    input_dir / "tensor_cache.npz",
                    input_dir / "train_indices.npz",
                    input_dir / "evaluation.npz",
                )
            },
            "label_counts": {
                str(key): value for key, value in sorted(label_counts.items())
            },
            "learning_rate": LEARNING_RATE,
            "namespace": namespace,
            "seed": seed,
            "train_indices_sha256": hashlib.sha256(
                np.ascontiguousarray(train_indices).tobytes()
            ).hexdigest(),
            "weight_decay": WEIGHT_DECAY,
        },
    )
    if not all(
        math.isfinite(float(metrics[name]))
        for name in ("accuracy", "cross_entropy", "seconds")
    ):
        raise FloatingPointError("Non-finite T6 final metric.")
    return metrics

