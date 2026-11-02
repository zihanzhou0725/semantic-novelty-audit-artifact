"""Shared deterministic training utilities for CLUTRR V3."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from toolkit.external_behavior.common import (
    ARMS,
    MODEL_SEEDS,
    PROJECT_ROOT,
    derive_seed,
    iter_jsonl,
    read_json,
    sha256_bytes,
    sha256_file,
    write_json,
)
from toolkit.external_behavior.model import (
    ByteTransformerClassifier,
    encode_normalized_input,
)


EXECUTION_ID = "external-behavior-v3-execution-v1"
DESIGN_DIR = PROJECT_ROOT / "manifests" / "external-behavior-v3"
DATA_DIR = DESIGN_DIR / "data"
EXECUTION_DIR = DESIGN_DIR / "execution-v1"
PROTOCOL_DIR = (
    PROJECT_ROOT
    / "protocols"
    / "external-behavior-v3"
    / "execution-v1"
)
RUN_ROOT = (
    PROJECT_ROOT
    / "runs"
    / "external-behavior-v3"
    / "execution-v1"
)
FIXED_LENGTH = 185
TRAIN_BATCH_SIZE = 128
EVAL_BATCH_SIZE = 256
EPOCHS = 30
TRAIN_ROWS = 24_000
EVAL_ROWS = 8_000
UPDATES_PER_EPOCH = 188
UPDATES_PER_RUN = 5_640
TORCH_INTRAOP_THREADS = 2
TORCH_INTEROP_THREADS = 1
MAX_CONCURRENT_RUNS = 8


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def deterministic_npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_STORED,
    ) as archive:
        for name in sorted(arrays):
            array_buffer = io.BytesIO()
            np.save(
                array_buffer,
                np.asarray(arrays[name]),
                allow_pickle=False,
            )
            info = zipfile.ZipInfo(
                f"{name}.npy",
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, array_buffer.getvalue())
    return output.getvalue()


def write_deterministic_npz(
    path: Path,
    arrays: dict[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(deterministic_npz_bytes(arrays))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="ascii") as handle:
        return list(csv.DictReader(handle))


def model_seed_value(model_seed: int) -> int:
    return derive_seed(f"clutrr:model:{model_seed}")


def epoch_order(model_seed: int, epoch: int) -> np.ndarray:
    if model_seed not in MODEL_SEEDS or not 1 <= epoch <= EPOCHS:
        raise ValueError("Invalid model seed or epoch.")
    seed = derive_seed(f"clutrr:epoch:{model_seed}:{epoch}")
    rng = np.random.Generator(np.random.PCG64(seed))
    return rng.permutation(TRAIN_ROWS).astype(np.int32)


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(canonical_json_bytes(list(value.shape)))
    digest.update(value.tobytes())
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous().numpy()
    return array_sha256(value)


def state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        value = tensor.numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(canonical_json_bytes(list(value.shape)))
        digest.update(value.tobytes())
    return digest.hexdigest()


def torch_rng_sha256() -> str:
    return tensor_sha256(torch.get_rng_state())


def initialize_model(model_seed: int) -> tuple[
    ByteTransformerClassifier,
    str,
    str,
]:
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(model_seed_value(model_seed))
    model = ByteTransformerClassifier(23)
    return (
        model,
        state_dict_sha256(model.state_dict()),
        torch_rng_sha256(),
    )


def classes() -> list[str]:
    manifest = read_json(
        PROJECT_ROOT
        / "manifests"
        / "external-behavior-v1"
        / "implementation_manifest.json"
    )
    result = list(manifest["model_output_classes"])
    if len(result) != 23:
        raise ValueError("Frozen CLUTRR class count changed.")
    return result


def _arm_path(seed: int, arm: str) -> Path:
    return DATA_DIR / f"clutrr_train_seed{seed}_arm{arm}.txt"


def arm_row_ids(seed: int, arm: str) -> list[str]:
    if seed not in MODEL_SEEDS or arm not in ARMS:
        raise ValueError("Unknown run.")
    return _arm_path(seed, arm).read_text(encoding="ascii").splitlines()


def needed_row_ids() -> set[str]:
    result: set[str] = set()
    for seed in MODEL_SEEDS:
        for arm in ARMS:
            result.update(arm_row_ids(seed, arm))
    result.update(
        row["surface_id"]
        for row in iter_jsonl(DATA_DIR / "clutrr_v3_evaluation.jsonl")
    )
    return result


def build_tensor_cache() -> dict[str, Any]:
    class_names = classes()
    label_to_id = {name: index for index, name in enumerate(class_names)}
    needed = needed_row_ids()
    selected: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(DATA_DIR / "clutrr_v3_candidates.jsonl"):
        if row["surface_id"] in needed:
            selected[row["surface_id"]] = row
    missing = sorted(needed - set(selected))
    if missing:
        raise ValueError(f"Tensor cache is missing {len(missing)} rows.")

    row_ids = sorted(selected)
    token_ids = np.zeros((len(row_ids), FIXED_LENGTH), dtype=np.uint16)
    attention_mask = np.zeros((len(row_ids), FIXED_LENGTH), dtype=np.bool_)
    labels = np.zeros(len(row_ids), dtype=np.int16)
    for index, row_id in enumerate(row_ids):
        row = selected[row_id]
        tokens = encode_normalized_input(row["normalized_input"])
        if len(tokens) > FIXED_LENGTH:
            raise ValueError("Frozen input exceeds fixed padding length.")
        token_ids[index, : len(tokens)] = tokens
        attention_mask[index, : len(tokens)] = True
        labels[index] = label_to_id[row["target_relation"]]
    row_id_array = np.asarray(row_ids, dtype="S64")
    write_deterministic_npz(
        EXECUTION_DIR / "tensor_cache.npz",
        {
            "attention_mask": attention_mask,
            "labels": labels,
            "row_ids": row_id_array,
            "token_ids": token_ids,
        },
    )
    cache_index = {row_id: index for index, row_id in enumerate(row_ids)}

    train_arrays: dict[str, np.ndarray] = {}
    for seed in MODEL_SEEDS:
        for arm in ARMS:
            ids = arm_row_ids(seed, arm)
            train_arrays[f"seed{seed}_arm{arm}"] = np.asarray(
                [cache_index[row_id] for row_id in ids],
                dtype=np.int32,
            )
    write_deterministic_npz(
        EXECUTION_DIR / "train_cache_indices.npz",
        train_arrays,
    )

    evaluation = list(
        iter_jsonl(DATA_DIR / "clutrr_v3_evaluation.jsonl")
    )
    write_deterministic_npz(
        EXECUTION_DIR / "evaluation_manifest.npz",
        {
            "cache_indices": np.asarray(
                [cache_index[row["surface_id"]] for row in evaluation],
                dtype=np.int32,
            ),
            "labels": np.asarray(
                [label_to_id[row["target_relation"]] for row in evaluation],
                dtype=np.int16,
            ),
            "pair_index": np.asarray(
                [row["pair_index"] for row in evaluation],
                dtype=np.int16,
            ),
            "pair_member": np.asarray(
                [row["pair_member"] for row in evaluation],
                dtype=np.int8,
            ),
            "row_ids": np.asarray(
                [row["surface_id"] for row in evaluation],
                dtype="S64",
            ),
            "semantic_ids": np.asarray(
                [row["semantic_id"] for row in evaluation],
                dtype="S64",
            ),
            "source_group_ids": np.asarray(
                [row["source_group_id"] for row in evaluation],
                dtype="S64",
            ),
        },
    )
    return {
        "cached_rows": len(row_ids),
        "evaluation_rows": len(evaluation),
        "needed_rows": len(needed),
        "tensor_cache_sha256": sha256_file(
            EXECUTION_DIR / "tensor_cache.npz"
        ),
        "train_cache_indices_sha256": sha256_file(
            EXECUTION_DIR / "train_cache_indices.npz"
        ),
        "evaluation_manifest_sha256": sha256_file(
            EXECUTION_DIR / "evaluation_manifest.npz"
        ),
    }


def initialization_manifest() -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for seed in MODEL_SEEDS:
        model, state_hash, rng_hash = initialize_model(seed)
        records.append(
            {
                "initial_model_sha256": state_hash,
                "model_seed": seed,
                "seed_value": model_seed_value(seed),
                "training_start_torch_rng_sha256": rng_hash,
            }
        )
        del model
    return {
        "records": records,
        "schema_version": "external-behavior-v3-initialization-v1",
    }


def epoch_order_manifest() -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for seed in MODEL_SEEDS:
        for epoch in range(1, EPOCHS + 1):
            order = epoch_order(seed, epoch)
            records.append(
                {
                    "epoch": epoch,
                    "model_seed": seed,
                    "order_sha256": array_sha256(order),
                    "seed_value": derive_seed(
                        f"clutrr:epoch:{seed}:{epoch}"
                    ),
                }
            )
    return {
        "records": records,
        "schema_version": "external-behavior-v3-epoch-orders-v1",
    }


def run_id(seed: int, arm: str) -> str:
    return f"seed{seed}_arm{arm}"


def build_run_manifest(
    initialization: dict[str, Any],
    epoch_orders: dict[str, Any],
) -> list[dict[str, Any]]:
    init_by_seed = {
        int(row["model_seed"]): row for row in initialization["records"]
    }
    order_hashes: dict[int, list[str]] = {seed: [] for seed in MODEL_SEEDS}
    for row in epoch_orders["records"]:
        order_hashes[int(row["model_seed"])].append(row["order_sha256"])
    rows: list[dict[str, Any]] = []
    for seed in MODEL_SEEDS:
        combined_order_hash = sha256_bytes(
            canonical_json_bytes(order_hashes[seed])
        )
        for arm in ARMS:
            rows.append(
                {
                    "arm": arm,
                    "combined_epoch_order_sha256": combined_order_hash,
                    "expected_initial_model_sha256": init_by_seed[seed][
                        "initial_model_sha256"
                    ],
                    "expected_training_start_torch_rng_sha256": init_by_seed[
                        seed
                    ]["training_start_torch_rng_sha256"],
                    "model_seed": seed,
                    "run_id": run_id(seed, arm),
                    "train_manifest": _arm_path(seed, arm).name,
                    "train_manifest_sha256": sha256_file(
                        _arm_path(seed, arm)
                    ),
                }
            )
    return rows


def finite_tree(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    if isinstance(value, float):
        return bool(np.isfinite(value))
    return True


def artifact_hashes(directory: Path, names: Iterable[str]) -> dict[str, str]:
    return {name: sha256_file(directory / name) for name in sorted(names)}


__all__ = [
    "ARMS",
    "DATA_DIR",
    "EPOCHS",
    "EVAL_BATCH_SIZE",
    "EVAL_ROWS",
    "EXECUTION_DIR",
    "EXECUTION_ID",
    "FIXED_LENGTH",
    "MAX_CONCURRENT_RUNS",
    "MODEL_SEEDS",
    "PROJECT_ROOT",
    "PROTOCOL_DIR",
    "RUN_ROOT",
    "TORCH_INTEROP_THREADS",
    "TORCH_INTRAOP_THREADS",
    "TRAIN_BATCH_SIZE",
    "TRAIN_ROWS",
    "UPDATES_PER_EPOCH",
    "UPDATES_PER_RUN",
    "array_sha256",
    "arm_row_ids",
    "artifact_hashes",
    "build_run_manifest",
    "build_tensor_cache",
    "canonical_json_bytes",
    "classes",
    "epoch_order",
    "epoch_order_manifest",
    "finite_tree",
    "initialization_manifest",
    "initialize_model",
    "load_npz",
    "read_csv",
    "read_json",
    "run_id",
    "sha256_file",
    "state_dict_sha256",
    "torch_rng_sha256",
    "write_csv",
    "write_deterministic_npz",
    "write_json",
]
