"""Shared deterministic training utilities for directional CLUTRR V4."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from toolkit.external_behavior.common import (
    derive_seed,
    iter_jsonl,
)
from toolkit.external_behavior.model import encode_normalized_input
from toolkit.external_behavior_v3.training_common import (
    array_sha256,
    artifact_hashes,
    canonical_json_bytes,
    classes,
    deterministic_npz_bytes,
    epoch_order,
    finite_tree,
    initialize_model,
    load_npz,
    read_csv,
    read_json,
    sha256_bytes,
    sha256_file,
    state_dict_sha256,
    tensor_sha256,
    torch_rng_sha256,
    write_csv,
    write_deterministic_npz,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXECUTION_ID = "external-behavior-v4-execution-v1"
MATERIALIZATION_DIR = (
    PROJECT_ROOT
    / "manifests"
    / "external-behavior-v4"
    / "materialization-v1"
)
DATA_DIR = MATERIALIZATION_DIR / "data"
EXECUTION_DIR = (
    PROJECT_ROOT
    / "manifests"
    / "external-behavior-v4"
    / "execution-v1"
)
PROTOCOL_DIR = (
    PROJECT_ROOT
    / "protocols"
    / "external-behavior-v4"
    / "execution-v1"
)
RUN_ROOT = (
    PROJECT_ROOT
    / "runs"
    / "external-behavior-v4"
    / "execution-v1"
)
BLOCKS = (0, 1)
MODEL_SEEDS = tuple(range(8))
ARMS = ("A", "B")
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
RUN_COUNT = 32


def run_id(block: int, seed: int, arm: str) -> str:
    return f"block{block}_seed{seed}_arm{arm}"


def model_seed_value(model_seed: int) -> int:
    return derive_seed(f"clutrr:model:{model_seed}")


def _materialized_rows() -> list[dict[str, str]]:
    rows = read_csv(MATERIALIZATION_DIR / "run_manifest.csv")
    if len(rows) != RUN_COUNT:
        raise RuntimeError("Frozen materialized run count changed.")
    return rows


def train_path(row: dict[str, str]) -> Path:
    return DATA_DIR / row["train_manifest"]


def needed_row_ids() -> set[str]:
    result: set[str] = set()
    for row in _materialized_rows():
        result.update(
            train_path(row).read_text(encoding="ascii").splitlines()
        )
    for block in BLOCKS:
        result.update(
            row["surface_id"]
            for row in iter_jsonl(
                DATA_DIR / f"evaluation_block{block}.jsonl"
            )
        )
    return result


def _evaluation_arrays(
    block: int,
    evaluation: list[dict[str, Any]],
    cache_index: dict[str, int],
    label_to_id: dict[str, int],
) -> dict[str, np.ndarray]:
    return {
        "block_index": np.asarray(
            [block for _ in evaluation],
            dtype=np.int8,
        ),
        "block_pair_index": np.asarray(
            [row["block_pair_index"] for row in evaluation],
            dtype=np.int16,
        ),
        "cache_indices": np.asarray(
            [cache_index[row["surface_id"]] for row in evaluation],
            dtype=np.int32,
        ),
        "global_pair_index": np.asarray(
            [row["global_v4_pair_index"] for row in evaluation],
            dtype=np.int16,
        ),
        "labels": np.asarray(
            [label_to_id[row["target_relation"]] for row in evaluation],
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
    }


def build_tensor_cache() -> dict[str, Any]:
    class_names = classes()
    label_to_id = {name: index for index, name in enumerate(class_names)}
    needed = needed_row_ids()
    selected: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(
        PROJECT_ROOT
        / "manifests"
        / "external-behavior-v3"
        / "data"
        / "clutrr_v3_candidates.jsonl"
    ):
        if row["surface_id"] in needed:
            selected[row["surface_id"]] = row
    missing = sorted(needed - set(selected))
    if missing:
        raise RuntimeError(f"Tensor cache is missing {len(missing)} rows.")
    row_ids = sorted(selected)
    token_ids = np.zeros((len(row_ids), FIXED_LENGTH), dtype=np.uint16)
    attention_mask = np.zeros(
        (len(row_ids), FIXED_LENGTH),
        dtype=np.bool_,
    )
    labels = np.zeros(len(row_ids), dtype=np.int16)
    for index, row_id in enumerate(row_ids):
        row = selected[row_id]
        tokens = encode_normalized_input(row["normalized_input"])
        if len(tokens) > FIXED_LENGTH:
            raise RuntimeError("Frozen V4 input exceeds fixed length.")
        token_ids[index, : len(tokens)] = tokens
        attention_mask[index, : len(tokens)] = True
        labels[index] = label_to_id[row["target_relation"]]
    write_deterministic_npz(
        EXECUTION_DIR / "tensor_cache.npz",
        {
            "attention_mask": attention_mask,
            "labels": labels,
            "row_ids": np.asarray(row_ids, dtype="S64"),
            "token_ids": token_ids,
        },
    )
    cache_index = {row_id: index for index, row_id in enumerate(row_ids)}
    train_arrays: dict[str, np.ndarray] = {}
    for row in _materialized_rows():
        ids = train_path(row).read_text(encoding="ascii").splitlines()
        train_arrays[row["run_id"]] = np.asarray(
            [cache_index[row_id] for row_id in ids],
            dtype=np.int32,
        )
    write_deterministic_npz(
        EXECUTION_DIR / "train_cache_indices.npz",
        train_arrays,
    )
    evaluation_hashes: dict[str, str] = {}
    for block in BLOCKS:
        evaluation = list(
            iter_jsonl(DATA_DIR / f"evaluation_block{block}.jsonl")
        )
        if len(evaluation) != EVAL_ROWS:
            raise RuntimeError("Frozen V4 evaluation row count changed.")
        path = EXECUTION_DIR / f"evaluation_block{block}.npz"
        write_deterministic_npz(
            path,
            _evaluation_arrays(
                block,
                evaluation,
                cache_index,
                label_to_id,
            ),
        )
        evaluation_hashes[str(block)] = sha256_file(path)
    return {
        "cached_rows": len(row_ids),
        "evaluation_manifest_sha256": evaluation_hashes,
        "evaluation_rows_per_block": EVAL_ROWS,
        "needed_rows": len(needed),
        "tensor_cache_sha256": sha256_file(
            EXECUTION_DIR / "tensor_cache.npz"
        ),
        "train_cache_indices_sha256": sha256_file(
            EXECUTION_DIR / "train_cache_indices.npz"
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
        "schema_version": "external-behavior-v4-initialization-v1",
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
        "schema_version": "external-behavior-v4-epoch-orders-v1",
    }


def build_run_manifest(
    initialization: dict[str, Any],
    epoch_orders: dict[str, Any],
) -> list[dict[str, Any]]:
    init_by_seed = {
        int(row["model_seed"]): row for row in initialization["records"]
    }
    order_hashes: dict[int, list[str]] = {
        seed: [] for seed in MODEL_SEEDS
    }
    for row in epoch_orders["records"]:
        order_hashes[int(row["model_seed"])].append(row["order_sha256"])
    rows: list[dict[str, Any]] = []
    for source in _materialized_rows():
        block = int(source["block_index"])
        seed = int(source["model_seed"])
        arm = source["arm"]
        identifier = run_id(block, seed, arm)
        if identifier != source["run_id"]:
            raise RuntimeError("Materialized run ID changed.")
        evaluation_path = EXECUTION_DIR / f"evaluation_block{block}.npz"
        rows.append(
            {
                "arm": arm,
                "block_index": block,
                "combined_epoch_order_sha256": sha256_bytes(
                    canonical_json_bytes(order_hashes[seed])
                ),
                "evaluation_manifest": evaluation_path.name,
                "evaluation_manifest_sha256": sha256_file(
                    evaluation_path
                ),
                "expected_initial_model_sha256": init_by_seed[seed][
                    "initial_model_sha256"
                ],
                "expected_training_start_torch_rng_sha256": init_by_seed[
                    seed
                ]["training_start_torch_rng_sha256"],
                "materialized_evaluation_sha256": source[
                    "evaluation_manifest_sha256"
                ],
                "model_seed": seed,
                "run_id": identifier,
                "train_manifest": source["train_manifest"],
                "train_manifest_sha256": source[
                    "train_manifest_sha256"
                ],
            }
        )
    if len(rows) != RUN_COUNT:
        raise RuntimeError("Execution run manifest count changed.")
    return rows


def tensor_cache_summary(path: Path) -> dict[str, Any]:
    cache = load_npz(path)
    return {
        name: {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
        for name, value in sorted(cache.items())
    }


__all__ = [
    "ARMS",
    "BLOCKS",
    "DATA_DIR",
    "EPOCHS",
    "EVAL_BATCH_SIZE",
    "EVAL_ROWS",
    "EXECUTION_DIR",
    "EXECUTION_ID",
    "FIXED_LENGTH",
    "MATERIALIZATION_DIR",
    "MAX_CONCURRENT_RUNS",
    "MODEL_SEEDS",
    "PROJECT_ROOT",
    "PROTOCOL_DIR",
    "RUN_COUNT",
    "RUN_ROOT",
    "TORCH_INTEROP_THREADS",
    "TORCH_INTRAOP_THREADS",
    "TRAIN_BATCH_SIZE",
    "TRAIN_ROWS",
    "UPDATES_PER_EPOCH",
    "UPDATES_PER_RUN",
    "array_sha256",
    "artifact_hashes",
    "build_run_manifest",
    "build_tensor_cache",
    "canonical_json_bytes",
    "classes",
    "deterministic_npz_bytes",
    "epoch_order",
    "epoch_order_manifest",
    "finite_tree",
    "initialization_manifest",
    "initialize_model",
    "load_npz",
    "read_csv",
    "read_json",
    "run_id",
    "sha256_bytes",
    "sha256_file",
    "state_dict_sha256",
    "tensor_cache_summary",
    "tensor_sha256",
    "torch_rng_sha256",
    "train_path",
    "write_csv",
    "write_deterministic_npz",
    "write_json",
]
