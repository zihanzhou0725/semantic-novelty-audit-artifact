#!/usr/bin/env python3
"""Shared deterministic construction utilities for quality-enhancement P1."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import sys
import zipfile
from argparse import Namespace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = ROOT / "external_semantic_overlap_audit"
PROTOCOL_DIR = EXTERNAL_ROOT / "protocols" / "quality-enhancement-v1"
P1_DIR = PROTOCOL_DIR / "p1"
KFWA_ROOT = (
    ROOT
    / "semantic_novelty_audit_supplement"
    / "src"
    / "kfwa_experiments"
)
if str(KFWA_ROOT) not in sys.path:
    sys.path.insert(0, str(KFWA_ROOT))

from kfwa.datasets import (  # noqa: E402
    SymbolicRule,
    build_symbolic_rules,
    group_symbolic_rules_by_function,
    symbolic_function_representatives,
)


SCHEMA_VERSION = "quality-enhancement-v1"
P1_SCHEMA_VERSION = "quality-enhancement-p1-v1"
MASTER_SEED = 20261009
NUM_SYMBOLS = 20
SUPPORT_SIZE = 8
SUPPORT_NOISE = 0.1
TRAIN_EPISODES = 20_000
EVAL_EPISODES_PER_FUNCTION = 250
EPOCHS = 60
BATCH_SIZE = 256
MODEL_SEEDS = (0, 1, 2, 3, 4)
ARMS = ("A", "B")
MODELS = (
    "symbolic_deepsets",
    "symbolic_relation_network",
    "symbolic_set_transformer",
    "symbolic_hopfield",
    "symbolic_perceiver",
    "symbolic_transformer",
)
FAMILIES = ("affine", "block", "parity")
FAMILY_TO_ID = {family: index for index, family in enumerate(FAMILIES)}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive_seed(namespace: str) -> int:
    payload = f"{SCHEMA_VERSION}:{MASTER_SEED}:{namespace}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def arm_assignment_bit(model_seed: int, intercept: int) -> int:
    payload = (
        f"{SCHEMA_VERSION}:{MASTER_SEED}:p1-arm:{model_seed}:{intercept}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 1


def deterministic_npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    """Return a byte-stable NPZ archive with fixed metadata and member order."""

    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            array_buffer = io.BytesIO()
            np.save(
                array_buffer,
                np.asarray(arrays[name]),
                allow_pickle=False,
            )
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, array_buffer.getvalue())
    return output.getvalue()


def write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(deterministic_npz_bytes(arrays))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                )
                + "\n"
            )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def symbolic_universe() -> dict[str, Any]:
    rules = build_symbolic_rules(
        num_symbols=NUM_SYMBOLS,
        families=",".join(FAMILIES),
    )
    groups = group_symbolic_rules_by_function(rules)
    representatives = symbolic_function_representatives(rules)
    target_ids = sorted(
        function_id
        for function_id, aliases in groups.items()
        if len({rule.family for rule in aliases}) > 1
    )
    base_ids = sorted(set(groups) - set(target_ids))

    target_records: list[dict[str, Any]] = []
    target_by_params: dict[tuple[int, int], int] = {}
    train_rule_by_function: dict[int, SymbolicRule] = {}
    eval_rule_by_function: dict[int, SymbolicRule] = {}
    for function_id in target_ids:
        aliases = groups[function_id]
        affine = [rule for rule in aliases if rule.family == "affine"]
        parity = [rule for rule in aliases if rule.family == "parity"]
        if len(affine) != 1 or len(parity) != 1:
            raise ValueError(
                f"Target {function_id} lacks a unique affine/parity identity."
            )
        multiplier, intercept = affine[0].params
        signature = sorted({rule.family for rule in aliases})
        target_by_params[(int(multiplier), int(intercept))] = function_id
        train_rule_by_function[function_id] = parity[0]
        eval_rule_by_function[function_id] = affine[0]
        target_records.append(
            {
                "function_id": function_id,
                "outputs": list(affine[0].outputs),
                "affine_rule_id": affine[0].rule_id,
                "affine_params": list(affine[0].params),
                "parity_rule_id": parity[0].rule_id,
                "parity_params": list(parity[0].params),
                "alias_rule_ids": [rule.rule_id for rule in aliases],
                "alias_families": [rule.family for rule in aliases],
                "alias_signature": "+".join(signature),
                "alias_count": len(aliases),
            }
        )

    pair_records: list[dict[str, Any]] = []
    for intercept in range(20):
        first = target_by_params[(1, intercept)]
        second = target_by_params[(11, intercept)]
        pair_records.append(
            {
                "pair_index": intercept,
                "intercept": intercept,
                "function_multiplier_1": first,
                "function_multiplier_11": second,
            }
        )

    return {
        "rules": rules,
        "groups": groups,
        "representatives": representatives,
        "target_ids": target_ids,
        "base_ids": base_ids,
        "target_records": sorted(
            target_records,
            key=lambda row: (
                row["affine_params"][0],
                row["affine_params"][1],
            ),
        ),
        "pair_records": pair_records,
        "target_by_params": target_by_params,
        "train_rule_by_function": train_rule_by_function,
        "eval_rule_by_function": eval_rule_by_function,
    }


def function_for_arm(
    pair: dict[str, Any],
    model_seed: int,
    arm: str,
) -> tuple[int, int]:
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    bit = arm_assignment_bit(model_seed, int(pair["intercept"]))
    first = int(pair["function_multiplier_1"])
    second = int(pair["function_multiplier_11"])
    if bit == 0:
        return (first, 1) if arm == "A" else (second, 11)
    return (second, 11) if arm == "A" else (first, 1)


def episode_components(namespace: str) -> tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    rng = np.random.Generator(np.random.PCG64(derive_seed(namespace)))
    support_digits = rng.choice(
        NUM_SYMBOLS,
        size=SUPPORT_SIZE,
        replace=False,
    ).astype(np.uint8)
    support_set = set(int(value) for value in support_digits)
    candidates = np.asarray(
        [value for value in range(NUM_SYMBOLS) if value not in support_set],
        dtype=np.uint8,
    )
    query_digit = int(rng.choice(candidates))
    corruption_mask = np.zeros(SUPPORT_SIZE, dtype=np.bool_)
    corruption_offsets = np.zeros(SUPPORT_SIZE, dtype=np.uint8)
    for index in range(SUPPORT_SIZE):
        if float(rng.random()) < SUPPORT_NOISE:
            corruption_mask[index] = True
            corruption_offsets[index] = np.uint8(rng.integers(1, NUM_SYMBOLS))
    return (
        support_digits,
        query_digit,
        corruption_mask,
        corruption_offsets,
    )


def labels_for_outputs(
    outputs: tuple[int, ...],
    support_digits: np.ndarray,
    query_digit: int,
    corruption_mask: np.ndarray,
    corruption_offsets: np.ndarray,
) -> tuple[np.ndarray, int]:
    clean = np.asarray(
        [outputs[int(digit)] for digit in support_digits],
        dtype=np.uint8,
    )
    labels = clean.copy()
    labels[corruption_mask] = (
        labels[corruption_mask] + corruption_offsets[corruption_mask]
    ) % NUM_SYMBOLS
    return labels, int(outputs[query_digit])


def build_train_arrays(model_seed: int) -> dict[str, np.ndarray]:
    universe = symbolic_universe()
    base_ids: list[int] = universe["base_ids"]
    pair_records: list[dict[str, Any]] = universe["pair_records"]
    representatives: dict[int, SymbolicRule] = universe["representatives"]
    train_rules: dict[int, SymbolicRule] = universe["train_rule_by_function"]
    groups: dict[int, list[SymbolicRule]] = universe["groups"]

    extra_rng = np.random.Generator(
        np.random.PCG64(derive_seed(f"p1-extra:{model_seed}"))
    )
    extra_order = np.asarray(base_ids, dtype=np.int64)
    extra_rng.shuffle(extra_order)
    extra_ids = set(int(value) for value in extra_order[:80])

    slot_kind: list[int] = []
    slot_reference: list[int] = []
    for function_id in base_ids:
        repetitions = 40 + int(function_id in extra_ids)
        slot_kind.extend([0] * repetitions)
        slot_reference.extend([function_id] * repetitions)
    for pair in pair_records:
        slot_kind.extend([1] * 40)
        slot_reference.extend([int(pair["pair_index"])] * 40)
    if len(slot_kind) != TRAIN_EPISODES:
        raise AssertionError(f"Unexpected train slot count: {len(slot_kind)}")

    permutation_rng = np.random.Generator(
        np.random.PCG64(derive_seed(f"p1-slots:{model_seed}"))
    )
    permutation = permutation_rng.permutation(TRAIN_EPISODES)
    kinds = np.asarray(slot_kind, dtype=np.uint8)[permutation]
    references = np.asarray(slot_reference, dtype=np.int64)[permutation]

    arrays: dict[str, np.ndarray] = {
        "support_digits": np.empty(
            (TRAIN_EPISODES, SUPPORT_SIZE),
            dtype=np.uint8,
        ),
        "query_digit": np.empty(TRAIN_EPISODES, dtype=np.uint8),
        "corruption_mask": np.empty(
            (TRAIN_EPISODES, SUPPORT_SIZE),
            dtype=np.bool_,
        ),
        "corruption_offset": np.empty(
            (TRAIN_EPISODES, SUPPORT_SIZE),
            dtype=np.uint8,
        ),
        "slot_kind": kinds,
        "pair_intercept": np.full(TRAIN_EPISODES, -1, dtype=np.int8),
        "function_id_A": np.empty(TRAIN_EPISODES, dtype=np.int64),
        "function_id_B": np.empty(TRAIN_EPISODES, dtype=np.int64),
        "identity_id_A": np.empty(TRAIN_EPISODES, dtype=np.int16),
        "identity_id_B": np.empty(TRAIN_EPISODES, dtype=np.int16),
        "family_id_A": np.empty(TRAIN_EPISODES, dtype=np.int8),
        "family_id_B": np.empty(TRAIN_EPISODES, dtype=np.int8),
        "function_alias_count_A": np.empty(TRAIN_EPISODES, dtype=np.int8),
        "function_alias_count_B": np.empty(TRAIN_EPISODES, dtype=np.int8),
        "function_multiplier_A": np.full(
            TRAIN_EPISODES,
            -1,
            dtype=np.int8,
        ),
        "function_multiplier_B": np.full(
            TRAIN_EPISODES,
            -1,
            dtype=np.int8,
        ),
        "support_labels_A": np.empty(
            (TRAIN_EPISODES, SUPPORT_SIZE),
            dtype=np.uint8,
        ),
        "support_labels_B": np.empty(
            (TRAIN_EPISODES, SUPPORT_SIZE),
            dtype=np.uint8,
        ),
        "target_A": np.empty(TRAIN_EPISODES, dtype=np.uint8),
        "target_B": np.empty(TRAIN_EPISODES, dtype=np.uint8),
    }

    for slot_index in range(TRAIN_EPISODES):
        support, query, corrupt, offsets = episode_components(
            f"p1-episode:{model_seed}:{slot_index}"
        )
        arrays["support_digits"][slot_index] = support
        arrays["query_digit"][slot_index] = query
        arrays["corruption_mask"][slot_index] = corrupt
        arrays["corruption_offset"][slot_index] = offsets

        if int(kinds[slot_index]) == 0:
            function_a = function_b = int(references[slot_index])
            rule_a = rule_b = representatives[function_a]
        else:
            pair = pair_records[int(references[slot_index])]
            function_a, multiplier_a = function_for_arm(
                pair,
                model_seed,
                "A",
            )
            function_b, multiplier_b = function_for_arm(
                pair,
                model_seed,
                "B",
            )
            rule_a = train_rules[function_a]
            rule_b = train_rules[function_b]
            arrays["pair_intercept"][slot_index] = int(pair["intercept"])
            arrays["function_multiplier_A"][slot_index] = multiplier_a
            arrays["function_multiplier_B"][slot_index] = multiplier_b

        for arm, function_id, rule in (
            ("A", function_a, rule_a),
            ("B", function_b, rule_b),
        ):
            labels, target = labels_for_outputs(
                representatives[function_id].outputs,
                support,
                query,
                corrupt,
                offsets,
            )
            arrays[f"function_id_{arm}"][slot_index] = function_id
            arrays[f"identity_id_{arm}"][slot_index] = rule.rule_id
            arrays[f"family_id_{arm}"][slot_index] = FAMILY_TO_ID[rule.family]
            arrays[f"function_alias_count_{arm}"][slot_index] = len(
                groups[function_id]
            )
            arrays[f"support_labels_{arm}"][slot_index] = labels
            arrays[f"target_{arm}"][slot_index] = target
    return arrays


def build_epoch_orders(model_seed: int) -> dict[str, np.ndarray]:
    orders = np.empty((EPOCHS, TRAIN_EPISODES), dtype=np.int32)
    base = np.arange(TRAIN_EPISODES, dtype=np.int32)
    for epoch in range(1, EPOCHS + 1):
        rng = np.random.Generator(
            np.random.PCG64(derive_seed(f"p1-epoch:{model_seed}:{epoch}"))
        )
        orders[epoch - 1] = rng.permutation(base)
    return {"epoch_order": orders}


def build_eval_arrays() -> dict[str, np.ndarray]:
    universe = symbolic_universe()
    representatives: dict[int, SymbolicRule] = universe["representatives"]
    eval_rules: dict[int, SymbolicRule] = universe["eval_rule_by_function"]
    groups: dict[int, list[SymbolicRule]] = universe["groups"]
    target_records: list[dict[str, Any]] = universe["target_records"]
    target_records = sorted(
        target_records,
        key=lambda row: (
            row["affine_params"][0],
            row["affine_params"][1],
        ),
    )
    size = len(target_records) * EVAL_EPISODES_PER_FUNCTION
    arrays: dict[str, np.ndarray] = {
        "support_digits": np.empty((size, SUPPORT_SIZE), dtype=np.uint8),
        "query_digit": np.empty(size, dtype=np.uint8),
        "corruption_mask": np.empty((size, SUPPORT_SIZE), dtype=np.bool_),
        "corruption_offset": np.empty((size, SUPPORT_SIZE), dtype=np.uint8),
        "support_labels": np.empty((size, SUPPORT_SIZE), dtype=np.uint8),
        "target": np.empty(size, dtype=np.uint8),
        "function_id": np.empty(size, dtype=np.int64),
        "identity_id": np.empty(size, dtype=np.int16),
        "family_id": np.empty(size, dtype=np.int8),
        "function_alias_count": np.empty(size, dtype=np.int8),
        "function_multiplier": np.empty(size, dtype=np.int8),
        "pair_intercept": np.empty(size, dtype=np.int8),
        "episode_within_function": np.empty(size, dtype=np.int16),
    }
    index = 0
    for record in target_records:
        function_id = int(record["function_id"])
        multiplier, intercept = (int(value) for value in record["affine_params"])
        rule = eval_rules[function_id]
        for episode_index in range(EVAL_EPISODES_PER_FUNCTION):
            support, query, corrupt, offsets = episode_components(
                f"p1-eval:{function_id}:{episode_index}"
            )
            labels, target = labels_for_outputs(
                representatives[function_id].outputs,
                support,
                query,
                corrupt,
                offsets,
            )
            arrays["support_digits"][index] = support
            arrays["query_digit"][index] = query
            arrays["corruption_mask"][index] = corrupt
            arrays["corruption_offset"][index] = offsets
            arrays["support_labels"][index] = labels
            arrays["target"][index] = target
            arrays["function_id"][index] = function_id
            arrays["identity_id"][index] = rule.rule_id
            arrays["family_id"][index] = FAMILY_TO_ID[rule.family]
            arrays["function_alias_count"][index] = len(groups[function_id])
            arrays["function_multiplier"][index] = multiplier
            arrays["pair_intercept"][index] = intercept
            arrays["episode_within_function"][index] = episode_index
            index += 1
    if index != size:
        raise AssertionError(f"Unexpected eval row count: {index}")
    return arrays


def build_run_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        for model_seed in MODEL_SEEDS:
            for arm in ARMS:
                run_id = f"{model}__seed{model_seed}__arm{arm}"
                rows.append(
                    {
                        "run_id": run_id,
                        "model": model,
                        "model_seed": model_seed,
                        "arm": arm,
                        "train_manifest": f"train_seed{model_seed}.npz",
                        "epoch_manifest": f"epoch_orders_seed{model_seed}.npz",
                        "eval_manifest": "eval.npz",
                        "epochs": EPOCHS,
                        "optimizer_updates": EPOCHS
                        * ((TRAIN_EPISODES + BATCH_SIZE - 1) // BATCH_SIZE),
                        "status": "not_started",
                    }
                )
    return rows


def model_args(model: str) -> Namespace:
    return Namespace(
        model=model,
        num_symbols=NUM_SYMBOLS,
        support_max_k=SUPPORT_SIZE,
        model_dim=64,
        hidden_dim=256,
        layers=2,
        heads=4,
        ff_dim=128,
        public_latent_count=32,
        dropout=0.05,
    )


def set_deterministic_state(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def parse_checksum_file(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        digest, filename = line.split("  ", maxsplit=1)
        checksums[filename] = digest
    return checksums


def validate_checksum_file(base_dir: Path, checksum_path: Path) -> None:
    for filename, expected in parse_checksum_file(checksum_path).items():
        observed = sha256_file(base_dir / filename)
        if observed != expected:
            raise ValueError(
                f"Checksum mismatch for {filename}: {observed} != {expected}"
            )
