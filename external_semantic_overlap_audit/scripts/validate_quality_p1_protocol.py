#!/usr/bin/env python3
"""Validate every frozen quality-enhancement P1 design invariant."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from quality_p1_common import (
    ARMS,
    EPOCHS,
    EVAL_EPISODES_PER_FUNCTION,
    MODEL_SEEDS,
    MODELS,
    P1_DIR,
    P1_SCHEMA_VERSION,
    PROTOCOL_DIR,
    ROOT,
    TRAIN_EPISODES,
    build_epoch_orders,
    build_eval_arrays,
    build_run_rows,
    build_train_arrays,
    canonical_json_bytes,
    load_npz,
    parse_checksum_file,
    sha256_bytes,
    sha256_file,
    symbolic_universe,
    write_json,
)


def fail(message: str) -> None:
    raise SystemExit(f"P1 validation failed: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="ascii") as handle:
        return [json.loads(line) for line in handle]


def compare_arrays(
    observed: dict[str, np.ndarray],
    expected: dict[str, np.ndarray],
    label: str,
) -> None:
    require(
        set(observed) == set(expected),
        f"{label} fields differ: {set(observed) ^ set(expected)}",
    )
    for name in sorted(expected):
        require(
            observed[name].dtype == expected[name].dtype,
            f"{label}.{name} dtype mismatch",
        )
        require(
            observed[name].shape == expected[name].shape,
            f"{label}.{name} shape mismatch",
        )
        require(
            np.array_equal(observed[name], expected[name]),
            f"{label}.{name} content mismatch",
        )


def validate_parent() -> str:
    report = json.loads(
        (PROTOCOL_DIR / "validation_report.json").read_text(encoding="ascii")
    )
    require(report.get("status") == "passed", "parent P0 validation is not passed")
    preregistration_hash = sha256_file(PROTOCOL_DIR / "preregistration.json")
    require(
        report["preregistration_sha256"] == preregistration_hash,
        "parent P0 validation refers to another preregistration",
    )
    return preregistration_hash


def validate_checksums() -> dict[str, str]:
    checksums = parse_checksum_file(P1_DIR / "SHA256SUMS")
    required = {
        "canary_report.json",
        "design_manifest.json",
        "targets.jsonl",
        "pairs.jsonl",
        "eval.npz",
        "run_manifest.csv",
        *{f"train_seed{seed}.npz" for seed in MODEL_SEEDS},
        *{f"epoch_orders_seed{seed}.npz" for seed in MODEL_SEEDS},
    }
    require(set(checksums) == required, "P1 checksum file set is incomplete")
    for filename, expected in checksums.items():
        observed = sha256_file(P1_DIR / filename)
        require(observed == expected, f"checksum mismatch for {filename}")
    return checksums


def validate_source_hashes(design: dict[str, Any]) -> None:
    for relative, expected in design["source_sha256"].items():
        path = ROOT / relative
        require(path.is_file(), f"missing frozen source: {relative}")
        require(
            sha256_file(path) == expected,
            f"frozen source changed: {relative}",
        )


def validate_census(
    design: dict[str, Any],
    universe: dict[str, Any],
) -> None:
    census = design["function_census"]
    require(len(universe["rules"]) == census["identity_count"] == 560, "identity census")
    require(len(universe["groups"]) == census["function_count"] == 518, "function census")
    require(len(universe["target_ids"]) == census["target_count"] == 40, "target census")
    require(len(universe["base_ids"]) == census["base_count"] == 478, "base census")
    require(len(universe["pair_records"]) == census["pair_count"] == 20, "pair census")
    signatures = Counter(
        row["alias_signature"] for row in universe["target_records"]
    )
    require(
        dict(sorted(signatures.items()))
        == census["alias_signature_counts"]
        == {"affine+block+parity": 2, "affine+parity": 38},
        "target alias signatures",
    )
    require(
        not (set(universe["target_ids"]) & set(universe["base_ids"])),
        "target/base semantic overlap",
    )
    require(
        set(universe["target_ids"]) | set(universe["base_ids"])
        == set(universe["groups"]),
        "target/base semantic partition",
    )

    target_hash = sha256_bytes(
        canonical_json_bytes(
            [
                {
                    "function_id": row["function_id"],
                    "outputs": row["outputs"],
                }
                for row in universe["target_records"]
            ]
        )
    )
    base_hash = sha256_bytes(
        canonical_json_bytes(
            [
                {
                    "function_id": function_id,
                    "outputs": list(
                        universe["representatives"][function_id].outputs
                    ),
                }
                for function_id in universe["base_ids"]
            ]
        )
    )
    require(target_hash == census["target_semantic_set_sha256"], "target set hash")
    require(base_hash == census["base_semantic_set_sha256"], "base set hash")

    observed_targets = load_jsonl(P1_DIR / "targets.jsonl")
    observed_pairs = load_jsonl(P1_DIR / "pairs.jsonl")
    require(observed_targets == universe["target_records"], "target rows changed")
    require(observed_pairs == universe["pair_records"], "pair rows changed")


def validate_eval(
    design: dict[str, Any],
    universe: dict[str, Any],
) -> dict[str, np.ndarray]:
    observed = load_npz(P1_DIR / "eval.npz")
    expected = build_eval_arrays()
    compare_arrays(observed, expected, "eval")
    size = len(universe["target_ids"]) * EVAL_EPISODES_PER_FUNCTION
    require(len(observed["target"]) == size == 10_000, "eval row count")
    require(np.all(observed["family_id"] == 0), "eval identities are not affine")
    function_counts = Counter(int(value) for value in observed["function_id"])
    require(
        set(function_counts) == set(universe["target_ids"]),
        "eval target set mismatch",
    )
    require(
        set(function_counts.values()) == {EVAL_EPISODES_PER_FUNCTION},
        "eval function weighting mismatch",
    )
    require(
        design["evaluation"]["manifest_sha256"]
        == sha256_file(P1_DIR / "eval.npz"),
        "eval hash differs from design manifest",
    )
    require(design["evaluation"]["common_across_all_runs"], "eval is not common")
    return observed


def validate_train_seed(
    seed: int,
    design: dict[str, Any],
    universe: dict[str, Any],
) -> dict[str, np.ndarray]:
    observed = load_npz(P1_DIR / f"train_seed{seed}.npz")
    expected = build_train_arrays(seed)
    compare_arrays(observed, expected, f"train_seed{seed}")
    epoch_observed = load_npz(P1_DIR / f"epoch_orders_seed{seed}.npz")
    epoch_expected = build_epoch_orders(seed)
    compare_arrays(
        epoch_observed,
        epoch_expected,
        f"epoch_orders_seed{seed}",
    )
    require(
        epoch_observed["epoch_order"].shape == (EPOCHS, TRAIN_EPISODES),
        f"seed {seed} epoch shape",
    )
    require(
        design["training"]["train_manifest_sha256"][str(seed)]
        == sha256_file(P1_DIR / f"train_seed{seed}.npz"),
        f"seed {seed} train hash",
    )
    require(
        design["training"]["epoch_manifest_sha256"][str(seed)]
        == sha256_file(P1_DIR / f"epoch_orders_seed{seed}.npz"),
        f"seed {seed} epoch hash",
    )

    base_mask = observed["slot_kind"] == 0
    target_mask = ~base_mask
    require(int(base_mask.sum()) == 19_200, f"seed {seed} base slots")
    require(int(target_mask.sum()) == 800, f"seed {seed} target slots")
    for field in (
        "function_id",
        "identity_id",
        "family_id",
        "function_alias_count",
        "support_labels",
        "target",
    ):
        require(
            np.array_equal(
                observed[f"{field}_A"][base_mask],
                observed[f"{field}_B"][base_mask],
            ),
            f"seed {seed} base {field} pairing",
        )
    require(
        np.all(observed["family_id_A"][target_mask] == 2)
        and np.all(observed["family_id_B"][target_mask] == 2),
        f"seed {seed} target identities are not parity",
    )
    require(
        np.all(
            observed["function_id_A"][target_mask]
            != observed["function_id_B"][target_mask]
        ),
        f"seed {seed} complementary target assignment",
    )

    base_ids = set(universe["base_ids"])
    target_ids = set(universe["target_ids"])
    for arm in ARMS:
        base_counts = Counter(
            int(value) for value in observed[f"function_id_{arm}"][base_mask]
        )
        target_counts = Counter(
            int(value) for value in observed[f"function_id_{arm}"][target_mask]
        )
        require(set(base_counts) == base_ids, f"seed {seed} arm {arm} base set")
        require(
            Counter(base_counts.values()) == Counter({40: 398, 41: 80}),
            f"seed {seed} arm {arm} base dose",
        )
        require(
            len(target_counts) == 20 and set(target_counts.values()) == {40},
            f"seed {seed} arm {arm} target dose",
        )
        require(
            set(target_counts).issubset(target_ids),
            f"seed {seed} arm {arm} target set",
        )

    exposed_a = set(
        int(value) for value in observed["function_id_A"][target_mask]
    )
    exposed_b = set(
        int(value) for value in observed["function_id_B"][target_mask]
    )
    require(not (exposed_a & exposed_b), f"seed {seed} arm target overlap")
    require(exposed_a | exposed_b == target_ids, f"seed {seed} target coverage")
    require(
        np.array_equal(
            np.sort(epoch_observed["epoch_order"], axis=1),
            np.broadcast_to(
                np.arange(TRAIN_EPISODES, dtype=np.int32),
                (EPOCHS, TRAIN_EPISODES),
            ),
        ),
        f"seed {seed} epoch permutations",
    )
    return observed


def validate_run_manifest(design: dict[str, Any]) -> None:
    with (P1_DIR / "run_manifest.csv").open(newline="", encoding="ascii") as handle:
        rows = list(csv.DictReader(handle))
    expected = [
        {key: str(value) for key, value in row.items()}
        for row in build_run_rows()
    ]
    require(rows == expected, "run manifest differs from registered matrix")
    require(len(rows) == design["training"]["run_count"] == 60, "run count")
    require(len({row["run_id"] for row in rows}) == 60, "run IDs are not unique")
    require(
        Counter(row["model"] for row in rows)
        == Counter({model: 10 for model in MODELS}),
        "model run balance",
    )
    require(
        Counter(int(row["model_seed"]) for row in rows)
        == Counter({seed: 12 for seed in MODEL_SEEDS}),
        "seed run balance",
    )
    require(
        Counter(row["arm"] for row in rows) == Counter({"A": 30, "B": 30}),
        "arm run balance",
    )


def main() -> int:
    parent_hash = validate_parent()
    checksums = validate_checksums()
    design = json.loads(
        (P1_DIR / "design_manifest.json").read_text(encoding="ascii")
    )
    require(design["schema_version"] == P1_SCHEMA_VERSION, "schema version")
    require(
        design["parent_preregistration_sha256"] == parent_hash,
        "parent preregistration hash",
    )
    require(
        design["status"] == "frozen_before_p1_training"
        and not design["training_started"],
        "P1 was not frozen before training",
    )
    canary = json.loads(
        (P1_DIR / "canary_report.json").read_text(encoding="ascii")
    )
    require(
        canary["status"] == "passed"
        and canary["training_updates_performed"] == 0,
        "pre-freeze canary status",
    )
    validate_source_hashes(design)
    universe = symbolic_universe()
    validate_census(design, universe)
    validate_eval(design, universe)
    for seed in MODEL_SEEDS:
        validate_train_seed(seed, design, universe)
    validate_run_manifest(design)

    report = {
        "checks": {
            "parent_p0_validation": "passed",
            "frozen_checksums": "passed",
            "source_checksums": "passed",
            "pre_freeze_canaries": "passed",
            "identity_function_target_census": "passed",
            "target_base_partition": "passed",
            "target_pairing": "passed",
            "train_stream_reconstruction": "passed",
            "paired_base_identity": "passed",
            "complementary_target_exposure": "passed",
            "equal_arm_dose_and_family": "passed",
            "epoch_order_reconstruction": "passed",
            "eval_stream_reconstruction": "passed",
            "eval_affine_identity": "passed",
            "run_matrix_60": "passed",
        },
        "design_gate": "passed",
        "files_checked": len(checksums),
        "p1_checksum_manifest_sha256": sha256_file(P1_DIR / "SHA256SUMS"),
        "schema_version": P1_SCHEMA_VERSION,
        "status": "passed",
        "training_may_start": True,
    }
    write_json(P1_DIR / "validation_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
