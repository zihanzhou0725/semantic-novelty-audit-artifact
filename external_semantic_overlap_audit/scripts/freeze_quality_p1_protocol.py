#!/usr/bin/env python3
"""Materialize and freeze all preregistered P1 design artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from quality_p1_common import (
    ARMS,
    BATCH_SIZE,
    EPOCHS,
    EVAL_EPISODES_PER_FUNCTION,
    EXTERNAL_ROOT,
    MASTER_SEED,
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
    sha256_bytes,
    sha256_file,
    symbolic_universe,
    write_csv,
    write_deterministic_npz,
    write_json,
    write_jsonl,
)
from validate_quality_p1_canary import run_canaries


GENERATED_FILES = (
    "canary_report.json",
    "design_manifest.json",
    "targets.jsonl",
    "pairs.jsonl",
    "eval.npz",
    "run_manifest.csv",
    *tuple(f"train_seed{seed}.npz" for seed in MODEL_SEEDS),
    *tuple(f"epoch_orders_seed{seed}.npz" for seed in MODEL_SEEDS),
)


def source_hashes() -> dict[str, str]:
    relative_paths = (
        "external_semantic_overlap_audit/scripts/quality_p1_common.py",
        "external_semantic_overlap_audit/scripts/validate_quality_p1_canary.py",
        "external_semantic_overlap_audit/scripts/freeze_quality_p1_protocol.py",
        "external_semantic_overlap_audit/scripts/validate_quality_p1_protocol.py",
        "external_semantic_overlap_audit/scripts/run_quality_p1.py",
        "external_semantic_overlap_audit/scripts/run_quality_p1_matrix.py",
        "semantic_novelty_audit_supplement/src/kfwa_experiments/kfwa/datasets.py",
        "semantic_novelty_audit_supplement/src/kfwa_experiments/kfwa/models.py",
        "semantic_novelty_audit_supplement/src/kfwa_experiments/kfwa/symbolic.py",
        "semantic_novelty_audit_supplement/src/kfwa_experiments/kfwa/metrics.py",
        "semantic_novelty_audit_supplement/src/kfwa_experiments/kfwa/training.py",
        "semantic_novelty_audit_supplement/src/kfwa_experiments/kfwa/utils.py",
    )
    result: dict[str, str] = {}
    for relative in relative_paths:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing P1 source file: {path}")
        result[relative] = sha256_file(path)
    return result


def target_rows(universe: dict[str, Any]) -> list[dict[str, Any]]:
    return list(universe["target_records"])


def pair_rows(universe: dict[str, Any]) -> list[dict[str, Any]]:
    return list(universe["pair_records"])


def design_manifest(
    universe: dict[str, Any],
    canary: dict[str, Any],
    train_hashes: dict[int, str],
    epoch_hashes: dict[int, str],
    eval_hash: str,
) -> dict[str, Any]:
    signatures = Counter(
        row["alias_signature"] for row in universe["target_records"]
    )
    return {
        "schema_version": P1_SCHEMA_VERSION,
        "parent_schema_version": "quality-enhancement-v1",
        "parent_preregistration_sha256": sha256_file(
            PROTOCOL_DIR / "preregistration.json"
        ),
        "master_seed": MASTER_SEED,
        "status": "frozen_before_p1_training",
        "training_started": False,
        "canary_status": canary["status"],
        "function_census": {
            "identity_count": len(universe["rules"]),
            "function_count": len(universe["groups"]),
            "target_count": len(universe["target_ids"]),
            "base_count": len(universe["base_ids"]),
            "pair_count": len(universe["pair_records"]),
            "alias_signature_counts": dict(sorted(signatures.items())),
            "target_semantic_set_sha256": sha256_bytes(
                canonical_json_bytes(
                    [
                        {
                            "function_id": row["function_id"],
                            "outputs": row["outputs"],
                        }
                        for row in universe["target_records"]
                    ]
                )
            ),
            "base_semantic_set_sha256": sha256_bytes(
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
            ),
        },
        "training": {
            "models": list(MODELS),
            "model_seeds": list(MODEL_SEEDS),
            "arms": list(ARMS),
            "run_count": len(MODELS) * len(MODEL_SEEDS) * len(ARMS),
            "episodes_per_arm": TRAIN_EPISODES,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "optimizer_updates": EPOCHS
            * ((TRAIN_EPISODES + BATCH_SIZE - 1) // BATCH_SIZE),
            "checkpoint_selection": "fixed_final_epoch_60",
            "train_manifest_sha256": {
                str(seed): train_hashes[seed] for seed in MODEL_SEEDS
            },
            "epoch_manifest_sha256": {
                str(seed): epoch_hashes[seed] for seed in MODEL_SEEDS
            },
        },
        "evaluation": {
            "target_functions": len(universe["target_ids"]),
            "episodes_per_function": EVAL_EPISODES_PER_FUNCTION,
            "rows": len(universe["target_ids"])
            * EVAL_EPISODES_PER_FUNCTION,
            "manifest_sha256": eval_hash,
            "common_across_all_runs": True,
        },
        "source_sha256": source_hashes(),
    }


def write_checksum_manifest(directory: Path) -> None:
    lines = [
        f"{sha256_file(directory / filename)}  {filename}"
        for filename in GENERATED_FILES
    ]
    (directory / "SHA256SUMS").write_text(
        "\n".join(lines) + "\n",
        encoding="ascii",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze quality-enhancement P1 manifests without training."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing P1 directory before training only.",
    )
    args = parser.parse_args()

    parent_validation = json.loads(
        (PROTOCOL_DIR / "validation_report.json").read_text(encoding="ascii")
    )
    if parent_validation.get("status") != "passed":
        raise SystemExit("Parent P0 protocol validation has not passed.")
    if P1_DIR.exists():
        if not args.force:
            raise SystemExit(f"P1 directory already exists: {P1_DIR}")
        existing_design = P1_DIR / "design_manifest.json"
        if existing_design.exists():
            existing = json.loads(existing_design.read_text(encoding="ascii"))
            if existing.get("training_started"):
                raise SystemExit("Refusing to replace P1 after training started.")

    canary = run_canaries()
    if canary["status"] != "passed":
        raise SystemExit("P1 canaries did not pass.")

    universe = symbolic_universe()
    staging = PROTOCOL_DIR / ".p1.build"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    write_json(staging / "canary_report.json", canary)
    write_jsonl(staging / "targets.jsonl", target_rows(universe))
    write_jsonl(staging / "pairs.jsonl", pair_rows(universe))

    eval_arrays = build_eval_arrays()
    write_deterministic_npz(staging / "eval.npz", eval_arrays)
    eval_hash = sha256_file(staging / "eval.npz")

    train_hashes: dict[int, str] = {}
    epoch_hashes: dict[int, str] = {}
    for seed in MODEL_SEEDS:
        train_path = staging / f"train_seed{seed}.npz"
        epoch_path = staging / f"epoch_orders_seed{seed}.npz"
        write_deterministic_npz(train_path, build_train_arrays(seed))
        write_deterministic_npz(epoch_path, build_epoch_orders(seed))
        train_hashes[seed] = sha256_file(train_path)
        epoch_hashes[seed] = sha256_file(epoch_path)

    write_csv(staging / "run_manifest.csv", build_run_rows())
    write_json(
        staging / "design_manifest.json",
        design_manifest(
            universe,
            canary,
            train_hashes,
            epoch_hashes,
            eval_hash,
        ),
    )
    write_checksum_manifest(staging)

    if P1_DIR.exists():
        shutil.rmtree(P1_DIR)
    staging.rename(P1_DIR)
    print(f"Frozen P1 protocol at {P1_DIR}")
    print(f"Artifacts: {len(GENERATED_FILES)} plus SHA256SUMS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
