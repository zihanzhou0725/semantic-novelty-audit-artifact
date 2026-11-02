#!/usr/bin/env python3
"""Validate the frozen Boolean ICL behavioral protocol and split manifests."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from freeze_boolean_behavioral_protocol import (
    EVAL_FUNCTION_DRAWS,
    MODEL_SEEDS,
    PUBLIC_SOURCE_HASHES,
    SCHEMA_VERSION,
    TASKS,
    TRAIN_FUNCTION_DRAWS,
    decode_parameter,
    ids_sha256,
    replace_excluded_slots,
    sample_ids,
    semantic_id,
    sha256_file,
)


def fail(message: str) -> None:
    raise SystemExit(f"Validation failed: {message}")


def parse_checksums(path: Path) -> dict[str, str]:
    checksums = {}
    for line in path.read_text().splitlines():
        digest, filename = line.split("  ", maxsplit=1)
        checksums[filename] = digest
    return checksums


def validate_checksums(protocol_dir: Path) -> None:
    expected = parse_checksums(protocol_dir / "SHA256SUMS")
    for filename, digest in expected.items():
        observed = sha256_file(protocol_dir / filename)
        if observed != digest:
            fail(f"checksum mismatch for {filename}: {observed} != {digest}")


def validate_source_hashes(root: Path) -> None:
    source_dir = root / "upstream" / "boolean_icl_public" / "src"
    for filename, digest in PUBLIC_SOURCE_HASHES.items():
        observed = sha256_file(source_dir / filename)
        if observed != digest:
            fail(f"source checksum mismatch for {filename}")


def load_eval_rows(path: Path) -> dict[str, list[dict]]:
    rows_by_task: dict[str, list[dict]] = defaultdict(list)
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            rows_by_task[row["task"]].append(row)
    return dict(rows_by_task)


def validate_eval_rows(
    rows_by_task: dict[str, list[dict]], split_manifest: dict
) -> dict[str, np.ndarray]:
    if set(rows_by_task) != set(TASKS):
        fail(f"unexpected eval tasks: {set(rows_by_task)}")

    result = {}
    for task_name, rows in rows_by_task.items():
        if len(rows) != EVAL_FUNCTION_DRAWS:
            fail(f"{task_name} has {len(rows)} eval rows")
        if [row["eval_index"] for row in rows] != list(range(EVAL_FUNCTION_DRAWS)):
            fail(f"{task_name} eval indices are not contiguous")

        encoded = np.asarray(
            [row["encoded_semantic_id"] for row in rows], dtype=np.uint64
        )
        for row in rows:
            value = int(row["encoded_semantic_id"])
            if row["semantic_id"] != semantic_id(task_name, value):
                fail(f"{task_name} semantic ID text mismatch")
            if row["parameter"] != decode_parameter(task_name, value):
                fail(f"{task_name} decoded parameter mismatch")
            if task_name == "sparse_disjunction_d50_k4_positive":
                if len(row["parameter"]["active_indices"]) != 4:
                    fail("Sparse Disjunction semantic ID does not contain four variables")

        summary = split_manifest["task_eval_summaries"][task_name]
        regenerated = sample_ids(
            task_name, EVAL_FUNCTION_DRAWS, summary["function_seed"]
        )
        if not np.array_equal(encoded, regenerated):
            fail(f"{task_name} eval stream is not reproducible")
        if ids_sha256(encoded) != summary["ordered_semantic_ids_sha256"]:
            fail(f"{task_name} ordered eval hash mismatch")
        if len(np.unique(encoded)) != summary["unique_semantic_id_count"]:
            fail(f"{task_name} unique eval count mismatch")
        result[task_name] = encoded
    return result


def validate_run_rows(
    rows: list[dict],
    eval_by_task: dict[str, np.ndarray],
    split_manifest: dict,
) -> None:
    expected_count = len(TASKS) * len(MODEL_SEEDS) * 2
    if len(rows) != expected_count:
        fail(f"expected {expected_count} run rows, found {len(rows)}")

    grouped: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        grouped[(row["task"], int(row["model_seed"]))][row["protocol"]] = row

    expected_pairs = {
        (task_name, seed) for task_name in TASKS for seed in MODEL_SEEDS
    }
    if set(grouped) != expected_pairs:
        fail("run task/seed pairs do not match the preregistration")

    for (task_name, model_seed), protocols in grouped.items():
        if set(protocols) != {"original", "strict_function_disjoint"}:
            fail(f"missing protocol for {task_name} seed {model_seed}")
        original_row = protocols["original"]
        strict_row = protocols["strict_function_disjoint"]
        for field in (
            "train_function_seed",
            "replacement_seed",
            "train_numpy_input_seed",
            "train_torch_input_seed",
            "function_draw_count",
        ):
            if original_row[field] != strict_row[field]:
                fail(f"paired field {field} differs for {task_name} seed {model_seed}")

        eval_unique = np.unique(eval_by_task[task_name])
        original = sample_ids(
            task_name,
            TRAIN_FUNCTION_DRAWS,
            int(original_row["train_function_seed"]),
        )
        strict, replacement_draws = replace_excluded_slots(
            task_name,
            original,
            eval_unique,
            int(strict_row["replacement_seed"]),
        )
        collision_mask = np.isin(original, eval_unique)
        if ids_sha256(original) != original_row["function_stream_sha256"]:
            fail(f"original stream hash mismatch for {task_name} seed {model_seed}")
        if ids_sha256(strict) != strict_row["function_stream_sha256"]:
            fail(f"strict stream hash mismatch for {task_name} seed {model_seed}")
        if np.any(np.isin(strict, eval_unique)):
            fail(f"strict/eval overlap for {task_name} seed {model_seed}")
        if not np.array_equal(strict[~collision_mask], original[~collision_mask]):
            fail(f"noncollision slots changed for {task_name} seed {model_seed}")
        if int(collision_mask.sum()) != int(strict_row["replaced_slot_count"]):
            fail(f"replacement count mismatch for {task_name} seed {model_seed}")
        if replacement_draws != int(strict_row["replacement_candidate_draw_count"]):
            fail(f"replacement draw count mismatch for {task_name} seed {model_seed}")

    if split_manifest["validation_invariants"]["strict_eval_overlap_count"] != 0:
        fail("split manifest does not freeze zero strict/eval overlap")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol_dir = root / "protocols" / SCHEMA_VERSION
    validate_checksums(protocol_dir)
    validate_source_hashes(root)

    preregistration = json.loads(
        (protocol_dir / "preregistration.json").read_text()
    )
    split_manifest = json.loads((protocol_dir / "split_manifest.json").read_text())
    if preregistration["schema_version"] != SCHEMA_VERSION:
        fail("preregistration schema version mismatch")
    if split_manifest["protocol_generator_sha256"] != sha256_file(
        root / "scripts" / "freeze_boolean_behavioral_protocol.py"
    ):
        fail("protocol generator checksum mismatch")
    if split_manifest["training_adapter_sha256"] != sha256_file(
        root / "scripts" / "run_boolean_behavioral.py"
    ):
        fail("training adapter checksum mismatch")
    if split_manifest["behavioral_requirements_sha256"] != sha256_file(
        root / "behavioral_requirements.txt"
    ):
        fail("behavioral requirements checksum mismatch")
    if preregistration["status"] != "frozen_before_training":
        fail("preregistration is not marked frozen")
    if preregistration["training_started"]:
        fail("preregistration says training has already started")
    if preregistration["training"]["core_run_count"] != 18:
        fail("core run count is not 18")

    eval_by_task = validate_eval_rows(
        load_eval_rows(protocol_dir / "eval_functions.jsonl"), split_manifest
    )
    with (protocol_dir / "run_manifest.csv").open(newline="") as handle:
        run_rows = list(csv.DictReader(handle))
    validate_run_rows(run_rows, eval_by_task, split_manifest)

    exposure_by_task = defaultdict(list)
    for row in run_rows:
        if row["protocol"] == "original":
            exposure_by_task[row["task"]].append(
                float(row["eval_draw_exposure_rate"])
            )
    ordered_means = {
        task: sum(values) / len(values)
        for task, values in exposure_by_task.items()
    }
    if not (
        ordered_means["conjunction_d28"]
        < ordered_means["majority_d28"]
        < ordered_means["sparse_disjunction_d50_k4_positive"]
    ):
        fail("empirical exposure levels are not ordered low < medium < high")

    report = {
        "status": "passed",
        "schema_version": SCHEMA_VERSION,
        "checks": {
            "frozen_file_checksums": "passed",
            "upstream_source_checksums": "passed",
            "eval_stream_reconstruction": "passed",
            "semantic_id_roundtrip": "passed",
            "run_stream_reconstruction": "passed",
            "strict_eval_disjointness": "passed",
            "paired_noncollision_identity": "passed",
            "empirical_exposure_ordering": "passed",
        },
        "eval_rows": sum(len(rows) for rows in eval_by_task.values()),
        "run_rows": len(run_rows),
        "mean_original_eval_exposure_rate": ordered_means,
        "protocol_counts": Counter(row["protocol"] for row in run_rows),
    }
    output_path = protocol_dir / "validation_report.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
