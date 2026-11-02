"""Frozen directional exposure-effect analysis for CLUTRR V4."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from toolkit.external_behavior_v3.effect_analysis import (
    array_sha256,
    classification_metrics,
    deterministic_npz_bytes,
    load_npz,
    read_csv,
    read_json,
    relative,
    sha256_file,
    write_csv,
    write_deterministic_npz,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_ID = "external-behavior-v4-effect-analysis-v1"
PROTOCOL_DIR = (
    PROJECT_ROOT
    / "protocols"
    / "external-behavior-v4"
    / "effect-analysis-v1"
)
ANALYSIS_DIR = (
    PROJECT_ROOT
    / "analysis"
    / "external-behavior-v4"
    / "effect-analysis-v1"
)
EXECUTION_DIR = (
    PROJECT_ROOT
    / "manifests"
    / "external-behavior-v4"
    / "execution-v1"
)
RUN_ROOT = (
    PROJECT_ROOT
    / "runs"
    / "external-behavior-v4"
    / "execution-v1"
)
MATERIALIZATION_DIR = (
    PROJECT_ROOT
    / "manifests"
    / "external-behavior-v4"
    / "materialization-v1"
)
DIRECTIONAL_DESIGN_DIR = (
    PROJECT_ROOT
    / "manifests"
    / "external-behavior-v4"
    / "directional-design-v1"
)
BLOCKS = (0, 1)
MODEL_SEEDS = tuple(range(8))
ARMS = ("A", "B")
PAIR_COUNT = 40
PATTERN_COUNT = 80
GROUPS_PER_PATTERN = 34
ROWS_PER_PATTERN = 200
EVAL_ROWS_PER_BLOCK = 8_000
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 202_610_132
BOOTSTRAP_BATCH_SIZE = 64
SESOI_PP = 1.0

IMPLEMENTATION_PATHS = (
    "scripts/freeze_external_behavior_v4_effect_analysis.py",
    "scripts/analyze_external_behavior_v4_effect.py",
    "scripts/validate_external_behavior_v4_effect.py",
    "tests/test_external_behavior_v4_effect_analysis.py",
    "toolkit/external_behavior_v3/effect_analysis.py",
    "toolkit/external_behavior_v4/effect_analysis.py",
    "protocols/external-behavior-v4/effect-analysis-v1/validate.py",
)

OUTPUT_NAMES = (
    "block_effects.csv",
    "bootstrap.npz",
    "pair_effects.csv",
    "pattern_effects.csv",
    "relation_effects.csv",
    "report_cn.md",
    "results.json",
    "seed_aligned_metrics.csv",
    "seed_effects.csv",
    "seed_pattern_effects.csv",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _checksum_entries(path: Path) -> list[tuple[str, Path]]:
    entries: list[tuple[str, Path]] = []
    for line in path.read_text(encoding="ascii").splitlines():
        if line:
            digest, name = line.split("  ", 1)
            entries.append((digest, path.parent / name))
    return entries


def _input_paths() -> list[Path]:
    paths = [
        PROJECT_ROOT
        / "protocols"
        / "external-behavior-v4"
        / "directional-design-v1"
        / "preregistration.json",
        PROJECT_ROOT
        / "protocols"
        / "external-behavior-v4"
        / "execution-v1"
        / "preregistration.json",
        EXECUTION_DIR / "EXECUTION_SHA256SUMS",
        EXECUTION_DIR / "execution_manifest.json",
        EXECUTION_DIR / "execution_validation_report.json",
        EXECUTION_DIR / "run_artifact_manifest.json",
        EXECUTION_DIR / "run_manifest.csv",
        EXECUTION_DIR / "evaluation_block0.npz",
        EXECUTION_DIR / "evaluation_block1.npz",
        MATERIALIZATION_DIR / "run_manifest.csv",
        MATERIALIZATION_DIR / "data" / "targets.jsonl",
        DIRECTIONAL_DESIGN_DIR / "selected_pairs.csv",
        DIRECTIONAL_DESIGN_DIR / "design_manifest.json",
    ]
    for row in read_csv(EXECUTION_DIR / "run_manifest.csv"):
        paths.append(
            RUN_ROOT / row["run_id"] / "eval_predictions.npz"
        )
    return sorted(set(paths), key=relative)


def _implementation_hashes() -> dict[str, str]:
    missing = [
        name
        for name in IMPLEMENTATION_PATHS
        if not (PROJECT_ROOT / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing V4 effect code: {missing}")
    return {
        name: sha256_file(PROJECT_ROOT / name)
        for name in IMPLEMENTATION_PATHS
    }


def _verify_prediction_hashes() -> None:
    artifacts = read_json(EXECUTION_DIR / "run_artifact_manifest.json")
    records = artifacts.get("runs")
    if not isinstance(records, list) or len(records) != 32:
        raise RuntimeError("V4 run-artifact manifest is incomplete.")
    for record in records:
        identifier = record["run_id"]
        expected = record["artifacts_sha256"]["eval_predictions.npz"]
        path = RUN_ROOT / identifier / "eval_predictions.npz"
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"V4 prediction hash mismatch: {identifier}")


def verify_protocol_bundle() -> None:
    for expected, path in _checksum_entries(PROTOCOL_DIR / "SHA256SUMS"):
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"V4 effect protocol checksum mismatch: {path}")
    report = read_json(PROTOCOL_DIR / "validation_report.json")
    if (
        report.get("status") != "pass"
        or report.get("stage") != "pre_unblinding"
        or report.get("failed_count") != 0
    ):
        raise RuntimeError("V4 pre-unblinding validation did not pass.")


def verify_frozen_inputs() -> None:
    manifest = read_json(PROTOCOL_DIR / "input_manifest.json")
    for name, expected in manifest["input_sha256"].items():
        path = PROJECT_ROOT / name
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Frozen V4 effect input changed: {name}")
    for name, expected in manifest["implementation_sha256"].items():
        path = PROJECT_ROOT / name
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Frozen V4 effect code changed: {name}")


def validate_protocol(*, require_outcomes_absent: bool) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: Any = None) -> None:
        checks.append({"detail": detail, "name": name, "pass": bool(condition)})

    prereg = read_json(PROTOCOL_DIR / "preregistration.json")
    execution = read_json(EXECUTION_DIR / "execution_validation_report.json")
    check(
        "execution_gate",
        execution.get("status") == "pass"
        and execution.get("execution_gate") == "pass"
        and execution.get("failed_count") == 0,
    )
    check(
        "execution_blinded",
        execution.get("primary_effect_computed") is False,
    )
    check(
        "estimand",
        prereg["estimand"]["seed_count"] == 8
        and prereg["estimand"]["pair_count"] == 40
        and prereg["estimand"]["pattern_count"] == 80
        and prereg["estimand"]["orientation"]
        == "exposed_minus_unexposed",
    )
    bootstrap = prereg["bootstrap"]
    check(
        "bootstrap",
        bootstrap["replicates"] == BOOTSTRAP_REPLICATES
        and bootstrap["seed"] == BOOTSTRAP_SEED
        and bootstrap["model_seed_draws_per_replicate"] == 8
        and bootstrap["pattern_pair_draws_per_replicate"] == 40
        and bootstrap["source_group_draws_per_pattern"] == 34,
    )
    check(
        "directional_gate",
        prereg["decision_gate"]["sesoi_pp"] == SESOI_PP
        and prereg["decision_gate"]["positive_rule"]
        == "point_estimate_pp >= 1.0 and 95_percent_interval_lower_pp > 0",
    )
    manifest_path = PROTOCOL_DIR / "input_manifest.json"
    receipt_path = PROTOCOL_DIR / "freeze_receipt.json"
    check(
        "freeze_artifacts",
        manifest_path.is_file() and receipt_path.is_file(),
    )
    input_failures: list[str] = []
    code_failures: list[str] = []
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        for name, expected in manifest["input_sha256"].items():
            path = PROJECT_ROOT / name
            if not path.is_file() or sha256_file(path) != expected:
                input_failures.append(name)
        for name, expected in manifest["implementation_sha256"].items():
            path = PROJECT_ROOT / name
            if not path.is_file() or sha256_file(path) != expected:
                code_failures.append(name)
    check("input_hashes", not input_failures, input_failures)
    check("implementation_hashes", not code_failures, code_failures)
    artifacts = (
        list(ANALYSIS_DIR.rglob("*")) if ANALYSIS_DIR.exists() else []
    )
    if require_outcomes_absent:
        check(
            "outcomes_absent",
            not artifacts,
            [relative(path) for path in artifacts],
        )
    failed = [item for item in checks if not item["pass"]]
    document_names = (
        "README.md",
        "checklist.md",
        "freeze_receipt.json",
        "input_manifest.json",
        "preregistration.json",
        "validate.py",
    )
    return {
        "analysis_id": ANALYSIS_ID,
        "check_count": len(checks),
        "checks": checks,
        "document_sha256": {
            name: sha256_file(PROTOCOL_DIR / name)
            for name in document_names
        },
        "failed_count": len(failed),
        "schema_version": (
            "external-behavior-v4-effect-protocol-validation-v1"
        ),
        "stage": (
            "pre_unblinding"
            if require_outcomes_absent
            else "post_unblinding_recheck"
        ),
        "status": "pass" if not failed else "fail",
        "validated_at_utc": _now(),
    }


def freeze_protocol() -> dict[str, Any]:
    if (PROTOCOL_DIR / "input_manifest.json").exists():
        raise RuntimeError("V4 effect-analysis protocol is already frozen.")
    if ANALYSIS_DIR.exists() and any(ANALYSIS_DIR.rglob("*")):
        raise RuntimeError("V4 effect outputs exist before protocol freeze.")
    execution = read_json(EXECUTION_DIR / "execution_validation_report.json")
    if (
        execution.get("status") != "pass"
        or execution.get("failed_count") != 0
        or execution.get("primary_effect_computed") is not False
    ):
        raise RuntimeError("V4 execution is not eligible for unblinding.")
    _verify_prediction_hashes()
    paths = _input_paths()
    missing = [relative(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing V4 analysis inputs: {missing}")
    manifest = {
        "analysis_id": ANALYSIS_ID,
        "frozen_at_utc": _now(),
        "implementation_sha256": _implementation_hashes(),
        "input_sha256": {
            relative(path): sha256_file(path) for path in paths
        },
        "prediction_file_count": 32,
        "schema_version": (
            "external-behavior-v4-effect-input-manifest-v1"
        ),
        "status": "frozen_before_outcome_contrast_computation",
    }
    write_json(PROTOCOL_DIR / "input_manifest.json", manifest)
    receipt = {
        "analysis_id": ANALYSIS_ID,
        "frozen_at_utc": _now(),
        "schema_version": (
            "external-behavior-v4-effect-freeze-receipt-v1"
        ),
        "state_at_freeze": {
            "analysis_output_directory_absent": not ANALYSIS_DIR.exists(),
            "execution_gate_passed": True,
            "outcome_contrast_computed": False,
            "outcome_values_loaded_by_freeze_process": False,
            "prediction_files_accessed_for_sha256_only": True,
        },
        "status": "frozen_before_unblinding",
    }
    write_json(PROTOCOL_DIR / "freeze_receipt.json", receipt)
    report = validate_protocol(require_outcomes_absent=True)
    if report["status"] != "pass":
        raise RuntimeError(f"V4 pre-unblinding validation failed: {report}")
    write_json(PROTOCOL_DIR / "validation_report.json", report)
    names = (
        "README.md",
        "checklist.md",
        "freeze_receipt.json",
        "input_manifest.json",
        "preregistration.json",
        "validate.py",
        "validation_report.json",
    )
    (PROTOCOL_DIR / "SHA256SUMS").write_text(
        "".join(
            f"{sha256_file(PROTOCOL_DIR / name)}  {name}\n"
            for name in names
        ),
        encoding="ascii",
    )
    return {
        "analysis_id": ANALYSIS_ID,
        "input_manifest_sha256": sha256_file(
            PROTOCOL_DIR / "input_manifest.json"
        ),
        "status": receipt["status"],
        "validation_status": report["status"],
    }


def patterns() -> tuple[list[dict[str, Any]], np.ndarray]:
    rows = read_csv(DIRECTIONAL_DESIGN_DIR / "selected_pairs.csv")
    pairs = sorted(rows, key=lambda row: int(row["global_v4_pair_index"]))
    if len(pairs) != PAIR_COUNT:
        raise RuntimeError("Frozen V4 pair count changed.")
    records: list[dict[str, Any]] = []
    pair_pattern_indices = np.empty((PAIR_COUNT, 2), dtype=np.int16)
    for pair in pairs:
        global_pair = int(pair["global_v4_pair_index"])
        for member in (0, 1):
            index = len(records)
            pair_pattern_indices[global_pair, member] = index
            records.append(
                {
                    "block_index": int(pair["block_index"]),
                    "block_pair_index": int(pair["block_pair_index"]),
                    "global_pair_index": global_pair,
                    "pair_member": member,
                    "pattern_index": index,
                    "semantic_id": pair[f"member_{member}"],
                    "target_relation": pair["target_relation"],
                }
            )
    if [row["pattern_index"] for row in records] != list(range(80)):
        raise RuntimeError("Frozen V4 pattern order changed.")
    return records, pair_pattern_indices


def exposed_arms(pattern_records: list[dict[str, Any]]) -> np.ndarray:
    rows = read_csv(MATERIALIZATION_DIR / "run_manifest.csv")
    by_key = {
        (
            int(row["block_index"]),
            int(row["model_seed"]),
            row["arm"],
        ): json.loads(row["exposed_pair_members"])
        for row in rows
    }
    result = np.empty((8, 80), dtype=np.int8)
    for seed in MODEL_SEEDS:
        for pattern in pattern_records:
            block = pattern["block_index"]
            pair = str(pattern["block_pair_index"])
            member = pattern["pair_member"]
            matches = [
                arm_index
                for arm_index, arm in enumerate(ARMS)
                if int(by_key[(block, seed, arm)][pair]) == member
            ]
            if len(matches) != 1:
                raise RuntimeError("V4 exposure alignment is not unique.")
            result[seed, pattern["pattern_index"]] = matches[0]
    return result


def evaluation_contexts(
    pattern_records: list[dict[str, Any]],
) -> tuple[
    dict[int, dict[str, np.ndarray]],
    list[tuple[int, np.ndarray]],
    list[list[np.ndarray]],
    np.ndarray,
]:
    evaluations = {
        block: load_npz(EXECUTION_DIR / f"evaluation_block{block}.npz")
        for block in BLOCKS
    }
    pattern_rows: list[tuple[int, np.ndarray]] = [
        (0, np.asarray([], dtype=np.int64)) for _ in range(80)
    ]
    groups: list[list[np.ndarray]] = [[] for _ in range(80)]
    group_sizes = np.zeros((80, 34), dtype=np.int16)
    coverage = {
        block: np.zeros(EVAL_ROWS_PER_BLOCK, dtype=np.int16)
        for block in BLOCKS
    }
    for pattern in pattern_records:
        block = pattern["block_index"]
        evaluation = evaluations[block]
        selected = (
            evaluation["global_pair_index"].astype(np.int64)
            == pattern["global_pair_index"]
        ) & (
            evaluation["pair_member"].astype(np.int64)
            == pattern["pair_member"]
        )
        rows = np.flatnonzero(selected)
        semantics = evaluation["semantic_ids"][rows].astype("U64")
        if (
            len(rows) != ROWS_PER_PATTERN
            or set(semantics) != {pattern["semantic_id"]}
        ):
            raise RuntimeError("Frozen V4 evaluation pattern changed.")
        source_ids = evaluation["source_group_ids"].astype("U64")
        group_ids = sorted(set(source_ids[rows]))
        if len(group_ids) != GROUPS_PER_PATTERN:
            raise RuntimeError("Frozen V4 source-group count changed.")
        pattern_groups: list[np.ndarray] = []
        for group_index, group_id in enumerate(group_ids):
            group_rows = rows[source_ids[rows] == group_id]
            if len(group_rows) not in (2, 6):
                raise RuntimeError("Unexpected V4 source-group size.")
            pattern_groups.append(group_rows)
            group_sizes[pattern["pattern_index"], group_index] = len(
                group_rows
            )
        if sorted(
            group_sizes[pattern["pattern_index"]].tolist()
        ) != [2] + [6] * 33:
            raise RuntimeError("Frozen V4 200-row allocation changed.")
        pattern_rows[pattern["pattern_index"]] = (block, rows)
        groups[pattern["pattern_index"]] = pattern_groups
        coverage[block][rows] += 1
    if any(
        not np.array_equal(
            coverage[block],
            np.ones(EVAL_ROWS_PER_BLOCK, dtype=np.int16),
        )
        for block in BLOCKS
    ):
        raise RuntimeError("V4 evaluation patterns do not partition each block.")
    return evaluations, pattern_rows, groups, group_sizes


def load_predictions(
    evaluations: dict[int, dict[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    logits = np.empty((8, 2, 2, 8000, 23), dtype=np.float32)
    correct = np.empty((8, 2, 2, 8000), dtype=np.bool_)
    for seed in MODEL_SEEDS:
        for block in BLOCKS:
            expected = evaluations[block]
            metadata = set(expected) - {"cache_indices"}
            for arm_index, arm in enumerate(ARMS):
                identifier = f"block{block}_seed{seed}_arm{arm}"
                prediction = load_npz(
                    RUN_ROOT / identifier / "eval_predictions.npz"
                )
                required = metadata | {"logits", "predictions"}
                if (
                    set(prediction) != required
                    or prediction["logits"].shape != (8000, 23)
                    or not np.isfinite(prediction["logits"]).all()
                    or any(
                        not np.array_equal(
                            prediction[name],
                            expected[name],
                        )
                        for name in metadata
                    )
                    or not np.array_equal(
                        prediction["predictions"],
                        prediction["logits"].argmax(axis=1),
                    )
                ):
                    raise RuntimeError(
                        f"Invalid V4 prediction artifact: {identifier}"
                    )
                logits[seed, block, arm_index] = prediction["logits"]
                correct[seed, block, arm_index] = (
                    prediction["predictions"].astype(np.int64)
                    == prediction["labels"].astype(np.int64)
                )
    return logits, correct


def effect_tables(
    pattern_records: list[dict[str, Any]],
    pair_pattern_indices: np.ndarray,
    pattern_rows: list[tuple[int, np.ndarray]],
    groups: list[list[np.ndarray]],
    group_sizes: np.ndarray,
    exposure: np.ndarray,
    correct: np.ndarray,
) -> tuple[dict[str, list[dict[str, Any]]], np.ndarray, np.ndarray]:
    seed_pattern_rows: list[dict[str, Any]] = []
    group_differences = np.zeros((8, 80, 34), dtype=np.int16)
    point_matrix = np.zeros((8, 80), dtype=np.float64)
    for seed in MODEL_SEEDS:
        for pattern in pattern_records:
            index = pattern["pattern_index"]
            block, rows = pattern_rows[index]
            exposed_arm = int(exposure[seed, index])
            unexposed_arm = 1 - exposed_arm
            exposed_correct = int(
                correct[seed, block, exposed_arm, rows].sum()
            )
            unexposed_correct = int(
                correct[seed, block, unexposed_arm, rows].sum()
            )
            effect_pp = (
                100.0
                * (exposed_correct - unexposed_correct)
                / ROWS_PER_PATTERN
            )
            point_matrix[seed, index] = effect_pp
            for group_index, group_rows in enumerate(groups[index]):
                group_differences[seed, index, group_index] = int(
                    correct[
                        seed,
                        block,
                        exposed_arm,
                        group_rows,
                    ].sum()
                    - correct[
                        seed,
                        block,
                        unexposed_arm,
                        group_rows,
                    ].sum()
                )
            seed_pattern_rows.append(
                {
                    "block_index": block,
                    "effect_pp": effect_pp,
                    "evaluation_rows": len(rows),
                    "exposed_accuracy": exposed_correct / len(rows),
                    "exposed_arm": ARMS[exposed_arm],
                    "exposed_correct": exposed_correct,
                    "global_pair_index": pattern["global_pair_index"],
                    "model_seed": seed,
                    "pair_member": pattern["pair_member"],
                    "pattern_index": index,
                    "semantic_id": pattern["semantic_id"],
                    "target_relation": pattern["target_relation"],
                    "unexposed_accuracy": unexposed_correct / len(rows),
                    "unexposed_arm": ARMS[unexposed_arm],
                    "unexposed_correct": unexposed_correct,
                }
            )
    pattern_table = [
        {
            "block_index": pattern["block_index"],
            "effect_pp": float(
                point_matrix[:, pattern["pattern_index"]].mean()
            ),
            "global_pair_index": pattern["global_pair_index"],
            "pair_member": pattern["pair_member"],
            "pattern_index": pattern["pattern_index"],
            "semantic_id": pattern["semantic_id"],
            "target_relation": pattern["target_relation"],
        }
        for pattern in pattern_records
    ]
    seed_table = [
        {
            "effect_pp": float(point_matrix[seed].mean()),
            "model_seed": seed,
        }
        for seed in MODEL_SEEDS
    ]
    block_table = [
        {
            "block_index": block,
            "effect_pp": float(
                point_matrix[
                    :,
                    [
                        row["pattern_index"]
                        for row in pattern_records
                        if row["block_index"] == block
                    ],
                ].mean()
            ),
            "pattern_count": 40,
        }
        for block in BLOCKS
    ]
    pair_table = [
        {
            "block_index": pattern_records[
                int(pair_pattern_indices[pair, 0])
            ]["block_index"],
            "effect_pp": float(
                point_matrix[:, pair_pattern_indices[pair]].mean()
            ),
            "global_pair_index": pair,
            "member_0_pattern_index": int(
                pair_pattern_indices[pair, 0]
            ),
            "member_1_pattern_index": int(
                pair_pattern_indices[pair, 1]
            ),
            "target_relation": pattern_records[
                int(pair_pattern_indices[pair, 0])
            ]["target_relation"],
        }
        for pair in range(PAIR_COUNT)
    ]
    relation_indices: dict[str, list[int]] = defaultdict(list)
    for pattern in pattern_records:
        relation_indices[pattern["target_relation"]].append(
            pattern["pattern_index"]
        )
    relation_table = [
        {
            "effect_pp": float(point_matrix[:, indices].mean()),
            "pattern_count": len(indices),
            "target_relation": relation,
        }
        for relation, indices in sorted(relation_indices.items())
    ]
    return (
        {
            "block": block_table,
            "pair": pair_table,
            "pattern": pattern_table,
            "relation": relation_table,
            "seed": seed_table,
            "seed_pattern": seed_pattern_rows,
        },
        group_differences,
        point_matrix,
    )


def hierarchical_bootstrap(
    group_differences: np.ndarray,
    group_sizes: np.ndarray,
    pair_pattern_indices: np.ndarray,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
    batch_size: int = BOOTSTRAP_BATCH_SIZE,
) -> tuple[np.ndarray, dict[str, str]]:
    differences = np.asarray(group_differences, dtype=np.float64)
    sizes = np.asarray(group_sizes, dtype=np.float64)
    pairs = np.asarray(pair_pattern_indices, dtype=np.int16)
    if (
        differences.shape != (8, 80, 34)
        or sizes.shape != (80, 34)
        or pairs.shape != (40, 2)
    ):
        raise ValueError("Invalid V4 bootstrap tensor shape.")
    rng = np.random.Generator(np.random.PCG64(seed))
    seed_draws = rng.integers(
        0,
        8,
        size=(replicates, 8),
        dtype=np.int16,
    )
    pair_draws = rng.integers(
        0,
        40,
        size=(replicates, 40),
        dtype=np.int16,
    )
    group_draws = rng.integers(
        0,
        34,
        size=(replicates, 40, 2, 34),
        dtype=np.int16,
    )
    effects = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        selected_patterns = pairs[pair_draws[start:stop]].reshape(
            stop - start,
            80,
        )
        selected_groups = group_draws[start:stop].reshape(
            stop - start,
            80,
            34,
        )
        denominators = sizes[
            selected_patterns[:, :, None],
            selected_groups,
        ].sum(axis=2)
        real_seed_effects = np.empty(
            (stop - start, 8, 80),
            dtype=np.float64,
        )
        for model_seed in MODEL_SEEDS:
            numerators = differences[model_seed][
                selected_patterns[:, :, None],
                selected_groups,
            ].sum(axis=2)
            real_seed_effects[:, model_seed] = (
                numerators / denominators
            )
        selected_seed_effects = np.take_along_axis(
            real_seed_effects,
            seed_draws[start:stop, :, None],
            axis=1,
        )
        effects[start:stop] = (
            100.0 * selected_seed_effects.mean(axis=(1, 2))
        )
    return effects, {
        "group_draws_sha256": array_sha256(group_draws),
        "pair_draws_sha256": array_sha256(pair_draws),
        "seed_draws_sha256": array_sha256(seed_draws),
    }


def classify_gate(
    point_pp: float,
    lower_pp: float,
    upper_pp: float,
) -> dict[str, Any]:
    passed = point_pp >= SESOI_PP and lower_pp > 0.0
    return {
        "classification": (
            "directional_positive_go"
            if passed
            else "directional_no_go"
        ),
        "interval_width_pp": upper_pp - lower_pp,
        "positive_gate_pass": passed,
    }


def aligned_metrics(
    logits: np.ndarray,
    evaluations: dict[int, dict[str, np.ndarray]],
    pattern_records: list[dict[str, Any]],
    pattern_rows: list[tuple[int, np.ndarray]],
    exposure: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    metric_names = (
        "accuracy",
        "brier_score",
        "cross_entropy",
        "ece_15",
        "mean_confidence",
        "signed_calibration_gap",
    )
    rows: list[dict[str, Any]] = []
    differences: dict[str, list[float]] = {
        name: [] for name in metric_names
    }
    labels = np.concatenate(
        [evaluations[block]["labels"] for block in BLOCKS]
    ).astype(np.int64)
    for seed in MODEL_SEEDS:
        exposed_logits = np.empty((16_000, 23), dtype=np.float32)
        unexposed_logits = np.empty((16_000, 23), dtype=np.float32)
        for pattern in pattern_records:
            index = pattern["pattern_index"]
            block, selected = pattern_rows[index]
            offset = block * EVAL_ROWS_PER_BLOCK
            output_rows = offset + selected
            arm = int(exposure[seed, index])
            exposed_logits[output_rows] = logits[
                seed,
                block,
                arm,
                selected,
            ]
            unexposed_logits[output_rows] = logits[
                seed,
                block,
                1 - arm,
                selected,
            ]
        exposed = classification_metrics(exposed_logits, labels)
        unexposed = classification_metrics(unexposed_logits, labels)
        row: dict[str, Any] = {"model_seed": seed}
        for name in metric_names:
            difference = exposed[name] - unexposed[name]
            differences[name].append(difference)
            row[f"exposed_{name}"] = exposed[name]
            row[f"unexposed_{name}"] = unexposed[name]
            row[f"difference_{name}"] = difference
        rows.append(row)
    aggregate = {
        f"mean_exposed_minus_unexposed_{name}": float(
            np.mean(differences[name])
        )
        for name in metric_names
    }
    return rows, aggregate


def _report(results: dict[str, Any]) -> str:
    primary = results["primary"]
    interval = primary["confidence_interval_95"]
    decision = results["decision"]
    secondary = results["secondary"]
    return "\n".join(
        [
            "# Directional V4 暴露效应解盲报告",
            "",
            "## 主结果",
            "",
            f"- 暴露减未暴露准确率效应：{primary['effect_pp']:.4f} pp。",
            (
                f"- 95% 层级 bootstrap 区间："
                f"[{interval['lower_pp']:.4f}, "
                f"{interval['upper_pp']:.4f}] pp。"
            ),
            f"- 预注册判定：`{decision['classification']}`。",
            "",
            "## 描述性结果",
            "",
            (
                "- Cross-entropy 暴露减未暴露："
                f"{secondary['mean_exposed_minus_unexposed_cross_entropy']:.6f}。"
            ),
            (
                "- Brier 暴露减未暴露："
                f"{secondary['mean_exposed_minus_unexposed_brier_score']:.6f}。"
            ),
            (
                "- ECE-15 暴露减未暴露："
                f"{secondary['mean_exposed_minus_unexposed_ece_15']:.6f}。"
            ),
            "",
            (
                "Gate 仅使用冻结的 accuracy estimand 与 directional rule。"
                "block、seed、pattern、relation 和 calibration 结果均为描述性。"
            ),
            "",
        ]
    )


def run_analysis() -> dict[str, Any]:
    verify_protocol_bundle()
    verify_frozen_inputs()
    if ANALYSIS_DIR.exists():
        raise RuntimeError("V4 effect output directory already exists.")
    pattern_records, pair_indices = patterns()
    exposure = exposed_arms(pattern_records)
    (
        evaluations,
        pattern_rows,
        groups,
        group_sizes,
    ) = evaluation_contexts(pattern_records)
    logits, correct = load_predictions(evaluations)
    tables, group_differences, point_matrix = effect_tables(
        pattern_records,
        pair_indices,
        pattern_rows,
        groups,
        group_sizes,
        exposure,
        correct,
    )
    point_pp = float(point_matrix.mean())
    bootstrap, draw_hashes = hierarchical_bootstrap(
        group_differences,
        group_sizes,
        pair_indices,
    )
    lower_pp, upper_pp = np.quantile(
        bootstrap,
        [0.025, 0.975],
        method="linear",
    )
    decision = classify_gate(
        point_pp,
        float(lower_pp),
        float(upper_pp),
    )
    metric_rows, secondary = aligned_metrics(
        logits,
        evaluations,
        pattern_records,
        pattern_rows,
        exposure,
    )
    if not math.isclose(
        100.0
        * secondary["mean_exposed_minus_unexposed_accuracy"],
        point_pp,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("V4 primary and aligned accuracy disagree.")
    ANALYSIS_DIR.mkdir(parents=True)
    write_csv(
        ANALYSIS_DIR / "seed_pattern_effects.csv",
        tables["seed_pattern"],
    )
    write_csv(ANALYSIS_DIR / "pattern_effects.csv", tables["pattern"])
    write_csv(ANALYSIS_DIR / "seed_effects.csv", tables["seed"])
    write_csv(ANALYSIS_DIR / "block_effects.csv", tables["block"])
    write_csv(ANALYSIS_DIR / "pair_effects.csv", tables["pair"])
    write_csv(
        ANALYSIS_DIR / "relation_effects.csv",
        tables["relation"],
    )
    write_csv(
        ANALYSIS_DIR / "seed_aligned_metrics.csv",
        metric_rows,
    )
    write_deterministic_npz(
        ANALYSIS_DIR / "bootstrap.npz",
        {"effects_pp": bootstrap},
    )
    results = {
        "analysis_id": ANALYSIS_ID,
        "analyzed_at_utc": _now(),
        "decision": decision,
        "integrity": {
            "blocks": 2,
            "evaluation_rows_per_run": 8_000,
            "model_seeds": 8,
            "pattern_pairs": 40,
            "patterns": 80,
            "prediction_files": 32,
            "source_groups_per_pattern": 34,
        },
        "primary": {
            "bootstrap": {
                "draw_sha256": draw_hashes,
                "effects_sha256": array_sha256(bootstrap),
                "replicates": len(bootstrap),
                "seed": BOOTSTRAP_SEED,
            },
            "confidence_interval_95": {
                "lower_pp": float(lower_pp),
                "upper_pp": float(upper_pp),
                "width_pp": float(upper_pp - lower_pp),
            },
            "effect_pp": point_pp,
            "estimand": (
                "equal-seed equal-pattern row-weighted "
                "exposed-minus-unexposed accuracy"
            ),
        },
        "protocol_input_manifest_sha256": sha256_file(
            PROTOCOL_DIR / "input_manifest.json"
        ),
        "schema_version": (
            "external-behavior-v4-effect-analysis-results-v1"
        ),
        "secondary": secondary,
        "status": "unblinded_analysis_complete",
    }
    write_json(ANALYSIS_DIR / "results.json", results)
    (ANALYSIS_DIR / "report_cn.md").write_text(
        _report(results),
        encoding="utf-8",
    )
    output_hashes = {
        name: sha256_file(ANALYSIS_DIR / name)
        for name in OUTPUT_NAMES
    }
    write_json(
        ANALYSIS_DIR / "analysis_manifest.json",
        {
            "analysis_id": ANALYSIS_ID,
            "completed_at_utc": _now(),
            "output_sha256": output_hashes,
            "protocol_input_manifest_sha256": sha256_file(
                PROTOCOL_DIR / "input_manifest.json"
            ),
            "schema_version": (
                "external-behavior-v4-effect-analysis-manifest-v1"
            ),
            "status": "analysis_complete_pending_validation",
        },
    )
    return results


def direct_primary_recompute() -> float:
    pattern_records, _ = patterns()
    exposure = exposed_arms(pattern_records)
    evaluations, pattern_rows, _, _ = evaluation_contexts(pattern_records)
    _, correct = load_predictions(evaluations)
    effects: list[float] = []
    for seed in MODEL_SEEDS:
        for pattern in pattern_records:
            index = pattern["pattern_index"]
            block, rows = pattern_rows[index]
            arm = int(exposure[seed, index])
            effects.append(
                100.0
                * (
                    float(correct[seed, block, arm, rows].mean())
                    - float(
                        correct[seed, block, 1 - arm, rows].mean()
                    )
                )
            )
    return float(np.mean(effects))


def _validation_checklist(checks: list[dict[str, Any]]) -> str:
    status = {item["name"]: item["pass"] for item in checks}
    entries = (
        ("protocol", "Validate the frozen pre-unblinding protocol."),
        ("inputs", "Validate every input and implementation hash."),
        ("outputs", "Validate all declared analysis output hashes."),
        ("primary", "Independently recompute the primary point estimate."),
        ("bootstrap", "Reproduce all 20,000 bootstrap draws."),
        ("interval", "Recompute the registered percentile interval."),
        ("decision", "Reapply the frozen directional Gate."),
        ("tables", "Validate all registered table dimensions."),
        ("finite", "Confirm finite numeric outputs."),
    )
    lines = ["# Directional V4 Effect Analysis Integrity", ""]
    for name, label in entries:
        lines.append(f"- [{'x' if status.get(name) else ' '}] {label}")
    lines.append("")
    return "\n".join(lines)


def validate_analysis() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: Any = None) -> None:
        checks.append({"detail": detail, "name": name, "pass": bool(condition)})

    try:
        verify_protocol_bundle()
        check("protocol", True)
    except Exception as error:
        check("protocol", False, repr(error))
    try:
        verify_frozen_inputs()
        check("inputs", True)
    except Exception as error:
        check("inputs", False, repr(error))
    manifest = read_json(ANALYSIS_DIR / "analysis_manifest.json")
    output_failures = [
        name
        for name, expected in manifest["output_sha256"].items()
        if not (ANALYSIS_DIR / name).is_file()
        or sha256_file(ANALYSIS_DIR / name) != expected
    ]
    check("outputs", not output_failures, output_failures)
    results = read_json(ANALYSIS_DIR / "results.json")
    direct = direct_primary_recompute()
    reported = float(results["primary"]["effect_pp"])
    check(
        "primary",
        math.isclose(direct, reported, rel_tol=0.0, abs_tol=1e-12),
        {"direct": direct, "reported": reported},
    )
    pattern_records, pair_indices = patterns()
    exposure = exposed_arms(pattern_records)
    evaluations, pattern_rows, groups, group_sizes = evaluation_contexts(
        pattern_records
    )
    _, correct = load_predictions(evaluations)
    _, group_differences, _ = effect_tables(
        pattern_records,
        pair_indices,
        pattern_rows,
        groups,
        group_sizes,
        exposure,
        correct,
    )
    reproduced, draw_hashes = hierarchical_bootstrap(
        group_differences,
        group_sizes,
        pair_indices,
    )
    with np.load(
        ANALYSIS_DIR / "bootstrap.npz",
        allow_pickle=False,
    ) as archive:
        stored = archive["effects_pp"]
    expected_bootstrap = results["primary"]["bootstrap"]
    check(
        "bootstrap",
        np.array_equal(reproduced, stored)
        and array_sha256(stored) == expected_bootstrap["effects_sha256"]
        and draw_hashes == expected_bootstrap["draw_sha256"],
    )
    lower_pp, upper_pp = np.quantile(
        stored,
        [0.025, 0.975],
        method="linear",
    )
    interval = results["primary"]["confidence_interval_95"]
    check(
        "interval",
        all(
            math.isclose(float(observed), float(expected), abs_tol=1e-12)
            for observed, expected in (
                (lower_pp, interval["lower_pp"]),
                (upper_pp, interval["upper_pp"]),
                (upper_pp - lower_pp, interval["width_pp"]),
            )
        ),
    )
    reproduced_decision = classify_gate(
        reported,
        float(lower_pp),
        float(upper_pp),
    )
    check(
        "decision",
        reproduced_decision == results["decision"],
        {
            "reported": results["decision"],
            "reproduced": reproduced_decision,
        },
    )
    expected_counts = {
        "block_effects.csv": 2,
        "pair_effects.csv": 40,
        "pattern_effects.csv": 80,
        "relation_effects.csv": len(
            {row["target_relation"] for row in pattern_records}
        ),
        "seed_aligned_metrics.csv": 8,
        "seed_effects.csv": 8,
        "seed_pattern_effects.csv": 640,
    }
    observed_counts = {
        name: len(read_csv(ANALYSIS_DIR / name))
        for name in expected_counts
    }
    check(
        "tables",
        expected_counts == observed_counts,
        {"expected": expected_counts, "observed": observed_counts},
    )

    def finite_tree(value: Any) -> bool:
        if isinstance(value, dict):
            return all(finite_tree(item) for item in value.values())
        if isinstance(value, list):
            return all(finite_tree(item) for item in value)
        if isinstance(value, float):
            return math.isfinite(value)
        return True

    check(
        "finite",
        finite_tree(results) and np.isfinite(stored).all(),
    )
    failed = [item for item in checks if not item["pass"]]
    report = {
        "analysis_gate": "pass" if not failed else "no_go",
        "analysis_id": ANALYSIS_ID,
        "check_count": len(checks),
        "checks": checks,
        "failed_checks": [item["name"] for item in failed],
        "failed_count": len(failed),
        "primary_decision": results["decision"]["classification"],
        "schema_version": (
            "external-behavior-v4-effect-analysis-validation-v1"
        ),
        "status": "pass" if not failed else "no_go",
        "validated_at_utc": _now(),
    }
    write_json(ANALYSIS_DIR / "analysis_validation_report.json", report)
    (ANALYSIS_DIR / "analysis_checklist.md").write_text(
        _validation_checklist(checks),
        encoding="ascii",
    )
    names = (
        *OUTPUT_NAMES,
        "analysis_manifest.json",
        "analysis_checklist.md",
        "analysis_validation_report.json",
    )
    (ANALYSIS_DIR / "ANALYSIS_SHA256SUMS").write_text(
        "".join(
            f"{sha256_file(ANALYSIS_DIR / name)}  {name}\n"
            for name in names
        ),
        encoding="ascii",
    )
    return report
