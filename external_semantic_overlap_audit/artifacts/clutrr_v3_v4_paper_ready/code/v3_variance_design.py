"""Exploratory V3 variance decomposition and prospective V4 design gate."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .effect_analysis import (
    ANALYSIS_DIR as EFFECT_ANALYSIS_DIR,
    ARMS,
    BOOTSTRAP_REPLICATES,
    DATA_DIR,
    EVAL_ROWS,
    EXECUTION_DIR,
    MODEL_SEEDS,
    PROJECT_ROOT,
    PROTOCOL_DIR as EFFECT_PROTOCOL_DIR,
    RUN_ROOT,
    _effect_tables,
    _evaluation_groups,
    _exposed_arms,
    _load_predictions,
    _patterns,
    array_sha256,
    iter_jsonl,
    load_npz,
    read_csv,
    read_json,
    relative,
    sha256_file,
    verify_frozen_inputs,
    verify_protocol_bundle,
    write_csv,
    write_deterministic_npz,
    write_json,
)


DESIGN_ID = "external-behavior-v3-variance-design-v1"
PROTOCOL_DIR = (
    PROJECT_ROOT
    / "protocols"
    / "external-behavior-v3"
    / "variance-design-v1"
)
ANALYSIS_DIR = (
    PROJECT_ROOT
    / "analysis"
    / "external-behavior-v3"
    / "variance-design-v1"
)
DECOMPOSITION_REPLICATES = 20_000
DECOMPOSITION_SEED_BASE = 202_610_130
PROJECTION_SEED_COUNTS = (8, 16, 32, 64)
PROJECTION_PAIR_COUNTS = (20, 40, 80, 120)
TARGET_INTERVAL_WIDTH_PP = 2.0
Z_975 = 1.959963984540054

FACTOR_BITS = {
    "model_seed": 1,
    "pattern_pair": 2,
    "source_group": 4,
}

IMPLEMENTATION_PATHS = (
    "scripts/freeze_external_behavior_v3_variance_design.py",
    "scripts/analyze_external_behavior_v3_variance_design.py",
    "scripts/validate_external_behavior_v3_variance_design.py",
    "tests/test_external_behavior_v3_variance_design.py",
    "toolkit/external_behavior_v3/effect_analysis.py",
    "toolkit/external_behavior_v3/variance_design.py",
    "protocols/external-behavior-v3/variance-design-v1/validate.py",
)

OUTPUT_NAMES = (
    "anova_components.csv",
    "bootstrap_component_summary.csv",
    "decomposition_distributions.npz",
    "precision_projection.csv",
    "remaining_pair_capacity.csv",
    "report_cn.md",
    "v4_design_gate_report.json",
    "variance_results.json",
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
        EFFECT_PROTOCOL_DIR / "SHA256SUMS",
        EFFECT_PROTOCOL_DIR / "input_manifest.json",
        EFFECT_PROTOCOL_DIR / "preregistration.json",
        EFFECT_PROTOCOL_DIR / "validation_report.json",
        EFFECT_ANALYSIS_DIR / "ANALYSIS_SHA256SUMS",
        EFFECT_ANALYSIS_DIR / "analysis_validation_report.json",
        EFFECT_ANALYSIS_DIR / "bootstrap.npz",
        EFFECT_ANALYSIS_DIR / "results.json",
        EFFECT_ANALYSIS_DIR / "seed_pattern_effects.csv",
        DATA_DIR / "clutrr_v3_eligible_patterns.jsonl",
        DATA_DIR / "clutrr_v3_pairs.jsonl",
        DATA_DIR / "clutrr_v3_evaluation.jsonl",
        DESIGN_DIR / "design_validation_report.json",
        EXECUTION_DIR / "evaluation_manifest.npz",
        EXECUTION_DIR / "run_artifact_manifest.json",
    ]
    return sorted(paths, key=relative)


DESIGN_DIR = PROJECT_ROOT / "manifests" / "external-behavior-v3"


def _implementation_hashes() -> dict[str, str]:
    missing = [
        name
        for name in IMPLEMENTATION_PATHS
        if not (PROJECT_ROOT / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing variance implementation: {missing}")
    return {
        name: sha256_file(PROJECT_ROOT / name)
        for name in IMPLEMENTATION_PATHS
    }


def verify_variance_protocol() -> None:
    for expected, path in _checksum_entries(PROTOCOL_DIR / "SHA256SUMS"):
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Variance protocol checksum mismatch: {path}")
    report = read_json(PROTOCOL_DIR / "validation_report.json")
    if (
        report.get("status") != "pass"
        or report.get("stage") != "pre_decomposition"
        or report.get("failed_count") != 0
    ):
        raise RuntimeError("Variance protocol validation did not pass.")


def verify_variance_inputs() -> None:
    manifest = read_json(PROTOCOL_DIR / "input_manifest.json")
    for name, expected in manifest["input_sha256"].items():
        path = PROJECT_ROOT / name
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Frozen variance input changed: {name}")
    for name, expected in manifest["implementation_sha256"].items():
        path = PROJECT_ROOT / name
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Frozen variance code changed: {name}")


def validate_protocol(*, require_outputs_absent: bool) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: Any = None) -> None:
        checks.append({"detail": detail, "name": name, "pass": bool(condition)})

    prereg = read_json(PROTOCOL_DIR / "preregistration.json")
    effect_validation = read_json(
        EFFECT_ANALYSIS_DIR / "analysis_validation_report.json"
    )
    effect_results = read_json(EFFECT_ANALYSIS_DIR / "results.json")
    check(
        "effect_analysis_integrity",
        effect_validation.get("status") == "pass"
        and effect_validation.get("failed_count") == 0,
    )
    check(
        "post_unblinding_status",
        prereg["epistemic_status"]
        == "post_unblinding_exploratory_design_planning"
        and effect_results["decision"]["classification"]
        == "imprecise_no_go",
    )
    check(
        "decomposition",
        prereg["variance_decomposition"]["replicates"]
        == DECOMPOSITION_REPLICATES
        and prereg["variance_decomposition"]["factors"]
        == list(FACTOR_BITS),
    )
    check(
        "projection",
        prereg["prospective_projection"]["target_interval_width_pp"]
        == TARGET_INTERVAL_WIDTH_PP
        and prereg["prospective_projection"]["pair_unit"]
        == "same-label semantic-pattern pair",
    )
    check(
        "no_retroactive_change",
        prereg["constraints"]["v3_decision_remains"]
        == "imprecise_no_go"
        and prereg["constraints"]["v4_training_released"] is False,
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
    if require_outputs_absent:
        check(
            "outputs_absent",
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
        "check_count": len(checks),
        "checks": checks,
        "design_id": DESIGN_ID,
        "document_sha256": {
            name: sha256_file(PROTOCOL_DIR / name)
            for name in document_names
        },
        "failed_count": len(failed),
        "schema_version": (
            "external-behavior-v3-variance-protocol-validation-v1"
        ),
        "stage": (
            "pre_decomposition"
            if require_outputs_absent
            else "post_decomposition_recheck"
        ),
        "status": "pass" if not failed else "fail",
        "validated_at_utc": _now(),
    }


def freeze_protocol() -> dict[str, Any]:
    if (PROTOCOL_DIR / "input_manifest.json").exists():
        raise RuntimeError("Variance-design protocol is already frozen.")
    if ANALYSIS_DIR.exists() and any(ANALYSIS_DIR.rglob("*")):
        raise RuntimeError("Variance outputs exist before protocol freeze.")
    verify_protocol_bundle()
    verify_frozen_inputs()
    effect_validation = read_json(
        EFFECT_ANALYSIS_DIR / "analysis_validation_report.json"
    )
    if effect_validation.get("status") != "pass":
        raise RuntimeError("Effect analysis integrity did not pass.")
    paths = _input_paths()
    missing = [relative(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing variance inputs: {missing}")
    manifest = {
        "design_id": DESIGN_ID,
        "frozen_at_utc": _now(),
        "implementation_sha256": _implementation_hashes(),
        "input_sha256": {
            relative(path): sha256_file(path) for path in paths
        },
        "schema_version": (
            "external-behavior-v3-variance-input-manifest-v1"
        ),
        "status": "frozen_before_variance_decomposition",
    }
    write_json(PROTOCOL_DIR / "input_manifest.json", manifest)
    receipt = {
        "design_id": DESIGN_ID,
        "frozen_at_utc": _now(),
        "schema_version": (
            "external-behavior-v3-variance-freeze-receipt-v1"
        ),
        "state_at_freeze": {
            "effect_analysis_already_unblinded": True,
            "exploratory_status_declared": True,
            "variance_decomposition_computed": False,
            "variance_output_directory_absent": not ANALYSIS_DIR.exists(),
            "v3_decision_changed": False,
            "v4_training_released": False,
        },
        "status": "frozen_before_exploratory_decomposition",
    }
    write_json(PROTOCOL_DIR / "freeze_receipt.json", receipt)
    report = validate_protocol(require_outputs_absent=True)
    if report["status"] != "pass":
        raise RuntimeError(f"Variance protocol validation failed: {report}")
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
        "design_id": DESIGN_ID,
        "input_manifest_sha256": sha256_file(
            PROTOCOL_DIR / "input_manifest.json"
        ),
        "status": receipt["status"],
        "validation_status": report["status"],
    }


def build_outcome_tensors() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    evaluation = load_npz(EXECUTION_DIR / "evaluation_manifest.npz")
    patterns, pair_pattern_indices = _patterns()
    pattern_by_row, groups, group_sizes = _evaluation_groups(
        evaluation,
        patterns,
    )
    exposed_arms = _exposed_arms(patterns)
    _, correct = _load_predictions(evaluation)
    *_, group_differences = _effect_tables(
        patterns,
        pair_pattern_indices,
        pattern_by_row,
        groups,
        group_sizes,
        exposed_arms,
        correct,
    )
    return group_differences, group_sizes, pair_pattern_indices


def sampling_distribution(
    group_differences: np.ndarray,
    group_sizes: np.ndarray,
    pair_pattern_indices: np.ndarray,
    *,
    mask: int,
    replicates: int = DECOMPOSITION_REPLICATES,
    seed: int | None = None,
) -> np.ndarray:
    differences = np.asarray(group_differences, dtype=np.float64)
    sizes = np.asarray(group_sizes, dtype=np.float64)
    pairs = np.asarray(pair_pattern_indices, dtype=np.int16)
    if differences.shape != (8, 40, 34):
        raise ValueError("Expected group differences with shape (8,40,34).")
    if sizes.shape != (40, 34) or pairs.shape != (20, 2):
        raise ValueError("Invalid source-group or pair shape.")
    rng = np.random.Generator(
        np.random.PCG64(
            DECOMPOSITION_SEED_BASE + mask if seed is None else seed
        )
    )
    if mask & FACTOR_BITS["model_seed"]:
        seed_indices = rng.integers(
            0,
            8,
            size=(replicates, 8),
            dtype=np.int16,
        )
    else:
        seed_indices = np.broadcast_to(
            np.arange(8, dtype=np.int16),
            (replicates, 8),
        )
    if mask & FACTOR_BITS["pattern_pair"]:
        pair_indices = rng.integers(
            0,
            20,
            size=(replicates, 20),
            dtype=np.int16,
        )
    else:
        pair_indices = np.broadcast_to(
            np.arange(20, dtype=np.int16),
            (replicates, 20),
        )
    selected_patterns = pairs[pair_indices].reshape(replicates, 40)
    if mask & FACTOR_BITS["source_group"]:
        group_indices = rng.integers(
            0,
            34,
            size=(replicates, 40, 34),
            dtype=np.int16,
        )
        denominators = sizes[
            selected_patterns[:, :, None],
            group_indices,
        ].sum(axis=2)
        real_seed_effects = np.empty(
            (replicates, 8, 40),
            dtype=np.float64,
        )
        for model_seed in MODEL_SEEDS:
            numerators = differences[model_seed][
                selected_patterns[:, :, None],
                group_indices,
            ].sum(axis=2)
            real_seed_effects[:, model_seed, :] = (
                100.0 * numerators / denominators
            )
    else:
        fixed_effects = (
            100.0
            * differences.sum(axis=2)
            / sizes.sum(axis=1)[None, :]
        )
        real_seed_effects = np.transpose(
            fixed_effects[:, selected_patterns],
            (1, 0, 2),
        )
    selected_seed_effects = np.take_along_axis(
        real_seed_effects,
        seed_indices[:, :, None],
        axis=1,
    )
    return selected_seed_effects.mean(axis=(1, 2))


def bootstrap_decomposition(
    group_differences: np.ndarray,
    group_sizes: np.ndarray,
    pair_pattern_indices: np.ndarray,
) -> tuple[
    dict[int, np.ndarray],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    distributions = {
        mask: sampling_distribution(
            group_differences,
            group_sizes,
            pair_pattern_indices,
            mask=mask,
        )
        for mask in range(8)
    }
    summary_rows: list[dict[str, Any]] = []
    variances: dict[int, float] = {}
    for mask, values in distributions.items():
        lower, upper = np.quantile(
            values,
            [0.025, 0.975],
            method="linear",
        )
        variance = float(np.var(values, ddof=1))
        variances[mask] = variance
        summary_rows.append(
            {
                "distribution_sha256": array_sha256(values),
                "interval_lower_pp": float(lower),
                "interval_upper_pp": float(upper),
                "interval_width_pp": float(upper - lower),
                "mask": mask,
                "mean_pp": float(values.mean()),
                "resample_model_seed": bool(
                    mask & FACTOR_BITS["model_seed"]
                ),
                "resample_pattern_pair": bool(
                    mask & FACTOR_BITS["pattern_pair"]
                ),
                "resample_source_group": bool(
                    mask & FACTOR_BITS["source_group"]
                ),
                "sd_pp": float(np.std(values, ddof=1)),
                "variance_pp2": variance,
            }
        )
    shapley_rows: list[dict[str, Any]] = []
    full_variance = variances[7]
    bits = tuple(FACTOR_BITS.items())
    for factor, bit in bits:
        contribution = 0.0
        for permutation in itertools.permutations(
            [item[1] for item in bits]
        ):
            prior = 0
            for current in permutation:
                if current == bit:
                    contribution += variances[prior | bit] - variances[prior]
                    break
                prior |= current
        contribution /= math.factorial(len(bits))
        shapley_rows.append(
            {
                "factor": factor,
                "shapley_share": (
                    contribution / full_variance
                    if full_variance > 0
                    else 0.0
                ),
                "shapley_variance_pp2": contribution,
            }
        )
    if not math.isclose(
        sum(row["shapley_variance_pp2"] for row in shapley_rows),
        full_variance,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise RuntimeError("Shapley variance contributions do not add up.")
    return distributions, summary_rows, shapley_rows


def pair_effect_matrix() -> np.ndarray:
    rows = read_csv(EFFECT_ANALYSIS_DIR / "seed_pattern_effects.csv")
    matrix = np.full((8, 20, 2), np.nan, dtype=np.float64)
    for row in rows:
        matrix[
            int(row["model_seed"]),
            int(row["pair_index"]),
            int(row["pair_member"]),
        ] = float(row["effect_pp"])
    if not np.isfinite(matrix).all():
        raise RuntimeError("Seed-pattern effect table is incomplete.")
    return matrix.mean(axis=2)


def random_effects_anova(
    matrix: np.ndarray,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (8, 20):
        raise ValueError("Expected an 8 by 20 seed-pair matrix.")
    seed_count, pair_count = values.shape
    grand = float(values.mean())
    seed_means = values.mean(axis=1)
    pair_means = values.mean(axis=0)
    residuals = (
        values
        - seed_means[:, None]
        - pair_means[None, :]
        + grand
    )
    ss_seed = float(
        pair_count * np.square(seed_means - grand).sum()
    )
    ss_pair = float(
        seed_count * np.square(pair_means - grand).sum()
    )
    ss_residual = float(np.square(residuals).sum())
    ms_seed = ss_seed / (seed_count - 1)
    ms_pair = ss_pair / (pair_count - 1)
    ms_residual = ss_residual / (
        (seed_count - 1) * (pair_count - 1)
    )
    seed_variance = max(
        (ms_seed - ms_residual) / pair_count,
        0.0,
    )
    pair_variance = max(
        (ms_pair - ms_residual) / seed_count,
        0.0,
    )
    interaction_variance = ms_residual
    mean_contributions = {
        "model_seed": seed_variance / seed_count,
        "pattern_pair": pair_variance / pair_count,
        "seed_pair_interaction": (
            interaction_variance / (seed_count * pair_count)
        ),
    }
    total_mean_variance = sum(mean_contributions.values())
    result = {
        "grand_mean_pp": grand,
        "interaction_variance_pp2": interaction_variance,
        "model_seed_variance_pp2": seed_variance,
        "pattern_pair_variance_pp2": pair_variance,
        "projected_grand_mean_variance_pp2": total_mean_variance,
    }
    rows = [
        {
            "component": name,
            "grand_mean_variance_contribution_pp2": contribution,
            "grand_mean_variance_share": (
                contribution / total_mean_variance
                if total_mean_variance > 0
                else 0.0
            ),
            "random_effect_variance_pp2": {
                "model_seed": seed_variance,
                "pattern_pair": pair_variance,
                "seed_pair_interaction": interaction_variance,
            }[name],
        }
        for name, contribution in mean_contributions.items()
    ]
    return result, rows


def all_candidate_pairs() -> list[dict[str, Any]]:
    eligible = list(
        iter_jsonl(DATA_DIR / "clutrr_v3_eligible_patterns.jsonl")
    )
    group_counts = {
        row["semantic_id"]: int(row["complete_source_groups"])
        for row in eligible
    }
    by_relation: dict[str, list[str]] = defaultdict(list)
    for row in eligible:
        by_relation[row["target_relation"]].append(row["semantic_id"])
    pairs: list[dict[str, Any]] = []
    for relation in sorted(by_relation):
        semantic_ids = sorted(by_relation[relation])
        for offset in range(0, len(semantic_ids) - 1, 2):
            first, second = semantic_ids[offset : offset + 2]
            pairs.append(
                {
                    "member_0": first,
                    "member_0_complete_source_groups": group_counts[first],
                    "member_1": second,
                    "member_1_complete_source_groups": group_counts[second],
                    "pair_sha256": hashlib.sha256(
                        (first + second).encode("ascii")
                    ).hexdigest(),
                    "target_relation": relation,
                }
            )
    return sorted(pairs, key=lambda row: row["pair_sha256"])


def capacity_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_pairs = all_candidate_pairs()
    selected = {
        row["pair_sha256"]
        for row in iter_jsonl(DATA_DIR / "clutrr_v3_pairs.jsonl")
    }
    remaining = [
        row for row in all_pairs if row["pair_sha256"] not in selected
    ]
    eligible_counts = Counter(
        row["target_relation"]
        for row in iter_jsonl(
            DATA_DIR / "clutrr_v3_eligible_patterns.jsonl"
        )
    )
    all_counts = Counter(row["target_relation"] for row in all_pairs)
    selected_counts = Counter(
        row["target_relation"]
        for row in iter_jsonl(DATA_DIR / "clutrr_v3_pairs.jsonl")
    )
    remaining_counts = Counter(
        row["target_relation"] for row in remaining
    )
    rows = [
        {
            "eligible_patterns": eligible_counts[relation],
            "original_pair_capacity": all_counts[relation],
            "remaining_disjoint_pairs": remaining_counts[relation],
            "target_relation": relation,
            "v3_selected_pairs": selected_counts[relation],
        }
        for relation in sorted(eligible_counts)
    ]
    if len(all_pairs) != 159 or len(selected) != 20 or len(remaining) != 139:
        raise RuntimeError("Frozen pair capacity changed.")
    return rows, remaining


def projected_mean_variance(
    anova: dict[str, float],
    seed_count: float,
    pair_count: int,
) -> float:
    seed_term = (
        0.0
        if math.isinf(seed_count)
        else anova["model_seed_variance_pp2"] / seed_count
    )
    interaction_term = (
        0.0
        if math.isinf(seed_count)
        else anova["interaction_variance_pp2"]
        / (seed_count * pair_count)
    )
    return (
        seed_term
        + anova["pattern_pair_variance_pp2"] / pair_count
        + interaction_term
    )


def precision_projection(
    anova: dict[str, float],
    registered_width_pp: float,
    remaining_pair_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_variance = projected_mean_variance(anova, 8.0, 20)
    raw_base_width = 2.0 * Z_975 * math.sqrt(base_variance)
    calibration_factor = registered_width_pp / raw_base_width
    pair_counts = sorted(
        set((*PROJECTION_PAIR_COUNTS, remaining_pair_count))
    )
    rows: list[dict[str, Any]] = []
    for seed_count in (*PROJECTION_SEED_COUNTS, math.inf):
        for pair_count in pair_counts:
            variance = projected_mean_variance(
                anova,
                float(seed_count),
                pair_count,
            )
            width = (
                2.0
                * Z_975
                * math.sqrt(variance)
                * calibration_factor
            )
            rows.append(
                {
                    "pair_count": pair_count,
                    "projected_interval_width_pp": width,
                    "projected_mean_variance_pp2": variance,
                    "seed_count": (
                        "infinite"
                        if math.isinf(seed_count)
                        else int(seed_count)
                    ),
                    "target_width_pass": (
                        width <= TARGET_INTERVAL_WIDTH_PP
                    ),
                }
            )
    target_variance = (
        base_variance
        * (TARGET_INTERVAL_WIDTH_PP / registered_width_pp) ** 2
    )
    required_rows: list[dict[str, Any]] = []
    for seed_count in (*PROJECTION_SEED_COUNTS, math.inf):
        if math.isinf(seed_count):
            denominator = target_variance
            numerator = anova["pattern_pair_variance_pp2"]
        else:
            denominator = (
                target_variance
                - anova["model_seed_variance_pp2"] / seed_count
            )
            numerator = (
                anova["pattern_pair_variance_pp2"]
                + anova["interaction_variance_pp2"] / seed_count
            )
        required = (
            math.inf
            if denominator <= 0
            else math.ceil(numerator / denominator)
        )
        required_rows.append(
            {
                "required_pair_count": (
                    "infinite" if math.isinf(required) else int(required)
                ),
                "seed_count": (
                    "infinite"
                    if math.isinf(seed_count)
                    else int(seed_count)
                ),
            }
        )
    infinite_seed_required = next(
        row["required_pair_count"]
        for row in required_rows
        if row["seed_count"] == "infinite"
    )
    max_capacity_width = next(
        row["projected_interval_width_pp"]
        for row in rows
        if row["seed_count"] == "infinite"
        and row["pair_count"] == remaining_pair_count
    )
    summary = {
        "anova_to_registered_width_calibration_factor": calibration_factor,
        "base_anova_width_pp": raw_base_width,
        "base_projected_mean_variance_pp2": base_variance,
        "maximum_remaining_pair_capacity": remaining_pair_count,
        "minimum_pairs_at_infinite_seeds": infinite_seed_required,
        "projected_width_at_infinite_seeds_and_max_pairs_pp": (
            max_capacity_width
        ),
        "required_pairs_by_seed_count": required_rows,
        "target_interval_width_pp": TARGET_INTERVAL_WIDTH_PP,
    }
    return rows, summary


def v4_gate(
    projection: dict[str, Any],
    remaining_pair_count: int,
) -> dict[str, Any]:
    required = projection["minimum_pairs_at_infinite_seeds"]
    width = projection[
        "projected_width_at_infinite_seeds_and_max_pairs_pp"
    ]
    capacity_pass = (
        isinstance(required, int) and required <= remaining_pair_count
    )
    width_pass = width <= TARGET_INTERVAL_WIDTH_PP
    decision = (
        "strict_v3_precision_replication_go"
        if capacity_pass and width_pass
        else "strict_v3_precision_replication_no_go"
    )
    return {
        "capacity_gate_pass": capacity_pass,
        "decision": decision,
        "projected_max_capacity_width_gate_pass": width_pass,
        "recommendation": (
            "release_v4_protocol_design"
            if decision == "strict_v3_precision_replication_go"
            else "do_not_release_same_gate_v4_training"
        ),
        "v3_decision_remains": "imprecise_no_go",
        "v4_training_released": False,
    }


def _report(
    results: dict[str, Any],
    gate: dict[str, Any],
) -> str:
    shapley = {
        row["factor"]: row["shapley_share"]
        for row in results["bootstrap_shapley"]
    }
    projection = results["prospective_projection"]
    max_capacity_width = projection[
        "projected_width_at_infinite_seeds_and_max_pairs_pp"
    ]
    return "\n".join(
        [
            "# CLUTRR V3 探索性方差分解与 V4 设计门控",
            "",
            "## 证据状态",
            "",
            "- 本分析在 V3 解盲后冻结，只能用于探索性诊断和未来设计。",
            "- V3 的预注册结论保持 `imprecise_no_go`，未作追溯修改。",
            "",
            "## 方差分解",
            "",
            (
                "- Bootstrap-Shapley 方差份额："
                f"seed {100 * shapley['model_seed']:.2f}%，"
                f"pair {100 * shapley['pattern_pair']:.2f}%，"
                f"source group {100 * shapley['source_group']:.2f}%。"
            ),
            (
                "- 该分解是抽样机制归因，不等同于因果方差组件。"
            ),
            "",
            "## V4 严格精度门",
            "",
            (
                "- 未参与 V3 的同标签语义 pair 容量："
                f"{projection['maximum_remaining_pair_capacity']}。"
            ),
            (
                "- 即使 seed 数趋于无穷，达到 2 pp 区间宽度的"
                f"投影最少 pair 数："
                f"{projection['minimum_pairs_at_infinite_seeds']}。"
            ),
            (
                "- 使用全部剩余 pairs 且 seed 数趋于无穷时，"
                "投影区间宽度："
                f"{max_capacity_width:.4f} pp。"
            ),
            f"- 设计门决策：`{gate['decision']}`。",
            f"- 建议：`{gate['recommendation']}`。",
            "",
            (
                "结论仅适用于复现 V3 的 2 pp 严格 precision gate。"
                "采用不同 estimand 或方向性门控必须建立新的前瞻性协议，"
                "不能由本探索性结果直接放行。"
            ),
            "",
        ]
    )


def run_variance_design() -> dict[str, Any]:
    verify_variance_protocol()
    verify_variance_inputs()
    if ANALYSIS_DIR.exists():
        raise RuntimeError("Variance output directory already exists.")
    group_differences, group_sizes, pair_indices = build_outcome_tensors()
    (
        distributions,
        component_rows,
        shapley_rows,
    ) = bootstrap_decomposition(
        group_differences,
        group_sizes,
        pair_indices,
    )
    anova, anova_rows = random_effects_anova(pair_effect_matrix())
    capacity, remaining_pairs = capacity_rows()
    effect_results = read_json(EFFECT_ANALYSIS_DIR / "results.json")
    registered_width = float(
        effect_results["primary"]["confidence_interval_95"]["width_pp"]
    )
    projection_rows, projection = precision_projection(
        anova,
        registered_width,
        len(remaining_pairs),
    )
    gate = v4_gate(projection, len(remaining_pairs))
    ANALYSIS_DIR.mkdir(parents=True)
    write_csv(
        ANALYSIS_DIR / "bootstrap_component_summary.csv",
        component_rows,
    )
    write_csv(ANALYSIS_DIR / "anova_components.csv", anova_rows)
    write_csv(
        ANALYSIS_DIR / "remaining_pair_capacity.csv",
        capacity,
    )
    write_csv(
        ANALYSIS_DIR / "precision_projection.csv",
        projection_rows,
    )
    write_deterministic_npz(
        ANALYSIS_DIR / "decomposition_distributions.npz",
        {
            f"mask_{mask}": values
            for mask, values in distributions.items()
        },
    )
    results = {
        "anova": anova,
        "bootstrap_shapley": shapley_rows,
        "bootstrap_subsets": component_rows,
        "computed_at_utc": _now(),
        "design_id": DESIGN_ID,
        "epistemic_status": (
            "post_unblinding_exploratory_design_planning"
        ),
        "prospective_projection": projection,
        "schema_version": (
            "external-behavior-v3-variance-results-v1"
        ),
        "status": "exploratory_decomposition_complete",
        "v3_primary_effect_pp": effect_results["primary"]["effect_pp"],
        "v3_registered_decision": "imprecise_no_go",
    }
    write_json(ANALYSIS_DIR / "variance_results.json", results)
    gate_report = {
        "capacity": {
            "all_original_pair_capacity": (
                len(remaining_pairs) + 20
            ),
            "remaining_disjoint_pair_capacity": len(remaining_pairs),
            "v3_used_pairs": 20,
        },
        "design_id": DESIGN_ID,
        "gate": gate,
        "projection": projection,
        "schema_version": "external-behavior-v4-design-gate-report-v1",
        "status": gate["decision"],
    }
    write_json(
        ANALYSIS_DIR / "v4_design_gate_report.json",
        gate_report,
    )
    (ANALYSIS_DIR / "report_cn.md").write_text(
        _report(results, gate),
        encoding="utf-8",
    )
    output_hashes = {
        name: sha256_file(ANALYSIS_DIR / name)
        for name in OUTPUT_NAMES
    }
    manifest = {
        "completed_at_utc": _now(),
        "design_id": DESIGN_ID,
        "output_sha256": output_hashes,
        "protocol_input_manifest_sha256": sha256_file(
            PROTOCOL_DIR / "input_manifest.json"
        ),
        "schema_version": (
            "external-behavior-v3-variance-analysis-manifest-v1"
        ),
        "status": "complete_pending_independent_validation",
    }
    write_json(ANALYSIS_DIR / "analysis_manifest.json", manifest)
    return gate_report


def _checklist(checks: list[dict[str, Any]]) -> str:
    status = {item["name"]: item["pass"] for item in checks}
    labels = (
        ("protocol", "Validate the frozen exploratory protocol."),
        ("inputs", "Validate all frozen inputs and implementation hashes."),
        ("outputs", "Validate all declared output hashes."),
        ("decomposition", "Reproduce all eight bootstrap subsets."),
        ("shapley", "Recompute additive Shapley variance attribution."),
        ("anova", "Recompute the two-way random-effects decomposition."),
        ("capacity", "Reconstruct all selected and remaining pairs."),
        ("projection", "Recompute calibrated prospective precision."),
        ("gate", "Reapply the frozen V4 strict-precision design gate."),
        ("finite", "Confirm finite numeric outputs."),
    )
    lines = ["# V3 Variance and V4 Design Integrity", ""]
    for name, label in labels:
        lines.append(f"- [{'x' if status.get(name) else ' '}] {label}")
    lines.append("")
    return "\n".join(lines)


def validate_variance_design() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: Any = None) -> None:
        checks.append({"detail": detail, "name": name, "pass": bool(condition)})

    try:
        verify_variance_protocol()
        check("protocol", True)
    except Exception as error:
        check("protocol", False, repr(error))
    try:
        verify_variance_inputs()
        check("inputs", True)
    except Exception as error:
        check("inputs", False, repr(error))
    manifest = read_json(ANALYSIS_DIR / "analysis_manifest.json")
    failures = [
        name
        for name, expected in manifest["output_sha256"].items()
        if not (ANALYSIS_DIR / name).is_file()
        or sha256_file(ANALYSIS_DIR / name) != expected
    ]
    check("outputs", not failures, failures)
    results = read_json(ANALYSIS_DIR / "variance_results.json")
    stored_distributions = load_npz(
        ANALYSIS_DIR / "decomposition_distributions.npz"
    )
    group_differences, group_sizes, pair_indices = build_outcome_tensors()
    distributions, component_rows, shapley_rows = bootstrap_decomposition(
        group_differences,
        group_sizes,
        pair_indices,
    )
    decomposition_ok = all(
        np.array_equal(
            values,
            stored_distributions[f"mask_{mask}"],
        )
        for mask, values in distributions.items()
    ) and component_rows == results["bootstrap_subsets"]
    check("decomposition", decomposition_ok)
    check("shapley", shapley_rows == results["bootstrap_shapley"])
    anova, _ = random_effects_anova(pair_effect_matrix())
    check("anova", anova == results["anova"])
    capacity, remaining = capacity_rows()
    stored_capacity = read_csv(
        ANALYSIS_DIR / "remaining_pair_capacity.csv"
    )
    check(
        "capacity",
        len(remaining) == 139
        and len(capacity) == len(stored_capacity)
        and all(
            str(expected[key]) == observed[key]
            for expected, observed in zip(capacity, stored_capacity)
            for key in expected
        ),
    )
    effect_results = read_json(EFFECT_ANALYSIS_DIR / "results.json")
    projection_rows, projection = precision_projection(
        anova,
        float(
            effect_results["primary"]["confidence_interval_95"][
                "width_pp"
            ]
        ),
        len(remaining),
    )
    check(
        "projection",
        projection == results["prospective_projection"]
        and len(projection_rows)
        == len(read_csv(ANALYSIS_DIR / "precision_projection.csv")),
    )
    gate_report = read_json(
        ANALYSIS_DIR / "v4_design_gate_report.json"
    )
    expected_gate = v4_gate(projection, len(remaining))
    check("gate", gate_report["gate"] == expected_gate)

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
        finite_tree(results)
        and all(
            np.isfinite(value).all()
            for value in stored_distributions.values()
        ),
    )
    failed = [item for item in checks if not item["pass"]]
    report = {
        "analysis_gate": "pass" if not failed else "no_go",
        "check_count": len(checks),
        "checks": checks,
        "design_id": DESIGN_ID,
        "failed_checks": [item["name"] for item in failed],
        "failed_count": len(failed),
        "schema_version": (
            "external-behavior-v3-variance-analysis-validation-v1"
        ),
        "status": "pass" if not failed else "no_go",
        "v4_design_decision": gate_report["gate"]["decision"],
        "validated_at_utc": _now(),
    }
    write_json(ANALYSIS_DIR / "analysis_validation_report.json", report)
    (ANALYSIS_DIR / "analysis_checklist.md").write_text(
        _checklist(checks),
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
