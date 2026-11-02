#!/usr/bin/env python3
"""Additional two-way cluster-robust sensitivity for CLUTRR V4."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = (
    PROJECT_ROOT
    / "analysis"
    / "external-behavior-v4"
    / "effect-analysis-v1"
    / "seed_pattern_effects.csv"
)
PRIMARY_RESULTS = (
    PROJECT_ROOT
    / "analysis"
    / "external-behavior-v4"
    / "effect-analysis-v1"
    / "results.json"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "analysis"
    / "external-behavior-v4"
    / "cluster-sensitivity-v1"
)
RESULTS_PATH = OUTPUT_DIR / "results.json"
REPORT_PATH = OUTPUT_DIR / "report.md"
CHECKSUM_PATH = OUTPUT_DIR / "SHA256SUMS"

EXPECTED_SEEDS = tuple(range(8))
EXPECTED_PAIRS = tuple(range(40))
EXPECTED_MEMBERS = (0, 1)
T_975_DF7 = 2.3646242515927844


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path = INPUT_CSV) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="ascii") as handle:
        raw = list(csv.DictReader(handle))
    rows: list[dict[str, Any]] = []
    for row in raw:
        rows.append(
            {
                "effect_pp": float(row["effect_pp"]),
                "global_pair_index": int(row["global_pair_index"]),
                "model_seed": int(row["model_seed"]),
                "pair_member": int(row["pair_member"]),
                "semantic_id": row["semantic_id"],
            }
        )
    return rows


def validate_grid(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cells = Counter(
        (
            row["model_seed"],
            row["global_pair_index"],
            row["pair_member"],
        )
        for row in rows
    )
    expected = {
        (seed, pair, member)
        for seed in EXPECTED_SEEDS
        for pair in EXPECTED_PAIRS
        for member in EXPECTED_MEMBERS
    }
    actual = set(cells)
    duplicates = sorted(cell for cell, count in cells.items() if count != 1)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    semantic_ids = [row["semantic_id"] for row in rows]
    if duplicates or missing or unexpected:
        raise RuntimeError(
            "Unbalanced seed-pair-member grid: "
            f"duplicates={duplicates}, missing={missing}, "
            f"unexpected={unexpected}"
        )
    if len(set(semantic_ids)) != 80:
        raise RuntimeError("Expected exactly 80 target semantic IDs.")
    return {
        "model_seed_count": len(EXPECTED_SEEDS),
        "observation_count": len(rows),
        "pair_count": len(EXPECTED_PAIRS),
        "pair_member_count": len(EXPECTED_MEMBERS),
        "semantic_id_count": len(set(semantic_ids)),
        "status": "pass",
    }


def cluster_variance(
    rows: list[dict[str, Any]],
    residuals: list[float],
    key: Callable[[dict[str, Any]], Any],
) -> dict[str, float | int]:
    scores: dict[Any, float] = defaultdict(float)
    for row, residual in zip(rows, residuals):
        scores[key(row)] += residual
    group_count = len(scores)
    if group_count < 2:
        raise RuntimeError("Cluster variance requires at least two groups.")
    observation_count = len(rows)
    correction = group_count / (group_count - 1)
    score_sum_squares = sum(score * score for score in scores.values())
    variance = (
        correction
        * score_sum_squares
        / (observation_count * observation_count)
    )
    return {
        "cluster_count": group_count,
        "cr1_correction": correction,
        "score_sum_squares": score_sum_squares,
        "variance_pp2": variance,
    }


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grid = validate_grid(rows)
    effects = [row["effect_pp"] for row in rows]
    observation_count = len(effects)
    effect_pp = sum(effects) / observation_count
    residuals = [effect - effect_pp for effect in effects]
    seed = cluster_variance(
        rows,
        residuals,
        lambda row: row["model_seed"],
    )
    pair = cluster_variance(
        rows,
        residuals,
        lambda row: row["global_pair_index"],
    )
    intersection = cluster_variance(
        rows,
        residuals,
        lambda row: (row["model_seed"], row["global_pair_index"]),
    )
    variance = (
        float(seed["variance_pp2"])
        + float(pair["variance_pp2"])
        - float(intersection["variance_pp2"])
    )
    if variance <= 0.0:
        raise RuntimeError("Two-way cluster variance is not positive.")
    standard_error = math.sqrt(variance)
    degrees_of_freedom = min(
        int(seed["cluster_count"]),
        int(pair["cluster_count"]),
    ) - 1
    if degrees_of_freedom != 7:
        raise RuntimeError("Unexpected conservative degrees of freedom.")
    half_width = T_975_DF7 * standard_error

    primary = json.loads(PRIMARY_RESULTS.read_text(encoding="ascii"))
    primary_effect = float(primary["primary"]["effect_pp"])
    if not math.isclose(effect_pp, primary_effect, abs_tol=1e-12):
        raise RuntimeError(
            f"Sensitivity effect {effect_pp} != primary {primary_effect}"
        )

    return {
        "analysis_id": "external-behavior-v4-cluster-sensitivity-v1",
        "confidence_interval_95": {
            "critical_value": T_975_DF7,
            "degrees_of_freedom": degrees_of_freedom,
            "lower_pp": effect_pp - half_width,
            "upper_pp": effect_pp + half_width,
        },
        "effect_pp": effect_pp,
        "epistemic_status": "post_hoc_reviewer_requested_sensitivity",
        "grid_validation": grid,
        "input_sha256": {
            str(INPUT_CSV.relative_to(PROJECT_ROOT)): sha256_file(INPUT_CSV),
            str(PRIMARY_RESULTS.relative_to(PROJECT_ROOT)): sha256_file(
                PRIMARY_RESULTS
            ),
        },
        "implementation_sha256": {
            str(Path(__file__).resolve().relative_to(PROJECT_ROOT)): sha256_file(
                Path(__file__).resolve()
            )
        },
        "method": {
            "description": (
                "Intercept-only Cameron-Gelbach-Miller two-way cluster "
                "sandwich over seed-pattern effects."
            ),
            "intersection_clusters": "model_seed x complete pattern_pair",
            "pair_clusters": "complete same-label pattern pairs",
            "seed_clusters": "model training seeds",
            "small_sample_correction": (
                "CR1 G/(G-1) applied separately to seed, pair, and "
                "seed-by-pair components"
            ),
            "two_way_formula": "V_seed + V_pair - V_seed_by_pair",
        },
        "primary_gate_impact": "none",
        "schema_version": (
            "external-behavior-v4-cluster-sensitivity-results-v1"
        ),
        "standard_error_pp": standard_error,
        "status": "complete",
        "variance_components": {
            "model_seed": seed,
            "pattern_pair": pair,
            "seed_by_pair_intersection": intersection,
            "two_way_variance_pp2": variance,
        },
    }


def write_outputs(results: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(results, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    interval = results["confidence_interval_95"]
    REPORT_PATH.write_text(
        "\n".join(
            [
                "# CLUTRR V4 two-way cluster sensitivity",
                "",
                "- Status: additional post-hoc sensitivity.",
                f"- Point estimate: {results['effect_pp']:.6f} pp.",
                (
                    "- Two-way CR1 cluster SE: "
                    f"{results['standard_error_pp']:.6f} pp."
                ),
                (
                    "- Conservative t(7) 95% interval: "
                    f"[{interval['lower_pp']:.6f}, "
                    f"{interval['upper_pp']:.6f}] pp."
                ),
                "- Clusters: 8 model seeds and 40 complete pattern pairs.",
                "- Primary Gate impact: none.",
                "",
            ]
        ),
        encoding="ascii",
    )
    files = (RESULTS_PATH, REPORT_PATH)
    CHECKSUM_PATH.write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in files),
        encoding="ascii",
    )


def main() -> None:
    results = analyze(read_rows())
    write_outputs(results)
    print(
        json.dumps(
            {
                "ci_lower_pp": results["confidence_interval_95"]["lower_pp"],
                "ci_upper_pp": results["confidence_interval_95"]["upper_pp"],
                "effect_pp": results["effect_pp"],
                "standard_error_pp": results["standard_error_pp"],
                "status": results["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
