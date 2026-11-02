#!/usr/bin/env python3
"""Freeze the P1 analysis implementation and machine-readable analysis spec."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quality_p1_common import (
    EXTERNAL_ROOT,
    MODEL_SEEDS,
    MODELS,
    P1_DIR,
    ROOT,
    sha256_file,
    write_json,
)


ANALYSIS_SCHEMA_VERSION = "quality-enhancement-p1-analysis-v1"
RUN_ROOT = EXTERNAL_ROOT / "runs" / "quality-enhancement-v1" / "p1"
ANALYSIS_DIR = (
    EXTERNAL_ROOT / "analysis" / "quality-enhancement-v1" / "p1"
)
ANALYSIS_SCRIPT = Path(__file__).with_name("analyze_quality_p1.py")


def fail(message: str) -> None:
    raise SystemExit(f"P1 analysis freeze failed: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def hash_inputs(paths: list[Path]) -> dict[str, str]:
    values: dict[str, str] = {}
    for path in paths:
        require(path.is_file(), f"missing analysis input: {path}")
        values[relative(path)] = sha256_file(path)
    return dict(sorted(values.items()))


def main() -> int:
    full_validation_path = RUN_ROOT / "full_validation.json"
    full_validation = json.loads(
        full_validation_path.read_text(encoding="ascii")
    )
    require(full_validation.get("status") == "passed", "full validation failed")
    require(
        full_validation.get("effect_metrics_compared") is False,
        "full validation already compared effect metrics",
    )
    require(
        full_validation.get("eligible_run_count") == 60,
        "full validation did not retain all 60 runs",
    )

    spec_path = ANALYSIS_DIR / "analysis_spec.json"
    outcome_names = (
        "bootstrap.npz",
        "function_effects.csv",
        "model_effects.csv",
        "report_cn.md",
        "results.json",
        "seed_effects.csv",
    )
    require(not spec_path.exists(), "analysis spec already exists")
    for name in outcome_names:
        require(
            not (ANALYSIS_DIR / name).exists(),
            f"outcome artifact exists before analysis freeze: {name}",
        )
    require(ANALYSIS_SCRIPT.is_file(), "analysis implementation is missing")

    preregistration_path = P1_DIR.parent / "preregistration.json"
    input_paths = [
        preregistration_path,
        P1_DIR / "SHA256SUMS",
        P1_DIR / "design_manifest.json",
        P1_DIR / "validation_report.json",
        P1_DIR / "run_manifest.csv",
        P1_DIR / "targets.jsonl",
        P1_DIR / "pairs.jsonl",
        P1_DIR / "eval.npz",
        RUN_ROOT / "execution_state.json",
        RUN_ROOT / "matrix_status.json",
        full_validation_path,
        Path(__file__).resolve(),
        ANALYSIS_SCRIPT.resolve(),
        Path(__file__).with_name(
            "validate_quality_p1_full_matrix.py"
        ).resolve(),
    ]
    spec: dict[str, Any] = {
        "analysis_implementation_sha256": sha256_file(ANALYSIS_SCRIPT),
        "bootstrap": {
            "architecture_resampling": False,
            "dependency_sharing": {
                "episode_indices_shared_across": [
                    "architectures",
                    "selected_model_seeds",
                    "exposed_and_unexposed_predictions",
                ],
                "function_pair_indices_shared_across": [
                    "architectures",
                    "selected_model_seeds",
                ],
                "model_seed_indices_shared_across": ["architectures"],
            },
            "draw_order": [
                "draw_all_replicate_by_5_model_seed_indices",
                "draw_all_replicate_by_20_function_pair_indices",
                "draw_episode_indices_in_contiguous_replicate_batches",
            ],
            "episode_draws_per_selected_function": 250,
            "function_pair_draws_per_replicate": 20,
            "implementation_batch_size": 32,
            "model_seed_draws_per_replicate": 5,
            "percentile_interpolation": "linear",
            "replicates": 20_000,
            "rng": "numpy.random.Generator(numpy.random.PCG64(seed))",
            "seed": 20_261_009,
            "type": "hierarchical_percentile",
        },
        "estimand": {
            "architecture_macro_weighting": "equal_fixed_weights",
            "architecture_specific_formula": (
                "tau_m_pp = 100 * mean_seed(mean_function("
                "mean_episode(correct_exposed - correct_unexposed)))"
            ),
            "exposure_alignment": (
                "For each model_seed and target_function, select the arm whose "
                "frozen exposure flag is true as exposed and the complementary "
                "arm as unexposed."
            ),
            "function_weighting": "equal_over_all_40_fixed_targets",
            "macro_formula": "tau_macro_pp = mean_over_6_fixed_architectures(tau_m_pp)",
            "name": "architecture_macro_exact_exposure_effect",
            "orientation": "exposed_minus_unexposed",
            "unit": "percentage_points",
        },
        "freeze_implementation_sha256": sha256_file(Path(__file__).resolve()),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "function_order": {
            "pair_order": "pairs.jsonl ascending pair_index 0..19",
            "within_pair_order": [
                "function_multiplier_1",
                "function_multiplier_11",
            ],
        },
        "gate": {
            "equivalence_margin_pp": [-0.5, 0.5],
            "evaluation_order": [
                "positive_behavioral_go",
                "negative_behavioral_go",
                "null_equivalence_go",
                "inconclusive_no_go",
            ],
            "minimum_architectures_same_direction": 4,
            "model_specific_absolute_effect_pp": 1.0,
            "model_specific_claim_rule": (
                "absolute point estimate >= 1.0 pp and the 99.1667% "
                "simultaneous interval excludes zero in the point-estimate "
                "direction"
            ),
            "negative_rule": (
                "tau_macro <= -1.0 pp, macro 95% upper bound < 0, and at "
                "least 4 of 6 architecture point estimates < 0"
            ),
            "null_rule": (
                "macro 95% interval contained in [-0.5,+0.5] pp, every "
                "architecture absolute point estimate < 1.0 pp, and no "
                "simultaneous architecture interval excludes zero"
            ),
            "positive_rule": (
                "tau_macro >= +1.0 pp, macro 95% lower bound > 0, and at "
                "least 4 of 6 architecture point estimates > 0"
            ),
            "sesoi_pp": 1.0,
        },
        "input_sha256": hash_inputs(input_paths),
        "intervals": {
            "model_confidence": 1.0 - 0.05 / 6.0,
            "model_method": (
                "percentile Bonferroni-simultaneous across six architectures"
            ),
            "primary_confidence": 0.95,
            "primary_method": "percentile",
        },
        "model_order": list(MODELS),
        "model_seed_order": list(MODEL_SEEDS),
        "protocol": "quality-enhancement-p1-v1",
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "secondary": {
            "cross_entropy_and_brier": {
                "orientation": "exposed_minus_unexposed",
                "reporting": (
                    "point estimates only; equal function, model-seed, and "
                    "architecture weights; descriptive and not Gate-eligible"
                ),
            },
            "effect_by_frozen_alias_signature": (
                "Within each signature, equally average its fixed target "
                "functions, then model seeds and architectures."
            ),
            "evaluation_calibration": {
                "aggregation": (
                    "Compute each summary separately for each architecture "
                    "and model seed after exposure alignment, then take an "
                    "equal architecture-seed mean."
                ),
                "bin_edges": "15 equal-width bins on [0,1], last bin includes 1",
                "bins": 15,
                "confidence": "maximum predicted class probability",
                "metrics": [
                    "accuracy",
                    "mean_confidence",
                    "signed_gap_confidence_minus_accuracy",
                    "expected_calibration_error",
                ],
            },
            "per_function_effect": (
                "For each fixed function, average paired episode correctness "
                "differences equally over the five seeds and six architectures."
            ),
            "training_calibration": {
                "reason": (
                    "The frozen training adapter persisted final-epoch loss "
                    "and accuracy but not training-set probabilities."
                ),
                "rule": (
                    "Report not_estimable_from_frozen_artifacts; do not "
                    "substitute loss or accuracy for calibration. Final-epoch "
                    "loss and accuracy may be listed as descriptive execution "
                    "summaries."
                ),
                "status": "registered_but_unavailable",
            },
        },
        "status": "frozen_before_outcome_contrast_computation",
    }
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    write_json(spec_path, spec)
    output = {
        "analysis_spec": relative(spec_path),
        "analysis_spec_sha256": sha256_file(spec_path),
        "analysis_implementation_sha256": spec[
            "analysis_implementation_sha256"
        ],
        "status": spec["status"],
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
