#!/usr/bin/env python3
"""Analyze quality-enhancement P1 under the frozen exposure-effect protocol."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from quality_p1_common import (
    EVAL_EPISODES_PER_FUNCTION,
    EXTERNAL_ROOT,
    MODEL_SEEDS,
    MODELS,
    P1_DIR,
    ROOT,
    load_npz,
    sha256_file,
    write_csv,
    write_deterministic_npz,
    write_json,
)


ANALYSIS_SCHEMA_VERSION = "quality-enhancement-p1-analysis-v1"
RUN_ROOT = EXTERNAL_ROOT / "runs" / "quality-enhancement-v1" / "p1"
ANALYSIS_DIR = (
    EXTERNAL_ROOT / "analysis" / "quality-enhancement-v1" / "p1"
)


def fail(message: str) -> None:
    raise SystemExit(f"P1 analysis failed: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="ascii") as handle:
        return [json.loads(line) for line in handle]


def percentile_interval(
    draws: np.ndarray,
    confidence: float,
    *,
    method: str,
) -> list[float]:
    alpha = 1.0 - confidence
    interval = np.quantile(
        draws,
        [alpha / 2.0, 1.0 - alpha / 2.0],
        method=method,
    )
    return [float(interval[0]), float(interval[1])]


def hierarchical_bootstrap(
    accuracy_delta: np.ndarray,
    pair_function_indices: np.ndarray,
    spec: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    bootstrap = spec["bootstrap"]
    replicates = int(bootstrap["replicates"])
    batch_size = int(bootstrap["implementation_batch_size"])
    rng = np.random.Generator(np.random.PCG64(int(bootstrap["seed"])))

    seed_indices = rng.integers(
        0,
        len(MODEL_SEEDS),
        size=(replicates, len(MODEL_SEEDS)),
        dtype=np.int16,
    )
    pair_indices = rng.integers(
        0,
        len(pair_function_indices),
        size=(replicates, len(pair_function_indices)),
        dtype=np.int16,
    )
    model_draws = np.empty((replicates, len(MODELS)), dtype=np.float64)
    flat_delta = accuracy_delta.reshape(
        len(MODELS),
        len(MODEL_SEEDS),
        -1,
    )

    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        count = stop - start
        selected_functions = pair_function_indices[
            pair_indices[start:stop].astype(np.int64)
        ]
        episode_indices = rng.integers(
            0,
            EVAL_EPISODES_PER_FUNCTION,
            size=(
                count,
                len(pair_function_indices),
                2,
                EVAL_EPISODES_PER_FUNCTION,
            ),
            dtype=np.int16,
        )
        flat_indices = (
            selected_functions[..., None] * EVAL_EPISODES_PER_FUNCTION
            + episode_indices
        )
        episode_means = flat_delta[:, :, flat_indices].mean(
            axis=-1,
            dtype=np.float64,
        )
        episode_means = episode_means.transpose(0, 2, 1, 3, 4)
        selected_seed_means = np.take_along_axis(
            episode_means,
            seed_indices[start:stop][None, :, :, None, None],
            axis=2,
        )
        model_draws[start:stop] = (
            selected_seed_means.mean(axis=(2, 3, 4)) * 100.0
        ).T

    macro_draws = model_draws.mean(axis=1)
    return model_draws, macro_draws, {
        "seed_resample_indices": seed_indices,
        "pair_resample_indices": pair_indices,
    }


def calibration_metrics(
    correct: np.ndarray,
    confidence: np.ndarray,
    bins: int,
) -> dict[str, float]:
    correct_flat = correct.reshape(-1).astype(np.float64)
    confidence_flat = confidence.reshape(-1).astype(np.float64)
    require(
        np.isfinite(confidence_flat).all()
        and np.all((confidence_flat >= 0.0) & (confidence_flat <= 1.0)),
        "invalid confidence values",
    )
    bin_index = np.minimum(
        np.floor(confidence_flat * bins).astype(np.int64),
        bins - 1,
    )
    ece = 0.0
    for index in range(bins):
        mask = bin_index == index
        if np.any(mask):
            ece += float(mask.mean()) * abs(
                float(correct_flat[mask].mean())
                - float(confidence_flat[mask].mean())
            )
    accuracy = float(correct_flat.mean())
    mean_confidence = float(confidence_flat.mean())
    return {
        "accuracy": accuracy,
        "mean_confidence": mean_confidence,
        "signed_gap_confidence_minus_accuracy": mean_confidence - accuracy,
        "ece": ece,
    }


def metric_summary(
    exposed: np.ndarray,
    unexposed: np.ndarray,
) -> dict[str, Any]:
    model_exposed = exposed.mean(axis=(1, 2, 3), dtype=np.float64)
    model_unexposed = unexposed.mean(axis=(1, 2, 3), dtype=np.float64)
    model_delta = model_exposed - model_unexposed
    return {
        "architecture_delta": {
            model: float(model_delta[index])
            for index, model in enumerate(MODELS)
        },
        "exposed_mean": float(model_exposed.mean()),
        "unexposed_mean": float(model_unexposed.mean()),
        "exposed_minus_unexposed": float(model_delta.mean()),
    }


def evaluate_gate(
    macro_effect: float,
    macro_interval: list[float],
    model_effects: np.ndarray,
    model_intervals: np.ndarray,
    spec: dict[str, Any],
) -> dict[str, Any]:
    thresholds = spec["gate"]
    positive_architectures = int(np.sum(model_effects > 0.0))
    negative_architectures = int(np.sum(model_effects < 0.0))
    model_intervals_excluding_zero = (
        (model_intervals[:, 0] > 0.0) | (model_intervals[:, 1] < 0.0)
    )
    positive = (
        macro_effect >= thresholds["sesoi_pp"]
        and macro_interval[0] > 0.0
        and positive_architectures >= thresholds["minimum_architectures_same_direction"]
    )
    negative = (
        macro_effect <= -thresholds["sesoi_pp"]
        and macro_interval[1] < 0.0
        and negative_architectures >= thresholds["minimum_architectures_same_direction"]
    )
    null_equivalence = (
        macro_interval[0] >= thresholds["equivalence_margin_pp"][0]
        and macro_interval[1] <= thresholds["equivalence_margin_pp"][1]
        and bool(np.all(np.abs(model_effects) < thresholds["sesoi_pp"]))
        and not bool(np.any(model_intervals_excluding_zero))
    )

    if positive:
        decision = "positive_behavioral_go"
        allowed_claim = (
            "Exact cross-family semantic exposure improves macro accuracy for "
            "the six registered architectures at the registered training budget."
        )
    elif negative:
        decision = "negative_behavioral_go"
        allowed_claim = (
            "Exact cross-family semantic exposure changes macro accuracy in the "
            "negative direction for the six registered architectures and budget; "
            "this is not an inflation claim."
        )
    elif null_equivalence:
        decision = "null_equivalence_go"
        allowed_claim = (
            "The registered architecture-macro exact-exposure effect is bounded "
            "inside the preregistered equivalence margin; this is not universal "
            "invariance."
        )
    else:
        decision = "inconclusive_no_go"
        allowed_claim = (
            "Only the numerical exposure-effect estimate and uncertainty may be "
            "reported; the registered result is neither behavioral Go nor "
            "null-equivalence Go."
        )
    return {
        "allowed_claim": allowed_claim,
        "decision": decision,
        "evaluation_order": thresholds["evaluation_order"],
        "negative_architecture_count": negative_architectures,
        "negative_behavioral_go": negative,
        "null_equivalence_go": null_equivalence,
        "positive_architecture_count": positive_architectures,
        "positive_behavioral_go": positive,
    }


def build_report(results: dict[str, Any]) -> str:
    primary = results["primary"]
    gate = results["gate"]
    model_rows = results["architectures"]
    decision_labels = {
        "positive_behavioral_go": "Positive behavioral Go",
        "negative_behavioral_go": "Negative behavioral Go",
        "null_equivalence_go": "Null-equivalence Go",
        "inconclusive_no_go": "Inconclusive No-Go",
    }
    lines = [
        "# P1 互补精确暴露实验结果",
        "",
        "分析协议：`quality-enhancement-p1-analysis-v1`  ",
        "输入：60/60 个冻结 epoch-60 runs；40 个目标函数，每个函数 250 个共同评估 episode。",
        "",
        "## 完整性",
        "",
        "- 全矩阵验证通过，所有 run 的 checkpoint、预测、元数据与哈希均有效。",
        "- 完整性审计未比较 A/B 效应；分析规范在效应计算前冻结。",
        "- 主区间使用 20,000 次层级 percentile bootstrap；架构不重采样。",
        "",
        "## 主结果",
        "",
        "效应定义为 `exposed - unexposed`，单位为 percentage points (pp)。",
        "",
        f"- `tau_macro` = **{primary['effect_pp']:.3f} pp**",
        f"- 95% CI = **[{primary['ci95_pp'][0]:.3f}, {primary['ci95_pp'][1]:.3f}] pp**",
        f"- Gate = **{decision_labels[gate['decision']]}**",
        "",
        "| Architecture | Exposed acc | Unexposed acc | Effect (pp) | 99.1667% simultaneous CI | Model claim |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in model_rows:
        lines.append(
            f"| `{row['architecture']}` | "
            f"{row['exposed_accuracy']:.4f} | "
            f"{row['unexposed_accuracy']:.4f} | "
            f"{row['effect_pp']:.3f} | "
            f"[{row['ci_lower_pp']:.3f}, {row['ci_upper_pp']:.3f}] | "
            f"{'eligible' if row['model_specific_claim_eligible'] else 'not eligible'} |"
        )
    secondary = results["secondary"]
    lines.extend(
        [
            "",
            "## 次指标",
            "",
            "| Metric | Exposed | Unexposed | Exposed - unexposed |",
            "|---|---:|---:|---:|",
            f"| Cross entropy | {secondary['cross_entropy']['exposed_mean']:.5f} | "
            f"{secondary['cross_entropy']['unexposed_mean']:.5f} | "
            f"{secondary['cross_entropy']['exposed_minus_unexposed']:.5f} |",
            f"| Brier score | {secondary['brier_score']['exposed_mean']:.5f} | "
            f"{secondary['brier_score']['unexposed_mean']:.5f} | "
            f"{secondary['brier_score']['exposed_minus_unexposed']:.5f} |",
            "",
            "训练阶段未持久化逐 episode 概率，因此预注册的训练校准摘要不可估；"
            "该缺失被保留，不以训练 loss 替代校准指标。评估校准摘要已按冻结的 15-bin 定义计算。",
            "",
            "## Gate 与 claim 边界",
            "",
            gate["allowed_claim"],
            "",
            "该实验识别的是注册架构与预算下的随机化 exact-exposure endpoint effect，"
            "不单独识别模型内部的 memorization mechanism。",
            "",
        ]
    )
    return "\n".join(lines)


def verify_frozen_spec(spec: dict[str, Any], spec_path: Path) -> None:
    require(
        spec.get("schema_version") == ANALYSIS_SCHEMA_VERSION,
        "analysis spec schema version",
    )
    require(
        spec.get("status") == "frozen_before_outcome_contrast_computation",
        "analysis spec is not frozen before outcome contrasts",
    )
    for relative, expected in spec["input_sha256"].items():
        path = ROOT / relative
        require(path.is_file(), f"missing frozen analysis input: {relative}")
        require(
            sha256_file(path) == expected,
            f"frozen analysis input changed: {relative}",
        )
    require(
        sha256_file(Path(__file__).resolve())
        == spec["analysis_implementation_sha256"],
        "analysis implementation differs from frozen spec",
    )
    require(spec_path.is_file(), "analysis spec is missing")


def main() -> int:
    spec_path = ANALYSIS_DIR / "analysis_spec.json"
    require(spec_path.is_file(), "analysis spec must be frozen first")
    spec = json.loads(spec_path.read_text(encoding="ascii"))
    verify_frozen_spec(spec, spec_path)

    full_validation = json.loads(
        (RUN_ROOT / "full_validation.json").read_text(encoding="ascii")
    )
    require(full_validation.get("status") == "passed", "full validation failed")
    require(
        full_validation.get("effect_metrics_compared") is False,
        "full validation compared effect metrics",
    )
    require(
        full_validation.get("eligible_run_count") == 60,
        "not all runs are eligible",
    )

    targets = load_jsonl(P1_DIR / "targets.jsonl")
    pairs = load_jsonl(P1_DIR / "pairs.jsonl")
    target_by_id = {int(row["function_id"]): row for row in targets}
    require(len(target_by_id) == 40 and len(pairs) == 20, "target/pair census")
    function_ids: list[int] = []
    for pair_index, pair in enumerate(pairs):
        require(int(pair["pair_index"]) == pair_index, "pair order")
        function_ids.extend(
            [
                int(pair["function_multiplier_1"]),
                int(pair["function_multiplier_11"]),
            ]
        )
    require(len(set(function_ids)) == 40, "pair functions are not unique")
    function_index = {
        function_id: index for index, function_id in enumerate(function_ids)
    }
    pair_function_indices = np.asarray(
        [
            [
                function_index[int(pair["function_multiplier_1"])],
                function_index[int(pair["function_multiplier_11"])],
            ]
            for pair in pairs
        ],
        dtype=np.int16,
    )

    eval_arrays = load_npz(P1_DIR / "eval.npz")
    eval_indices: dict[int, np.ndarray] = {}
    for function_id in function_ids:
        indices = np.flatnonzero(eval_arrays["function_id"] == function_id)
        require(
            len(indices) == EVAL_EPISODES_PER_FUNCTION,
            f"eval count for function {function_id}",
        )
        require(
            np.array_equal(
                eval_arrays["episode_within_function"][indices],
                np.arange(EVAL_EPISODES_PER_FUNCTION, dtype=np.int16),
            ),
            f"eval episode order for function {function_id}",
        )
        eval_indices[function_id] = indices

    shape = (
        len(MODELS),
        len(MODEL_SEEDS),
        len(function_ids),
        EVAL_EPISODES_PER_FUNCTION,
    )
    exposed_accuracy = np.empty(shape, dtype=np.bool_)
    unexposed_accuracy = np.empty(shape, dtype=np.bool_)
    exposed_ce = np.empty(shape, dtype=np.float32)
    unexposed_ce = np.empty(shape, dtype=np.float32)
    exposed_brier = np.empty(shape, dtype=np.float32)
    unexposed_brier = np.empty(shape, dtype=np.float32)
    exposed_confidence = np.empty(shape, dtype=np.float32)
    unexposed_confidence = np.empty(shape, dtype=np.float32)
    training_final_rows: list[dict[str, float]] = []

    for model_index, model in enumerate(MODELS):
        for seed_index, seed in enumerate(MODEL_SEEDS):
            predictions: dict[str, dict[str, np.ndarray]] = {}
            for arm in ("A", "B"):
                run_id = f"{model}__seed{seed}__arm{arm}"
                path = RUN_ROOT / run_id / "eval_predictions.npz"
                expected_hash = full_validation["runs"][run_id][
                    "artifact_sha256"
                ]["eval_predictions.npz"]
                require(
                    sha256_file(path) == expected_hash,
                    f"{run_id} prediction artifact changed",
                )
                predictions[arm] = load_npz(path)
                with (RUN_ROOT / run_id / "history.csv").open(
                    newline="", encoding="ascii"
                ) as handle:
                    history = list(csv.DictReader(handle))
                require(len(history) == 60, f"{run_id} history changed")
                training_final_rows.append(
                    {
                        "accuracy": float(history[-1]["train_accuracy"]),
                        "loss": float(history[-1]["train_loss"]),
                    }
                )

            for function_position, function_id in enumerate(function_ids):
                indices = eval_indices[function_id]
                exposed_a = predictions["A"]["exposed"][indices]
                exposed_b = predictions["B"]["exposed"][indices]
                require(
                    bool(np.all(exposed_a == exposed_a[0]))
                    and np.array_equal(exposed_a, ~exposed_b),
                    f"{model} seed {seed} function {function_id} exposure",
                )
                exposed_arm, unexposed_arm = (
                    ("A", "B") if bool(exposed_a[0]) else ("B", "A")
                )
                exposed_run = predictions[exposed_arm]
                unexposed_run = predictions[unexposed_arm]
                exposed_accuracy[
                    model_index, seed_index, function_position
                ] = exposed_run["correct"][indices]
                unexposed_accuracy[
                    model_index, seed_index, function_position
                ] = unexposed_run["correct"][indices]
                exposed_ce[
                    model_index, seed_index, function_position
                ] = exposed_run["per_episode_ce"][indices]
                unexposed_ce[
                    model_index, seed_index, function_position
                ] = unexposed_run["per_episode_ce"][indices]
                exposed_brier[
                    model_index, seed_index, function_position
                ] = exposed_run["per_episode_brier"][indices]
                unexposed_brier[
                    model_index, seed_index, function_position
                ] = unexposed_run["per_episode_brier"][indices]
                exposed_confidence[
                    model_index, seed_index, function_position
                ] = exposed_run["probabilities"][indices].max(axis=1)
                unexposed_confidence[
                    model_index, seed_index, function_position
                ] = unexposed_run["probabilities"][indices].max(axis=1)

    accuracy_delta = (
        exposed_accuracy.astype(np.int8) - unexposed_accuracy.astype(np.int8)
    )
    model_exposed_accuracy = exposed_accuracy.mean(
        axis=(1, 2, 3), dtype=np.float64
    )
    model_unexposed_accuracy = unexposed_accuracy.mean(
        axis=(1, 2, 3), dtype=np.float64
    )
    model_effects = (
        model_exposed_accuracy - model_unexposed_accuracy
    ) * 100.0
    macro_effect = float(model_effects.mean())

    model_draws, macro_draws, resample_indices = hierarchical_bootstrap(
        accuracy_delta,
        pair_function_indices,
        spec,
    )
    quantile_method = spec["bootstrap"]["percentile_interpolation"]
    macro_interval = percentile_interval(
        macro_draws,
        float(spec["intervals"]["primary_confidence"]),
        method=quantile_method,
    )
    model_confidence = float(spec["intervals"]["model_confidence"])
    model_intervals = np.asarray(
        [
            percentile_interval(
                model_draws[:, model_index],
                model_confidence,
                method=quantile_method,
            )
            for model_index in range(len(MODELS))
        ],
        dtype=np.float64,
    )
    gate = evaluate_gate(
        macro_effect,
        macro_interval,
        model_effects,
        model_intervals,
        spec,
    )

    architecture_rows: list[dict[str, Any]] = []
    model_specific_threshold = float(
        spec["gate"]["model_specific_absolute_effect_pp"]
    )
    for model_index, model in enumerate(MODELS):
        lower, upper = model_intervals[model_index]
        effect = float(model_effects[model_index])
        interval_excludes_zero = bool(lower > 0.0 or upper < 0.0)
        interval_same_direction = bool(
            (effect > 0.0 and lower > 0.0)
            or (effect < 0.0 and upper < 0.0)
        )
        architecture_rows.append(
            {
                "architecture": model,
                "ci_lower_pp": float(lower),
                "ci_upper_pp": float(upper),
                "effect_pp": effect,
                "exposed_accuracy": float(model_exposed_accuracy[model_index]),
                "interval_confidence": model_confidence,
                "interval_excludes_zero": interval_excludes_zero,
                "model_specific_claim_eligible": bool(
                    abs(effect) >= model_specific_threshold
                    and interval_same_direction
                ),
                "unexposed_accuracy": float(
                    model_unexposed_accuracy[model_index]
                ),
            }
        )

    seed_rows: list[dict[str, Any]] = []
    for model_index, model in enumerate(MODELS):
        for seed_index, seed in enumerate(MODEL_SEEDS):
            exposed = float(
                exposed_accuracy[model_index, seed_index].mean()
            )
            unexposed = float(
                unexposed_accuracy[model_index, seed_index].mean()
            )
            seed_rows.append(
                {
                    "architecture": model,
                    "model_seed": seed,
                    "exposed_accuracy": exposed,
                    "unexposed_accuracy": unexposed,
                    "effect_pp": 100.0 * (exposed - unexposed),
                }
            )

    function_rows: list[dict[str, Any]] = []
    for position, function_id in enumerate(function_ids):
        exposed = float(exposed_accuracy[:, :, position].mean())
        unexposed = float(unexposed_accuracy[:, :, position].mean())
        target = target_by_id[function_id]
        multiplier, intercept = (int(value) for value in target["affine_params"])
        function_rows.append(
            {
                "function_id": function_id,
                "multiplier": multiplier,
                "intercept": intercept,
                "alias_signature": target["alias_signature"],
                "exposed_accuracy": exposed,
                "unexposed_accuracy": unexposed,
                "effect_pp": 100.0 * (exposed - unexposed),
            }
        )

    signature_results: dict[str, Any] = {}
    signatures = sorted({row["alias_signature"] for row in targets})
    for signature in signatures:
        positions = [
            position
            for position, function_id in enumerate(function_ids)
            if target_by_id[function_id]["alias_signature"] == signature
        ]
        exposed = float(exposed_accuracy[:, :, positions].mean())
        unexposed = float(unexposed_accuracy[:, :, positions].mean())
        signature_results[signature] = {
            "function_count": len(positions),
            "exposed_accuracy": exposed,
            "unexposed_accuracy": unexposed,
            "effect_pp": 100.0 * (exposed - unexposed),
        }

    calibration_bins = int(spec["secondary"]["evaluation_calibration"]["bins"])
    calibration_by_architecture_seed: list[dict[str, Any]] = []
    for model_index, model in enumerate(MODELS):
        for seed_index, seed in enumerate(MODEL_SEEDS):
            exposed_calibration = calibration_metrics(
                exposed_accuracy[model_index, seed_index],
                exposed_confidence[model_index, seed_index],
                calibration_bins,
            )
            unexposed_calibration = calibration_metrics(
                unexposed_accuracy[model_index, seed_index],
                unexposed_confidence[model_index, seed_index],
                calibration_bins,
            )
            calibration_by_architecture_seed.append(
                {
                    "architecture": model,
                    "model_seed": seed,
                    "exposed": exposed_calibration,
                    "unexposed": unexposed_calibration,
                }
            )
    calibration_macro: dict[str, Any] = {}
    for metric_name in (
        "accuracy",
        "mean_confidence",
        "signed_gap_confidence_minus_accuracy",
        "ece",
    ):
        exposed_value = float(
            np.mean(
                [
                    row["exposed"][metric_name]
                    for row in calibration_by_architecture_seed
                ]
            )
        )
        unexposed_value = float(
            np.mean(
                [
                    row["unexposed"][metric_name]
                    for row in calibration_by_architecture_seed
                ]
            )
        )
        calibration_macro[metric_name] = {
            "exposed": exposed_value,
            "unexposed": unexposed_value,
            "exposed_minus_unexposed": exposed_value - unexposed_value,
        }

    secondary = {
        "brier_score": metric_summary(exposed_brier, unexposed_brier),
        "cross_entropy": metric_summary(exposed_ce, unexposed_ce),
        "effect_by_frozen_alias_signature": signature_results,
        "evaluation_calibration": {
            "bin_count": calibration_bins,
            "by_architecture_seed": calibration_by_architecture_seed,
            "macro_equal_architecture_seed": calibration_macro,
        },
        "training_calibration": {
            "available_descriptive_final_epoch": {
                "mean_train_accuracy": float(
                    np.mean([row["accuracy"] for row in training_final_rows])
                ),
                "mean_train_objective": float(
                    np.mean([row["loss"] for row in training_final_rows])
                ),
                "run_count": len(training_final_rows),
            },
            "reason": (
                "The frozen adapter did not persist training-set probabilities; "
                "loss and accuracy cannot identify calibration."
            ),
            "status": "not_estimable_from_frozen_artifacts",
        },
    }

    results = {
        "analysis_spec_sha256": sha256_file(spec_path),
        "architectures": architecture_rows,
        "gate": gate,
        "primary": {
            "bootstrap_replicates": int(spec["bootstrap"]["replicates"]),
            "ci95_pp": macro_interval,
            "effect_pp": macro_effect,
            "estimand": "architecture_macro_exact_exposure_effect",
            "orientation": "exposed_minus_unexposed",
            "unit": "percentage_points",
        },
        "protocol": "quality-enhancement-p1-v1",
        "provenance": {
            "full_validation_sha256": sha256_file(
                RUN_ROOT / "full_validation.json"
            ),
            "p1_checksum_manifest_sha256": sha256_file(
                P1_DIR / "SHA256SUMS"
            ),
            "preregistration_sha256": sha256_file(
                P1_DIR.parent / "preregistration.json"
            ),
        },
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "secondary": secondary,
        "status": "complete",
    }

    require(
        math.isclose(
            results["primary"]["effect_pp"],
            float(np.mean([row["effect_pp"] for row in architecture_rows])),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "macro point estimate is not the equal architecture mean",
    )
    require(len(seed_rows) == 30, "seed output row count")
    require(len(function_rows) == 40, "function output row count")

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    write_json(ANALYSIS_DIR / "results.json", results)
    write_csv(ANALYSIS_DIR / "model_effects.csv", architecture_rows)
    write_csv(ANALYSIS_DIR / "seed_effects.csv", seed_rows)
    write_csv(ANALYSIS_DIR / "function_effects.csv", function_rows)
    write_deterministic_npz(
        ANALYSIS_DIR / "bootstrap.npz",
        {
            "macro_effect_pp": macro_draws,
            "model_effect_pp": model_draws,
            "seed_resample_indices": resample_indices[
                "seed_resample_indices"
            ],
            "pair_resample_indices": resample_indices[
                "pair_resample_indices"
            ],
        },
    )
    (ANALYSIS_DIR / "report_cn.md").write_text(
        build_report(results),
        encoding="utf-8",
    )

    output_names = (
        "analysis_spec.json",
        "bootstrap.npz",
        "function_effects.csv",
        "model_effects.csv",
        "report_cn.md",
        "results.json",
        "seed_effects.csv",
    )
    checksum_lines = [
        f"{sha256_file(ANALYSIS_DIR / name)}  {name}" for name in output_names
    ]
    (ANALYSIS_DIR / "SHA256SUMS").write_text(
        "\n".join(checksum_lines) + "\n",
        encoding="ascii",
    )
    print(json.dumps(results, indent=2, sort_keys=True, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
