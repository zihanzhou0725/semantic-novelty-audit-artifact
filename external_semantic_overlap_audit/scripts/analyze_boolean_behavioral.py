#!/usr/bin/env python3
"""Analyze the frozen Boolean behavioral matrix under the registered Gate."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from freeze_boolean_behavioral_protocol import SCHEMA_VERSION, sha256_file


ANALYSIS_SCHEMA_VERSION = "boolean-behavioral-analysis-v1"
TASK_ORDER = (
    "conjunction_d28",
    "majority_d28",
    "sparse_disjunction_d50_k4_positive",
)
TASK_LABELS = {
    "conjunction_d28": "Conjunction-28",
    "majority_d28": "Majority-28",
    "sparse_disjunction_d50_k4_positive": "Sparse Disjunction-50",
}
PRIMARY_METRIC = "tail_accuracy"
SECONDARY_METRICS = (
    "final_position_accuracy",
    "all_positions_1_through_39_accuracy",
    "tail_binary_cross_entropy",
)


def fail(message: str) -> None:
    raise SystemExit(f"Behavioral analysis failed: {message}")


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def derive_seed(namespace: str) -> int:
    payload = f"{ANALYSIS_SCHEMA_VERSION}:{namespace}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def cluster_components(
    values: np.ndarray, semantic_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    _, inverse = np.unique(semantic_ids, return_inverse=True)
    cluster_count = int(inverse.max()) + 1
    sums = np.bincount(inverse, weights=values, minlength=cluster_count)
    counts = np.bincount(inverse, minlength=cluster_count)
    return sums, counts


def cluster_bootstrap_draws(
    values: np.ndarray,
    semantic_ids: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> np.ndarray:
    sums, counts = cluster_components(values, semantic_ids)
    cluster_count = len(sums)
    rng = np.random.Generator(np.random.PCG64(seed))
    result = np.empty(replicates, dtype=np.float64)
    batch_size = 256
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        indices = rng.integers(
            0, cluster_count, size=(stop - start, cluster_count)
        )
        result[start:stop] = sums[indices].sum(axis=1) / counts[indices].sum(
            axis=1
        )
    return result


def percentile_interval(
    draws: np.ndarray, confidence_level: float
) -> tuple[float, float]:
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(draws, [alpha / 2, 1.0 - alpha / 2])
    return float(lower), float(upper)


def hierarchical_bootstrap(
    paired_by_seed: dict[int, dict],
    *,
    replicates: int,
    confidence_level: float,
    task_name: str,
) -> dict:
    seeds = sorted(paired_by_seed)
    if seeds != [0, 1, 2]:
        fail(f"{task_name} does not contain model seeds 0, 1, and 2")
    rng = np.random.Generator(
        np.random.PCG64(derive_seed(f"{task_name}:hierarchical:seed-choice"))
    )
    seed_choices = rng.integers(0, len(seeds), size=(replicates, len(seeds)))
    conditional_draws = np.empty(
        (len(seeds), replicates, len(seeds)), dtype=np.float64
    )
    for seed_index, model_seed in enumerate(seeds):
        data = paired_by_seed[model_seed]
        draws = cluster_bootstrap_draws(
            data["primary_delta_fraction"],
            data["semantic_ids"],
            replicates=replicates * len(seeds),
            seed=derive_seed(
                f"{task_name}:hierarchical:clusters:model-seed-{model_seed}"
            ),
        )
        conditional_draws[seed_index] = draws.reshape(
            replicates, len(seeds)
        )
    selected = np.empty((replicates, len(seeds)), dtype=np.float64)
    row_indices = np.arange(replicates)
    for draw_index in range(len(seeds)):
        selected[:, draw_index] = conditional_draws[
            seed_choices[:, draw_index], row_indices, draw_index
        ]
    means_pp = selected.mean(axis=1) * 100.0
    lower, upper = percentile_interval(means_pp, confidence_level)
    return {
        "replicates": replicates,
        "mean_pp": float(means_pp.mean()),
        "ci95_pp": [lower, upper],
    }


def original_exposure_summary(rows: list[dict]) -> dict:
    seen = [row[PRIMARY_METRIC] for row in rows if row["seen_in_actual_training_stream"]]
    unseen = [
        row[PRIMARY_METRIC]
        for row in rows
        if not row["seen_in_actual_training_stream"]
    ]
    return {
        "seen_n": len(seen),
        "unseen_n": len(unseen),
        "seen_tail_accuracy": statistics.fmean(seen) if seen else None,
        "unseen_tail_accuracy": statistics.fmean(unseen) if unseen else None,
        "seen_minus_unseen_pp": (
            100.0 * (statistics.fmean(seen) - statistics.fmean(unseen))
            if seen and unseen
            else None
        ),
        "interpretation": "descriptive_only_not_a_randomized_contrast",
    }


def format_number(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def build_markdown(results: dict) -> str:
    lines = [
        "# Boolean ICL 严格语义划分行为实验",
        "",
        "分析日期：2026-10-08  ",
        "输入：18 个冻结 full runs；每个 run 20,000 steps、1,280 个固定 eval episodes。",
        "",
        "## 完整性",
        "",
        "- 18/18 runs 通过 checkpoint、输出哈希与有限值检查。",
        "- Strict train/eval semantic overlap 为 0。",
        "- Original/Strict paired semantic IDs、labels 与 exposure flags 完全一致。",
        "",
        "## 主指标",
        "",
        r"主指标为位置 20--39 的 tail accuracy，效应定义为",
        "",
        r"\[",
        r"\Delta_{\rm pp}=100(A_{\rm original}-A_{\rm strict}).",
        r"\]",
        "",
        "| Task | Seed 0 | Seed 1 | Seed 2 | Mean | Hierarchical 95% CI | Core Gate |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for task_name in TASK_ORDER:
        summary = results["tasks"][task_name]
        seed_values = [
            summary["per_seed"][str(seed)]["primary_delta_pp"]
            for seed in (0, 1, 2)
        ]
        ci = summary["hierarchical_bootstrap"]["ci95_pp"]
        lines.append(
            "| "
            + " | ".join(
                (
                    TASK_LABELS[task_name],
                    *(format_number(value) for value in seed_values),
                    format_number(summary["three_seed"]["mean_delta_pp"]),
                    f"[{format_number(ci[0])}, {format_number(ci[1])}]",
                    "Go" if summary["core_gate_pass"] else "No-Go",
                )
            )
            + " |"
        )

    gate = results["gate"]
    lines.extend(
        [
            "",
            "置信区间不参与 Gate；Gate 仅使用预注册的 `>=2 pp` 与 3/3 seeds "
            "同方向规则。",
            "",
            "## Gate",
            "",
            f"- Overall Core Gate: **{'Go' if gate['overall_core_go'] else 'No-Go'}**",
            f"- Passing tasks: {', '.join(TASK_LABELS[t] for t in gate['passing_tasks']) if gate['passing_tasks'] else 'none'}",
            f"- Strong-pattern Gate: **{'Go' if gate['strong_pattern_pass'] else 'No-Go'}**",
            f"- Transformer seeds 3/4 expansion: {', '.join(TASK_LABELS[t] for t in gate['transformer_expansion_tasks']) if gate['transformer_expansion_tasks'] else 'none'}",
            f"- LSTM ranking module: {'triggered' if gate['lstm_ranking_module_triggered'] else 'not triggered'}",
            "",
            "## 次指标",
            "",
            "| Task | Final acc Δ (pp) | Positions 1--39 Δ (pp) | Tail BCE Δ |",
            "|---|---:|---:|---:|",
        ]
    )
    for task_name in TASK_ORDER:
        secondary = results["tasks"][task_name]["secondary_three_seed_means"]
        lines.append(
            f"| {TASK_LABELS[task_name]} | "
            f"{format_number(secondary['final_position_accuracy_pp'])} | "
            f"{format_number(secondary['all_positions_1_through_39_accuracy_pp'])} | "
            f"{format_number(secondary['tail_binary_cross_entropy'], 4)} |"
        )
    lines.extend(
        [
            "",
            "BCE 使用 Original - Strict；负值表示 Original 更好。",
            "",
            "## Claim 边界",
            "",
            results["claim_decision"],
            "",
            "Original 内部 seen/unseen 分层仅作描述，因为 exposure 不是随机分配，"
            "不能替代 Original/Strict 受控对照。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    analysis_dir = root / "analysis" / SCHEMA_VERSION
    spec_path = analysis_dir / "analysis_spec.json"
    full_dir = root / "runs" / SCHEMA_VERSION / "full"
    validation_path = full_dir / "full_validation.json"
    protocol_dir = root / "protocols" / SCHEMA_VERSION
    analysis_dir.mkdir(parents=True, exist_ok=True)

    spec = json.loads(spec_path.read_text())
    if spec["schema_version"] != ANALYSIS_SCHEMA_VERSION:
        fail("analysis spec schema version differs")
    if spec["status"] != "frozen_before_paired_outcome_analysis":
        fail("analysis spec is not frozen")
    validation = json.loads(validation_path.read_text())
    if validation["status"] != "passed":
        fail("full matrix validation did not pass")
    if validation["effect_metrics_compared"]:
        fail("full validation unexpectedly compared effect metrics")

    with (protocol_dir / "run_manifest.csv").open(newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    rows_by_key = {
        (
            row["task"],
            int(row["model_seed"]),
            row["protocol"],
        ): row
        for row in manifest_rows
    }

    prediction_rows = {}
    for key, manifest_row in rows_by_key.items():
        prediction_rows[key] = load_jsonl(
            full_dir / manifest_row["run_id"] / "eval_predictions.jsonl"
        )

    per_episode_output = []
    task_results = {}
    paired_for_hierarchy = {}
    cluster_replicates = spec["per_seed_cluster_bootstrap"]["replicates"]
    confidence_level = spec["per_seed_cluster_bootstrap"]["confidence_level"]
    for task_name in TASK_ORDER:
        per_seed = {}
        hierarchy_input = {}
        secondary_seed_effects = defaultdict(list)
        for model_seed in (0, 1, 2):
            original = prediction_rows[(task_name, model_seed, "original")]
            strict = prediction_rows[
                (task_name, model_seed, "strict_function_disjoint")
            ]
            if len(original) != len(strict):
                fail(f"{task_name} seed {model_seed} paired row counts differ")

            semantic_ids = []
            primary_delta = []
            metric_values = defaultdict(list)
            for original_row, strict_row in zip(original, strict):
                for field in (
                    "eval_index",
                    "semantic_id",
                    "labels",
                    "seen_in_original_training_stream",
                ):
                    if original_row[field] != strict_row[field]:
                        fail(
                            f"{task_name} seed {model_seed} paired field {field} differs"
                        )
                semantic_ids.append(original_row["semantic_id"])
                episode = {
                    "task": task_name,
                    "model_seed": model_seed,
                    "eval_index": original_row["eval_index"],
                    "semantic_id": original_row["semantic_id"],
                    "seen_in_original_training_stream": original_row[
                        "seen_in_original_training_stream"
                    ],
                }
                for metric_name in (PRIMARY_METRIC, *SECONDARY_METRICS):
                    original_value = original_row[metric_name]
                    strict_value = strict_row[metric_name]
                    delta = original_value - strict_value
                    metric_values[metric_name].append(delta)
                    episode[f"original_{metric_name}"] = original_value
                    episode[f"strict_{metric_name}"] = strict_value
                    episode[f"delta_{metric_name}"] = delta
                primary_delta.append(episode[f"delta_{PRIMARY_METRIC}"])
                per_episode_output.append(episode)

            semantic_array = np.asarray(semantic_ids)
            primary_array = np.asarray(primary_delta, dtype=np.float64)
            bootstrap_pp = (
                cluster_bootstrap_draws(
                    primary_array,
                    semantic_array,
                    replicates=cluster_replicates,
                    seed=derive_seed(
                        f"{task_name}:model-seed-{model_seed}:cluster-bootstrap"
                    ),
                )
                * 100.0
            )
            lower, upper = percentile_interval(
                bootstrap_pp, confidence_level
            )
            original_accuracy = statistics.fmean(
                row[PRIMARY_METRIC] for row in original
            )
            strict_accuracy = statistics.fmean(
                row[PRIMARY_METRIC] for row in strict
            )
            seed_result = {
                "original_tail_accuracy": original_accuracy,
                "strict_tail_accuracy": strict_accuracy,
                "primary_delta_pp": float(primary_array.mean() * 100.0),
                "cluster_bootstrap": {
                    "replicates": cluster_replicates,
                    "semantic_cluster_count": len(np.unique(semantic_array)),
                    "ci95_pp": [lower, upper],
                },
                "secondary_effects": {
                    metric_name: float(np.mean(values))
                    for metric_name, values in metric_values.items()
                    if metric_name != PRIMARY_METRIC
                },
                "original_exposure_stratification": original_exposure_summary(
                    original
                ),
            }
            per_seed[str(model_seed)] = seed_result
            hierarchy_input[model_seed] = {
                "primary_delta_fraction": primary_array,
                "semantic_ids": semantic_array,
            }
            for metric_name in SECONDARY_METRICS:
                secondary_seed_effects[metric_name].append(
                    seed_result["secondary_effects"][metric_name]
                )

        seed_deltas = [
            per_seed[str(seed)]["primary_delta_pp"] for seed in (0, 1, 2)
        ]
        mean_delta = statistics.fmean(seed_deltas)
        sample_sd = statistics.stdev(seed_deltas)
        half_width = (
            spec["three_seed_summary"]["t_interval"]["critical_value"]
            * sample_sd
            / math.sqrt(3)
        )
        hierarchical = hierarchical_bootstrap(
            hierarchy_input,
            replicates=spec["hierarchical_bootstrap"]["replicates"],
            confidence_level=spec["hierarchical_bootstrap"][
                "confidence_level"
            ],
            task_name=task_name,
        )
        core_gate_pass = mean_delta >= 2.0 and all(
            delta > 0.0 for delta in seed_deltas
        )
        task_results[task_name] = {
            "per_seed": per_seed,
            "three_seed": {
                "mean_delta_pp": mean_delta,
                "sample_sd_pp": sample_sd,
                "t_ci95_pp": [
                    mean_delta - half_width,
                    mean_delta + half_width,
                ],
            },
            "hierarchical_bootstrap": hierarchical,
            "secondary_three_seed_means": {
                "final_position_accuracy_pp": 100.0
                * statistics.fmean(
                    secondary_seed_effects["final_position_accuracy"]
                ),
                "all_positions_1_through_39_accuracy_pp": 100.0
                * statistics.fmean(
                    secondary_seed_effects[
                        "all_positions_1_through_39_accuracy"
                    ]
                ),
                "tail_binary_cross_entropy": statistics.fmean(
                    secondary_seed_effects["tail_binary_cross_entropy"]
                ),
            },
            "core_gate_pass": core_gate_pass,
        }
        paired_for_hierarchy[task_name] = hierarchy_input

    passing_tasks = [
        task_name
        for task_name in TASK_ORDER
        if task_results[task_name]["core_gate_pass"]
    ]
    task_means = {
        task_name: task_results[task_name]["three_seed"]["mean_delta_pp"]
        for task_name in TASK_ORDER
    }
    strong_pattern = (
        task_means["sparse_disjunction_d50_k4_positive"] >= 5.0
        and abs(task_means["conjunction_d28"]) <= 2.0
        and task_means["conjunction_d28"]
        <= task_means["majority_d28"]
        <= task_means["sparse_disjunction_d50_k4_positive"]
    )
    sparse_pass = (
        "sparse_disjunction_d50_k4_positive" in passing_tasks
    )
    overall_core_go = bool(passing_tasks)
    if overall_core_go:
        claim_decision = (
            "Core Go：允许仅针对通过 Gate 的任务表述“该受测 Transformer "
            "受益于训练期接触相同 evaluation semantics”。不得外推到原论文"
            "未公开 checkpoint、其他模型族或所有存在 overlap 的 benchmark。"
        )
    else:
        claim_decision = (
            "Core No-Go：不得声称该受测 Transformer 依赖 semantic exposure，"
            "行为影响部分按预注册规则停止扩展。"
        )

    results = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis_spec_sha256": sha256_file(spec_path),
        "full_validation_sha256": sha256_file(validation_path),
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "primary_metric": spec["primary_effect"],
        "tasks": task_results,
        "gate": {
            "overall_core_go": overall_core_go,
            "no_go": not overall_core_go,
            "passing_tasks": passing_tasks,
            "strong_pattern_pass": strong_pattern,
            "transformer_expansion_tasks": passing_tasks,
            "lstm_ranking_module_triggered": sparse_pass,
            "ranking_change_claim_allowed_now": False,
        },
        "claim_decision": claim_decision,
    }

    results_path = analysis_dir / "results.json"
    paired_path = analysis_dir / "paired_episode_results.csv"
    report_path = analysis_dir / "report_cn.md"
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    with paired_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(per_episode_output[0])
        )
        writer.writeheader()
        writer.writerows(per_episode_output)
    report_path.write_text(build_markdown(results))

    checksum_paths = (spec_path, results_path, paired_path, report_path)
    (analysis_dir / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in checksum_paths)
    )
    print(json.dumps(results, indent=2, sort_keys=True))
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
