"""Function-level heterogeneity for the internal exact-exposure study."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def _group_summary(
    rows: list[dict[str, object]],
    column: str,
) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row[column])].append(float(row["effect_pp"]))
    return {
        key: {
            "count": len(values),
            "effect_mean_pp": float(np.mean(values)),
        }
        for key, values in sorted(grouped.items())
    }


def analyze_internal(
    function_effects_path: Path,
) -> tuple[
    list[dict[str, object]],
    dict[str, float | int],
    dict[str, dict[str, dict[str, float | int]]],
]:
    """Return typed function rows, aggregate statistics, and group summaries."""

    with function_effects_path.open(newline="", encoding="ascii") as handle:
        source = list(csv.DictReader(handle))
    required = {
        "function_id",
        "multiplier",
        "intercept",
        "alias_signature",
        "exposed_accuracy",
        "unexposed_accuracy",
        "effect_pp",
    }
    if len(source) != 40 or not required <= set(source[0]):
        raise ValueError("Expected the frozen 40-row function-effects table.")

    table: list[dict[str, object]] = []
    for row in source:
        ea = float(row["exposed_accuracy"])
        ua = float(row["unexposed_accuracy"])
        effect = float(row["effect_pp"])
        if abs(100.0 * (ea - ua) - effect) >= 1e-9:
            raise ValueError(f"Effect mismatch for {row['function_id']}")
        table.append(
            {
                "function_id": row["function_id"],
                "multiplier": int(row["multiplier"]),
                "intercept": int(row["intercept"]),
                "alias_signature": row["alias_signature"],
                "exposed_accuracy": ea,
                "unexposed_accuracy": ua,
                "effect_pp": effect,
                "norm_gain": (
                    math.nan if ua == 1.0 else (ea - ua) / (1.0 - ua)
                ),
            }
        )

    effects = np.asarray([float(row["effect_pp"]) for row in table])
    baselines = np.asarray(
        [float(row["unexposed_accuracy"]) for row in table]
    )
    gains = np.asarray([float(row["norm_gain"]) for row in table])
    finite_gains = gains[np.isfinite(gains)]
    correlation = spearmanr(effects, baselines)
    summary: dict[str, float | int] = {
        "function_count": len(table),
        "effect_mean_pp": float(effects.mean()),
        "effect_sd_pp": float(effects.std(ddof=1)),
        "effect_median_pp": float(np.median(effects)),
        "effect_min_pp": float(effects.min()),
        "effect_max_pp": float(effects.max()),
        "positive_effect_count": int((effects > 0).sum()),
        "baseline_mean": float(baselines.mean()),
        "effect_baseline_spearman": float(correlation.statistic),
        "effect_baseline_spearman_pvalue": float(correlation.pvalue),
        "norm_gain_mean": float(finite_gains.mean()),
        "norm_gain_median": float(np.median(finite_gains)),
        "norm_gain_min": float(finite_gains.min()),
        "norm_gain_max": float(finite_gains.max()),
        "norm_gain_nan_count": int((~np.isfinite(gains)).sum()),
    }
    groups = {
        "multiplier": _group_summary(table, "multiplier"),
        "alias_signature": _group_summary(table, "alias_signature"),
    }
    return table, summary, groups

