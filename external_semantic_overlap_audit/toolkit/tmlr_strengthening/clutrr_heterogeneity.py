"""Pattern-level heterogeneity analysis for the directional CLUTRR study."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.stats import spearmanr


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="ascii") as handle:
        return list(csv.DictReader(handle))


def _load_classes(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="ascii"))
    if isinstance(value, dict):
        value = value.get("classes", next(iter(value.values())))
    classes = [str(item) for item in value]
    if len(classes) != 23 or "no-relation" not in classes:
        raise ValueError("Expected the frozen 23-class CLUTRR vocabulary.")
    return classes


def _semantic_id(target_relation: str, edges: Sequence[str]) -> str:
    payload = json.dumps(
        [target_relation, *edges],
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def decode_semantic_id(
    semantic_id: str,
    target_relation: str,
    classes: Sequence[str],
) -> tuple[str, str, str]:
    """Invert one frozen four-relation semantic hash by exhaustive search."""

    for edges in itertools.product(classes, repeat=3):
        if _semantic_id(target_relation, edges) == semantic_id:
            return edges
    raise ValueError(f"Could not decode semantic_id {semantic_id}")


def _decoder(
    targets: Iterable[str],
    classes: Sequence[str],
) -> dict[tuple[str, str], tuple[str, str, str]]:
    result: dict[tuple[str, str], tuple[str, str, str]] = {}
    for target in sorted(set(targets)):
        for edges in itertools.product(classes, repeat=3):
            result[(target, _semantic_id(target, edges))] = edges
    return result


def eta_squared(values: Sequence[float], groups: Sequence[str]) -> float:
    """Return the one-way between-group share of total sum of squares."""

    y = np.asarray(values, dtype=float)
    if len(y) != len(groups) or not len(y):
        raise ValueError("values and groups must be nonempty and aligned")
    grand = float(y.mean())
    total = float(np.square(y - grand).sum())
    if total == 0.0:
        return 0.0
    between = 0.0
    for group in sorted(set(groups)):
        selected = y[np.asarray([item == group for item in groups])]
        between += len(selected) * float(selected.mean() - grand) ** 2
    return between / total


def _r_squared(y: np.ndarray, design: np.ndarray) -> float:
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ coefficients
    total = y - y.mean()
    denominator = float(total @ total)
    return 1.0 if denominator == 0.0 else 1.0 - float(residual @ residual) / denominator


def analyze_clutrr(
    pattern_effects_path: Path,
    seed_pattern_effects_path: Path,
    classes_path: Path,
) -> tuple[list[dict[str, object]], dict[str, object], list[dict[str, object]]]:
    """Compute pattern rows, aggregate summary, and structural covariates."""

    patterns = _read_csv(pattern_effects_path)
    seed_rows = _read_csv(seed_pattern_effects_path)
    if len(patterns) != 80 or len(seed_rows) != 640:
        raise ValueError("Expected 80 patterns and 640 seed-pattern rows.")
    required_pattern = {
        "semantic_id",
        "target_relation",
        "effect_pp",
        "block_index",
        "pattern_index",
    }
    required_seed = {
        "semantic_id",
        "model_seed",
        "exposed_accuracy",
        "unexposed_accuracy",
    }
    if not required_pattern <= set(patterns[0]) or not required_seed <= set(
        seed_rows[0]
    ):
        raise ValueError("CLUTRR analysis CSV columns changed.")

    seed_by_semantic: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in seed_rows:
        seed_by_semantic[row["semantic_id"]].append(row)
    classes = _load_classes(classes_path)
    decoder = _decoder(
        (row["target_relation"] for row in patterns),
        classes,
    )

    table: list[dict[str, object]] = []
    for row in sorted(patterns, key=lambda item: int(item["pattern_index"])):
        semantic_id = row["semantic_id"]
        observations = seed_by_semantic[semantic_id]
        if len(observations) != 8:
            raise ValueError(f"Expected eight seeds for {semantic_id}")
        ea = float(
            np.mean([float(item["exposed_accuracy"]) for item in observations])
        )
        ua = float(
            np.mean([float(item["unexposed_accuracy"]) for item in observations])
        )
        effect = float(row["effect_pp"])
        if abs(100.0 * (ea - ua) - effect) >= 1e-6:
            raise ValueError(f"Effect mismatch for {semantic_id}")
        edges = decoder.get((row["target_relation"], semantic_id))
        if edges is None:
            raise ValueError(f"Could not decode semantic_id {semantic_id}")
        norm_gain = math.nan if ua == 1.0 else (ea - ua) / (1.0 - ua)
        table.append(
            {
                "semantic_id": semantic_id,
                "target_relation": row["target_relation"],
                "e1": edges[0],
                "e2": edges[1],
                "e3": edges[2],
                "ea": ea,
                "ua": ua,
                "effect_pp": effect,
                "norm_gain": norm_gain,
                "block_index": int(row["block_index"]),
                "pattern_index": int(row["pattern_index"]),
                "in_edges": int(row["target_relation"] in edges),
                "distinct": len(set(edges)),
            }
        )

    effects = np.asarray([float(row["effect_pp"]) for row in table])
    baselines = 100.0 * np.asarray([float(row["ua"]) for row in table])
    gains = np.asarray(
        [float(row["norm_gain"]) for row in table],
        dtype=float,
    )
    relations = [str(row["target_relation"]) for row in table]
    spearman = spearmanr(effects, baselines)

    relation_levels = sorted(set(relations))
    relation_design = np.column_stack(
        [
            np.ones(len(table)),
            baselines,
            *[
                np.asarray([value == relation for value in relations], dtype=float)
                for relation in relation_levels[1:]
            ],
        ]
    )
    baseline_design = np.column_stack([np.ones(len(table)), baselines])
    leave_one_out = [
        float(effects[np.asarray([value != relation for value in relations])].mean())
        for relation in relation_levels
    ]

    covariates: list[dict[str, object]] = []
    covariate_summary: dict[str, object] = {}
    for name in ("in_edges", "distinct"):
        values = np.asarray([int(row[name]) for row in table])
        correlation = spearmanr(effects, values)
        grouped: list[dict[str, object]] = []
        for value in sorted(set(values.tolist())):
            selected = effects[values == value]
            group = {
                "covariate": name,
                "value": int(value),
                "count": int(len(selected)),
                "effect_mean_pp": float(selected.mean()),
            }
            grouped.append(group)
            covariates.append(group)
        covariate_summary[name] = {
            "groups": grouped,
            "spearman": float(correlation.statistic),
            "pvalue": float(correlation.pvalue),
        }

    block_means = {
        str(block): float(
            effects[
                np.asarray(
                    [int(row["block_index"]) == block for row in table]
                )
            ].mean()
        )
        for block in (0, 1)
    }
    finite_gains = gains[np.isfinite(gains)]
    summary: dict[str, object] = {
        "pattern_count": len(table),
        "decoded_count": len(table),
        "effect_mean_pp": float(effects.mean()),
        "effect_median_pp": float(np.median(effects)),
        "effect_sd_pp": float(effects.std(ddof=1)),
        "effect_min_pp": float(effects.min()),
        "effect_max_pp": float(effects.max()),
        "positive_effect_count": int((effects > 0).sum()),
        "baseline_mean_pp": float(baselines.mean()),
        "effect_baseline_spearman": float(spearman.statistic),
        "effect_baseline_spearman_pvalue": float(spearman.pvalue),
        "norm_gain_mean": float(finite_gains.mean()),
        "norm_gain_median": float(np.median(finite_gains)),
        "norm_gain_min": float(finite_gains.min()),
        "norm_gain_max": float(finite_gains.max()),
        "norm_gain_nan_count": int((~np.isfinite(gains)).sum()),
        "relation_eta_squared": eta_squared(effects, relations),
        "leave_one_relation_mean_min_pp": min(leave_one_out),
        "leave_one_relation_mean_max_pp": max(leave_one_out),
        "baseline_ols_r_squared": _r_squared(effects, baseline_design),
        "baseline_relation_ols_r_squared": _r_squared(
            effects,
            relation_design,
        ),
        "block_effect_mean_pp": block_means,
        "covariates": covariate_summary,
    }
    return table, summary, covariates

