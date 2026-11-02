#!/usr/bin/env python3
"""Generate exploratory CLUTRR pattern-level heterogeneity artifacts."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toolkit.external_behavior.common import iter_jsonl  # noqa: E402
from toolkit.tmlr_strengthening.artifacts import (  # noqa: E402
    run_info,
    write_csv,
    write_json,
    write_macros,
    write_sha256sums,
)
from toolkit.tmlr_strengthening.clutrr_heterogeneity import (  # noqa: E402
    analyze_clutrr,
)
from toolkit.paper_style import (  # noqa: E402
    PALETTE,
    apply,
    categorical_colors,
    figsize,
)


INPUT_DIR = ROOT / "analysis" / "external-behavior-v4" / "effect-analysis-v1"
CLASSES_PATH = (
    ROOT
    / "manifests"
    / "external-behavior-v4"
    / "execution-v1"
    / "classes.json"
)
COMMON_BASE = (
    ROOT
    / "manifests"
    / "external-behavior-v4"
    / "materialization-v1"
    / "data"
    / "common_base.jsonl"
)
OUTPUT_DIR = (
    ROOT / "analysis" / "tmlr-strengthening-v1" / "t2_clutrr_heterogeneity"
)


def main() -> None:
    apply()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pattern_path = INPUT_DIR / "pattern_effects.csv"
    seed_path = INPUT_DIR / "seed_pattern_effects.csv"
    table, summary, covariates = analyze_clutrr(
        pattern_path,
        seed_path,
        CLASSES_PATH,
    )
    input_paths = [pattern_path, seed_path, CLASSES_PATH]
    compact_counts = ROOT / "artifacts" / "tmlr_clutrr_common_relation_counts.json"
    if COMMON_BASE.exists() or compact_counts.exists():
        if COMMON_BASE.exists():
            counts = Counter(
                row["target_relation"] for row in iter_jsonl(COMMON_BASE)
            )
            input_paths.append(COMMON_BASE)
        else:
            counts = json.loads(compact_counts.read_text(encoding="ascii"))
            input_paths.append(compact_counts)
        values = np.asarray(
            [counts[str(row["target_relation"])] for row in table],
            dtype=float,
        )
        effects = np.asarray(
            [float(row["effect_pp"]) for row in table],
            dtype=float,
        )
        correlation = spearmanr(effects, values)
        summary["common_relation_count"] = {
            "status": "computed",
            "spearman": float(correlation.statistic),
            "pvalue": float(correlation.pvalue),
        }
        for row, value in zip(table, values):
            row["common_relation_count"] = int(value)
    else:
        summary["common_relation_count"] = {
            "status": "skipped: common_base.jsonl not present"
        }

    for row in covariates:
        details = summary["covariates"][str(row["covariate"])]
        row["spearman"] = details["spearman"]
        row["pvalue"] = details["pvalue"]
    output_columns = (
        "semantic_id",
        "target_relation",
        "e1",
        "e2",
        "e3",
        "ea",
        "ua",
        "effect_pp",
        "norm_gain",
        "block_index",
        "pattern_index",
        "in_edges",
        "distinct",
        "common_relation_count",
    )
    output_rows = [
        {name: row.get(name, "") for name in output_columns}
        for row in table
    ]
    write_csv(OUTPUT_DIR / "pattern_table.csv", output_rows)
    write_json(OUTPUT_DIR / "summary.json", summary)
    write_csv(OUTPUT_DIR / "covariates.csv", covariates)

    effects = np.asarray([float(row["effect_pp"]) for row in table])
    baselines = 100.0 * np.asarray([float(row["ua"]) for row in table])
    gains = np.asarray([float(row["norm_gain"]) for row in table])
    relations = sorted({str(row["target_relation"]) for row in table})
    colors = categorical_colors(len(relations))
    color_by_relation = dict(zip(relations, colors))
    figure, axes = plt.subplots(1, 2, figsize=figsize("full"))
    for relation in relations:
        selected = np.asarray(
            [row["target_relation"] == relation for row in table]
        )
        axes[0].scatter(
            baselines[selected],
            effects[selected],
            s=22,
            alpha=0.82,
            color=color_by_relation[relation],
            edgecolors="none",
        )
    x_values = np.linspace(0.0, 100.0, 101)
    axes[0].plot(
        x_values,
        100.0 - x_values,
        color=PALETTE["charcoal"],
        linewidth=1.0,
        linestyle="--",
        label="headroom ceiling",
    )
    axes[0].set(
        xlabel="Unexposed accuracy (%)",
        ylabel="Exposure effect (pp)",
        xlim=(0.0, 102.0),
    )
    axes[0].legend(frameon=False, fontsize=7, loc="upper right")
    axes[0].text(
        -0.04,
        1.04,
        "(a)",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontweight="bold",
        clip_on=False,
        zorder=10,
    )
    axes[1].hist(
        gains[np.isfinite(gains)],
        bins=20,
        color=PALETTE["blue"],
        edgecolor="white",
        linewidth=0.5,
    )
    axes[1].axvline(0.0, color=PALETTE["charcoal"], linewidth=0.8)
    axes[1].set(
        xlabel="Normalized gain",
        ylabel="Patterns",
    )
    axes[1].text(
        -0.04,
        1.04,
        "(b)",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontweight="bold",
        clip_on=False,
        zorder=10,
    )
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=7)
        axis.xaxis.label.set_size(8)
        axis.yaxis.label.set_size(8)
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "fig_clutrr_heterogeneity.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)

    in_edges = {
        int(row["value"]): row
        for row in summary["covariates"]["in_edges"]["groups"]
    }
    distinct = {
        int(row["value"]): row
        for row in summary["covariates"]["distinct"]["groups"]
    }
    write_macros(
        OUTPUT_DIR / "macros.tex",
        {
            "TTwoPatterns": str(summary["pattern_count"]),
            "TTwoMean": f"{summary['effect_mean_pp']:.2f}",
            "TTwoMedian": f"{summary['effect_median_pp']:.1f}",
            "TTwoSD": f"{summary['effect_sd_pp']:.1f}",
            "TTwoPositive": str(summary["positive_effect_count"]),
            "TTwoMin": f"{summary['effect_min_pp']:.1f}",
            "TTwoMax": f"{summary['effect_max_pp']:.1f}",
            "TTwoSpearman": f"{summary['effect_baseline_spearman']:.3f}",
            "TTwoNGMedian": f"{summary['norm_gain_median']:.3f}",
            "TTwoBaseRSquared": f"{summary['baseline_ols_r_squared']:.3f}",
            "TTwoRelationRSquared": (
                f"{summary['baseline_relation_ols_r_squared']:.3f}"
            ),
            "TTwoLeaveMin": (
                f"{summary['leave_one_relation_mean_min_pp']:.2f}"
            ),
            "TTwoLeaveMax": (
                f"{summary['leave_one_relation_mean_max_pp']:.2f}"
            ),
            "TTwoBaseline": f"{summary['baseline_mean_pp'] / 100.0:.3f}",
            "TTwoEta": f"{summary['relation_eta_squared']:.3f}",
            "TTwoInAbsentN": str(in_edges[0]["count"]),
            "TTwoInAbsentMean": f"{in_edges[0]['effect_mean_pp']:.2f}",
            "TTwoInPresentN": str(in_edges[1]["count"]),
            "TTwoInPresentMean": f"{in_edges[1]['effect_mean_pp']:.2f}",
            "TTwoInSpearman": (
                f"{summary['covariates']['in_edges']['spearman']:.3f}"
            ),
            "TTwoInP": (
                f"{summary['covariates']['in_edges']['pvalue']:.3f}"
            ),
            "TTwoDistinctOneN": str(distinct[1]["count"]),
            "TTwoDistinctOneMean": (
                f"{distinct[1]['effect_mean_pp']:.2f}"
            ),
            "TTwoDistinctTwoN": str(distinct[2]["count"]),
            "TTwoDistinctTwoMean": (
                f"{distinct[2]['effect_mean_pp']:.2f}"
            ),
            "TTwoDistinctThreeN": str(distinct[3]["count"]),
            "TTwoDistinctThreeMean": (
                f"{distinct[3]['effect_mean_pp']:.2f}"
            ),
            "TTwoDistinctSpearman": (
                f"{summary['covariates']['distinct']['spearman']:.3f}"
            ),
            "TTwoDistinctP": (
                f"{summary['covariates']['distinct']['pvalue']:.3f}"
            ),
        },
    )
    write_json(
        OUTPUT_DIR / "run_info.json",
        run_info(
            seed=None,
            script_path=(
                "scripts/tmlr_strengthening/"
                "run_t2_clutrr_heterogeneity.py"
            ),
            input_paths=input_paths,
        ),
    )
    write_sha256sums(OUTPUT_DIR)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
