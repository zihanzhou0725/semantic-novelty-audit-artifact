#!/usr/bin/env python3
"""Generate exploratory internal function-level heterogeneity artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toolkit.tmlr_strengthening.artifacts import (  # noqa: E402
    run_info,
    write_csv,
    write_json,
    write_macros,
    write_sha256sums,
)
from toolkit.tmlr_strengthening.internal_heterogeneity import (  # noqa: E402
    analyze_internal,
)
from toolkit.paper_style import PALETTE, apply, figsize  # noqa: E402


INPUT_DIR = ROOT / "analysis" / "quality-enhancement-v1" / "p1"
TWO_DIR = (
    ROOT / "analysis" / "tmlr-strengthening-v1" / "t2_clutrr_heterogeneity"
)
OUTPUT_DIR = (
    ROOT / "analysis" / "tmlr-strengthening-v1" / "t3_internal_heterogeneity"
)


def main() -> None:
    apply()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    function_path = INPUT_DIR / "function_effects.csv"
    model_path = INPUT_DIR / "model_effects.csv"
    t2_summary_path = TWO_DIR / "summary.json"
    if not t2_summary_path.exists():
        raise FileNotFoundError("Run task 2 before task 3.")
    table, summary, groups = analyze_internal(function_path)
    summary["groups"] = groups
    summary["displacement_status"] = (
        "not run: excluded from the user-approved strengthening scope"
    )
    write_csv(OUTPUT_DIR / "function_table.csv", table)
    write_json(OUTPUT_DIR / "summary_heterogeneity.json", summary)

    t2 = json.loads(t2_summary_path.read_text(encoding="ascii"))
    cross_setting = [
        {
            "setting": "CLUTRR",
            "unit_count": t2["pattern_count"],
            "mean_unexposed_accuracy": t2["baseline_mean_pp"] / 100.0,
            "mean_effect_pp": t2["effect_mean_pp"],
            "effect_baseline_spearman": t2[
                "effect_baseline_spearman"
            ],
            "norm_gain_median": t2["norm_gain_median"],
        },
        {
            "setting": "Internal finite-function intervention",
            "unit_count": summary["function_count"],
            "mean_unexposed_accuracy": summary["baseline_mean"],
            "mean_effect_pp": summary["effect_mean_pp"],
            "effect_baseline_spearman": summary[
                "effect_baseline_spearman"
            ],
            "norm_gain_median": summary["norm_gain_median"],
        },
    ]
    write_csv(OUTPUT_DIR / "cross_setting_table.csv", cross_setting)

    baselines = 100.0 * np.asarray(
        [float(row["unexposed_accuracy"]) for row in table]
    )
    effects = np.asarray([float(row["effect_pp"]) for row in table])
    multipliers = np.asarray([int(row["multiplier"]) for row in table])
    figure, axis = plt.subplots(figsize=figsize("panel"))
    for multiplier, color in (
        (1, PALETTE["blue"]),
        (11, PALETTE["orange"]),
    ):
        selected = multipliers == multiplier
        axis.scatter(
            baselines[selected],
            effects[selected],
            s=28,
            alpha=0.82,
            color=color,
            edgecolors="none",
            label=f"multiplier {multiplier}",
        )
    axis.set(
        xlabel="Unexposed accuracy (%)",
        ylabel="Exposure effect (pp)",
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=7)
    axis.xaxis.label.set_size(8)
    axis.yaxis.label.set_size(8)
    axis.legend(frameon=False, fontsize=7)
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "fig_internal_heterogeneity.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)

    write_macros(
        OUTPUT_DIR / "macros.tex",
        {
            "TThreeFunctions": str(summary["function_count"]),
            "TThreeBaseline": f"{summary['baseline_mean']:.3f}",
            "TThreeMean": f"{summary['effect_mean_pp']:.2f}",
            "TThreeSD": f"{summary['effect_sd_pp']:.2f}",
            "TThreeMedian": f"{summary['effect_median_pp']:.2f}",
            "TThreeMin": f"{summary['effect_min_pp']:.2f}",
            "TThreeMax": f"{summary['effect_max_pp']:.2f}",
            "TThreePositive": str(summary["positive_effect_count"]),
            "TThreeSpearman": (
                f"{summary['effect_baseline_spearman']:.3f}"
            ),
            "TThreeNGMean": f"{summary['norm_gain_mean']:.3f}",
            "TThreeNGMedian": f"{summary['norm_gain_median']:.3f}",
        },
    )
    write_json(
        OUTPUT_DIR / "run_info.json",
        run_info(
            seed=None,
            script_path=(
                "scripts/tmlr_strengthening/"
                "run_t3_internal_heterogeneity.py"
            ),
            input_paths=[
                function_path,
                model_path,
                t2_summary_path,
            ],
        ),
    )
    write_sha256sums(OUTPUT_DIR)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
