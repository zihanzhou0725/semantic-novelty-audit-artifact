#!/usr/bin/env python3
"""Generate parameterized sensitivity bounds using structural SLR counts only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toolkit.tmlr_strengthening.artifacts import (  # noqa: E402
    run_info, write_csv, write_json, write_macros, write_sha256sums,
)
from toolkit.paper_style import PALETTE, apply, figsize  # noqa: E402

INPUT_PATH = ROOT / "artifacts" / "slr_overlap_results.json"
OUTPUT_DIR = ROOT / "analysis" / "tmlr-strengthening-v1" / "t10_novelty_bounds"


def main() -> None:
    apply()
    audit = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    item_count = (
        audit["rows"]["test"]
        - audit["sound_semantic_overlap_union"]["test_overlap_rows"]
    )
    if item_count != 548:
        raise ValueError("Expected 548 unresolved items from the structural audit.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    grid = []
    # The widest binary interval is attained at p=1/2 because M is even.
    for upper_exposed in range(item_count):
        novel_count = item_count - upper_exposed
        grid.append({
            "M": item_count,
            "U": upper_exposed,
            "n": novel_count,
            "rho": upper_exposed / item_count,
            "maximum_width": min(1.0, upper_exposed / novel_count),
        })
    write_csv(OUTPUT_DIR / "slr_novelty_grid.csv", grid)
    summary = {
        "analysis_type": "parameterized_sensitivity_not_model_performance",
        "unresolved_item_count": item_count,
        "lower_formula": "max(0,p*M-U)/(M-U)",
        "upper_formula": "min(p*M,M-U)/(M-U)",
        "maximum_width_formula": "min(1,U/(M-U))",
        "finite_binary_domain": "p*M must be an integer",
        "score_artifacts_used": False,
    }
    write_json(OUTPUT_DIR / "summary.json", summary)

    figure, axis = plt.subplots(figsize=figsize("panel"))
    axis.plot(
        [row["rho"] for row in grid],
        [row["maximum_width"] for row in grid],
        color=PALETTE["blue"], linewidth=1.2,
    )
    axis.set(
        xlabel=r"Assumed exposure upper share $\rho=U/M$",
        ylabel="Maximum interval width",
        xlim=(0.0, 1.0), ylim=(0.0, 1.05),
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=7)
    axis.xaxis.label.set_size(8)
    axis.yaxis.label.set_size(8)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "fig_slr_novelty_bounds.pdf", bbox_inches="tight")
    plt.close(figure)
    write_macros(OUTPUT_DIR / "macros.tex", {"TTenM": str(item_count)})
    write_json(
        OUTPUT_DIR / "run_info.json",
        run_info(
            seed=None,
            script_path="scripts/tmlr_strengthening/run_t10_novelty_bounds.py",
            input_paths=[INPUT_PATH],
        ),
    )
    write_sha256sums(OUTPUT_DIR)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
