#!/usr/bin/env python3
"""Copy generated macros and figures into the TMLR source tree."""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT.parent / "semantic_novelty_audit_paper" / "tmlr"
ANALYSIS_DIR = ROOT / "analysis" / "tmlr-strengthening-v1"
TASKS = {
    "t2": "t2_clutrr_heterogeneity",
    "t3": "t3_internal_heterogeneity",
    "t9": "t9_overlapping_certificates",
    "t10": "t10_novelty_bounds",
    "t6": "t6_semantic_vs_surface",
}
FIGURES = {
    "t2_clutrr_heterogeneity/fig_clutrr_heterogeneity.pdf":
        "tmlrs_clutrr_heterogeneity.pdf",
    "t3_internal_heterogeneity/fig_internal_heterogeneity.pdf":
        "tmlrs_internal_heterogeneity.pdf",
    "t10_novelty_bounds/fig_slr_novelty_bounds.pdf":
        "tmlrs_slr_novelty_bounds.pdf",
    "t6_semantic_vs_surface/fig_semantic_vs_surface.pdf":
        "tmlrs_semantic_vs_surface.pdf",
}


def main() -> None:
    generated = PAPER_DIR / "generated"
    figure_dir = PAPER_DIR / "figures"
    generated.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    inputs = []
    for short_name, directory_name in TASKS.items():
        source = ANALYSIS_DIR / directory_name / "macros.tex"
        if not source.exists():
            continue
        destination = generated / f"tmlrs_{short_name}_macros.tex"
        shutil.copyfile(source, destination)
        inputs.append(
            f"\\input{{generated/tmlrs_{short_name}_macros}}"
        )
    (generated / "tmlr_strengthening_macros.tex").write_text(
        "\n".join(inputs) + "\n",
        encoding="ascii",
    )
    for relative_source, destination_name in FIGURES.items():
        source = ANALYSIS_DIR / relative_source
        if source.exists():
            shutil.copyfile(source, figure_dir / destination_name)


if __name__ == "__main__":
    main()

