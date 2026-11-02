#!/usr/bin/env python3
"""Build the cross-benchmark summary CSV and three manuscript figures."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toolkit.paper_style import PALETTE, apply as apply_paper_style  # noqa: E402


COLORS = {
    "blue": PALETTE["blue"],
    "green": PALETTE["green"],
    "orange": PALETTE["orange"],
    "vermillion": PALETTE["vermillion"],
    "sky": PALETTE["sky"],
    "charcoal": PALETTE["charcoal"],
    "mid": PALETTE["mid"],
    "light": PALETTE["light"],
    "pale_blue": PALETTE["pale_blue"],
    "pale_green": PALETTE["pale_green"],
    "pale_orange": PALETTE["pale_orange"],
}

TASK_LABELS = {
    "conjunction_d28": "Conjunction",
    "majority_d28": "Majority",
    "sparse_disjunction_d50_k4_positive": "Sparse disjunction",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def configure_matplotlib() -> None:
    apply_paper_style()


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    fig.savefig(
        output_base.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={
            "Creator": "build_cross_benchmark_figures.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(
        output_base.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        metadata={"Software": "build_cross_benchmark_figures.py"},
    )


def rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    facecolor: str,
    edgecolor: str,
    linewidth: float = 1.0,
    radius: float = 0.02,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=3,
    )
    ax.add_patch(patch)
    return patch


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["mid"],
    linewidth: float = 1.0,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=linewidth,
            color=color,
            shrinkA=0,
            shrinkB=0,
            connectionstyle="arc3,rad=0",
            zorder=2,
        )
    )


def build_framework_figure(output_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    columns = [
        (0.035, 0.29, "Parameter identity", "generator settings / rule text"),
        (0.355, 0.29, "Semantic identity", "complete input-output function"),
        (0.675, 0.29, "Family identity", "rule class / grammar family"),
    ]
    header_colors = (
        (COLORS["pale_blue"], COLORS["blue"]),
        (COLORS["pale_green"], COLORS["green"]),
        (COLORS["pale_orange"], COLORS["orange"]),
    )
    for (x, width, title, subtitle), (face, edge) in zip(columns, header_colors):
        rounded_box(
            ax,
            (x, 0.81),
            width,
            0.13,
            facecolor=face,
            edgecolor=edge,
            linewidth=1.2,
        )
        ax.text(
            x + width / 2,
            0.885,
            title,
            ha="center",
            va="center",
            weight="bold",
            color=COLORS["charcoal"],
        )
        ax.text(
            x + width / 2,
            0.842,
            subtitle,
            ha="center",
            va="center",
            fontsize=7.5,
            color=COLORS["mid"],
        )

    parameter_nodes = [
        (0.075, 0.700, r"$r_1$: $x > 0$"),
        (0.075, 0.635, r"$r_2$: $0 < x$"),
        (0.075, 0.570, r"$r_3$: $x + 1 > 1$"),
        (0.075, 0.445, r"$r_4$: $x^2 > 0$"),
        (0.075, 0.365, r"$r_5$: $x \times x > 0$"),
    ]
    parameter_width = 0.215
    parameter_height = 0.038
    for x, center_y, text in parameter_nodes:
        rounded_box(
            ax,
            (x, center_y - parameter_height / 2),
            parameter_width,
            parameter_height,
            facecolor="white",
            edgecolor=COLORS["blue"],
            radius=0.012,
        )
        ax.text(
            x + parameter_width / 2,
            center_y,
            text,
            ha="center",
            va="center",
            fontsize=7.5,
            zorder=4,
        )

    semantic_nodes = [
        (0.395, 0.635, r"$f_1(x)=\mathbf{1}[x>0]$"),
        (0.395, 0.405, r"$f_2(x)=\mathbf{1}[x^2>0]$"),
    ]
    semantic_width = 0.215
    semantic_height = 0.086
    for x, center_y, text in semantic_nodes:
        rounded_box(
            ax,
            (x, center_y - semantic_height / 2),
            semantic_width,
            semantic_height,
            facecolor="white",
            edgecolor=COLORS["green"],
            radius=0.014,
        )
        ax.text(
            x + semantic_width / 2,
            center_y,
            text,
            ha="center",
            va="center",
            fontsize=8,
            zorder=4,
        )

    family_nodes = [
        (0.745, 0.635, "Threshold family"),
        (0.745, 0.405, "Polynomial family"),
    ]
    family_width = 0.18
    family_height = 0.086
    for x, center_y, text in family_nodes:
        rounded_box(
            ax,
            (x, center_y - family_height / 2),
            family_width,
            family_height,
            facecolor="white",
            edgecolor=COLORS["orange"],
            radius=0.016,
        )
        ax.text(
            x + family_width / 2,
            center_y,
            text,
            ha="center",
            va="center",
            fontsize=8,
            zorder=4,
        )

    box_pad = 0.012
    parameter_right = 0.075 + parameter_width + box_pad
    semantic_left = 0.395 - box_pad
    semantic_right = 0.395 + semantic_width + box_pad
    family_left = 0.745 - box_pad
    f1_ports = (0.655, 0.635, 0.615)
    for (_, center_y, _), port_y in zip(parameter_nodes[:3], f1_ports):
        arrow(
            ax,
            (parameter_right, center_y),
            (semantic_left, port_y),
            color=COLORS["blue"],
        )
    f2_ports = (0.420, 0.390)
    for (_, center_y, _), port_y in zip(parameter_nodes[3:], f2_ports):
        arrow(
            ax,
            (parameter_right, center_y),
            (semantic_left, port_y),
            color=COLORS["blue"],
        )
    arrow(
        ax,
        (semantic_right, 0.635),
        (family_left, 0.635),
        color=COLORS["green"],
    )
    arrow(
        ax,
        (semantic_right, 0.405),
        (family_left, 0.405),
        color=COLORS["green"],
    )

    ax.text(
        0.3425,
        0.755,
        r"$q:\mathcal{R}\rightarrow\mathcal{F}$  (many-to-one)",
        ha="center",
        va="center",
        fontsize=7.5,
        color=COLORS["mid"],
    )
    ax.text(
        0.6775,
        0.755,
        r"$g:\mathcal{F}\rightarrow\mathcal{G}$",
        ha="center",
        va="center",
        fontsize=8,
        color=COLORS["mid"],
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
    )

    bands = [
        (
            0.035,
            0.29,
            "Parameter-disjoint",
            "new representation\nmay preserve the same function",
            COLORS["pale_blue"],
            COLORS["blue"],
        ),
        (
            0.355,
            0.29,
            "Function-disjoint",
            "new complete mapping\nwithin a known family",
            COLORS["pale_green"],
            COLORS["green"],
        ),
        (
            0.675,
            0.29,
            "Family-disjoint",
            "new rule class\nor grammar family",
            COLORS["pale_orange"],
            COLORS["orange"],
        ),
    ]
    for x, width, title, subtitle, face, edge in bands:
        rounded_box(
            ax,
            (x, 0.06),
            width,
            0.17,
            facecolor=face,
            edgecolor=edge,
            linewidth=1.0,
        )
        ax.text(
            x + width / 2,
            0.172,
            title,
            ha="center",
            va="center",
            fontsize=8.3,
            weight="bold",
            color=COLORS["charcoal"],
        )
        ax.text(
            x + width / 2,
            0.105,
            subtitle,
            ha="center",
            va="center",
            fontsize=7.2,
            color=COLORS["mid"],
            linespacing=1.25,
        )

    save_figure(fig, output_base)
    plt.close(fig)


def build_slr_figure(slr: dict, output_base: Path) -> None:
    total = slr["rows"]["test"]
    token_rows = slr["sound_semantic_overlap_union"]["token_identity_rows"]
    flat_rows = slr["sound_semantic_overlap_union"]["flat_rule_equivalence_rows"]
    union_rows = slr["sound_semantic_overlap_union"]["test_overlap_rows"]
    verbatim_rows = slr["raw_identity"]["test_rows_found_verbatim_in_train"]
    values = [verbatim_rows / total, token_rows / total, union_rows / total]

    fig, ax = plt.subplots(figsize=(6.0, 2.4))
    y = np.arange(3)
    ax.barh(y, [1, 1, 1], color=COLORS["light"], height=0.52, zorder=0)
    ax.barh(
        y[0],
        values[0],
        color=COLORS["mid"],
        height=0.52,
        zorder=2,
    )
    ax.barh(
        y[1],
        values[1],
        color=COLORS["blue"],
        height=0.52,
        label="Token-identical",
        zorder=2,
    )
    ax.barh(
        y[2],
        token_rows / total,
        color=COLORS["blue"],
        height=0.52,
        zorder=2,
    )
    ax.barh(
        y[2],
        flat_rows / total,
        left=token_rows / total,
        color=COLORS["orange"],
        height=0.52,
        label="Additional sound semantic aliases",
        zorder=2,
    )
    counts = (verbatim_rows, token_rows, union_rows)
    for index, (value, count) in enumerate(zip(values, counts)):
        x = max(value + 0.018, 0.025)
        ax.text(
            x,
            index,
            f"{100 * value:.1f}%  ({count:,}/{total:,})",
            va="center",
            ha="left",
            fontsize=8.5,
            weight="bold" if index == 2 else "normal",
            color=COLORS["charcoal"],
        )

    ax.set_yticks(y)
    ax.set_yticklabels(
        ["Raw string match", "Prolog token identity", "Sound semantic union"]
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xticklabels([f"{int(value * 100)}%" for value in np.linspace(0, 1, 6)])
    ax.set_xlabel("Test rules overlapping the training set")
    ax.grid(axis="x", color="#D0D5DD", linewidth=0.6, alpha=0.8, zorder=1)
    ax.legend(
        loc="upper right",
        frameon=False,
        ncol=1,
        bbox_to_anchor=(0.99, 0.83),
    )
    fig.text(
        0.5,
        0.035,
        "Lower bound: unresolved aggregate, negation, arithmetic, and disjunctive rules are not counted.",
        ha="center",
        va="center",
        fontsize=7.2,
        color=COLORS["mid"],
    )
    fig.subplots_adjust(bottom=0.24)
    save_figure(fig, output_base)
    plt.close(fig)


def build_boolean_figure(
    boolean_rows: list[dict], output_base: Path
) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    marker_styles = ("o", "s", "D")
    colors = (COLORS["blue"], COLORS["green"], COLORS["vermillion"])
    offsets = ((8, 10), (8, -18), (-122, 10))
    for row, marker, color, offset in zip(
        boolean_rows, marker_styles, colors, offsets
    ):
        exposure_pct = 100.0 * row["overlap_or_exposure_rate"]
        effect = row["behavior_effect_pp"]
        lower = row["behavior_ci_low_pp"]
        upper = row["behavior_ci_high_pp"]
        ax.errorbar(
            exposure_pct,
            effect,
            yerr=[[effect - lower], [upper - effect]],
            fmt=marker,
            markersize=7,
            color=color,
            ecolor=color,
            elinewidth=1.5,
            capsize=3,
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=3,
        )
        ax.annotate(
            f"{row['case_label']}\n{exposure_pct:.2f}% exposed",
            (exposure_pct, effect),
            xytext=offset,
            textcoords="offset points",
            fontsize=7.8,
            color=COLORS["charcoal"],
            arrowprops={
                "arrowstyle": "-",
                "color": color,
                "linewidth": 0.8,
                "shrinkA": 2,
                "shrinkB": 4,
            },
        )

    ax.axhline(0, color=COLORS["charcoal"], linewidth=0.9, zorder=1)
    ax.fill_between(
        [0.5, 110],
        -0.3,
        0.3,
        color=COLORS["light"],
        alpha=0.45,
        zorder=0,
    )
    ax.set_xscale("log")
    ax.set_xlim(0.5, 110)
    ax.set_ylim(-0.31, 0.53)
    ax.set_xticks([1, 10, 100])
    ax.set_xticklabels(["1%", "10%", "100%"])
    ax.set_yticks(np.arange(-0.3, 0.51, 0.2))
    ax.set_xlabel("Original-protocol evaluation functions seen during training")
    ax.set_ylabel("Tail-accuracy effect: Original − Strict (pp)")
    ax.grid(axis="both", color="#D0D5DD", linewidth=0.6, alpha=0.75)
    ax.text(
        107,
        0.49,
        "Registered Go threshold: +2 pp (outside panel)",
        ha="right",
        va="top",
        fontsize=7.5,
        color=COLORS["mid"],
    )
    ax.text(
        0.52,
        -0.285,
        "Shaded band: ±0.3 pp",
        ha="left",
        va="bottom",
        fontsize=7.2,
        color=COLORS["mid"],
    )
    save_figure(fig, output_base)
    plt.close(fig)


def build_summary_rows(
    internal: dict,
    slr: dict,
    split_manifest: dict,
    boolean_results: dict,
) -> list[dict]:
    rows = [
        {
            "benchmark": "Internal symbolic benchmark",
            "case_id": "legacy_identity_public_audit",
            "case_label": "Internal identity split",
            "semantic_object": "complete input-output function",
            "protocol": "parameter-identity split",
            "contamination_measure": "held-out identities duplicating a train function",
            "overlap_or_exposure_rate": internal[
                "legacy_leaked_unseen_identity_fraction_mean"
            ],
            "rate_status": "exact_matched_five_seed_audit",
            "behavior_model": "six-public-architecture macro",
            "behavior_effect_pp": 100.0
            * internal["architecture_macro_difference"],
            "behavior_ci_low_pp": 100.0
            * internal["architecture_macro_ci"][0],
            "behavior_ci_high_pp": 100.0
            * internal["architecture_macro_ci"][1],
            "behavior_gate": (
                "go" if internal["cross_architecture_inflation_supported"] else "no_go"
            ),
            "source_overlap": "semantic_novelty_audit_supplement/results/public_baseline_audit/analysis/claim_gates.json",
            "source_behavior": "semantic_novelty_audit_supplement/results/public_baseline_audit/analysis/claim_gates.json",
        },
        {
            "benchmark": "SLR-Bench v1-All",
            "case_id": "slr_sound_semantic_union",
            "case_label": "SLR-Bench",
            "semantic_object": "Prolog rule semantics",
            "protocol": "raw rule-string exclusion",
            "contamination_measure": "sound train-test semantic-overlap lower bound",
            "overlap_or_exposure_rate": slr["sound_semantic_overlap_union"][
                "test_overlap_rate"
            ],
            "rate_status": "sound_lower_bound",
            "behavior_model": "",
            "behavior_effect_pp": None,
            "behavior_ci_low_pp": None,
            "behavior_ci_high_pp": None,
            "behavior_gate": "not_evaluable_from_reported_outputs",
            "source_overlap": "external_semantic_overlap_audit/artifacts/slr_overlap_results.json",
            "source_behavior": "",
        },
    ]

    for task_name in TASK_LABELS:
        training_summaries = split_manifest["task_training_stream_summaries"][
            task_name
        ]
        exposure = float(
            np.mean(
                [
                    item["eval_draw_exposure_rate_in_original"]
                    for item in training_summaries
                ]
            )
        )
        task_result = boolean_results["tasks"][task_name]
        interval = task_result["hierarchical_bootstrap"]["ci95_pp"]
        rows.append(
            {
                "benchmark": "Boolean ICL released generator",
                "case_id": task_name,
                "case_label": TASK_LABELS[task_name],
                "semantic_object": "Boolean truth table",
                "protocol": "same-distribution resampling",
                "contamination_measure": "evaluation functions sampled during training",
                "overlap_or_exposure_rate": exposure,
                "rate_status": "exact_empirical_frozen_stream",
                "behavior_model": "6-layer Transformer",
                "behavior_effect_pp": task_result["three_seed"]["mean_delta_pp"],
                "behavior_ci_low_pp": interval[0],
                "behavior_ci_high_pp": interval[1],
                "behavior_gate": (
                    "go" if task_result["core_gate_pass"] else "no_go"
                ),
                "source_overlap": "external_semantic_overlap_audit/protocols/boolean-behavioral-v1/split_manifest.json",
                "source_behavior": "external_semantic_overlap_audit/analysis/boolean-behavioral-v1/results.json",
            }
        )
    return rows


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    audit_root = Path(__file__).resolve().parents[1]
    workspace = audit_root.parent
    output_dir = audit_root / "analysis" / "cross_benchmark"
    paper_figure_dirs = {
        "historical_snapshot": workspace / "paper_sources" / "figures",
        "cross_benchmark_paper": workspace
        / "semantic_novelty_audit_paper"
        / "figures",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for paper_figure_dir in paper_figure_dirs.values():
        paper_figure_dir.mkdir(parents=True, exist_ok=True)

    sources = {
        "internal_claim_gates": workspace
        / "semantic_novelty_audit_supplement/results/public_baseline_audit/analysis/claim_gates.json",
        "slr_overlap": audit_root / "artifacts/slr_overlap_results.json",
        "boolean_split_manifest": audit_root
        / "protocols/boolean-behavioral-v1/split_manifest.json",
        "boolean_behavioral_results": audit_root
        / "analysis/boolean-behavioral-v1/results.json",
    }
    internal = load_json(sources["internal_claim_gates"])
    slr = load_json(sources["slr_overlap"])
    split_manifest = load_json(sources["boolean_split_manifest"])
    boolean_results = load_json(sources["boolean_behavioral_results"])

    rows = build_summary_rows(
        internal, slr, split_manifest, boolean_results
    )
    summary_path = output_dir / "cross_benchmark_summary.csv"
    write_summary_csv(summary_path, rows)

    figure_data = {
        "generator_sha256": sha256_file(Path(__file__).resolve()),
        "source_sha256": {
            name: sha256_file(path) for name, path in sources.items()
        },
        "summary_rows": rows,
        "slr_figure": {
            "test_rows": slr["rows"]["test"],
            "verbatim_rows": slr["raw_identity"][
                "test_rows_found_verbatim_in_train"
            ],
            "token_identity_rows": slr["sound_semantic_overlap_union"][
                "token_identity_rows"
            ],
            "flat_rule_equivalence_rows": slr[
                "sound_semantic_overlap_union"
            ]["flat_rule_equivalence_rows"],
            "semantic_union_rows": slr["sound_semantic_overlap_union"][
                "test_overlap_rows"
            ],
        },
    }
    figure_data_path = output_dir / "figure_data.json"
    figure_data_path.write_text(
        json.dumps(figure_data, indent=2, sort_keys=True) + "\n"
    )

    configure_matplotlib()
    figure_bases = {
        "identity": output_dir / "fig1_semantic_identity_framework",
        "slr": output_dir / "fig2_slr_semantic_overlap",
        "boolean": output_dir / "fig3_boolean_exposure_effect",
    }
    build_framework_figure(figure_bases["identity"])
    build_slr_figure(slr, figure_bases["slr"])
    boolean_rows = [
        row
        for row in rows
        if row["benchmark"] == "Boolean ICL released generator"
    ]
    build_boolean_figure(boolean_rows, figure_bases["boolean"])

    paper_names = {
        "identity": "fig_semantic_identity_framework.pdf",
        "slr": "fig_slr_semantic_overlap.pdf",
        "boolean": "fig_boolean_exposure_effect.pdf",
    }
    for paper_figure_dir in paper_figure_dirs.values():
        for key, paper_name in paper_names.items():
            shutil.copyfile(
                figure_bases[key].with_suffix(".pdf"),
                paper_figure_dir / paper_name,
            )

    output_paths = [
        output_dir / "README.md",
        summary_path,
        figure_data_path,
        *(base.with_suffix(suffix) for base in figure_bases.values() for suffix in (".pdf", ".png")),
    ]
    checksum_path = output_dir / "SHA256SUMS"
    checksum_path.write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in output_paths)
    )
    print(
        json.dumps(
            {
                "summary_rows": len(rows),
                "output_dir": str(output_dir),
                "paper_figures": {
                    destination: {
                        key: str(directory / name)
                        for key, name in paper_names.items()
                    }
                    for destination, directory in paper_figure_dirs.items()
                },
                "sha256s": {
                    path.name: sha256_file(path) for path in output_paths
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
