#!/usr/bin/env python3
"""Generate public-architecture audit figures from compact CSV results."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if (ROOT / "results").exists():
    RESULTS_ROOT = ROOT / "results"
    DEFAULT_PAPER_DIR = ROOT / "paper_sources"
else:
    RESULTS_ROOT = ROOT / "semantic_novelty_audit_supplement" / "results"
    DEFAULT_PAPER_DIR = ROOT / "paper_sources"

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_PAPER_DIR / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODELS = (
    "symbolic_deepsets",
    "symbolic_relation_network",
    "symbolic_set_transformer",
    "symbolic_hopfield",
    "symbolic_perceiver",
    "symbolic_transformer",
)
LABELS = (
    "Deep Sets",
    "Relation Net",
    "Set Transformer",
    "Hopfield",
    "Perceiver",
    "Transformer",
)
COLORS = {
    "legacy": "#D55E00",
    "strict": "#0072B2",
    "heldout": "#009E73",
    "bayes": "#333333",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def configure() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def by_key(
    rows: list[dict[str, str]],
    *fields: str,
) -> dict[tuple[str, ...], dict[str, str]]:
    return {tuple(row[field] for field in fields): row for row in rows}


def generate(output_dir: Path) -> Path:
    audit_root = RESULTS_ROOT / "public_baseline_audit"
    split_rows = read_rows(audit_root / "analysis" / "group_summary.csv")
    transfer_rows = read_rows(
        audit_root / "family_transfer" / "group_summary.csv"
    )
    split = by_key(split_rows, "protocol", "model")
    transfer = by_key(transfer_rows, "family", "model")

    legacy = np.asarray(
        [
            100.0
            * float(split[("legacy_identity", model)]["primary_accuracy_mean"])
            for model in MODELS
        ]
    )
    legacy_sd = np.asarray(
        [
            100.0
            * float(split[("legacy_identity", model)]["primary_accuracy_sd"])
            for model in MODELS
        ]
    )
    strict = np.asarray(
        [
            100.0
            * float(split[("strict_function", model)]["primary_accuracy_mean"])
            for model in MODELS
        ]
    )
    strict_sd = np.asarray(
        [
            100.0
            * float(split[("strict_function", model)]["primary_accuracy_sd"])
            for model in MODELS
        ]
    )
    heldout = np.asarray(
        [
            100.0
            * float(transfer[("heldout_macro", model)]["accuracy_mean"])
            for model in MODELS
        ]
    )
    heldout_sd = np.asarray(
        [
            100.0
            * float(transfer[("heldout_macro", model)]["accuracy_sd"])
            for model in MODELS
        ]
    )
    bayes = 100.0 * float(
        transfer[
            ("heldout_macro", "family_conditioned_bayes")
        ]["accuracy_mean"]
    )

    configure()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75))
    y = np.arange(len(MODELS))

    for index in range(len(MODELS)):
        axes[0].plot(
            [strict[index], legacy[index]],
            [y[index], y[index]],
            color="#B8B8B8",
            linewidth=1.2,
            zorder=1,
        )
    axes[0].errorbar(
        strict,
        y,
        xerr=strict_sd,
        fmt="o",
        color=COLORS["strict"],
        capsize=2,
        markersize=4.5,
        linewidth=1,
        label="Strict function",
        zorder=3,
    )
    axes[0].errorbar(
        legacy,
        y,
        xerr=legacy_sd,
        fmt="s",
        color=COLORS["legacy"],
        capsize=2,
        markersize=4.2,
        linewidth=1,
        label="Legacy identity",
        zorder=2,
    )
    axes[0].set_title("(a) Split protocol effect")
    axes[0].set_xlabel("Held-out accuracy (%)")
    axes[0].set_yticks(y, LABELS)
    axes[0].set_xlim(34, 79)
    axes[0].invert_yaxis()
    axes[0].grid(axis="x", linestyle="--", alpha=0.3)
    axes[0].legend(frameon=False, loc="lower left")

    x = np.arange(len(MODELS))
    width = 0.36
    axes[1].bar(
        x - width / 2,
        strict,
        width,
        yerr=strict_sd,
        capsize=2,
        color=COLORS["strict"],
        label="Known-family functions",
    )
    axes[1].bar(
        x + width / 2,
        heldout,
        width,
        yerr=heldout_sd,
        capsize=2,
        color=COLORS["heldout"],
        label="Held-out families",
    )
    axes[1].axhline(
        bayes,
        color=COLORS["bayes"],
        linewidth=1.2,
        linestyle="--",
        label=f"Held-out Bayes ({bayes:.1f}%)",
    )
    axes[1].set_title("(b) Function vs. family generalization")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_xticks(x, ("DS", "RN", "ST", "MH", "P", "T"))
    axes[1].set_ylim(0, 105)
    axes[1].set_yticks([0, 20, 40, 60, 80, 100])
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)
    axes[1].legend(frameon=False, loc="upper left")

    fig.subplots_adjust(
        left=0.14,
        right=0.99,
        bottom=0.17,
        top=0.88,
        wspace=0.3,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "fig_audit_summary.pdf"
    fig.savefig(output, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PAPER_DIR / "figures",
    )
    args = parser.parse_args()
    output = generate(args.output_dir)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
