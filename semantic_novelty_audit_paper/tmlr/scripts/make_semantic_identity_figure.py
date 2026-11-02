#!/usr/bin/env python3
"""Render a vector overview of claim-dependent semantic identity."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "figures" / "fig_semantic_identity_framework.pdf"

# Reuse the shared paper style module when the supplement tree is present;
# fall back to an inline equivalent when this script is run standalone.
_AUDIT_ROOT = (
    Path(__file__).resolve().parents[3] / "external_semantic_overlap_audit"
)
if str(_AUDIT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AUDIT_ROOT))

try:  # pragma: no cover - exercised on the full supplement tree
    from toolkit.paper_style import PALETTE, apply as apply_paper_style
except Exception:  # pragma: no cover - standalone fallback
    PALETTE = {
        "blue": "#2E6FA7", "green": "#3E8E6E", "gold": "#C9A13B",
        "sky": "#7FB6D9", "charcoal": "#2B2F36", "mid": "#6B7280",
        "pale_blue": "#E9F1F8", "pale_green": "#E8F3EE",
        "pale_orange": "#FBF1E2",
    }

    def apply_paper_style() -> None:
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "pdf.fonttype": 42,
        })


INK = PALETTE["charcoal"]
MUTED = PALETTE["mid"]
BLUE = PALETTE["blue"]
GREEN = PALETTE["green"]
AMBER = PALETTE["gold"]


def node(ax, x, y, width, height, label, color, fill="white", subtitle=None):
    ax.add_patch(FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.006,rounding_size=0.009",
        linewidth=0.8, edgecolor=color, facecolor=fill, zorder=3,
    ))
    ax.text(
        x + width / 2, y + height * (0.62 if subtitle else 0.5),
        label, ha="center", va="center", fontsize=9, color=INK, zorder=4,
    )
    if subtitle:
        ax.text(
            x + width / 2, y + height * 0.27, subtitle,
            ha="center", va="center", fontsize=7.6, color=MUTED, zorder=4,
        )


def arrow(ax, start, end, color, rad=0):
    ax.add_patch(FancyArrowPatch(
        start, end, connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>", mutation_scale=9, linewidth=1.0,
        color=color, shrinkA=1, shrinkB=1, zorder=2,
    ))


def main():
    apply_paper_style()
    fig = plt.figure(figsize=(6.0, 2.8))
    ax = fig.add_axes([0.01, 0.02, 0.98, 0.96])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")

    columns = [
        (0.015, BLUE, PALETTE["pale_blue"], "Representation identity",
         "Generator setting or rule text", "Parameter-disjoint",
         "New representation;\nsemantic aliases may remain"),
        (0.355, GREEN, PALETTE["pale_green"], "Semantic identity",
         r"Observable behavior under $\mathcal{O}$", "Function-disjoint",
         "New observable\nsemantic object"),
        (0.695, AMBER, PALETTE["pale_orange"], "Family identity",
         r"Registered class $a(r)$", "Family-disjoint",
         "No registered family\nshared across the split"),
    ]
    for x, color, fill, title, subtitle, criterion, detail in columns:
        ax.add_patch(FancyBboxPatch(
            (x, 0.025), 0.29, 0.94,
            boxstyle="round,pad=0.004,rounding_size=0.012",
            facecolor=fill, edgecolor="none", zorder=0,
        ))
        ax.plot([x + 0.025, x + 0.265], [0.945, 0.945],
                color=color, linewidth=2, solid_capstyle="round")
        ax.text(x + 0.145, 0.885, title, fontsize=9.2, color=INK,
                fontweight="bold", ha="center", va="center")
        ax.text(x + 0.145, 0.827, subtitle, fontsize=7.6,
                color=MUTED, ha="center", va="center")
        ax.plot([x + 0.025, x + 0.265], [0.235, 0.235],
                color=color, linewidth=0.55, alpha=0.4)
        ax.text(x + 0.145, 0.177, criterion, fontsize=8.8,
                color=color, ha="center", va="center", fontweight="bold")
        ax.text(x + 0.145, 0.09, detail, fontsize=7.5, color=MUTED,
                ha="center", va="center", linespacing=1.35)

    left_x, middle_x, right_x = 0.04, 0.38, 0.72
    width = 0.24
    for y, label in [
        (0.65, r"$r_1$: affine rule"),
        (0.49, r"$r_2$: parity alias"),
        (0.30, r"$r_3$: threshold rule"),
    ]:
        node(ax, left_x, y, width, 0.10, label, PALETTE["sky"])
    node(ax, middle_x, 0.52, width, 0.19,
         r"$f_1(x)=\mathbf{1}[x>0]$", GREEN,
         subtitle="Shared semantic object")
    node(ax, middle_x, 0.30, width, 0.10,
         r"$f_2(x)=\mathbf{1}[x>2]$", GREEN)
    for y, label in [
        (0.65, "Affine family"),
        (0.49, "Parity family"),
        (0.30, "Threshold family"),
    ]:
        node(ax, right_x, y, width, 0.10, label, PALETTE["gold"])

    arrow(ax, (0.286, 0.70), (0.374, 0.655), BLUE, rad=-0.08)
    arrow(ax, (0.286, 0.54), (0.374, 0.575), BLUE, rad=0.08)
    arrow(ax, (0.286, 0.35), (0.374, 0.35), BLUE)
    arrow(ax, (0.626, 0.655), (0.714, 0.70), AMBER, rad=-0.08)
    arrow(ax, (0.626, 0.575), (0.714, 0.54), AMBER, rad=0.08)
    arrow(ax, (0.626, 0.35), (0.714, 0.35), AMBER)
    ax.text(0.67, 0.617, r"$\Gamma(f_1)$", fontsize=8.3,
            color=AMBER, ha="center", va="center")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUTPUT, bbox_inches="tight", pad_inches=0.03,
        metadata={"Creator": "make_semantic_identity_figure.py",
                  "CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
