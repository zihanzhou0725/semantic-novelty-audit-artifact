#!/usr/bin/env python3
"""Unified figure style for the semantic-novelty audit paper.

Usage
-----
Copy this file into the supplement at ``external_semantic_overlap_audit/
toolkit/paper_style.py`` (the figure scripts already put that root on
``sys.path``), then at the top of every figure script::

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from toolkit.paper_style import (
        PALETTE, apply, clean, figsize, panel_label, save, categorical_colors,
    )
    apply()                       # before creating any figure

and at the end of each figure::

    save(fig, output_path)        # vector PDF, tight bbox

Design rules this module enforces
---------------------------------
1. Serif text (STIX) so figures sit inside a Computer Modern paper without
   a visible font break; math uses the same family.
2. One muted, print-safe palette derived from Okabe--Ito; no default
   matplotlib cycle, no tab20.
3. Figure widths aligned to the paper's 6in text block: full (6.0in),
   wide (6.0x1.8), half (2.9in).
4. No in-figure titles.  The caption carries the title; scripts should not
   call ``ax.set_title(...)``.
5. Lean axes: top/right spines off, 0.6pt rules, small outward ticks,
   dotted grid only where it earns its place.
6. Vector output: PDF with ``fonttype 42`` (embedded TrueType), 300dpi PNG
   previews on request.
"""

from __future__ import annotations

from typing import Iterable

import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib import colors as mcolors

# ---------------------------------------------------------------------------
# Palette: Okabe--Ito hues, desaturated ~15% for print.
# ---------------------------------------------------------------------------
PALETTE = {
    "blue": "#2E6FA7",
    "orange": "#D98E2B",
    "green": "#3E8E6E",
    "vermillion": "#C2543D",
    "sky": "#7FB6D9",
    "gold": "#C9A13B",
    "charcoal": "#2B2F36",
    "mid": "#6B7280",
    "light": "#E6E8EC",
    # pale panel fills for schematic figures (e.g. the identity framework)
    "pale_blue": "#E9F1F8",
    "pale_green": "#E8F3EE",
    "pale_orange": "#FBF1E2",
}

#: Cycle used by matplotlib when no explicit color is given.
CYCLE = [
    PALETTE["blue"],
    PALETTE["orange"],
    PALETTE["green"],
    PALETTE["vermillion"],
    PALETTE["gold"],
    PALETTE["mid"],
]

#: Sequential map for continuous magnitudes.
SEQUENTIAL = "viridis"

# ---------------------------------------------------------------------------
# Figure sizes aligned to a 6in text block (DMLR 11pt layout).
# ---------------------------------------------------------------------------
SIZES = {
    "full": (6.0, 2.4),        # ~0.98\textwidth
    "figure1": (6.0, 2.3),     # headline figures at 0.94\textwidth
    "wide": (6.0, 1.8),
    "full_tall": (6.0, 3.0),
    "half": (2.9, 2.2),
    "panel": (3.1, 2.15),      # 0.52\textwidth appendix panels
    "panel_tall": (3.1, 2.45),
    "third": (1.9, 1.9),
}


def figsize(kind: str = "full") -> tuple[float, float]:
    """Return a standard figure size in inches."""
    if kind not in SIZES:
        raise KeyError(f"unknown size {kind!r}; choose from {sorted(SIZES)}")
    return SIZES[kind]


def apply(preset: str = "serif", base_size: float = 8.0) -> None:
    """Install the paper style into matplotlib's rcParams.

    Parameters
    ----------
    preset:
        ``"serif"`` (default, matches the paper body) or ``"sans"``
        (fallback if a sans-serif house style is preferred).
    base_size:
        Base font size in points.  Figures included at ~0.95x their natural
        width render at ~7.5pt effective text.
    """
    if preset == "serif":
        family = {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
            "mathtext.fontset": "stix",
        }
    elif preset == "sans":
        family = {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
        }
    else:
        raise ValueError("preset must be 'serif' or 'sans'")

    rc = {
        "figure.figsize": SIZES["full"],
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.pad_inches": 0.02,
        "font.size": base_size,
        "axes.labelsize": base_size,
        "axes.titlesize": base_size,  # discouraged; captions carry titles
        "xtick.labelsize": base_size - 1,
        "ytick.labelsize": base_size - 1,
        "legend.fontsize": base_size - 1,
        "axes.linewidth": 0.6,
        "axes.edgecolor": PALETTE["charcoal"],
        "axes.labelcolor": PALETTE["charcoal"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "axes.grid": False,
        "grid.color": PALETTE["light"],
        "grid.linewidth": 0.5,
        "grid.linestyle": ":",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.color": PALETTE["charcoal"],
        "ytick.color": PALETTE["charcoal"],
        "text.color": PALETTE["charcoal"],
        "lines.linewidth": 1.0,
        "lines.markersize": 3.0,
        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "legend.borderaxespad": 0.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.prop_cycle": cycler(color=CYCLE),
    }
    rc.update(family)
    plt.rcParams.update(rc)


def clean(axes: Iterable) -> None:
    """Apply axis-level polish that rcParams cannot express.

    Accepts a single axis or an iterable of axes.  Removes the top/right
    spines, thins the remaining rules, and keeps ticks small.
    """
    if isinstance(axes, matplotlib.axes.Axes):
        axes = [axes]
    for ax in axes:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_linewidth(0.6)
            ax.spines[side].set_color(PALETTE["charcoal"])
        ax.tick_params(
            direction="out",
            length=2.5,
            width=0.6,
            colors=PALETTE["charcoal"],
        )


def panel_label(ax, label: str, x: float = -0.02, y: float = 1.02) -> None:
    """Place a bold panel tag such as ``(a)`` just outside the axes."""
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontweight="bold",
        fontsize=plt.rcParams["font.size"],
        color=PALETTE["charcoal"],
        clip_on=False,
    )


def save(fig, path, *, dpi: int = 300, pad: float = 0.02):
    """Save ``fig`` to ``path`` with a tight bounding box.

    Use a ``.pdf`` path for the manuscript (vector, embedded fonts).  A
    ``.png`` path is fine for quick previews; ``dpi`` applies only there.
    """
    fig.savefig(path, bbox_inches="tight", pad_inches=pad, dpi=dpi)
    return path


def categorical_colors(n: int, blend: float = 0.2) -> list[tuple]:
    """Return ``n`` muted, distinguishable colors (for target-relation codes).

    Blends the tab20 cycle towards white by ``blend`` so that categorical
    scatter plots stay legible in print without changing hue ordering.
    """
    cmap = plt.get_cmap("tab20")
    colors = []
    for i in range(n):
        r, g, b, _ = cmap(i % 20)
        colors.append(
            (
                (1.0 - blend) * r + blend,
                (1.0 - blend) * g + blend,
                (1.0 - blend) * b + blend,
            )
        )
    return colors


def _self_check() -> None:
    apply()
    print("figure size (full):", figsize("full"))
    print("prop cycle:", plt.rcParams["axes.prop_cycle"].by_key()["color"])
    print("font:", plt.rcParams["font.family"], plt.rcParams["font.serif"][0])
    print("categorical colors (3):", categorical_colors(3))
    print("OK")


if __name__ == "__main__":
    _self_check()
