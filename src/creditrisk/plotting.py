"""Shared figure style, so that every notebook produces charts that look alike.

Colours: one blue for a single series or for magnitude (light to dark), blue and orange when two
series are compared. Grid and axes stay light so that the data stands out.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from creditrisk.config import REPO_ROOT

FIGURES_DIR = REPO_ROOT / "reports" / "figures"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

# Single-hue scale for magnitudes (heatmaps): light = small, dark = large.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "creditrisk_blues", [SURFACE, "#cde2fb", "#5598e7", "#104281"]
)


def apply_style() -> None:
    """Set the matplotlib defaults used across the project. Call once at the top of a notebook."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 100,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 10,
            "text.color": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelcolor": MUTED,
            "axes.edgecolor": GRID,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "legend.frameon": False,
        }
    )


def save_figure(fig: Figure, name: str, figures_dir: Path | None = None) -> Path:
    """Save a figure as ``<name>.png`` in ``figures_dir`` (default ``reports/figures``)."""
    folder = FIGURES_DIR if figures_dir is None else Path(figures_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.png"
    fig.savefig(path)
    return path
