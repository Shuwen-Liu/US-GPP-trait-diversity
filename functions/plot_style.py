"""Shared matplotlib styling used by the figure scripts.

Figure widths follow the journal's column widths (inches).
"""
import matplotlib.pyplot as plt
import seaborn as sns

FIG_FULL = 7.5      # full-page width
FIG_SINGLE = 3.75   # single-column width


def apply_style(font: str = "Helvetica", context: str = "paper") -> None:
    """Paper-style defaults: seaborn 'ticks', tight layout, 400 dpi output.

    The requested font is used if it is installed; otherwise matplotlib falls back
    to its default sans-serif font.
    """
    plt.style.use("default")
    sns.set_style("ticks")
    sns.set_context(context)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [font, "Arial", "DejaVu Sans"]
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = font
    plt.rcParams["mathtext.it"] = f"{font}:italic"
    plt.rcParams["mathtext.bf"] = f"{font}:bold"
    plt.rcParams["lines.linewidth"] *= 1.25
    plt.rcParams["axes.linewidth"] *= 1.25
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.rcParams["savefig.dpi"] = 400
    plt.rcParams["savefig.bbox"] = "tight"
