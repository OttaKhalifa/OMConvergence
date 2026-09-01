"""Figure style shared by the three notebooks.

Kept apart from the computation so that changing how a figure looks cannot change what it
shows. Applied through ``plt.rc_context(PAPER_STYLE)``.
"""

#: rcParams shared by every figure of the paper, applied through plt.rc_context.
PAPER_STYLE = {
    "figure.dpi": 130, "savefig.dpi": 300, "font.family": "serif", "font.size": 10,
    "axes.labelsize": 10, "axes.titlesize": 10, "legend.fontsize": 8,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.4, "legend.frameon": False,
}

#: single-hue light-to-dark ramp for magnitudes in [0, 1] (probabilities)
SEQUENTIAL_CMAP = "Blues"

#: two-hue ramp with a neutral midpoint, for signed quantities pivoting at 0
DIVERGING_CMAP = "RdBu"
