"""Figure style and plotting helpers shared by the notebooks.

Kept apart from the computation so that changing how a figure looks cannot change what it
shows. Applied through ``plt.rc_context(PAPER_STYLE)``.

The helpers below carry the layout of the published figures, so that the same grid reads the
same way whether it was produced by a mixture of Markov chains or of hidden Markov models,
and a reader can put two of them side by side.
"""

from pathlib import Path

import numpy as np

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


def frame_grid(ax, rows, cols, title, row_label=r"$\alpha$", col_label=r"$K$"):
    """Axis furniture of the published (alpha, K) heatmaps."""
    ax.set_xticks(range(len(cols)), [str(c) for c in cols])
    ax.set_yticks(range(len(rows)), [f"{r:g}" for r in rows])
    ax.set_xlabel(col_label)
    ax.set_ylabel(row_label)
    ax.set_title(title, fontsize=9, pad=6)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.4)
        spine.set_color("0.7")


def heatmap(ax, values, rows, cols, title, annot=None, cmap=None, vmin=0.0, vmax=1.0,
            fmt="{:.2f}", annot_fmt="{:+.2f}"):
    """Colour is `values`; the cell carries that number and, below it, `annot`.

    Two numbers per cell rather than one: a cell at 0 still has to say how far it is from
    the boundary, which a colour alone cannot do.
    """
    im = ax.imshow(values, origin="lower", aspect="auto",
                   cmap=SEQUENTIAL_CMAP if cmap is None else cmap, vmin=vmin, vmax=vmax)
    frame_grid(ax, rows, cols, title)
    span = max(abs(vmin), abs(vmax))
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if not np.isfinite(values[i, j]):
                continue
            light = abs(values[i, j]) > 0.6 * span if vmin < 0 else values[i, j] > 0.55
            colour = "white" if light else "0.25"
            ax.text(j, i + 0.13, fmt.format(values[i, j]), ha="center", va="center",
                    fontsize=5.5, color=colour)
            if annot is not None and np.isfinite(annot[i, j]):
                ax.text(j, i - 0.17, annot_fmt.format(annot[i, j]), ha="center", va="center",
                        fontsize=4.6, color=colour, alpha=0.85)
    return im


def plot_paths(ax, horizons, replicates, group_means, colour, ylabel, title):
    """One thin line per replicate, the group means between, the grand mean on top.

    The layout of `plot_gamma_convergence` in om_convergence.ipynb. A replicate is a sample
    path -- the same sequences read at growing lengths -- so the spread between thin lines is
    the variability of one experiment, not an interval around the mean.
    """
    ax.plot(horizons, replicates.T, color=colour, lw=0.6, alpha=0.13)
    if group_means is not None:
        ax.plot(horizons, group_means.T, color=colour, lw=0.9, alpha=0.5)
        ax.plot([], [], color=colour, lw=0.9, alpha=0.5,
                label=rf"{group_means.shape[0]} mixture means")
    ax.plot(horizons, replicates.mean(0), color=colour, lw=1.9,
            label=rf"mean over {replicates.shape[0]} replicates")
    ax.set_xlabel(r"$n$, length of a sequence")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.04, 1.04)
    ax.axhline(0.5, color="0.6", lw=0.7, ls=(0, (1, 3)), zorder=0)
    ax.set_title(title, fontsize=10)
    ax.legend(loc="lower right", handlelength=2.6, borderaxespad=0.6)


def save(fig, directory, name):
    """Write both formats the paper needs, and say where."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(directory / f"{name}.{ext}", bbox_inches="tight")
    print("figure written to", directory / f"{name}.pdf")
