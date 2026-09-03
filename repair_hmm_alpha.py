"""Restore the alpha column of the HMM tables, which HMMMixture.alpha wrote as NaN.

`run_dataset` and `eta_rows` record `mixture.alpha`, and the HMM mixture returned NaN for it
before the field was added -- so the grid axis is missing from every row the running sweep
wrote. The draws themselves are fine: `alpha_B` did reach the generator, and the stream key
never contained alpha, so nothing about the mixtures changes.

The coordinate is recoverable because the drivers iterate cells in a fixed order,
``[(a, K) for a in ALPHAS for K in KS]``, and append as they go. This script walks the rows
in that order and writes the alpha each belongs to, checking as it goes that the K it finds
is the K that order predicts -- if that check fails the ordering assumption is wrong and
nothing is written.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ALPHAS = (0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 5.0, 10.0)
KS = (2, 3, 4, 5, 6, 7, 8, 9, 10)
RESULTS = Path("results")


def repair(path, rows_per_mixture, n_mixtures=20):
    """Assign alpha by walking the cells in the order the driver wrote them."""
    if not path.exists():
        print(f"{path.name}: absent")
        return
    df = pd.read_csv(path)
    if df.alpha.notna().all():
        print(f"{path.name}: alpha already present, nothing to do")
        return

    cursor, assigned, mismatch = 0, [], 0
    for alpha in ALPHAS:
        for K in KS:
            width = rows_per_mixture(K) * n_mixtures
            block = df.iloc[cursor:cursor + width]
            if block.empty:
                break
            if not (block.K == K).all():
                mismatch += 1
            assigned.extend([alpha] * len(block))
            cursor += len(block)
        if cursor >= len(df):
            break

    if mismatch:
        print(f"{path.name}: {mismatch} blocks do not match the expected K -- NOT written")
        return
    if cursor != len(df):
        print(f"{path.name}: covered {cursor} of {len(df)} rows -- NOT written")
        return

    df["alpha"] = assigned
    df.to_csv(path, index=False)
    counts = df.groupby(["alpha", "K"]).mixture_id.nunique()
    print(f"{path.name}: {len(df)} rows, {counts.size} cells, "
          f"{sorted(counts.unique())} mixtures per cell, alphas {sorted(df.alpha.unique())}")


def main():
    n_horizons = 2                       # the sweeps store n = 400 and n = 1000
    n_algorithms = 4
    repair(RESULTS / "recovery_cluster_hmm.csv", lambda K: n_algorithms * n_horizons)
    repair(RESULTS / "recovery_eta_hmm.csv", lambda K: n_horizons)
    repair(RESULTS / "recovery_gamma_hmm.csv", lambda K: K * (K + 1) // 2 * n_horizons)

    for name in ("path_cluster_hmm.csv", "path_eta_hmm.csv"):
        path = RESULTS / name
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.alpha.notna().all():
            print(f"{name}: alpha already present")
            continue
        df["alpha"] = 0.3                # sweep_path.py runs one alpha, its --alpha default
        df.to_csv(path, index=False)
        print(f"{name}: {len(df)} rows set to alpha = 0.3")


if __name__ == "__main__":
    sys.exit(main())
