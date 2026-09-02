"""Sweep the (alpha, K) grid for the four ways of estimating K.

    python sweep_khat.py [--n-mixtures 20] [--horizons 400 1000] [--N 200]

Compares, on the same dissimilarity matrices:

  stated      the threshold of Theorem 3.9, a_{N,n} = M (log N / n)^(1/4)
  geomean     sqrt(h_median * h_max) of the profile merge heights, no constant
  ratio       the largest multiplicative jump in those heights, no constant
  ratio-half  the same restricted to l >= ceil(N/2)

The rules read only the dissimilarity matrix, and cost under 1% of the sweep -- all four are
evaluated on every matrix, so comparing them is free once one of them is measured.

eta_n is estimated separately, on an independent Monte Carlo sample, and stored alongside:
the point of the grid is to relate K_hat to the finite-horizon geometry, not merely to
alpha and K.

Restartable: mixtures already present in the output are skipped, so the sweep can be killed
and resumed. Rows are flushed as they are produced.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from clustering import (exact_recovery, geomean_threshold, profile_distances,
                        profile_graph_k, profile_heights, profile_threshold,
                        ratio_threshold)
from experiments import (ResultsWriter, draw_markov_mixture, estimate_gamma_paths, eta_rows,
                         save_mixture, seed_key, stream, univariate_om)

SEED = 20260901
ALPHAS = (0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 5.0, 10.0)     # the grid of the published figures
KS = (2, 3, 4, 5, 6, 7, 8, 9, 10)
D_STATES = 5

GRID_FIELDS = ["alpha", "K", "mixture_id", "dataset_id", "n", "N", "d", "cost_scheme",
               "rule", "threshold", "k_hat", "k_correct", "exact_recovery", "mixture_key"]
ETA_FIELDS = ["alpha", "K", "mixture_id", "n", "cost_scheme", "eta_hat", "eta_ci_low",
              "eta_ci_high", "separation_status", "n_pairs", "level", "mixture_key"]


def rules_for(rho, heights, N, n, M):
    """(name, threshold) for each rule, on one precomputed profile matrix."""
    half = (N + 1) // 2 - 1
    return (
        ("stated", profile_threshold(N, n, M)),
        ("geomean", geomean_threshold(rho, heights)),
        ("ratio", ratio_threshold(rho, heights)),
        ("ratio-half", ratio_threshold(rho, heights, floor=half)),
    )


def done_mixtures(path):
    """Mixtures already written, so a killed run resumes instead of restarting."""
    if not Path(path).exists():
        return set()
    with open(path, newline="", encoding="utf-8") as handle:
        return {(float(r["alpha"]), int(r["K"]), int(r["mixture_id"]))
                for r in csv.DictReader(handle)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-mixtures", type=int, default=20)
    parser.add_argument("--n-datasets", type=int, default=1)
    parser.add_argument("--n-gamma", type=int, default=60)
    parser.add_argument("--horizons", type=int, nargs="+", default=[400, 1000])
    parser.add_argument("--N", type=int, default=200)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    n_grid = np.asarray(sorted(args.horizons), dtype=np.int64)
    om = univariate_om("constant", D_STATES)
    out = Path(args.out)
    grid_path, eta_path = out / "khat_grid.csv", out / "khat_eta.csv"
    already = done_mixtures(grid_path)

    cells = [(a, K) for a in ALPHAS for K in KS]
    total = len(cells) * args.n_mixtures
    print(f"{len(cells)} cells x {args.n_mixtures} mixtures x {args.n_datasets} dataset(s), "
          f"N = {args.N}, horizons {list(n_grid)}")
    print(f"{len(already)} mixtures already done, skipping those\n", flush=True)

    grid_writer = ResultsWriter(grid_path, GRID_FIELDS)
    eta_writer = ResultsWriter(eta_path, ETA_FIELDS)
    start = time.perf_counter()
    finished = 0

    try:
        for alpha, K in cells:
            for m in range(args.n_mixtures):
                finished += 1
                if (float(alpha), int(K), m) in already:
                    continue
                mixture = draw_markov_mixture(K, D_STATES, alpha, m, SEED)
                save_mixture(mixture, out / "mixtures")

                estimate = estimate_gamma_paths(mixture, om, n_grid, args.n_gamma,
                                                stream(SEED, "khat-gamma", alpha, K, m))
                eta_writer.write(eta_rows(mixture, om, estimate))

                rows = []
                for r in range(args.n_datasets):
                    key = seed_key("khat-data", alpha, K, m, r)
                    X, truth = mixture.sample_dataset(args.N, int(n_grid[-1]),
                                                      stream(SEED, key))
                    matrices = om.matrices(X, n_grid)
                    for g, n in enumerate(n_grid):
                        rho = profile_distances(matrices[g])
                        heights = profile_heights(rho)
                        for name, threshold in rules_for(rho, heights, args.N, int(n), om.M):
                            k_hat, labels = profile_graph_k(None, rho=rho,
                                                            threshold=threshold,
                                                            return_labels=True)
                            rows.append({
                                "alpha": alpha, "K": K, "mixture_id": m, "dataset_id": r,
                                "n": int(n), "N": args.N, "d": D_STATES,
                                "cost_scheme": om.name, "rule": name,
                                "threshold": threshold, "k_hat": k_hat,
                                "k_correct": int(k_hat == K),
                                "exact_recovery": int(exact_recovery(labels, truth)),
                                "mixture_key": mixture.key,
                            })
                grid_writer.write(rows)

                if finished % 20 == 0:
                    elapsed = time.perf_counter() - start
                    rate = elapsed / max(finished - len(already), 1)
                    left = (total - finished) * rate
                    print(f"  {finished:>5}/{total}  alpha={alpha:<5} K={K:<2}  "
                          f"{elapsed/60:6.1f} min elapsed, ~{left/60:6.1f} min left",
                          flush=True)
    finally:
        grid_writer.close()
        eta_writer.close()

    print(f"\ndone in {(time.perf_counter() - start)/60:.1f} min")
    print(f"  {grid_path}\n  {eta_path}")


if __name__ == "__main__":
    main()
