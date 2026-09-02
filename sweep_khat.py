"""Sweep the (alpha, K) grid for the four ways of estimating K.

    python sweep_khat.py [--n-mixtures 20] [--horizons 400 1000] [--N 200]

Compares, on the same dissimilarity matrices:

  stated      the threshold of the current Theorem 3.9, M (log N / n)^(1/4)
  safeguard   the proposed replacement, max{sqrt(h_med h_max), h_max (log N / n)^(1/4)}
  asw-pam     K by maximal average silhouette width over PAM, the applied default
  asw-average the same over average linkage
  geomean     sqrt(h_med h_max) alone, carried as a diagnostic of what the floor costs

The profile rules cost under 1% of the sweep, so carrying the diagnostic is free.

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

from clustering import (asw_select_k, exact_recovery, geomean_threshold,
                        profile_distances, profile_graph_k, profile_heights,
                        profile_threshold, safeguard_threshold)
from experiments import (ResultsWriter, draw_hmm_mixture, draw_markov_mixture,
                         estimate_gamma_paths, eta_rows, multichannel_om, save_mixture,
                         seed_key, stream, univariate_om)

SEED = 20260901
ALPHAS = (0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 5.0, 10.0)     # the grid of the published figures
KS = (2, 3, 4, 5, 6, 7, 8, 9, 10)
D_STATES = 5

GRID_FIELDS = ["alpha", "K", "mixture_id", "dataset_id", "n", "N", "d", "cost_scheme",
               "rule", "threshold", "k_hat", "k_correct", "exact_recovery", "mixture_key"]

K_MAX_ASW = 12          # the grid tops out at K = 10; 12 leaves the ASW rules room to overshoot
ETA_FIELDS = ["alpha", "K", "mixture_id", "n", "cost_scheme", "eta_hat", "eta_ci_low",
              "eta_ci_high", "separation_status", "n_pairs", "level", "mixture_key"]


def evaluate_rules(D, rho, heights, N, n, M, with_asw):
    """(name, threshold, k_hat, labels) for every rule, on one dissimilarity matrix.

    Three rules have a status and are reported as such: `stated` is the threshold of the
    current Theorem 3.9, `safeguard` is the proposed replacement, and the ASW rules are the
    applied default. `geomean` is carried only as a diagnostic -- `safeguard` is exactly
    `geomean` whenever the floor does not bind, so the pair measures what the floor costs.
    Variants without a consistency proof are deliberately not run: the comparison the paper
    makes is a consistent rule against the one practice uses, not a search over heuristics.

    The profile rules differ only in where they cut the same rho, so they share one O(N^3)
    profile computation. The ASW rules do not read rho at all: they cluster at every k and
    score the result, which is why they cost an order of magnitude more.
    """
    out = []
    for name, threshold in (("stated", profile_threshold(N, n, M)),
                            ("geomean", geomean_threshold(rho, heights)),
                            ("safeguard", safeguard_threshold(rho, n, heights))):
        k_hat, labels = profile_graph_k(None, rho=rho, threshold=threshold,
                                        return_labels=True)
        out.append((name, threshold, k_hat, labels))
    if with_asw:
        for method, name in (("pam", "asw-pam"), ("average", "asw-average")):
            k_hat, labels = asw_select_k(D, k_max=K_MAX_ASW, method=method,
                                         return_labels=True)
            out.append((name, "", k_hat, labels))
    return out


# --- the mechanism the sweep runs on ----------------------------------------
#
# Markov chains index difficulty by the Dirichlet concentration alpha. HMMs have no such
# knob, so the same axis becomes the emission concentration alpha_B, read the same way.
# The rules for K read only rho, which reads only the dissimilarity matrix, so none of them
# knows which mechanism produced it.

HMM_STATES = 4
HMM_VARS = 5
HMM_CATEGORIES = [5] * 5


def build_om(kind, cost="constant"):
    if kind == "markov":
        return univariate_om(cost, D_STATES)
    return multichannel_om(cost, HMM_CATEGORIES, sub=2.0, indel=1.0)


def build_mixture(kind, K, alpha, mixture_id, seed):
    if kind == "markov":
        return draw_markov_mixture(K, D_STATES, alpha, mixture_id, seed)
    return draw_hmm_mixture(K, HMM_STATES, HMM_VARS, HMM_CATEGORIES, mixture_id, seed,
                            alpha_A=0.5, alpha_B=alpha)


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
    parser.add_argument("--mechanism", default="markov", choices=("markov", "hmm"),
                        help="markov: mixtures of first-order chains. hmm: mixtures of "
                             "homogeneous multichannel HMMs, five channels of five "
                             "letters, with alpha read as alpha_B")
    parser.add_argument("--tag", default="", help="suffix for the output files")
    parser.add_argument("--asw", action="store_true",
                        help="also run the applied default, K by maximal silhouette width")
    parser.add_argument("--alphas", type=float, nargs="+", default=None,
                        help="restrict the alpha grid")
    args = parser.parse_args()

    n_grid = np.asarray(sorted(args.horizons), dtype=np.int64)
    om = build_om(args.mechanism)
    out = Path(args.out)
    suffix = f"_{args.tag}" if args.tag else ""
    grid_path = out / f"khat_grid{suffix}.csv"
    eta_path = out / f"khat_eta{suffix}.csv"
    already = done_mixtures(grid_path)

    alphas = tuple(args.alphas) if args.alphas else ALPHAS
    cells = [(a, K) for a in alphas for K in KS]
    total = len(cells) * args.n_mixtures
    print(f"{len(cells)} cells x {args.n_mixtures} mixtures x {args.n_datasets} dataset(s), "
          f"N = {args.N}, horizons {list(n_grid)}, alphas {list(alphas)}"
          f"{', with ASW' if args.asw else ''}")
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
                mixture = build_mixture(args.mechanism, K, alpha, m, SEED)
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
                        for name, threshold, k_hat, labels in evaluate_rules(
                                matrices[g], rho, heights, args.N, int(n), om.M, args.asw):
                            rows.append({
                                "alpha": alpha, "K": K, "mixture_id": m, "dataset_id": r,
                                "n": int(n), "N": args.N, "d": mixture.d,
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
