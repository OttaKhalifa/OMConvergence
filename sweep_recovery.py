"""Sweep the (alpha, K) grid for clustering recovery at known K -- figure block B.

    python sweep_recovery.py [--n-mixtures 20] [--n-datasets 1] [--horizons 400 1000]

The chain the numerical section is built on is

    generative difficulty (alpha, K)  ->  finite-horizon geometry eta_n  ->  recovery,

and this driver fills its last link. For each cell it draws `R_mix` mixtures, estimates
Gamma^(n) on an independent sample, then simulates `R_data` clustering datasets and scores
single, complete and average linkage and PAM on each -- the same dissimilarity matrix for
all four, so what separates them is the algorithm and nothing else.

Exact recovery is the primary outcome, being what the theorems are statements about; ARI is
kept as a graded secondary reading of the same partition. Every PAM run carries the flag
saying whether its medoid set was certified one-swap stationary, which is the hypothesis of
the theorem rather than a detail of the implementation.

K_hat from the safeguarded rule is recorded on the same matrices, so the two questions of
the K-selection block -- does the rule find K, and does the partition it implies recover the
truth -- are answered from this run too.

Restartable at the level of (cell, mixture, dataset), so a second pass with a larger
--n-datasets adds to the table rather than redoing it.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from experiments import (ALGORITHMS, ResultsWriter, draw_hmm_mixture,
                         draw_markov_mixture, estimate_gamma_paths, eta_rows,
                         gamma_rows, multichannel_om, run_dataset, save_mixture,
                         seed_key, stream, univariate_om)

SEED = 20260901
ALPHAS = (0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 5.0, 10.0)     # the grid of the published figures
KS = (2, 3, 4, 5, 6, 7, 8, 9, 10)
D_STATES = 5

CLUSTER_FIELDS = ["alpha", "K", "d", "mixture_id", "n", "N", "cost_scheme", "algorithm",
                  "dataset_id", "exact_recovery", "ari", "k_hat", "k_correct",
                  "exact_recovery_at_k_hat", "pam_one_swap_certified", "pam_hit_cap",
                  "mixture_key"]
ETA_FIELDS = ["alpha", "K", "mixture_id", "n", "cost_scheme", "eta_hat", "eta_ci_low",
              "eta_ci_high", "separation_status", "n_pairs", "level", "mixture_key"]
GAMMA_FIELDS = ["alpha", "K", "mixture_id", "n", "cost_scheme", "k", "l", "gamma_hat",
                "se", "ci_low", "ci_high", "n_pairs", "level", "mixture_key"]


# --- the mechanism the sweep runs on ----------------------------------------
#
# Markov chains index difficulty by the Dirichlet concentration alpha. HMMs have no such
# knob, so the same axis becomes the emission concentration alpha_B, read the same way:
# small values give sharply peaked emissions, hence components that are easy to tell apart.
# Everything downstream -- the geometry, the four algorithms, the rules for K -- reads only
# a dissimilarity matrix and does not know which mechanism produced it.

HMM_STATES = 4
HMM_VARS = 5
HMM_CATEGORIES = [5] * 5


def build_om(kind, cost, seed):
    """The dissimilarity, built once: it does not depend on the mixture."""
    if kind == "markov":
        return univariate_om(cost, D_STATES, rng=stream(seed, "cost", cost))
    return multichannel_om(cost, HMM_CATEGORIES, sub=2.0, indel=1.0)


def build_mixture(kind, K, alpha, mixture_id, seed):
    """One mixture of the requested mechanism, drawn on its own stream."""
    if kind == "markov":
        return draw_markov_mixture(K, D_STATES, alpha, mixture_id, seed)
    return draw_hmm_mixture(K, HMM_STATES, HMM_VARS, HMM_CATEGORIES, mixture_id, seed,
                            alpha_A=0.5, alpha_B=alpha)


def done_runs(path):
    """(cell, mixture, dataset) triples already written, so a run resumes where it stopped."""
    if not Path(path).exists():
        return set()
    with open(path, newline="", encoding="utf-8") as handle:
        return {(float(r["alpha"]), int(r["K"]), int(r["mixture_id"]), int(r["dataset_id"]))
                for r in csv.DictReader(handle)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-mixtures", type=int, default=20)
    parser.add_argument("--n-datasets", type=int, default=1)
    parser.add_argument("--n-gamma", type=int, default=60)
    parser.add_argument("--horizons", type=int, nargs="+", default=[400, 1000])
    parser.add_argument("--N", type=int, default=200)
    parser.add_argument("--cost", default="constant", choices=("constant", "random", "trate"))
    parser.add_argument("--alphas", type=float, nargs="+", default=None)
    parser.add_argument("--pam-restarts", type=int, default=1)
    parser.add_argument("--mechanism", default="markov", choices=("markov", "hmm"),
                        help="markov: mixtures of first-order chains. hmm: mixtures of "
                             "homogeneous multichannel HMMs, five channels of five "
                             "letters, with alpha read as alpha_B")
    parser.add_argument("--tag", default="")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    n_grid = np.asarray(sorted(args.horizons), dtype=np.int64)
    om = build_om(args.mechanism, args.cost, SEED)
    out = Path(args.out)
    suffix = f"_{args.tag}" if args.tag else ""
    cluster_path = out / f"recovery_cluster{suffix}.csv"
    eta_path = out / f"recovery_eta{suffix}.csv"
    gamma_path = out / f"recovery_gamma{suffix}.csv"
    already = done_runs(cluster_path)

    assumption = om.assumption_1()
    metric_ok = all(v for k, v in assumption.items() if k.startswith("("))
    alphas = tuple(args.alphas) if args.alphas else ALPHAS
    cells = [(a, K) for a in alphas for K in KS]
    total = len(cells) * args.n_mixtures * args.n_datasets

    print(f"{len(cells)} cells x {args.n_mixtures} mixtures x {args.n_datasets} dataset(s), "
          f"N = {args.N}, horizons {list(n_grid)}, cost scheme {args.cost}")
    print(f"Assumption 1 holds for this cost scheme: {metric_ok}   (M = {om.M})")
    if not metric_ok:
        failed = [k for k, v in assumption.items() if k.startswith("(") and not v]
        print(f"  !! outside the metric framework: {failed} -- report as robustness, "
              f"not as confirmation of the theorems")
    print(f"{len(already)} runs already done, skipping those\n", flush=True)

    cluster_writer = ResultsWriter(cluster_path, CLUSTER_FIELDS)
    eta_writer = ResultsWriter(eta_path, ETA_FIELDS)
    gamma_writer = ResultsWriter(gamma_path, GAMMA_FIELDS)
    start = time.perf_counter()
    finished = fresh = 0

    try:
        for alpha, K in cells:
            for m in range(args.n_mixtures):
                mixture = build_mixture(args.mechanism, K, alpha, m, SEED)
                wrote_geometry = False
                for r in range(args.n_datasets):
                    finished += 1
                    if (float(alpha), int(K), m, r) in already:
                        continue
                    if not wrote_geometry:
                        save_mixture(mixture, out / "mixtures")
                        estimate = estimate_gamma_paths(
                            mixture, om, n_grid, args.n_gamma,
                            stream(SEED, "recovery-gamma", alpha, K, m))
                        eta_writer.write(eta_rows(mixture, om, estimate))
                        gamma_writer.write(gamma_rows(mixture, om, estimate))
                        wrote_geometry = True

                    rows = run_dataset(mixture, om, args.N, n_grid,
                                       stream(SEED, "recovery-data", alpha, K, m, r), r,
                                       algorithms=ALGORITHMS,
                                       pam_restarts=args.pam_restarts)
                    for row in rows:                     # keep only the declared columns
                        row.pop("dataset_id", None)
                        row["dataset_id"] = r
                    cluster_writer.write([{k: row.get(k, "") for k in CLUSTER_FIELDS}
                                          for row in rows])
                    fresh += 1

                    if fresh % 20 == 0:
                        elapsed = time.perf_counter() - start
                        left = (total - finished) * elapsed / fresh
                        print(f"  {finished:>5}/{total}  alpha={alpha:<5} K={K:<2}  "
                              f"{elapsed/60:6.1f} min elapsed, ~{left/60:6.1f} min left",
                              flush=True)
    finally:
        for writer in (cluster_writer, eta_writer, gamma_writer):
            writer.close()

    print(f"\ndone in {(time.perf_counter() - start)/60:.1f} min")
    for path in (cluster_path, eta_path, gamma_path):
        print(f"  {path}")


if __name__ == "__main__":
    main()
