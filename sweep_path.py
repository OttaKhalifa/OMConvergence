"""ARI and exact recovery against the horizon, at fixed N -- figure block C.

    python sweep_path.py [--n-mixtures 6] [--n-datasets 20] [--N 200]

The published Figure 4 fixed the kernels once and replicated the data 50 times at
alpha = 0.3, K = 4, N = 800 -- so it described one draw of kernels. Nothing in it separated
what is a property of the mixture from what is a property of that particular draw. This
driver keeps the design and adds the outer level: several mixtures, each held fixed while
its datasets are replicated.

Each replicate is a genuine sample path: the same N sequences read at growing lengths, via
the nested prefixes of `om_matrices`. The spread between replicates is therefore the
variability of one experiment, not an interval around a mean.

`eta_n` is estimated on an independent sample at the same horizons, so a curve can be read
against the geometry of the mixture that produced it.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from experiments import (ALGORITHMS, ResultsWriter, draw_hmm_mixture,
                         draw_markov_mixture, estimate_gamma_paths, eta_rows,
                         multichannel_om, run_dataset, save_mixture, stream,
                         univariate_om)

SEED = 20260901
D_STATES = 5

PATH_FIELDS = ["alpha", "K", "d", "mixture_id", "dataset_id", "n", "N", "cost_scheme",
               "algorithm", "exact_recovery", "ari", "k_hat", "k_correct",
               "exact_recovery_at_k_hat", "pam_one_swap_certified", "pam_hit_cap",
               "mixture_key"]
ETA_FIELDS = ["alpha", "K", "mixture_id", "n", "cost_scheme", "eta_hat", "eta_ci_low",
              "eta_ci_high", "separation_status", "n_pairs", "level", "mixture_key"]


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
    if not Path(path).exists():
        return set()
    with open(path, newline="", encoding="utf-8") as handle:
        return {(float(r["alpha"]), int(r["K"]), int(r["mixture_id"]), int(r["dataset_id"]))
                for r in csv.DictReader(handle)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.3,
                        help="the published path used 0.3: distinguishable chains that "
                             "still mix fast")
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--n-mixtures", type=int, default=6)
    parser.add_argument("--n-datasets", type=int, default=20)
    parser.add_argument("--n-gamma", type=int, default=60)
    parser.add_argument("--N", type=int, default=200)
    parser.add_argument("--horizons", type=int, nargs="+",
                        default=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    parser.add_argument("--cost", default="constant", choices=("constant", "random", "trate"))
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
    path_file = out / f"path_cluster{suffix}.csv"
    eta_file = out / f"path_eta{suffix}.csv"
    already = done_runs(path_file)

    total = args.n_mixtures * args.n_datasets
    print(f"alpha = {args.alpha}, K = {args.K}, N = {args.N}, cost scheme {args.cost}")
    print(f"{args.n_mixtures} mixtures x {args.n_datasets} datasets, horizons "
          f"{list(n_grid)}")
    print(f"{len(already)} runs already done, skipping those\n", flush=True)

    path_writer = ResultsWriter(path_file, PATH_FIELDS)
    eta_writer = ResultsWriter(eta_file, ETA_FIELDS)
    start = time.perf_counter()
    finished = fresh = 0

    try:
        for m in range(args.n_mixtures):
            mixture = build_mixture(args.mechanism, args.K, args.alpha, m, SEED)
            wrote_geometry = False
            for r in range(args.n_datasets):
                finished += 1
                if (float(args.alpha), int(args.K), m, r) in already:
                    continue
                if not wrote_geometry:
                    save_mixture(mixture, out / "mixtures")
                    estimate = estimate_gamma_paths(
                        mixture, om, n_grid, args.n_gamma,
                        stream(SEED, "path-gamma", args.alpha, args.K, m))
                    eta_writer.write(eta_rows(mixture, om, estimate))
                    wrote_geometry = True

                rows = run_dataset(mixture, om, args.N, n_grid,
                                   stream(SEED, "path-data", args.alpha, args.K, m, r), r,
                                   algorithms=ALGORITHMS, pam_restarts=args.pam_restarts)
                for row in rows:
                    row["dataset_id"] = r
                path_writer.write([{k: row.get(k, "") for k in PATH_FIELDS} for row in rows])
                fresh += 1

                if fresh % 10 == 0:
                    elapsed = time.perf_counter() - start
                    left = (total - finished) * elapsed / fresh
                    print(f"  {finished:>4}/{total}  mixture {m}  "
                          f"{elapsed/60:5.1f} min elapsed, ~{left/60:5.1f} min left",
                          flush=True)
    finally:
        path_writer.close()
        eta_writer.close()

    print(f"\ndone in {(time.perf_counter() - start)/60:.1f} min")
    print(f"  {path_file}\n  {eta_file}")


if __name__ == "__main__":
    main()
