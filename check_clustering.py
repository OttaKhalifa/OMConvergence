"""Check the clustering routines of `om_lib` against references.

Nothing here belongs in the experiment notebooks: this is about the code being right, not
about the mixtures. Run it once after touching `om_lib`, and cite it in the paper for the two
claims it establishes.

    python3 check_clustering.py            # default sizes, a few seconds
    python3 check_clustering.py --big      # up to C(40, 4) medoid sets, about a minute

What it establishes
-------------------
1. `single_linkage_tree` returns the merge heights of `scipy.cluster.hierarchy.linkage`
   (method="single") and `adjusted_rand_index` matches `sklearn.metrics.adjusted_rand_score`,
   to machine precision.
2. `kmedoids` -- PAM's BUILD followed by alternating refinement, best of several restarts --
   reaches the *global* minimiser of Phi found by `kmedoids_exhaustive`, wherever enumerating
   the C(N, K) medoid sets is affordable. The consistency theorem is about that global
   minimiser, so this is what licenses using the heuristic at the sizes where enumeration is
   hopeless: C(800, 4) = 1.7e10.
"""

from __future__ import annotations

import argparse
import sys
from math import comb

import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

from om_lib import (adjusted_rand_index, cost_scheme, cut_at_k, kmedoids, kmedoids_exhaustive,
                    om_matrices, sample_mixture, single_linkage_tree)

D_STATES = 5
ALPHA = 0.3
SEED = 20260803
MED_RESTARTS = 10          # keep in step with the notebook
MED_SEED = 0

S_COST, DELTA_COST = cost_scheme("constant", D_STATES, sub=2.0, indel=1.0)


def _matrix(K, N, n, rng):
    """One OM dissimilarity matrix from a fresh K-component mixture, with its labels."""
    mix = sample_mixture(K, N, n, D_STATES, ALPHA, rng)
    return om_matrices(mix["X"], [n], S_COST, DELTA_COST)[0], mix["labels"]


def check_single_linkage(rng, cases, tol=1e-12):
    """Merge heights against scipy, and the ARI of the cut at K against sklearn."""
    print("single linkage vs scipy.cluster.hierarchy, ARI vs sklearn.metrics")
    ok = True
    for K, N, n in cases:
        D, labels = _matrix(K, N, n, rng)
        heights, edges = single_linkage_tree(D)
        Z = sch.linkage(squareform(D, checks=False), method="single")
        dh = float(np.abs(heights - Z[:, 2]).max())
        ours = cut_at_k(edges, N, K)
        dari = abs(adjusted_rand_index(ours, labels) - adjusted_rand_score(ours, labels))
        good = dh <= tol and dari <= tol
        ok &= good
        print(f"    K={K}, N={N:>4}, n={n:>4}: max |height difference| = {dh:.2e}, "
              f"|ARI difference| = {dari:.2e}   {'ok' if good else 'FAILED'}")
    return ok


def check_kmedoids(rng, cases, tol=1e-9):
    """The heuristic against the global minimiser of Phi, by enumeration."""
    print("K-medoids: heuristic vs exhaustive minimisation of Phi")
    ok = True
    for K, N, n, reps in cases:
        reached = same = 0
        worst = 0.0
        for _ in range(reps):
            D, _ = _matrix(K, N, n, rng)
            lab_h, _, obj_h = kmedoids(D, K, np.random.default_rng(MED_SEED),
                                       n_restarts=MED_RESTARTS)
            lab_e, _, obj_e = kmedoids_exhaustive(D, K)
            reached += obj_h <= obj_e + tol
            same += adjusted_rand_index(lab_h, lab_e) > 1 - tol
            worst = max(worst, (obj_h - obj_e) / max(obj_e, 1e-12))
        ok &= reached == reps
        print(f"    K={K}, N={N:>3}, n={n:>4}, C(N,K)={comb(N, K):>8}: optimum reached "
              f"{reached}/{reps}, same partition {same}/{reps}, worst excess "
              f"{100 * worst:.3f}%   {'ok' if reached == reps else 'FAILED'}")
    return ok


def check_on_gaussians(rng, K=8, per_cluster=50, dim=4, spread=2.2, tol=1e-12):
    """The same routines on a mixture of K overlapping Gaussians, against scikit-learn.

    Nothing here involves Markov chains or the OM distance: the point is a testbed in general
    position, where the distances are all distinct, so a disagreement can only be a bug and
    never a tie broken differently. `spread` is the distance between cluster centres in units
    of the within-cluster standard deviation -- small enough that the clusters overlap.
    """
    from scipy.spatial.distance import pdist
    from sklearn.cluster import AgglomerativeClustering

    centres = rng.normal(scale=spread, size=(K, dim))
    labels = np.repeat(np.arange(K), per_cluster)
    X = rng.normal(loc=centres[labels], scale=1.0)
    N = X.shape[0]
    D = squareform(pdist(X))
    off = D[np.triu_indices(N, 1)]

    heights, edges = single_linkage_tree(D)
    Z = sch.linkage(pdist(X), method="single")
    ours = cut_at_k(edges, N, K)
    theirs = AgglomerativeClustering(n_clusters=K, linkage="single",
                                     metric="precomputed").fit_predict(D)

    dh = float(np.abs(heights - Z[:, 2]).max())
    ari_vs_sklearn = adjusted_rand_index(ours, theirs)
    dari = abs(adjusted_rand_index(ours, labels) - adjusted_rand_score(ours, labels))
    ok = dh <= tol and ari_vs_sklearn > 1 - tol and dari <= tol

    print(f"{K} overlapping Gaussians in dimension {dim}, {N} points, centres {spread} sd apart")
    print(f"    distinct distances                       : {np.unique(off).size} / {off.size}")
    print(f"    max |height difference| vs scipy         : {dh:.2e}")
    print(f"    ARI between our cut and sklearn's        : {ari_vs_sklearn:.12f}")
    print(f"    |our ARI - sklearn ARI| against the truth: {dari:.2e}")
    print(f"    ARI against the true labels              : "
          f"{adjusted_rand_index(ours, labels):.3f}   {'ok' if ok else 'FAILED'}")

    # and the medoids, against the global optimum, on a subsample small enough to enumerate
    sub = rng.choice(N, size=24, replace=False)
    Ds = D[np.ix_(sub, sub)]
    lab_h, _, obj_h = kmedoids(Ds, 3, np.random.default_rng(MED_SEED), n_restarts=MED_RESTARTS)
    lab_e, _, obj_e = kmedoids_exhaustive(Ds, 3)
    med_ok = obj_h <= obj_e + 1e-9 and adjusted_rand_index(lab_h, lab_e) > 1 - 1e-9
    print(f"    K-medoids on 24 of them, K=3, C(24,3)={comb(24, 3)}: Phi = {obj_h:.6f} vs "
          f"{obj_e:.6f} exhaustive   {'ok' if med_ok else 'FAILED'}")
    return ok and med_ok


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--big", action="store_true",
                        help="add the (N, K) where enumeration costs about a minute")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    linkage_cases = [(2, 60, 300), (4, 120, 300), (10, 200, 200)]
    medoid_cases = [(2, 40, 300, 6), (3, 40, 300, 6), (4, 30, 300, 6)]
    if args.big:
        linkage_cases.append((5, 400, 400))
        medoid_cases.append((4, 40, 400, 4))

    ok = check_on_gaussians(rng)
    print()
    ok &= check_single_linkage(rng, linkage_cases)
    print()
    ok &= check_kmedoids(rng, medoid_cases)

    print(f"\nat the sizes the notebook actually uses, C(800, 4) = {comb(800, 4):.2e} and "
          f"C(800, 10) = {comb(800, 10):.2e}:\nenumeration is out of reach, which is why the "
          f"heuristic has to be validated here instead.")
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
