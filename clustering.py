"""Clustering of a precomputed dissimilarity matrix, and how it is scored.

Everything downstream of the dissimilarity matrix: the three estimators compared in the
paper, and the quantities the figures report about their output. None of these functions
knows how D was computed, so they apply unchanged to either OM path of ``om``.

Contents
--------
Single linkage : ``single_linkage_tree``, ``cut_at_k``, ``largest_gap_k``
Average linkage : ``average_linkage_labels``
K-medoids      : ``kmedoids``, ``kmedoids_objective``
Scoring        : ``adjusted_rand_index``, ``gamma_block_means``, ``separation_levels``
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import linkage as scipy_linkage
from scipy.spatial.distance import squareform

# ---------------------------------------------------------------------------
# Single-linkage clustering
# ---------------------------------------------------------------------------


def single_linkage_tree(D):
    """Merge heights and merged pairs of the single-linkage dendrogram of D.

    Single linkage coincides with Kruskal's algorithm: the l-th merge occurs at the l-th
    smallest edge of a minimum spanning tree of the complete graph weighted by D. Returns
    (heights, edges) with `heights` the non-decreasing (N-1,) vector of merge heights
    h_1 <= ... <= h_{N-1} of the paper and `edges` the corresponding (N-1, 2) pairs.
    """
    D = np.asarray(D, dtype=float)
    N = D.shape[0]
    if N < 2:
        return np.empty(0), np.empty((0, 2), dtype=np.int64)
    ii, jj = np.triu_indices(N, k=1)
    order = np.argsort(D[ii, jj], kind="stable")
    parent = np.arange(N)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    heights = np.empty(N - 1)
    edges = np.empty((N - 1, 2), dtype=np.int64)
    m = 0
    for p in order:
        a, b = find(ii[p]), find(jj[p])
        if a != b:
            parent[a] = b
            heights[m] = D[ii[p], jj[p]]
            edges[m] = (ii[p], jj[p])
            m += 1
            if m == N - 1:
                break
    return heights, edges


def _components(edges, N):
    """Labels of the connected components of the graph on N vertices with these edges."""
    parent = np.arange(N)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    roots = np.array([find(i) for i in range(N)])
    return np.unique(roots, return_inverse=True)[1]


def cut_at_k(edges, N, K):
    """Single-linkage partition into K blocks: the first N-K merges, i.e. Definition 3.2
    stopped at N-K steps. K >= N returns the singletons."""
    return _components(edges[:max(N - K, 0)], N)


def largest_gap_k(heights):
    """The largest-gap estimator of K, equation (6): K = N - argmax_l (h_{l+1} - h_l),
    over 1 <= l <= N-2, taking the smallest index in case of tie."""
    N = heights.size + 1
    if N < 3:
        return N
    gaps = np.diff(heights)                  # gaps[l - 1] = h_{l+1} - h_l, l = 1, ..., N-2
    return int(N - (1 + int(np.argmax(gaps))))


def average_linkage_labels(D, K):
    """Average-linkage partition of a precomputed dissimilarity matrix into K blocks.

    Delegated to scipy, whose implementation is the reference one. This estimator is *not*
    covered by the paper's theory -- Remark 3.4 covers bracketed linkages, of which it is one -- and
    is here only as an empirical control for single linkage. The contrast is the point: single
    linkage merges two blocks on the *smallest* dissimilarity between them, so one aberrant
    sequence suffices to chain them together, and the cut at K then spends a whole block on
    that sequence; average linkage merges on the mean over all pairs between the two blocks,
    where no single sequence can carry the decision.

    The tree comes from scipy, the reference implementation of UPGMA. Note that `squareform`
    hands it the *condensed* form, so it reads a dissimilarity matrix and not a set of
    observation vectors. The cut, however, is ours: exactly the first N-K merges, as in
    `cut_at_k`, rather than `fcluster(..., "maxclust")`, which returns *at most* K blocks by
    choosing a threshold and can therefore return fewer when merge heights are tied -- and
    integer costs make them tied constantly. Both linkages then obey literally the same rule.
    """
    D = np.asarray(D, dtype=float)
    N = D.shape[0]
    if N < 2:
        return np.zeros(N, dtype=np.int64)
    Z = scipy_linkage(squareform(D, checks=False), method="average")
    parent = np.arange(2 * N - 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for m in range(max(N - K, 0)):                    # the first N - K merges of the tree
        parent[find(int(Z[m, 0]))] = N + m
        parent[find(int(Z[m, 1]))] = N + m
    seen = {}
    return np.array([seen.setdefault(find(i), len(seen)) for i in range(N)], dtype=np.int64)



# ---------------------------------------------------------------------------
# K-medoids
# ---------------------------------------------------------------------------


def kmedoids_objective(D, medoids):
    """Phi(m) = sum_i min_k D[i, m_k], the criterion the K-medoids theorem minimises."""
    return float(np.asarray(D, dtype=float)[:, list(medoids)].min(axis=1).sum())


def kmedoids(D, K, rng, n_restarts=10, max_iter=100):
    """K-medoids on a precomputed dissimilarity matrix. Returns (labels, medoids, objective).

    The theorem is about the *global* minimiser of Phi, which would require the C(N, K) medoid
    sets to be enumerated -- out of reach past a few dozen sequences, C(800, 4) being 1.7e10.
    The minimisation is therefore heuristic, in the classical two stages:

    * PAM's BUILD, deterministic: start from the medoid minimising the total dissimilarity,
      then repeatedly add the point that reduces Phi the most;
    * alternating refinement: assign every point to its nearest medoid, then replace each
      medoid by the member of its cluster minimising the within-cluster sum of
      dissimilarities, until the medoid set is stable.

    `n_restarts - 1` further runs start from random medoid sets and the lowest Phi wins, which
    is what `rng` is for. Checked against the global optimum by enumeration wherever that was
    affordable: on this project's mixtures the heuristic reached it every time at N = 40,
    K <= 4.
    """
    D = np.asarray(D, dtype=float)
    N = D.shape[0]
    if K < 1 or K > N:
        raise ValueError("K must lie between 1 and N")
    best = None
    for r in range(n_restarts):
        if r == 0:
            med = [int(np.argmin(D.sum(axis=1)))]
            while len(med) < K:
                gain = np.maximum(D[:, med].min(axis=1)[None, :] - D, 0.0).sum(axis=1)
                gain[med] = -np.inf
                med.append(int(np.argmax(gain)))
            med = np.array(med, dtype=np.int64)
        else:
            med = rng.choice(N, size=K, replace=False)
        for _ in range(max_iter):
            labels = np.argmin(D[:, med], axis=1)
            new = med.copy()
            for k in range(K):
                idx = np.flatnonzero(labels == k)
                if idx.size:
                    new[k] = idx[np.argmin(D[np.ix_(idx, idx)].sum(axis=1))]
            if np.array_equal(np.sort(new), np.sort(med)):
                break
            med = new
        obj = kmedoids_objective(D, med)
        if best is None or obj < best[2]:
            best = (np.argmin(D[:, med], axis=1), med, obj)
    return best



# ---------------------------------------------------------------------------
# Agreement between partitions, and plug-in estimates of Gamma
# ---------------------------------------------------------------------------


def adjusted_rand_index(a, b):
    """Adjusted Rand Index between two labellings of the same N items.

    Equals 1 exactly when the two partitions coincide, and has expectation 0 under the
    permutation model, so ARI = 1 is the exact-recovery event of Theorem 3.3.
    """
    a = np.unique(np.asarray(a), return_inverse=True)[1]
    b = np.unique(np.asarray(b), return_inverse=True)[1]
    N = a.size
    if N < 2:
        return 1.0
    table = np.zeros((a.max() + 1, b.max() + 1), dtype=np.int64)
    np.add.at(table, (a, b), 1)

    def comb2(x):
        return (x * (x - 1) / 2.0).sum()

    index = comb2(table.astype(float))
    exp_a, exp_b = comb2(table.sum(1).astype(float)), comb2(table.sum(0).astype(float))
    expected = exp_a * exp_b / (N * (N - 1) / 2.0)
    maximum = 0.5 * (exp_a + exp_b)
    if maximum == expected:              # both partitions trivial: they then coincide
        return 1.0
    return float((index - expected) / (maximum - expected))


def gamma_block_means(D, labels, K):
    """Plug-in estimate of Gamma from a dissimilarity matrix and the true labels.

    Gamma_hat[k, l] is the average of hat-gamma_n(i, j) over the pairs with
    (Z_i, Z_j) = (k, l), the diagonal of D being excluded from the within-class blocks.
    Averaging over the |C_k|(|C_k|-1)/2 or |C_k||C_l| available pairs makes this far less
    noisy than the two-realisations-per-component estimate of `assumptions.ipynb`, at no
    extra cost once D is computed. Blocks with no available pair are NaN.
    """
    D = np.asarray(D, dtype=float)
    labels = np.asarray(labels)
    onehot = (labels[:, None] == np.arange(K)[None, :]).astype(float)
    sums = onehot.T @ D @ onehot                      # diagonal of D is zero, so it drops out
    counts = np.outer(onehot.sum(0), onehot.sum(0))
    np.fill_diagonal(counts, onehot.sum(0) * (onehot.sum(0) - 1.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / np.maximum(counts, 1e-300), np.nan)


def separation_levels(D, labels, K):
    """Delta_in, Delta_out and Delta_out^max from the plug-in Gamma of `gamma_block_means`."""
    G = gamma_block_means(D, labels, K)
    offdiag = G[~np.eye(K, dtype=bool)]
    if np.all(np.isnan(np.diag(G))) or np.all(np.isnan(offdiag)):
        return {"in": np.nan, "out": np.nan, "out_max": np.nan}
    return {"in": float(np.nanmax(np.diag(G))),
            "out": float(np.nanmin(offdiag)),
            "out_max": float(np.nanmax(offdiag))}
