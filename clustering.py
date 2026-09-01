"""Clustering of a precomputed dissimilarity matrix, and how it is scored.

Everything downstream of the dissimilarity matrix: the three estimators compared in the
paper, and the quantities the figures report about their output. None of these functions
knows how D was computed, so they apply unchanged to either OM path of ``om``.

Contents
--------
Hierarchical   : ``hac_labels`` (single, complete, average), ``single_linkage_tree``,
                 ``cut_at_k``
PAM            : ``pam``, ``pam_objective``, ``pam_certify_one_swap``
Selecting K    : ``profile_graph_k``, ``profile_distances``, ``profile_threshold``
Scoring        : ``exact_recovery`` (the primary outcome), ``adjusted_rand_index``

Two estimators the paper states are implemented here in the form it states them: the
HAC cut is *exactly* the first N-K merges, and PAM is the strictly-improving one-swap
algorithm whose stationary point Theorem 3.8 is about -- not an alternating heuristic
that happens to converge nearby.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import linkage as scipy_linkage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import squareform

# Only for the O(N^3) profile distances; `om` owns the pure-Python fallback.
from om import njit, prange

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


def _cut_scipy_tree(Z, N, K):
    """Labels of the partition left by the first N-K merges of a scipy linkage tree."""
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


HAC_METHODS = ("single", "complete", "average")


def hac_labels(D, K, method="average"):
    """Hierarchical partition of a precomputed dissimilarity matrix into K blocks.

    `method` is one of `HAC_METHODS`. All three are *bracketed* linkages in the sense of
    Remark 3.4 -- single and complete are the two extremes of the bracket, average lies
    between them -- so the consistency theorem covers all three, and comparing them
    measures only what the theory leaves free.

    The tree comes from scipy, the reference implementation; `squareform` hands it the
    *condensed* form, so it reads a dissimilarity matrix and not a set of observation
    vectors. The cut is ours: exactly the first N-K merges, as in `cut_at_k`, rather than
    `fcluster(..., "maxclust")`, which returns *at most* K blocks by choosing a threshold
    and can therefore return fewer when merge heights are tied -- and integer costs make
    them tied constantly. Every linkage then obeys literally the same rule, which is what
    makes the comparison between them meaningful.
    """
    if method not in HAC_METHODS:
        raise ValueError(f"method must be one of {HAC_METHODS}, got {method!r}")
    D = np.asarray(D, dtype=float)
    N = D.shape[0]
    if N < 2:
        return np.zeros(N, dtype=np.int64)
    Z = scipy_linkage(squareform(D, checks=False), method=method)
    return _cut_scipy_tree(Z, N, K)



# ---------------------------------------------------------------------------
# Selecting K: the profile graph of Theorem 3.9
# ---------------------------------------------------------------------------


@njit(cache=True, parallel=True)
def _profile_distances(D):
    N = D.shape[0]
    out = np.zeros((N, N), dtype=np.float64)
    for i in prange(N):
        for j in range(i + 1, N):
            best = 0.0
            for l in range(N):
                if l == i or l == j:
                    continue
                v = D[i, l] - D[j, l]
                if v < 0.0:
                    v = -v
                if v > best:
                    best = v
            out[i, j] = best
            out[j, i] = best
    return out


def profile_distances(D):
    """rho(i, j) = max over l not in {i, j} of |gamma_hat(i, l) - gamma_hat(j, l)|.

    How differently i and j see the rest of the sample. Two sequences of the same component
    have the same finite-horizon mean against every third sequence, so their profiles differ
    only by the fluctuation of gamma_hat; two sequences of different components differ by at
    least eta against any third one drawn from either of their components. That gap, not the
    dissimilarity between i and j themselves, is what separates the two cases -- which is why
    the rule needs no dendrogram and no candidate value of K.

    O(N^3), parallel over i. At N = 800 that is 5e8 inner steps, under a second.
    """
    return _profile_distances(np.ascontiguousarray(D, dtype=np.float64))


def profile_threshold(N, n, M):
    """a_{N,n} = M (log N / n)^{1/4}, equation (8).

    M is the largest substitution cost -- `check_assumption_metric(...)["M = max c_sub"]`,
    or M^mc for multichannel costs. The exponent 1/4 is what makes the threshold sit
    strictly between the two scales it must separate: it goes to 0, so it eventually falls
    below eta, while n a_{N,n}^2 / log N -> infinity, so the uniform concentration still
    covers it.
    """
    if N < 2 or n < 1:
        raise ValueError("need N >= 2 and n >= 1")
    return float(M) * (np.log(N) / n) ** 0.25


def profile_graph_k(D, n, M, threshold=None, return_labels=False):
    """Estimate K as the number of connected components of H_{N,n}, Theorem 3.9.

    i and j are adjacent when rho(i, j) <= a_{N,n}. On the concentration event the graph has
    an edge exactly between same-component pairs, so its components *are* the true clusters:
    the rule returns a partition as well as a count, and `return_labels` hands it back. That
    partition is not what the paper cuts a dendrogram to -- Remark 3.10 estimates K here and
    then runs single linkage or PAM at K_hat -- but it is free, and disagreement between the
    two is a useful diagnostic.

    `threshold` overrides a_{N,n}. The theorem fixes the threshold up to the constant M, and
    that constant is what makes it unusable at feasible horizons: with the default scheme
    M = 2, a_{N,n} is 0.54 at N = 200, n = 1000, while the profile distances it has to
    separate sit at 0.04 within a component and 0.44 between -- so the threshold lands above
    *both* and joins everything into one component. The mechanism is sound and the gap is
    wide; only the scale is wrong, and it closes like n^(-1/4), so an eta of 0.3 would need
    n > 50000. Pass an explicit threshold to run the rule at a calibrated scale, and say so:
    that is outside the constants the theorem fixes.

    Meaningful only for N >= 3: with N = 2 there is no third sequence to profile against,
    every rho is an empty maximum, and the two points are always joined.
    """
    D = np.asarray(D, dtype=float)
    N = D.shape[0]
    if threshold is None:
        threshold = profile_threshold(N, n, M)
    adjacency = profile_distances(D) <= threshold
    np.fill_diagonal(adjacency, False)
    k_hat, labels = connected_components(csr_matrix(adjacency), directed=False)
    return (int(k_hat), labels.astype(np.int64)) if return_labels else int(k_hat)


# ---------------------------------------------------------------------------
# PAM (K-medoids)
# ---------------------------------------------------------------------------


def pam_objective(D, medoids):
    """Phi(M) = (1/N) sum_i min_{m in M} D[i, m], the criterion of Definition 3.7.

    Normalised by N, as in the paper. The normalisation is irrelevant to which medoid set
    minimises Phi, but it makes the value comparable across N, and it is the number the
    theory speaks about.
    """
    D = np.asarray(D, dtype=float)
    return float(D[:, np.asarray(medoids, dtype=np.int64)].min(axis=1).mean())


def _pam_build(D, K):
    """PAM's BUILD: the deterministic greedy initialisation, K medoids.

    Start from the point minimising the total dissimilarity, then repeatedly add the point
    that reduces Phi the most. This is only an initialisation -- the theory constrains the
    *output* of SWAP, not where it starts -- but it is the standard choice and it makes the
    algorithm deterministic when no restarts are asked for.
    """
    medoids = [int(np.argmin(D.sum(axis=1)))]
    while len(medoids) < K:
        nearest = D[:, medoids].min(axis=1)               # (N,) current cost of each point
        gain = np.maximum(nearest[:, None] - D, 0.0).sum(axis=0)   # gain[h], summed over i
        gain[medoids] = -np.inf
        medoids.append(int(np.argmax(gain)))
    return np.array(medoids, dtype=np.int64)


def _swap_table(D, medoids):
    """(table, current): N*Phi after every one-medoid swap, and N*Phi as it stands.

    `table[j, h]` is N*Phi of the medoid set with `medoids[j]` replaced by `h`, for every
    h at once -- including the h that are themselves medoids, which the caller masks out.

    Recomputing Phi from scratch for each of the K(N-K) candidate swaps would make a sweep
    O(K^2 N^2). The bookkeeping below brings it to O(K N^2): removing medoid j changes
    nothing for the points it does not currently serve, and the points it does serve fall
    back on their second nearest medoid. So the cost of point i once j is gone and h is in
    is min(base_j[i], D[i, h]), with base_j[i] the fallback distance -- and the whole row
    of the table is one min-reduction of an (N, N) array.
    """
    N = D.shape[0]
    Dm = D[:, medoids]                                    # (N, K)
    K = Dm.shape[1]
    order = np.argsort(Dm, axis=1, kind="stable")
    rows = np.arange(N)
    assign = order[:, 0]                                  # index *within* `medoids`
    nearest = Dm[rows, assign]
    second = Dm[rows, order[:, 1]] if K >= 2 else np.full(N, np.inf)

    table = np.empty((K, N), dtype=float)
    for j in range(K):
        base = np.where(assign == j, second, nearest)
        table[j] = np.minimum(base[:, None], D).sum(axis=0)
    return table, float(nearest.sum())


def _best_swap(D, medoids):
    """Steepest one-medoid swap: (j, h, delta), delta = N*(Phi_new - Phi_current).

    `delta >= 0` means no swap strictly improves Phi, i.e. the medoid set is one-swap
    stationary. When K = N there is no non-medoid to swap in and delta is +inf.
    """
    table, current = _swap_table(D, medoids)
    table[:, medoids] = np.inf                            # h must be a non-medoid
    flat = int(np.argmin(table))
    j, h = flat // table.shape[1], flat % table.shape[1]
    return int(j), int(h), float(table[j, h] - current)


def pam_certify_one_swap(D, medoids, tol=0.0):
    """True iff no single medoid/non-medoid swap strictly decreases Phi.

    The exhaustive check the theory needs. Theorem 3.8 is not about a global minimiser of
    Phi -- that would need the C(N, K) medoid sets enumerated, C(800, 4) being 1.7e10 --
    but about a set on which no improving one-swap exists, which is exactly what
    Lemma 3.7 then turns into "one medoid per cluster". A run that stopped on an iteration
    cap rather than on exhaustion has not established that, so it is worth certifying
    separately from the loop that produced it.
    """
    D = np.asarray(D, dtype=float)
    medoids = np.asarray(medoids, dtype=np.int64)
    if medoids.size >= D.shape[0]:
        return True                                       # no non-medoid to swap in
    return _best_swap(D, medoids)[2] >= -tol


@dataclass(frozen=True)
class PAMResult:
    """Output of `pam`, with what is needed to certify it against the theory."""

    labels: np.ndarray          # (N,) nearest-medoid partition
    medoids: np.ndarray         # (K,) indices
    objective: float            # Phi, normalised by N
    one_swap_certified: bool    # no improving one-swap exists at the returned medoids
    n_swaps: int                # swaps accepted by the winning run
    hit_cap: bool               # a run stopped on `max_swaps` rather than on exhaustion


def pam(D, K, rng=None, n_restarts=1, max_swaps=10_000, tol=0.0):
    """Partitioning Around Medoids on a precomputed dissimilarity matrix.

    This is the algorithm Theorem 3.8 analyses, and not an approximation of it: BUILD, then
    *strictly improving one-medoid swaps* until no improving swap exists. The alternating
    "assign, then re-centre each cluster" heuristic often called K-medoids is a different
    algorithm; its fixed points need not be one-swap stationary, so the theorem does not
    apply to them.

    Each sweep takes the *steepest* improving swap. Taking the first improving one instead
    would also terminate at a one-swap stationary set -- the theory covers either -- but
    steepest descent makes the run deterministic given its starting point.

    `n_restarts` beyond the first starts from random medoid sets (hence `rng`) and the
    lowest Phi wins. This departs from BUILD-then-SWAP, but every candidate is run to
    exhaustion, so the winner is one-swap stationary whatever it started from and the
    theorem still applies. The default is a single deterministic run: the paper's PAM.

    Phi cannot increase and there are finitely many medoid sets, so the loop terminates;
    `max_swaps` only guards against a pathological tie-driven cycle, and `hit_cap` reports
    whether it ever bound.
    """
    D = np.asarray(D, dtype=float)
    N = D.shape[0]
    if K < 1 or K > N:
        raise ValueError(f"K must lie between 1 and N = {N}, got {K}")
    if n_restarts < 1:
        raise ValueError("n_restarts must be >= 1")
    if n_restarts > 1 and rng is None:
        raise ValueError("rng is required when n_restarts > 1")

    best, hit_cap = None, False
    for r in range(n_restarts):
        medoids = _pam_build(D, K) if r == 0 else np.sort(rng.choice(N, size=K, replace=False))
        swaps = 0
        while swaps < max_swaps:
            j, h, delta = _best_swap(D, medoids)
            if delta >= -tol:
                break
            medoids = medoids.copy()
            medoids[j] = h
            swaps += 1
        else:
            hit_cap = True
        objective = pam_objective(D, medoids)
        if best is None or objective < best[0]:
            best = (objective, medoids, swaps)

    objective, medoids, swaps = best
    return PAMResult(
        labels=np.argmin(D[:, medoids], axis=1).astype(np.int64),
        medoids=medoids,
        objective=objective,
        one_swap_certified=pam_certify_one_swap(D, medoids, tol=tol),
        n_swaps=swaps,
        hit_cap=hit_cap,
    )


# ---------------------------------------------------------------------------
# Agreement between partitions
# ---------------------------------------------------------------------------


def exact_recovery(labels, truth):
    """True iff the two labellings induce the *same partition*, labels aside.

    The primary outcome of the paper: Theorems 3.5, 3.8 and 3.9 are statements about
    P(partition = P*), not about a similarity score. Label values are meaningless -- every
    estimator here numbers its blocks by order of appearance -- so the comparison is between
    partitions, and no permutation has to be searched for: two labellings agree iff the
    number of distinct (a_i, b_i) pairs equals the number of blocks on each side, which says
    exactly that the map from one labelling to the other is a bijection.
    """
    a = np.asarray(labels).reshape(-1)
    b = np.asarray(truth).reshape(-1)
    if a.size != b.size:
        raise ValueError(f"labellings have different lengths: {a.size} and {b.size}")
    pairs = len(set(zip(a.tolist(), b.tolist())))
    return pairs == len(set(a.tolist())) == len(set(b.tolist()))


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
