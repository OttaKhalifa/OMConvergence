"""Shared utilities for the numerical illustrations of

    "Consistency of Optimal Matching-based Clustering for Mixtures of Markov Chains".

Notation follows the paper: the alphabet is Sigma = {0, ..., d-1}, the substitution cost
matrix is S = (c_sub(a, b))_{a,b}, and the gap cost is delta = (delta(a))_a, so that the
extended cost dbar of equation (2.1) is recovered from the pair (S, delta).

Contents
--------
Cost schemes and Assumption 1 : ``cost_scheme``, ``check_assumption_metric``
Markov chains                 : ``sample_markov_model``, ``sample_chain_order1``,
                                ``simulate_markov_sequence``, ``stationary_distribution_markov``,
                                ``stationary_symbol_distribution``, ``spectral_gap``
OM dissimilarity              : ``om_distance``, ``gamma_hat_paths``
Bounds of Proposition 2.6     : ``wasserstein_lower_bound``, ``product_upper_bound``
Mixtures and clustering       : ``sample_mixture``, ``om_matrices``, ``om_matrix``,
                                ``single_linkage_tree``, ``cut_at_k``, ``cut_at_threshold``,
                                ``largest_gap_k``, ``adjusted_rand_index``,
                                ``gamma_block_means``, ``separation_levels``
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

try:
    from numba import njit, prange
except ImportError:  # pragma: no cover - pure Python fallback, orders of magnitude slower
    print("Warning: numba not available, falling back to pure Python.")

    def njit(*args, **kwargs):
        def deco(fn):
            return fn
        return deco

    def prange(x):
        return range(x)


# ---------------------------------------------------------------------------
# Cost schemes and Assumption 1 (metric cost scheme)
# ---------------------------------------------------------------------------


def compute_constant_subst_matrix(n_states, cost=1.0):
    """c_sub(a, b) = cost for a != b."""
    if n_states is None or n_states < 2:
        raise ValueError("n_states must be >= 2")
    S = np.full((n_states, n_states), float(cost))
    np.fill_diagonal(S, 0.0)
    return S


def compute_random_subst_matrix(n_states, rng, low=0.5, high=2.0):
    """Symmetric costs drawn uniformly on [low, high] off the diagonal."""
    if n_states is None or n_states < 2:
        raise ValueError("n_states must be >= 2")
    if low <= 0 or high <= low:
        raise ValueError("low and high must satisfy 0 < low < high")
    A = rng.uniform(low, high, size=(n_states, n_states))
    S = 0.5 * (A + A.T)
    np.fill_diagonal(S, 0.0)
    return S


def compute_trate_subst_matrix(sequences, n_states=None, pad_value=None, smoothing=1e-8):
    """TRATE costs c_sub(a, b) = 2 - Phat(a, b) - Phat(b, a), estimated from `sequences`.

    Being data-driven, these costs are outside the theoretical framework unless they are
    estimated on a sample independent of the sequences on which the OM dissimilarity is
    then computed.
    """
    if n_states is None:
        n_states = 1 + max(int(np.max(s)) for s in sequences if len(s) > 0)
    counts = np.zeros((n_states, n_states), dtype=float)
    for seq in sequences:
        arr = np.asarray(seq, dtype=np.int64).reshape(-1)
        if pad_value is not None:
            arr = arr[arr != pad_value]
        if arr.size >= 2:
            idx = arr[:-1] * n_states + arr[1:]
            counts += np.bincount(idx, minlength=n_states ** 2).reshape(n_states, n_states)
    row_sums = counts.sum(axis=1, keepdims=True)
    P = (counts + smoothing) / (row_sums + smoothing * n_states)
    S = 2.0 - P - P.T
    np.fill_diagonal(S, 0.0)
    return S


def choose_indel_cost(subst_cost, strategy="median_nonzero"):
    """Scalar gap cost derived from the substitution costs."""
    d = subst_cost.shape[0]
    offdiag = subst_cost[~np.eye(d, dtype=bool)]
    offdiag = offdiag[offdiag > 0]
    if offdiag.size == 0:
        return 1.0
    if strategy == "mean_nonzero":
        return float(offdiag.mean())
    return float(np.median(offdiag))


def cost_scheme(name, d, rng=None, pilot_sequences=None,
                sub=2.0, indel=1.0, low=1.2, high=2.0):
    """Return the pair (S, delta) for one of the three cost schemes.

    - "constant" : c_sub = `sub` off the diagonal (the usual indel=1, sub=2 default);
    - "random"   : symmetric c_sub drawn uniformly on [low, high];
    - "trate"    : c_sub = 2 - Phat - Phat^T estimated on `pilot_sequences`.

    In all three cases the gap cost is constant, equal to `indel`.
    """
    if name == "constant":
        S = compute_constant_subst_matrix(d, cost=sub)
    elif name == "random":
        S = compute_random_subst_matrix(d, rng, low=low, high=high)
    elif name == "trate":
        S = compute_trate_subst_matrix(pilot_sequences, n_states=d)
    else:
        raise ValueError(f"unknown cost scheme: {name}")
    return S, np.full(d, float(indel))


def check_assumption_metric(S, delta, tol=1e-12):
    """Check conditions (i)-(vi) of Assumption 1 for (c_sub, delta) = (S, delta)."""
    d = S.shape[0]
    offdiag = ~np.eye(d, dtype=bool)
    T = (S[:, :, None] + S[None, :, :]).min(axis=1)   # T[a, c] = min_b S[a, b] + S[b, c]
    return {
        "(i) delta > 0": bool(np.all(delta > 0)),
        "(ii) symmetry": bool(np.allclose(S, S.T, atol=tol)),
        "(iii) triangle inequality": bool(np.all(S <= T + tol)),
        "(iv) c_sub <= delta + delta": bool(np.all(S <= delta[:, None] + delta[None, :] + tol)),
        "(v) delta <= c_sub + delta": bool(np.all(delta[:, None] <= S + delta[None, :] + tol)),
        "(vi) c_sub > 0 off-diagonal": bool(np.all(S[offdiag] > 0)),
        "zero diagonal": bool(np.allclose(np.diag(S), 0.0, atol=tol)),
        "M = max c_sub": float(S.max()),
    }


# ---------------------------------------------------------------------------
# Markov chains
# ---------------------------------------------------------------------------


def _expand_dirichlet_alpha(alpha, n_rows, n_cols, name="alpha"):
    if np.isscalar(alpha):
        if alpha <= 0:
            raise ValueError(f"{name} must be > 0")
        return np.full((n_rows, n_cols), float(alpha))
    arr = np.asarray(alpha, dtype=float)
    if arr.ndim == 1:
        if arr.shape[0] != n_cols:
            raise ValueError(f"{name} must have length {n_cols}")
        arr = np.tile(arr, (n_rows, 1))
    elif arr.ndim != 2 or arr.shape != (n_rows, n_cols):
        raise ValueError(f"{name} must be a scalar, a 1D or a ({n_rows}, {n_cols}) array")
    if np.any(arr <= 0):
        raise ValueError(f"all values of {name} must be > 0")
    return arr


def _context_to_index(context, n_states):
    idx = 0
    for s in context:
        idx = idx * n_states + int(s)
    return idx


def _index_to_context(idx, n_states, order):
    context = [0] * order
    for pos in range(order - 1, -1, -1):
        context[pos] = int(idx % n_states)
        idx //= n_states
    return context


def sample_markov_model(n_states, order_k, alpha, rng, alpha_init=None):
    """Order-`order_k` Markov model with rows drawn from a Dirichlet(`alpha`) prior.

    The returned dictionary holds the (n_states ** order_k, n_states) transition array; for
    order_k = 1 this array is the transition matrix P itself.
    """
    if order_k < 1:
        raise ValueError("order_k must be >= 1")
    if n_states < 2:
        raise ValueError("n_states must be >= 2")
    n_contexts = n_states ** order_k
    trans_alpha = _expand_dirichlet_alpha(alpha, n_contexts, n_states)
    transitions = np.empty((n_contexts, n_states), dtype=float)
    for i in range(n_contexts):
        transitions[i] = rng.dirichlet(trans_alpha[i])
    if alpha_init is None:
        alpha_init = alpha
    if order_k == 1:
        init_probs = rng.dirichlet(_expand_dirichlet_alpha(alpha_init, 1, n_states)[0])
    else:
        init_probs = rng.dirichlet(_expand_dirichlet_alpha(alpha_init, 1, n_contexts)[0])
    return {"n_states": n_states, "order": order_k,
            "transitions": transitions, "init_probs": init_probs}


def _sample_initial_context(model, rng):
    order, n_states, p = model["order"], model["n_states"], model["init_probs"]
    if order == 1:
        return [int(rng.choice(n_states, p=p))]
    return _index_to_context(int(rng.choice(p.size, p=p)), n_states, order)


def _sample_next_state(model, context, rng):
    idx = _context_to_index(context, model["n_states"])
    return int(rng.choice(model["n_states"], p=model["transitions"][idx]))


def simulate_markov_sequence(model, n, rng):
    """Length-n trajectory of a model of arbitrary order, started from `init_probs`."""
    order = model["order"]
    seq = list(_sample_initial_context(model, rng))
    while len(seq) < n:
        seq.append(_sample_next_state(model, seq[-order:], rng))
    return np.asarray(seq[:n], dtype=np.int64)


@njit(cache=True)
def _walk(cum, u, x0):
    """Inverse-cdf walk: cum is the row-wise cumulative kernel, u the uniforms."""
    n = u.shape[0]
    x = np.empty(n, dtype=np.int64)
    x[0] = x0
    for t in range(1, n):
        x[t] = np.searchsorted(cum[x[t - 1]], u[t])
    return x


def sample_chain_order1(P, n, rng, init=None):
    """Length-n trajectory of the first-order chain with kernel P.

    `init` is either None (start from the stationary law), an integer (Dirac initial law), or
    a probability vector.
    """
    d = P.shape[0]
    if init is None:
        x0 = int(rng.choice(d, p=stationary_distribution_markov(P)))
    elif np.isscalar(init):
        x0 = int(init)
    else:
        x0 = int(rng.choice(d, p=np.asarray(init, dtype=float)))
    return _walk(np.cumsum(P, axis=1), rng.random(n), x0)


def stationary_distribution_markov(P, init=None, tol=1e-12, max_iter=200_000):
    """Stationary law of an irreducible kernel, by power iteration."""
    n = P.shape[0]
    v = np.full(n, 1.0 / n) if init is None else np.asarray(init, dtype=float).reshape(-1)
    if v.size != n or v.sum() <= 0:
        raise ValueError("init must be a non-negative vector of length P.shape[0]")
    v = v / v.sum()
    for _ in range(max_iter):
        v_next = v @ P
        if np.abs(v_next - v).sum() < tol:
            v = v_next
            break
        v = v_next
    return v / v.sum()


def build_context_transition_matrix(model):
    """Transition matrix of the context chain of an order > 1 model."""
    n_states, order_k = model["n_states"], model["order"]
    if order_k <= 1:
        raise ValueError("order_k must be > 1 for a context transition matrix")
    n_contexts = n_states ** order_k
    Q = np.zeros((n_contexts, n_contexts), dtype=float)
    for i in range(n_contexts):
        ctx = _index_to_context(i, n_states, order_k)
        for a, p in enumerate(model["transitions"][i]):
            if p > 0:
                Q[i, _context_to_index(ctx[1:] + [a], n_states)] += p
    return Q


def stationary_symbol_distribution(model):
    """Stationary law of the observed symbol, for a model of arbitrary order."""
    if model["order"] == 1:
        return stationary_distribution_markov(model["transitions"])
    Q = build_context_transition_matrix(model)
    pi_ctx = stationary_distribution_markov(Q, init=model.get("init_probs"))
    pi_sym = np.zeros(model["n_states"], dtype=float)
    for idx in range(Q.shape[0]):
        pi_sym[_index_to_context(idx, model["n_states"], model["order"])[-1]] += pi_ctx[idx]
    return pi_sym / pi_sym.sum()


def spectral_gap(P):
    """1 - |lambda_2|, a proxy for the mixing speed of P."""
    ev = np.sort(np.abs(np.linalg.eigvals(P)))
    return float(1.0 - ev[-2])


# ---------------------------------------------------------------------------
# OM dissimilarity
# ---------------------------------------------------------------------------


@njit(cache=True, fastmath=True)
def om_distance(x, y, S, delta):
    """d_OM(x, y) for the cost scheme (S, delta), in O(nm) time and O(m) memory.

    The two-row dynamic program matters here: the full (n+1) x (m+1) table would take 800 MB
    per pair at n = 1e4, which rules out running several pairs in parallel.
    """
    n = x.shape[0]
    m = y.shape[0]
    prev = np.empty(m + 1, dtype=np.float64)
    curr = np.empty(m + 1, dtype=np.float64)
    prev[0] = 0.0
    for j in range(1, m + 1):
        prev[j] = prev[j - 1] + delta[y[j - 1]]
    for i in range(1, n + 1):
        xi = x[i - 1]
        curr[0] = prev[0] + delta[xi]
        for j in range(1, m + 1):
            yj = y[j - 1]
            d_del = prev[j] + delta[xi]
            d_ins = curr[j - 1] + delta[yj]
            d_sub = prev[j - 1] + S[xi, yj]
            best = d_del if d_del < d_ins else d_ins
            if d_sub < best:
                best = d_sub
            curr[j] = best
        for j in range(m + 1):
            prev[j] = curr[j]
    return prev[m]


@njit(cache=True, parallel=True)
def gamma_hat_pairs(X, Y, S, delta):
    """hat-gamma_n for each aligned pair of rows of X and Y, in parallel over the pairs.

    X, Y are (R, n) integer arrays; the result has shape (R,). Used when only the final
    horizon is needed, e.g. to fill a whole dissimilarity matrix.
    """
    R = X.shape[0]
    n = X.shape[1]
    out = np.empty(R, dtype=np.float64)
    for r in prange(R):
        out[r] = om_distance(X[r], Y[r], S, delta) / n
    return out


@njit(cache=True, parallel=True)
def gamma_hat_paths(X, Y, grid, S, delta):
    """Sample paths n -> d_OM(X[r, :n], Y[r, :n]) / n, in parallel over the replicates.

    X, Y are (R, n_max) integer arrays and `grid` an increasing array of horizons; the result
    has shape (R, len(grid)). Since the horizons index nested prefixes of the same two
    trajectories, each row is a genuine sample path of hat-gamma_n.
    """
    R = X.shape[0]
    G = grid.shape[0]
    out = np.empty((R, G), dtype=np.float64)
    for r in prange(R):
        for g in range(G):
            n = grid[g]
            out[r, g] = om_distance(X[r, :n], Y[r, :n], S, delta) / n
    return out


# ---------------------------------------------------------------------------
# Mixtures of Markov chains
# ---------------------------------------------------------------------------


def sample_mixture(K, N, n, d, alpha, rng, weights=None, kernels=None):
    """N sequences drawn from a K-component mixture of order-1 chains, with their labels.

    Latent labels are i.i.d. with law `weights` (uniform if None); conditionally on Z_i = k,
    sequence i is a length-n realisation of kernel k. As in the other experiments, every
    sequence starts from its own initial law, the Dirac law at a state drawn uniformly on
    Sigma, so no two sequences share an initial condition.

    Pass `kernels` (a (K, d, d) array) to keep the mixture fixed across replicates and let
    only the labels and the trajectories be redrawn; otherwise K kernels are drawn row-wise
    from a Dirichlet(`alpha`) prior.
    """
    if kernels is None:
        kernels = np.stack([sample_markov_model(d, 1, alpha, rng)["transitions"]
                            for _ in range(K)])
    else:
        kernels = np.asarray(kernels, dtype=float)
        if kernels.shape != (K, d, d):
            raise ValueError(f"kernels must have shape ({K}, {d}, {d})")
    w = np.full(K, 1.0 / K) if weights is None else np.asarray(weights, dtype=float)
    if w.size != K or np.any(w <= 0):
        raise ValueError("weights must be K positive numbers")
    w = w / w.sum()
    labels = rng.choice(K, size=N, p=w)
    X = np.empty((N, n), dtype=np.int64)
    for i, z in enumerate(labels):
        X[i] = sample_chain_order1(kernels[z], n, rng, init=int(rng.integers(d)))
    return {"X": X, "labels": labels, "kernels": kernels, "weights": w,
            "counts": np.bincount(labels, minlength=K)}


# ---------------------------------------------------------------------------
# Dissimilarity matrices
# ---------------------------------------------------------------------------


@njit(cache=True, parallel=True)
def _om_matrix_kernel(X, grid, ii, jj, S, delta, out):
    """Fill out[g, i, j] = d_OM(X[i, :grid[g]], X[j, :grid[g]]) / grid[g], in parallel."""
    for p in prange(ii.shape[0]):
        i = ii[p]
        j = jj[p]
        for g in range(grid.shape[0]):
            n = grid[g]
            v = om_distance(X[i, :n], X[j, :n], S, delta) / n
            out[g, i, j] = v
            out[g, j, i] = v


def om_matrices(X, grid, S, delta):
    """Normalised OM dissimilarity matrices of the rows of X, on nested prefixes.

    X is an (N, n_max) integer array and `grid` an increasing array of horizons; the result
    has shape (len(grid), N, N), each slice being symmetric with a zero diagonal. Since the
    horizons index nested prefixes of the same trajectories, the slices are the successive
    states of one dissimilarity matrix rather than independent draws.

    The parallel loop runs over the N(N-1)/2 pairs, all horizons of a pair being handled by
    the same thread: the two rows of X are then read once per pair rather than once per
    horizon.
    """
    X = np.ascontiguousarray(X, dtype=np.int64)
    grid = np.atleast_1d(np.asarray(grid, dtype=np.int64))
    if grid[-1] > X.shape[1]:
        raise ValueError("the largest horizon exceeds the length of the sequences")
    ii, jj = np.triu_indices(X.shape[0], k=1)
    out = np.zeros((grid.size, X.shape[0], X.shape[0]), dtype=np.float64)
    _om_matrix_kernel(X, grid, ii.astype(np.int64), jj.astype(np.int64), S, delta, out)
    return out


def om_matrix(X, S, delta):
    """The (N, N) normalised OM dissimilarity matrix at the full length of the sequences."""
    return om_matrices(X, [X.shape[1]], S, delta)[0]


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
    """Single-linkage partition into K blocks: the first N-K merges, i.e. Definition 4.2
    stopped at N-K steps. K >= N returns the singletons."""
    return _components(edges[:max(N - K, 0)], N)


def cut_at_threshold(heights, edges, N, t):
    """Single-linkage partition at level t: the connected components of {D <= t}, which is
    the graph G_t of Theorem 4.4. The number of blocks is whatever the data gives."""
    return _components(edges[heights <= t], N)


def largest_gap_k(heights):
    """The largest-gap estimator of K, equation (4.4): K = N - argmax_l (h_{l+1} - h_l),
    over 1 <= l <= N-2, taking the smallest index in case of tie."""
    N = heights.size + 1
    if N < 3:
        return N
    gaps = np.diff(heights)                  # gaps[l - 1] = h_{l+1} - h_l, l = 1, ..., N-2
    return int(N - (1 + int(np.argmax(gaps))))


# ---------------------------------------------------------------------------
# Agreement between partitions, and plug-in estimates of Gamma
# ---------------------------------------------------------------------------


def adjusted_rand_index(a, b):
    """Adjusted Rand Index between two labellings of the same N items.

    Equals 1 exactly when the two partitions coincide, and has expectation 0 under the
    permutation model, so ARI = 1 is the exact-recovery event of Theorem 4.4.
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


# ---------------------------------------------------------------------------
# Bounds of Proposition 2.6
# ---------------------------------------------------------------------------


def wasserstein_lower_bound(pi_P, pi_Q, S):
    """W_dbar(pi_P, pi_Q) on (Sigma_bot, dbar), by linear programming.

    Both marginals are supported on Sigma and dbar restricted to Sigma x Sigma is c_sub, so
    the transport problem can be written on Sigma alone.
    """
    d = pi_P.size
    A_eq = np.zeros((2 * d, d * d))
    for i in range(d):
        A_eq[i, i * d:(i + 1) * d] = 1.0
    for j in range(d):
        A_eq[d + j, j::d] = 1.0
    res = linprog(S.reshape(-1), A_eq=A_eq, b_eq=np.concatenate([pi_P, pi_Q]),
                  bounds=[(0.0, None)] * (d * d), method="highs")
    if not res.success:
        raise RuntimeError("Wasserstein LP failed: " + res.message)
    return float(res.fun)


def product_upper_bound(pi_P, pi_Q, S):
    """pi_P^T S pi_Q."""
    return float(pi_P @ S @ pi_Q)


# ---------------------------------------------------------------------------
# Shared figure style
# ---------------------------------------------------------------------------

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
