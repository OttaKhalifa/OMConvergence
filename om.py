"""The OM dissimilarity: cost schemes, distances, and the bounds of Proposition 2.8.

Notation follows the paper: the alphabet is Sigma = {0, ..., d-1}, the substitution cost
matrix is S = (c_sub(a, b))_{a,b}, and the gap cost is delta = (delta(a))_a, so that the
extended cost dbar of equation (1) is recovered from the pair (S, delta).

Two OM paths live here, and they are not interchangeable:

- the **paper's**, on univariate sequences, parametrised by an explicit (S, delta) --
  ``om_distance`` and everything built on it. This is what the three notebooks use;
- the **benchmark's**, on multichannel sequences, which fixes the whole configuration
  (per-channel TRATE costs, indel = sum_v 0.5 max(S_v), ``norm="maxlength"``) --
  ``om_trate_distances``. This is the TraMineR default applied to several channels at
  once, and it takes no cost scheme because it derives its own from the data.

Both reproduce ``TraMineR/src/OMdistance.cpp`` bit for bit; ``validation/`` checks the
first one against R, on the three cost schemes.

Contents
--------
Cost schemes and Assumption 1 : ``cost_scheme``, ``check_assumption_metric``,
                                ``compute_constant_subst_matrix``,
                                ``compute_random_subst_matrix``,
                                ``compute_trate_subst_matrix``
OM dissimilarity (univariate) : ``om_distance``, ``gamma_hat_pairs``, ``gamma_hat_paths``,
                                ``om_matrices``
OM dissimilarity (multichannel) : ``om_trate_distances``, ``OMResult``
Bounds of Proposition 2.8     : ``wasserstein_lower_bound``, ``product_upper_bound``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

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


def compute_trate_subst_matrix(sequences, n_states=None):
    """TRATE costs c_sub(a, b) = 2 - Phat(a, b) - Phat(b, a), estimated from `sequences`.

    Being data-driven, these costs are outside the theoretical framework unless they are
    estimated on a sample independent of the sequences on which the OM dissimilarity is
    then computed.

    Reproduces ``seqcost(method="TRATE", cval=2)`` bit for bit, which needs two things
    beyond the formula (``validation/`` checks both):

    - **no smoothing.** A state that is never the source of a transition keeps a row of
      zeros, as ``seqtrate`` leaves it. The former 1e-8 smoothing made that row uniform
      instead -- and on an empty row ``(0 + eps) / (0 + eps * d) = 1 / d`` whatever eps
      is, so a parameter advertised as a numerical guard was in fact choosing the row.
    - **mirror rather than recompute.** ``seqcost`` evaluates ``2 - P[a,b] - P[b,a]`` on
      the upper triangle and copies it down; recomputing the lower cell gives
      ``(2 - b) - a`` against its ``(2 - a) - b``, one ulp apart.
    """
    if n_states is None:
        n_states = 1 + max(int(np.max(s)) for s in sequences if len(s) > 0)
    counts = np.zeros((n_states, n_states), dtype=float)
    for seq in sequences:
        arr = np.asarray(seq, dtype=np.int64).reshape(-1)
        if arr.size >= 2:
            idx = arr[:-1] * n_states + arr[1:]
            counts += np.bincount(idx, minlength=n_states ** 2).reshape(n_states, n_states)
    row_sums = counts.sum(axis=1, keepdims=True)
    P = np.zeros_like(counts)
    np.divide(counts, row_sums, out=P,
              where=np.broadcast_to(row_sums > 0, counts.shape))
    S = 2.0 - P - P.T
    np.fill_diagonal(S, 0.0)
    iu = np.triu_indices(n_states, 1)
    S[iu[1], iu[0]] = S[iu]
    return S


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
# OM dissimilarity
# ---------------------------------------------------------------------------


@njit(cache=True)
def om_distance(x, y, S, delta):
    """d_OM(x, y) for the cost scheme (S, delta), in O(nm) time and O(m) memory.

    Follows ``TraMineR/src/OMdistance.cpp`` so that the result is bit-identical to
    ``seqdist(method="OM", sm=S, indel=delta, norm="none")``, and not merely close to it;
    ``validation/`` checks that on the three cost schemes. Three details of that file are
    what the last few ulp hang on, none of them visible in the recurrence itself:

    - the common prefix and the common suffix are skipped before the dynamic program, so
      its border restarts inside the differing zone;
    - the border is multiplied, ``i * gap``, where accumulating ``i`` additions rounds
      ``i`` times. Only defined for a constant gap cost, which is the only case TraMineR
      has; a gap cost that varies with the state falls back to the running sum;
    - two equal states skip the addition of a zero substitution cost, which matters only
      for a signed zero but costs nothing to reproduce.

    ``fastmath`` is off on purpose: it lets the compiler reassociate the additions, which
    is exactly what bit-identity forbids, and measurement put its benefit at about 4% of
    runtime. ``validation/README.md`` also says what that check does *not* pin down.

    The two-row dynamic program holds the same values as the full table, so it is neutral
    on the bits. It matters for memory: the full (n+1) x (m+1) table would take 800 MB per
    pair at n = 1e4, which rules out running several pairs in parallel.
    """
    n = x.shape[0]
    m = y.shape[0]

    # Matching two identical states costs nothing, and Assumption 1 (iv) makes it optimal
    # (c_sub(a, a) = 0 <= delta(a) + delta(a)), so the common prefix and suffix can be
    # dropped. TraMineR does it to go faster; here it is also what moves the rounding.
    p = 0
    while p < n and p < m and x[p] == y[p]:
        p += 1
    end_x = n
    end_y = m
    while end_x > p and end_y > p and x[end_x - 1] == y[end_y - 1]:
        end_x -= 1
        end_y -= 1
    rows = end_x - p
    cols = end_y - p

    # Assumption 1 lets delta vary with the state; every scheme of `cost_scheme` makes it
    # constant, and that is also the only shape TraMineR has, so the multiplied border is
    # available whenever the comparison with TraMineR is meaningful at all.
    gap = delta[0]
    const_gap = True
    for a in range(1, delta.shape[0]):
        if delta[a] != gap:
            const_gap = False
            break

    prev = np.empty(cols + 1, dtype=np.float64)
    curr = np.empty(cols + 1, dtype=np.float64)
    prev[0] = 0.0
    if const_gap:
        for j in range(1, cols + 1):
            prev[j] = j * gap
    else:
        for j in range(1, cols + 1):
            prev[j] = prev[j - 1] + delta[y[p + j - 1]]
    for i in range(1, rows + 1):
        xi = x[p + i - 1]
        if const_gap:
            curr[0] = i * gap
        else:
            curr[0] = prev[0] + delta[xi]
        for j in range(1, cols + 1):
            yj = y[p + j - 1]
            d_del = prev[j] + delta[xi]
            d_ins = curr[j - 1] + delta[yj]
            # min(a, c) + b == min(a + b, c + b) bit for bit, IEEE rounding being
            # monotone, so the three-way min below is TraMineR's "best indel, then
            # substitution" on a constant gap cost, and the general one otherwise.
            if xi == yj:
                d_sub = prev[j - 1]
            else:
                d_sub = prev[j - 1] + S[xi, yj]
            best = d_del if d_del < d_ins else d_ins
            if d_sub < best:
                best = d_sub
            curr[j] = best
        for j in range(cols + 1):
            prev[j] = curr[j]
    return prev[cols]


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



# -------------------------------------------------------------------------
# Multichannel TRATE dissimilarity (the benchmark's OM configuration)
# -------------------------------------------------------------------------

# The per-channel TRATE costs below duplicate the formula of `compute_trate_subst_matrix`
# above. The two are kept apart on purpose: each is validated bit for bit against a
# different R call -- `seqcost` on one channel there, on V channels here -- and the
# univariate one also has to accept the ragged sequence lists of the notebooks.

@dataclass(frozen=True)
class OMResult:
    """Outputs needed by Ward and by the TraMineR equivalence check."""

    distances: np.ndarray
    raw_distances: np.ndarray
    substitution_costs: tuple[np.ndarray, ...]
    indel: float
    lengths: np.ndarray


def _validate_inputs(
    sequences: Sequence[np.ndarray],
    n_categories: Sequence[int],
) -> tuple[list[np.ndarray], tuple[int, ...]]:
    """Return contiguous int64 ``(L_i, V)`` arrays and validate their alphabet."""
    if not sequences:
        raise ValueError("sequences must not be empty")

    categories = tuple(int(c) for c in n_categories)
    if not categories or any(c < 2 for c in categories):
        raise ValueError("n_categories must contain at least 2 categories per channel")
    n_channels = len(categories)

    validated = []
    for i, sequence in enumerate(sequences):
        source = np.asarray(sequence)
        if source.ndim != 2 or source.shape[1] != n_channels:
            raise ValueError(
                f"sequence {i} must have shape (L, {n_channels}), got {source.shape}")
        if source.shape[0] == 0:
            raise ValueError(f"sequence {i} is empty")
        if not np.issubdtype(source.dtype, np.integer):
            raise TypeError(f"sequence {i} must contain integer category ids")

        array = np.ascontiguousarray(source, dtype=np.int64)
        for channel, count in enumerate(categories):
            values = array[:, channel]
            if np.any(values < 0) or np.any(values >= count):
                raise ValueError(
                    f"sequence {i}, channel {channel}: category ids must be in [0, {count})")
        validated.append(array)
    return validated, categories


def _trate_substitution_costs(
    sequences: Sequence[np.ndarray],
    n_categories: Sequence[int],
) -> tuple[np.ndarray, ...]:
    """Reproduce ``seqcost(method="TRATE", cval=2)`` for every channel."""
    costs = []
    for channel, count in enumerate(n_categories):
        previous = [sequence[:-1, channel] for sequence in sequences if len(sequence) >= 2]
        following = [sequence[1:, channel] for sequence in sequences if len(sequence) >= 2]

        if previous:
            prev = np.concatenate(previous)
            nxt = np.concatenate(following)
            transitions = np.bincount(
                prev * count + nxt, minlength=count * count
            ).reshape(count, count).astype(np.float64)
        else:
            transitions = np.zeros((count, count), dtype=np.float64)

        row_sums = transitions.sum(axis=1, keepdims=True)
        rates = np.zeros_like(transitions)
        observed = row_sums > 0
        np.divide(
            transitions,
            row_sums,
            out=rates,
            where=np.broadcast_to(observed, transitions.shape),
        )

        # TraMineR computes each upper-triangle cell as (2 - P[i,j]) - P[j,i]
        # and copies it to the symmetric cell. Recomputing the lower triangle can
        # differ by one ulp because the two subtractions are then reversed.
        substitution = 2.0 - rates - rates.T
        np.fill_diagonal(substitution, 0.0)
        upper = np.triu_indices(count, 1)
        substitution[upper[1], upper[0]] = substitution[upper]
        costs.append(substitution)
    return tuple(costs)


def _trate_indel_cost(substitution_costs: Sequence[np.ndarray]) -> float:
    """Reproduce R's extended-precision ``sum(0.5 * max(S_v))``."""
    total = np.longdouble(0.0)
    for substitution in substitution_costs:
        total += np.longdouble(0.5) * np.longdouble(np.max(substitution))
    indel = float(total)
    if indel <= 0.0:
        raise ValueError("TRATE produced a non-positive indel cost")
    return indel


def _pack_numba_inputs(
    sequences: Sequence[np.ndarray],
    substitution_costs: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack variable-length sequences and channel-specific square matrices."""
    n_sequences = len(sequences)
    n_channels = sequences[0].shape[1]
    lengths = np.asarray([len(sequence) for sequence in sequences], dtype=np.int64)

    padded = np.full(
        (n_sequences, int(lengths.max()), n_channels), -1, dtype=np.int64
    )
    for i, sequence in enumerate(sequences):
        padded[i, : len(sequence)] = sequence

    max_categories = max(cost.shape[0] for cost in substitution_costs)
    packed_costs = np.zeros(
        (n_channels, max_categories, max_categories), dtype=np.float64
    )
    for channel, cost in enumerate(substitution_costs):
        count = cost.shape[0]
        packed_costs[channel, :count, :count] = cost
    return padded, lengths, packed_costs


@njit(cache=True)
def _om_pair(sequence_a, sequence_b, substitution_costs, indel):
    """Transcription of ``TraMineR/src/OMdistance.cpp::distance``.

    Common prefixes and suffixes are skipped, borders use multiplication, and
    equal multichannel states bypass the zero-cost addition. These details are
    required for bit identity, not merely numerical proximity.
    """
    length_a, n_channels = sequence_a.shape
    length_b = sequence_b.shape[0]

    prefix = 0
    while prefix < length_a and prefix < length_b:
        same = True
        for channel in range(n_channels):
            if sequence_a[prefix, channel] != sequence_b[prefix, channel]:
                same = False
                break
        if not same:
            break
        prefix += 1

    end_a, end_b = length_a, length_b
    while end_a > prefix and end_b > prefix:
        same = True
        for channel in range(n_channels):
            if sequence_a[end_a - 1, channel] != sequence_b[end_b - 1, channel]:
                same = False
                break
        if not same:
            break
        end_a -= 1
        end_b -= 1

    rows, columns = end_a - prefix, end_b - prefix
    matrix = np.empty((rows + 1, columns + 1), dtype=np.float64)
    matrix[0, 0] = 0.0
    for i in range(1, rows + 1):
        matrix[i, 0] = i * indel
    for j in range(1, columns + 1):
        matrix[0, j] = j * indel

    for i in range(1, rows + 1):
        index_a = prefix + i - 1
        for j in range(1, columns + 1):
            index_b = prefix + j - 1

            best_indel = matrix[i, j - 1]
            if matrix[i - 1, j] < best_indel:
                best_indel = matrix[i - 1, j]
            best_indel += indel

            same = True
            for channel in range(n_channels):
                if sequence_a[index_a, channel] != sequence_b[index_b, channel]:
                    same = False
                    break
            if same:
                substitution = matrix[i - 1, j - 1]
            else:
                cost = 0.0
                for channel in range(n_channels):
                    cost += substitution_costs[
                        channel,
                        sequence_a[index_a, channel],
                        sequence_b[index_b, channel],
                    ]
                substitution = matrix[i - 1, j - 1] + cost
            matrix[i, j] = substitution if substitution < best_indel else best_indel
    return matrix[rows, columns]


@njit(cache=True, parallel=True)
def _pairwise_om_distances(padded, lengths, substitution_costs, indel):
    """Compute the raw symmetric matrix, balancing work over sequence pairs."""
    n_sequences = padded.shape[0]
    distances = np.zeros((n_sequences, n_sequences), dtype=np.float64)
    for pair in prange(n_sequences * (n_sequences - 1) // 2):
        row = int(
            (2 * n_sequences - 1 - np.sqrt(
                np.float64((2 * n_sequences - 1) ** 2 - 8 * pair)
            )) / 2.0
        )
        if row * (2 * n_sequences - row - 1) // 2 > pair:
            row -= 1
        column = row + 1 + pair - row * (2 * n_sequences - row - 1) // 2

        distance = _om_pair(
            padded[row, : lengths[row]],
            padded[column, : lengths[column]],
            substitution_costs,
            indel,
        )
        distances[row, column] = distance
        distances[column, row] = distance
    return distances


def _normalize_maxlength(
    distances: np.ndarray,
    lengths: np.ndarray,
    indel: float,
) -> np.ndarray:
    """Reproduce ``seqdist(norm="maxlength")`` with one rounded division."""
    lengths_float = np.asarray(lengths, dtype=np.float64)
    denominator = indel * np.maximum(lengths_float[:, None], lengths_float[None, :])
    return distances / denominator


def om_trate_distances(
    sequences: Sequence[np.ndarray],
    n_categories: Sequence[int],
) -> OMResult:
    """Compute the benchmark's unweighted TRATE + OM ``maxlength`` distances."""
    validated, categories = _validate_inputs(sequences, n_categories)
    substitution_costs = _trate_substitution_costs(validated, categories)
    indel = _trate_indel_cost(substitution_costs)
    padded, lengths, packed_costs = _pack_numba_inputs(validated, substitution_costs)
    raw_distances = _pairwise_om_distances(padded, lengths, packed_costs, indel)
    distances = _normalize_maxlength(raw_distances, lengths, indel)
    return OMResult(
        distances=distances,
        raw_distances=raw_distances,
        substitution_costs=substitution_costs,
        indel=indel,
        lengths=lengths,
    )

# ---------------------------------------------------------------------------
# Bounds of Proposition 2.8
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
