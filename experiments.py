"""Two-level Monte Carlo engine for the numerical experiments.

The chain the experiments implement is

    generative difficulty (alpha, K)  ->  finite-horizon geometry eta_n  ->  recovery,

and the whole point of this module is to keep its three links honestly separated.

**Two levels of randomness.** A parameter cell draws `R_mix` mixtures; *conditionally on
each fixed mixture* it estimates the finite-horizon matrix, then simulates `R_data`
clustering datasets. The kernels never change in between, so every algorithm, every cost
scheme and every horizon sees the same mixture and the comparison between them is paired.
Redrawing kernels per algorithm -- as the superseded notebooks did -- puts the variance of
the prior into what reads as a difference between methods.

**Gamma^(n) is not read off the clustering data.** `Gamma^(n)_kl` is a finite-horizon mean,

    Gamma^(n)_kl = E[ d_OM(X^(k)_{1:n}, X^(l)_{1:n}) / n ],

estimated from `R_gamma` *independent* trajectory pairs. Averaging the same dissimilarity
matrix the clustering ran on would correlate the estimated geometry with the outcome it is
supposed to explain, and no confidence interval could be attached to it.

**The sign of eta_n is a decision, so it carries uncertainty.** eta_n is a min minus a max
over K(K+1)/2 estimated entries, so a single noisy point estimate crossing zero says very
little. Entries get *simultaneous* intervals, and the mixture is classified separated,
nonseparated or uncertain from the conservative bounds those induce on eta_n. This is a
statement about the finite horizon n. It is not a proof that the asymptotic condition
eta > 0 holds, and nothing here should be reported as one.

**Every horizon comes from nested prefixes of the same trajectories.** `om_matrices` and
`gamma_hat_paths` evaluate the whole grid on increasing prefixes, so n -> eta_hat_n and
n -> recovery are genuine sample paths of one experiment rather than independent draws, and
the whole grid costs about 1.33 times its largest horizon alone.

Contents
--------
Reproducibility : ``stream``, ``seed_key``
Uncertainty     : ``wilson_interval``, ``simultaneous_intervals``
Mixtures        : ``MarkovMixture``, ``draw_markov_mixture``, ``save_mixture``,
                  ``HMMMixture``, ``draw_hmm_mixture``
Dissimilarity   : ``UnivariateOM``, ``MultichannelOM``
Geometry        : ``GammaEstimate``, ``estimate_gamma_paths``
Clustering      : ``run_dataset``, ``labels_at_k``, ``ALGORITHMS``
Storage         : ``ResultsWriter``
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import norm, t

from clustering import (adjusted_rand_index, exact_recovery, hac_labels, pam,
                        profile_graph_k, profile_heights, profile_distances,
                        safeguard_threshold)
from generators import MixtureOfHMMGenerator, sample_markov_model, sample_mixture
from om import (check_assumption_metric, cost_scheme, gamma_hat_paths,
                multichannel_cost_scheme, om_matrices, om_multichannel_matrices,
                om_multichannel_paths)

ALGORITHMS = ("single", "complete", "average", "pam")

SEPARATED = "separated"
NONSEPARATED = "nonseparated"
UNCERTAIN = "uncertain"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def _key_int(part):
    """Map one key element to the integer a SeedSequence spawn key needs."""
    if isinstance(part, (int, np.integer)):
        return int(part) % (2 ** 32)
    digest = hashlib.blake2b(repr(part).encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big")


def seed_key(*parts):
    """The human-readable form of a stream key, stored next to every result row."""
    return "/".join(str(p) for p in parts)


def stream(base_seed, *parts):
    """An independent generator addressed by a key rather than by a counter.

    Two runs that ask for the same key get the same stream, whatever else the program did
    in between and in whatever order the cells were executed. That is what makes a single
    row of a results table reproducible on its own: `stream(SEED, "grid", alpha, K, 3)`
    rebuilds the exact draw that produced it, without replaying the whole sweep.
    """
    spawn = tuple(_key_int(p) for p in parts)
    return np.random.default_rng(np.random.SeedSequence(entropy=base_seed, spawn_key=spawn))


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------


def wilson_interval(successes, trials, level=0.95):
    """Wilson score interval for a binomial proportion.

    Used for every Monte Carlo proportion reported, including the heatmap cells. The
    textbook normal interval is not an option here: the interesting cells are exactly those
    where the proportion sits near 0 or 1, where that interval has poor coverage and can
    leave [0, 1]. Wilson stays inside and does not collapse to a point at p_hat = 0 or 1.
    """
    if trials <= 0:
        return (float("nan"), float("nan"))
    z = norm.ppf(1.0 - (1.0 - level) / 2.0)
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denominator
    half = z * np.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials ** 2)) / denominator
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def simultaneous_intervals(mean, se, n_obs, n_entries, level=0.95):
    """Bonferroni-simultaneous Student intervals for `n_entries` estimated entries.

    eta_n is a *minimum over off-diagonal entries minus a maximum over diagonal ones*, so
    its sign depends on several entries at once and per-entry intervals at level 0.95 would
    not give a level-0.95 statement about it. Splitting the error budget over the
    K(K+1)/2 distinct entries does, conservatively, and costs a factor of about 3 in width
    at K = 10 -- affordable, since `R_gamma` is cheap.

    Student rather than normal because `R_gamma` is in the tens, not the thousands.
    """
    if n_obs < 2:
        raise ValueError("need at least 2 observations for an interval")
    quantile = t.ppf(1.0 - (1.0 - level) / (2.0 * n_entries), n_obs - 1)
    return mean - quantile * se, mean + quantile * se


# ---------------------------------------------------------------------------
# Mixtures: drawn once, then held fixed
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarkovMixture:
    """K first-order kernels on an alphabet of size d, with their mixture weights.

    The unit that is held fixed: one of these is drawn per `mixture_id`, and everything
    downstream -- every horizon, every cost scheme, every algorithm, every dataset -- is
    conditional on it.
    """

    mixture_id: int
    alpha: float
    K: int
    d: int
    kernels: np.ndarray         # (K, d, d)
    weights: np.ndarray         # (K,)
    key: str

    def sample_component(self, k, n_sequences, seq_len, rng):
        """`n_sequences` independent trajectories of component k, as an (R, n) array.

        Each starts from its own initial law -- the Dirac mass at a uniformly drawn state --
        so no two share an initial condition. Proposition 2.2 makes the limit independent of
        that choice; drawing it afresh is what lets the experiment show it rather than
        assume it.
        """
        from generators import sample_chain_order1
        P = self.kernels[k]
        return np.stack([sample_chain_order1(P, seq_len, rng, init=int(rng.integers(self.d)))
                         for _ in range(n_sequences)])

    def sample_dataset(self, N, seq_len, rng):
        """One labelled clustering dataset from the fixed kernels: ((N, n) array, labels)."""
        data = sample_mixture(self.K, N, seq_len, self.d, self.alpha, rng,
                              weights=self.weights, kernels=self.kernels)
        return data["X"], data["labels"]


def draw_markov_mixture(K, d, alpha, mixture_id, base_seed, weights=None):
    """Draw one mixture: K kernels with rows i.i.d. Dirichlet(alpha), on its own stream."""
    rng = stream(base_seed, "mixture", alpha, K, d, mixture_id)
    kernels = np.stack([sample_markov_model(d, 1, alpha, rng)["transitions"] for _ in range(K)])
    w = np.full(K, 1.0 / K) if weights is None else np.asarray(weights, dtype=float) / np.sum(weights)
    return MarkovMixture(mixture_id=mixture_id, alpha=float(alpha), K=int(K), d=int(d),
                         kernels=kernels, weights=w,
                         key=seed_key("mixture", alpha, K, d, mixture_id))


def save_mixture(mixture, directory):
    """Store the kernels themselves, not only the seed that produced them.

    A seed reproduces a draw only as long as the drawing code is untouched; the kernels are
    what the results are actually conditional on.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"mixture_{mixture.mixture_id:04d}_alpha{mixture.alpha}_K{mixture.K}.npz"
    payload = {"weights": mixture.weights, "alpha": mixture.alpha, "K": mixture.K,
               "d": mixture.d, "key": mixture.key}
    if hasattr(mixture, "kernels"):
        payload["kernels"] = mixture.kernels
    else:                                    # an HMM: store what defines each component
        for k, component in enumerate(mixture.generator.components):
            payload[f"pi_{k}"] = component.pi
            payload[f"A_{k}"] = component.A
            for c, B in enumerate(component.B):
                payload[f"B_{k}_{c}"] = B
    np.savez_compressed(path, **payload)
    return path


# ---------------------------------------------------------------------------
# The dissimilarity the experiment runs on
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UnivariateOM:
    """One cost scheme (S, delta), and the OM quantities computed with it."""

    name: str
    S: np.ndarray
    delta: np.ndarray

    @property
    def M(self):
        """max c_sub, the constant of the profile threshold a_{N,n}."""
        return float(np.max(self.S))

    def assumption_1(self):
        """Conditions (i)-(vi) of Assumption 1, to be recorded rather than assumed.

        The constant and random schemes satisfy them by construction. TRATE need not: it is
        estimated from data, and nothing makes `2 - Phat - Phat^T` obey the triangle
        inequality. Cells where it fails are outside the metric framework and must be
        reported as such, not as a numerical confirmation of the theorems.
        """
        return check_assumption_metric(self.S, self.delta)

    def gamma_paths(self, X, Y, n_grid):
        """(R, G) normalised dissimilarities between aligned rows, on nested prefixes."""
        X = np.ascontiguousarray(X, dtype=np.int64)
        Y = np.ascontiguousarray(Y, dtype=np.int64)
        return gamma_hat_paths(X, Y, np.asarray(n_grid, dtype=np.int64), self.S, self.delta)

    def matrices(self, X, n_grid):
        """(G, N, N) dissimilarity matrices of one dataset, on nested prefixes."""
        return om_matrices(X, np.asarray(n_grid, dtype=np.int64), self.S, self.delta)


def univariate_om(name, d, rng=None, pilot_sequences=None, **kwargs):
    """Build a `UnivariateOM` from one of the three schemes of `om.cost_scheme`."""
    S, delta = cost_scheme(name, d, rng=rng, pilot_sequences=pilot_sequences, **kwargs)
    return UnivariateOM(name=name, S=S, delta=delta)


@dataclass(frozen=True)
class HMMMixture:
    """K homogeneous multichannel HMMs, drawn once and held fixed.

    The same contract as `MarkovMixture` -- `sample_component` for an independent estimate of
    Gamma^(n), `sample_dataset` for a labelled clustering dataset -- so the drivers do not
    know which mechanism they are running.

    What changes is what a sequence is: an (n, V) array of V conditionally independent
    channels driven by one latent chain, rather than an (n,) array of observed states. The
    dissimilarity changes with it, but nothing downstream of the matrix does.
    """

    mixture_id: int
    K: int
    n_states: int
    n_vars: int
    n_categories: tuple
    generator: MixtureOfHMMGenerator
    key: str

    @property
    def alpha(self):
        """No Dirichlet concentration here; the field exists so the tables share a schema."""
        return float("nan")

    @property
    def d(self):
        return int(np.prod(self.n_categories))

    @property
    def weights(self):
        return self.generator.w

    def sample_component(self, k, n_sequences, seq_len, rng):
        return self.generator.sample_component(k, n_sequences, seq_len, rng=rng)

    def sample_dataset(self, N, seq_len, rng):
        saved = self.generator.rng
        self.generator.rng = rng
        try:
            data = self.generator.sample_dataset(N, seq_len)
        finally:
            self.generator.rng = saved
        return data["X"], data["y"]


def draw_hmm_mixture(K, n_states, n_vars, n_categories, mixture_id, base_seed,
                     alpha_A=0.5, alpha_B=0.3, min_weight=0.10):
    """Draw one mixture of homogeneous multichannel HMMs, on its own stream.

    `alpha_A` and `alpha_B` play the role `alpha` plays for Markov chains: small values give
    sharply peaked transition and emission laws, hence components that are easy to tell
    apart, large ones near-uniform laws and components that are not.
    """
    rng = stream(base_seed, "hmm-mixture", K, n_states, n_vars, mixture_id)
    generator = MixtureOfHMMGenerator(
        n_components=K, n_states=n_states, n_vars=n_vars, n_categories=n_categories,
        alpha_A=alpha_A, alpha_B=alpha_B, min_weight=min_weight, rng=rng)
    return HMMMixture(mixture_id=mixture_id, K=int(K), n_states=int(n_states),
                      n_vars=int(n_vars), n_categories=tuple(generator.C_list),
                      generator=generator,
                      key=seed_key("hmm-mixture", K, n_states, n_vars, mixture_id))


@dataclass(frozen=True)
class MultichannelOM:
    """Multichannel OM with costs fixed in advance, the interface of `UnivariateOM`.

    Costs are *not* estimated from the data, unlike `om.om_trate_distances`. Gamma^(n) is
    estimated on an independent sample, and a cost scheme derived from the sample would make
    the dissimilarity depend on which sequences happened to be drawn.
    """

    name: str
    costs: np.ndarray           # (V, C_max, C_max), the packed per-channel matrices
    indel: float                # the aggregated gap cost, sum_c lambda_c delta^(c)
    M_mc: float
    n_categories: tuple
    channel_indel: np.ndarray   # (V,), the per-channel gap costs

    @property
    def M(self):
        """M^mc, the aggregated largest substitution cost."""
        return self.M_mc

    def assumption_1(self):
        """Conditions (i)-(vi), checked on each channel.

        The paper notes right after Definition 4.1 that Assumption 1 is stable under the
        aggregation: if every ``(c_sub^(c), delta^(c))`` satisfies (i)-(vi) on Sigma_c, so
        does the aggregate on the product alphabet. Checking the channels is therefore
        equivalent, and it is the only tractable route -- five channels of five letters make
        a product alphabet of 3125, on which the triangle inequality alone is a 3125^3 table,
        227 GiB.

        Returns the per-condition conjunction over channels, plus ``M = max c_sub`` for the
        aggregate, which is the constant the profile threshold needs.
        """
        verdicts = []
        for c, size in enumerate(self.n_categories):
            S = self.costs[c][:size, :size]
            verdicts.append(check_assumption_metric(S, np.full(size, self.channel_indel[c])))
        out = {k: bool(all(v[k] for v in verdicts)) for k in verdicts[0] if k.startswith("(")}
        out["zero diagonal"] = bool(all(v["zero diagonal"] for v in verdicts))
        out["M = max c_sub"] = float(self.M_mc)
        return out

    def gamma_paths(self, X, Y, n_grid):
        return om_multichannel_paths(X, Y, np.asarray(n_grid, dtype=np.int64),
                                     self.costs, self.indel)

    def matrices(self, X, n_grid):
        return om_multichannel_matrices(X, np.asarray(n_grid, dtype=np.int64),
                                        self.costs, self.indel)


def multichannel_om(name, n_categories, **kwargs):
    """Build a `MultichannelOM` from one of the fixed schemes of `om`."""
    costs, indel, M, channel_indel = multichannel_cost_scheme(name, n_categories, **kwargs)
    return MultichannelOM(name=f"{name}-mc", costs=costs, indel=indel, M_mc=M,
                          n_categories=tuple(int(c) for c in n_categories),
                          channel_indel=channel_indel)


# ---------------------------------------------------------------------------
# The finite-horizon geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GammaEstimate:
    """Gamma^(n) for one fixed mixture, over a grid of horizons.

    `mean` and `se` are (K, K, G). Symmetry is used rather than assumed: each unordered
    pair is estimated once and written into both cells, so Gamma_kl and Gamma_lk are the
    same number and not two noisy estimates of it.
    """

    n_grid: np.ndarray          # (G,)
    mean: np.ndarray            # (K, K, G)
    se: np.ndarray              # (K, K, G)
    n_pairs: int                # R_gamma

    @property
    def K(self):
        return self.mean.shape[0]

    @property
    def n_entries(self):
        """The K(K+1)/2 distinct entries the simultaneous correction is spread over."""
        return self.K * (self.K + 1) // 2

    def intervals(self, level=0.95):
        """(lower, upper), each (K, K, G), simultaneous over the distinct entries."""
        return simultaneous_intervals(self.mean, self.se, self.n_pairs, self.n_entries, level)

    def eta(self):
        """Point estimate of eta_n = min_{k != l} Gamma_kl - max_k Gamma_kk, per horizon."""
        offdiag = ~np.eye(self.K, dtype=bool)
        return self.mean[offdiag].min(axis=0) - np.diagonal(self.mean).T.max(axis=0)

    def eta_bounds(self, level=0.95):
        """Conservative simultaneous bounds (L, U) on eta_n, per horizon.

        On the event that every entry lies in its interval -- which has probability at least
        `level` -- eta_n lies between these. Both are attained, so they cannot be tightened
        without assuming which entries achieve the min and the max.
        """
        lower, upper = self.intervals(level)
        offdiag = ~np.eye(self.K, dtype=bool)
        low = lower[offdiag].min(axis=0) - np.diagonal(upper).T.max(axis=0)
        high = upper[offdiag].min(axis=0) - np.diagonal(lower).T.max(axis=0)
        return low, high

    def separation_status(self, level=0.95):
        """One of SEPARATED / NONSEPARATED / UNCERTAIN per horizon.

        A classification of *finite-horizon* separation at level `level`. It says nothing
        directly about the asymptotic condition eta > 0; intervals that settle away from
        zero as n grows are evidence for it, and should be described that way.
        """
        low, high = self.eta_bounds(level)
        return np.where(low > 0.0, SEPARATED, np.where(high < 0.0, NONSEPARATED, UNCERTAIN))


def estimate_gamma_paths(mixture, om, n_grid, n_pairs, rng):
    """Estimate Gamma^(n) for every component pair, on independent trajectory pairs.

    For k = l the two trajectories are independent realisations of the *same* component:
    gamma(P, P) is not zero, and the diagonal of Gamma is precisely the within-component
    dispersion that eta_n subtracts.
    """
    n_grid = np.asarray(n_grid, dtype=np.int64)
    if np.any(np.diff(n_grid) <= 0):
        raise ValueError("n_grid must be increasing")
    K, G, n_max = mixture.K, n_grid.size, int(n_grid[-1])

    mean = np.empty((K, K, G))
    se = np.empty((K, K, G))
    for k in range(K):
        for l in range(k, K):
            X = mixture.sample_component(k, n_pairs, n_max, rng)
            Y = mixture.sample_component(l, n_pairs, n_max, rng)
            paths = om.gamma_paths(X, Y, n_grid)                  # (n_pairs, G)
            mean[k, l] = mean[l, k] = paths.mean(axis=0)
            se[k, l] = se[l, k] = paths.std(axis=0, ddof=1) / np.sqrt(n_pairs)
    return GammaEstimate(n_grid=n_grid, mean=mean, se=se, n_pairs=int(n_pairs))


# ---------------------------------------------------------------------------
# Clustering one dataset
# ---------------------------------------------------------------------------


def labels_at_k(D, k, algorithm, rng=None, pam_restarts=1):
    """Partition of D into k blocks by `algorithm`, with the degenerate k guarded.

    K_hat from the profile graph is data-driven and can come out at 1 or at N -- those are
    its two documented failure modes -- and running PAM at k = N would cost O(N^3) to
    return the singletons. Both ends are short-circuited.
    """
    N = D.shape[0]
    k = int(k)
    if k <= 1:
        return np.zeros(N, dtype=np.int64)
    if k >= N:
        return np.arange(N, dtype=np.int64)
    if algorithm == "pam":
        return pam(D, k, rng=rng, n_restarts=pam_restarts).labels
    return hac_labels(D, k, method=algorithm)


def run_dataset(mixture, om, N, n_grid, rng, dataset_id, algorithms=ALGORITHMS,
                pam_restarts=1, extra=None):
    """Simulate one clustering dataset and score every algorithm at every horizon.

    The dissimilarity matrix is computed once per horizon and reused by all algorithms, so
    what separates them is the algorithm alone. Exact recovery is the primary outcome, being
    what Theorems 3.5, 3.8 and 3.9 are statements about; ARI is kept as a secondary,
    graded reading of the same partition.

    K_hat is estimated once per horizon from the same matrix, and every algorithm is scored
    twice: at the true K, and at K_hat. The two questions of Section 7 -- does the rule find
    K, and does the resulting partition recover P* -- then come from the same run.
    """
    n_grid = np.asarray(n_grid, dtype=np.int64)
    X, truth = mixture.sample_dataset(N, int(n_grid[-1]), rng)
    matrices = om.matrices(X, n_grid)

    rows = []
    for g, n in enumerate(n_grid):
        D = matrices[g]
        # The safeguarded rule, not the threshold of the current Theorem 3.9: the latter
        # returns K_hat = 1 at every horizon these experiments can reach.
        rho = profile_distances(D)
        k_hat = profile_graph_k(None, rho=rho,
                                threshold=safeguard_threshold(rho, int(n),
                                                              profile_heights(rho)))
        for algorithm in algorithms:
            labels = labels_at_k(D, mixture.K, algorithm, rng=rng, pam_restarts=pam_restarts)
            at_k_hat = labels_at_k(D, k_hat, algorithm, rng=rng, pam_restarts=pam_restarts)
            certified = hit_cap = None
            if algorithm == "pam" and 1 < mixture.K < N:
                result = pam(D, mixture.K, rng=rng, n_restarts=pam_restarts)
                certified, hit_cap = result.one_swap_certified, result.hit_cap
            row = {
                "alpha": mixture.alpha, "K": mixture.K, "d": mixture.d,
                "mixture_id": mixture.mixture_id, "n": int(n), "N": int(N),
                "cost_scheme": om.name, "algorithm": algorithm, "dataset_id": dataset_id,
                "exact_recovery": int(exact_recovery(labels, truth)),
                "ari": adjusted_rand_index(labels, truth),
                "k_hat": int(k_hat),
                "k_correct": int(k_hat == mixture.K),
                "exact_recovery_at_k_hat": int(exact_recovery(at_k_hat, truth)),
                "pam_one_swap_certified": "" if certified is None else int(certified),
                "pam_hit_cap": "" if hit_cap is None else int(hit_cap),
                "mixture_key": mixture.key,
            }
            row.update(extra or {})
            rows.append(row)
    return rows


def gamma_rows(mixture, om, estimate, level=0.95, extra=None):
    """Tidy rows for Gamma^(n): one per (mixture, horizon, k <= l)."""
    lower, upper = estimate.intervals(level)
    rows = []
    for g, n in enumerate(estimate.n_grid):
        for k in range(estimate.K):
            for l in range(k, estimate.K):
                row = {
                    "alpha": mixture.alpha, "K": mixture.K, "mixture_id": mixture.mixture_id,
                    "n": int(n), "cost_scheme": om.name, "k": k, "l": l,
                    "gamma_hat": estimate.mean[k, l, g], "se": estimate.se[k, l, g],
                    "ci_low": lower[k, l, g], "ci_high": upper[k, l, g],
                    "n_pairs": estimate.n_pairs, "level": level,
                    "mixture_key": mixture.key,
                }
                row.update(extra or {})
                rows.append(row)
    return rows


def eta_rows(mixture, om, estimate, level=0.95, extra=None):
    """Tidy rows for eta_n and its finite-horizon separation verdict: one per horizon."""
    eta = estimate.eta()
    low, high = estimate.eta_bounds(level)
    status = estimate.separation_status(level)
    rows = []
    for g, n in enumerate(estimate.n_grid):
        row = {
            "alpha": mixture.alpha, "K": mixture.K, "mixture_id": mixture.mixture_id,
            "n": int(n), "cost_scheme": om.name,
            "eta_hat": eta[g], "eta_ci_low": low[g], "eta_ci_high": high[g],
            "separation_status": status[g],
            "n_pairs": estimate.n_pairs, "level": level,
            "mixture_key": mixture.key,
        }
        row.update(extra or {})
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


class ResultsWriter:
    """Append tidy rows to a CSV, flushing as they arrive.

    Figures are rebuilt from these tables, never from a rerun: the expensive sweeps are
    hours long and nothing in a figure should depend on repeating them. Flushing per batch
    also means a run killed halfway keeps everything it had already produced.
    """

    def __init__(self, path, fieldnames):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = list(fieldnames)
        existing = self.path.exists() and self.path.stat().st_size > 0
        self._handle = self.path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=self.fieldnames)
        if not existing:
            self._writer.writeheader()
        self.n_rows = 0

    def write(self, rows):
        for row in rows:
            self._writer.writerow(row)
        self.n_rows += len(rows)
        self._handle.flush()

    def close(self):
        self._handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
