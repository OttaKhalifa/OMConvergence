# Numerical illustrations — OM-based clustering for mixtures of Markov chains

The code behind the figures of Section 5 and of the appendix of *Consistency of Optimal
Matching-based Clustering for Mixtures of Markov Chains*. Four modules hold the shared library
and the three notebooks are the three experiments. Each fixes its seed in its first cell and
writes its figures to `Figures/`, so a full run reproduces the published ones.

| module | what it holds |
|---|---|
| `om.py` | cost schemes and Assumption 1, the OM dissimilarity — univariate `om_distance` as in the paper, multichannel `om_trate_distances` as in the benchmark — and the bounds of Proposition 2.8 |
| `generators.py` | what the sequences are drawn from: Markov chains and their mixtures, and mixtures of homogeneous multichannel HMMs |
| `clustering.py` | everything downstream of a dissimilarity matrix: single and average linkage, K-medoids, ARI and the plug-in estimates of Gamma |
| `figures.py` | the rcParams and colormaps shared by the three notebooks |

```
pip install -r requirements.txt
jupyter lab
```

| notebook | paper | figures written to `Figures/` |
|---|---|---|
| `om_convergence.ipynb` | §5.1 | `Convergence/gamma_convergence_constant` (Fig. 1), `_trate`, `_random` (Figs. 7, 8) |

**The experiments of §5.2 to §5.4 are being refactored.** They studied the *asymptotic*
matrix $\Gamma$ through a single long trajectory; they are being replaced by experiments that
estimate the finite-horizon $\Gamma^{(n)}$ from an independent Monte Carlo sample, attach
uncertainty to the sign of the separation margin $\eta_n$, and read clustering recovery at the
same $n$. The two superseded notebooks are parked in [`legacy/`](legacy/README.md) with the
reasons; git history keeps the runs that produced the published figures.

Numba caches its compiled kernels on disk, so only the first run of a session pays the ~10 s of
compilation. Without numba the library still runs, in pure Python and orders of magnitude slower.

## Agreement with TraMineR

The OM dissimilarity is not ours to define: `om.om_distance` and
`om.compute_trate_subst_matrix` reproduce **TraMineR 2.2.12** — `seqdist(method="OM")` and
`seqcost(method="TRATE")` — *bit for bit*, on 80 draws covering the alphabet sizes, horizons and
mixtures of the three experiments.

```bash
validation/run.sh --seeds 10    # needs R with TraMineR + jsonlite; about 3 min
```

[`validation/README.md`](validation/README.md) states the claim, its scope, what the four
details of `OMdistance.cpp` it rests on are worth, and what the check does *not* pin down.
Running the experiments does not need R — only re-checking the equivalence does.

## PAM against the algorithm the theory analyses

Theorem 3.8 is about a medoid set on which no improving one-swap exists, not about a global
minimiser of $\Phi$. `clustering.pam` is that algorithm, and every set it returns carries a
`one_swap_certified` flag established by exhaustive search over all $K(N-K)$ swaps.

```bash
python validation/check_pam.py    # no R needed; about a minute
```

The check verifies the incremental swap bookkeeping against $\Phi$ recomputed from scratch,
the normalisation of $\Phi$, one-swap stationarity of every output, and the edge cases; it
also reports how often PAM reaches the global optimum where enumeration is affordable.

## The Monte Carlo engine

`experiments.py` holds the two-level machinery the refactored experiments run on: mixtures
drawn once and held fixed, $\Gamma^{(n)}$ estimated from an independent sample, simultaneous
intervals on its entries, and the finite-horizon separation verdict they induce on $\eta_n$.

```bash
python validation/check_experiments.py    # about three minutes
```

It checks stream reproducibility, the empirical coverage of the binomial and simultaneous
intervals, the symmetry and coverage of $\hat\Gamma^{(n)}$, that the $\eta_n$ bounds bracket
the point estimate and cover a high-precision reference, that a separated mixture is
classified separated and a near-identical one is not, and that the tidy tables survive a
round trip.
