# Numerical illustrations — OM-based clustering for mixtures of Markov chains

The code behind the figures of Section 5 and of the appendix of *Consistency of Optimal
Matching-based Clustering for Mixtures of Markov Chains*. Five modules hold the shared library,
four notebooks are the four experiments, and three sweeps fill the tables the notebooks read.
Each notebook fixes its seed in its first cell and writes its figures to `Figures/`, so a full
run reproduces the published ones.

```bash
pip install -r requirements.txt
jupyter lab
```

## The library

| module | what it holds |
|---|---|
| `om.py` | cost schemes and Assumption 1, the OM dissimilarity — univariate `om_distance` as in the paper, multichannel `om_multichannel_paths` and `om_trate_distances` — and the bounds of Proposition 2.8 |
| `generators.py` | what the sequences are drawn from: Markov chains and their mixtures, and mixtures of homogeneous multichannel HMMs |
| `clustering.py` | everything downstream of a dissimilarity matrix: single, average and complete linkage, `pam` with its one-swap certificate, the profile graph $\hat K$ of Theorem 3.9 and its thresholds, ASW, exact recovery and ARI |
| `experiments.py` | the two-level Monte Carlo engine: key-addressed streams, mixtures drawn once and held fixed, $\Gamma^{(n)}$ estimated from an independent sample, simultaneous intervals on its entries and the separation verdict they induce on $\eta_n$ |
| `figures.py` | the rcParams, colormaps and layout helpers shared by the notebooks |

## The experiments

Every notebook reads tables from `results/` and writes figures to `Figures/`; none of them
computes an OM matrix. The sweeps do that, and they are the slow part.

| notebook | paper | reads | writes to `Figures/` |
|---|---|---|---|
| `om_convergence.ipynb` | §5.1 | — (samples its own paths) | `Convergence/gamma_convergence_{constant,trate,random}` |
| `difficulty_grid.ipynb` | §5.2–§5.3 | `recovery_cluster_main`, `recovery_eta_main`, `khat_grid_final` | `Grid/{separation,recovery,recovery_vs_eta,k_selection,recovery_single_linkage}` |
| `recovery_path.ipynb` | §5.4 | `path_cluster_main`, `path_eta_main` | `Recovery/ari_path_{average_linkage,pam,single_linkage}` |
| `hmm.ipynb` | appendix | `*_hmm.csv` | `HMM/{separation_hmm_vs_markov,recovery_hmm,k_selection_hmm,ari_path_*_hmm}` |

| sweep | fills |
|---|---|
| `sweep_recovery.py` | the $(\alpha, K)$ grid: `recovery_{cluster,eta,gamma}_*.csv` |
| `sweep_khat.py` | selecting $K$ — our rule against ASW: `khat_{grid,eta}_*.csv` |
| `sweep_path.py` | recovery against the horizon $n$: `path_{cluster,eta}_*.csv` |

Each takes `--mechanism markov|hmm`, appends as it goes and skips the cells already present in
its output, so an interrupted run resumes. `run_hmm_night.sh` replays all three on mixtures of
five-channel HMMs, which is what the appendix reports.

## State of the tables

`results/*_main.csv` are complete at $N = 200$ and back every figure of the body.

On the HMM side only `path_{cluster,eta}_hmm.csv` are complete — six mixtures at $\alpha = 3$.
`recovery_*_hmm.csv` hold one cell of the 72 the grid needs, from an interrupted run, and the
$\hat K$ sweep never started. The files already in `Figures/HMM/` predate the recalibration of
the HMM $\alpha$ axis and do not match the grid `run_hmm_night.sh` now uses.

`Figures/Recovery/` also still holds `ari_grid_*`, `k_hat_grid` and the `kmedoids` figures the
superseded notebooks in `../legacy/` produced; `Figures/grid_800/` is what replaces them.

Numba caches its compiled kernels on disk, so only the first run of a session pays the ~10 s of
compilation. Without numba the library still runs, in pure Python and orders of magnitude slower.

## Agreement with TraMineR

The OM dissimilarity is not ours to define: `om.om_distance` and
`om.compute_trate_subst_matrix` reproduce **TraMineR 2.2.12** — `seqdist(method="OM")` and
`seqcost(method="TRATE")` — *bit for bit*, on 80 draws covering the alphabet sizes, horizons and
mixtures of the four experiments.

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
