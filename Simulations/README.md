# Numerical illustrations — OM-based clustering for mixtures of Markov chains

The code behind the figures of Section 5 and of the appendix of *Consistency of Optimal
Matching-based Clustering for Mixtures of Markov Chains*. Five modules hold the shared library
and four notebooks are the four experiments. Each notebook computes everything it reports --
mixtures, geometry, OM matrices, clustering -- fixes its seed in its first cell and writes its
figures to `Figures/`, so a full run reproduces the published ones.

```bash
pip install -r requirements.txt
jupyter lab
```

## The library

| module | what it holds |
|---|---|
| `om.py` | cost schemes and Assumption 1, the OM dissimilarity — univariate `om_distance` as in the paper, multichannel `om_multichannel_paths` and `om_trate_distances` — and the bounds of Proposition 2.10 |
| `generators.py` | what the sequences are drawn from: Markov chains and their mixtures, and mixtures of homogeneous multichannel HMMs |
| `clustering.py` | everything downstream of a dissimilarity matrix: single, average and complete linkage, `pam` with its one-swap certificate, the profile graph $\hat K$ of Theorem 3.8 and its thresholds, ASW, exact recovery and ARI |
| `experiments.py` | the two-level Monte Carlo engine: key-addressed streams, mixtures drawn once and held fixed, $\Gamma^{(n)}$ estimated from an independent sample, simultaneous intervals on its entries and the separation verdict they induce on $\eta_n$ |
| `figures.py` | the rcParams, colormaps and layout helpers shared by the notebooks |

## The experiments

Each notebook runs its own sweep and writes figures to `Figures/`. None of them reads a grid
computed earlier: a notebook is here to reproduce its section, not to store it. The knobs are
in the first code cell of each -- set `N` small to watch a section rebuild itself in seconds.
The OM matrices are the expensive part, so a full run is hours.

| notebook | paper | writes to `results/` | writes to `Figures/` |
|---|---|---|---|
| `om_convergence.ipynb` | §5.1 | — (samples its own paths) | `Convergence/gamma_convergence_{constant,trate,random}` |
| `difficulty_grid.ipynb` | §5.2–§5.3 | `grid_N{N}_{cluster,eta,khat}` | `grid_N800/{separation,recovery,k_selection,recovery_single_linkage}` |
| `recovery_path.ipynb` | §5.4 | `recovery_path_notebook_cluster` | `Recovery/ari_path_{average_linkage,pam,single_linkage}` |
| `hmm.ipynb` | appendix | `hmm_notebook_{cluster,eta,khat,path}` | `HMM/{gamma_convergence_*_hmm,separation_hmm,recovery_hmm,k_selection_hmm,ari_path_*_hmm}` |

`difficulty_grid` and `hmm` score the same two rules for $K$: `safeguard`, the rule of the
paper, against `asw-pam`, the default of applied sequence analysis. That is the comparison the
paper makes -- a consistent rule against the one practice uses -- and not a search over
heuristics, so nothing else is run.

## What is in `results/`

Exactly the eight tables the notebooks write, and nothing else: a sweep fills them, the
figures below it read them back within the same run, and the next run overwrites them. They
are committed so that a figure can be redrawn without repeating the sweep that produced it,
which is hours. Earlier runs are not kept -- the archive of the superseded `*_main.csv` and
`khat_*` tables was removed, git history holding what it was worth.

`Figures/` is laid out by experiment rather than by paper section, and `grid_N800/` carries the
$N$ of the run behind it: redrawing that grid at another $N$ writes elsewhere and cannot
silently overwrite the published figures. `grid_N800/grid_N800_report.txt` records the parameters
of that run.

`results/mixtures/` holds the drawn kernels, one file per mixture, and is not tracked: it lets
a single draw be rebuilt without replaying a sweep.

Numba caches its compiled kernels on disk, so only the first run of a session pays the ~10 s of
compilation. Without numba the library still runs, in pure Python and orders of magnitude slower.

## Agreement with TraMineR

The OM dissimilarity is not ours to define: `om.om_distance` and the two substitution
matrices reproduce **TraMineR** — `seqdist(method="OM")`, `seqcost(method="TRATE")` and
`seqcost(method="CONSTANT")` — *bit for bit*, on 80 draws covering the alphabet sizes, horizons
and mixtures of the four experiments, on releases 2.2.12 and 2.2.11 alike.

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
