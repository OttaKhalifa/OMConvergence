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
| `assumptions.ipynb` | §5.2 | `Assumptions/assumption_separation` (Fig. 2), `_strong_separation_a`, `_b` (Figs. 3a, 3b) |
| `recovery.ipynb` | §5.3–5.4 | `Recovery/ari_path_average_linkage`, `ari_grid_average_linkage` (Figs. 4a, 5a), the same two `_kmedoids` (Figs. 4b, 5b) and `_single_linkage` (Figs. 9, 10), `k_hat_grid` (Fig. 6) |

Wall-clock on eight cores: about a minute, two minutes, and about 30 h. The last one is the
$(\alpha, K)$ grid — $72 \times 30$ dissimilarity matrices at $N = 800$, $n = 1000$, some 45 s
each — and its outputs are therefore not stored in the notebook.

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
