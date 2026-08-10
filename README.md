# Numerical illustrations — OM-based clustering for mixtures of Markov chains

Simulation code for the paper *Consistency of Optimal Matching-based Clustering for Mixtures of
Markov Chains*. It contains exactly what produces the figures of Section 5 and of the appendix,
and nothing else.

The OM (edit) dissimilarity is given by a substitution cost matrix `S` =
$(c_{\mathrm{sub}}(a,b))_{a,b}$ and a gap cost vector `delta` = $(\delta(a))_a$ on a finite
alphabet $\Sigma$, $|\Sigma| = d$, with $M := \max_{a,b} c_{\mathrm{sub}}(a,b)$; the pair
`(S, delta)` is the extended cost $\bar d$ of equation (1).

## Layout

| file | role |
|---|---|
| `om_lib.py` | shared library: cost schemes, Assumption 1 checker, Markov sampling, OM dissimilarity, mixtures, clustering, ARI, the bounds of Proposition 2.8, figure style |
| `om_convergence.ipynb` | Section 5.1 — almost sure convergence of $\hat\gamma_n$ and the two bounds |
| `assumptions.ipynb` | Section 5.2 — how often Assumptions 3 and 4 hold under a Dirichlet prior |
| `recovery.ipynb` | Sections 5.3 and 5.4 — ARI of the three clustering procedures, and the largest-gap estimator of $K$ |
| `Figures/` | the generated figures, `.pdf` for the paper and `.png` for viewing |

## Figures

| paper | file | notebook |
|---|---|---|
| 1 | `Figures/Convergence/gamma_convergence_constant` | `om_convergence.ipynb` |
| 2 | `Figures/Assumptions/assumption_separation` | `assumptions.ipynb` |
| 3a, 3b | `Figures/Assumptions/assumption_strong_separation_a`, `..._b` | `assumptions.ipynb` |
| 4a, 4b | `Figures/Recovery/ari_path_average_linkage`, `ari_path_kmedoids` | `recovery.ipynb` |
| 5a, 5b | `Figures/Recovery/ari_grid_average_linkage`, `ari_grid_kmedoids` | `recovery.ipynb` |
| 6 | `Figures/Recovery/k_hat_grid` | `recovery.ipynb` |
| 7, 8 | `Figures/Convergence/gamma_convergence_trate`, `..._random` | `om_convergence.ipynb` |
| 9, 10 | `Figures/Recovery/ari_path_single_linkage`, `ari_grid_single_linkage` | `recovery.ipynb` |

## Running

Requires `numpy`, `scipy`, `matplotlib`, `numba` and `tqdm`:

```
pip install -r requirements.txt
jupyter lab
```

Every notebook fixes its seed at the top and writes its figures to `Figures/`, so a full run
reproduces the published ones. Wall-clock, on eight cores: about a minute for
`om_convergence.ipynb`, two minutes for `assumptions.ipynb`, and about 30 h for `recovery.ipynb`
— one $800 \times 800$ dissimilarity matrix at $n = 1000$ takes some 45 s, and the $(\alpha, K)$
grid needs $72 \times 30$ of them. The outputs of `recovery.ipynb` are therefore not stored in
the file; the figures it wrote are in `Figures/Recovery/`.

The notebooks import `om_lib` and use `%autoreload`, so edits to the library take effect without
restarting the kernel. Numba caches its compiled kernels on disk — possible because the code
lives in a `.py` file and not in a cell — so only the first run of a session pays the ~10 s of
compilation. Without numba the library still runs, in pure Python and orders of magnitude slower.

## `om_lib.py`

Cost schemes (`cost_scheme`) come in three flavours: `constant` ($c_{\mathrm{sub}} \equiv 2$,
$\delta \equiv 1$, the usual sequence-analysis default), `random` (symmetric $c_{\mathrm{sub}}
\sim \mathcal U[1.2, 2]$) and `trate` ($c_{\mathrm{sub}} = 2 - \hat P - \hat P^\top$).
`check_assumption_metric` verifies conditions (i)–(vi) of Assumption 1 numerically and returns
$M$; every scheme used in the notebooks passes it.

The OM dissimilarity (`om_distance`) is a **two-row** dynamic program, $O(nm)$ time and $O(m)$
memory. The full $(n+1)\times(m+1)$ table would take 800 MB per pair at $n = 10^4$, which would
rule out running pairs in parallel. `gamma_hat_pairs` computes $\hat\gamma_n$ for a batch of
pairs and `gamma_hat_paths` whole sample paths $n \mapsto \hat\gamma_n$; both parallelise over
the batch with `prange`. `om_matrices` returns the full $N \times N$ matrix at each horizon of a
grid of nested prefixes, parallelising over the $\binom{N}{2}$ pairs. Nested prefixes share no
work: the two-row program keeps no table to snapshot, so a matrix over a grid of horizons costs
the sum of the $n_g^2$, not the largest of them.

For the clustering: `sample_mixture` draws $N$ sequences from a $K$-component mixture with their
latent labels; `single_linkage_tree` is Kruskal's algorithm, which returns the merge heights
$h_1 \le \dots \le h_{N-1}$ and the merged pairs in one pass; `cut_at_k` and `largest_gap_k` are
the two ways the paper cuts that tree; `average_linkage_labels` delegates the tree to `scipy` and
cuts it at $K$ blocks; `kmedoids` is PAM's BUILD followed by alternating refinement, best of
several restarts. `adjusted_rand_index` avoids a `scikit-learn` dependency.
`gamma_block_means` and `separation_levels` give the plug-in $\hat\Gamma$,
$\hat\Delta_{\mathrm{in}}$, $\hat\Delta_{\mathrm{out}}$ from a dissimilarity matrix and the true
labels. `wasserstein_lower_bound` (a linear program) and `product_upper_bound`
($\pi_P^\top S \pi_Q$) are the two bounds of Proposition 2.8.

Two points were checked against reference implementations while the code was written:
`single_linkage_tree` reproduces the merge heights of `scipy.cluster.hierarchy.linkage` and
`adjusted_rand_index` those of `sklearn.metrics.adjusted_rand_score`, to machine precision; and
the `kmedoids` heuristic reaches the *global* minimiser of $\Phi_{N,n}$ — the object Theorem 3.8
speaks about — wherever enumerating the $\binom{N}{K}$ medoid sets was affordable, up to $N = 40$
and $K \le 4$. At the sizes the notebooks use, $\binom{800}{4} = 1.7 \cdot 10^{10}$ puts
enumeration out of reach.

One consequence of integer costs: with $c_{\mathrm{sub}} \equiv 2$, $\delta \equiv 1$ every
$d_{\mathrm{OM}}$ is an integer, so $\hat\gamma_n$ takes at most $2n+1$ values and ex æquo merge
heights are common. The dendrogram is then genuinely non-unique, and a cut falling exactly on a
tie is arbitrary — deterministic and reproducible, but implementation-dependent. It only matters
when the tie straddles the cut, hence only near the transition.
