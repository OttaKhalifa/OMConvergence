# Experiment 3 — ARI of single-linkage clustering: complete protocol

Self-contained description of the numerical experiment implemented in `recovery.ipynb`, for the
paper *Consistency of Optimal Matching-based Clustering for Mixtures of Markov Chains*. Everything
below is what the code actually does, not what an idealised version of it would do; the places
where the implementation constrains the design are flagged.

---

## 1. Question the experiment answers

The separation condition $\Delta_{\mathrm{out}} > \Delta_{\mathrm{in}}$ (Assumption 3) is
**sufficient** for single linkage to recover the true partition. Experiment 2
(`assumptions.ipynb`) measured *how often it holds* over a grid of mixtures, and found it fails
increasingly past $K = 4$. Experiment 3 asks the converse question in two parts:

1. **Where does recovery set in** as the two sample sizes $N$ (number of sequences) and $n$
   (length of a sequence) grow together?
2. **How much recovery survives where the condition fails** — i.e. is Assumption 3 necessary, or
   only sufficient?
3. **What does knowing $K$ buy?** Everything else assumes $K$ known; a final section drops that
   assumption and runs the largest-gap estimator of $K$ on the same trees, which is governed by
   Assumption 4 rather than Assumption 3.

Nothing is conditioned on an assumption being satisfied: the mixture of the first figure is a
first, unselected draw, and the $(\alpha, K)$ grid of the second is kept in full.

---

## 2. Statistical model

* Alphabet $\Sigma = \{0,1,2,3,4\}$, so $d = |\Sigma| = 5$. First-order Markov chains.
* A mixture has $K$ components. Each component's kernel $P_k$ is drawn **row-wise** from a
  Dirichlet$(\alpha,\dots,\alpha)$ prior: the $d$ rows are i.i.d. Dirichlet draws on the simplex.
  Small $\alpha$ gives near-deterministic, slowly mixing kernels; large $\alpha$ gives kernels
  close to uniform, hence mutually indistinguishable.
* Mixing weights are **balanced**, $w_k = 1/K$. Single linkage needs no balance condition
  (Assumption 5 is only required by $K$-medoids), so nothing is gained by unbalancing them here.
* $N$ sequences with i.i.d. latent labels $Z_i \sim \mathrm{Unif}\{1,\dots,K\}$; conditionally on
  $Z_i = k$, sequence $X_i$ is a length-$n$ trajectory of the chain with kernel $P_k$.
* **Every sequence starts from its own initial law**: the Dirac law at a state drawn uniformly on
  $\Sigma$, independently across sequences. This is a deliberate protocol choice, see §7.

Implementation: `om_lib.sample_mixture(K, N, n, d, alpha, rng, weights=None, kernels=None)`.
Passing `kernels` keeps the mixture fixed across replicates and redraws only labels and
trajectories; omitting it redraws the kernels too. Trajectory sampling is a vectorised
inverse-cdf walk (`sample_chain_order1`, numba-compiled).

## 3. Dissimilarity

* **Cost scheme**, fixed in advance and deterministic, as the theory requires: substitution cost
  $c_{\mathrm{sub}}(a,b) = 2$ for $a \neq b$ and $0$ otherwise; gap (indel) cost
  $\delta(a) = 1$ for every $a$. Hence $M := \max_{a,b} c_{\mathrm{sub}}(a,b) = 2$.
  Conditions (i)–(vi) of Assumption 1 are verified numerically at run time by
  `check_assumption_metric` and asserted.
* $d_{\mathrm{OM}}(x,y)$ is the optimal-matching (edit) distance for that scheme, computed by a
  **two-row dynamic program**, $O(nm)$ time and $O(m)$ memory (`om_distance`, numba, `fastmath`).
  The full $(n+1)\times(m+1)$ table is never materialised — at $n = 10^4$ it would be 800 MB per
  pair and would rule out running pairs in parallel.
* The normalised dissimilarity is $\hat\gamma_n(i,j) = d_{\mathrm{OM}}(X_{i,1:n}, X_{j,1:n})/n$.
* `om_matrices(X, grid, S, delta)` returns the $(\,|grid|, N, N)$ array of these matrices over a
  grid of horizons. The parallel loop (`prange`) runs over the $\binom{N}{2}$ pairs, all horizons
  of a pair being handled by one thread.

> **Implementation fact that constrains the design.** Nested prefixes share **no work**: the
> kernel calls `om_distance` once per pair *and per horizon*, because the two-row DP keeps no
> table from which the value at length $n$ could be read off a longer run. One matrix over a
> horizon grid $(n_1,\dots,n_G)$ therefore costs $\propto N^2 \sum_g n_g^2$, **not** $N^2 n_G^2$.
> For the 11 horizons of the path below, $\sum_g n_g^2 = 3.9\,n_{\max}^2$.

## 4. Estimator and measurements

**Clustering.** Single linkage, computed as Kruskal's algorithm on the complete graph weighted by
$D$ (`single_linkage_tree`): the $\ell$-th merge is the $\ell$-th smallest edge of a minimum
spanning tree, which returns the merge heights $h_1 \le \cdots \le h_{N-1}$ and the merged pairs in
one pass. The partition is the **cut at $K$ blocks** (`cut_at_k`: the first $N-K$ merges,
Definition 4.2 of the paper), with $K$ known.

*Why the cut at $K$ and not at a level $t$.* The consistency theorem is stated for the cut at a
fixed $t \in (\Delta_{\mathrm{in}}, \Delta_{\mathrm{out}})$, but its proof shows that on the event
$\mathcal E_{N,n}(\varepsilon)$ the two cuts coincide; cutting at $K$ avoids an oracle threshold no
practitioner could pick. When a class comes out empty, the cut is taken at the number of non-empty
classes, which is what $P^\star$ counts.

**Quantities reported.**
* the **ARI** between the true partition $P^\star$ and the estimated one — it degrades gracefully,
  so it separates "two clusters merged" from "partition unrelated to the truth";
* the **exact-recovery event** $\{\mathrm{ARI} = 1\}$ (numerically $\mathrm{ARI} > 1 - 10^{-12}$),
  which is the event the theorem bounds;
* $\hat K$, the **largest-gap estimator** of $K$, and the ARI of the partition cut at $\hat K$ —
  a second partition read off the *same* tree, see §5.3.

`adjusted_rand_index` is a local implementation (no `scikit-learn` dependency); merge heights and
ARI agree with `scipy.cluster.hierarchy.linkage` and `sklearn.metrics.adjusted_rand_score` to
machine precision.

**The separation margin, measured on the same data.** Once the $N \times N$ matrix is computed,
the true labels give a plug-in estimate of $\Gamma$ at no extra cost (`gamma_block_means`,
`separation_levels`):

$$\hat\Gamma_{k\ell} = \operatorname*{avg}_{Z_i = k,\, Z_j = \ell,\, i \neq j} \hat\gamma_n(i,j),
\qquad
\hat\Delta_{\mathrm{in}} = \max_k \hat\Gamma_{kk}, \qquad
\hat\Delta_{\mathrm{out}} = \min_{k \neq \ell} \hat\Gamma_{k\ell},$$

and the **margin** $\hat\eta = \hat\Delta_{\mathrm{out}} - \hat\Delta_{\mathrm{in}}$
($\hat\Delta_{\mathrm{out}}^{\max} = \max_{k\neq\ell}\hat\Gamma_{k\ell}$ is also available).
The diagonal of $D$ is excluded from the within-class blocks; blocks with no available pair are
`NaN` and are skipped by `nanmax`/`nanmin`. **ARI and margin are thus measured on the same
repetitions**, which is what makes the scatter of §5.2 legitimate.

Note for the paper: this plug-in averages over $|C_k|(|C_k|-1)/2$ or $|C_k||C_\ell|$ pairs, whereas
`assumptions.ipynb` uses **one pair per block** (two independent realisations per component). It is
far less noisy, so the two notebooks are not expected to agree digit for digit on
$\Delta_{\mathrm{in}}$, $\Delta_{\mathrm{out}}$.

---

## 5. The two experiments

### 5.1 A path of increasing $(N, n)$

**What the bound says.** The exact-recovery bound is $N^2\exp(-\varepsilon^2 n/(2C^\star))$, which
rewrites as

$$\exp\Big(2\log N\Big(1 - \frac{\lambda}{\lambda^\star}\Big)\Big), \qquad
\lambda := \frac{n}{\log N}, \qquad \lambda^\star := \frac{4C^\star}{\varepsilon^2},$$

with $\varepsilon = \eta/2$ and $C^\star = C_{\mathrm{Pau}} M^2 \tau_{\mathrm{mix}}$. The two
sample sizes enter **only through the ratio $\lambda$**; the remaining $\log N$ multiplies the
exponent, so it sharpens the transition without moving it. The statement is therefore about
neither $n \to \infty$ at fixed $N$ nor $N \to \infty$ at fixed $n$, but about a **joint regime**.

**Design.** $N$ and $n$ are the two quantities a practitioner actually has, so they are the two
that are set: an explicit list of strictly increasing pairs $(N_i, n_i)$ — no cross product, and no
horizon derived from a target $\lambda$. $\lambda$ is a *readout* of the design, not a control.
$N$ is evenly spaced on the hundreds; $n$ lives on a grid of 50 and is chosen so that the realised
$\lambda_i$ grows as linearly as the rounding allows (`make_path`).

| $i$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| $N_i$ | 100 | 150 | 250 | 300 | 400 | 450 | 500 | 600 | 650 | 750 | 800 |
| $n_i$ | 100 | 150 | 250 | 350 | 450 | 500 | 600 | 700 | 800 | 900 | 1000 |
| $\lambda_i$ | 21.7 | 29.9 | 45.3 | 61.4 | 75.1 | 81.8 | 96.5 | 109.4 | 123.5 | 136.0 | 149.6 |

so $N$ spans a factor 8, $n$ a factor 10 and $\lambda$ a factor 6.9. Increments of $\lambda$: 8.2,
15.3, 16.1, 13.7, 6.7, 14.7, 12.9, 14.1, 12.4, 13.6 — least-squares slope 12.9, maximum deviation
from the straight line 3.4, i.e. 26 % of one step. That irregularity is entirely a rounding
artefact: eleven evenly spaced values of $N$ between 100 and 800 do not land on multiples of 50, so
both coordinates get nudged and $\lambda$ inherits the jitter. Nothing is biased by it — the
abscissa of the figure is the **realised** $\lambda$, so every point is plotted where it belongs;
only the sampling density along the axis is affected.

*Why $n$ grows only slightly faster than $N$.* Exactly linear $\lambda$ would require
$n_i = \lambda_i \log N_i$; over this range $\lambda$ is multiplied by 6.9 while $\log N$ only goes
from 4.6 to 6.7, so the growth of $n$ is dominated by that of $\lambda$ and $n/N$ moves little.
Rounding erases most of the remainder.

**The path is nested in both directions.** One replicate draws $N_{\max} = 800$ sequences of
length $n_{\max} = 1000$ **once**; the point $i$ reads the *first* $N_i$ of them truncated to their
*first* $n_i$ symbols. A replicate is therefore a single dataset growing in both directions — the
two-dimensional analogue of the nested prefixes used in experiments 1 and 2 — and a curve
$i \mapsto \mathrm{ARI}_i$ is a genuine sample path, not a sequence of independent draws.

**Fixed mixture.** $K = 4$, $\alpha = 0.3$, the $K$ kernels drawn once (first draw, no selection);
across replicates only labels and trajectories are redrawn. Everything the bound depends on besides
$N$ and $n$ ($\varepsilon$ through $\eta$, $C^\star$ through the mixing times) is held constant
along the path.

*Why $K = 4$, $\alpha = 0.3$.* Measuring a threshold needs the threshold inside the observation
window. Rounding $n$ to a grid of 50 with $n \ge 100$ puts the low end of the window at
$\lambda \approx 20$, and since $\lambda^\star \propto 1/\eta^2$, an easy mixture would sit at
$\mathrm{ARI} = 1$ from the first point. $K = 4$ is the largest $K$ at which experiment 2 still
found the separation condition satisfied most of the time — the edge of the region the theory
covers — and $\alpha = 0.3$ keeps the chains distinguishable while still mixing fast, which
experiment 2 identified as the most reliable regime.

**Replicates.** $R = 100$. The figure draws **all** replicate curves as thin lines
(`lw=0.6, alpha=0.16`) with the mean on top (`lw=1.9`), the charte of `om_convergence.ipynb`; the
spread between them is the variability of one experiment, not an interval around a mean.

**Printed readouts.** A table of $N_i$, $n_i$, $\lambda_i$, mean ARI, $\hat{\mathbb P}(\mathrm{ARI}=1)$,
median margin; and the $\lambda$ at which the mean ARI first reaches 0.5 and 0.99, by linear
interpolation, together with the $N$ and $n$ interpolated at that $\lambda$.

**Limit to state explicitly in the paper.** A single path **cannot** separate "the transition is
governed by $\lambda$" from "governed by $n$", because $N$ and $n$ move together. Telling those
apart requires several $N$ at a common $\lambda$ — a design that was implemented and deliberately
dropped (see §8). What the path does show is the weaker and more directly useful statement: that
recovery sets in along the joint regime the theorem describes, and where.

### 5.2 The $(\alpha, K)$ grid

* $\alpha \in \{0.1, 0.2, 0.3, 0.4, 0.5, 1, 5, 10\}$ and $K \in \{2,\dots,10\}$: **72 cells**, the
  same grid as `assumptions.ipynb`, so the heatmaps can be read cell against cell.
* **New chains at every repetition** (unlike the path, whose mixture is fixed): each repetition
  draws its own $K$ kernels from the Dirichlet$(\alpha)$ prior, so a cell averages over mixtures,
  not over trajectories of one mixture.
* Run at the **last point of the path**, $N = 800$ and a single horizon $n = 1000$, tied to it in
  the code (`N_SWEEP = int(PATH_N[-1])`, `GRID_N = np.array([PATH_LEN[-1]])`) rather than
  transcribed. The grid therefore sits at an $(N, n)$ the curve of §5.1 actually visits, and one
  where that curve says whether the sample size suffices for at least one mixture. Note that
  $n = 1000$ is *not* the $n = 1500$ of `assumptions.ipynb`: the comparison with that notebook is
  at equal $(\alpha, K)$, not at equal sequence length.
* $R = 30$ repetitions per cell. A cell costs $O(N^2 n^2)$ **independently of $K$**, so the grid is
  uniform in price.
* Per cell: mean ARI, $\hat{\mathbb P}(\mathrm{ARI}=1)$, median margin $\hat\eta$,
  $\hat{\mathbb P}(\hat\eta > 0)$, median $\hat\Delta_{\mathrm{in}}$ and
  $\hat\Delta_{\mathrm{out}}$.

**Figures.**
1. `ari_grid_single_linkage` — a single figure, two heatmaps: mean ARI and
   $\hat{\mathbb P}(\mathrm{ARI}=1)$. Nothing else is drawn on them. The separation margin is
   **deliberately not shown**: it describes the assumption, not the performance of the algorithm,
   and `assumptions.ipynb` already reports it on this very grid. The comparison "is Assumption 3
   necessary" is therefore made by **putting the two figures side by side** — same $\alpha$ rows,
   same $K$ columns — and reading where the ARI is near 1 while the Assumption 3 panel says the
   condition fails.
2. `design_N_n` — the whole design in the $(N,n)$ plane: the path as a staircase, the grid as a
   single marker on its last point, and the level sets of $\lambda$ as light curves.

The margin is nevertheless still measured on the same repetitions and kept in `sweep["margin"]`
and `sweep["p_sep"]`, whose ranges are printed after the sweep; it costs nothing once the matrix
exists (§4). Two earlier outputs were dropped: a scatter of mean ARI against median margin (one
marker per cell, coloured by $K$ and by $\alpha$) and a cell counting the cells with a non-positive
margin.

### 5.3 Estimating $K$: the largest-gap rule

Everything above assumes $K$ known — that is what the theorem is stated for, and it is what lets
the tree be cut at $K$ blocks instead of at an oracle level $t$. This section drops the assumption.

* **Estimator.** $\hat K = N - \arg\max_{1 \le \ell \le N-2}(h_{\ell+1} - h_\ell)$, equation (4.4):
  the dendrogram is cut where it opens widest (`om_lib.largest_gap_k`). It is computed from the
  merge heights `single_linkage_tree` already returns, so it costs **no extra alignment** and is
  evaluated on the *same tree, same data, same repetitions* as everything above — in both the path
  and the grid.
* **What governs it.** Not Assumption 3 but **Assumption 4** (strong separation), in either form
  $\Delta_{\mathrm{out}} > 2\Delta_{\mathrm{in}}$ (4a) or
  $2\Delta_{\mathrm{out}} > \Delta_{\mathrm{out}}^{\max} + \Delta_{\mathrm{in}}$ (4b). Experiment 2
  found those essentially confined to $K = 2$: 4(a) peaks at $0.78$ at $(\alpha,K) = (0.1,2)$ and
  is zero for $K \ge 4$. The prediction under test is that $\hat K$ collapses well before the ARI
  at known $K$ does.
* **Reported.** $\hat{\mathbb P}(\hat K = K)$, the median $\hat K$ (which says *how* the rule
  fails, not only that it does), and the ARI of the partition cut at $\hat K$ next to the ARI at
  known $K$ — their difference being the price of not knowing $K$.
* **Figures.** `k_hat_path` — along the path, the ARI at $K$ and at $\hat K$ (all replicate curves
  plus the mean), and $\hat{\mathbb P}(\hat K = K)$ with the median $\hat K$. `k_hat_grid` — a
  single heatmap over the $(\alpha,K)$ grid, in the same charte as the ARI grid:
  $\hat{\mathbb P}(\hat K = K)$ in colour and in the upper figure, the median $\hat K$ below.
  It is designed to be read cell against cell with the Assumption 4(a) panel of
  `assumptions.ipynb`. The mean ARI achieved at $\hat K$ is measured and kept in
  `sweep["ari_k_hat"]` but not drawn — the question the grid asks is whether $K$ itself is
  recoverable; the per-$K$ table printed alongside gives it against the ARI at known $K$,
  averaged over $\alpha$.

> **Caveat that must be stated in the paper.** With $c_{\mathrm{sub}} \equiv 2$, $\delta \equiv 1$,
> every $\hat\gamma_n$ is a multiple of $1/n$, so the gaps $h_{\ell+1}-h_\ell$ are massively tied.
> `largest_gap_k` resolves a tie by taking the **smallest** index $\ell$, i.e. the **largest**
> $\hat K$, so whenever the true gap does not dominate the quantum $1/n$ the estimator is biased
> hard towards $\hat K \approx N$. On a reduced test run ($N \le 400$, $n \le 400$, $K = 4$) the
> median $\hat K$ came out at 50, 78, 114, 294 — a degenerate answer produced by tie-breaking, not
> by the geometry of the dendrogram; at $K = 2$ on the grid, where the problem is easy, the rule
> worked ($\hat{\mathbb P} = 0.75$). At $n = 1000$ the quantum is $1/1000$ and a margin of $0.09$
> spans some 90 quanta, so the true gap should dominate — but this has to be checked on the real
> run rather than assumed, and the alternative tie-breaks (towards the largest $\ell$, or
> restricting the argmax to $\ell \ge N - K_{\max}$) change the estimator's definition and were
> deliberately *not* applied.

### 5.4 $K$-medoids on the same matrices

Single linkage is not the only estimator the paper studies. $K$-medoids minimises

$$\Phi_{N,n}(m) = \sum_{i=1}^{N} \min_{1 \le k \le K} d_{\mathrm{OM}}(X_i, m_k)$$

over the medoid sets $m$, and its consistency needs **Assumption 5** (balance),
$\Delta_{\mathrm{out}} > (1 + w_{\max}/w_{\min})\Delta_{\mathrm{in}}$ — with the balanced weights
used throughout, $\Delta_{\mathrm{out}} > 2\Delta_{\mathrm{in}}$, i.e. exactly Assumption 4(a),
far more demanding than the $\Delta_{\mathrm{out}} > \Delta_{\mathrm{in}}$ single linkage needs.
It is run on the *same matrices*, the *same repetitions* and the *same mixtures*, on both the path
and the grid, so the comparison isolates the algorithm. It is free: a medoid search is a few
hundredths of a second against the tens of seconds one matrix costs.

* **Algorithm** (`om_lib.kmedoids`). The theorem is about the *global* minimiser of $\Phi$, which
  would need the $\binom{N}{K}$ medoid sets enumerated — $\binom{800}{4} = 1.7\cdot10^{10}$, out of
  reach. The minimisation is therefore heuristic: PAM's deterministic BUILD (start from the medoid
  minimising the total dissimilarity, then repeatedly add the point that reduces $\Phi$ most), then
  alternate assignment to the nearest medoid with within-cluster re-centring until the medoid set
  is stable; best of `MED_RESTARTS = 10` runs, the others started at random. The seed is fixed, so
  the partition is a deterministic function of $D$.
* **Validation** (`check_clustering.py`, outside the notebook). Against `kmedoids_exhaustive`, the
  heuristic reached the global optimum in **18/18** draws at $(N,K) = (40,2), (40,3), (30,4)$, with
  the same partition in 17 of them. On an independent, tie-free testbed — 8 overlapping Gaussians
  in dimension 4, 400 points, all $79\,800$ distances distinct — `single_linkage_tree` reproduces
  the merge heights of `scipy.cluster.hierarchy.linkage` to $0$ exactly, `cut_at_k` returns exactly
  the partition of `sklearn.cluster.AgglomerativeClustering(linkage="single", metric="precomputed")`
  (ARI $= 1$), and `adjusted_rand_index` matches `sklearn.metrics.adjusted_rand_score` to
  $1.6\cdot10^{-17}$.
* **Figures** (appendix material): `medoids_path`, single linkage against $K$-medoids along the
  path, all replicate curves plus the means; `medoids_grid`, a single heatmap of the mean ARI of
  $K$-medoids with $\hat{\mathbb P}(\mathrm{ARI}=1)$ below, in the same charte as the
  single-linkage grid so the two can be read cell against cell. Per-$K$ tables of the gap between
  the two estimators are printed alongside.

> **What this does not do.** The weights are balanced at every point, $w_{\max}/w_{\min} = 1$, so
> Assumption 5 is probed at its most favourable setting and never along the axis it is really
> about. Varying $w_{\max}/w_{\min}$ until
> $A = w_{\min}(\Delta_{\mathrm{out}}-\Delta_{\mathrm{in}}) - w_{\max}\Delta_{\mathrm{in}}$ changes
> sign, with single linkage as the control, remains a separate experiment.

**An early signal, from a reduced test run** ($N \le 300$, $n \le 300$, $K = 4$, $\alpha = 0.3$, 2
replicates — indicative only): $K$-medoids is **far better** than single linkage at these sizes,
mean ARI $0.89$ against $0.35$ at the first point and $0.98$ against $0.73$ at the last. If that
survives the full run it is worth a paragraph: the estimator requiring the *stronger* assumption is
the one that performs better at finite sample, single linkage paying for its chaining behaviour
long before its assumption fails.

---

## 6. Reproducibility and cost

* `SEED = 20260803`. The path's kernels come from `default_rng(SEED)`, the path sweep from
  `SEED + 1`, the grid sweep from `SEED + 2`.
* Cost model used to size the run: one $N\times N$ matrix over horizons $(n_g)$ costs
  $\mathrm{RATE}\cdot N^2 \sum_g n_g^2$, with `RATE` measured on the machine at run time and
  multiplied by an `OVERHEAD` of **1.17** covering what the alignment kernel does not: mixture
  sampling, Kruskal's algorithm in pure Python, memory traffic. Calibrated on a completed run:
  1080 matrices at $N = 800$ over horizons $(400, 800)$ took 37 753 s, i.e. 35.0 s each against
  30.0 s predicted.
* Projected for the run described above, on a 32-core machine: path 3.9 h (≈ 2 min per replicate),
  grid 21.4 h (36 s per cell-repetition), **total ≈ 25 h ≈ 1 day**. The raw rate has been observed
  to vary by 40 % with machine load, and the projection with it. For the record, the ratio
  $\sum_g n_g^2 / n_{\max}^2$ for this path is 3.97, i.e. the eleven horizons cost four times the
  last one alone.

## 7. Methodological decisions that changed the results

These are worth a sentence each in the paper; each one is a case where an earlier version of the
code gave materially different numbers.

* **Costs are fixed in advance.** The theory requires a deterministic cost scheme. Data-driven
  (TRATE) costs are therefore confined to an appendix of experiment 1, and estimated there on a
  *pilot sample independent* of the sequences entering $\hat\gamma_n$.
* **Every trajectory has its own initial law.** An earlier version started both realisations of a
  component from the model's own `init_probs`, drawn once per model from Dirichlet$(\alpha)$ — a
  near-Dirac for small $\alpha$, hence the *same* starting state. With near-deterministic kernels
  both realisations then follow the same trajectory and $\Gamma_{kk} \approx 0$ mechanically: at
  $\alpha = 10^{-4}$ that protocol gives $\Delta_{\mathrm{in}} = 0.001$ against $1.269$ with
  independent starts, and the separation condition appears to hold in 90 % of draws instead of
  17 %. Both protocols are legitimate and consistent as $n \to \infty$, but the shared start
  creates an artificial coupling that masks slow mixing.
* **Fixed horizon rather than an adaptive stopping rule.** An earlier version stopped when
  $D_n/n$ had moved by less than $\delta$ over a window; that makes the horizon data-dependent and
  stops early precisely when the chains mix slowly, biasing $\hat\gamma$ downwards.
* **The $\alpha$ grid starts at 0.1.** Below $\alpha \approx 0.1$ the relaxation time of the
  kernels explodes ($\approx 5\cdot10^4$ at $\alpha = 0.01$ against $5.1$ at $\alpha = 0.1$), so
  $\hat\gamma_n$ at any reachable horizon says nothing about $\gamma$.
* **Resolution of the margin.** At $n = 1500$ the estimation noise on a single $\hat\gamma_n$ is of
  order 0.02–0.03, and taking a max over $K$ and a min over $K(K-1)/2$ estimates amplifies it. In
  `assumptions.ipynb` any cell whose median margin is below $\approx 0.05$ in absolute value should
  be read as undetermined rather than as a violation; the plug-in used here is much less noisy
  (§4), but the caveat still applies to the smallest margins.
* **Ties in the merge heights.** With $c_{\mathrm{sub}} \equiv 2$, $\delta \equiv 1$ every
  $d_{\mathrm{OM}}$ is an integer, so $\hat\gamma_n$ takes at most $2n+1$ values and ex æquo merge
  heights are common. The dendrogram is then genuinely non-unique and a cut falling exactly on a
  tie is arbitrary — deterministic and reproducible, but implementation-dependent. It only matters
  when a tie straddles the cut, hence only near the transition.

## 8. Designs that were tried and dropped

* **A cross product $N \times \lambda$** (several $N$, each with horizons
  $n = \lfloor \lambda \log N \rceil$ over a common $\lambda$ grid), whose point was to show the
  curves collapsing onto one when plotted against $\lambda$. Dropped in favour of the single path,
  which controls $N$ and $n$ directly. Consequence, stated in §5.1: the collapse is no longer
  tested. Note that the curves would not superpose exactly anyway — the bound
  $\exp(2\log N(1-\lambda/\lambda^\star))$ has a common threshold $\lambda^\star$ but an exponent
  proportional to $\log N$, so the transition *sharpens* with $N$ and the curves pivot around
  $\lambda^\star$. The toy model
  $\mathbb P(\text{success}) \approx \exp(-\frac12 e^{2\log N(1-\lambda/\lambda^\star)})$ even puts
  the crossing height at $e^{-1/2} \approx 0.61$ independently of $N$, which is the sharpest form
  of the prediction and remains untested.
* **A second, shorter horizon on the grid**, with a companion figure showing the mean ARI at both.
  Dropped because nested prefixes share no work (§3): the second horizon costs its own $n^2$, and
  what it showed — the ARI rising with $n$ — is already read along the path at eleven lengths.
* **$K = 2$ or $K = 3$ for the path.** At $K = 3$ an unselected Dirichlet draw is a lottery: over 9
  draws at $\alpha \in \{0.2,0.3,0.4\}$ the margin ranged from $-0.001$ to $+0.397$, and three
  never reached $\mathrm{ARI} = 0.99$ by $n = 800$. All 9 draws at $K = 2$ reached it, between
  $n = 32$ and $n = 504$ — too easy once the design moved to $n \ge 100$ with large $N$.

## 9. Prior run (smaller, completed) — indicative results

A completed run of the same code with **different sizes** gives the shape of the answer. Do not
quote these numbers for the final version; they come from: path $N = 100 \to 1000$,
$n = 100 \to 1000$ (10 points, all hundreds), $R = 8$; grid $N = 800$, horizons $(400, 800)$,
$R = 15$; same mixture $K = 4$, $\alpha = 0.3$.

Path (mean over 8 replicates):

| $N$ | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 | 1000 |
|---|---|---|---|---|---|---|---|---|---|---|
| $n$ | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 | 1000 |
| $\lambda$ | 21.7 | 37.7 | 52.6 | 66.8 | 80.5 | 93.8 | 106.9 | 119.7 | 132.3 | 144.8 |
| mean ARI | 0.259 | 0.582 | 0.623 | 0.723 | 0.755 | 0.857 | 0.863 | 0.902 | 1.000 | 1.000 |
| $\hat{\mathbb P}(\mathrm{ARI}=1)$ | 0 | 0 | 0 | 0 | 0.12 | 0.50 | 0.50 | 0.62 | 1.00 | 1.00 |
| median $\hat\eta$ | +0.055 | +0.076 | +0.082 | +0.088 | +0.091 | +0.094 | +0.095 | +0.096 | +0.097 | +0.098 |

Mean ARI crosses 0.5 at $\lambda = 33.7$ ($N \approx 168$, $n \approx 175$) and 0.99 at
$\lambda = 131$ ($N \approx 889$, $n \approx 890$). The mixture: spectral gaps
$(0.383, 0.293, 0.222, 0.232)$, relaxation times $(2.6, 3.4, 4.5, 4.3)$; at $N = 100$, $n = 800$ it
gives $\hat\Delta_{\mathrm{in}} = 0.730$, $\hat\Delta_{\mathrm{out}} = 0.822$, margin $+0.092$.

Note the two readings this already supports: the margin is **positive and essentially flat** along
the whole path (it is a property of the mixture, converging as $n$ grows), while the ARI climbs
from 0.26 to 1 — so the margin being positive is *not* what decides recovery at finite sample; the
sample sizes are. And exact recovery arrives much later than a high ARI, as the theorem's union
bound over $\binom{N}{2}$ pairs predicts.

Grid ranges at $n = 800$, $N = 800$, $R = 15$: mean ARI over $[0.00, 1.00]$,
$\hat{\mathbb P}(\mathrm{ARI}=1)$ over $[0.00, 1.00]$, $\hat{\mathbb P}(\hat\eta>0)$ over
$[0.00, 1.00]$, median margin over $[-0.43, +0.75]$. The margin therefore does go substantially
negative somewhere on the grid, which is what makes the "is Assumption 3 necessary" scatter
informative.

## 10. Code map

| object | role |
|---|---|
| `om_lib.cost_scheme` | builds $(S, \delta)$; `constant`, `random`, `trate` |
| `om_lib.check_assumption_metric` | conditions (i)–(vi) of Assumption 1, returns $M$ |
| `om_lib.sample_markov_model`, `sample_chain_order1`, `sample_mixture` | model and data |
| `om_lib.om_distance`, `om_matrices` | two-row DP, and the $N\times N$ matrices over a horizon grid |
| `om_lib.single_linkage_tree`, `cut_at_k`, `cut_at_threshold`, `largest_gap_k` | the tree and the three ways of cutting it |
| `om_lib.adjusted_rand_index`, `gamma_block_means`, `separation_levels` | the measurements |
| `om_lib.spectral_gap`, `stationary_distribution_markov` | mixture diagnostics |
| `recovery.ipynb: make_path` | the $(N,n)$ design |
| `recovery.ipynb: EVAL_KEYS, evaluate_matrix / evaluate_paths` | both ARIs, $\hat K$ and the separation levels, from a matrix / along prefixes |
| `recovery.ipynb: sweep_path` | the path, nested in both directions |
| `recovery.ipynb: sweep_ari` | the $(\alpha, K)$ grid |
| `recovery.ipynb: plot_design, plot_ari_path, plot_ari_grid, plot_k_hat_path, plot_k_hat_grid` | the figures |

Still open in the paper's outline and **not** covered here: the $K$-medoids experiment under a
marginal violation of the balance condition (Assumption 5). The largest-gap estimator of $K$ was
the other gap and is now covered, by §5.3.
