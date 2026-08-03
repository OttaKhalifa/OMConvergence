# Numerical illustrations — OM-based clustering for mixtures of Markov chains

Simulation code for the paper *Consistency of Optimal Matching-based Clustering for Mixtures of
Markov Chains* (`paper.txt`). The OM (edit) dissimilarity is defined by a substitution cost matrix
`S` = $(c_{\mathrm{sub}}(a,b))$ and a gap cost vector `delta` = $(\delta(a))$ on a finite alphabet
$\Sigma$, $|\Sigma| = d$, and $M := \max_{a,b} c_{\mathrm{sub}}(a,b)$.

## Layout

| file | role |
|---|---|
| `om_lib.py` | shared utilities: cost schemes, Assumption 1 checker, Markov sampling, OM dissimilarity, mixtures, single linkage, ARI, bounds, figure style |
| `om_convergence.ipynb` | experiment 1 — almost sure convergence of $\hat\gamma_n$ |
| `assumptions.ipynb` | experiment 2 — the assumptions bearing on $\gamma$ |
| `recovery.ipynb` | experiment 3 — ARI of single-linkage clustering, and whether Assumption 3 is necessary |
| `Figures/Convergence/`, `Figures/Assumptions/`, `Figures/Recovery/`, `Figures/Old/` | generated figures, one folder per experiment plus the superseded ones (`.pdf` for the paper, `.png` for viewing) |
| `paper.txt` | the LaTeX source of the paper (not modified by the code) |

Requires `numpy`, `scipy`, `matplotlib`, `numba`, `tqdm`. The notebooks import `om_lib` and use
`%autoreload`, so edits to the library take effect without restarting the kernel. Numba caches its
compiled kernels on disk (possible because the code lives in a `.py` file, not in a cell), so the
first run of a session pays ~10 s of compilation and later ones do not.

## `om_lib.py`

Cost schemes (`cost_scheme`) come in three flavours — `constant` ($c_{\mathrm{sub}} \equiv 2$,
$\delta \equiv 1$, the usual sequence-analysis default), `random` (symmetric
$c_{\mathrm{sub}} \sim \mathcal U[1.2, 2]$) and `trate` ($c_{\mathrm{sub}} = 2 - \hat P - \hat P^\top$).
`check_assumption_metric` verifies conditions (i)–(vi) of Assumption 1 numerically and returns $M$;
every scheme used below passes it.

The OM dissimilarity (`om_distance`) is a **two-row** dynamic program, $O(nm)$ time and $O(m)$
memory. The full $(n+1)\times(m+1)$ table would take 800 MB per pair at $n = 10^4$, which would rule
out running pairs in parallel. `gamma_hat_pairs` computes $\hat\gamma_n$ for a batch of pairs and
`gamma_hat_paths` computes whole sample paths $n \mapsto \hat\gamma_n$; both parallelise over the
batch with `prange`.

Also included: Dirichlet sampling of Markov models of arbitrary order, a vectorised order-1
trajectory sampler, stationary laws, the spectral gap, and the two bounds of the paper —
`wasserstein_lower_bound` (linear program) and `product_upper_bound` ($\pi_P^\top S \pi_Q$).
`PAPER_STYLE`, `SEQUENTIAL_CMAP` and `DIVERGING_CMAP` hold the shared figure style.

For the clustering experiments: `sample_mixture` draws $N$ sequences from a $K$-component mixture
together with the latent labels; `om_matrices` returns the whole $N \times N$ dissimilarity matrix
at each horizon of a grid of nested prefixes, parallelising over the $\binom{N}{2}$ pairs with all
horizons of a pair on one thread. `single_linkage_tree` is Kruskal's algorithm, which gives the
single-linkage merge heights $h_1 \le \cdots \le h_{N-1}$ and the merged pairs in one pass;
`cut_at_k`, `cut_at_threshold` and `largest_gap_k` are the three ways the paper cuts that tree.
`adjusted_rand_index` avoids a `scikit-learn` dependency. `gamma_block_means` and
`separation_levels` give the plug-in $\hat\Gamma$, $\hat\Delta_{\mathrm{in}}$,
$\hat\Delta_{\mathrm{out}}$ from a dissimilarity matrix and the true labels. The merge heights and
the ARI agree with `scipy.cluster.hierarchy.linkage` and
`sklearn.metrics.adjusted_rand_score` to machine precision.

One consequence of integer costs: with $c_{\mathrm{sub}} \equiv 2$, $\delta \equiv 1$ every
$d_{\mathrm{OM}}$ is an integer, so $\hat\gamma_n$ takes at most $2n+1$ values and ex æquo merge
heights are common. The dendrogram is then genuinely non-unique, and a cut falling exactly on a tie
is arbitrary — deterministic and reproducible, but implementation-dependent (this is the one point
where our cut and `sklearn`'s can disagree). It only matters when the tie straddles the cut, hence
only near the transition.

## Experiment 1 — convergence of $\hat\gamma_n$

Illustrates that $\hat\gamma_n = d_{\mathrm{OM}}(X_{1:n}, Y_{1:n})/n \to \gamma(P,Q)$ almost surely
with a limit independent of the initial laws, that $\gamma(P,P) > 0$, and that both computable
bounds hold.

$d = 5$, $P$ and $Q$ drawn row-wise from Dirichlet($0.25$) — **first draw, no selection**. Two
configurations, *within* ($X, Y$ independent with kernel $P$) and *between* ($X \sim P$, $Y \sim Q$),
$R = 30$ replicates each, $n$ up to $10^4$ on a 24-point logarithmic grid of **nested prefixes of
the same trajectories**, so each thin curve is a genuine sample path rather than a sequence of
independent draws. Every one of the 120 trajectories starts from its own initial law, the Dirac law
at a state drawn uniformly on $\Sigma$: the proposition allows arbitrary initial laws, Dirac laws
are the informative case since any initial law is a mixture of them, and the spread of the curves at
small $n$ therefore contains the transient due to the initial condition. Its disappearance is what
the second step of the proof asserts.

Results for the drawn pair: $\pi_P = (0.076, 0.035, 0.600, 0.152, 0.136)$ with spectral gap $0.476$,
$\pi_Q = (0.488, 0.020, 0.056, 0.106, 0.330)$ with spectral gap $0.558$. The limits are
$\gamma(P,P) \approx 0.54$ and $\gamma(P,Q) \approx 1.29$, stable well before $n = 10^4$. The bounds
are $\pi_P^\top S \pi_P = 1.182$, $W_{\bar d}(\pi_P, \pi_Q) = 1.211$ and
$\pi_P^\top S \pi_Q = 1.735$: both hold, the Wasserstein lower bound is much the tighter of the two,
and here $\pi_P^\top S \pi_P < W_{\bar d}(\pi_P, \pi_Q)$, so the two computable bounds alone already
separate the within- and between-pair limits without any simulation.

The three cost schemes are applied to the **same** 120 trajectories, so the three figures are
comparable term by term. Runtime ~1 min.

## Experiment 2 — assumptions on $\gamma$

With $\Gamma_{k\ell} = \gamma(\mathbb P_k, \mathbb P_\ell)$,
$\Delta_{\mathrm{in}} = \max_k \Gamma_{kk}$, $\Delta_{\mathrm{out}} = \min_{k \neq \ell} \Gamma_{k\ell}$
and $\Delta_{\mathrm{out}}^{\max} = \max_{k \neq \ell} \Gamma_{k\ell}$, four conditions are tested
one by one, each written as a signed margin $m > 0$ in OM cost units so that $|m| \le M = 2$ and the
four are on a common scale:

| assumption | margin $m$ | needed by |
|---|---|---|
| 3 separation | $\Delta_{\mathrm{out}} - \Delta_{\mathrm{in}}$ | single linkage |
| 4(a) strong separation | $\Delta_{\mathrm{out}} - 2\Delta_{\mathrm{in}}$ | largest-gap rule for $K$ |
| 4(b) strong separation | $2\Delta_{\mathrm{out}} - \Delta_{\mathrm{out}}^{\max} - \Delta_{\mathrm{in}}$ | largest-gap rule for $K$ |
| 5 balance | $\Delta_{\mathrm{out}} - (1 + w_{\max}/w_{\min})\Delta_{\mathrm{in}}$ | $K$-medoids |

$K$ components on $d = 5$ states, rows drawn from Dirichlet($\alpha$), new chains at every
repetition; each $\Gamma_{k\ell}$ estimated by $\hat\gamma_n$ at a fixed $n = 1500$ from two
independent realisations per component. Grid $\alpha \in \{0.1, 0.2, 0.3, 0.4, 0.5, 1, 5, 10\}$,
$K \in \{2,\dots,10\}$, $R = 40$, $w_{\max}/w_{\min} = 2$. Each cell reports both the proportion of
repetitions with $m > 0$ and the median of $m$ — the probability alone saturates at $0$ over much of
the grid, where it can no longer tell a condition missed by a hair from one that is hopeless.

What comes out:

- **Separation is the only practically satisfiable condition.** Probability $\ge 0.9$ for $K \le 4$
  over $\alpha \in [0.2, 1]$, still $0.5$–$0.6$ at $K = 10$.
- **4(a) and 5 are essentially confined to $K = 2$.** 4(a) reaches $0.78$ at $(\alpha, K) = (0.1, 2)$
  and is zero for $K \ge 4$; the balance condition at $w_{\max}/w_{\min} = 2$ peaks at $0.50$ in that
  same cell and is zero almost everywhere else. The conditions needed to estimate $K$ and to run
  $K$-medoids are far more demanding than the one needed for single linkage.
- **The decay in $K$ is structural.** $\Delta_{\mathrm{in}}$ is a max over $K$ terms and
  $\Delta_{\mathrm{out}}$ a min over $K(K-1)/2$: one rises and the other falls mechanically as $K$
  grows, whatever the quality of the components.
- **The collapse at $\alpha \ge 5$ is a degeneracy, not a large violation.** All margins there are
  within $0.03$ of zero, i.e. $1.5\%$ of $M$: the chains become indistinguishable and
  $\Delta_{\mathrm{in}}, \Delta_{\mathrm{out}}$ converge to a common value. The clustering problem
  itself becomes ill-posed rather than the assumption being badly violated — a distinction the
  probability heatmap cannot show and the margin does.
- **The best regime is not the most separated one.** At $K = 5$ the median margin decreases
  monotonically with $\alpha$ ($+0.11$ at $\alpha = 0.1$ down to $-0.02$ at $\alpha = 10$), yet the
  probability *peaks* at $\alpha \approx 0.4$ ($0.98$, against $0.68$ at $\alpha = 0.1$). Same median
  margin, very different failure rates: at small $\alpha$ the margin distribution has a heavy left
  tail. Moderate randomness is more reliable than maximal distinguishability.
- **The $K = 2$ column of 4(b) is degenerate**: with a single between-pair,
  $\Delta_{\mathrm{out}}^{\max} = \Delta_{\mathrm{out}}$ and the condition reduces exactly to
  separation. It duplicates the separation panel digit for digit and should be greyed out.

Runtime scales as $n^2 \times R \times \sum_K (K + K(K-1)/2)$; a few minutes for the grid above.

## Experiment 3 — ARI of single-linkage clustering

Experiment 2 measured how often the separation condition holds; this one asks the converse
question, **how much of the recovery survives where it fails**. The conditions of the paper are
sufficient, and the point of the experiment is to locate the boundary of what they cover from
below. Nothing is conditioned on an assumption being satisfied: the mixture of the first figure is a
first draw with no selection, and the $(\alpha, K)$ grid is kept in full.

$K$ components on $d = 5$ states, rows drawn from Dirichlet($\alpha$), balanced weights
$w_k = 1/K$ (single linkage needs no balance condition), $N$ sequences of length $n$ with i.i.d.
latent labels, constant costs. Each sequence starts from its own initial law, a Dirac law at a state
drawn uniformly on $\Sigma$, as in the other two experiments.

**Estimator.** Single linkage cut at $K$ blocks, $K$ known. Theorem "consistency of single-linkage
clustering" is stated for the cut at a fixed level $t \in (\Delta_{\mathrm{in}},
\Delta_{\mathrm{out}})$, but its proof shows that on the event
$\mathcal E_{N,n}(\varepsilon)$ the two cuts coincide, and cutting at $K$ needs no oracle threshold.
When a class comes out empty the cut is taken at the number of non-empty classes, which is what
$P^\star$ counts. Two quantities are reported: the ARI, which degrades gracefully and therefore
distinguishes "two clusters merged" from "partition unrelated to the truth", and the exact-recovery
event $\{\mathrm{ARI} = 1\}$, which is what the theorem bounds.

**The margin comes for free.** Once the $N \times N$ matrix is computed, the true labels give
$\hat\Gamma_{k\ell}$ = average of $\hat\gamma_n(i,j)$ over the pairs of the block $(k,\ell)$, hence
$\hat\Delta_{\mathrm{in}}$, $\hat\Delta_{\mathrm{out}}$ and the margin
$\hat\eta = \hat\Delta_{\mathrm{out}} - \hat\Delta_{\mathrm{in}}$ **on the same data as the ARI**.
Averaging over $|C_k|(|C_k|-1)/2$ or $|C_k||C_\ell|$ pairs makes this much less noisy than the
one-pair-per-block estimate of experiment 2, so its numbers are not expected to match that notebook
digit for digit.

Two figures:

1. **One axis for both $n$ and $N$.** The exact-recovery bound
   $N^2\exp(-\varepsilon^2 n/(2C^\star))$ rewrites as
   $\exp\bigl(2\log N\,(1 - \lambda/\lambda^\star)\bigr)$ with $\lambda := n/\log N$ and
   $\lambda^\star := 4C^\star/\varepsilon^2$: the two sample sizes enter **only through the ratio
   $\lambda$**, the remaining $\log N$ sharpening the transition without moving it. The figure has
   two panels — mean ARI against $n$, one curve per $N$, and the *same* points against $n/\log N$ —
   so the collapse is shown rather than asserted.

   Two choices make the test clean. The **mixture is fixed** ($K = 2$, $\alpha = 0.3$, kernels drawn
   once, first draw, no selection) and only the labels and trajectories are redrawn, so everything
   the bound depends on besides $n$ and $N$ is held constant and $\lambda$ is the only free
   combination left. And the **grid is in $\lambda$, not in $n$**: for each $N$ the horizons are
   $n = \lfloor\lambda\log N\rceil$ over a common $\lambda \in [3, 70]$ grid, so all the $N$ land on
   the same abscissae in the right panel by construction, and the budget adapts on its own — short
   sequences for small $N$, where the transition happens earlier. Horizons within one $N$ are again
   nested prefixes; the band is the 10th-to-90th percentile over replicates, whose number decreases
   with $N$ ($R = \max(8, 1600/N)$) since the ARI of a large sample already averages over many pairs.

   $N \in \{10, 25, 60, 150, 400\}$, a factor $2.6$ in $\log N$, which is the lever of the
   rescaling. Printed alongside: the $n$ and the $\lambda$ at which the mean ARI crosses $0.5$ for
   each $N$, and the coefficient of variation of each across $N$ — the rescaling is the right one to
   the extent that the second is the smaller. Runtime ~2 min, of which 108 s is $N = 400$ alone;
   adding $N = 1000$ widens the lever to $3.0$ and costs ~13 min more on its own.

   **Why $K = 2$ here.** Measuring a threshold needs the threshold inside the observation window,
   and at $K = 3$ an unselected Dirichlet draw is a lottery: over 9 draws at
   $\alpha \in \{0.2, 0.3, 0.4\}$ the margin $\hat\eta$ ranged from $-0.001$ to $+0.397$ and three
   never reached $\mathrm{ARI} = 0.99$ by $n = 800$, which puts $n^\star$ outside any affordable grid
   since the cost is $O(N^2n^2)$. All 9 draws at $K = 2$ reached it, between $n = 32$ and $n = 504$.
   $K = 2$ is also the minimal instance of the theorem, and the scaling this figure is about comes
   from the union bound over the $\binom{N}{2}$ pairs, which does not involve $K$ — the dependence on
   $K$ is what the grid below is for. The draw actually used has $\hat\Delta_{\mathrm{in}} = 0.729$,
   $\hat\Delta_{\mathrm{out}} = 1.030$, margin $+0.301$ at $n = 800$, and the mean ARI crosses $0.5$
   around $\lambda \approx 25$ — hence the upper end of the $\lambda$ grid, chosen to leave the
   plateau at $\mathrm{ARI} = 1$ visible.
2. **The $(\alpha, K)$ grid.** Same grid as experiment 2 — $\alpha \in \{0.1, \dots, 10\}$,
   $K \in \{2,\dots,10\}$, new chains at every repetition — with $N = 60$, $R = 15$ and two horizons
   $n \in \{150, 600\}$ evaluated on nested prefixes. Mean ARI and $\mathbb P(\mathrm{ARI} = 1)$ per cell,
   each annotated with the median margin measured on the same repetitions, so the heatmaps can be
   read cell against cell versus the Assumption 3 panel of experiment 2. A companion scatter puts
   the mean ARI against the median margin, one marker per cell, coloured by $K$ on one panel and by
   $\alpha$ on the other: the top-left quadrant — negative margin, ARI near 1 — is the region the
   theory does not cover and the algorithm handles anyway. Runtime ~6 min.

The horizon is smaller here than the $n = 1500$ of experiment 2: a full matrix costs
$\binom{N}{2}$ dissimilarities instead of $K + K(K-1)/2$, so the budget goes into $N$ rather than
into $n$. The bottom rows of the heatmaps ($\alpha \le 0.2$, where the relaxation time is largest)
are therefore finite-$n$ statements even more than they were there. Runtime scales as
$N^2 n^2 R$ per cell, independently of $K$.

*Results not yet filled in — the notebook has not been run.*

## Methodological decisions that changed the results

These are the choices that make the current numbers differ from earlier versions of this code, and
they are worth a sentence in the paper.

**Costs are fixed in advance.** The theory requires a deterministic cost scheme. TRATE costs are
data-driven, so they are used only in the appendix of experiment 1 and estimated there on a **pilot
sample independent** of the sequences entering $\hat\gamma_n$; experiment 2 uses the constant scheme
throughout, since per-repetition estimated costs would make the cost scheme random.

**Every trajectory has its own initial law.** An earlier version started both realisations of a
component from the model's own `init_probs`, drawn once per model from Dirichlet($\alpha$) — a
near-Dirac for small $\alpha$, hence the *same* starting state. With near-deterministic kernels both
realisations then follow the same trajectory and $\Gamma_{kk} \approx 0$ mechanically: at
$\alpha = 10^{-4}$ that protocol gives $\Delta_{\mathrm{in}} = 0.001$ against $1.269$ with
independent starts, and the separation condition appears to hold in 90% of draws instead of 17%.
Both protocols are legitimate under the theory and consistent as $n \to \infty$, but the shared
start creates an artificial coupling that masks slow mixing. The two agree from $\alpha \ge 0.1$
onwards, which is why the grid starts there.

**Fixed horizon rather than an adaptive stopping rule.** An earlier version stopped when $D_n/n$ had
moved by less than $\delta$ over a window; that makes the horizon data-dependent and stops early
precisely when the chains mix slowly, biasing $\hat\gamma$ downwards.

**The $\alpha$ grid avoids the slow-mixing regime.** Below $\alpha \approx 0.1$ the relaxation time
of the kernels explodes ($\approx 5 \cdot 10^4$ at $\alpha = 0.01$, against $5.1$ at $\alpha = 0.1$),
so $\hat\gamma_n$ at any reachable horizon says nothing about $\gamma$. `assumptions.ipynb` ends with
a diagnostic cell tabulating $1 - |\lambda_2|$, the relaxation time and $n / \text{relaxation}$ per
$\alpha$.

**Resolution of the margin.** At $n = 1500$ the estimation noise on a single $\hat\gamma_n$ is of
order $0.02$–$0.03$, and taking a max over $K$ and a min over $K(K-1)/2$ estimates amplifies it. Any
cell whose median margin is below $\approx 0.05$ in absolute value should be read as undetermined
rather than as a violation. Raising $n$ to $4000$ roughly halves the noise at $\approx 7$ times the
compute; measured on the same trajectories, the residual bias of $\hat\gamma_n$ at $\alpha = 0.1$ is
$+0.025$ at $n = 1500$ and $+0.001$ at $n = 4000$, and the within/between ordering never flipped
between $n = 1500$ and $n = 8000$ over 25 draws.

## Figures

| file | content |
|---|---|
| `Convergence/gamma_convergence_constant.pdf` | experiment 1, constant costs — main figure |
| `Convergence/gamma_convergence_trate.pdf`, `Convergence/gamma_convergence_random.pdf` | experiment 1, appendix cost schemes |
| `Assumptions/assumptions_gamma_all.pdf` | the four conditions, colour = probability the condition holds |
| `Assumptions/assumptions_gamma_margins.pdf` | the four conditions, colour = median signed margin (diverging, pivot at 0) |
| `Assumptions/assumption_<name>.pdf` | one panel per condition, probability in colour, probability and margin annotated per cell |
| `Assumptions/assumptions_gamma_levels.pdf` | median $\Delta_{\mathrm{in}}$ and $\Delta_{\mathrm{out}}$ separately |
| `Recovery/ari_collapse_single_linkage.pdf` | experiment 3, mean ARI against $n$ and against $n/\log N$, one curve per $N$ |
| `Recovery/ari_grid_single_linkage.pdf` | experiment 3, mean ARI and $\mathbb P(\mathrm{ARI}=1)$ on the $(\alpha,K)$ grid, annotated with the median margin |
| `Recovery/ari_vs_separation.pdf` | experiment 3, mean ARI against the median separation margin, one marker per cell |
| `Recovery/ari_grid_horizons.pdf` | experiment 3, mean ARI at the two horizons side by side |

Sequential single-hue ramps for probabilities, a diverging ramp with a neutral midpoint for signed
margins, no dashed gridlines, legends outside the axes when the data fills the panel.

## Legacy

`Figures/Old/` holds `Separation assumption heatmap.png` and `Balancing assumption heatmap.png`,
which come from the earlier protocol (shared initial laws, adaptive horizon, $\alpha \le 1$,
viridis) and are kept for reference only; they are superseded by `Figures/Assumptions/`.
`gamma_vs_spectral_gap(1).ipynb` and `strong_assumptions_bounds.ipynb` are earlier explorations, not
maintained against `om_lib`. `om_convergence_and_assumptions.ipynb` was split into the two current
notebooks and removed. `ari_convergence.ipynb` is superseded by `recovery.ipynb`: it plotted ARI
against $n$ with one curve per $N$ and against $N$ with one curve per $n$, redrawing the chains at
every repetition (so the curves were not sample paths), with its own inlined copy of the sampling
and OM code and a `scikit-learn`/`joblib` dependency.

## Still to do

**$K$-medoids under a marginal violation of the balance condition**, the last of the four
experiments in the paper's outline. The natural design: fix a mixture, vary $w_{\max}/w_{\min}$ so
that $A = w_{\min}(\Delta_{\mathrm{out}}-\Delta_{\mathrm{in}}) - w_{\max}\Delta_{\mathrm{in}}$
crosses zero, and plot the ARI against $A$, with single linkage — which needs no balance condition —
as the control. Since the theorem is about the *global* minimiser of $\Phi_{N,n}$, the medoid set
should be found by enumerating the $\binom{N}{K}$ candidates whenever that is affordable
(trivial at $N = 40$, $K = 2$), otherwise a PAM heuristic measures the failure of the heuristic
rather than the violation of the assumption.

Also open: the largest-gap estimator of $K$ (`largest_gap_k` exists but no experiment uses it), and
the section "Numerical Illustrations" of `paper.txt`, still a placeholder — its item list has a
stray duplicated `\item` at the third bullet.

One extension of the rescaling of experiment 3, left aside for now. The curves for different $N$ are
not expected to coincide exactly: the bound $\exp(2\log N(1-\lambda/\lambda^\star))$ has a common
threshold $\lambda^\star$ but an exponent proportional to $\log N$, so the transition **sharpens**
with $N$ and the curves pivot around $\lambda^\star$ rather than superposing. The toy model
$\mathbb P(\text{success}) \approx \exp(-\tfrac12 e^{2\log N(1-\lambda/\lambda^\star)})$ even puts the
crossing height at $e^{-1/2} \approx 0.61$, independently of $N$; testing that is a natural
refinement of the right-hand panel. Going further, $\lambda^\star = 4C^\star/\varepsilon^2$ with
$\varepsilon = \eta/2$ and $C^\star = C_{\mathrm{Pau}}M^2\tau_{\mathrm{mix}}$, so
$n\eta^2/(M^2\tau_{\mathrm{mix}}\log N)$ would put every $(N, \alpha, K)$ on one master curve whose
threshold is the universal constant $16\,C_{\mathrm{Pau}}$ — but that measures how tight Paulin's
inequality is, not whether the assumptions are needed, which is what experiment 3 is about.
