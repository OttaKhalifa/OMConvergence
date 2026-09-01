# Superseded experiments

These two notebooks produced Figures 2, 3a, 3b, 4, 5, 6, 9 and 10 of the current draft.
They are kept here only until the refactored experiments replace them, and should not be
run: both rest on protocols the refactor rules out.

`assumptions.ipynb` — the (alpha, K) separation heatmaps. Estimates each `Gamma_kl` from
**two** realisations at a single horizon n = 1500 and treats the result as the asymptotic
`Gamma`, with no uncertainty attached to the sign of the margin. Also computes the two
"strong separation" conditions, which belong to a largest-gap theorem the paper no longer
states.

`recovery.ipynb` — the ARI paths and the (alpha, K) recovery grids. Redraws kernels
independently for each algorithm, so the comparison between linkage and K-medoids is
unpaired; reads `Gamma` off the clustering dataset itself; reports ARI rather than exact
recovery; and selects K by the largest-gap rule.

The replacements are the four notebooks of figure blocks A to F. Once those land, delete
this directory — git history keeps everything.
