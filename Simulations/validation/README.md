# The OM dissimilarity of `om`, against TraMineR

`om.om_distance` and `om.compute_trate_subst_matrix` are checked against
**TraMineR 2.2.12**, the reference implementation of optimal matching for sequence
analysis, on the same sequences and the same cost schemes.

## Run

```bash
validation/run.sh              # one draw per case, about 30 s
validation/run.sh --seeds 10   # the ten draws the claim below is made on, about 3 min
```

Needs R with `TraMineR` and `jsonlite`, and a python with `numpy` and `numba`.
`compare_om.py` exits non-zero if any comparison fails.

`make_sequences.py` writes `cases.json` (the contract: shape, alphabet, cost schemes)
and `sequences.csv` (long format, one row per cell). The R script reads those two and
writes `reference.json`. `compare_om.py` replays `om` on the same files and
compares. Both sides reading the *same* file is the point: a mismatch can only be an
algorithm difference, never a data difference. The three generated files are
deterministic and not tracked by git — wipe them, re-run, byte-identical output.

## The claim

> On 80 draws (eight cases × ten seeds) covering the alphabet sizes, horizons and
> mixtures of the experiments, `om` reproduces TraMineR 2.2.12's
> `seqdist(method="OM", sm, indel, norm="none")` **bit for bit**, on each of the four
> cost schemes, and its TRATE substitution matrix reproduces
> `seqcost(method="TRATE", cval=2)` bit for bit. 320 comparisons, all
> `np.array_equal` — bit-identical, not "close". Nothing in the checker is asserted
> by tolerance.

Scope: one distance (OM), the four cost schemes below, alphabets `d ≤ 15`, horizons
`n ≤ 2000`, complete sequences of equal length. Not covered: other TraMineR versions
(`seqcost` changed between releases), weighted sequences, missing values, other
machines or a numba-free run.

## What is compared

Two layers, checked separately because they fail separately.

| | ours | TraMineR |
|---|---|---|
| substitution | `compute_trate_subst_matrix` | `seqcost(method="TRATE", cval=2)` |
| alignment | `om_distance` | `seqdist(method="OM", norm="none")` |

The paper's normalisation is `d_OM / n`, applied by `gamma_hat_pairs` and
`om_matrices` after the fact. At a fixed horizon it is the same division on both
sides, so there is nothing for TraMineR to say about it and `norm="none"` is what is
compared.

### Cost schemes

`constant` and `random` ship their matrix to R inside `cases.json` at full precision —
R cannot redraw a numpy uniform, and agreeing on a matrix we handed over would prove
nothing, so only the alignment layer is under test there. The two `trate` schemes
estimate the matrix independently on each side, which is what makes the cost layer a
real comparison.

`trate_halfmax` uses TraMineR's own `indel = 0.5 · max(S)` instead of the paper's
`δ ≡ 1`. It is there for one reason: **1 is a power of two**, so a border accumulated
cell by cell and a border multiplied agree on it no matter what the implementation
does. Every measured difference in the table below comes from the schemes where the
gap cost is not dyadic.

### Cases

| case | N | n | d | what it is for |
|---|---|---|---|---|
| `mix_d5_k4` | 40 | 120 | 5 | the recovery experiment of §5.3–5.4, in miniature |
| `mix_d5_k2` | 40 | 200 | 5 | the pair setting of the convergence experiment |
| `mix_d2_k2` | 40 | 150 | 2 | the smallest alphabet |
| `mix_d15_k3` | 40 | 100 | 15 | a large alphabet, so TRATE reads sparse counts |
| `absorbing` | 40 | 120 | 4 | a state that is never the source of a transition |
| `shared_runs` | 40 | 140 | 5 | a long common prefix and suffix |
| `long_n500` | 30 | 500 | 5 | enough cells for the accumulation path to matter |
| `long_n2000` | 12 | 2000 | 5 | the order of the horizons §5.1 runs to |

`N` is deliberately small. Equivalence is decided pair by pair — the dynamic program
never sees more than two sequences — so `N` only multiplies the number of
comparisons. What changes the computation is `n` and `d`.

## What it took

Four things, none of them visible in the recurrence itself. The count after each is
how many of the 320 comparisons break when it is put back the way it was, measured by
doing exactly that and re-running the check.

* **No smoothing** in `compute_trate_subst_matrix`; the old value was `1e-8`.
  **320/320.** It put `P` about 1e-10 away from `seqtrate`, so nothing at all agreed
  at the bit.
* **Empty rows stay zero.** A state that is never the source of a transition keeps a
  row of zeros, as `seqtrate` leaves it; the smoothing made it uniform instead — and
  on an empty row `(0 + ε)/(0 + εd) = 1/d` for *any* ε, so a parameter advertised as a
  numerical guard was in fact choosing the row. This is the one difference that was
  not an ulp, and `absorbing` is the case built to see it.
* **Mirror the upper triangle.** `seqcost` evaluates `2 - P[a,b] - P[b,a]` for `a < b`
  and copies the cell down; recomputing `(b,a)` gives `(2-β)-α` against its `(2-α)-β`,
  one ulp apart. **186/320.**
* **Multiplied border.** `dp[i][0] = i · δ` rounds once where `i` accumulated additions
  round `i` times. **25/320.**
* **Common prefix and suffix skipped** before the DP, so its border restarts inside
  the differing zone. `OMdistance.cpp` does it for speed, and it moves the
  accumulation path. **19/320**, seven of them on `shared_runs`.

## What this check does *not* pin down

Stated because a check that is silent on something should say so rather than let the
"bit for bit" above be read as covering it.

* **The suffix skip on its own: 0/320.** Removing it while keeping the prefix skip
  changes nothing on any comparison. Unlike the prefix, a common suffix does not move
  the border — the DP has already accumulated its value by the time it reaches the
  matched tail, which then adds exactly zero. It is kept because TraMineR does it and
  because that is where the speed is, not because this check defends it.
* **`fastmath=True`: 0/320.** Turning it back on changes nothing here. It is off in
  `om_distance` because it lets the compiler reassociate the additions, which is
  precisely what bit-identity forbids; earlier measurement on this code put its
  benefit at about 4% of runtime and caught a cold-cache run differing from the runs
  after it by 5e-07. A warm, single-process check like this one cannot reproduce that,
  so the flag rests on the argument, not on these 320 comparisons.
