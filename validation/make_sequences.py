"""Test sequences shared by TraMineR and by `om`.

    python validation/make_sequences.py [--seeds K]

Writes TWO files, read as-is by `tramineR_reference.R` and by `compare_om.py`:

  cases.json     per-case metadata -- shape, alphabet, and the cost schemes to run.
                 This is the CONTRACT. Deriving the alphabet from the data would
                 make it depend on what the draw happened to contain, and a state
                 missing from a draw would then vanish from both sides at once,
                 invisible to the comparison.
  sequences.csv  long format `case,seq,t,state`, one row per cell.

Both implementations reading the SAME file is the point of the protocol: a mismatch
can then only be an algorithm difference, never a data difference. Both files are
deterministic -- wipe them, re-run, byte-identical output.

States are written `s00`, `s01`, ...: `seqdef` SORTS the alphabet, and unpadded labels
would put `s10` before `s2`, permuting the substitution matrix against our 0..d-1
indices.

The draws come from `generators.sample_mixture`, the generator of the experiments
themselves, so the empirical transition rates that TRATE reads have the structure of
a mixture of Markov chains rather than uniform noise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from generators import sample_mixture  # noqa: E402
from om import cost_scheme  # noqa: E402


def case_mixture(N, n, d, K, alpha, seed):
    """N sequences of length n from a K-component mixture of Dirichlet(alpha) chains."""
    rng = np.random.default_rng(seed)
    return sample_mixture(K, N, n, d, alpha, rng)["X"]


def case_absorbing(N=40, n=120, d=4, K=2, alpha=0.3, seed=404):
    """Edge case: state d-1 occupies the last position and appears nowhere else.

    It is therefore never the source of a transition. `seqtrate` zeroes that whole row
    (its `PA == 0` branch); the smoothing `compute_trate_subst_matrix` used to apply
    made it uniform at 1/d instead. This is the one place where the two implementations
    could differ by something other than an ulp, so it is measured, not assumed.
    """
    X = case_mixture(N, n, d - 1, K, alpha, seed)
    X[:, -1] = d - 1
    return X


def case_shared_runs(N=40, n=140, d=5, K=2, alpha=0.3, seed=808, head=30, tail=20):
    """Every sequence starts on the same 30 states and ends on the same 20.

    The dynamic program skips the common prefix and the common suffix before it runs.
    Between two independent chains those runs are one or two positions long, so on the
    other cases the shortcut is almost never the thing being tested; here it covers a
    third of the alignment. Measured, by removing the skip and re-running the check: it
    breaks 19 of the 320 comparisons, 7 of them on this case -- more than any other.
    """
    X = case_mixture(N, n, d, K, alpha, seed)
    rng = np.random.default_rng(seed + 1)
    X[:, :head] = rng.integers(0, d, head)
    X[:, n - tail:] = rng.integers(0, d, tail)
    return X


# Each entry is a function of the replicate index `k`, so the whole grid can be redrawn
# on fresh seeds: bit identity on one draw could be a coincidence of that draw, on ten
# it is a property of the implementations.
CASES = {
    "mix_d5_k4":   (lambda k: case_mixture(40, 120, 5, 4, 0.3, 101 + 1000 * k), 5,
                    "d=5, K=4: the recovery experiment of 5.3-5.4, in miniature"),
    "mix_d5_k2":   (lambda k: case_mixture(40, 200, 5, 2, 0.25, 202 + 1000 * k), 5,
                    "d=5, K=2, alpha=0.25: the pair setting of the convergence experiment"),
    "mix_d2_k2":   (lambda k: case_mixture(40, 150, 2, 2, 0.5, 303 + 1000 * k), 2,
                    "d=2: the smallest alphabet, where every pair of states is adjacent"),
    "mix_d15_k3":  (lambda k: case_mixture(40, 100, 15, 3, 0.2, 505 + 1000 * k), 15,
                    "d=15: a large alphabet, so TRATE reads sparse transition counts"),
    "absorbing":   (lambda k: case_absorbing(seed=404 + 1000 * k), 4,
                    "a state that is never the source of a transition: seqtrate zeroes its row"),
    "shared_runs": (lambda k: case_shared_runs(seed=808 + 1000 * k), 5,
                    "a long common prefix and suffix: what the DP skips before it runs"),
    # --- the horizons, where floating-point accumulation in the DP is the whole worry ---
    # N is deliberately small: equivalence is decided PAIR BY PAIR (the dynamic program
    # never sees more than two sequences), so N only multiplies the number of
    # comparisons. What changes the computation is n.
    "long_n500":   (lambda k: case_mixture(30, 500, 5, 2, 0.3, 606 + 1000 * k), 5,
                    "n=500: enough cells for the accumulation path to matter"),
    "long_n2000":  (lambda k: case_mixture(12, 2000, 5, 2, 0.3, 707 + 1000 * k), 5,
                    "n=2000: the order of the horizons the convergence experiment runs to"),
}


def schemes_for(d, seed):
    """The cost schemes each case is checked on, as `tramineR_reference.R` reads them.

    `source` says who builds the substitution matrix. "given" ships it in this file, at
    full precision, because R cannot redraw a numpy uniform; "TRATE" means each side
    estimates it from the sequences with its own code, which is what makes the cost
    layer a genuine comparison rather than a shared input.

    `indel` is 1.0 in the three schemes the paper uses. `half_max` adds the one thing
    those three cannot test: delta = 1 is a power of two, so a border accumulated cell
    by cell and a border multiplied agree on it whatever the implementation does.
    0.5 * max(S) is not, and it is TraMineR's own default.
    """
    rng = np.random.default_rng(seed)
    S_const, _ = cost_scheme("constant", d, sub=2.0, indel=1.0)
    S_rand, _ = cost_scheme("random", d, rng=rng, low=1.2, high=2.0)
    return {
        "constant":      {"source": "given", "sub": S_const.tolist(), "indel": 1.0},
        "random":        {"source": "given", "sub": S_rand.tolist(), "indel": 1.0},
        "trate":         {"source": "TRATE", "indel": 1.0},
        "trate_halfmax": {"source": "TRATE", "indel": "half_max"},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=1, metavar="K",
                    help="draw each case on K different seeds (default 1). "
                         "K > 1 suffixes the case names `_s0`.. `_s{K-1}`.")
    args = ap.parse_args(argv)

    meta, lines = {}, ["case,seq,t,state"]
    for base, (builder, d, note) in CASES.items():
        for k in range(args.seeds):
            name = base if args.seeds == 1 else f"{base}_s{k}"
            X = builder(k)
            N, n = X.shape
            assert X.max() < d, f"{name}: state outside [0, d)"
            labels = [f"s{c:02d}" for c in range(d)]
            lines += [f"{name},{i},{t},{labels[X[i, t]]}"
                      for i in range(N) for t in range(n)]
            meta[name] = {"n_sequences": int(N), "seq_len": int(n), "n_states": int(d),
                          "alphabet": labels, "note": note,
                          "schemes": schemes_for(d, 90_000 + 13 * k)}
            if args.seeds == 1 or k == 0:
                print(f"{name:14} N={N:3d} n={n:5d} d={d:2d} "
                      f"{'' if args.seeds == 1 else f'(x{args.seeds} seeds) '}{note}")

    (HERE / "cases.json").write_text(json.dumps(meta, indent=2) + "\n", "utf-8")
    (HERE / "sequences.csv").write_text("\n".join(lines) + "\n", "utf-8")
    print(f"\n-> cases.json, sequences.csv ({len(meta)} cases, {len(lines) - 1} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
