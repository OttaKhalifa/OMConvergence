"""Assert bit identity between `om`'s univariate OM path and the TraMineR reference.

    python validation/compare_om.py

Reads the same cases.json + sequences.csv that `tramineR_reference.R` read, replays
`om` on them, and compares against reference.json. Every comparison is
`np.array_equal` -- bit-identical, not "close". Nothing here is asserted by tolerance.
Exits non-zero on the first failing case.

Two layers are checked separately, because they fail separately:

  costs      compute_trate_subst_matrix vs seqcost(method="TRATE", cval=2).
             Only the two `trate` schemes have anything to check: `constant` and
             `random` ship their matrix to R in cases.json, so agreeing on it would
             prove nothing.
  distance   om_distance vs seqdist(method="OM", norm="none"), on whichever cost
             matrix the scheme uses.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from om import compute_trate_subst_matrix, om_distance  # noqa: E402


def _load_sequences(cases: dict) -> dict[str, np.ndarray]:
    """The (N, n) integer arrays of every case, read back from sequences.csv."""
    arrays = {name: np.full((meta["n_sequences"], meta["seq_len"]), -1, dtype=np.int64)
              for name, meta in cases.items()}
    state_ids = {name: {state: i for i, state in enumerate(meta["alphabet"])}
                 for name, meta in cases.items()}
    with (HERE / "sequences.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            name = row["case"]
            arrays[name][int(row["seq"]), int(row["t"])] = state_ids[name][row["state"]]
    for name, array in arrays.items():
        if (array < 0).any():
            raise AssertionError(f"{name}: sequences.csv leaves cells unset")
    return arrays


def _distance_matrix(X: np.ndarray, S: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """The full symmetric d_OM matrix, the way `om_matrices` fills one."""
    N = X.shape[0]
    D = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(i + 1, N):
            D[i, j] = D[j, i] = om_distance(X[i], X[j], S, delta)
    return D


def main() -> int:
    cases = json.loads((HERE / "cases.json").read_text(encoding="utf-8"))
    reference = json.loads((HERE / "reference.json").read_text(encoding="utf-8"))
    sequences = _load_sequences(cases)
    failures = []

    print(f"{'case':14} {'scheme':14} {'costs':>9} {'indel':>9} {'distances':>10}")
    for name, meta in cases.items():
        X = sequences[name]
        d = meta["n_states"]
        for scheme, spec in meta["schemes"].items():
            expected = reference[name][scheme]
            expected_sub = np.asarray(expected["sub"], dtype=np.float64)
            expected_indel = float(np.asarray(expected["indel"]).ravel()[0])

            if spec["source"] == "TRATE":
                S = compute_trate_subst_matrix(X, n_states=d)
                costs_ok = np.array_equal(S, expected_sub)
            else:
                S = np.asarray(spec["sub"], dtype=np.float64)
                costs_ok = None  # shipped to R in cases.json, nothing to compare

            indel = 0.5 * np.max(S) if spec["indel"] == "half_max" else float(spec["indel"])
            indel_ok = indel == expected_indel

            # Compare the distances on OUR costs. Where they are also ours to get
            # right, `costs_ok` above has already said whether they are.
            D = _distance_matrix(X, S, np.full(d, indel))
            dist_ok = np.array_equal(D, np.asarray(expected["dist"], dtype=np.float64))

            checks = {"costs": costs_ok, "indel": indel_ok, "distances": dist_ok}
            failures.extend(f"{name}/{scheme}: {label}"
                            for label, ok in checks.items() if ok is False)
            marks = ["-" if ok is None else "EXACT" if ok else "FAIL"
                     for ok in checks.values()]
            print(f"{name:14} {scheme:14} {marks[0]:>9} {marks[1]:>9} {marks[2]:>10}")

    if failures:
        raise AssertionError("\n".join(failures))
    n_schemes = sum(len(meta["schemes"]) for meta in cases.values())
    print(f"\nOK - {len(cases)} cases x 4 cost schemes = {n_schemes} comparisons, "
          "bit-identical:\n     TRATE substitution costs, indel, and raw OM distances.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
