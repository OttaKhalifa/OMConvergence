"""Assert that `clustering.pam` is the algorithm Theorem 3.8 analyses.

    python validation/check_pam.py [--instances R]

Theorem 3.8 is not about a global minimiser of Phi. It is about the output of PAM, and
Lemma 3.7 -- "every cluster has exactly one medoid" -- is proved by exhibiting an improving
one-swap whenever a cluster holds two medoids or none. So the property the implementation
owes the theory is precisely:

    no medoid/non-medoid swap strictly decreases Phi.

Six checks, in the order in which a failure would matter:

  swap table    the O(K N^2) bookkeeping used to pick a swap must agree with Phi recomputed
                from scratch for every one of the K*N candidates. Everything else rests on
                it, and an error here would be silent: the search would simply take wrong
                swaps and still terminate.
  objective     Phi is (1/N) sum_i min_m D[i, m], the normalisation of Definition 3.7.
  stationarity  every returned medoid set is certified one-swap stationary. This is the
                theorem's hypothesis, and the run must reach it by exhaustion, not by
                hitting an iteration cap.
  optimum       on instances small enough to enumerate all C(N, K) medoid sets, how often
                PAM's Phi equals the global minimum. Reported, not asserted: the theory
                does not ask for the global optimum, and PAM is not expected to find it.
  alternating   the same rate of one-swap stationarity for the alternating "assign, then
                re-centre" heuristic that PAM is often confused with. Reported to show what
                the change buys; a low rate here is the reason the heuristic cannot stand in
                for PAM.
  edge cases    K = 1, K = N, N = 1, all-equal dissimilarities, and determinism.

Instances are of two kinds: random symmetric matrices, which stress ties and asymmetry of
the search, and real OM dissimilarity matrices from a mixture of Markov chains, which are
what the experiments actually feed it. Exits non-zero on the first failing assertion.
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from clustering import (_best_swap, _swap_table, exact_recovery, pam,  # noqa: E402
                        pam_certify_one_swap, pam_objective)
from generators import sample_mixture  # noqa: E402
from om import cost_scheme, om_matrices  # noqa: E402


# ---------------------------------------------------------------------------
# Instances
# ---------------------------------------------------------------------------


def random_matrix(N, rng, integer=False):
    """Symmetric, zero-diagonal, non-negative. Integer values to force ties."""
    A = rng.integers(0, 4, size=(N, N)) if integer else rng.uniform(0.0, 1.0, size=(N, N))
    D = ((A + A.T) / 2.0).astype(float)
    np.fill_diagonal(D, 0.0)
    return D


def om_matrix(N, n, K, rng):
    """A real normalised OM dissimilarity matrix, as the experiments produce them."""
    data = sample_mixture(K, N, n, 4, 0.3, rng)
    S, delta = cost_scheme("constant", 4)
    return om_matrices(data["X"], np.array([n], dtype=np.int64), S, delta)[0], data["labels"]


# ---------------------------------------------------------------------------
# References, deliberately written the slow and obvious way
# ---------------------------------------------------------------------------


def brute_phi(D, medoids):
    return float(D[:, list(medoids)].min(axis=1).sum())


def brute_swap_table(D, medoids):
    """N*Phi after replacing medoids[j] by h, recomputed from scratch for every (j, h)."""
    K, N = len(medoids), D.shape[0]
    table = np.empty((K, N))
    for j in range(K):
        for h in range(N):
            candidate = list(medoids)
            candidate[j] = h
            table[j, h] = brute_phi(D, candidate)
    return table


def brute_global_optimum(D, K):
    """min Phi over all C(N, K) medoid sets. Only for N small enough to enumerate."""
    N = D.shape[0]
    return min(brute_phi(D, m) for m in combinations(range(N), K)) / N


def alternating_kmedoids(D, K, rng, max_iter=100):
    """The heuristic PAM is often confused with: assign, then re-centre, until stable.

    Reproduced here only as a foil. Its fixed points are stable under *simultaneous*
    re-centring of every cluster, which is a different condition from one-swap
    stationarity, and Lemma 3.7 needs the latter.
    """
    N = D.shape[0]
    medoids = rng.choice(N, size=K, replace=False)
    for _ in range(max_iter):
        labels = np.argmin(D[:, medoids], axis=1)
        new = medoids.copy()
        for k in range(K):
            idx = np.flatnonzero(labels == k)
            if idx.size:
                new[k] = idx[np.argmin(D[np.ix_(idx, idx)].sum(axis=1))]
        if np.array_equal(np.sort(new), np.sort(medoids)):
            break
        medoids = new
    return medoids


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_swap_table(instances, report):
    instances = [(D, K) for D, K, _ in instances]
    worst = 0.0
    for D, K in instances:
        rng = np.random.default_rng(0)
        for _ in range(3):
            medoids = np.sort(rng.choice(D.shape[0], size=K, replace=False))
            table, current = _swap_table(D, medoids)
            reference = brute_swap_table(D, medoids)
            worst = max(worst, float(np.abs(table - reference).max()))
            assert abs(current - brute_phi(D, medoids)) < 1e-9, "current Phi disagrees"
            # the swap actually chosen must be the argmin of the reference table too
            j, h, _ = _best_swap(D, medoids)
            masked = reference.copy()
            masked[:, medoids] = np.inf
            assert abs(masked[j, h] - masked.min()) < 1e-9, "steepest swap is not the argmin"
    assert worst < 1e-9, f"swap table disagrees with brute force by {worst:.3e}"
    report("swap table", f"EXACT (max |diff| = {worst:.1e} over {len(instances)} instances)")


def check_objective(instances, report):
    instances = [(D, K) for D, K, _ in instances]
    worst = 0.0
    for D, K in instances:
        medoids = np.arange(K)
        worst = max(worst, abs(pam_objective(D, medoids) - brute_phi(D, medoids) / D.shape[0]))
    assert worst < 1e-12, f"Phi disagrees with its definition by {worst:.3e}"
    report("objective", f"Phi = (1/N) sum_i min_m D[i,m]  (max |diff| = {worst:.1e})")


def check_stationarity(instances, report):
    instances = [(D, K) for D, K, _ in instances]
    caps = 0
    for D, K in instances:
        for n_restarts in (1, 5):
            rng = np.random.default_rng(1)
            result = pam(D, K, rng=rng, n_restarts=n_restarts)
            assert result.one_swap_certified, "PAM returned a non-stationary medoid set"
            assert pam_certify_one_swap(D, result.medoids), "certification disagrees with itself"
            caps += int(result.hit_cap)
    assert caps == 0, f"{caps} runs stopped on the iteration cap"
    report("stationarity", f"CERTIFIED on {2 * len(instances)} runs, none hit the swap cap")


def check_optimum(report, seed=0):
    """PAM's Phi against the global minimum, wherever enumeration is affordable."""
    rng = np.random.default_rng(seed)
    rows, hits1, hits5, total = [], 0, 0, 0
    for N, K in ((10, 2), (12, 3), (14, 3), (14, 4)):
        n_hit1 = n_hit5 = 0
        reps = 20
        for _ in range(reps):
            D = random_matrix(N, rng)
            best = brute_global_optimum(D, K)
            phi1 = pam(D, K).objective
            phi5 = pam(D, K, rng=np.random.default_rng(2), n_restarts=5).objective
            assert phi1 >= best - 1e-9 and phi5 >= best - 1e-9, "Phi below the global minimum"
            n_hit1 += int(abs(phi1 - best) < 1e-9)
            n_hit5 += int(abs(phi5 - best) < 1e-9)
        rows.append((N, K, n_hit1, n_hit5, reps))
        hits1 += n_hit1
        hits5 += n_hit5
        total += reps
    for N, K, h1, h5, reps in rows:
        report(f"  N={N}, K={K}", f"BUILD+SWAP {h1}/{reps}   with 5 restarts {h5}/{reps}")
    report("optimum", f"global optimum reached {hits1}/{total} (1 run), "
                      f"{hits5}/{total} (5 restarts)")


def check_alternating(instances, report):
    """How often the alternating heuristic lands on a one-swap stationary set.

    Split by instance kind, because the two are not comparable: an unstructured random
    matrix has no clusters for either algorithm to find, whereas an OM matrix from a
    genuine mixture does, and it is the second that says what the old figures were worth.
    """
    counts = {}
    for D, K, kind in instances:
        rng = np.random.default_rng(3)
        hit, tot = counts.get(kind, (0, 0))
        for _ in range(5):
            medoids = alternating_kmedoids(D, K, rng)
            hit += int(pam_certify_one_swap(D, medoids))
            tot += 1
        counts[kind] = (hit, tot)
    for kind, (hit, tot) in counts.items():
        report(f"  {kind}", f"one-swap stationary in {hit}/{tot} runs ({100 * hit / tot:.0f}%)")
    report("alternating", "PAM is one-swap stationary by construction, 100% in every case")


def check_recovery_gap(report, reps=12, seed=17):
    """Does the algorithm change the answer? Exact recovery on real mixtures, paired.

    Same dissimilarity matrix, same true labels, two algorithms -- so the comparison is
    paired and any difference is the algorithm alone. This is the question the refactor
    actually needs answered: whether the K-medoids figures have to be recomputed, or only
    relabelled.
    """
    rng = np.random.default_rng(seed)
    pam_hits = alt_hits = both = 0
    for _ in range(reps):
        D, labels = om_matrix(60, 300, 3, rng)
        K = len(np.unique(labels))
        p = exact_recovery(pam(D, K).labels, labels)
        medoids = alternating_kmedoids(D, K, np.random.default_rng(5))
        a = exact_recovery(np.argmin(D[:, medoids], axis=1), labels)
        pam_hits += int(p)
        alt_hits += int(a)
        both += int(p == a)
    report("recovery", f"exact recovery on {reps} real mixtures: PAM {pam_hits}/{reps}, "
                       f"alternating {alt_hits}/{reps}, same verdict {both}/{reps}")


def check_edge_cases(report):
    rng = np.random.default_rng(4)
    D = random_matrix(12, rng)

    r = pam(D, 1)
    assert r.medoids.size == 1 and r.one_swap_certified
    assert r.medoids[0] == int(np.argmin(D.sum(axis=1))), "K=1 must return the total-cost medoid"

    r = pam(D, 12)
    assert r.objective == 0.0 and r.one_swap_certified, "K=N must give Phi = 0"

    r = pam(np.zeros((1, 1)), 1)
    assert r.labels.tolist() == [0]

    flat = np.ones((8, 8))                       # every off-diagonal equal: ties everywhere
    np.fill_diagonal(flat, 0.0)
    r = pam(flat, 3)
    assert r.one_swap_certified and not r.hit_cap, "ties must not cycle"

    for bad, K in ((D, 0), (D, 13)):
        try:
            pam(bad, K)
            raise AssertionError(f"K={K} should have raised")
        except ValueError:
            pass
    try:
        pam(D, 3, n_restarts=5)
        raise AssertionError("n_restarts > 1 without rng should have raised")
    except ValueError:
        pass

    a = pam(D, 3, rng=np.random.default_rng(7), n_restarts=4)
    b = pam(D, 3, rng=np.random.default_rng(7), n_restarts=4)
    assert np.array_equal(a.medoids, b.medoids) and a.objective == b.objective
    assert np.array_equal(pam(D, 3).medoids, pam(D, 3).medoids), "BUILD+SWAP must be deterministic"

    report("edge cases", "K=1, K=N, N=1, all-tied, bad arguments, determinism: all OK")


# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=int, default=6,
                        help="random matrices and OM matrices of each size")
    args = parser.parse_args()

    rng = np.random.default_rng(11)
    instances = []
    for _ in range(args.instances):
        instances.append((random_matrix(30, rng), 3, "random uniform"))
        instances.append((random_matrix(24, rng, integer=True), 4, "random integer"))
    for _ in range(args.instances):
        D, _labels = om_matrix(40, 200, 3, rng)
        instances.append((D, 3, "real OM     "))

    width = 14
    def report(name, verdict):
        print(f"{name:<{width}} {verdict}")

    print(f"{len(instances)} instances: random uniform, random integer (ties), and real OM\n")
    check_swap_table(instances, report)
    check_objective(instances, report)
    check_stationarity(instances, report)
    check_edge_cases(report)
    print()
    check_optimum(report)
    print()
    check_alternating(instances, report)
    print()
    check_recovery_gap(report)
    print("\nOK - PAM is the strictly-improving one-swap algorithm of Theorem 3.8,\n"
          "     and every medoid set it returns is certified one-swap stationary.")


if __name__ == "__main__":
    main()
