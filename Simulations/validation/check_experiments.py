"""Assert that the Monte Carlo engine measures what it claims to measure.

    python validation/check_experiments.py [--quick]

The engine turns a mixture into three claims: a finite-horizon matrix Gamma^(n), a verdict
on the sign of eta_n, and a recovery probability. Each is a statement with a stated error
rate, so each is checked by simulation against a reference computed the expensive way.

  streams        the same key rebuilds the same generator, different keys do not collide,
                 and neither depends on the order in which keys were requested. Everything
                 reproducible about the tables rests on this.
  wilson         empirical coverage of the binomial interval used for every reported
                 proportion, including at p near 0 and 1 where the normal interval fails.
  simultaneous   empirical *joint* coverage of the Bonferroni-Student intervals: the
                 probability that all m entries are covered at once, which is what the sign
                 of eta_n depends on. Must be at least the nominal level, and conservative.
  gamma          Gamma_hat against a high-precision reference: symmetry exactly, and the
                 coverage of each entry's interval.
  eta            L <= eta_hat <= U always, and the conservative bounds cover the reference
                 eta at least at the nominal level. Then that a clearly separated mixture is
                 classified separated, and an unseparated one is not classified separated --
                 the direction that matters, since a false "separated" is the error that
                 would corrupt a conclusion.
  recovery       on a mixture separated by construction, every algorithm recovers the
                 partition exactly at a long enough horizon; on one where the components are
                 near-identical, none does. Bounds the engine at both ends.
  storage        rows survive a round trip through the CSV with their identifiers, the
                 header is written once, and an interrupted run keeps what it wrote.

The reference Gamma is itself a Monte Carlo estimate, but with 40 times the sample size, so
its own standard error is about a sixth of the one under test and it can be treated as the
truth for a coverage check.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from experiments import (ALGORITHMS, NONSEPARATED, SEPARATED, GammaEstimate,  # noqa: E402
                         ResultsWriter, draw_markov_mixture, estimate_gamma_paths,
                         run_dataset, seed_key, simultaneous_intervals, stream,
                         univariate_om, wilson_interval)

SEED = 20260901


def report(name, verdict):
    print(f"{name:<14} {verdict}")


# ---------------------------------------------------------------------------


def check_streams():
    a = stream(SEED, "grid", 0.3, 4, 7).normal(size=5)
    b = stream(SEED, "grid", 0.3, 4, 7).normal(size=5)
    assert np.array_equal(a, b), "the same key must rebuild the same stream"

    # order-independence: asking for other keys in between changes nothing
    stream(SEED, "other", 1).normal(size=100)
    c = stream(SEED, "grid", 0.3, 4, 7).normal(size=5)
    assert np.array_equal(a, c), "streams must not depend on request order"

    keys = [("grid", 0.3, 4, i) for i in range(200)] + [("grid", a_, 4, 0) for a_ in
                                                        np.linspace(0.1, 10, 200)]
    draws = np.array([stream(SEED, *k).normal(size=3) for k in keys])
    assert len(np.unique(draws, axis=0)) == len(keys), "distinct keys collided"
    assert not np.array_equal(stream(SEED, "grid", 0.3, 4, 7).normal(size=5),
                              stream(SEED + 1, "grid", 0.3, 4, 7).normal(size=5))
    report("streams", f"reproducible, order-independent, {len(keys)} keys with no collision")


def check_wilson(reps=20000, level=0.95):
    """Coverage of Wilson, against the normal interval it replaces.

    No binomial interval attains its nominal coverage exactly -- the support is discrete, so
    coverage oscillates with n and p, and dips near the ends are unavoidable (Brown, Cai and
    DasGupta, 2001). The claim being tested is therefore not "Wilson reaches 0.95" but the
    two things the engine actually relies on: coverage never collapses, and it beats the
    normal interval precisely where the heatmap cells live, at p near 0 and 1. There the
    normal interval degenerates -- at p_hat = 0 it is the single point {0} -- which is why
    it cannot be used for a proportion of exact recoveries.
    """
    rng = np.random.default_rng(1)

    def coverage(interval, p, n):
        successes = rng.binomial(n, p, size=reps)
        hits = 0
        for s in np.unique(successes):
            low, high = interval(int(s), n, level)
            assert 0.0 <= low <= high <= 1.0, "interval left [0, 1]"
            hits += int(np.sum(successes == s)) * int(low <= p <= high)
        return hits / reps

    def wald(successes, trials, lvl):
        z = norm.ppf(1.0 - (1.0 - lvl) / 2.0)
        p = successes / trials
        half = z * np.sqrt(p * (1.0 - p) / trials)
        return max(0.0, p - half), min(1.0, p + half)

    worst, beaten = 1.0, 0
    cases = ((0.5, 20), (0.05, 20), (0.02, 50), (0.95, 30), (0.999, 40))
    for p, n in cases:
        cov_w = coverage(wilson_interval, p, n)
        cov_n = coverage(wald, p, n)
        worst = min(worst, cov_w)
        beaten += int(cov_w >= cov_n)
        print(f"  p={p:<5} n={n:<3} wilson {cov_w:.3f}   normal {cov_n:.3f}")
    assert worst >= 0.90, f"coverage collapsed to {worst:.3f}"
    assert beaten == len(cases), "the normal interval covered better somewhere"
    report("wilson", f"coverage in [{worst:.3f}, 1], never below the normal interval "
                     f"(nominal {level}; dips near p=0 are inherent to any binomial interval)")


def check_simultaneous(reps=4000, m=10, n_obs=40, level=0.95):
    """Joint coverage of the m intervals: the event the eta verdict depends on."""
    rng = np.random.default_rng(2)
    truth = rng.uniform(0.5, 1.5, size=m)
    joint = per_entry = 0
    for _ in range(reps):
        sample = rng.normal(truth, 0.3, size=(n_obs, m))
        mean = sample.mean(axis=0)
        se = sample.std(axis=0, ddof=1) / np.sqrt(n_obs)
        low, high = simultaneous_intervals(mean, se, n_obs, m, level)
        inside = (low <= truth) & (truth <= high)
        joint += int(inside.all())
        per_entry += int(inside[0])
    joint /= reps
    per_entry /= reps
    assert joint >= level, f"joint coverage {joint:.3f} below nominal {level}"
    report("simultaneous", f"joint coverage {joint:.3f} >= {level} "
                           f"(per entry {per_entry:.3f}, conservative as intended)")


# ---------------------------------------------------------------------------


def _mixture(alpha, K, d, mixture_id=0):
    return draw_markov_mixture(K, d, alpha, mixture_id, SEED)


def check_gamma(n=200, n_pairs=25, reference_pairs=1000, reps=60, level=0.95):
    om = univariate_om("constant", 4)
    mixture = _mixture(0.3, 3, 4)
    grid = np.array([n], dtype=np.int64)

    reference = estimate_gamma_paths(mixture, om, grid, reference_pairs,
                                     stream(SEED, "reference"))
    assert np.allclose(reference.mean[..., 0], reference.mean[..., 0].T, atol=0.0), \
        "Gamma_hat must be exactly symmetric"

    covered = total = 0
    for r in range(reps):
        estimate = estimate_gamma_paths(mixture, om, grid, n_pairs, stream(SEED, "cov", r))
        low, high = estimate.intervals(level)
        for k in range(mixture.K):
            for l in range(k, mixture.K):
                covered += int(low[k, l, 0] <= reference.mean[k, l, 0] <= high[k, l, 0])
                total += 1
    coverage = covered / total
    assert coverage >= level - 0.02, f"entrywise coverage {coverage:.3f}"
    report("gamma", f"exactly symmetric; {coverage:.3f} of {total} intervals cover the "
                    f"reference (nominal {level}, simultaneous so conservative)")
    return mixture, om, reference


def check_eta(mixture, om, reference, n=200, n_pairs=25, reps=60, level=0.95):
    grid = np.array([n], dtype=np.int64)
    reference_eta = float(reference.eta()[0])

    covered = 0
    for r in range(reps):
        estimate = estimate_gamma_paths(mixture, om, grid, n_pairs, stream(SEED, "cov", r))
        low, high = estimate.eta_bounds(level)
        assert low[0] <= estimate.eta()[0] <= high[0], "bounds do not bracket the estimate"
        covered += int(low[0] <= reference_eta <= high[0])
    coverage = covered / reps
    assert coverage >= level, f"eta coverage {coverage:.3f} below nominal {level}"
    report("eta", f"L <= eta_hat <= U always; bounds cover the reference eta in "
                  f"{coverage:.3f} of {reps} runs (nominal {level})")

    # the two ends of the classification, on mixtures built to sit at each
    easy = _mixture(0.1, 2, 5, mixture_id=1)          # sharp kernels, well apart
    est = estimate_gamma_paths(easy, om, np.array([400]), 60, stream(SEED, "easy"))
    status_easy = est.separation_status(level)[0]

    hard = _mixture(50.0, 4, 5, mixture_id=2)         # near-uniform kernels, near-identical
    est = estimate_gamma_paths(hard, om, np.array([400]), 60, stream(SEED, "hard"))
    status_hard = est.separation_status(level)[0]

    assert status_easy == SEPARATED, f"a separated mixture was called {status_easy}"
    assert status_hard != SEPARATED, f"an unseparated mixture was called {status_hard}"
    report("  classify", f"alpha=0.1, K=2 -> {status_easy};  alpha=50, K=4 -> {status_hard}")


def check_recovery(N=60, n=400, reps=4):
    om = univariate_om("constant", 5)
    easy = _mixture(0.1, 3, 5, mixture_id=1)
    hard = _mixture(50.0, 3, 5, mixture_id=2)

    for label, mixture, expected in (("separated", easy, 1.0), ("near-identical", hard, 0.0)):
        hits = {a: 0 for a in ALGORITHMS}
        for r in range(reps):
            rows = run_dataset(mixture, om, N, np.array([n]), stream(SEED, label, r), r)
            for row in rows:
                hits[row["algorithm"]] += row["exact_recovery"]
        rates = {a: hits[a] / reps for a in ALGORITHMS}
        print("  " + f"{label:<14} " +
              "  ".join(f"{a}={rates[a]:.2f}" for a in ALGORITHMS))
        if expected == 1.0:
            assert all(v == 1.0 for v in rates.values()), "a separated mixture was not recovered"
        else:
            assert all(v == 0.0 for v in rates.values()), "near-identical components recovered"
    report("recovery", "every algorithm recovers the separated mixture and none the "
                       "near-identical one")


def check_storage(tmp=None):
    tmp = Path(tmp or HERE / "_check_storage.csv")
    if tmp.exists():
        tmp.unlink()
    fields = ["alpha", "K", "mixture_id", "n", "value"]
    rows = [{"alpha": 0.3, "K": 4, "mixture_id": i, "n": 100 * i, "value": i / 3}
            for i in range(5)]
    with ResultsWriter(tmp, fields) as writer:
        writer.write(rows[:2])
    with ResultsWriter(tmp, fields) as writer:          # reopening must not repeat the header
        writer.write(rows[2:])

    text = tmp.read_text(encoding="utf-8").strip().split("\n")
    assert text[0] == ",".join(fields), "header missing or wrong"
    assert sum(line.startswith("alpha,") for line in text) == 1, "header written twice"
    assert len(text) == 1 + len(rows), f"expected {len(rows)} rows, got {len(text) - 1}"

    import csv as _csv
    back = list(_csv.DictReader(tmp.read_text(encoding="utf-8").splitlines()))
    assert [int(r["mixture_id"]) for r in back] == list(range(5))
    assert abs(float(back[4]["value"]) - 4 / 3) < 1e-12, "float lost precision"
    tmp.unlink()
    report("storage", "header once, appends across sessions, identifiers and floats intact")


def check_reproducible(N=40, n=200):
    om = univariate_om("constant", 4)
    mixture = _mixture(0.3, 3, 4)
    key = seed_key("dataset", 0.3, 3, 0, 5)
    a = run_dataset(mixture, om, N, np.array([n]), stream(SEED, key), 5)
    b = run_dataset(mixture, om, N, np.array([n]), stream(SEED, key), 5)
    assert a == b, "the same key produced different results"
    other = run_dataset(mixture, om, N, np.array([n]), stream(SEED, "dataset", 0.3, 3, 0, 6), 6)
    assert [r["ari"] for r in a] != [r["ari"] for r in other], "different keys gave the same data"
    report("reproducible", "one key rebuilds a row exactly; a different key does not")


# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="fewer replications")
    args = parser.parse_args()
    scale = 4 if args.quick else 1

    check_streams()
    print()
    check_wilson(reps=20000 // scale)
    print()
    check_simultaneous(reps=4000 // scale)
    print()
    mixture, om, reference = check_gamma(reps=60 // scale)
    check_eta(mixture, om, reference, reps=60 // scale)
    print()
    check_recovery()
    print()
    check_storage()
    check_reproducible()
    print("\nOK - streams, intervals, the eta verdict, recovery and storage all behave as\n"
          "     documented; the intervals are conservative in the direction that matters.")


if __name__ == "__main__":
    main()
