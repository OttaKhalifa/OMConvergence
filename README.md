# Consistency of Optimal Matching-based Clustering for Mixtures of Markov Chains

Everything the numerical section of the paper needs is in
[`Simulations/`](Simulations/README.md): the library, the four notebooks that are the four
experiments, the sweeps that fill their tables, and the checks that pin the OM dissimilarity
to TraMineR and PAM to the algorithm Theorem 3.8 analyses.

```bash
cd Simulations
pip install -r requirements.txt
jupyter lab
```

[`legacy/`](legacy/README.md) parks the two notebooks the refactor superseded, with the
reasons; git history keeps the runs that produced the figures they published.
