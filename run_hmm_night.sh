#!/bin/bash
# The whole numerical section, replayed on mixtures of multichannel HMMs.
#
# Five channels of five letters, four hidden states, constant substitution costs. The grid
# axis ties the three Dirichlet concentrations -- initial law, transitions, emissions -- to
# one alpha. Turning the emissions alone leaves every mixture separated with a margin
# between 0.38 and 2.83, because five channels of five letters carry enough that components
# stay apart whatever their emissions look like; the latent chain has to be blurred too.
#
# Measured on the tied knob at n = 1000: eta is 0.34 at alpha = 1, 0.08 at 3, 0.016 at 10,
# and crosses zero near 20, saturating around -0.01 beyond 100. The grid
# {0.5, 1, 2, 3, 5, 10, 30, 100} spans that, with the crossing inside it -- the same shape as
# the Markov grid, translated onto this mechanism's scale, where alpha = 1 here is about
# alpha = 0.3 there. The paths run at alpha = 3, where recovery climbs rather than
# saturating at once.
#
# Sequential rather than parallel: both sweeps saturate the cores, so running them at once
# only makes each slower.
set -e
cd "$(dirname "$0")"

echo "=== paths against n, HMM ($(date +%H:%M)) ==="
python3 -u sweep_path.py --mechanism hmm --K 4 --alpha 3.0 \
    --n-mixtures 6 --n-datasets 20 --N 200 --tag hmm --out results

echo "=== (alpha, K) grid, HMM ($(date +%H:%M)) ==="
python3 -u sweep_recovery.py --mechanism hmm --n-mixtures 20 --n-datasets 1 \
    --horizons 400 1000 --N 200 --tag hmm --out results

echo "=== selecting K, HMM ($(date +%H:%M)) ==="
python3 -u sweep_khat.py --mechanism hmm --n-mixtures 20 --horizons 1000 \
    --N 200 --asw --tag hmm --out results

echo "=== done ($(date +%H:%M)) ==="
