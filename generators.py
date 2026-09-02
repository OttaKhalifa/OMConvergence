"""Generators: Markov chains, mixtures of Markov chains, and hidden Markov models.

Three mechanisms, one shared convention: a sequence is drawn once and for all from a
single component, and the component labels are the ground truth against which a
clustering is scored.

- **Markov chains** -- ``sample_markov_model`` draws a kernel from a Dirichlet prior,
  ``sample_chain_order1`` a trajectory from it, and ``sample_mixture`` a whole labelled
  sample from a mixture of such kernels. This is the mechanism of the paper.
- **Hidden Markov models** -- ``MixtureOfHMMGenerator`` adds a latent state between the
  chain and what is observed, and observes V channels rather than one. The chain is
  homogeneous: no term of the model depends on t.
- **Mixture weights** -- ``sample_mixture_weights`` and ``sample_component_labels`` are
  shared by the mechanisms above, since two mechanisms may differ by the law of the
  observations but not by the way sequences are split between components.

Contents
--------
Mixture weights and labels : ``weights_rng``, ``sample_mixture_weights``,
                             ``sample_component_labels``
Markov chains              : ``sample_markov_model``, ``sample_chain_order1``,
                             ``stationary_distribution_markov``, ``spectral_gap``
Mixtures of Markov chains  : ``sample_mixture``
Hidden Markov models       : ``HMMComponent``, ``MixtureOfHMMGenerator``
                             (``sample_dataset``, ``sample_component``)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

# Only for the inner loop of `_walk`; `om` owns the pure-Python fallback.
from om import njit

# -------------------------------------------------------------------------
# Mixture weights and component labels
# -------------------------------------------------------------------------

# Les proportions sont tirees dans une Dirichlet symetrique plutot que forcees egales. Des
# groupes de tailles inegales sont la regle sur donnees reelles, et un banc a groupes
# exactement equilibres flatterait les methodes precisement la ou elles sont les plus
# fragiles. Le prix est une variance inter-replication plus forte -- a K = 4, la plus petite
# composante vaut 5 % de N en mediane contre 25 % si elle etait equilibree -- qu'il faut
# absorber par le nombre de replications. Passer mixture_weights permet de fixer les
# proportions, et donc de mesurer cet effet plutot que de le subir.

def _as_list(value: Union[int, Sequence[int]], length: int, name: str) -> List[int]:
    """Repete un entier ou valide une valeur par canal strictement positive."""
    if isinstance(value, (int, np.integer)):
        values = [int(value)] * length
    else:
        values = [int(v) for v in value]
        if len(values) != length:
            raise ValueError(f"{name} must have length {length}, got {len(values)}")
    if any(v <= 0 for v in values):
        raise ValueError(f"All values of {name} must be > 0")
    return values


def weights_rng(rng: np.random.Generator) -> np.random.Generator:
    """Flux dedie au tirage des proportions, derive de `rng`.

    `sample_mixture_weights` ne consomme le RNG que si `weights is None`.
    Appele directement sur le flux principal, imposer les proportions ne le
    consommait donc pas, et tout ce qui est tire ensuite -- les parametres des
    composantes -- se retrouvait decale. Deux runs a graine identique n'avaient
    pas les memes clusters selon qu'on imposait l'equilibre ou non, ce qui
    rendait la comparaison equilibre / desequilibre non appariee : elle melangeait
    l'effet des proportions et celui d'un autre tirage de parametres.

    On consomme donc ici un entier, inconditionnellement, et les proportions sont
    tirees dans le flux qu'il ensemence. Le flux principal avance de la meme
    facon dans les deux cas, et les parametres des composantes ne dependent plus
    du mode de tirage des proportions.
    """
    return np.random.default_rng(int(rng.integers(1 << 63)))


def sample_mixture_weights(
    rng: np.random.Generator,
    n_components: int,
    weights: Optional[Sequence[float]] = None,
    min_weight: float = 0.10,
) -> np.ndarray:
    """Proportions des composantes, tirees ou imposees.

    None tire dans une Dirichlet symetrique -- loi uniforme sur le simplexe --
    puis applique un plancher affine : chaque composante recoit au moins
    min_weight, la masse restante (1 - K*min_weight) etant repartie par la
    Dirichlet. Le desequilibre survit dans ce residu ; min_weight=0 redonne la
    Dirichlet nue, min_weight=1/K l'equilibre parfait. Sans effet si weights est
    impose. Le plancher porte sur les PROBABILITES, pas les effectifs realises.
    Sinon les poids fournis sont valides et renormalises.
    """
    if n_components <= 0:
        raise ValueError("n_components must be > 0")
    if weights is None:
        if not (0.0 <= min_weight <= 1.0 / n_components):
            raise ValueError(f"min_weight must be in [0, 1/K], got {min_weight}")
        return min_weight + (1.0 - n_components * min_weight) * rng.dirichlet(np.ones(n_components))

    w = np.asarray(weights, dtype=float)
    if w.shape != (n_components,):
        raise ValueError(f"mixture_weights must have shape ({n_components},), got {w.shape}")
    if np.any(w < 0):
        raise ValueError("mixture_weights must be >= 0")
    total = w.sum()
    if total <= 0:
        raise ValueError("mixture_weights must sum to > 0")
    return w / total


def sample_component_labels(
    rng: np.random.Generator,
    weights: np.ndarray,
    n_sequences: int,
) -> np.ndarray:
    """Tire l'appartenance de chaque sequence, une fois pour toutes.

    Tirage i.i.d. et non repartition deterministe : les effectifs realises
    fluctuent donc autour de n_sequences * weights, et une composante peut
    ressortir vide. C'est le mecanisme, pas un defaut -- mais c'est a garder en
    tete en lisant la variance d'une cellule.
    """
    if n_sequences <= 0:
        raise ValueError("n_sequences must be > 0")
    return rng.choice(len(weights), size=n_sequences, p=weights)

# ---------------------------------------------------------------------------
# Markov chains
# ---------------------------------------------------------------------------


def _expand_dirichlet_alpha(alpha, n_rows, n_cols, name="alpha"):
    if np.isscalar(alpha):
        if alpha <= 0:
            raise ValueError(f"{name} must be > 0")
        return np.full((n_rows, n_cols), float(alpha))
    arr = np.asarray(alpha, dtype=float)
    if arr.ndim == 1:
        if arr.shape[0] != n_cols:
            raise ValueError(f"{name} must have length {n_cols}")
        arr = np.tile(arr, (n_rows, 1))
    elif arr.ndim != 2 or arr.shape != (n_rows, n_cols):
        raise ValueError(f"{name} must be a scalar, a 1D or a ({n_rows}, {n_cols}) array")
    if np.any(arr <= 0):
        raise ValueError(f"all values of {name} must be > 0")
    return arr


def sample_markov_model(n_states, order_k, alpha, rng, alpha_init=None):
    """Order-`order_k` Markov model with rows drawn from a Dirichlet(`alpha`) prior.

    The returned dictionary holds the (n_states ** order_k, n_states) transition array; for
    order_k = 1 this array is the transition matrix P itself, which is the only case the
    notebooks use. `init_probs` is left in place, and drawn, because every figure of the paper
    is reproduced from a fixed seed: removing the draw would shift the random stream.
    """
    if order_k < 1:
        raise ValueError("order_k must be >= 1")
    if n_states < 2:
        raise ValueError("n_states must be >= 2")
    n_contexts = n_states ** order_k
    trans_alpha = _expand_dirichlet_alpha(alpha, n_contexts, n_states)
    transitions = np.empty((n_contexts, n_states), dtype=float)
    for i in range(n_contexts):
        transitions[i] = rng.dirichlet(trans_alpha[i])
    if alpha_init is None:
        alpha_init = alpha
    if order_k == 1:
        init_probs = rng.dirichlet(_expand_dirichlet_alpha(alpha_init, 1, n_states)[0])
    else:
        init_probs = rng.dirichlet(_expand_dirichlet_alpha(alpha_init, 1, n_contexts)[0])
    return {"n_states": n_states, "order": order_k,
            "transitions": transitions, "init_probs": init_probs}


@njit(cache=True)
def _walk(cum, u, x0):
    """Inverse-cdf walk: cum is the row-wise cumulative kernel, u the uniforms."""
    n = u.shape[0]
    x = np.empty(n, dtype=np.int64)
    x[0] = x0
    for t in range(1, n):
        x[t] = np.searchsorted(cum[x[t - 1]], u[t])
    return x


def sample_chain_order1(P, n, rng, init=None):
    """Length-n trajectory of the first-order chain with kernel P.

    `init` is either None (start from the stationary law), an integer (Dirac initial law), or
    a probability vector.
    """
    d = P.shape[0]
    if init is None:
        x0 = int(rng.choice(d, p=stationary_distribution_markov(P)))
    elif np.isscalar(init):
        x0 = int(init)
    else:
        x0 = int(rng.choice(d, p=np.asarray(init, dtype=float)))
    return _walk(np.cumsum(P, axis=1), rng.random(n), x0)


def stationary_distribution_markov(P, init=None, tol=1e-12, max_iter=200_000):
    """Stationary law of an irreducible kernel, by power iteration."""
    n = P.shape[0]
    v = np.full(n, 1.0 / n) if init is None else np.asarray(init, dtype=float).reshape(-1)
    if v.size != n or v.sum() <= 0:
        raise ValueError("init must be a non-negative vector of length P.shape[0]")
    v = v / v.sum()
    for _ in range(max_iter):
        v_next = v @ P
        if np.abs(v_next - v).sum() < tol:
            v = v_next
            break
        v = v_next
    return v / v.sum()


def spectral_gap(P):
    """1 - |lambda_2|, a proxy for the mixing speed of P."""
    ev = np.sort(np.abs(np.linalg.eigvals(P)))
    return float(1.0 - ev[-2])



# ---------------------------------------------------------------------------
# Mixtures of Markov chains
# ---------------------------------------------------------------------------


def sample_mixture(K, N, n, d, alpha, rng, weights=None, kernels=None):
    """N sequences drawn from a K-component mixture of order-1 chains, with their labels.

    Latent labels are i.i.d. with law `weights` (uniform if None); conditionally on Z_i = k,
    sequence i is a length-n realisation of kernel k. As in the other experiments, every
    sequence starts from its own initial law, the Dirac law at a state drawn uniformly on
    Sigma, so no two sequences share an initial condition.

    Pass `kernels` (a (K, d, d) array) to keep the mixture fixed across replicates and let
    only the labels and the trajectories be redrawn; otherwise K kernels are drawn row-wise
    from a Dirichlet(`alpha`) prior.
    """
    if kernels is None:
        kernels = np.stack([sample_markov_model(d, 1, alpha, rng)["transitions"]
                            for _ in range(K)])
    else:
        kernels = np.asarray(kernels, dtype=float)
        if kernels.shape != (K, d, d):
            raise ValueError(f"kernels must have shape ({K}, {d}, {d})")
    w = np.full(K, 1.0 / K) if weights is None else np.asarray(weights, dtype=float)
    if w.size != K or np.any(w <= 0):
        raise ValueError("weights must be K positive numbers")
    w = w / w.sum()
    labels = rng.choice(K, size=N, p=w)
    X = np.empty((N, n), dtype=np.int64)
    for i, z in enumerate(labels):
        X[i] = sample_chain_order1(kernels[z], n, rng, init=int(rng.integers(d)))
    return {"X": X, "labels": labels, "kernels": kernels, "weights": w,
            "counts": np.bincount(labels, minlength=K)}



# -------------------------------------------------------------------------
# Hidden Markov models: mixture of homogeneous multichannel HMMs
# -------------------------------------------------------------------------

# Chaque sequence recoit une composante k une fois pour toutes, puis suit une chaine de
# Markov latente sur S etats dont la loi ne depend pas du temps :
#
#     z_1 ~ pi[k],  z_t | z_{t-1}=s ~ A[k][s],  P(X_{t,d}=c | z_t=s) = B[k,d][s, c]
#
# Les V canaux sont conditionnellement independants sachant l'etat : toute la dependance
# entre canaux passe par z_t. C'est ce qui rend le cas multicanal lisible -- ce que l'OM
# multicanal doit retrouver d'un groupe est exactement la structure que cet etat latent
# commun impose aux V canaux a la fois. Les sequences produites sont *completes*, de
# longueur seq_len ; toute observation partielle s'applique ensuite.

@dataclass
class HMMComponent:
    """Parametres d'une composante : S etats, V canaux.

    Ne porte que ce qui varie d'une composante a l'autre ; les dimensions et les
    hyperparametres de tirage vivent sur le generateur.
    """

    pi: np.ndarray            # (S,)                  loi initiale
    A: np.ndarray             # (S, S)                transitions
    B: List[np.ndarray]       # par canal : (S, C_d)  emissions


class MixtureOfHMMGenerator:
    """Melange de HMM homogenes categoriels multivaries."""

    def __init__(
        self,
        n_components: int,
        n_states: int,
        n_vars: int,
        n_categories: Union[int, Sequence[int]],
        mixture_weights: Optional[Sequence[float]] = None,
        min_weight: float = 0.10,
        alpha_pi: float = 1.0,
        alpha_A: float = 1.0,
        alpha_B: float = 1.0,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
    ) -> None:
        if n_components <= 0:
            raise ValueError("n_components must be > 0")
        if n_vars <= 0:
            raise ValueError("n_vars must be > 0")
        if n_states <= 0:
            raise ValueError("n_states must be > 0")

        self.K = int(n_components)
        self.S = int(n_states)
        self.V = int(n_vars)
        self.C_list = _as_list(n_categories, self.V, "n_categories")
        # X et Z sont stockes en int8 : au-dela, le cast serait silencieux.
        if max(self.S, max(self.C_list)) > 127:
            raise ValueError("n_states and n_categories must be <= 127 (X, Z are int8)")

        self.alpha_pi = float(alpha_pi)
        self.alpha_A = float(alpha_A)
        self.alpha_B = float(alpha_B)

        self.rng = rng if rng is not None else np.random.default_rng(seed)
        # Flux dedie : voir `weights_rng`. Le flux principal avance de la meme
        # facon que les proportions soient imposees ou tirees, donc les composantes
        # ci-dessous ne dependent pas de ce choix.
        self.w = sample_mixture_weights(weights_rng(self.rng), self.K,
                                        mixture_weights, min_weight)
        self.components = [self._sample_component() for _ in range(self.K)]

    # ---- parametres ----

    def _sample_component(self) -> HMMComponent:
        """Tire pi, A et les B canal par canal dans des Dirichlet symetriques.

        Les trois alpha reglent chacun la concentration d'une brique : alpha < 1
        donne des lois piquees -- chaine quasi deterministe, etats bien separes
        par leurs emissions -- alpha > 1 les rapproche de l'uniforme et rend les
        composantes d'autant plus difficiles a distinguer.
        """
        S = self.S
        pi = self.rng.dirichlet(np.full(S, self.alpha_pi))
        A = np.vstack([self.rng.dirichlet(np.full(S, self.alpha_A)) for _ in range(S)])
        B = [np.vstack([self.rng.dirichlet(np.full(C_d, self.alpha_B))
                        for _ in range(S)])
             for C_d in self.C_list]
        return HMMComponent(pi=pi, A=A, B=B)

    # ---- echantillonnage ----

    def sample_dataset(self, n_sequences: int, seq_len: int) -> Dict[str, Any]:
        """Genere n_sequences sequences completes de longueur seq_len.

        Renvoie {"X": (N, T, V) int8, "y": (N,), "Z": (N, T) int8}, ou Z donne
        les etats caches. Z est de toute facon calcule pour produire X : le
        renvoyer systematiquement ne coute qu'une recopie.
        """
        T = int(seq_len)
        if T <= 0:
            raise ValueError("seq_len must be > 0")

        y = sample_component_labels(self.rng, self.w, n_sequences)

        X = np.empty((n_sequences, T, self.V), dtype=np.int8)
        Z = np.empty((n_sequences, T), dtype=np.int8)
        for k, comp in enumerate(self.components):
            idx = np.where(y == k)[0]
            if idx.size:
                self._sample_batch(idx, comp, T, X, Z)

        return {"X": X, "y": y, "Z": Z}

    def _sample_batch(
        self,
        idx: np.ndarray,
        comp: HMMComponent,
        T: int,
        X_out: np.ndarray,
        Z_out: np.ndarray,
    ) -> None:
        """Genere en bloc toutes les sequences d'une meme composante.

        La chaine se deroule pas a pas -- z_t depend de z_{t-1} -- mais groupee
        par etat de depart : un seul tirage par (t, etat) au lieu d'un par
        sequence. Les emissions, elles, ne dependent pas du temps : sachant Z,
        toutes les positions occupant le meme etat sont i.i.d., et un unique
        tirage par (etat, canal) couvre le bloc (n_k, T) entier.
        """
        n_k, S = len(idx), self.S

        # --- Chaine de Markov : (n_k, T) ---
        z = np.empty((n_k, T), dtype=np.int8)
        z[:, 0] = self.rng.choice(S, size=n_k, p=comp.pi)
        for t in range(1, T):
            for s in range(S):
                mask = z[:, t - 1] == s
                n_s = int(mask.sum())
                if n_s:
                    z[mask, t] = self.rng.choice(S, size=n_s, p=comp.A[s])

        # --- Emissions : groupees par etat, sur tout le bloc ---
        X_k = np.empty((n_k, T, self.V), dtype=np.int8)
        for s in range(S):
            mask_s = z == s
            n_s = int(mask_s.sum())
            if n_s == 0:
                continue
            for d, B_d in enumerate(comp.B):
                X_k[:, :, d][mask_s] = self.rng.choice(len(B_d[s]), size=n_s, p=B_d[s])

        X_out[idx] = X_k
        Z_out[idx] = z

    def sample_component(self, k, n_sequences, seq_len, rng=None):
        """`n_sequences` trajectories of component k alone, as an (R, T, V) int8 array.

        `sample_dataset` draws the component labels; this bypasses them. Estimating
        Gamma^(n)_kl needs trajectories of a *named* pair of components, on an independent
        sample, which the labelled dataset cannot provide without wasting most of it.

        Passing `rng` runs the draw on that stream and leaves the generator's own untouched,
        so parameters stay fixed while the data varies.
        """
        if not 0 <= k < self.K:
            raise ValueError(f"component index must lie in [0, {self.K}), got {k}")
        T = int(seq_len)
        if T <= 0:
            raise ValueError("seq_len must be > 0")
        if n_sequences <= 0:
            raise ValueError("n_sequences must be > 0")

        saved = self.rng
        if rng is not None:
            self.rng = rng
        try:
            X = np.empty((n_sequences, T, self.V), dtype=np.int8)
            Z = np.empty((n_sequences, T), dtype=np.int8)
            self._sample_batch(np.arange(n_sequences), self.components[k], T, X, Z)
        finally:
            self.rng = saved
        return X

    def params_summary(self) -> Dict[str, Any]:
        return {
            "K": self.K,
            "weights": self.w.copy(),
            "n_states": self.S,
            "n_vars": self.V,
            "n_categories": list(self.C_list),
            "alpha_pi": self.alpha_pi,
            "alpha_A": self.alpha_A,
            "alpha_B": self.alpha_B,
        }
