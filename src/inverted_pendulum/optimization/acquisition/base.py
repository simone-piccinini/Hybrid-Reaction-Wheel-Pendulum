"""AcquisitionFunction — the abstract strategy that picks the next θ.

``data_contracts.md`` §8 (the critical contract): ``select`` receives the
**whole** :class:`GaussianProcess`, not pre-computed moments, because Entropy
Search needs the full posterior to sample functions; EI/UCB call ``gp.predict``
internally. The seeded ``rng`` makes candidate sampling and GP draws
reproducible (determinism contract §6).

All three acquisitions share the §4.5 maximisation scheme: "evaluate α on a
moderate candidate set (random or proposal-drawn) … the maximiser becomes the
next controller configuration." This base class implements that loop once —
draw candidates from the :class:`SearchSpace`, score them, return the best —
and leaves only the per-candidate score to subclasses. (Local refinement of
the best candidate, mentioned in §4.5, is omitted for clarity; resolution is
controlled by ``n_candidates``.)

Everything here works in **minimisation** convention — the project minimises
the rollout cost ``J`` (``optimization.md`` §1) — so each ``_scores`` returns a
"higher is better" value and ``select`` takes the ``argmax``.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np

from ..gaussian_process import GaussianProcess
from ...core.types import SearchSpace

# Default candidate-set size for the §4.5 acquisition maximisation.
DEFAULT_N_CANDIDATES: int = 512

_SQRT_2PI = math.sqrt(2.0 * math.pi)


def standard_normal_pdf(z: np.ndarray) -> np.ndarray:
    """Standard-normal density ``φ(z) = e^{−z²/2}/√(2π)`` (elementwise)."""
    z = np.asarray(z, dtype=np.float64)
    return np.exp(-0.5 * z**2) / _SQRT_2PI


def standard_normal_cdf(z: np.ndarray) -> np.ndarray:
    """Standard-normal CDF ``Φ(z) = ½(1 + erf(z/√2))`` via stdlib ``math.erf``.

    No SciPy (``AGENTS.md`` §3): ``math.erf`` is standard library, applied
    elementwise.
    """
    z = np.asarray(z, dtype=np.float64)
    erf = np.vectorize(math.erf, otypes=[np.float64])
    return 0.5 * (1.0 + erf(z / math.sqrt(2.0)))


class AcquisitionFunction(ABC):
    """Abstract acquisition (``data_contracts.md`` §8; ``class_diagram.md``).

    Parameters
    ----------
    n_candidates : int
        Size of the candidate set the acquisition is maximised over (§4.5).
    """

    def __init__(self, n_candidates: int = DEFAULT_N_CANDIDATES) -> None:
        if n_candidates < 1:
            raise ValueError("n_candidates must be >= 1")
        self.n_candidates = int(n_candidates)

    @abstractmethod
    def _scores(
        self,
        gp: GaussianProcess,
        space: SearchSpace,
        candidates: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Per-candidate acquisition values (higher = more desirable to query).

        ``candidates`` is ``(n_candidates, d)``; returns a length-``n_candidates``
        vector. ``space`` is passed for acquisitions that need the whole domain
        (Entropy Search draws representer points from it); EI/UCB ignore it.
        """

    def select(
        self,
        gp: GaussianProcess,
        space: SearchSpace,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return the next θ to evaluate (``data_contracts.md`` §8).

        Draws a candidate set from ``space`` (in the GP's own search
        coordinates), scores it, and returns the highest-scoring candidate.
        """
        if not isinstance(gp, GaussianProcess):
            raise TypeError("gp must be a GaussianProcess")
        if not isinstance(space, SearchSpace):
            raise TypeError("space must be a SearchSpace")
        if space.dimension != gp.kernel.dim:
            raise ValueError(
                f"space dimension {space.dimension} != GP dimension {gp.kernel.dim}"
            )
        candidates = space.sample(self.n_candidates, rng)
        values = self._scores(gp, space, candidates, rng)
        return candidates[int(np.argmax(values))].copy()


def expected_improvement_values(
    gp: GaussianProcess, candidates: np.ndarray, incumbent: float
) -> np.ndarray:
    """Expected improvement over ``incumbent`` for a **minimisation** objective.

    Shared by :class:`ExpectedImprovement` and used as the Entropy-Search
    representer proposal (``optimization.md`` §4.3). With ``μ, σ`` the GP
    posterior at a candidate and ``f⁺`` the incumbent (best cost so far):

        z = (f⁺ − μ) / σ,   EI = (f⁺ − μ) Φ(z) + σ φ(z)   (σ > 0),

    and ``EI = max(f⁺ − μ, 0)`` where ``σ = 0`` (Jones et al. 1998, adapted to
    minimisation). EI ≥ 0 everywhere — it doubles as a sampling density.
    """
    posterior = gp.predict(candidates)
    mean = posterior.mean
    std = np.sqrt(posterior.variance)
    improvement = incumbent - mean
    ei = np.empty_like(mean)
    positive = std > 0.0
    z = np.zeros_like(mean)
    z[positive] = improvement[positive] / std[positive]
    ei[positive] = (
        improvement[positive] * standard_normal_cdf(z[positive])
        + std[positive] * standard_normal_pdf(z[positive])
    )
    ei[~positive] = np.maximum(improvement[~positive], 0.0)
    # EI ≥ 0 analytically; clip the round-off negatives so it is a valid
    # sampling density (Entropy Search resamples representers ∝ EI)
    return np.maximum(ei, 0.0)
