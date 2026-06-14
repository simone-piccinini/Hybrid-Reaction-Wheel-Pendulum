"""Confidence-bound acquisition — the optimism baseline.

``optimization.md`` §4.1 lists Upper Confidence Bound alongside EI as a local
utility. Named ``UpperConfidenceBound`` per ``class_diagram.md``; because the
project **minimises** the cost ``J``, the relevant bound is the *lower*
confidence bound ``μ(θ) − β σ(θ)`` (GP-LCB, Srinivas et al. 2010), and the
candidate minimising it is preferred. To fit the base class's "higher is
better" convention the score returned is its negation,

    score(θ) = β σ(θ) − μ(θ),

so a candidate is attractive when its predicted cost is low (exploitation)
*or* its uncertainty is high (exploration); ``β`` sets the trade-off.
"""

from __future__ import annotations

import numpy as np

from ..gaussian_process import GaussianProcess
from ...core.types import SearchSpace
from .base import AcquisitionFunction, DEFAULT_N_CANDIDATES


class UpperConfidenceBound(AcquisitionFunction):
    """Lower-confidence-bound selection for minimisation (``class_diagram.md``).

    Parameters
    ----------
    beta : float
        Exploration weight ``β ≥ 0``; larger favours uncertain regions.
    n_candidates : int
        Candidate-set size for the §4.5 maximisation.
    """

    def __init__(self, beta: float = 2.0, n_candidates: int = DEFAULT_N_CANDIDATES):
        super().__init__(n_candidates=n_candidates)
        beta = float(beta)
        if beta < 0.0:
            raise ValueError("beta must be >= 0")
        self.beta = beta

    def _scores(
        self,
        gp: GaussianProcess,
        space: SearchSpace,
        candidates: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """``β σ − μ`` at each candidate (maximised ⇔ minimising ``μ − β σ``)."""
        posterior = gp.predict(candidates)
        std = np.sqrt(posterior.variance)
        return self.beta * std - posterior.mean
