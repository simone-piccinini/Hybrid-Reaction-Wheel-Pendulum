"""Expected Improvement — the classic value-greedy acquisition (baseline).

``optimization.md`` §4.1 names EI as one of the "local utilities" Entropy
Search departs from: it scores a candidate purely by how much its predicted
cost is expected to beat the incumbent. Cheap and effective, it is the project
baseline against the goal acquisition (Entropy Search) and the proposal
density that ES draws its representer points from (§4.3).
"""

from __future__ import annotations

import numpy as np

from ..gaussian_process import GaussianProcess
from ...core.types import SearchSpace
from .base import AcquisitionFunction, expected_improvement_values


class ExpectedImprovement(AcquisitionFunction):
    """EI for the minimisation of the rollout cost (Jones et al. 1998).

    The incumbent is the best (lowest) **observed** cost so far. Under noise
    this is the simplest classical choice; ``optimization.md`` §5 separately
    reports the posterior-mean minimiser as the final answer.
    """

    def _scores(
        self,
        gp: GaussianProcess,
        space: SearchSpace,
        candidates: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """EI at each candidate (``space``/``rng`` unused — EI is deterministic)."""
        incumbent = float(np.min(gp.observed_targets))
        return expected_improvement_values(gp, candidates, incumbent)
