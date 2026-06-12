"""Trajectory entropy — the *physical* occupancy entropy of a response.

⚠ This is NOT the Entropy-Search entropy. AGENTS §5 / optimization.md §7: two
entropies exist in this project and must never be cross-wired. The
information-theoretic ``H[p(θ*|D)]`` over the optimum's location lives only in
``optimization/acquisition/entropy_search.py``. This module is the *metric*: a
scalar describing how spread-out a state trajectory is, consumed by the
scoring layer like any other metric.

Definition: the Shannon entropy of the trajectory's occupancy histogram —
bin the visited values over their own range, normalise the counts to a
probability vector ``p``, and return ``H = −Σ p_b ln p_b`` (nats). A regulated
trajectory parked at the equilibrium occupies one bin and scores 0; a wandering
or ringing one spreads mass across bins and scores up to ``ln(n_bins)``. The
metric is bounded, hence always finite for the cost surface
(``numerical_standards.md`` §7).
"""

from __future__ import annotations

import numpy as np

from ..core.constants import ATOL
from ..core.types import SimulationResult

# Default histogram resolution: 32 bins bounds the metric at ln 32 ≈ 3.47 nats.
# A definition parameter (overridable per call), not a numerical tolerance.
DEFAULT_BINS: int = 32


def trajectory_entropy(
    result: SimulationResult, state_index: int = 0, n_bins: int = DEFAULT_BINS
) -> float:
    """Occupancy (Shannon) entropy of one state trajectory, in nats.

    Histogram over ``[min θ, max θ]`` with ``n_bins`` equal bins; a trajectory
    whose total range is within ``ATOL`` is treated as constant and scores
    exactly 0 (the range would otherwise be pure round-off noise).

    Parameters
    ----------
    result : SimulationResult
        The rollout to score.
    state_index : int
        Column of ``true_states`` to evaluate (default 0 = ``theta_p``).
    n_bins : int
        Histogram resolution, ≥ 2; the metric lies in ``[0, ln n_bins]``.
    """
    if not isinstance(result, SimulationResult):
        raise TypeError("result must be a SimulationResult")
    if not 0 <= state_index < result.n_x:
        raise ValueError(f"state_index must be in [0, {result.n_x}), got {state_index}")
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")
    theta = result.true_states[:, state_index]
    lo, hi = float(np.min(theta)), float(np.max(theta))
    if hi - lo <= ATOL:
        return 0.0
    counts, _ = np.histogram(theta, bins=n_bins, range=(lo, hi))
    p = counts / float(theta.shape[0])
    nonzero = p[p > 0.0]
    return float(-(nonzero * np.log(nonzero)).sum())
