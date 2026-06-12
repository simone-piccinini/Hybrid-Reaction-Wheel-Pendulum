"""Stability metrics — percentage overshoot M_p and settling time T_s.

Pure functions of a :class:`SimulationResult` (``dependency_rules.md`` §3:
"metrics are pure functions of a SimulationResult"), evaluated on the **true**
state trajectory — performance is judged on what the plant actually did, not
on the estimate. Code names follow ``notation.md`` §6 (``overshoot``,
``settling_time``); the scalar cost combining them lives with the optimization
layer's ``ObjectiveFunction``, not here.

The runs scored here are **regulation transients** (an initial perturbation
driven back to the upright equilibrium at 0), so the classical step-response
definitions (Ogata, *Modern Control Engineering*, §5-4) are restated with the
initial displacement playing the role of the step size: the response travels
from ``θ₀`` to a final value of 0, overshoot is the excursion past 0 on the
far side, and the settling band is a fraction of ``|θ₀|``.

Metrics are computed regardless of ``result.diverged`` — mapping divergence to
the finite penalty is the ObjectiveFunction's job (``numerical_standards.md`` §7).
"""

from __future__ import annotations

import numpy as np

from ..core.constants import ATOL
from ..core.types import SimulationResult

# Ogata §5-4 settling criterion: response stays within ±2% of the reference
# scale. Overridable per call (a definition parameter, not a tolerance).
SETTLING_BAND_FRACTION: float = 0.02


def _trajectory(result: SimulationResult, state_index: int) -> np.ndarray:
    if not isinstance(result, SimulationResult):
        raise TypeError("result must be a SimulationResult")
    if not 0 <= state_index < result.n_x:
        raise ValueError(f"state_index must be in [0, {result.n_x}), got {state_index}")
    return result.true_states[:, state_index]


def overshoot(result: SimulationResult, state_index: int = 0) -> float:
    """Percentage overshoot ``M_p`` of a regulation transient (``notation.md`` §6).

    The largest excursion past the equilibrium on the side *opposite* the
    initial displacement, as a percentage of that displacement (Ogata §5-4
    with final value 0 and step size ``θ₀``):

        M_p = 100 · max(0, max_t −sign(θ₀)·θ(t)) / |θ₀|.

    A monotone decay never crosses zero and scores 0. When the run starts
    already at the equilibrium (``|θ₀| ≤ ATOL``) there is no transient to
    overshoot and the metric is defined as 0.

    Parameters
    ----------
    result : SimulationResult
        The rollout to score.
    state_index : int
        Column of ``true_states`` to evaluate (default 0 = ``theta_p``, the
        angle being stabilised).
    """
    theta = _trajectory(result, state_index)
    theta_0 = float(theta[0])
    if abs(theta_0) <= ATOL:
        return 0.0
    sign = 1.0 if theta_0 > 0 else -1.0
    worst_crossing = float(np.max(-sign * theta))
    return 100.0 * max(0.0, worst_crossing) / abs(theta_0)


def settling_time(
    result: SimulationResult,
    state_index: int = 0,
    band_fraction: float = SETTLING_BAND_FRACTION,
) -> float:
    """Settling time ``T_s`` (``notation.md`` §6): when the response enters the band for good.

    The earliest time after which ``|θ(t)|`` stays within
    ``band_fraction · |θ₀|`` for the rest of the run (Ogata §5-4, the 2%
    criterion by default, taken relative to the initial displacement since the
    regulation target is 0). Returns a time on the run's own clock
    (``result.time``).

    Edge conventions, chosen to keep the metric finite for the cost surface
    (``numerical_standards.md`` §7):
    - never settles within the horizon → returns ``time[-1]`` (capped);
    - already inside the band at every step → returns ``time[0]``;
    - starts at the equilibrium (``|θ₀| ≤ ATOL``) → 0.0 if it stays within
      ``ATOL`` throughout, else ``time[-1]`` (it left a regulated point).
    """
    if band_fraction <= 0.0:
        raise ValueError("band_fraction must be > 0")
    theta = _trajectory(result, state_index)
    time = result.time
    theta_0 = float(theta[0])
    if abs(theta_0) <= ATOL:
        return 0.0 if float(np.max(np.abs(theta))) <= ATOL else float(time[-1])
    band = band_fraction * abs(theta_0)
    violations = np.abs(theta) > band
    if not bool(violations.any()):
        return float(time[0])
    last_violation = int(np.flatnonzero(violations)[-1])
    if last_violation == theta.shape[0] - 1:
        return float(time[-1])
    return float(time[last_violation + 1])
