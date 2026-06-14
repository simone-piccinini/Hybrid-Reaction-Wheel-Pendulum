"""Performance metrics — control effort and oscillation energy.

Pure functions of a :class:`SimulationResult` (``dependency_rules.md`` §3).
Both are discrete left-Riemann approximations of the continuous integrals over
the rollout's own (strictly increasing) time grid, so they are well-defined on
non-uniform grids too. As with the stability metrics, divergence handling
belongs to the ObjectiveFunction, not here.
"""

from __future__ import annotations

import numpy as np

from ..core.types import SimulationResult


def _time_steps(result: SimulationResult) -> np.ndarray:
    if not isinstance(result, SimulationResult):
        raise TypeError("result must be a SimulationResult")
    dt = np.diff(result.time)
    if dt.size and float(np.min(dt)) <= 0.0:
        raise ValueError("result.time must be strictly increasing")
    return dt


def control_effort(result: SimulationResult) -> float:
    """Integrated squared control ``∫ ‖u(t)‖² dt`` (``class_diagram.md``; AGENTS §4).

    Left-Riemann sum of the squared input held over each step (the input IS
    zero-order-held in this project, so the sum is exact for the applied
    signal): ``Σ_k ‖u_k‖² Δt_k`` over the first ``T−1`` samples. Units V²·s
    for the single-voltage plant. A single-sample run has zero effort.
    """
    dt = _time_steps(result)
    if dt.size == 0:
        return 0.0
    squared = np.sum(result.controls[:-1] ** 2, axis=1)
    return float(squared @ dt)


def oscillation_energy(result: SimulationResult, state_index: int = 1) -> float:
    """Integrated squared rate ``∫ θ̇(t)² dt`` — how much the response oscillates.

    The kinetic-energy-like measure of residual motion (README: metrics layer,
    "oscillation energy"): a well-damped transient dies quickly and scores
    low; a ringing one keeps accumulating. Left-Riemann over the time grid,
    default ``state_index=1`` = ``theta_p_dot`` (``notation.md`` §3).
    """
    dt = _time_steps(result)
    if not 0 <= state_index < result.n_x:
        raise ValueError(f"state_index must be in [0, {result.n_x}), got {state_index}")
    if dt.size == 0:
        return 0.0
    rate = result.true_states[:-1, state_index]
    return float((rate**2) @ dt)


def itae(result: SimulationResult, state_index: int = 0) -> float:
    """Integral of Time-multiplied Absolute Error ``∫₀^T t · |e(t)| dt``.

    A global performance index over the whole error history, not just two
    points of the step response (``notation.md`` §6). The explicit ``t`` factor
    weights *late* error far more than early error, so minimising ITAE pushes
    the response to settle quickly and drives the steady-state error to zero —
    the property that makes ITAE a standard tuning criterion (Graham & Lathrop,
    1953).

    The error is the deviation of the regulated state from its upright target
    (zero): ``e(t) = θ(t)`` with ``state_index=0`` = ``theta_p`` by default.
    Computed by the trapezoidal rule on the rollout's own time grid (exact for
    a piecewise-linear integrand, and well-defined on non-uniform grids).
    """
    if not isinstance(result, SimulationResult):
        raise TypeError("result must be a SimulationResult")
    if not 0 <= state_index < result.n_x:
        raise ValueError(f"state_index must be in [0, {result.n_x}), got {state_index}")
    time = result.time
    if time.shape[0] < 2:
        return 0.0
    integrand = time * np.abs(result.true_states[:, state_index])
    dt = np.diff(time)
    if float(np.min(dt)) <= 0.0:
        raise ValueError("result.time must be strictly increasing")
    return float(np.sum(0.5 * (integrand[:-1] + integrand[1:]) * dt))  # trapezoid
