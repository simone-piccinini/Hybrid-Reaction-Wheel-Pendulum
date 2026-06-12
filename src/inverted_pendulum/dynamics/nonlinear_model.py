"""Discrete-time stepper for the true nonlinear plant.

Wraps :class:`~inverted_pendulum.physical.pendulum.ReactionWheelPendulum` and a
``numerics`` integrator into the form the simulation layer drives: one
zero-order-hold step ``x_{k+1} = step(x_k, u_k)`` with the motor voltage held
constant across the step (``numerics/integrators.py``, module docstring;
``dependency_rules.md`` §2 — ``dynamics`` may import ``physical``, ``core``,
``numerics``).

This is the ground-truth propagator: it integrates the full ``sin θ`` dynamics
(``model.md``), saturations included, and is what the simulator uses for
``SimulationResult.true_states``. Its small-angle behaviour must agree with the
:class:`~inverted_pendulum.dynamics.linearized_model.LinearizedPlantModel`,
which exposes the same ``step(state, voltage)`` protocol so the two models are
interchangeable to the simulation layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..numerics.integrators import euler, rk4
from ..physical.pendulum import ReactionWheelPendulum

_INTEGRATORS = {"rk4": rk4, "euler": euler}


@dataclass(frozen=True)
class NonlinearPlantModel:
    """One-step ZOH integration of the nonlinear plant (``model.md``).

    Parameters
    ----------
    plant : ReactionWheelPendulum
        The physical plant whose ``state_derivative`` is integrated.
    dt : float
        Control sampling period Δt (s), > 0 — supplied by config
        (``numerical_standards.md`` preamble: no inline step sizes).
    method : str
        Integrator from ``numerics/integrators.py``: ``"rk4"`` (default,
        global error O(Δt⁴)) or ``"euler"`` (O(Δt), for comparison).
    """

    plant: ReactionWheelPendulum
    dt: float
    method: str = "rk4"

    def __post_init__(self) -> None:
        object.__setattr__(self, "dt", float(self.dt))
        if not isinstance(self.plant, ReactionWheelPendulum):
            raise TypeError("plant must be a ReactionWheelPendulum")
        if self.dt <= 0.0:
            raise ValueError("dt must be > 0")
        if self.method not in _INTEGRATORS:
            raise ValueError(
                f"method must be one of {sorted(_INTEGRATORS)}, got {self.method!r}"
            )

    def derivative(self, state, voltage: float) -> np.ndarray:
        """The vector field ``ẋ = f(x, u)`` being integrated.

        Delegates to :meth:`ReactionWheelPendulum.state_derivative` (the
        voltage-input nonlinear dynamics, motor saturations included).
        """
        return self.plant.state_derivative(state, voltage)

    def step(self, state, voltage: float) -> np.ndarray:
        """Advance one sampling period under zero-order-hold control.

        Integrates ``ẋ = f(x, u_k)`` from ``t_k`` to ``t_k + Δt`` with the
        voltage ``u_k`` held constant — the usage prescribed by
        ``numerics/integrators.py``. The dynamics are time-invariant, so the
        integrator's ``t`` argument is immaterial and passed as 0.
        """
        integrate = _INTEGRATORS[self.method]
        u_k = float(voltage)
        return integrate(
            lambda _t, x: self.plant.state_derivative(x, u_k), 0.0, state, self.dt
        )
