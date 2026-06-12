"""Fixed-step explicit ODE integrators.

Pure-NumPy single-step methods for the initial-value problem ``dx/dt = f(t, x)``
(``numerical_standards.md`` §3, "ODE integrators"). The simulation layer uses
these to advance the nonlinear plant under a zero-order-hold control: it wraps
its dynamics as ``f(t, x) = plant_dynamics(x, u_k)`` with ``u_k`` held constant
across the step.

References
----------
Hairer, Nørsett & Wanner, *Solving Ordinary Differential Equations I*
(2nd ed.), §II.1 — the explicit Euler method and the classical four-stage,
fourth-order Runge-Kutta scheme.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# A vector field f(t, x) -> dx/dt, with x and the return value the same shape.
VectorField = Callable[[float, np.ndarray], np.ndarray]


def euler(f: VectorField, t: float, x, dt: float) -> np.ndarray:
    """One explicit (forward) Euler step of ``dx/dt = f(t, x)``.

    ``x_{n+1} = x_n + dt · f(t_n, x_n)``. First-order accurate: local
    truncation error ``O(dt²)``, global error ``O(dt)``. Hairer-Nørsett-Wanner
    §II.1; ``numerical_standards.md`` §3.

    Parameters
    ----------
    f : callable
        Vector field ``f(t, x) -> dx/dt``.
    t : float
        Current time.
    x : array_like
        Current state.
    dt : float
        Step size.

    Returns
    -------
    ndarray
        The state at ``t + dt`` (float64).
    """
    x = np.asarray(x, dtype=np.float64)
    return x + dt * np.asarray(f(t, x), dtype=np.float64)


def rk4(f: VectorField, t: float, x, dt: float) -> np.ndarray:
    """One step of the classical four-stage, fourth-order Runge-Kutta method.

    ::

        k1 = f(t,        x)
        k2 = f(t + dt/2, x + (dt/2) k1)
        k3 = f(t + dt/2, x + (dt/2) k2)
        k4 = f(t + dt,   x +  dt    k3)
        x_{n+1} = x_n + (dt/6)(k1 + 2 k2 + 2 k3 + k4)

    Global error ``O(dt⁴)``. Hairer-Nørsett-Wanner §II.1 (classical RK4
    Butcher tableau); ``numerical_standards.md`` §3.

    Parameters
    ----------
    f : callable
        Vector field ``f(t, x) -> dx/dt``.
    t : float
        Current time.
    x : array_like
        Current state.
    dt : float
        Step size.

    Returns
    -------
    ndarray
        The state at ``t + dt`` (float64).
    """
    x = np.asarray(x, dtype=np.float64)
    half = 0.5 * dt
    k1 = np.asarray(f(t, x), dtype=np.float64)
    k2 = np.asarray(f(t + half, x + half * k1), dtype=np.float64)
    k3 = np.asarray(f(t + half, x + half * k2), dtype=np.float64)
    k4 = np.asarray(f(t + dt, x + dt * k3), dtype=np.float64)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
