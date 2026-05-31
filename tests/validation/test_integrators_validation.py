"""Validation of numerics.integrators against scipy.integrate.solve_ivp.

Per numerical_standards.md §8, the integrators are cross-checked against the
reference to VALIDATION_RTOL. ``solve_ivp`` is run at very tight tolerances so
it serves as ground truth; the fixed step ``dt`` of each method is chosen small
enough that its global error falls below VALIDATION_RTOL (Euler O(dt) needs a
much smaller step than RK4 O(dt^4)). Only tests/validation/ imports the
reference library.
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from inverted_pendulum.numerics.constants import ATOL, VALIDATION_RTOL
from inverted_pendulum.numerics.integrators import euler, rk4


def ground_truth(fun, x0, T):
    """Accurate reference state at time ``T`` via adaptive RK45 (tight tols)."""
    sol = solve_ivp(fun, (0.0, T), np.asarray(x0, float),
                    rtol=1e-12, atol=1e-13, t_eval=[T])
    return sol.y[:, -1]


def roll_final(method, f, x0, dt, T):
    x = np.asarray(x0, dtype=np.float64)
    t = 0.0
    n = round(T / dt)
    for _ in range(n):
        x = method(f, t, x, dt)
        t += dt
    return x


def test_rk4_linear_oscillator_matches_scipy():
    omega = 2.0
    f = lambda t, x: np.array([x[1], -(omega**2) * x[0]])  # noqa: E731
    x0, T, dt = [1.0, 0.0], 2.0, 5e-3
    ours = roll_final(rk4, f, x0, dt, T)
    ref = ground_truth(f, x0, T)
    assert np.allclose(ours, ref, rtol=VALIDATION_RTOL, atol=ATOL)


def test_rk4_nonlinear_pendulum_matches_scipy():
    # Undamped nonlinear pendulum, no closed form -> scipy is the reference.
    f = lambda t, x: np.array([x[1], -np.sin(x[0])])  # noqa: E731
    x0, T, dt = [0.5, 0.0], 3.0, 2e-3
    ours = roll_final(rk4, f, x0, dt, T)
    ref = ground_truth(f, x0, T)
    assert np.allclose(ours, ref, rtol=VALIDATION_RTOL, atol=ATOL)


def test_euler_linear_decay_matches_scipy():
    # Euler is O(dt): use a short horizon and small step so error < 1e-6.
    k = 0.5
    f = lambda t, x: -k * x  # noqa: E731
    x0, T, dt = [1.0], 0.05, 5e-5
    ours = roll_final(euler, f, x0, dt, T)
    ref = ground_truth(f, x0, T)
    assert np.allclose(ours, ref, rtol=VALIDATION_RTOL, atol=ATOL)
