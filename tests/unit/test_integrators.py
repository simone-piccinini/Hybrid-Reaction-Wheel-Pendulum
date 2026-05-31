"""Unit tests for numerics.integrators.

Closed-form solutions and convergence-order checks only (no reference library);
the scipy cross-check lives in tests/validation/test_integrators_validation.py.
"""

import numpy as np
import pytest

from inverted_pendulum.numerics.integrators import euler, rk4


def roll(method, f, x0, dt, n_steps, t0=0.0):
    """Integrate ``n_steps`` steps with a single-step ``method``; return states."""
    x = np.asarray(x0, dtype=np.float64)
    t = t0
    states = [x]
    for _ in range(n_steps):
        x = method(f, t, x, dt)
        t += dt
        states.append(x)
    return np.array(states)


# --------------------------------------------------------------------------- #
# single-step formulas
# --------------------------------------------------------------------------- #
def test_euler_single_step_formula():
    f = lambda t, x: 2.0 * np.ones_like(x)  # noqa: E731
    x0 = np.array([1.0, -3.0])
    out = euler(f, 0.0, x0, 0.1)
    assert np.allclose(out, x0 + 0.1 * 2.0)
    assert out.dtype == np.float64


def test_rk4_matches_simpson_on_state_independent_field():
    # f independent of x reduces RK4 to Simpson's rule.
    f = lambda t, x: np.array([1.0])  # noqa: E731  (x'(t)=1 -> x=t)
    out = rk4(f, 0.0, np.array([0.0]), 0.3)
    assert np.allclose(out, [0.3])


# --------------------------------------------------------------------------- #
# exactness on a polynomial solution (RK4 is order 4)
# --------------------------------------------------------------------------- #
def test_rk4_exact_on_cubic_solution():
    # x'(t) = 3 t^2  =>  x(t) = t^3.  RK4 (Simpson) integrates this exactly.
    f = lambda t, x: np.array([3.0 * t * t])  # noqa: E731
    traj = roll(rk4, f, np.array([0.0]), 0.1, 10)  # integrate to T = 1
    assert np.allclose(traj[-1], [1.0], atol=1e-12)


# --------------------------------------------------------------------------- #
# harmonic oscillator: analytic cos/sin
# --------------------------------------------------------------------------- #
def test_rk4_harmonic_oscillator_matches_analytic():
    omega = 2.0
    f = lambda t, x: np.array([x[1], -(omega**2) * x[0]])  # noqa: E731
    dt, n = 0.005, 400  # T = 2.0
    traj = roll(rk4, f, np.array([1.0, 0.0]), dt, n)
    t = np.linspace(0.0, dt * n, n + 1)
    analytic = np.stack([np.cos(omega * t), -omega * np.sin(omega * t)], axis=1)
    assert np.allclose(traj, analytic, atol=1e-7)


# --------------------------------------------------------------------------- #
# convergence order: Euler ~ O(dt), RK4 ~ O(dt^4)
# --------------------------------------------------------------------------- #
def _final_error(method, dt):
    k, T = 1.3, 1.0
    f = lambda t, x: -k * x  # noqa: E731
    x0 = np.array([2.0])
    exact = x0 * np.exp(-k * T)
    n = round(T / dt)
    xT = roll(method, f, x0, dt, n)[-1]
    return abs(xT[0] - exact[0])


def test_euler_is_first_order():
    ratio = _final_error(euler, 1e-2) / _final_error(euler, 5e-3)
    assert 1.7 < ratio < 2.3  # halving dt ~ halves the error


def test_rk4_is_fourth_order():
    ratio = _final_error(rk4, 2e-2) / _final_error(rk4, 1e-2)
    assert 12.0 < ratio < 20.0  # halving dt ~ /16


def test_state_shape_preserved():
    f = lambda t, x: x  # noqa: E731
    for method in (euler, rk4):
        out = method(f, 0.0, np.zeros((3,)), 0.01)
        assert out.shape == (3,)
