"""Unit tests for numerics.optimizers.

Properties only (convergence to known minima, restart selection, determinism,
non-convergence handling); the scipy cross-check lives in
tests/validation/test_optimizers_validation.py.
"""

import numpy as np
import pytest

from inverted_pendulum.numerics.constants import ML2_GRAD_TOL
from inverted_pendulum.numerics.optimizers import (
    OptimizeResult,
    minimize,
    minimize_with_restarts,
)


# ---- test objectives (with analytic gradients) ---------------------------- #
def quadratic(A, x_star):
    """f(x) = ½ (x-x*)ᵀ A (x-x*), grad = A (x-x*)."""
    fun = lambda x: 0.5 * (x - x_star) @ A @ (x - x_star)  # noqa: E731
    grad = lambda x: A @ (x - x_star)  # noqa: E731
    return fun, grad


def rosenbrock():
    fun = lambda x: 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2  # noqa: E731

    def grad(x):
        return np.array(
            [
                -400.0 * x[0] * (x[1] - x[0] ** 2) - 2.0 * (1.0 - x[0]),
                200.0 * (x[1] - x[0] ** 2),
            ]
        )

    return fun, grad


# --------------------------------------------------------------------------- #
def test_minimize_quadratic_reaches_minimum():
    A = np.array([[3.0, 0.5], [0.5, 2.0]])
    x_star = np.array([1.0, -2.0])
    fun, grad = quadratic(A, x_star)
    res = minimize(fun, np.zeros(2), grad)
    assert res.converged
    assert np.max(np.abs(res.grad)) < ML2_GRAD_TOL
    assert np.allclose(res.x, x_star, atol=1e-3)


def test_minimize_decreases_objective():
    A = np.array([[10.0, 0.0], [0.0, 1.0]])  # ill-conditioned bowl
    fun, grad = quadratic(A, np.array([2.0, 3.0]))
    x0 = np.array([-5.0, 5.0])
    res = minimize(fun, x0, grad)
    assert res.fun < fun(x0)
    assert res.converged


def test_minimize_rosenbrock():
    fun, grad = rosenbrock()
    res = minimize(fun, np.array([-1.2, 1.0]), grad, max_iter=500)
    assert res.converged
    assert np.allclose(res.x, [1.0, 1.0], atol=1e-2)


def test_initial_point_already_optimal():
    fun, grad = quadratic(np.eye(2), np.array([0.0, 0.0]))
    res = minimize(fun, np.zeros(2), grad)
    assert res.converged and res.n_iter == 0


def test_minimize_with_restarts_finds_global():
    # 1-D double well, global minimum near x ≈ -1.06 (deeper than x ≈ 0.92).
    fun = lambda x: (x[0] ** 2 - 1.0) ** 2 + 0.5 * x[0]  # noqa: E731
    grad = lambda x: np.array([4.0 * x[0] * (x[0] ** 2 - 1.0) + 0.5])  # noqa: E731
    rng = np.random.default_rng(0)
    sample = lambda r: r.uniform(-2.0, 2.0, size=1)  # noqa: E731
    res = minimize_with_restarts(fun, grad, sample, rng, n_restarts=8)
    assert res.x[0] < 0.0
    assert res.fun < -0.4  # deeper well; the shallow one sits near +0.48


def test_restarts_are_deterministic():
    fun = lambda x: (x[0] ** 2 - 1.0) ** 2 + 0.5 * x[0]  # noqa: E731
    grad = lambda x: np.array([4.0 * x[0] * (x[0] ** 2 - 1.0) + 0.5])  # noqa: E731
    sample = lambda r: r.uniform(-2.0, 2.0, size=1)  # noqa: E731
    a = minimize_with_restarts(fun, grad, sample, np.random.default_rng(42), n_restarts=5)
    b = minimize_with_restarts(fun, grad, sample, np.random.default_rng(42), n_restarts=5)
    assert np.array_equal(a.x, b.x) and a.fun == b.fun


def test_non_convergence_returns_best_and_warns():
    fun, grad = rosenbrock()
    with pytest.warns(UserWarning):
        res = minimize(fun, np.array([-1.2, 1.0]), grad, max_iter=1)
    assert not res.converged
    assert isinstance(res, OptimizeResult)
    assert np.all(np.isfinite(res.x))


def test_restarts_require_at_least_one():
    fun, grad = quadratic(np.eye(1), np.array([0.0]))
    with pytest.raises(ValueError):
        minimize_with_restarts(
            fun, grad, lambda r: r.uniform(-1, 1, 1), np.random.default_rng(0), n_restarts=0
        )
