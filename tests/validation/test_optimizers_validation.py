"""Validation of numerics.optimizers against scipy.optimize.

Per numerical_standards.md §8, the optimiser is cross-checked against
``scipy.optimize.minimize``. An iterative optimiser's accuracy is governed by
its gradient tolerance, not VALIDATION_RTOL, so both optimisers are driven to a
tight gradient tolerance here; their minimisers then agree to VALIDATION_RTOL.
Only tests/validation/ imports the reference library.
"""

import numpy as np
import pytest
from scipy.optimize import minimize as scipy_minimize

from inverted_pendulum.numerics.constants import ATOL, VALIDATION_RTOL
from inverted_pendulum.numerics.optimizers import minimize

TIGHT = 1e-9  # gradient tolerance for both optimisers in these comparisons


def rosenbrock():
    fun = lambda x: 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2  # noqa: E731
    grad = lambda x: np.array(  # noqa: E731
        [
            -400.0 * x[0] * (x[1] - x[0] ** 2) - 2.0 * (1.0 - x[0]),
            200.0 * (x[1] - x[0] ** 2),
        ]
    )
    return fun, grad


def test_quadratic_matches_scipy_and_analytic():
    rng = np.random.default_rng(5)
    n = 4
    root = rng.standard_normal((n, n))
    A = root @ root.T + n * np.eye(n)  # SPD
    x_star = rng.standard_normal(n)
    fun = lambda x: 0.5 * (x - x_star) @ A @ (x - x_star)  # noqa: E731
    grad = lambda x: A @ (x - x_star)  # noqa: E731

    ours = minimize(fun, np.zeros(n), grad, grad_tol=TIGHT, max_iter=500)
    ref = scipy_minimize(fun, np.zeros(n), jac=grad, method="BFGS",
                         options={"gtol": TIGHT, "maxiter": 500})
    assert np.allclose(ours.x, x_star, rtol=VALIDATION_RTOL, atol=1e-6)
    assert np.allclose(ours.x, ref.x, rtol=VALIDATION_RTOL, atol=1e-6)


def test_rosenbrock_matches_scipy():
    fun, grad = rosenbrock()
    x0 = np.array([-1.2, 1.0])
    ours = minimize(fun, x0, grad, grad_tol=TIGHT, max_iter=2000)
    ref = scipy_minimize(fun, x0, jac=grad, method="BFGS",
                         options={"gtol": TIGHT, "maxiter": 2000})
    assert ours.converged
    assert np.allclose(ours.x, [1.0, 1.0], atol=1e-6)
    assert np.allclose(ours.x, ref.x, rtol=VALIDATION_RTOL, atol=1e-6)
