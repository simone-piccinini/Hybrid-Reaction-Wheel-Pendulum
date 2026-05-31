"""Unit tests for numerics.riccati (discrete-time DARE).

Properties only (DARE residual, scalar closed form, duality, error paths).
Residuals use the project's own chol_solve, not a reference solver; the scipy
cross-check lives in tests/validation/test_riccati_validation.py.

Accuracy note: the DARE is solved by the *value iteration* mandated by lqr.md /
numerical_standards.md §5, which is only *linearly* convergent. At its step gate
(RTOL = 1e-6 relative) the solution error is ≈ step/(1−ρ) ≈ a few ×1e-6 relative,
so tolerances here are set to that achievable accuracy, not machine precision.
"""

import math

import numpy as np
import pytest

from inverted_pendulum.numerics.linalg import NotPositiveDefiniteError, chol_solve
from inverted_pendulum.numerics.riccati import (
    DareSolution,
    RiccatiNotConverged,
    solve_dare,
)


def _relnorm(M, P):
    """‖M‖∞ / ‖P‖∞ (max-abs), a scale-free residual measure."""
    return float(np.max(np.abs(M))) / float(np.max(np.abs(P)))


def dare_residual(A, B, Q, R, P):
    """P − [Q + AᵀPA − AᵀPB (R+BᵀPB)⁻¹ BᵀPA], using the project's own SPD solve."""
    S = R + B.T @ P @ B
    return P - (Q + A.T @ P @ A - A.T @ P @ B @ chol_solve(S, B.T @ P @ A))


def test_dare_scalar_closed_form():
    # a=b=q=r=1: P²−P−1=0 ⇒ P = (1+√5)/2 (golden ratio); K = P/(1+P) = 1/φ.
    sol = solve_dare([[1.0]], [[1.0]], [[1.0]], [[1.0]])
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    assert np.allclose(sol.P, [[phi]], rtol=1e-5)
    assert np.allclose(sol.K, [[phi / (1.0 + phi)]], rtol=1e-5)
    assert abs(1.0 - sol.K[0, 0]) < 1.0  # closed loop |A−BK| < 1 (stable)


def test_dare_residual_on_unstable_discrete_plant():
    A = np.array([[1.1, 0.1], [0.0, 1.05]])  # eigenvalues outside unit circle
    B = np.array([[0.0], [0.1]])
    Q = np.eye(2)
    R = np.array([[1.0]])
    sol = solve_dare(A, B, Q, R)
    assert _relnorm(dare_residual(A, B, Q, R, sol.P), sol.P) < 1e-5
    assert np.allclose(sol.P, sol.P.T, atol=1e-10)
    assert sol.K.shape == (1, 2)
    assert 1 <= sol.iterations < 1000


def test_dare_duality_gives_estimator_solution():
    # Same solver with (A→Aᵀ, B→Cᵀ, Q→W, R→V) solves the estimator DARE
    # P = W + A P Aᵀ − A P Cᵀ (V + C P Cᵀ)⁻¹ C P Aᵀ.
    A = np.array([[1.1, 0.1], [0.0, 0.95]])
    C = np.array([[1.0, 0.0]])
    W = np.eye(2)
    V = np.array([[0.25]])
    sol = solve_dare(A.T, C.T, W, V)
    P = sol.P
    resid = P - (W + A @ P @ A.T - A @ P @ C.T @ chol_solve(V + C @ P @ C.T, C @ P @ A.T))
    assert _relnorm(resid, P) < 1e-5


def test_dare_rejects_non_pd_R():
    A = np.array([[1.1, 0.1], [0.0, 1.05]])
    B = np.array([[0.0], [0.1]])
    with pytest.raises(NotPositiveDefiniteError):
        solve_dare(A, B, np.eye(2), np.array([[0.0]]))


def test_dare_rejects_asymmetric_Q():
    A = np.array([[1.1, 0.1], [0.0, 1.05]])
    B = np.array([[0.0], [0.1]])
    Q = np.array([[1.0, 0.5], [0.0, 1.0]])  # asymmetric
    with pytest.raises(ValueError):
        solve_dare(A, B, Q, np.array([[1.0]]))


def test_dare_raises_when_not_converged():
    A = np.array([[1.1, 0.1], [0.0, 1.05]])
    B = np.array([[0.0], [0.1]])
    with pytest.raises(RiccatiNotConverged):
        solve_dare(A, B, np.eye(2), np.array([[1.0]]), max_iter=1)


def test_dare_returns_namedtuple():
    sol = solve_dare([[1.0]], [[1.0]], [[1.0]], [[1.0]])
    assert isinstance(sol, DareSolution)
    assert sol.P.dtype == np.float64 and sol.K.dtype == np.float64
