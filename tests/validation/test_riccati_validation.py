"""Validation of numerics.riccati against scipy.linalg.solve_discrete_are.

Per numerical_standards.md §8: the DARE solver is cross-checked against the
reference (its gain derived with numpy.linalg). The DARE is solved by the
linearly-convergent value iteration mandated by lqr.md / §5, whose solution
error at the RTOL=1e-6 step gate is ≈ step/(1−ρ) ≈ a few ×1e-6 relative; the
comparison tolerance below (1e-4) reflects that achievable accuracy rather than
machine precision (cf. the analogous note for the iterative optimiser). Only
tests/validation/ imports the reference libraries.
"""

import numpy as np
import numpy.linalg as npl
import pytest
from scipy.linalg import solve_discrete_are

from inverted_pendulum.numerics.riccati import solve_dare

RICCATI_VALIDATION_RTOL = 1e-4  # see module docstring (linear value iteration)


def reference_gain(A, B, Q, R, P):
    """K = (R + BᵀPB)⁻¹ BᵀPA via the reference solver."""
    return npl.solve(R + B.T @ P @ B, B.T @ P @ A)


def test_dare_matches_scipy_unstable_plant():
    A = np.array([[1.1, 0.1], [0.0, 1.05]])  # open-loop unstable (discrete)
    B = np.array([[0.0], [0.1]])
    Q = np.eye(2)
    R = np.array([[1.0]])
    sol = solve_dare(A, B, Q, R)
    P_ref = solve_discrete_are(A, B, Q, R)
    assert np.allclose(sol.P, P_ref, rtol=RICCATI_VALIDATION_RTOL)
    assert np.allclose(sol.K, reference_gain(A, B, Q, R, P_ref), rtol=RICCATI_VALIDATION_RTOL)


def test_dare_scalar_matches_scipy():
    A, B, Q, R = [[1.2]], [[0.5]], [[2.0]], [[1.0]]
    sol = solve_dare(A, B, Q, R)
    P_ref = solve_discrete_are(np.array(A), np.array(B), np.array(Q), np.array(R))
    assert np.allclose(sol.P, P_ref, rtol=RICCATI_VALIDATION_RTOL)


@pytest.mark.parametrize("seed", [0, 3, 7])
def test_dare_matches_scipy_random_mimo(seed):
    r = np.random.default_rng(seed)
    n_x, n_u = 4, 2
    A = 0.5 * r.standard_normal((n_x, n_x))  # moderate spectral radius
    B = r.standard_normal((n_x, n_u))
    Qroot = r.standard_normal((n_x, n_x))
    Q = Qroot @ Qroot.T + np.eye(n_x)  # SPD
    R = np.diag(np.exp(r.standard_normal(n_u)))  # SPD diagonal
    sol = solve_dare(A, B, Q, R)
    P_ref = solve_discrete_are(A, B, Q, R)
    assert np.allclose(sol.P, P_ref, rtol=RICCATI_VALIDATION_RTOL)
    assert np.allclose(sol.K, reference_gain(A, B, Q, R, P_ref), rtol=RICCATI_VALIDATION_RTOL)
