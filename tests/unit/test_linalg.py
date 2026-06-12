"""Unit tests for numerics.linalg.

These assert the mathematical *properties* of each primitive (reconstruction,
triangularity, residuals, error handling) without importing any reference
solver — that cross-check lives in tests/validation/test_linalg_validation.py.
All randomness flows from a seeded Generator (determinism contract, §6).
"""

import numpy as np
import pytest

from inverted_pendulum.numerics import linalg
from inverted_pendulum.numerics.constants import SYM_TOL


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20240530)


def random_spd(n: int, rng: np.random.Generator) -> np.ndarray:
    """A well-conditioned symmetric positive-definite matrix."""
    A = rng.standard_normal((n, n))
    return A @ A.T + n * np.eye(n)


# --------------------------------------------------------------------------- #
# symmetry helpers
# --------------------------------------------------------------------------- #
def test_symmetrize_produces_symmetric(rng):
    A = rng.standard_normal((4, 4))
    S = linalg.symmetrize(A)
    assert np.array_equal(S, S.T)
    assert linalg.is_symmetric(S)
    assert not linalg.is_symmetric(A)  # a generic Gaussian matrix is asymmetric


def test_is_symmetric_respects_tolerance():
    A = np.eye(3)
    A[0, 1] = SYM_TOL / 2
    assert linalg.is_symmetric(A)
    A[0, 1] = 10 * SYM_TOL
    assert not linalg.is_symmetric(A)


# --------------------------------------------------------------------------- #
# cholesky
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [1, 2, 5, 8])
def test_cholesky_reconstructs_and_is_lower(rng, n):
    M = random_spd(n, rng)
    L = linalg.cholesky(M)
    assert L.dtype == np.float64
    assert np.allclose(np.triu(L, 1), 0.0)        # strictly upper part is zero
    assert np.all(np.diag(L) > 0.0)               # positive diagonal
    assert np.allclose(L @ L.T, M, rtol=1e-10, atol=1e-12)


def test_cholesky_raises_on_non_pd():
    M = np.array([[1.0, 2.0], [2.0, 1.0]])  # symmetric, eigenvalues 3 and -1
    with pytest.raises(linalg.NotPositiveDefiniteError):
        linalg.cholesky(M)


def test_cholesky_raises_on_asymmetric():
    M = np.array([[2.0, 1.0], [0.0, 2.0]])
    with pytest.raises(ValueError):
        linalg.cholesky(M)


def test_cholesky_rejects_negative_jitter(rng):
    with pytest.raises(ValueError):
        linalg.cholesky(random_spd(3, rng), jitter=-1e-9)


def test_cholesky_jitter_rescues_singular_psd():
    v = np.array([1.0, 2.0, 3.0])
    M = np.outer(v, v)  # rank-1 PSD => singular, Cholesky must fail
    with pytest.raises(linalg.NotPositiveDefiniteError):
        linalg.cholesky(M)
    L = linalg.cholesky(M, jitter=1e-6)
    assert np.allclose(L @ L.T, M + 1e-6 * np.eye(3), atol=1e-9)


# --------------------------------------------------------------------------- #
# triangular solves
# --------------------------------------------------------------------------- #
def test_solve_lower_and_upper_vector(rng):
    n = 6
    L = linalg.cholesky(random_spd(n, rng))
    x_true = rng.standard_normal(n)
    assert np.allclose(linalg.solve_lower(L, L @ x_true), x_true, atol=1e-10)
    U = L.T
    assert np.allclose(linalg.solve_upper(U, U @ x_true), x_true, atol=1e-10)


def test_triangular_solve_multiple_rhs(rng):
    n, m = 5, 3
    L = linalg.cholesky(random_spd(n, rng))
    X = rng.standard_normal((n, m))
    out = linalg.solve_lower(L, L @ X)
    assert out.shape == (n, m)
    assert np.allclose(out, X, atol=1e-10)


def test_triangular_solve_singular_raises():
    L = np.array([[1.0, 0.0], [2.0, 0.0]])  # zero diagonal entry
    with pytest.raises(linalg.SingularMatrixError):
        linalg.solve_lower(L, np.array([1.0, 2.0]))
    with pytest.raises(linalg.SingularMatrixError):
        linalg.solve_upper(L.T, np.array([1.0, 2.0]))


# --------------------------------------------------------------------------- #
# chol_solve
# --------------------------------------------------------------------------- #
def test_chol_solve_residual(rng):
    n = 7
    M = random_spd(n, rng)
    x_true = rng.standard_normal(n)
    x = linalg.chol_solve(M, M @ x_true)
    assert np.allclose(x, x_true, atol=1e-9)
    assert np.allclose(M @ x, M @ x_true, atol=1e-9)


# --------------------------------------------------------------------------- #
# LU
# --------------------------------------------------------------------------- #
def test_lu_factorisation_invariant(rng):
    n = 6
    A = rng.standard_normal((n, n))
    f = linalg.lu_factor(A)
    assert np.allclose(np.triu(f.L, 1), 0.0)      # L lower-triangular ...
    assert np.allclose(np.diag(f.L), 1.0)         # ... and unit diagonal
    assert np.allclose(np.tril(f.U, -1), 0.0)     # U upper-triangular
    assert np.allclose(A[f.perm], f.L @ f.U, atol=1e-12)  # P A = L U


def test_lu_solve_vector(rng):
    n = 6
    A = rng.standard_normal((n, n))
    x_true = rng.standard_normal(n)
    x = linalg.lu_solve_matrix(A, A @ x_true)
    assert np.allclose(x, x_true, atol=1e-9)


def test_lu_solve_multiple_rhs(rng):
    n, m = 5, 4
    A = rng.standard_normal((n, n))
    X = rng.standard_normal((n, m))
    out = linalg.lu_solve_matrix(A, A @ X)
    assert out.shape == (n, m)
    assert np.allclose(out, X, atol=1e-9)


def test_lu_pivots_on_zero_leading_entry():
    # Leading entry is zero: requires a row swap, which partial pivoting handles.
    A = np.array([[0.0, 1.0], [1.0, 0.0]])
    x_true = np.array([2.0, 3.0])
    assert np.allclose(linalg.lu_solve_matrix(A, A @ x_true), x_true, atol=1e-12)


def test_lu_singular_raises():
    A = np.array([[1.0, 2.0], [2.0, 4.0]])  # rank 1
    with pytest.raises(linalg.SingularMatrixError):
        linalg.lu_factor(A)


# --------------------------------------------------------------------------- #
# shape guards
# --------------------------------------------------------------------------- #
def test_non_square_raises():
    rect = np.ones((2, 3))
    with pytest.raises(ValueError):
        linalg.cholesky(rect)
    with pytest.raises(ValueError):
        linalg.lu_factor(rect)
    with pytest.raises(ValueError):
        linalg.eigvals(rect)


# --------------------------------------------------------------------------- #
# eigvals / spectral_radius (shifted QR)
# --------------------------------------------------------------------------- #
def sorted_eigs(values) -> np.ndarray:
    return np.sort_complex(np.asarray(values, dtype=np.complex128))


def test_eigvals_diagonal():
    d = np.array([3.0, -1.0, 0.5, 7.0])
    np.testing.assert_allclose(
        sorted_eigs(linalg.eigvals(np.diag(d))), sorted_eigs(d), atol=1e-12
    )


def test_eigvals_triangular(rng):
    T = np.triu(rng.standard_normal((5, 5)))
    np.testing.assert_allclose(
        sorted_eigs(linalg.eigvals(T)), sorted_eigs(np.diag(T)), atol=1e-10
    )


def test_eigvals_rotation_complex_pair():
    # planar rotation by ψ: eigenvalues cos ψ ± i sin ψ
    psi = 0.7
    R = np.array(
        [[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]]
    )
    expected = [np.cos(psi) + 1j * np.sin(psi), np.cos(psi) - 1j * np.sin(psi)]
    np.testing.assert_allclose(
        sorted_eigs(linalg.eigvals(R)), sorted_eigs(expected), atol=1e-12
    )


def test_eigvals_companion_matrix():
    # companion of (λ−1)(λ−2)(λ−3) = λ³ − 6λ² + 11λ − 6
    C = np.array([[6.0, -11.0, 6.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    np.testing.assert_allclose(
        sorted_eigs(linalg.eigvals(C)), sorted_eigs([1.0, 2.0, 3.0]), atol=1e-9
    )


def test_eigvals_cyclic_permutation_needs_exceptional_shift():
    # the 3-cycle permutation: eigenvalues are the cube roots of unity; the
    # plain Wilkinson shift stalls on it, exercising the exceptional shift
    P3 = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    expected = [np.exp(2j * np.pi * k / 3) for k in range(3)]
    np.testing.assert_allclose(
        sorted_eigs(linalg.eigvals(P3)), sorted_eigs(expected), atol=1e-10
    )


def test_eigvals_symmetric_are_real(rng):
    M = random_spd(5, rng)
    e = linalg.eigvals(M)
    assert np.max(np.abs(e.imag)) < 1e-9
    assert np.all(e.real > 0.0)  # SPD: strictly positive spectrum


@pytest.mark.parametrize("n", [1, 2, 3, 6])
def test_eigvals_sum_matches_trace(rng, n):
    A = rng.standard_normal((n, n))
    e = linalg.eigvals(A)
    assert e.shape == (n,) and e.dtype == np.complex128
    np.testing.assert_allclose(np.sum(e), np.trace(A), rtol=1e-9, atol=1e-9)


def test_eigvals_is_deterministic(rng):
    A = rng.standard_normal((5, 5))
    np.testing.assert_array_equal(linalg.eigvals(A), linalg.eigvals(A))


def test_eigvals_rejects_nonfinite():
    A = np.array([[1.0, np.nan], [0.0, 1.0]])
    with pytest.raises(ValueError):
        linalg.eigvals(A)


def test_eigvals_budget_exhaustion_raises(rng):
    A = rng.standard_normal((4, 4))
    with pytest.raises(linalg.EigNotConverged):
        linalg.eigvals(A, max_sweeps=0)


def test_spectral_radius_known_values():
    assert linalg.spectral_radius(0.5 * np.eye(3)) == pytest.approx(0.5)
    psi = 1.1  # rotations are isometries: ρ = 1
    R = np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])
    assert linalg.spectral_radius(R) == pytest.approx(1.0, abs=1e-12)
