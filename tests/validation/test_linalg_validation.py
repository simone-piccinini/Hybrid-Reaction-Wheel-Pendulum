"""Validation of numerics.linalg against reference libraries.

Per numerical_standards.md §8, every primitive with a known reference is
cross-checked here to ``VALIDATION_RTOL``. Tests under ``tests/validation/``
are the *only* place permitted to import the banned solver libraries
(``numpy.linalg``, ``scipy.linalg``); nothing under ``src/`` imports them.
"""

import numpy as np
import numpy.linalg as npl
import pytest
from scipy.linalg import solve_triangular

from inverted_pendulum.numerics import linalg
from inverted_pendulum.numerics.constants import ATOL, VALIDATION_RTOL


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(7)


def random_spd(n: int, rng: np.random.Generator) -> np.ndarray:
    A = rng.standard_normal((n, n))
    return A @ A.T + n * np.eye(n)


@pytest.mark.parametrize("n", [2, 4, 9])
def test_cholesky_matches_numpy(rng, n):
    M = random_spd(n, rng)
    L_ours = linalg.cholesky(M)
    L_ref = npl.cholesky(M)  # lower-triangular, same convention
    assert np.allclose(L_ours, L_ref, rtol=VALIDATION_RTOL, atol=ATOL)


def test_triangular_solves_match_scipy(rng):
    n = 8
    L = linalg.cholesky(random_spd(n, rng))
    b = rng.standard_normal(n)
    assert np.allclose(
        linalg.solve_lower(L, b),
        solve_triangular(L, b, lower=True),
        rtol=VALIDATION_RTOL,
        atol=ATOL,
    )
    U = L.T
    assert np.allclose(
        linalg.solve_upper(U, b),
        solve_triangular(U, b, lower=False),
        rtol=VALIDATION_RTOL,
        atol=ATOL,
    )


def test_chol_solve_matches_numpy_solve(rng):
    n = 10
    M = random_spd(n, rng)
    b = rng.standard_normal(n)
    assert np.allclose(
        linalg.chol_solve(M, b), npl.solve(M, b), rtol=VALIDATION_RTOL, atol=ATOL
    )


def test_lu_solve_matches_numpy_solve(rng):
    n = 10
    A = rng.standard_normal((n, n))
    b = rng.standard_normal(n)
    assert np.allclose(
        linalg.lu_solve_matrix(A, b), npl.solve(A, b), rtol=VALIDATION_RTOL, atol=ATOL
    )


def test_lu_solve_matrix_rhs_matches_numpy_solve(rng):
    n, m = 9, 5
    A = rng.standard_normal((n, n))
    B = rng.standard_normal((n, m))
    assert np.allclose(
        linalg.lu_solve_matrix(A, B), npl.solve(A, B), rtol=VALIDATION_RTOL, atol=ATOL
    )
