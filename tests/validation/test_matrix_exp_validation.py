"""Validation of numerics.matrix_exp against scipy.linalg.expm.

Per numerical_standards.md §8, the matrix-exponential primitive is cross-checked
against the reference to VALIDATION_RTOL. Only tests/validation/ may import the
banned reference libraries; nothing under src/ does.
"""

import numpy as np
import pytest
from scipy.linalg import expm as expm_ref

from inverted_pendulum.numerics.constants import ATOL, VALIDATION_RTOL
from inverted_pendulum.numerics.matrix_exp import expm


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(2024)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
def test_matches_scipy_small_norm(rng, n):
    A = rng.standard_normal((n, n))
    assert np.allclose(expm(A), expm_ref(A), rtol=VALIDATION_RTOL, atol=ATOL)


def test_matches_scipy_large_norm_exercises_scaling(rng):
    # Norm well above θ_13 so the scaling-and-squaring path runs.
    A = 10.0 * rng.standard_normal((6, 6))
    assert np.allclose(expm(A), expm_ref(A), rtol=VALIDATION_RTOL, atol=ATOL)


def test_matches_scipy_nonnormal_triangular(rng):
    A = np.triu(rng.standard_normal((5, 5)))
    assert np.allclose(expm(A), expm_ref(A), rtol=VALIDATION_RTOL, atol=ATOL)


def test_block_augmented_discretization(rng):
    # Dynamics-layer use: expm([[A, B], [0, 0]] * dt) has top-left A_d and
    # top-right B_d (model.md, "Discretise"). Verify the whole block matches.
    nx, nu, dt = 3, 1, 0.01
    A = rng.standard_normal((nx, nx))
    B = rng.standard_normal((nx, nu))
    M = np.zeros((nx + nu, nx + nu))
    M[:nx, :nx] = A
    M[:nx, nx:] = B
    assert np.allclose(expm(M * dt), expm_ref(M * dt), rtol=VALIDATION_RTOL, atol=ATOL)
