"""Unit tests for numerics.matrix_exp.

Properties and closed-form exponentials only (no reference library); the scipy
cross-check lives in tests/validation/test_matrix_exp_validation.py. Randomness
flows from a seeded Generator (determinism contract, §6).
"""

import math

import numpy as np
import pytest

from inverted_pendulum.numerics.matrix_exp import expm


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(123)


def test_zero_matrix_gives_identity():
    assert np.allclose(expm(np.zeros((4, 4))), np.eye(4))


def test_scalar_multiple_of_identity():
    # exp(cI) = e^c I
    c = 0.7
    assert np.allclose(expm(c * np.eye(3)), math.exp(c) * np.eye(3))


def test_diagonal_is_elementwise_exp():
    d = np.array([0.5, -1.3, 2.0])
    assert np.allclose(expm(np.diag(d)), np.diag(np.exp(d)))


def test_nilpotent_closed_form():
    # A^2 = 0  =>  exp(A) = I + A
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    assert np.allclose(expm(A), np.eye(2) + A)


def test_rotation_generator_closed_form():
    # exp([[0,-t],[t,0]]) = [[cos t, -sin t], [sin t, cos t]]
    t = 0.9
    A = np.array([[0.0, -t], [t, 0.0]])
    expected = np.array(
        [[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]]
    )
    assert np.allclose(expm(A), expected)


def test_inverse_identity(rng):
    # A and -A commute, so exp(A) exp(-A) = I.
    A = rng.standard_normal((5, 5))
    assert np.allclose(expm(A) @ expm(-A), np.eye(5), atol=1e-8)


def test_scaling_branch_triggers_for_large_norm():
    # ‖.‖_1 above θ_13 forces the squaring phase (s > 0).
    d = np.array([6.0, -6.0])  # ℓ1-norm 6 > 5.3719…
    assert np.allclose(expm(np.diag(d)), np.diag(np.exp(d)), rtol=1e-10)
    d2 = np.array([20.0, -3.0, 0.0])
    assert np.allclose(expm(np.diag(d2)), np.diag(np.exp(d2)), rtol=1e-10)


def test_dtype_and_shape_guards():
    assert expm(np.eye(2)).dtype == np.float64
    with pytest.raises(ValueError):
        expm(np.ones((2, 3)))
    with pytest.raises(ValueError):
        expm(np.array([[np.inf, 0.0], [0.0, 0.0]]))
