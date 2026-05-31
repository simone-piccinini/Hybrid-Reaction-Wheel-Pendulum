"""Unit tests for core.types (StateSpaceModel) and core.constants."""

import dataclasses
import math

import numpy as np
import pytest

from inverted_pendulum.core import constants
from inverted_pendulum.core.types import StateSpaceModel


def simple_model(is_discrete=False, dt=None):
    A = np.array([[0.0, 1.0], [2.0, -0.5]])
    B = np.array([[0.0], [1.0]])
    C = np.array([[1.0, 0.0]])
    D = np.array([[0.0]])
    return StateSpaceModel(A, B, C, D, is_discrete=is_discrete, dt=dt)


# --------------------------------------------------------------------------- #
# construction / dimensions / dtype
# --------------------------------------------------------------------------- #
def test_dimensions_and_dtype():
    m = simple_model()
    assert (m.n_x, m.n_u, m.n_y) == (2, 1, 1)
    for arr in (m.A, m.B, m.C, m.D):
        assert arr.dtype == np.float64
    assert m.is_discrete is False and m.dt is None


def test_arrays_are_read_only_and_copied():
    A = np.array([[0.0, 1.0], [2.0, -0.5]])
    m = StateSpaceModel(A, [[0.0], [1.0]], [[1.0, 0.0]], [[0.0]])
    with pytest.raises(ValueError):  # writeable=False
        m.A[0, 0] = 9.0
    A[0, 0] = 9.0  # mutating the original must not affect the model's copy
    assert m.A[0, 0] == 0.0


def test_frozen_instance():
    m = simple_model()
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.A = np.eye(2)


# --------------------------------------------------------------------------- #
# shape / consistency validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "A,B,C,D",
    [
        (np.ones((2, 3)), np.ones((2, 1)), np.ones((1, 2)), np.ones((1, 1))),  # A not square
        (np.eye(2), np.ones((3, 1)), np.ones((1, 2)), np.ones((1, 1))),  # B rows != n_x
        (np.eye(2), np.ones((2, 1)), np.ones((1, 3)), np.ones((1, 1))),  # C cols != n_x
        (np.eye(2), np.ones((2, 1)), np.ones((1, 2)), np.ones((1, 2))),  # D wrong shape
    ],
)
def test_shape_validation(A, B, C, D):
    with pytest.raises(ValueError):
        StateSpaceModel(A, B, C, D)


def test_discrete_requires_positive_dt():
    with pytest.raises(ValueError):
        simple_model(is_discrete=True, dt=None)
    with pytest.raises(ValueError):
        simple_model(is_discrete=True, dt=-0.01)


def test_continuous_must_not_set_dt():
    with pytest.raises(ValueError):
        simple_model(is_discrete=False, dt=0.01)


# --------------------------------------------------------------------------- #
# discretize
# --------------------------------------------------------------------------- #
def test_discretize_scalar_closed_form():
    # ẋ = a x + b u  ->  A_d = e^{a dt},  B_d = b (e^{a dt} − 1) / a
    a, b, dt = 0.5, 2.0, 0.1
    m = StateSpaceModel([[a]], [[b]], [[1.0]], [[0.0]])
    d = m.discretize(dt)
    assert d.is_discrete and d.dt == dt
    assert np.allclose(d.A, [[math.exp(a * dt)]])
    assert np.allclose(d.B, [[b * (math.exp(a * dt) - 1.0) / a]])


def test_discretize_preserves_C_D_and_does_not_mutate_self():
    m = simple_model()
    d = m.discretize(0.02)
    assert np.allclose(d.C, m.C) and np.allclose(d.D, m.D)
    assert d.A.shape == (2, 2) and d.B.shape == (2, 1)
    assert m.is_discrete is False and m.dt is None  # original untouched


def test_discretize_rejects_bad_input():
    with pytest.raises(ValueError):
        simple_model().discretize(0.0)
    with pytest.raises(ValueError):
        simple_model(is_discrete=True, dt=0.01).discretize(0.01)


# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
def test_constants_reexported_and_domain_values():
    from inverted_pendulum.numerics.constants import ATOL as NUM_ATOL

    assert constants.ATOL == NUM_ATOL  # re-export matches the numerics source
    assert constants.GRAVITY == 9.81
    assert constants.PENALTY == 1e3
