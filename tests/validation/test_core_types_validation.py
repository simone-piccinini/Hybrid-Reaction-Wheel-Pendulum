"""Validation of StateSpaceModel.discretize against scipy.signal.cont2discrete.

Per numerical_standards.md §8: zero-order-hold discretisation is cross-checked
against the reference (method='zoh') to VALIDATION_RTOL. Only tests/validation/
imports the reference library.
"""

import numpy as np
import pytest
from scipy.signal import cont2discrete

from inverted_pendulum.core.types import StateSpaceModel
from inverted_pendulum.numerics.constants import ATOL, VALIDATION_RTOL


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_zoh_discretization_matches_scipy(seed):
    rng = np.random.default_rng(seed)
    n_x, n_u, n_y = 4, 1, 2
    A = rng.standard_normal((n_x, n_x))
    B = rng.standard_normal((n_x, n_u))
    C = rng.standard_normal((n_y, n_x))
    D = np.zeros((n_y, n_u))
    dt = 0.01

    d = StateSpaceModel(A, B, C, D).discretize(dt)
    A_ref, B_ref, C_ref, D_ref, _ = cont2discrete((A, B, C, D), dt, method="zoh")
    assert np.allclose(d.A, A_ref, rtol=VALIDATION_RTOL, atol=ATOL)
    assert np.allclose(d.B, B_ref, rtol=VALIDATION_RTOL, atol=ATOL)
