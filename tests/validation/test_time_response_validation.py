"""Validation of dynamics.time_response against scipy.signal.

The hand-written step and free responses (built on the project's own expm /
ZOH discretisation) are cross-checked against scipy.signal on stable test
systems. Only tests/validation/ may import the banned reference libraries.
"""

import numpy as np
from scipy import signal

from inverted_pendulum.core.types import StateSpaceModel
from inverted_pendulum.dynamics.time_response import (
    free_response,
    step_response,
)
from inverted_pendulum.numerics.constants import ATOL, VALIDATION_RTOL


def damped_second_order(wn=4.0, zeta=0.5):
    A = [[0.0, 1.0], [-wn**2, -2.0 * zeta * wn]]
    B = [[0.0], [wn**2]]
    C = [[1.0, 0.0]]
    return StateSpaceModel(A, B, C, [[0.0]])


def test_step_response_matches_scipy():
    model = damped_second_order()
    dt = 1e-3
    time, y = step_response(model, t_end=6.0, dt=dt)
    sys = signal.StateSpace(model.A, model.B, model.C, model.D)
    _, y_ref = signal.step(sys, T=time)
    assert np.allclose(y[:, 0], y_ref, rtol=VALIDATION_RTOL, atol=1e-4)


def test_free_response_matches_scipy_lsim():
    # initial-condition response = zero-input lsim with X0
    model = damped_second_order()
    dt = 1e-3
    x0 = np.array([1.0, 0.0])
    time, states, outputs = free_response(model, x0, t_end=6.0, dt=dt)
    sys = signal.StateSpace(model.A, model.B, model.C, model.D)
    _, y_ref, _ = signal.lsim(sys, U=np.zeros_like(time), T=time, X0=x0)
    assert np.allclose(outputs[:, 0], y_ref, rtol=VALIDATION_RTOL, atol=1e-4)


def test_free_response_states_match_matrix_exponential_reference():
    # x(t_k) = expm(A t_k) x0, checked against scipy.linalg.expm directly
    from scipy.linalg import expm as expm_ref

    model = damped_second_order(wn=3.0, zeta=0.3)
    x0 = np.array([0.5, -1.0])
    dt = 1e-2
    time, states, _ = free_response(model, x0, t_end=2.0, dt=dt)
    A = np.asarray(model.A)
    for k, t in enumerate(time):
        ref = expm_ref(A * t) @ x0
        assert np.allclose(states[k], ref, rtol=VALIDATION_RTOL, atol=ATOL)
