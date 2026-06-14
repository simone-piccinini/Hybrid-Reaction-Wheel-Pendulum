"""Validation of dynamics.frequency_response against reference libraries.

The hand-written transfer function G(s) = C(sI−A)⁻¹B + D — whose complex solve
goes through the project's real LU via the 2n embedding — is cross-checked
against a numpy.linalg.inv reference (continuous and discrete) and against
scipy.signal for a SISO channel. Only tests/validation/ may import the banned
reference libraries.
"""

import numpy as np
import numpy.linalg as npl
import pytest
from scipy import signal

from inverted_pendulum.core.types import StateSpaceModel
from inverted_pendulum.dynamics.frequency_response import (
    bode,
    log_frequencies,
    transfer_function,
)
from inverted_pendulum.numerics.constants import ATOL, VALIDATION_RTOL
from inverted_pendulum.physical.motor import DCMotor
from inverted_pendulum.physical.pendulum import ReactionWheelPendulum
from inverted_pendulum.physical.wheel import ReactionWheel


def pendulum_model():
    plant = ReactionWheelPendulum(
        pendulum_mass=0.3, pendulum_length=0.15, body_inertia=0.02,
        pivot_friction=0.01,
        wheel=ReactionWheel(mass=0.1, radius=0.05, inertia=1e-4,
                            friction_coefficient=1e-4),
        motor=DCMotor(resistance=2.0, torque_constant=0.05,
                      back_emf_constant=0.05),
    )
    return plant.linearize()


def test_continuous_matches_numpy_inverse():
    model = pendulum_model()
    A, B, C, D = model.A, model.B, model.C, model.D
    omega = log_frequencies(1e-2, 1e3, 40)
    G = transfer_function(model, omega)
    for k, w in enumerate(omega):
        ref = C @ npl.inv(1j * w * np.eye(model.n_x) - A) @ B + D
        assert np.allclose(G[k], ref, rtol=VALIDATION_RTOL, atol=ATOL)


def test_discrete_matches_numpy_inverse():
    model = pendulum_model().discretize(0.01)
    A, B, C, D = model.A, model.B, model.C, model.D
    omega = log_frequencies(1e-2, 100.0, 30)  # below Nyquist π/dt ≈ 314 rad/s
    G = transfer_function(model, omega)
    for k, w in enumerate(omega):
        z = np.exp(1j * w * model.dt)
        ref = C @ npl.inv(z * np.eye(model.n_x) - A) @ B + D
        assert np.allclose(G[k], ref, rtol=VALIDATION_RTOL, atol=ATOL)


def test_bode_matches_scipy_signal_siso():
    # extract the voltage -> theta_p SISO channel and compare to scipy.signal
    model = pendulum_model()
    C_siso = model.C[0:1, :]  # first output row
    sys = signal.StateSpace(model.A, model.B, C_siso, model.D[0:1, :])
    omega = log_frequencies(1e-1, 1e2, 50)
    _, mag_ref, phase_ref = signal.bode(sys, w=omega)  # dB, degrees
    data = bode(model, omega, output_index=0, input_index=0)
    assert np.allclose(data.magnitude_db, mag_ref, rtol=VALIDATION_RTOL, atol=1e-6)
    # scipy wraps phase to (−180, 180]; compare as wrapped angles
    ours = np.angle(np.exp(1j * np.radians(data.phase_deg)))
    theirs = np.angle(np.exp(1j * np.radians(phase_ref)))
    assert np.allclose(ours, theirs, atol=1e-6)
