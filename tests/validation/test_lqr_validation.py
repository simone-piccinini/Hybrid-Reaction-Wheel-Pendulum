"""Validation of the LQR design against scipy/numpy references.

The controller's gain is cross-checked against scipy.linalg.solve_discrete_are
on the actual discretised plant, at the tolerance achievable by the
linearly-convergent value iteration (see test_riccati_validation.py); the
closed-loop spectrum and stored spectral radius are checked against
numpy.linalg.eigvals. Only tests/validation/ imports the reference libraries.
"""

import numpy as np
import numpy.linalg as npl
from scipy.linalg import solve_discrete_are

from inverted_pendulum.control.lqr_controller import LQRController
from inverted_pendulum.dynamics.linearized_model import LinearizedPlantModel
from inverted_pendulum.numerics.constants import VALIDATION_RTOL
from inverted_pendulum.physical.motor import DCMotor
from inverted_pendulum.physical.pendulum import ReactionWheelPendulum
from inverted_pendulum.physical.wheel import ReactionWheel

# The value iteration's tail converges as ρ(A−BK)² per sweep; this plant's
# slowest closed-loop pole is ≈0.991 (the weakly-weighted wheel angle), so the
# §5 step gate leaves a ~2e-4 relative solution error — looser than the 1e-4
# of test_riccati_validation's better-conditioned cases.
LQR_VALIDATION_RTOL = 1e-3

DT = 0.01
Q_LQR = np.diag([20.0, 2.0, 1e-2, 1e-2])
R_LQR = np.array([[1.0]])


def make_design():
    plant = ReactionWheelPendulum(
        pendulum_mass=0.3, pendulum_length=0.15, body_inertia=0.02,
        pivot_friction=0.01,
        wheel=ReactionWheel(mass=0.1, radius=0.05, inertia=1e-4,
                            friction_coefficient=1e-4),
        motor=DCMotor(resistance=2.0, torque_constant=0.05,
                      back_emf_constant=0.05),
    )
    linear = LinearizedPlantModel.from_plant(plant, DT)
    return linear, LQRController.from_model(linear.discrete, Q_LQR, R_LQR)


def test_gain_matches_scipy_dare():
    linear, c = make_design()
    A, B = linear.A_d, linear.B_d
    P_ref = solve_discrete_are(A, B, Q_LQR, R_LQR)
    K_ref = npl.solve(R_LQR + B.T @ P_ref @ B, B.T @ P_ref @ A)
    assert np.allclose(c.P_dare, P_ref, rtol=LQR_VALIDATION_RTOL)
    assert np.allclose(c.K_gain, K_ref, rtol=LQR_VALIDATION_RTOL)


def test_closed_loop_spectrum_inside_unit_circle():
    linear, c = make_design()
    eigs = npl.eigvals(linear.A_d - linear.B_d @ c.K_gain)
    assert np.max(np.abs(eigs)) < 1.0
    np.testing.assert_allclose(
        c.closed_loop_spectral_radius, np.max(np.abs(eigs)), rtol=VALIDATION_RTOL
    )
