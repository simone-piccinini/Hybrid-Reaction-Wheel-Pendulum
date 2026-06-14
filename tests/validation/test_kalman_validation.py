"""Validation of estimation.kalman_filter against filterpy and scipy.

numerical_standards.md §8: "kalman_filter → compare a short run to a filterpy
reference". The recursion is compared state-by-state and covariance-by-
covariance to filterpy (which uses the same Joseph-form update); the dual-DARE
steady state is compared to scipy.linalg.solve_discrete_are. Only
tests/validation/ imports the reference libraries.
"""

import numpy as np
from filterpy.kalman import KalmanFilter as FilterpyKF
from scipy.linalg import solve_discrete_are

from inverted_pendulum.core.types import StateSpaceModel
from inverted_pendulum.dynamics.linearized_model import LinearizedPlantModel
from inverted_pendulum.estimation.kalman_filter import (
    KalmanFilter,
    steady_state_kalman_gain,
)
from inverted_pendulum.numerics.constants import ATOL, VALIDATION_RTOL
from inverted_pendulum.physical.motor import DCMotor
from inverted_pendulum.physical.pendulum import ReactionWheelPendulum
from inverted_pendulum.physical.wheel import ReactionWheel

DT = 0.01


def make_plant_model() -> LinearizedPlantModel:
    plant = ReactionWheelPendulum(
        pendulum_mass=0.3, pendulum_length=0.15, body_inertia=0.02,
        pivot_friction=0.01,
        wheel=ReactionWheel(mass=0.1, radius=0.05, inertia=1e-4,
                            friction_coefficient=1e-4),
        motor=DCMotor(resistance=2.0, torque_constant=0.05,
                      back_emf_constant=0.05),
    )
    return LinearizedPlantModel.from_plant(plant, DT)


def test_recursion_matches_filterpy():
    m = make_plant_model()
    W = np.diag([1e-8, 1e-6, 1e-8, 1e-4])
    V = np.diag([1e-6, 1e-4])
    x0 = np.array([0.05, 0.0, 0.0, 0.0])
    P0 = 1e-2 * np.eye(4)

    ours = KalmanFilter(m.discrete, W, V, x0=x0, P0=P0)

    ref = FilterpyKF(dim_x=4, dim_z=2)
    ref.x = x0.copy()
    ref.F = np.asarray(m.A_d)
    ref.B = np.asarray(m.B_d)
    ref.H = np.asarray(m.C)
    ref.Q = W
    ref.R = V
    ref.P = P0.copy()

    rng = np.random.default_rng(2026)
    for _ in range(50):
        u = rng.standard_normal(1)
        z = rng.standard_normal(2)
        ours.step(u, z)
        ref.predict(u=u)
        ref.update(z)
        assert np.allclose(ours.x_hat, ref.x.ravel(),
                           rtol=VALIDATION_RTOL, atol=ATOL)
        assert np.allclose(ours.P_est, ref.P, rtol=VALIDATION_RTOL, atol=ATOL)


def test_steady_state_matches_scipy_dare():
    # detectable pair (full-state sensing) with O(1) covariances: the dual
    # DARE solution scales linearly with (W, V), and at this scale the value
    # iteration's §5 step gate is dominated by its relative term
    m = make_plant_model()
    model = StateSpaceModel(
        m.A_d, m.B_d, np.eye(4), np.zeros((4, 1)), is_discrete=True, dt=DT
    )
    W = np.diag([1.0, 2.0, 0.5, 3.0])
    V = 0.5 * np.eye(4)
    gain = steady_state_kalman_gain(model, W, V)
    P_ref = solve_discrete_are(model.A.T, model.C.T, W, V)
    assert np.allclose(gain.P_pred, P_ref, rtol=1e-4)
    L_ref = P_ref @ model.C.T @ np.linalg.inv(model.C @ P_ref @ model.C.T + V)
    assert np.allclose(gain.L_gain, L_ref, rtol=1e-4)
