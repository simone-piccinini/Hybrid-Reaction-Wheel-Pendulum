"""Unit tests for control.lqr_controller.LQRController (DARE design + control law)."""

import dataclasses

import numpy as np
import pytest

from inverted_pendulum.control.lqr_controller import (
    LQRController,
    UnstableClosedLoopError,
)
from inverted_pendulum.core.types import LQGConfig, StateSpaceModel
from inverted_pendulum.dynamics.linearized_model import LinearizedPlantModel
from inverted_pendulum.dynamics.nonlinear_model import NonlinearPlantModel
from inverted_pendulum.numerics.linalg import is_symmetric, spectral_radius
from inverted_pendulum.physical.motor import DCMotor
from inverted_pendulum.physical.pendulum import ReactionWheelPendulum
from inverted_pendulum.physical.wheel import ReactionWheel

DT = 0.01
# wheel-angle weight small but positive: zero leaves the wheel integrator on
# the unit circle (see test_zero_wheel_angle_weight_is_rejected)
Q_DIAG = np.array([20.0, 2.0, 1e-2, 1e-2])
R_LQR = np.array([[1.0]])


def make_plant():
    return ReactionWheelPendulum(
        pendulum_mass=0.3, pendulum_length=0.15, body_inertia=0.02,
        pivot_friction=0.01,
        wheel=ReactionWheel(mass=0.1, radius=0.05, inertia=1e-4,
                            friction_coefficient=1e-4),
        motor=DCMotor(resistance=2.0, torque_constant=0.05,
                      back_emf_constant=0.05),
    )


@pytest.fixture(scope="module")
def design():
    plant = make_plant()
    linear = LinearizedPlantModel.from_plant(plant, DT)
    controller = LQRController.from_model(linear.discrete, np.diag(Q_DIAG), R_LQR)
    return plant, linear, controller


# --------------------------------------------------------------------------- #
# gain design
# --------------------------------------------------------------------------- #
def test_design_shapes_and_symmetry(design):
    _, _, c = design
    assert c.K_gain.shape == (1, 4) and c.P_dare.shape == (4, 4)
    assert (c.n_x, c.n_u) == (4, 1)
    assert is_symmetric(c.P_dare)
    assert not c.K_gain.flags.writeable and not c.P_dare.flags.writeable


def test_dare_fixed_point_residual(design):
    # P = Q + AᵀPA − AᵀPB (R + BᵀPB)⁻¹ BᵀPA (lqr.md); n_u = 1 so the inner
    # inverse is a scalar division — no solver needed for the check
    _, linear, c = design
    A, B, P = linear.A_d, linear.B_d, c.P_dare
    Q = np.diag(Q_DIAG)
    s = float((R_LQR + B.T @ P @ B)[0, 0])
    residual = Q + A.T @ P @ A - (A.T @ P @ B) @ (B.T @ P @ A) / s - P
    # P entries reach ~1e6 (the weakly-weighted wheel angle is expensive to
    # regulate), so the residual is judged relative to ‖P‖
    assert np.max(np.abs(residual)) < 1e-6 * np.max(np.abs(P))


def test_gain_matches_converged_P(design):
    # K = (R + BᵀPB)⁻¹ BᵀPA read off the converged P (lqr.md, "Gain")
    _, linear, c = design
    A, B, P = linear.A_d, linear.B_d, c.P_dare
    s = float((R_LQR + B.T @ P @ B)[0, 0])
    np.testing.assert_allclose(c.K_gain, (B.T @ P @ A) / s, rtol=1e-9)


def test_closed_loop_is_stable(design):
    _, linear, c = design
    rho = spectral_radius(linear.A_d - linear.B_d @ c.K_gain)
    assert rho < 1.0
    assert c.closed_loop_spectral_radius == pytest.approx(rho)


def test_from_config_equals_from_model(design):
    _, linear, c = design
    config = LQGConfig(
        Q_lqr=np.diag(Q_DIAG),
        R_lqr=R_LQR,
        W_process=1e-3 * np.eye(4),
        V_measure=1e-2 * np.eye(2),
    )
    from_config = LQRController.from_config(linear.discrete, config)
    np.testing.assert_allclose(from_config.K_gain, c.K_gain)
    np.testing.assert_allclose(from_config.P_dare, c.P_dare)


# --------------------------------------------------------------------------- #
# control law
# --------------------------------------------------------------------------- #
def test_compute_control_is_minus_K_xhat(design):
    _, _, c = design
    x_hat = np.array([0.02, -0.1, 1.0, 5.0])
    u = c.compute_control(x_hat)
    assert u.shape == (1,)
    np.testing.assert_allclose(u, -(c.K_gain @ x_hat))


def test_zero_state_zero_control(design):
    _, _, c = design
    np.testing.assert_allclose(c.compute_control(np.zeros(4)), np.zeros(1))


def test_compute_control_rejects_bad_shape(design):
    _, _, c = design
    with pytest.raises(ValueError):
        c.compute_control(np.zeros(3))


# --------------------------------------------------------------------------- #
# closed-loop regulation — the layer's purpose
# --------------------------------------------------------------------------- #
def test_regulates_the_linear_plant(design):
    # the wheel states ride the slow ρ≈0.991 mode (the cheap wheel-angle
    # weight), so they settle last — thresholds reflect the two time scales
    _, linear, c = design
    x = np.array([0.05, 0.0, 0.0, 0.0])
    for _ in range(1000):
        x = linear.step(x, c.compute_control(x)[0])
    assert abs(x[0]) < 1e-5 and abs(x[1]) < 1e-5
    assert abs(x[3]) < 1e-2


def test_stabilizes_the_nonlinear_plant_from_a_tilt(design):
    # full-state feedback u = −K x on the true sin-theta plant: 0.05 rad tilt
    # regulated to ~1e-6 rad in 10 s (LQG separation: the Kalman estimate
    # replaces x once estimation/ exists)
    plant, _, c = design
    sim = NonlinearPlantModel(plant=plant, dt=DT)
    x = np.array([0.05, 0.0, 0.0, 0.0])
    for _ in range(1000):
        x = sim.step(x, c.compute_control(x)[0])
    assert abs(x[0]) < 1e-5      # theta_p back upright
    assert abs(x[1]) < 1e-4      # theta_p_dot at rest
    assert abs(x[3]) < 1e-1      # wheel spun down
    assert np.all(np.isfinite(x))


# --------------------------------------------------------------------------- #
# design-time guards
# --------------------------------------------------------------------------- #
def test_rejects_continuous_model(design):
    _, linear, _ = design
    with pytest.raises(ValueError):
        LQRController.from_model(linear.continuous, np.diag(Q_DIAG), R_LQR)


def test_rejects_non_model():
    with pytest.raises(TypeError):
        LQRController.from_model("not a model", np.diag(Q_DIAG), R_LQR)


def test_from_config_rejects_raw_matrices(design):
    _, linear, _ = design
    with pytest.raises(TypeError):
        LQRController.from_config(linear.discrete, {"Q": np.diag(Q_DIAG)})


def test_zero_wheel_angle_weight_is_rejected(design):
    # Q with zero weight on theta_w leaves the wheel-angle integrator
    # unregulated: a closed-loop eigenvalue sits ON the unit circle and the
    # lqr.md caution must trip
    _, linear, _ = design
    Q = np.diag([20.0, 2.0, 0.0, 1e-2])
    with pytest.raises(UnstableClosedLoopError):
        LQRController.from_model(linear.discrete, Q, R_LQR)


def test_constructor_enforces_stability_invariant(design):
    _, _, c = design
    with pytest.raises(UnstableClosedLoopError):
        LQRController(
            Q_lqr=c.Q_lqr, R_lqr=c.R_lqr, K_gain=c.K_gain, P_dare=c.P_dare,
            closed_loop_spectral_radius=1.0,
        )


def test_constructor_rejects_bad_gain_shape(design):
    _, _, c = design
    with pytest.raises(ValueError):
        LQRController(
            Q_lqr=c.Q_lqr, R_lqr=c.R_lqr, K_gain=np.zeros((2, 4)),
            P_dare=c.P_dare, closed_loop_spectral_radius=0.5,
        )


def test_frozen(design):
    _, _, c = design
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.K_gain = np.zeros((1, 4))
