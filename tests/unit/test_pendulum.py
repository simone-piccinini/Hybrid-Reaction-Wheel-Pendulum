"""Unit tests for physical.pendulum.ReactionWheelPendulum (plant + linearisation)."""

import dataclasses
import math

import numpy as np
import pytest

from inverted_pendulum.physical.motor import DCMotor
from inverted_pendulum.physical.pendulum import ReactionWheelPendulum
from inverted_pendulum.physical.wheel import ReactionWheel

# plausible bench-scale parameters (SI units)
MOTOR = dict(resistance=2.0, torque_constant=0.05, back_emf_constant=0.05)
WHEEL = dict(mass=0.1, radius=0.05, inertia=1e-4, friction_coefficient=1e-4)
PLANT = dict(pendulum_mass=0.3, pendulum_length=0.15, body_inertia=0.02,
             pivot_friction=0.01)


def make(motor=None, wheel=None, **overrides):
    params = dict(PLANT)
    params.update(overrides)
    return ReactionWheelPendulum(
        wheel=ReactionWheel(**(wheel or WHEEL)),
        motor=DCMotor(**(motor or MOTOR)),
        **params,
    )


# --------------------------------------------------------------------------- #
# equilibria and structure of the nonlinear dynamics
# --------------------------------------------------------------------------- #
def test_upright_is_an_equilibrium():
    p = make()
    np.testing.assert_allclose(p.nonlinear_dynamics(np.zeros(4), 0.0), np.zeros(4))
    np.testing.assert_allclose(p.state_derivative(np.zeros(4), 0.0), np.zeros(4))


def test_hanging_is_an_equilibrium():
    # θ = π: sin θ = 0, so gravity exerts no torque there either. Float sin(π)
    # is ~1.2e-16, which mgl/I_b amplifies to ~3e-15 — hence the atol.
    p = make()
    x = np.array([math.pi, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(p.nonlinear_dynamics(x, 0.0), np.zeros(4), atol=1e-13)


def test_kinematic_rows():
    # rows 1 and 3: ẋ₁ = x₂ and ẋ₃ = x₄ (model.md)
    p = make()
    x = np.array([0.3, -1.2, 4.0, 7.5])
    dx = p.nonlinear_dynamics(x, 0.2)
    assert dx[0] == pytest.approx(x[1])
    assert dx[2] == pytest.approx(x[3])


def test_gravity_destabilizes_upright():
    # tilted and at rest with no input: the pendulum accelerates away
    p = make()
    dx = p.nonlinear_dynamics(np.array([0.1, 0.0, 0.0, 0.0]), 0.0)
    assert dx[1] > 0.0


def test_gravity_enters_as_sine_not_angle():
    p = make(pivot_friction=0.0)
    theta = 1.0  # large angle: sin θ clearly ≠ θ
    dx = p.nonlinear_dynamics(np.array([theta, 0.0, 0.0, 0.0]), 0.0)
    mgl = p.pendulum_mass * p.gravity * p.pendulum_length
    assert dx[1] == pytest.approx(mgl * math.sin(theta) / p.body_inertia)
    assert dx[1] != pytest.approx(mgl * theta / p.body_inertia)


def test_action_reaction_coupling():
    # at the origin, a positive wheel torque pushes the body the other way
    p = make()
    tau = 0.05
    dx = p.nonlinear_dynamics(np.zeros(4), tau)
    I_b, I_w = p.body_inertia, p.wheel.inertia
    assert dx[1] == pytest.approx(-tau / I_b)
    assert dx[3] == pytest.approx(tau * (I_b + I_w) / (I_b * I_w))


def test_wheel_torque_balance():
    # I_w · (θ̈ + φ̈) = τ_w for an arbitrary state (the wheel equation of motion)
    p = make()
    x = np.array([0.4, -0.8, 2.0, 30.0])
    tau = 0.07
    dx = p.nonlinear_dynamics(x, tau)
    tau_net = p.wheel.apply_torque(tau, x[3])
    assert p.wheel.inertia * (dx[1] + dx[3]) == pytest.approx(tau_net)


def test_pivot_friction_damps_the_body():
    p = make(pivot_friction=0.02)
    frictionless = make(pivot_friction=0.0)
    x = np.array([0.0, 3.0, 0.0, 0.0])
    drag = p.nonlinear_dynamics(x, 0.0)[1] - frictionless.nonlinear_dynamics(x, 0.0)[1]
    assert drag == pytest.approx(-0.02 * 3.0 / p.body_inertia)


def test_wheel_angle_does_not_enter_dynamics():
    p = make()
    x1 = np.array([0.2, 0.5, 0.0, 10.0])
    x2 = np.array([0.2, 0.5, 123.0, 10.0])
    np.testing.assert_allclose(
        p.nonlinear_dynamics(x1, 0.1), p.nonlinear_dynamics(x2, 0.1)
    )


def test_state_derivative_composes_motor_and_plant():
    p = make()
    x = np.array([0.1, -0.4, 1.0, 20.0])
    V = 6.0
    tau_m = p.motor.compute_torque(V, x[3])
    np.testing.assert_allclose(p.state_derivative(x, V), p.nonlinear_dynamics(x, tau_m))


def test_bad_state_shape_raises():
    p = make()
    with pytest.raises(ValueError):
        p.nonlinear_dynamics(np.zeros(3), 0.0)
    with pytest.raises(ValueError):
        p.state_derivative(np.zeros((4, 1)), 0.0)


# --------------------------------------------------------------------------- #
# linearisation — model.md's explicit matrices
# --------------------------------------------------------------------------- #
def test_linearize_matches_model_md_matrices():
    p = make()
    model = p.linearize()
    I_b, I_w = p.body_inertia, p.wheel.inertia
    b_p = p.pivot_friction
    E = p.motor.torque_constant / p.motor.resistance
    D = (p.motor.torque_constant * p.motor.back_emf_constant / p.motor.resistance
         + p.wheel.friction_coefficient)
    mgl = p.pendulum_mass * p.gravity * p.pendulum_length
    coupling = (I_w + I_b) / (I_w * I_b)
    A_expected = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [mgl / I_b, -b_p / I_b, 0.0, D / I_b],
            [0.0, 0.0, 0.0, 1.0],
            [-mgl / I_b, b_p / I_b, 0.0, -D * coupling],
        ]
    )
    B_expected = np.array([[0.0], [-E / I_b], [0.0], [E * coupling]])
    np.testing.assert_allclose(model.A, A_expected)
    np.testing.assert_allclose(model.B, B_expected)
    assert not model.is_discrete and model.dt is None


def test_linearize_output_matrices():
    # C selects [theta_p, theta_w_dot] (n_y = 2), zero feedthrough
    model = make().linearize()
    np.testing.assert_allclose(
        model.C, [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    np.testing.assert_allclose(model.D, np.zeros((2, 1)))
    assert (model.n_x, model.n_u, model.n_y) == (4, 1, 2)


def test_wheel_angle_column_is_zero():
    model = make().linearize()
    np.testing.assert_allclose(model.A[:, 2], np.zeros(4))


def test_linearize_consistent_with_nonlinear_dynamics():
    # central finite differences of ẋ = f(x, u) at the origin reproduce (A, B)
    p = make()
    model = p.linearize()
    h = 1e-6
    A_fd = np.zeros((4, 4))
    for j in range(4):
        dx = np.zeros(4)
        dx[j] = h
        A_fd[:, j] = (
            p.state_derivative(dx, 0.0) - p.state_derivative(-dx, 0.0)
        ) / (2 * h)
    B_fd = (
        (p.state_derivative(np.zeros(4), h) - p.state_derivative(np.zeros(4), -h))
        / (2 * h)
    ).reshape(4, 1)
    np.testing.assert_allclose(A_fd, model.A, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(B_fd, model.B, rtol=1e-6, atol=1e-6)


def test_linearize_accepts_upright_operating_point():
    p = make()
    np.testing.assert_allclose(p.linearize(np.zeros(4)).A, p.linearize().A)
    # the wheel angle is free (column 3 of A is zero)
    np.testing.assert_allclose(
        p.linearize(np.array([0.0, 0.0, 42.0, 0.0])).A, p.linearize().A
    )


@pytest.mark.parametrize(
    "x0",
    [
        np.array([0.2, 0.0, 0.0, 0.0]),   # tilted
        np.array([0.0, 0.5, 0.0, 0.0]),   # swinging
        np.array([0.0, 0.0, 0.0, 10.0]),  # wheel spinning
        np.zeros(3),                      # wrong shape
    ],
)
def test_linearize_rejects_non_upright_operating_point(x0):
    with pytest.raises(ValueError):
        make().linearize(x0)


# --------------------------------------------------------------------------- #
# validation / immutability
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "bad",
    [
        dict(pendulum_mass=0.0),
        dict(pendulum_mass=-0.3),
        dict(pendulum_length=0.0),
        dict(body_inertia=0.0),
        dict(body_inertia=-0.02),
        dict(pivot_friction=-0.01),
        dict(gravity=0.0),
    ],
)
def test_validation_rejects_bad_params(bad):
    with pytest.raises(ValueError):
        make(**bad)


def test_validation_rejects_wrong_component_types():
    with pytest.raises(TypeError):
        ReactionWheelPendulum(wheel="not a wheel", motor=DCMotor(**MOTOR), **PLANT)
    with pytest.raises(TypeError):
        ReactionWheelPendulum(
            wheel=ReactionWheel(**WHEEL), motor="not a motor", **PLANT
        )


def test_lumped_scalar_properties():
    p = make()
    assert p.voltage_to_torque_gain == pytest.approx(0.05 / 2.0)
    assert p.lumped_damping == pytest.approx(0.05 * 0.05 / 2.0 + 1e-4)
    assert p.wheel_inertia == pytest.approx(1e-4)


def test_frozen():
    p = make()
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.gravity = 1.62
