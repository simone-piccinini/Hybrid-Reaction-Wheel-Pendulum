"""Unit tests for dynamics.nonlinear_model.NonlinearPlantModel (ZOH stepper)."""

import dataclasses

import numpy as np
import pytest

from inverted_pendulum.dynamics.nonlinear_model import NonlinearPlantModel
from inverted_pendulum.physical.motor import DCMotor
from inverted_pendulum.physical.pendulum import ReactionWheelPendulum
from inverted_pendulum.physical.wheel import ReactionWheel

DT = 1e-3


def make_plant(**overrides):
    params = dict(pendulum_mass=0.3, pendulum_length=0.15, body_inertia=0.02,
                  pivot_friction=0.01)
    params.update(overrides)
    return ReactionWheelPendulum(
        wheel=ReactionWheel(mass=0.1, radius=0.05, inertia=1e-4,
                            friction_coefficient=1e-4),
        motor=DCMotor(resistance=2.0, torque_constant=0.05,
                      back_emf_constant=0.05),
        **params,
    )


def make(dt=DT, method="rk4"):
    return NonlinearPlantModel(plant=make_plant(), dt=dt, method=method)


# --------------------------------------------------------------------------- #
# stepping
# --------------------------------------------------------------------------- #
def test_step_is_one_rk4_step_of_the_plant():
    # hand-rolled classical RK4 on f(x) = state_derivative(x, V) with V held
    m = make()
    f = lambda x: m.plant.state_derivative(x, 6.0)  # noqa: E731
    x = np.array([0.05, -0.2, 1.0, 30.0])
    k1 = f(x)
    k2 = f(x + 0.5 * DT * k1)
    k3 = f(x + 0.5 * DT * k2)
    k4 = f(x + DT * k3)
    expected = x + (DT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    np.testing.assert_allclose(m.step(x, 6.0), expected, rtol=1e-15)


def test_euler_method_option():
    m = make(method="euler")
    x = np.array([0.05, -0.2, 1.0, 30.0])
    expected = x + DT * m.plant.state_derivative(x, 3.0)
    np.testing.assert_allclose(m.step(x, 3.0), expected, rtol=1e-15)


def test_equilibrium_is_a_fixed_point():
    m = make()
    np.testing.assert_allclose(m.step(np.zeros(4), 0.0), np.zeros(4))


def test_derivative_delegates_to_plant():
    m = make()
    x = np.array([0.1, 0.3, -2.0, 15.0])
    np.testing.assert_allclose(
        m.derivative(x, 4.0), m.plant.state_derivative(x, 4.0)
    )


def test_step_is_deterministic():
    m = make()
    x = np.array([0.02, 0.1, 0.0, 5.0])
    np.testing.assert_array_equal(m.step(x, 2.0), m.step(x, 2.0))


def test_rk4_more_accurate_than_euler():
    # both integrate the same vector field; against a tiny-step reference,
    # rk4's one-step error must be far below euler's
    fine = make(dt=DT / 100)
    x = np.array([0.1, 0.0, 0.0, 0.0])
    ref = x.copy()
    for _ in range(100):
        ref = fine.step(ref, 1.0)
    e_rk4 = np.max(np.abs(make(method="rk4").step(x, 1.0) - ref))
    e_euler = np.max(np.abs(make(method="euler").step(x, 1.0) - ref))
    assert e_rk4 < e_euler / 100


# --------------------------------------------------------------------------- #
# validation / immutability
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad_dt", [0.0, -1e-3])
def test_rejects_bad_dt(bad_dt):
    with pytest.raises(ValueError):
        NonlinearPlantModel(plant=make_plant(), dt=bad_dt)


def test_rejects_unknown_method():
    with pytest.raises(ValueError):
        NonlinearPlantModel(plant=make_plant(), dt=DT, method="rk45")


def test_rejects_wrong_plant_type():
    with pytest.raises(TypeError):
        NonlinearPlantModel(plant="not a plant", dt=DT)


def test_frozen():
    m = make()
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.dt = 0.01
