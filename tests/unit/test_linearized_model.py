"""Unit tests for dynamics.linearized_model.LinearizedPlantModel."""

import dataclasses

import numpy as np
import pytest

from inverted_pendulum.core.types import StateSpaceModel
from inverted_pendulum.dynamics.linearized_model import LinearizedPlantModel
from inverted_pendulum.dynamics.nonlinear_model import NonlinearPlantModel
from inverted_pendulum.physical.motor import DCMotor
from inverted_pendulum.physical.pendulum import ReactionWheelPendulum
from inverted_pendulum.physical.wheel import ReactionWheel

DT = 1e-3


def make_plant():
    return ReactionWheelPendulum(
        pendulum_mass=0.3, pendulum_length=0.15, body_inertia=0.02,
        pivot_friction=0.01,
        wheel=ReactionWheel(mass=0.1, radius=0.05, inertia=1e-4,
                            friction_coefficient=1e-4),
        motor=DCMotor(resistance=2.0, torque_constant=0.05,
                      back_emf_constant=0.05),
    )


def make(dt=DT):
    return LinearizedPlantModel.from_plant(make_plant(), dt)


# --------------------------------------------------------------------------- #
# construction — model.md steps 3–6
# --------------------------------------------------------------------------- #
def test_from_plant_pairs_linearization_with_its_discretization():
    plant = make_plant()
    m = LinearizedPlantModel.from_plant(plant, DT)
    continuous = plant.linearize()
    discrete = continuous.discretize(DT)
    np.testing.assert_allclose(m.continuous.A, continuous.A)
    np.testing.assert_allclose(m.continuous.B, continuous.B)
    np.testing.assert_allclose(m.A_d, discrete.A)
    np.testing.assert_allclose(m.B_d, discrete.B)
    assert m.dt == DT and (m.n_x, m.n_u, m.n_y) == (4, 1, 2)


def test_output_map_carries_over():
    m = make()
    np.testing.assert_array_equal(m.continuous.C, m.discrete.C)
    np.testing.assert_array_equal(m.C, [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])


def test_from_plant_forwards_operating_point():
    plant = make_plant()
    LinearizedPlantModel.from_plant(plant, DT, operating_point=np.zeros(4))
    with pytest.raises(ValueError):
        LinearizedPlantModel.from_plant(
            plant, DT, operating_point=np.array([0.3, 0.0, 0.0, 0.0])
        )


def test_from_plant_rejects_wrong_plant_type():
    with pytest.raises(TypeError):
        LinearizedPlantModel.from_plant("not a plant", DT)


# --------------------------------------------------------------------------- #
# constructor invariants
# --------------------------------------------------------------------------- #
def test_rejects_swapped_models():
    plant = make_plant()
    continuous = plant.linearize()
    discrete = continuous.discretize(DT)
    with pytest.raises(ValueError):
        LinearizedPlantModel(continuous=discrete, discrete=discrete)
    with pytest.raises(ValueError):
        LinearizedPlantModel(continuous=continuous, discrete=continuous)


def test_rejects_mismatched_output_map():
    plant = make_plant()
    continuous = plant.linearize()
    discrete = continuous.discretize(DT)
    other_C = StateSpaceModel(
        discrete.A, discrete.B, np.eye(4), np.zeros((4, 1)),
        is_discrete=True, dt=DT,
    )
    with pytest.raises(ValueError):
        LinearizedPlantModel(continuous=continuous, discrete=other_C)


def test_rejects_non_models():
    with pytest.raises(TypeError):
        LinearizedPlantModel(continuous="A", discrete="B")


# --------------------------------------------------------------------------- #
# stepping
# --------------------------------------------------------------------------- #
def test_step_formula():
    m = make()
    x = np.array([0.01, -0.05, 2.0, 10.0])
    V = 3.0
    np.testing.assert_allclose(m.step(x, V), m.A_d @ x + m.B_d[:, 0] * V)


def test_equilibrium_is_a_fixed_point():
    m = make()
    np.testing.assert_allclose(m.step(np.zeros(4), 0.0), np.zeros(4))


def test_step_rejects_bad_state_shape():
    with pytest.raises(ValueError):
        make().step(np.zeros(3), 0.0)


def test_linear_step_matches_nonlinear_near_upright():
    # the swap symmetry: for small angles the two steppers must agree —
    # ZOH-exact linear update vs rk4 of the sin-theta dynamics
    plant = make_plant()
    linear = LinearizedPlantModel.from_plant(plant, DT)
    nonlinear = NonlinearPlantModel(plant=plant, dt=DT)
    x = np.array([1e-3, -2e-3, 0.5, 5e-2])
    V = 1e-2
    np.testing.assert_allclose(
        linear.step(x, V), nonlinear.step(x, V), rtol=1e-7, atol=1e-9
    )


def test_linear_step_diverges_from_nonlinear_at_large_angle():
    # sanity of the comparison above: at large angle sin(theta) != theta and
    # the two models must NOT agree
    plant = make_plant()
    linear = LinearizedPlantModel.from_plant(plant, dt=0.05)
    nonlinear = NonlinearPlantModel(plant=plant, dt=0.05)
    x = np.array([1.0, 0.0, 0.0, 0.0])
    assert not np.allclose(linear.step(x, 0.0), nonlinear.step(x, 0.0), rtol=1e-3)


# --------------------------------------------------------------------------- #
# immutability
# --------------------------------------------------------------------------- #
def test_frozen():
    m = make()
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.discrete = m.continuous
