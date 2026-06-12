"""Validation of the dynamics-layer steppers against scipy references.

The composed plant + rk4 propagator is cross-checked against a high-accuracy
scipy.integrate.solve_ivp trajectory, and the linearise-and-discretise
pipeline against scipy.signal.cont2discrete (numerical_standards.md §8; only
tests/validation/ may import the banned reference libraries).
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import cont2discrete

from inverted_pendulum.dynamics.linearized_model import LinearizedPlantModel
from inverted_pendulum.dynamics.nonlinear_model import NonlinearPlantModel
from inverted_pendulum.numerics.constants import ATOL, VALIDATION_RTOL
from inverted_pendulum.physical.motor import DCMotor
from inverted_pendulum.physical.pendulum import ReactionWheelPendulum
from inverted_pendulum.physical.wheel import ReactionWheel


def make_plant() -> ReactionWheelPendulum:
    return ReactionWheelPendulum(
        pendulum_mass=0.3, pendulum_length=0.15, body_inertia=0.02,
        pivot_friction=0.01,
        wheel=ReactionWheel(mass=0.1, radius=0.05, inertia=1e-4,
                            friction_coefficient=1e-4),
        motor=DCMotor(resistance=2.0, torque_constant=0.05,
                      back_emf_constant=0.05),
    )


def test_rk4_trajectory_matches_solve_ivp():
    # 200 ms of free fall from a small tilt under constant voltage, stepped at
    # 1 ms ZOH (the voltage is constant, so a single solve_ivp integration of
    # the same vector field is a valid reference for the whole horizon)
    plant = make_plant()
    model = NonlinearPlantModel(plant=plant, dt=1e-3)
    x0 = np.array([0.05, 0.0, 0.0, 0.0])
    V = 0.5
    n_steps = 200

    x = x0.copy()
    for _ in range(n_steps):
        x = model.step(x, V)

    ref = solve_ivp(
        lambda _t, s: plant.state_derivative(s, V),
        (0.0, n_steps * 1e-3),
        x0,
        rtol=1e-12,
        atol=1e-12,
        dense_output=False,
    )
    assert ref.success
    assert np.allclose(x, ref.y[:, -1], rtol=VALIDATION_RTOL, atol=ATOL)


def test_from_plant_discretization_matches_cont2discrete():
    plant = make_plant()
    dt = 0.01
    model = LinearizedPlantModel.from_plant(plant, dt)
    A_d_ref, B_d_ref, C_ref, D_ref, _ = cont2discrete(
        (model.continuous.A, model.continuous.B,
         model.continuous.C, model.continuous.D),
        dt, method="zoh",
    )
    assert np.allclose(model.A_d, A_d_ref, rtol=VALIDATION_RTOL, atol=ATOL)
    assert np.allclose(model.B_d, B_d_ref, rtol=VALIDATION_RTOL, atol=ATOL)
    assert np.allclose(model.C, C_ref, rtol=VALIDATION_RTOL, atol=ATOL)
