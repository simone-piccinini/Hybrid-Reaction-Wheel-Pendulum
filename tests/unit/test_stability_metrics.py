"""Unit tests for metrics.stability_metrics (overshoot M_p, settling time T_s)."""

import numpy as np
import pytest

from inverted_pendulum.metrics.stability_metrics import overshoot, settling_time


# --------------------------------------------------------------------------- #
# overshoot
# --------------------------------------------------------------------------- #
def test_overshoot_known_value(make_result):
    # from +1.0, worst crossing to -0.2 → 20%
    r = make_result([1.0, 0.5, -0.2, 0.05, 0.0])
    assert overshoot(r) == pytest.approx(20.0)


def test_overshoot_monotone_decay_is_zero(make_result):
    r = make_result([1.0, 0.5, 0.2, 0.1, 0.05])
    assert overshoot(r) == 0.0


def test_overshoot_sign_symmetric(make_result):
    # from -1.0, worst crossing to +0.3 → 30%
    r = make_result([-1.0, -0.4, 0.3, -0.05])
    assert overshoot(r) == pytest.approx(30.0)


def test_overshoot_picks_the_worst_crossing(make_result):
    r = make_result([1.0, -0.1, 0.2, -0.35, 0.0])
    assert overshoot(r) == pytest.approx(35.0)


def test_overshoot_scales_with_initial_displacement(make_result):
    r = make_result([0.05, -0.01, 0.0])  # -0.01/0.05 = 20%
    assert overshoot(r) == pytest.approx(20.0)


def test_overshoot_zero_start_is_zero(make_result):
    assert overshoot(make_result(np.zeros(5))) == 0.0


def test_overshoot_other_state_index(make_result):
    states = np.zeros((4, 4))
    states[:, 3] = [10.0, 2.0, -1.0, 0.0]
    r = make_result(states=states)
    assert overshoot(r, state_index=3) == pytest.approx(10.0)
    assert overshoot(r, state_index=0) == 0.0


def test_overshoot_input_guards(make_result):
    r = make_result([1.0, 0.0])
    with pytest.raises(ValueError):
        overshoot(r, state_index=4)
    with pytest.raises(TypeError):
        overshoot("not a result")


# --------------------------------------------------------------------------- #
# settling time
# --------------------------------------------------------------------------- #
def test_settling_time_known_value(make_result):
    # band = 2% of 1.0 = 0.02; last violation at index 2 → settles at time[3]
    r = make_result([1.0, 0.5, 0.1, 0.015, 0.01, 0.005], dt=0.01)
    assert settling_time(r) == pytest.approx(0.03)


def test_settling_time_never_settles_caps_at_horizon(make_result):
    r = make_result([1.0, 0.9, 0.8], dt=0.01)
    assert settling_time(r) == pytest.approx(0.02)  # time[-1]


def test_settling_time_reentry_counts_the_last_violation(make_result):
    # dips inside the band, leaves again: only the final entry counts
    r = make_result([1.0, 0.01, 0.5, 0.01, 0.005], dt=0.01)
    assert settling_time(r) == pytest.approx(0.03)


def test_settling_time_zero_start(make_result):
    assert settling_time(make_result(np.zeros(5))) == 0.0
    # starts regulated but leaves: capped at the horizon
    r = make_result([0.0, 0.5, 0.0], dt=0.01)
    assert settling_time(r) == pytest.approx(0.02)


def test_settling_time_custom_band(make_result):
    # 50% band of 1.0: only the first sample violates → settles at time[1]
    r = make_result([1.0, 0.4, 0.3, 0.2], dt=0.01)
    assert settling_time(r, band_fraction=0.5) == pytest.approx(0.01)


def test_settling_time_respects_run_clock(make_result):
    # time need not start at zero: the metric reads the run's own clock
    r = make_result([1.0, 0.5, 0.01, 0.005], time=np.array([2.0, 2.1, 2.2, 2.3]))
    assert settling_time(r) == pytest.approx(2.2)


def test_settling_time_rejects_bad_band(make_result):
    with pytest.raises(ValueError):
        settling_time(make_result([1.0, 0.0]), band_fraction=0.0)


# --------------------------------------------------------------------------- #
# integration: metrics on a real stabilised rollout
# --------------------------------------------------------------------------- #
def test_metrics_on_the_lqr_stabilised_rollout(make_result):
    from inverted_pendulum.control.lqr_controller import LQRController
    from inverted_pendulum.dynamics.linearized_model import LinearizedPlantModel
    from inverted_pendulum.dynamics.nonlinear_model import NonlinearPlantModel
    from inverted_pendulum.metrics.performance_metrics import control_effort
    from inverted_pendulum.metrics.trajectory_entropy import trajectory_entropy
    from inverted_pendulum.physical.motor import DCMotor
    from inverted_pendulum.physical.pendulum import ReactionWheelPendulum
    from inverted_pendulum.physical.wheel import ReactionWheel

    plant = ReactionWheelPendulum(
        pendulum_mass=0.3, pendulum_length=0.15, body_inertia=0.02,
        pivot_friction=0.01,
        wheel=ReactionWheel(mass=0.1, radius=0.05, inertia=1e-4,
                            friction_coefficient=1e-4),
        motor=DCMotor(resistance=2.0, torque_constant=0.05,
                      back_emf_constant=0.05),
    )
    dt = 0.01
    linear = LinearizedPlantModel.from_plant(plant, dt)
    controller = LQRController.from_model(
        linear.discrete, np.diag([20.0, 2.0, 1e-2, 1e-2]), [[1.0]]
    )
    sim = NonlinearPlantModel(plant=plant, dt=dt)
    horizon = 1000
    states = np.empty((horizon, 4))
    controls = np.empty((horizon, 1))
    x = np.array([0.05, 0.0, 0.0, 0.0])
    for k in range(horizon):
        u = controller.compute_control(x)
        states[k], controls[k] = x, u
        x = sim.step(x, u[0])
    r = make_result(states=states, controls=controls, dt=dt)

    assert overshoot(r) < 50.0                      # mild crossing at most
    assert 0.0 < settling_time(r) < 5.0             # settles well inside 10 s
    assert 0.0 < control_effort(r) < 100.0          # finite, nonzero effort
    assert 0.0 < trajectory_entropy(r) < np.log(32)  # transient, then parked
