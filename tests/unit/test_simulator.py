"""Unit tests for simulation.simulator.SimulationEngine (the LQG rollout oracle)."""

import dataclasses
import math

import numpy as np
import pytest

from inverted_pendulum.control.lqr_controller import LQRController
from inverted_pendulum.core.types import LQGConfig
from inverted_pendulum.numerics.riccati import RiccatiNotConverged
from inverted_pendulum.physical.motor import DCMotor
from inverted_pendulum.physical.pendulum import ReactionWheelPendulum
from inverted_pendulum.physical.wheel import ReactionWheel
from inverted_pendulum.simulation.disturbances import Disturbances
from inverted_pendulum.simulation.simulator import SimulationEngine

DT = 0.01


def make_plant(**motor_overrides):
    motor = dict(resistance=2.0, torque_constant=0.05, back_emf_constant=0.05)
    motor.update(motor_overrides)
    return ReactionWheelPendulum(
        pendulum_mass=0.3, pendulum_length=0.15, body_inertia=0.02,
        pivot_friction=0.01,
        wheel=ReactionWheel(mass=0.1, radius=0.05, inertia=1e-4,
                            friction_coefficient=1e-4),
        motor=DCMotor(**motor),
    )


def make_config(**overrides):
    params = dict(
        Q_lqr=np.diag([20.0, 2.0, 1e-2, 1e-2]),
        R_lqr=np.eye(1),
        W_process=np.diag([1e-8, 1e-6, 1e-8, 1e-4]),
        V_measure=np.diag([1e-6, 1e-4]),
    )
    params.update(overrides)
    return LQGConfig(**params)


def make_disturbances(**overrides):
    params = dict(process_std=[0.0, 1e-4, 0.0, 1e-2], measurement_std=[1e-3, 1e-2])
    params.update(overrides)
    return Disturbances(**params)


def make_engine(plant=None, disturbances=None, **overrides):
    params = dict(dt=DT, simulation_time=10.0, seed=7)
    params.update(overrides)
    return SimulationEngine(
        plant=plant or make_plant(),
        disturbances=disturbances or make_disturbances(),
        **params,
    )


# --------------------------------------------------------------------------- #
# determinism — AGENTS §7
# --------------------------------------------------------------------------- #
def test_same_seed_is_byte_reproducible():
    engine = make_engine()
    config = make_config()
    a, b = engine.run(config), engine.run(config)
    for name in ("time", "true_states", "estimated_states", "controls",
                 "measurements"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    assert a.seed == b.seed and a.diverged == b.diverged


def test_explicit_seed_overrides_and_is_recorded():
    engine = make_engine(seed=7)
    config = make_config()
    default = engine.run(config)
    other = engine.run(config, seed=99)
    assert default.seed == 7 and other.seed == 99
    assert not np.array_equal(default.true_states, other.true_states)


# --------------------------------------------------------------------------- #
# the nominal rollout — the LQG loop regulates
# --------------------------------------------------------------------------- #
def test_nominal_rollout_regulates():
    result = make_engine().run(make_config())
    assert not result.diverged
    assert result.horizon == 1000
    assert abs(result.true_states[-1, 0]) < 5e-3       # held upright
    estimation_error = np.abs(
        result.estimated_states[500:, 0] - result.true_states[500:, 0]
    )
    assert float(np.mean(estimation_error)) < 1e-3     # filter tracks
    assert result.config is make_config() or result.config.n_x == 4


def test_result_grids_and_contract():
    engine = make_engine(simulation_time=2.0)
    config = make_config()
    result = engine.run(config)
    assert engine.horizon == 200
    np.testing.assert_allclose(result.time, DT * np.arange(200))
    assert result.true_states.shape == (200, 4)
    assert result.controls.shape == (200, 1)
    assert result.measurements.shape == (200, 2)
    assert result.config is config


def test_metrics_consume_the_rollout():
    from inverted_pendulum.metrics.performance_metrics import control_effort
    from inverted_pendulum.metrics.stability_metrics import settling_time
    result = make_engine().run(make_config())
    assert 0.0 < settling_time(result) < 10.0
    assert 0.0 < control_effort(result) < 1e3


# --------------------------------------------------------------------------- #
# the noiseless world — exact bookkeeping
# --------------------------------------------------------------------------- #
def test_noiseless_world_bookkeeping():
    silent = Disturbances(process_std=np.zeros(4), measurement_std=np.zeros(2))
    engine = make_engine(disturbances=silent, simulation_time=3.0)
    config = make_config()
    result = engine.run(config)
    # measurements are exactly C x (no sensor noise)
    C = engine.linear_model.C
    np.testing.assert_allclose(
        result.measurements, result.true_states @ C.T, atol=1e-12
    )
    # recorded controls are exactly the LQR law on the recorded estimates
    controller = LQRController.from_config(engine.linear_model.discrete, config)
    np.testing.assert_allclose(
        result.controls, -result.estimated_states @ controller.K_gain.T, atol=1e-12
    )
    assert not result.diverged


# --------------------------------------------------------------------------- #
# divergence
# --------------------------------------------------------------------------- #
def test_starting_fallen_diverges_immediately():
    engine = make_engine(initial_state=np.array([1.8, 0.0, 0.0, 0.0]))
    result = engine.run(make_config())
    assert result.diverged
    assert result.horizon == 1
    np.testing.assert_allclose(result.true_states[0], [1.8, 0.0, 0.0, 0.0])


def test_saturated_motor_falls_dynamically():
    # 2 V of authority cannot catch a 0.3 rad tilt: the pendulum genuinely
    # falls mid-rollout and the arrays are truncated at the falling sample
    engine = make_engine(
        plant=make_plant(max_voltage=2.0),
        initial_state=np.array([0.3, 0.0, 0.0, 0.0]),
    )
    result = engine.run(make_config())
    assert result.diverged
    assert 1 < result.horizon < 1000
    assert abs(result.true_states[-1, 0]) > math.pi / 2
    assert result.time.shape[0] == result.horizon


def test_design_failure_becomes_a_diverged_result(monkeypatch):
    # a controller-design failure is reported as a one-sample diverged result,
    # not raised — the optimiser sees the stack as a black box (dependency
    # rules §4). Force the failure deterministically by stubbing the gain
    # design to raise.
    from inverted_pendulum.control.lqr_controller import LQRController
    from inverted_pendulum.simulation import simulator as sim_module

    def boom(*args, **kwargs):
        raise RiccatiNotConverged("forced for the test")

    monkeypatch.setattr(sim_module.LQRController, "from_config", boom)
    result = make_engine().run(make_config())
    assert result.diverged
    assert result.horizon == 1
    np.testing.assert_array_equal(result.true_states[0], make_engine().initial_state)


# --------------------------------------------------------------------------- #
# guards / construction
# --------------------------------------------------------------------------- #
def test_rejects_bad_construction():
    with pytest.raises(TypeError):
        make_engine(plant="not a plant")
    with pytest.raises(TypeError):
        SimulationEngine(plant=make_plant(), dt=DT, simulation_time=1.0,
                         disturbances="not disturbances")
    with pytest.raises(ValueError):
        make_engine(dt=0.0)
    with pytest.raises(ValueError):
        make_engine(simulation_time=1e-3)  # less than one sample
    with pytest.raises(ValueError):
        make_engine(divergence_angle=0.0)
    with pytest.raises(ValueError):
        make_engine(initial_state=np.zeros(3))
    with pytest.raises(ValueError):
        make_engine(disturbances=Disturbances(process_std=np.zeros(3),
                                              measurement_std=np.zeros(2)))


def test_run_rejects_bad_config():
    engine = make_engine()
    with pytest.raises(TypeError):
        engine.run({"Q": np.eye(4)})
    mismatched = LQGConfig(Q_lqr=np.eye(4), R_lqr=np.eye(1),
                           W_process=np.eye(4), V_measure=np.eye(3))
    with pytest.raises(ValueError):
        engine.run(mismatched)


def test_engine_is_frozen():
    engine = make_engine()
    with pytest.raises(dataclasses.FrozenInstanceError):
        engine.seed = 1
    assert not engine.initial_state.flags.writeable
