"""Unit tests for experiment.manager.ExperimentManager (config-driven runs)."""

import numpy as np
import pytest

from inverted_pendulum.control.lqr_controller import LQRController
from inverted_pendulum.core.types import LQGConfig, SearchSpace
from inverted_pendulum.io.config_loader import ExperimentConfig
from inverted_pendulum.io.run_logging import load_metadata
from inverted_pendulum.optimization.acquisition.expected_improvement import (
    ExpectedImprovement,
)
from inverted_pendulum.optimization.bayes_optimizer import BayesianOptimizer
from inverted_pendulum.optimization.gaussian_process import GaussianProcess
from inverted_pendulum.physical.pendulum import ReactionWheelPendulum
from inverted_pendulum.simulation.simulator import SimulationEngine
from inverted_pendulum.experiment.manager import ExperimentManager, ExperimentResult


def small_config_dict(**overrides):
    # log-space bounds (length 11) around stabilising LQG weights
    lower = [0.0, -1.0, -7.0, -7.0, -1.0, -19, -15, -19, -11, -15, -11]
    upper = [4.0, 2.0, -2.0, -2.0, 2.0, -8, -8, -8, -6, -8, -6]
    d = {
        "name": "small",
        "seed": 1,
        "plant": {
            "pendulum_mass": 0.3, "pendulum_length": 0.15, "body_inertia": 0.02,
            "pivot_friction": 0.01,
            "wheel": {"mass": 0.1, "radius": 0.05, "inertia": 1e-4,
                      "friction_coefficient": 1e-4},
            "motor": {"resistance": 2.0, "torque_constant": 0.05,
                      "back_emf_constant": 0.05},
        },
        "simulation": {
            "dt": 0.01, "simulation_time": 1.0, "initial_state": [0.05, 0, 0, 0],
            "disturbances": {"process_std": [0, 1e-4, 0, 1e-2],
                             "measurement_std": [1e-3, 1e-2]},
        },
        "objective": {"Mp_desired": 5.0, "Ts_desired": 0.5},
        "search_space": {"lower_bounds": lower, "upper_bounds": upper},
        "gp": {"kernel": "matern52"},
        "acquisition": {"kind": "expected_improvement",
                        "params": {"n_candidates": 80}},
        "optimization": {"n_initial": 4, "n_iterations": 3,
                         "optimize_hyperparameters": False},
    }
    d.update(overrides)
    return d


def small_manager(**overrides):
    return ExperimentManager(ExperimentConfig.from_dict(small_config_dict(**overrides)))


# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #
def test_builders_produce_the_right_objects():
    mgr = small_manager()
    assert isinstance(mgr.build_plant(), ReactionWheelPendulum)
    assert isinstance(mgr.build_engine(), SimulationEngine)
    space = mgr.build_search_space()
    assert isinstance(space, SearchSpace) and space.dimension == 11
    assert isinstance(mgr.build_gp(11), GaussianProcess)
    assert isinstance(mgr.build_acquisition(), ExpectedImprovement)
    assert isinstance(mgr.build_optimizer(), BayesianOptimizer)


def test_engine_seed_comes_from_config():
    mgr = small_manager(seed=42)
    assert mgr.build_engine().seed == 42
    assert mgr.seed == 42


def test_unknown_kernel_and_acquisition_raise():
    with pytest.raises(ValueError, match="kernel"):
        small_manager(gp={"kernel": "no_such"}).build_gp(11)
    with pytest.raises(ValueError, match="acquisition"):
        small_manager(acquisition={"kind": "no_such"}).build_acquisition()


def test_rejects_non_config():
    with pytest.raises(TypeError):
        ExperimentManager({"seed": 0})


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
def test_run_returns_a_well_formed_result():
    result = small_manager().run()
    assert isinstance(result, ExperimentResult)
    assert isinstance(result.best_config, LQGConfig)
    assert result.best_theta.shape == (11,)
    # history grows to n_initial + n_iterations
    assert result.history_X.shape == (7, 11)
    assert result.history_y.shape == (7,)
    assert np.all(np.isfinite(result.history_y))
    for key in ("objective", "overshoot", "settling_time", "control_effort",
                "trajectory_entropy", "best_observed_cost", "n_evaluations"):
        assert key in result.metrics
        assert np.isfinite(result.metrics[key])
    assert result.metrics["n_evaluations"] == 7
    assert result.git_hash == small_manager().git_hash


def test_search_finds_a_stabilising_design():
    # at least one evaluated config stabilised (cost below the penalty) — the
    # search is exploring usable controllers, not only diverging ones
    result = small_manager().run()
    assert result.metrics["best_observed_cost"] < 1e3
    # and the reported config is a valid, designable controller
    engine = small_manager().build_engine()
    controller = LQRController.from_config(
        engine.linear_model.discrete, result.best_config
    )
    assert controller.closed_loop_spectral_radius < 1.0


def test_run_is_deterministic():
    a = small_manager().run()
    b = small_manager().run()
    np.testing.assert_array_equal(a.history_X, b.history_X)
    np.testing.assert_array_equal(a.history_y, b.history_y)
    assert a.best_config == b.best_config


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #
def test_save_results_writes_the_record(tmp_path):
    mgr = small_manager()
    result = mgr.run()
    out = mgr.save_results(result, tmp_path / "run")
    assert (out / "metadata.json").is_file()
    assert (out / "trajectories.npz").is_file()
    assert (out / "history.npz").is_file()
    md = load_metadata(out)
    assert md["seed"] == 1
    assert md["git_hash"] == mgr.git_hash
    assert md["config"]["name"] == "small"
    assert md["metrics"]["n_evaluations"] == 7
    assert len(md["summary"]["best_theta"]) == 11


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_save_results_with_plots(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    mgr = small_manager()
    result = mgr.run()
    out = mgr.save_results(result, tmp_path / "run_plots", plots=True)
    assert (out / "trajectory.png").is_file()
    assert (out / "convergence.png").is_file()
