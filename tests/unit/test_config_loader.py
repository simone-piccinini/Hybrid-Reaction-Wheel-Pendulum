"""Unit tests for io.config_loader (YAML → typed ExperimentConfig)."""

from pathlib import Path

import numpy as np
import pytest

from inverted_pendulum.io.config_loader import (
    ExperimentConfig,
    SearchSpaceConfig,
    load_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_YAML = REPO_ROOT / "configs" / "default.yaml"


def minimal_dict():
    return {
        "name": "t",
        "seed": 3,
        "plant": {
            "pendulum_mass": 0.3, "pendulum_length": 0.15, "body_inertia": 0.02,
            "wheel": {"mass": 0.1, "radius": 0.05, "inertia": 1e-4},
            "motor": {"resistance": 2.0, "torque_constant": 0.05,
                      "back_emf_constant": 0.05},
        },
        "simulation": {
            "dt": 0.01, "simulation_time": 2.0, "initial_state": [0.05, 0, 0, 0],
            "disturbances": {"process_std": [0, 1e-4, 0, 1e-2],
                             "measurement_std": [1e-3, 1e-2]},
        },
        "objective": {"Mp_desired": 5.0, "Ts_desired": 1.0},
        "search_space": {"lower_bounds": [0] * 11, "upper_bounds": [1] * 11},
        "optimization": {"n_initial": 4, "n_iterations": 2},
    }


# --------------------------------------------------------------------------- #
# the shipped default config
# --------------------------------------------------------------------------- #
def test_default_yaml_loads():
    config = load_config(DEFAULT_YAML)
    assert isinstance(config, ExperimentConfig)
    assert config.name == "default"
    assert config.seed == 0
    assert config.plant.pendulum_mass == 0.3
    assert config.plant.wheel.inertia == pytest.approx(1e-4)
    assert config.plant.motor.resistance == 2.0
    assert config.simulation.dt == 0.01
    assert config.simulation.initial_state.shape == (4,)
    assert config.objective.Mp_desired == 5.0
    assert config.search_space.lower_bounds.shape == (11,)
    assert config.gp.kernel == "matern52"
    assert config.acquisition.kind == "entropy_search"
    assert config.acquisition.params["n_representers"] == 50
    assert config.optimization.n_initial == 8


# --------------------------------------------------------------------------- #
# from_dict
# --------------------------------------------------------------------------- #
def test_from_dict_minimal_and_defaults():
    config = ExperimentConfig.from_dict(minimal_dict())
    assert config.seed == 3
    # defaults filled in
    assert config.plant.gravity == 9.81
    assert config.plant.pivot_friction == 0.0
    assert config.plant.motor.max_voltage == float("inf")
    assert config.objective.w_error == 1.0
    assert config.objective.w_overshoot == 100.0
    assert config.objective.Mp_max == 20.0
    assert config.objective.U_max is None
    assert config.objective.penalty is None
    assert config.gp.kernel == "matern52"  # gp section omitted -> defaults
    assert config.acquisition.kind == "entropy_search"
    assert config.optimization.optimize_hyperparameters is True
    assert config.simulation.disturbances.initial_state_std is None


def test_missing_required_key_raises():
    d = minimal_dict()
    del d["plant"]["pendulum_mass"]
    with pytest.raises(ValueError, match="pendulum_mass"):
        ExperimentConfig.from_dict(d)


def test_missing_top_level_section_raises():
    d = minimal_dict()
    del d["objective"]
    with pytest.raises(ValueError, match="objective"):
        ExperimentConfig.from_dict(d)


def test_missing_seed_raises():
    d = minimal_dict()
    del d["seed"]
    with pytest.raises(ValueError, match="seed"):
        ExperimentConfig.from_dict(d)


def test_vectors_are_readonly_arrays():
    config = ExperimentConfig.from_dict(minimal_dict())
    assert isinstance(config.simulation.initial_state, np.ndarray)
    assert not config.simulation.initial_state.flags.writeable
    assert not config.search_space.lower_bounds.flags.writeable


def test_search_space_log_scale_optional():
    base = SearchSpaceConfig.from_dict(
        {"lower_bounds": [0, 0], "upper_bounds": [1, 1]}
    )
    assert base.log_scale is None
    explicit = SearchSpaceConfig.from_dict(
        {"lower_bounds": [0, 0], "upper_bounds": [1, 1], "log_scale": [True, False]}
    )
    assert explicit.log_scale.tolist() == [True, False]


def test_bad_vector_shape_raises():
    d = minimal_dict()
    d["simulation"]["disturbances"]["process_std"] = [[1, 2], [3, 4]]
    with pytest.raises(ValueError, match="process_std"):
        ExperimentConfig.from_dict(d)


# --------------------------------------------------------------------------- #
# load_config file handling
# --------------------------------------------------------------------------- #
def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("/no/such/config.yaml")


def test_load_config_empty_file(tmp_path):
    empty = tmp_path / "empty.yaml"
    empty.write_text("")
    with pytest.raises(ValueError, match="empty"):
        load_config(empty)
