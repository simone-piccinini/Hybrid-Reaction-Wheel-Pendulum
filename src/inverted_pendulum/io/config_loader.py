"""Configuration loading — YAML on disk to a typed :class:`ExperimentConfig`.

``io`` is a cross-cutting leaf that may depend on ``core`` only
(``dependency_rules.md`` §2). So this module does **not** build domain objects
(plant, engine, optimiser) — it only parses YAML into validated, plain-data
dataclasses. The ``experiment`` layer, which may import every layer, turns this
data into the actual objects (``experiment/manager.py``).

A YAML parser is explicitly allowed (``AGENTS.md`` §3: "config I/O is not the
problem"). The schema is documented field-by-field in ``configs/default.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml


def _require(section: dict, key: str, where: str):
    """Fetch ``section[key]`` or raise a clear, located error."""
    if key not in section:
        raise ValueError(f"config: missing required key '{key}' in {where}")
    return section[key]


def _vector(values, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"config: '{name}' must be a 1-D list, got shape {arr.shape}")
    arr.flags.writeable = False
    return arr


@dataclass(frozen=True)
class MotorConfig:
    """DC-motor parameters (``notation.md`` §2)."""

    resistance: float
    torque_constant: float
    back_emf_constant: float
    inductance: float = 0.0
    rotor_inertia: float = 0.0
    friction_coefficient: float = 0.0
    max_voltage: float = float("inf")
    max_current: float = float("inf")

    @classmethod
    def from_dict(cls, d: dict) -> "MotorConfig":
        return cls(
            resistance=float(_require(d, "resistance", "plant.motor")),
            torque_constant=float(_require(d, "torque_constant", "plant.motor")),
            back_emf_constant=float(_require(d, "back_emf_constant", "plant.motor")),
            inductance=float(d.get("inductance", 0.0)),
            rotor_inertia=float(d.get("rotor_inertia", 0.0)),
            friction_coefficient=float(d.get("friction_coefficient", 0.0)),
            max_voltage=float(d.get("max_voltage", float("inf"))),
            max_current=float(d.get("max_current", float("inf"))),
        )


@dataclass(frozen=True)
class WheelConfig:
    """Reaction-wheel parameters (``notation.md`` §2)."""

    mass: float
    radius: float
    inertia: float
    friction_coefficient: float = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> "WheelConfig":
        return cls(
            mass=float(_require(d, "mass", "plant.wheel")),
            radius=float(_require(d, "radius", "plant.wheel")),
            inertia=float(_require(d, "inertia", "plant.wheel")),
            friction_coefficient=float(d.get("friction_coefficient", 0.0)),
        )


@dataclass(frozen=True)
class PlantConfig:
    """Reaction-wheel-pendulum parameters plus its wheel and motor."""

    pendulum_mass: float
    pendulum_length: float
    body_inertia: float
    wheel: WheelConfig
    motor: MotorConfig
    pivot_friction: float = 0.0
    gravity: float = 9.81

    @classmethod
    def from_dict(cls, d: dict) -> "PlantConfig":
        return cls(
            pendulum_mass=float(_require(d, "pendulum_mass", "plant")),
            pendulum_length=float(_require(d, "pendulum_length", "plant")),
            body_inertia=float(_require(d, "body_inertia", "plant")),
            wheel=WheelConfig.from_dict(_require(d, "wheel", "plant")),
            motor=MotorConfig.from_dict(_require(d, "motor", "plant")),
            pivot_friction=float(d.get("pivot_friction", 0.0)),
            gravity=float(d.get("gravity", 9.81)),
        )


@dataclass(frozen=True)
class DisturbanceConfig:
    """True-world noise standard deviations (``simulation/disturbances.py``)."""

    process_std: np.ndarray
    measurement_std: np.ndarray
    initial_state_std: np.ndarray | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "DisturbanceConfig":
        initial = d.get("initial_state_std")
        return cls(
            process_std=_vector(_require(d, "process_std", "simulation.disturbances"),
                                "process_std"),
            measurement_std=_vector(
                _require(d, "measurement_std", "simulation.disturbances"),
                "measurement_std",
            ),
            initial_state_std=(None if initial is None
                               else _vector(initial, "initial_state_std")),
        )


@dataclass(frozen=True)
class SimulationConfig:
    """Rollout settings (``simulation/simulator.py``)."""

    dt: float
    simulation_time: float
    initial_state: np.ndarray
    disturbances: DisturbanceConfig
    divergence_angle: float = float(np.pi / 2.0)

    @classmethod
    def from_dict(cls, d: dict) -> "SimulationConfig":
        return cls(
            dt=float(_require(d, "dt", "simulation")),
            simulation_time=float(_require(d, "simulation_time", "simulation")),
            initial_state=_vector(_require(d, "initial_state", "simulation"),
                                  "initial_state"),
            disturbances=DisturbanceConfig.from_dict(
                _require(d, "disturbances", "simulation")
            ),
            divergence_angle=float(d.get("divergence_angle", np.pi / 2.0)),
        )


@dataclass(frozen=True)
class ObjectiveConfig:
    """Cost-function settings (``notation.md`` §6).

    The cost is ``w_error·ITAE + w_control·∫u² + hinge penalties on M_p, T_s``
    (and optionally on the peak input past ``U_max``). The hinge penalties are
    normalised by the maximum-acceptable values ``Mp_max``/``Ts_max``.
    """

    Mp_desired: float
    Ts_desired: float
    Mp_max: float = 20.0
    Ts_max: float = 3.0
    w_error: float = 1.0
    w_control: float = 1.0
    w_overshoot: float = 100.0
    w_settling: float = 100.0
    U_max: float | None = None
    w_saturation: float = 100.0
    error_state_index: int = 0
    penalty: float | None = None  # None → ObjectiveFunction's PENALTY default

    @classmethod
    def from_dict(cls, d: dict) -> "ObjectiveConfig":
        penalty = d.get("penalty")
        u_max = d.get("U_max")
        return cls(
            Mp_desired=float(_require(d, "Mp_desired", "objective")),
            Ts_desired=float(_require(d, "Ts_desired", "objective")),
            Mp_max=float(d.get("Mp_max", 20.0)),
            Ts_max=float(d.get("Ts_max", 3.0)),
            w_error=float(d.get("w_error", 1.0)),
            w_control=float(d.get("w_control", 1.0)),
            w_overshoot=float(d.get("w_overshoot", 100.0)),
            w_settling=float(d.get("w_settling", 100.0)),
            U_max=None if u_max is None else float(u_max),
            w_saturation=float(d.get("w_saturation", 100.0)),
            error_state_index=int(d.get("error_state_index", 0)),
            penalty=None if penalty is None else float(penalty),
        )


@dataclass(frozen=True)
class SearchSpaceConfig:
    """BO search box in log-space coordinates (``data_contracts.md`` §2)."""

    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    log_scale: np.ndarray | None = None  # None → all-True (every dim log-space)

    @classmethod
    def from_dict(cls, d: dict) -> "SearchSpaceConfig":
        log_scale = d.get("log_scale")
        return cls(
            lower_bounds=_vector(_require(d, "lower_bounds", "search_space"),
                                "lower_bounds"),
            upper_bounds=_vector(_require(d, "upper_bounds", "search_space"),
                                "upper_bounds"),
            log_scale=(None if log_scale is None
                       else np.asarray(log_scale, dtype=bool)),
        )


@dataclass(frozen=True)
class GPConfig:
    """GP surrogate settings (``optimization/gaussian_process.py``)."""

    kernel: str = "matern52"  # 'matern52' | 'squared_exponential'
    lengthscales: np.ndarray | None = None  # None → ones(d)
    signal_variance: float = 1.0
    noise_variance: float = 1e-2

    @classmethod
    def from_dict(cls, d: dict) -> "GPConfig":
        ell = d.get("lengthscales")
        return cls(
            kernel=str(d.get("kernel", "matern52")),
            lengthscales=None if ell is None else _vector(ell, "lengthscales"),
            signal_variance=float(d.get("signal_variance", 1.0)),
            noise_variance=float(d.get("noise_variance", 1e-2)),
        )


@dataclass(frozen=True)
class AcquisitionConfig:
    """Acquisition-function selection and its parameters."""

    kind: str = "entropy_search"  # 'entropy_search' | 'expected_improvement' | 'ucb'
    params: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "AcquisitionConfig":
        return cls(
            kind=str(d.get("kind", "entropy_search")),
            params=dict(d.get("params", {})),
        )


@dataclass(frozen=True)
class OptimizationConfig:
    """Outer-loop budget (``optimization.md`` §5)."""

    n_initial: int
    n_iterations: int
    n_rollouts_per_eval: int = 1
    optimize_hyperparameters: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "OptimizationConfig":
        return cls(
            n_initial=int(_require(d, "n_initial", "optimization")),
            n_iterations=int(_require(d, "n_iterations", "optimization")),
            n_rollouts_per_eval=int(d.get("n_rollouts_per_eval", 1)),
            optimize_hyperparameters=bool(d.get("optimize_hyperparameters", True)),
        )


@dataclass(frozen=True)
class ExperimentConfig:
    """A whole experiment, fully determining a reproducible run (AGENTS §7).

    Plain data only — the ``experiment`` layer builds domain objects from it.
    """

    name: str
    seed: int
    plant: PlantConfig
    simulation: SimulationConfig
    objective: ObjectiveConfig
    search_space: SearchSpaceConfig
    gp: GPConfig
    acquisition: AcquisitionConfig
    optimization: OptimizationConfig

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        if not isinstance(d, dict):
            raise ValueError("config: top level must be a mapping")
        return cls(
            name=str(d.get("name", "experiment")),
            seed=int(_require(d, "seed", "top level")),
            plant=PlantConfig.from_dict(_require(d, "plant", "top level")),
            simulation=SimulationConfig.from_dict(
                _require(d, "simulation", "top level")
            ),
            objective=ObjectiveConfig.from_dict(_require(d, "objective", "top level")),
            search_space=SearchSpaceConfig.from_dict(
                _require(d, "search_space", "top level")
            ),
            gp=GPConfig.from_dict(d.get("gp", {})),
            acquisition=AcquisitionConfig.from_dict(d.get("acquisition", {})),
            optimization=OptimizationConfig.from_dict(
                _require(d, "optimization", "top level")
            ),
        )


def load_config(path) -> ExperimentConfig:
    """Read a YAML file and parse it into a validated :class:`ExperimentConfig`.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the YAML is malformed or a required field is missing.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open("r") as handle:
        raw = yaml.safe_load(handle)
    if raw is None:
        raise ValueError(f"config file is empty: {path}")
    return ExperimentConfig.from_dict(raw)
