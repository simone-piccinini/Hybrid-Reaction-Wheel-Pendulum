"""ExperimentManager — config-driven, reproducible LQG-tuning runs.

The top-level orchestration layer (``class_diagram.md``;
``dependency_rules.md`` §1: ``experiment`` may import ``optimization``,
``simulation``, ``io`` and ``core``). It turns a plain-data
:class:`ExperimentConfig` into the whole object graph — plant, engine,
objective, GP, acquisition, Bayesian optimiser — runs the §5 tuning loop, and
records the AGENTS §7 reproducibility bundle (git hash, full config, seed,
metrics, trajectories).

A run is fully determined by ``(config, seed)``: the single seeded generator is
derived from ``config.seed`` and threaded through the optimiser, so re-running
the same config reproduces the same result.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..core.types import LQGConfig, SearchSpace, SimulationResult
from ..io.config_loader import ExperimentConfig, load_config
from ..io.run_logging import current_git_hash, save_run
from ..metrics.performance_metrics import control_effort, itae, oscillation_energy
from ..metrics.stability_metrics import overshoot, settling_time
from ..metrics.trajectory_entropy import trajectory_entropy
from ..optimization.acquisition.entropy_search import EntropySearch
from ..optimization.acquisition.expected_improvement import ExpectedImprovement
from ..optimization.acquisition.ucb import UpperConfidenceBound
from ..optimization.bayes_optimizer import BayesianOptimizer
from ..optimization.gaussian_process import GaussianProcess
from ..optimization.kernels.matern import Matern52ARD
from ..optimization.kernels.squared_exponential import SquaredExponentialARD
from ..optimization.objective import ObjectiveFunction
from ..physical.motor import DCMotor
from ..physical.pendulum import ReactionWheelPendulum
from ..physical.wheel import ReactionWheel
from ..simulation.disturbances import Disturbances
from ..simulation.simulator import SimulationEngine

_KERNELS = {
    "matern52": Matern52ARD,
    "squared_exponential": SquaredExponentialARD,
}
_ACQUISITIONS = {
    "entropy_search": EntropySearch,
    "expected_improvement": ExpectedImprovement,
    "ucb": UpperConfidenceBound,
}


@dataclass(frozen=True)
class ExperimentResult:
    """The outcome of one experiment (returned by :meth:`ExperimentManager.run`).

    Fields
    ------
    name, seed, git_hash :
        Identity and provenance of the run (AGENTS §7).
    best_config : LQGConfig
        The reported optimum — the posterior-mean minimiser (``optimization.md`` §5).
    best_theta : ndarray
        Its packed log-space vector (``LQGConfig.to_vector``).
    history_X, history_y : ndarray
        The Bayesian-optimisation dataset: evaluated points and their costs.
    best_result : SimulationResult
        A representative rollout of ``best_config`` at the experiment seed.
    metrics : dict
        Scalar metrics of ``best_result`` plus the search summary.
    """

    name: str
    seed: int
    git_hash: str
    best_config: LQGConfig
    best_theta: np.ndarray
    history_X: np.ndarray
    history_y: np.ndarray
    best_result: SimulationResult
    metrics: dict


class ExperimentManager:
    """Builds and runs a configured LQG-tuning experiment (``class_diagram.md``).

    Parameters
    ----------
    config : ExperimentConfig
        The parsed experiment configuration (see :func:`load_config`).
    """

    def __init__(self, config: ExperimentConfig) -> None:
        if not isinstance(config, ExperimentConfig):
            raise TypeError("config must be an ExperimentConfig")
        self.config = config
        self.seed = int(config.seed)
        self.git_hash = current_git_hash()

    @classmethod
    def from_config_file(cls, path) -> "ExperimentManager":
        """Load a YAML config from ``path`` and construct the manager."""
        return cls(load_config(path))

    # ------------------------------------------------------------------ #
    # builders — turn config data into domain objects
    # ------------------------------------------------------------------ #
    def build_plant(self) -> ReactionWheelPendulum:
        """Assemble the plant from the ``plant`` config section."""
        p = self.config.plant
        return ReactionWheelPendulum(
            pendulum_mass=p.pendulum_mass,
            pendulum_length=p.pendulum_length,
            body_inertia=p.body_inertia,
            pivot_friction=p.pivot_friction,
            gravity=p.gravity,
            wheel=ReactionWheel(
                mass=p.wheel.mass, radius=p.wheel.radius,
                inertia=p.wheel.inertia,
                friction_coefficient=p.wheel.friction_coefficient,
            ),
            motor=DCMotor(
                resistance=p.motor.resistance,
                torque_constant=p.motor.torque_constant,
                back_emf_constant=p.motor.back_emf_constant,
                inductance=p.motor.inductance,
                rotor_inertia=p.motor.rotor_inertia,
                friction_coefficient=p.motor.friction_coefficient,
                max_voltage=p.motor.max_voltage,
                max_current=p.motor.max_current,
            ),
        )

    def build_engine(self) -> SimulationEngine:
        """Assemble the rollout engine from the ``simulation`` config section."""
        s = self.config.simulation
        disturbances = Disturbances(
            process_std=s.disturbances.process_std,
            measurement_std=s.disturbances.measurement_std,
            initial_state_std=s.disturbances.initial_state_std,
        )
        return SimulationEngine(
            plant=self.build_plant(),
            dt=s.dt,
            simulation_time=s.simulation_time,
            disturbances=disturbances,
            initial_state=s.initial_state,
            seed=self.seed,
            divergence_angle=s.divergence_angle,
        )

    def build_objective(self) -> ObjectiveFunction:
        """Assemble the cost function from the ``objective`` config section."""
        o = self.config.objective
        kwargs = dict(
            Mp_desired=o.Mp_desired, Ts_desired=o.Ts_desired,
            Mp_max=o.Mp_max, Ts_max=o.Ts_max,
            w_error=o.w_error, w_control=o.w_control,
            w_overshoot=o.w_overshoot, w_settling=o.w_settling,
            U_max=o.U_max, w_saturation=o.w_saturation,
            error_state_index=o.error_state_index,
        )
        if o.penalty is not None:
            kwargs["penalty"] = o.penalty
        return ObjectiveFunction(**kwargs)

    def build_search_space(self) -> SearchSpace:
        """Assemble the BO search box from the ``search_space`` config section."""
        sp = self.config.search_space
        d = sp.lower_bounds.shape[0]
        log_scale = (np.ones(d, dtype=bool) if sp.log_scale is None
                     else sp.log_scale)
        return SearchSpace(
            dimension=d, lower_bounds=sp.lower_bounds,
            upper_bounds=sp.upper_bounds, log_scale=log_scale,
        )

    def build_gp(self, dim: int) -> GaussianProcess:
        """Assemble the GP surrogate from the ``gp`` config section."""
        g = self.config.gp
        if g.kernel not in _KERNELS:
            raise ValueError(
                f"unknown kernel '{g.kernel}'; choose from {sorted(_KERNELS)}"
            )
        lengthscales = np.ones(dim) if g.lengthscales is None else g.lengthscales
        kernel = _KERNELS[g.kernel](
            lengthscales=lengthscales, signal_variance=g.signal_variance
        )
        return GaussianProcess(kernel, noise_variance=g.noise_variance)

    def build_acquisition(self):
        """Assemble the acquisition from the ``acquisition`` config section."""
        a = self.config.acquisition
        if a.kind not in _ACQUISITIONS:
            raise ValueError(
                f"unknown acquisition '{a.kind}'; choose from {sorted(_ACQUISITIONS)}"
            )
        return _ACQUISITIONS[a.kind](**a.params)

    def build_optimizer(self) -> BayesianOptimizer:
        """Assemble the full :class:`BayesianOptimizer` from the config."""
        space = self.build_search_space()
        return BayesianOptimizer(
            engine=self.build_engine(),
            objective=self.build_objective(),
            gp=self.build_gp(space.dimension),
            acquisition=self.build_acquisition(),
            space=space,
            n_rollouts_per_eval=self.config.optimization.n_rollouts_per_eval,
            optimize_hyperparameters=self.config.optimization.optimize_hyperparameters,
        )

    # ------------------------------------------------------------------ #
    # the run
    # ------------------------------------------------------------------ #
    def run(self) -> ExperimentResult:
        """Run the §5 tuning loop and return the :class:`ExperimentResult`.

        Deterministic from ``config.seed``: one generator threads the whole
        optimisation. After the search, a representative rollout of the
        reported best config (at the experiment seed) is scored for the
        metrics record.
        """
        rng = np.random.default_rng(self.seed)
        optimizer = self.build_optimizer()
        opt = self.config.optimization

        best_config = optimizer.optimize(
            n_initial=opt.n_initial, n_iterations=opt.n_iterations, rng=rng
        )
        best_theta = best_config.to_vector()

        # a representative rollout of the reported optimum, at the run seed
        best_result = optimizer.engine.run(best_config, seed=self.seed)
        metrics = self._metrics(best_result, optimizer)

        return ExperimentResult(
            name=self.config.name,
            seed=self.seed,
            git_hash=self.git_hash,
            best_config=best_config,
            best_theta=best_theta,
            history_X=optimizer.data.X.copy(),
            history_y=optimizer.data.y.copy(),
            best_result=best_result,
            metrics=metrics,
        )

    def _metrics(self, result: SimulationResult, optimizer: BayesianOptimizer) -> dict:
        """Scalar metrics of the representative rollout plus the search summary."""
        objective = optimizer.objective
        return {
            "objective": objective.evaluate(result),
            "itae": itae(result),
            "overshoot": overshoot(result),
            "settling_time": settling_time(result),
            "control_effort": control_effort(result),
            "oscillation_energy": oscillation_energy(result),
            "trajectory_entropy": trajectory_entropy(result),
            "diverged": bool(result.diverged),
            "best_observed_cost": float(np.min(optimizer.data.y)),
            "n_evaluations": int(optimizer.data.size()),
        }

    # ------------------------------------------------------------------ #
    # persistence
    # ------------------------------------------------------------------ #
    def save_results(self, result: ExperimentResult, directory, *, plots: bool = False):
        """Write the AGENTS §7 record (and optional plots) for ``result``.

        Returns the run directory. With ``plots=True`` also writes
        ``trajectory.png`` and ``convergence.png`` (requires matplotlib).
        """
        directory = Path(directory)
        summary = {
            "best_theta": result.best_theta,
            "best_config_Q_diag": np.diag(result.best_config.Q_lqr),
            "best_config_R_diag": np.diag(result.best_config.R_lqr),
            "best_config_W_diag": np.diag(result.best_config.W_process),
            "best_config_V_diag": np.diag(result.best_config.V_measure),
        }
        save_run(
            directory,
            config=asdict(self.config),
            seed=result.seed,
            metrics=result.metrics,
            result=result.best_result,
            history=(result.history_X, result.history_y),
            summary=summary,
            git_hash=result.git_hash,
        )
        if plots:
            from ..io.plotting import (
                plot_convergence, plot_trajectory, save_figure,
            )
            save_figure(plot_trajectory(result.best_result),
                        directory / "trajectory.png")
            save_figure(plot_convergence(result.history_y),
                        directory / "convergence.png")
        return directory
