"""SimulationEngine — the closed-loop LQG rollout, and the BO loop's oracle.

Composes the layers below into one experiment (``class_diagram.md``;
``dependency_rules.md`` §2): the true nonlinear plant stepped under
zero-order-hold (:class:`NonlinearPlantModel`), noisy sensors
(:class:`Disturbances`), the Kalman filter, and the LQR law acting on the
**estimate** — the LQG separation. ``run(config, seed)`` is the single entry
point the optimisation layer consumes (``dependency_rules.md`` §4: the
optimiser reaches this stack only through here), producing the typed
:class:`SimulationResult`.

Determinism contract (AGENTS §7): a rollout is a pure function of
``(LQGConfig, seed, plant params, sim params)``. One ``numpy.random.Generator``
seeded per rollout supplies, in a fixed order, the initial-state perturbation
and the per-step process and measurement noise; the result records the seed.
The diagram's ``step()``/``reset(seed)`` are folded into the atomic ``run``
for exactly this reason — there is no mutable engine state to drift.

Per-step sequencing (standard discrete LQG cycle): at each instant the sensors
are read first, the filter conditions on them, the controller acts on the
posterior estimate, and the world advances:

    z_k = C x_k + v_k                       (measure the true state)
    x̂_k = filter.update / step              (posterior given z_k)
    u_k = −K x̂_k                            (LQR on the estimate)
    x_{k+1} = step(x_k, u_k) + w_k          (true plant + process noise)

Row ``k`` of every ``SimulationResult`` array refers to instant ``t_k = k·dt``.

Divergence: the small-angle model is meaningless once the pendulum has
fallen, so the rollout stops early when ``|theta_p|`` exceeds
``divergence_angle`` (default π/2 — fallen sideways) or the state stops being
finite, returning the truncated arrays with ``diverged=True``
(``data_contracts.md`` §3; the ObjectiveFunction maps that to the finite
PENALTY, ``numerical_standards.md`` §7). Controller/filter *design* failures
(``UnstableClosedLoopError`` from an unstabilising gain,
``RiccatiNotConverged`` from a non-convergent sweep) are caught here and
returned as a one-sample ``diverged=True`` result, so the optimisation layer
sees the stack as a pure black box reached only through ``engine.run`` →
``objective.evaluate`` (``dependency_rules.md`` §4) — it never has to know
about, or import, the control/estimation exception types.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..control.lqr_controller import LQRController, UnstableClosedLoopError
from ..core.types import LQGConfig, SimulationResult
from ..dynamics.linearized_model import LinearizedPlantModel
from ..dynamics.nonlinear_model import NonlinearPlantModel
from ..estimation.kalman_filter import KalmanFilter
from ..numerics.riccati import RiccatiNotConverged
from ..physical.pendulum import ReactionWheelPendulum
from .disturbances import Disturbances

# The linearisation (and hence the whole LQG design) is a small-angle model:
# past upright ± π/2 the pendulum has fallen and the rollout is over.
DEFAULT_DIVERGENCE_ANGLE: float = math.pi / 2.0


@dataclass(frozen=True)
class SimulationEngine:
    """One simulated world: plant + sensors + clock (``class_diagram.md``).

    The engine itself is immutable — every candidate :class:`LQGConfig` is
    evaluated against the *same* world, which is what makes rollout costs
    comparable across the Bayesian-optimisation search.

    Parameters
    ----------
    plant : ReactionWheelPendulum
        The true nonlinear plant.
    dt : float
        Control sampling period Δt (s), > 0.
    simulation_time : float
        Rollout horizon (s); the number of samples is ``round(T/Δt)``.
    disturbances : Disturbances
        True process/measurement/initial noise levels (NOT the config's W/V).
    initial_state : (n_x,) array_like
        Nominal initial true state (default: 0.05 rad tilt, at rest) — the
        experiment's test condition; the filter starts from it too.
    seed : int
        Default rollout seed when ``run`` is not given one explicitly.
    divergence_angle : float
        ``|theta_p|`` bound (rad) past which the pendulum counts as fallen.
    """

    plant: ReactionWheelPendulum
    dt: float
    simulation_time: float
    disturbances: Disturbances
    initial_state: np.ndarray = field(
        default_factory=lambda: np.array([0.05, 0.0, 0.0, 0.0])
    )
    seed: int = 0
    divergence_angle: float = DEFAULT_DIVERGENCE_ANGLE

    def __post_init__(self) -> None:
        if not isinstance(self.plant, ReactionWheelPendulum):
            raise TypeError("plant must be a ReactionWheelPendulum")
        if not isinstance(self.disturbances, Disturbances):
            raise TypeError("disturbances must be a Disturbances")
        object.__setattr__(self, "dt", float(self.dt))
        object.__setattr__(self, "simulation_time", float(self.simulation_time))
        object.__setattr__(self, "divergence_angle", float(self.divergence_angle))
        object.__setattr__(self, "seed", int(self.seed))
        if self.dt <= 0.0:
            raise ValueError("dt must be > 0")
        if self.horizon < 1:
            raise ValueError("simulation_time must cover at least one sample")
        if self.divergence_angle <= 0.0:
            raise ValueError("divergence_angle must be > 0")
        x0 = np.array(self.initial_state, dtype=np.float64)  # copy -> read-only
        x0.flags.writeable = False
        object.__setattr__(self, "initial_state", x0)
        # build the models once: the linearisation depends only on the plant
        linear = LinearizedPlantModel.from_plant(self.plant, self.dt)
        object.__setattr__(self, "_linear", linear)
        object.__setattr__(
            self, "_true_model", NonlinearPlantModel(plant=self.plant, dt=self.dt)
        )
        if x0.shape != (linear.n_x,):
            raise ValueError(
                f"initial_state must have shape ({linear.n_x},), got {x0.shape}"
            )
        if (self.disturbances.n_x, self.disturbances.n_y) != (
            linear.n_x,
            linear.n_y,
        ):
            raise ValueError(
                "disturbances dimensions must match the plant: expected "
                f"({linear.n_x}, {linear.n_y}), got "
                f"({self.disturbances.n_x}, {self.disturbances.n_y})"
            )

    @property
    def horizon(self) -> int:
        """Number of samples ``T = round(simulation_time / dt)``."""
        return int(round(self.simulation_time / self.dt))

    @property
    def linear_model(self) -> LinearizedPlantModel:
        """The linearise-and-discretise pipeline the LQG design consumes."""
        return self._linear

    def run(self, config: LQGConfig, seed: int | None = None) -> SimulationResult:
        """One closed-loop LQG rollout of the candidate ``config``.

        Designs the controller and filter from the config, then plays the
        per-step cycle in the module docstring for ``horizon`` samples or
        until divergence. ``seed`` defaults to the engine's; the result
        records whichever was used.

        Raises
        ------
        UnstableClosedLoopError, RiccatiNotConverged
            Design failures, propagated for the optimisation layer to map to
            its finite penalty.
        """
        if not isinstance(config, LQGConfig):
            raise TypeError("config must be an LQGConfig")
        linear = self._linear
        if (config.n_x, config.n_u, config.n_y) != (
            linear.n_x,
            linear.n_u,
            linear.n_y,
        ):
            raise ValueError(
                "config dimensions must match the plant: expected "
                f"({linear.n_x}, {linear.n_u}, {linear.n_y}), got "
                f"({config.n_x}, {config.n_u}, {config.n_y})"
            )
        used_seed = self.seed if seed is None else int(seed)
        rng = np.random.default_rng(used_seed)  # the single threaded generator

        # a config that cannot yield a stabilising gain (or whose Riccati sweep
        # does not converge) is a failed run, not an exception the optimiser
        # must handle: report it as diverged (dependency_rules.md §4).
        try:
            controller = LQRController.from_config(linear.discrete, config)
            kalman = KalmanFilter.from_config(
                linear.discrete, config, x0=self.initial_state
            )
        except (UnstableClosedLoopError, RiccatiNotConverged):
            return self._diverged_result(config, used_seed)

        horizon = self.horizon
        time = self.dt * np.arange(horizon)
        true_states = np.zeros((horizon, linear.n_x))
        estimated_states = np.zeros((horizon, linear.n_x))
        controls = np.zeros((horizon, linear.n_u))
        measurements = np.zeros((horizon, linear.n_y))

        x = self.initial_state + self.disturbances.initial_perturbation(rng)
        u_prev: np.ndarray | None = None
        diverged = False
        samples = horizon
        for k in range(horizon):
            true_states[k] = x
            if abs(x[0]) > self.divergence_angle or not np.all(np.isfinite(x)):
                diverged = True
                samples = k + 1
                break
            z = linear.C @ x + self.disturbances.measurement_noise(rng)
            measurements[k] = z
            if u_prev is None:
                x_hat = kalman.update(z)  # first instant: condition the prior
            else:
                x_hat = kalman.step(u_prev, z)
            estimated_states[k] = x_hat
            u = controller.compute_control(x_hat)
            controls[k] = u
            x = self._true_model.step(x, u[0]) + self.disturbances.process_noise(rng)
            u_prev = u

        return SimulationResult(
            time=time[:samples],
            true_states=true_states[:samples],
            estimated_states=estimated_states[:samples],
            controls=controls[:samples],
            measurements=measurements[:samples],
            seed=used_seed,
            diverged=diverged,
            config=config,
        )

    def _diverged_result(self, config: LQGConfig, used_seed: int) -> SimulationResult:
        """A one-sample ``diverged=True`` result for a failed controller design.

        The single sample sits at the nominal initial state (no physics ran);
        the ObjectiveFunction maps ``diverged`` to the finite PENALTY.
        """
        n_x, n_u, n_y = self._linear.n_x, self._linear.n_u, self._linear.n_y
        x0 = self.initial_state.reshape(1, n_x)
        return SimulationResult(
            time=np.zeros(1),
            true_states=x0,
            estimated_states=x0,
            controls=np.zeros((1, n_u)),
            measurements=np.zeros((1, n_y)),
            seed=used_seed,
            diverged=True,
            config=config,
        )
