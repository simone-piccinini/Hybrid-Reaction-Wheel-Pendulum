"""BayesianOptimizer — the outer loop that tunes the LQG weights.

The integrated Algorithm of ``optimization.md`` §5, tuning
``θ = vec(Q, R, W, V)`` over the realised closed-loop cost. It honours the
optimiser-isolation rule (``dependency_rules.md`` §4): it reaches the
physical/control/estimation stack **only** through ``engine.run(config)`` →
``objective.evaluate(result)``, never instantiating a controller or filter
itself. The same loop would tune any controller behind the same
``SimulationEngine``.

The §5 procedure, verbatim:

    sample n_init initial θ; query the oracle (n_r averaged rollouts each)
    standardise targets; fit GP hyperparameters φ by ML-II
    for t = n_init+1 … T:
        condition the GP on D_{t-1}
        θ_t ← argmax_θ α(θ)                 (the acquisition)
        y_t ← oracle(θ_t), averaged over n_r rollouts
        append (θ_t, y_t); refit φ by ML-II  (inner loop every step)
    return argmin_θ μ_T(θ)                   (the posterior-mean minimiser)

The reported answer is the minimiser of the **posterior mean**, not the best
observed point: under noise the lowest observation may be a lucky draw, whereas
the posterior mean integrates the evidence (§5).

The search coordinates are exactly :meth:`LQGConfig.to_vector`'s log-space
packing, so a candidate ``θ`` maps to a config by
``LQGConfig.from_vector(θ, n_x, n_u, n_y)`` and the SearchSpace must use
``log_scale = True`` on every dimension. A design that yields an unstable
gain or a non-convergent Riccati sweep surfaces as a ``diverged`` rollout from
the engine, which the objective maps to its finite ``PENALTY``
(``dependency_rules.md`` §4; ``numerical_standards.md`` §7) — the oracle is a
pure black box and the search never crashes on a bad candidate.

Determinism (§6): the whole run is reproducible from the ``rng`` passed to
:meth:`optimize` — it seeds the initial design, every rollout, the acquisition
draws, and the ML-II restarts.
"""

from __future__ import annotations

import numpy as np

from ..core.types import Dataset, LQGConfig, SearchSpace
from ..simulation.simulator import SimulationEngine
from .acquisition.base import AcquisitionFunction
from .gaussian_process import GaussianProcess
from .objective import ObjectiveFunction

# Default candidate-set size for the §5 "argmin posterior mean" report.
DEFAULT_REPORT_CANDIDATES: int = 1024


class BayesianOptimizer:
    """GP-BO with a pluggable acquisition over the LQG weights (``class_diagram.md``).

    Parameters
    ----------
    engine : SimulationEngine
        The rollout oracle's dynamics half.
    objective : ObjectiveFunction
        The rollout oracle's scoring half (cost ``y``; ``notation.md`` §6).
    gp : GaussianProcess
        The surrogate over the cost surface; its kernel dimension must equal
        ``space.dimension``.
    acquisition : AcquisitionFunction
        The strategy proposing each next θ (EI / UCB / Entropy Search).
    space : SearchSpace
        The bounded domain in log-space search coordinates.
    n_rollouts_per_eval : int
        ``n_r`` — rollouts averaged per oracle query (CLT noise reduction, §1).
    optimize_hyperparameters : bool
        Whether to refit the GP hyperparameters by ML-II each step (§5); may
        be disabled to hold the kernel fixed.
    """

    def __init__(
        self,
        engine: SimulationEngine,
        objective: ObjectiveFunction,
        gp: GaussianProcess,
        acquisition: AcquisitionFunction,
        space: SearchSpace,
        *,
        n_rollouts_per_eval: int = 1,
        optimize_hyperparameters: bool = True,
    ) -> None:
        if not isinstance(engine, SimulationEngine):
            raise TypeError("engine must be a SimulationEngine")
        if not isinstance(objective, ObjectiveFunction):
            raise TypeError("objective must be an ObjectiveFunction")
        if not isinstance(gp, GaussianProcess):
            raise TypeError("gp must be a GaussianProcess")
        if not isinstance(acquisition, AcquisitionFunction):
            raise TypeError("acquisition must be an AcquisitionFunction")
        if not isinstance(space, SearchSpace):
            raise TypeError("space must be a SearchSpace")
        model = engine.linear_model
        expected_dim = 2 * model.n_x + model.n_u + model.n_y
        if space.dimension != expected_dim:
            raise ValueError(
                f"space dimension {space.dimension} != packed LQG dimension "
                f"{expected_dim} (2·n_x + n_u + n_y)"
            )
        if gp.kernel.dim != space.dimension:
            raise ValueError("GP kernel dimension must equal space dimension")
        if int(n_rollouts_per_eval) < 1:
            raise ValueError("n_rollouts_per_eval must be >= 1")

        self.engine = engine
        self.objective = objective
        self.gp = gp
        self.acquisition = acquisition
        self.space = space
        self.n_rollouts_per_eval = int(n_rollouts_per_eval)
        self.optimize_hyperparameters = bool(optimize_hyperparameters)
        self.data = Dataset(space.dimension)
        self._n_x, self._n_u, self._n_y = model.n_x, model.n_u, model.n_y

    # ------------------------------------------------------------------ #
    # the oracle — dependency_rules §4: only engine.run → objective.evaluate
    # ------------------------------------------------------------------ #
    def evaluate_candidate(self, theta, rng: np.random.Generator) -> float:
        """Average rollout cost of the config packed in ``theta`` (search coords).

        Maps ``θ`` to an :class:`LQGConfig`, runs ``n_rollouts_per_eval``
        seeded rollouts through the black-box oracle (``engine.run`` →
        ``objective.evaluate``), and averages the costs. The engine reports a
        failed design as a diverged rollout, which the objective scores as the
        finite ``penalty`` — so the returned cost is always finite
        (``dependency_rules.md`` §4).
        """
        config = LQGConfig.from_vector(theta, self._n_x, self._n_u, self._n_y)
        costs = []
        for _ in range(self.n_rollouts_per_eval):
            seed = int(rng.integers(0, 2**31 - 1))
            result = self.engine.run(config, seed=seed)
            costs.append(self.objective.evaluate(result))
        return float(np.mean(costs))

    def propose_next(self, rng: np.random.Generator) -> np.ndarray:
        """Maximise the acquisition over the current GP (``optimization.md`` §4.5)."""
        return self.acquisition.select(self.gp, self.space, rng)

    # ------------------------------------------------------------------ #
    # the integrated loop — optimization.md §5
    # ------------------------------------------------------------------ #
    def _refit(self, rng: np.random.Generator) -> None:
        """Condition the GP on the data; refit φ by ML-II if enabled (§5)."""
        self.gp.fit(self.data.X, self.data.y)
        if self.optimize_hyperparameters:
            self.gp.optimize_hyperparameters(rng)

    def optimize(
        self, n_initial: int, n_iterations: int, rng: np.random.Generator
    ) -> LQGConfig:
        """Run the §5 procedure and return the posterior-mean minimiser.

        Parameters
        ----------
        n_initial : int
            ``n_init`` — Latin-hypercube initial designs queried before the
            model-driven phase.
        n_iterations : int
            ``T − n_init`` — acquisition-driven steps.
        rng : numpy.random.Generator
            The single seeded generator threading the whole run (§6).
        """
        if n_initial < 1:
            raise ValueError("n_initial must be >= 1")
        if n_iterations < 0:
            raise ValueError("n_iterations must be >= 0")

        # initial design: Latin-hypercube θ, each scored by the averaged oracle
        for theta in self.space.sample(n_initial, rng):
            self.data.append(theta, self.evaluate_candidate(theta, rng))
        self._refit(rng)

        # model-driven phase: propose, query, append, refit — every step
        for _ in range(n_iterations):
            theta = self.propose_next(rng)
            self.data.append(theta, self.evaluate_candidate(theta, rng))
            self._refit(rng)

        return self.best_config(rng)

    # ------------------------------------------------------------------ #
    # reporting — argmin of the posterior mean (§5)
    # ------------------------------------------------------------------ #
    def best_theta(
        self, rng: np.random.Generator, n_candidates: int = DEFAULT_REPORT_CANDIDATES
    ) -> np.ndarray:
        """The posterior-mean minimiser over a candidate set (search coords; §5)."""
        candidates = self.space.sample(n_candidates, rng)
        means = self.gp.predict(candidates).mean
        return candidates[int(np.argmin(means))].copy()

    def best_config(
        self, rng: np.random.Generator, n_candidates: int = DEFAULT_REPORT_CANDIDATES
    ) -> LQGConfig:
        """The reported optimum as an :class:`LQGConfig` (§5)."""
        return LQGConfig.from_vector(
            self.best_theta(rng, n_candidates), self._n_x, self._n_u, self._n_y
        )
