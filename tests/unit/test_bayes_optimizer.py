"""Unit tests for optimization.bayes_optimizer.BayesianOptimizer (the §5 loop)."""

import numpy as np
import pytest

from inverted_pendulum.control.lqr_controller import LQRController
from inverted_pendulum.core.types import LQGConfig, SearchSpace
from inverted_pendulum.numerics.riccati import RiccatiNotConverged
from inverted_pendulum.optimization.acquisition.entropy_search import EntropySearch
from inverted_pendulum.optimization.acquisition.expected_improvement import (
    ExpectedImprovement,
)
from inverted_pendulum.optimization.bayes_optimizer import BayesianOptimizer
from inverted_pendulum.optimization.gaussian_process import GaussianProcess
from inverted_pendulum.optimization.kernels.matern import Matern52ARD
from inverted_pendulum.optimization.objective import ObjectiveFunction
from inverted_pendulum.physical.motor import DCMotor
from inverted_pendulum.physical.pendulum import ReactionWheelPendulum
from inverted_pendulum.physical.wheel import ReactionWheel
from inverted_pendulum.simulation.disturbances import Disturbances
from inverted_pendulum.simulation.simulator import SimulationEngine

DIM = 11  # 2·n_x + n_u + n_y = 8 + 1 + 2

# log-space search box (Q[4], R[1], W[4], V[2]) containing stabilising configs
LOWER = np.array([0.0, -1.0, -7.0, -7.0, -1.0, -19, -15, -19, -11, -15, -11], float)
UPPER = np.array([4.0, 2.0, -2.0, -2.0, 2.0, -8, -8, -8, -6, -8, -6], float)


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


def make_engine(**overrides):
    params = dict(dt=0.01, simulation_time=2.0, seed=0)
    params.update(overrides)
    return SimulationEngine(
        plant=overrides.pop("plant", make_plant()),
        disturbances=Disturbances(process_std=[0, 1e-4, 0, 1e-2],
                                  measurement_std=[1e-3, 1e-2]),
        **{k: v for k, v in params.items() if k != "plant"},
    )


def make_space():
    return SearchSpace(dimension=DIM, lower_bounds=LOWER, upper_bounds=UPPER,
                       log_scale=np.ones(DIM, dtype=bool))


def make_gp(noise=1e-2):
    return GaussianProcess(
        Matern52ARD(lengthscales=np.ones(DIM), signal_variance=1.0),
        noise_variance=noise,
    )


def make_objective():
    return ObjectiveFunction(Mp_desired=5.0, Ts_desired=0.5)


def make_bo(acquisition=None, optimize_hyperparameters=False, **engine_kwargs):
    return BayesianOptimizer(
        make_engine(**engine_kwargs),
        make_objective(),
        make_gp(),
        acquisition or ExpectedImprovement(n_candidates=100),
        make_space(),
        optimize_hyperparameters=optimize_hyperparameters,
    )


# --------------------------------------------------------------------------- #
# loop structure — optimization.md §5
# --------------------------------------------------------------------------- #
def test_initial_design_only():
    bo = make_bo()
    best = bo.optimize(n_initial=4, n_iterations=0, rng=np.random.default_rng(0))
    assert bo.data.size() == 4
    assert isinstance(best, LQGConfig)


def test_full_loop_grows_the_dataset_and_returns_a_config():
    bo = make_bo()
    best = bo.optimize(n_initial=5, n_iterations=4, rng=np.random.default_rng(1))
    assert bo.data.size() == 9  # n_init + n_iterations
    assert isinstance(best, LQGConfig)
    assert np.all(np.isfinite(bo.data.y))  # finite cost surface (§7)


def test_reported_config_stabilizes_the_plant():
    # the loop's payoff: the posterior-mean minimiser is a usable controller
    bo = make_bo()
    best = bo.optimize(n_initial=6, n_iterations=5, rng=np.random.default_rng(2))
    controller = LQRController.from_config(bo.engine.linear_model.discrete, best)
    assert controller.closed_loop_spectral_radius < 1.0
    # at least one evaluated design was stabilising (cost below the penalty)
    assert float(np.min(bo.data.y)) < make_objective().penalty


def test_optimize_is_deterministic_given_the_seed():
    a = make_bo()
    b = make_bo()
    config_a = a.optimize(n_initial=4, n_iterations=3, rng=np.random.default_rng(7))
    config_b = b.optimize(n_initial=4, n_iterations=3, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(a.data.X, b.data.X)
    np.testing.assert_array_equal(a.data.y, b.data.y)
    assert config_a == config_b


# --------------------------------------------------------------------------- #
# the oracle — dependency_rules §4
# --------------------------------------------------------------------------- #
def test_evaluate_candidate_is_finite():
    bo = make_bo()
    rng = np.random.default_rng(0)
    for theta in bo.space.sample(10, rng):
        assert np.isfinite(bo.evaluate_candidate(theta, rng))


def test_divergent_rollout_maps_to_penalty():
    # a 2 V motor cannot catch a 0.4 rad start: the rollout diverges and the
    # objective returns the finite PENALTY through the oracle
    bo = make_bo(
        plant=make_plant(max_voltage=2.0),
        initial_state=np.array([0.4, 0.0, 0.0, 0.0]),
    )
    # a reasonable, stabilising-by-design θ still diverges dynamically here
    theta = LQGConfig(
        Q_lqr=np.diag([20.0, 2.0, 1e-2, 1e-2]), R_lqr=np.eye(1),
        W_process=np.diag([1e-8, 1e-6, 1e-8, 1e-4]), V_measure=np.diag([1e-6, 1e-4]),
    ).to_vector()
    assert bo.evaluate_candidate(theta, np.random.default_rng(0)) == pytest.approx(
        make_objective().penalty
    )


def test_design_failure_maps_to_penalty(monkeypatch):
    # a design failure surfaces from the engine as a diverged rollout, which
    # the objective scores as the finite penalty — the optimiser stays a pure
    # black-box consumer (dependency_rules §4)
    from inverted_pendulum.simulation import simulator as sim_module

    def boom(*args, **kwargs):
        raise RiccatiNotConverged("forced for the test")

    monkeypatch.setattr(sim_module.LQRController, "from_config", boom)
    bo = make_bo()
    theta = bo.space.sample(1, np.random.default_rng(0))[0]
    assert bo.evaluate_candidate(theta, np.random.default_rng(0)) == pytest.approx(
        make_objective().penalty
    )


def test_n_rollouts_averages():
    bo = make_bo(optimize_hyperparameters=False)
    bo.n_rollouts_per_eval = 3
    theta = bo.space.sample(1, np.random.default_rng(0))[0]
    # just exercises the averaging path; result must stay finite
    assert np.isfinite(bo.evaluate_candidate(theta, np.random.default_rng(0)))


def test_propose_next_returns_a_point_in_the_box():
    bo = make_bo()
    bo.optimize(n_initial=4, n_iterations=0, rng=np.random.default_rng(0))
    theta = bo.propose_next(np.random.default_rng(1))
    assert bo.space.contains(theta)


# --------------------------------------------------------------------------- #
# acquisitions in the loop
# --------------------------------------------------------------------------- #
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_full_loop_with_ml2_and_ei():
    # the §5 loop with ML-II refitting every step (EI acquisition)
    bo = BayesianOptimizer(
        make_engine(), make_objective(), make_gp(),
        ExpectedImprovement(n_candidates=150), make_space(),
        optimize_hyperparameters=True,
    )
    best = bo.optimize(n_initial=5, n_iterations=4, rng=np.random.default_rng(3))
    controller = LQRController.from_config(bo.engine.linear_model.discrete, best)
    assert controller.closed_loop_spectral_radius < 1.0


def test_full_loop_with_entropy_search():
    # the project-goal acquisition driving the real oracle, tiny ES params
    es = EntropySearch(
        n_optimum_samples=150, n_representers=20, n_fantasies=4, n_candidates=30
    )
    bo = BayesianOptimizer(
        make_engine(), make_objective(), make_gp(), es, make_space(),
        optimize_hyperparameters=False,
    )
    best = bo.optimize(n_initial=5, n_iterations=2, rng=np.random.default_rng(5))
    assert isinstance(best, LQGConfig)
    assert bo.data.size() == 7


# --------------------------------------------------------------------------- #
# construction guards
# --------------------------------------------------------------------------- #
def test_construction_validates_dimensions_and_types():
    engine, objective, gp = make_engine(), make_objective(), make_gp()
    acq, space = ExpectedImprovement(), make_space()
    with pytest.raises(TypeError):
        BayesianOptimizer("nope", objective, gp, acq, space)
    with pytest.raises(TypeError):
        BayesianOptimizer(engine, objective, gp, "nope", space)
    # wrong-dimension space (and GP) vs the packed LQG dimension
    bad_space = SearchSpace(dimension=5, lower_bounds=np.zeros(5),
                            upper_bounds=np.ones(5), log_scale=np.ones(5, bool))
    bad_gp = GaussianProcess(Matern52ARD(np.ones(5), 1.0))
    with pytest.raises(ValueError):
        BayesianOptimizer(engine, objective, bad_gp, acq, bad_space)
    with pytest.raises(ValueError):
        make_bo().__class__(
            engine, objective, gp, acq, space, n_rollouts_per_eval=0
        )


def test_optimize_rejects_bad_counts():
    bo = make_bo()
    with pytest.raises(ValueError):
        bo.optimize(n_initial=0, n_iterations=1, rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        bo.optimize(n_initial=1, n_iterations=-1, rng=np.random.default_rng(0))
