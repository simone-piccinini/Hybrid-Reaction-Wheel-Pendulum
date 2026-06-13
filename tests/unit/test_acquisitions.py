"""Unit tests for optimization.acquisition (EI, UCB, Entropy Search)."""

import numpy as np
import pytest

from inverted_pendulum.core.types import SearchSpace
from inverted_pendulum.optimization.acquisition.base import (
    AcquisitionFunction,
    standard_normal_cdf,
    standard_normal_pdf,
    expected_improvement_values,
)
from inverted_pendulum.optimization.acquisition.entropy_search import (
    EntropySearch,
    shannon_entropy,
)
from inverted_pendulum.optimization.acquisition.expected_improvement import (
    ExpectedImprovement,
)
from inverted_pendulum.optimization.acquisition.ucb import UpperConfidenceBound
from inverted_pendulum.optimization.gaussian_process import GaussianProcess
from inverted_pendulum.optimization.kernels.squared_exponential import (
    SquaredExponentialARD,
)


@pytest.fixture
def rng():
    return np.random.default_rng(20260612)


def fitted_gp(rng, n=8, d=1, noise=1e-3):
    X = rng.uniform(-3.0, 3.0, size=(n, d))
    y = (X[:, 0] - 1.0) ** 2  # a clear parabola, minimum near x = 1
    gp = GaussianProcess(
        SquaredExponentialARD(lengthscales=np.ones(d), signal_variance=1.0),
        noise_variance=noise,
    )
    gp.fit(X, y)
    return gp


def box(d=1, lo=-3.0, hi=3.0):
    return SearchSpace(
        dimension=d, lower_bounds=lo * np.ones(d), upper_bounds=hi * np.ones(d),
        log_scale=np.zeros(d, dtype=bool),
    )


# --------------------------------------------------------------------------- #
# normal helpers (no scipy)
# --------------------------------------------------------------------------- #
def test_normal_helpers_known_values():
    assert standard_normal_pdf(np.array([0.0]))[0] == pytest.approx(
        1.0 / np.sqrt(2 * np.pi)
    )
    assert standard_normal_cdf(np.array([0.0]))[0] == pytest.approx(0.5)
    assert standard_normal_cdf(np.array([5.0]))[0] == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Expected Improvement
# --------------------------------------------------------------------------- #
def test_ei_is_nonnegative_and_selects_in_box(rng):
    gp = fitted_gp(rng)
    ei = ExpectedImprovement(n_candidates=300)
    space = box()
    candidates = space.sample(300, np.random.default_rng(0))
    values = ei._scores(gp, space, candidates, rng)
    assert np.all(values >= 0.0)
    choice = ei.select(gp, space, np.random.default_rng(1))
    assert space.contains(choice)


def test_ei_prefers_the_promising_region(rng):
    # EI should point near the parabola's minimum (x ≈ 1), away from the data
    gp = fitted_gp(rng)
    choice = ExpectedImprovement(n_candidates=800).select(gp, box(), rng)
    assert abs(choice[0] - 1.0) < 1.0


def test_ei_value_matches_closed_form(rng):
    gp = fitted_gp(rng)
    query = np.array([[0.5], [1.0], [2.0]])
    incumbent = float(np.min(gp.observed_targets))
    posterior = gp.predict(query)
    mu, sigma = posterior.mean, np.sqrt(posterior.variance)
    z = (incumbent - mu) / sigma
    expected = (incumbent - mu) * standard_normal_cdf(z) + sigma * standard_normal_pdf(z)
    np.testing.assert_allclose(
        expected_improvement_values(gp, query, incumbent), expected, rtol=1e-9
    )


def test_ei_zero_variance_gives_plain_improvement(rng):
    gp = fitted_gp(rng, noise=1e-10)
    # at a training point the posterior std ≈ 0 → EI ≈ max(incumbent − μ, 0)
    x_train = gp.train_inputs[:1]
    incumbent = float(np.min(gp.observed_targets))
    ei = expected_improvement_values(gp, x_train, incumbent)
    assert ei[0] >= 0.0 and ei[0] < 1e-3


# --------------------------------------------------------------------------- #
# Upper/Lower Confidence Bound
# --------------------------------------------------------------------------- #
def test_ucb_score_is_beta_sigma_minus_mu(rng):
    gp = fitted_gp(rng)
    space = box()
    candidates = space.sample(50, np.random.default_rng(0))
    ucb = UpperConfidenceBound(beta=2.5)
    posterior = gp.predict(candidates)
    expected = 2.5 * np.sqrt(posterior.variance) - posterior.mean
    np.testing.assert_allclose(
        ucb._scores(gp, space, candidates, rng), expected, rtol=1e-12
    )


def test_ucb_beta_zero_is_pure_exploitation(rng):
    # β = 0 → maximise −μ → minimise the posterior mean
    gp = fitted_gp(rng)
    space = box()
    choice = UpperConfidenceBound(beta=0.0, n_candidates=800).select(gp, space, rng)
    grid = np.linspace(-3, 3, 400).reshape(-1, 1)
    best_mean_point = grid[np.argmin(gp.predict(grid).mean)]
    assert abs(choice[0] - best_mean_point[0]) < 0.2


def test_ucb_high_beta_explores(rng):
    # large β pushes selection toward high-uncertainty regions (away from data)
    gp = fitted_gp(rng)
    space = box()
    exploit = UpperConfidenceBound(beta=0.0, n_candidates=800).select(
        gp, space, np.random.default_rng(2)
    )
    explore = UpperConfidenceBound(beta=20.0, n_candidates=800).select(
        gp, space, np.random.default_rng(2)
    )
    nearest_data = lambda p: np.min(np.abs(gp.train_inputs[:, 0] - p[0]))  # noqa: E731
    assert nearest_data(explore) >= nearest_data(exploit)


def test_ucb_rejects_negative_beta():
    with pytest.raises(ValueError):
        UpperConfidenceBound(beta=-1.0)


# --------------------------------------------------------------------------- #
# Entropy Search
# --------------------------------------------------------------------------- #
def test_shannon_entropy_known_values():
    assert shannon_entropy(np.array([1.0, 0.0, 0.0])) == pytest.approx(0.0)
    assert shannon_entropy(np.array([0.5, 0.5])) == pytest.approx(np.log(2.0))
    n = 8
    assert shannon_entropy(np.full(n, 1.0 / n)) == pytest.approx(np.log(n))


def test_pmin_concentrates_with_more_data(rng):
    # a sharply-determined parabola: p_min should put most mass near x = 1
    gp = fitted_gp(rng, n=15)
    es = EntropySearch(n_optimum_samples=2000, n_representers=40)
    representers, pmin = es.optimum_distribution(gp, box(), rng)
    assert pmin.shape == (40,)
    assert pmin.sum() == pytest.approx(1.0)
    # the representer carrying the most mass sits near the true minimiser
    assert abs(representers[np.argmax(pmin)][0] - 1.0) < 1.0


def test_expected_entropy_reduction_is_nonnegative(rng):
    # observing anywhere cannot, in expectation, increase entropy about θ*
    # (mutual information ≥ 0); allow a small MC tolerance
    gp = fitted_gp(rng)
    es = EntropySearch(n_optimum_samples=800, n_representers=30, n_fantasies=8)
    representers, _ = es.optimum_distribution(gp, box(), rng)
    for theta in (np.array([0.0]), np.array([1.0]), np.array([2.5])):
        alpha = es.expected_entropy_reduction(gp, theta, representers, rng)
        assert alpha >= -0.05


def test_entropy_search_selects_within_box(rng):
    gp = fitted_gp(rng)
    es = EntropySearch(
        n_optimum_samples=300, n_representers=20, n_fantasies=5, n_candidates=40
    )
    choice = es.select(gp, box(), rng)
    assert box().contains(choice)
    assert choice.shape == (1,)


def test_entropy_search_values_where_belief_is_unresolved(rng):
    # ES targets where p_min is still spread, not where the value is best:
    # querying near the unexplored region should reduce entropy more than
    # re-querying a well-pinned training point
    gp = fitted_gp(rng, n=6)
    es = EntropySearch(n_optimum_samples=1500, n_representers=40, n_fantasies=10)
    representers, _ = es.optimum_distribution(gp, box(), np.random.default_rng(3))
    # a point sitting on top of existing data (well-determined) ...
    near_data = gp.train_inputs[0]
    gain_known = es.expected_entropy_reduction(
        gp, near_data, representers, np.random.default_rng(4)
    )
    # ... vs a point in the under-sampled tail
    gap = np.array([-2.5])
    gain_unknown = es.expected_entropy_reduction(
        gp, gap, representers, np.random.default_rng(4)
    )
    assert gain_unknown >= gain_known - 0.02


def test_entropy_search_is_deterministic_given_seed(rng):
    gp = fitted_gp(rng)
    es = EntropySearch(
        n_optimum_samples=200, n_representers=15, n_fantasies=4, n_candidates=30
    )
    a = es.select(gp, box(), np.random.default_rng(7))
    b = es.select(gp, box(), np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


def test_entropy_search_param_validation():
    with pytest.raises(ValueError):
        EntropySearch(n_optimum_samples=0)
    with pytest.raises(ValueError):
        EntropySearch(n_representers=0)


# --------------------------------------------------------------------------- #
# shared base behaviour
# --------------------------------------------------------------------------- #
def test_select_validates_dimensions(rng):
    gp = fitted_gp(rng, d=1)
    with pytest.raises(ValueError):
        ExpectedImprovement().select(gp, box(d=2), rng)
    with pytest.raises(TypeError):
        ExpectedImprovement().select("not a gp", box(), rng)


def test_acquisition_is_abstract():
    with pytest.raises(TypeError):
        AcquisitionFunction()  # _scores unimplemented


def test_select_is_reproducible(rng):
    gp = fitted_gp(rng)
    space = box()
    ei = ExpectedImprovement(n_candidates=100)
    a = ei.select(gp, space, np.random.default_rng(11))
    b = ei.select(gp, space, np.random.default_rng(11))
    np.testing.assert_array_equal(a, b)
