"""Unit tests for optimization.gaussian_process (R&W Algorithm 2.1 surrogate)."""

import numpy as np
import pytest

from inverted_pendulum.core.types import GPPosterior
from inverted_pendulum.optimization.gaussian_process import (
    GaussianProcess,
    cholesky_with_escalating_jitter,
)
from inverted_pendulum.optimization.kernels.squared_exponential import (
    SquaredExponentialARD,
)


def make_kernel(d=2, ell=1.0, sf2=1.0):
    return SquaredExponentialARD(
        lengthscales=ell * np.ones(d), signal_variance=sf2
    )


@pytest.fixture
def rng():
    return np.random.default_rng(20260612)


def toy_data(rng, n=12, d=2):
    X = rng.uniform(-2.0, 2.0, size=(n, d))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2
    return X, y


# --------------------------------------------------------------------------- #
# fit / predict — Algorithm 2.1 behaviour
# --------------------------------------------------------------------------- #
def test_interpolates_training_data_with_small_noise(rng):
    X, y = toy_data(rng)
    gp = GaussianProcess(make_kernel(), noise_variance=1e-8)
    gp.fit(X, y)
    posterior = gp.predict(X)
    np.testing.assert_allclose(posterior.mean, y, atol=1e-3)
    assert np.all(posterior.variance < 1e-3)


def test_reverts_to_prior_far_from_data(rng):
    X, y = toy_data(rng)
    gp = GaussianProcess(make_kernel(sf2=2.0), noise_variance=1e-4)
    gp.fit(X, y)
    far = np.array([[50.0, 50.0]])
    posterior = gp.predict(far)
    # standardised prior: mean reverts to the data mean, variance to σ_f²·σ_y²
    assert posterior.mean[0] == pytest.approx(np.mean(y), rel=1e-6)
    assert posterior.variance[0] == pytest.approx(2.0 * np.var(y), rel=1e-3)


def test_predict_returns_the_contract(rng):
    X, y = toy_data(rng)
    gp = GaussianProcess(make_kernel(), noise_variance=1e-4)
    gp.fit(X, y)
    posterior = gp.predict(rng.uniform(-2, 2, size=(5, 2)))
    assert isinstance(posterior, GPPosterior)
    assert len(posterior) == 5
    assert np.all(posterior.variance >= 0.0)


def test_standardization_makes_predictions_affine_equivariant(rng):
    # fitting on a + b·y must shift/scale predictions exactly: the transform
    # is stored and inverted (optimization.md §6)
    X, y = toy_data(rng)
    query = rng.uniform(-2, 2, size=(4, 2))
    base = GaussianProcess(make_kernel(), noise_variance=1e-4)
    base.fit(X, y)
    scaled = GaussianProcess(make_kernel(), noise_variance=1e-4)
    scaled.fit(X, 3.0 - 7.0 * y)
    p_base, p_scaled = base.predict(query), scaled.predict(query)
    np.testing.assert_allclose(p_scaled.mean, 3.0 - 7.0 * p_base.mean, rtol=1e-9)
    np.testing.assert_allclose(p_scaled.variance, 49.0 * p_base.variance, rtol=1e-9)


def test_constant_targets_are_handled(rng):
    X, _ = toy_data(rng)
    gp = GaussianProcess(make_kernel(), noise_variance=1e-4)
    gp.fit(X, np.full(X.shape[0], 5.0))  # σ_y = 0 → guard kicks in
    posterior = gp.predict(X[:3])
    np.testing.assert_allclose(posterior.mean, 5.0 * np.ones(3), atol=1e-6)


# --------------------------------------------------------------------------- #
# joint posterior and sampling — the Entropy-Search prerequisites
# --------------------------------------------------------------------------- #
def test_joint_posterior_diagonal_matches_predict(rng):
    X, y = toy_data(rng)
    gp = GaussianProcess(make_kernel(), noise_variance=1e-3)
    gp.fit(X, y)
    query = rng.uniform(-2, 2, size=(6, 2))
    mean, cov = gp.joint_posterior(query)
    posterior = gp.predict(query)
    np.testing.assert_allclose(mean, posterior.mean, rtol=1e-9)
    np.testing.assert_allclose(np.diag(cov), posterior.variance, atol=1e-9)
    np.testing.assert_allclose(cov, cov.T)


def test_sample_posterior_matches_its_moments(rng):
    X, y = toy_data(rng)
    gp = GaussianProcess(make_kernel(), noise_variance=1e-3)
    gp.fit(X, y)
    query = rng.uniform(-2, 2, size=(3, 2))
    mean, cov = gp.joint_posterior(query)
    draws = gp.sample_posterior(query, n_samples=40000, rng=rng)
    assert draws.shape == (40000, 3)
    np.testing.assert_allclose(draws.mean(axis=0), mean, atol=0.05)
    np.testing.assert_allclose(np.cov(draws.T), cov, atol=0.05)


def test_sampling_is_deterministic_given_the_seed(rng):
    X, y = toy_data(rng)
    gp = GaussianProcess(make_kernel(), noise_variance=1e-3)
    gp.fit(X, y)
    query = np.zeros((2, 2))
    a = gp.sample_posterior(query, 5, np.random.default_rng(1))
    b = gp.sample_posterior(query, 5, np.random.default_rng(1))
    np.testing.assert_array_equal(a, b)


def test_observation_noise_maps_to_raw_units(rng):
    X, y = toy_data(rng)
    gp = GaussianProcess(make_kernel(), noise_variance=1e-2)
    gp.fit(X, y)
    assert gp.observation_noise_variance == pytest.approx(1e-2 * np.std(y) ** 2)


# --------------------------------------------------------------------------- #
# marginal likelihood and hyperparameter optimisation
# --------------------------------------------------------------------------- #
def test_log_marginal_likelihood_prefers_the_generating_kernel(rng):
    # data drawn from a known GP: the true hyperparameters should score a
    # higher marginal likelihood than absurd ones (the Occam balance, §3)
    kernel = make_kernel(d=1, ell=1.0, sf2=1.0)
    X = np.linspace(-3, 3, 25).reshape(-1, 1)
    K = kernel.matrix(X, X) + 1e-6 * np.eye(25)
    from inverted_pendulum.numerics.linalg import cholesky
    y = cholesky(K) @ rng.standard_normal(25)

    good = GaussianProcess(kernel, noise_variance=1e-4)
    good.fit(X, y)
    bad = GaussianProcess(make_kernel(d=1, ell=100.0), noise_variance=1e-4)
    bad.fit(X, y)
    assert good.log_marginal_likelihood() > bad.log_marginal_likelihood()


def test_optimize_hyperparameters_does_not_worsen_the_fit(rng):
    X, y = toy_data(rng)
    gp = GaussianProcess(make_kernel(ell=0.1, sf2=5.0), noise_variance=0.5)
    gp.fit(X, y)
    before = gp.log_marginal_likelihood()
    after = gp.optimize_hyperparameters(rng, n_restarts=2)
    assert after >= before - 1e-9  # warm start guarantees no regression


# --------------------------------------------------------------------------- #
# conditioning and guards
# --------------------------------------------------------------------------- #
def test_duplicate_points_are_rescued_by_the_baseline_jitter(rng):
    # duplicated rows make K exactly singular; the always-on PSD_JITTER of
    # numerical_standards §4 must absorb it without escalation
    X = np.vstack([np.zeros((3, 2)), rng.uniform(-1, 1, (3, 2))])
    y = rng.standard_normal(6)
    gp = GaussianProcess(make_kernel(), noise_variance=1e-12)
    gp.fit(X, y)
    assert np.all(np.isfinite(gp.predict(X).mean))


def test_escalating_jitter_helper_warns_then_succeeds():
    # eigenvalue −1e-8 defeats the 1e-9 baseline but yields to escalation
    nearly_psd = np.diag([1.0, -1e-8])
    with pytest.warns(UserWarning):
        L = cholesky_with_escalating_jitter(nearly_psd)
    assert np.all(np.isfinite(L))


def test_guards(rng):
    gp = GaussianProcess(make_kernel(), noise_variance=1e-3)
    with pytest.raises(RuntimeError):
        gp.predict(np.zeros((1, 2)))  # unfitted
    with pytest.raises(ValueError):
        gp.fit(np.zeros((3, 2)), np.zeros(2))  # mismatched
    with pytest.raises(ValueError):
        gp.fit(np.zeros((2, 3)), np.zeros(2))  # wrong dim
    with pytest.raises(ValueError):
        gp.fit(np.zeros((2, 2)), np.array([1.0, np.nan]))  # non-finite
    with pytest.raises(ValueError):
        GaussianProcess(make_kernel(), noise_variance=0.0)
    with pytest.raises(TypeError):
        GaussianProcess("not a kernel")


def test_escalating_jitter_helper_raises_past_the_cap():
    from inverted_pendulum.numerics.linalg import NotPositiveDefiniteError
    hopeless = np.array([[1.0, 0.0], [0.0, -1.0]])  # eigenvalue −1 ≪ −JITTER_MAX
    with pytest.raises(NotPositiveDefiniteError), pytest.warns(UserWarning):
        cholesky_with_escalating_jitter(hopeless)
