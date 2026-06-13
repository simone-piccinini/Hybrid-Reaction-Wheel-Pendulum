"""Unit tests for optimization.marginal_likelihood (ML-II value and gradient)."""

import numpy as np
import pytest

from inverted_pendulum.optimization.gaussian_process import GaussianProcess
from inverted_pendulum.optimization.kernels.matern import Matern52ARD
from inverted_pendulum.optimization.kernels.squared_exponential import (
    SquaredExponentialARD,
)
from inverted_pendulum.optimization.marginal_likelihood import (
    fit_hyperparameters,
    negative_log_marginal_likelihood,
)


@pytest.fixture
def rng():
    return np.random.default_rng(99)


def standardized_data(rng, n=10, d=2):
    X = rng.uniform(-2.0, 2.0, size=(n, d))
    y = np.sin(X[:, 0]) - 0.3 * X[:, 1]
    return X, (y - y.mean()) / y.std()


@pytest.mark.parametrize("cls", [SquaredExponentialARD, Matern52ARD])
def test_gradient_matches_finite_differences(rng, cls):
    # the load-bearing check: R&W eq. 5.9 + log-space chain rule against
    # central finite differences of eq. 5.8
    X, y = standardized_data(rng)
    template = cls(lengthscales=np.ones(2), signal_variance=1.0)
    log_phi = rng.uniform(-0.5, 0.5, size=4)  # (log ℓ₁, log ℓ₂, log σ_f, log σ_n)
    value, gradient = negative_log_marginal_likelihood(log_phi, X, y, template)
    assert np.isfinite(value)
    h = 1e-6
    for j in range(4):
        up, dn = log_phi.copy(), log_phi.copy()
        up[j] += h
        dn[j] -= h
        fd = (
            negative_log_marginal_likelihood(up, X, y, template)[0]
            - negative_log_marginal_likelihood(dn, X, y, template)[0]
        ) / (2 * h)
        assert gradient[j] == pytest.approx(fd, rel=1e-4, abs=1e-7)


def test_value_matches_the_gp_computation(rng):
    # eq. 5.8 here must equal Algorithm 2.1 line 7 in the GP (same φ)
    X, y = standardized_data(rng)
    kernel = SquaredExponentialARD(lengthscales=[0.8, 1.2], signal_variance=1.5)
    noise_variance = 1e-2
    gp = GaussianProcess(kernel, noise_variance=noise_variance)
    gp.fit(X, y)  # y already standardised → internal transform ≈ identity
    log_phi = np.log(
        np.concatenate([kernel.hyperparams, [np.sqrt(noise_variance)]])
    )
    nll, _ = negative_log_marginal_likelihood(log_phi, X, y, kernel)
    assert -nll == pytest.approx(gp.log_marginal_likelihood(), rel=1e-6)


def test_fit_never_regresses_from_the_warm_start(rng):
    X, y = standardized_data(rng)
    kernel = SquaredExponentialARD(lengthscales=[2.0, 2.0], signal_variance=0.5)
    start_nll, _ = negative_log_marginal_likelihood(
        np.log(np.concatenate([kernel.hyperparams, [0.1]])), X, y, kernel
    )
    fitted_kernel, fitted_noise, result = fit_hyperparameters(
        kernel, 0.01, X, y, rng, n_restarts=3
    )
    assert result.fun <= start_nll + 1e-9
    assert fitted_noise > 0.0
    assert type(fitted_kernel) is SquaredExponentialARD


def test_fit_learns_the_noise_on_pure_noise_targets(rng):
    # white-noise targets: ML-II should attribute most variance to σ_n²
    X = rng.uniform(-2, 2, size=(30, 2))
    y = rng.standard_normal(30)  # standardised white noise
    kernel = SquaredExponentialARD(lengthscales=np.ones(2), signal_variance=1.0)
    _, fitted_noise, _ = fit_hyperparameters(kernel, 0.5, X, y, rng, n_restarts=3)
    assert fitted_noise > 0.05


def test_is_deterministic_given_the_seed(rng):
    X, y = standardized_data(rng)
    kernel = Matern52ARD(lengthscales=np.ones(2), signal_variance=1.0)
    k1, n1, _ = fit_hyperparameters(
        kernel, 0.1, X, y, np.random.default_rng(5), n_restarts=2
    )
    k2, n2, _ = fit_hyperparameters(
        kernel, 0.1, X, y, np.random.default_rng(5), n_restarts=2
    )
    np.testing.assert_array_equal(k1.lengthscales, k2.lengthscales)
    assert n1 == n2


def test_rejects_malformed_log_phi(rng):
    X, y = standardized_data(rng)
    kernel = SquaredExponentialARD(lengthscales=np.ones(2), signal_variance=1.0)
    with pytest.raises(ValueError):
        negative_log_marginal_likelihood(np.zeros(3), X, y, kernel)
    with pytest.raises(ValueError):
        fit_hyperparameters(kernel, 0.1, X, y, rng, n_restarts=0)
