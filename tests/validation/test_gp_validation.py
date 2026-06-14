"""Validation of the GaussianProcess against scikit-learn.

numerical_standards.md §8: "gaussian_process predict → compare mean/variance to
a GPy/sklearn GP with an identical kernel." Targets are pre-standardised so
our internal transform is the identity and the models are exactly comparable;
sklearn's ``alpha`` absorbs our σ_n² + PSD_JITTER. Only tests/validation/
imports the reference libraries.
"""

import numpy as np
import pytest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern

from inverted_pendulum.core.constants import PSD_JITTER
from inverted_pendulum.numerics.constants import ATOL, VALIDATION_RTOL
from inverted_pendulum.optimization.gaussian_process import GaussianProcess
from inverted_pendulum.optimization.kernels.matern import Matern52ARD
from inverted_pendulum.optimization.kernels.squared_exponential import (
    SquaredExponentialARD,
)

LENGTHSCALES = np.array([0.7, 1.4, 2.1])
SIGNAL_VARIANCE = 1.8
NOISE_VARIANCE = 1e-2


@pytest.fixture
def data():
    rng = np.random.default_rng(31)
    X = rng.uniform(-2.0, 2.0, size=(15, 3))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] - 0.2 * X[:, 2] ** 2
    y = (y - y.mean()) / y.std()  # identity internal transform
    query = rng.uniform(-2.0, 2.0, size=(7, 3))
    return X, y, query


def reference(kernel_sk, X, y):
    return GaussianProcessRegressor(
        kernel=kernel_sk,
        alpha=NOISE_VARIANCE + PSD_JITTER,
        optimizer=None,
        normalize_y=False,
    ).fit(X, y)


CASES = [
    (
        SquaredExponentialARD(LENGTHSCALES, SIGNAL_VARIANCE),
        ConstantKernel(SIGNAL_VARIANCE, "fixed") * RBF(LENGTHSCALES, "fixed"),
    ),
    (
        Matern52ARD(LENGTHSCALES, SIGNAL_VARIANCE),
        ConstantKernel(SIGNAL_VARIANCE, "fixed")
        * Matern(LENGTHSCALES, "fixed", nu=2.5),
    ),
]


@pytest.mark.parametrize("ours_kernel,sk_kernel", CASES)
def test_predictive_moments_match_sklearn(data, ours_kernel, sk_kernel):
    X, y, query = data
    ours = GaussianProcess(ours_kernel, noise_variance=NOISE_VARIANCE)
    ours.fit(X, y)
    posterior = ours.predict(query)
    ref = reference(sk_kernel, X, y)
    mean_ref, std_ref = ref.predict(query, return_std=True)
    assert np.allclose(posterior.mean, mean_ref, rtol=VALIDATION_RTOL, atol=ATOL)
    assert np.allclose(
        posterior.variance, std_ref**2, rtol=VALIDATION_RTOL, atol=1e-8
    )


@pytest.mark.parametrize("ours_kernel,sk_kernel", CASES)
def test_log_marginal_likelihood_matches_sklearn(data, ours_kernel, sk_kernel):
    X, y, _ = data
    ours = GaussianProcess(ours_kernel, noise_variance=NOISE_VARIANCE)
    ours.fit(X, y)
    ref = reference(sk_kernel, X, y)
    assert ours.log_marginal_likelihood() == pytest.approx(
        ref.log_marginal_likelihood(), rel=VALIDATION_RTOL
    )


def test_joint_posterior_matches_sklearn_full_covariance(data):
    X, y, query = data
    ours = GaussianProcess(
        SquaredExponentialARD(LENGTHSCALES, SIGNAL_VARIANCE),
        noise_variance=NOISE_VARIANCE,
    )
    ours.fit(X, y)
    _, cov = ours.joint_posterior(query)
    sk_kernel = ConstantKernel(SIGNAL_VARIANCE, "fixed") * RBF(LENGTHSCALES, "fixed")
    _, cov_ref = reference(sk_kernel, X, y).predict(query, return_cov=True)
    assert np.allclose(cov, cov_ref, rtol=VALIDATION_RTOL, atol=1e-8)
