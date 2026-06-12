"""Unit tests for optimization.kernels (ARD SE and Matérn-5/2)."""

import dataclasses
import math

import numpy as np
import pytest

from inverted_pendulum.optimization.kernels.base import Kernel
from inverted_pendulum.optimization.kernels.matern import Matern52ARD
from inverted_pendulum.optimization.kernels.squared_exponential import (
    SquaredExponentialARD,
)

KERNELS = [SquaredExponentialARD, Matern52ARD]


def make(cls, d=3, **overrides):
    params = dict(lengthscales=np.array([1.0, 2.0, 0.5][:d]), signal_variance=1.5)
    params.update(overrides)
    return cls(**params)


@pytest.fixture
def rng():
    return np.random.default_rng(20260612)


# --------------------------------------------------------------------------- #
# covariance values and properties
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls", KERNELS)
def test_self_covariance_is_signal_variance(cls):
    k = make(cls)
    x = np.array([0.3, -1.0, 2.0])
    assert k.covariance(x, x) == pytest.approx(1.5)


def test_se_known_value():
    k = make(SquaredExponentialARD, lengthscales=[1.0, 2.0, 0.5], signal_variance=2.0)
    x1 = np.array([1.0, 0.0, 0.0])
    x2 = np.array([0.0, 2.0, 0.5])
    r2 = (1.0 / 1.0) ** 2 + (2.0 / 2.0) ** 2 + (0.5 / 0.5) ** 2  # = 3
    assert k.covariance(x1, x2) == pytest.approx(2.0 * math.exp(-1.5))


def test_matern_known_value():
    k = make(Matern52ARD, lengthscales=[1.0, 1.0, 1.0], signal_variance=1.0)
    x1, x2 = np.zeros(3), np.array([1.0, 0.0, 0.0])  # r = 1
    s5 = math.sqrt(5.0)
    assert k.covariance(x1, x2) == pytest.approx((1 + s5 + 5 / 3) * math.exp(-s5))


@pytest.mark.parametrize("cls", KERNELS)
def test_symmetry_and_decay(cls, rng):
    k = make(cls)
    a, b = rng.standard_normal(3), rng.standard_normal(3)
    assert k.covariance(a, b) == pytest.approx(k.covariance(b, a))
    far = a + np.array([50.0, 0.0, 0.0])
    assert k.covariance(a, far) < 1e-6 * k.covariance(a, a)


@pytest.mark.parametrize("cls", KERNELS)
def test_ard_lengthscales_weight_dimensions(cls):
    # a long lengthscale makes that dimension irrelevant
    k = make(cls, lengthscales=[1e3, 1.0, 1.0])
    x = np.zeros(3)
    moved_irrelevant = np.array([5.0, 0.0, 0.0])
    moved_relevant = np.array([0.0, 5.0, 0.0])
    assert k.covariance(x, moved_irrelevant) > k.covariance(x, moved_relevant)


@pytest.mark.parametrize("cls", KERNELS)
def test_gram_matrix_is_symmetric_psd(cls, rng):
    from inverted_pendulum.numerics.linalg import eigvals, is_symmetric

    k = make(cls)
    X = rng.standard_normal((8, 3))
    K = k.matrix(X, X)
    assert K.shape == (8, 8)
    assert is_symmetric(K)
    assert float(np.min(eigvals(K).real)) >= -1e-9  # Mercer: PSD


@pytest.mark.parametrize("cls", KERNELS)
def test_cross_matrix_entries(cls, rng):
    k = make(cls)
    X1, X2 = rng.standard_normal((4, 3)), rng.standard_normal((2, 3))
    K = k.matrix(X1, X2)
    assert K.shape == (4, 2)
    assert K[2, 1] == pytest.approx(k.covariance(X1[2], X2[1]))


# --------------------------------------------------------------------------- #
# gradients vs central finite differences — the ML-II prerequisite
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls", KERNELS)
def test_gradient_matches_finite_differences(cls, rng):
    k = make(cls)
    x1, x2 = rng.standard_normal(3), rng.standard_normal(3)
    analytic = k.gradient(x1, x2)
    h = 1e-6
    phi = k.hyperparams
    for j in range(phi.shape[0]):
        bumped_up, bumped_dn = phi.copy(), phi.copy()
        bumped_up[j] += h
        bumped_dn[j] -= h
        fd = (
            k.with_hyperparams(bumped_up).covariance(x1, x2)
            - k.with_hyperparams(bumped_dn).covariance(x1, x2)
        ) / (2 * h)
        assert analytic[j] == pytest.approx(fd, rel=1e-5, abs=1e-9)


@pytest.mark.parametrize("cls", KERNELS)
def test_gradient_at_coincident_points(cls):
    # at x1 == x2 the lengthscale gradients vanish; ∂k/∂σ_f = 2σ_f
    k = make(cls)
    x = np.array([0.1, 0.2, 0.3])
    grad = k.gradient(x, x)
    np.testing.assert_allclose(grad[:-1], np.zeros(3), atol=1e-12)
    assert grad[-1] == pytest.approx(2.0 * math.sqrt(1.5))


# --------------------------------------------------------------------------- #
# hyperparameter plumbing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls", KERNELS)
def test_hyperparams_round_trip(cls):
    k = make(cls)
    rebuilt = k.with_hyperparams(k.hyperparams)
    np.testing.assert_allclose(rebuilt.lengthscales, k.lengthscales)
    assert rebuilt.signal_variance == pytest.approx(k.signal_variance)
    assert type(rebuilt) is cls


@pytest.mark.parametrize("cls", KERNELS)
def test_validation_and_immutability(cls):
    with pytest.raises(ValueError):
        make(cls, lengthscales=[1.0, -1.0, 1.0])
    with pytest.raises(ValueError):
        make(cls, signal_variance=0.0)
    with pytest.raises(ValueError):
        make(cls).with_hyperparams(np.ones(2))  # wrong length
    with pytest.raises(ValueError):
        make(cls).covariance(np.zeros(2), np.zeros(2))  # wrong point dim
    k = make(cls)
    with pytest.raises(dataclasses.FrozenInstanceError):
        k.signal_variance = 2.0


def test_kernel_is_abstract():
    with pytest.raises(TypeError):
        Kernel(lengthscales=np.ones(2), signal_variance=1.0)
