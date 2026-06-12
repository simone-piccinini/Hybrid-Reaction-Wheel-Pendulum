"""Unit tests for core.types.Dataset and core.types.GPPosterior."""

import numpy as np
import pytest

from inverted_pendulum.core.constants import ATOL
from inverted_pendulum.core.types import Dataset, GPPosterior


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def test_dataset_starts_empty():
    d = Dataset(dim=3)
    assert d.size() == 0 and len(d) == 0
    assert d.X.shape == (0, 3) and d.y.shape == (0,)
    assert d.dim == 3


def test_dataset_append_grows_and_preserves_order():
    d = Dataset(dim=2)
    d.append([1.0, 2.0], 0.5)
    d.append(np.array([3.0, 4.0]), -1.0)
    assert d.size() == 2
    assert np.allclose(d.X, [[1.0, 2.0], [3.0, 4.0]])
    assert np.allclose(d.y, [0.5, -1.0])
    assert d.X.dtype == np.float64 and d.y.dtype == np.float64


def test_dataset_views_are_read_only():
    d = Dataset(dim=2)
    d.append([1.0, 2.0], 0.5)
    with pytest.raises(ValueError):
        d.X[0, 0] = 9.0
    with pytest.raises(ValueError):
        d.y[0] = 9.0


def test_dataset_rejects_wrong_shape():
    d = Dataset(dim=2)
    with pytest.raises(ValueError):
        d.append([1.0, 2.0, 3.0], 0.0)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_dataset_rejects_non_finite(bad):
    d = Dataset(dim=2)
    with pytest.raises(ValueError):
        d.append([bad, 1.0], 0.0)
    with pytest.raises(ValueError):
        d.append([1.0, 2.0], bad)


def test_dataset_requires_positive_dim():
    with pytest.raises(ValueError):
        Dataset(dim=0)


# --------------------------------------------------------------------------- #
# GPPosterior
# --------------------------------------------------------------------------- #
def test_posterior_basic():
    p = GPPosterior(mean=np.array([1.0, 2.0]), variance=np.array([0.1, 0.2]))
    assert len(p) == 2
    assert np.allclose(p.mean, [1.0, 2.0])
    assert np.allclose(p.variance, [0.1, 0.2])


def test_posterior_single_query_coerced_to_length_one():
    p = GPPosterior(mean=1.5, variance=0.3)
    assert p.mean.shape == (1,) and p.variance.shape == (1,)


def test_posterior_clips_roundoff_negative_variance():
    p = GPPosterior(mean=np.array([0.0]), variance=np.array([-1e-12]))  # > -ATOL
    assert p.variance[0] == 0.0


def test_posterior_raises_on_large_negative_variance():
    with pytest.raises(ValueError):
        GPPosterior(mean=np.array([0.0]), variance=np.array([-1e-6]))  # < -ATOL


def test_posterior_rejects_length_mismatch():
    with pytest.raises(ValueError):
        GPPosterior(mean=np.array([1.0, 2.0]), variance=np.array([0.1]))


def test_posterior_arrays_read_only():
    p = GPPosterior(mean=np.array([1.0]), variance=np.array([0.1]))
    with pytest.raises(ValueError):
        p.variance[0] = 9.0


def test_posterior_negative_threshold_is_atol():
    # exactly at -ATOL is clipped (not a bug); just beyond it raises.
    ok = GPPosterior(mean=np.array([0.0]), variance=np.array([-ATOL]))
    assert ok.variance[0] == 0.0
