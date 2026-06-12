"""Unit tests for core.types.SearchSpace."""

import numpy as np
import pytest

from inverted_pendulum.core.types import SearchSpace


def make(d=3):
    return SearchSpace(
        dimension=d,
        lower_bounds=np.full(d, -2.0),
        upper_bounds=np.full(d, 1.0),
        log_scale=np.array([True, False, True])[:d],
    )


# --------------------------------------------------------------------------- #
# construction / validation
# --------------------------------------------------------------------------- #
def test_fields_and_width():
    s = make()
    assert s.dimension == 3
    assert s.lower_bounds.dtype == np.float64
    assert s.log_scale.dtype == np.bool_
    assert np.allclose(s.width, 3.0)


def test_arrays_read_only():
    s = make()
    with pytest.raises(ValueError):
        s.lower_bounds[0] = 5.0


def test_rejects_length_mismatch():
    with pytest.raises(ValueError):
        SearchSpace(3, np.zeros(2), np.ones(3), np.array([True, False, True]))


def test_rejects_lower_ge_upper():
    with pytest.raises(ValueError):
        SearchSpace(2, np.array([0.0, 1.0]), np.array([1.0, 1.0]), np.array([True, True]))


# --------------------------------------------------------------------------- #
# sample (Latin hypercube)
# --------------------------------------------------------------------------- #
def test_sample_shape_and_within_bounds():
    s = make()
    X = s.sample(20, np.random.default_rng(0))
    assert X.shape == (20, 3)
    assert np.all(X >= s.lower_bounds) and np.all(X <= s.upper_bounds)


def test_sample_is_latin_hypercube():
    s = make()
    n = 12
    X = s.sample(n, np.random.default_rng(1))
    unit = (X - s.lower_bounds) / s.width
    for j in range(s.dimension):
        strata = np.floor(unit[:, j] * n).astype(int)
        assert sorted(strata.tolist()) == list(range(n))  # exactly one point per stratum


def test_sample_is_deterministic():
    s = make()
    a = s.sample(10, np.random.default_rng(7))
    b = s.sample(10, np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_sample_rejects_non_positive_n():
    with pytest.raises(ValueError):
        make().sample(0, np.random.default_rng(0))


# --------------------------------------------------------------------------- #
# clip / contains
# --------------------------------------------------------------------------- #
def test_clip_projects_into_box():
    s = make()
    out = s.clip(np.array([5.0, -9.0, 0.5]))
    assert np.allclose(out, [1.0, -2.0, 0.5])  # clamped, in-box value untouched
    assert s.contains(out)


def test_contains_in_and_out():
    s = make()
    assert s.contains(np.zeros(3))
    assert not s.contains(np.array([2.0, 0.0, 0.0]))
    assert not s.contains(np.zeros(2))  # wrong dimension


def test_clip_rejects_wrong_shape():
    with pytest.raises(ValueError):
        make().clip(np.zeros(4))


# --------------------------------------------------------------------------- #
# to_natural
# --------------------------------------------------------------------------- #
def test_to_natural_exponentiates_log_dims_only():
    s = make()  # log_scale = [True, False, True]
    theta = np.array([0.0, 0.5, np.log(10.0)])
    natural = s.to_natural(theta)
    assert np.allclose(natural, [1.0, 0.5, 10.0])  # exp on dims 0,2; identity on dim 1
