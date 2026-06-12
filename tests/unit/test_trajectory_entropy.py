"""Unit tests for metrics.trajectory_entropy (the physical occupancy entropy)."""

import numpy as np
import pytest

from inverted_pendulum.metrics.trajectory_entropy import trajectory_entropy


def test_constant_trajectory_has_zero_entropy(make_result):
    assert trajectory_entropy(make_result(0.3 * np.ones(50))) == 0.0


def test_two_level_trajectory_is_ln2(make_result):
    theta = np.array([0.0, 1.0] * 25)  # half the mass in each extreme bin
    assert trajectory_entropy(make_result(theta)) == pytest.approx(np.log(2.0))


def test_four_equally_visited_levels_is_ln4(make_result):
    theta = np.array([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0] * 10)
    assert trajectory_entropy(make_result(theta)) == pytest.approx(np.log(4.0))


def test_uniform_spread_approaches_ln_nbins(make_result):
    theta = np.arange(3200) / 3200.0  # uniform occupancy of all 32 bins
    h = trajectory_entropy(make_result(theta))
    assert h == pytest.approx(np.log(32.0), rel=1e-3)


def test_bounded_by_ln_nbins(make_result):
    rng = np.random.default_rng(11)
    theta = rng.standard_normal(500)
    h = trajectory_entropy(make_result(theta), n_bins=16)
    assert 0.0 < h <= np.log(16.0)


def test_regulated_scores_below_persistent_oscillation(make_result):
    # a brief transient parked at 0 occupies few bins; a persistent sine
    # sweeps its full range forever
    t = np.arange(200)
    regulated = np.where(t < 5, 1.0 - 0.2 * t, 0.0)
    ringing = np.sin(0.3 * t)
    h_reg = trajectory_entropy(make_result(regulated))
    h_ring = trajectory_entropy(make_result(ringing))
    assert h_reg < h_ring


def test_custom_bins_change_the_bound(make_result):
    theta = np.arange(1000) / 1000.0
    assert trajectory_entropy(make_result(theta), n_bins=4) == pytest.approx(
        np.log(4.0), rel=1e-2
    )


def test_other_state_index(make_result):
    states = np.zeros((40, 4))
    states[:, 3] = np.array([0.0, 1.0] * 20)
    r = make_result(states=states)
    assert trajectory_entropy(r, state_index=3) == pytest.approx(np.log(2.0))
    assert trajectory_entropy(r, state_index=0) == 0.0


def test_input_guards(make_result):
    r = make_result(np.zeros(5))
    with pytest.raises(ValueError):
        trajectory_entropy(r, state_index=9)
    with pytest.raises(ValueError):
        trajectory_entropy(r, n_bins=1)
    with pytest.raises(TypeError):
        trajectory_entropy(np.zeros(5))
