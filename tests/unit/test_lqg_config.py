"""Unit tests for core.types.LQGConfig (the vec(Q,R,W,V) decision vector)."""

import numpy as np
import pytest

from inverted_pendulum.core.types import LQGConfig

# Project dimensions: n_x=4, n_u=1, n_y=2  ->  d = 2*4 + 1 + 2 = 11.
QD = [2.0, 3.0, 4.0, 5.0]
RD = [1.5]
WD = [0.1, 0.2, 0.3, 0.4]
VD = [0.05, 0.06]


def make(qd=QD, rd=RD, wd=WD, vd=VD):
    return LQGConfig(np.diag(qd), np.diag(rd), np.diag(wd), np.diag(vd))


def test_dimensions():
    c = make()
    assert (c.n_x, c.n_u, c.n_y) == (4, 1, 2)
    assert c.dim == 11


def test_to_vector_is_log_of_diagonals_in_packing_order():
    theta = make().to_vector()
    assert theta.shape == (11,)
    expected = np.log(np.array(QD + RD + WD + VD))
    assert np.allclose(theta, expected)


def test_from_vector_builds_positive_diagonal_matrices():
    theta = np.log(np.array(QD + RD + WD + VD))
    c = LQGConfig.from_vector(theta, 4, 1, 2)
    assert np.allclose(np.diag(c.Q_lqr), QD)
    assert np.allclose(np.diag(c.R_lqr), RD)
    assert np.allclose(np.diag(c.W_process), WD)
    assert np.allclose(np.diag(c.V_measure), VD)
    assert np.allclose(c.Q_lqr - np.diag(np.diag(c.Q_lqr)), 0.0)  # strictly diagonal


def test_round_trip_config_vector_config():
    c = make()
    rebuilt = LQGConfig.from_vector(c.to_vector(), c.n_x, c.n_u, c.n_y)
    assert rebuilt == c


def test_round_trip_vector_config_vector():
    rng = np.random.default_rng(0)
    theta = rng.standard_normal(11)  # arbitrary log-space point
    assert np.allclose(LQGConfig.from_vector(theta, 4, 1, 2).to_vector(), theta)


def test_equality_is_tolerant_but_distinguishing():
    assert make() == make()
    assert not (make() == make(qd=[2.0, 3.0, 4.0, 5.001]))


def test_equality_with_other_type():
    assert (make() == 42) is False


def test_rejects_non_diagonal_matrix():
    Q = np.diag(QD).copy()
    Q[0, 1] = 0.1  # off-diagonal term
    with pytest.raises(ValueError):
        LQGConfig(Q, np.diag(RD), np.diag(WD), np.diag(VD))


def test_rejects_non_positive_diagonal():
    with pytest.raises(ValueError):  # zero entry -> log undefined
        LQGConfig(np.diag([2.0, 0.0, 4.0, 5.0]), np.diag(RD), np.diag(WD), np.diag(VD))


def test_rejects_non_square():
    with pytest.raises(ValueError):
        LQGConfig(np.ones((4, 3)), np.diag(RD), np.diag(WD), np.diag(VD))


def test_from_vector_rejects_wrong_length():
    with pytest.raises(ValueError):
        LQGConfig.from_vector(np.zeros(10), 4, 1, 2)


def test_arrays_are_read_only():
    c = make()
    with pytest.raises(ValueError):
        c.Q_lqr[0, 0] = 99.0
