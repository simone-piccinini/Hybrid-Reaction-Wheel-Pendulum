"""Unit tests for metrics.performance_metrics (control effort, oscillation energy)."""

import numpy as np
import pytest

from inverted_pendulum.metrics.performance_metrics import (
    control_effort,
    itae,
    oscillation_energy,
)


# --------------------------------------------------------------------------- #
# control effort
# --------------------------------------------------------------------------- #
def test_effort_constant_control(make_result):
    # u ≡ 3 V over 5 steps of 10 ms: 9 · 0.01 · 4 intervals
    r = make_result(np.zeros(5), controls=3.0 * np.ones((5, 1)), dt=0.01)
    assert control_effort(r) == pytest.approx(9.0 * 0.01 * 4)


def test_effort_zero_control(make_result):
    assert control_effort(make_result(np.zeros(10))) == 0.0


def test_effort_nonuniform_grid(make_result):
    # left-Riemann: 2²·1 + 3²·2 = 22 (the last sample is never integrated)
    r = make_result(
        np.zeros(3),
        controls=np.array([[2.0], [3.0], [99.0]]),
        time=np.array([0.0, 1.0, 3.0]),
    )
    assert control_effort(r) == pytest.approx(22.0)


def test_effort_single_sample_is_zero(make_result):
    r = make_result([1.0], controls=np.array([[5.0]]))
    assert control_effort(r) == 0.0


def test_effort_requires_increasing_time(make_result):
    r = make_result(np.zeros(3), time=np.array([0.0, 0.0, 1.0]))
    with pytest.raises(ValueError):
        control_effort(r)


def test_effort_rejects_non_result():
    with pytest.raises(TypeError):
        control_effort([[1.0]])


# --------------------------------------------------------------------------- #
# oscillation energy
# --------------------------------------------------------------------------- #
def test_oscillation_constant_rate(make_result):
    states = np.zeros((6, 4))
    states[:, 1] = 2.0  # theta_p_dot ≡ 2 rad/s
    r = make_result(states=states, dt=0.1)
    assert oscillation_energy(r) == pytest.approx(4.0 * 0.1 * 5)


def test_oscillation_sine_rate_integral(make_result):
    # ∫₀¹ sin²(2πt) dt = 1/2
    t = np.arange(0.0, 1.0, 1e-3)
    states = np.zeros((t.shape[0], 4))
    states[:, 1] = np.sin(2.0 * np.pi * t)
    r = make_result(states=states, time=t)
    assert oscillation_energy(r) == pytest.approx(0.5, rel=1e-2)


def test_oscillation_at_rest_is_zero(make_result):
    assert oscillation_energy(make_result(np.ones(8))) == 0.0


def test_oscillation_other_state_index(make_result):
    states = np.zeros((4, 4))
    states[:, 3] = 1.0  # wheel rate
    r = make_result(states=states, dt=0.5)
    assert oscillation_energy(r, state_index=3) == pytest.approx(1.5)
    assert oscillation_energy(r, state_index=1) == 0.0


def test_oscillation_rejects_bad_index(make_result):
    with pytest.raises(ValueError):
        oscillation_energy(make_result(np.zeros(3)), state_index=7)


# --------------------------------------------------------------------------- #
# ITAE  —  ∫ t·|e(t)| dt
# --------------------------------------------------------------------------- #
def test_itae_constant_error(make_result):
    # e(t) = 1 over [0, T]:  ∫ t·1 dt = T²/2  (trapezoid exact for linear t)
    t = np.linspace(0.0, 2.0, 2001)
    r = make_result(np.ones(t.shape[0]), time=t)
    assert itae(r) == pytest.approx(2.0**2 / 2.0, rel=1e-9)


def test_itae_zero_error_is_zero(make_result):
    assert itae(make_result(np.zeros(50))) == 0.0


def test_itae_time_weights_late_error_more(make_result):
    # the same error pulse counts more when it occurs later in the run
    n = 200
    early = np.zeros(n); early[10:20] = 0.1
    late = np.zeros(n); late[180:190] = 0.1
    assert itae(make_result(late)) > itae(make_result(early))


def test_itae_uses_absolute_value(make_result):
    # sign of the error must not matter
    theta = np.array([0.3, -0.5, 0.2, -0.1])
    assert itae(make_result(theta)) == pytest.approx(itae(make_result(-theta)))


def test_itae_single_sample_is_zero(make_result):
    assert itae(make_result([0.5])) == 0.0


def test_itae_rejects_bad_index(make_result):
    with pytest.raises(ValueError):
        itae(make_result(np.zeros(5)), state_index=9)
    with pytest.raises(TypeError):
        itae("not a result")
