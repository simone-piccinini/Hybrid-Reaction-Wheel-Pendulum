"""Unit tests for dynamics.time_response (step/free response, metrics, modal)."""

import numpy as np
import pytest

from inverted_pendulum.core.types import StateSpaceModel
from inverted_pendulum.dynamics.time_response import (
    Mode,
    TransientMetrics,
    closed_loop,
    free_response,
    modal_analysis,
    step_response,
    transient_metrics,
)


def first_order(tau=0.5):
    """ẋ = −x/τ + (1/τ)u, y = x  ->  G(s) = 1/(τs+1), unit DC gain."""
    return StateSpaceModel([[-1.0 / tau]], [[1.0 / tau]], [[1.0]], [[0.0]])


def second_order(wn, zeta):
    """Canonical 2nd-order: poles at −ζω_n ± jω_n√(1−ζ²), DC gain 1."""
    A = [[0.0, 1.0], [-wn**2, -2.0 * zeta * wn]]
    B = [[0.0], [wn**2]]
    C = [[1.0, 0.0]]
    return StateSpaceModel(A, B, C, [[0.0]])


# --------------------------------------------------------------------------- #
# step response — analytic first/second order
# --------------------------------------------------------------------------- #
def test_first_order_step_matches_closed_form():
    tau = 0.5
    time, y = step_response(first_order(tau), t_end=3.0, dt=1e-3)
    expected = 1.0 - np.exp(-time / tau)
    np.testing.assert_allclose(y[:, 0], expected, atol=1e-3)
    assert y[-1, 0] == pytest.approx(1.0, abs=1e-2)  # settles to DC gain


def test_second_order_step_overshoot_matches_formula():
    # %OS = 100 exp(−ζπ/√(1−ζ²)), peak time = π/(ω_n√(1−ζ²))
    wn, zeta = 2.0, 0.5
    time, y = step_response(second_order(wn, zeta), t_end=8.0, dt=1e-3)
    metrics = transient_metrics(time, y[:, 0])  # initial 0, final ~1
    expected_os = 100.0 * np.exp(-zeta * np.pi / np.sqrt(1 - zeta**2))
    expected_tp = np.pi / (wn * np.sqrt(1 - zeta**2))
    assert metrics.percent_overshoot == pytest.approx(expected_os, abs=1.0)
    assert metrics.peak_time == pytest.approx(expected_tp, abs=0.05)
    assert metrics.steady_state == pytest.approx(1.0, abs=1e-2)


def test_overdamped_step_has_no_overshoot():
    time, y = step_response(second_order(2.0, 1.5), t_end=8.0, dt=1e-3)
    metrics = transient_metrics(time, y[:, 0])
    assert metrics.percent_overshoot == pytest.approx(0.0, abs=0.5)


def test_step_response_input_guard():
    with pytest.raises(ValueError):
        step_response(first_order(), t_end=1.0, dt=1e-2, input_index=1)
    with pytest.raises(ValueError):
        step_response(first_order(), t_end=-1.0, dt=1e-2)


# --------------------------------------------------------------------------- #
# free (initial-condition) response — x(t) = e^{At} x0
# --------------------------------------------------------------------------- #
def test_free_response_first_order_decay():
    model = StateSpaceModel([[-2.0]], [[0.0]], [[1.0]], [[0.0]])
    time, states, outputs = free_response(model, [3.0], t_end=2.0, dt=1e-2)
    expected = 3.0 * np.exp(-2.0 * time)
    np.testing.assert_allclose(states[:, 0], expected, atol=1e-6)
    np.testing.assert_allclose(outputs[:, 0], expected, atol=1e-6)


def test_free_response_oscillator():
    # undamped oscillator ẍ + x = 0, x(0)=1, ẋ(0)=0  ->  x(t) = cos t
    model = StateSpaceModel([[0.0, 1.0], [-1.0, 0.0]], [[0.0], [0.0]],
                            [[1.0, 0.0]], [[0.0]])
    time, states, _ = free_response(model, [1.0, 0.0], t_end=2 * np.pi, dt=1e-3)
    np.testing.assert_allclose(states[:, 0], np.cos(time), atol=1e-4)


def test_free_response_requires_dt_for_continuous():
    with pytest.raises(ValueError):
        free_response(first_order(), [1.0], t_end=1.0)  # dt missing


def test_free_response_discrete_uses_model_dt():
    disc = StateSpaceModel([[0.5]], [[0.0]], [[1.0]], [[0.0]], is_discrete=True, dt=0.1)
    time, states, _ = free_response(disc, [1.0], t_end=0.5)
    np.testing.assert_allclose(states[:, 0], 0.5 ** np.arange(states.shape[0]))
    assert time[1] - time[0] == pytest.approx(0.1)


# --------------------------------------------------------------------------- #
# transient metrics
# --------------------------------------------------------------------------- #
def test_metrics_on_regulation_transient():
    # a clean decaying signal from 1 to 0 with one overshoot to −0.2
    time = np.linspace(0, 10, 1001)
    signal = np.exp(-time) * np.cos(2 * time)  # decays to 0, crosses below 0
    metrics = transient_metrics(time, signal, final=0.0)
    assert isinstance(metrics, TransientMetrics)
    assert metrics.percent_overshoot > 0.0   # it swings past zero
    assert metrics.settling_time < 10.0
    assert metrics.steady_state == 0.0


def test_metrics_flat_signal_is_zero():
    time = np.linspace(0, 1, 100)
    metrics = transient_metrics(time, np.full_like(time, 0.7))
    assert metrics == TransientMetrics(0.0, 0.0, 0.0, 0.0, 0.7)


def test_metrics_settling_band():
    # steps from 0 to 1, stays within 2% after t = 0.5
    time = np.linspace(0, 1, 1001)
    signal = np.where(time < 0.5, 0.5, 1.0)
    metrics = transient_metrics(time, signal, final=1.0, initial=0.0)
    assert metrics.settling_time == pytest.approx(0.5, abs=2e-3)


def test_metrics_input_guard():
    with pytest.raises(ValueError):
        transient_metrics(np.zeros(3), np.zeros(4))


# --------------------------------------------------------------------------- #
# closed loop
# --------------------------------------------------------------------------- #
def test_closed_loop_substitutes_feedback():
    A = np.array([[0.0, 1.0], [2.0, 0.0]])
    B = np.array([[0.0], [1.0]])
    model = StateSpaceModel(A, B, [[1.0, 0.0]], [[0.0]])
    K = np.array([[3.0, 4.0]])
    cl = closed_loop(model, K)
    np.testing.assert_allclose(cl.A, A - B @ K)
    np.testing.assert_allclose(cl.B, B)
    assert not cl.is_discrete


def test_closed_loop_validates_gain_shape():
    with pytest.raises(ValueError):
        closed_loop(first_order(), np.zeros((1, 2)))  # wrong n_x


# --------------------------------------------------------------------------- #
# modal analysis
# --------------------------------------------------------------------------- #
def test_modal_analysis_recovers_wn_and_zeta():
    wn, zeta = 3.0, 0.4
    modes = modal_analysis(second_order(wn, zeta))
    assert len(modes) == 2
    for mode in modes:
        assert isinstance(mode, Mode)
        assert mode.natural_frequency == pytest.approx(wn, rel=1e-6)
        assert mode.damping_ratio == pytest.approx(zeta, rel=1e-6)


def test_modal_analysis_flags_instability():
    # a positive real pole -> negative damping ratio
    unstable = StateSpaceModel([[1.0]], [[0.0]], [[1.0]], [[0.0]])
    mode = modal_analysis(unstable)[0]
    assert mode.damping_ratio < 0.0
    assert mode.natural_frequency == pytest.approx(1.0)


def test_modal_analysis_integrator_pole():
    integrator = StateSpaceModel([[0.0]], [[0.0]], [[1.0]], [[0.0]])
    mode = modal_analysis(integrator)[0]
    assert mode.natural_frequency == pytest.approx(0.0)
    assert mode.damping_ratio == 0.0
