"""Unit tests for optimization.objective.ObjectiveFunction (notation §6 cost)."""

import numpy as np
import pytest

from inverted_pendulum.core.constants import PENALTY
from inverted_pendulum.metrics.stability_metrics import overshoot, settling_time
from inverted_pendulum.optimization.objective import ObjectiveFunction


def make(**overrides):
    params = dict(Mp_desired=10.0, Ts_desired=2.0, w1=1.0, w2=1.0)
    params.update(overrides)
    return ObjectiveFunction(**params)


def test_cost_formula_matches_notation(make_result):
    # known trajectory: Mp = 20%, Ts known from the band entry
    r = make_result([1.0, 0.5, -0.2, 0.015, 0.01, 0.005], dt=0.01)
    obj = make(Mp_desired=10.0, Ts_desired=0.02, w1=2.0, w2=3.0)
    Mp, Ts = overshoot(r), settling_time(r)
    expected = 2.0 * ((Mp - 10.0) / 10.0) ** 2 + 3.0 * ((Ts - 0.02) / 0.02) ** 2
    assert obj.evaluate(r) == pytest.approx(expected)


def test_perfect_match_costs_zero(make_result):
    r = make_result([1.0, 0.5, -0.1, 0.015, 0.01, 0.005], dt=0.01)
    obj = make(Mp_desired=overshoot(r), Ts_desired=settling_time(r))
    assert obj.evaluate(r) == pytest.approx(0.0)


def test_diverged_run_gets_the_finite_penalty(make_result):
    r = make_result([1.0, 2.0], diverged=True)
    obj = make()
    assert obj.evaluate(r) == PENALTY
    assert np.isfinite(obj.evaluate(r))
    custom = make(penalty=42.0)
    assert custom.evaluate(r) == 42.0


def test_evaluate_is_always_finite(make_result):
    # never settles (Ts capped at horizon), huge overshoot: still finite
    r = make_result([1.0, -50.0, 40.0, -30.0])
    assert np.isfinite(make().evaluate(r))


def test_metric_passthroughs(make_result):
    r = make_result([1.0, 0.5, -0.2, 0.0], controls=2.0 * np.ones((4, 1)), dt=0.01)
    obj = make()
    assert obj.compute_overshoot(r) == pytest.approx(overshoot(r))
    assert obj.compute_settling_time(r) == pytest.approx(settling_time(r))
    assert obj.compute_control_effort(r) == pytest.approx(4.0 * 0.01 * 3)


def test_weights_scale_their_terms(make_result):
    r = make_result([1.0, 0.5, -0.2, 0.015, 0.01, 0.005], dt=0.01)
    only_mp = make(w1=1.0, w2=0.0)
    only_ts = make(w1=0.0, w2=1.0)
    both = make(w1=1.0, w2=1.0)
    assert both.evaluate(r) == pytest.approx(only_mp.evaluate(r) + only_ts.evaluate(r))


@pytest.mark.parametrize(
    "bad",
    [
        dict(Mp_desired=0.0),
        dict(Ts_desired=-1.0),
        dict(w1=-0.1),
        dict(w2=-0.1),
        dict(penalty=np.inf),
        dict(penalty=0.0),
    ],
)
def test_validation_rejects_bad_params(bad):
    with pytest.raises(ValueError):
        make(**bad)


def test_rejects_non_result():
    with pytest.raises(TypeError):
        make().evaluate("not a result")
