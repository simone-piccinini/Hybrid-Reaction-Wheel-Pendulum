"""Unit tests for optimization.objective.ObjectiveFunction (notation §6 cost).

The cost is  y = w_e·ITAE + w_u·∫u² + w_p·hinge(Mp) + w_t·hinge(Ts) [+ sat],
with asymmetric (hinge) spec penalties normalised by the max-acceptable value.
"""

import numpy as np
import pytest

from inverted_pendulum.core.constants import PENALTY
from inverted_pendulum.metrics.performance_metrics import control_effort, itae
from inverted_pendulum.metrics.stability_metrics import overshoot, settling_time
from inverted_pendulum.optimization.objective import ObjectiveFunction


def make(**overrides):
    params = dict(
        Mp_desired=5.0, Ts_desired=1.0, Mp_max=20.0, Ts_max=3.0,
        w_error=1.0, w_control=1.0, w_overshoot=100.0, w_settling=100.0,
    )
    params.update(overrides)
    return ObjectiveFunction(**params)


def hinge(value, desired, scale):
    return max(0.0, (value - desired) / scale) ** 2


# --------------------------------------------------------------------------- #
# the cost is the weighted sum of its terms
# --------------------------------------------------------------------------- #
def test_cost_is_the_weighted_sum_of_terms(make_result):
    r = make_result([1.0, 0.5, -0.3, 0.05, 0.0],
                    controls=2.0 * np.ones((5, 1)), dt=0.01)
    obj = make(w_error=3.0, w_control=0.5, w_overshoot=10.0, w_settling=7.0,
               Mp_desired=5.0, Mp_max=20.0, Ts_desired=0.01, Ts_max=0.05)
    expected = (
        3.0 * itae(r)
        + 0.5 * control_effort(r)
        + 10.0 * hinge(overshoot(r), 5.0, 20.0)
        + 7.0 * hinge(settling_time(r), 0.01, 0.05)
    )
    assert obj.evaluate(r) == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# the hinge is asymmetric — zero while within spec
# --------------------------------------------------------------------------- #
def test_no_penalty_when_within_spec(make_result):
    # monotone decay: no overshoot, and a generous Ts target -> both hinges 0;
    # zero control -> cost is exactly w_error·ITAE
    r = make_result([1.0, 0.5, 0.2, 0.05, 0.0], dt=0.01)
    obj = make(Mp_desired=50.0, Ts_desired=10.0, w_error=1.0, w_control=1.0)
    assert overshoot(r) == 0.0
    assert obj.evaluate(r) == pytest.approx(itae(r))


def test_beating_the_target_is_not_rewarded_below_zero(make_result):
    # a response well inside spec pays no spec penalty at all (no negative cost)
    r = make_result([1.0, 0.3, 0.0, 0.0], dt=0.01)
    obj = make(Mp_desired=50.0, Ts_desired=10.0, w_error=0.0, w_control=0.0)
    assert obj.evaluate(r) == pytest.approx(0.0)


def test_overshoot_penalty_grows_past_the_target(make_result):
    # 30% overshoot with Mp_desired=10, Mp_max=20 -> hinge ((30-10)/20)² = 1
    r = make_result([1.0, 0.5, -0.3, 0.0], dt=0.01)
    obj = make(Mp_desired=10.0, Mp_max=20.0, w_error=0.0, w_control=0.0,
               w_overshoot=10.0, w_settling=0.0)
    assert overshoot(r) == pytest.approx(30.0)
    assert obj.evaluate(r) == pytest.approx(10.0 * ((30.0 - 10.0) / 20.0) ** 2)


def test_lower_overshoot_costs_less(make_result):
    mild = make_result([1.0, 0.5, -0.05, 0.0], dt=0.01)   # 5% overshoot
    severe = make_result([1.0, 0.5, -0.40, 0.0], dt=0.01)  # 40% overshoot
    obj = make(Mp_desired=0.0 + 1e-9, w_error=0.0, w_control=0.0, w_settling=0.0)
    assert obj.evaluate(mild) < obj.evaluate(severe)


# --------------------------------------------------------------------------- #
# control energy and saturation
# --------------------------------------------------------------------------- #
def test_control_energy_term(make_result):
    r = make_result(np.zeros(5), controls=3.0 * np.ones((5, 1)), dt=0.01)
    obj = make(w_error=0.0, w_control=2.0, w_overshoot=0.0, w_settling=0.0)
    assert obj.evaluate(r) == pytest.approx(2.0 * control_effort(r))


def test_saturation_penalty_fires_only_past_U_max(make_result):
    obj = make(U_max=24.0, w_saturation=2.0,
               w_error=0.0, w_control=0.0, w_overshoot=0.0, w_settling=0.0)
    over = make_result(np.zeros(2), controls=np.array([[30.0], [0.0]]), dt=0.01)
    assert obj.evaluate(over) == pytest.approx(2.0 * (30.0 - 24.0) ** 2)
    under = make_result(np.zeros(2), controls=np.array([[20.0], [0.0]]), dt=0.01)
    assert obj.evaluate(under) == pytest.approx(0.0)


def test_no_saturation_term_when_U_max_is_none(make_result):
    r = make_result(np.zeros(2), controls=np.array([[100.0], [0.0]]), dt=0.01)
    obj = make(w_error=0.0, w_control=0.0, w_overshoot=0.0, w_settling=0.0)
    assert obj.U_max is None
    assert obj.evaluate(r) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# robustness (stability-margin) penalty — the reversed hinge
# --------------------------------------------------------------------------- #
def only_margins(**overrides):
    """Objective with only the robustness term active (performance weights 0)."""
    base = dict(w_error=0.0, w_control=0.0, w_overshoot=0.0, w_settling=0.0)
    base.update(overrides)
    return make(**base)


def test_no_margin_penalty_by_default(make_result):
    # weights default to 0 -> a fragile margin costs nothing, cost unchanged
    r = make_result(np.zeros(3), phase_margin_deg=1.0, gain_margin_db=0.1)
    assert make().w_phase_margin == 0.0 and make().w_gain_margin == 0.0
    assert make().evaluate(r) == pytest.approx(0.0)


def test_phase_margin_shortfall_is_penalised(make_result):
    # PM = 12° below PM_min = 30°: hinge w·((30-12)/30)²
    r = make_result(np.zeros(3), phase_margin_deg=12.0)
    obj = only_margins(PM_min=30.0, w_phase_margin=50.0)
    assert obj.evaluate(r) == pytest.approx(50.0 * ((30.0 - 12.0) / 30.0) ** 2)


def test_gain_margin_shortfall_is_penalised(make_result):
    # GM = 2 dB below GM_min = 6 dB: hinge w·((6-2)/6)²
    r = make_result(np.zeros(3), gain_margin_db=2.0)
    obj = only_margins(GM_min=6.0, w_gain_margin=20.0)
    assert obj.evaluate(r) == pytest.approx(20.0 * ((6.0 - 2.0) / 6.0) ** 2)


def test_margin_at_or_above_threshold_is_free(make_result):
    # PM/GM meeting the bar pay nothing (the controller may be MORE robust)
    r = make_result(np.zeros(3), phase_margin_deg=45.0, gain_margin_db=8.0)
    obj = only_margins(PM_min=30.0, GM_min=6.0,
                       w_phase_margin=50.0, w_gain_margin=20.0)
    assert obj.evaluate(r) == pytest.approx(0.0)


def test_thinner_margin_costs_more(make_result):
    obj = only_margins(PM_min=30.0, w_phase_margin=10.0)
    thin = make_result(np.zeros(3), phase_margin_deg=5.0)
    healthy = make_result(np.zeros(3), phase_margin_deg=25.0)
    assert obj.evaluate(thin) > obj.evaluate(healthy) > 0.0


def test_non_finite_margins_contribute_no_penalty(make_result):
    # nan (not computed) and +inf (never hits -180°, fully gain-robust) -> 0
    obj = only_margins(w_phase_margin=50.0, w_gain_margin=20.0)
    nan_pm = make_result(np.zeros(3), phase_margin_deg=float("nan"),
                         gain_margin_db=float("inf"))
    assert obj.evaluate(nan_pm) == pytest.approx(0.0)
    assert np.isfinite(obj.evaluate(nan_pm))


def test_margin_penalty_adds_on_top_of_performance(make_result):
    # the robustness term is additive with the rest of the cost
    r = make_result([1.0, 0.5, 0.2, 0.05, 0.0], dt=0.01,
                    phase_margin_deg=15.0)
    base = make(Mp_desired=50.0, Ts_desired=10.0, w_error=1.0, w_control=0.0)
    withpen = make(Mp_desired=50.0, Ts_desired=10.0, w_error=1.0, w_control=0.0,
                   PM_min=30.0, w_phase_margin=8.0)
    extra = 8.0 * ((30.0 - 15.0) / 30.0) ** 2
    assert withpen.evaluate(r) == pytest.approx(base.evaluate(r) + extra)


def test_diverged_run_ignores_margins(make_result):
    # a diverged rollout short-circuits to the flat PENALTY, margins irrelevant
    r = make_result([1.0, 2.0], diverged=True, phase_margin_deg=1.0)
    assert only_margins(w_phase_margin=1e6).evaluate(r) == PENALTY


# --------------------------------------------------------------------------- #
# divergence and finiteness
# --------------------------------------------------------------------------- #
def test_diverged_run_gets_the_finite_penalty(make_result):
    r = make_result([1.0, 2.0], diverged=True)
    assert make().evaluate(r) == PENALTY
    assert make(penalty=42.0).evaluate(r) == 42.0


def test_evaluate_is_always_finite(make_result):
    r = make_result([1.0, -50.0, 40.0, -30.0], controls=1e3 * np.ones((4, 1)))
    assert np.isfinite(make(U_max=24.0).evaluate(r))


# --------------------------------------------------------------------------- #
# metric passthroughs (class_diagram.md)
# --------------------------------------------------------------------------- #
def test_metric_passthroughs(make_result):
    r = make_result([1.0, 0.5, -0.2, 0.0], controls=2.0 * np.ones((4, 1)), dt=0.01)
    obj = make()
    assert obj.compute_overshoot(r) == pytest.approx(overshoot(r))
    assert obj.compute_settling_time(r) == pytest.approx(settling_time(r))
    assert obj.compute_control_effort(r) == pytest.approx(control_effort(r))
    assert obj.compute_itae(r) == pytest.approx(itae(r))


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "bad",
    [
        dict(Mp_max=0.0),
        dict(Ts_max=-1.0),
        dict(w_error=-0.1),
        dict(w_overshoot=-1.0),
        dict(U_max=0.0),
        dict(penalty=np.inf),
        dict(penalty=0.0),
        dict(PM_min=0.0),
        dict(GM_min=-1.0),
        dict(w_phase_margin=-1.0),
        dict(w_gain_margin=-2.0),
    ],
)
def test_validation_rejects_bad_params(bad):
    with pytest.raises(ValueError):
        make(**bad)


def test_rejects_non_result():
    with pytest.raises(TypeError):
        make().evaluate("not a result")
