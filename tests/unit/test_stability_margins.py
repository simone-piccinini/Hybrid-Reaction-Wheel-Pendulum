"""Unit tests for the loop transfer function and stability margins."""

import numpy as np
import pytest

from inverted_pendulum.control.lqr_controller import LQRController
from inverted_pendulum.core.types import StateSpaceModel
from inverted_pendulum.dynamics.frequency_response import (
    StabilityMargins,
    log_frequencies,
    loop_transfer_function,
    stability_margins,
)
from inverted_pendulum.estimation.kalman_filter import steady_state_kalman_gain
from inverted_pendulum.physical.motor import DCMotor
from inverted_pendulum.physical.pendulum import ReactionWheelPendulum
from inverted_pendulum.physical.wheel import ReactionWheel

DT = 0.01
KEEP = [0, 1, 3]  # drop the decoupled wheel angle


# --------------------------------------------------------------------------- #
# stability_margins on analytic loops with known margins
# --------------------------------------------------------------------------- #
def test_margins_first_order_integrator():
    # L(s) = 1/(s(s+1)): PM = 51.83° at ω = 0.786, infinite gain margin
    w = log_frequencies(1e-2, 1e2, 4000)
    L = 1.0 / ((1j * w) * (1j * w + 1.0))
    m = stability_margins(w, L)
    assert isinstance(m, StabilityMargins)
    assert m.phase_margin_deg == pytest.approx(51.83, abs=0.2)
    assert m.gain_crossover == pytest.approx(0.786, rel=1e-2)
    assert not np.isfinite(m.gain_margin_db)  # phase never reaches −180°


def test_margins_third_order_has_finite_gain_margin():
    # L(s) = 2/(s+1)³: GM = 12.04 dB at ω = 1.732, PM = 67.6°
    w = log_frequencies(1e-2, 1e2, 6000)
    L = 2.0 / ((1j * w + 1.0) ** 3)
    m = stability_margins(w, L)
    assert m.gain_margin_db == pytest.approx(12.04, abs=0.1)
    assert m.phase_crossover == pytest.approx(1.732, rel=1e-2)
    assert m.phase_margin_deg == pytest.approx(67.6, abs=0.3)


def test_margins_infinite_when_gain_below_one():
    # |L| < 1 everywhere: no gain crossover, infinite phase margin
    w = log_frequencies(1e-1, 1e1, 500)
    L = 0.1 / (1j * w + 1.0)
    m = stability_margins(w, L)
    assert not np.isfinite(m.phase_margin_deg)
    assert np.isnan(m.gain_crossover)


# --------------------------------------------------------------------------- #
# loop_transfer_function for the LQG loop
# --------------------------------------------------------------------------- #
def reduced_plant():
    plant = ReactionWheelPendulum(
        pendulum_mass=0.5, pendulum_length=0.2, body_inertia=0.025,
        pivot_friction=0.005,
        wheel=ReactionWheel(mass=0.2, radius=0.06, inertia=3.5e-4,
                            friction_coefficient=1e-4),
        motor=DCMotor(resistance=1.2, torque_constant=0.08,
                      back_emf_constant=0.08),
    )
    cont = plant.linearize()
    A, B, C = cont.A, cont.B, cont.C
    return StateSpaceModel(
        A[np.ix_(KEEP, KEEP)], B[KEEP], C[:, KEEP], np.zeros((2, 1))
    ).discretize(DT)


def lqg_gains(model):
    K = LQRController.from_model(model, np.diag([50.0, 5.0, 0.05]), [[1.0]]).K_gain
    L = steady_state_kalman_gain(
        model, np.diag([1e-4, 1e-3, 1e-2]), np.diag([1e-6, 1e-4])
    ).L_gain
    return K, L


def test_loop_transfer_function_shape_and_finite():
    model = reduced_plant()
    K, L = lqg_gains(model)
    w = log_frequencies(1e-2, 100.0, 200)
    loop = loop_transfer_function(model, K, L, w)
    assert loop.shape == (200,)
    assert loop.dtype == np.complex128
    assert np.all(np.isfinite(loop))


def test_stable_lqg_loop_has_positive_phase_margin():
    # the LQG design is stabilising (separation principle), so the loop must
    # have a positive phase margin
    model = reduced_plant()
    K, L = lqg_gains(model)
    w = log_frequencies(1e-2, 0.9 * np.pi / DT, 2000)
    margins = stability_margins(w, loop_transfer_function(model, K, L, w))
    assert margins.phase_margin_deg > 0.0
    # this particular design is known fragile (Doyle): small PM, finite GM
    assert margins.phase_margin_deg < 30.0
    assert np.isfinite(margins.gain_margin_db)


def test_loop_transfer_function_guards():
    model = reduced_plant()
    K, L = lqg_gains(model)
    with pytest.raises(ValueError):  # continuous model rejected
        loop_transfer_function(
            StateSpaceModel(model.A, model.B, model.C, model.D), K, L, [1.0]
        )
    with pytest.raises(ValueError):  # wrong K shape
        loop_transfer_function(model, np.zeros((1, 2)), L, [1.0])
    with pytest.raises(ValueError):  # wrong L shape
        loop_transfer_function(model, K, np.zeros((3, 1)), [1.0])
