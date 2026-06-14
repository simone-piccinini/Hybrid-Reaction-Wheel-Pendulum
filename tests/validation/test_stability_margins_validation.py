"""Validation of the LQG loop gain and stability margins against python-control.

The hand-written loop transfer function is cross-checked against
``control.frequency_response`` of the same controller·plant series (they agree
to machine precision), and the margin extractor against ``control.margin`` on
analytic continuous loops where the reference is reliable.

A documented caveat: ``control.margin`` *misses* a genuine −180° phase crossing
of this discrete LQG loop (it reports an infinite gain margin), whereas the
from-scratch grid-based extractor finds it. The crossing is real — the unwrapped
phase passes through −180° transversally with |L| < 1 — so the from-scratch
gain margin (~3.3 dB) is the correct one. Only ``tests/validation`` imports the
banned reference library.
"""

import numpy as np
import pytest
import control

from inverted_pendulum.control.lqr_controller import LQRController
from inverted_pendulum.core.types import StateSpaceModel
from inverted_pendulum.dynamics.frequency_response import (
    log_frequencies,
    loop_transfer_function,
    stability_margins,
)
from inverted_pendulum.estimation.kalman_filter import steady_state_kalman_gain
from inverted_pendulum.numerics.constants import ATOL, VALIDATION_RTOL
from inverted_pendulum.physical.motor import DCMotor
from inverted_pendulum.physical.pendulum import ReactionWheelPendulum
from inverted_pendulum.physical.wheel import ReactionWheel

DT = 0.01
KEEP = [0, 1, 3]


def reduced_plant_and_gains():
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
    model = StateSpaceModel(
        A[np.ix_(KEEP, KEEP)], B[KEEP], C[:, KEEP], np.zeros((2, 1))
    ).discretize(DT)
    K = LQRController.from_model(model, np.diag([50.0, 5.0, 0.05]), [[1.0]]).K_gain
    L = steady_state_kalman_gain(
        model, np.diag([1e-4, 1e-3, 1e-2]), np.diag([1e-6, 1e-4])
    ).L_gain
    return model, K, L


def control_loop_system(model, K, L):
    """The same loop L = −K_c·G built with python-control, for reference."""
    Ad, Bd, Cd = np.asarray(model.A), np.asarray(model.B), np.asarray(model.C)
    n_x = Ad.shape[0]
    closed, filt = Ad - Bd @ K, np.eye(n_x) - L @ Cd
    controller = control.ss(closed @ filt, closed @ L, -K @ filt, -K @ L, DT)
    plant = control.ss(Ad, Bd, Cd, np.zeros((Cd.shape[0], 1)), DT)
    return -(controller * plant)


def test_loop_transfer_function_matches_control():
    model, K, L = reduced_plant_and_gains()
    omega = log_frequencies(1e-2, 0.9 * np.pi / DT, 500)
    ours = loop_transfer_function(model, K, L, omega)
    mag, phase, _ = control.frequency_response(control_loop_system(model, K, L), omega)
    reference = np.squeeze(mag) * np.exp(1j * np.squeeze(phase))
    assert np.allclose(ours, reference, rtol=VALIDATION_RTOL, atol=ATOL)


def test_margins_match_control_on_analytic_loops():
    # control.margin is reliable on these continuous loops
    w = log_frequencies(1e-3, 1e3, 8000)
    for tf, Lc in [
        (control.tf([1], [1, 1, 0]), 1.0 / ((1j * w) * (1j * w + 1.0))),
        (control.tf([2], [1, 3, 3, 1]), 2.0 / ((1j * w + 1.0) ** 3)),
    ]:
        gm_c, pm_c, _, _ = control.margin(tf)
        m = stability_margins(w, Lc)
        assert m.phase_margin_deg == pytest.approx(pm_c, abs=0.3)
        if np.isfinite(gm_c):
            assert m.gain_margin_db == pytest.approx(20 * np.log10(gm_c), abs=0.2)


def test_phase_margin_matches_control_for_the_lqg_loop():
    # python-control gets the PHASE margin of the discrete LQG loop right (it is
    # the GAIN margin it misses — see module docstring)
    model, K, L = reduced_plant_and_gains()
    omega = log_frequencies(1e-2, 0.9 * np.pi / DT, 4000)
    ours = stability_margins(omega, loop_transfer_function(model, K, L, omega))
    _, pm_c, _, _ = control.margin(control_loop_system(model, K, L))
    assert ours.phase_margin_deg == pytest.approx(pm_c, abs=0.5)
