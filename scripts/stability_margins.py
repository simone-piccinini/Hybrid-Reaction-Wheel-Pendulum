#!/usr/bin/env python3
"""Open-loop Bode and stability margins of the LQG loop (the expert's plot).

Instead of the closed-loop time response, this plots the **loop transfer
function** L(jω) = −K_c(jω)·G(jω) of the broken loop (plant + LQG controller),
read at the plant input, and reports the two margins an expert checks:

    Phase Margin (PM)  — how much extra phase lag (e.g. microcontroller latency)
                         the loop tolerates before instability.   Want ≳ 30–45°.
    Gain Margin (GM)   — how much the loop gain may change (K_t, battery droop)
                         before instability.                       Want ≳ 6 dB.

The wheel angle is a decoupled integrator (uncontrollable and unobservable from
the sensors), so it is dropped to obtain the controllable + observable model on
which the steady-state LQG controller — and hence the loop gain — is defined.

NOTE (Doyle 1978, "Guaranteed Margins for LQG Regulators: None"): LQR and the
Kalman filter are each robust, but their combination need not be — this plot is
exactly how you find out.

Usage
-----
    PYTHONPATH=src python scripts/stability_margins.py [CONFIG] [-o OUTDIR]
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np

from inverted_pendulum.control.lqr_controller import LQRController
from inverted_pendulum.core.types import StateSpaceModel
from inverted_pendulum.dynamics.frequency_response import (
    log_frequencies,
    loop_transfer_function,
    stability_margins,
)
from inverted_pendulum.estimation.kalman_filter import steady_state_kalman_gain
from inverted_pendulum.experiment.manager import ExperimentManager

# A sensible LQG weighting on the reduced [theta_p, theta_p_dot, theta_w_dot]
# model (the optimiser would tune these; fixed here for a clear demonstration).
Q_RED = np.diag([50.0, 5.0, 0.05])
R_LQR = np.array([[1.0]])
W_RED = np.diag([1.0e-4, 1.0e-3, 1.0e-2])
V_MEAS = np.diag([1.0e-6, 1.0e-4])
KEEP = [0, 1, 3]  # drop the wheel angle (state index 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default="configs/pendulum_sensible.yaml")
    parser.add_argument("-o", "--out", default="results/stability_margins")
    args = parser.parse_args()

    manager = ExperimentManager.from_config_file(args.config)
    continuous = manager.build_engine().linear_model.continuous
    dt = manager.config.simulation.dt

    # reduced, controllable + observable model (drop the decoupled wheel angle)
    A, B, C = continuous.A, continuous.B, continuous.C
    reduced = StateSpaceModel(
        A[np.ix_(KEEP, KEEP)], B[KEEP], C[:, KEEP], np.zeros((C.shape[0], 1)),
    ).discretize(dt)

    # design the LQG: LQR gain K and steady-state Kalman gain L
    K = LQRController.from_model(reduced, Q_RED, R_LQR).K_gain
    L = steady_state_kalman_gain(reduced, W_RED, V_MEAS).L_gain

    # loop gain L(jω) up to just below the Nyquist frequency π/dt. The margins
    # match dynamics.frequency_response.loop_margins (the same sequence the
    # SimulationEngine records per design); the curve is kept for the Bode plot.
    omega = log_frequencies(1e-2, 0.9 * np.pi / dt, 2000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loop = loop_transfer_function(reduced, K, L, omega)
    margins = stability_margins(omega, loop)

    gm = ("infinite" if not np.isfinite(margins.gain_margin_db)
          else f"{margins.gain_margin_db:.2f} dB")
    pm = ("infinite" if not np.isfinite(margins.phase_margin_deg)
          else f"{margins.phase_margin_deg:.2f} deg")
    print(f"Plant '{manager.config.name}' — LQG loop stability margins:")
    print(f"  Gain margin  GM = {gm}"
          + (f"  at ω = {margins.phase_crossover:.2f} rad/s"
             if np.isfinite(margins.phase_crossover) else "")
          + f"   (want ≳ 6 dB)")
    print(f"  Phase margin PM = {pm}"
          + (f"  at ω = {margins.gain_crossover:.2f} rad/s"
             if np.isfinite(margins.gain_crossover) else "")
          + f"   (want ≳ 30–45°)")
    if np.isfinite(margins.phase_margin_deg) and margins.phase_margin_deg < 30.0:
        print("  ⚠ low phase margin — fragile to delay (cf. Doyle 1978 on LQG)")

    from inverted_pendulum.io.plotting import plot_loop_bode, save_figure

    fig = plot_loop_bode(omega, loop, margins=margins,
                         title=f"LQG open-loop gain — {manager.config.name}")
    path = save_figure(fig, Path(args.out) / "loop_bode.png")
    print(f"\nBode diagram written to: {path}")


if __name__ == "__main__":
    main()
