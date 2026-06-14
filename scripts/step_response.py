#!/usr/bin/env python3
"""Time-domain analysis of the closed-loop reaction-wheel pendulum.

Builds the plant from a config, designs an LQR controller, and characterises the
closed loop in the time domain (see docs/guides/time_domain_response.md):

1. closed-loop modal analysis — poles, natural frequencies, damping ratios;
2. the regulation transient — the linear initial-condition response x(t)=e^{A_cl t}x0
   from a small tilt, compared against the true nonlinear rollout;
3. the unit-step response of the closed-loop output;

with the standard transient metrics (rise/peak/overshoot/settling) printed and
the curves saved as figures.

Usage
-----
    PYTHONPATH=src python scripts/step_response.py [CONFIG] [-o OUTDIR] [--tilt RAD]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from inverted_pendulum.control.lqr_controller import LQRController
from inverted_pendulum.dynamics.nonlinear_model import NonlinearPlantModel
from inverted_pendulum.dynamics.time_response import (
    closed_loop,
    free_response,
    modal_analysis,
    step_response,
    transient_metrics,
)
from inverted_pendulum.experiment.manager import ExperimentManager

# A sensible LQR weighting for the demonstration (the optimiser would tune these).
Q_DIAG = np.array([20.0, 2.0, 1e-2, 1e-2])
R_LQR = np.array([[1.0]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default="configs/default.yaml")
    parser.add_argument("-o", "--out", default="results/time_response")
    parser.add_argument("--tilt", type=float, default=0.05,
                        help="initial pendulum tilt for the regulation transient (rad)")
    parser.add_argument("--t-end", type=float, default=5.0)
    args = parser.parse_args()

    manager = ExperimentManager.from_config_file(args.config)
    plant = manager.build_plant()
    linear = manager.build_engine().linear_model
    dt = manager.config.simulation.dt
    controller = LQRController.from_model(linear.discrete, np.diag(Q_DIAG), R_LQR)

    # 1. modal analysis of the continuous closed loop A_cl = A − BK
    cl = closed_loop(linear.continuous, controller.K_gain)
    print("Closed-loop modes (continuous):")
    for mode in modal_analysis(cl):
        kind = ("unstable" if mode.damping_ratio < 0 else
                "overdamped" if mode.damping_ratio >= 1 else "underdamped")
        print(f"  λ = {mode.eigenvalue:+.3f}   ω_n = {mode.natural_frequency:6.3f} rad/s"
              f"   ζ = {mode.damping_ratio:+.3f}  ({kind})")

    # 2. regulation transient: linear (e^{A_cl t} x0) vs nonlinear rollout
    x0 = np.array([args.tilt, 0.0, 0.0, 0.0])
    time, states, _ = free_response(cl, x0, t_end=args.t_end, dt=dt)
    theta_linear = states[:, 0]

    sim = NonlinearPlantModel(plant=plant, dt=dt)
    theta_nonlinear = np.empty_like(theta_linear)
    x = x0.copy()
    for k in range(time.size):
        theta_nonlinear[k] = x[0]
        x = sim.step(x, controller.compute_control(x)[0])

    metrics = transient_metrics(time, theta_linear, final=0.0)
    print(f"\nRegulation transient from θ₀ = {args.tilt} rad (linear model):")
    print(f"  overshoot      = {metrics.percent_overshoot:.1f} %")
    print(f"  peak time      = {metrics.peak_time:.3f} s")
    print(f"  settling time  = {metrics.settling_time:.3f} s")
    print(f"  linear vs nonlinear max |Δθ| = "
          f"{np.max(np.abs(theta_linear - theta_nonlinear)):.2e} rad")

    # 3. closed-loop unit-step response (output 0 = θ_p)
    step_time, step_out = step_response(cl, t_end=args.t_end, dt=dt, input_index=0)

    from inverted_pendulum.io.plotting import plot_time_response, save_figure

    out_dir = Path(args.out)
    fig1 = plot_time_response(
        time, np.stack([theta_linear, theta_nonlinear], axis=1),
        labels=["linear  (eᴬᵗ x₀)", "nonlinear (true plant)"],
        ylabel=r"$\theta_p$ (rad)", reference=0.0,
        title=f"Regulation transient from a {args.tilt} rad tilt",
    )
    save_figure(fig1, out_dir / "regulation_transient.png")
    fig2 = plot_time_response(
        step_time, step_out[:, 0], ylabel="output", reference=None,
        title="Closed-loop unit-step response (θ_p channel)",
    )
    save_figure(fig2, out_dir / "step_response.png")
    print(f"\nFigures written to: {out_dir}")


if __name__ == "__main__":
    main()
