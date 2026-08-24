#!/usr/bin/env python3
"""Swing up the reaction-wheel pendulum from hanging, then balance it with the LQG.

Demonstrates the full maneuver the LQG-tuning stack does *not* cover on its own:
a global energy-shaping swing-up (``control/swingup_controller.py``) that pumps
the pendulum from hanging (``theta_p = pi``) up into the small-angle basin, a
switching supervisor that hands off near upright, and the project's balancing LQG
(recursive Kalman filter + LQR ``u = -K x_hat``) that catches and holds it. The
rollout runs on the *full nonlinear* plant (``dynamics/nonlinear_model.py``), so
it exercises the ``sin theta`` regime the linear model cannot describe.

Two phases, one switch:
    swing-up  : V = k (E - E_up) theta_dot / E, saturated   (energy control)
    [switch]  : |wrap(theta_p)| < angle AND |theta_dot| < rate  -> latch to balance
    balance   : measure -> Kalman -> u = -K x_hat             (the tuned LQG)

Usage
-----
    PYTHONPATH=src python scripts/swing_up.py [CONFIG] [-o OUTDIR] \
        [--gain K] [--switch-angle RAD] [--switch-rate RAD_S] [--sim-time S] \
        [--no-plots]

Notes
-----
- This build's peak torque is well below the gravity torque at the horizontal,
  so the energy law is bang-bang and swings up over several pumps (resonant).
- The wheel spins up while pumping; the catch must happen before it saturates.
  Balance uses a near-zero LQR weight on the (unobservable) wheel angle so the
  filter's drifting wheel-angle estimate does not fight the controller.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
from pathlib import Path

import numpy as np

from inverted_pendulum.control.lqr_controller import LQRController
from inverted_pendulum.control.swingup_controller import EnergySwingUpController
from inverted_pendulum.core.types import LQGConfig
from inverted_pendulum.dynamics.nonlinear_model import NonlinearPlantModel
from inverted_pendulum.estimation.kalman_filter import KalmanFilter
from inverted_pendulum.experiment.manager import ExperimentManager

# A sensible balancing LQG (the optimiser would tune these; fixed here for the
# demo). The wheel-angle weight is deliberately tiny: that state is unobservable
# from the sensors, so its Kalman estimate drifts and must not be regulated hard.
BALANCE_CONFIG = LQGConfig(
    Q_lqr=np.diag([50.0, 5.0, 1.0e-6, 0.05]),
    R_lqr=np.array([[1.0]]),
    W_process=np.diag([1.0e-4, 1.0e-3, 1.0e-6, 1.0e-2]),
    V_measure=np.diag([1.0e-6, 1.0e-4]),
)


def wrap(angle: float) -> float:
    """Wrap an angle to ``(-pi, pi]`` — the pendulum error from upright."""
    return math.atan2(math.sin(angle), math.cos(angle))


def _feasibility(plant) -> None:
    """Print the torque-vs-friction feasibility of swinging this plant up.

    Energy pumping stops at the bottom speed where the peak reaction torque
    balances pivot friction, ``theta_dot_max ~ E*V_max / b_p``. Reaching upright
    needs ``theta_dot_need = sqrt(4 m g l / I_b)`` at the bottom. Swing-up is only
    possible when the former exceeds the latter — i.e. ``b_p`` below a critical
    value ``b_p_crit = E*V_max / theta_dot_need``. A weak motor or a stiff pivot
    (this build's *unmeasured* ``b_p`` guess) puts it out of reach.
    """
    mgl = plant.pendulum_mass * plant.gravity * plant.pendulum_length
    tau = plant.voltage_to_torque_gain * plant.motor.max_voltage  # peak reaction
    theta_dot_need = math.sqrt(4.0 * mgl / plant.body_inertia)
    b_p = plant.pivot_friction
    theta_dot_max = tau / b_p if b_p > 0 else math.inf
    b_crit = tau / theta_dot_need
    ok = theta_dot_max > theta_dot_need
    print(f"  feasibility: peak reaction torque {tau:.3f} N*m; need "
          f"theta_dot~{theta_dot_need:.1f} rad/s at the bottom to reach the top; "
          f"friction caps it at ~{theta_dot_max:.1f} rad/s (b_p={b_p:g}).")
    print(f"    -> swing-up is {'FEASIBLE' if ok else 'INFEASIBLE'} "
          f"(needs b_p < ~{b_crit:.4f}; "
          f"{'ok' if ok else 'try --pivot-friction ' + f'{b_crit / 2:.4f}'}).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default="configs/pendulum_measured.yaml")
    parser.add_argument("-o", "--out", default="results/swing_up")
    parser.add_argument("--gain", type=float, default=8.0,
                        help="energy-pumping gain k (default 8.0)")
    parser.add_argument("--switch-angle", type=float, default=0.20,
                        help="|wrap(theta_p)| (rad) below which to hand off (default 0.20)")
    parser.add_argument("--switch-rate", type=float, default=4.0,
                        help="|theta_p_dot| (rad/s) below which to hand off (default 4.0)")
    parser.add_argument("--sim-time", type=float, default=8.0,
                        help="rollout horizon (s, default 8.0)")
    parser.add_argument("--pivot-friction", type=float, default=None,
                        help="override the plant's b_p (N*m*s/rad) — swing-up "
                             "feasibility is gated by it; see the feasibility line")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    manager = ExperimentManager.from_config_file(args.config)
    engine = manager.build_engine()
    plant = engine.plant
    dt = manager.config.simulation.dt
    discrete = engine.linear_model.discrete
    if args.pivot_friction is not None:
        plant = dataclasses.replace(plant, pivot_friction=args.pivot_friction)
    true_model = NonlinearPlantModel(plant=plant, dt=dt)

    _feasibility(plant)

    swingup = EnergySwingUpController(
        mass=plant.pendulum_mass, length=plant.pendulum_length,
        gravity=plant.gravity, body_inertia=plant.body_inertia,
        voltage_to_torque_gain=plant.voltage_to_torque_gain,
        max_voltage=plant.motor.max_voltage, energy_gain=args.gain,
    )
    controller = LQRController.from_config(discrete, BALANCE_CONFIG)
    v_max = plant.motor.max_voltage

    horizon = int(round(args.sim_time / dt))
    time = dt * np.arange(horizon)
    states = np.zeros((horizon, 4))
    controls = np.zeros(horizon)
    energy = np.zeros(horizon)
    phase = np.zeros(horizon, dtype=int)  # 0 = swing-up, 1 = balance

    x = np.array([math.pi, 0.0, 0.0, 0.0])  # hanging, at rest
    kalman: KalmanFilter | None = None
    just_switched = False
    switch_step: int | None = None
    u_prev = 0.0

    for k in range(horizon):
        states[k] = x
        theta_p, theta_p_dot = x[0], x[1]
        energy[k] = swingup.energy(theta_p, theta_p_dot)

        # --- supervisor: latch to balance once inside the capture window ------
        if kalman is None and abs(wrap(theta_p)) < args.switch_angle \
                and abs(theta_p_dot) < args.switch_rate:
            x0_hat = np.array([wrap(theta_p), theta_p_dot, x[2], x[3]])
            kalman = KalmanFilter(discrete, BALANCE_CONFIG.W_process,
                                  BALANCE_CONFIG.V_measure, x0=x0_hat)
            just_switched = True
            switch_step = k

        # --- control law for this phase ---------------------------------------
        if kalman is None:
            u = swingup.compute_voltage(theta_p, theta_p_dot)
            phase[k] = 0
        else:
            z = np.array([wrap(theta_p), x[3]])  # sensors: [theta_p, theta_w_dot]
            x_hat = kalman.update(z) if just_switched else kalman.step(u_prev, z)
            just_switched = False
            u = float(controller.compute_control(x_hat)[0])
            u = float(np.clip(u, -v_max, v_max))
            phase[k] = 1

        controls[k] = u
        x = true_model.step(x, u)
        u_prev = u

    # ---- report ------------------------------------------------------------
    final_angle = abs(wrap(states[-1, 0]))
    final_rate = abs(states[-1, 1])
    balanced = switch_step is not None and final_angle < 0.10 and final_rate < 0.5
    peak_wheel = float(np.max(np.abs(states[:, 3])))
    print(f"Swing-up on '{manager.config.name}' (k={args.gain}, dt={dt}s):")
    min_error = math.degrees(min(abs(wrap(a)) for a in states[:, 0]))
    if switch_step is None:
        print(f"  did NOT reach the capture window in {args.sim_time:.0f}s "
              f"(closest approach to upright: {min_error:.0f} deg). See the "
              f"feasibility line above — if INFEASIBLE the pivot friction caps "
              f"the pump; otherwise raise --gain or --switch-rate.")
    else:
        print(f"  swing-up took {switch_step * dt:.2f}s "
              f"({switch_step} steps), then handed off to the LQG")
        print(f"  final upright error = {math.degrees(final_angle):.2f} deg, "
              f"rate = {final_rate:.3f} rad/s")
        print(f"  peak wheel speed during pumping = {peak_wheel:.0f} rad/s "
              f"({peak_wheel * 60 / (2 * math.pi):.0f} RPM)")
        print(f"  => {'BALANCED' if balanced else 'caught but not settled'}")

    if not args.no_plots:
        _plot(time, states, controls, phase, energy, swingup.upright_energy,
              switch_step, dt, v_max, Path(args.out), manager.config.name)


def _plot(time, states, controls, phase, energy, e_up, switch_step, dt, v_max,
          out_dir, name) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    t_switch = None if switch_step is None else switch_step * dt

    axes[0].plot(time, np.degrees([wrap(a) for a in states[:, 0]]), color="C0")
    axes[0].axhline(0, color="grey", lw=0.6, ls=":")
    axes[0].set_ylabel("upright error (deg)")
    axes[1].plot(time, states[:, 1], color="C1")
    axes[1].set_ylabel(r"$\dot\theta_p$ (rad/s)")
    axes[2].plot(time, states[:, 3], color="C2")
    axes[2].set_ylabel(r"wheel $\dot\theta_w$ (rad/s)")
    axes[3].plot(time, controls, color="C3")
    axes[3].axhline(v_max, color="grey", lw=0.6, ls=":")
    axes[3].axhline(-v_max, color="grey", lw=0.6, ls=":")
    axes[3].set_ylabel("voltage (V)")
    axes[3].set_xlabel("time (s)")

    for ax in axes:
        if t_switch is not None:
            ax.axvline(t_switch, color="k", lw=1.0, ls="--", alpha=0.7)
        ax.grid(alpha=0.25)
    axes[0].set_title(f"Swing-up + LQG catch — {name} "
                      f"(dashed = hand-off to balance)")
    fig.tight_layout()
    path = out_dir / "swing_up.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"\nFigure written to: {path}")


if __name__ == "__main__":
    main()
