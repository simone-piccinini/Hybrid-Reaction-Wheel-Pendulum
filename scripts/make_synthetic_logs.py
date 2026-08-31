#!/usr/bin/env python3
"""Generate synthetic ``swing_test_v1`` logs from known parameters.

Lets :mod:`identify_parameters` be validated **without hardware**: the traces
come from parameters this script chooses, so the identified values can be
compared against ground truth. Run it before trusting the analysis on real
captures, and again after changing any estimator in that script.

Usage
-----
    python scripts/make_synthetic_logs.py OUTDIR
    python scripts/identify_parameters.py OUTDIR

The identified values should match the TRUTH block this prints, to within a few
percent. The traces carry realistic encoder noise (about one LSB of a 12-bit
AS5600), so exact agreement is not expected and would be suspicious.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

# ---- ground truth (the numbers identify_parameters.py must recover) ----
MASS, LENGTH, G = 0.101, 0.15, 9.81
I_B = 0.0022725          # body inertia about the pivot (kg*m^2)
B_P = 0.004              # pivot viscous friction (N*m*s/rad)
J_W = 6.11725e-5         # wheel inertia (kg*m^2)
K_T = 0.037              # torque constant (N*m/A)
B_W = 1.0e-4             # wheel bearing friction (N*m*s/rad)
RESISTANCE, RAIL, PWM = 10.0, 12.0, 1023.0
NOISE_PIVOT_DEG = 0.09   # ~1 LSB of a 12-bit encoder
NOISE_WHEEL_DPS = 3.0
NOISE_WHEEL_DEG = 0.30   # analog channel: 1-2 bits lost to noise
DT = 0.005               # 200 Hz, matching the firmware's log rate


def _write(out: Path, name: str, kind: str, t, piv, dps, wdeg, whl, duty) -> None:
    with (out / name).open("w") as f:
        f.write(f"# TEST,{kind}\n# DURATION_S,{t[-1]:.0f}\n")
        f.write("t_s,pivot_deg,pivot_dps,wheel_deg,wheel_dps,duty\n")
        for i in range(len(t)):
            f.write(f"{t[i]:.4f},{piv[i]:.3f},{dps[i]:.2f},"
                    f"{wdeg[i]:.3f},{whl[i]:.1f},{int(duty[i])}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("outdir")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    mgl = MASS * G * LENGTH

    # stage 0 - stationary noise
    t = np.arange(0, 10, DT)
    z = np.zeros_like(t)
    _write(out, "00_noise.csv", "noise", t,
           rng.normal(0, NOISE_PIVOT_DEG, t.size), z,
           rng.normal(0, NOISE_WHEEL_DEG, t.size), z, z)

    # stage 1 - free swing about the hanging equilibrium
    t = np.arange(0, 20, DT)
    th, om, piv, dps = math.radians(25.0), 0.0, [], []
    for _ in t:
        piv.append(math.degrees(th))
        dps.append(math.degrees(om))
        om += ((-B_P * om - mgl * math.sin(th)) / I_B) * DT
        th += om * DT
    _write(out, "01_freeswing.csv", "freeswing", t,
           np.array(piv) + rng.normal(0, NOISE_PIVOT_DEG, t.size),
           np.array(dps), np.zeros_like(t), np.zeros_like(t),
           np.zeros_like(t))

    # stage 2 - wheel spin-up then coast
    t = np.arange(0, 8, DT)
    duty = np.where(t < 3.0, 300.0, 0.0)
    b_emf = K_T**2 / RESISTANCE
    w, whl = 0.0, []
    for k in range(t.size):
        whl.append(math.degrees(w))
        v = duty[k] / PWM * RAIL
        w += ((K_T * v / RESISTANCE - (b_emf + B_W) * w) / J_W) * DT
    _write(out, "02_spin.csv", "spin", t, np.zeros_like(t), np.zeros_like(t),
           np.zeros_like(t),
           np.array(whl) + rng.normal(0, NOISE_WHEEL_DPS, t.size), duty)

    # stage 3 - torque pulse on the hanging arm
    t = np.arange(0, 6, DT)
    duty = np.where((t >= 0.5) & (t < 0.7), 150.0, 0.0)
    th, om, piv, dps = 0.0, 0.0, [], []
    for k in range(t.size):
        piv.append(math.degrees(th))
        dps.append(math.degrees(om))
        tau_w = K_T * (duty[k] / PWM * RAIL) / RESISTANCE
        om += ((-tau_w - B_P * om - mgl * math.sin(th)) / I_B) * DT
        th += om * DT
    _write(out, "03_pulse.csv", "pulse", t,
           np.array(piv) + rng.normal(0, NOISE_PIVOT_DEG, t.size),
           np.array(dps), np.zeros_like(t), np.zeros_like(t), duty)

    print(f"wrote 4 captures to {out}\n")
    print("TRUTH — identify_parameters.py should recover these:")
    print(f"  body_inertia          {I_B:.6g}")
    print(f"  pivot_friction        {B_P:.6g}")
    print(f"  torque_constant       {K_T:.6g}")
    print(f"  wheel friction_coeff  {B_W:.6g}")
    print(f"  pivot angle noise     {math.radians(NOISE_PIVOT_DEG):.3g} rad "
          f"({NOISE_PIVOT_DEG} deg)")
    print(f"  wheel angle noise     {math.radians(NOISE_WHEEL_DEG):.3g} rad "
          f"({NOISE_WHEEL_DEG} deg)")


if __name__ == "__main__":
    main()
