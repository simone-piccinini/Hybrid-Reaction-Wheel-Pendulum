#!/usr/bin/env python3
"""Turn hardware logs from ``firmware/swing_test_v1`` into config parameters.

Stage 4 of ``docs/hardware/bringup_pipeline.md``: read the CSV traces captured
over serial, identify the plant parameters the simulation currently guesses, and
emit a ready YAML fragment.

The physics behind each estimate is cited inline; nothing here is a black box.
NumPy is used as a calculator only (AGENTS.md §3) — the peak finder and the
exponential fits are written out.

Usage
-----
    python scripts/identify_parameters.py LOGDIR [-o OUT.yaml]
                                          [--mass KG] [--length M]
                                          [--wheel-inertia KGM2] [--rail V]

    LOGDIR   directory of .csv captures (or a single .csv)
    -o       write a YAML fragment here (otherwise print to stdout)

Each capture must keep the firmware's ``# TEST,<kind>`` header line, which is
how the test type is recognised.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

# Defaults describe the as-built v2 machine; override from the command line.
DEF_MASS = 0.101        # m_tip (kg)
DEF_LENGTH = 0.15       # L, pivot -> centre of mass (m)
DEF_JW = 6.11725e-5     # wheel inertia from CAD (kg*m^2)
DEF_RAIL = 12.0         # supply rail (V)
DEF_R = 10.0            # phase resistance (ohm) - VERIFY, see thermal doc §5
PWM_MAX = 1023.0
G = 9.81


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load(path: Path) -> tuple[str, dict[str, np.ndarray]]:
    """Read one capture; return ``(test_kind, columns)``."""
    kind, header, rows = "unknown", None, []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            if line.upper().startswith("# TEST,"):
                kind = line.split(",", 1)[1].strip()
            continue
        parts = line.split(",")
        if header is None:
            if parts[0] == "t_s":
                header = parts
            continue
        if len(parts) != len(header):
            continue                      # partial line from serial, drop it
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            continue
    if header is None or not rows:
        raise ValueError(f"{path.name}: no usable CSV rows")
    arr = np.asarray(rows, dtype=float)
    return kind, {name: arr[:, i] for i, name in enumerate(header)}


# --------------------------------------------------------------------------
# stage 0 - sensor noise
# --------------------------------------------------------------------------
def identify_noise(c: dict[str, np.ndarray], dt: float) -> dict:
    """Stationary noise floor -> the ``measurement_std`` / Kalman ``V`` entries.

    Both channels are measured as ANGLES, because that is what the encoders
    actually report. The simulation's second measurement is a wheel *rate*, and
    a rate obtained by differencing successive angles over ``dt`` has

        sigma_rate = sqrt(2) * sigma_angle / dt,

    which is where the rate noise entry comes from. It is strongly ``dt``
    dependent — at 100 Hz a 0.3 deg angle noise becomes ~0.74 rad/s of rate
    noise — so the sample rate is reported alongside it.
    """
    piv = np.radians(c["pivot_deg"])
    out = {
        "samples": int(piv.size),
        "pivot_std_rad": float(piv.std(ddof=1)),
        "pivot_std_deg": float(np.degrees(piv.std(ddof=1))),
        "pivot_p2p_deg": float(np.ptp(c["pivot_deg"])),
    }
    if "wheel_deg" in c:
        whl = np.radians(c["wheel_deg"])
        out["wheel_std_deg"] = float(np.degrees(whl.std(ddof=1)))
        out["wheel_std_rad"] = float(whl.std(ddof=1))
        out["assumed_dt_s"] = dt
        out["wheel_rate_std_rad_s"] = float(math.sqrt(2.0) * whl.std(ddof=1) / dt)
    else:
        # older captures without the wheel_deg column
        out["wheel_rate_std_rad_s"] = float(np.radians(c["wheel_dps"]).std(ddof=1))
        out["note"] = "no wheel_deg column; rate std taken directly"

    # A drifting mean over a supposedly stationary log means the channel is not
    # merely white. On the analog encoder that usually means slip-ring supply
    # modulation - correlated noise, which a Kalman filter handles worst.
    for key, col in (("pivot", "pivot_deg"), ("wheel", "wheel_deg")):
        if col not in c:
            continue
        x = c[col]
        half = x.size // 2
        drift = float(abs(x[half:].mean() - x[:half].mean()))
        out[f"{key}_mean_drift_deg"] = drift
        if drift > 3.0 * float(x.std(ddof=1)):
            out[f"{key}_WARNING"] = ("mean drifts by more than 3 sigma - this "
                                     "channel is not white noise")
    return out


# --------------------------------------------------------------------------
# stage 1 - free swing  ->  I_b and b_p
# --------------------------------------------------------------------------
def _smooth(x: np.ndarray, n: int) -> np.ndarray:
    """Centred moving average; the cheapest way to stop noise faking peaks."""
    if n < 3:
        return x
    k = np.ones(n) / n
    return np.convolve(x, k, mode="same")


def _period_from_crossings(t: np.ndarray, x: np.ndarray) -> float:
    """Period from mean-crossings detected with HYSTERESIS.

    A plain sign test is useless here: encoder noise makes the trace cross zero
    dozens of times per genuine crossing, which reports a period of a few
    samples instead of the real one. A Schmitt trigger — require the signal to
    reach ±band before accepting the opposite state — rejects that entirely,
    because the noise (a fraction of a degree) is far smaller than the swing.
    """
    amp = float(np.max(np.abs(x)))
    if amp <= 0:
        raise ValueError("flat trace")
    band = 0.25 * amp
    state, times = 0, []
    for i in range(x.size):
        if state <= 0 and x[i] > band:
            if state != 0:
                times.append(t[i])
            state = 1
        elif state >= 0 and x[i] < -band:
            if state != 0:
                times.append(t[i])
            state = -1
    if len(times) < 3:
        raise ValueError("fewer than 3 hysteresis crossings - displace further "
                         "or log for longer")
    return float(2.0 * (times[-1] - times[0]) / (len(times) - 1))


def _peaks(t: np.ndarray, x: np.ndarray, min_sep: float) -> tuple[np.ndarray, np.ndarray]:
    """Local maxima of ``x`` separated by at least ``min_sep`` seconds."""
    idx: list[int] = []
    for i in range(1, x.size - 1):
        if x[i] > x[i - 1] and x[i] >= x[i + 1] and x[i] > 0:
            if idx and (t[i] - t[idx[-1]]) < min_sep:
                if x[i] > x[idx[-1]]:
                    idx[-1] = i           # keep the taller of two close peaks
                continue
            idx.append(i)
    return t[idx], x[idx]


def identify_free_swing(c: dict[str, np.ndarray], mass: float, length: float) -> dict:
    """Damped free oscillation about the HANGING equilibrium.

        I_b θ̈ + b_p θ̇ + m g L θ = 0        (small angle)
        ω_n = sqrt(m g L / I_b),  ζ = b_p / (2 sqrt(m g L I_b))

    Period gives ω_d = ω_n sqrt(1-ζ²); the log decrement gives ζ. Together with
    the weighed ``m`` and measured ``L`` they pin I_b and b_p — an estimate that
    does not depend on CAD at all.
    """
    t = c["t_s"]
    x = c["pivot_deg"] - np.mean(c["pivot_deg"])
    dt = float(np.median(np.diff(t)))

    # 1) robust period from mean-crossings of a lightly smoothed trace
    T0 = _period_from_crossings(t, _smooth(x, max(3, int(0.01 / dt))))
    # 2) real peaks, forced at least half a period apart
    xs = _smooth(x, max(3, int(0.05 * T0 / dt)))
    tp, xp = _peaks(t, xs, min_sep=0.5 * T0)
    keep = xp > 0.05 * np.max(np.abs(xs))         # drop the noise tail
    tp, xp = tp[keep], xp[keep]
    if tp.size < 3:
        raise ValueError("free swing: fewer than 3 clean peaks - "
                         "displace further, or log longer")

    periods = np.diff(tp)
    Td = float(np.median(periods))
    if abs(Td - T0) / T0 > 0.25:
        raise ValueError(f"free swing: peak period {Td:.3f}s disagrees with "
                         f"crossing period {T0:.3f}s - trace too noisy to trust")
    omega_d = 2.0 * math.pi / Td

    # log decrement over the whole usable envelope (least sensitive to noise)
    n = tp.size - 1
    delta = float(np.log(xp[0] / xp[-1]) / n)
    zeta = delta / math.sqrt(4.0 * math.pi**2 + delta**2)
    omega_n = omega_d / math.sqrt(max(1e-9, 1.0 - zeta**2))

    mgl = mass * G * length
    I_b = mgl / omega_n**2
    b_p = 2.0 * zeta * omega_n * I_b
    return {
        "period_s": Td,
        "period_spread_pct": float(100 * np.std(periods) / Td) if periods.size > 1 else 0.0,
        "n_peaks": int(tp.size),
        "omega_n_rad_s": float(omega_n),
        "zeta": float(zeta),
        "body_inertia_kgm2": float(I_b),
        "pivot_friction_Nms": float(b_p),
        "mgl_used": float(mgl),
    }


# --------------------------------------------------------------------------
# stage 2 - wheel  ->  K_t and b_w
# --------------------------------------------------------------------------
def identify_spin(c: dict[str, np.ndarray], J_w: float, rail: float,
                  resistance: float) -> dict:
    """Wheel spin-up step and/or coast-down.

    Spin-up:  J ω̇ = (K_t/R)·V - (K_t K_e/R + b_w)·ω, so ω(t) rises
    exponentially to ω_ss with time constant τ. The INITIAL slope isolates the
    drive term alone (ω≈0, no back-EMF):

        K_t = J · ω̇(0) · R / V_eff

    Coast-down (V=0):  J ω̇ = -b_w ω  ->  b_w = J / τ_coast.
    """
    t = c["t_s"]
    w = np.radians(c["wheel_dps"])
    duty = c["duty"]
    out: dict = {}

    driven = np.abs(duty) > 1
    if driven.any() and np.abs(w[driven]).max() > 0.1:
        d = float(np.median(np.abs(duty[driven])))
        v_eff = d / PWM_MAX * rail
        seg_t, seg_w = t[driven], np.abs(w[driven])
        tail = seg_w[int(0.7 * seg_w.size):]
        w_ss = float(np.mean(tail)) if tail.size else float(seg_w.max())
        # time constant: first crossing of 63.2 % of steady state
        target = 0.632 * w_ss
        above = np.nonzero(seg_w >= target)[0]
        if above.size and w_ss > 1e-6:
            tau = float(seg_t[above[0]] - seg_t[0])
            if tau > 1e-4:
                alpha0 = w_ss / tau                        # initial acceleration
                out["duty_used"] = d
                out["v_eff_V"] = float(v_eff)
                out["omega_ss_rad_s"] = w_ss
                out["tau_s"] = tau
                out["alpha0_rad_s2"] = float(alpha0)
                out["Kt_Nm_per_A"] = float(J_w * alpha0 * resistance / v_eff)
                # total damping from the steady state
                b_tot = (out["Kt_Nm_per_A"] * v_eff / resistance) / w_ss
                out["b_total_Nms"] = float(b_tot)
                b_emf = out["Kt_Nm_per_A"] ** 2 / resistance
                out["b_wheel_Nms"] = float(max(0.0, b_tot - b_emf))

    coast = (~driven) & (np.abs(w) > 0.05 * (np.abs(w).max() + 1e-12))
    if coast.sum() > 20:
        ct, cw = t[coast], np.abs(w[coast])
        good = cw > 0.15 * cw[0]
        if good.sum() > 10:
            # slope of ln(w) vs t -> -1/tau_coast (least squares, by hand)
            x, y = ct[good] - ct[good][0], np.log(cw[good])
            A = np.vstack([x, np.ones_like(x)]).T
            # normal equations: (A^T A) p = A^T y, solved 2x2 in closed form
            ata, aty = A.T @ A, A.T @ y
            det = ata[0, 0] * ata[1, 1] - ata[0, 1] * ata[1, 0]
            if abs(det) > 1e-12:
                slope = (aty[0] * ata[1, 1] - ata[0, 1] * aty[1]) / det
                if slope < -1e-6:
                    tau_c = float(-1.0 / slope)
                    out["tau_coast_s"] = tau_c
                    # Coasting at duty 0 still leaves the windings driven, so the
                    # decay is J/(b_emf + b_w) - NOT b_w alone. Subtract the
                    # electrical term or the config double-counts it, since the
                    # simulator models back-EMF from K_t/K_e separately.
                    b_tot_coast = J_w / tau_c
                    kt_est = out.get("Kt_Nm_per_A")
                    out["b_total_coast_Nms"] = float(b_tot_coast)
                    if kt_est:
                        b_emf = kt_est ** 2 / resistance
                        out["b_wheel_coast_Nms"] = float(max(0.0, b_tot_coast - b_emf))
    if not out:
        raise ValueError("spin log: no usable driven or coasting segment")
    return out


# --------------------------------------------------------------------------
# stage 3 - torque pulse  ->  I_b, independent of CAD
# --------------------------------------------------------------------------
def identify_pulse(c: dict[str, np.ndarray], Kt: float, rail: float,
                   resistance: float, mass: float, length: float,
                   b_p: float | None, period: float | None) -> dict:
    """Angular-momentum balance on the arm during a commanded wheel torque.

    Integrating the body equation over the pulse,

        I_b·Δθ̇ = −τ_w·Δt − m g L ∫sin θ dt − b_p ∫θ̇ dt,

    so gravity and friction are *subtracted out* rather than assumed negligible.
    That correction matters: a 200 ms pulse on a 0.78 s pendulum covers a quarter
    of a period, over which the naive impulse form ``I_b = τ_w Δt / Δθ̇``
    overestimates I_b by more than 2×.

    The sign of Δθ̇ against the commanded duty is the other output, and the more
    important one — it is the reaction-coupling convention.
    """
    t, duty = c["t_s"], c["duty"]
    rate = np.radians(c["pivot_dps"])
    ang = np.radians(c["pivot_deg"])
    on = np.abs(duty) > 1
    if not on.any():
        raise ValueError("pulse log: no commanded duty found")
    i0, i1 = int(np.argmax(on)), int(on.size - 1 - np.argmax(on[::-1]))
    dt = float(t[i1] - t[i0])
    if dt <= 0:
        raise ValueError("pulse log: zero-length pulse")

    d = float(np.median(duty[i0:i1 + 1]))
    v_eff = d / PWM_MAX * rail
    tau_w = Kt * v_eff / resistance
    dv = float(rate[i1] - rate[i0])

    out: dict = {
        "duty": d, "pulse_s": dt, "tau_w_Nm": float(tau_w),
        "delta_pivot_rate_rad_s": dv,
        "sign_consistent": bool(np.sign(dv) == -np.sign(d)) if abs(dv) > 1e-3 else None,
    }
    if period:
        frac = dt / period
        out["pulse_frac_of_period"] = float(frac)
        if frac > 0.10:
            out["warning"] = (f"pulse is {frac*100:.0f}% of the {period:.2f}s period; "
                              "gravity correction applied, but <=10% is better")
    if abs(dv) <= 1e-3:
        return out

    seg = slice(i0, i1 + 1)
    grav = mass * G * length * float(np.trapezoid(np.sin(ang[seg]), t[seg]))
    fric = (b_p or 0.0) * float(np.trapezoid(rate[seg], t[seg]))
    out["gravity_impulse_Nms"] = grav
    out["body_inertia_naive_kgm2"] = float(abs(tau_w * dt / dv))
    out["body_inertia_kgm2"] = float(abs((-tau_w * dt - grav - fric) / dv))
    return out


# --------------------------------------------------------------------------
def emit_yaml(res: dict, mass: float, length: float, J_w: float) -> str:
    """Render the identified numbers as a config fragment."""
    fs, sp, ns = res.get("freeswing"), res.get("spin"), res.get("noise")
    pu = res.get("pulse")
    I_b = (fs or {}).get("body_inertia_kgm2")
    if I_b is None and pu:
        I_b = pu.get("body_inertia_kgm2")
    b_p = (fs or {}).get("pivot_friction_Nms")
    Kt = (sp or {}).get("Kt_Nm_per_A")
    b_w = (sp or {}).get("b_wheel_coast_Nms") or (sp or {}).get("b_wheel_Nms")

    def num(v, d):
        return f"{v:.6g}" if v is not None else f"{d}   # NOT MEASURED - still a guess"

    lines = [
        "# Identified from hardware logs by scripts/identify_parameters.py.",
        "# Paste over the corresponding keys in your experiment config.",
        "plant:",
        f"  pendulum_mass: {mass:.6g}",
        f"  pendulum_length: {length:.6g}",
        f"  body_inertia: {num(I_b, mass * length**2)}",
        f"  pivot_friction: {num(b_p, 0.01)}",
        "  wheel:",
        f"    inertia: {J_w:.6g}",
        f"    friction_coefficient: {num(b_w, 1.0e-4)}",
        "  motor:",
        f"    torque_constant: {num(Kt, 0.037)}",
        f"    back_emf_constant: {num(Kt, 0.037)}",
    ]
    if ns:
        lines += [
            "simulation:",
            "  disturbances:",
            f"    measurement_std: [{ns['pivot_std_rad']:.3g}, "
            f"{ns['wheel_rate_std_rad_s']:.3g}]   # MEASURED [theta_p, theta_w_dot]",
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logdir", help="directory of .csv captures, or one .csv")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--mass", type=float, default=DEF_MASS)
    ap.add_argument("--length", type=float, default=DEF_LENGTH)
    ap.add_argument("--wheel-inertia", type=float, default=DEF_JW)
    ap.add_argument("--rail", type=float, default=DEF_RAIL)
    ap.add_argument("--resistance", type=float, default=DEF_R)
    ap.add_argument("--dt", type=float, default=0.02,
                    help="control period the rate noise is quoted for "
                         "(default 0.02 = the firmware config's 50 Hz)")
    a = ap.parse_args()

    p = Path(a.logdir)
    files = sorted(p.glob("*.csv")) if p.is_dir() else [p]
    if not files:
        raise SystemExit(f"no .csv files in {p}")

    # Load everything first, then analyse in DEPENDENCY order: the pulse needs
    # b_p and the period from the free swing, and K_t from the spin test. Do not
    # rely on filenames to impose that.
    loaded: list[tuple[Path, str, dict]] = []
    for f in files:
        try:
            kind, c = load(f)
            loaded.append((f, kind, c))
        except ValueError as e:
            print(f"  {f.name}: SKIPPED ({e})")
    order = {"noise": 0, "freeswing": 1, "spin": 2, "pulse": 3}
    loaded.sort(key=lambda r: order.get(r[1], 99))

    res: dict = {}
    print(f"Reading {len(files)} capture(s) from {p}\n")
    for f, kind, c in loaded:
        try:
            if kind == "noise":
                res["noise"] = identify_noise(c, a.dt)
            elif kind == "freeswing":
                res["freeswing"] = identify_free_swing(c, a.mass, a.length)
            elif kind == "spin":
                res["spin"] = identify_spin(c, a.wheel_inertia, a.rail, a.resistance)
            elif kind == "pulse":
                fs = res.get("freeswing", {})
                res["pulse"] = identify_pulse(
                    c, res.get("spin", {}).get("Kt_Nm_per_A", 0.037),
                    a.rail, a.resistance, a.mass, a.length,
                    fs.get("pivot_friction_Nms"), fs.get("period_s"),
                )
            else:
                print(f"  {f.name}: kind '{kind}' has no analyser, skipped")
                continue
            print(f"  {f.name}: {kind}, {len(c['t_s'])} samples  OK")
        except ValueError as e:
            print(f"  {f.name}: {kind} FAILED - {e}")

    for name, block in res.items():
        print(f"\n--- {name} ---")
        for k, v in block.items():
            print(f"    {k:<26} {v}")

    # sanity flags worth seeing before the numbers are trusted
    print()
    if "spin" in res and "Kt_Nm_per_A" in res["spin"]:
        kt = res["spin"]["Kt_Nm_per_A"]
        if not 0.030 <= kt <= 0.070:
            print(f"  !! K_t = {kt:.4g} is outside the plausible 0.030-0.070 band.")
            print("     Check J_w and the phase-resistance assumption (thermal doc §5).")
    if "freeswing" in res and "pulse" in res:
        a1 = res["freeswing"].get("body_inertia_kgm2")
        a2 = res["pulse"].get("body_inertia_kgm2")
        if a1 and a2:
            err = abs(a1 - a2) / a1 * 100
            print(f"  I_b cross-check: free swing {a1:.4g} vs pulse {a2:.4g} "
                  f"({err:.0f}% apart)")
            if err > 25:
                print("     >25% apart - do not trust either until resolved.")
    if "pulse" in res and res["pulse"].get("sign_consistent") is False:
        print("  !! Pulse sign is INVERTED relative to the model's convention.")
        print("     Fix before any balancing attempt: it is positive feedback.")

    text = emit_yaml(res, a.mass, a.length, a.wheel_inertia)
    print("\n" + "=" * 60 + "\n" + text)
    if a.out:
        Path(a.out).write_text(text)
        print(f"written to {a.out}")


if __name__ == "__main__":
    main()
