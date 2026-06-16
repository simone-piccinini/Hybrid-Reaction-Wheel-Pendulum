# Sizing the pendulum — the engineering reasoning

> "How long can the arm be with a GBM2804H-100T?" — and, more usefully, *how an
> engineer reasons* when choosing hardware for a robot / drone / pendulum.

The answer is never "as long as it looks nice." You start from a **requirement**,
turn it into a **load**, check the **actuator** against that load **with a safety
margin**, and **iterate**. This file does that for the reaction-wheel pendulum
and pulls out the general method.

---

## 1. The method (works for any actuated machine)

| Step | Pendulum | Drone | Robot arm |
|---|---|---|---|
| 1. **Requirement** | recover from a tilt of `θ_c` | hover + manoeuvre | lift a payload at a reach |
| 2. **Worst-case load** | gravity torque at `θ_c` | weight (thrust-to-weight ≥ 2) | payload × reach (+ arm weight) |
| 3. **Actuator limit** | motor reaction torque, wheel saturation, heating | motor thrust, battery C-rate | joint stall torque, gearbox |
| 4. **Margin** | use ~50 % of peak for control authority | T/W ≥ 2 | ×1.5–2 stall margin |
| 5. **Iterate + verify** | shrink arm / mass, re-check | resize props/motor | bigger gearbox | 

The number you solve for (here, the arm length `L`) is whatever makes step 3 ≥
step 2 with the step-4 margin. Then you **verify in simulation** before buying.

---

## 2. The three limits of a reaction-wheel pendulum

The pendulum balances because the motor accelerates the wheel and the wheel
pushes back on the body (action–reaction). Three independent things can run out.

### (a) Torque — can it hold against gravity?

Gravity makes a torque about the pivot at tilt `θ`:

$$
\tau_{grav}(\theta) = M\,g\,\ell\,\sin\theta
$$

with `M` the pendulum mass above the pivot, `ℓ` the pivot→centre-of-mass
distance, `g = 9.81`. The wheel can push back with at most the motor torque
`τ_max`. To **recover from a maximum tilt `θ_c`** you need

$$
\boxed{\;M\,g\,\ell\,\sin\theta_c \;\le\; \tau_{max}\;}
$$

If almost all the mass sits at the tip (motor + wheel assembly at distance `L`),
then `ℓ ≈ L` and `M ≈ m_tip`, so the **gravity torque grows linearly with the
arm length** and

$$
L_{max} \;=\; \frac{\tau_{max}}{m_{tip}\,g\,\sin\theta_c}.
$$

This is almost always the **binding** limit, and the one that sets `L`.

### (b) Momentum — can the wheel absorb the fall before saturating?

Even with enough torque, the wheel speeds up while it pushes and **saturates** at
`ω_max`. Its momentum budget is

$$
H_{max} = I_w\,\omega_{max}.
$$

To catch a fall it must absorb the angular momentum the body builds up,
`≈ I_b\,\dot\theta`. So you also need `I_w\,\omega_{max} \gtrsim I_b\,\dot\theta_{catch}`.
Note: a bigger wheel inertia `I_w` buys **momentum capacity**, *not* more torque
(the reaction torque equals the motor torque regardless of `I_w`). Spin the wheel
faster or make it heavier to widen the recoverable angle.

### (c) Dynamics — how fast does it fall?

For a tip mass, `I_b ≈ M L²`, so the inverted-pendulum natural rate is

$$
\omega_n = \sqrt{\frac{M g \ell}{I_b}} \approx \sqrt{\frac{g}{L}}.
$$

A **longer arm falls slower** (more reaction time — easier on the controller
bandwidth and the 100 Hz loop), but needs **more torque** to hold. Torque
(limit a) usually wins, so longer ⇒ harder overall on this small motor.

---

## 3. What this motor can actually deliver

The datasheet quotes a *rated* load torque, but the real ceiling comes from the
electronics — a classic gotcha:

- **Torque constant** from the load point: `K_t ≈ τ/i = 0.0294 N·m / 0.8 A ≈ 0.037 N·m/A`
  (using 300 g·cm ≈ 0.0294 N·m; 1 g·cm ≈ 9.81×10⁻⁵ N·m).
- **Peak current is limited by the winding, not the driver.** The winding is
  `R = 10 Ω`; on a 12 V bus the stall current is only `i ≈ V/R = 12/10 = 1.2 A`
  — *well below* the SimpleFOCMini's 2.5 A rating. So the driver's headline
  current is irrelevant here; `R` is the limit.
- **Peak (stall) torque** `τ_peak ≈ K_t · V/R = 0.037 × 1.2 ≈ 0.044 N·m` (brief).
- **Continuous torque** is set by heating (`P = i²R`; `≤25 W` total): keep
  `i ≲ 0.8–1 A`, so `τ_cont ≈ 0.03 N·m`.
- **Design torque** — leave half for control authority and disturbance
  rejection (you must do *more* than barely hold): `τ_design ≈ 0.02 N·m`.
- **Wheel speed** `ω_max ≈ 1550 rpm ≈ 162 rad/s` no-load; usable ~120 rad/s.

---

## 4. Worked answer — how long the arm can be

Assume a realistic tip assembly: motor (39 g) + reaction wheel (~50 g) +
bracket/sensor (~20 g) ⇒ **`m_tip ≈ 0.11 kg`**, concentrated at `L`. Then
`L_max = τ / (m_tip · g · sin θ_c)`:

| Catch angle `θ_c` | with `τ_design = 0.02 N·m` | with `τ_peak = 0.04 N·m` (brief) |
|---|---|---|
| 5° | ~21 cm | ~42 cm |
| **10°** | **~11 cm** | **~21 cm** |
| 15° | ~7 cm | ~14 cm |
| 20° | ~5 cm | ~10 cm |

Reading the table the way an engineer does: with a sensible **10–12° catch
angle** and proper torque margin, this motor comfortably balances an arm of
**~12–15 cm**; you can stretch toward **~20 cm** only if you (i) keep the tip
light (~0.10 kg), (ii) accept a small recoverable angle (~10°), and (iii) lean
on brief peak torque. Past ~20–25 cm the GBM2804H-100T runs out of authority.

> **Recommended: arm `L ≈ 15 cm`, tip mass ≤ ~0.12 kg, design catch angle ~12°.**
> This matches the published SimpleFOC pendulum builds and the `pendulum_sensible`
> config (`ℓ_p = 0.15–0.20 m`).

### Check the momentum limit too

Wheel: `m_w = 50 g`, `r = 0.05 m` ⇒ `I_w = ½ m_w r² ≈ 6.3×10⁻⁵ kg·m²`, so
`H_max = I_w ω_max ≈ 6.3×10⁻⁵ × 120 ≈ 7.5×10⁻³ kg·m²/s`. A 15 cm arm falling
through ~12° builds momentum `I_b θ̇ ≈ (m_tip L²)(ω_n θ_c) ≈ (0.11·0.15²)(8·0.21)
≈ 4×10⁻³ kg·m²/s`. Since `H_max > ` that, the wheel **won't saturate** for this
manoeuvre (≈1.8× margin). Good — torque is the binding limit, as expected. To
widen the recoverable angle, raise `H_max`: a heavier-rimmed wheel (more `I_w`)
or a faster spin.

---

## 5. The trade-offs, in one picture

- **Longer arm** → linearly more gravity torque (worse), but slower fall (mildly
  better). Net: torque-limited, so shorter is safer.
- **Heavier tip** → proportionally more gravity torque (worse). Keep the wheel,
  bracket, and sensors light.
- **Bigger / faster wheel** → more momentum capacity (wider catch angle), same
  torque. Use it to recover from bigger kicks, not to hold a longer arm.
- **Higher bus voltage** → more stall current `V/R` → more peak torque. 12 V is
  already near this motor's `2–3S` rating; don't exceed it.
- Want a genuinely long/heavy pendulum? That's a bigger-motor decision
  (GM4108-120T), exactly the step-5 iteration.

---

## 6. Close the loop with the simulator

Numbers on paper are step 4; **step 5 is verification**. Put your chosen
geometry into a config and run the optimiser — if the motor lacks authority the
closed loop will diverge and you'll see it before spending a euro:

```yaml
# configs/pendulum_real.yaml  (fill from your build)
plant:
  pendulum_mass: 0.12        # m_tip you measured
  pendulum_length: 0.15      # the arm L you chose
  body_inertia: 0.0027       # ~ m_tip · L²  (or measure by swing test)
  wheel: {mass: 0.05, radius: 0.05, inertia: 6.3e-5, ...}
  motor: {resistance: 10.0, torque_constant: 0.037, back_emf_constant: 0.037,
          max_voltage: 12.0}
objective: {U_max: 12.0, ...}   # the saturation penalty enforces the V limit
```

Then `scripts/run_experiment.py` tunes the LQG for *that* hardware, and
`scripts/stability_margins.py` tells you how robust it is. If `diverged = True`
or the control saturates `U_max`, the arm is too long / heavy — shorten it and
re-run. That iteration loop, on the cheap, *is* the engineering.

---

## Sources

- Datasheet figures as supplied for the iPower GBM2804H-100T.
- Reaction-wheel pendulum reference build: [GitHub](https://github.com/simplefoc/Arduino-FOC-reaction-wheel-inverted-pendulum), [project docs](https://docs.simplefoc.com/simplefoc_pendulum).
- Plant model and parameters: [`docs/theory/model.md`](../theory/model.md), [`configs/pendulum_sensible.yaml`](../../configs/pendulum_sensible.yaml).
