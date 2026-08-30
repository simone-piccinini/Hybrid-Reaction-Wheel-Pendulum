# Thermal limits, the duty ceiling, and why the firmware throttles the wrong thing

The v6 firmware caps commanded duty at `dutyCap = 250` (throttled in practice to
~223) with a `HARD_DUTY_CEILING` of 450, because during bring-up the motor was
"held locked at duty 237 for several seconds and became too hot to touch"
(the v6 firmware handoff, §2 — not yet in this repo).

That budget is **calibrated against a wiring fault, not a thermal limit**, and
the cap it produces is the single largest gap between this project's simulation
and its hardware. This note shows the arithmetic, what the real limit is, and
what the cap costs in control authority.

---

## 1. Duty 237 cannot produce that symptom

PWM sets the *average* phase voltage, `V_eff = (duty/1023)·12`. Locked rotor is
the worst case — no back-EMF, no airflow — so current is `V_eff/R` with
`R = 10 Ω`:

| duty | V_eff | I = V/R | P = I²R | |
|---|---|---|---|---|
| 150 | 1.76 V | 0.176 A | **0.31 W** | the `f 150` test |
| **237** | **2.78 V** | **0.278 A** | **0.77 W** | the "too hot" event |
| 450 | 5.28 V | 0.528 A | 2.79 W | `HARD_DUTY_CEILING` |
| 1023 | 12.00 V | 1.200 A | 14.40 W | full rail |
| **GND/VCC short** | 12.00 V | 1.200 A | **14.40 W** | PWM bypassed — **19× duty 237** |

Heating rate follows from the winding's own thermal mass (~10 g of copper,
`c_p` 385 J/kg·K → ~3.9 J/K), which is what heats first:

| case | P | K/s | rise in 10 s |
|---|---|---|---|
| duty 237, correct wiring | 0.77 W | 0.20 | **2 K** — barely warm |
| duty 450, correct wiring | 2.79 W | 0.72 | 7 K |
| **GND/VCC short** | 14.40 W | 3.74 | **37 K** — too hot to touch |

**Duty 237 with correct wiring dissipates 0.77 W and cannot heat this motor
appreciably in seconds.** A supply fault dissipates 19× that and matches the
observed symptom exactly. The budget was fitted to a phantom.

---

## 2. The real sustained limit

Steady state, lumped convection over the 35 × 15 mm can (`A ≈ 3.6×10⁻³ m²`):

| cooling | `hA` | τ | duty for +40 K | for +60 K |
|---|---|---|---|---|
| still air (`h≈10`) | 36 mW/K | 12.7 min | 322 | 395 |
| wheel spinning (`h≈25`) | 89 mW/K | 5.1 min | 510 | 624 |

Note the **12-minute time constant**. Nothing thermal happens in seconds; the
winding's local rise is the only fast effect, and §1 bounds it.

---

## 3. The firmware throttles the wrong quantity

Heating depends on the **average** duty. Catching a falling pendulum depends on
the **peak**. The firmware caps the peak. Measured on a real balancing rollout
(catch from 5°, full rail available, `configs/pendulum_build_v2.yaml`):

| | value |
|---|---|
| peak duty | 1023 — saturates, but only **1.1 % of the time** |
| peak power | 14.4 W |
| **mean power, whole rollout** | **0.118 W** (equivalent sustained duty **93**) |
| mean power once settled | 0.008 W (duty 24) |
| total energy over 10 s | 1.1 J → **0.3 K** winding rise |

Against the §2 sustained budget of 1.43 W (still air, +40 K), **balancing uses
about 8 % of the thermal envelope while still hitting the rail on peaks.**
Holding upright uses well under 1 %.

The rolling-average budget the firmware already implements
(`DUTY_BUDGET_THROTTLE` / `DUTY_BUDGET_CUT`) is exactly the right mechanism. It
is the instantaneous `dutyCap` that does the damage.

---

## 4. What the cap costs

Largest recoverable initial tilt, nonlinear plant, one fixed balancing LQG:

| peak duty | effective V | catch angle |
|---|---|---|
| 223 (current default, throttled) | 2.62 V | **2.28°** |
| 450 (current hard ceiling) | 5.28 V | 4.62° |
| 850 | 9.97 V | 8.75° |
| **1023 (full rail, peaks only)** | **12.00 V** | **10.54°** |

**~4.6× the catch angle for 0.3 K of extra heating.** At 2.28° the pendulum
cannot be balanced in practice — that is inside sensor noise and stiction. This
is the difference between a system that works and one that does not.

---

## 5. Recommended change

Separate the two limits the firmware currently conflates:

```c
const int HARD_DUTY_CEILING = 1000;   // was 450 — brief peaks, for the catch
int dutyCap = 1000;                   // was 250
```

Keep `DUTY_BUDGET_THROTTLE` / `DUTY_BUDGET_CUT` doing the thermal work, re-tuned
against measured balancing duty (mean ≈ 93 equivalent). Keep **every** other
safety layer — stall detection matters *more* with a higher ceiling, since a
locked rotor at duty 1000 is the one case that genuinely does dissipate 14 W.

### Verify on hardware first

This analysis assumes the winding is 10 Ω **per phase** and the wiring is now
correct. The handoff flags that gimbal specs are ambiguous about phase vs
phase-to-phase resistance — if it is phase-to-phase, actual phase R is 5 Ω,
every current below **doubles** and power **quadruples**.

Locked rotor, bench supply with a current display, `f <duty>`, ≤10 s per step:

| duty | predicted I (10 Ω phase) |
|---|---|
| 150 | 0.18 A |
| 250 | 0.29 A |
| 450 | 0.53 A |
| 850 | 1.00 A |

Stop at the first step where measured current exceeds prediction by >30 %. If
the readings track, the wiring is sound and the headroom is real. If they come
in at ~2×, halve every ceiling in this note.

---

## 6. Scope of this analysis

Honest about what it is: a **lumped thermal model with estimated convection
coefficients**, not a measured one.

- **Solid**: the 19× ratio in §1 is plain `I²R` with no modelling assumptions,
  and §3's finding that balancing is a low-average-duty task is measured from
  the rollout.
- **Less certain**: the absolute sustained ceilings in §2 (duty 322–624) depend
  on `h`, which is estimated. This is precisely why the average-duty guard
  should stay in place rather than being replaced by a fixed higher cap.

---

## See also

- [reaction_wheel_design.md §9](reaction_wheel_design.md) — the companion
  finding that the wheel is torque-limited, not momentum-limited, and that
  adding rim mass is the wrong fix. Both notes converge on the same conclusion:
  **this build is short of actuator authority, not of inertia.**
- [`configs/pendulum_build_v2_firmware.yaml`](../../configs/pendulum_build_v2_firmware.yaml)
  — the plant with the firmware's real limits applied.
