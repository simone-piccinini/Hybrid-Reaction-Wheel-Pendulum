> **Imported verbatim** from the hardware bring-up work, so this repository is
> self-contained. This is the record of the *as-flashed* v6 firmware
> ([`firmware/closed_loop_v6_dualsensor.ino`](../../firmware/closed_loop_v6_dualsensor.ino))
> and the plant it drives.
>
> Two of its conclusions have since been revised by analysis in this repo — read
> them alongside it:
> - **§2 "thermal limits"** — the duty cap is calibrated against a wiring fault,
>   not a thermal limit. See [thermal_and_duty_limits.md](thermal_and_duty_limits.md).
> - **§2 "torque budget for a direct-drive pendulum"** — that table is for a
>   *direct-drive* pendulum and does **not** apply to this reaction-wheel build.
>   See [reaction_wheel_design.md §9](reaction_wheel_design.md).
>
> The bring-up plan that supersedes its §8 is
> [bringup_pipeline.md](bringup_pipeline.md).

# Handoff: Sensored BLDC Firmware (v6) for LQR + Kalman Integration

**Audience.** An agent holding a working LQR + Kalman simulation of an inverted
pendulum, tasked with realising it on this hardware.

**What this document is.** A description of the physical plant, the firmware
that currently drives it (`closed_loop_v6_dualsensor.ino`), the measured
parameters available, and the specific places where the control law should be
substituted. It also records failure modes discovered during bring-up that cost
significant time, because several of them are invisible in simulation and will
recur if the assumptions behind them are not carried forward.

**Status.** Motor, driver, and both encoders are verified working. Closed-loop
commutation is functional. Torque mode and velocity mode both run. The pendulum
mechanics are not yet attached; nothing below has been validated under load.

---

## 1. Hardware inventory

| Item | Part | Interface |
|---|---|---|
| MCU | Teensy 4.1 (i.MX RT1062, 600 MHz) | USB serial, 115200 |
| Driver | 3-phase BLDC driver, 8–30 V input | 3× PWM + enable + fault |
| Motor | iPower GBM2804H-100T gimbal motor | 3 phases |
| Encoder A | AS5600, **analog** output | ADC pin 23 (A9) |
| Encoder B | AS5600, **I2C** | SDA 18, SCL 19 |
| Supply | 12 V DC to driver VCC | — |
| Slip ring | 6-circuit capsule, gold contacts, ≤10 mΩ noise | 3 circuits motor, 3 sensor |

### Pin map

```
Driver          Teensy
  GND      ->   GND
  IN1      ->   2      (PWM, phase A)
  IN2      ->   3      (PWM, phase B)
  IN3      ->   4      (PWM, phase C)
  EN       ->   5      (digital out, active HIGH)
  FAULT    ->   8      (digital in, active LOW)
  RESET    ->   driver's own 3.3V   (active low, held high)
  SLEEP    ->   driver's own 3.3V   (active low, held high)

Encoder A (analog)       Encoder B (I2C)
  VCC  -> 3.3V             VCC  -> 3.3V
  GND  -> GND              GND  -> GND
  DIR  -> GND              DIR  -> GND
  GPO  -> FLOATING         GPO  -> FLOATING
  OUT  -> pin 23           SDA  -> pin 18
                           SCL  -> pin 19
```

**Two constraints that are not optional:**

- **GPO must float.** Grounding it puts the AS5600 into programming mode and
  disables the analog output entirely. Many breakout boards ship with a 1 kΩ
  resistor (often R4) tying PGO to ground; on this hardware that resistor was
  physically removed. If a board is replaced, check for it.
- **Encoder VCC must be 3.3 V, not 5 V.** The AS5600 analog output is
  ratiometric to its own supply, and the Teensy 4.1 is not 5 V tolerant on
  analog inputs.

### Why two different interfaces

Every AS5600 is fixed at I2C address 0x36 and the address is not configurable
on the standard part. Two of them cannot share a bus. Running one on analog and
one on I2C is the workaround. It also matches the mechanical constraint: the
slip ring has three spare circuits, and I2C needs four (VCC, GND, SDA, SCL)
while analog needs three (VCC, GND, OUT).

---

## 2. Motor specification

```
Model                      GBM2804H-100T
Configuration              12N / 14P  ->  7 POLE PAIRS
Outer diameter             35 mm
Height                     15 mm (including encoder seat)
Hollow shaft OD / ID       7 mm / 5 mm
Mass                       39 g
Phase resistance           10 Ω ±5%  (varies with temperature)
No-load current            0.07 ±0.05 A
No-load voltage            10 V
No-load speed              1468–1622 RPM
Loaded current             0.8 A
Loaded voltage             10 V
Loaded torque              250–350 g·cm
Maximum power              ≤25 W
Operating cells            2–3S
Rotor bell runout          ≤0.1 mm
Operating temperature      -20 to +60 °C
Lead wire                  #28 AWG, 300 mm
Nominal rotation direction CW (shaft extension)
```

### Derived electrical constants

Rated torque, upper bound: 350 g·cm = **34.3 mN·m** at 0.8 A.

```
Kt  =  0.0343 N·m / 0.8 A     =  0.0429 N·m/A
Ke  =  Kt (SI)                =  0.0429 V·s/rad
```

Cross-check against the no-load figures gives a different answer:
1545 RPM ≈ 161.8 rad/s at 10 V implies Ke ≈ 0.0618 V·s/rad. The two disagree by
~45%. Manufacturer specs for gimbal motors are inconsistent about whether
voltage is line-to-line or phase, and whether resistance is phase or
phase-to-phase. **Treat both figures as order-of-magnitude only and identify Kt
empirically before relying on it.** Section 8 describes how.

Electrical damping, which sets the only passive damping the plant has:

```
b_emf  =  Kt · Ke / R  =  0.0429² / 10  ≈  1.84e-4  N·m·s/rad
```

This is very small — it is the reason open-loop drive rings badly and why the
derivative term in any closed loop is doing essentially all of the damping.

### Rotor inertia

Not measured. A crude estimate from geometry (bell ≈ 20 g at ≈ 15 mm effective
radius) gives **J ≈ 4.5e-6 kg·m²**, order of magnitude only. This must be
identified on the real hardware; it is a direct input to the LQR and a 2–3×
error here will invalidate the gains.

### Thermal limits — the binding constraint

This is the single most important non-obvious limit on the design.

At 0.8 A through 10 Ω windings, copper loss is **6.4 W**. The 25 W ceiling is a
peak figure, not continuous. A gimbal motor holding position against gravity
draws current at zero speed with **no rotor movement and therefore no airflow**,
which is the classic way these motors are destroyed. During bring-up the motor
was held locked at duty 237 for several seconds and became too hot to touch.

The firmware therefore budgets a rolling average of commanded duty
(Section 6.5). This is not decorative — it should be preserved, and the
thresholds re-tuned once the pendulum load is attached and real duty cycles are
known.

**Consequence for control design:** the actuator has a peak torque limit *and* a
separate, lower, sustained torque limit. An LQR that assumes unbounded control
will saturate; one that assumes the peak is continuously available will overheat
the motor. Both limits belong in the simulation.

### Torque budget for a direct-drive pendulum

Holding a pendulum at horizontal requires `τ = m·g·L` (L to centre of mass):

| Pendulum | Torque required | Feasible? |
|---|---|---|
| 20 g at 5 cm | 9.8 mN·m | yes, comfortable |
| 50 g at 5 cm | 24.5 mN·m | marginal, ~70% of rated |
| 50 g at 10 cm | 49 mN·m | no |
| 100 g at 10 cm | 98 mN·m | no |

If peak commanded torque in simulation exceeds roughly half of 34 mN·m, the
design has no margin for the parameter errors described in Section 9.

---

## 3. Sensors

### AS5600, both units

12-bit absolute over one revolution. **Single-turn only** — it reports position
within a revolution and has no idea which revolution it is on. Multi-turn
tracking is done in software by unwrapping, and is lost on power cycle.

| | Encoder A | Encoder B |
|---|---|---|
| Interface | analog, ratiometric to 3.3 V | I2C @ 400 kHz, addr 0x36 |
| Pin | 23 (A9) | SDA 18 / SCL 19 |
| Role | rotor angle, drives commutation | second axis, read only |
| Resolution | 12-bit ADC, ~1–2 bits lost to noise | full 12-bit |
| Effective resolution | ≈ 0.2–0.35° | 0.088° |
| Fault detection | plausibility check only | missing ACK is immediate |
| Diagnostics | none | MD/ML/MH status, AGC |

### Analog mode characteristics

- Default output mode is **full range**: 0 V to VCC across 360°. A calibrated
  span near 4000 ADC counts is correct and expected.
- The output is **nonlinear and compressed within a few degrees of the 0/360
  wrap point**. For a bounded working range, orient the magnet so the wrap sits
  away from where the mechanism dwells. For continuous rotation it is handled by
  unwrapping and is not a problem.
- Being ratiometric, output scales with the sensor's own VCC. Supply variation —
  including brush contact resistance if the sensor is fed through a slip ring —
  appears directly as angle noise.
- **A disconnected analog wire reads near 0 V, which is indistinguishable from a
  valid 0° reading.** I2C gives a missing ACK; analog gives a confident wrong
  answer. The plausibility check in `readAngleA()` exists for this and should not
  be removed.

### Magnet requirements

Diametrically magnetised, 0.5–3 mm from the chip face, centred on the rotation
axis. An axially magnetised magnet produces a weak compressed output rather than
nothing, which is easy to mistake for a wiring fault. Encoder B can report
magnet status directly (`b` command) — use it rather than inferring.

---

## 4. Firmware architecture (v6)

Single file, no external libraries beyond `Wire`. Runs a fixed-rate control loop
at **1 kHz**.

```
setup()
  ├─ configure PWM: 20 kHz, 10-bit (0..1023), pins 2/3/4
  ├─ configure ADC: 12-bit, hardware averaging 8
  ├─ Wire.begin() @ 400 kHz
  └─ scanSensorB()            reports presence, magnet status, AGC

loop()
  ├─ handleSerial()           command parsing
  ├─ updateSensorB()          every 20 ms, independent of arm state
  └─ if armed:
       ├─ session / FAULT checks
       ├─ rate gate: return unless CONTROL_PERIOD_US elapsed
       ├─ readAngleA()  -> unwrap -> contAngle
       ├─ velocity = EMA(Δangle / dt)
       ├─ mode dispatch: TORQUE (direct) | VELOCITY (FF + PI)
       ├─ duty ramp -> cap -> thermal throttle
       ├─ stall / overspeed / thermal checks
       └─ writePhases(elec + ±90°, |duty|)
```

### 4.1 Commutation — the core of the closed loop

```c
float elec = wrap360(sensorDir * pos * POLE_PAIRS);
writePhases(wrap360(elec + (cmd >= 0 ? 90.0f : -90.0f)), (int)fabsf(cmd));
```

Mechanical angle × 7 gives electrical angle. The stator field is placed 90
electrical degrees ahead of the measured rotor position, which is the maximum
torque point of `τ = τ_max · sin(δ)`. Because the field is referenced to the
*measured* rotor angle rather than to a timer, the load angle stays at 90°
regardless of speed, and step-out cannot occur.

`writePhases()` generates three sinusoids at 120° spacing, offset to unipolar
and scaled by duty:

```c
a = (sin(θ)          + 1) / 2 * duty
b = (sin(θ + 2π/3)   + 1) / 2 * duty
c = (sin(θ + 4π/3)   + 1) / 2 * duty
```

PWM is 20 kHz (above audible) at 10-bit resolution.

### 4.2 `sensorDir` — why it exists

Whether the encoder counts in the same rotational sense as the motor's
electrical rotation depends on how the phases happen to be wired and how the
magnet is oriented. It cannot be known in advance.

If the sign is wrong, the computed field swings *backwards* as the rotor turns.
Instead of leading and pulling, it lags and pushes, and the system settles at an
equilibrium where torque cancels: **the rotor locks in place, buzzes, and draws
full current while going nowhere.** This exact failure consumed a long debugging
session and is easily mistaken for insufficient torque, a weak driver, or a
mechanical jam.

It is now measured automatically during arming, and can be flipped at runtime
with `z`.

### 4.3 Self-calibration (`selfCalibrate()`) — read this carefully

Arming sweeps the field through `POLE_PAIRS` full electrical revolutions, which
is exactly **one mechanical revolution**, at low duty, recording:

- raw ADC minimum and maximum → the analog scale
- net direction of travel → `sensorDir`
- total span → verification that the rotor actually followed

**This replaced hand calibration, and the reason matters.** Manually spinning a
magnet gave spans of 4027, 4004, and then 2759 counts on consecutive attempts
with the same sensor, because a hand sweep does not reliably cover the full
range. A span of 2759 where the truth is 4030 inflates every computed angle by
46%. Since `elec = pos × 7`, the field then advances 46% too fast, outruns the
rotor, loses synchronisation, and the motor locks — presenting as a commutation
fault when the actual defect was a scale error established minutes earlier.

The symptom appeared as "shaft moved 75.8° where 51.4° was expected", which
looked like a wrong pole-pair count. Dividing 75.8 by 1.46 gives 51.9°. The
geometry was correct throughout; only the scale was wrong.

**Any future change to how the analog scale is established must preserve the
property that the sweep is machine-driven and cannot be under-run.**

The sweep also fails loudly if the span is below `MIN_VALID_SPAN` (2500 counts),
which catches a loose magnet, an obstruction, or `SWEEP_DUTY` being too low for
the load — all of which would otherwise silently store a bad scale.

### 4.4 Velocity estimation

```c
velocity = (1 - VEL_ALPHA) * velocity + VEL_ALPHA * (d / dt);   // α = 0.02
```

Time constant ≈ 50 ms at 1 kHz. This is heavy filtering, and it is deliberate.

Differentiating an analog angle amplifies its noise severely. With a
free-running loop (dt ≈ 200 µs) a single noisy ADC sample produced apparent
velocities of ±5000 deg/s while the shaft moved under a degree, which tripped
the overspeed cut immediately. Two mitigations are in place:

- **Fixed 1 kHz rate**, so dt is consistent and large enough that one ADC count
  is a small velocity error.
- **Spike rejection**: any single sample implying more than 8° of movement in
  one millisecond is discarded as noise rather than fed to the derivative.

**Implication for the Kalman filter:** this filtered velocity has ~50 ms of
phase lag. If the state estimator derives velocity itself from the raw angle,
that lag disappears — which is preferable — but the estimator must then handle
the raw noise directly. Do not feed the already-filtered `velocity` into a
Kalman filter that also models process noise; that double-filters and the lag
will eat stability margin.

### 4.5 Velocity mode: feedforward plus PI trim

```c
float ff   = FF_GAIN * speedTarget;        // FF_GAIN = 0.15
float verr = speedTarget - vSigned;
viTerm    += VKI * verr * dt;              // clamped to ±VI_LIMIT
dutyRequest = ff + VKP * verr + viTerm;    // VKP 0.12, VKI 0.06
```

`FF_GAIN` comes from measurement: duty 150 produced ≈1000 deg/s steady-state
no-load, so duty ≈ 0.15 × deg/s.

Before feedforward existed, the PI loop had to generate the entire command. At
90 deg/s that meant operating at duty 15–60, where cogging torque is a large
fraction of the command, and the loop surged between 57 and 132 deg/s chasing
its own ripple. The integral gain was also far too high (0.40) to sit behind a
50 ms filter lag.

**This section is the one an LQR will replace.** See Section 7.

---

## 5. Measured plant data

Everything below was measured on the real hardware, unloaded.

| Quantity | Value | Notes |
|---|---|---|
| Duty → no-load speed | duty 150 → ≈1000 deg/s | steady state, unloaded |
| Implied gain | ≈ 0.15 duty per deg/s | basis of `FF_GAIN` |
| Analog span, full turn | 4060–4065 counts | consistent across runs |
| Analog resolution | ≈ 0.089 °/count nominal | 1–2 bits lost to noise |
| Minimum duty to break away | ≈ 100–150 unloaded | cogging-dependent |
| Duty at which locked motor overheated | 237 held for seconds | do not repeat |
| Control loop rate | 1 kHz | fixed |
| PWM frequency | 20 kHz | above audible |
| PWM resolution | 10-bit, 0–1023 | |
| Hard duty ceiling | 450 (≈44%) | software limit |

**Not yet measured, and needed for LQR design:**

- Rotor inertia J
- Kt, empirically (the spec figures disagree by 45%)
- Friction: Coulomb and viscous terms
- Thermal time constant under realistic duty cycles
- Analog sensor noise floor in counts, RMS

---

## 6. Safety layers

Preserve these. Each corresponds to a failure actually encountered.

| Layer | Trigger | Catches |
|---|---|---|
| Disarmed at boot | requires `r` | uncommanded motion at power-up |
| Sensor verified before arm | refuses if A not calibrated | driving blind |
| Sweep span check | < 2500 counts | loose magnet, jam, insufficient duty |
| Spike rejection | Δ > 8°/ms | ADC noise entering the derivative |
| Overspeed | filtered vel > 1500 deg/s | runaway |
| Stall detect | duty > 100 and vel < 15 deg/s for 1.5 s | jam, inverted `sensorDir` |
| Thermal budget | rolling mean duty > 340, throttle at 200 | winding damage |
| Sensor loss | implausible reading | disconnected analog wire |
| Driver FAULT | pin 8 LOW | over-current, over-temperature |
| Session limit | 120 s | unattended operation |
| Unknown command | any unparsed input | disarms rather than ignores |

The thermal budget throttles before it cuts: above a rolling mean duty of 200
the cap is squeezed linearly, and above 340 the run aborts. Note that this makes
the duty cap self-limiting — commanding 250 settles near 223 effective,
commanding 300 settles near 242. Raising the cap alone buys much less torque
than it appears to.

---

## 7. Integration guidance for LQR + Kalman

### 7.1 The interface to use

**Use TORQUE mode.** It takes a signed duty command directly, applies ramping and
all safety limits, and handles commutation from the measured rotor angle. That is
exactly the interface an LQR wants: it emits a torque, and something else worries
about how to produce it.

Replace the velocity PI block (Section 4.5) with:

```c
// LQR: u = -K · x
float u = -(K1*x1 + K2*x2 + K3*x3 + K4*x4);   // in N·m
dutyRequest = u / TORQUE_PER_DUTY;             // convert to duty units
```

`TORQUE_PER_DUTY` must be identified empirically (Section 8). Do **not** derive
it from the datasheet Kt, given the 45% internal disagreement in the spec.

Keep: commutation, `sensorDir`, self-calibration, all safety layers, the fixed
1 kHz rate.

Remove or bypass: `FF_GAIN`, `VKP`, `VKI`, `viTerm`, the duty ramp if the LQR
already limits its own rate of change.

### 7.2 State vector and sensor mapping

For a reaction-wheel pendulum the usual state is:

```
x = [ θ_pendulum, θ̇_pendulum, θ̇_wheel ]
```

- **θ_pendulum** — encoder B (I2C), `sensorB_cont`, already unwrapped, degrees
- **θ̇_pendulum** — derive in the estimator, not from firmware
- **θ̇_wheel** — encoder A, `velocity × sensorDir`, degrees/second

Wheel *angle* is a cyclic coordinate and does not appear in the dynamics; wheel
*rate* does, because it carries stored momentum and determines saturation
proximity.

For a directly driven pendulum (motor shaft is the pivot), encoder A serves both
commutation and pendulum angle, and encoder B is available for a second joint.

**Units.** The firmware works in **degrees and degrees/second** throughout.
Convert at the boundary; do not mix conventions inside the loop.

### 7.3 Sampling and latency budget

| Stage | Cost |
|---|---|
| Control period | 1000 µs |
| ADC read, 8× hardware averaging | ≈ 30 µs |
| I2C read of encoder B @ 400 kHz | ≈ 500 µs |
| Sinusoid + 3 PWM writes | < 20 µs |

Encoder B is read every 20 ms, not every cycle, precisely because a 500 µs I2C
transaction is half the control period. **If the Kalman filter needs the pendulum
angle at 1 kHz, this must be restructured** — either non-blocking I2C, a faster
bus, or accepting 50 Hz updates on that channel with the estimator predicting
between them. This is the most likely place where a simulation assuming
instantaneous full-state feedback will diverge from reality.

Direct-drive inverted pendulums typically want 500 Hz–1 kHz. The Teensy 4.1 has
ample compute; I2C is the bottleneck.

### 7.4 Measurement noise for the filter

The simulation's noise covariances were tuned against simulated noise. Real
characteristics differ:

- **Encoder A (analog)** — effective resolution 0.2–0.35°, and the noise is
  **correlated with rotation** if the supply passes through a slip ring, because
  brush contact resistance modulates a ratiometric output. Correlated noise
  violates the core Kalman assumption and is handled worst by the filter.
  Characterise it: hold the shaft fixed, rotate the slip ring, record variance.
- **Encoder B (I2C)** — clean 12-bit, 0.088°/count, quantisation-dominated.

Measure both on the real hardware and set R accordingly rather than carrying the
simulated values across.

### 7.5 What the firmware does not model

- **Friction and stiction.** Not modelled anywhere. Dominant near equilibrium,
  which is exactly where a balancer operates. Expect limit cycles: the controller
  commands a torque too small to break away, error grows until it does, then
  overshoots. No amount of gain tuning removes this, because the plant is
  nonlinear in a way the linear model is not.
- **Cogging torque.** 12 slots, 14 poles — cogging has a strong component at 42
  cycles per mechanical revolution. Significant at low duty.
- **Gravity torque varying with angle.** A fixed-gain LQR linearised about
  upright degrades as the pendulum moves away. A feedforward term
  `+ Kg·sin(θ)` addresses the steady-state part cheaply and is worth adding
  before reaching for gain scheduling.
- **Thermal derating.** Available torque falls as the windings heat. The budget
  in Section 6 handles safety, not performance prediction.

---

## 8. Recommended parameter identification before closing the loop

Do these on the real hardware and re-run the simulation with the measured values
**before** transferring gains. This is the highest-value step available, and it
is cheap.

**Torque constant.** In TORQUE mode with the rotor free, command a known duty
step and record angular acceleration from encoder A. `τ = J·α` gives Kt once J is
known; running several duty levels gives the linearity too. Combined with a
separate J measurement, this yields `TORQUE_PER_DUTY` directly.

**Inertia.** Attach a known additional inertia (a machined disc of computed J),
repeat the acceleration test, and solve the two equations. This removes the
dependence on the geometric estimate.

**Friction.** Ramp duty slowly upward until motion begins — that threshold is the
Coulomb friction torque. Then measure steady-state speed at several duty levels;
the slope gives the viscous coefficient, the intercept confirms Coulomb.

**Pendulum mass and length.** Let it swing freely as a pendulum with the motor
disabled and measure the natural period. `T = 2π√(L/g)` for a simple pendulum
gives effective L; combined with weighed mass this gives `m·g·L`, the torque
needed to hold horizontal — which determines feasibility per Section 2.

**Sensor noise.** Log both encoders stationary for 60 s at full rate. Compute
variance. These are the R entries.

**Loop timing.** Log actual dt over 10 s and check for jitter, particularly if
I2C reads are moved into the control path.

Then re-run the simulation with measured J, Kt, friction, measured delay, and
measured noise. If the gains no longer stabilise, that is a twenty-minute
discovery rather than a week of hardware debugging.

---

## 9. Failure modes encountered during bring-up

Recorded because several are invisible in simulation and expensive to
rediscover.

**GPO tied to ground disables the analog output.** Board-level defect present on
common AS5600 breakouts (1 kΩ resistor, often R4). Symptom: OUT reads as a
floating pin. Survived three sensor replacements before being identified.

**Hand calibration silently corrupts the angle scale.** See Section 4.3. Presents
as a commutation fault or a wrong pole-pair count.

**Inverted `sensorDir` locks the rotor.** See Section 4.2. Presents as
insufficient torque.

**Velocity noise trips overspeed instantly.** With a free-running loop, one noisy
ADC sample implies thousands of deg/s. Fixed by the rate gate and spike
rejection.

**Position-tracking cannot spin a motor.** An earlier version advanced a position
target at the desired speed and let a PD chase it. The target ran away, the rotor
never accelerated, duty stayed at 36 because that is all `KP × error` asked for,
and it tripped the tracking limit. Continuous rotation requires commanding
torque, not position.

**Open-loop drive rings and cannot be damped.** A fixed electrical angle is a
magnetic spring of stiffness `k = τ_max·p`. With load inertia and
`b_emf ≈ 1.8e-4`, damping ratio ζ ≈ 0.1. Raising duty raises k, which *lowers* ζ.
More power makes it worse. Only closed-loop velocity feedback provides damping.

**Twisted wire joints work for motor phases and fail for sensor signals.** Phase
current (0.8 A) punches through contact oxide; a high-impedance analog signal
does not. A joint adequate for the motor can be completely dead for OUT. All
sensor connections should be soldered or properly crimped, never twisted.

**Hot-plugging damages components.** Connecting or moving wiring on a live
circuit destroyed a sensor and stressed the MCU. Power down for every wiring
change.

**Analog sensor faults are silent.** A disconnected wire reads as a valid 0°.
Only the plausibility check distinguishes them.

---

## 10. Command reference (v6)

| Command | Effect |
|---|---|
| `r` | Arm. Runs self-calibration sweep, ~8 s. Shaft must be free to turn. |
| `f <duty>` | Torque mode, signed, −450..450. `f 150` spins at ≈1000 deg/s unloaded. |
| `v <dps>` | Velocity mode with feedforward. `v 90`. |
| `h` | Coast to a stop, zero torque. |
| `z` | Flip `sensorDir` at runtime. |
| `c <duty>` | Duty cap, 0–450. |
| `p <val>` | Velocity loop KP. |
| `i <val>` | Velocity loop KI. |
| `g <val>` | Feedforward gain. |
| `b` | Rescan encoder B, report magnet status and AGC. |
| `s` | Status. |
| `x` | Disarm. |

Telemetry format while armed:

```
  TORQ  A 1002 dps  duty 150/250   B 187.3 deg  avg 150
  VEL   A 88/90 dps  duty 21/250   B 187.3 deg  avg 45
         │  │         │             │            └─ rolling mean duty (thermal)
         │  │         │             └─ encoder B angle
         │  │         └─ commanded / effective cap
         │  └─ target (velocity mode only)
         └─ encoder A speed, direction-corrected
```

---

## 11. Open items

- Pendulum mechanics not attached; nothing validated under load.
- J, Kt, and friction not identified.
- Encoder B not yet assigned to a mechanical axis.
- Slip ring carries motor phases only at present; sensor routing through it is
  designed but not commissioned. Multi-turn cable protection exists as a separate
  module (`turn_tracker.ino`) with EEPROM persistence and a dirty-shutdown flag.
- Analog sensor noise through the slip ring not characterised.
- Thermal budget thresholds tuned for an unloaded motor and will need revisiting.
