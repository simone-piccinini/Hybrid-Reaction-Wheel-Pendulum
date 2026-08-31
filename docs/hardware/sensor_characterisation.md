# Sensor characterisation — measured on the as-built machine

The lab record for Stage 0 of [bringup_pipeline.md](bringup_pipeline.md). Every
number here was measured on the real hardware with
[`firmware/swing_test_v1/`](../../firmware/swing_test_v1/) and
[`firmware/y_wobble_v1/`](../../firmware/y_wobble_v1/).

Encoder roles: **A** = analog, on the wheel/rotor. **B** = I²C, on the pivot,
i.e. the pendulum angle `θ_p`.

---

## 1. Noise floor (Stage 0) — both channels pass

Stationary, 10 s at 200 Hz, motor never enabled. Two independent runs:

| channel | run 1 | run 2 | limit | verdict |
|---|---|---|---|---|
| pivot (B) | 0.044° | 0.052° | < 0.15° | ✅ |
| wheel (A) | 0.178° (2.03 LSB) | 0.189° (2.14 LSB) | < 0.5° | ✅ |

The pivot is exceptional: across an entire 10 s log it took only **two distinct
values, one LSB apart** (0.000 / 0.088). The noise is sub-LSB. The wheel loses
about one bit to noise, exactly as the v6 handoff predicted for the analog
channel. Neither channel drifted (0.02σ over the log).

**These two runs are the trap this document exists to record: the noise floor
passed twice while the pivot encoder was reporting 100° for a true 180°.**
Measuring noise says nothing about whether the reported angle is the real one.

### The derived-rate problem

`pivot_dps` in run 1 swung ±12 °/s on a **stationary** arm. That is not motion:
one LSB step at the 1 kHz sample rate, through the firmware's 0.15 EMA, is

```
0.0879 deg / 0.001 s * 0.15  =  13.2 deg/s
```

which matches the observed range. Differencing a quantised angle at 1 kHz
produces pure artefact. Since the pendulum's own rate during balancing peaks
around 17 °/s, **a naively differenced pivot rate would sit at or below its own
noise floor for most of the task.** `θ̇_p` must be reconstructed by the
estimator from the model and the angle history, never by differencing
consecutive samples (cf. [handoff §4.4](firmware_v6_handoff.md)).

---

## 2. Linearity (Stage 0b) — the fault the noise test missed

Slow manual rotation, stopping at known mechanical angles:

| mechanical | measured | error | local gain |
|---|---|---|---|
| 0° | 0.0 | — | — |
| 90° | 74.2 | −15.8 | 0.82 |
| **180°** | **100.5** | **−79.5** | **0.29** |
| 270° | 235.0 | −35.0 | 1.49 |
| 360° | 366.0 | +6.0 | 1.46 |

A full turn totals **366°** — only 1.7% off — so the sensor, wiring, I²C and
unwrapping are all correct. But the local gain varies **5.1× around the
circle**. Right in total, wrong everywhere in between: the signature of an
**off-axis magnet**. The AS5600 reads field *direction*; a magnet displaced from
the rotation axis orbits the chip instead of spinning in place, so the
distortion cancels over a full revolution but not locally.

**Why it blocks everything downstream:** the tuned LQG's measured gain margin is
6.85 dB (×2.20), so the loop tolerates sensor gain in **[0.45, 2.20]**. A local
gain of **0.29 is outside that**. In that region the controller sees under a
third of the tilt that exists and pushes back with under a third of the torque —
not a tuning problem, and no `(Q, R, W, V)` fixes it.

Concentricity tolerance is ~**0.25 mm**, far tighter than the 0.5–3 mm gap spec.

---

## 3. Out-of-plane wobble (Stage 0c) — the binding limitation

The arm can tilt slightly out of its swing plane, and the mount cannot be fully
rebuilt. The question that decides whether that is correctable: does the reading
depend on the swing angle **alone** (invertible by a calibration map) or on the
swing angle **and** the tilt independently (not invertible from one sensor)?

Method: pin the swing angle against a stop, then push only out of plane.
`q <deg>` = quiet baseline, `g <deg>` = wobble.

### Before the magnet was re-centred

| capture | spread | 1σ |
|---|---|---|
| quiet @ 0° | 0.088° | 0.044° |
| **wobble @ 0°** | **39.463°** | **10.788°** |

y-tilt contribution **39.46°** — 3.9× the 10° threshold. For context the
measured stability basin of this plant is **10.54°**, so **one sigma of
measurement error equalled the entire basin (102%)**. The sensor could not
distinguish "upright and fine" from "already past saving".

### After re-centring the magnet

| capture | spread | 1σ |
|---|---|---|
| quiet @ 0° | 0.176° | 0.048° |
| **wobble @ 0°** | **9.141°** | **2.631°** |

**y-tilt contribution 9.14° — a 4.3× improvement**, and now inside the marginal
band (3–10°). 1σ is **25% of the stability basin**, down from 102%.

This confirms the mechanism: the error is the *product* of eccentricity and
tilt. A centred magnet loses field *amplitude* under tilt but preserves field
*direction* to second order; an off-centre one has a first-order sensitivity
amplified by the offset. Centring collapsed most of the error without changing
the y-play at all.

> **Only 0° has been measured.** Sensitivity varies around the circle, and the
> position that matters for balancing is near upright. Repeat `q 180` / `g 180`
> before trusting any balancing design.

---

## 4. What this means for the configuration

| parameter | previous guess | **measured** | ratio |
|---|---|---|---|
| `measurement_std[0]` (θ_p) | 1.0e-3 rad | **0.04592 rad** | **46×** |
| `measurement_std[1]` (θ̇_w) | 1.0e-2 rad/s | **0.4665 rad/s** (at dt = 0.01) | **47×** |

The wheel-rate entry is derived, not measured directly: both encoders report
*angles*, and a rate obtained by differencing over `dt` has
`σ_rate = √2·σ_angle/dt`. It is therefore strongly `dt`-dependent and must be
re-derived whenever the sample rate changes.

**The θ_p figure is not really sensor noise, and treating it as such is
optimistic.** The pivot encoder reads to one LSB when still; the 2.63° is
*mechanical* — the arm's out-of-plane wobble moving the reading at a fixed true
angle. It is correlated with arm motion, not white, which is the assumption a
Kalman filter is least robust to. Carrying it as measurement noise is the honest
way to tell the filter not to trust that channel, but it flatters the real
situation.

### The search space could not express any of this

Both `V` bounds excluded the measured values:

| parameter | needed `ln(var)` | original box | |
|---|---|---|---|
| `V[0]` pivot | −6.16 | [−15.0, −8.0] | outside |
| `V[1]` wheel rate | −1.52 | [−11.0, −6.0] | outside |

The bounds had been set around the *guessed* noise, so the optimiser was
physically unable to represent how bad the real sensors are and would have been
forced to over-trust them at any evaluation budget. Widened in
[`configs/pendulum_build_v2_measured.yaml`](../../configs/pendulum_build_v2_measured.yaml).

**This generalises:** whenever a measured parameter lands outside its search
box, the tuner fails in a way that looks like a control problem and is not.
Check box membership every time a parameter is measured.

---

## 5. Re-tuned against the measured noise — the verdict

With the measured `measurement_std` and the widened `V` box, the Entropy-Search
tuner **does** find stabilising controllers (best observed 883.6, well inside the
5000 divergence penalty). But the cost is severe:

| | guessed noise | **measured noise** |
|---|---|---|
| max recoverable tilt | 10.54° | **6.12°** |
| sensor 1σ as a share of that basin | 2% | **43%** |
| phase margin | +15.9° | **−2.6°** |
| gain margin | 6.30 dB | ∞ |

Three things to read here.

1. **The basin shrank by 42%** (10.54° → 6.12°), because the filter must now
   distrust the angle channel and therefore responds more slowly.
2. **Sensor 1σ is 43% of the remaining basin.** As a rule of thumb you want that
   well under ~25%: the estimator should be resolving the state far more finely
   than the distance to falling over. At 43% the controller spends much of its
   authority reacting to measurement error.
3. **Phase margin is negative.** The loop is still closed-loop stable in
   simulation (hence a finite catch angle and an infinite gain margin — it is
   conditionally stable), but it has **no delay tolerance whatsoever**. The real
   hardware carries 11–21 ms of latency stack. This controller would not survive
   contact with it.

So: 9.14° of wobble is a genuine improvement over 39.5°, and it moves the
problem from *impossible* to *merely unsurvivable*. It is not yet enough.

**Target: get the wobble spread under ~3°** — roughly another 3× — which is the
"second order" band where a calibration map works and the noise stops dominating
the design. Either further centring, a second bearing to constrain the y-play,
or an IMU on the arm (which also fixes the `θ̇_p` differencing problem in §1).

---

## 6. Status and next steps

- ✅ Noise floor — both channels pass with margin
- ⚠️ Linearity — improved but **only 0° re-verified**; repeat the 4-point check
- ⚠️ Wobble — 39.5° → 9.14°, marginal; measure at **180° (upright)** next
- ⛔ Balancing — not until the upright-position wobble is known

**Independent of all of the above**, Stage 2 (wheel spin-up / coast-down) uses
only encoder A on the motor rotor and requires the arm clamped anyway. It yields
`K_t`, wheel friction, and the phase-vs-phase-to-phase resistance answer, and
can be run while the pivot mount is being improved.
