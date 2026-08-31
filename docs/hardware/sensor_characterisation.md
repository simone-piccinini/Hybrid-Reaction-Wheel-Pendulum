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

### At 180° (upright) — the position balancing actually lives in

| capture | spread | 1σ | detrended jitter |
|---|---|---|---|
| quiet @ 180° | 2.988° | 0.519° | 0.405° |
| wobble @ 180° | 3.340° | 0.534° | 0.497° |

y-tilt contribution **1.49°** — inside the "second order" band. On the face of
it, far better than the 9.14° at hanging.

**Two caveats keep this from being a clean pass.**

1. **The quiet baseline is contaminated.** At 0° the quiet spread was 0.176°
   (1–2 LSB). Here it is 2.988°, seventeen times worse, and the quarter-means
   climb 2.359 → 2.969 → 3.151 → 3.169 with a +0.106 °/s linear trend. That is
   an arm still settling against its stop, not sensor noise. Removing the trend
   leaves 0.405° of jitter. Since the quadrature subtraction assumes the two
   captures share a baseline, a settling transient in the *quiet* run inflates
   what gets subtracted and flatters the result.

2. **A small wobble response may mean an insensitive sensor, not a good one.**
   §2 measured the 90–180° segment at local gain **0.29**. Where the sensor
   under-responds to angle it also under-responds to tilt — so a small apparent
   wobble at upright is exactly what a *compressed* region would produce. That
   would be bad news wearing good news' clothes: gain 0.29 at upright is outside
   the loop's [0.45, 2.20] tolerance.

**What settles it:** re-run the §2 four-point linearity check now that the
magnet has been re-centred. If the local gain near upright is close to 1.0, the
1.49° is real and upright is in good shape. If it is still ~0.3, the wobble
number is an artefact of compression and the mount needs more work regardless.
That check costs two minutes and gates the balancing attempt.

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

## 6. Stage 1 — free swing, measured

Motor unpowered, arm displaced ~25° from hanging and released, 20 s at 200 Hz.
Raw capture: [`data/hw/s1_freeswing.log`](../../data/hw/s1_freeswing.log).

| parameter | previous | **measured** | |
|---|---|---|---|
| `body_inertia` `I_b` | 0.0022725 (CAD) | **0.00224935** | **1.0% apart — CAD validated** |
| `pivot_friction` `b_p` | 0.01 (**guess**) | **0.000268852** | **37× smaller than guessed** |

Derived: `ω_n = 8.129 rad/s`, period **0.773 s** against 0.777 s predicted from
CAD (0.5% apart), damping ratio `ζ = 0.0074` — a very lightly damped pendulum.

Two things follow.

**The CAD model is trustworthy.** `I_b` from the swing period agrees with the
solid model to 1%, and the period matches the prediction to 0.5%. That is a real
validation of the mass and length figures, obtained from timing alone and so
immune to every angle-distortion problem in §2 and §3.

**Swing-up flips from impossible to comfortable.** The feasibility condition in
[swingup.md](../theory/swingup.md) needs `b_p < ~0.0027`. At the guessed 0.01 the
pivot friction bled energy faster than the wheel could pump it in, and the
maneuver could not work at any gain. At the measured 0.000269 there is a **10×
margin**: the analysis now reports the friction cap at 165 rad/s against the
16.3 rad/s actually needed at the bottom. Simulated on the identified plant, the
arm swings up in **2.34 s** and the LQG catches it.

### Re-tuned on the fully identified plant

| | guessed `I_b`/`b_p` | **measured `I_b`/`b_p`** |
|---|---|---|
| best observed cost | 883.6 | 1539.9 |
| max recoverable tilt | 6.12° | **10.69°** |
| phase margin | −2.6° | **−21.2°** |
| peak commanded `u` | 12.9 V | **17.4 V** (past the 12 V rail) |

A mixed result, and worth stating plainly: the real plant is **harder to
balance** than the guessed one. Removing 37× of assumed pivot friction removes
37× of free damping, so the controller has to supply all of it — the same point
the v6 handoff makes about `b_emf` being tiny and the loop's derivative action
doing essentially all the damping.

The larger catch angle is not a straight win either: it is reached with the
command saturating well past the rail, and the two rows are different
controllers on different plants, not one controller compared fairly.

**The phase margin has now come out negative on three separate tunings**
(+2.3°, −2.6°, −21.2°) whenever the measured pivot noise is included. Each
rollout is closed-loop stable in simulation — these are conditionally stable
loops, hence the infinite gain margins — but a loop with no phase margin at its
gain crossover has *no delay tolerance*, and the real firmware carries 11–21 ms.
The exact value is not to be trusted at this noise level; the pattern is.

**The binding constraint is the pivot sensor, not the plant parameters.** No
amount of re-tuning recovers phase margin while `measurement_std[0]` is
0.046 rad.

> `b_p` here includes a little drag from the wheel turning in its own bearing,
> since the wheel was left free. That biases the figure *high*, so the true pivot
> friction is at most this — which only strengthens the swing-up conclusion.

This is the clearest payoff of the pipeline so far: a parameter that was a pure
guess, wrong by 37×, and it was silently deciding whether an entire capability
was possible.

---

## 7. Status and next steps

- ✅ Noise floor — both channels pass with margin
- ⚠️ Linearity — **not re-measured since the magnet was re-centred.** This is
  now the single outstanding Stage 0 item, and it gates balancing.
- 🟡 Wobble — measured at both ends: **9.14° at hanging** (marginal),
  **1.49° at upright** (looks good, but see the two caveats in §3)
- ⛔ Balancing — gated on the linearity re-check
- ✅ **Stage 1 done** (§6): `I_b` measured and agreeing with CAD to 1%, `b_p`
  measured at 37× below the guess, and swing-up consequently feasible.
- ⬜ Stage 2 (wheel: `K_t`, wheel friction, the 10 Ω question) — needs the arm
  clamped, uses only encoder A, unaffected by the pivot faults above.

**Independent of all of the above**, Stage 2 (wheel spin-up / coast-down) uses
only encoder A on the motor rotor and requires the arm clamped anyway. It yields
`K_t`, wheel friction, and the phase-vs-phase-to-phase resistance answer, and
can be run while the pivot mount is being improved.
