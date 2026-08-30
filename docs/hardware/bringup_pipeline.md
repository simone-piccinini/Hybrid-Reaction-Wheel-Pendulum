# Bring-up pipeline: from a built pendulum to a balancing controller

**Do not skip to balancing.** The simulation currently rests on three numbers
that were never measured — pivot friction `b_p`, the torque constant `K_t`, and
the sensor noise floor — and on a fourth (`I_b`) that comes from CAD rather than
the assembled machine. Gains derived from wrong parameters fail on hardware in
ways that look like control problems and are not.

This pipeline runs the machine in stages of increasing risk. **Every stage
before balancing produces a number that goes back into the config**, so the
simulation you finally tune against is the machine you actually built.

Firmware for stages 0–3: [`firmware/swing_test_v1/`](../../firmware/swing_test_v1/).
Analysis: [`scripts/identify_parameters.py`](../../scripts/identify_parameters.py).

---

## Why these parameters and not others

Ranked by how much a wrong value costs you:

| Parameter | Now | How wrong it could be | What it breaks |
|---|---|---|---|
| **`b_p`** pivot friction | `0.01` **placeholder** | unbounded — never measured | swing-up feasibility flips entirely below `b_p ≈ 0.0027`; sets all passive damping |
| **`K_t`** torque constant | `0.037` | datasheet rows disagree **0.037 / 0.0429 / 0.0618** (67 %) | scales the whole `B` matrix, so every LQR gain |
| **`V_measure`** noise | guessed | unknown; analog channel may be slip-ring correlated | Kalman trust; correlated noise breaks the filter's core assumption |
| **`I_b`** body inertia | `0.0022725` (CAD) | ±10–20 % | natural frequency, catch dynamics |
| **`J_w`** wheel + rotor | `6.117e-5` (CAD) | rotor bell adds ~4.5e-6, unmeasured | wheel authority |

`b_p` is first because it is a pure guess *and* the analysis is most sensitive
to it. Measuring it costs one stage-1 test and about two minutes.

---

## Stage 0 — Sensors only, motor never enabled

**Risk: none.** Nothing moves under power.

1. `s` — confirm encoder A calibrated, encoder B present, magnet status OK.
2. `n` — 10 s stationary log of both encoders. Reports mean, std, peak-to-peak.
3. By hand: rotate the pendulum through its full travel and watch B track
   continuously; spin the wheel by hand and watch A track.

**Pass:** B std < 0.15°, A std < 0.5°, neither channel jumps or freezes.

**Produces:** `simulation.disturbances.measurement_std` — *measured*, replacing
the guessed `[1.0e-3, 1.0e-2]`.

> If A's noise rises when the slip ring turns, the analog supply is being
> modulated by brush resistance. That is **correlated** noise, which a Kalman
> filter handles worst of all. Record it now; it changes how much you can trust
> the wheel-rate channel.

---

## Stage 1 — Free swing, motor unpowered

**Risk: low.** Motor unpowered; the pendulum is just a pendulum.

Hang the pendulum **downward** (its stable equilibrium), displace by ~20–30°,
release, let it decay:

```
w 20        # log free swing for 20 s at 200 Hz
```

**Produces, from one trace:**

- damped period `T_d` → `ω_n = sqrt(m g L / I_b)` → **validates `I_b`** against CAD
- log-decrement of the decay envelope → `ζ` → **`b_p`, measured**

**Pass:** at least 6 clean peaks; period stable within 2 %.

This is the highest-value two minutes in the whole pipeline. Run it **before**
attaching wheel wiring, so nothing tugs on the arm.

---

## Stage 2 — Wheel alone, pendulum clamped

**Risk: medium — clamp the arm rigidly before enabling the motor.**

```
r           # arm (self-calibrates encoder A, ~8 s; wheel must spin freely)
k 300       # torque step to duty 300, log wheel spin-up
d 300       # spin up to duty 300, then coast; log spin-down
```

**Produces:**

- spin-up initial acceleration → `K_t` given `J_w`, **or** the ratio `K_t/J_w`
  directly if you would rather trust neither
- steady-state speed and time constant → `K_t`, `K_e`, wheel viscous friction
- coast-down decay → `b_w` (viscous) plus the Coulomb breakaway term

**This also settles the 10 Ω question.** Run stage 2 with a bench supply showing
current and compare against the predicted stall currents in
[thermal_and_duty_limits.md](thermal_and_duty_limits.md) §5. If measured current
is about 2× predicted, the datasheet's 10 Ω is phase-to-phase and every duty
ceiling in that note halves.

**Pass:** `K_t` lands inside 0.030–0.070; spin-up is smooth, not surging.

---

## Stage 3 — Coupled swing (the first time the wheel moves the arm)

**Risk: medium. The arm will move. Clear the space and keep hands clear.**

Pendulum **hanging** and free to swing, wheel powered. Sign check first:

```
c 200       # conservative duty cap
t 150 200   # one torque pulse: duty 150 for 200 ms, log the response
```

**Confirm this before anything else:** a *positive* duty must move the arm in a
*consistent, known* direction. Reaction torque opposes the wheel's acceleration,
so a positive command should swing the arm the same way every time. If the sign
is inconsistent, or inverted relative to the model, fix it here — an inverted
sign in a balancing controller is positive feedback, and the arm slams over
immediately.

Then the small resonant swing:

```
e 8 15      # energy pump, gain 8, abort above 15 deg
```

This drives the wheel in phase with the arm's velocity to add a little energy
each swing — the same energy-shaping law as [swingup.md](../theory/swingup.md),
but with a **hard amplitude cap** so it builds a gentle oscillation and then
stops. It is deliberately *not* a swing-up.

**Produces:**

- confirmed torque sign and the arm's direction convention
- pulse response `Δθ̇ / (τ·Δt)` → **the coupling gain**, an independent
  measurement of `I_b` that does not depend on CAD at all
- practical confirmation that the available torque actually moves the arm

**Pass:** the arm responds to every pulse in the same direction; amplitude grows
smoothly under `e`, and the abort fires cleanly at the cap.

> **Expect the abort to be your friend here.** If `e` cannot build amplitude at
> all, that is real information: torque authority is marginal, and
> [thermal_and_duty_limits.md](thermal_and_duty_limits.md) §4 says raise the peak
> duty ceiling before concluding the design is wrong.

---

## Stage 4 — Back to simulation (no hardware)

Feed the measurements in and re-tune. `identify_parameters.py` emits a ready
YAML fragment:

```bash
python scripts/identify_parameters.py results/hw/ -o configs/pendulum_identified.yaml
```

```bash
PYTHONPATH=src python scripts/run_experiment.py configs/pendulum_identified.yaml -o results/identified
```

```bash
PYTHONPATH=src python scripts/stability_margins.py configs/pendulum_identified.yaml -o results/identified_margins
```

**Gate — do not proceed unless all four hold:**

1. the tuned controller stabilises, and **peak |u| stays inside the rail**;
2. **phase margin ≥ 25°** — with ~10–20 ms of measured latency stack to spend,
   less than that will not survive the real loop;
3. **max recoverable tilt ≥ 2×** the angle you can realistically place the arm
   at by hand;
4. the reported optimum is checked as the **best-observed** point, not only the
   posterior-mean minimiser — in 11-D at a 28-evaluation budget the latter is
   routinely far worse (this repo has hit that three times).

If the gate fails, the fix is a parameter or the actuator ceiling — not gain
tweaking.

---

## Stage 5 — Balancing (a later, separate step)

Only after stage 4 passes. In outline, and deliberately not built yet:

- port the LQR gain `K` and the steady-state Kalman gain `L` as constants;
- **derive wheel rate in the estimator from raw encoder A**, never from the
  firmware's filtered `velocity` — its ~50 ms lag is 105 % of the delay budget
  and is unconditionally destabilising
  ([handoff §4.4](firmware_v6_handoff.md));
- run the estimator at 1 kHz, predicting between encoder-B updates;
- start with the arm **held by hand** near upright, controller running, and feel
  whether it pushes the right way before letting go;
- keep every v6 safety layer, and add an angle-based abort.

---

## The loop this closes

```
   stage 0-3  ──►  measured b_p, K_t, I_b, J_w, noise
                        │
                        ▼
   stage 4    ──►  config ──► BO tuning ──► margins ──► GATE
                        │                                │
                        └── fails: fix the parameter     │
                            or the actuator ceiling      │
                                                         ▼
   stage 5    ──────────────────────────────────►   balancing
```

Each stage is cheap, and each one removes a way the balancing attempt could fail
for a reason that has nothing to do with the controller.
