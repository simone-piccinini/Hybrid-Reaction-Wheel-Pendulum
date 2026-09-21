# Robustness of the Tuned LQG on the Identified Plant

**What changed when the plant stopped being a CAD estimate and became a set of
*measured* numbers — and why the pivot encoder, not the physics, is now the
binding constraint on balancing (`configs/pendulum_identified.yaml`).**

This is the sequel to [Robustness on the measured build](robustness_lqg_measured.md),
which analysed the controller the optimiser found for the *CAD/measured*
simulation. Here the plant parameters and the sensor noise come from **hardware
measurement** (the [bring-up pipeline](../hardware/bringup_pipeline.md), stages
0–1), and the story flips: measuring the machine *validated* the model and
*unlocked* swing-up, yet made the balancing problem **harder**, because the honest
sensor noise is worse than any guess had assumed.

---

## 1. Summary

| Quantity | CAD / guess | Measured | Consequence |
|---|---|---|---|
| `I_b` body inertia | 0.0022725 | **0.00224935** (stage 1) | CAD **validated** — 1.0 % apart |
| `b_p` pivot friction | 0.01 *(guess, "not measured")* | **0.000269** (ζ = 0.0074) | **37× smaller** — swing-up now feasible |
| pivot-angle noise | 1 LSB assumed | **2.63° (1σ), and not white** | the new binding constraint |
| Tuned-weight phase margin | 11.6° (measured build) | **−21.2°** (identified) | the nominal loop is now *unstable* |

In words: the two parameters people worried about (`I_b`, `b_p`) turned out to be
either right or better than assumed, so **the plant is not the problem.** What the
measurement exposed is the **pivot encoder**: out-of-plane wobble of the arm
moves the angle reading by ±4.6° *at a fixed true angle*
([sensor characterisation](../hardware/sensor_characterisation.md)). Feeding that
honest noise to the tuner drives the reported phase margin **negative** — and it
does so on *every* tuning that includes the measured pivot noise. The bring-up
robustness gate (stage 4, phase margin ≥ 25°) correctly **fails**: this build is
not ready to balance until the sensing improves.

---

## 2. What the measurement changed

Three things moved from estimate to measured (`configs/pendulum_identified.yaml`):

- **`I_b` — validated.** The free-swing period gives `I_b` from the period
  *alone*, so it is immune to every angle-distortion problem in the pivot
  encoder: 0.00224935 vs the CAD 0.0022725, **1.0 % apart**. The Lagrangian model
  of [model.md](../theory/model.md) is confirmed on the real arm.
- **`b_p` — 37× smaller.** Log-decrement on the same ring-down gives
  `b_p = 2.69e-4` (ζ = 0.0074, period 0.773 s vs 0.777 s predicted, 0.5 %). The
  0.01 placeholder was 3.7× the *wrong* side of the swing-up threshold
  (`b_p < 0.0027`, [swingup.md](../theory/swingup.md)); the measurement is 10× the
  *right* side. Simulated on the identified plant the arm **swings up in 2.34 s
  and the LQG catches it** (`results/identified_swingup/`).
- **The reaction wheel — lighter as built.** The printed PLA wheel came out at
  `I_w = 6.12e-5` (41 g), not the CAD `1.04e-4` (64 g); the reduced momentum
  capacity is analysed in [reaction_wheel_design.md](../hardware/reaction_wheel_design.md)
  (adding rim mass is *not* the fix).

And one thing got honestly worse: the **measured sensor noise**. The pivot
channel carries a 2.63° (1σ) disturbance that is not electrical noise but
mechanical wobble, and is **not white**, so representing it as a Gaussian
measurement covariance `V` is already optimistic. The search-space bounds on `V`
had to be *widened* for the tuner to even express how much to distrust the
sensor — the original bounds excluded the measured value.

---

## 3. What the measurement bought, and what it cost

**Bought:** a validated model, and swing-up feasibility. Removing 37× of assumed
pivot friction moves the energy-pumping feasibility bound from *impossible* to
*comfortable* (§1 of the swing-up brief).

**Cost:** the same 37× is 37× of *free passive damping* the balancing loop had
silently been leaning on. With the guessed friction, the plant damped its own
oscillation; with the real bearing it does not, so the **controller now has to
supply that damping actively** — through the very sensor channel that is 2.63°
noisy. That is the trade the rest of this note quantifies.

---

## 4. The re-tune on the identified plant

The tuner was re-run on `pendulum_identified.yaml` with the **robustness term on**
(`w_phase_margin = 50`, `w_gain_margin = 20`, `PM_min = 30°`, `GM_min = 6 dB`),
targets relaxed to achievable values (`Mp ≤ 15 %`, `Ts ≤ 2 s`), the divergence
penalty raised clear of the achievable cost range, and the `V` bounds widened so
the tuner *can* down-weight the sensor. Even so, the reported optimum is poor:

| Metric | Value |
|---|---|
| best objective (posterior mean) | 3356 |
| overshoot `M_p` | **218 %** |
| settling `T_s` | **9.99 s** (never settles in the 10 s window) |
| control effort ∫u² | 277.9 |
| peak command | **17.4 V** (past the 12 V rail) |

And its stability margins, read the two ways (cf. the measured-build paper §5):

| Weighting | Phase margin | Gain margin |
|---|---|---|
| **Tuned** `(Q,R,W,V)` from the run | **−21.2°** @ 1.19 rad/s | ∞ (no −180° crossing) |
| Demonstration weights (stock script) | 23.0° @ 6.79 rad/s | 11.7 dB @ 33.3 rad/s |

The tuned-weight **phase margin is negative**: the nominal LQG loop is *unstable*.
The 10 s rollout does not formally diverge (the pendulum never crosses ±π/2), but
it also never settles — a 218 % overshoot and a command that clips the rail are
what a nominally-unstable loop looks like when a finite horizon and saturation
hold it barely together. Note the *gain* robustness is, if anything, excellent
(demo GM 11.7 dB ≈ +280 % gain tolerance): as on the measured build, **gain error
is not the risk — phase is** — only here the phase margin has crossed zero.

---

## 5. Why the sensor is the binding constraint

The phase margin comes out **negative on all three tunings that include the
measured pivot noise** (the identified plant here, and the two `build_v2`
variants). That invariance is the tell: it is not a property of one parameter set,
it is the sensor.

Mechanistically: the pivot encoder's ±4.6° wobble sits right in the band the loop
must act on. The filter's only defence is to distrust that channel (a large `V`),
but a filter that trusts the pendulum-angle measurement less is slower to react to
a real tilt — and slowness *is* phase lag near the crossover, which is exactly
what eats phase margin. So the sensor noise sets a floor on achievable lag, and on
this build that floor is above the stability boundary. No `(Q,R,W,V)` in the
searched space escapes it, which is why widening the box and turning the
robustness penalty on did not rescue the margin.

This is Doyle (1978) once more — LQG carries no guaranteed margin — but with a
sharper edge than the measured-build paper: there the tuner *chose* a thin
margin the objective had not forbidden; here the achievable margin is **negative
regardless**, because the plant it must control through a 2.63°-noisy sensor
cannot be stabilised with headroom by output feedback at this sample rate.

---

## 6. What this means for the build (documentation of the path)

The bring-up pipeline's **stage-4 gate — "phase margin ≥ 25° on the identified
plant before touching stage 5 (balancing)" — fails**, and it is *right* to fail:
it is doing its job of blocking a hardware attempt that would not hold. The
parameters are settled; the open work is **sensing**, not tuning:

1. **Reduce the out-of-plane wobble mechanically** — it is a rigidity/mounting
   problem, not an electrical one; a stiffer arm or pivot removes the disturbance
   at source (the only fix that recovers phase margin cleanly).
2. **A calibration map has limited reach** — it only helps if the distortion is a
   single-valued function of angle, and only while the magnet stays put
   ([sensor_characterisation.md §0c](../hardware/sensor_characterisation.md)); the
   residual still lands in `V` and still costs phase.
3. **A second/better angle sensor** (e.g. an IMU-derived tilt fused with the
   encoder) attacks the noise floor directly.

None of these is a controller change — which is the point of writing this down:
the tuning stack has done what it can, and the next gain is upstream of it.

---

## 7. Reproduce

```bash
# the identified-plant tuning run (robustness term on):
PYTHONPATH=src python scripts/run_experiment.py configs/pendulum_identified.yaml -o results/identified

# demonstration-weight margins (the stock script's fixed weighting):
PYTHONPATH=src python scripts/stability_margins.py configs/pendulum_identified.yaml -o results/identified_margins
#   for the TUNED-weight margin of §4 (the -21.2 deg), use the (Q,R,W,V) from
#   results/identified/metadata.json on the reduced [theta_p, theta_p_dot,
#   theta_w_dot] model, as in robustness_lqg_measured.md §3.

# swing-up on the identified plant (feasible at the measured b_p):
PYTHONPATH=src python scripts/swing_up.py configs/pendulum_identified.yaml -o results/identified_swingup

# the identification itself, from the raw hardware logs:
PYTHONPATH=src python scripts/identify_parameters.py results/hw/ -o configs/pendulum_identified.yaml
```

---

## 8. Bottom line

Identifying the plant on hardware **validated the model** (`I_b` to 1 %) and
**unlocked swing-up** (`b_p` 37× below the feasibility threshold) — real progress.
But it also removed the free damping the balancer had been leaning on and put the
**true pivot-encoder noise (2.63°, non-white)** into the loop, and against that the
best achievable phase margin is **negative (−21.2°)**. The binding constraint has
moved from the *physics*, which is now well-known, to the *sensing*, which is not
good enough. The bring-up robustness gate fails accordingly, and the next work is
mechanical and sensory, not another tuning run.
