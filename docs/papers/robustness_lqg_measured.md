# Robustness of the Tuned LQG on the Measured Build

**A stability-margin analysis of the Bayesian-optimised controller for the real,
CAD/measured reaction-wheel pendulum (`configs/pendulum_measured.yaml`).**

This report answers a question a clean simulation cannot: *will the controller
that the optimiser found survive contact with real hardware* — firmware latency,
a drooping battery, a warm motor? It reads the **open-loop gain of the LQG loop**
and its **gain/phase margins** (the method and theory: [the analysis-scripts
guide](../guides/analysis_scripts.md) and [frequency_analysis.md
§7](../guides/frequency_analysis.md#7-the-experts-plot-open-loop-gain-and-stability-margins)).

---

## 1. Summary

The BO-tuned controller for the measured plant is **stable and gain-robust, but
delay-fragile**:

| Margin | Tuned controller | Rule of thumb | Verdict |
|---|---|---|---|
| **Gain margin (GM)** | **6.85 dB** at ω = 14.6 rad/s | ≳ 6 dB | ✅ **passes** |
| **Phase margin (PM)** | **11.6°** at ω = 3.56 rad/s | ≳ 30–45° | ⚠️ **low** |
| Delay margin (derived) | **56.8 ms** (≈ 5.7 control cycles at 10 ms) | — | comfortable |
| Gain tolerance (derived) | **+120 %** loop gain (×2.2) | — | strong |

In words: the loop can absorb a **doubling of loop gain** (a much weaker `K_t`, a
sagging battery, a mis-estimated inertia) before going unstable — a genuinely
healthy gain margin that *clears* the textbook 6 dB bar. What it cannot absorb is
much extra **phase lag**: only ~57 ms of round-trip delay, or any unmodelled lag
near the 3.6 rad/s crossover, before stability is lost. **Latency is the binding
risk on this build; gain error is not.**

---

## 2. Why this is the test that matters

The tuner minimises a *simulated* cost on the model in
`configs/pendulum_measured.yaml`. A simulation is, by construction, an honest
test only of the model you wrote down. The hardware differs in ways the model
omits:

- the microcontroller reads two AS5600 over I²C, computes, and writes the motor
  command with **latency**;
- the motor's `K_t` drifts with temperature and the battery **droops** under
  load, so the **real loop gain ≠ the designed loop gain**;
- unmodelled mechanical resonances and friction add **phase lag**.

The closed-loop rollout ([`results/measured/trajectory.png`]) looks immaculate —
recovery in ~0.3 s, ~7 V peak well inside the 12 V rail. That tells you nothing
about how much of the above the hardware can take. The **margins do**, and they
are the quantity to trust before powering the real pendulum.

---

## 3. Method

The robustness is read from the **open-loop loop gain** of the broken loop,
`L(z) = −K_c(z)·G(z)` with `z = e^{jωΔt}`, where `G` is the measured plant and
`K_c` is the **full LQG** (steady-state Kalman observer + LQR law
`u = −K x̂`). Built in **discrete time**, so it is the *implemented* controller
— sampling lag included. Two details:

- **Tuned weights, not demonstration weights.** The stock
  `scripts/stability_margins.py` uses a *fixed* demonstration weighting; the
  numbers here instead use the **actual BO-tuned** `(Q, R, W, V)` from
  [`results/measured/metadata.json`](../../results/measured/metadata.json), so
  the margins describe the controller the experiment really produced. (See §5
  for why this distinction changes the verdict.)
- **Reduced model.** The wheel *angle* `θ_w` is a decoupled integrator —
  uncontrollable from the motor and unobservable from the sensors — so a
  steady-state Kalman filter does not exist for the full plant. It is dropped
  (`KEEP = [0,1,3]`), leaving the controllable + observable
  `[θ_p, θ̇_p, θ̇_w]` model on which the LQG, and hence the loop gain, is
  defined. The hidden mode cancels in `L`, so margins are unaffected.

The tuned weights used (full diag; reduced where noted):

```
Q = [4.137, 1.477, 0.0176, 0.00205]   R = [3.231]
W = [1.77e-4, 6.60e-6, 1.04e-5, 2.71e-4]   V = [9.32e-6, 9.64e-5]
```

---

## 4. Results and interpretation

**Gain margin — healthy.** GM = 6.85 dB at the phase crossover ω ≈ 14.6 rad/s.
A factor of `10^(6.85/20) ≈ 2.2`: the loop survives a **+120 % change in loop
gain**. On the bench, loop gain ∝ `K_t · V_supply / R_a`, so this covers a hot
motor losing torque constant, a LiPo sagging from 12 V toward 8–9 V under a
current spike, and ordinary `K_t`/inertia identification error — comfortably. For
a control loop this is a reassuring number, and it clears the 6 dB rule.

**Phase margin — low.** PM = 11.6° at the gain crossover ω ≈ 3.56 rad/s. Phase
lag is what latency looks like in the frequency domain, so this converts to a
maximum tolerable round-trip delay

```
τ_max = PM[rad] / ω_gc = (11.6 · π/180) / 3.56 ≈ 0.057 s ≈ 57 ms.
```

At the 10 ms control period that is **~5.7 control cycles** of *additional* delay
beyond what the discrete model already accounts for. So a slow I²C burst, a
heavier control computation, one or two dropped samples, or filtering lag on the
sensors is the failure mode to watch — not gain error. 11.6° is below the 30–45°
comfort band: the controller works, but with little phase headroom.

**Geometrically** (the loop-gain Bode, regenerate with the command in §7): the
magnitude crosses 0 dB while the phase is still ~11° above −180° (the small PM),
and the phase only reaches −180° out at ~15 rad/s where the magnitude has fallen
6.85 dB below unity (the healthy GM). The gain curve rolls off steeply between
the two crossings — which is exactly why this loop is gain-robust yet
phase-poor.

---

## 5. Why "tuned vs demonstration weights" matters

Running the stock script (fixed demo weights) on the same plant gives a *different
and more pessimistic* gain margin — the tuning genuinely improved gain robustness:

| Controller · plant | PM | GM | delay margin | gain tolerance |
|---|---|---|---|---|
| **Tuned · measured** (this report) | 11.6° @ 3.56 | **6.85 dB** @ 14.6 | 57 ms (5.7 cyc) | +120 % |
| Demo weights · measured | 9.8° @ 5.57 | 2.61 dB @ 15.9 | 31 ms | +35 % |
| Demo weights · sensible default | 7.2° @ 3.29 | 3.30 dB @ 14.0 | 38 ms | +46 % |

The lesson: **margins are a property of the specific `(Q,R,W,V)`, not of the
plant alone.** Reporting the demo-weight margins would have *understated* this
controller's gain robustness by ~4 dB. Always read the margins of the weights you
will actually deploy — which is what this report does, and a small improvement
worth folding back into `scripts/stability_margins.py`.

---

## 6. Doyle's warning, and what to do about it

The low PM is the textbook signature of Doyle (1978), *"Guaranteed Margins for
LQG Regulators: None"*: LQR **alone** guarantees ≥ 6 dB / ≥ 60°, and the Kalman
filter is robust on its own, but **their combination guarantees nothing**. Here
the optimiser recovered a strong *gain* margin but left the *phase* margin thin —
no contradiction, just the absence of any guarantee. Practical consequences for
this build:

1. **Keep the loop deterministic and fast.** The 10 ms period is fine (the budget
   is ~5.7 cycles), but protect it: bound the I²C read time, avoid blocking calls
   in the control ISR, and don't add heavy sensor filtering that injects lag near
   3.6 rad/s.
2. **Gain drift is not the worry.** With +120 % gain tolerance, battery droop and
   `K_t` variation are well covered — no need to over-engineer the supply.
3. **Tune *for* phase margin.** The objective currently has no robustness term, so
   the optimiser is free to return an 11.6° controller. Folding a hinge penalty
   for `PM < 30°` (and `GM < 6 dB`) into the cost would steer the search toward
   controllers that keep phase headroom — the natural next experiment.

---

## 7. Reproduce

```bash
# tuned-controller margins (this report's numbers):
PYTHONPATH=src python scripts/stability_margins.py configs/pendulum_measured.yaml -o results/measured_margins
#   ^ stock script uses demonstration weights; for the TUNED-weight margins of
#     §4 use the actual (Q,R,W,V) from results/measured/metadata.json.

# the experiment that produced the controller:
PYTHONPATH=src python scripts/run_experiment.py configs/pendulum_measured.yaml -o results/measured
```

> On Windows, prefix with `PYTHONIOENCODING=utf-8` — `stability_margins.py`
> prints `ω`/`≳`, which the default cp1252 console cannot encode.

---

## 8. Bottom line

The tuned LQG **stabilises the measured pendulum with strong gain robustness
(6.85 dB, +120 %) but thin phase robustness (11.6°, ~57 ms of delay)**. It is
safe to bring up on hardware *provided the control loop's latency is kept tight*;
the next design iteration should add a phase-margin term to the tuning objective.
