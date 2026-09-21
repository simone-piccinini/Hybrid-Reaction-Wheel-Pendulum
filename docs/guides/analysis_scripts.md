# The Analysis Scripts — and Why `stability_margins.py` Is the One That Tells You the Truth

The `scripts/` directory holds the project's **entry points**: small,
config-driven programs that build the plant from a YAML file, do one specific
analysis, print a summary, and write figures. They are allowed to import any
layer (`AGENTS.md` §1), so each is a short, readable demonstration of the stack.

This guide tours all of them, then spends most of its length on
[`stability_margins.py`](../../scripts/stability_margins.py) — because that is
the script that predicts how the controller will behave **on real hardware**,
not just in a noiseless simulation.

All scripts run with `PYTHONPATH=src` and take an optional config path (default
noted per script) and `-o OUTDIR`.

---

## 1. The scripts at a glance

| Script | Question it answers | Domain |
|---|---|---|
| [`run_experiment.py`](../../scripts/run_experiment.py) | What `(Q, R, W, V)` weights tune the LQG best? | the BO tuning loop |
| [`two_stage_experiment.py`](../../scripts/two_stage_experiment.py) | Can a warm-started, focused second search beat the first? | BO, refinement |
| [`compare_acquisitions.py`](../../scripts/compare_acquisitions.py) | Which acquisition (EI / UCB / Entropy Search) finds the best controller? | BO, benchmarking |
| [`frequency_analysis.py`](../../scripts/frequency_analysis.py) | Where does the **open-loop plant** resonate? | Bode of $G$ |
| [`step_response.py`](../../scripts/step_response.py) | How does the **closed loop** move in time — poles, overshoot, settling? | time domain |
| [`stability_margins.py`](../../scripts/stability_margins.py) | **How much delay / gain error can the real loop survive before it falls?** | Bode of $L=GK$ |
| [`swing_up.py`](../../scripts/swing_up.py) | Can it get from *hanging* to upright, and will the LQG catch it? | nonlinear, global |
| [`evaluation_noise.py`](../../scripts/evaluation_noise.py) | How repeatable is one cost evaluation — how much noise must the BO absorb? | BO, diagnostics |
| [`ard_relevance.py`](../../scripts/ard_relevance.py) | Which of the 11 LQG weights does the cost actually depend on? | BO, diagnostics |

The first three are the **optimisation** story (covered in
[running_experiments.md](running_experiments.md) and
[experiments.md](experiments.md)). The last three are the **analysis** story:
they take a *fixed* controller and characterise it.

### `run_experiment.py` — the tuning loop
Loads a config, runs the Bayesian-optimisation loop
(`ExperimentManager.run`), and writes the full reproducibility bundle (git
hash, config, seed, metrics, trajectories, convergence + trajectory plots) to
`results/<name>/`. This is the project's headline workflow.

### `two_stage_experiment.py` — explore, then refine
Runs a broad exploration, then a **second** search that (1) warm-starts a fresh
GP with every evaluation from the first, and (2) shrinks the search box to a
tight window around the first run's best point — spending the remaining budget
refining the good region. It re-scores both winners over many seeds for a fair
comparison.

### `compare_acquisitions.py` — the acquisition benchmark
Runs the same problem under Expected Improvement, UCB, and Entropy Search across
several seeds and reports mean/spread of the best cost, the fraction of runs
whose reported optimum actually stabilises the plant, and wall-clock time.
Entropy Search is the project goal; EI and UCB are the baselines. `--plot PNG`
writes a best-cost-so-far-vs-evaluation overlay (mean ± min–max band per
acquisition); `--from-json PATH` re-builds the table and figure from a saved run
without recomputing (e.g. re-plot `results/acquisition_comparison.json`).

### `evaluation_noise.py` — how noisy is one score?
Fixes one controller (the tuned optimum from a run's `metadata.json`) and
re-evaluates it over many seeds, reporting the mean, spread, and coefficient of
variation of the cost and the headline metrics. This measures the *observation
noise* the surrogate models directly (`optimization.md` §1): on the measured
build the cost has a ~20 % CV, almost all of it settling-time scatter — which is
why the optimiser reports the posterior-mean minimiser, not the best single
sample.

### `ard_relevance.py` — which weights matter?
Evaluates a space-filling design over the search box, fits the GP once with
ML-II, and reads the **ARD lengthscales** (`optimization.md` §2–3): a short
lengthscale means the cost is sensitive to that weight, a long one means it is
nearly flat. On the measured build the cost depends most on the LQR weights on
the *reaction-wheel* states and on `R`, while the pendulum-angle weight comes out
nearly irrelevant — so the effective problem is lower-dimensional than 11-D. It
also exposes a modelling pitfall the script warns about: below ~120 averaged
design points the ML-II fit is degenerate (drives `σ_n → 0` and inflates the
irrelevant lengthscales), so the ranking is an indication, not a precise number.

### `frequency_analysis.py` — the plant's Bode diagram
Draws one Bode diagram per input→output channel of the **open-loop plant** $G$.
It exposes the unstable pendulum mode (~4 rad/s) and the wheel-angle integrator.
This is the *plant*, with no controller in the loop. Full derivation:
[frequency_analysis.md](frequency_analysis.md) §§1–6.

### `swing_up.py` — the whole maneuver, on the nonlinear plant
Everything else in this list lives near the upright equilibrium, where the
small-angle linear model is valid. This one does not: it runs **energy-shaping
swing-up** from hanging ($\theta_p=\pi$), a switching supervisor that latches
near upright, and then the balancing **LQG** that catches and holds it — all on
the full nonlinear plant. It also front-loads a **feasibility check**: comparing
the peak reaction torque against the speed the pendulum needs at the bottom, it
will tell you the maneuver is *impossible* on the configured plant, and name the
pivot-friction threshold that would make it possible, rather than silently
failing to swing up. Theory: [swingup.md](../theory/swingup.md).

### `step_response.py` — the closed loop in time
Designs an LQR, forms $A_{cl}=A-BK$, and reports the closed-loop **modes**
(poles → natural frequency $\omega_n$, damping $\zeta$), the regulation
transient from a small tilt (linear $e^{A_{cl}t}x_0$ **vs** the true nonlinear
rollout, a check that the linearisation is faithful), and the unit-step
response with rise/peak/overshoot/settling. Full write-up:
[time_domain_response.md](time_domain_response.md).

---

## 2. Why `stability_margins.py` is the most important analysis script

Everything else either tunes the controller against a *simulated* cost or shows
how it behaves in a *clean, modelled* world. A simulation is, by construction,
an honest test only of the model you wrote down. The reaction-wheel pendulum you
actually build differs from that model in ways you did not — and cannot fully —
capture:

- the microcontroller reads sensors, computes, and writes the motor command with
  **latency** (I²C transactions, the control-loop period, PWM/driver lag);
- the motor's torque constant $K_t$ drifts with temperature, and the battery
  **voltage droops** under load — so the **real loop gain is not exactly the one
  you designed**;
- there are unmodelled mechanical resonances, sensor quantisation, and friction
  nonlinearities.

A time-domain rollout that looks perfect (like the clean `trajectory.png` the
tuner produces) tells you nothing about how much of *this* the hardware can
absorb before it becomes unstable and the pendulum falls. **Stability margins
are exactly that budget.** This is why a control engineer trusts the margin plot
over a pretty step response: it is the script that anticipates real life.

### 2.1 The loop transfer function — the right object

The script does **not** plot the closed-loop time response. It plots the
**open-loop gain of the broken loop**,

$$
L(z) = -K_c(z)\,G(z), \qquad z = e^{j\omega\Delta t},
$$

where $G$ is the plant and $K_c$ is the **full LQG controller** (Kalman observer
+ LQR law $u=-K\hat{\mathbf{x}}$), realised as a system from measurement to
control. Conceptually: cut the loop at the plant input, inject a sinusoid, and
follow it once around — plant → sensors → observer → control law — back to where
you cut. $L(j\omega)$ is what comes back. (Construction and the exact
$A_c,B_c,C_c,D_c$ realisation: [frequency_analysis.md](frequency_analysis.md)
§7.) The loop is built in **discrete time** with $z=e^{j\omega\Delta t}$, so it
is the *implemented* controller — sampling lag included — not an idealised
continuous one.

**Why a reduced model.** The wheel *angle* $\theta_w$ is a decoupled
integrator: uncontrollable from the motor and unobservable from the two sensors.
A steady-state Kalman filter does not exist for the full plant (it is
undetectable), so the script drops $\theta_w$ (state index 2, `KEEP = [0,1,3]`)
and works on the controllable + observable
$[\theta_p,\dot\theta_p,\dot\theta_w]$ model on which the LQG — and hence the
loop gain — is well-defined. The hidden mode cancels in $L$, so the margins are
unchanged.

### 2.2 The two margins, in plain hardware terms

Read off where $L(j\omega)$ approaches the critical point $-1$:

- **Phase margin (PM)** $=180^\circ + \angle L(j\omega_{gc})$, measured at the
  **gain crossover** $|L|=1$. It is the extra **phase lag** the loop tolerates
  before instability. Lag is what *latency* looks like in the frequency domain,
  so PM converts directly into a **maximum tolerable time delay**

  $$
  \tau_{\max} \;=\; \frac{\mathrm{PM}\,[\text{rad}]}{\omega_{gc}}.
  $$

  Rule of thumb: want PM $\gtrsim 30\text{–}45^\circ$.

- **Gain margin (GM)** $=-20\log_{10}|L(j\omega_{pc})|$, measured at the **phase
  crossover** $\angle L=-180^\circ$. It is how much the **loop gain** may change
  — a weaker $K_t$, a drooping battery, a mis-estimated inertia — before
  instability. Want GM $\gtrsim 6$ dB.

### 2.3 What this pendulum reports — and the warning baked into the script

Run it on the sensible bench design:

```bash
PYTHONPATH=src python scripts/stability_margins.py configs/pendulum_sensible.yaml -o results/margins
```

```
Plant 'pendulum_sensible' — LQG loop stability margins:
  Gain margin  GM = 3.30 dB  at ω = 13.96 rad/s   (want ≳ 6 dB)
  Phase margin PM = 7.24 deg  at ω = 3.29 rad/s   (want ≳ 30–45°)
  ⚠ low phase margin — fragile to delay (cf. Doyle 1978 on LQG)
```

Translate those two numbers into the language of the bench:

- **PM = 7.24°** at $\omega_{gc}=3.29$ rad/s. The delay budget is
  $$
  \tau_{\max}=\frac{7.24\times\pi/180}{3.29}\approx 0.038\ \text{s} \approx 38\ \text{ms}.
  $$
  With a 10 ms control period (`dt: 0.01`), **only ~3–4 control cycles of extra
  latency** — a slow I²C read, a heavier control computation, one dropped sample —
  would tip this loop into instability. That is a razor-thin margin for real
  firmware.
- **GM = 3.30 dB**, i.e. a gain factor of $10^{3.3/20}\approx 1.46$. A **~46 %
  increase in loop gain** destabilises it — well within the range a warm motor or
  a sagging battery can produce.

Both are **far below** the textbook rules of thumb, and the script prints the
explicit `⚠` when PM $< 30^\circ$. The loop *stabilises in simulation* yet is
**fragile in the world** — precisely the gap that a noiseless rollout hides and
this script exposes.

### 2.4 Doyle's warning — why you cannot skip this plot

The note in the script header is not decoration. Doyle (1978), *"Guaranteed
Margins for LQG Regulators: None"*, proved that although **LQR alone** has
excellent guaranteed margins ($\geq 6$ dB, $\geq 60^\circ$) and the **Kalman
filter** is robust on its own, **their combination carries no such guarantee**.
You can design a beautiful LQR, a beautiful observer, bolt them together into an
LQG, and end up with the 7° margin above. There is no shortcut: the only way to
*know* the real robustness is to form $L=GK$ for the **full** LQG and look — which
is exactly what this script does.

This is also a direct argument for the project's optimisation goal, and the
argument was acted on. The tuner minimises a *simulated* cost; while that cost
said nothing about robustness it was free to return a controller with a 7°
margin. The objective now carries an optional **margin penalty** — a hinge on
PM/GM below their floors, computed on this very loop gain — so the Bayesian
optimiser can be told to search for controllers that survive contact with
hardware. Turning it on roughly doubles the measured build's delay budget; the
A/B, and where it still falls short, is
[§7 of the robustness study](../papers/robustness_lqg_measured.md).

### 2.5 Reading the figure

The script writes `results/margins/loop_bode.png` (a generated artifact, not
version-controlled — regenerate with the command in §2.3). Reading it:
the magnitude crosses 0 dB (green **PM** line) at $\omega\approx3.3$ rad/s while
the phase there is just shy of $-180^\circ$ — a small gap, hence the small PM.
The phase touches $-180^\circ$ (red **GM** line) near $\omega\approx14$ rad/s
where the magnitude is only $-3.3$ dB below unity — a small gap, hence the small
GM. The two crossings sitting so close to the critical point *is* the fragility,
shown geometrically.

> The from-scratch, grid-based margin extractor actually finds a genuine
> $-180^\circ$ crossing of the **discrete** loop (the ~3.3 dB GM) that the
> reference `control.margin` misses — a small vindication of computing it
> directly on a dense grid. See [frequency_analysis.md](frequency_analysis.md)
> §8 for the validation story.

---

## 3. How they fit together

```
frequency_analysis.py   →  the PLANT alone         (is it unstable? where?)
step_response.py        →  the CLOSED LOOP in time  (does it settle nicely?)
stability_margins.py    →  the CLOSED LOOP's ROBUSTNESS (will it survive hardware?)
run_experiment.py …     →  TUNE the weights that all three then analyse
swing_up.py             →  the WHOLE MANEUVER, nonlinear (can it even get up there?)
```

A complete study is: tune the weights (`run_experiment.py`), confirm the
transient (`step_response.py`), **check the margins (`stability_margins.py`)
before trusting any of it on a real pendulum** — and, if the pendulum has to
start from hanging rather than being placed upright by hand, confirm the
maneuver is feasible at all (`swing_up.py`).

---

## 4. See also

- [frequency_analysis.md](frequency_analysis.md) — the full derivation of $G(s)$
  and of the loop gain $L=GK$, the complex-to-real embedding solve, and the
  validation against `scipy.signal` / `control`.
- [time_domain_response.md](time_domain_response.md) — the theory behind
  `step_response.py`.
- [running_experiments.md](running_experiments.md) — the config → run → outputs
  workflow behind `run_experiment.py`.
- [control_and_estimation.md](control_and_estimation.md) — the LQR and Kalman
  implementations whose *combination* the margins test.
