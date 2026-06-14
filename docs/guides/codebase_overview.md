# Codebase Overview — A Guided Tour

This document is the map of the repository: what each layer does, how data
flows between them, and how to read the code. It is meant to be the first thing
a new reader opens. For *why* each algorithm is the way it is, follow the links
into [`docs/theory/`](../theory/); for *how to run* the system, see
[running_experiments.md](running_experiments.md).

---

## 1. What the system does, in one paragraph

An inverted pendulum carries a motor-driven reaction wheel. Spinning the wheel
produces a reaction torque on the pendulum body — the only way to balance it.
We linearise the plant about the upright equilibrium, estimate its state from
noisy sensors with a **Kalman filter**, and stabilise it with an **LQR** law
(the two together are **LQG**). The LQG design has four weight matrices
`(Q, R, W, V)` that trade off regulation tightness, control effort, and trust in
the model vs the sensors. Choosing them by hand is hard, so we **tune them
automatically**: a **Bayesian optimiser** treats "run a closed-loop simulation
and score it" as an expensive black-box function and searches the weight space
with a **Gaussian-process surrogate** and an **Entropy-Search** acquisition.

---

## 2. The layered architecture

The code is a strict stack. A lower layer never imports a higher one; the rule
is enforced by design and documented in
[dependency_rules.md](../architecture/dependency_rules.md).

```
numpy
 └─ numerics/      hand-written math (no SciPy, no numpy.linalg solvers)
     └─ core/      domain value objects + the data contracts
         └─ physical/      motor, wheel, the nonlinear plant
             └─ dynamics/      ZOH stepper + linearise-and-discretise
                 ├─ control/       LQR gain
                 └─ estimation/    Kalman filter
                     └─ simulation/    the closed-loop rollout (the oracle)
                         ├─ metrics/       overshoot, settling, effort, entropy
                         └─ optimization/  GP, ML-II, kernels, acquisitions, BO loop
                             └─ experiment/    config-driven runs
  io/  is a cross-cutting leaf: only experiment/ and scripts/ may use it.
```

Reading order that mirrors the dependencies (bottom-up) is also the order the
project was built and the easiest way to understand it:
**numerics → core → physical → dynamics → control/estimation → simulation →
metrics → optimization → experiment**.

---

## 3. Layer-by-layer

### `numerics/` — the from-scratch mathematics
The academic heart. Everything that "solves a problem" is implemented here so
no banned library is needed (see [numerics_from_scratch.md](numerics_from_scratch.md)):

- `linalg.py` — Cholesky, triangular solves, `chol_solve`, partial-pivot LU,
  and a shifted-QR `eigvals` / `spectral_radius`.
- `matrix_exp.py` — `expm` by scaling-and-squaring with a Padé approximant.
- `integrators.py` — `rk4` and `euler`.
- `riccati.py` — `solve_dare`, the discrete Riccati solver (used by **both** LQR
  and Kalman, via duality).
- `optimizers.py` — a BFGS minimiser with restarts (for ML-II).

### `core/` — the typed vocabulary
Frozen dataclasses passed between layers ([data_contracts.md](../architecture/data_contracts.md)):
`StateSpaceModel`, `LQGConfig` (the decision vector `θ` and its log-space
packing), `SearchSpace`, `Dataset`, `GPPosterior`, `SimulationResult`. No
algorithms live here — only validated, immutable data.

### `physical/` — the plant
`DCMotor` (steady-state voltage→torque), `ReactionWheel` (flywheel + bearing
friction), and `ReactionWheelPendulum`, which holds the full nonlinear `sin θ`
dynamics and a `linearize()` producing the continuous `StateSpaceModel`
([model.md](../theory/model.md)).

### `dynamics/` — true model vs design model
`NonlinearPlantModel` advances the *true* plant one ZOH step with `rk4`;
`LinearizedPlantModel` pairs the continuous linearisation with its discrete
form `(A_d, B_d)`. Both expose the same `step(state, voltage)` so the simulator
can use either.

### `control/` and `estimation/` — the LQG halves
`LQRController` reads the feedback gain `K` off the discrete Riccati solution and
applies `u = −K x̂` ([lqr.md](../theory/lqr.md)). `KalmanFilter` runs the
predict/update recursion and can compute its steady-state gain from the *dual*
Riccati equation ([kalman.md](../theory/kalman.md)). They are siblings — coupled
only inside the simulator. See [control_and_estimation.md](control_and_estimation.md).

### `simulation/` — the oracle
`SimulationEngine.run(config, seed)` plays the closed loop: measure → filter →
`u = −K x̂` → step the true plant + injected noise. It returns a typed
`SimulationResult`. `Disturbances` carries the *true* world noise, deliberately
separate from the filter's assumed `W`/`V`. This is the single entry point the
optimiser is allowed to touch.

### `metrics/` — scoring a rollout
Pure functions of a `SimulationResult`: `overshoot`, `settling_time`,
`control_effort`, `oscillation_energy`, `itae`, and the (physical) `trajectory_entropy`
— not to be confused with the information-theoretic entropy in Entropy Search.

### `optimization/` — the tuner
The GP surrogate (`gaussian_process.py`, R&W Algorithm 2.1), the ML-II inner
loop (`marginal_likelihood.py`), the ARD `kernels/`, the `acquisition/`
strategies (Expected Improvement, UCB, and the goal **Entropy Search**), the
scalar `objective.py`, and `bayes_optimizer.py` — the outer loop. Walkthrough:
[bayesian_optimization_walkthrough.md](bayesian_optimization_walkthrough.md).

### `experiment/` and `io/` — orchestration and I/O
`ExperimentManager` turns a config into the whole object graph, runs the loop,
and records the result. `io/` parses YAML (`config_loader`), persists runs
(`run_logging`), and plots (`plotting`). See [running_experiments.md](running_experiments.md).

---

## 4. The data-flow, end to end

```
ReactionWheelPendulum.linearize()  ─▶  StateSpaceModel
StateSpaceModel.discretize(dt)     ─▶  (A_d, B_d)  ─▶  LQRController, KalmanFilter
LQGConfig  ─▶  SimulationEngine.run()  ─▶  SimulationResult
SimulationResult  ─▶  ObjectiveFunction.evaluate()  ─▶  y  (a scalar cost)
(θ, y)  ─▶  Dataset  ─▶  GaussianProcess.fit()
GaussianProcess  ─▶  AcquisitionFunction.select()  ─▶  next θ
next θ  ─▶  LQGConfig.from_vector()  ─▶  (loop)
```

Every arrow is one of the typed `core` objects — no raw dicts or unlabelled
tuples cross a layer boundary.

## 5. The two nested optimisation loops (do not confuse them)

- **Outer loop** (Bayesian optimisation, `bayes_optimizer.py`): searches the
  LQG weights `θ = vec(Q, R, W, V)` over the *true closed-loop cost*.
- **Inner loop** (ML-II, `marginal_likelihood.py`): searches the GP kernel
  hyperparameters `φ = (ℓ, σ_f, σ_n)` over the *marginal likelihood*, every time
  a new observation is added.

Likewise there are **two distinct "entropies"**: the physical
`metrics/trajectory_entropy.py` and the information-theoretic `H[p_min]` inside
`acquisition/entropy_search.py`. They never mix.

---

## 6. Conventions worth knowing before reading code

- **No-library rule.** Inside `src/`, NumPy is a calculator only. No
  `numpy.linalg` solver, no SciPy. The hand-written equivalents live in
  `numerics/`. Reference libraries appear *only* in `tests/validation/`.
- **Every public function cites its equation** — a docstring pointing at an
  equation number or a `docs/theory/` section.
- **Determinism.** Every stochastic path takes an explicit seeded
  `numpy.random.Generator`; a run is reproducible from `(config, seed)`.
- **Names come from [notation.md](../theory/notation.md).** `Q_lqr` vs
  `W_process`, `P_dare` vs `P_est`, `L_gain` vs `L_chol` — the glossary resolves
  the dangerous collisions.
- **Tolerances come from [numerical_standards.md](../conventions/numerical_standards.md)**,
  never inline magic numbers.

---

## 7. Where to look for a given question

| Question | Look at |
|---|---|
| How is the plant modelled? | `physical/pendulum.py`, [model.md](../theory/model.md) |
| How is the LQR gain computed? | `control/lqr_controller.py`, `numerics/riccati.py`, [lqr.md](../theory/lqr.md) |
| How does the Kalman filter work here? | `estimation/kalman_filter.py`, [kalman.md](../theory/kalman.md) |
| What is the cost being optimised? | `optimization/objective.py`, [notation.md](../theory/notation.md) §6 |
| How does Entropy Search choose a point? | `optimization/acquisition/entropy_search.py`, [optimization.md](../theory/optimization.md) §4 |
| How do I run / configure an experiment? | [running_experiments.md](running_experiments.md), `configs/default.yaml` |
| Which library is allowed where? | [numerical_standards.md](../conventions/numerical_standards.md) §3 |
| What does each result file contain? | [running_experiments.md](running_experiments.md) §4 |
