# Dependency Rules

> The import graph is part of the design. A lower layer must never import an upper layer. These rules are enforceable in CI (see §5) so that an architectural breach fails the build rather than silently accumulating.

---

## 1. Layer ordering (low → high)

```
numpy
  └─ numerics/          (hand-written math; depends ONLY on numpy)
       └─ core/         (domain types & interfaces; depends on numerics)
            └─ physical/        (motor, wheel, plant)
                 └─ dynamics/   (state-space, linearisation)
                      ├─ control/      (LQR)
                      └─ estimation/   (Kalman, EKF)
                           └─ simulation/   (orchestration)
                                ├─ metrics/      (Mp, Ts, effort, traj-entropy)
                                └─ optimization/ (GP, kernels, acquisition, BO)
                                     └─ experiment/  (runs, logging, artifacts)
io/  is a cross-cutting leaf: usable by experiment/ and scripts/ only.
```

A module may import from its own layer and any layer **below** it. It may never import from a layer **above**.

---

## 2. Allowed edges

| From | May import | Why |
|---|---|---|
| `numerics` | `numpy` only | foundational primitives, no domain knowledge |
| `core` | `numerics` | types may use numeric helpers (e.g. PSD checks) |
| `physical` | `core`, `numerics` | physics expressed with primitives & types |
| `dynamics` | `physical`, `core`, `numerics` | linearisation of the plant |
| `control` | `dynamics`, `core`, `numerics` | LQR needs the linear model |
| `estimation` | `dynamics`, `core`, `numerics` | Kalman needs the linear model |
| `simulation` | `physical`, `dynamics`, `control`, `estimation`, `core`, `numerics` | orchestrates a closed loop |
| `metrics` | `core` (consumes `SimulationResult`), `numerics` | pure post-processing |
| `optimization` | `simulation`, `metrics`, `core`, `numerics` | runs experiments, scores them, models the surface |
| `experiment` | `optimization`, `simulation`, `io`, `core` | top-level orchestration |
| `io` | `core` | serialisation of typed objects |
| `scripts/` | anything | entry points |
| `tests/validation/` | banned reference libs (scipy/control/…) | ground-truth checks only |

---

## 3. Forbidden edges (will fail CI)

| Forbidden | Reason |
|---|---|
| `numerics → anything but numpy` | the math core must stay dependency-free and academically pure |
| `core → physical/control/estimation/optimization/...` | types must not depend on the logic that uses them |
| `dynamics → control` / `dynamics → estimation` / `dynamics → optimization` / `dynamics → simulation` | the model must not know about its consumers |
| `control → estimation` / `estimation → control` | controller and filter are siblings; couple them only in `simulation` |
| `control → experiments` / `control → optimization` | a controller never drives experiments |
| `optimization → physical` / `optimization → control` / `optimization → estimation` (directly) | the optimiser proposes parameters and runs experiments **through `simulation`**, never by reaching into components |
| `metrics → optimization` / `metrics → simulation` | metrics are pure functions of a `SimulationResult` |
| `metrics → io` (plotting) | keep metric computation free of visualisation side effects |
| `* → io` except `experiment/`, `scripts/` | logging/plotting belong at the orchestration edge |
| `src/* → scipy / control / GPy / filterpy / sklearn / torch / jax` | the no-library rule (see `AGENTS.md §3` and `numerical_standards.md`) |
| `src/* → numpy.linalg.{solve,inv,cholesky,eig,...}` | these solve the problem; reimplement in `numerics/` |
| `tests/validation → src ... → reference lib` (indirect) | reference libs must never leak into `src` via any path |

---

## 4. The optimiser-isolation rule (most important)

`optimization/` must treat the physical/control/estimation stack as a **black box reached only through `simulation`**. Concretely:

- `BayesianOptimizer` holds a `SimulationEngine`, an `ObjectiveFunction`, a `GaussianProcess`, an `AcquisitionFunction`, and a `SearchSpace`.
- To score a candidate it calls `engine.run(config)` → `objective.evaluate(result)`. It does **not** instantiate or configure `LQRController`/`KalmanFilter` directly.
- This keeps the optimiser agnostic to *what* it tunes: the same loop would tune an MPC controller if `SimulationEngine` were swapped, with no change to `optimization/`.

---

## 5. Enforcement

Use [`import-linter`](https://import-linter.readthedocs.io/) with a layered contract mirroring §1, plus "forbidden module" contracts for the banned libraries. A minimal `pyproject.toml` / `.importlinter` sketch:

```ini
[importlinter]
root_package = inverted_pendulum

[importlinter:contract:layers]
name = Architectural layers
type = layers
layers =
    experiment
    optimization
    metrics
    simulation
    control | estimation
    dynamics
    physical
    core
    numerics

[importlinter:contract:no-solver-libs]
name = No problem-solving libraries in src
type = forbidden
source_modules = inverted_pendulum
forbidden_modules =
    scipy
    control
    GPy
    gpytorch
    botorch
    filterpy
    sklearn
    torch
    jax
```

(The `numpy.linalg` ban is checked by a small custom lint rule or a grep gate in CI, since `import-linter` operates at module granularity.)

Run the contract check in CI on every PR. A failing contract blocks merge.

---

## 6. Quick decision guide for an agent

Before adding an `import`, ask:
1. Is the target a **lower** layer than the current file? If not — stop, you are inverting the dependency.
2. Is it one of the **banned libraries**? If yes — implement it in `numerics/` instead.
3. Does it reach into `control`/`estimation`/`physical` from `optimization`? If yes — route through `simulation` instead.
4. Is it `io` from a non-edge layer? If yes — move the side effect up to `experiment/`.

If all four pass, the import is allowed.
