# Inverted Reaction-Wheel Pendulum — LQG + Bayesian Optimisation

A from-scratch, academic implementation of a stabilising control system for an
**inverted reaction-wheel pendulum**. The plant is linearised about the upright
equilibrium, its state is estimated with a **Kalman filter** and controlled
with an **LQR** law (together: **LQG**), and the LQG weight matrices
`(Q, R, W, V)` are **tuned automatically by Bayesian Optimisation** with a
hand-written **Gaussian-process** surrogate and an **Entropy-Search**
acquisition function.

The point of the project is the *derivation and implementation*: correctness,
transparency, and traceability to the theory matter more than raw speed.

## The golden rule: no library that solves the problem

Every numerical algorithm is written from scratch in `numerics/` — Cholesky and
triangular solves, LU, a shifted-QR eigensolver, the matrix exponential, RK4,
the discrete Riccati solver, and a BFGS optimiser. NumPy is used as a
calculator (`@`, broadcasting, slicing, seeded RNG) — never as a solver. SciPy,
scikit-learn, `control`, `filterpy`, GPy and friends appear **only** in
`tests/validation/`, as ground truth to check the hand-written code against.
See [`AGENTS.md`](AGENTS.md) and [`docs/conventions/numerical_standards.md`](docs/conventions/numerical_standards.md).

## Architecture

The code is a strict dependency stack — a lower layer never imports an upper
one ([`docs/architecture/dependency_rules.md`](docs/architecture/dependency_rules.md)):

```
numerics → core → physical → dynamics → control ┐
                                        estimation ┘→ simulation → metrics
                                                                 → optimization → experiment
                                                                       io ──────────┘ (leaf)
```

| Layer | What it holds |
|---|---|
| `numerics` | hand-written linear algebra, Riccati, integrators, optimiser, eigensolver |
| `core` | domain value objects (`StateSpaceModel`, `LQGConfig`, `SearchSpace`, `Dataset`, `GPPosterior`, `SimulationResult`) |
| `physical` | `DCMotor`, `ReactionWheel`, `ReactionWheelPendulum` (nonlinear plant + linearisation) |
| `dynamics` | the ZOH stepper for the true plant and the linearise-and-discretise pipeline |
| `control` | `LQRController` (gain via the discrete Riccati equation) |
| `estimation` | `KalmanFilter` (predict/update + dual-DARE steady state) |
| `simulation` | `SimulationEngine` — the closed-loop rollout and the optimiser's oracle |
| `metrics` | overshoot, settling time, control effort, trajectory entropy |
| `optimization` | GP surrogate, ML-II, kernels, acquisitions (EI/UCB/Entropy Search), the BO loop |
| `experiment` / `io` | config-driven reproducible runs; YAML loading, logging, plotting |

The theory each layer implements is written up in [`docs/theory/`](docs/theory/)
(`model.md`, `lqr.md`, `kalman.md`, `optimization.md`, `notation.md`).

## Documentation

Full documentation is indexed in [`docs/README.md`](docs/README.md). New readers
should start with the [**codebase overview**](docs/guides/codebase_overview.md) —
a guided tour of every layer, the data flow, and the conventions. Topic guides
cover the [hand-written numerics](docs/guides/numerics_from_scratch.md),
[control and estimation](docs/guides/control_and_estimation.md), the
[Bayesian-optimisation layer](docs/guides/bayesian_optimization_walkthrough.md),
[running experiments](docs/guides/running_experiments.md), and an empirical
[comparison of the acquisition functions](docs/guides/experiments.md).

## Running an experiment

Tests and scripts run with `PYTHONPATH=src` (there is no installed package yet).

```bash
PYTHONPATH=src python scripts/run_experiment.py configs/default.yaml -o results/run01
```

This tunes the LQG weights on the configured plant and writes a full,
reproducible run record (git hash, config, seed, metrics, trajectories, and
diagnostic plots) to `results/run01/`. The configuration schema is documented
field-by-field in [`configs/default.yaml`](configs/default.yaml), and the whole
workflow — config → run → outputs, determinism, extending — is covered in
[`docs/guides/running_experiments.md`](docs/guides/running_experiments.md).

A run is fully determined by `(config, seed)`: re-running the same config
reproduces the same result.

## Tests

```bash
PYTHONPATH=src python -m pytest tests/ -q
```

- `tests/unit/` — properties of each primitive and component (no reference libs).
- `tests/validation/` — cross-checks against SciPy / scikit-learn / filterpy to
  a stated tolerance; the only place the banned libraries may be imported.

## Repository layout

```
src/inverted_pendulum/   the package, one directory per layer above
docs/theory/             the derivations each layer implements
docs/architecture/       data contracts, dependency rules, class diagram
docs/conventions/        numerical standards (tolerances, the no-library rule)
docs/guides/             how to run experiments
configs/                 experiment configurations (default.yaml is the schema)
scripts/                 entry points (run_experiment.py)
tests/                   unit and validation suites
```
