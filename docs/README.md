# Documentation Index

All project documentation lives here, grouped by purpose. New readers should
start with the **codebase overview**, then dip into theory or guides as needed.

## Start here

- [**Codebase Overview**](guides/codebase_overview.md) — the guided tour: every
  layer, the data flow, the conventions, and where to look for any question.

## Guides (practical)

- [Running Experiments](guides/running_experiments.md) — the `experiment`/`io`
  layers: config → run → outputs, reproducibility, extending.
- [Numerics From Scratch](guides/numerics_from_scratch.md) — the hand-written
  linear algebra, Riccati, integrators, eigensolver and optimiser, and how the
  validation tests anchor them.
- [Control and Estimation](guides/control_and_estimation.md) — the LQR law and
  the Kalman filter as implemented, and the Riccati duality they share.
- [Bayesian Optimisation Walkthrough](guides/bayesian_optimization_walkthrough.md)
  — a reading guide to the `optimization/` layer: GP surrogate, ML-II, kernels,
  acquisitions, and the outer loop.
- [Experiments](guides/experiments.md) — an empirical comparison of the three
  acquisition functions, reproducible from `scripts/compare_acquisitions.py`.

## Theory (the derivations each layer implements)

- [notation.md](theory/notation.md) — the symbol ↔ code glossary. **Consult
  before naming anything.**
- [model.md](theory/model.md) — the reaction-wheel-pendulum plant and its
  linearisation.
- [lqr.md](theory/lqr.md) — the LQR controller via the discrete Riccati equation.
- [kalman.md](theory/kalman.md) — the Kalman filter (and the Riccati duality).
- [optimization.md](theory/optimization.md) — the GP Bayesian optimisation with
  Entropy Search (the binding specification of the optimisation layer).

## Architecture (the contracts)

- [data_contracts.md](architecture/data_contracts.md) — the exact shape of every
  object passed between layers.
- [dependency_rules.md](architecture/dependency_rules.md) — the allowed and
  forbidden imports, layer by layer.
- [class_diagram.md](architecture/class_diagram.md) — the class diagram (Mermaid
  source).

## Conventions

- [numerical_standards.md](conventions/numerical_standards.md) — tolerances,
  conditioning, convergence criteria, and the authoritative no-library boundary.

## Top-level

- [../README.md](../README.md) — project summary and quick start.
- [../AGENTS.md](../AGENTS.md) — the working contract (read before changing code).
- [../configs/default.yaml](../configs/default.yaml) — the configuration schema,
  documented field by field.
