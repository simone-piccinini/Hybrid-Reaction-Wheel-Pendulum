# AGENTS.md

> **Read this file before writing or modifying any code.** It is the contract for working in this repository. If any instruction elsewhere conflicts with this file, this file wins. When in doubt, stop and ask rather than guess.

---

## 1. What this project is

A from-scratch, academic implementation of a stabilising control system for an **inverted reaction-wheel pendulum**. The system linearises the plant, estimates state with a **Kalman filter**, controls it with an **LQR** law (together: LQG), and **automatically tunes the LQG weight matrices** $(Q, R, W, V)$ using **Bayesian Optimisation** with a hand-written **Gaussian-Process surrogate** and an **Entropy-Search** acquisition function.

The point of the project is the *derivation and implementation*, not the result. Correctness, transparency, and traceability to theory matter more than speed.

---

## 2. The five golden rules

1. **No library that solves the problem for you.** See §3. You implement the algorithms. NumPy is a calculator, not a solver.
2. **All numerical algorithms live in `numerics/`.** Never inline a Cholesky, a Riccati iteration, an integrator, or a matrix inverse inside a controller, filter, or GP. If you need a primitive that isn't in `numerics/`, add it there.
3. **Every public function cites the equation it implements.** Docstrings reference the source, e.g. `"""Predictive mean, Eq. 2.25 (Rasmussen & Williams)."""`. If there is no equation, cite the doc section in `docs/theory/`.
4. **Respect the dependency rules.** See `docs/architecture/dependency_rules.md`. A lower layer never imports an upper layer. Violations must fail CI.
5. **Determinism is mandatory.** Every stochastic path takes an explicit seeded generator. No bare `np.random.*` calls. A run must be reproducible from its config + seed.

---

## 3. The no-library rule (most common failure mode)

This is the boundary an agent is most likely to cross. Be exact.

**Allowed anywhere:**
- `numpy` for array storage, broadcasting, elementwise ops, and `@` / `matmul`.
- `numpy.random.Generator` (via a seed) for sampling — never the legacy global `np.random.*`.
- Python standard library (`math`, `dataclasses`, `typing`, `pathlib`, `json`, etc.).
- A YAML parser for config loading (`pyyaml`) — config I/O is not "the problem."

**Banned inside `src/`** (these *solve the problem* and defeat the academic goal):
- `numpy.linalg.solve`, `inv`, `cholesky`, `eig`, `eigh`, `lstsq`, `pinv`, `det`, `qr`, `svd`
- `scipy.*` entirely (`scipy.linalg`, `scipy.integrate`, `scipy.optimize`, `scipy.stats`)
- `control`, `python-control`, `slycot`
- `GPy`, `gpytorch`, `botorch`, `sklearn.gaussian_process`
- `filterpy` or any prebuilt Kalman/EKF
- any autodiff framework used to avoid deriving gradients (`torch`, `jax`, `tensorflow`)

**Allowed in `tests/validation/` ONLY:** the banned libraries above, used *exclusively* as ground truth to verify your hand-written implementations match to a stated tolerance. They must never be imported by anything under `src/`.

If a task seems to require a banned import in `src/`, that is a signal to implement the primitive in `numerics/` — not to import the library.

---

## 4. Layer map (where things go)

| Layer | Path | Responsibility |
|---|---|---|
| Numerics | `src/inverted_pendulum/numerics/` | Hand-written linear algebra, Riccati, integrators, optimisers |
| Core | `src/inverted_pendulum/core/` | Domain types & interfaces (no algorithms) |
| Physical | `src/inverted_pendulum/physical/` | Motor, wheel, plant — nonlinear physics |
| Dynamics | `src/inverted_pendulum/dynamics/` | State-space models, linearisation, discretisation |
| Control | `src/inverted_pendulum/control/` | LQR gain & control law |
| Estimation | `src/inverted_pendulum/estimation/` | Kalman / EKF |
| Simulation | `src/inverted_pendulum/simulation/` | Time-stepping orchestration, noise injection |
| Metrics | `src/inverted_pendulum/metrics/` | $M_p$, $T_s$, control effort, trajectory entropy |
| Optimization | `src/inverted_pendulum/optimization/` | GP, kernels, acquisition, BO loop, ML-II |
| Experiment | `src/inverted_pendulum/experiment/` | Config-driven runs, logging, artifacts |
| IO | `src/inverted_pendulum/io/` | Config loading, logging, plotting |

The companion docs:
- `docs/theory/notation.md` — symbol ↔ code names. **Consult before naming any matrix or variable.**
- `docs/architecture/data_contracts.md` — exact shape of every object passed between layers.
- `docs/architecture/dependency_rules.md` — allowed/forbidden imports.
- `docs/conventions/numerical_standards.md` — tolerances, conditioning, convergence.

---

## 5. The two optimisation levels (do not confuse them)

This project has **two nested optimisation loops** with different parameters. Mixing them is a critical error.

| Loop | Optimises | Symbol | Lives in |
|---|---|---|---|
| **Outer** (Bayesian Optimisation) | LQG weight matrices | $\boldsymbol{\theta} = \mathrm{vec}(Q, R, W, V)$ | `optimization/bayes_optimizer.py` + `acquisition/` |
| **Inner** (ML-II / type-II max likelihood) | GP kernel hyperparameters | $\boldsymbol{\phi} = (\ell, \sigma_f, \sigma_n)$ | `optimization/marginal_likelihood.py` |

The inner loop runs *every time* a new observation is added to the GP, before the acquisition function is evaluated.

Likewise, **two different "entropies" exist** and must never be cross-wired:
- `metrics/trajectory_entropy.py` — a *physical* metric of a response trajectory.
- The information-theoretic entropy $H[p(\boldsymbol{\theta}^*\mid\mathcal{D})]$ — lives *only* inside `optimization/acquisition/entropy_search.py`.

---

## 6. Workflow expectations for an agent

- **One responsibility per change.** Do not refactor across layers in a single edit.
- **Write the test with the code.** Every `numerics/` primitive needs a unit test; every primitive that has a reference implementation needs a `tests/validation/` cross-check.
- **No silent magic numbers.** Tolerances and limits come from `docs/conventions/numerical_standards.md` or a config — never hard-coded inline.
- **Prefer explicit types.** Use the dataclasses in `core/types.py`; do not pass raw untyped dicts or tuples between layers.
- **Run the checks before declaring done:** unit tests, validation tests, the import-linter (dependency rules), and the formatter.

---

## 7. Determinism contract

- A simulation is fully determined by `(LQGConfig, seed, plant params, sim params)`.
- All randomness (process noise, measurement noise, GP-sample draws in Entropy Search, initial-state perturbations) flows from a single seeded `numpy.random.Generator` threaded through call signatures.
- `SimulationResult` records the seed that produced it.
- `ExperimentManager` records: git commit hash, full config, seed, metrics, and trajectories for every run.

---

## 8. Definition of done

A change is complete when: it obeys the no-library rule, lives in the correct layer, cites its equations, has unit + (where applicable) validation tests passing, introduces no forbidden import, is deterministic, and uses the canonical names from `notation.md` and the types from `data_contracts.md`.