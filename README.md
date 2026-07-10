# Hybrid Reaction-Wheel Pendulum

**Automatic LQG tuning by Entropy Search — a from-scratch implementation of
[Marco et al., *"Automatic LQR Tuning Based on Gaussian Process Global
Optimization"*, ICRA 2016](https://arxiv.org/abs/1605.01950), on an inverted
reaction-wheel pendulum.**

An inverted pendulum carries a motor-driven reaction wheel; spinning the wheel
is the *only* way to keep it upright. The controller is classical — a **Kalman
filter** estimates the state, an **LQR** law stabilises it (together: **LQG**)
— but classical design leaves a hard, human problem open: choosing the four
weight matrices `(Q, R, W, V)` that decide how the controller trades regulation
tightness against control effort, and trust in the model against trust in the
sensors.

**The key of this project is closing that loop automatically.** Following the
paper, controller tuning is treated as a black-box optimisation problem: run a
closed-loop *simulation*, score the resulting trajectory with a single cost,
and let **Bayesian optimisation** — a hand-written **Gaussian-process
surrogate** with an **Entropy Search** acquisition — decide which weights to
try next. Entropy Search picks each evaluation to maximise *information about
the location of the optimum*, so good weights are found in tens of rollouts,
not thousands.

```
              ┌──────────────────────────────────────────────────┐
              │              BAYESIAN OPTIMISATION               │
              │      GP surrogate (ML-II) + Entropy Search       │
              └────────▲──────────────────────────┬──────────────┘
       rollout cost    │                          │   next candidate
       y = f(θ)        │                          │   θ = log(Q, R, W, V)   (11-D)
              ┌────────┴──────────────────────────▼──────────────┐
              │              CLOSED-LOOP SIMULATION              │
              │                                                  │
              │    noisy sensors ──▶ Kalman filter ──▶ x̂         │
              │    u = −K x̂ (LQR) ──▶ DC motor ──▶ reaction wheel │
              │                    ──▶ nonlinear pendulum plant  │
              └──────────────────────────────────────────────────┘
```

"Hybrid" is the project's trajectory: everything is developed and tuned in
simulation, but against a plant whose parameters come from the *real* bench
build — CAD inertias, measured masses, the actual motor
([`configs/pendulum_measured.yaml`](configs/pendulum_measured.yaml),
[`docs/hardware/`](docs/hardware/)) — so the tuned controller is meant to
transfer to hardware.

The point of the project is the *derivation and implementation*: correctness,
transparency, and traceability to the theory matter more than raw speed.

---

## The golden rule: no library that solves the problem

Every numerical algorithm is written from scratch in
[`src/inverted_pendulum/numerics/`](src/inverted_pendulum/numerics/) — Cholesky
and triangular solves, LU, a shifted-QR eigensolver, the matrix exponential,
RK4, the discrete Riccati solver, and a BFGS optimiser. NumPy is used as a
calculator (`@`, broadcasting, slicing, seeded RNG) — never as a solver. SciPy,
scikit-learn, `control`, `filterpy`, GPy and friends appear **only** in
[`tests/validation/`](tests/validation/), as ground truth to check the
hand-written code against. See [`AGENTS.md`](AGENTS.md) and
[`docs/conventions/numerical_standards.md`](docs/conventions/numerical_standards.md).

The rule covers the optimisation layer too: the Gaussian process, the kernels,
the ML-II hyperparameter fit, and Entropy Search itself (representer points,
fantasised observations, the change in the optimum's distribution) are all
implemented here, not imported.

---

## How a tuning run works

1. **Plant.** The nonlinear reaction-wheel pendulum is assembled from a YAML
   config (masses, inertias, motor constants) and linearised about the upright
   equilibrium ([`docs/theory/model.md`](docs/theory/model.md)).
2. **Candidate controller.** A candidate `θ` packs the log-diagonals of `Q, R`
   (LQR weights) and `W, V` (Kalman noise covariances) into an 11-D vector.
   Each candidate becomes a concrete LQG: gain `K` via the discrete Riccati
   equation, filter via the dual DARE
   ([`docs/theory/lqr.md`](docs/theory/lqr.md),
   [`docs/theory/kalman.md`](docs/theory/kalman.md)).
3. **Rollout.** The closed loop runs on the *nonlinear* plant with process and
   measurement noise; the trajectory is scored with one scalar cost — ITAE +
   control energy + hinge penalties on overshoot, settling time, and actuator
   saturation ([`configs/default.yaml`](configs/default.yaml) documents every
   term inline).
4. **Learn and propose.** The GP surrogate is refit by marginal likelihood;
   the acquisition — **Entropy Search** by default, Expected Improvement and
   UCB as baselines — proposes the next `θ`
   ([`docs/theory/optimization.md`](docs/theory/optimization.md)).
5. **Repeat** for a fixed budget (default: 8 Latin-hypercube seeds + 20
   acquisition steps), then report the posterior-mean minimiser and write a
   fully reproducible run record — git hash, config, seed, metrics,
   trajectories, plots.

A run is fully determined by `(config, seed)`: re-running the same config
reproduces the same result.

---

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
| `dynamics` | ZOH stepping, linearise-and-discretise, frequency & time response |
| `control` | `LQRController` (gain via the discrete Riccati equation) |
| `estimation` | `KalmanFilter` (predict/update + dual-DARE steady state) |
| `simulation` | `SimulationEngine` — the closed-loop rollout, the optimiser's oracle |
| `metrics` | overshoot, settling time, control effort, trajectory entropy |
| `optimization` | **the heart**: GP surrogate, ML-II, kernels, EI/UCB/**Entropy Search**, the BO loop |
| `experiment` / `io` | config-driven reproducible runs; YAML loading, logging, plotting |

---

## Repository structure

```
├── src/inverted_pendulum/     the package — one directory per layer above
├── configs/                   experiment configurations
│   ├── default.yaml               ← the documented config schema
│   ├── pendulum_sensible.yaml     a realistic bench-scale plant
│   └── pendulum_measured.yaml     the REAL build (CAD + measured parameters)
├── scripts/                   entry points (docs/guides/analysis_scripts.md)
│   ├── run_experiment.py          run one BO tuning experiment end to end
│   ├── two_stage_experiment.py    explore, then warm-started refinement
│   ├── compare_acquisitions.py    Entropy Search vs EI vs UCB benchmark
│   ├── frequency_analysis.py      Bode diagrams of the open-loop plant
│   ├── step_response.py           closed-loop poles & transient metrics
│   └── stability_margins.py       gain/phase margins of the LQG loop
├── docs/
│   ├── theory/                 the derivations each layer implements
│   ├── guides/                 practical walkthroughs (start here)
│   ├── architecture/           data contracts, dependency rules, class diagram
│   ├── conventions/            numerical standards, the no-library rule
│   ├── hardware/               the physical build: BOM, sensors, sizing, wheel
│   └── papers/                 the inspiration paper + this repo's own analyses
└── tests/
    ├── unit/                   properties of every primitive (no reference libs)
    └── validation/             cross-checks vs SciPy/sklearn/filterpy/control
```

Full documentation is indexed in [`docs/README.md`](docs/README.md); new
readers should start with the
[**codebase overview**](docs/guides/codebase_overview.md).

---

## Installation

No build step — just install the dependencies (a virtual environment is
recommended). Runtime needs only NumPy, PyYAML and matplotlib:

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt          # runtime
pip install -r requirements-dev.txt      # + reference libs, to run the tests
```

## Running an experiment

Tests and scripts run with `PYTHONPATH=src` (there is no installed package yet).

```bash
# tune (Q, R, W, V) on the measured build by Entropy Search:
PYTHONPATH=src python scripts/run_experiment.py configs/pendulum_measured.yaml -o results/run01
```

This writes a full, reproducible run record (git hash, config, seed, metrics,
trajectories, convergence and trajectory plots) to `results/run01/`. The
default Entropy-Search budget takes a few minutes; use `--no-plots` or an
Expected-Improvement config for a quick first run. Then analyse the result:

```bash
PYTHONPATH=src python scripts/step_response.py      -o results/transient  # time domain
PYTHONPATH=src python scripts/frequency_analysis.py -o results/bode       # frequency domain
PYTHONPATH=src python scripts/stability_margins.py  configs/pendulum_measured.yaml -o results/margins
PYTHONPATH=src python scripts/compare_acquisitions.py --json results/cmp.json
```

The whole workflow — config → run → outputs, determinism, extending — is
covered in [`docs/guides/running_experiments.md`](docs/guides/running_experiments.md);
every script is explained in
[`docs/guides/analysis_scripts.md`](docs/guides/analysis_scripts.md).

## Tests

```bash
PYTHONPATH=src python -m pytest tests/ -q      # 539 tests
```

- `tests/unit/` — properties of each primitive and component (no reference libs).
- `tests/validation/` — the hand-written numerics cross-checked against SciPy /
  scikit-learn / filterpy / python-control to a stated tolerance; the only
  place the banned libraries may be imported.

---

## Results on the measured build

Tuning the real bench plant
([`configs/pendulum_measured.yaml`](configs/pendulum_measured.yaml)) finds an
LQG that stabilises a 0.05 rad tilt in ~0.3 s at a ~7 V peak on the 12 V rail,
in 28 simulated evaluations. The follow-up robustness study —
[`docs/papers/robustness_lqg_measured.md`](docs/papers/robustness_lqg_measured.md)
— reads the tuned loop's stability margins: **gain-robust (6.85 dB, +120 %
gain tolerance) but delay-fragile (PM 11.6°, ≈ 57 ms)** — the textbook LQG
caveat of Doyle (1978), and the motivation for folding a margin term into the
tuning objective next. An empirical comparison of the three acquisition
functions is in [`docs/guides/experiments.md`](docs/guides/experiments.md).

---

## References

- A. Marco, P. Hennig, J. Bohg, S. Schaal, S. Trimpe, **"Automatic LQR Tuning
  Based on Gaussian Process Global Optimization"**, *IEEE ICRA*, 2016.
  [arXiv:1605.01950](https://arxiv.org/abs/1605.01950) — **the paper this
  project implements and adapts** (LQR → full LQG; robot arm → reaction-wheel
  pendulum; hardware evaluations → seeded simulation rollouts).
- P. Hennig, C. J. Schuler, "Entropy Search for Information-Efficient Global
  Optimization", *JMLR* 13, 2012 — the acquisition function.
- C. E. Rasmussen, C. K. I. Williams, *Gaussian Processes for Machine
  Learning*, MIT Press, 2006 — the GP equations cited throughout the code.
- J. C. Doyle, "Guaranteed Margins for LQG Regulators: None", *IEEE TAC*, 1978
  — why the robustness analysis exists.

See [`docs/papers/README.md`](docs/papers/README.md) for how each reference
maps onto the code.
