# Inversed Wheeled pendulum

This project is dividen in the following steps:

1) model description and physical world
    - deriving lagrangian equations



## Repository structure

```text
inverted-pendulum/
│
├── README.md
├── AGENTS.md                       # ★ agent guardrails (read-first)
├── pyproject.toml                  # single source of build/deps
├── requirements.txt                # pinned RUNTIME deps  → numpy only
├── requirements-dev.txt            # scipy, control, matplotlib
├── .gitignore
│
├── docs/
│   ├── architecture/
│   │   ├── system_design.md
│   │   ├── class_diagram.md         # mermaid SOURCE (diffable) + exported .pdf
│   │   ├── control_pipeline.md
│   │   ├── dependency_rules.md      # ★ allowed/forbidden edges
│   │   └── data_contracts.md        # ★ LQGConfig, SimulationResult, GPPosterior…
│   │
│   ├── theory/
│   │   ├── notation.md              # ★ symbol ↔ code glossary
│   │   ├── lagrangian_model.md
│   │   ├── linearization.md
│   │   ├── lqr_riccati.md
│   │   ├── kalman_filter.md
│   │   ├── gaussian_process.md
│   │   ├── bayesian_optimization.md
│   │   └── entropy_search.md        # the ES acquisition (information-theoretic)
│   │
│   ├── conventions/
│   │   ├── coding_standards.md
│   │   ├── numerical_standards.md   # ★ tolerances, conditioning, no-library rule
│   │   └── reproducibility.md       # seeds, config, git hash
│   │
│   └── papers/
│       └── annotated/
│
├── src/inverted_pendulum/           # importable package (src-layout)
│   ├── core/                        # shared domain types & interfaces
│   │   ├── types.py                 # LQGConfig, SimulationResult, Dataset, GPPosterior
│   │   ├── search_space.py          # bounds, log-scale
│   │   └── interfaces.py            # abstract base classes
│   │
│   ├── numerics/                    # ★ FROM-SCRATCH math primitives
│   │   ├── linalg.py                # cholesky, triangular_solve, solve, inverse
│   │   ├── riccati.py               # CARE / DARE solver
│   │   ├── matrix_exp.py            # discretization (expm)
│   │   ├── integrators.py           # rk4, euler
│   │   └── optimizers.py            # gradient descent / L-BFGS-style for ML-II
│   │
│   ├── physical/                    # the plant (matches class diagram)
│   │   ├── dc_motor.py
│   │   ├── reaction_wheel.py
│   │   └── reaction_wheel_pendulum.py
│   │
│   ├── dynamics/
│   │   ├── state_space.py
│   │   ├── nonlinear_model.py
│   │   └── linearized_model.py
│   │
│   ├── control/
│   │   ├── base.py                  # controller interface
│   │   ├── lqr_controller.py
│   │   └── cost_matrices.py
│   │
│   ├── estimation/
│   │   ├── base.py                  # filter interface
│   │   ├── kalman_filter.py
│   │   ├── ekf.py                   # extension
│   │   └── noise_models.py
│   │
│   ├── optimization/
│   │   ├── bayes_optimizer.py
│   │   ├── gaussian_process.py
│   │   ├── marginal_likelihood.py   # ★ ML-II objective + gradients
│   │   ├── kernels/                 # ★ swappable kernels
│   │   │   ├── base.py
│   │   │   ├── squared_exponential.py
│   │   │   └── matern.py
│   │   ├── acquisition/             # ★ swappable strategies
│   │   │   ├── base.py
│   │   │   ├── entropy_search.py    # the project goal
│   │   │   ├── expected_improvement.py
│   │   │   └── ucb.py
│   │   └── objective.py             # Mp/Ts/effort → scalar cost
│   │
│   ├── simulation/
│   │   ├── simulator.py
│   │   ├── disturbances.py
│   │   └── environment.py
│   │
│   ├── metrics/
│   │   ├── stability_metrics.py     # overshoot Mp, settling time Ts
│   │   ├── performance_metrics.py   # control effort, oscillation energy
│   │   └── trajectory_entropy.py    # ← renamed (NOT the ES entropy)
│   │
│   ├── experiment/
│   │   └── manager.py               # ExperimentManager
│   │
│   └── io/                          # replaces vague utils/
│       ├── config_loader.py
│       ├── logging.py
│       └── plotting.py
│
├── configs/
│   ├── default.yaml
│   ├── schema.md                    # documents every config field
│   ├── plant/
│   ├── lqr/
│   ├── kalman/
│   ├── optimization/
│   └── experiments/
│
├── experiments/
│   ├── exp001_baseline/
│   │   ├── config.yaml
│   │   └── results/                 # gitignored run outputs
│   └── exp002_entropy_search/
│       ├── config.yaml
│       └── results/
│
├── notebooks/                       # promoted out of experiments/
│
├── results/                         # gitignored: global artifacts 
│
├── tests/
│   ├── unit/                        # mirrors src/ one-to-one
│   ├── validation/                  # ★ cross-check vs scipy/contro
│   │   ├── test_riccati_vs_scipy.py
│   │   ├── test_cholesky_reconstruction.py
│   │   └── test_kalman_vs_reference.py
│   └── conftest.py
│
└── scripts/
    ├── run_simulation.py
    ├── train_optimizer.py
    └── benchmark.py

```