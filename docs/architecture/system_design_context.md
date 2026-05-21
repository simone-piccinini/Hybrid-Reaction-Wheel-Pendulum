# System Design Context

## Project Goal

The goal of this project is to design, simulate, optimize, and eventually deploy a control system for an inverted reaction wheel pendulum.

The system combines:

- nonlinear rigid body dynamics
- DC motor actuation
- state estimation through Kalman filtering
- optimal control using LQR
- Bayesian optimization for automatic hyperparameter tuning

The final objective is to obtain a robust and physically deployable stabilization system capable of balancing around the unstable upright equilibrium while minimizing uncertainty, oscillations, and control effort.

---

# Why This Architecture Exists

The architecture is intentionally modular.

Control systems rapidly become unmaintainable when:
- dynamics are mixed with optimization logic
- estimators contain simulation code
- controllers directly manipulate experiments
- metrics are scattered throughout the project

This design separates the project into independent layers with explicit responsibilities.

The architecture is designed to achieve:

- mathematical clarity
- reproducibility
- scalability
- hardware portability
- deterministic experimentation
- safe collaboration with AI coding agents

---

# High-Level System Pipeline

The system follows the following conceptual pipeline:

```text
Physical System
    ↓
Dynamics Model
    ↓
Simulation Engine
    ↓
State Estimation (Kalman Filter)
    ↓
Control Law (LQR)
    ↓
Optimization Loop (Bayesian Optimization)
    ↓
Metrics & Evaluation
```

Each layer is isolated and has a single responsibility.

---

# Core Design Philosophy

The project follows the following engineering principles:

## 1. Separation of Concerns

Each module handles exactly one conceptual responsibility.

Examples:
- dynamics/ only models physics
- control/ only computes control actions
- estimation/ only estimates state
- optimization/ only searches hyperparameters

This prevents coupling and architectural drift.

---

## 2. Deterministic Simulations

Every experiment must be reproducible.

All stochastic elements must:
- expose seeds
- expose covariance matrices
- be logged

Optimization results must always be reproducible from saved configurations.

---

## 3. Mathematical Transparency

The architecture reflects the mathematical structure of the system.

The software structure mirrors:
- physical dynamics
- estimation theory
- optimal control theory
- probabilistic optimization

This makes the codebase easier to validate against theoretical derivations.

---

## 4. Hardware Portability

The architecture separates:
- physical system models
from
- simulation orchestration

This allows future migration from:
- simulated pendulum
to
- real embedded hardware

without rewriting control or optimization logic.

---

# Class Diagram Motivation

The class diagram models the real physical and computational hierarchy of the system.

---

# Physical Layer

## DCMotor

Represents the electromechanical actuator.

Responsibilities:
- electrical dynamics
- torque generation
- back EMF modeling
- motor constraints

The motor is modeled independently because:
- it may later be replaced
- actuator dynamics affect control quality
- motor limitations influence optimization

---

## ReactionWheel

Represents the inertial wheel attached to the pendulum.

Responsibilities:
- angular momentum
- rotational inertia
- torque application

The wheel is isolated because:
- inertia may change during experimentation
- different wheel geometries may be tested
- wheel dynamics affect stabilization performance

---

## ReactionWheelPendulum

Represents the complete physical plant.

Contains:
- pendulum body
- reaction wheel
- DC motor

Responsibilities:
- nonlinear dynamics
- state evolution
- system linearization
- state-space generation

This is the core physical model of the project.

The class aggregates:
- motor
- wheel

because these components physically belong to the same system.

---

# Dynamics Layer

## StateSpaceModel

Represents the linearized system:

```math
ẋ = Ax + Bu
y = Cx + Du
```

Responsibilities:
- storing system matrices
- discretization
- linear control compatibility

This layer exists because:
- controllers and estimators require linear models
- linearization must remain independent from controllers

---

# Estimation Layer

## KalmanFilter

Responsible for probabilistic state estimation.

Responsibilities:
- prediction step
- correction step
- covariance propagation
- noisy measurement handling

This class is isolated because:
- estimation logic must remain independent from control
- different filters may later replace it (EKF, UKF, Particle Filter)

The Kalman filter operates on the state-space model rather than directly on nonlinear dynamics.

---

# Control Layer

## LQRController

Responsible for optimal feedback control.

Responsibilities:
- Riccati equation solution
- feedback gain computation
- control action generation

The controller:
- consumes estimated state
- outputs control torque/voltage

This separation allows:
- future MPC implementation
- controller benchmarking
- multiple control strategies

without modifying the plant model.

---

# Simulation Layer

## SimulationEngine

The orchestrator of the complete system.

Responsibilities:
- timestep execution
- synchronization
- numerical integration
- noise injection
- trajectory propagation

The engine coordinates:
- plant
- estimator
- controller

without embedding their logic.

This is critical for:
- modularity
- reproducibility
- experimentation

---

# Optimization Layer

## BayesianOptimizer

Responsible for automatic hyperparameter tuning.

Optimizes:
- LQR Q/R matrices
- Kalman covariance matrices
- potentially simulation parameters

Uses:
- Gaussian Processes
- acquisition functions
- iterative experiment evaluation

The optimizer never directly modifies system internals.

Instead:
- it proposes parameters
- runs experiments
- evaluates metrics

This preserves modularity.

---

## GaussianProcess

Models the probabilistic surrogate function:

```math
f(x) ~ GP(μ(x), k(x, x'))
```

Responsibilities:
- uncertainty modeling
- posterior updates
- mean/variance prediction

This layer is isolated because:
- surrogate models may later change
- kernels may evolve independently

---

## AcquisitionFunction

Responsible for exploration/exploitation tradeoff.

Examples:
- Expected Improvement
- UCB
- Probability of Improvement

This abstraction allows:
- interchangeable acquisition strategies
- optimizer experimentation

without changing optimization infrastructure.

---

# Metrics Layer

## ObjectiveFunction

Responsible for evaluating experiment quality.

Metrics may include:
- entropy minimization
- stabilization time
- overshoot
- control effort
- oscillation energy

The optimization system depends entirely on this class.

This separation allows:
- changing objectives
- multi-objective optimization
- benchmarking

without touching optimization algorithms.

---

# Experiment Management

## ExperimentManager

Responsible for:
- configuration loading
- result logging
- artifact saving
- metadata tracking

This layer guarantees:
- reproducibility
- experiment traceability
- scientific rigor

Every experiment should store:
- git commit hash
- configuration
- seeds
- metrics
- trajectories

---

# Architectural Constraints

The following dependency rules must always be respected:

Allowed:

```text
control → dynamics
estimation → dynamics
optimization → simulation
simulation → control
simulation → estimation
```

Forbidden:

```text
dynamics → optimization
control → experiments
metrics → visualization
optimization → hardware drivers
```

---

# Long-Term Vision

This architecture is intentionally designed to support future extensions:

Future possibilities include:
- nonlinear MPC
- reinforcement learning controllers
- hardware-in-the-loop simulation
- real-time embedded deployment
- adaptive filtering
- multi-objective Bayesian optimization
- automatic model identification

The architecture must therefore remain:
- modular
- extensible
- mathematically consistent

at every stage of development.

---

# Final Principle

The architecture exists to preserve a direct mapping between:

- physical system
- mathematical model
- software implementation

A developer should always be able to trace:
- a control law
- a covariance matrix
- a dynamic equation
- an optimization objective

from theory to implementation without ambiguity.