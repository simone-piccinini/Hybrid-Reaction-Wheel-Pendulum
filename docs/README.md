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
- [Frequency Analysis](guides/frequency_analysis.md) — from the state-space form
  to the Bode diagram via the Laplace transfer function `G(s)=C(sI−A)⁻¹B+D`,
  with the derivation and the reading of the plant's poles; plus the **open-loop
  LQG loop gain and stability margins** (phase/gain margin, Doyle's caveat).
- [Time-Domain Response](guides/time_domain_response.md) — step response,
  initial-condition (regulation) transient via `x(t)=e^{At}x₀`, the standard
  transient metrics, and modal analysis (poles → `ω_n`, `ζ`), linear vs nonlinear.
- [Experiments](guides/experiments.md) — an empirical comparison of the three
  acquisition functions, reproducible from `scripts/compare_acquisitions.py`.
- [The Analysis Scripts](guides/analysis_scripts.md) — a tour of every entry
  point in `scripts/` (including `swing_up.py`), with a deep dive on
  **`stability_margins.py`**: why the open-loop LQG gain (phase/gain margin,
  Doyle's caveat) predicts how the controller survives real-hardware latency
  and gain drift — the script that tells you the truth a clean simulation
  hides, and that motivated the objective's margin penalty.

## Hardware

- [Bill of Materials](hardware/bill_of_materials.md) — the complete parts list to
  physically build the pendulum (motor, driver, sensors, power, mechanics),
  alternatives within a €100–150 budget, and the ESP32 ↔ SimpleFOCMini ↔ AS5600
  wiring/pinout.
- [Sensors, the I²C bus, and the magnets](hardware/sensors_and_i2c.md) — why two
  AS5600 are needed, how I²C works and the fixed-address clash (and its fixes),
  and what the diametric magnets are for.
- [Sizing the pendulum](hardware/sizing_the_pendulum.md) — the engineering
  reasoning and the math behind how long the arm can be for a given motor
  (torque vs gravity, wheel saturation, margins) — the method an engineer uses
  for any robot / drone / pendulum.
- [Sensor characterisation](hardware/sensor_characterisation.md) — the measured
  Stage 0 lab record for the real machine: noise floors, the encoder linearity
  fault the noise test missed, the out-of-plane wobble, and what the measured
  numbers do to the achievable controller.
- [**Bring-up pipeline**](hardware/bringup_pipeline.md) — the staged path from a
  built machine to a balancing controller: measure `b_p`, `K_t`, `I_b` and the
  sensor noise on hardware (stages 0-3), feed them back into the config, and
  only then attempt balancing. Firmware: `firmware/swing_test_v1/`.
- [v6 firmware handoff](hardware/firmware_v6_handoff.md) — the as-flashed
  firmware and the plant it drives, imported verbatim, with the two conclusions
  this repo has since revised flagged at the top.
- [Thermal limits and the duty ceiling](hardware/thermal_and_duty_limits.md) — why
  the firmware's duty cap, not the wheel, is the binding constraint: the "too hot
  to touch" event was a wiring fault (19x the power of legitimate duty 237),
  balancing costs 0.118 W mean, and raising the *peak* ceiling buys 4.6x the
  catch angle for 0.3 K of extra heating.
- [Designing the reaction wheel](hardware/reaction_wheel_design.md) — sizing the
  wheel from its moment of inertia `I_w` (momentum capacity `H_max = I_w·ω_max`),
  the "mass at the rim" rule, recommended dimensions, the concentric outrunner
  mounting, and modelling it in Onshape.

## Papers & analyses

- [**Papers index**](papers/README.md) — the literature this project is built
  on, starting with the inspiration:
  [Marco et al., *Automatic LQR Tuning Based on Gaussian Process Global
  Optimization*, ICRA 2016 (arXiv:1605.01950)](https://arxiv.org/abs/1605.01950)
  — and how each reference maps onto the code.
- [Robustness of the Tuned LQG on the Measured Build](papers/robustness_lqg_measured.md)
  — a stability-margin study of the BO-tuned controller for the real,
  CAD/measured pendulum: gain-robust (6.85 dB, +120 %) but delay-fragile (11.6°,
  ~57 ms), with Doyle's caveat and what it means for the firmware.

## Theory (the derivations each layer implements)

- [notation.md](theory/notation.md) — the symbol ↔ code glossary. **Consult
  before naming anything.**
- [model.md](theory/model.md) — the reaction-wheel-pendulum plant and its
  linearisation.
- [lqr.md](theory/lqr.md) — the LQR controller via the discrete Riccati equation.
- [kalman.md](theory/kalman.md) — the Kalman filter (and the Riccati duality).
- [swingup.md](theory/swingup.md) — energy-shaping swing-up from hanging and the
  hand-off to the balancing LQG (the one nonlinear, global controller), with the
  torque-vs-friction feasibility condition.
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
