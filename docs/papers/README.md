# Papers

The literature this project is built on — and the analyses it produced.

## The inspiration

**A. Marco, P. Hennig, J. Bohg, S. Schaal, S. Trimpe —
"Automatic LQR Tuning Based on Gaussian Process Global Optimization",
IEEE ICRA 2016.** [arXiv:1605.01950](https://arxiv.org/abs/1605.01950)

This is the paper the whole project implements and adapts. Its idea: stop
hand-tuning LQR weight matrices; instead treat "run the controller, score the
trajectory" as an expensive black-box function `f(θ)`, model it with a
**Gaussian process**, and choose each next experiment with **Entropy Search** —
the acquisition that maximises information about *where the optimum is*, rather
than greedily sampling where the cost looks low. The authors tune a
seven-DOF robot arm balancing an inverted pole in a handful of hardware
experiments.

This repository transplants that framework, with three deliberate adaptations:

| Marco et al. (2016) | This project |
|---|---|
| LQR weights `(Q, R)` | full **LQG**: `(Q, R)` *and* Kalman covariances `(W, V)` — an 11-D search space |
| seven-DOF arm + inverted pole | inverted **reaction-wheel pendulum** (the real bench build in [`configs/pendulum_measured.yaml`](../../configs/pendulum_measured.yaml)) |
| hardware experiments as evaluations | seeded, reproducible **simulation rollouts** as evaluations |
| GP / ES from research libraries | **everything from scratch** — GP, ML-II, kernels, Entropy Search, Riccati, all of it ([`AGENTS.md`](../../AGENTS.md) §3) |

Where it lives in the code: the loop is
[`optimization/bayes_optimizer.py`](../../src/inverted_pendulum/optimization/bayes_optimizer.py),
the surrogate [`optimization/gaussian_process.py`](../../src/inverted_pendulum/optimization/gaussian_process.py),
the acquisition [`optimization/acquisition/entropy_search.py`](../../src/inverted_pendulum/optimization/acquisition/entropy_search.py),
and the oracle it queries is the closed-loop
[`simulation/simulator.py`](../../src/inverted_pendulum/simulation/simulator.py).
The binding derivation is [`docs/theory/optimization.md`](../theory/optimization.md).

## Supporting references

- **P. Hennig, C. J. Schuler — "Entropy Search for Information-Efficient Global
  Optimization", JMLR 13, 2012.** The acquisition function itself: the
  distribution over the minimiser `p_min`, representer points, and fantasised
  observations. Implemented in
  [`acquisition/entropy_search.py`](../../src/inverted_pendulum/optimization/acquisition/entropy_search.py).
- **C. E. Rasmussen, C. K. I. Williams — *Gaussian Processes for Machine
  Learning*, MIT Press, 2006.** The GP posterior and marginal-likelihood
  equations cited by number throughout
  [`gaussian_process.py`](../../src/inverted_pendulum/optimization/gaussian_process.py)
  and [`marginal_likelihood.py`](../../src/inverted_pendulum/optimization/marginal_likelihood.py).
- **J. C. Doyle — "Guaranteed Margins for LQG Regulators: None", IEEE TAC,
  1978.** LQR and the Kalman filter are each robust; their combination need
  not be. The reason `scripts/stability_margins.py` exists, and the frame for
  the robustness analysis below.

## This repo's own analyses

- [**Robustness of the Tuned LQG on the Measured Build**](robustness_lqg_measured.md)
  — stability margins of the Entropy-Search-tuned controller on the real
  plant: gain-robust (6.85 dB, +120 %) but delay-fragile (PM 11.6°, ≈ 57 ms),
  and what that means for the firmware.
- [Acquisition-function comparison](../guides/experiments.md) — Entropy Search
  vs Expected Improvement vs UCB on the same tuning problem, multi-seed.
