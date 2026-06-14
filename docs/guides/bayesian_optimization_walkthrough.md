# Bayesian Optimisation — A Walkthrough of the `optimization/` Layer

This is a reading guide to the layer that tunes the LQG weights: how the pieces
fit together and what each one does, following one outer-loop step from start to
finish. The mathematical specification is [optimization.md](../theory/optimization.md);
this document is the code-side companion to it.

The problem: minimise `J(θ)`, the realised closed-loop cost of the controller
built from weights `θ = vec(Q, R, W, V)`. `J` is **expensive** (each evaluation
is a full simulated rollout), **noisy** (random disturbances per rollout), and
**black-box** (no gradient). Bayesian optimisation is the right tool: model `J`
with a cheap surrogate, and use the surrogate to decide where to spend the next
expensive evaluation.

---

## 1. The pieces

| File | Class / function | Role |
|---|---|---|
| `objective.py` | `ObjectiveFunction` | turns a `SimulationResult` into the scalar cost `y` |
| `kernels/` | `SquaredExponentialARD`, `Matern52ARD` | the GP covariance functions (ARD) |
| `gaussian_process.py` | `GaussianProcess` | the surrogate model of the cost surface |
| `marginal_likelihood.py` | `fit_hyperparameters` | the ML-II inner loop (learns the kernel hyperparameters) |
| `acquisition/` | `ExpectedImprovement`, `UpperConfidenceBound`, `EntropySearch` | decide the next `θ` |
| `bayes_optimizer.py` | `BayesianOptimizer` | the outer loop tying it all together |

---

## 2. The decision vector and its coordinates

`θ` is the four LQG weight matrices, modelled as diagonal and packed in
**log-space** (every weight is positive, so its log is the natural
unconstrained coordinate):

```
θ = [ log diag(Q) (4) | log diag(R) (1) | log diag(W) (4) | log diag(V) (2) ]   # length 11
```

`LQGConfig.to_vector` / `from_vector` are the exact inverse pair, and the
`SearchSpace` operates in these same log coordinates. The optimiser only ever
sees `θ`; converting back to an `LQGConfig` happens at the oracle boundary.

---

## 3. The surrogate — `GaussianProcess`

A Gaussian process models `J` as a distribution over functions. Given the data
seen so far, it returns at any query point a **mean** (best guess of the cost)
and a **variance** (how unsure it is). The implementation is Rasmussen &
Williams' Algorithm 2.1 on the project's own Cholesky:

- `fit(X, y)` standardises the targets, forms `K + (σ_n² + jitter) I`, and
  factorises it **once**.
- `predict` returns the marginal mean/variance (eqs. 2.25–2.26).
- `joint_posterior` / `sample_posterior` give the *full* covariance and joint
  function draws — Entropy Search needs whole sampled functions, not just
  per-point variances.

The kernel is **ARD** (automatic relevance determination): one lengthscale per
dimension, so the model can learn that some weights matter far more than others.

---

## 4. The inner loop — ML-II (`marginal_likelihood.py`)

The kernel itself has hyperparameters `φ = (ℓ₁…ℓ_d, σ_f, σ_n)`. They are not
guessed: they are fit by **type-II maximum likelihood** — maximising the log
marginal likelihood of the data under the model (R&W eq. 5.8), by gradient
ascent in log-space from several random restarts (the objective is non-convex).
The gradient is analytic (eq. 5.9); the project BFGS does the optimisation on
the *negative* log marginal likelihood.

This inner loop runs **every time** a new observation is added, before the
acquisition is evaluated. It is a different optimisation from the outer loop —
different parameters (`φ`, not `θ`), different objective (marginal likelihood,
not closed-loop cost). Conflating the two is the classic error this project's
notation guards against.

---

## 5. The acquisition — where to look next

The acquisition turns the posterior into a choice of the next `θ`. All three
are maximised over a candidate set drawn from the search space (`§4.5`), in
minimisation convention.

- **Expected Improvement** — `E[max(f⁺ − J(θ), 0)]`, the expected amount by
  which a query beats the best cost so far. Greedy and cheap; the baseline.
- **Upper/Lower Confidence Bound** — prefers low predicted mean *or* high
  uncertainty (`β σ − μ`); the tunable explore/exploit baseline.
- **Entropy Search** — the goal. It keeps an explicit belief `p_min(θ)` over the
  **location of the optimum** and picks the query expected to shrink that
  belief's entropy the most:

  ```
  α_ES(θ) = H[p_min] − E_{y_θ}[ H[p_min | y_θ] ]
  ```

  `p_min` is estimated by Monte Carlo (draw joint GP function samples over a set
  of *representer points*, count where each sample's minimum falls); the
  hypothetical outcome `y_θ` is marginalised by a *fantasy update* — exact
  Gaussian conditioning where the posterior covariance change is deterministic
  and only the mean shift carries the random innovation. Crucially, ES can
  value a point that is not itself promising if probing it would most sharply
  collapse the belief about where the optimum is — the right behaviour when each
  rollout is precious.

---

## 6. The outer loop — `BayesianOptimizer.optimize`

Following [optimization.md](../theory/optimization.md) §5, one call does:

```
1. sample n_initial θ from the search space; evaluate the oracle on each
2. fit the GP; refit φ by ML-II
3. repeat n_iterations times:
     θ_next = acquisition.select(gp, space, rng)     # §4.5
     y = oracle(θ_next)                              # engine.run → objective
     append (θ_next, y); refit the GP and φ
4. return argmin_θ μ(θ)                              # the posterior-mean minimiser
```

Two design points:

- **The oracle is a black box.** The optimiser reaches the plant *only* through
  `engine.run(config) → objective.evaluate(result)` — it never builds a
  controller or filter itself (`dependency_rules.md` §4). A design that cannot
  be stabilised comes back as a `diverged` result scored at the finite penalty,
  so the search never crashes on a bad candidate.
- **The answer is the posterior-mean minimiser, not the best observation.**
  Under noise the lowest observed cost may be a lucky draw; the posterior mean
  integrates the evidence.

---

## 7. Using it

Through the experiment layer (recommended — handles config, seeds, recording):

```python
from inverted_pendulum.experiment.manager import ExperimentManager
result = ExperimentManager.from_config_file("configs/default.yaml").run()
```

Or directly, wiring the pieces yourself:

```python
import numpy as np
from inverted_pendulum.optimization.gaussian_process import GaussianProcess
from inverted_pendulum.optimization.kernels.matern import Matern52ARD
from inverted_pendulum.optimization.acquisition.entropy_search import EntropySearch
from inverted_pendulum.optimization.bayes_optimizer import BayesianOptimizer
# ... build engine, objective, search space (see ExperimentManager.build_*) ...

gp = GaussianProcess(Matern52ARD(np.ones(11), 1.0), noise_variance=1e-2)
optimizer = BayesianOptimizer(engine, objective, gp, EntropySearch(), space)
best_config = optimizer.optimize(n_initial=8, n_iterations=20,
                                 rng=np.random.default_rng(0))
```

See the empirical comparison of the three acquisitions in
[experiments.md](experiments.md).
