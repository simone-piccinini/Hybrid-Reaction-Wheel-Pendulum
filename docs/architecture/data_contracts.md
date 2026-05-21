# Data Contracts — Domain Types

> The exact shape of every object passed between layers. An agent must use these types rather than untyped dicts, tuples, or bare arrays. All types live in `src/inverted_pendulum/core/`. Shapes use the dimension symbols from `notation.md` ($n_x, n_u, n_y, d, n$). All arrays are `float64` unless stated.

Implement value objects as frozen `@dataclass` where possible (immutability supports the determinism contract). Interfaces are abstract base classes (`abc.ABC`).

---

## 1. `LQGConfig` — the decision vector $\boldsymbol{\theta}$

The single source of truth for the four LQG weight matrices and their packing into / out of the flat vector the GP sees.

| Field | Type | Shape | Invariant |
|---|---|---|---|
| `Q_lqr` | array | $n_x \times n_x$ | symmetric PSD |
| `R_lqr` | array | $n_u \times n_u$ | symmetric PD |
| `W_process` | array | $n_x \times n_x$ | symmetric PSD |
| `V_measure` | array | $n_y \times n_y$ | symmetric PD |

**Methods**
- `to_vector() -> array[d]` — flatten the free parameters into $\boldsymbol{\theta}$. By default the diagonals (and any modelled off-diagonals) of the four matrices, in a fixed, documented order. Positive entries are stored in **log-space** (see `SearchSpace`).
- `from_vector(theta: array[d]) -> LQGConfig` — inverse of `to_vector`; rebuilds valid symmetric matrices.

**Contract:** `from_vector(c.to_vector()) == c` (round-trip). The packing order is fixed once and documented in `core/types.py`; changing it is a breaking change.

---

## 2. `SearchSpace` — bounds for the BO outer loop

Defines the domain over which $\boldsymbol{\theta}$ is optimised. Bayesian optimisation is undefined without it.

| Field | Type | Shape | Meaning |
|---|---|---|---|
| `dimension` | int | — | $d$ |
| `lower_bounds` | array | $d$ | per-dimension lower limit |
| `upper_bounds` | array | $d$ | per-dimension upper limit |
| `log_scale` | array[bool] | $d$ | whether the dimension is optimised in log-space |

**Methods**
- `sample(n, rng) -> array[n, d]` — draw `n` points (Latin-hypercube or uniform) using the seeded generator `rng`.
- `clip(theta) -> array[d]` — project a point back into the box.

**Contract:** every $\boldsymbol{\theta}$ entering `GaussianProcess` or `AcquisitionFunction` lies within these bounds. Positive-only quantities (variances, cost weights) use `log_scale = True`.

---

## 3. `SimulationResult` — output of one rollout

Produced by `SimulationEngine.run`; consumed by `ObjectiveFunction` and `metrics/`. Never returned as a bare tuple.

| Field | Type | Shape | Meaning |
|---|---|---|---|
| `time` | array | $T$ | time stamps |
| `true_states` | array | $T \times n_x$ | ground-truth state trajectory |
| `estimated_states` | array | $T \times n_x$ | Kalman estimate trajectory |
| `controls` | array | $T \times n_u$ | applied control history |
| `measurements` | array | $T \times n_y$ | noisy measurements |
| `seed` | int | — | RNG seed that produced this run |
| `diverged` | bool | — | `True` if the pendulum fell / state blew up |
| `config` | `LQGConfig` | — | the configuration that produced this run |

**Contract:** if `diverged is True`, the `ObjectiveFunction` assigns a large finite penalty cost (never `inf`/`nan`, which would poison the GP). Divergence detection threshold lives in `numerical_standards.md`.

---

## 4. `Dataset` — the BO observation history $\mathcal{D}_n$

Owned by `GaussianProcess`; visible (read-only) to `BayesianOptimizer`.

| Field | Type | Shape |
|---|---|---|
| `X` | array | $n \times d$ |
| `y` | array | $n$ |

**Methods**
- `append(theta: array[d], y: float) -> None`
- `size() -> int`

**Contract:** `X` rows are points in `SearchSpace` coordinates (log-space where applicable), consistent with what the kernel consumes.

---

## 5. `GPPosterior` — output of `GaussianProcess.predict`

The predictive distribution at one or more query points. Returning *both* moments is mandatory — every acquisition function needs them.

| Field | Type | Shape |
|---|---|---|
| `mean` | array | $m$ (one per query point) |
| `variance` | array | $m$ |

**Contract:** `variance >= 0` elementwise (clip tiny negative values from round-off to 0; see `numerical_standards.md`). For a single query, `m = 1`.

---

## 6. `StateSpaceModel` — linear plant

Produced by `ReactionWheelPendulum.linearize`; consumed by `LQRController` and `KalmanFilter`.

| Field | Type | Shape | Meaning |
|---|---|---|---|
| `A` | array | $n_x \times n_x$ | state matrix |
| `B` | array | $n_x \times n_u$ | input matrix |
| `C` | array | $n_y \times n_x$ | output matrix |
| `D` | array | $n_y \times n_u$ | feedthrough |
| `is_discrete` | bool | — | continuous vs discrete |
| `dt` | float \| None | — | step (set iff discrete) |

**Methods**
- `discretize(dt) -> StateSpaceModel` — returns a **new** discrete model (`A_d`, `B_d`) via the matrix exponential in `numerics/matrix_exp.py`. Does not mutate `self`.

**Contract:** controllers/filters must check `is_discrete` and use the matrix variant matching their formulation.

---

## 7. `Kernel` — interface (abstract)

Lives in `optimization/kernels/base.py`. Implementations: `SquaredExponentialARD`, `Matern52ARD`.

| Member | Type | Meaning |
|---|---|---|
| `lengthscales` | array[$d$] | ARD: one length-scale per dimension |
| `signal_variance` | float | $\sigma_f^2$ |
| `covariance(x1, x2) -> float` | method | $k(x_1, x_2)$ |
| `matrix(X1, X2) -> array` | method | full Gram matrix $k(X_1, X_2)$ |
| `gradient(x1, x2) -> array` | method | $\partial k / \partial \boldsymbol{\phi}$ (for ML-II) |

**Contract:** `matrix(X, X)` is symmetric PSD. The kernel exposes its hyperparameters as a flat vector for the ML-II optimiser.

---

## 8. `AcquisitionFunction` — interface (abstract)

Lives in `optimization/acquisition/base.py`. Implementations: `EntropySearch`, `ExpectedImprovement`, `UpperConfidenceBound`.

| Member | Type | Meaning |
|---|---|---|
| `select(gp, space, rng) -> array[d]` | method | returns the next $\boldsymbol{\theta}$ to evaluate |

**Contract — critical:** `select` receives the **whole `GaussianProcess`**, not `(mean, variance)`. Entropy Search requires the full posterior to build $p(\boldsymbol{\theta}^*\mid\mathcal{D})$ by sampling functions. EI/UCB call `gp.predict` internally over candidates drawn from `space.sample(rng)`. The `rng` makes candidate sampling and GP draws reproducible.

---

## 9. Hand-off summary

```
ReactionWheelPendulum.linearize() ──▶ StateSpaceModel
StateSpaceModel ──▶ LQRController, KalmanFilter
LQGConfig ──▶ SimulationEngine.run() ──▶ SimulationResult
SimulationResult ──▶ ObjectiveFunction.evaluate() ──▶ y (float)
(theta, y) ──▶ Dataset.append() ──▶ GaussianProcess.fit()
GaussianProcess ──▶ AcquisitionFunction.select() ──▶ next theta
next theta ──▶ LQGConfig.from_vector() ──▶ (loop)
```

Every arrow is one of the typed objects above. No layer passes a raw dict or unlabelled tuple across a boundary.