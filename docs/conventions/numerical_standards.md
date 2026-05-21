# Numerical Standards

> Tolerances, conditioning practices, convergence criteria, and the precise boundary of the no-library rule. These exist so that an agent never invents its own thresholds inline. Every magic number related to numerical behaviour comes from this file or a config — never hard-coded ad hoc.

---

## 1. Precision & types

- All real arrays are `float64`. No `float32` anywhere in `src/`.
- Dimensions (`n_x`, `d`, `n`) are Python `int`.
- Prefer `@` (matmul) and explicit shapes; assert shapes at function entry in `numerics/` and at layer boundaries.
- Never compare floats with `==`. Use `abs(a - b) <= ATOL + RTOL * abs(b)` with the tolerances below.

---

## 2. Global tolerances

| Constant | Value | Use |
|---|---|---|
| `ATOL` | `1e-9` | absolute tolerance, general comparisons |
| `RTOL` | `1e-6` | relative tolerance, general comparisons |
| `SYM_TOL` | `1e-8` | max allowed asymmetry `‖M − Mᵀ‖∞` before symmetrising |
| `PSD_JITTER` | `1e-9` | baseline diagonal jitter added before Cholesky |
| `VALIDATION_RTOL` | `1e-6` | tolerance when checking against reference libraries in `tests/validation/` |

These live as named constants in `numerics/` (or a `core/constants.py`), not scattered as literals.

---

## 3. The no-library rule (authoritative boundary)

Mirrors `AGENTS.md §3`; this section is the technical reference.

**You must implement, in `numerics/`:**

| Primitive | File | Notes |
|---|---|---|
| Cholesky factorisation | `linalg.py` | lower-triangular `L`, `L Lᵀ = M`; raises if not PD |
| Triangular solves (fwd/back) | `linalg.py` | `solve_lower`, `solve_upper` |
| SPD solve via Cholesky | `linalg.py` | `chol_solve(M, b)` = `Lᵀ\(L\b)` — used by GP & Kalman |
| General linear solve (LU) | `linalg.py` | partial-pivot LU for non-SPD systems |
| Matrix inverse | `linalg.py` | only where unavoidable; prefer solves |
| Eigen-decomposition (symmetric) | `linalg.py` | QR/Jacobi iteration; for conditioning checks |
| Continuous Riccati (CARE) | `riccati.py` | for LQR gain — iterative (Kleinman) or Hamiltonian-Schur |
| Discrete Riccati (DARE) | `riccati.py` | if controller is discretised |
| Matrix exponential | `matrix_exp.py` | scaling-and-squaring + Padé; for discretisation |
| ODE integrators | `integrators.py` | `rk4`, `euler` |
| Gradient optimiser | `optimizers.py` | for ML-II; gradient descent or L-BFGS-style line search |

**Banned in `src/`:** `numpy.linalg.{solve,inv,cholesky,eig,eigh,lstsq,pinv,det,qr,svd}`, all of `scipy`, `control`, `GPy`, `gpytorch`, `botorch`, `filterpy`, `sklearn`, `torch`, `jax`. Allowed in `tests/validation/` only, as ground truth.

`numpy` element-wise ops, broadcasting, `@`, `.T`, slicing, and seeded `Generator` random draws are always fine — they are array bookkeeping, not problem-solving.

---

## 4. Conditioning & SPD handling

The recurring numerical risk is forming and factoring covariance / kernel matrices that are near-singular.

- **Always symmetrise** a matrix expected to be symmetric before factoring: `M = 0.5 * (M + M.T)` if `‖M − Mᵀ‖∞ > SYM_TOL`.
- **Always add jitter before Cholesky** on kernel matrices: factor `K + (σ_n² + PSD_JITTER) I`, never `K` alone. The `σ_n²` term is the model noise; `PSD_JITTER` guards round-off. (This is exactly why Algorithm 2.1 factors `K + σ²ₙI`.)
- If Cholesky still fails, escalate jitter geometrically (`×10`) up to `1e-3`, log a warning, and record it in the result. If it exceeds `1e-3`, raise — the model is misspecified.
- **Clip negative predictive variance** from round-off: `variance = max(variance, 0.0)`. A large negative value (`< -ATOL`) is a bug, not round-off — raise.
- Track condition numbers of `A`, `K+σ²I` in debug logs; warn above `1e12`.

---

## 5. Convergence criteria

| Iteration | Converged when | Max iters | On failure |
|---|---|---|---|
| Riccati (Kleinman) | `‖P_{k+1} − P_k‖∞ < ATOL + RTOL·‖P_k‖∞` | `1000` | raise `RiccatiNotConverged` |
| Matrix exp (squaring) | Padé order + squaring count fixed by norm | — | n/a (direct) |
| ML-II hyperparameter opt | `‖∇ log p‖∞ < 1e-4` **or** step `< 1e-8` | `200` | return best-so-far, log warning |
| Acquisition inner-opt | candidate set exhausted or no improvement `> RTOL` | configured | return best candidate |

ML-II is non-convex: run from **multiple random restarts** (default `5`, configurable), keep the best log-marginal-likelihood. Optimise hyperparameters in **log-space** (lengthscales and variances are positive).

---

## 6. Determinism rules

- A single `numpy.random.Generator`, seeded from the experiment config, is threaded through every stochastic call: process/measurement noise, initial-state perturbation, `SearchSpace.sample`, and the function draws inside Entropy Search.
- No function under `src/` calls the legacy global `np.random.*`. Pass `rng` explicitly.
- `SimulationResult.seed` records the seed; the run must be byte-reproducible from `(config, seed)`.
- Floating-point summation order matters for reproducibility: do not parallelise reductions in a way that changes ordering without recording it.

---

## 7. Cost-surface hygiene (protecting the GP)

The GP is poisoned by non-finite or wildly-scaled targets. Therefore:

- `ObjectiveFunction.evaluate` must return a **finite** float. Map a diverged run (`SimulationResult.diverged`) to a large finite penalty (e.g. `PENALTY = 1e3`, configurable) — never `inf` or `nan`.
- Standardise targets `y` to zero mean / unit variance inside the GP before fitting; store the transform so predictions can be mapped back.
- Reject `nan`/`inf` in `Dataset.append` with an explicit error.

---

## 8. Validation policy

For every `numerics/` primitive with a known reference, add a `tests/validation/` test comparing against the banned library to `VALIDATION_RTOL`:

- `cholesky` → reconstruct `L @ L.T ≈ M`, and compare `L` to `numpy.linalg.cholesky`.
- `riccati` (CARE) → compare gain `K` to `scipy.linalg.solve_continuous_are`-derived gain.
- `matrix_exp` → compare to `scipy.linalg.expm`.
- `kalman_filter` → compare a short run to a `filterpy` reference.
- `gaussian_process` predict → compare mean/variance to a `GPy`/`sklearn` GP with an identical kernel.

These tests import the reference libs; nothing in `src/` ever may. A primitive without a passing validation test is not "done."

---

## 9. Summary for an agent

When writing numerical code: use `float64`; pull tolerances from §2; never call a banned solver (§3) — implement in `numerics/`; symmetrise + jitter before Cholesky (§4); honour the convergence limits (§5); thread the seeded `rng` (§6); keep the cost surface finite (§7); and add a validation cross-check (§8).
