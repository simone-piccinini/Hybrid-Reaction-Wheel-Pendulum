# The `numerics/` Layer — Mathematics From Scratch

The whole point of this project is to *implement* the algorithms, not to call a
library that hides them. Every numerical routine the control and optimisation
layers need is written by hand in `src/inverted_pendulum/numerics/`, on top of
NumPy used purely as an array calculator (`@`, broadcasting, slicing, a seeded
RNG). This document explains each primitive, the algorithm behind it, and where
it is used.

The boundary is exact ([numerical_standards.md](../conventions/numerical_standards.md) §3):
**banned in `src/`** are `numpy.linalg.{solve, inv, cholesky, eig, eigh, lstsq,
pinv, det, qr, svd}`, all of SciPy, and every prebuilt control/GP/Kalman
library. They are allowed **only** in `tests/validation/`, as the ground truth
the hand-written code is checked against to a stated tolerance.

---

## `linalg.py` — dense linear algebra

### Cholesky factorisation — `cholesky(M)`
The Cholesky–Banachiewicz algorithm (Golub & Van Loan, Alg. 4.2.1): factor a
symmetric positive-definite `M = L Lᵀ` with `L` lower-triangular, one column at
a time. A non-positive pivot means `M` is not positive-definite and raises
`NotPositiveDefiniteError`. An optional diagonal `jitter` supports the
conditioning policy of §4.

**Used by:** every SPD solve — the GP kernel system, the Kalman innovation
covariance, the Riccati gain.

### Triangular solves — `solve_lower`, `solve_upper`
Forward and back substitution (GVL Alg. 3.1.1–3.1.2). They accept a vector or a
matrix right-hand side (columns solved together).

### SPD solve — `chol_solve(M, b)`
`x = Lᵀ \ (L \ b)`: factor once, then one forward and one back substitution.
This is the workhorse for "apply `M⁻¹`" without ever forming an inverse — used
by the GP predictive equations and the Kalman gain.

### General linear solve — `lu_factor`, `lu_solve`, `lu_solve_matrix`
Partial-pivot LU (GVL Alg. 3.4.1) for non-symmetric systems, with the row
permutation tracked so `P M = L U`.

### Eigenvalues — `eigvals(M)`, `spectral_radius(M)`
The practical **shifted-QR algorithm** (GVL §7.5), complex-capable:
1. Householder reduction to upper Hessenberg form (Alg. 7.4.2);
2. Wilkinson-shifted QR iterations in complex arithmetic, with 1×1 and 2×2
   deflation and the LAPACK-style "exceptional shift" that breaks the cycles a
   plain Wilkinson shift stalls on (e.g. a cyclic permutation matrix).

`spectral_radius` wraps `max |λ|`. **Used by:** the LQR closed-loop stability
check (all eigenvalues of `A_d − B_d K` strictly inside the unit circle) and the
open-loop instability sanity check of [model.md](../theory/model.md).

> Why a general eigensolver and not just a symmetric one? The closed-loop
> matrix `A_d − B_d K` is non-symmetric and can have complex eigenvalues, so the
> unit-circle test genuinely needs the general algorithm. This was a conscious
> architectural decision recorded against [model.md](../theory/model.md)'s
> "confirm the closed loop is stable" caution.

---

## `matrix_exp.py` — the matrix exponential `expm(A)`

Scaling-and-squaring with a degree-13 Padé approximant (Higham's method, the
standard behind `scipy.linalg.expm`): scale `A` by a power of two until its norm
is small, evaluate the Padé rational approximant there, then square the result
back up. **Used by:** zero-order-hold discretisation — `expm([[A, B],[0, 0]]·dt)`
has `A_d` and `B_d` in its top blocks (`StateSpaceModel.discretize`).

---

## `integrators.py` — `rk4`, `euler`

Fixed-step explicit ODE integrators (Hairer–Nørsett–Wanner §II.1). `rk4` is the
classical four-stage, fourth-order Runge–Kutta scheme (global error `O(Δt⁴)`);
`euler` is first-order, kept for comparison. **Used by:**
`NonlinearPlantModel.step` to advance the true `sin θ` dynamics under a
constant (ZOH) control across each sampling period.

---

## `riccati.py` — the discrete Riccati solver `solve_dare(A, B, Q, R)`

Solves the **Discrete Algebraic Riccati Equation**

```
P = Q + Aᵀ P A − Aᵀ P B (R + Bᵀ P B)⁻¹ Bᵀ P A
```

by **backward value iteration** (the Kleinman-style sweep of
[lqr.md](../theory/lqr.md)): start from `P ← Q`, iterate the recursion until
`‖P_next − P‖∞` falls below the `ATOL + RTOL·‖P‖∞` gate, then read off the gain
`K = (R + Bᵀ P B)⁻¹ Bᵀ P A`. The inner inverse is applied with `chol_solve`,
never formed explicitly, and `P` is re-symmetrised each sweep to fight
round-off.

**The duality trick** ([kalman.md](../theory/kalman.md)): the Kalman filter's
steady-state covariance solves the *same* equation with `A → Aᵀ, B → Cᵀ,
Q → W, R → V`. So this one routine is called twice — once for the LQR gain, once
for the estimator gain. There is deliberately no second solver.

> **Convergence note.** Value iteration is *linearly* convergent, so at the §5
> step gate the solution matches a reference DARE solver to ~`1e-4…1e-6`
> relative, not to machine precision. The validation tests use a tolerance that
> reflects this honestly. A marginally-damped closed loop (spectral radius near
> 1) converges slowly and may need a raised iteration cap.

---

## `optimizers.py` — `minimize`, `minimize_with_restarts`

A from-scratch **BFGS** quasi-Newton minimiser with an Armijo backtracking line
search (Nocedal & Wright: BFGS update eq. 6.17, initial scaling 6.20, Armijo
condition 3.4), plus a seeded multi-restart driver. It **minimises**, so the
ML-II inner loop hands it the *negative* log marginal likelihood and its negated
gradient — both supplied analytically (no autodiff). **Used by:** GP
hyperparameter fitting (`marginal_likelihood.fit_hyperparameters`).

---

## How the validation tests anchor all of this

For every primitive with a known reference, `tests/validation/` cross-checks the
hand-written result against the banned library to `VALIDATION_RTOL`:

| Primitive | Checked against |
|---|---|
| `cholesky`, solves, LU | `numpy.linalg`, `scipy.linalg.solve_triangular` |
| `eigvals` | `numpy.linalg.eigvals` (one-to-one spectrum pairing) |
| `expm` | `scipy.linalg.expm` |
| `solve_dare` | `scipy.linalg.solve_discrete_are` |
| `rk4` trajectory | `scipy.integrate.solve_ivp` |
| Kalman recursion | `filterpy` |
| GP predict / log-marginal | `scikit-learn` GP with an identical fixed kernel |

A primitive without a passing validation test is, by the project's own
definition, not done.
