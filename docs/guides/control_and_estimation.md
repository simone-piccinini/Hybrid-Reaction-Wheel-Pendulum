# Control and Estimation — LQG as Implemented

This document explains the two halves of the LQG controller as they appear in
the code: the **LQR** law in `control/lqr_controller.py` and the **Kalman
filter** in `estimation/kalman_filter.py`. Both consume the discretised plant
from `dynamics/` and both rest on the single discrete Riccati solver in
`numerics/riccati.py`. Theory references: [lqr.md](../theory/lqr.md),
[kalman.md](../theory/kalman.md), [model.md](../theory/model.md).

---

## 1. The plant the controllers see

The controllers never touch the nonlinear plant directly. They consume a
**discrete** `StateSpaceModel` produced by the dynamics pipeline:

```
ReactionWheelPendulum.linearize()  →  continuous (A, B, C, D)
   .discretize(dt)                 →  discrete  (A_d, B_d, C, D)   (zero-order hold)
```

The state is `x = [θ_p, θ̇_p, θ_w, θ̇_w]` (`n_x = 4`); the input is the motor
voltage (`n_u = 1`); the sensors are `[θ_p, θ̇_w]` (`n_y = 2`). The wheel angle
`θ_w` does not enter the dynamics (column 3 of `A` is zero) — it is carried for
tracking only. `LinearizedPlantModel` keeps the continuous and discrete models
together so the two are never accidentally mixed.

---

## 2. LQR — `control/lqr_controller.py`

### What it computes
The constant state-feedback gain `K` that minimises the infinite-horizon
quadratic cost `Σ (xᵀ Q x + uᵀ R u)`. By dynamic programming the optimal cost-to-go
is quadratic, `½ xᵀ P x`, with `P` the stabilising solution of the **discrete
Riccati equation**, and the gain is read straight off it:

```
K = (R + B_dᵀ P B_d)⁻¹ B_dᵀ P A_d        u = −K x̂
```

### How the code does it
`LQRController.from_model(discrete_model, Q_lqr, R_lqr)`:
1. rejects a continuous model (the brief runs in discrete time);
2. calls `numerics.solve_dare(A_d, B_d, Q, R)` — the value-iteration sweep;
3. records `K_gain` and `P_dare`;
4. **verifies the closed loop**: computes `spectral_radius(A_d − B_d K)` with the
   from-scratch eigensolver and raises `UnstableClosedLoopError` if any
   eigenvalue is on or outside the unit circle (within tolerance).

`from_config(model, LQGConfig)` is the typed entry point used by the rest of the
system; it consumes only `Q_lqr`/`R_lqr` (the `W`/`V` in the same config belong
to the filter). `compute_control(x_hat)` returns `u = −K x̂`.

### A real consequence captured in a test
Giving the wheel angle zero weight leaves its integrator unregulated: a
closed-loop eigenvalue sits exactly on the unit circle and
`UnstableClosedLoopError` fires. This is not a contrived check — it is the
physics, and the test asserts it.

---

## 3. Kalman filter — `estimation/kalman_filter.py`

### The recursion
Standard discrete predict/update (Welch & Bishop eqs. 1.9–1.13;
Anderson & Moore §3.1):

```
predict:  x̂⁻ = A_d x̂ + B_d u            P⁻ = A_d P A_dᵀ + W
update:   S  = C P⁻ Cᵀ + V               L  = P⁻ Cᵀ S⁻¹
          x̂  = x̂⁻ + L (z − C x̂⁻)
          P  = (I − L C) P⁻ (I − L C)ᵀ + L V Lᵀ      (Joseph form)
```

Implementation choices:
- The gain `L` is applied as an SPD solve (`chol_solve`), never an explicit
  inverse.
- The covariance update uses the **Joseph form**, which stays symmetric and
  positive-semidefinite under round-off (re-symmetrised each step).
- The class is mutable (`x_hat`, `P_est` evolve) but **deterministic**: the
  trajectory of estimates is a pure function of the inputs.

`from_config(model, LQGConfig)` consumes `W_process`/`V_measure`. `predict(u)`,
`update(z)`, and `step(u, z)` (predict then update) drive it.

### Steady state via duality
`steady_state_kalman_gain` reuses the **same** Riccati solver with the dual
substitution `A → A_dᵀ, B → Cᵀ, Q → W, R → V` — one routine, two uses.

### A documented physical subtlety
With the real sensor set `[θ_p, θ̇_w]` the wheel **angle** `θ_w` is
*unobservable* at a unit-circle eigenvalue: no steady-state filter exists for
it. The dual Riccati correctly refuses to converge (a test pins this), and the
recursive `P[2,2]` grows without bound. Finite-horizon filtering is unaffected —
the rollouts are finite — and this is exactly why the controller keeps the
wheel-angle weight small.

---

## 4. LQG — putting them together

LQR and Kalman are **siblings**: neither imports the other. They are coupled
only inside `SimulationEngine`, which closes the loop each step:

```
z_k = C x_k + v_k                 # measure the true state (noisy)
x̂_k = kalman.update / step         # posterior estimate
u_k = −K x̂_k                       # LQR acts on the estimate, not the truth
x_{k+1} = plant.step(x_k, u_k) + w_k
```

Feeding the gain back on the **estimate** rather than the true state — the
separation principle — is what makes output-feedback control optimal. The
end-to-end test in `tests/unit/test_kalman_filter.py` closes this loop on the
true nonlinear plant with noisy sensors and shows the pendulum held upright from
a 0.05 rad tilt with millradian estimation error.

---

## 5. Why the Riccati solver is shared

Both the LQR gain and the Kalman steady-state gain are fixed points of the same
algebraic equation; the only difference is which matrices play which role. The
project implements `solve_dare` once and calls it from both layers — the
clearest possible expression of the LQR/Kalman duality, and one fewer place for
a bug to hide. See [numerics_from_scratch.md](numerics_from_scratch.md#riccatipy--the-discrete-riccati-solver-solve_darea-b-q-r).
