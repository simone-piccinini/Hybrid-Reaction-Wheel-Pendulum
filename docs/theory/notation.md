# Notation — Symbol ↔ Code Glossary

> **Consult this file before naming any variable, matrix, or function.** Its purpose is to guarantee that one mathematical symbol maps to exactly one code identifier across the whole repository. The most dangerous ambiguity it resolves: the letters $Q$ and $R$ mean *different things* in LQR and in the Kalman filter.

All matrices are NumPy 2-D arrays of dtype `float64`; all vectors are 1-D `float64` arrays. Time series are 2-D arrays with time along axis 0.

---

## 1. The Q / R / W / V disambiguation (read first)

| Symbol | Meaning | **Canonical code name** | Shape | Owner |
|---|---|---|---|---|
| $Q$ | LQR **state-error** cost weight | `Q_lqr` | $n_x \times n_x$ | `LQRController` |
| $R$ | LQR **control-effort** cost weight | `R_lqr` | $n_u \times n_u$ | `LQRController` |
| $W$ | Kalman **process-noise** covariance | `W_process` | $n_x \times n_x$ | `KalmanFilter` |
| $V$ | Kalman **measurement-noise** covariance | `V_measure` | $n_y \times n_y$ | `KalmanFilter` |

**Never** name a variable bare `Q` or `R`. The four matrices above are the components of the Bayesian-optimisation decision vector $\boldsymbol{\theta}$ (see §6).

---

## 2. Physical system

| Symbol | Meaning | Code name | Unit |
|---|---|---|---|
| $\theta_p$ | pendulum angle from upright | `theta_p` | rad |
| $\dot\theta_p$ | pendulum angular velocity | `theta_p_dot` | rad/s |
| $\theta_w$ ($\phi$) | reaction-wheel angle (rel. to pendulum) | `theta_w` | rad |
| $\dot\theta_w$ | reaction-wheel angular velocity | `theta_w_dot` | rad/s |
| $m_p$ | pendulum mass | `pendulum_mass` | kg |
| $\ell_p$ | pendulum length (pivot→CoM) | `pendulum_length` | m |
| $I_b$ | pendulum body inertia | `body_inertia` | kg·m² |
| $I_w$ | reaction-wheel inertia | `wheel_inertia` | kg·m² |
| $\tau$ | motor torque on the wheel | `torque` | N·m |
| $g$ | gravitational acceleration | `GRAVITY` (const) | m/s² |

### DC motor

| Symbol | Meaning | Code name | Unit |
|---|---|---|---|
| $V_m$ | motor input voltage | `voltage` | V |
| $R_m$ | armature resistance | `resistance` | Ω |
| $L_m$ | armature inductance | `inductance` | H |
| $K_t$ | torque constant | `torque_constant` | N·m/A |
| $K_e$ | back-EMF constant | `back_emf_constant` | V·s/rad |
| $i$ | armature current | `current` | A |
| $V_{m,\max}$ | voltage saturation limit | `max_voltage` | V |

---

## 3. State-space model

| Symbol | Meaning | Code name | Shape |
|---|---|---|---|
| $\mathbf{x}$ | state vector | `x` | $n_x$ |
| $\mathbf{u}$ | control input | `u` | $n_u$ |
| $\mathbf{y}$ | measured output | `y` (or `z` in filter) | $n_y$ |
| $A$ | continuous state matrix | `A` | $n_x \times n_x$ |
| $B$ | continuous input matrix | `B` | $n_x \times n_u$ |
| $C$ | output matrix | `C` | $n_y \times n_x$ |
| $D$ | feedthrough matrix | `D` | $n_y \times n_u$ |
| $A_d, B_d$ | discrete-time matrices | `A_d`, `B_d` | as above |
| $\Delta t$ | sample / integration step | `dt` | s |
| $n_x, n_u, n_y$ | dimensions | `n_x`, `n_u`, `n_y` | int |

Convention: `x` is the **true** state; `x_hat` is the Kalman **estimate**. The default state ordering is `x = [theta_p, theta_p_dot, theta_w, theta_w_dot]` ($n_x = 4$), matching model.md's $[\theta, \dot\theta, \phi, \dot\phi]$: pendulum angle, pendulum rate, wheel angle, wheel rate. The wheel angle `theta_w` ($\phi$) is uncontrollable (column 3 of $A$ is zero) but is retained in the state for tracking.

---

## 4. Control (LQR)

| Symbol | Meaning | Code name | Shape |
|---|---|---|---|
| $Q$ | state cost (see §1) | `Q_lqr` | $n_x \times n_x$ |
| $R$ | control cost (see §1) | `R_lqr` | $n_u \times n_u$ |
| $K$ | optimal feedback gain | `K_gain` | $n_u \times n_x$ |
| $P$ | Riccati solution (control) | `P_care` | $n_x \times n_x$ |
| $\mathbf{u} = -K\hat{\mathbf{x}}$ | control law | `compute_control(x_hat)` | — |

---

## 5. Estimation (Kalman filter)

| Symbol | Meaning | Code name | Shape |
|---|---|---|---|
| $W$ | process-noise cov (see §1) | `W_process` | $n_x \times n_x$ |
| $V$ | measurement-noise cov (see §1) | `V_measure` | $n_y \times n_y$ |
| $\hat{\mathbf{x}}$ | state estimate | `x_hat` | $n_x$ |
| $P$ | estimate-error covariance | `P_est` | $n_x \times n_x$ |
| $L$ | Kalman gain | `L_gain` | $n_x \times n_y$ |
| $\mathbf{z}$ | measurement | `z` | $n_y$ |

Note: the control Riccati solution `P_care` and the estimator covariance `P_est` are **distinct**; never share the name `P`.

---

## 6. Bayesian optimisation — outer loop

| Symbol | Meaning | Code name | Shape |
|---|---|---|---|
| $\boldsymbol{\theta}$ | BO decision vector $\mathrm{vec}(Q,R,W,V)$ | `theta` | $d$ |
| $\boldsymbol{\theta}^*$ | (believed) global optimum location | `theta_star` | $d$ |
| $d$ | decision-space dimension | `dim` | int |
| $J(\boldsymbol{\theta})$ | true (unknown) cost surface | — | scalar |
| $y$ | observed cost of one simulation | `y` | scalar |
| $\mathcal{D}_n$ | dataset $\{(\theta_i, y_i)\}$ | `Dataset` | — |
| $M_p$ | percentage overshoot | `overshoot` | — |
| $T_s$ | settling time | `settling_time` | s |
| $e(t)$ | regulated error $\theta_p(t) - 0$ | `true_states[:,0]` | rad |
| $\mathrm{ITAE}$ | $\int_0^{T} t\,\lvert e(t)\rvert\,dt$ | `itae` | rad·s² |
| $M_p^{\max}, T_s^{\max}$ | max **acceptable** $M_p$, $T_s$ (hinge scales) | `Mp_max`, `Ts_max` | —, s |
| $U_{\max}$ | actuation limit (optional) | `U_max` | V |
| $w_e, w_u$ | weights on ITAE, control energy | `w_error`, `w_control` | scalar |
| $w_p, w_t$ | **penalty** weights on $M_p$, $T_s$ (large) | `w_overshoot`, `w_settling` | scalar |

**Cost function** (`ObjectiveFunction.evaluate`). Rather than a symmetric
squared error on two points of the step response, the cost combines a global
error index, a control-energy term, and **asymmetric hinge penalties** that
fire only when a specification is *violated*:

$$
y \;=\; w_e\,\underbrace{\int_0^{T} t\,\lvert e(t)\rvert\,dt}_{\text{ITAE}}
\;+\; w_u \int_0^{T} u(t)^2\,dt
\;+\; w_p\left[\max\!\Big(0,\tfrac{M_p - M_p^{\text{des}}}{M_p^{\max}}\Big)\right]^2
\;+\; w_t\left[\max\!\Big(0,\tfrac{T_s - T_s^{\text{des}}}{T_s^{\max}}\Big)\right]^2
$$

with an optional actuation-saturation penalty $w_{\text{sat}}\big[\max(0,\;\max_t\lvert u(t)\rvert - U_{\max})\big]^2$.

- **ITAE** weights *late* error by $t$, so minimising it speeds up settling and
  drives the steady-state error to zero — judging the whole history, not two
  isolated points.
- The **hinge** terms are zero while the response meets the spec (the controller
  is free to be *better* than target) and grow quadratically past it.
  Normalising by the *maximum acceptable* value $M_p^{\max}$/$T_s^{\max}$ keeps
  them scale-free and avoids the division-by-zero of a desired-value denominator.
- $w_p, w_t$ are deliberately **large**: a spec violation must make $y$ spike so
  the optimiser discards those hyper-parameters. $w_e, w_u$ set the general
  trade-off among spec-satisfying designs (the system-level analogue of the
  LQR's $Q$/$R$).

Code: `Mp_desired`, `Mp_max`, `Ts_desired`, `Ts_max`, `w_error`, `w_control`,
`w_overshoot`, `w_settling`, `U_max`, `w_saturation`. A diverged rollout maps to
a large finite `PENALTY` (§7).

---

## 7. Gaussian process — inner loop & surrogate

| Symbol | Meaning | Code name | Shape |
|---|---|---|---|
| $k(\cdot,\cdot)$ | covariance / kernel function | `Kernel.covariance` | scalar |
| $K$ | training kernel matrix $k(X,X)$ | `K_matrix` | $n \times n$ |
| $\mathbf{k}_*$ | cross-kernel $k(X, \theta_*)$ | `k_star` | $n$ |
| $k_{**}$ | prior variance $k(\theta_*,\theta_*)$ | `k_star_star` | scalar |
| $L$ | Cholesky factor of $K+\sigma_n^2 I$ | `L_chol` | $n \times n$ |
| $\boldsymbol{\alpha}$ | $(K+\sigma_n^2 I)^{-1}\mathbf{y}$ | `alpha` | $n$ |
| $\bar{f}_*$ | predictive mean | `mean` | scalar/array |
| $\mathbb{V}[f_*]$ | predictive variance | `variance` | scalar/array |
| $\boldsymbol{\phi}$ | GP hyperparameters | `hyperparams` | — |
| $\ell$ | length-scales (ARD: one per dim) | `lengthscales` | $d$ |
| $\sigma_f^2$ | signal variance | `signal_variance` | scalar |
| $\sigma_n^2$ | noise variance | `noise_variance` | scalar |
| $\log p(\mathbf{y}\mid X)$ | log marginal likelihood | `log_marginal_likelihood` | scalar |

**Note the `L` collision:** `L_gain` is the Kalman gain (§5); `L_chol` is the GP Cholesky factor. Always suffix.

---

## 8. Entropy Search — acquisition

| Symbol | Meaning | Code name |
|---|---|---|
| $p(\boldsymbol{\theta}^*\mid\mathcal{D})$ | belief over optimum location | `optimum_distribution` |
| $H[\cdot]$ | Shannon (differential) entropy | `entropy` |
| $\alpha_{\text{ES}}(\boldsymbol{\theta})$ | expected entropy reduction | `acquisition_value` |
| $\mathbb{E}_y[\cdot]$ | expectation over predicted $y$ | (Monte-Carlo in code) |

Acquisition objective:
$$
\boldsymbol{\theta}_{n+1} = \arg\max_{\boldsymbol{\theta}}\Big[H[p(\boldsymbol{\theta}^*\mid\mathcal{D}_n)] - \mathbb{E}_y\big[H[p(\boldsymbol{\theta}^*\mid\mathcal{D}_n\cup\{(\boldsymbol{\theta},y)\})]\big]\Big]
$$

---

## 9. General conventions

- Constants are `UPPER_SNAKE_CASE` (`GRAVITY`).
- Discrete-time quantities take a `_d` suffix (`A_d`).
- Estimated quantities take a `_hat` suffix (`x_hat`).
- Desired/target quantities take a `_desired` suffix (`Ts_desired`).
- Matrices are capitalised in math but use descriptive snake_case in code (`Q_lqr`, not `Q`).
- Function docstrings cite the implementing equation: `Eq. 2.25 (R&W)` or `docs/theory/<file>.md §x`.
