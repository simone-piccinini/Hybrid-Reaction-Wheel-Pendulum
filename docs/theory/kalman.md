# Implementation Brief — Kalman Filter (the LQG State Estimator)

## Goal

Recover the full state $\hat{\mathbf{x}}$ of the pendulum from a handful of
noisy sensors, so the LQR law can feed back on an *estimate* it does not measure
directly. The controller wants $[\theta_p, \dot\theta_p, \theta_w, \dot\theta_w]$
but the hardware senses only a subset (an angle and a wheel rate). The Kalman
filter fills the gap: it is the optimal linear estimator that fuses the plant
*model* with each incoming *measurement*, weighting the two by their relative
uncertainty. Its output $\hat{\mathbf{x}}$ is exactly what `LQRController`
consumes in $u = -K\hat{\mathbf{x}}$ — the two halves of **LQG**.

## Where it comes from (the one-paragraph intuition)

At every step we hold a Gaussian belief about the state: a mean $\hat{\mathbf{x}}$
and a covariance $P$ that measures how unsure we are. Two things happen each
cycle. First we **predict**: push the belief forward through the known dynamics
$(A_d, B_d)$, which moves the mean and — because the model is imperfect and the
world is noisy — *inflates* the covariance by the process noise $W$. Then we
**update**: a measurement $\mathbf{z}$ arrives, we compare it to what the model
expected ($C\hat{\mathbf{x}}^-$), and we nudge the mean toward the data by an
amount set by the **Kalman gain** $L$ — large when the sensors are trusted
(small $V$), small when they are noisy. The update *shrinks* the covariance,
because a measurement always buys information. Predict inflates, update deflates;
the gain is the optimal balance point between the two, and for a time-invariant
plant the covariance settles to a steady state so the gain can be precomputed.

## What it consumes

- $(A_d, B_d, C)$ — the **discretised** linear plant from the dynamics layer.
  The recursion runs in discrete time: pass `model.discretize(dt)`, never the
  continuous $(A, B)$. $C$ is the measurement matrix, $\mathbf{y} = C\mathbf{x} +
  \mathbf{v}$, selecting what the hardware actually senses (see `model.md`).
- $W \succeq 0$ — **process-noise** covariance (`W_process`, $n_x \times n_x$).
  How much we distrust the model; larger $W$ makes the filter lean on the sensors.
- $V \succ 0$ — **measurement-noise** covariance (`V_measure`, $n_y \times n_y$).
  How much we distrust the sensors; larger $V$ makes the filter lean on the model.

$W$ and $V$ are **not** the LQR's $Q, R$ — see `notation.md` §1. Like $Q, R$ they
are **tuning parameters owned by the optimisation layer**: the Bayesian-optimisation
loop searches $(Q, R, W, V)$ jointly for the best real closed-loop performance.
This brief treats them as inputs.

## The equations to implement

Write $\hat{\mathbf{x}}^-, P^-$ for the *a-priori* (predicted) estimate and
covariance, $\hat{\mathbf{x}}, P$ for the *a-posteriori* (corrected) ones.

**Predict (time update)** — propagate the belief through the dynamics:

$$
\hat{\mathbf{x}}^- = A_d\,\hat{\mathbf{x}} + B_d\,u, \qquad
P^- = A_d\,P\,A_d^\top + W.
$$

**Update (measurement update)** — fold in the measurement $\mathbf{z}$:

$$
S = C\,P^-\,C^\top + V, \qquad
L = P^-\,C^\top S^{-1},
$$
$$
\hat{\mathbf{x}} = \hat{\mathbf{x}}^- + L\,\big(\mathbf{z} - C\,\hat{\mathbf{x}}^-\big), \qquad
P = (I - L C)\,P^-\,(I - L C)^\top + L\,V\,L^\top.
$$

$S$ is the **innovation covariance**, $\mathbf{z} - C\hat{\mathbf{x}}^-$ the
**innovation** (measurement minus prediction), and $L$ the **Kalman gain**
(`L_gain`, $n_x \times n_y$). The covariance update is written in **Joseph form**
— $(I - LC)P^-(I - LC)^\top + LVL^\top$ rather than the algebraically equal but
fragile $(I - LC)P^-$ — because it stays symmetric and positive-semidefinite
under round-off (`numerical_standards.md` §4). One full cycle is `predict(u)`
then `update(z)`.

References: Anderson & Moore, *Optimal Filtering*, §3.1; Welch & Bishop, *An
Introduction to the Kalman Filter*, eqs. 1.9–1.13.

## Duality note (reuse the Riccati solver)

For a time-invariant plant the a-priori covariance $P^-$ converges to a fixed
point that solves a **discrete algebraic Riccati equation** — the *same* equation
the LQR solves, under the substitution $A_d \to A_d^\top$, $B_d \to C^\top$,
$Q \to W$, $R \to V$:

$$
P^- = W + A_d\,P^-\,A_d^\top - A_d\,P^-\,C^\top\big(C\,P^-\,C^\top + V\big)^{-1} C\,P^-\,A_d^\top .
$$

So the steady-state filter calls the **one** DARE routine from `numerics/riccati.py`
as `solve_dare(A_dᵀ, Cᵀ, W, V)`, reads the converged $P^-$, and forms the
steady-state gain $L = P^- C^\top (C P^- C^\top + V)^{-1}$ from it. Do **not**
write a second solver, and do not confuse the covariances: the control Riccati
solution is `P_dare` / `P_care`; the estimator's is `P_est` (recursive) or
`P_pred` (steady-state a-priori). The gain the DARE returns for the dual problem
is *not* itself the Kalman gain — recompute $L$ from $P^-$ as above.

## The detectability caveat for this plant

The steady-state filter exists only for a **detectable** pair $(A_d, C)$ — every
unstable/marginal mode must be visible in some measurement. This plant's locked
sensor set $[\theta_p, \dot\theta_w]$ **does not see the wheel angle** $\theta_w$:
column 3 of $A$ is zero (the angle drives nothing) and no sensor reads it, so it
is an *unobservable integrator* sitting on the unit circle. Consequently **no
steady-state Kalman filter exists for the full 4-state plant** — `solve_dare` on
the dual would not converge to a stabilising solution.

This is a property of the physics, not a bug. Two consequences, both intended:

- The **recursive** filter (`predict`/`update`) stays perfectly well-defined; only
  the *estimate-error covariance entry* $P_{\text{est}}[2,2]$ (the wheel angle)
  grows without bound over an infinite horizon. Over the finite simulation
  horizon it is harmless.
- Analyses that *need* a steady-state observer (e.g. the loop-gain / stability
  margins) drop $\theta_w$ and work on the controllable-and-observable reduced
  model $[\theta_p, \dot\theta_p, \dot\theta_w]$ (`KEEP = [0,1,3]`), on which the
  dual DARE is well-posed. The hidden mode cancels in the loop gain, so margins
  are unaffected (see `docs/papers/robustness_lqg_measured.md`).

## Implementation order

1. **Receive** the discrete $(A_d, B_d, C)$ from dynamics and $(W, V)$ from
   config / optimiser. Validate: $W$ symmetric PSD, $V$ symmetric PD, shapes
   $n_x \times n_x$ and $n_y \times n_y$.
2. **Initialise** $\hat{\mathbf{x}}_0$ (default zeros — the upright equilibrium)
   and $P_0$ (default $W$, the one-step prior uncertainty).
3. **Predict** with the applied $u$: propagate $\hat{\mathbf{x}}^-, P^-$.
4. **Update** with the measurement $\mathbf{z}$: form $S$, solve for $L$, correct
   the mean, apply the Joseph-form covariance update.
5. **Hand off** $\hat{\mathbf{x}}$ to the LQR law each step. For a fixed design,
   optionally precompute the steady-state $L$ via the dual DARE (§ "Duality note")
   instead of recursing $P$.

## Practical cautions

- **Never invert directly.** $L = P^- C^\top S^{-1}$ is applied by solving the SPD
  system $S\,L^\top = C\,P^-$ with the project's Cholesky solve (`chol_solve`),
  not by forming $S^{-1}$ (`AGENTS.md` §3).
- **Keep $P$ symmetric.** Re-symmetrise ($P \leftarrow \tfrac12(P + P^\top)$) after
  each update; round-off drifts $P$ off symmetry and corrupts the next $S$. Use
  the Joseph form, not $(I - LC)P^-$.
- **Discrete only.** The recursion runs on $(A_d, B_d, C)$; passing the continuous
  model is a configuration error the constructor rejects.
- **The gain feeds the controller, not the truth.** The LQR consumes $\hat{\mathbf{x}}$,
  never the true state — this separation (LQR + Kalman = LQG) is what makes
  output-feedback control optimal. But note **Doyle (1978)**: the LQR and the
  filter are each robust, yet their combination guarantees *no* stability margin
  — the reason `scripts/stability_margins.py` and the robustness study exist.
- **Determinism.** The estimate trajectory is a pure function of
  $(\mathbf{x}_0, P_0, u\text{-sequence}, z\text{-sequence})$; all randomness lives
  upstream in the seeded simulator, never inside the filter.
