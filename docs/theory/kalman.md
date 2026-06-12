# Implementation Brief — LQR Controller via the Discrete Riccati Equation

## Goal

Compute the constant feedback gain $K$ that stabilises the pendulum at the upright equilibrium while balancing two competing costs: how far the state strays from zero, and how much control effort is spent. The control law is a single matrix multiply, $u = -K\hat{\mathbf{x}}$, applied to the state estimate from the Kalman filter.

## Where it comes from (the one-paragraph intuition)

LQR minimises an infinite sum of quadratic costs over a linear system. Solving for all future controls at once is intractable, so dynamic programming solves it backward one step at a time via the "cost-to-go" value function. Because the dynamics are linear and the cost quadratic, that value function is itself quadratic — $V(\mathbf{x}) = \tfrac{1}{2}\mathbf{x}^\top P \mathbf{x}$ — and minimising it over the control falls out as a *linear* feedback law $u = -K\mathbf{x}$. The matrix $P$ is the unique fixed point of the **Riccati equation**, and the gain is read directly off it. For a time-invariant plant the backward recursion converges to a steady state, so $P$ and $K$ are computed once and held constant.

## What it consumes

- $(A_d, B_d)$ — the **discretised** plant matrices from the dynamics layer (the brief runs in discrete time; do not pass continuous $A, B$).
- $Q \succeq 0$ — state cost weight. Penalises deviation of each state from zero; larger entries demand tighter regulation of that state.
- $R \succ 0$ — control cost weight. Penalises voltage; larger $R$ produces gentler, more energy-efficient control.

$Q$ and $R$ are **tuning parameters owned by the optimization layer** — not physical constants. The whole Bayesian-optimization loop exists to discover the $(Q, R)$ (alongside the filter's $W, V$) that make the real closed loop perform best. This brief treats them as inputs.

## The equations to implement

**Discrete Algebraic Riccati Equation (DARE)** — solve for the symmetric positive-definite fixed point $P$:

$$

P = Q + A_d^\top P A_d - A_d^\top P B_d\left(R + B_d^\top P B_d\right)^{-1} B_d^\top P A_d.

$$

**Gain** — read off the converged $P$:

$$

K = \left(R + B_d^\top P B_d\right)^{-1} B_d^\top P A_d.

$$

**Control law** — applied every step to the estimate:

$$

u = -K\,\hat{\mathbf{x}}.

$$

## How to solve the DARE — backward sweep to steady state

The fixed point is found by iterating the recursion until it stops moving:

1. **Initialise** $P \leftarrow Q$ (any PSD seed works; $Q$ is standard).
2. **Iterate** the Riccati update:
   $$

   P_{\text{next}} = Q + A_d^\top P A_d - A_d^\top P B_d\left(R + B_d^\top P B_d\right)^{-1} B_d^\top P A_d.

   $$
3. **Check convergence:** stop when $\lVert P_{\text{next}} - P \rVert_\infty < \texttt{ATOL} + \texttt{RTOL}\cdot\lVert P \rVert_\infty$ (tolerances from the numerical-standards config). Otherwise set $P \leftarrow P_{\text{next}}$ and repeat.
4. **Cap iterations** at the configured maximum; if it has not converged, raise `RiccatiNotConverged` rather than returning a half-baked $P$.
5. **Compute $K$** once from the converged $P$.

This is the Kleinman-style value-iteration route. It lives in the numerics layer and is written from scratch — no library DARE solver.

## Duality note (reuse the same solver)

The Kalman filter's steady-state covariance solves the **same equation** with substitutions $A_d \to A_d^\top$, $B_d \to C^\top$, $Q \to W$, $R \to V$. Implement one Riccati routine and call it twice — once for the LQR gain, once for the estimator gain. Do not write two solvers.

## Implementation order

1. **Receive** $(A_d, B_d)$ from dynamics and $(Q, R)$ from config / optimizer.
2. **Validate:** $Q$ symmetric PSD, $R$ symmetric positive-definite; check $(A_d, B_d)$ is controllable (otherwise no stabilising gain exists).
3. **Solve the DARE** by the backward sweep above to obtain the steady-state $P$.
4. **Compute** $K$ from $P$.
5. **Hand off** the constant $K$ to the control law, which is invoked each timestep as $u = -K\hat{\mathbf{x}}$.

## Practical cautions

- **Never invert directly.** The term $(R + B_d^\top P B_d)^{-1}$ is applied by solving an SPD linear system with the project's factorisation primitive, not by forming an explicit inverse.
- **Keep $P$ symmetric.** Round-off drifts $P$ off symmetry; re-symmetrise (`P ← ½(P + Pᵀ)`) each iteration to keep the inner inverse well-conditioned.
- **Confirm the closed loop is stable.** After computing $K$, the eigenvalues of $A_d - B_d K$ must all lie inside the unit circle. If any sits on or outside it, the gain does not stabilise — treat as a configuration error (and, in the optimizer, map to a large finite penalty rather than crashing).
- **The gain feeds back on the estimate, not the true state.** LQR consumes $\hat{\mathbf{x}}$ from the Kalman filter; this separation (LQR + Kalman = LQG) is what makes output-feedback control optimal.
- **Continuous vs discrete.** This brief is the discrete DARE. If the controller were left in continuous time it would instead be the continuous (CARE) form $0 = Q + A^\top P + P A - P B R^{-1} B^\top P$ — keep the two distinct and use the one matching your discretisation choice.