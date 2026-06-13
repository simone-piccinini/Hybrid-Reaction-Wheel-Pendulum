"""ML-II — type-II maximum likelihood for the GP hyperparameters.

The **inner loop** of ``optimization.md`` §3 (never to be confused with the
outer BO loop over the controller weights θ, AGENTS §5): fit
``φ = (ℓ₁ … ℓ_d, σ_f, σ_n)`` by maximising the log marginal likelihood

    log p(y | X, φ) = −½ yᵀ K_n⁻¹ y − ½ log|K_n| − (n/2) log 2π,
    K_n = K(φ) + σ_n² I,

(R&W eq. 5.8) by gradient ascent in **log-space** (the hyperparameters are
positive) from **multiple seeded restarts** (non-convexity;
``numerical_standards.md`` §5: defaults ``ML2_RESTARTS``, gradient/step
tolerances, 200-iteration cap, best-so-far on failure).

The analytic gradient is R&W eq. 5.9, in the numerically convenient form

    ∂ log p / ∂φⱼ = ½ tr( (α αᵀ − K_n⁻¹) · ∂K_n/∂φⱼ ),   α = K_n⁻¹ y,

with the per-pair kernel derivatives supplied by ``Kernel.gradient`` in
natural units and the log-space chain rule ``∂K/∂(log φⱼ) = φⱼ · ∂K/∂φⱼ``
applied here. No autodiff (``AGENTS.md`` §3); the minimiser is the project's
own BFGS (``numerics/optimizers``), given the *negative* log marginal
likelihood.

Targets are expected **standardised** (the GP standardises before calling),
which is what makes the fixed restart box below meaningful: the outputs have
unit scale and the inputs live in the SearchSpace box (coordinates of order
one to ten).
"""

from __future__ import annotations

import numpy as np

from ..core.constants import ML2_RESTARTS, PSD_JITTER
from ..numerics.linalg import (
    NotPositiveDefiniteError,
    chol_solve,
    cholesky,
    solve_lower,
    solve_upper,
    symmetrize,
)
from ..numerics.optimizers import OptimizeResult, minimize
from .kernels.base import Kernel

# Restart-sampling box in log-space (numerical_standards.md §5 "multiple random
# restarts"; the ranges assume standardised targets and O(1-10) inputs).
LOG_LENGTHSCALE_RANGE: tuple[float, float] = (np.log(1e-2), np.log(1e2))
LOG_SIGNAL_STD_RANGE: tuple[float, float] = (np.log(1e-2), np.log(1e1))
LOG_NOISE_STD_RANGE: tuple[float, float] = (np.log(1e-4), np.log(1e0))

# Finite stand-in objective value when K_n is not factorisable at a trial φ
# (kept finite per §7 so the line search can back away from it).
_INFEASIBLE_NLL: float = 1e12


def negative_log_marginal_likelihood(
    log_phi, X, y_standardized, kernel_template: Kernel
) -> tuple[float, np.ndarray]:
    """``(−log p, −∇ log p)`` at the log-hyperparameters ``log φ``.

    ``log_phi`` packs ``(log ℓ₁ … log ℓ_d, log σ_f, log σ_n)``. Value from
    R&W eq. 5.8 via the Cholesky route (Alg. 2.1 line 7); gradient from
    eq. 5.9 with the log-space chain rule. ``K_n⁻¹`` is formed explicitly for
    the trace term — an O(n³) step that is transparent and cheap at BO sample
    sizes. An infeasible ``φ`` (non-PD ``K_n``) returns a large finite value
    with a zero gradient, so restarts elsewhere win.
    """
    log_phi = np.asarray(log_phi, dtype=np.float64)
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    y = np.asarray(y_standardized, dtype=np.float64).ravel()
    d = kernel_template.dim
    if log_phi.shape != (d + 2,):
        raise ValueError(f"log_phi must have shape ({d + 2},), got {log_phi.shape}")

    with np.errstate(over="ignore", under="ignore"):  # extreme φ handled below
        phi = np.exp(log_phi)  # (ℓ₁…ℓ_d, σ_f, σ_n) — lengthscales and std-devs
        signal_variance = float(phi[d] ** 2)
        noise_variance = float(phi[d + 1] ** 2)
    # the BFGS line search may probe extreme log φ: exp/square can overflow to
    # inf or underflow to 0, leaving the kernel ill-defined — finite stand-in
    # (§7) so the search backs away rather than crashing
    feasible = (
        np.all(np.isfinite(phi[:d]))
        and np.all(phi[:d] > 0.0)
        and np.isfinite(signal_variance)
        and signal_variance > 0.0
        and np.isfinite(noise_variance)
        and noise_variance > 0.0
    )
    if not feasible:
        return _INFEASIBLE_NLL, np.zeros(d + 2)
    kernel = kernel_template.with_hyperparams(phi[: d + 1])
    n = X.shape[0]

    K_n = kernel.matrix(X, X) + (noise_variance + PSD_JITTER) * np.eye(n)
    # a huge-but-finite σ_f can still overflow the kernel entries to inf; that
    # φ is infeasible, treated like a non-PD K_n (finite stand-in, §7)
    if not np.all(np.isfinite(K_n)):
        return _INFEASIBLE_NLL, np.zeros(d + 2)
    try:
        L = cholesky(symmetrize(K_n))
    except NotPositiveDefiniteError:
        return _INFEASIBLE_NLL, np.zeros(d + 2)

    alpha = solve_upper(L.T, solve_lower(L, y))
    log_p = (
        -0.5 * float(y @ alpha)
        - float(np.sum(np.log(np.diag(L))))
        - 0.5 * n * np.log(2.0 * np.pi)
    )

    # eq. 5.9: ∂log p/∂φⱼ = ½ tr(A · ∂K_n/∂φⱼ) with A = ααᵀ − K_n⁻¹
    K_n_inv = chol_solve(symmetrize(K_n), np.eye(n))
    A = np.outer(alpha, alpha) - K_n_inv

    # per-pair kernel derivatives, assembled into the d+1 derivative matrices
    dK = np.zeros((d + 1, n, n))
    for i in range(n):
        for j in range(i, n):
            g = kernel.gradient(X[i], X[j])
            dK[:, i, j] = g
            dK[:, j, i] = g

    grad_log_p = np.empty(d + 2)
    for p_idx in range(d + 1):  # log ℓᵢ and log σ_f: chain rule ·φ
        grad_log_p[p_idx] = 0.5 * phi[p_idx] * float(np.sum(A * dK[p_idx]))
    # log σ_n: ∂K_n/∂(log σ_n) = 2 σ_n² I
    grad_log_p[-1] = 0.5 * 2.0 * noise_variance * float(np.trace(A))

    return -log_p, -grad_log_p


def fit_hyperparameters(
    kernel: Kernel,
    noise_variance: float,
    X,
    y_standardized,
    rng: np.random.Generator,
    *,
    n_restarts: int = ML2_RESTARTS,
) -> tuple[Kernel, float, OptimizeResult]:
    """ML-II: maximise the log marginal likelihood over ``log φ`` (``optimization.md`` §3).

    Runs the project BFGS from the **current** hyperparameters first (the
    warm start — the model should never get worse than it was), then from
    ``n_restarts − 1`` points sampled in the documented log-space box, and
    keeps the best (``numerical_standards.md`` §5). Returns the winning
    kernel, noise variance, and the optimiser result.
    """
    if n_restarts < 1:
        raise ValueError("n_restarts must be >= 1")
    d = kernel.dim

    def objective(log_phi) -> float:
        value, _ = negative_log_marginal_likelihood(
            log_phi, X, y_standardized, kernel
        )
        return value

    def gradient(log_phi) -> np.ndarray:
        _, grad = negative_log_marginal_likelihood(
            log_phi, X, y_standardized, kernel
        )
        return grad

    def sample_start(generator: np.random.Generator) -> np.ndarray:
        start = np.empty(d + 2)
        start[:d] = generator.uniform(*LOG_LENGTHSCALE_RANGE, size=d)
        start[d] = generator.uniform(*LOG_SIGNAL_STD_RANGE)
        start[d + 1] = generator.uniform(*LOG_NOISE_STD_RANGE)
        return start

    current = np.log(
        np.concatenate([kernel.hyperparams, [np.sqrt(float(noise_variance))]])
    )
    best = minimize(objective, current, gradient)
    for _ in range(n_restarts - 1):
        result = minimize(objective, sample_start(rng), gradient)
        if result.fun < best.fun:
            best = result

    phi_best = np.exp(best.x)
    return (
        kernel.with_hyperparams(phi_best[: d + 1]),
        float(phi_best[-1] ** 2),
        best,
    )
