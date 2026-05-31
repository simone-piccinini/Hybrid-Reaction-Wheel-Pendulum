"""Gradient-based unconstrained optimiser for ML-II hyperparameter fitting.

A from-scratch **BFGS** quasi-Newton method with an Armijo backtracking line
search, plus a seeded multi-restart driver (``numerical_standards.md`` §3,
"Gradient optimiser"; convergence rule and restarts §5). The hyperparameter
space is low-dimensional (per-dimension lengthscales plus signal and noise),
so full BFGS — maintaining a dense inverse-Hessian approximation — is cheap and
converges superlinearly.

This routine **minimises**. ML-II *maximises* the log marginal likelihood, so
the optimization layer passes the *negative* log marginal likelihood (and its
negated gradient). Gradients are supplied analytically by the caller; no
autodiff and no banned optimiser library are used (``AGENTS.md`` §3).

Reference
---------
Nocedal & Wright, *Numerical Optimization* (2nd ed.): BFGS update eq. (6.17),
initial Hessian scaling eq. (6.20), Armijo sufficient-decrease eq. (3.4).
"""

from __future__ import annotations

import warnings
from typing import Callable, NamedTuple

import numpy as np

from .constants import ML2_GRAD_TOL, ML2_MAX_ITER, ML2_RESTARTS, ML2_STEP_TOL

# Line-search / update constants (Nocedal & Wright); algorithmic, not tolerances.
_ARMIJO_C1: float = 1e-4  # sufficient-decrease constant, eq. (3.4)
_BACKTRACK_RHO: float = 0.5  # step contraction factor
_MAX_LINE_SEARCH: int = 50  # backtracking steps before declaring failure
_CURVATURE_EPS: float = 1e-10  # skip BFGS update if sᵀy ≤ this (keeps H ≻ 0)

ScalarFn = Callable[[np.ndarray], float]
GradFn = Callable[[np.ndarray], np.ndarray]


class OptimizeResult(NamedTuple):
    """Outcome of :func:`minimize`.

    ``x`` minimiser, ``fun`` objective there, ``grad`` gradient there,
    ``n_iter`` iterations taken, ``converged`` whether a §5 criterion was met,
    and a human-readable ``message``.
    """

    x: np.ndarray
    fun: float
    grad: np.ndarray
    n_iter: int
    converged: bool
    message: str


def _max_abs(v: np.ndarray) -> float:
    """Vector ∞-norm ``‖v‖∞`` = max absolute component."""
    return float(np.max(np.abs(v))) if v.size else 0.0


def _backtracking_line_search(fun: ScalarFn, x, f0, g0, p):
    """Armijo backtracking along ``p``; Nocedal & Wright eq. (3.4), Alg. 3.1.

    Returns ``(success, alpha, x_new, f_new)``. Halves ``alpha`` from 1 until
    ``f(x + alpha p) ≤ f0 + c1 alpha (g0·p)`` holds with a finite value.
    """
    slope = float(g0 @ p)  # directional derivative; < 0 for a descent direction
    alpha = 1.0
    for _ in range(_MAX_LINE_SEARCH):
        x_new = x + alpha * p
        f_new = float(fun(x_new))
        if np.isfinite(f_new) and f_new <= f0 + _ARMIJO_C1 * alpha * slope:
            return True, alpha, x_new, f_new
        alpha *= _BACKTRACK_RHO
    return False, alpha, x + alpha * p, float(fun(x + alpha * p))


def minimize(
    fun: ScalarFn,
    x0,
    grad: GradFn,
    *,
    max_iter: int = ML2_MAX_ITER,
    grad_tol: float = ML2_GRAD_TOL,
    step_tol: float = ML2_STEP_TOL,
) -> OptimizeResult:
    """Minimise ``fun`` from ``x0`` by BFGS with an Armijo line search.

    Iterate ``p = −H ∇f`` (``H`` the inverse-Hessian approximation), line-search
    along ``p``, then apply the BFGS update (Nocedal & Wright eq. 6.17) with the
    eq. (6.20) initial scaling and a curvature guard. Stop when
    ``‖∇f‖∞ < grad_tol`` **or** the step ``‖Δx‖∞ < step_tol`` (both count as
    convergence, ``numerical_standards.md`` §5); on reaching ``max_iter`` return
    the best-so-far point and warn.

    Parameters
    ----------
    fun : callable
        Scalar objective ``fun(x) -> float``.
    x0 : array_like
        Starting point.
    grad : callable
        Gradient ``grad(x) -> ndarray`` (analytic; same shape as ``x``).
    max_iter, grad_tol, step_tol :
        Defaults from ``numerical_standards.md`` §5.

    Returns
    -------
    OptimizeResult
    """
    x = np.asarray(x0, dtype=np.float64).copy()
    n = x.size
    ident = np.eye(n, dtype=np.float64)
    f = float(fun(x))
    g = np.asarray(grad(x), dtype=np.float64)

    if _max_abs(g) < grad_tol:
        return OptimizeResult(x, f, g, 0, True, "initial point met gradient tolerance")

    H = ident.copy()
    converged = False
    message = f"reached max_iter={max_iter} without convergence"
    iteration = 0
    for iteration in range(1, max_iter + 1):
        p = -H @ g
        if g @ p >= 0.0:  # not a descent direction — reset to steepest descent
            H = ident.copy()
            p = -g
        success, _alpha, x_new, f_new = _backtracking_line_search(fun, x, f, g, p)
        if not success:
            converged = _max_abs(g) < grad_tol
            message = "line search could not satisfy the Armijo condition"
            break

        s = x_new - x
        g_new = np.asarray(grad(x_new), dtype=np.float64)
        y = g_new - g
        sy = float(s @ y)
        if iteration == 1 and sy > _CURVATURE_EPS:
            H = (sy / float(y @ y)) * ident  # initial scaling, eq. (6.20)
        if sy > _CURVATURE_EPS:  # curvature guard keeps H positive-definite
            rho = 1.0 / sy
            left = ident - rho * np.outer(s, y)
            H = left @ H @ left.T + rho * np.outer(s, s)

        x, f, g = x_new, f_new, g_new
        if _max_abs(g) < grad_tol:
            converged, message = True, "gradient tolerance satisfied"
            break
        if _max_abs(s) < step_tol:
            converged, message = True, "step below step tolerance"
            break

    if not converged:
        warnings.warn(f"minimize: {message}", stacklevel=2)
    return OptimizeResult(
        x=x, fun=f, grad=g, n_iter=iteration, converged=converged, message=message
    )


def minimize_with_restarts(
    fun: ScalarFn,
    grad: GradFn,
    sample_start: Callable[[np.random.Generator], np.ndarray],
    rng: np.random.Generator,
    *,
    n_restarts: int = ML2_RESTARTS,
    **minimize_kwargs,
) -> OptimizeResult:
    """Run :func:`minimize` from ``n_restarts`` seeded starting points; keep the best.

    The ML-II objective is non-convex, so multiple restarts are run and the
    result with the lowest ``fun`` is returned (``numerical_standards.md`` §5).
    Starting points come from ``sample_start(rng)``, so the whole search is
    reproducible from ``rng`` (determinism contract §6). ``sample_start`` owns
    the domain (e.g. hyperparameter bounds in log-space); this routine stays
    agnostic to it.
    """
    if n_restarts < 1:
        raise ValueError("n_restarts must be >= 1")
    best: OptimizeResult | None = None
    for _ in range(n_restarts):
        x0 = np.asarray(sample_start(rng), dtype=np.float64)
        result = minimize(fun, x0, grad, **minimize_kwargs)
        if best is None or result.fun < best.fun:
            best = result
    assert best is not None  # n_restarts >= 1 guarantees a result
    return best
