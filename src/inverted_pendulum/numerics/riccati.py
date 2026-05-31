"""Discrete algebraic Riccati equation (DARE) via backward value iteration.

Solves the DARE

    P = Q + Aᵀ P A − Aᵀ P B (R + Bᵀ P B)⁻¹ Bᵀ P A

for the symmetric stabilising solution ``P``, and returns the optimal feedback
gain

    K = (R + Bᵀ P B)⁻¹ Bᵀ P A.

This is the discrete-time form used by the LQR brief (``docs/theory/lqr.md``):
the plant is discretised to ``(A_d, B_d)`` and the controller/estimator run in
discrete time. The fixed point is found by the backward value iteration
(``lqr.md``, "How to solve the DARE"); convergence and the iteration cap follow
``numerical_standards.md`` §5. The implicit inverse ``(R + Bᵀ P B)⁻¹`` is applied
as a symmetric-positive-definite solve via :mod:`inverted_pendulum.numerics.linalg`,
never formed explicitly (``AGENTS.md`` §3, ``lqr.md`` "Practical cautions").

Duality (``lqr.md`` "Duality note"): the discrete Kalman steady-state error
covariance solves the **same** equation with ``A → Aᵀ``, ``B → Cᵀ``, ``Q → W``,
``R → V``. Call :func:`solve_dare` twice rather than writing a second solver.

Reference
---------
docs/theory/lqr.md (DARE and the value-iteration sweep); numerical_standards.md §5.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .constants import ATOL, RTOL
from .linalg import chol_solve, cholesky, is_symmetric, symmetrize

# numerical_standards.md §5: Riccati iteration cap.
RICCATI_MAX_ITER: int = 1000


class RiccatiNotConverged(Exception):
    """Raised when the value iteration does not meet the §5 tolerance in time."""


class DareSolution(NamedTuple):
    """Result of :func:`solve_dare`.

    ``P`` is the symmetric stabilising DARE solution; ``K = (R + Bᵀ P B)⁻¹ Bᵀ P A``
    is the discrete-time feedback gain (shape ``n_u × n_x``); ``iterations`` is the
    number of value-iteration sweeps taken.
    """

    P: np.ndarray
    K: np.ndarray
    iterations: int


def _inf_norm(M: np.ndarray) -> float:
    """Induced ∞-norm ``‖M‖∞`` = max absolute row sum (numerical_standards.md §5)."""
    return float(np.max(np.sum(np.abs(M), axis=1))) if M.size else 0.0


def _gain(A, B, Q, R, P):
    """K = (R + Bᵀ P B)⁻¹ Bᵀ P A, via an SPD solve (no explicit inverse)."""
    S = symmetrize(R + B.T @ P @ B)
    return chol_solve(S, B.T @ P @ A)


def solve_dare(A, B, Q, R, *, max_iter: int = RICCATI_MAX_ITER) -> DareSolution:
    """Solve the DARE for the stabilising ``P`` and feedback gain ``K``.

    Backward value iteration (``lqr.md``): initialise ``P ← Q`` and iterate

        P ← Q + Aᵀ P A − Aᵀ P B (R + Bᵀ P B)⁻¹ Bᵀ P A

    until ``‖P_next − P‖∞ ≤ ATOL + RTOL · ‖P‖∞`` (numerical_standards.md §5),
    then read ``K`` off the converged ``P``.

    Parameters
    ----------
    A : (n_x, n_x) array_like
        Discrete-time state matrix ``A_d``.
    B : (n_x, n_u) array_like
        Discrete-time input matrix ``B_d``.
    Q : (n_x, n_x) array_like
        Symmetric positive-semidefinite state cost.
    R : (n_u, n_u) array_like
        Symmetric positive-definite control cost.
    max_iter : int
        Iteration cap before raising :class:`RiccatiNotConverged` (default 1000).

    Returns
    -------
    DareSolution
        ``P``, ``K = (R + Bᵀ P B)⁻¹ Bᵀ P A``, and the sweep count.

    Raises
    ------
    ValueError
        On shape mismatch or non-symmetric ``Q``/``R``.
    NotPositiveDefiniteError
        If ``R`` is not positive-definite.
    RiccatiNotConverged
        If the tolerance is not met within ``max_iter`` sweeps.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    n_x = A.shape[0]
    if A.shape != (n_x, n_x):
        raise ValueError(f"A must be square, got {A.shape}")
    if B.ndim != 2 or B.shape[0] != n_x:
        raise ValueError(f"B must be (n_x, n_u) with n_x={n_x}, got {B.shape}")
    n_u = B.shape[1]
    if Q.shape != (n_x, n_x):
        raise ValueError(f"Q must be {(n_x, n_x)}, got {Q.shape}")
    if R.shape != (n_u, n_u):
        raise ValueError(f"R must be {(n_u, n_u)}, got {R.shape}")
    if not is_symmetric(Q):
        raise ValueError("Q must be symmetric (within SYM_TOL)")
    if not is_symmetric(R):
        raise ValueError("R must be symmetric (within SYM_TOL)")
    cholesky(symmetrize(R))  # validates R ≻ 0 (raises NotPositiveDefiniteError otherwise)

    P = symmetrize(Q)
    for iteration in range(1, max_iter + 1):
        K = _gain(A, B, Q, R, P)            # (R + BᵀPB)⁻¹ BᵀPA at current P
        P_next = symmetrize(Q + A.T @ P @ A - (B.T @ P @ A).T @ K)
        if _inf_norm(P_next - P) <= ATOL + RTOL * _inf_norm(P):
            return DareSolution(
                P=P_next, K=_gain(A, B, Q, R, P_next), iterations=iteration
            )
        P = P_next

    raise RiccatiNotConverged(
        f"DARE value iteration did not converge within {max_iter} sweeps"
    )
